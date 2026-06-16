"""Unit tests for F3 (index_repo advisory lock) and F8b (dedup fingerprint).

F3 -- a non-blocking per-(repo, branch) advisory lock prevents two overlapping
``index_repo`` runs from racing (an earlier run's deletion sweep could DELETE
chunks a later run hasn't re-touched). On contention the overlapping run SKIPS
(returns 0) instead of blocking a pooled connection.

F8b -- the per-file dedup ``content_hash`` folds in an index-config fingerprint
(chunker sizing + chars/token ratios + embedder model/dimension) so a chunking
or embed-model change auto-invalidates dedup (the recomputed hash no longer
matches the recorded one -> the file is re-indexed, not skipped).
"""
from __future__ import annotations

import psycopg
import pytest

from opsrag.indexed_files.postgres import (
    PostgresIndexedFilesTracker,
    _repo_lock_key,
)
from opsrag.ingestion.pipeline import IngestionPipeline

# --------------------------------------------------------------------------- #
# F3 -- advisory-lock key + repo_lock context-manager behaviour
# --------------------------------------------------------------------------- #


def test_repo_lock_key_is_stable_signed_64bit():
    k1 = _repo_lock_key("svc", "main")
    k2 = _repo_lock_key("svc", "main")
    assert k1 == k2  # deterministic
    # Signed 64-bit (Postgres bigint) domain -- must never overflow.
    assert -(2**63) <= k1 < 2**63


def test_repo_lock_key_differs_per_repo_and_branch():
    assert _repo_lock_key("svc", "main") != _repo_lock_key("other", "main")
    assert _repo_lock_key("svc", "main") != _repo_lock_key("svc", "dev")


class _FakeCursor:
    def __init__(self, conn: _FakeStandaloneConn) -> None:
        self._conn = conn
        self._result = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, sql, params=None):
        if "pg_try_advisory_lock" in sql:
            # Grant the lock only if nobody else holds this key.
            key = params[0]
            if key in self._conn.held:
                self._result = (False,)
            else:
                self._conn.held.add(key)
                self._conn.acquired.append(key)
                self._result = (True,)
        elif "pg_advisory_unlock" in sql:
            key = params[0]
            self._conn.held.discard(key)
            self._conn.released.append(key)
            self._result = (True,)

    async def fetchone(self):
        return self._result


class _FakeStandaloneConn:
    """Stand-in for a DEDICATED ``psycopg.AsyncConnection`` (not pool-acquired).

    repo_lock now opens its own standalone connection via
    ``psycopg.AsyncConnection.connect(dsn, ...)`` and later ``await conn.close()``
    -- it no longer borrows from the worker pool. ``shared`` carries the
    advisory-lock state across every connection so a second concurrent connect
    observes a key the first one holds.
    """

    def __init__(self, shared: dict) -> None:
        self.held: set = shared["held"]
        self.acquired: list = shared["acquired"]
        self.released: list = shared["released"]
        self.closed = False

    def cursor(self):
        return _FakeCursor(self)

    async def close(self):
        self.closed = True


class _SharedLockState:
    """Holds the advisory-lock state asserted by the tests (``.shared``)."""

    def __init__(self) -> None:
        self.shared = {"held": set(), "acquired": [], "released": []}


def _make_tracker(monkeypatch) -> PostgresIndexedFilesTracker:
    # Construct without opening a real pool. repo_lock connects via
    # psycopg.AsyncConnection.connect(self._dsn, ...) -> patch that to hand back
    # a fake standalone connection backed by shared lock-state.
    t = PostgresIndexedFilesTracker.__new__(PostgresIndexedFilesTracker)
    t._dsn = "postgresql://test"  # type: ignore[attr-defined]
    t._ready = True
    t._pool = _SharedLockState()  # type: ignore[attr-defined]  # shared-state holder for asserts

    async def _fake_connect(dsn, **kwargs):
        return _FakeStandaloneConn(t._pool.shared)  # type: ignore[attr-defined]

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", _fake_connect)
    return t


@pytest.mark.asyncio
async def test_repo_lock_acquires_then_releases(monkeypatch):
    t = _make_tracker(monkeypatch)
    key = _repo_lock_key("svc", "main")
    async with t.repo_lock("svc", "main") as got:
        assert got is True
        assert key in t._pool.shared["held"]  # type: ignore[attr-defined]
    # Released on exit, on the SAME (dedicated) connection.
    assert key not in t._pool.shared["held"]  # type: ignore[attr-defined]
    assert t._pool.shared["released"] == [key]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_repo_lock_skips_on_contention(monkeypatch):
    t = _make_tracker(monkeypatch)
    async with t.repo_lock("svc", "main") as first:
        assert first is True
        # A concurrent run on the same (repo, branch) cannot get the try-lock.
        async with t.repo_lock("svc", "main") as second:
            assert second is False
    # The contended run never acquired, so nothing extra to unlock.
    assert len(t._pool.shared["acquired"]) == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_repo_lock_not_ready_yields_true():
    t = PostgresIndexedFilesTracker.__new__(PostgresIndexedFilesTracker)
    t._ready = False
    async with t.repo_lock("svc", "main") as got:
        assert got is True


@pytest.mark.asyncio
async def test_index_repo_skips_when_lock_not_acquired():
    """The pipeline returns 0 and does NOT list files when the lock is held."""

    class _LockedTracker:
        from contextlib import asynccontextmanager as _acm

        @_acm
        async def repo_lock(self, repo, branch):
            yield False  # always contended

    class _BoomSCM:
        async def list_files(self, *a, **k):  # pragma: no cover - must not run
            raise AssertionError("index_repo must skip before listing files")

    pipe = IngestionPipeline(
        scm=_BoomSCM(),
        parsers=[],
        chunker=None,
        embedder=None,
        vector_store=object(),
        indexed_files=_LockedTracker(),
    )
    assert await pipe.index_repo("svc", "main") == 0


# --------------------------------------------------------------------------- #
# F8b -- index-config fingerprint folded into the dedup content_hash
# --------------------------------------------------------------------------- #


class _FakeChunker:
    def __init__(self, child_size=256, child_overlap=32):
        self.child_size = child_size
        self.child_overlap = child_overlap

    def chunk(self, doc):  # pragma: no cover - unused in fingerprint test
        return []


class _FakeEmbedder:
    def __init__(self, model_name="cohere-embed-v4", dimension=1536):
        self._model = model_name
        self._dim = dimension

    @property
    def model_name(self):
        return self._model

    @property
    def dimension(self):
        return self._dim

    async def embed_texts(self, texts):  # pragma: no cover
        return [[0.0] for _ in texts]

    async def embed_query(self, query):  # pragma: no cover
        return [0.0]


def _pipe(chunker, embedder) -> IngestionPipeline:
    return IngestionPipeline(
        scm=object(),
        parsers=[],
        chunker=chunker,
        embedder=embedder,
        vector_store=object(),
    )


def test_fingerprint_is_deterministic_and_memoized():
    p = _pipe(_FakeChunker(), _FakeEmbedder())
    fp1 = p._index_fingerprint()
    fp2 = p._index_fingerprint()
    assert fp1 == fp2
    assert len(fp1) == 64  # sha256 hex


def test_fingerprint_changes_when_chunk_size_changes():
    a = _pipe(_FakeChunker(child_size=256), _FakeEmbedder())._index_fingerprint()
    b = _pipe(_FakeChunker(child_size=512), _FakeEmbedder())._index_fingerprint()
    assert a != b


def test_fingerprint_changes_when_embed_model_changes():
    base = _FakeChunker()
    a = _pipe(base, _FakeEmbedder(model_name="cohere-embed-v4"))._index_fingerprint()
    b = _pipe(base, _FakeEmbedder(model_name="titan-embed-v2"))._index_fingerprint()
    assert a != b


def test_fingerprint_changes_when_embed_dimension_changes():
    base = _FakeChunker()
    a = _pipe(base, _FakeEmbedder(dimension=1536))._index_fingerprint()
    b = _pipe(base, _FakeEmbedder(dimension=768))._index_fingerprint()
    assert a != b


# --------------------------------------------------------------------------- #
# F8 (3) -- pgvector allow_dimension_change=true actually DROPs the table
# --------------------------------------------------------------------------- #


class _FakeAsyncpgConn:
    """Minimal asyncpg-conn stand-in for _assert_dimension_compatible."""

    def __init__(self, existing_type: str | None) -> None:
        self._existing_type = existing_type
        self.executed: list[str] = []

    async def fetchval(self, sql, *params):
        return self._existing_type

    async def execute(self, sql, *params):
        self.executed.append(sql)


@pytest.mark.asyncio
async def test_pgvector_allow_change_drops_table_on_mismatch():
    pytest.importorskip("asyncpg")  # pgvector extra; skip when absent (CI unit job)
    from opsrag.vectorstores.pgvector import PgVectorStore

    store = PgVectorStore.__new__(PgVectorStore)
    store._table = "opsrag_chunks"
    store._dimension = 768
    store._allow_dimension_change = True
    conn = _FakeAsyncpgConn("vector(3072)")
    await store._assert_dimension_compatible(conn)
    assert any("DROP TABLE IF EXISTS opsrag_chunks" in s for s in conn.executed)


@pytest.mark.asyncio
async def test_pgvector_mismatch_without_allow_change_raises():
    pytest.importorskip("asyncpg")  # pgvector extra; skip when absent (CI unit job)
    from opsrag.vectorstores.pgvector import PgVectorStore

    store = PgVectorStore.__new__(PgVectorStore)
    store._table = "opsrag_chunks"
    store._dimension = 768
    store._allow_dimension_change = False
    conn = _FakeAsyncpgConn("vector(3072)")
    with pytest.raises(RuntimeError):
        await store._assert_dimension_compatible(conn)
    assert conn.executed == []  # never drops when not opted in


@pytest.mark.asyncio
async def test_pgvector_matching_dimension_is_noop():
    pytest.importorskip("asyncpg")  # pgvector extra; skip when absent (CI unit job)
    from opsrag.vectorstores.pgvector import PgVectorStore

    store = PgVectorStore.__new__(PgVectorStore)
    store._table = "opsrag_chunks"
    store._dimension = 768
    store._allow_dimension_change = True
    conn = _FakeAsyncpgConn("vector(768)")
    await store._assert_dimension_compatible(conn)
    assert conn.executed == []  # nothing to drop
