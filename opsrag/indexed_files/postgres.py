"""Postgres-backed `indexed_files` tracker.

Schema (created on first ``open()`` call -- idempotent):

    CREATE TABLE indexed_files (
      repo         TEXT NOT NULL,
      branch       TEXT NOT NULL,
      path         TEXT NOT NULL,
      source_type  TEXT NOT NULL DEFAULT 'gitlab',
      content_hash TEXT NOT NULL,
      indexed_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      chunk_count  INT NOT NULL DEFAULT 0,
      PRIMARY KEY (repo, branch, path)
    );

The ``last_seen_at`` column drives the repo-level deletion sweep: at the END
of a successful ``index_repo`` run, rows whose ``last_seen_at`` predates the
run start were not seen this pass (deleted from source) and are purged --
see ``sweep_deleted`` and ``IngestionPipeline.index_repo``.
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from datetime import datetime

import psycopg
from psycopg_pool import AsyncConnectionPool

_log = logging.getLogger("opsrag.indexed_files.postgres")


def _repo_lock_key(repo: str, branch: str) -> int:
    """Derive a stable SIGNED 64-bit advisory-lock key for a (repo, branch).

    ``pg_try_advisory_lock`` takes a bigint (signed 64-bit). We hash the
    ``repo@branch`` identity with blake2b (digest_size=8 -> 8 bytes), then
    interpret those bytes as a signed big-endian integer so the value always
    fits Postgres' bigint domain (the unsigned interpretation overflows for
    high bytes and raises ``numeric out of range``). Mirrors the fixed-key
    advisory-lock pattern in ``opsrag.db.migrate`` but scoped per repo/branch
    so distinct repos never contend with one another.
    """
    digest = hashlib.blake2b(
        f"{repo}@{branch}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big", signed=True)

# Two statements run separately -- psycopg with prepare_threshold=0 rejects
# multi-statement prepared queries.
_SCHEMA_TABLE = """
CREATE TABLE IF NOT EXISTS indexed_files (
    repo         TEXT NOT NULL,
    branch       TEXT NOT NULL,
    path         TEXT NOT NULL,
    source_type  TEXT NOT NULL DEFAULT 'gitlab',
    content_hash TEXT NOT NULL,
    indexed_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    chunk_count  INT NOT NULL DEFAULT 0,
    PRIMARY KEY (repo, branch, path)
)
"""

_SCHEMA_INDEX = (
    "CREATE INDEX IF NOT EXISTS indexed_files_repo_branch "
    "ON indexed_files (repo, branch)"
)


class PostgresIndexedFilesTracker:
    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 5):
        self._dsn = dsn
        self._pool = AsyncConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        self._ready = False

    async def open(self) -> None:
        await self._pool.open()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SCHEMA_TABLE)
                await cur.execute(_SCHEMA_INDEX)
        self._ready = True
        _log.info("indexed_files schema ready")

    async def close(self) -> None:
        await self._pool.close()
        self._ready = False

    @asynccontextmanager
    async def repo_lock(self, repo: str, branch: str) -> AsyncIterator[bool]:
        """Non-blocking per-(repo, branch) advisory lock for index_repo runs.

        Yields True if THIS caller acquired the lock, False if another run
        already holds it (the caller should SKIP its overlapping run rather
        than block). Uses ``pg_try_advisory_lock`` (non-blocking) so we never
        park a pooled connection for minutes waiting on a concurrent reindex.

        The lock is session-scoped to a DEDICATED, standalone connection (NOT
        the shared worker pool) held for the whole ``async with`` body, and
        released with ``pg_advisory_unlock`` in ``finally`` on that SAME
        connection. A dedicated connection matters: the lock is held for the
        entire (minutes-long) index run, and the indexing workers
        (``OPSRAG_FILE_PARALLEL``) draw from the shared tracker pool -- parking
        the lock on a pooled connection peaks demand at 1+workers and can
        exhaust/deadlock the pool (zero headroom at the default max_size~5, and
        raising file_parallel deadlocks outright). The standalone connection
        never competes with workers. If the tracker isn't ready (no schema yet)
        we yield True so behaviour matches the no-op tracker -- the lock is an
        optimization, never a correctness gate that could wedge indexing on a
        transient Postgres hiccup.
        """
        if not self._ready:
            yield True
            return
        key = _repo_lock_key(repo, branch)
        conn: psycopg.AsyncConnection | None = None
        # Scope the broad except to ONLY the acquisition (open connection +
        # pg_try_advisory_lock). A locking failure must never block indexing
        # (parity with the no-op tracker) -- treat it as "acquired" so the run
        # proceeds; the operator can investigate Postgres separately. Crucially
        # the body's `yield True` is OUTSIDE this except, so an exception raised
        # by the caller's `async with` body propagates cleanly instead of being
        # swallowed here (which would mask the real error as a confusing
        # "generator didn't stop after athrow()").
        try:
            conn = await psycopg.AsyncConnection.connect(
                self._dsn, autocommit=True, prepare_threshold=0
            )
            async with conn.cursor() as cur:
                await cur.execute("SELECT pg_try_advisory_lock(%s)", (key,))
                row = await cur.fetchone()
            got = bool(row and row[0])
        except Exception as exc:
            if conn is not None:
                await conn.close()
            _log.warning(
                "repo_lock failed repo=%s branch=%s: %s -- proceeding without lock",
                repo, branch, exc,
            )
            yield True
            return
        if not got:
            # Lost a real contention race: another run holds the lock. Close the
            # spare connection and signal the caller to skip its overlapping run.
            await conn.close()
            yield False
            return
        try:
            yield True
        finally:
            # Release on the SAME session/connection that holds it, then close.
            # Swallow a release hiccup: Postgres auto-releases session advisory
            # locks on disconnect (the conn.close() below triggers that), and an
            # unlock error must never override a caller-body exception that is
            # being unwound through this finally.
            try:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT pg_advisory_unlock(%s)", (key,))
            except Exception as exc:  # noqa: BLE001 -- release is best-effort
                _log.warning(
                    "repo_lock release failed repo=%s branch=%s: %s "
                    "(session lock auto-released on close)",
                    repo, branch, exc,
                )
            finally:
                await conn.close()

    async def should_skip(
        self, repo: str, branch: str, path: str, content_hash: str
    ) -> bool:
        if not self._ready:
            return False
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT content_hash FROM indexed_files "
                    "WHERE repo = %s AND branch = %s AND path = %s",
                    (repo, branch, path),
                )
                row = await cur.fetchone()
        return bool(row and row[0] == content_hash)

    async def record(
        self,
        repo: str,
        branch: str,
        path: str,
        content_hash: str,
        chunk_count: int,
    ) -> None:
        if not self._ready:
            return
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO indexed_files
                        (repo, branch, path, content_hash, chunk_count,
                         indexed_at, last_seen_at)
                    VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                    ON CONFLICT (repo, branch, path) DO UPDATE SET
                        content_hash = EXCLUDED.content_hash,
                        chunk_count  = EXCLUDED.chunk_count,
                        indexed_at   = NOW(),
                        last_seen_at = NOW()
                    """,
                    (repo, branch, path, content_hash, chunk_count),
                )

    async def repo_branches(self) -> dict[str, str]:
        """Best-known branch per repo, from the actually-indexed files.

        The startup backfill rebuilds /indexing/status from Qdrant payloads,
        which carry `repo` but not `branch`. Without this it defaulted every
        unconfigured repo to "main", mislabeling repos indexed on `master`
        (e.g. genapp) and creating duplicate rows. We pick the branch with the
        most indexed files per repo as canonical. Empty on any failure."""
        if not self._ready:
            return {}
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT DISTINCT ON (repo) repo, branch "
                        "FROM indexed_files "
                        "GROUP BY repo, branch "
                        "ORDER BY repo, count(*) DESC"
                    )
                    rows = await cur.fetchall()
            return {r[0]: r[1] for r in rows}
        except Exception as exc:
            _log.warning("repo_branches lookup failed: %s", exc)
            return {}

    async def mark_seen(
        self, repo: str, branch: str, paths: Iterable[str]
    ) -> None:
        if not self._ready:
            return
        path_list = list(paths)
        if not path_list:
            return
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                # Single bulk update -- psycopg expands the list to ANY ($1).
                await cur.execute(
                    "UPDATE indexed_files SET last_seen_at = NOW() "
                    "WHERE repo = %s AND branch = %s AND path = ANY(%s)",
                    (repo, branch, path_list),
                )

    async def sweep_deleted(
        self, repo: str, branch: str, run_started_at: datetime
    ) -> list[str]:
        """Purge tracker rows for files that vanished from source this run.

        A file present last run but NOT seen this run keeps its older
        ``last_seen_at`` (every seen file -- re-indexed via ``record`` or
        skipped via ``mark_seen`` -- gets bumped to NOW()). So rows with
        ``last_seen_at < run_started_at`` are exactly the deleted files.
        DELETE...RETURNING gives us their paths in one round-trip so the
        caller can drop the matching chunks from the vector store.
        """
        if not self._ready:
            return []
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM indexed_files "
                    "WHERE repo = %s AND branch = %s AND last_seen_at < %s "
                    "RETURNING path",
                    (repo, branch, run_started_at),
                )
                rows = await cur.fetchall()
        return [r[0] for r in rows]
