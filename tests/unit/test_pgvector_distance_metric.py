"""H4 / M2 / D1-alpha coverage for the pgvector vector store.

These run fully OFFLINE against a fake asyncpg pool/conn (no Postgres). They
prove:

  * H4 -- the configured distance metric is threaded into the HNSW ops class,
    the ORDER BY operator, the score expression, and the score_threshold clause:
      - cosine (default) is BYTE-IDENTICAL to the pre-H4 hardcoded SQL
        (vector_cosine_ops / '<=>' / '1 - (...)').
      - dot  -> vector_ip_ops / '<#>'  / negated score.
      - euclid -> vector_l2_ops / '<->' / negated score.
  * H4 metric guard -- an existing HNSW index built for a different ops class
    fails closed (parity with the dimension guard), unless
    allow_dimension_change drops+rebuilds it.
  * M2 -- a failing HNSW index build LOGS A WARNING (does not raise, does not
    swallow silently).
  * D1-alpha -- hybrid_search no longer accepts an `alpha` kwarg.
"""
from __future__ import annotations

import inspect

import pytest

pytest.importorskip("asyncpg")  # pgvector extra; skip in the minimal unit job

from opsrag.interfaces.parser import DocType
from opsrag.vectorstores.pgvector import (
    _DISTANCE_OPS,
    PgVectorStore,
    _score_expr,
)


# --------------------------------------------------------------------------
# Fake asyncpg pool / connection -- records executed SQL, returns canned rows.
# --------------------------------------------------------------------------
class _FakeConn:
    def __init__(self, *, fetch_rows=None, fetchval_map=None,
                 create_index_raises: Exception | None = None):
        self.executed: list[str] = []
        self.fetched: list[tuple[str, tuple]] = []
        self._fetch_rows = fetch_rows or []
        self._fetchval_map = fetchval_map or {}
        self._create_index_raises = create_index_raises

    async def execute(self, sql, *args):
        # Simulate the HNSW build failing when asked (M2).
        if (
            self._create_index_raises is not None
            and "USING hnsw" in sql
        ):
            raise self._create_index_raises
        self.executed.append(sql)
        return "EXECUTE"

    async def fetch(self, sql, *args):
        self.fetched.append((sql, args))
        return list(self._fetch_rows)

    async def fetchval(self, sql, *args):
        # Match by a substring key so callers can stub specific lookups.
        for key, val in self._fetchval_map.items():
            if key in sql:
                return val
        return None

    async def fetchrow(self, sql, *args):
        return None


class _FakeAcquireCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _FakeAcquireCtx(self._conn)


class _FakeRow(dict):
    """asyncpg Record stand-in: dict subscripting + .get()."""


def _row(score: float = 0.5) -> _FakeRow:
    return _FakeRow(
        chunk_id="c1",
        content="acme-notes rollback runbook",
        doc_type=DocType.RUNBOOK.value,
        source_path="runbooks/001.md",
        repo="samples",
        parent_chunk_id=None,
        chunk_type="child",
        token_count=10,
        metadata={},
        priority=None,
        score=score,
    )


def _store_with_conn(conn, *, distance="cosine") -> PgVectorStore:
    store = PgVectorStore(dsn="postgres://x", dimension=8, distance=distance)
    store._pool = _FakePool(conn)
    return store


# --------------------------------------------------------------------------
# _score_expr unit mapping (the heart of H4).
# --------------------------------------------------------------------------
def test_score_expr_cosine_is_byte_identical():
    # cosine MUST be the exact pre-H4 string so existing deployments don't drift.
    assert _score_expr("cosine", "<=>") == "1 - (embedding <=> $1::vector)"


def test_score_expr_dot_and_euclid_are_negated():
    assert _score_expr("dot", "<#>") == "-(embedding <#> $1::vector)"
    assert _score_expr("euclid", "<->") == "-(embedding <-> $1::vector)"


def test_distance_ops_table_maps_each_metric():
    assert _DISTANCE_OPS["cosine"] == ("vector_cosine_ops", "<=>")
    assert _DISTANCE_OPS["dot"] == ("vector_ip_ops", "<#>")
    assert _DISTANCE_OPS["euclid"] == ("vector_l2_ops", "<->")


def test_unknown_distance_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown pgvector distance"):
        PgVectorStore(dsn="postgres://x", distance="manhattan")


# --------------------------------------------------------------------------
# H4 -- ensure_table builds the HNSW index with the metric's ops class.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ensure_table_cosine_uses_cosine_ops_class():
    conn = _FakeConn()
    store = _store_with_conn(conn, distance="cosine")
    await store.ensure_table()
    hnsw = [s for s in conn.executed if "USING hnsw" in s]
    assert hnsw, "HNSW index build SQL should have run"
    assert "vector_cosine_ops" in hnsw[0]
    assert "vector_ip_ops" not in hnsw[0]
    assert "vector_l2_ops" not in hnsw[0]


@pytest.mark.asyncio
async def test_ensure_table_dot_uses_ip_ops_class():
    conn = _FakeConn()
    store = _store_with_conn(conn, distance="dot")
    await store.ensure_table()
    hnsw = [s for s in conn.executed if "USING hnsw" in s]
    assert hnsw and "vector_ip_ops" in hnsw[0]


@pytest.mark.asyncio
async def test_ensure_table_euclid_uses_l2_ops_class():
    conn = _FakeConn()
    store = _store_with_conn(conn, distance="euclid")
    await store.ensure_table()
    hnsw = [s for s in conn.executed if "USING hnsw" in s]
    assert hnsw and "vector_l2_ops" in hnsw[0]


# --------------------------------------------------------------------------
# H4 -- search() threads the operator + score expr + threshold clause.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_search_cosine_sql_byte_identical():
    conn = _FakeConn(fetch_rows=[_row(0.9)])
    store = _store_with_conn(conn, distance="cosine")
    store._ensured = True  # skip ensure_table SQL noise
    results = await store.search([0.1] * 8, top_k=5, score_threshold=0.2)
    sql = conn.fetched[-1][0]
    # Pre-H4 hardcoded cosine strings must appear verbatim.
    assert "1 - (embedding <=> $1::vector) AS score" in sql
    assert "ORDER BY embedding <=> $1::vector" in sql
    assert "AND 1 - (embedding <=> $1::vector) >=" in sql
    assert results[0].distance_metric == "cosine"


@pytest.mark.asyncio
async def test_search_dot_uses_ip_operator_and_negated_score():
    conn = _FakeConn(fetch_rows=[_row(0.9)])
    store = _store_with_conn(conn, distance="dot")
    store._ensured = True
    results = await store.search([0.1] * 8, top_k=5, score_threshold=0.2)
    sql = conn.fetched[-1][0]
    assert "-(embedding <#> $1::vector) AS score" in sql
    assert "ORDER BY embedding <#> $1::vector" in sql
    assert "AND -(embedding <#> $1::vector) >=" in sql
    # No cosine operator leaked into a dot store.
    assert "<=>" not in sql
    assert results[0].distance_metric == "dot"


@pytest.mark.asyncio
async def test_search_euclid_uses_l2_operator_and_negated_score():
    conn = _FakeConn(fetch_rows=[_row(0.9)])
    store = _store_with_conn(conn, distance="euclid")
    store._ensured = True
    results = await store.search([0.1] * 8, top_k=5)
    sql = conn.fetched[-1][0]
    assert "-(embedding <-> $1::vector) AS score" in sql
    assert "ORDER BY embedding <-> $1::vector" in sql
    assert "<=>" not in sql and "<#>" not in sql
    assert results[0].distance_metric == "euclid"


# --------------------------------------------------------------------------
# H4 -- hybrid_search dense lane threads the operator + score expr.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_hybrid_dense_lane_dot_operator():
    conn = _FakeConn(fetch_rows=[])
    store = _store_with_conn(conn, distance="dot")
    store._ensured = True
    store._trgm_available = False
    await store.hybrid_search([0.1] * 8, "rollback", top_k=5)
    # The first fetched query is the dense lane (issued before the lexical lane).
    dense_sql = conn.fetched[0][0]
    assert "-(embedding <#> $1::vector) AS score" in dense_sql
    assert "ORDER BY embedding <#> $1::vector" in dense_sql


@pytest.mark.asyncio
async def test_hybrid_dense_lane_cosine_byte_identical():
    conn = _FakeConn(fetch_rows=[])
    store = _store_with_conn(conn, distance="cosine")
    store._ensured = True
    store._trgm_available = False
    await store.hybrid_search([0.1] * 8, "rollback", top_k=5)
    dense_sql = conn.fetched[0][0]
    assert "1 - (embedding <=> $1::vector) AS score" in dense_sql
    assert "ORDER BY embedding <=> $1::vector" in dense_sql


# --------------------------------------------------------------------------
# H4 metric guard -- fail closed on an ops-class mismatch.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_metric_mismatch_fails_closed():
    # Existing index reports cosine ops, but the store is configured for dot.
    conn = _FakeConn(fetchval_map={"opcname": "vector_cosine_ops"})
    store = _store_with_conn(conn, distance="dot")
    with pytest.raises(RuntimeError, match="distance-metric mismatch"):
        await store.ensure_table()


@pytest.mark.asyncio
async def test_metric_match_no_error():
    # Existing index already built for the configured metric -> no raise.
    conn = _FakeConn(fetchval_map={"opcname": "vector_ip_ops"})
    store = _store_with_conn(conn, distance="dot")
    await store.ensure_table()  # should not raise


@pytest.mark.asyncio
async def test_metric_mismatch_drops_index_under_allow_change():
    conn = _FakeConn(fetchval_map={"opcname": "vector_cosine_ops"})
    store = PgVectorStore(
        dsn="postgres://x", dimension=8, distance="dot",
        allow_dimension_change=True,
    )
    store._pool = _FakePool(conn)
    await store.ensure_table()  # drops index loudly, rebuilds, no raise
    assert any("DROP INDEX IF EXISTS idx_chunks_embedding" in s
               for s in conn.executed), "mismatched index should be dropped"
    hnsw = [s for s in conn.executed if "USING hnsw" in s]
    assert hnsw and "vector_ip_ops" in hnsw[0], "rebuilt with the new ops class"


# --------------------------------------------------------------------------
# M2 -- a failing HNSW build logs a WARNING and does NOT raise.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_hnsw_build_failure_warns_not_raises(caplog):
    conn = _FakeConn(create_index_raises=RuntimeError("no hnsw in pgvector<0.5"))
    store = _store_with_conn(conn, distance="cosine")
    import logging
    with caplog.at_level(logging.WARNING, logger="opsrag.vectorstores.pgvector"):
        # Must NOT raise -- a missing HNSW index degrades to a seq scan.
        await store.ensure_table()
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "HNSW index build failed" in msgs
    assert "sequential scan" in msgs
    # ensure_table still completed (state marked ensured).
    assert store._ensured is True


# --------------------------------------------------------------------------
# D1-alpha -- the vestigial alpha kwarg is gone.
# --------------------------------------------------------------------------
def test_hybrid_search_signature_has_no_alpha():
    params = inspect.signature(PgVectorStore.hybrid_search).parameters
    assert "alpha" not in params, "alpha must be removed from pgvector hybrid_search"
    # The real knobs remain.
    assert {"embedding", "query_text", "top_k", "filters"} <= set(params)
