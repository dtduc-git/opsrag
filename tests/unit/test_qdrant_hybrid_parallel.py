"""Behavior-equivalence tests for the PARALLELIZED hybrid_search lanes.

The dense / BM25-sparse / code lanes of `QdrantVectorStore.hybrid_search` used
to run as three serial `await query_points(...)` round-trips; they now fire
CONCURRENTLY via `asyncio.gather`. These tests prove the optimization is a pure
latency change:

  * the FUSED RRF ordering + scores are bit-identical to a serial reference
    computed from the same per-lane candidate lists,
  * every per-lane GUARD still short-circuits (zero-vec dense, empty/blank
    query_text, missing code_store / zero-vec code embedding) exactly as before
    -- a skipped lane issues NO network call and contributes nothing,
  * per-lane error semantics match the prior SERIAL code bit-for-bit: a BM25
    or code lane exception degrades that one lane to [] without sinking the
    others (those lanes had their own try/except -> [] fallbacks), while a
    DENSE lane exception PROPAGATES (the dense lane had NO try/except, so a
    dense failure errored the whole query -- gather's return_exceptions=True
    must NOT silently swallow it into a degraded BM25/code-only result).

No fastembed / no Qdrant server: we inject a fake AsyncQdrantClient that returns
deterministic point lists and stub out bm25_sparse.encode_query.
"""
from __future__ import annotations

import asyncio

import pytest
from qdrant_client import models as qm

from opsrag.vectorstores import bm25_sparse
from opsrag.vectorstores.lane_weights import compute_lane_weights
from opsrag.vectorstores.qdrant import _RRF_K, QdrantVectorStore, _priority_rrf_bonus

# --- Fakes ------------------------------------------------------------------


class _FakePoint:
    """Minimal qdrant ScoredPoint stand-in: id + payload (for _hit_to_result)."""

    def __init__(self, pid: str, *, priority: str | None = None):
        self.id = pid
        payload: dict = {"chunk_id": pid, "content": pid, "source_path": f"{pid}.md"}
        if priority is not None:
            payload["priority"] = priority
        self.payload = payload


class _FakeQueryResult:
    def __init__(self, points: list[_FakePoint]):
        self.points = points


class _FakeClient:
    """AsyncQdrantClient stand-in that returns canned per-lane results.

    `lane_results` maps the `using` vector name -> list of _FakePoint. The code
    lane (a separate store/collection) is keyed by ("code", collection_name).
    Records call order + concurrency so we can assert the lanes overlapped.
    """

    def __init__(self, lane_results: dict, *, raise_on: set[str] | None = None,
                 gate: asyncio.Event | None = None):
        self._lane_results = lane_results
        self._raise_on = raise_on or set()
        self._gate = gate
        self.calls: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def query_points(self, *, collection_name, query, using, query_filter=None,
                           search_params=None, limit=None, with_payload=True):
        key = using
        self.calls.append(key)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self._gate is not None:
                # Force overlap: every lane parks here until released, so a
                # serial implementation would deadlock-block (proving gather).
                await self._gate.wait()
            else:
                await asyncio.sleep(0)
            if key in self._raise_on:
                raise RuntimeError(f"boom-{key}")
            return _FakeQueryResult(list(self._lane_results.get(key, [])))
        finally:
            self.in_flight -= 1


def _make_store(client) -> QdrantVectorStore:
    """Build a QdrantVectorStore WITHOUT __init__'s network client setup."""
    store = object.__new__(QdrantVectorStore)
    store._client = client
    store._collection = "opsrag"
    store._ensured = True  # skip ensure_collection's network calls
    return store


# --- Serial reference RRF (the pre-optimization fusion math) -----------------


def _serial_reference(query_text, dense_pts, sparse_pts, code_pts, top_k):
    """Recompute the fused ordering the OLD serial code would produce from the
    same per-lane candidate lists. Mirrors hybrid_search's fusion loop exactly
    (lane weights, RRF k, additive priority bonus, sort)."""
    lane_weights = compute_lane_weights(query_text)
    rrf_score: dict[str, float] = {}
    seen: dict[str, object] = {}
    for hit_list, w in (
        (dense_pts, lane_weights["dense"]),
        (sparse_pts, lane_weights["sparse"]),
        ([], lane_weights["graph"]),
        (code_pts, lane_weights["code"]),
    ):
        for rank, h in enumerate(hit_list, start=1):
            key = str(h.id)
            rrf_score[key] = rrf_score.get(key, 0.0) + w / (_RRF_K + rank)
            if key not in seen:
                seen[key] = h
    for key in list(rrf_score.keys()):
        h = seen.get(key)
        payload = getattr(h, "payload", None) if h else None
        rrf_score[key] += _priority_rrf_bonus((payload or {}).get("priority"))
    ranked = sorted(rrf_score.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return ranked


@pytest.fixture(autouse=True)
def _stub_bm25(monkeypatch):
    """encode_query needs fastembed; stub it to a non-empty sparse vector so the
    BM25 lane's `if sparse_query.indices` guard passes without the model."""
    monkeypatch.setattr(
        bm25_sparse, "encode_query",
        lambda text: qm.SparseVector(indices=[1, 2, 3], values=[1.0, 1.0, 1.0]),
    )


# --- Tests ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_fusion_matches_serial_reference_three_lanes():
    """Dense + BM25 + code all active: parallel fused order == serial reference."""
    dense = [_FakePoint("a"), _FakePoint("b"), _FakePoint("c")]
    sparse = [_FakePoint("b"), _FakePoint("d"), _FakePoint("a")]
    code = [_FakePoint("e"), _FakePoint("a")]

    main_client = _FakeClient({"dense": dense, "bm25": sparse})
    code_client = _FakeClient({"dense": code})
    store = _make_store(main_client)
    code_store = _make_store(code_client)

    query = "where is handle_webhook defined"
    embedding = [0.1, 0.2, 0.3]
    code_embedding = [0.4, 0.5, 0.6]
    top_k = 10

    results = await store.hybrid_search(
        embedding=embedding, query_text=query, top_k=top_k,
        code_embedding=code_embedding, code_store=code_store,
    )

    ref = _serial_reference(query, dense, sparse, code, top_k)
    got = [(r.chunk.id, r.score) for r in results]
    assert [k for k, _ in ref] == [cid for cid, _ in got], "fused ORDER diverged"
    for (rk, rs), (gk, gs) in zip(ref, got):
        assert rk == gk
        assert gs == pytest.approx(rs), "fused SCORE diverged"


@pytest.mark.asyncio
async def test_priority_bonus_preserved_under_parallel():
    """The additive priority bonus must still reorder identically in parallel."""
    # `b` is a single-lane sparse hit but carries priority=high; the bonus must
    # still apply post-fusion exactly as the serial code did.
    dense = [_FakePoint("a"), _FakePoint("c")]
    sparse = [_FakePoint("b", priority="high"), _FakePoint("a")]
    main_client = _FakeClient({"dense": dense, "bm25": sparse})
    store = _make_store(main_client)

    query = "plain prose query without identifiers"
    results = await store.hybrid_search(
        embedding=[1.0, 0.0], query_text=query, top_k=10,
    )
    ref = _serial_reference(query, dense, sparse, [], 10)
    got = [(r.chunk.id, r.score) for r in results]
    assert [k for k, _ in ref] == [cid for cid, _ in got]
    for (rk, rs), (gk, gs) in zip(ref, got):
        assert rk == gk and gs == pytest.approx(rs)


@pytest.mark.asyncio
async def test_lanes_actually_run_concurrently():
    """With a shared gate, all active lanes must be in flight simultaneously --
    a serial implementation could never reach max_in_flight == 2."""
    gate = asyncio.Event()
    dense = [_FakePoint("a")]
    sparse = [_FakePoint("b")]
    main_client = _FakeClient({"dense": dense, "bm25": sparse}, gate=gate)
    store = _make_store(main_client)

    task = asyncio.create_task(store.hybrid_search(
        embedding=[0.1, 0.2], query_text="some query", top_k=5,
    ))
    # Let both lanes reach the gate, then release.
    for _ in range(20):
        await asyncio.sleep(0)
        if main_client.in_flight >= 2:
            break
    assert main_client.in_flight == 2, "dense + bm25 did not overlap"
    gate.set()
    await task
    assert main_client.max_in_flight == 2


@pytest.mark.asyncio
async def test_zero_vec_dense_lane_skipped_no_network():
    """Zero-vector embedding -> dense lane short-circuits (no query_points)."""
    sparse = [_FakePoint("b"), _FakePoint("a")]
    main_client = _FakeClient({"bm25": sparse})
    store = _make_store(main_client)

    query = "bm25 only intent"
    results = await store.hybrid_search(
        embedding=[0.0, 0.0, 0.0], query_text=query, top_k=10,
    )
    assert "dense" not in main_client.calls, "zero-vec dense lane must not query"
    ref = _serial_reference(query, [], sparse, [], 10)
    assert [k for k, _ in ref] == [r.chunk.id for r in results]


@pytest.mark.asyncio
async def test_blank_query_skips_sparse_lane():
    """Empty/blank query_text -> BM25 lane short-circuits (no query_points)."""
    dense = [_FakePoint("a"), _FakePoint("b")]
    main_client = _FakeClient({"dense": dense})
    store = _make_store(main_client)

    results = await store.hybrid_search(
        embedding=[0.1, 0.2], query_text="   ", top_k=10,
    )
    assert "bm25" not in main_client.calls, "blank query must skip BM25 lane"
    ref = _serial_reference("   ", dense, [], [], 10)
    assert [k for k, _ in ref] == [r.chunk.id for r in results]


@pytest.mark.asyncio
async def test_code_lane_skipped_without_store():
    """code_embedding present but code_store None -> code lane short-circuits."""
    dense = [_FakePoint("a")]
    sparse = [_FakePoint("b")]
    main_client = _FakeClient({"dense": dense, "bm25": sparse})
    store = _make_store(main_client)

    query = "q"
    results = await store.hybrid_search(
        embedding=[0.1, 0.2], query_text=query, top_k=10,
        code_embedding=[0.3, 0.4], code_store=None,
    )
    ref = _serial_reference(query, dense, sparse, [], 10)
    assert [k for k, _ in ref] == [r.chunk.id for r in results]


@pytest.mark.asyncio
async def test_one_lane_exception_degrades_to_empty_not_abort():
    """A BM25 lane error degrades that lane to [] (Exception->[]); dense + code
    still fuse, matching the serial try/except fallback semantics."""
    dense = [_FakePoint("a"), _FakePoint("c")]
    code = [_FakePoint("e"), _FakePoint("a")]
    main_client = _FakeClient({"dense": dense, "bm25": [_FakePoint("z")]},
                              raise_on={"bm25"})
    code_client = _FakeClient({"dense": code})
    store = _make_store(main_client)
    code_store = _make_store(code_client)

    query = "query text"
    results = await store.hybrid_search(
        embedding=[0.1, 0.2], query_text=query, top_k=10,
        code_embedding=[0.3, 0.4], code_store=code_store,
    )
    # Sparse contributed nothing (errored) -> reference uses [] for sparse.
    ref = _serial_reference(query, dense, [], code, 10)
    got = [(r.chunk.id, r.score) for r in results]
    assert [k for k, _ in ref] == [cid for cid, _ in got]
    for (rk, rs), (gk, gs) in zip(ref, got):
        assert rk == gk and gs == pytest.approx(rs)
    assert "z" not in [cid for cid, _ in got], "errored sparse lane leaked hits"


@pytest.mark.asyncio
async def test_dense_lane_exception_propagates_not_silently_empty():
    """A DENSE lane failure must RAISE -- it must NOT be swallowed into a
    degraded BM25/code-only result.

    The serial code had NO try/except around the dense query, so a dense
    failure propagated and the whole query errored. The parallel rewrite
    wraps all lanes in `gather(..., return_exceptions=True)`; without an
    explicit re-raise that would coerce a dense exception to [] and the query
    would silently return degraded (BM25-only) results -- a retrieval-quality
    behavior change. This guards against that regression.
    """
    sparse = [_FakePoint("b"), _FakePoint("a")]
    # Dense lane (using="dense" on the MAIN client) raises; BM25 succeeds.
    main_client = _FakeClient({"dense": [_FakePoint("x")], "bm25": sparse},
                              raise_on={"dense"})
    store = _make_store(main_client)

    with pytest.raises(RuntimeError, match="boom-dense"):
        await store.hybrid_search(
            embedding=[0.1, 0.2, 0.3], query_text="some query", top_k=10,
        )


@pytest.mark.asyncio
async def test_code_lane_exception_degrades_to_empty_not_abort():
    """A CODE lane failure degrades that lane to [] (its serial try/except ->
    [] fallback); dense + BM25 still fuse and the query does NOT raise."""
    dense = [_FakePoint("a"), _FakePoint("c")]
    sparse = [_FakePoint("b"), _FakePoint("a")]
    main_client = _FakeClient({"dense": dense, "bm25": sparse})
    # Code lane lives on a SEPARATE store/client; it queries using="dense".
    code_client = _FakeClient({"dense": [_FakePoint("z")]}, raise_on={"dense"})
    store = _make_store(main_client)
    code_store = _make_store(code_client)

    query = "query text"
    results = await store.hybrid_search(
        embedding=[0.1, 0.2], query_text=query, top_k=10,
        code_embedding=[0.3, 0.4], code_store=code_store,
    )
    # Code contributed nothing (errored) -> reference uses [] for code.
    ref = _serial_reference(query, dense, sparse, [], 10)
    got = [(r.chunk.id, r.score) for r in results]
    assert [k for k, _ in ref] == [cid for cid, _ in got]
    for (rk, rs), (gk, gs) in zip(ref, got):
        assert rk == gk and gs == pytest.approx(rs)
    assert "z" not in [cid for cid, _ in got], "errored code lane leaked hits"
