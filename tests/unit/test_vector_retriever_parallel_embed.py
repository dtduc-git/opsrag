"""Behavior-equivalence tests for the PARALLELIZED embed lanes in the vector
retriever node.

`vector_retrieve_node` used to compute the main-query embedding (HyDE doc-space
or raw query-space) and the optional code-lane embedding in two serial awaits;
they now run CONCURRENTLY via `asyncio.gather`. These tests prove the change is
pure latency:

  * the embeddings fed to `hybrid_search` are identical to the serial path
    (correct HyDE-vs-raw branch, code embedder gets the RAW query),
  * the two embeds actually overlap (gather, not serial),
  * the code-lane best-effort fallback is preserved -- a code-embedder failure
    degrades `code_embedding` to None without sinking the main retrieval,
  * a MAIN-embed failure still propagates (no best-effort there, as before).

The happy-path threading is already covered by test_vector_retriever_hybrid.py;
these focus on the concurrency + failure semantics that changed.
"""
from __future__ import annotations

import asyncio

import pytest

from opsrag.agent.nodes.vector_retriever import vector_retrieve_node


class _QdrantLikeStore:
    """hybrid_search accepts the optional code lane kwargs (like Qdrant)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def hybrid_search(
        self, embedding, query_text, top_k=10, alpha=0.7,
        filters=None, code_embedding=None, code_store=None,
    ):
        self.calls.append({
            "embedding": embedding, "query_text": query_text,
            "code_embedding": code_embedding, "code_store": code_store,
        })
        return []

    async def search(self, *a, **k):  # pragma: no cover
        raise AssertionError("dense-only search() must not be called")

    async def search_by_text(self, *a, **k):  # pragma: no cover
        return []


class _RecordingEmbedder:
    dimension = 4

    def __init__(self, vec=None, *, gate: asyncio.Event | None = None,
                 raise_exc: Exception | None = None) -> None:
        self.seen: list[str] = []
        self._vec = vec or [0.1, 0.2, 0.3, 0.4]
        self._gate = gate
        self._raise = raise_exc
        self.in_flight = 0

    async def _maybe(self):
        self.in_flight += 1
        try:
            if self._gate is not None:
                await self._gate.wait()
            else:
                await asyncio.sleep(0)
            if self._raise is not None:
                raise self._raise
        finally:
            self.in_flight -= 1

    async def embed_query(self, text: str):
        self.seen.append(text)
        await self._maybe()
        return list(self._vec)

    async def embed_texts(self, texts: list[str]):
        self.seen.extend(texts)
        await self._maybe()
        return [list(self._vec) for _ in texts]


class _FakeObs:
    async def log_retrieval(self, **kwargs):
        return None


@pytest.mark.asyncio
async def test_concurrent_embed_feeds_identical_values_hyde():
    """HyDE branch: dense gets the doc-space HyDE embed, code gets the RAW query
    embed -- identical to the serial path."""
    store = _QdrantLikeStore()
    code_store = _QdrantLikeStore()
    main_emb = _RecordingEmbedder(vec=[1.0, 0.0, 0.0, 0.0])
    code_emb = _RecordingEmbedder(vec=[0.0, 1.0, 0.0, 0.0])
    node = vector_retrieve_node(
        store, main_emb, _FakeObs(), top_k=5,
        code_embedder=code_emb, code_store=code_store,
    )
    await node({
        "query": "where is handle_webhook defined",
        "hyde_text": "A hypothetical answer about webhooks",
    })

    call = store.calls[0]
    # Dense lane fed the HyDE doc-space embedding.
    assert call["embedding"] == [1.0, 0.0, 0.0, 0.0]
    assert main_emb.seen == ["A hypothetical answer about webhooks"]
    # Code lane fed the code-embedder over the RAW query.
    assert call["code_embedding"] == [0.0, 1.0, 0.0, 0.0]
    assert code_emb.seen == ["where is handle_webhook defined"]
    assert call["code_store"] is code_store


@pytest.mark.asyncio
async def test_concurrent_embed_feeds_identical_values_no_hyde():
    """No-HyDE branch: dense gets the raw query embed (embed_query)."""
    store = _QdrantLikeStore()
    code_store = _QdrantLikeStore()
    main_emb = _RecordingEmbedder(vec=[2.0, 0.0, 0.0, 0.0])
    code_emb = _RecordingEmbedder(vec=[0.0, 2.0, 0.0, 0.0])
    node = vector_retrieve_node(
        store, main_emb, _FakeObs(), top_k=5,
        code_embedder=code_emb, code_store=code_store,
    )
    await node({"query": "deploy the gateway"})

    call = store.calls[0]
    assert call["embedding"] == [2.0, 0.0, 0.0, 0.0]
    assert main_emb.seen == ["deploy the gateway"]
    assert call["code_embedding"] == [0.0, 2.0, 0.0, 0.0]
    assert code_emb.seen == ["deploy the gateway"]


@pytest.mark.asyncio
async def test_embeds_run_concurrently():
    """With a shared gate, both embedders must be in flight simultaneously --
    a serial implementation could never reach in_flight==1 on both at once."""
    gate = asyncio.Event()
    store = _QdrantLikeStore()
    code_store = _QdrantLikeStore()
    main_emb = _RecordingEmbedder(gate=gate)
    code_emb = _RecordingEmbedder(gate=gate)
    node = vector_retrieve_node(
        store, main_emb, _FakeObs(), top_k=5,
        code_embedder=code_emb, code_store=code_store,
    )
    task = asyncio.create_task(node({"query": "concurrent please"}))
    for _ in range(50):
        await asyncio.sleep(0)
        if main_emb.in_flight == 1 and code_emb.in_flight == 1:
            break
    assert main_emb.in_flight == 1 and code_emb.in_flight == 1, (
        "main + code embed did not overlap"
    )
    gate.set()
    await task


@pytest.mark.asyncio
async def test_code_embed_failure_degrades_to_none_best_effort():
    """A code-embedder failure must NOT sink retrieval: code_embedding -> None,
    main lane still runs (best-effort fallback preserved)."""
    store = _QdrantLikeStore()
    code_store = _QdrantLikeStore()
    main_emb = _RecordingEmbedder(vec=[3.0, 0.0, 0.0, 0.0])
    code_emb = _RecordingEmbedder(raise_exc=RuntimeError("code embed down"))
    node = vector_retrieve_node(
        store, main_emb, _FakeObs(), top_k=5,
        code_embedder=code_emb, code_store=code_store,
    )
    out = await node({"query": "find handle_event"})

    assert len(store.calls) == 1, "main retrieval must still run"
    call = store.calls[0]
    assert call["embedding"] == [3.0, 0.0, 0.0, 0.0]
    # Code lane disabled because its embed failed -> None, NOT threaded through.
    assert call["code_embedding"] is None
    assert call["code_store"] is None
    assert "code" not in out["sources_searched"]


@pytest.mark.asyncio
async def test_main_embed_failure_propagates():
    """The main embedding has no best-effort fallback -- a failure must still
    raise out of the node, as in the serial path."""
    store = _QdrantLikeStore()
    main_emb = _RecordingEmbedder(raise_exc=RuntimeError("main embed down"))
    node = vector_retrieve_node(store, main_emb, _FakeObs(), top_k=5)
    with pytest.raises(RuntimeError, match="main embed down"):
        await node({"query": "anything"})
