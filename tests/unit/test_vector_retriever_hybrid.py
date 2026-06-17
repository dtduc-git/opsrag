"""Vector-retriever hybrid + code-lane wiring tests.

Guards the regression that the main retrieval path was dense-only
(`vector_store.search`), bypassing BM25 and the code collection. The node must
now call `hybrid_search`, feed the BM25 lane the RAW user query (not HyDE
prose), and thread the optional code lane through -- but only to stores whose
`hybrid_search` actually accepts the code kwargs (Qdrant), never pgvector.
"""
from __future__ import annotations

from opsrag.agent.nodes.vector_retriever import vector_retrieve_node


class _QdrantLikeStore:
    """hybrid_search accepts the optional code lane kwargs (like Qdrant)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def hybrid_search(
        self, embedding, query_text, top_k=10,
        filters=None, code_embedding=None, code_store=None,
    ):
        self.calls.append({
            "query_text": query_text, "top_k": top_k, "filters": filters,
            "code_embedding": code_embedding, "code_store": code_store,
        })
        return []

    async def search(self, *a, **k):  # pragma: no cover - must not be hit
        raise AssertionError("dense-only search() must not be called on the main path")


class _PgLikeStore:
    """hybrid_search WITHOUT the code kwargs (like pgvector)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def hybrid_search(self, embedding, query_text, top_k=10, filters=None):
        self.calls.append({"query_text": query_text, "filters": filters})
        return []

    async def search(self, *a, **k):  # pragma: no cover
        raise AssertionError("search() must not be called")


class _FakeEmbedder:
    dimension = 4

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def embed_query(self, text: str):
        self.seen.append(text)
        return [0.1, 0.2, 0.3, 0.4]

    async def embed_texts(self, texts: list[str]):
        # HyDE now embeds the hypothetical as a DOCUMENT (embed_texts), not a
        # query -- record it the same way so assertions on `seen` still hold.
        self.seen.extend(texts)
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _FakeObs:
    async def log_retrieval(self, **kwargs):
        return None


async def test_main_path_uses_hybrid_with_raw_query_and_code_lane():
    store = _QdrantLikeStore()
    code_store = _QdrantLikeStore()
    emb = _FakeEmbedder()
    code_emb = _FakeEmbedder()
    node = vector_retrieve_node(
        store, emb, _FakeObs(), top_k=5,
        code_embedder=code_emb, code_store=code_store,
    )
    out = await node({
        "query": "where is handle_webhook defined",
        "hyde_text": "A hypothetical answer about webhooks",
    })

    assert len(store.calls) == 1, "main lane must call hybrid_search exactly once"
    call = store.calls[0]
    # BM25 lane gets the RAW query (exact identifier match), NOT the HyDE prose.
    assert call["query_text"] == "where is handle_webhook defined"
    # Dense lane embedded the HyDE text.
    assert emb.seen[0] == "A hypothetical answer about webhooks"
    # Code lane threaded through, embedded with the code embedder -- using the
    # RAW query, NOT the HyDE prose: CODE_RETRIEVAL_QUERY wants identifier/code-
    # shaped query text, so a prose hypothetical would defeat the code lane.
    assert call["code_embedding"] is not None
    assert call["code_store"] is code_store
    assert code_emb.seen == ["where is handle_webhook defined"]
    assert "bm25" in out["sources_searched"]
    assert "code" in out["sources_searched"]


async def test_pgvector_store_gets_no_code_kwargs():
    store = _PgLikeStore()
    emb = _FakeEmbedder()
    code_emb = _FakeEmbedder()
    node = vector_retrieve_node(
        store, emb, _FakeObs(), top_k=5,
        code_embedder=code_emb, code_store=_QdrantLikeStore(),
    )
    # Must not raise TypeError despite a code lane being configured.
    out = await node({"query": "foo bar baz"})

    assert len(store.calls) == 1
    assert store.calls[0]["query_text"] == "foo bar baz"
    # Code lane silently disabled -> code embedder never invoked.
    assert code_emb.seen == []
    assert "code" not in out["sources_searched"]


async def test_no_code_lane_configured_still_hybrid():
    store = _QdrantLikeStore()
    emb = _FakeEmbedder()
    node = vector_retrieve_node(store, emb, _FakeObs(), top_k=5)
    out = await node({"query": "deploy the gateway"})

    assert len(store.calls) == 1
    assert store.calls[0]["code_embedding"] is None
    assert "bm25" in out["sources_searched"]
    assert "code" not in out["sources_searched"]
