"""M1 -- knowledge_search must apply the SAME rerank enrichments as the
LangGraph rerank node (path-anchor boost, MMR diversity, pre-rerank
content-dedup, weak-retrieval signal).

The fix factored the enrichment logic into the shared
``apply_rerank_enrichments`` helper and calls it from BOTH ``rerank_node``
and ``_h_knowledge_search`` so the two paths can't re-diverge. These tests
pin the parity: given the SAME candidate pool, the tool path's final order
matches the rerank node's, and the dedup / MMR / anchor-boost / weak-retrieval
behaviours all fire on the tool path too.
"""
from __future__ import annotations

import pytest

import opsrag.mcp.knowledge as knowledge
from opsrag.agent.nodes.reranker import rerank_node
from opsrag.interfaces.chunker import Chunk
from opsrag.interfaces.parser import DocType
from opsrag.interfaces.reranker import RerankResult
from opsrag.interfaces.vectorstore import SearchResult

# --- shared fakes ----------------------------------------------------------


def _chunk(cid: str, content: str, *, repo: str = "acme/configs",
           source_path: str | None = None) -> Chunk:
    return Chunk(
        id=cid,
        content=content,
        doc_type=DocType.GENERIC_MARKDOWN,
        source_path=source_path if source_path is not None else f"repo/{cid}.yaml",
        repo=repo,
    )


class _FakeReranker:
    """Descending scores in input order so post-rerank ordering is
    deterministic (mirrors the rerank-node unit test's fake)."""

    score_floor = 0.05
    trust_score = 0.65

    async def rerank(self, query, results, top_k=5):  # noqa: ANN001
        n = len(results)
        return [
            RerankResult(chunk=r.chunk, relevance_score=0.9 - i * 0.01)
            for i, r in enumerate(results[:top_k] if top_k else results)
        ][:n]


class _NullObservability:
    async def log_retrieval(self, *a, **k):
        return None


class _FakeEmbedder:
    async def embed_query(self, query):  # noqa: ANN001
        return [0.0, 0.0, 0.0]


def _make_vector_store(pool: list[Chunk]):
    """A vector store whose hybrid_search returns the given chunk pool as
    SearchResults (bi-encoder order), regardless of query/embedding."""

    class _VS:
        async def hybrid_search(self, embedding, query_text, top_k=10, **kw):  # noqa: ANN001
            return [SearchResult(chunk=c, score=0.5) for c in pool][:top_k]

        async def search(self, embedding, top_k=5):  # noqa: ANN001
            return [SearchResult(chunk=c, score=0.5) for c in pool][:top_k]

    return _VS()


@pytest.fixture
def bind_knowledge():
    """Bind the module-level knowledge_search state and restore it after."""
    saved = (
        knowledge._embedder, knowledge._vector_store, knowledge._code_embedder,
        knowledge._code_vector_store, knowledge._llm, knowledge._reranker,
        knowledge._rerank_diversity, knowledge._rerank_content_dedup,
        knowledge._rerank_content_dedup_threshold,
    )

    def _bind(pool, *, reranker, diversity, content_dedup, content_dedup_threshold):
        knowledge.bind(
            _FakeEmbedder(),
            _make_vector_store(pool),
            reranker=reranker,
            rerank_diversity=diversity,
            rerank_content_dedup=content_dedup,
            rerank_content_dedup_threshold=content_dedup_threshold,
        )

    yield _bind

    (
        knowledge._embedder, knowledge._vector_store, knowledge._code_embedder,
        knowledge._code_vector_store, knowledge._llm, knowledge._reranker,
        knowledge._rerank_diversity, knowledge._rerank_content_dedup,
        knowledge._rerank_content_dedup_threshold,
    ) = saved


# --- parity: same pool -> same final order ---------------------------------


@pytest.mark.asyncio
async def test_knowledge_search_matches_rerank_node_with_mmr(bind_knowledge):
    """Three near-identical configs + one distinct doc. With MMR diversity ON,
    BOTH paths must promote the distinct doc into the top-k ahead of a
    redundant config variant -- and produce the SAME final order."""
    pool = [
        _chunk("cfg-a", "replicas 3 image nginx port 8080 env prod"),
        _chunk("cfg-b", "replicas 3 image nginx port 8080 env staging"),
        _chunk("cfg-c", "replicas 3 image nginx port 8080 env dev"),
        _chunk("runbook", "oncall escalation pagerduty severity rollback"),
    ]
    query = "nginx config"

    # rerank node order on this pool (diversity on, dedup default-on).
    node = rerank_node(_FakeReranker(), _NullObservability(), top_k=3,
                       diversity=0.7, content_dedup=True)
    node_out = await node({"query": query, "retrieved_chunks": list(pool)})
    node_ids = [c.id for c in node_out["merged_results"]]

    # knowledge_search order on the SAME pool (same reranker + same config).
    bind_knowledge(list(pool), reranker=_FakeReranker(), diversity=0.7,
                   content_dedup=True, content_dedup_threshold=0.0)
    ks = await knowledge._h_knowledge_search(None, {"query": query, "k": 3})
    ks_ids = [h["source"].split("/")[-1].replace(".yaml", "") for h in ks["results"]]

    # Both rescued the distinct runbook into the top-3 and agree on order.
    assert node_ids[0] == "cfg-a"
    assert "runbook" in node_ids
    assert ks_ids == node_ids


@pytest.mark.asyncio
async def test_knowledge_search_applies_content_dedup_before_rerank(bind_knowledge):
    """Byte-identical content from two paths must collapse to one BEFORE the
    reranker is called on the tool path (same as the node)."""
    same = "replicas 3 image nginx port 8080 env prod"
    pool = [
        _chunk("copy-a", same, source_path="repo-a/values.yaml"),
        _chunk("copy-b", same, source_path="repo-b/values.yaml"),
        _chunk("distinct", "oncall escalation pagerduty severity rollback",
               source_path="repo-c/runbook.md"),
    ]

    class _SpyReranker(_FakeReranker):
        def __init__(self):
            self.seen: list[list[str]] = []

        async def rerank(self, query, results, top_k=5):  # noqa: ANN001
            self.seen.append([r.chunk.id for r in results])
            return await super().rerank(query, results, top_k=top_k)

    spy = _SpyReranker()
    bind_knowledge(list(pool), reranker=spy, diversity=0.0,
                   content_dedup=True, content_dedup_threshold=0.0)
    ks = await knowledge._h_knowledge_search(None, {"query": "nginx config", "k": 5})

    # The reranker saw the DEDUPED pool -- copy-b dropped, first occurrence kept.
    assert spy.seen == [["copy-a", "distinct"]]
    sources = [h["source"] for h in ks["results"]]
    assert sources == ["repo-a/values.yaml", "repo-c/runbook.md"]


@pytest.mark.asyncio
async def test_knowledge_search_applies_path_anchor_boost(bind_knowledge):
    """An anchor-matching doc the cross-encoder ranked LAST must be promoted by
    the path-anchor boost on the tool path (same additive +bonus as the node)."""
    # The reranker scores in input order (cfg first, anchor-doc last), but the
    # anchor doc's path contains the anchor 'acme-notes-be' -> +0.15 boost should
    # lift it past the close-scoring non-anchor docs.
    pool = [
        _chunk("top", "deployment replicas image config values prod",
               source_path="other/values.yaml", repo="acme/other"),
        _chunk("mid", "deployment replicas image config values staging",
               source_path="other/values2.yaml", repo="acme/other"),
        _chunk("anchor", "service deployment notes backend config",
               source_path="acme-notes-be/deploy.yaml", repo="acme-notes-be"),
    ]
    query = "how is acme-notes-be deployed"

    node = rerank_node(_FakeReranker(), _NullObservability(), top_k=3,
                       diversity=0.0, content_dedup=True)
    node_out = await node({"query": query, "retrieved_chunks": list(pool)})
    node_ids = [c.id for c in node_out["merged_results"]]

    bind_knowledge(list(pool), reranker=_FakeReranker(), diversity=0.0,
                   content_dedup=True, content_dedup_threshold=0.0)
    ks = await knowledge._h_knowledge_search(None, {"query": query, "k": 3})
    ks_ids = [h["source"].split("/")[0] for h in ks["results"]]

    # Node boosted the anchor doc to the top; tool path agrees.
    assert node_ids[0] == "anchor"
    assert ks_ids[0] == "acme-notes-be"


@pytest.mark.asyncio
async def test_knowledge_search_flags_weak_retrieval(bind_knowledge):
    """When the query NAMES an entity (anchor) but NO chunk path/repo matches
    AND the best raw rerank score is below the floor, the tool path must set
    weak_retrieval=True (reasoner treats as 'not in corpus')."""

    class _LowScoreReranker:
        score_floor = 0.5  # high floor so the 0.1 scores fall below it
        trust_score = 0.65

        async def rerank(self, query, results, top_k=5):  # noqa: ANN001
            return [
                RerankResult(chunk=r.chunk, relevance_score=0.1)
                for r in results
            ]

    pool = [
        _chunk("x", "unrelated content about kafka and zookeeper",
               source_path="infra/kafka.yaml", repo="acme/infra"),
        _chunk("y", "unrelated content about redis caching layer",
               source_path="infra/redis.yaml", repo="acme/infra"),
    ]
    # Anchor 'acme-payments-be' is named but no chunk path/repo contains it.
    query = "how does acme-payments-be handle retries"

    bind_knowledge(list(pool), reranker=_LowScoreReranker(), diversity=0.0,
                   content_dedup=True, content_dedup_threshold=0.0)
    ks = await knowledge._h_knowledge_search(None, {"query": query, "k": 5})
    assert ks.get("weak_retrieval") is True


@pytest.mark.asyncio
async def test_knowledge_search_no_weak_flag_when_anchor_matches(bind_knowledge):
    """The weak-retrieval flag must NOT fire when a chunk path matches the
    anchor, even if rerank scores are low -- the entity IS in the corpus."""

    class _LowScoreReranker:
        score_floor = 0.5
        trust_score = 0.65

        async def rerank(self, query, results, top_k=5):  # noqa: ANN001
            return [RerankResult(chunk=r.chunk, relevance_score=0.1) for r in results]

    pool = [
        _chunk("hit", "retry policy backoff config for the payments service",
               source_path="acme-payments-be/retry.yaml", repo="acme-payments-be"),
    ]
    query = "how does acme-payments-be handle retries"

    bind_knowledge(list(pool), reranker=_LowScoreReranker(), diversity=0.0,
                   content_dedup=True, content_dedup_threshold=0.0)
    ks = await knowledge._h_knowledge_search(None, {"query": query, "k": 5})
    assert "weak_retrieval" not in ks
