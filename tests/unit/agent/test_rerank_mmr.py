"""Tests for post-rerank MMR diversity re-ordering.

Covers the standalone helper (opsrag/rerankers/mmr.py) and its wiring
into the rerank node (opsrag/agent/nodes/reranker.py). The node must be
a strict pass-through when the diversity flag is off (default), and must
break up near-duplicate candidates when it's on.
"""
from __future__ import annotations

import pytest

from opsrag.agent.nodes.reranker import rerank_node
from opsrag.interfaces.chunker import Chunk
from opsrag.interfaces.parser import DocType
from opsrag.interfaces.reranker import RerankResult
from opsrag.rerankers.mmr import jaccard_similarity, mmr_reorder


# --- standalone MMR helper -------------------------------------------------


def test_jaccard_identical_and_disjoint():
    assert jaccard_similarity("alpha beta gamma", "alpha beta gamma") == 1.0
    assert jaccard_similarity("alpha beta", "delta epsilon") == 0.0
    # Partial overlap: {a,b,c} vs {a,b,d} -> 2/4 = 0.5
    assert jaccard_similarity("a b c", "a b d") == pytest.approx(0.5)


def test_mmr_reorders_near_duplicates():
    """Three near-identical configs at the top + one distinct doc ranked
    last. With diversity on, the distinct doc should be promoted ahead of
    the redundant near-duplicates."""
    # Candidates in rerank (relevance) order. The first three are
    # near-duplicate config variants; the fourth is a distinct doc the
    # reranker scored lowest.
    items = ["cfg-a", "cfg-b", "cfg-c", "runbook"]
    texts = {
        "cfg-a": "replicas 3 image nginx port 8080 env prod",
        "cfg-b": "replicas 3 image nginx port 8080 env staging",
        "cfg-c": "replicas 3 image nginx port 8080 env dev",
        "runbook": "oncall escalation pagerduty severity rollback steps",
    }
    relevance = [0.90, 0.89, 0.88, 0.50]

    out = mmr_reorder(
        items,
        relevance=relevance,
        diversity=0.7,
        text_of=lambda i: texts[i],
        top_k=3,
    )

    # First pick is always the most relevant (cfg-a).
    assert out[0] == "cfg-a"
    # The distinct runbook must be promoted into the top-3 ahead of at
    # least one near-duplicate config -- that is the whole point of MMR.
    assert "runbook" in out
    assert len(out) == 3


def test_mmr_disabled_is_passthrough():
    """diversity in (None, 0.0) must return relevance order untouched
    (sliced to top_k). This guards the default-OFF contract."""
    items = ["a", "b", "c", "d"]
    relevance = [0.9, 0.8, 0.7, 0.6]
    txt = {"a": "x", "b": "y", "c": "z", "d": "w"}

    for div in (0.0, None):
        out = mmr_reorder(
            items, relevance=relevance, diversity=div,
            text_of=lambda i: txt[i], top_k=3,
        )
        assert out == ["a", "b", "c"]


# --- rerank node wiring ----------------------------------------------------


def _chunk(cid: str, content: str) -> Chunk:
    return Chunk(
        id=cid,
        content=content,
        doc_type=DocType.GENERIC_MARKDOWN,
        source_path=f"repo/{cid}.yaml",
        repo="acme/configs",
    )


class _FakeReranker:
    """Returns the candidates with descending scores in input order, so we
    can deterministically assert what the node does AFTER reranking."""

    score_floor = 0.05
    trust_score = 0.65

    async def rerank(self, query, results, top_k=5):
        n = len(results)
        return [
            RerankResult(chunk=r.chunk, relevance_score=0.9 - i * 0.01)
            for i, r in enumerate(results[:top_k] if top_k else results)
        ][:n]


class _NullObservability:
    async def log_retrieval(self, *a, **k):
        return None


@pytest.mark.asyncio
async def test_node_default_flag_off_is_passthrough():
    """With no rerank_diversity in state and diversity arg defaulting to
    0.0, the node must keep the cross-encoder's order verbatim."""
    chunks = [
        _chunk("cfg-a", "replicas 3 image nginx port 8080 env prod"),
        _chunk("cfg-b", "replicas 3 image nginx port 8080 env staging"),
        _chunk("cfg-c", "replicas 3 image nginx port 8080 env dev"),
        _chunk("runbook", "oncall escalation pagerduty severity rollback"),
    ]
    node = rerank_node(_FakeReranker(), _NullObservability(), top_k=3)
    out = await node({"query": "nginx config", "retrieved_chunks": chunks})
    kept_ids = [c.id for c in out["merged_results"]]
    # Pure rerank order, no diversity shuffle.
    assert kept_ids == ["cfg-a", "cfg-b", "cfg-c"]


@pytest.mark.asyncio
async def test_node_diversity_on_promotes_distinct_doc():
    """With rerank_diversity set in state, the distinct runbook should be
    pulled into the top-3 ahead of a redundant config variant."""
    chunks = [
        _chunk("cfg-a", "replicas 3 image nginx port 8080 env prod"),
        _chunk("cfg-b", "replicas 3 image nginx port 8080 env staging"),
        _chunk("cfg-c", "replicas 3 image nginx port 8080 env dev"),
        _chunk("runbook", "oncall escalation pagerduty severity rollback"),
    ]
    node = rerank_node(_FakeReranker(), _NullObservability(), top_k=3)
    out = await node(
        {"query": "nginx config", "retrieved_chunks": chunks, "rerank_diversity": 0.7}
    )
    kept_ids = [c.id for c in out["merged_results"]]
    assert kept_ids[0] == "cfg-a"  # most relevant still first
    assert "runbook" in kept_ids   # diversity rescued the distinct doc
    assert len(kept_ids) == 3
