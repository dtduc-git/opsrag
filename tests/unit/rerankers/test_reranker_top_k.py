"""The Vertex + FastEmbed rerankers must HONOR the caller's `top_k`.

The rerank node calls `reranker.rerank(query, results, top_k=len(results))`
deliberately -- it scores the full candidate pool so the path-anchor boost
can rescue an anchor-matching doc the cross-encoder ranked deep
(opsrag/agent/nodes/reranker.py). Cohere/Bedrock honor that wide top_k;
these two must not silently cap the result list below the requested count.

This guards against a regression where vertex/fastembed truncate to their
own (small) default top_k internally.
"""
from __future__ import annotations

import json

import httpx
import pytest

from opsrag.interfaces.chunker import Chunk
from opsrag.interfaces.parser import DocType
from opsrag.interfaces.vectorstore import SearchResult


def _results(n: int) -> list[SearchResult]:
    out = []
    for i in range(n):
        chunk = Chunk(
            id=f"c{i}",
            content=f"document number {i} about service alpha replicas {i}",
            doc_type=DocType.GENERIC_MARKDOWN,
            source_path=f"repo/doc{i}.md",
            repo="acme/configs",
        )
        out.append(SearchResult(chunk=chunk, score=0.0))
    return out


# --- Vertex ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_vertex_honors_caller_top_k_returns_all():
    """With top_k == len(results), Vertex must return one RerankResult per
    candidate (max-pooled), not cap below the request."""
    from opsrag.rerankers import vertex as vertex_mod

    n = 12
    results = _results(n)

    def _handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        # Echo a score for every record the reranker sent. IDs are
        # "{candidate_idx}:{window_idx}"; give later candidates lower scores
        # so ordering is deterministic but every candidate is represented.
        records = []
        for rec in payload["records"]:
            cand_idx = int(str(rec["id"]).split(":", 1)[0])
            records.append({"id": rec["id"], "score": 1.0 - cand_idx * 0.01})
        return httpx.Response(200, json={"records": records})

    rr = vertex_mod.VertexReranker(project="test-project")
    # Avoid ADC + real network: inject a MockTransport and stub the token.
    rr._client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    rr._get_token = lambda: "fake-token"  # type: ignore[method-assign]

    out = await rr.rerank("alpha service", results, top_k=len(results))
    await rr.close()

    assert len(out) == n, "vertex capped below the requested top_k"
    # Descending by max-pooled score, every candidate present exactly once.
    ids = [r.chunk.id for r in out]
    assert sorted(ids) == sorted(f"c{i}" for i in range(n))


@pytest.mark.asyncio
async def test_vertex_small_top_k_still_caps():
    """When the caller asks for a small top_k, Vertex must still respect it
    (the cap is the *caller's* number, not an internal default)."""
    from opsrag.rerankers import vertex as vertex_mod

    results = _results(8)

    def _handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        records = [
            {"id": rec["id"], "score": 1.0 - int(str(rec["id"]).split(":", 1)[0]) * 0.01}
            for rec in payload["records"]
        ]
        return httpx.Response(200, json={"records": records})

    rr = vertex_mod.VertexReranker(project="test-project")
    rr._client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    rr._get_token = lambda: "fake-token"  # type: ignore[method-assign]

    out = await rr.rerank("alpha service", results, top_k=3)
    await rr.close()
    assert len(out) == 3


# --- FastEmbed -------------------------------------------------------------


class _StubCrossEncoder:
    """Stub TextCrossEncoder: returns a descending logit per document so we
    avoid the ~90MB ONNX model download in unit tests."""

    def __init__(self, *a, **k):
        pass

    @staticmethod
    def list_supported_models():
        return [{"model": "Xenova/ms-marco-MiniLM-L-6-v2"}]

    def rerank(self, query, documents):
        # Raw logits (cross-encoder native scale), descending.
        return [5.0 - i for i in range(len(documents))]


@pytest.mark.asyncio
async def test_fastembed_honors_caller_top_k_returns_all(monkeypatch):
    """With top_k == len(results), FastEmbed must return every candidate."""
    pytest.importorskip("fastembed")  # optional extra; skip when absent (CI unit job)
    import opsrag.rerankers.fastembed_reranker as fe

    monkeypatch.setattr(fe, "TextCrossEncoder", _StubCrossEncoder)

    n = 15
    results = _results(n)
    rr = fe.FastEmbedReranker()
    out = await rr.rerank("alpha service", results, top_k=len(results))

    assert len(out) == n, "fastembed capped below the requested top_k"
    ids = {r.chunk.id for r in out}
    assert ids == {f"c{i}" for i in range(n)}
    # Sigmoid-mapped into (0, 1), still descending order.
    scores = [r.relevance_score for r in out]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 < s < 1.0 for s in scores)


@pytest.mark.asyncio
async def test_fastembed_small_top_k_still_caps(monkeypatch):
    """A small caller top_k must still cap FastEmbed's output."""
    pytest.importorskip("fastembed")  # optional extra; skip when absent (CI unit job)
    import opsrag.rerankers.fastembed_reranker as fe

    monkeypatch.setattr(fe, "TextCrossEncoder", _StubCrossEncoder)

    results = _results(10)
    rr = fe.FastEmbedReranker()
    out = await rr.rerank("alpha service", results, top_k=4)
    assert len(out) == 4
