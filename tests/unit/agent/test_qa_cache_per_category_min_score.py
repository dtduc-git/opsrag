"""M4 (c): the per-category qa_threshold reaches qa_cache.lookup.

The dead-wired `policy_for(category)['qa_threshold']` is now threaded into
`QAVectorCache.lookup(..., min_score=...)` on BOTH cache-lookup lanes
(`query_with_session` and `query_with_session_events`), so tighter
categories (MIXED 0.96, INFRA_GRAPH 0.94, FORENSIC 0.92, ...) enforce
their own cosine floor instead of the one global cache threshold.

We monkeypatch `classify_query` to a deterministic category and assert on
the `min_score` kwarg the lookup actually receives -- testing the seam
where the value was dropped, not the classifier itself.
"""
from __future__ import annotations

import asyncio

import pytest

import opsrag.agent.classifier as classifier
from opsrag.agent.classifier import (
    ClassificationResult,
    QueryCategory,
    policy_for,
)
from opsrag.agent.graph import (
    query_with_session,
    query_with_session_events,
)


class _FakeEmbedder:
    async def embed_query(self, text):  # noqa: ANN001
        return [0.0, 1.0, 0.0]


class _RecordingCache:
    """Records the `min_score` kwarg seen by `lookup`; never returns a hit."""

    def __init__(self) -> None:
        self.min_scores_seen: list[object] = []

    async def lookup(self, *a, min_score=None, **k):  # noqa: ANN001, ANN002
        self.min_scores_seen.append(min_score)
        return None

    async def store(self, *a, **k):  # noqa: ANN001, ANN002
        return "qa-id"


class _InvokeGraph:
    async def ainvoke(self, initial, config=None):  # noqa: ANN001
        return {
            "query": initial.get("query"),
            "generation": "an answer",
            "generation_grounded": True,
            "grounding_checked": True,
            "final_chunks": [],
            "current_step": "done",
        }

    async def astream_events(self, initial, config=None, version=None):  # noqa: ANN001
        # Minimal stream: emit nothing graph-ish; the cache lane runs before this.
        if False:  # pragma: no cover - never yields; lookup happens pre-stream
            yield {}


def _patch_category(monkeypatch, category: QueryCategory) -> None:
    async def _fake_classify(query, **kwargs):  # noqa: ANN001, ANN003
        return ClassificationResult(category=category, layer="test")

    monkeypatch.setattr(classifier, "classify_query", _fake_classify)


@pytest.mark.parametrize(
    "category",
    [QueryCategory.MIXED, QueryCategory.INFRA_GRAPH, QueryCategory.FORENSIC],
)
def test_non_streaming_passes_per_category_min_score(monkeypatch, category):
    _patch_category(monkeypatch, category)
    cache = _RecordingCache()
    asyncio.run(
        query_with_session(
            compiled_graph=_InvokeGraph(),
            query="why did the deploy fail in cycle 7",
            user_id="u1",
            thread_id="t1",
            embedder=_FakeEmbedder(),
            qa_cache=cache,
            session_store=None,
            llm=None,
        )
    )
    expected = policy_for(category)["qa_threshold"]
    assert cache.min_scores_seen == [expected]


@pytest.mark.parametrize(
    "category",
    [QueryCategory.MIXED, QueryCategory.INFRA_GRAPH, QueryCategory.FORENSIC],
)
def test_streaming_passes_per_category_min_score(monkeypatch, category):
    _patch_category(monkeypatch, category)
    cache = _RecordingCache()

    async def _drain():
        async for _ in query_with_session_events(
            compiled_graph=_InvokeGraph(),
            query="why did the deploy fail in cycle 7",
            user_id="u1",
            thread_id="t1",
            embedder=_FakeEmbedder(),
            qa_cache=cache,
            session_store=None,
            llm=None,
        ):
            pass

    asyncio.run(_drain())
    expected = policy_for(category)["qa_threshold"]
    assert cache.min_scores_seen == [expected]


def test_min_score_none_when_no_classification(monkeypatch):
    """Embedder present but classification fails -> min_score is None
    (falls back to the cache's own global threshold)."""
    async def _boom(query, **kwargs):  # noqa: ANN001, ANN003
        raise RuntimeError("classifier down")

    monkeypatch.setattr(classifier, "classify_query", _boom)
    cache = _RecordingCache()
    asyncio.run(
        query_with_session(
            compiled_graph=_InvokeGraph(),
            query="how do I roll back the deploy",
            user_id="u1",
            thread_id="t1",
            embedder=_FakeEmbedder(),
            qa_cache=cache,
            session_store=None,
            llm=None,
        )
    )
    assert cache.min_scores_seen == [None]
