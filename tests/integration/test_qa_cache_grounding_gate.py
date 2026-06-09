"""Regression: ungrounded answers must NOT be written to the Q&A cache.

The hallucination check sets `generation_grounded` and a DURABLE
`grounding_checked` flag. An answer that fails grounding but ships anyway via
the `max_retries_hit` route still flows into the terminal `save_memory` node,
which overwrites `current_step` to "memory_saved". The old cache-write gate
keyed on `current_step == "hallucination_checked"`, so with memory enabled the
"skip caching ungrounded answers" guard was ~always False and ungrounded
answers leaked into the cache (90-day forensic TTL) and were re-served.

These tests drive the real `query_with_session` cache-write gate with a fake
compiled graph, embedder, and cache:

  - ungrounded (max-retries, then save_memory clobbers current_step) -> NOT cached
  - grounded -> cached
  - minimal mode (no hallucination check ran) -> cached (absent check != failure)
"""
from __future__ import annotations

import pytest

from opsrag.agent.graph import query_with_session


class _FakeEmbedder:
    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class _FakeCache:
    """Records store() calls; never returns a lookup hit (forces fresh path)."""

    def __init__(self) -> None:
        self.stored: list[dict] = []

    async def lookup(self, *a, **k):
        return None

    async def store(self, **kwargs) -> None:
        self.stored.append(kwargs)


class _FakeGraph:
    """Stand-in compiled graph: ainvoke returns a scripted final state."""

    def __init__(self, final_state: dict) -> None:
        self._final = final_state

    async def ainvoke(self, initial, config=None) -> dict:
        return dict(self._final)


# A benign question: UNKNOWN category (skip_cache=False), not user-scoped or
# time-sensitive, so it actually reaches the cache-write gate.
_QUERY = "explain the deploy pipeline architecture for the platform"


async def _run(final_state: dict) -> _FakeCache:
    cache = _FakeCache()
    await query_with_session(
        compiled_graph=_FakeGraph(final_state),
        query=_QUERY,
        user_id="anonymous",
        embedder=_FakeEmbedder(),
        qa_cache=cache,
        llm=None,
        session_store=None,
        semantic_router=None,
    )
    return cache


async def test_ungrounded_max_retries_answer_is_not_cached():
    # Exactly the leaky scenario: grounding check ran and FAILED, the answer
    # shipped via max_retries_hit, then save_memory clobbered current_step.
    cache = await _run(
        {
            "generation": "An answer the grounding check rejected.",
            "generation_grounded": False,
            "grounding_checked": True,
            "current_step": "memory_saved",  # save_memory overwrote it
        }
    )
    assert cache.stored == [], "ungrounded answer must NOT be cached"


async def test_grounded_answer_is_cached():
    cache = await _run(
        {
            "generation": "A well-grounded answer.",
            "generation_grounded": True,
            "grounding_checked": True,
            "current_step": "memory_saved",
        }
    )
    assert len(cache.stored) == 1, "grounded answer must be cached"
    assert cache.stored[0]["answer"] == "A well-grounded answer."


async def test_minimal_mode_no_grounding_check_is_cached():
    # Minimal graph has no hallucination check: generation_grounded stays at its
    # initial False and grounding_checked is never set. Absent check is NOT a
    # grounding failure -> still cacheable.
    cache = await _run(
        {
            "generation": "A minimal-mode answer.",
            "generation_grounded": False,
            "current_step": "verified",
        }
    )
    assert len(cache.stored) == 1, "minimal-mode answer (no check) must be cached"
