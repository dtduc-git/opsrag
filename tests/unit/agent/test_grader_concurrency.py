"""Tests for the CRAG grader's bounded fan-out concurrency.

The grader issues one relevance LLM call per candidate chunk (up to ~50 after
retrieval + merge). An unbounded asyncio.gather would fire them all at once and
trip provider 429 rate limits, so each per-chunk grade must run behind a
semaphore. These tests instrument a fake LLM that records the peak number of
concurrent in-flight calls and assert it never exceeds the cap.
"""
from __future__ import annotations

import asyncio

import pytest

from opsrag.agent.nodes.grader import _GRADE_CONCURRENCY, grade_documents_node
from opsrag.interfaces.chunker import Chunk
from opsrag.interfaces.parser import DocType


def _chunk(cid: str) -> Chunk:
    return Chunk(
        id=cid,
        content=f"content for {cid}",
        doc_type=DocType.GENERIC_MARKDOWN,
        source_path=f"repo/{cid}.md",
        repo="acme/configs",
    )


class _InstrumentedLLM:
    """Fake LLMProvider that tracks the peak number of concurrent
    generate_structured calls. Each call yields control (sleep 0) so the event
    loop interleaves every coroutine the gather has admitted past the semaphore;
    the recorded max therefore equals the real in-flight ceiling.
    """

    def __init__(self) -> None:
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls = 0

    async def generate_structured(self, *, purpose, messages, schema, system_prompt):
        self.in_flight += 1
        self.calls += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            # Yield repeatedly so other admitted coroutines get to run before we
            # release -- this is what surfaces a missing/too-large cap.
            for _ in range(3):
                await asyncio.sleep(0)
            return schema(relevant=True)
        finally:
            self.in_flight -= 1


class _NullObservability:
    async def log_retrieval(self, *a, **k):
        return None


@pytest.mark.asyncio
async def test_grader_caps_concurrent_llm_calls():
    """50 candidates must NOT produce 50 simultaneous LLM calls."""
    llm = _InstrumentedLLM()
    node = grade_documents_node(llm, _NullObservability())
    candidates = [_chunk(f"c{i}") for i in range(50)]

    out = await node({"query": "q", "merged_results": candidates})

    # Every chunk was graded (and all kept, since the fake says relevant=True).
    assert llm.calls == 50
    assert len(out["graded_chunks"]) == 50
    # The whole point: in-flight calls never exceed the default cap.
    assert llm.max_in_flight <= _GRADE_CONCURRENCY
    # And we actually reached the cap (otherwise the assertion is vacuous).
    assert llm.max_in_flight == _GRADE_CONCURRENCY


@pytest.mark.asyncio
async def test_grader_concurrency_override_from_state():
    """The `grader_concurrency` state key tunes the cap without a code change."""
    llm = _InstrumentedLLM()
    node = grade_documents_node(llm, _NullObservability())
    candidates = [_chunk(f"c{i}") for i in range(20)]

    out = await node(
        {"query": "q", "merged_results": candidates, "grader_concurrency": 3}
    )

    assert llm.calls == 20
    assert len(out["graded_chunks"]) == 20
    assert llm.max_in_flight <= 3
    assert llm.max_in_flight == 3


@pytest.mark.asyncio
async def test_grader_explicit_one_serializes():
    """An explicit cap of 1 must serialize the grade calls (max in-flight 1)."""
    llm = _InstrumentedLLM()
    node = grade_documents_node(llm, _NullObservability())
    candidates = [_chunk(f"c{i}") for i in range(8)]

    out = await node(
        {"query": "q", "merged_results": candidates, "grader_concurrency": 1}
    )

    assert llm.calls == 8
    assert len(out["graded_chunks"]) == 8
    assert llm.max_in_flight == 1


@pytest.mark.asyncio
async def test_grader_falsy_override_stays_bounded():
    """A falsy/garbage override (0, None) must NOT revert to an unbounded
    gather -- it falls back to the safe default cap, never > _GRADE_CONCURRENCY."""
    for bad in (0, None):
        llm = _InstrumentedLLM()
        node = grade_documents_node(llm, _NullObservability())
        candidates = [_chunk(f"c{i}") for i in range(50)]

        out = await node(
            {"query": "q", "merged_results": candidates, "grader_concurrency": bad}
        )

        assert llm.calls == 50
        assert len(out["graded_chunks"]) == 50
        # Still bounded by the default -- the key invariant is "never unbounded".
        assert llm.max_in_flight <= _GRADE_CONCURRENCY
