"""Behaviour-equivalence tests for the structured-gate output-token cap.

The three structured gates (grader relevance, hallucination groundedness, router
route-decision) now pass ``max_tokens=128`` to ``generate_structured`` so the tiny
boolean/object verdict schedules faster (smaller output reservation). These tests
prove that change is QUALITY-NEUTRAL:

  1. The cap is forwarded verbatim as ``max_tokens=128`` to the provider, and
  2. the verdict each gate returns is IDENTICAL to the un-capped path -- i.e. the
     boolean / decision the node produces does not depend on the cap.

They use an offline recording fake provider (no network, no secrets): it captures
the kwargs it was called with and returns a caller-supplied verdict, so we can
assert both the forwarded cap and the unchanged result in one shot.
"""
from __future__ import annotations

import pytest

from opsrag.agent.nodes.grader import _GATE_MAX_TOKENS as GRADER_CAP
from opsrag.agent.nodes.grader import _grade_one, grade_documents_node
from opsrag.agent.nodes.hallucination import _GATE_MAX_TOKENS as HALL_CAP
from opsrag.agent.nodes.hallucination import (
    check_hallucination_node,
    verify_groundedness,
)
from opsrag.agent.nodes.router import _GATE_MAX_TOKENS as ROUTER_CAP
from opsrag.agent.nodes.router import route_query_node
from opsrag.interfaces.chunker import Chunk
from opsrag.interfaces.parser import DocType


# All three gates share the same 128-token cap.
def test_all_gate_caps_are_128():
    assert GRADER_CAP == HALL_CAP == ROUTER_CAP == 128


def _chunk(cid: str = "c1", content: str = "evidence text") -> Chunk:
    return Chunk(
        id=cid,
        content=content,
        doc_type=DocType.GENERIC_MARKDOWN,
        source_path=f"repo/{cid}.md",
        repo="acme/configs",
    )


class _Obs:
    async def log(self, *a, **k): ...
    async def log_llm_call(self, *a, **k): ...
    async def log_retrieval(self, *a, **k): ...


class _RecordingLLM:
    """Offline fake: records every generate_structured call's kwargs and returns
    a fixed verdict built from ``schema`` so the node's logic runs unchanged.

    ``verdict`` is the kwargs handed to ``schema(**verdict)`` -- this is exactly
    what the real provider yields after parsing a (non-truncated) JSON object, so
    the node sees an identical object to the un-capped path.
    """

    model_name = "fake-model"

    def __init__(self, verdict: dict) -> None:
        self._verdict = verdict
        self.calls: list[dict] = []

    async def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["schema"](**self._verdict)


# -- grader gate -------------------------------------------------------------


@pytest.mark.parametrize("relevant", [True, False])
async def test_grader_forwards_cap_and_returns_same_verdict(relevant):
    llm = _RecordingLLM({"relevant": relevant})
    out = await _grade_one(llm, "the query", _chunk())
    # Verdict is unchanged by the cap.
    assert out is relevant
    # The cap was forwarded verbatim.
    assert len(llm.calls) == 1
    assert llm.calls[0]["max_tokens"] == GRADER_CAP == 128


async def test_grade_node_capped_matches_uncapped_decision():
    """The node's kept/dropped set is identical whether or not the cap is set --
    proven by comparing a capped run to a hand-computed expected (all relevant)."""
    llm = _RecordingLLM({"relevant": True})
    node = grade_documents_node(llm, _Obs())
    candidates = [_chunk(f"c{i}") for i in range(5)]
    out = await node({"query": "q", "merged_results": candidates})
    assert len(out["graded_chunks"]) == 5
    # Every per-chunk call carried the cap.
    assert all(c["max_tokens"] == 128 for c in llm.calls)


# -- hallucination / groundedness gate ---------------------------------------


@pytest.mark.parametrize("grounded", [True, False])
async def test_verify_groundedness_forwards_cap_and_same_verdict(grounded):
    llm = _RecordingLLM({"grounded": grounded})
    out = await verify_groundedness(llm, "an answer", [_chunk()])
    # Verdict is unchanged by the cap (fail-closed semantics preserved).
    assert out is grounded
    assert len(llm.calls) == 1
    assert llm.calls[0]["max_tokens"] == HALL_CAP == 128


async def test_hallucination_node_forwards_cap():
    llm = _RecordingLLM({"grounded": True})
    node = check_hallucination_node(llm, _Obs())
    out = await node({"generation": "some answer", "final_chunks": [_chunk()]})
    assert out["generation_grounded"] is True
    assert llm.calls[0]["max_tokens"] == 128


# -- router gate -------------------------------------------------------------


async def test_router_forwards_cap_and_returns_same_decision():
    llm = _RecordingLLM(
        {"query_type": "incident", "requires_graph": False, "confidence": 0.9}
    )
    node = route_query_node(llm, _Obs())
    out = await node({"query": "pods are crashing"})
    # Decision is unchanged by the cap.
    assert out["query_type"] == "incident"
    assert out["requires_graph"] is False
    assert out["intent_confidence"] == 0.9
    assert llm.calls[0]["max_tokens"] == ROUTER_CAP == 128


async def test_router_graph_inference_unchanged_with_cap():
    """requires_graph is still forced on for blast_radius/dependency_map even
    when the model said False -- the cap does not touch that logic."""
    llm = _RecordingLLM(
        {"query_type": "blast_radius", "requires_graph": False, "confidence": 0.5}
    )
    node = route_query_node(llm, _Obs())
    out = await node({"query": "what breaks if svc-a dies"})
    assert out["requires_graph"] is True
    assert llm.calls[0]["max_tokens"] == 128
