"""Grounding gates must FAIL CLOSED, and the default multi_agent path must
actually run the shared groundedness check.

Covers the F6 remediation:
  - `verify_groundedness` returns False on LLM error (no silent grounding).
  - `check_hallucination_node` marks an answer NOT grounded when the check
    errors (was: silently grounded=True).
  - `answer_verifier` appends a caution instead of returning the answer clean
    when the verifier LLM errors / returns malformed output.
  - `generator_node` (default multi_agent path) runs the shared gate when
    `verify_grounding=True` and sets `generation_grounded` from the real
    result -- and does NOT hardcode True.
  - `grader._grade_one` still fails OPEN (keeps the doc) on error.
"""
from __future__ import annotations

from opsrag.agent.nodes import answer_verifier as av
from opsrag.agent.nodes.grader import _grade_one
from opsrag.agent.nodes.hallucination import (
    check_hallucination_node,
    verify_groundedness,
)
from opsrag.agent.nodes.multi_agent import generator_node
from opsrag.interfaces.chunker import Chunk
from opsrag.interfaces.parser import DocType


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


class _BoomStructuredLLM:
    """generate_structured always raises -- the grounding/grader error path."""

    model_name = "fake-model"

    async def generate_structured(self, **kw):
        raise RuntimeError("llm down")


class _GroundedLLM:
    model_name = "fake-model"

    def __init__(self, grounded: bool) -> None:
        self._grounded = grounded

    async def generate_structured(self, **kw):
        g = self._grounded

        class R:  # noqa: D401
            grounded = g

        return R()


# -- verify_groundedness (the shared helper) --------------------------------


async def test_verify_groundedness_fails_closed_on_error():
    assert await verify_groundedness(_BoomStructuredLLM(), "an answer", [_chunk()]) is False


async def test_verify_groundedness_true_when_llm_says_grounded():
    assert await verify_groundedness(_GroundedLLM(True), "an answer", [_chunk()]) is True


async def test_verify_groundedness_false_when_llm_says_not_grounded():
    assert await verify_groundedness(_GroundedLLM(False), "an answer", [_chunk()]) is False


# -- check_hallucination_node fail-closed -----------------------------------


async def test_hallucination_node_fails_closed_marks_not_grounded():
    node = check_hallucination_node(_BoomStructuredLLM(), _Obs())
    out = await node({"generation": "some answer", "final_chunks": [_chunk()]})
    # The whole point of F6: an unverifiable answer must NOT be silently grounded.
    assert out["generation_grounded"] is False
    assert out["grounding_checked"] is True


# -- answer_verifier fail-closed (append caution, don't pass clean) ----------


class _BoomGenLLM:
    """`generate` raises -> verifier can't run."""

    model_name = "fake-model"

    async def generate(self, **kw):
        raise RuntimeError("verifier llm down")


class _MalformedGenLLM:
    """`generate` returns un-parseable content -> verdict is None."""

    model_name = "fake-model"

    async def generate(self, **kw):
        class R:
            content = "not json at all -- no braces here"

        return R()


async def test_answer_verifier_appends_caution_on_llm_error():
    node = av.verify_answer_node(_BoomGenLLM(), None, _Obs())
    answer = "We deploy via `apps/foo/values.yaml`."
    out = await node({"generation": answer, "final_chunks": [_chunk()]})
    assert out["generation"].startswith(answer)
    assert av._CAUTION in out["generation"]
    assert out["verification_result"]["fail_closed"] is True


async def test_answer_verifier_appends_caution_on_malformed_verdict():
    node = av.verify_answer_node(_MalformedGenLLM(), None, _Obs())
    answer = "We deploy via `apps/foo/values.yaml`."
    out = await node({"generation": answer, "final_chunks": [_chunk()]})
    assert out["generation"].endswith(av._CAUTION)
    assert out["verification_result"]["skipped"] is True


# -- grader still fails OPEN (recall > precision on grader outage) -----------


async def test_grader_fails_open_keeps_doc_on_error():
    # _grade_one returns True (keep) when the LLM errors -- dropping recall on a
    # grader outage is worse than keeping an extra chunk.
    assert await _grade_one(_BoomStructuredLLM(), "the query", _chunk()) is True


# -- default multi_agent generator path runs the shared gate -----------------


class _GenLLM:
    """generate() returns a fixed answer; generate_structured drives grounding."""

    model_name = "fake-model"

    def __init__(self, *, answer: str, grounded: bool) -> None:
        self._answer = answer
        self._grounded = grounded

    async def generate(self, **kw):
        ans = self._answer

        class R:
            content = ans

        return R()

    async def generate_structured(self, **kw):
        g = self._grounded

        class R:
            grounded = g

        return R()


def _gen_state(answer_chunk: Chunk) -> dict:
    return {
        "query": "how does the deploy pipeline work",
        "tool_message_history": [
            {"role": "assistant", "content": "draft", "response": {}},
        ],
        "tool_retrieved_chunks": [answer_chunk],
        "tool_call_audit": [],
        "model_route_decision": {},
    }


async def test_generator_default_path_sets_grounded_from_real_check():
    llm = _GenLLM(answer="The pipeline lives in repo/c1.md.", grounded=True)
    node = generator_node(llm, _Obs(), verify_grounding=True)
    out = await node(_gen_state(_chunk()))
    assert out["generation_grounded"] is True
    assert out["grounding_checked"] is True


async def test_generator_default_path_fails_closed_when_not_grounded():
    llm = _GenLLM(answer="A claim not in the evidence.", grounded=False)
    node = generator_node(llm, _Obs(), verify_grounding=True)
    out = await node(_gen_state(_chunk()))
    # No longer hardcoded True -- the real (failed) verdict wins.
    assert out["generation_grounded"] is False
    assert out["grounding_checked"] is True
    # Fail-closed: a caution was appended so the user isn't shown an unverified
    # answer as if it were clean.
    assert "could not be verified" in out["generation"]


async def test_generator_default_path_skips_gate_when_disabled():
    # verify_grounding=False -> no check runs; answer is NOT claimed grounded
    # (the old hardcoded True is gone), and grounding_checked stays False.
    llm = _GenLLM(answer="An answer.", grounded=True)
    node = generator_node(llm, _Obs(), verify_grounding=False)
    out = await node(_gen_state(_chunk()))
    assert out["generation_grounded"] is False
    assert out["grounding_checked"] is False


async def test_generator_live_tool_only_answer_is_unverified_not_failed():
    # No retrieved doc chunks (live-tool-only answer): nothing to ground against
    # -> unverified (grounded False, grounding_checked False), not a failure.
    llm = _GenLLM(answer="Pod is CrashLoopBackOff.", grounded=True)
    node = generator_node(llm, _Obs(), verify_grounding=True)
    state = _gen_state(_chunk())
    state["tool_retrieved_chunks"] = []  # live tools only, no doc chunks
    out = await node(state)
    assert out["generation_grounded"] is False
    assert out["grounding_checked"] is False
