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
from opsrag.agent.nodes.generator import generate_node
from opsrag.agent.nodes.grader import _grade_one
from opsrag.agent.nodes.hallucination import (
    check_hallucination_node,
    verify_groundedness,
)
from opsrag.agent.nodes.multi_agent import generator_node, triage_route
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
    # A GENUINE live-tool answer: no retrieved doc chunks, but real tool calls
    # fired (tool_call_audit non-empty). Nothing to ground against doc-side, and
    # the tool output IS the grounding -> unverified (grounded False,
    # grounding_checked False), NOT a no-evidence failure (H5 exemption).
    llm = _GenLLM(answer="Pod is CrashLoopBackOff.", grounded=True)
    node = generator_node(llm, _Obs(), verify_grounding=True)
    state = _gen_state(_chunk())
    state["tool_retrieved_chunks"] = []  # live tools only, no doc chunks
    state["tool_call_audit"] = [{"name": "k8s_get_pods"}]  # a real tool fired
    out = await node(state)
    assert out["generation_grounded"] is False
    assert out["grounding_checked"] is False
    # Exempt: a true tool turn is grounded in its tool output -- no caution.
    assert "could not be verified" not in out["generation"]
    assert "without retrieving any" not in out["generation"]


async def test_generator_no_evidence_answer_fails_closed_and_not_cacheable():
    # H5 core: a tool-path turn that emitted NO tool calls AND retrieved NO doc
    # chunks is answering purely from parametric memory. Pre-fix it slipped past
    # the retrieved_chunks guard and shipped clean (grounded False,
    # grounding_checked False) -> cache-ELIGIBLE and unflagged. Now it must fail
    # CLOSED: grounding_checked True + generation_grounded False (so the qa_cache
    # and investigation_cache write gates -- which exclude grounding_checked &&
    # not generation_grounded -- both skip it) and a caution appended.
    llm = _GenLLM(answer="The deploy uses argocd.", grounded=True)
    node = generator_node(llm, _Obs(), verify_grounding=True)
    state = _gen_state(_chunk())
    state["tool_retrieved_chunks"] = []  # no doc chunks
    state["tool_call_audit"] = []        # no tools fired -> NO evidence at all
    out = await node(state)
    assert out["generation_grounded"] is False
    assert out["grounding_checked"] is True  # DURABLE flag -> cache write gate trips
    assert "could not be verified" in out["generation"]
    # Mirror the qa_cache / investigation_cache write gate predicate: an answer
    # the grounding node explicitly failed is NOT cacheable.
    grounded_explicitly_failed = (
        out.get("generation_grounded") is False
        and out.get("grounding_checked") is True
    )
    assert grounded_explicitly_failed is True


# -- C1: multi_agent RETRIEVAL-branch generate_node runs the shared gate ------
#
# The retrieval branch (vector_retrieve -> [rerank] -> generate -> END) is
# terminal -- there is NO downstream check_hallucination_node -- so the gate
# must run INSIDE generate_node, exactly as F6 did for the tool-path
# generator_node. build_full_graph passes verify_grounding=False because it
# runs its own check_hallucination_node (no double-gating).


class _CountingGenLLM(_GenLLM):
    """Like _GenLLM but records how many times the grounding gate
    (generate_structured) was invoked, to prove no double-gating."""

    def __init__(self, *, answer: str, grounded: bool) -> None:
        super().__init__(answer=answer, grounded=grounded)
        self.structured_calls = 0

    async def generate_structured(self, **kw):
        self.structured_calls += 1
        return await super().generate_structured(**kw)


def _retrieval_state(chunk: Chunk) -> dict:
    # generate_node reads chunks from graded_chunks/merged_results/
    # retrieved_chunks (in that order). Use retrieved_chunks here.
    return {
        "query": "how does the deploy pipeline work",
        "retrieved_chunks": [chunk],
    }


async def test_retrieval_generate_sets_grounding_checked_and_grounded_true():
    llm = _GenLLM(answer="The pipeline lives in repo/c1.md.", grounded=True)
    node = generate_node(llm, _Obs(), verify_grounding=True)
    out = await node(_retrieval_state(_chunk()))
    assert out["grounding_checked"] is True
    assert out["generation_grounded"] is True
    # Clean answer: no caution appended.
    assert "could not be verified" not in out["generation"]


async def test_retrieval_generate_fails_closed_when_not_grounded():
    llm = _GenLLM(answer="A claim not in the evidence.", grounded=False)
    node = generate_node(llm, _Obs(), verify_grounding=True)
    out = await node(_retrieval_state(_chunk()))
    # The real (failed) verdict wins and grounding_checked is set True so the
    # qa_cache write gate (generation_grounded False AND grounding_checked True)
    # excludes this answer automatically.
    assert out["generation_grounded"] is False
    assert out["grounding_checked"] is True
    assert "could not be verified" in out["generation"]


async def test_retrieval_generate_fails_closed_on_verify_error():
    # generate() returns an answer; generate_structured (the gate) raises ->
    # verify_groundedness fails CLOSED -> grounded False, grounding_checked True.
    class _Boom(_GenLLM):
        async def generate_structured(self, **kw):
            raise RuntimeError("gate llm down")

    llm = _Boom(answer="Some answer about repo/c1.md.", grounded=True)
    node = generate_node(llm, _Obs(), verify_grounding=True)
    out = await node(_retrieval_state(_chunk()))
    assert out["generation_grounded"] is False
    assert out["grounding_checked"] is True
    assert "could not be verified" in out["generation"]


async def test_retrieval_generate_no_chunks_is_unverified_not_failed():
    # No retrieved chunks -> nothing to ground against -> unverified
    # (grounded False, grounding_checked False), not a failure. Matches the
    # generator_node / tool_synthesize convention so absent-evidence answers
    # still cache (grounding_checked stays False).
    llm = _GenLLM(answer="(no context retrieved)", grounded=True)
    node = generate_node(llm, _Obs(), verify_grounding=True)
    out = await node({"query": "q", "retrieved_chunks": []})
    assert out["generation_grounded"] is False
    assert out["grounding_checked"] is False


# -- H5: triage_route hardening (0-tool tool path -> retrieval, not generator) --


def test_triage_route_no_tools_tool_path_goes_to_retrieval():
    # H5: triage flagged the tool path but emitted NO tool calls. Pre-fix this
    # routed straight to `generator`, which had no evidence and answered from
    # parametric memory. Now it routes to `retrieval` so it gets real corpus
    # grounding first.
    assert triage_route({"tool_path_active": True, "tool_calls": []}) == "retrieval"


def test_triage_route_with_tools_goes_to_tool_caller():
    assert (
        triage_route({"tool_path_active": True, "tool_calls": [{"name": "x"}]})
        == "tool_caller"
    )


def test_triage_route_non_tool_path_goes_to_retrieval():
    assert triage_route({"tool_path_active": False}) == "retrieval"


async def test_full_graph_generate_does_not_double_gate():
    # build_full_graph passes verify_grounding=False because it runs its own
    # check_hallucination_node after generate. The retrieval-branch gate must
    # NOT run here, so generate_structured is never called (no double-gating)
    # and grounding stays unchecked for the downstream node to set.
    llm = _CountingGenLLM(answer="The pipeline lives in repo/c1.md.", grounded=True)
    node = generate_node(llm, _Obs(), verify_grounding=False)
    out = await node(_retrieval_state(_chunk()))
    assert llm.structured_calls == 0
    assert out["grounding_checked"] is False
    assert out["generation_grounded"] is False
    # No caution appended when the gate is disabled.
    assert "could not be verified" not in out["generation"]
