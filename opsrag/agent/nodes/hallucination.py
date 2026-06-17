"""Hallucination / groundedness check node."""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from opsrag.agent.prompts import HALLUCINATION_SYSTEM
from opsrag.interfaces.chunker import Chunk
from opsrag.interfaces.llm import LLMProvider
from opsrag.interfaces.observability import ObservabilityProvider

_log = logging.getLogger("opsrag.agent.hallucination")

# Output token cap for the SHARED, fail-closed groundedness gate. The structured
# payload is the one-field ``_GroundedResult`` ({"grounded": true/false}) -- ~10-20
# tokens including JSON punctuation -- so a 128-token ceiling lets the tiny verdict
# schedule faster WITHOUT any chance of truncating a valid boolean. Kept at 128
# (not lower) precisely because ``verify_groundedness`` fails CLOSED: a truncated /
# unparseable response is treated as NOT grounded, so the cap is sized with ample
# headroom to never risk a spurious "not grounded" verdict. This helper backs BOTH
# build_full_graph's check_hallucination_node AND the F6 default multi_agent-path
# gate, so the single cap here covers both. Quality-neutral: the cap is HONORED on
# non-thinking providers (Anthropic/Bedrock-Claude/OpenAI), while Vertex and
# LiteLLM-Gemini IGNORE it (floor it at their default) to avoid truncating Gemini
# thinking tokens -- which count against max_output_tokens unless a response_schema
# is set, and the in-prompt-schema structured path sets none. Were the cap honored
# on Gemini, truncated thinking would yield empty output -> a spurious fail-closed
# "not grounded" and a needless regen. So the verdict is unchanged on every
# provider.
_GATE_MAX_TOKENS = 128


class _GroundedResult(BaseModel):
    grounded: bool = Field(description="True if every factual claim is supported by the context")


def _grounding_chunks(state: dict) -> list[Chunk]:
    """The evidence the generator actually answered from.

    After parent-child substitution the LLM sees ``final_chunks`` (1024-tok
    parents), not ``graded_chunks`` (256-tok children); checking the children
    produced spurious "not grounded" verdicts. Mirror answer_verifier's
    fallback chain so every grounding path scores the same evidence."""
    return (
        state.get("final_chunks")
        or state.get("graded_chunks")
        or state.get("merged_results")
        or state.get("retrieved_chunks")
        or []
    )


async def verify_groundedness(
    llm: LLMProvider, answer: str, chunks: list[Chunk]
) -> bool:
    """Shared, FAIL-CLOSED groundedness check.

    Single source of truth for "is every factual claim in `answer` supported by
    `chunks`". Used by both ``check_hallucination_node`` (build_full_graph) and
    the default multi_agent generator path so the two never diverge.

    Returns True only when the LLM affirmatively judges the answer grounded.
    Any error (LLM failure, malformed response) returns False -- we do NOT
    silently ground an answer we could not verify."""
    context = "\n\n---\n\n".join(
        f"[Source: {c.source_path}]\n{c.content}" for c in chunks
    ) or "(no context)"
    prompt = (
        f"Context:\n{context}\n\n"
        f"Answer:\n{answer}\n\n"
        "Is every factual claim in the answer supported by the context?"
    )
    try:
        result = await llm.generate_structured(
            purpose="hallucination-check",
            messages=[{"role": "user", "content": prompt}],
            schema=_GroundedResult,
            system_prompt=HALLUCINATION_SYSTEM,
            max_tokens=_GATE_MAX_TOKENS,
        )
        return bool(result.grounded)
    except Exception as exc:  # noqa: BLE001 -- fail closed, never silently ground
        _log.warning(
            "groundedness check errored (%s); failing CLOSED (unverified -> not grounded)",
            exc,
        )
        return False


def check_hallucination_node(llm: LLMProvider, observability: ObservabilityProvider):
    async def _check(state: dict) -> dict:
        answer = state.get("generation", "")
        # Ground against the SAME evidence the generator answered from (see
        # `_grounding_chunks`).
        chunks = _grounding_chunks(state)
        if not answer:
            # No answer -> not grounded; count the attempt on the REGENERATE
            # counter (separate from the CRAG rewrite's retry_count) so the two
            # loops don't cannibalize one budget -- 2 rewrites then a grounding
            # failure used to hit max_retries with ZERO regenerate attempts.
            return {
                "generation_grounded": False,
                "current_step": "hallucination_checked",
                "grounding_checked": True,
                "regen_count": state.get("regen_count", 0) + 1,
            }

        # Shared, FAIL-CLOSED check: on any error this returns False (the helper
        # logs + warns), so an unverifiable answer is treated as not grounded
        # rather than silently shipped as grounded.
        grounded = await verify_groundedness(llm, answer, chunks)

        out: dict = {
            "generation_grounded": grounded,
            "current_step": "hallucination_checked",
            "grounding_checked": True,
        }
        # Bound the regenerate loop on its OWN counter (regen_count), not the
        # shared retry_count -- otherwise CRAG rewrites spend the same budget and
        # a grounding failure after them ships ungrounded with no regen attempt.
        if not grounded:
            out["regen_count"] = state.get("regen_count", 0) + 1
        return out

    return _check


def hallucination_decision(state: dict) -> str:
    if state.get("generation_grounded", True):
        return "grounded"
    # Regenerate loop budget, independent of the CRAG rewrite budget. Defaults to
    # max_retries (the prod-seeded value) when max_regens isn't set explicitly.
    regens = state.get("regen_count", 0)
    max_regens = state.get("max_regens", state.get("max_retries", 3))
    if regens >= max_regens:
        return "max_retries_hit"
    return "not_grounded"
