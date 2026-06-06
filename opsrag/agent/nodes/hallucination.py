"""Hallucination / groundedness check node."""
from __future__ import annotations

from pydantic import BaseModel, Field

from opsrag.agent.prompts import HALLUCINATION_SYSTEM
from opsrag.interfaces.llm import LLMProvider
from opsrag.interfaces.observability import ObservabilityProvider


class _GroundedResult(BaseModel):
    grounded: bool = Field(description="True if every factual claim is supported by the context")


def check_hallucination_node(llm: LLMProvider, observability: ObservabilityProvider):
    async def _check(state: dict) -> dict:
        answer = state.get("generation", "")
        # Ground against the SAME evidence the generator answered from. After
        # parent-child substitution the LLM sees `final_chunks` (1024-tok
        # parents), not `graded_chunks` (256-tok children); checking the
        # children produced spurious "not grounded" verdicts -> wasted
        # regenerate loops. Mirror answer_verifier's fallback chain.
        chunks = (
            state.get("final_chunks")
            or state.get("graded_chunks")
            or state.get("merged_results")
            or state.get("retrieved_chunks")
            or []
        )
        if not answer:
            # No answer -> not grounded; count the attempt so the regenerate
            # loop (check_hallucination -> generate) is bounded by max_retries.
            return {
                "generation_grounded": False,
                "current_step": "hallucination_checked",
                "retry_count": state.get("retry_count", 0) + 1,
            }

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
            )
            grounded = result.grounded
        except Exception:
            grounded = True  # Fail open

        out: dict = {
            "generation_grounded": grounded,
            "current_step": "hallucination_checked",
        }
        # Bound the regenerate loop: each not-grounded verdict counts as a
        # retry so `hallucination_decision` hits `max_retries_hit` instead of
        # looping generate -> verify -> check forever (the loop never
        # incremented retry_count before, so a persistently-strict grounding
        # check spun indefinitely).
        if not grounded:
            out["retry_count"] = state.get("retry_count", 0) + 1
        return out

    return _check


def hallucination_decision(state: dict) -> str:
    if state.get("generation_grounded", True):
        return "grounded"
    retries = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    if retries >= max_retries:
        return "max_retries_hit"
    return "not_grounded"
