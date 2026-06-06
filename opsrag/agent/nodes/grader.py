"""Document relevance grader node."""
from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, Field

_log = logging.getLogger("opsrag.agent.grader")

from opsrag.agent.prompts import GRADER_SYSTEM
from opsrag.interfaces.chunker import Chunk
from opsrag.interfaces.llm import LLMProvider
from opsrag.interfaces.observability import ObservabilityProvider


class _GradeResult(BaseModel):
    relevant: bool = Field(description="True if the document is directly relevant")


async def _grade_one(llm: LLMProvider, query: str, chunk: Chunk) -> bool:
    prompt = (
        f"Question:\n{query}\n\n"
        f"Document:\n[Source: {chunk.source_path}]\n{chunk.content}\n\n"
        "Is this document directly relevant to the question?"
    )
    try:
        result = await llm.generate_structured(
            purpose="grade",
            messages=[{"role": "user", "content": prompt}],
            schema=_GradeResult,
            system_prompt=GRADER_SYSTEM,
        )
        return result.relevant
    except Exception:
        return True  # Fail open -- don't drop docs when the grader errors


def grade_documents_node(
    llm: LLMProvider,
    observability: ObservabilityProvider,
    min_relevant: int = 1,
):
    async def _grade(state: dict) -> dict:
        query = state["query"]
        candidates: list[Chunk] = state.get("merged_results") or state.get("retrieved_chunks") or []

        if not candidates:
            return {
                "graded_chunks": [],
                "current_step": "graded",
            }

        verdicts = await asyncio.gather(
            *(_grade_one(llm, query, c) for c in candidates)
        )
        kept = [c for c, ok in zip(candidates, verdicts) if ok]

        # Floor: retrieval returned candidates but the strict grader rejected
        # ALL of them. Rather than force an (often unproductive) CRAG rewrite
        # cycle, keep the top `min_relevant` best-ranked candidates -- the
        # hallucination check + answer verifier downstream still guard against
        # ungrounded generation. When retrieval returned NOTHING (candidates
        # empty), we exit above and the rewrite path still fires.
        if not kept and min_relevant > 0:
            kept = candidates[:min_relevant]
            _log.info(
                "grader floor: kept top %d of %d candidates (all failed strict grade)",
                len(kept), len(candidates),
            )

        return {
            "graded_chunks": kept,
            "current_step": "graded",
        }

    return _grade


def grade_decision(state: dict) -> str:
    """Conditional edge: decide whether to rewrite, generate, or fall back.

    CRAG decision tree (Yan et al. 2024 + Phase 2 Step 3 / ADR-004):
    - relevant chunks present -> generate
    - no relevant + retries left -> rewrite query, try again
    - no relevant + max retries hit -> emit honest "insufficient information"
      answer rather than generate with bad chunks (Phase 2 Step 3 addition).
    """
    graded = state.get("graded_chunks") or []
    retries = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if graded:
        return "has_relevant"
    if retries >= max_retries:
        return "insufficient_info"
    return "needs_rewrite"
