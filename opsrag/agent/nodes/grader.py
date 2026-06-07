"""Document relevance grader node."""
from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, Field

_log = logging.getLogger("opsrag.agent.grader")

# If the cross-encoder's best (un-boosted) score is at least this, we trust the
# reranker's top hit even when the binary relevance grader rejects everything --
# rather than burn a CRAG rewrite on retrieval the reranker was confident about.
# Scores are sigmoid-normalized [0,1]; 0.5 = a genuinely relevant match.
_TRUST_RERANK_SCORE = 0.5

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

        # Floor, but ONLY as a last resort once the CRAG rewrite budget is
        # spent. The strict grader rejected everything; on the FIRST attempts we
        # leave `graded_chunks` empty so `grade_decision` routes to
        # `needs_rewrite` (the whole point of CRAG -- rewrite the query and
        # re-retrieve when the retrieved docs are irrelevant). Only when retries
        # are exhausted do we keep the top `min_relevant` best-ranked candidates
        # rather than bail to "insufficient info" -- a best-effort answer the
        # hallucination check + verifier still guard. (Unconditionally flooring
        # here made graded_chunks always non-empty, which silently disabled the
        # rewrite loop entirely.)
        if not kept and min_relevant > 0:
            retries = state.get("retry_count", 0)
            max_retries = state.get("max_retries", 2)
            # Keep the reranker's top hit when the cross-encoder was CONFIDENT
            # about it: the binary per-chunk grader runs AFTER rerank and can
            # only remove recall, so nuking a strong cross-encoder #1 to rewrite
            # the query wastes the budget on already-good retrieval. Fall back to
            # the rewrite loop only when the rerank was weak (genuinely-bad
            # retrieval, what CRAG is for) OR the retry budget is spent.
            best_rr = float(state.get("best_rerank_score", 0.0))
            if best_rr >= _TRUST_RERANK_SCORE or retries >= max_retries:
                kept = candidates[:min_relevant]
                _log.info(
                    "grader floor (%s): kept top %d of %d candidates "
                    "(all failed strict grade)",
                    "confident rerank" if best_rr >= _TRUST_RERANK_SCORE else "retries exhausted",
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
