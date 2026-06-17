"""Document relevance grader node."""
from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, Field

_log = logging.getLogger("opsrag.agent.grader")

# If the cross-encoder's best (un-boosted) score is at least this, we trust the
# reranker's top hit even when the binary relevance grader rejects everything --
# rather than burn a CRAG rewrite on retrieval the reranker was confident about.
# Default-reranker scores are sigmoid(logit): 0.5 is the NEUTRAL midpoint
# (logit 0 = "no opinion"), so trusting at 0.5 defeated CRAG. Require a genuine
# positive lean (~logit +0.6). NOTE: this is calibrated for the FastEmbed
# default; Cohere/Vertex have different scales -- make it per-reranker if those
# become the default.
_TRUST_RERANK_SCORE = 0.65

# Max simultaneous per-chunk grade LLM calls. The grader fans out one call per
# candidate (up to ~50 after retrieval/merge); an unbounded asyncio.gather would
# fire all of them at once and trip provider 429 rate limits. Bound it like the
# other fan-outs in the codebase (ingestion/rerank use a Semaphore). State key
# `grader_concurrency` overrides it for tuning without a code change.
_GRADE_CONCURRENCY = 6

# Output token cap for the binary relevance gate. The structured payload is the
# one-field ``_GradeResult`` ({"relevant": true/false}) -- ~10-20 tokens including
# JSON punctuation -- so a 128-token ceiling lets the tiny answer schedule faster
# (shorter max_tokens reservation) WITHOUT any chance of truncating a valid
# verdict. Quality-neutral: the cap is HONORED on non-thinking providers
# (Anthropic/Bedrock-Claude/OpenAI), while Vertex and LiteLLM-Gemini IGNORE it
# (floor it at their default) to avoid truncating Gemini thinking tokens -- which
# count against max_output_tokens unless a response_schema is set, and the
# in-prompt-schema structured path sets none. So the boolean the grader returns
# is identical on every provider, with or without the cap.
_GATE_MAX_TOKENS = 128

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
            max_tokens=_GATE_MAX_TOKENS,
        )
        return result.relevant
    except Exception as exc:
        # Fail OPEN here (deliberately, unlike the groundedness gate): the grader
        # only *removes* recall, so dropping docs on a grader error is strictly
        # worse than keeping them -- a missing chunk yields a wrong/empty answer,
        # whereas an extra chunk is filtered downstream by rerank + the
        # fail-closed groundedness check. Log so silent grader outages are
        # visible (they'd otherwise look like "everything is relevant").
        _log.warning("grader errored for %s; failing OPEN (keeping doc): %s",
                     getattr(chunk, "source_path", "?"), exc)
        return True


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

        # Bound concurrency: one grade call per candidate can be up to ~50, and an
        # unbounded gather would issue them all at once -> 429 cascades. Gate each
        # coroutine behind a semaphore so at most N run in flight.
        cap = max(1, int(state.get("grader_concurrency") or _GRADE_CONCURRENCY))
        sem = asyncio.Semaphore(cap)

        async def _grade_bounded(c: Chunk) -> bool:
            async with sem:
                return await _grade_one(llm, query, c)

        verdicts = await asyncio.gather(
            *(_grade_bounded(c) for c in candidates)
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
            max_retries = state.get("max_retries", 3)
            # Keep the reranker's top hit when the cross-encoder was CONFIDENT
            # about it: the binary per-chunk grader runs AFTER rerank and can
            # only remove recall, so nuking a strong cross-encoder #1 to rewrite
            # the query wastes the budget on already-good retrieval. Fall back to
            # the rewrite loop only when the rerank was weak (genuinely-bad
            # retrieval, what CRAG is for) OR the retry budget is spent.
            best_rr = float(state.get("best_rerank_score", 0.0))
            # Per-reranker trust floor (set by the rerank node from the active
            # reranker's trust_score); falls back to the FastEmbed-calibrated
            # default. A hard-coded 0.65 was wrong for Cohere/Bedrock, whose
            # relevant scores cluster low -- it never fired, so CRAG always
            # rewrote even on good retrieval.
            trust_floor = float(state.get("rerank_trust_score", _TRUST_RERANK_SCORE))
            if best_rr >= trust_floor or retries >= max_retries:
                # Scale the floor for multi-fact / synthesis queries: the strict
                # binary grader over-prunes, and rescuing only min_relevant=1 can
                # ship a 1-chunk context for a query that needs several. Use the
                # distinct-entity (anchor) count and any decomposed sub-queries as
                # a proxy for how many groundings the answer needs, capped at the
                # generation budget (top_k, default 5).
                anchors = state.get("anchors") or []
                sub_queries = state.get("sub_queries") or []
                gen_budget = int(state.get("top_k") or 5)
                floor_n = min(
                    max(min_relevant, len(anchors), len(sub_queries)), gen_budget
                )
                kept = candidates[:floor_n]
                _log.info(
                    "grader floor (%s): kept top %d of %d candidates "
                    "(all failed strict grade; anchors=%d sub_queries=%d)",
                    "confident rerank" if best_rr >= trust_floor else "retries exhausted",
                    len(kept), len(candidates), len(anchors), len(sub_queries),
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
    max_retries = state.get("max_retries", 3)

    if graded:
        return "has_relevant"
    if retries >= max_retries:
        return "insufficient_info"
    return "needs_rewrite"
