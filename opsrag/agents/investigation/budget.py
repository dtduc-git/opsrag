"""Circuit breakers, duplicate-ancestor detection, helpers.

`check_budget()` runs before every new node is spawned or tested. It
mutates `BudgetState.circuit_breakers_hit` and raises `BudgetExceeded`
when any hard limit is tripped -- callers translate that into a
graceful termination (mark remaining pending nodes inconclusive +
synthesize).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from opsrag.agents.investigation.limits import (
    DUPLICATE_ANCESTOR_COSINE_THRESHOLD,
    DUPLICATE_SIBLING_COSINE_THRESHOLD,
    MAX_INVESTIGATION_DURATION_SEC,
    MAX_LLM_TOKENS_PER_INVESTIGATION,
    MAX_TOTAL_NODES,
    MAX_TOTAL_TOOL_CALLS,
)
from opsrag.agents.investigation.state import BudgetState, InvestigationState


class BudgetExceeded(Exception):
    """Raised when a hard circuit breaker is tripped.

    The `breaker` attribute is one of:
      circuit_breaker_max_nodes |
      circuit_breaker_max_tool_calls |
      circuit_breaker_max_duration |
      circuit_breaker_max_tokens
    Callers should catch, record on state, and route to synthesis.
    """

    def __init__(self, breaker: str, detail: str = ""):
        super().__init__(f"{breaker}: {detail}" if detail else breaker)
        self.breaker = breaker
        self.detail = detail


def check_budget(budget: BudgetState) -> None:
    """Verify every hard limit. Mutate `circuit_breakers_hit` and raise
    BudgetExceeded on the first trip.

    Called before each new node spawn and before each tool call so a
    runaway loop can't burn past the cap.
    """
    if budget.total_nodes >= MAX_TOTAL_NODES:
        _trip(budget, "circuit_breaker_max_nodes",
              f"{budget.total_nodes} >= {MAX_TOTAL_NODES}")
    if budget.total_tool_calls >= MAX_TOTAL_TOOL_CALLS:
        _trip(budget, "circuit_breaker_max_tool_calls",
              f"{budget.total_tool_calls} >= {MAX_TOTAL_TOOL_CALLS}")
    if budget.total_llm_tokens >= MAX_LLM_TOKENS_PER_INVESTIGATION:
        _trip(budget, "circuit_breaker_max_tokens",
              f"{budget.total_llm_tokens} >= {MAX_LLM_TOKENS_PER_INVESTIGATION}")
    elapsed = budget.elapsed_seconds()
    if elapsed >= MAX_INVESTIGATION_DURATION_SEC:
        _trip(budget, "circuit_breaker_max_duration",
              f"{elapsed:.1f}s >= {MAX_INVESTIGATION_DURATION_SEC}s")


def _trip(budget: BudgetState, breaker: str, detail: str) -> None:
    if breaker not in budget.circuit_breakers_hit:
        budget.circuit_breakers_hit.append(breaker)
    raise BudgetExceeded(breaker, detail)


def record_tool_call(
    budget: BudgetState,
    purpose: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """Bookkeeping helper -- increments the right per-purpose counter so
    the dashboard can split retrieval vs LLM cost."""
    # Dedup embeddings are cheap and bounded by the node cap. Count them
    # separately and do NOT charge them to total_tool_calls -- otherwise a
    # handful of per-hypothesis embeds inflate the 300-call breaker and
    # pollute `retrieval_calls`, making the per-purpose dashboard lie.
    if purpose == "embed_dedup":
        budget.embed_dedup_calls += 1
        return
    budget.total_tool_calls += 1
    if purpose == "retrieval":
        budget.retrieval_calls += 1
    elif purpose == "llm_query_gen":
        budget.llm_query_gen_calls += 1
    elif purpose == "llm_judge":
        budget.llm_judge_calls += 1
    elif purpose == "llm_synth":
        budget.llm_synth_calls += 1
    elif purpose == "tool_dispatch":
        budget.tool_dispatch_calls += 1
    elif purpose == "llm_tool_select":
        budget.llm_tool_select_calls += 1
    budget.input_tokens += input_tokens
    budget.output_tokens += output_tokens
    budget.total_llm_tokens += input_tokens + output_tokens


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; returns 0.0 when either vector is empty or zero-norm."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def is_duplicate_ancestor(
    state: InvestigationState,
    candidate_id: str,
    candidate_embedding: Sequence[float],
    threshold: float = DUPLICATE_ANCESTOR_COSINE_THRESHOLD,
) -> tuple[bool, str | None, float]:
    """True if `candidate_embedding` cosines above `threshold` with any
    ancestor on the path-to-root.

    Returns `(is_duplicate, matched_ancestor_id, best_score)`. Used to
    block the LLM-rephrase loop where a child node restates its parent
    under a slightly different label and the agent recurses forever.
    """
    ancestors = state.ancestors(candidate_id)
    best_score = 0.0
    best_id: str | None = None
    for anc in ancestors:
        anc_emb = state.statement_embeddings.get(anc.id)
        if not anc_emb:
            continue
        score = cosine(candidate_embedding, anc_emb)
        if score > best_score:
            best_score = score
            best_id = anc.id
    return best_score >= threshold, best_id, best_score


def is_duplicate_sibling(
    state: InvestigationState,
    candidate_embedding: Sequence[float],
    parent_id: str | None,
    threshold: float = DUPLICATE_SIBLING_COSINE_THRESHOLD,
) -> tuple[bool, str | None, float]:
    """True if `candidate_embedding` cosines above `threshold` with any
    already-added sibling (i.e. any existing child of `parent_id`).

    Mirrors the shape of `is_duplicate_ancestor` -- returns
    `(is_duplicate, matched_sibling_id, best_score)`. The caller is
    responsible for ordering: this function only inspects siblings that
    are ALREADY in `parent.children`, so it must be called after each
    accepted sibling is committed and before the next candidate is
    embedded.

    Special cases:
    - `parent_id is None` (root-level siblings) -> walks `state.root_ids`
      so the same dedup applies to the initial hypothesis batch.
    - Sibling has no embedding registered -> silently skipped (we can't
      compare without a vector; better to keep the candidate than drop
      it on a missing-embedding edge case).
    """
    if parent_id is None:
        sibling_ids: list[str] = list(state.root_ids)
    else:
        parent = state.nodes_by_id.get(parent_id)
        sibling_ids = list(parent.children) if parent is not None else []

    best_score = 0.0
    best_id: str | None = None
    for sib_id in sibling_ids:
        sib_emb = state.statement_embeddings.get(sib_id)
        if not sib_emb:
            continue
        score = cosine(candidate_embedding, sib_emb)
        if score > best_score:
            best_score = score
            best_id = sib_id
    return best_score >= threshold, best_id, best_score
