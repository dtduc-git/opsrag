"""LangGraph wiring for the hypothesis-driven investigation subgraph.

Phases (linear; the DFS happens inside `decide_next`'s loop back to
`test_hypothesis`):

    START
      |
      v
    bootstrap_context          <- runbook + past-incident lookups
      |
      v
    generate_hypotheses        <- 3-5 diverse root hypotheses
      |
      v
    test_hypothesis  <---------+
      |                        |
      v                        |
    decide_next  --------------+   loop until tree exhausted or budget hit
      | pending? -> test_hypothesis
      | recurse? -> generate_sub_hypotheses -> test_hypothesis
      | done?    -> synthesize_root_cause
      |
      v
    synthesize_root_cause
      |
      v
    END

The tree itself is stored flat in `InvestigationState.nodes_by_id`.
`current_node_id` is the DFS cursor -- `decide_next` advances it to the
next pending node, applying circuit breakers + duplicate-ancestor
checks along the way.

Datadog Bits AI SRE quotes we honor here:
- "breaks down complex hypotheses into sub-hypotheses ... If a
  sub-hypothesis is supported by evidence, the agent digs deeper. If
  not, it looks elsewhere."
- "focuses on the causal relationship between the monitor alert and
  specific telemetry data pertaining to a hypothesis, rather than
  looking at all of the available telemetry data at once."
- "Each hypothesis is classified as validated, invalidated, or
  inconclusive."
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from opsrag.agents.investigation.budget import (
    BudgetExceeded,
    check_budget,
    is_duplicate_ancestor,
    is_duplicate_sibling,
    record_tool_call,
)
from opsrag.agents.investigation.limits import (
    BOOTSTRAP_TOP_K,
    EVIDENCE_TOP_K,
    INCONCLUSIVE_CONFIDENCE_CEILING,
    INVALIDATED_CONFIDENCE_FLOOR,
    MAX_DEPTH,
    fanout_for_depth,
    threshold_for_depth,
)
from opsrag.agents.investigation.observability import (
    emit_circuit_breaker,
    emit_investigation_summary,
    emit_node_decision,
)
from opsrag.agents.investigation.prompts import (
    EVIDENCE_JUDGE_PROMPT,
    HYPOTHESIS_GEN_PROMPT,
    ROOT_CAUSE_SYNTH_PROMPT,
    SUB_HYPOTHESIS_GEN_PROMPT,
)
from opsrag.agents.investigation.state import (
    Citation,
    HypothesisNode,
    InvestigationState,
    TraceEvent,
)

_log = logging.getLogger("opsrag.agents.investigation.graph")

# Retriever signature contract: takes a query string + top_k, returns
# a list of (chunk_id, source_id, snippet, score, repo) dicts. We
# define a callable type alias so callers can plug in any retriever
# (existing Confluence/Slack/Rootly/Git pipeline) without us needing
# to import vector store directly.
RetrieveFn = Callable[[str, int], Awaitable[list[dict[str, Any]]]]


# --- T1.1 -- structured-output schemas ------------------------------
# These get forwarded to Vertex Gemini's `response_schema` so the model
# is forced to emit valid JSON matching this shape. Thinking-tokens no
# longer compete with the visible-output budget -- truncated JSON
# becomes categorically impossible.
#
# Vertex's response_schema doesn't accept a top-level array, so the
# hypothesis list is wrapped in a `HypothesisList` envelope. The verdict
# is already an object, so it maps cleanly.


class HypothesisCandidate(BaseModel):
    statement: str
    rationale: str


class HypothesisList(BaseModel):
    hypotheses: list[HypothesisCandidate]


class VerdictResponse(BaseModel):
    status: Literal["validated", "invalidated", "inconclusive"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    supporting_chunk_ids: list[str] = []
    refuting_chunk_ids: list[str] = []


# --- 1. bootstrap_context -------------------------------------------


def bootstrap_node(
    retrieve: RetrieveFn,
    embed_query: Callable[[str], Awaitable[list[float]]] | None = None,
):
    """Fetch runbook + past-incident snippets relevant to the alert,
    BEFORE we generate any hypothesis. This anchors the LLM in real
    context (Step 2 of the hypothesis-driven flow: gather initial
    context from past investigations and runbooks)."""

    async def _run(state: InvestigationState) -> InvestigationState:
        ac = state.alert_context
        # Two queries -- one targeting runbooks, one targeting past
        # Rootly incidents. Both are short, focused; we never dump.
        runbook_query = f"runbook {ac.service_hint or ''} {ac.alert_text}".strip()
        history_query = f"past incident {ac.service_hint or ''} {ac.alert_text}".strip()

        findings: list[str] = []
        citations: list[Citation] = []

        for q in (runbook_query, history_query):
            try:
                check_budget(state.budget_state)
                hits = await retrieve(q, BOOTSTRAP_TOP_K)
            except BudgetExceeded as exc:
                emit_circuit_breaker(state, exc.breaker, exc.detail)
                break
            record_tool_call(state.budget_state, "retrieval")
            for h in hits:
                snippet = (h.get("snippet") or "")[:280]
                if snippet and snippet not in findings:
                    findings.append(snippet)
                citations.append(Citation(
                    source_id=str(h.get("source_id") or ""),
                    chunk_id=str(h.get("chunk_id") or ""),
                    snippet=snippet,
                    score=float(h.get("score") or 0.0),
                    repo=str(h.get("repo") or ""),
                ))

        state.bootstrap_findings = findings
        state.bootstrap_citations = citations
        state.agent_trace.append(TraceEvent(
            event_type="bootstrap",
            payload={
                "findings_count": len(findings),
                "citations_count": len(citations),
            },
        ))
        return state

    return _run


# --- 2. generate_hypotheses (root level) ----------------------------


def generate_hypotheses_node(
    llm,
    embed_query: Callable[[str], Awaitable[list[float]]] | None = None,
):
    """LLM emits 3-5 diverse root hypotheses. Each becomes a `pending`
    node attached to the (empty) tree root list."""

    async def _run(state: InvestigationState) -> InvestigationState:
        try:
            check_budget(state.budget_state)
        except BudgetExceeded as exc:
            emit_circuit_breaker(state, exc.breaker, exc.detail)
            return state

        max_hypotheses = fanout_for_depth(depth=0)
        # Runbook-sourced hypotheses may have already been attached by
        # the route handler before graph.astream() -- count them against
        # the depth-0 fanout cap so we don't overshoot.
        existing_root_count = len([n for n in state.nodes_by_id.values() if n.depth == 0])
        remaining = max(0, max_hypotheses - existing_root_count)
        if remaining == 0:
            state.agent_trace.append(TraceEvent(
                event_type="hypothesis_gen",
                payload={"depth": 0, "count": 0, "skipped": "fanout_full_from_runbook"},
            ))
            return state
        ac = state.alert_context
        bootstrap_block = (
            "\n".join(f"- {f}" for f in state.bootstrap_findings)
            or "(no runbook or past-incident snippets retrieved)"
        )
        past_block = _render_past_investigations(state.past_investigations)
        prompt = HYPOTHESIS_GEN_PROMPT.format(
            alert_text=ac.alert_text,
            service_hint=ac.service_hint or "(unknown)",
            namespace_hint=ac.namespace_hint or "(unknown)",
            env_hint=ac.env_hint or "(unknown)",
            bootstrap_findings=bootstrap_block,
            past_investigations=past_block,
            max_hypotheses=max_hypotheses,
        )

        statements = await _call_llm_for_hypotheses(state, llm, prompt, purpose="llm_query_gen")
        statements = statements[:remaining]

        for stmt in statements:
            node = HypothesisNode(statement=stmt, depth=0, parent_id=None)
            state.add_node(node)
            if embed_query is not None:
                try:
                    emb = await embed_query(stmt)
                    state.statement_embeddings[node.id] = emb
                    record_tool_call(state.budget_state, "retrieval")
                except Exception as exc:
                    _log.warning("embed_query failed for root hypothesis: %s", exc)

        state.agent_trace.append(TraceEvent(
            event_type="hypothesis_gen",
            payload={
                "depth": 0,
                "count": len(statements),
                "runbook_seeded": existing_root_count,
                "fanout_cap": max_hypotheses,
            },
        ))
        return state

    return _run


# --- 3. test_hypothesis ---------------------------------------------


def test_hypothesis_node(retrieve: RetrieveFn, llm):
    """Targeted retrieval + LLM judge for the current pending node."""

    async def _run(state: InvestigationState) -> InvestigationState:
        nid = state.current_node_id
        if nid is None:
            return state
        node = state.nodes_by_id.get(nid)
        if node is None or node.status != "pending":
            return state

        # Circuit-breaker check BEFORE the targeted retrieval.
        try:
            check_budget(state.budget_state)
        except BudgetExceeded as exc:
            emit_circuit_breaker(state, exc.breaker, exc.detail)
            node.status = "inconclusive"
            node.termination_reason = exc.breaker  # type: ignore[assignment]
            return state

        # Retrieval scoped to THIS hypothesis only -- no dump-all-at-once.
        try:
            hits = await retrieve(node.statement, EVIDENCE_TOP_K)
        except BudgetExceeded as exc:
            emit_circuit_breaker(state, exc.breaker, exc.detail)
            node.status = "inconclusive"
            node.termination_reason = exc.breaker  # type: ignore[assignment]
            return state
        record_tool_call(state.budget_state, "retrieval")

        evidence_block_lines: list[str] = []
        snippet_index: dict[str, Citation] = {}
        for h in hits:
            cid = str(h.get("chunk_id") or "")
            citation = Citation(
                source_id=str(h.get("source_id") or ""),
                chunk_id=cid,
                snippet=(h.get("snippet") or "")[:400],
                score=float(h.get("score") or 0.0),
                repo=str(h.get("repo") or ""),
            )
            snippet_index[cid] = citation
            evidence_block_lines.append(
                f"[chunk_id={cid} source={citation.source_id}]\n{citation.snippet}"
            )
        evidence_block = "\n\n".join(evidence_block_lines) or "(no snippets returned)"

        # LLM judge -- strict three-state.
        # Service-anchor: pass alert context so the judge can reject
        # evidence from unrelated services (a major source of drift --
        # earlier runs validated Helm-template-bug hypotheses for an
        # unrelated service outage using citations from completely
        # different tools' changelogs).
        ac = state.alert_context
        prompt = EVIDENCE_JUDGE_PROMPT.format(
            alert_text=ac.alert_text[:500],
            service_hint=ac.service_hint or "(unknown)",
            namespace_hint=ac.namespace_hint or "(unknown)",
            env_hint=ac.env_hint or "(unknown)",
            hypothesis_statement=node.statement,
            evidence_snippets=evidence_block,
        )
        verdict = await _call_llm_for_verdict(state, llm, prompt, purpose="llm_judge")
        _apply_verdict(
            node, verdict, snippet_index,
            alert_service=ac.service_hint, alert_namespace=ac.namespace_hint,
        )
        emit_node_decision(state, node)

        state.agent_trace.append(TraceEvent(
            event_type="judge",
            node_id=node.id,
            payload={
                "status": node.status,
                "confidence": node.confidence,
                "evidence_count": len(node.evidence),
            },
        ))
        return state

    return _run


# --- 4. decide_next (router) ----------------------------------------

DecisionLabel = Literal[
    "generate_sub_hypotheses",
    "test_hypothesis",
    "synthesize_root_cause",
]


def decide_next_node(
    embed_query: Callable[[str], Awaitable[list[float]]] | None = None,
):
    """Pure routing -- picks the next phase based on tree + budget state.

    Order of checks (matches the spec):
      1. Budget tripped -> synthesize
      2. Current node was validated AND can recurse -> generate_sub_hypotheses
      3. Otherwise advance DFS cursor to the next pending node; if none
         remain -> synthesize.
    """

    async def _run(state: InvestigationState) -> InvestigationState:
        # Budget first.
        try:
            check_budget(state.budget_state)
        except BudgetExceeded as exc:
            emit_circuit_breaker(state, exc.breaker, exc.detail)
            _mark_remaining_inconclusive(state, exc.breaker)
            return state  # router below sees outcome already set by caller

        # Where are we in the DFS?
        nid = state.current_node_id
        if nid is not None and nid in state.nodes_by_id:
            node = state.nodes_by_id[nid]
            if node.status == "validated":
                if node.depth + 1 > MAX_DEPTH:
                    node.termination_reason = "max_depth_reached"
                elif node.confidence < threshold_for_depth(node.depth):
                    node.termination_reason = "below_recurse_threshold"
                else:
                    # Recurse -- leave current_node_id pointing at the
                    # parent so generate_sub_hypotheses can attach.
                    return state

        # Advance cursor.
        next_id = state.next_pending_id()
        state.current_node_id = next_id
        return state

    return _run


def decide_next_router(state: InvestigationState) -> DecisionLabel:
    """Conditional edge -- read state shape and pick the next graph node.

    Kept separate from `decide_next_node` so LangGraph's edge resolver
    sees a pure function (no I/O, no embedding calls)."""
    if state.budget_state.circuit_breakers_hit and state.current_node_id is None:
        return "synthesize_root_cause"

    nid = state.current_node_id
    if nid and nid in state.nodes_by_id:
        node = state.nodes_by_id[nid]
        if (
            node.status == "validated"
            and node.depth + 1 <= MAX_DEPTH
            and node.confidence >= threshold_for_depth(node.depth)
        ):
            return "generate_sub_hypotheses"
        if node.status == "pending":
            return "test_hypothesis"
    return "synthesize_root_cause"


# --- 5. generate_sub_hypotheses -------------------------------------


def generate_sub_hypotheses_node(
    llm,
    embed_query: Callable[[str], Awaitable[list[float]]] | None = None,
):
    """Decompose the current validated node into narrower mechanisms."""

    async def _run(state: InvestigationState) -> InvestigationState:
        try:
            check_budget(state.budget_state)
        except BudgetExceeded as exc:
            emit_circuit_breaker(state, exc.breaker, exc.detail)
            return state

        parent_id = state.current_node_id
        parent = state.nodes_by_id.get(parent_id) if parent_id else None
        if parent is None:
            return state

        # Build the ancestor chain string for the prompt.
        ancestors = list(reversed(state.ancestors(parent.id))) + [parent]
        ancestor_chain = "\n".join(
            f"  depth={n.depth} status={n.status} -> {n.statement}"
            for n in ancestors
        ) or "(no ancestors -- parent is root)"

        evidence_summary = "\n".join(
            f"- [{c.chunk_id}] {c.snippet[:140]}" for c in parent.evidence[:3]
        ) or "(parent has no attached evidence; sub-hypotheses are speculative)"

        next_depth = parent.depth + 1
        # fanout_for_depth takes the depth of nodes being GENERATED -- the
        # children sit at next_depth, not at parent.depth.
        max_hypotheses = fanout_for_depth(next_depth)

        prompt = SUB_HYPOTHESIS_GEN_PROMPT.format(
            parent_statement=parent.statement,
            parent_evidence_summary=evidence_summary,
            ancestor_chain=ancestor_chain,
            max_hypotheses=max_hypotheses,
        )

        statements = await _call_llm_for_hypotheses(state, llm, prompt, purpose="llm_query_gen")
        statements = statements[:max_hypotheses]

        added = 0
        for stmt in statements:
            node = HypothesisNode(statement=stmt, depth=next_depth, parent_id=parent.id)
            # Dedup checks -- embed first, then run ancestor check, then
            # sibling check. Each check that fires wires the node into
            # parent.children for traceability but skips add_node so the
            # budget node count is untouched (these never ran).
            if embed_query is not None:
                try:
                    emb = await embed_query(stmt)
                    record_tool_call(state.budget_state, "retrieval")
                    # Pre-register the embedding so is_duplicate_ancestor
                    # can find ancestors by id (the new node is not yet
                    # added to the tree).
                    state.statement_embeddings[node.id] = emb
                    state.nodes_by_id[node.id] = node  # temp insert for ancestor walk
                    is_dup, anc_id, score = is_duplicate_ancestor(state, node.id, emb)
                    if is_dup:
                        node.status = "inconclusive"
                        node.termination_reason = "duplicate_ancestor"
                        node.judge_rationale = (
                            f"semantic duplicate of ancestor {anc_id} (cos={score:.2f})"
                        )
                        # Wire into the parent for traceability but DON'T
                        # add to budget node total -- it never ran. Leave
                        # the temp tree insert + embedding in place so
                        # state.nodes_by_id can be inspected for dups.
                        if node.id not in parent.children:
                            parent.children.append(node.id)
                        state.agent_trace.append(TraceEvent(
                            event_type="duplicate_ancestor",
                            node_id=node.id,
                            payload={"matched_ancestor": anc_id, "cosine": score},
                        ))
                        continue
                    # Ancestor check passed -- now sibling check against
                    # children that already landed in this same batch.
                    # NB: we pass parent.id, NOT node.id, so siblings are
                    # resolved by parent's children list (which the temp
                    # insert is NOT yet in -- so the candidate can't
                    # self-match).
                    sib_dup, sib_id, sib_score = is_duplicate_sibling(
                        state, emb, parent.id,
                    )
                    if sib_dup:
                        node.status = "inconclusive"
                        node.termination_reason = "duplicate_sibling"
                        node.judge_rationale = (
                            f"semantic duplicate of sibling {sib_id} (cos={sib_score:.2f})"
                        )
                        # Wire into parent.children for traceability --
                        # the temp insert in nodes_by_id stays so this
                        # rejected node is inspectable for dashboards
                        # and tests. Budget total_nodes is NOT touched.
                        if node.id not in parent.children:
                            parent.children.append(node.id)
                        state.agent_trace.append(TraceEvent(
                            event_type="duplicate_sibling",
                            node_id=node.id,
                            payload={"matched_sibling": sib_id, "cosine": sib_score},
                        ))
                        continue
                    # Not a dup -- finalize via add_node so children/
                    # root_ids/budget_state stay consistent. Remove the
                    # temp insert first so add_node's bookkeeping fires
                    # exactly once.
                    state.nodes_by_id.pop(node.id, None)
                except Exception as exc:
                    _log.warning("embed_query failed in sub-hypothesis gen: %s", exc)
                    # Defensive cleanup so a failed embed doesn't leave a
                    # stray temp insert in the tree.
                    state.nodes_by_id.pop(node.id, None)
                    state.statement_embeddings.pop(node.id, None)
            state.add_node(node)
            added += 1

        state.agent_trace.append(TraceEvent(
            event_type="hypothesis_gen",
            node_id=parent.id,
            payload={"depth": next_depth, "count": added},
        ))
        # Cursor advances naturally on the next decide_next pass.
        state.current_node_id = state.next_pending_id()
        return state

    return _run


# --- 6. synthesize_root_cause ---------------------------------------


def synthesize_root_cause_node(llm):
    """Pick the deepest validated chain -> LLM writes the final answer."""

    async def _run(state: InvestigationState) -> InvestigationState:
        chain = _deepest_validated_chain(state)
        state.final_chain_node_ids = [n.id for n in chain]

        if not chain:
            state.outcome = (
                "circuit_breaker_terminated"
                if state.budget_state.circuit_breakers_hit
                else "inconclusive"
            )
            state.final_root_cause = (
                "Investigation completed without a validated causal chain. "
                "Escalate to oncall with the alert context."
            )
            emit_investigation_summary(state)
            return state

        # Render chain + evidence for the prompt.
        chain_lines = "\n".join(
            f"  depth={n.depth} status={n.status} conf={n.confidence:.2f} -> {n.statement}"
            for n in chain
        )
        all_evidence = [c for n in chain for c in n.evidence]
        evidence_lines = "\n".join(
            f"- [{c.source_id}:{c.chunk_id}] {c.snippet[:200]}"
            for c in all_evidence
        ) or "(chain has no attached evidence -- inconclusive)"
        budget_summary = (
            f"duration={state.budget_state.elapsed_seconds():.1f}s "
            f"nodes={state.budget_state.total_nodes} "
            f"tool_calls={state.budget_state.total_tool_calls} "
            f"breakers={','.join(state.budget_state.circuit_breakers_hit) or 'none'}"
        )

        prompt = ROOT_CAUSE_SYNTH_PROMPT.format(
            alert_text=state.alert_context.alert_text,
            validated_chain=chain_lines,
            evidence_block=evidence_lines,
            budget_summary=budget_summary,
        )

        # When a circuit breaker has already tripped, we still need a
        # final answer for the user -- but we can't burn more budget on
        # an LLM call. Fall back to a deterministic chain summary.
        try:
            # Flash burns ~half this budget on thinking tokens before
            # emitting visible output, so a 1500-cap clips conclusions
            # mid-markdown (observed: "**2" cut at start of "Causal chain"
            # header). 6000 gives ~2.5k visible chars -- plenty for the
            # standard Root cause / Causal chain / Confidence / Next
            # action / Caveats sections.
            #
            # `force=True`: always run this terminal synth, even when a
            # duration/cost breaker has already tripped. Without this,
            # any overrun-by-1s investigation falls back to the raw
            # deterministic chain dump -- useless for the operator.
            resp = await _call_llm_raw(
                state, llm, prompt,
                purpose="llm_synth", max_tokens=6000, force=True,
            )
            state.final_root_cause = (resp or "").strip() or None
        except BudgetExceeded as exc:
            emit_circuit_breaker(state, exc.breaker, exc.detail)
            state.final_root_cause = _deterministic_summary(chain, all_evidence, budget_summary)
        except Exception as exc:  # noqa: BLE001 -- synth failures shouldn't kill the answer
            _log.warning("synthesis LLM failed (%s) -- using deterministic summary", exc)
            state.final_root_cause = _deterministic_summary(chain, all_evidence, budget_summary)

        if state.budget_state.circuit_breakers_hit:
            state.outcome = "circuit_breaker_terminated"
        else:
            state.outcome = "validated_root_cause"

        emit_investigation_summary(state)
        return state

    return _run


# --- helpers --------------------------------------------------------


def _is_evidence_on_topic(citation: Citation, service: str | None, namespace: str | None) -> bool:
    """Heuristic: is this citation plausibly about the alert's service?
    Used as a defensive cap on judge over-confidence. We trust the
    judge's verdict but downgrade `validated` to `inconclusive` when
    NO supporting citation mentions the alert's service/namespace and
    the citation isn't from a clearly generic SRE doc (runbook, sre-kb,
    confluence:SRE space).
    """
    if not service and not namespace:
        return True  # can't filter -- be permissive
    haystack = f"{citation.source_id} {citation.repo}".lower()
    if service and service.lower() in haystack:
        return True
    if namespace and namespace.lower() in haystack:
        return True
    # Generic SRE knowledge -- always allowed as supporting evidence.
    generic_markers = ("sre-knowledge-base", "confluence:sre",
                       "confluence:runbook", "runbook")
    if any(m in haystack for m in generic_markers):
        return True
    return False


def _apply_verdict(
    node: HypothesisNode,
    verdict: dict[str, Any],
    snippet_index: dict[str, Citation],
    alert_service: str | None = None,
    alert_namespace: str | None = None,
) -> None:
    status = verdict.get("status", "inconclusive")
    if status not in ("validated", "invalidated", "inconclusive"):
        status = "inconclusive"
    raw_conf = verdict.get("confidence")
    try:
        conf = float(raw_conf) if raw_conf is not None else 0.0
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    if status == "invalidated":
        conf = INVALIDATED_CONFIDENCE_FLOOR
    elif status == "inconclusive":
        conf = min(conf, INCONCLUSIVE_CONFIDENCE_CEILING)

    node.status = status  # type: ignore[assignment]
    node.confidence = conf
    node.judge_rationale = str(verdict.get("rationale") or "")[:280]
    supporting = verdict.get("supporting_chunk_ids") or []
    refuting = verdict.get("refuting_chunk_ids") or []
    cited_ids: list[str] = []
    for cid in supporting:
        cited_ids.append(str(cid))
    for cid in refuting:
        cited_ids.append(str(cid))
    for cid in cited_ids:
        c = snippet_index.get(cid)
        if c is not None and c not in node.evidence:
            node.evidence.append(c)

    # Defensive cap: if the judge said `validated` but NO supporting
    # citation mentions the alert's service/namespace AND none are
    # generic SRE docs, downgrade to `inconclusive` with low confidence.
    # Prevents the drift mode where each hypothesis is "validated" by
    # unrelated-service evidence (e.g. a third-party tool's changelog
    # justifying an unrelated service's alert investigation).
    if status == "validated" and (alert_service or alert_namespace):
        supporting_cites = [snippet_index.get(str(cid)) for cid in supporting]
        supporting_cites = [c for c in supporting_cites if c is not None]
        if supporting_cites and not any(
            _is_evidence_on_topic(c, alert_service, alert_namespace) for c in supporting_cites
        ):
            node.status = "inconclusive"  # type: ignore[assignment]
            node.confidence = min(node.confidence, 0.3)
            note = (
                f"[off-topic cap] all supporting evidence is from outside "
                f"`{alert_service or alert_namespace}` -- downgraded."
            )
            node.judge_rationale = (node.judge_rationale + " " + note)[:280]


def _deepest_validated_chain(state: InvestigationState) -> list[HypothesisNode]:
    """Return the root -> leaf chain whose terminal node has the
    maximum depth AND was validated. Tie-break by terminal confidence."""
    best: list[HypothesisNode] = []
    best_key: tuple[int, float] = (-1, -1.0)

    def _walk(node_id: str, path: list[HypothesisNode]) -> None:
        nonlocal best, best_key
        node = state.nodes_by_id.get(node_id)
        if node is None:
            return
        next_path = path + [node]
        if node.status == "validated":
            key = (node.depth, node.confidence)
            if key > best_key:
                best_key = key
                best = next_path
        for child in node.children:
            _walk(child, next_path)

    for rid in state.root_ids:
        _walk(rid, [])
    return best


def _render_past_investigations(items: list[dict]) -> str:
    """Format prior-investigation summaries for the hypothesis-gen prompt.

    Each entry comes pre-populated by the route handler with keys
    {alert_text, final_root_cause, similarity, age_days, tool_calls_used,
     outcome, validated_chain_summary}.
    """
    if not items:
        return "(no prior similar investigations found)"
    lines: list[str] = []
    for i, p in enumerate(items, 1):
        age = p.get("age_days")
        sim = p.get("similarity")
        rc = (p.get("final_root_cause") or "")[:280]
        tools = p.get("tool_calls_used") or []
        tools_short = ", ".join(tools[:5]) or "(none)"
        chain = p.get("validated_chain_summary") or []
        chain_short = " -> ".join(chain[:3]) or "(empty)"
        lines.append(
            f"{i}. [age={age}d, similarity={sim:.2f}] root cause: {rc}\n"
            f"   chain: {chain_short}\n"
            f"   tools: {tools_short}"
        )
    return "\n".join(lines)


def _deterministic_summary(
    chain: list[HypothesisNode],
    evidence: list[Citation],
    budget_summary: str,
) -> str:
    """Fallback used when both the regular synth AND the forced retry
    fail (e.g. total LLM provider outage). Produces a compact human-
    readable Markdown block -- never the raw chunk dump that earlier
    versions emitted.

    Today's path with `force=True` on the synth call makes this rare,
    but it's still the safety net.
    """
    if not chain:
        return (
            "## Investigation incomplete\n\n"
            "The investigation terminated before any hypothesis could be validated.\n\n"
            "**Budget consumed:** "
            f"{budget_summary}\n\n"
            "**Suggested next step:** Escalate to on-call; check the alert's runbook URL manually."
        )

    # Best guess: deepest validated node -- typically the most specific.
    best = chain[-1]
    parts: list[str] = []
    parts.append("## Likely root cause\n")
    parts.append(f"{best.statement}\n")
    if best.judge_rationale:
        parts.append(f"\n**Judge rationale:** {best.judge_rationale}\n")
    parts.append("\n_Note: LLM-synthesis path was unavailable; this is the deepest validated hypothesis as a best-effort summary._\n")

    parts.append("\n## Causal chain\n")
    for n in chain:
        parts.append(f"- **d{n.depth}** ({int(n.confidence * 100)}%) -- {n.statement}\n")

    if evidence:
        parts.append("\n## Top evidence\n")
        # Shorten source IDs aggressively -- the raw chunk-id chain is
        # noise in the conclusion view.
        seen: set[str] = set()
        for c in evidence:
            tag = (c.source_id or "").split(":", 2)
            short = ":".join(tag[:2]) if len(tag) > 1 else (c.source_id or "?")
            if short in seen:
                continue
            seen.add(short)
            parts.append(f"- `{short}` -- {(c.snippet or '').strip()[:240]}\n")
            if len(seen) >= 5:
                break

    parts.append(f"\n---\n_Budget: {budget_summary}_\n")
    return "".join(parts)


def _mark_remaining_inconclusive(state: InvestigationState, reason: str) -> None:
    for node in state.nodes_by_id.values():
        if node.status == "pending":
            node.status = "inconclusive"
            node.termination_reason = reason  # type: ignore[assignment]
    state.current_node_id = None


# -- LLM call wrappers -- single point that records token usage --

_JSON_BLOCK_RE = re.compile(r"\[.*\]|\{.*\}", re.DOTALL)


async def _call_llm_raw(
    state: InvestigationState,
    llm,
    prompt: str,
    *,
    purpose: str,
    max_tokens: int = 800,
    response_schema: dict | None = None,
    force: bool = False,
) -> str:
    """Single point for LLM calls -- budget check + token accounting.

    When `response_schema` is provided AND the underlying LLM is Vertex
    Gemini, structured-output mode is enforced (T1.1). Non-Gemini
    providers ignore the param and fall back to prompt-only JSON.

    `force=True` skips the budget check. Reserved for the terminal
    synthesis call so a duration-breaker termination still produces
    an LLM-rephrased conclusion (otherwise users get the raw
    deterministic chain dump).
    """
    if not force:
        check_budget(state.budget_state)
    # ScriptedLLM in tests doesn't accept response_schema; only forward
    # the kwarg when caller asked for it so the default code path stays
    # backwards-compatible.
    extra_kwargs: dict[str, Any] = {}
    if response_schema is not None:
        extra_kwargs["response_schema"] = response_schema
    resp = await llm.generate(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
        purpose=f"investigation.{purpose}",
        **extra_kwargs,
    )
    in_t = int((resp.usage or {}).get("input_tokens", 0))
    out_t = int((resp.usage or {}).get("output_tokens", 0))
    record_tool_call(
        state.budget_state, purpose,
        input_tokens=in_t, output_tokens=out_t,
    )
    return resp.content or ""


async def _call_llm_for_hypotheses(
    state: InvestigationState, llm, prompt: str, *, purpose: str,
) -> list[str]:
    """Returns a list of hypothesis statements parsed from the LLM JSON.

    T1.1: enforces Vertex Gemini structured output via response_schema.
    The schema wraps the array in `HypothesisList.hypotheses` because
    Vertex's response_schema doesn't accept a top-level array. With
    structured output, JSON parse failures are categorically impossible
    on Gemini; the regex/loose-parse fallback below stays in place for
    non-Gemini providers and for resilience against schema mismatches.

    Token budget note: with structured output, thinking-tokens no longer
    eat into the visible-output budget (Vertex tracks them separately).
    Dropped max_tokens from the paranoid 8000 to 3000 -- still plenty of
    headroom for 5 hypotheses x ~100 tokens each plus rationales.
    """
    raw = await _call_llm_raw(
        state, llm, prompt,
        purpose=purpose, max_tokens=3000,
        response_schema=HypothesisList.model_json_schema(),
    )
    parsed = _parse_hypothesis_list(raw)
    if not parsed:
        _log.warning(
            "hypothesis parse returned 0 statements; raw[:200]=%r",
            raw[:200] if raw else "(empty)",
        )
    return parsed


async def _call_llm_for_verdict(
    state: InvestigationState, llm, prompt: str, *, purpose: str,
) -> dict[str, Any]:
    # T1.1: structured-output enforcement. With response_schema set,
    # Gemini emits a valid VerdictResponse JSON deterministically.
    # Dropped max_tokens from 5000 to 2000 -- the verdict object is
    # small (~150 tokens of visible output) and thinking is now a
    # separate budget.
    raw = await _call_llm_raw(
        state, llm, prompt,
        purpose=purpose, max_tokens=2000,
        response_schema=VerdictResponse.model_json_schema(),
    )
    return _parse_verdict(raw)


def _parse_hypothesis_list(raw: str) -> list[str]:
    """Parse either the new T1.1 envelope shape (`{"hypotheses": [...]}`)
    or the legacy top-level array. Tolerant of code fences and
    surrounding prose, since structured-output isn't guaranteed when the
    underlying LLM is not Vertex Gemini."""
    if not raw:
        return []
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.MULTILINE).strip()
    statements: list[str] = []
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RE.search(s)
        if not m:
            return []
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    # New shape: {"hypotheses": [{"statement": ..., "rationale": ...}, ...]}
    if isinstance(parsed, dict) and isinstance(parsed.get("hypotheses"), list):
        parsed = parsed["hypotheses"]
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                stmt = item.get("statement") or item.get("hypothesis")
                if isinstance(stmt, str) and stmt.strip():
                    statements.append(stmt.strip())
            elif isinstance(item, str) and item.strip():
                statements.append(item.strip())
    return statements


def _parse_verdict(raw: str) -> dict[str, Any]:
    if not raw:
        return {"status": "inconclusive", "confidence": 0.0, "rationale": "empty LLM response"}
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RE.search(s)
        if not m:
            return {"status": "inconclusive", "confidence": 0.0, "rationale": "unparseable JSON"}
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"status": "inconclusive", "confidence": 0.0, "rationale": "unparseable JSON"}
    if not isinstance(parsed, dict):
        return {"status": "inconclusive", "confidence": 0.0, "rationale": "non-dict verdict"}
    return parsed


# --- graph factory --------------------------------------------------


def build_investigation_graph(
    *,
    retrieve: RetrieveFn,
    llm_flash,
    llm_pro=None,
    embed_query: Callable[[str], Awaitable[list[float]]] | None = None,
):
    """Compile the LangGraph subgraph.

    Args:
        retrieve: an async callable taking (query_text, top_k) and
            returning a list of dicts with keys
            {chunk_id, source_id, snippet, score, repo}. Adapter onto
            the existing OpsRAG retriever stack lives at the caller.
        llm_flash: any object implementing the project's `LLMProvider`
            protocol -- `.generate(messages, temperature, max_tokens,
            purpose) -> LLMResponse`. Used for hypothesis generation,
            sub-hypothesis generation, and evidence judging. Cost-
            optimal default since these are JSON-constrained outputs.
        llm_pro: optional Pro-tier LLM. When provided, used for the
            final root-cause synthesis (user-facing narrative + tighter
            citation discipline). Falls back to `llm_flash` if None.
        embed_query: async callable for ancestor-similarity check.
            Optional -- when None, duplicate-ancestor pruning is
            disabled (use only for tests that don't need it).

    Flash/Pro split rationale: hypothesis generation + evidence judge
    are short JSON tasks where Pro's extra capability is wasted; final
    synthesis is the one user-facing answer per investigation, so
    spending Pro budget on it is cheap and improves quality.
    """
    synth_llm = llm_pro if llm_pro is not None else llm_flash
    graph = StateGraph(InvestigationState)
    graph.add_node("bootstrap_context", bootstrap_node(retrieve, embed_query))
    graph.add_node("generate_hypotheses", generate_hypotheses_node(llm_flash, embed_query))
    graph.add_node("test_hypothesis", test_hypothesis_node(retrieve, llm_flash))
    graph.add_node("decide_next", decide_next_node(embed_query))
    graph.add_node(
        "generate_sub_hypotheses",
        generate_sub_hypotheses_node(llm_flash, embed_query),
    )
    graph.add_node("synthesize_root_cause", synthesize_root_cause_node(synth_llm))

    graph.add_edge(START, "bootstrap_context")
    graph.add_edge("bootstrap_context", "generate_hypotheses")
    graph.add_edge("generate_hypotheses", "decide_next")
    graph.add_edge("test_hypothesis", "decide_next")
    graph.add_edge("generate_sub_hypotheses", "decide_next")
    graph.add_conditional_edges(
        "decide_next",
        decide_next_router,
        {
            "test_hypothesis": "test_hypothesis",
            "generate_sub_hypotheses": "generate_sub_hypotheses",
            "synthesize_root_cause": "synthesize_root_cause",
        },
    )
    graph.add_edge("synthesize_root_cause", END)
    return graph.compile()
