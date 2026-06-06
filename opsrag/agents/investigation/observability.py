"""Structured logging + metric emission for the investigation agent.

Two surfaces:
  1. `emit_investigation_summary()` -- single JSON log line at the end of
     each run. Datadog log-to-metric pipelines pick this up to build
     the dashboards in `dashboards/investigation_metrics.md`.
  2. `emit_circuit_breaker()` -- fires when any hard limit trips. Tagged
     with `breaker:<name>` so an alert can fan to oncall.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from opsrag.agents.investigation.state import HypothesisNode, InvestigationState

_log = logging.getLogger("opsrag.agents.investigation")


def _tree_size_breakdown(state: InvestigationState) -> dict[str, int]:
    counts = {
        "total_nodes": len(state.nodes_by_id),
        "validated": 0,
        "invalidated": 0,
        "inconclusive": 0,
        "pending": 0,
    }
    for node in state.nodes_by_id.values():
        counts[node.status] = counts.get(node.status, 0) + 1
    return counts


def _max_depth_reached(state: InvestigationState) -> int:
    if not state.nodes_by_id:
        return 0
    return max(n.depth for n in state.nodes_by_id.values())


def build_investigation_summary(state: InvestigationState) -> dict[str, Any]:
    """Build the structured summary blob. Returned as a dict so callers
    can ship it to whichever sink (stdout JSON, Datadog log, Phoenix
    span attribute) without coupling to a specific transport."""
    b = state.budget_state
    return {
        "event": "opsrag.investigation.complete",
        "investigation_id": state.alert_context.investigation_id,
        "duration_sec": round(b.elapsed_seconds(), 2),
        "tree_size": _tree_size_breakdown(state),
        "max_depth_reached": _max_depth_reached(state),
        "tool_calls": {
            "total": b.total_tool_calls,
            "retrieval": b.retrieval_calls,
            "llm_query_gen": b.llm_query_gen_calls,
            "llm_judge": b.llm_judge_calls,
            "llm_synth": b.llm_synth_calls,
        },
        "tokens": {
            "input": b.input_tokens,
            "output": b.output_tokens,
            "total": b.total_llm_tokens,
        },
        "circuit_breakers_hit": list(b.circuit_breakers_hit),
        "outcome": state.outcome,
        "final_chain_length": len(state.final_chain_node_ids),
        "service": state.alert_context.service_hint,
        "env": state.alert_context.env_hint,
    }


def emit_investigation_summary(state: InvestigationState) -> dict[str, Any]:
    """Single JSON log line -- Datadog parses this with a JSON pipeline.

    Tagged at INFO. Callers also receive the summary dict so they can
    attach it to the agent_trace export.
    """
    summary = build_investigation_summary(state)
    _log.info("investigation.complete %s", json.dumps(summary, default=str))
    return summary


def emit_circuit_breaker(state: InvestigationState, breaker: str, detail: str = "") -> None:
    """One log line per breaker hit. Datadog `@breaker` facet pivots to
    a per-breaker count metric for the dashboard."""
    payload = {
        "event": "opsrag.investigation.circuit_breaker_hit",
        "investigation_id": state.alert_context.investigation_id,
        "breaker": breaker,
        "detail": detail,
        "elapsed_sec": round(state.budget_state.elapsed_seconds(), 2),
        "total_nodes": state.budget_state.total_nodes,
        "total_tool_calls": state.budget_state.total_tool_calls,
    }
    _log.warning("investigation.circuit_breaker %s", json.dumps(payload, default=str))


def emit_node_decision(state: InvestigationState, node: HypothesisNode) -> None:
    """Per-node trace event -- useful when debugging an investigation
    that produced a surprising tree. Logged at DEBUG so it can be
    silenced in prod by default."""
    payload = {
        "event": "opsrag.investigation.node_decision",
        "investigation_id": state.alert_context.investigation_id,
        "node_id": node.id,
        "depth": node.depth,
        "status": node.status,
        "confidence": round(node.confidence, 3),
        "evidence_count": len(node.evidence),
        "termination_reason": node.termination_reason,
    }
    _log.debug("investigation.node %s", json.dumps(payload, default=str))
