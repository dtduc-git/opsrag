"""Investigation event type constants.

Each constant is a string the frontend filters on (`payload.type` in the
SSE event stream). Keep these stable -- the FE EventSource reducer maps
them to UI state changes; renaming breaks live updates.

Payload shape per type is documented inline. The UI is forgiving (unknown
types are no-ops), so adding new types is safe.
"""
from __future__ import annotations

from typing import Final


class EventType:
    # ---- Lifecycle ----
    # payload: {alert_text, incident_target?}
    INVESTIGATION_STARTED: Final = "investigation_started"
    # payload: {root_cause?, outcome?}  -- last event before stream closes
    INVESTIGATION_COMPLETED: Final = "investigation_completed"
    # payload: {error: str}
    INVESTIGATION_FAILED: Final = "investigation_failed"

    # ---- Lane probes (initial investigation, 3-lane fan-out) ----
    # payload: {} -- just signals the fan-out kicked off
    INITIAL_INVESTIGATION_STARTED: Final = "initial_investigation_started"
    # payload: {hits: [{title, source, snippet, url?}], elapsed_ms}
    LANE_A_COMPLETED: Final = "lane_a_completed"  # runbooks
    # payload: {hits: [{investigation_id, similarity, summary}], elapsed_ms}
    LANE_B_COMPLETED: Final = "lane_b_completed"  # historical investigations
    # payload: {summary, errors?: [str], elapsed_ms}
    LANE_C_COMPLETED: Final = "lane_c_completed"  # live MCP probe

    # ---- Insight synthesis (Pro fusion of A+B+C) ----
    # payload: {insight_card: {what_we_know, what_weve_seen, what_runbook_says,
    #          open_questions, seeded_hypotheses: [str], error?, elapsed_ms}}
    INSIGHT_READY: Final = "insight_ready"

    # ---- Hypothesis enumeration (Pro) ----
    # payload: {hypotheses: [{id, text, discriminating_tools: [str], status: 'open'}],
    #          incident_target?: str}
    HYPOTHESES_GENERATED: Final = "hypotheses_generated"

    # ---- Reasoner / tool loop ----
    # payload: {thinking_text: str, more_tools: bool, pending_tools: [str]}
    REASONER_STEP: Final = "reasoner_step"
    # payload: {name: str, args: dict}
    TOOL_CALLED: Final = "tool_called"
    # payload: {name: str, latency_ms, error?: str, summary?: str}
    TOOL_RESULT: Final = "tool_result"

    # ---- Per-hypothesis verdict (CORE FIX -- structured Pydantic, NOT text parse) ----
    # payload: {hypothesis_id, status: 'confirmed'|'ruled_out'|'untested'|'open',
    #          evidence: str, confidence: float}
    HYPOTHESIS_EVALUATED: Final = "hypothesis_evaluated"

    # ---- Generator (final prose answer for human; no longer a card-state source) ----
    # payload: {answer: str, elapsed_ms}
    CONCLUSION_READY: Final = "conclusion_ready"

    # ---- Critic (post-generator) ----
    # payload: {ok: bool, named_cause?: str, gaps?: [str]}
    CRITIC_VERDICT: Final = "critic_verdict"
