"""Regression: the CHAT multi-agent reasoner loop must have a per-turn
wall-clock breaker, not just a hop-count cap.

History (FINDING #9): the chat path bounded the tool loop only by
`MAX_TOOL_CALLS` (the number of hops) -- nothing capped how LONG a turn
could run. Ten back-to-back Pro calls could burn unbounded latency/cost
on a single turn. The investigation lane has budget.py (wall-clock +
token breakers); the chat lane had none.

Fix: `MAX_TURN_WALL_CLOCK_SEC` module constant + `turn_started_at`
(seeded at triage, time.monotonic) checked at the top of every reasoner
hop. On breach the reasoner stops looping and hands off to the generator
with whatever evidence exists -- clean termination, never a crash, and
WITHOUT burning another LLM call.
"""
from __future__ import annotations

import time

import pytest

from opsrag.agent.nodes.multi_agent import MAX_TURN_WALL_CLOCK_SEC, reasoner_node
from opsrag.llms.vertex import ToolCall, ToolCallingResponse


class _AlwaysMoreToolsLLM:
    """Stub LLM that ALWAYS asks for another tool call. Without a breaker
    a loop driven by this would run until MAX_TOOL_CALLS hops. Tracks
    invocation count so the test can prove the breaker short-circuits
    BEFORE any LLM call."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate_with_tools(self, **_kwargs) -> ToolCallingResponse:
        self.calls += 1
        return ToolCallingResponse(
            tool_calls=[ToolCall(name="k8s_list_pods", args={})],
            text="",
            model="stub",
        )

    # No `generate_with_tools_stream` -> reasoner takes the non-stream path.


@pytest.mark.asyncio
async def test_reasoner_wall_clock_breaker_terminates_before_hop_cap():
    """A turn that has already exceeded MAX_TURN_WALL_CLOCK_SEC stops the
    loop at the breaker -- hands off to the generator (no tool_calls),
    flags wall_clock_exceeded, and does NOT burn another LLM call (so it
    cannot run all 10 hops)."""
    llm = _AlwaysMoreToolsLLM()
    reason = reasoner_node(llm, observability=None, model_router=None)

    # Turn started well in the past -> breaker should trip on this hop.
    # Only 1 hop spent, so the MAX_TOOL_CALLS hop cap is NOT the thing
    # that stops it -- the wall-clock cap is.
    state = {
        "tool_message_history": [
            {"role": "user", "content": "investigate the SSO outage"},
        ],
        "tool_call_count": 1,
        "turn_started_at": time.monotonic() - (MAX_TURN_WALL_CLOCK_SEC + 5.0),
    }

    out = await reason(state)

    # Clean hand-off to generator: no further tool calls.
    assert out["tool_calls"] == []
    # Breaker fired (not the LLM "declined more tools" path).
    assert out["agent_event"]["metadata"].get("wall_clock_exceeded") is True
    # Crucially: the breaker short-circuited BEFORE invoking the LLM, so a
    # runaway turn can't keep paying for Pro calls.
    assert llm.calls == 0
    # Evidence preserved for the generator.
    assert out["tool_message_history"] == state["tool_message_history"]


@pytest.mark.asyncio
async def test_reasoner_within_budget_still_calls_llm():
    """A normal turn (well under the wall-clock cap) is unaffected -- the
    breaker does NOT fire and the reasoner proceeds to call the LLM."""
    llm = _AlwaysMoreToolsLLM()
    reason = reasoner_node(llm, observability=None, model_router=None)

    state = {
        "tool_message_history": [
            {"role": "user", "content": "what pods are running in prod?"},
        ],
        "tool_call_count": 1,
        "turn_started_at": time.monotonic(),  # just started
    }

    out = await reason(state)

    # Breaker did NOT fire -> LLM was consulted and asked for more tools.
    assert llm.calls == 1
    assert out["tool_calls"], "reasoner should have proceeded to call a tool"
    assert out["agent_event"]["metadata"].get("wall_clock_exceeded") is None


@pytest.mark.asyncio
async def test_reasoner_missing_turn_start_does_not_trip_breaker():
    """Defensive: if `turn_started_at` is absent (reasoner invoked in
    isolation / triage bypassed), the breaker must NOT fire spuriously --
    the reasoner proceeds normally."""
    llm = _AlwaysMoreToolsLLM()
    reason = reasoner_node(llm, observability=None, model_router=None)

    state = {
        "tool_message_history": [
            {"role": "user", "content": "list namespaces"},
        ],
        "tool_call_count": 1,
        # no turn_started_at
    }

    out = await reason(state)

    assert llm.calls == 1
    assert out["agent_event"]["metadata"].get("wall_clock_exceeded") is None
