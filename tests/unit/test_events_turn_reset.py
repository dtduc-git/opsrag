"""Regression: the streaming entrypoint must reset per-turn tool state.

`query_with_session_events` (used by BOTH the web /query stream AND every chat
channel bot) builds the per-turn `initial` graph state. LangGraph's checkpointer
persists state per `thread_id`, so any per-turn field NOT explicitly reset in
`initial` leaks forward across turns on the same thread.

The bug this guards against: `tool_call_count` (and the other tool-loop scratch
fields) were omitted from the events-path reset, so on a long-lived chat thread
(e.g. a Telegram DM where the user asks many questions) the count accumulated
until every new turn hit the loop cap (10) on its FIRST tool call -> the agent
could call no tools -> ungrounded/hallucinated answers. The non-streaming
`query_with_session` always reset them; the events path must match.
"""
from __future__ import annotations

import pytest

from opsrag.agent.graph import query_with_session, query_with_session_events

# The per-turn tool/scratch fields that MUST be reset on every turn (else the
# LangGraph checkpointer leaks them across turns on a thread_id). Both graph
# entrypoints must reset all of these identically.
_PER_TURN_TOOL_FIELDS = (
    "tool_call_count",
    "tool_message_history",
    "tool_path_active",
    "tool_call_audit",
    "model_route_decision",
    "past_investigations",
)


class _FakeEmbedder:
    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class _CapturingGraph:
    """Captures the `initial` state handed to astream_events, then ends cleanly."""

    def __init__(self) -> None:
        self.captured_initial: dict | None = None

    async def astream_events(self, initial, config=None, version=None):  # noqa: ANN001
        self.captured_initial = initial
        # Emit the outermost graph-end event carrying a minimal final state
        # (must contain "query" so the entrypoint recognises it as the result).
        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "data": {
                "output": {
                    "query": initial.get("query"),
                    "generation": "ok",
                    "generation_grounded": True,
                    "final_chunks": [],
                    "sources_searched": [],
                }
            },
        }


@pytest.mark.asyncio
async def test_events_path_resets_per_turn_tool_state() -> None:
    graph = _CapturingGraph()
    async for _ in query_with_session_events(
        graph,
        query="what is the acme-notes-be service?",
        user_id="telegram-bot:555:1158782396",
        thread_id="telegram-dm:555",
        embedder=_FakeEmbedder(),
        qa_cache=None,
        llm=None,
        session_store=None,
        semantic_router=None,
    ):
        pass

    init = graph.captured_initial
    assert init is not None, "graph was never invoked"
    # The per-turn tool budget + scratch MUST be reset every turn; otherwise the
    # checkpointer leaks them across turns on the same thread_id.
    assert init["tool_call_count"] == 0
    assert init["tool_message_history"] == []
    assert init["tool_path_active"] is False
    assert init["tool_call_audit"] == []
    assert init["model_route_decision"] == {}
    assert init["past_investigations"] == []


class _CapturingInvokeGraph:
    """Captures the `initial` state handed to ainvoke (non-streaming path)."""

    def __init__(self) -> None:
        self.captured_initial: dict | None = None

    async def ainvoke(self, initial, config=None) -> dict:  # noqa: ANN001
        self.captured_initial = initial
        return {
            "query": initial.get("query"),
            "generation": "ok",
            "generation_grounded": True,
            "final_chunks": [],
            "current_step": "done",
        }


@pytest.mark.asyncio
async def test_both_entrypoints_reset_the_same_per_turn_state() -> None:
    """Drift guard: the streaming (channels + web-stream) and non-streaming
    (web) entrypoints must reset the SAME per-turn tool fields. The original bug
    was exactly this drift -- `query_with_session` reset them, the events path
    didn't -- so every chat channel (Slack/Telegram/Discord/Teams, which all go
    through `query_with_session_events`) leaked `tool_call_count` across turns.
    """
    common = dict(
        query="what is the acme-notes-be service?",
        user_id="telegram-bot:555:1158782396",
        thread_id="dm:555",
        embedder=_FakeEmbedder(),
        qa_cache=None,
        llm=None,
        session_store=None,
        semantic_router=None,
    )

    events_graph = _CapturingGraph()
    async for _ in query_with_session_events(events_graph, **common):
        pass

    invoke_graph = _CapturingInvokeGraph()
    await query_with_session(compiled_graph=invoke_graph, **common)

    ev_init = events_graph.captured_initial
    ns_init = invoke_graph.captured_initial
    assert ev_init is not None and ns_init is not None

    for field in _PER_TURN_TOOL_FIELDS:
        assert field in ev_init, f"events path missing per-turn reset of {field!r}"
        assert field in ns_init, f"non-stream path missing per-turn reset of {field!r}"
        assert ev_init[field] == ns_init[field], (
            f"entrypoints drifted on per-turn field {field!r}: "
            f"events={ev_init[field]!r} non-stream={ns_init[field]!r}"
        )
    # And the budget specifically must start at zero on both.
    assert ev_init["tool_call_count"] == 0
    assert ns_init["tool_call_count"] == 0
