"""Ephemerality guarantee (spec SC-002 / FR-003).

Image bytes for a turn must ride in the LangGraph runnable ``config``
(``config["configurable"]["turn_images"]``), NEVER in the graph ``state`` /
``initial`` dict -- because LangGraph's Postgres checkpointer persists ``state``
per ``thread_id``, so any bytes placed there would survive the turn. The only
thing that may land in persisted state is a TEXT marker recording that an image
was attached (``"<query> [attached image: <names>]"``) -- no bytes.

These tests assert that split directly against both agent entry points by
injecting a minimal fake graph that captures the ``initial`` and ``config`` it
is handed.
"""
from __future__ import annotations

import asyncio

import pytest

from opsrag.agent.graph import query_with_session, query_with_session_events
from opsrag.llms.content import ImagePart

PNG = b"\x89PNG\r\nfake"


class _FakeStreamGraph:
    """Captures the config + initial state passed to astream_events."""

    def __init__(self) -> None:
        self.seen_config = None
        self.seen_initial = None

    async def astream_events(self, initial, config=None, version=None):  # noqa: ANN001
        self.seen_initial = initial
        self.seen_config = config
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


class _FakeInvokeGraph:
    """Captures the config + initial state passed to ainvoke."""

    def __init__(self) -> None:
        self.seen_config = None
        self.seen_initial = None

    async def ainvoke(self, initial, config=None) -> dict:  # noqa: ANN001
        self.seen_initial = initial
        self.seen_config = config
        return {
            "query": initial.get("query"),
            "generation": "ok",
            "generation_grounded": True,
            "final_chunks": [],
            "current_step": "done",
        }


def _assert_ephemeral(seen_initial: dict, seen_config: dict, vision_sentinel) -> None:
    # Ephemeral guarantee: bytes are NOT in the checkpointed state...
    assert "turn_images" not in (seen_initial or {})
    assert "vision_llm" not in (seen_initial or {})
    assert repr(PNG) not in repr(seen_initial)
    # ...they live in the runnable config instead.
    cfg = seen_config["configurable"]
    assert cfg["turn_images"][0].data == PNG
    assert cfg["vision_llm"] is vision_sentinel
    # The persisted query carries only a text marker, never bytes.
    assert "[attached image" in seen_initial["query"]
    assert "a.png" in seen_initial["query"]
    # Existing configurable keys preserved.
    assert cfg["thread_id"] == "web:u1:t1"
    assert cfg["user_id"] == "u1"


def test_events_path_images_ride_in_config_not_state() -> None:
    g = _FakeStreamGraph()
    sentinel = object()

    async def _run() -> None:
        async for _ in query_with_session_events(
            g,
            query="look",
            user_id="u1",
            thread_id="web:u1:t1",
            images=[ImagePart(PNG, "image/png", "a.png")],
            vision_llm=sentinel,
        ):
            pass

    asyncio.run(_run())
    _assert_ephemeral(g.seen_initial, g.seen_config, sentinel)


def test_invoke_path_images_ride_in_config_not_state() -> None:
    g = _FakeInvokeGraph()
    sentinel = object()

    asyncio.run(
        query_with_session(
            compiled_graph=g,
            query="look",
            user_id="u1",
            thread_id="web:u1:t1",
            images=[ImagePart(PNG, "image/png", "a.png")],
            vision_llm=sentinel,
        )
    )
    _assert_ephemeral(g.seen_initial, g.seen_config, sentinel)


@pytest.mark.parametrize("entry", ["events", "invoke"])
def test_no_images_keeps_plain_query_and_empty_turn_images(entry: str) -> None:
    """Text-only fast path is byte-identical: no marker, turn_images == []."""
    if entry == "events":
        g = _FakeStreamGraph()

        async def _run() -> None:
            async for _ in query_with_session_events(
                g, query="plain question", user_id="u1", thread_id="t1"
            ):
                pass

        asyncio.run(_run())
    else:
        g = _FakeInvokeGraph()
        asyncio.run(
            query_with_session(
                compiled_graph=g, query="plain question", user_id="u1", thread_id="t1"
            )
        )

    assert g.seen_initial["query"] == "plain question"
    assert "[attached image" not in g.seen_initial["query"]
    cfg = g.seen_config["configurable"]
    assert cfg["turn_images"] == []
    assert cfg["vision_llm"] is None
