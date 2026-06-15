"""Task 10: dispatcher fetch-after-permission + image caps + bare-image.

Covers:
  * FR-007 -- images are fetched ONLY after the permission check passes
    (a denied message never calls ``adapter.fetch_image``).
  * FR-009 -- the count cap clamps before fetching, and oversize fetched
    bytes are skipped, both read from ``VisionConfig``.
  * FR-006 -- a bare image (no caption) is rewritten to a default analyze
    prompt so the agent still runs.
  * FR-014 -- a fetch that raises degrades to text-only (never crashes).

The agent is stubbed by monkeypatching
``opsrag.channels.dispatcher.query_with_session_events`` with an async
generator -- no real graph, no network.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

import opsrag.channels.dispatcher as dispatcher_mod
from opsrag.channels.adapters.fake import FakeAdapter
from opsrag.channels.dispatcher import ChannelDispatcher
from opsrag.channels.permission import ChannelPermission
from opsrag.channels.types import ImageRef, InboundMessage
from opsrag.config import VisionConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass
class _ImageAdapter(FakeAdapter):
    """FakeAdapter that records fetch_image calls and returns scripted bytes.

    ``payloads`` maps an ImageRef.file_id -> bytes to return; missing ids
    return ``default_payload``. ``raise_on`` is a set of file_ids whose fetch
    raises (FR-014 path).
    """

    payloads: dict[str, bytes] = field(default_factory=dict)
    default_payload: bytes = b"IMG"
    raise_on: set[str] = field(default_factory=set)
    fetched: list[ImageRef] = field(default_factory=list)

    async def fetch_image(self, ref: ImageRef) -> bytes | None:
        self.fetched.append(ref)
        if ref.file_id in self.raise_on:
            raise RuntimeError("boom")
        return self.payloads.get(ref.file_id, self.default_payload)


def _stub_final(monkeypatch, *, captured: dict) -> None:
    async def fake_query(graph, **kwargs):  # noqa: ANN001
        captured.update(kwargs)
        yield {"type": "final", "answer": "ok"}

    monkeypatch.setattr(dispatcher_mod, "query_with_session_events", fake_query)


def _make_dispatcher(
    adapter: FakeAdapter,
    *,
    permission: ChannelPermission | None = None,
    vision: VisionConfig | None = None,
    providers: Any = None,
) -> ChannelDispatcher:
    return ChannelDispatcher(
        adapter=adapter,
        agent_graph=object(),
        providers=providers if providers is not None else object(),
        permission=permission or ChannelPermission(allowed_channels={"C123"}),
        vision=vision,
    )


def _msg(*, text: str = "look", images: tuple[ImageRef, ...] = ()) -> InboundMessage:
    return InboundMessage(
        channel_id="C123", user_id="U1", text=text, message_id="m1",
        thread_id=None, is_dm=False, workspace="W1", images=images,
    )


# ---------------------------------------------------------------------------
# FR-007: no fetch on denial
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_denied_message_never_fetches_image(monkeypatch) -> None:
    captured: dict = {}
    _stub_final(monkeypatch, captured=captured)
    adapter = _ImageAdapter()
    # Channel not allowlisted => deny BEFORE any fetch.
    disp = _make_dispatcher(
        adapter, permission=ChannelPermission(allowed_channels={"C999"}),
    )

    await disp.on_message(
        _msg(images=(ImageRef(file_id="F", mime_type="image/png"),)),
    )

    assert adapter.fetched == []        # FR-007: no fetch on denial
    assert captured == {}               # agent never ran
    assert adapter.posted == []


# ---------------------------------------------------------------------------
# FR-009: count cap clamps before fetch
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_count_cap_clamps_before_fetch(monkeypatch) -> None:
    captured: dict = {}
    _stub_final(monkeypatch, captured=captured)
    adapter = _ImageAdapter()
    disp = _make_dispatcher(adapter, vision=VisionConfig(max_images=2))

    refs = tuple(ImageRef(file_id=f"F{i}", mime_type="image/png") for i in range(5))
    await disp.on_message(_msg(images=refs))

    # Only the first 2 were ever fetched (clamp happens pre-fetch).
    assert [r.file_id for r in adapter.fetched] == ["F0", "F1"]
    assert len(captured["images"]) == 2


# ---------------------------------------------------------------------------
# FR-009: oversize fetched bytes are skipped
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_oversize_image_is_skipped(monkeypatch) -> None:
    captured: dict = {}
    _stub_final(monkeypatch, captured=captured)
    adapter = _ImageAdapter(
        payloads={"small": b"ok", "big": b"X" * 100},
    )
    disp = _make_dispatcher(adapter, vision=VisionConfig(max_bytes=10))

    refs = (
        ImageRef(file_id="small", mime_type="image/png"),
        ImageRef(file_id="big", mime_type="image/png"),
    )
    await disp.on_message(_msg(images=refs))

    # Both fetched, but the oversize one is dropped from the agent payload.
    assert {r.file_id for r in adapter.fetched} == {"small", "big"}
    assert [p.name for p in captured["images"]] == ["small"]


# ---------------------------------------------------------------------------
# FR-014: a fetch that raises degrades to text-only (no crash)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_failed_fetch_degrades_to_text_only(monkeypatch) -> None:
    captured: dict = {}
    _stub_final(monkeypatch, captured=captured)
    adapter = _ImageAdapter(raise_on={"bad"}, payloads={"good": b"ok"})
    disp = _make_dispatcher(adapter)

    refs = (
        ImageRef(file_id="bad", mime_type="image/png"),
        ImageRef(file_id="good", mime_type="image/png"),
    )
    await disp.on_message(_msg(text="explain", images=refs))

    # The failed ref is skipped; the good one survives; the turn still runs.
    assert [p.name for p in captured["images"]] == ["good"]
    assert captured["query"] == "explain"


# ---------------------------------------------------------------------------
# FR-006: a bare image (no caption) gets a default analyze prompt
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_bare_image_gets_default_prompt(monkeypatch) -> None:
    captured: dict = {}
    _stub_final(monkeypatch, captured=captured)
    adapter = _ImageAdapter()
    disp = _make_dispatcher(adapter)

    await disp.on_message(
        _msg(text="", images=(ImageRef(file_id="F", mime_type="image/png"),)),
    )

    assert captured["query"] == "Please analyze this image."
    assert len(captured["images"]) == 1


# ---------------------------------------------------------------------------
# FR-009: vision.enabled=False is a kill-switch -- no image is fetched
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_disabled_vision_never_fetches(monkeypatch) -> None:
    captured: dict = {}
    _stub_final(monkeypatch, captured=captured)
    adapter = _ImageAdapter()
    disp = _make_dispatcher(adapter, vision=VisionConfig(enabled=False))

    await disp.on_message(
        _msg(text="look", images=(ImageRef(file_id="F", mime_type="image/png"),)),
    )

    assert adapter.fetched == []                # kill-switch: no download
    assert captured["images"] == []            # nothing reaches the agent
    assert captured["query"] == "look"         # text turn still runs


# ---------------------------------------------------------------------------
# FR-010: a disallowed mime type is dropped BEFORE fetch (web-path parity)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_disallowed_mime_dropped_before_fetch(monkeypatch) -> None:
    captured: dict = {}
    _stub_final(monkeypatch, captured=captured)
    adapter = _ImageAdapter()
    disp = _make_dispatcher(adapter)        # default allow-list (no svg/pdf)

    refs = (
        ImageRef(file_id="ok", mime_type="image/png"),
        ImageRef(file_id="bad", mime_type="image/svg+xml"),
    )
    await disp.on_message(_msg(text="look", images=refs))

    # The svg ref is never fetched; only the png reaches the agent.
    assert [r.file_id for r in adapter.fetched] == ["ok"]
    assert [p.name for p in captured["images"]] == ["ok"]


# ---------------------------------------------------------------------------
# vision_llm is threaded from providers
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_vision_llm_passed_from_providers(monkeypatch) -> None:
    captured: dict = {}
    _stub_final(monkeypatch, captured=captured)
    adapter = _ImageAdapter()
    providers = type("P", (), {"vision_llm": "VISION"})()
    disp = _make_dispatcher(adapter, providers=providers)

    await disp.on_message(
        _msg(text="hi", images=(ImageRef(file_id="F", mime_type="image/png"),)),
    )

    assert captured["vision_llm"] == "VISION"
