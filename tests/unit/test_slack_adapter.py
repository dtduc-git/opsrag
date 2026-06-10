"""Unit tests for the Slack ChannelAdapter (P1 refactor -- WRAP, don't rewrite).

Covers design 8.4 for the Slack adapter, with NO network and NO real
``slack_sdk`` transport: a ``_FakeSlackClient`` records the leaf-module calls
the adapter delegates to. Assertions are on REAL behaviour:

  * inbound normalization: app_mention vs message.im -> InboundMessage
    (mention strip, is_dm, thread_id, workspace);
  * feedback parse: block_actions payload -> FeedbackEvent (+ malformed drop);
  * outbound: post_placeholder / edit / react (ACK/DONE/ERROR -> emoji) /
    finalize (renders Block Kit) / send_denial;
  * fetch_thread maps raw replies to ThreadMessage reusing the thread_context
    self-filter (our own past reply -> is_self=True);
  * identity oid format (slack-bot:<workspace>:<user>).

The adapter is constructed with a Slack channel config and its private
``_client`` is swapped for the fake (so ``connect`` -- which needs real tokens
+ Socket Mode -- is exercised separately via the event shim, below).
"""
from __future__ import annotations

import pytest

from opsrag.channels.adapters.slack.adapter import (
    SlackAdapter,
    _event_to_inbound,
    _payload_to_feedback,
    _SlackEventShim,
    _SlackHandle,
)
from opsrag.channels.config import SlackChannelConfig
from opsrag.channels.types import (
    AgentResult,
    FeedbackEvent,
    InboundMessage,
    ReactionKind,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeSlackClient:
    """Records the leaf-module calls the adapter delegates to (no network)."""

    def __init__(self) -> None:
        self.self_user_id = "UBOT"
        self.self_bot_id = "BBOT"
        self.posted: list[dict] = []
        self.updated: list[dict] = []
        self.reactions: list[tuple[str, str, str]] = []
        self.thread_replies: list[dict] = []
        self.user_info: dict[str, dict] = {}
        self._ts_seq = 0

    async def post_message(self, channel, text, thread_ts=None, blocks=None):
        self._ts_seq += 1
        ts = f"ts{self._ts_seq}"
        self.posted.append(
            {"channel": channel, "text": text, "thread_ts": thread_ts, "ts": ts},
        )
        return ts

    async def update_message(self, channel, ts, text, blocks=None):
        self.updated.append(
            {"channel": channel, "ts": ts, "text": text, "blocks": blocks},
        )

    async def add_reaction(self, channel, ts, emoji):
        self.reactions.append((channel, ts, emoji))

    async def fetch_thread_replies(self, channel, thread_ts, limit=30):
        return list(self.thread_replies)

    async def get_user_info(self, user_id):
        return self.user_info.get(user_id, {})


def _adapter_with_fake() -> tuple[SlackAdapter, _FakeSlackClient]:
    adapter = SlackAdapter(SlackChannelConfig(web_ui_base_url="https://opsrag.example.com"))
    fake = _FakeSlackClient()
    adapter._client = fake  # noqa: SLF001 -- inject the fake transport
    return adapter, fake


# ---------------------------------------------------------------------------
# Inbound normalization
# ---------------------------------------------------------------------------
def test_app_mention_strips_mention_and_sets_thread() -> None:
    event = {
        "channel": "C123",
        "user": "U1",
        "text": "<@UBOT> why is prod down?",
        "ts": "111.1",
        "thread_ts": "100.0",
        "team": "T9",
    }
    msg = _event_to_inbound(event, is_dm=False)
    assert msg.channel_id == "C123"
    assert msg.user_id == "U1"
    assert msg.text == "why is prod down?"  # mention stripped
    assert msg.message_id == "111.1"
    assert msg.thread_id == "100.0"
    assert msg.is_dm is False
    assert msg.workspace == "T9"


def test_message_im_is_dm_no_thread() -> None:
    event = {"channel": "D123", "user": "U1", "text": "hello", "ts": "200.0"}
    msg = _event_to_inbound(event, is_dm=True)
    assert msg.is_dm is True
    assert msg.thread_id is None
    assert msg.text == "hello"
    assert msg.workspace is None


# ---------------------------------------------------------------------------
# Feedback parse
# ---------------------------------------------------------------------------
def test_feedback_parse_up() -> None:
    payload = {
        "actions": [{"action_id": "opsrag_feedback_up", "value": "up:inv-42"}],
        "user": {"id": "U7"},
        "container": {"thread_ts": "100.0"},
        "response_url": "https://hooks.example.com/x",
    }
    fb = _payload_to_feedback(payload)
    assert fb is not None
    assert fb.thumbs == "up"
    assert fb.investigation_id == "inv-42"
    assert fb.user_id == "U7"
    assert fb.thread_id == "100.0"
    assert fb.raw["response_url"] == "https://hooks.example.com/x"


def test_feedback_parse_malformed_returns_none() -> None:
    # Not our action_id
    assert _payload_to_feedback(
        {"actions": [{"action_id": "other", "value": "up:x"}]},
    ) is None
    # No colon in value
    assert _payload_to_feedback(
        {"actions": [{"action_id": "opsrag_feedback_up", "value": "garbage"}]},
    ) is None
    # Empty actions
    assert _payload_to_feedback({"actions": []}) is None


# ---------------------------------------------------------------------------
# Outbound primitives
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_post_placeholder_and_edit() -> None:
    adapter, fake = _adapter_with_fake()
    handle = await adapter.post_placeholder("C123", "100.0", "thinking")
    assert isinstance(handle, _SlackHandle)
    assert fake.posted[0]["thread_ts"] == "100.0"
    await adapter.edit(handle, "still thinking")
    assert fake.updated[0]["channel"] == "C123"
    assert fake.updated[0]["ts"] == handle.ts
    assert fake.updated[0]["text"] == "still thinking"


@pytest.mark.asyncio
async def test_react_maps_kinds_to_emoji() -> None:
    adapter, fake = _adapter_with_fake()
    await adapter.react("C123", "111.1", ReactionKind.ACK)
    await adapter.react("C123", "111.1", ReactionKind.DONE)
    await adapter.react("C123", "111.1", ReactionKind.ERROR)
    emojis = [e for _, _, e in fake.reactions]
    assert emojis == ["eyes", "white_check_mark", "x"]


@pytest.mark.asyncio
async def test_finalize_renders_block_kit() -> None:
    adapter, fake = _adapter_with_fake()
    handle = await adapter.post_placeholder("C123", None, "thinking")
    result = AgentResult(
        answer="**Root cause**: bad deploy",
        sources=[{"title": "runbook", "url": "https://example.com/rb"}],
        diagram_present=False,
        session_id="slack-thread:C123:111.1",
        investigation_id="inv-9",
    )
    await adapter.finalize(handle, result)
    last = fake.updated[-1]
    assert last["ts"] == handle.ts
    assert isinstance(last["blocks"], list) and last["blocks"]
    # The feedback buttons row must carry our investigation id.
    flat = str(last["blocks"])
    assert "up:inv-9" in flat and "down:inv-9" in flat


@pytest.mark.asyncio
async def test_send_denial_dms_user() -> None:
    adapter, fake = _adapter_with_fake()
    msg = InboundMessage(
        channel_id="C123", user_id="U1", text="hi", message_id="m1",
        thread_id=None, is_dm=False, workspace="T9",
    )
    await adapter.send_denial(msg, "not allowed here")
    # DM uses the user_id as the channel target.
    assert fake.posted[-1]["channel"] == "U1"
    assert fake.posted[-1]["text"] == "not allowed here"


# ---------------------------------------------------------------------------
# fetch_thread: self-filter + display names
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fetch_thread_marks_self_and_resolves_names() -> None:
    adapter, fake = _adapter_with_fake()
    fake.user_info = {"U2": {"profile": {"display_name": "Alice"}}}
    fake.thread_replies = [
        {"user": "U2", "text": "the alert fired", "ts": "1"},
        {"user": "UBOT", "text": "our past answer", "ts": "2"},  # our own -> is_self
        {"username": "Rootly", "text": "INC-1 sev2", "ts": "3"},  # other bot kept
        {"user": "U2", "text": "   ", "ts": "4"},                 # empty -> skipped
    ]
    msgs = await adapter.fetch_thread("C123", "1", cap=20)
    # The empty message is dropped; 3 remain.
    assert len(msgs) == 3
    by_text = {m.text: m for m in msgs}
    assert by_text["the alert fired"].author == "Alice"
    assert by_text["the alert fired"].is_self is False
    assert by_text["our past answer"].is_self is True
    assert by_text["INC-1 sev2"].author == "Rootly"
    assert by_text["INC-1 sev2"].is_self is False


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resolve_identity_oid_format() -> None:
    adapter, _ = _adapter_with_fake()
    msg = InboundMessage(
        channel_id="C123", user_id="U1", text="hi", message_id="m1",
        thread_id=None, is_dm=False, workspace="T9",
    )
    user = await adapter.resolve_identity(msg)
    assert user.oid == "slack-bot:T9:U1"
    assert user.is_anonymous is True


# ---------------------------------------------------------------------------
# Event shim -> CoreSink
# ---------------------------------------------------------------------------
class _RecordingSink:
    def __init__(self) -> None:
        self.messages: list[InboundMessage] = []
        self.feedbacks: list[FeedbackEvent] = []

    async def on_message(self, msg: InboundMessage) -> None:
        self.messages.append(msg)

    async def on_feedback(self, fb: FeedbackEvent) -> None:
        self.feedbacks.append(fb)


@pytest.mark.asyncio
async def test_event_shim_routes_to_sink() -> None:
    adapter, _ = _adapter_with_fake()
    sink = _RecordingSink()
    shim = _SlackEventShim(adapter, sink)

    await shim.on_app_mention(
        {"channel": "C1", "user": "U1", "text": "<@UBOT> q", "ts": "1", "team": "T"},
    )
    await shim.on_message_im({"channel": "D1", "user": "U1", "text": "dm q", "ts": "2"})
    await shim.on_block_action(
        {
            "actions": [{"action_id": "opsrag_feedback_down", "value": "down:inv-1"}],
            "user": {"id": "U1"},
            "container": {},
            "response_url": "https://hooks.example.com/y",
        },
    )

    assert len(sink.messages) == 2
    assert sink.messages[0].is_dm is False and sink.messages[0].text == "q"
    assert sink.messages[1].is_dm is True and sink.messages[1].text == "dm q"
    assert len(sink.feedbacks) == 1
    assert sink.feedbacks[0].thumbs == "down" and sink.feedbacks[0].investigation_id == "inv-1"
