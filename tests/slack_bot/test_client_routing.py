"""Routing tests for SlackBotClient._on_request (channel-message + bot-skip)."""
from __future__ import annotations

import pytest

from opsrag.slack_bot.client import SlackBotClient

pytestmark = pytest.mark.asyncio


class RecordingDispatcher:
    def __init__(self, with_channel=True):
        self.mentions: list[dict] = []
        self.ims: list[dict] = []
        self.channels: list[dict] = []
        if with_channel:
            self.on_channel_message = self._on_channel  # type: ignore[attr-defined]

    async def on_app_mention(self, event):
        self.mentions.append(event)

    async def on_message_im(self, event):
        self.ims.append(event)

    async def on_block_action(self, payload):
        pass

    async def _on_channel(self, event):
        self.channels.append(event)


class FakeSocketClient:
    async def send_socket_mode_response(self, resp):
        pass


class FakeReq:
    def __init__(self, event, envelope_id="e1"):
        self.type = "events_api"
        self.envelope_id = envelope_id
        self.payload = {"event": event}


def _client(dispatcher):
    c = SlackBotClient(bot_token="xoxb-test", app_token="xapp-test")
    c._dispatcher = dispatcher
    c.self_bot_id = "BSELF"
    c.self_user_id = "USELF"
    return c


async def test_workflow_bot_message_routes_to_channel_handler():
    d = RecordingDispatcher()
    c = _client(d)
    event = {"type": "message", "subtype": "bot_message", "bot_id": "B0EXAMPLE01",
             "channel_type": "channel", "channel": "C0EXAMPLE01", "ts": "1.1"}
    await c._on_request(FakeSocketClient(), FakeReq(event))
    assert d.channels == [event]
    assert d.mentions == [] and d.ims == []


async def test_app_mention_still_routes_and_ignores_bots():
    d = RecordingDispatcher()
    c = _client(d)
    human = {"type": "app_mention", "user": "U1", "channel": "C1", "ts": "1.1"}
    await c._on_request(FakeSocketClient(), FakeReq(human))
    assert d.mentions == [human]
    # a bot-authored app_mention is still skipped (feedback-loop guard).
    d.mentions.clear()
    botmention = {"type": "app_mention", "bot_id": "BX", "channel": "C1", "ts": "1.2"}
    await c._on_request(FakeSocketClient(), FakeReq(botmention))
    assert d.mentions == []


async def test_dm_still_routes_and_ignores_bots():
    d = RecordingDispatcher()
    c = _client(d)
    im = {"type": "message", "channel_type": "im", "user": "U1", "channel": "D1", "ts": "1.1"}
    await c._on_request(FakeSocketClient(), FakeReq(im))
    assert d.ims == [im]
    d.ims.clear()
    botim = {"type": "message", "channel_type": "im", "bot_id": "BX", "channel": "D1", "ts": "1.2"}
    await c._on_request(FakeSocketClient(), FakeReq(botim))
    assert d.ims == []


async def test_own_post_is_self_guarded_out_of_channel_route():
    d = RecordingDispatcher()
    c = _client(d)
    own = {"type": "message", "bot_id": "BSELF", "channel_type": "channel",
           "channel": "C0EXAMPLE01", "ts": "1.1"}
    await c._on_request(FakeSocketClient(), FakeReq(own))
    assert d.channels == []


async def test_mpim_is_not_routed_to_channel_handler():
    d = RecordingDispatcher()
    c = _client(d)
    event = {"type": "message", "subtype": "bot_message", "bot_id": "B1",
             "channel_type": "mpim", "channel": "G1", "ts": "1.1"}
    await c._on_request(FakeSocketClient(), FakeReq(event))
    assert d.channels == []


async def test_channel_message_noop_when_dispatcher_lacks_handler():
    d = RecordingDispatcher(with_channel=False)  # no on_channel_message
    c = _client(d)
    event = {"type": "message", "subtype": "bot_message", "bot_id": "B1",
             "channel_type": "channel", "channel": "C1", "ts": "1.1"}
    # Must not raise -- getattr(..., None) makes it a silent no-op.
    await c._on_request(FakeSocketClient(), FakeReq(event))
