"""SlackAdapter <-> FirstResponder wiring tests (no real Slack)."""
from __future__ import annotations

import pytest

from opsrag.channels.adapters.slack.adapter import SlackAdapter, _SlackEventShim

pytestmark = pytest.mark.asyncio


class FakeFR:
    def __init__(self):
        self.bound = None
        self.events: list[dict] = []

    def bind_client(self, client):
        self.bound = client

    async def on_channel_message(self, event):
        self.events.append(event)


def test_attach_first_responder_stores_it():
    adapter = SlackAdapter(config=object())
    fr = FakeFR()
    adapter.attach_first_responder(fr)
    assert adapter._first_responder is fr


async def test_shim_delegates_channel_message_to_fr():
    adapter = SlackAdapter(config=object())
    fr = FakeFR()
    adapter.attach_first_responder(fr)
    shim = _SlackEventShim(adapter, sink=object())
    event = {"type": "message", "channel": "C0EXAMPLE01", "ts": "1.1"}
    await shim.on_channel_message(event)
    assert fr.events == [event]


async def test_shim_channel_message_noop_without_fr():
    adapter = SlackAdapter(config=object())
    shim = _SlackEventShim(adapter, sink=object())
    # No FirstResponder attached -- must be a silent no-op.
    await shim.on_channel_message({"type": "message", "ts": "1.1"})
