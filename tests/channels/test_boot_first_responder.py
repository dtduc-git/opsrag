"""boot.build_and_start attaches a FirstResponder for the slack channel."""
from __future__ import annotations

import pytest

from opsrag.channels import boot as boot_mod
from opsrag.channels.config import (
    FirstResponderChannelConfig,
    FirstResponderConfig,
    SlackChannelConfig,
)

pytestmark = pytest.mark.asyncio


class _Caches:
    qa_cache = None
    investigation_cache = None
    semantic_router = None
    feedback_store = None


class _Cfg:
    def __init__(self, slack_cfg):
        self.channels = type("C", (), {"slack": slack_cfg})()
        self.vision = None


class FakeAdapter:
    last_instance = None

    def __init__(self, channel_cfg):
        self.channel_cfg = channel_cfg
        self.attached = None
        self.connected_with = None
        FakeAdapter.last_instance = self

    def attach_first_responder(self, fr):
        self.attached = fr

    async def connect(self, sink):
        self.connected_with = sink


async def test_boot_attaches_first_responder_when_enabled(monkeypatch):
    monkeypatch.setitem(boot_mod.ROLE_TO_CHANNEL, "slackworker", "slack")
    monkeypatch.setattr(boot_mod, "_load_adapter_class", lambda name: FakeAdapter)

    slack_cfg = SlackChannelConfig(
        enabled=True,
        allowlist=["C0EXAMPLE01"],
        first_responder=FirstResponderConfig(
            enabled=True,
            channels={"C0EXAMPLE01": FirstResponderChannelConfig(
                request_app_allowlist=["A0EXAMPLE01"], oncall_handle="S0EXAMPLE01",
            )},
        ),
    )
    adapter = await boot_mod.build_and_start(
        "slackworker", _Cfg(slack_cfg), agent_graph=object(),
        providers=object(), caches=_Caches(),
    )
    assert adapter is FakeAdapter.last_instance
    assert adapter.attached is not None
    from opsrag.slack_bot.first_responder import FirstResponder
    assert isinstance(adapter.attached, FirstResponder)


async def test_boot_does_not_attach_when_first_responder_disabled(monkeypatch):
    monkeypatch.setitem(boot_mod.ROLE_TO_CHANNEL, "slackworker", "slack")
    monkeypatch.setattr(boot_mod, "_load_adapter_class", lambda name: FakeAdapter)

    slack_cfg = SlackChannelConfig(enabled=True, allowlist=["C0EXAMPLE01"])
    adapter = await boot_mod.build_and_start(
        "slackworker", _Cfg(slack_cfg), agent_graph=object(),
        providers=object(), caches=_Caches(),
    )
    assert adapter.attached is None
