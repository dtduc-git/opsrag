"""Unit tests for the SlackBotPermission back-compat shim (P1).

``tests/unit/test_slack_bot_channel_resolution.py`` already pins the
event-dict behaviour and MUST stay green unchanged. This file asserts the
*refactor invariant*: ``SlackBotPermission`` is now a thin facade over the
neutral ``ChannelPermission`` -- the allowlist/quota logic is shared, only the
Slack-event-dict normalization + the Slack-only bot-loop / channel-kind
pre-checks live in the subclass.

No network, no Slack client.
"""
from __future__ import annotations

import pytest

from opsrag.channels.permission import ChannelPermission
from opsrag.slack_bot.permission import SlackBotPermission


def test_slack_permission_is_channel_permission_subclass() -> None:
    perm = SlackBotPermission(allowed_channels={"C123"})
    assert isinstance(perm, ChannelPermission)


@pytest.mark.asyncio
async def test_quota_logic_shared_with_neutral_core() -> None:
    # record_usage / usage_count are inherited unchanged from ChannelPermission.
    perm = SlackBotPermission(allowed_channels={"C123"}, per_user_daily_quota=1)
    e = {"channel": "C123", "user": "U1"}
    assert (await perm.allow(e))[0] is True
    perm.record_usage("U1")
    assert perm.usage_count("U1") == 1
    ok, reason = await perm.allow(e)  # quota of 1 now hit
    assert ok is False and isinstance(reason, str)


@pytest.mark.asyncio
async def test_subtype_bot_message_silently_denied() -> None:
    # The Slack-only bot-loop guard (subtype) lives in the shim, not the core.
    perm = SlackBotPermission(allowed_channels={"C123"})
    ok, reason = await perm.allow(
        {"channel": "C123", "user": "U1", "subtype": "bot_message"},
    )
    assert ok is False and reason is None
