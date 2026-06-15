"""Unit test (T147): Slack-bot channel allowlist + quota resolution.

Exercises SlackBotPermission.allow() against synthetic events - no Slack client
or network. Covers the allowlist choke point (cost control), DM bypass, bot-loop
prevention, unknown-channel fail-closed, and the per-user daily quota.
"""
from __future__ import annotations

import pytest

from opsrag.slack_bot.permission import SlackBotPermission


@pytest.mark.asyncio
async def test_allowlisted_channel_allowed() -> None:
    perm = SlackBotPermission(allowed_channels={"C123"})
    ok, reason = await perm.allow({"channel": "C123", "user": "U1"})
    assert ok is True and reason is None


@pytest.mark.asyncio
async def test_channel_not_in_allowlist_denied_with_message() -> None:
    perm = SlackBotPermission(allowed_channels={"C123"})
    ok, reason = await perm.allow({"channel": "C999", "user": "U1"})
    assert ok is False
    assert isinstance(reason, str) and reason  # user-facing deny message


@pytest.mark.asyncio
async def test_empty_allowlist_denies_all_public_channels() -> None:
    perm = SlackBotPermission(allowed_channels=set())
    ok, _ = await perm.allow({"channel": "C123", "user": "U1"})
    assert ok is False


@pytest.mark.asyncio
async def test_dm_gated_by_dm_allowlist() -> None:
    # Deny-by-default: an empty dm_allowlist denies the DM silently; listing
    # the user allows it. Inherited from ChannelPermission.
    perm = SlackBotPermission(allowed_channels=set())
    ok, reason = await perm.allow({"channel": "D123", "user": "U1"})
    assert ok is False and reason is None

    allowed = SlackBotPermission(allowed_channels=set(), allowed_dm_users={"U1"})
    ok2, reason2 = await allowed.allow({"channel": "D123", "user": "U1"})
    assert ok2 is True and reason2 is None


@pytest.mark.asyncio
async def test_bot_messages_silently_denied() -> None:
    perm = SlackBotPermission(allowed_channels={"C123"})
    ok, reason = await perm.allow({"channel": "C123", "user": "U1", "bot_id": "B1"})
    assert ok is False and reason is None  # silent (loop prevention)


@pytest.mark.asyncio
async def test_unknown_channel_kind_fail_closed() -> None:
    perm = SlackBotPermission(allowed_channels={"C123"})
    ok, reason = await perm.allow({"channel": "X999", "user": "U1"})
    assert ok is False and reason is None


@pytest.mark.asyncio
async def test_per_user_quota_enforced() -> None:
    # allow() only checks the quota; the handler calls record_usage() after a
    # successful run, so errors/denials never burn quota.
    perm = SlackBotPermission(allowed_channels={"C123"}, per_user_daily_quota=2)
    e = {"channel": "C123", "user": "U1"}
    assert (await perm.allow(e))[0] is True
    perm.record_usage("U1")
    assert (await perm.allow(e))[0] is True
    perm.record_usage("U1")
    ok, reason = await perm.allow(e)  # 2 already recorded -> quota of 2 hit
    assert ok is False
    assert isinstance(reason, str)
