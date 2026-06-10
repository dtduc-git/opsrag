"""Unit tests for the channel-neutral ChannelPermission.

Ports the 7 Slack permission cases (allowlist choke point, DM bypass,
empty-allowlist deny-all, unknown/missing channel fail-closed, rolling
per-user quota) to the neutral InboundMessage. Bot-loop filtering is NOT
tested here -- it moved to the adapter (design 3.5), so ChannelPermission
only ever sees real user messages.

No network, no Slack client.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from opsrag.channels.permission import ChannelPermission
from opsrag.channels.types import InboundMessage


def _msg(
    *,
    channel_id: str = "C123",
    user_id: str = "U1",
    is_dm: bool = False,
    text: str = "hi",
) -> InboundMessage:
    return InboundMessage(
        channel_id=channel_id,
        user_id=user_id,
        text=text,
        message_id="m1",
        thread_id=None,
        is_dm=is_dm,
        workspace="W1",
    )


@pytest.mark.asyncio
async def test_allowlisted_channel_allowed() -> None:
    perm = ChannelPermission(allowed_channels={"C123"})
    ok, reason = await perm.allow(_msg(channel_id="C123"))
    assert ok is True and reason is None


@pytest.mark.asyncio
async def test_channel_not_in_allowlist_denied_with_message() -> None:
    perm = ChannelPermission(allowed_channels={"C123"})
    ok, reason = await perm.allow(_msg(channel_id="C999"))
    assert ok is False
    assert isinstance(reason, str) and reason  # user-facing deny message


@pytest.mark.asyncio
async def test_empty_allowlist_denies_all_public_channels() -> None:
    perm = ChannelPermission(allowed_channels=set())
    ok, _ = await perm.allow(_msg(channel_id="C123"))
    assert ok is False


@pytest.mark.asyncio
async def test_dm_bypasses_allowlist() -> None:
    # DM => is_dm True; empty allowlist still allows it (quota only).
    perm = ChannelPermission(allowed_channels=set())
    ok, reason = await perm.allow(_msg(channel_id="D123", is_dm=True))
    assert ok is True and reason is None


@pytest.mark.asyncio
async def test_missing_channel_on_non_dm_fail_closed() -> None:
    perm = ChannelPermission(allowed_channels={"C123"})
    ok, reason = await perm.allow(_msg(channel_id="", is_dm=False))
    assert ok is False and reason is None  # silent fail-closed


@pytest.mark.asyncio
async def test_non_inbound_message_rejected() -> None:
    perm = ChannelPermission(allowed_channels={"C123"})
    ok, reason = await perm.allow({"channel": "C123"})  # type: ignore[arg-type]
    assert ok is False and reason is None


@pytest.mark.asyncio
async def test_per_user_quota_enforced() -> None:
    # allow() only checks the quota; record_usage() is what moves it, so
    # errors/denials never burn quota.
    perm = ChannelPermission(allowed_channels={"C123"}, per_user_daily_quota=2)
    m = _msg(channel_id="C123", user_id="U1")
    assert (await perm.allow(m))[0] is True
    perm.record_usage("U1")
    assert (await perm.allow(m))[0] is True
    perm.record_usage("U1")
    ok, reason = await perm.allow(m)  # 2 recorded -> quota of 2 hit
    assert ok is False
    assert isinstance(reason, str)


@pytest.mark.asyncio
async def test_quota_is_per_user_not_global() -> None:
    perm = ChannelPermission(allowed_channels={"C123"}, per_user_daily_quota=1)
    perm.record_usage("U1")
    # U1 is now at quota; U2 is fresh.
    assert (await perm.allow(_msg(user_id="U1")))[0] is False
    assert (await perm.allow(_msg(user_id="U2")))[0] is True


@pytest.mark.asyncio
async def test_dm_still_quota_limited() -> None:
    # DMs bypass the allowlist but NOT the quota.
    perm = ChannelPermission(allowed_channels=set(), per_user_daily_quota=1)
    m = _msg(channel_id="D123", is_dm=True, user_id="U1")
    assert (await perm.allow(m))[0] is True
    perm.record_usage("U1")
    ok, reason = await perm.allow(m)
    assert ok is False and isinstance(reason, str)


@pytest.mark.asyncio
async def test_quota_zero_disables_quota() -> None:
    perm = ChannelPermission(allowed_channels={"C123"}, per_user_daily_quota=0)
    for _ in range(50):
        perm.record_usage("U1")
    ok, _ = await perm.allow(_msg(user_id="U1"))
    assert ok is True  # quota of 0 => unlimited


def test_record_usage_ignores_empty_user() -> None:
    perm = ChannelPermission(allowed_channels={"C123"})
    perm.record_usage("")  # no-op, must not raise
    assert perm.usage_count("") == 0


def test_usage_count_trims_old_entries() -> None:
    perm = ChannelPermission(allowed_channels={"C123"})
    now = time.time()
    # Inject one entry just over 24h old + one fresh.
    perm._usage["U1"] = [now - (25 * 60 * 60), now]  # noqa: SLF001
    assert perm.usage_count("U1") == 1  # the stale one is trimmed from the count


@pytest.mark.asyncio
async def test_concurrent_allow_and_record_usage_no_lost_updates() -> None:
    # The dispatcher may invoke allow()/record_usage() concurrently when a
    # burst of events arrives. ChannelPermission guards the rolling window
    # with an asyncio.Lock; N concurrent allow-then-record cycles must leave
    # the window at exactly N (no lost updates, no exceptions).
    perm = ChannelPermission(allowed_channels={"C123"}, per_user_daily_quota=1000)
    m = _msg(channel_id="C123", user_id="U1")

    async def cycle() -> None:
        ok, _ = await perm.allow(m)
        assert ok is True
        perm.record_usage("U1")

    await asyncio.gather(*[cycle() for _ in range(100)])
    assert perm.usage_count("U1") == 100
