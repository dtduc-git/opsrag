"""Permission + quota checks for inbound Slack events.

Two enforcement layers -- both required, because Slack's own
``/invite`` model is a *capability* gate, not a *cost* gate. Anyone
with channel-admin can invite the bot, and one wide rollout would
burn the LLM budget. The allowlist is our cost-control choke point.

1. **Channel allowlist** -- public/private channels (``C...`` / ``G...``)
   must appear in the allowlist. An empty allowlist means "no public
   channels allowed" -- populate it explicitly for every channel that
   should be able to ask the bot questions. DMs (``D...``) are always
   allowed (sender is implicitly identified, harder to abuse at scale).
   Bot-to-bot messages (``event.bot_id`` set) are silently denied to
   avoid feedback loops.

2. **Per-user daily quota** -- an in-memory ring buffer of recent
   request timestamps per Slack user. Pod restart resets the counters
   (this is intentional for v1 -- persistence is a Phase 2 concern).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

_log = logging.getLogger("opsrag.slack_bot.permission")

_ONE_DAY_S = 24 * 60 * 60


class SlackBotPermission:
    """Channel allowlist + per-user daily quota.

    State note
    ----------
    ``_usage`` is an *in-memory* dict ``{user_id: [ts, ts, ...]}`` of
    request epochs in seconds, trimmed to the last 24h on every check.
    There is no persistence -- a pod restart clears the quota state.
    This is acceptable for v1 (200/user/day is generous; abuse-tier
    enforcement is not the threat model). If you raise the quota much
    higher or care about precise rolling windows across restarts,
    promote this to Redis.
    """

    def __init__(
        self,
        allowed_channels: set[str] | list[str],
        per_user_daily_quota: int = 200,
        *,
        deny_dm_message: str = (
            "Sorry, I'm not enabled in that channel. "
            "Ping #devops if you want me added."
        ),
    ) -> None:
        self._allowed_channels: set[str] = set(allowed_channels)
        self._quota = int(per_user_daily_quota)
        self._deny_dm_message = deny_dm_message
        self._usage: dict[str, list[float]] = {}
        # Coarse lock -- the dispatcher is single-event-at-a-time per
        # Slack process today, but this keeps us safe if the sister
        # client ever fans events out concurrently.
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    async def allow(
        self, event: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """Decide whether to act on ``event``.

        Returns
        -------
        (ok, deny_reason)
            ``ok=True`` -> caller proceeds, ``deny_reason`` is None.
            ``ok=False`` -> caller MUST NOT run the agent.
            ``deny_reason`` is a user-facing string when we want the
            handler to DM/reply with an explanation; it is ``None``
            for silent denials (other bots, malformed events).
        """
        if not isinstance(event, dict):
            _log.debug("permission: rejecting non-dict event %r", type(event))
            return False, None

        # Silent deny: messages from other bots (incl. ourselves). This
        # is the canonical Slack-bot loop-prevention check.
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return False, None

        channel = event.get("channel") or ""
        user_id = event.get("user") or ""

        # DMs always allowed (no allowlist gate). Quota still applies.
        is_dm = channel.startswith("D")
        if not is_dm:
            if not channel.startswith(("C", "G")):
                # Unknown channel kind -- fail closed silently.
                _log.debug("permission: unknown channel kind %r", channel)
                return False, None
            if channel not in self._allowed_channels:
                _log.info(
                    "permission: deny channel=%s user=%s reason=not-in-allowlist",
                    channel, user_id,
                )
                return False, self._deny_dm_message

        # Per-user quota (DMs included -- we don't want a single user
        # hammering the bot in DM either).
        if user_id and self._quota > 0:
            async with self._lock:
                now = time.time()
                bucket = self._usage.get(user_id, [])
                bucket = [t for t in bucket if (now - t) < _ONE_DAY_S]
                self._usage[user_id] = bucket
                if len(bucket) >= self._quota:
                    _log.info(
                        "permission: deny channel=%s user=%s reason=quota count=%d quota=%d",
                        channel, user_id, len(bucket), self._quota,
                    )
                    return (
                        False,
                        (
                            f"You've hit the daily quota of {self._quota} "
                            "questions. Try again tomorrow."
                        ),
                    )

        return True, None

    def record_usage(self, user_id: str) -> None:
        """Append the current timestamp to the user's rolling bucket.

        The handler MUST call this after a *successful* agent run for
        the quota to actually move. Denied/errored events are not
        counted (this matches the user expectation that errors don't
        "burn" their quota).
        """
        if not user_id:
            return
        now = time.time()
        bucket = self._usage.get(user_id, [])
        # Trim opportunistically so the bucket doesn't grow unbounded
        # between explicit `allow` checks.
        bucket = [t for t in bucket if (now - t) < _ONE_DAY_S]
        bucket.append(now)
        self._usage[user_id] = bucket

    # ------------------------------------------------------------------
    # Introspection (used by tests + ops endpoints)
    # ------------------------------------------------------------------
    def usage_count(self, user_id: str) -> int:
        """Return current 24h request count for ``user_id``."""
        now = time.time()
        bucket = self._usage.get(user_id, [])
        return sum(1 for t in bucket if (now - t) < _ONE_DAY_S)
