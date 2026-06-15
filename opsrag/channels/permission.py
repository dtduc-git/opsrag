"""Channel-neutral allowlist + per-user daily quota.

This is ``SlackBotPermission`` lifted off the Slack ``event`` dict and
typed over the neutral :class:`~opsrag.channels.types.InboundMessage`.
The two enforcement layers are unchanged in spirit:

1. **Channel allowlist** -- public/group channels must appear in the
   ``allowed_channels`` allowlist (cost-control choke point; an empty
   allowlist denies all public channels).
2. **DM allowlist** -- DMs are NOT covered by the channel allowlist, so they
   have their own ``allowed_dm_users`` gate. DENY-BY-DEFAULT: an empty DM
   allowlist denies every DM (a stranger who finds the bot must not be able to
   query internal data); list platform user ids to allow them, or ``"*"`` to
   allow anyone. Unauthorized DMs are denied SILENTLY (logged, no reply).
3. **Per-user daily quota** -- an in-memory rolling 24h ring buffer of
   request timestamps per user. ``record_usage`` is called by the
   dispatcher only after a *successful* agent run, so errors/denials
   never burn a user's quota.

Bot-loop filtering does NOT live here: the adapter knows its own bot id
and drops its own + other bots' messages before the core ever sees them
(design section 3.5). So this class only ever sees real user messages.

See design doc ``specs/002-channel-bots/design.md`` section 3.5.
"""
from __future__ import annotations

import asyncio
import logging
import time

from opsrag.channels.types import InboundMessage

_log = logging.getLogger("opsrag.channels.permission")

_ONE_DAY_S = 24 * 60 * 60


class ChannelPermission:
    """Channel allowlist + per-user daily quota over ``InboundMessage``.

    State note
    ----------
    ``_usage`` is an *in-memory* dict ``{user_id: [ts, ...]}`` of request
    epochs in seconds, trimmed to the last 24h on every check. There is no
    persistence -- a worker restart clears the quota state. Acceptable for
    v1 (the quota is generous; abuse-tier enforcement is not the threat
    model). Promote to Redis if you need precise windows across restarts.
    """

    def __init__(
        self,
        allowed_channels: set[str] | list[str],
        per_user_daily_quota: int = 200,
        *,
        allowed_dm_users: set[str] | list[str] | None = None,
        deny_dm_message: str = (
            "Sorry, I'm not enabled in that channel. "
            "Ping #devops if you want me added."
        ),
    ) -> None:
        self._allowed_channels: set[str] = set(allowed_channels)
        self._quota = int(per_user_daily_quota)
        # Per-user DM allowlist (platform user ids). DENY-BY-DEFAULT: an empty
        # set means NO ONE may DM the bot -- otherwise any stranger who finds
        # the bot could extract internal answers + burn budget. The literal
        # "*" opts a deployment back into open DMs.
        self._allowed_dm_users: set[str] = set(allowed_dm_users or ())
        self._deny_dm_message = deny_dm_message
        self._usage: dict[str, list[float]] = {}
        # Coarse lock -- keeps the rolling window consistent if an adapter
        # ever fans inbound events out concurrently.
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    async def allow(self, msg: InboundMessage) -> tuple[bool, str | None]:
        """Decide whether to act on ``msg``.

        Returns ``(ok, deny_reason)``:
          * ``ok=True`` -> caller proceeds, ``deny_reason`` is ``None``.
          * ``ok=False`` -> caller MUST NOT run the agent. ``deny_reason``
            is a user-facing string when we want the dispatcher to DM the
            user an explanation; it is ``None`` for silent denials.
        """
        if not isinstance(msg, InboundMessage):
            _log.debug("permission: rejecting non-InboundMessage %r", type(msg))
            return False, None

        channel = msg.channel_id or ""
        user_id = msg.user_id or ""

        if msg.is_dm:
            # DM access gate (deny-by-default). Only users in the DM allowlist
            # (or "*") may DM the bot -- DMs aren't covered by the channel
            # allowlist, so without this any stranger could query internal data.
            # Silent deny (no reply) so the bot's existence isn't confirmed to
            # an unauthorized user; the attempt is logged so an operator can
            # see the id and add it to the allowlist.
            if "*" not in self._allowed_dm_users and user_id not in self._allowed_dm_users:
                _log.info(
                    "permission: deny dm user=%s reason=not-in-dm-allowlist",
                    user_id,
                )
                return False, None
        else:
            if not channel:
                # No channel id on a non-DM message -- fail closed silently.
                _log.debug("permission: missing channel id on non-DM message")
                return False, None
            if channel not in self._allowed_channels:
                _log.info(
                    "permission: deny channel=%s user=%s reason=not-in-allowlist",
                    channel, user_id,
                )
                return False, self._deny_dm_message

        # Per-user quota (DMs included -- a single user shouldn't hammer
        # the bot in DM either).
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

        The dispatcher MUST call this after a *successful* agent run for
        the quota to actually move. Denied/errored events are not counted
        (matches the expectation that errors don't "burn" quota).
        """
        if not user_id:
            return
        now = time.time()
        bucket = self._usage.get(user_id, [])
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
