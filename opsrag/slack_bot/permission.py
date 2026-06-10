"""Back-compat ``SlackBotPermission`` -- the Slack event-dict shim.

The permission/quota *logic* now lives in the channel-neutral
:class:`opsrag.channels.permission.ChannelPermission`, typed over
:class:`~opsrag.channels.types.InboundMessage`. This module keeps the
historical ``SlackBotPermission`` API alive for one release: it accepts the
Slack **event dict** shape (``{"channel", "user", "bot_id", "subtype", ...}``),
performs the two Slack-specific pre-checks the neutral core no longer does
(bot-loop drop + Slack ``C``/``G``/``D`` channel-kind classification), then
delegates the allowlist + rolling-quota decision to ``ChannelPermission``.

``tests/unit/test_slack_bot_channel_resolution.py`` exercises THIS class with
synthetic event dicts and must stay green unchanged. Bot-loop filtering and the
``X999`` unknown-channel-kind fail-closed therefore stay HERE (in the Slack
adapter's normalization layer), not in the neutral core.
"""
from __future__ import annotations

import logging
from typing import Any

from opsrag.channels.permission import ChannelPermission
from opsrag.channels.types import InboundMessage

_log = logging.getLogger("opsrag.slack_bot.permission")


class SlackBotPermission(ChannelPermission):
    """Slack event-dict facade over :class:`ChannelPermission`.

    Subclasses the neutral permission so ``record_usage`` / ``usage_count`` /
    the rolling-window state are all inherited unchanged. Only ``allow`` is
    overridden, to translate a Slack event dict into an ``InboundMessage`` (and
    apply the Slack-only bot-loop + channel-kind pre-checks) before delegating.
    """

    async def allow(  # type: ignore[override]
        self, event: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """Decide whether to act on a Slack ``event`` dict.

        Returns ``(ok, deny_reason)`` exactly as before:
          * silent deny (``deny_reason is None``) for other bots / our own
            messages (loop prevention) and unknown channel kinds (fail-closed).
          * user-facing ``deny_reason`` string for not-in-allowlist / quota.
        """
        if not isinstance(event, dict):
            _log.debug("permission: rejecting non-dict event %r", type(event))
            return False, None

        # Slack-only bot-loop guard (the neutral core never sees bot messages
        # because the adapter drops them, but the legacy event-dict path must
        # still classify them itself).
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return False, None

        channel = event.get("channel") or ""
        user_id = event.get("user") or ""

        # Slack channel-kind classification: C/G = public/group, D = DM.
        # Anything else (e.g. "X999") is an unknown kind -> fail closed.
        is_dm = channel.startswith("D")
        if not is_dm and not channel.startswith(("C", "G")):
            _log.debug("permission: unknown channel kind %r", channel)
            return False, None

        msg = InboundMessage(
            channel_id=channel,
            user_id=user_id,
            text=event.get("text") or "",
            message_id=event.get("ts") or "",
            thread_id=event.get("thread_ts") or None,
            is_dm=is_dm,
            workspace=event.get("team") or None,
            raw=dict(event),
        )
        return await super().allow(msg)
