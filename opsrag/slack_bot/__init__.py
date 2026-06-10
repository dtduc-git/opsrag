"""OpsRAG Slack chatbot package -- now a thin back-compat shim.

The Slack bot has been refactored onto the channel-neutral Ports & Adapters
core (``opsrag.channels``). The transport / render / thread / identity leaf
modules in this package are kept **byte-for-byte** and are wrapped by
:class:`opsrag.channels.adapters.slack.adapter.SlackAdapter`. This package now
exists only to keep historical imports working for one release:

  * ``SlackBotConfig`` / ``SlackBotClient`` / ``SlackProgressStreamer`` /
    ``assemble_thread_context`` -- unchanged leaf modules, re-exported.
  * ``SlackBotPermission`` -- now a Slack-event-dict facade over
    ``channels.permission.ChannelPermission`` (see ``permission.py``).
  * ``format_answer_as_slack_blocks`` / ``slack_user_to_current_user`` --
    unchanged leaf re-exports.
  * ``SlackBotClient`` boots the bot via ``channels.boot.build_and_start`` now,
    not via ``SlackEventDispatcher``; the old dispatcher is kept as a
    deprecated re-export only (``handler.py`` is no longer used by the server).

New code should import from ``opsrag.channels`` directly.
"""
from __future__ import annotations

# Unchanged leaf modules -- safe to import unconditionally.
from opsrag.slack_bot.client import SlackBotClient
from opsrag.slack_bot.config import SlackBotConfig
from opsrag.slack_bot.identity import slack_user_to_current_user
from opsrag.slack_bot.permission import SlackBotPermission
from opsrag.slack_bot.render import format_answer_as_slack_blocks
from opsrag.slack_bot.streaming import SlackProgressStreamer
from opsrag.slack_bot.thread_context import assemble_thread_context

# Deprecated: the old Slack-specific dispatcher. The server now boots Slack via
# ``channels.boot.build_and_start`` + ``SlackAdapter``; this is kept only so
# any lingering import doesn't break. Guarded because handler.py is otherwise
# unused and may be removed.
try:
    from opsrag.slack_bot.handler import SlackEventDispatcher  # noqa: F401
except ImportError:  # pragma: no cover - handler is optional/deprecated
    SlackEventDispatcher = None  # type: ignore[misc,assignment]

__all__ = [
    "SlackBotConfig",
    "SlackBotClient",
    "SlackProgressStreamer",
    "assemble_thread_context",
    "SlackBotPermission",
    "format_answer_as_slack_blocks",
    "slack_user_to_current_user",
    "SlackEventDispatcher",
]
