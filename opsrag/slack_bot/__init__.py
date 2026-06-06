"""OpsRAG Slack chatbot package.

See SESSION-SLACK-BOT-PLAN.md for the full design. This package is
split across two concurrent agents:

  * THIS agent owns: config.py, client.py, streaming.py, thread_context.py
  * Sister agent owns: handler.py, render.py, identity.py, permission.py

The sister-agent imports below are TYPE_CHECKING-style -- they're
guarded so this package imports cleanly before the sister's files
exist. Once the sister's PR lands, the try-block succeeds and the
re-exports become real.
"""
from __future__ import annotations

# Own modules -- safe to import unconditionally.
from opsrag.slack_bot.client import SlackBotClient
from opsrag.slack_bot.config import SlackBotConfig
from opsrag.slack_bot.streaming import SlackProgressStreamer
from opsrag.slack_bot.thread_context import assemble_thread_context

# Sister-agent modules -- guarded so this package imports cleanly even
# before those files exist. Once the sister's PR lands they become
# real re-exports.
try:
    from opsrag.slack_bot.handler import SlackEventDispatcher  # type: ignore[import-not-found]
    from opsrag.slack_bot.identity import (
        slack_user_to_current_user,  # type: ignore[import-not-found]
    )
    from opsrag.slack_bot.permission import SlackBotPermission  # type: ignore[import-not-found]
    from opsrag.slack_bot.render import (
        format_answer_as_slack_blocks,  # type: ignore[import-not-found]
    )
except ImportError:
    SlackEventDispatcher = None  # type: ignore[misc,assignment]
    format_answer_as_slack_blocks = None  # type: ignore[misc,assignment]
    SlackBotPermission = None  # type: ignore[misc,assignment]
    slack_user_to_current_user = None  # type: ignore[misc,assignment]

__all__ = [
    "SlackBotConfig",
    "SlackBotClient",
    "SlackProgressStreamer",
    "assemble_thread_context",
    "SlackEventDispatcher",
    "format_answer_as_slack_blocks",
    "SlackBotPermission",
    "slack_user_to_current_user",
]
