"""Slack channel adapter.

``SlackAdapter`` implements the :class:`~opsrag.channels.base.ChannelAdapter`
port by WRAPPING the existing, byte-for-byte-unchanged Slack leaf modules
(``opsrag.slack_bot.client`` / ``render`` / ``thread_context`` / ``identity``).
No Slack behaviour changes here -- the adapter only re-shapes those modules'
calls into the neutral port surface.

The ``slack_sdk`` import is reached only through ``SlackBotClient`` (lazy at
``connect`` time), so importing this module on the ``api`` role is cheap.
"""
from __future__ import annotations

from opsrag.channels.adapters.slack.adapter import SlackAdapter

__all__ = ["SlackAdapter"]
