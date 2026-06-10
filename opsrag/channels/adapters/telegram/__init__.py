"""Telegram channel adapter.

``TelegramAdapter`` implements the :class:`~opsrag.channels.base.ChannelAdapter`
port over the Telegram **Bot API**, spoken directly with ``httpx`` (already a
project dependency -- no ``python-telegram-bot``). The transport is a long-poll
loop calling ``getUpdates``; there is no public ingress to operate.

Importing this module touches nothing beyond the stdlib + the neutral channel
types: ``httpx`` is imported lazily inside ``connect`` / the ``_api`` helper, so
the package imports cleanly on the ``api`` role (and in unit CI) even if the
Telegram channel is disabled.
"""
from __future__ import annotations

from opsrag.channels.adapters.telegram.adapter import TelegramAdapter

__all__ = ["TelegramAdapter"]
