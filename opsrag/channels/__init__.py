"""Channel-agnostic core for the OpsRAG multi-channel chat bot.

Ports & Adapters (hexagonal): the platform-neutral flow lives here
(``dispatcher`` / ``streaming`` / ``permission`` / ``feedback``); each
platform is a thin adapter under ``adapters/`` behind the
:class:`~opsrag.channels.base.ChannelAdapter` port.

See design doc ``specs/002-channel-bots/design.md``.
"""
from __future__ import annotations
