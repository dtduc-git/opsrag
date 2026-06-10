"""Discord channel adapter (``discord.py`` gateway worker).

``DiscordAdapter`` implements the :class:`~opsrag.channels.base.ChannelAdapter`
port over the Discord gateway websocket. It runs as the ``discordbot`` outbound
worker role (no public ingress -- the gateway is an outbound connection).

The ``discord`` SDK is imported LAZILY (inside ``connect`` / the methods that
actually touch the client), so importing this package on the ``api`` role -- or
in unit CI, where ``discord.py`` is NOT installed -- never raises
``ModuleNotFoundError``. All neutral-mapping logic lives in module-level helper
functions that take duck-typed objects, so the normalization + render paths are
fully testable without the SDK.
"""
from __future__ import annotations

from opsrag.channels.adapters.discord.adapter import DiscordAdapter

__all__ = ["DiscordAdapter"]
