"""Per-platform channel adapters.

Real adapters (Slack/Telegram/Discord/Teams) live in their own
subpackages and are resolved lazily by ``channels.boot`` so a disabled
channel never imports its SDK. ``FakeAdapter`` is the in-memory adapter
used by the core test suite -- it imports nothing platform-specific.
"""
from __future__ import annotations
