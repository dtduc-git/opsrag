"""Channel-conversation origin helpers.

A conversation's privacy is encoded in its ``thread_id`` prefix, which the
channel dispatcher stamps in ``_session_thread_id`` (see
``opsrag/channels/dispatcher.py``):

    <platform>-thread:<channel_id>:<anchor>   # SHARED channel/group thread
    <platform>-dm:<channel_id>                # PRIVATE 1:1 DM

Only ``-thread:`` conversations are considered public (everyone in that channel
already saw them). ``-dm:`` and plain web threads stay private. This module is
the single source of truth for that predicate so the API + store agree.
"""
from __future__ import annotations

_PLATFORMS = ("slack", "discord", "telegram", "teams")

# Prefixes of SHARED (public) channel conversations -- one per platform.
PUBLIC_CHANNEL_THREAD_PREFIXES: tuple[str, ...] = tuple(
    f"{p}-thread:" for p in _PLATFORMS
)


def is_public_channel_thread(thread_id: str | None) -> bool:
    """True iff ``thread_id`` is a shared (public) channel thread.

    Private 1:1 DMs (``<platform>-dm:``) and web threads return False, so this
    is safe to use as the server-side authorization predicate for the
    public-channel read endpoints.
    """
    return bool(thread_id) and thread_id.startswith(PUBLIC_CHANNEL_THREAD_PREFIXES)


def platform_of(thread_id: str | None) -> str | None:
    """The platform (``slack``/``discord``/``telegram``/``teams``) for a public
    channel thread, or None if ``thread_id`` isn't a public channel thread."""
    if not thread_id:
        return None
    for platform in _PLATFORMS:
        if thread_id.startswith(f"{platform}-thread:"):
            return platform
    return None
