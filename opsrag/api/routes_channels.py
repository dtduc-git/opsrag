"""Public channel-conversation read API.

Lets any authorized web user BROWSE + READ conversations that happened in
SHARED chat channels (Slack/Discord/Teams channels, Telegram groups) via the
bots. Private 1:1 DMs and private web conversations are NOT exposed here.

Two endpoints, both read-only and gated on the ``chat`` scope (mirroring the
"investigations are shared, scope-gated, no owner" pattern rather than relaxing
the per-owner gate):

  GET /channels/conversations
      Sidebar listing of shared-channel conversations, most-recent-first.

  GET /channels/conversations/{thread_id}/messages
      Replayed message history for one shared-channel conversation.

Security: a conversation's privacy is encoded in its ``thread_id`` prefix
(``<platform>-thread:`` = shared, ``<platform>-dm:`` = private; web threads have
no such prefix). Both endpoints restrict to ``is_public_channel_thread`` SERVER
SIDE, so a private DM or a web thread_id can never be read through them.

Nginx strips /api/ before forwarding, so the router uses prefix='/channels'.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from opsrag.auth.scopes import Scope, require_scope
from opsrag.channels.origin import (
    PUBLIC_CHANNEL_THREAD_PREFIXES,
    is_public_channel_thread,
    platform_of,
)

_log = logging.getLogger("opsrag.api.routes_channels")

channels_router = APIRouter(
    prefix="/channels",
    tags=["channels"],
    dependencies=[Depends(require_scope(Scope.CHAT))],
)


def _get_store(request: Request):
    store = getattr(request.app.state, "session_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="session store unavailable")
    return store


@channels_router.get("/conversations")
async def list_channel_conversations(request: Request) -> dict:
    """Shared-channel conversations, most-recent-first.

    The synthetic bot ``user_id`` is dropped from each item -- it's an opaque
    platform oid, not useful to a reader -- and replaced with a ``platform``
    label derived from the thread_id prefix.
    """
    store = _get_store(request)
    rows = await store.list_sessions_by_prefixes(PUBLIC_CHANNEL_THREAD_PREFIXES)
    conversations = []
    for row in rows:
        item = {k: v for k, v in row.items() if k != "user_id"}
        item["platform"] = platform_of(row.get("thread_id"))
        conversations.append(item)
    # Newest-first (the store walk yields newest-first already; sort defensively
    # on updated_at so the contract is explicit).
    conversations.sort(key=lambda c: c.get("updated_at") or "", reverse=True)
    return {"conversations": conversations}


@channels_router.get("/conversations/{thread_id}/messages")
async def channel_conversation_messages(thread_id: str, request: Request) -> dict:
    """Replayed history for ONE shared-channel conversation.

    Returns 404 for any non-public thread_id (a private DM or a web thread),
    so this endpoint can only ever read shared-channel conversations.
    """
    if not is_public_channel_thread(thread_id):
        # 404 (not 403) -- don't reveal whether a private/web thread exists.
        raise HTTPException(status_code=404, detail="conversation not found")
    store = _get_store(request)
    messages = await store.get_messages(thread_id)
    return {
        "thread_id": thread_id,
        "platform": platform_of(thread_id),
        "messages": messages,
    }
