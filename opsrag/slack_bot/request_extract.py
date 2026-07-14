"""Pure request classification + query extraction for the first-responder.

No I/O, no Slack calls -- these are the cheap predicates run BEFORE any
agent/LLM work so bot noise is rejected early (channel-cost-fanout defense).
"""
from __future__ import annotations

import enum
import re
from typing import Any

from opsrag.channels.config import FirstResponderChannelConfig

# Slack prefixes an app_mention's text with "<@U0BOT>"; strip it so the agent
# never sees the mention token.
_MENTION_RE = re.compile(r"^(?:<@[A-Z0-9]+>\s*)+")

# Requester attribution. WORKFLOW posts carry the human in a labeled field
# ("Requester:/From:/Submitted by: <@U..>"); anchor to that rather than the
# FIRST mention (which may be an unrelated @teammate) so we never ping/attribute
# the wrong person. When no labeled field AND >1 distinct mention -> ambiguous
# -> return None (degrade to no greeting / no author metadata).
_REQUESTER_FIELD_RE = re.compile(
    r"(?:requester|requested\s+by|reported\s+by|reporter|from|submitted\s+by|"
    r"raised\s+by|created\s+by|opened\s+by|on\s+behalf\s+of)\s*[:\-]?\s*"
    r"<@([UW][A-Z0-9]+)>",
    re.IGNORECASE,
)
_ANY_USER_RE = re.compile(r"<@([UW][A-Z0-9]+)(?:\|[^>]+)?>")

# Message subtypes that are never a request. NOTE: "bot_message" is
# deliberately absent -- that is the workflow post we WANT to answer.
# "message_changed" is the critical one: OpsRAG's own chat.update edits
# arrive as message_changed and must be dropped to avoid a self-loop.
_IGNORED_SUBTYPES: frozenset[str] = frozenset(
    {
        "message_changed",
        "message_deleted",
        "message_replied",
        "thread_broadcast",
        "channel_join",
        "channel_leave",
        "channel_archive",
        "channel_unarchive",
        "channel_name",
        "channel_topic",
        "channel_purpose",
        "pinned_item",
        "unpinned_item",
        "bot_add",
        "bot_remove",
        "file_share",
        "reminder_add",
    }
)


class RequestKind(enum.Enum):
    """What a channel message is, for first-responder routing."""

    WORKFLOW = "workflow"
    DIRECT = "direct"
    IGNORE = "ignore"


def is_ignorable_subtype(event: dict[str, Any]) -> bool:
    """True for edits/joins/system subtypes we must never treat as requests."""
    return (event or {}).get("subtype") in _IGNORED_SUBTYPES


def classify(
    event: dict[str, Any], chan_cfg: FirstResponderChannelConfig
) -> RequestKind:
    """Decide WORKFLOW / DIRECT / IGNORE for a top-level channel message.

    Assumes ``is_ignorable_subtype`` already returned False (caller drops
    those first). A bot post is WORKFLOW only when its app_id or bot_id is in
    ``request_app_allowlist`` -- everything else (other bots, OpsRAG itself)
    is IGNORE. A human post is DIRECT when ``include_direct`` is set.
    """
    event = event or {}
    bot_id = event.get("bot_id")
    app_id = event.get("app_id")
    is_bot = bool(bot_id) or bool(app_id) or event.get("subtype") == "bot_message"
    if is_bot:
        allow = set(chan_cfg.request_app_allowlist)
        if (app_id and app_id in allow) or (bot_id and bot_id in allow):
            return RequestKind.WORKFLOW
        return RequestKind.IGNORE
    if chan_cfg.include_direct and (event.get("user") or "").strip():
        return RequestKind.DIRECT
    return RequestKind.IGNORE


def _flatten_blocks(event: dict[str, Any]) -> str:
    """Pull mrkdwn/plain_text out of section blocks (workflow layout fallback)."""
    parts: list[str] = []
    for block in (event.get("blocks") or []):
        if not isinstance(block, dict):
            continue
        txt = block.get("text")
        if isinstance(txt, dict) and txt.get("text"):
            parts.append(str(txt["text"]).strip())
        for field in (block.get("fields") or []):
            if isinstance(field, dict) and field.get("text"):
                parts.append(str(field["text"]).strip())
    return "\n".join(p for p in parts if p)


def extract_query(event: dict[str, Any]) -> str:
    """Normalized prompt from a channel event.

    Prefers ``text`` (workflow posts carry Basic Information + Full
    Description there). Falls back to flattening section blocks when text is
    empty (rich Workflow layouts). Strips any leading bot-mention token.
    """
    event = event or {}
    text = _MENTION_RE.sub("", (event.get("text") or "")).strip()
    if not text:
        text = _flatten_blocks(event)
    return text.strip()


def extract_requester(event: dict[str, Any]) -> str | None:
    """Bare Slack user id (U…/W…) of the human who made the request, else None.

    DIRECT human post:  ``event['user']``.
    WORKFLOW/bot post:   the id in a labeled Requester/From/Submitted-by field;
    if no labeled field but exactly ONE distinct mention exists, that id; else
    None (ambiguous -> don't guess). Returns the BARE id so the caller can both
    look it up (get_user_info) and wrap it (<@id>) for the ack.
    """
    event = event or {}
    is_bot = (
        bool(event.get("bot_id"))
        or bool(event.get("app_id"))
        or event.get("subtype") == "bot_message"
    )
    if not is_bot:
        return (event.get("user") or "").strip() or None
    text = event.get("text") or _flatten_blocks(event) or ""
    m = _REQUESTER_FIELD_RE.search(text)
    if m:
        return m.group(1)
    ids = _ANY_USER_RE.findall(text)
    if len(set(ids)) == 1:
        return ids[0]
    return None
