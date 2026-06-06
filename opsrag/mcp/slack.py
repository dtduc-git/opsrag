"""Slack MCP-style tools for OpsRAG.

Lets the agent fetch specific Slack messages by permalink, hydrate full
threads, search messages, and list channels. Complements the indexed
Slack source (`opsrag/sources/slack`) by allowing on-demand fetch when
a user pastes a Slack URL.

## Read-only enforcement

All four tools issue Slack Web API GETs (`conversations.history`,
`conversations.replies`, `conversations.info`, `conversations.list`,
`users.info`, `search.messages`). No write endpoints, ever.

## Tool list (4)

| Tool                          | Slack API                       |
|-------------------------------|---------------------------------|
| `slack_get_message_by_url`    | `conversations.history` (single message by ts) |
| `slack_get_thread_by_url`     | `conversations.replies`         |
| `slack_search_messages`       | `search.messages`               |
| `slack_list_channels`         | `conversations.list` (cached)   |

## Tokens

- `SLACK_BOT_TOKEN` (xoxb-...) -- required, reused from indexing source.
- `SLACK_USER_TOKEN` (xoxp-...) -- optional. `search.messages` requires
  a user token (`search:read` scope); if absent, the search tool returns
  a clear error rather than failing silently.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

from opsrag.mcp.gitlab import MCPTool
from opsrag.sources.slack.client import SlackClient

_log = logging.getLogger("opsrag.mcp.slack")

_BOT_TOKEN_ENV_KEYS = ("OPSRAG_SLACK_BOT_TOKEN", "SLACK_BOT_TOKEN")
_USER_TOKEN_ENV_KEYS = ("OPSRAG_SLACK_USER_TOKEN", "SLACK_USER_TOKEN")
# Per Constitution Principle VI, no default workspace URL: operators
# must set OPSRAG_SLACK_WORKSPACE_URL or supply it via
# DeploymentContext.source_urls.slack. _workspace_url() returns None
# when neither is set; callers that build deep-links MUST skip link
# rendering rather than substitute a placeholder.
_CHANNELS_CACHE_TTL_S = 300.0  # 5 min, per spec
_MESSAGE_TRUNCATE_CHARS = 2000
_DEFAULT_SEARCH_LIMIT = 20
_MAX_SEARCH_LIMIT = 100


class SlackMCPError(Exception):
    """Raised for Slack MCP-tool errors. Includes a short `reason`
    machine code (e.g. `bad_url`, `not_in_channel`, `no_user_token`) so
    callers can render a helpful response."""

    def __init__(self, message: str, *, reason: str = "error"):
        self.reason = reason
        super().__init__(message)


# --- token resolution -----------------------------------------------


def _resolve_token(keys: tuple[str, ...]) -> str | None:
    for key in keys:
        v = os.environ.get(key)
        if v and v.strip():
            return v.strip().strip('"').strip("'")
    return None


def _resolve_bot_token() -> str:
    tok = _resolve_token(_BOT_TOKEN_ENV_KEYS)
    if not tok:
        raise SlackMCPError(
            "Slack bot token not configured. Set one of: "
            + ", ".join(_BOT_TOKEN_ENV_KEYS),
            reason="no_bot_token",
        )
    return tok


def _resolve_user_token() -> str | None:
    return _resolve_token(_USER_TOKEN_ENV_KEYS)


def _workspace_url() -> str | None:
    raw = os.environ.get("OPSRAG_SLACK_WORKSPACE_URL")
    return raw.rstrip("/") if raw else None


# --- URL parser -----------------------------------------------------

# Match `/archives/<CHAN>/p<TS>` anywhere in the path. Channels are
# upper-case alphanumerics starting with C/G/D; `p<TS>` is `p` followed
# by digits (Slack squashes the dot from `1714939200.000100` into
# `p1714939200000100`).
_URL_RE = re.compile(
    r"/archives/(?P<channel>[A-Z0-9]+)/p(?P<ts>\d{6,})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedSlackURL:
    channel: str
    ts: str  # canonical "1714939200.000100"
    thread_ts: str | None  # from ?thread_ts=...&cid=... if present
    original_url: str


def parse_slack_url(url: str) -> ParsedSlackURL:
    """Parse a Slack permalink into channel + message ts (+ thread).

    Accepts:
      - `https://example.slack.com/archives/<CHAN>/p<TS>`
      - `https://app.slack.com/client/T.../<CHAN>/p<TS>` (alternate host)
      - `example.slack.com/archives/<CHAN>/p<TS>` (no scheme)
      - any of the above with `?thread_ts=<TS>&cid=<CHAN>` query

    Raises `SlackMCPError(reason="bad_url")` on any unparseable input.
    """
    if not url or not isinstance(url, str):
        raise SlackMCPError(
            "Empty URL; expected a Slack permalink.", reason="bad_url"
        )
    raw = url.strip()
    # Tolerate scheme-less URLs and the alternate `app.slack.com/client/<TEAM>/<CHAN>/p<TS>` form.
    candidate = raw if "://" in raw else f"https://{raw}"
    try:
        parsed = urlparse(candidate)
    except ValueError as exc:
        raise SlackMCPError(
            f"This doesn't look like a Slack permalink: {url!r} ({exc})",
            reason="bad_url",
        ) from exc

    path = parsed.path or ""
    m = _URL_RE.search(path)
    if not m:
        # Alternate `app.slack.com/client/<TEAM>/<CHAN>/p<TS>` -- no "/archives/" segment.
        alt = re.search(r"/client/[A-Z0-9]+/(?P<channel>[A-Z0-9]+)/p(?P<ts>\d{6,})", path, re.IGNORECASE)
        if not alt:
            raise SlackMCPError(
                f"This doesn't look like a Slack permalink (missing /archives/<channel>/p<ts>): {url!r}",
                reason="bad_url",
            )
        channel = alt.group("channel").upper()
        ts_digits = alt.group("ts")
    else:
        channel = m.group("channel").upper()
        ts_digits = m.group("ts")

    # Slack's `pNNNNNNNNNN.NNNNNN` format: digits before the last 6 are
    # the epoch seconds, last 6 are the fractional component. Real
    # Slack permalinks always have >=10 digits total.
    if len(ts_digits) < 7:
        raise SlackMCPError(
            f"Malformed Slack ts {ts_digits!r}; expected >=7 digits after 'p'.",
            reason="bad_url",
        )
    ts = f"{ts_digits[:-6]}.{ts_digits[-6:]}"

    # Thread permalinks pass the root via `?thread_ts=<TS>&cid=<CHAN>`.
    thread_ts: str | None = None
    if parsed.query:
        q = parse_qs(parsed.query)
        t = q.get("thread_ts", [None])[0]
        if t:
            thread_ts = t

    return ParsedSlackURL(
        channel=channel,
        ts=ts,
        thread_ts=thread_ts,
        original_url=raw,
    )


# --- channel-list cache ---------------------------------------------

_channels_cache: dict[str, Any] = {"ts": 0.0, "channels": []}


async def _list_channels_cached(client: SlackClient) -> list[dict]:
    """Cached `conversations.list` (5-min TTL). Returns the full
    channel list (id, name, is_member, etc.). Cache key is global --
    one workspace per process -- so a refresh hits when stale."""
    now = time.time()
    if (now - _channels_cache["ts"]) < _CHANNELS_CACHE_TTL_S and _channels_cache["channels"]:
        return _channels_cache["channels"]
    channels: list[dict] = []
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {
            "limit": 200,
            "exclude_archived": True,
            "types": "public_channel,private_channel",
        }
        if cursor:
            params["cursor"] = cursor
        try:
            data = await client._get("conversations.list", params)  # noqa: SLF001 (intended reuse)
        except RuntimeError as exc:
            # Surface clearly; don't poison the cache.
            raise SlackMCPError(
                f"conversations.list failed: {exc}",
                reason="slack_api_error",
            ) from exc
        for c in data.get("channels", []):
            channels.append({
                "id": c.get("id"),
                "name": c.get("name"),
                "is_member": bool(c.get("is_member")),
                "is_private": bool(c.get("is_private")),
                "is_archived": bool(c.get("is_archived")),
                "num_members": int(c.get("num_members", 0)),
                "topic": ((c.get("topic") or {}).get("value") or "")[:200],
                "purpose": ((c.get("purpose") or {}).get("value") or "")[:200],
            })
        cursor = (data.get("response_metadata") or {}).get("next_cursor") or None
        if not cursor:
            break
    _channels_cache["ts"] = now
    _channels_cache["channels"] = channels
    return channels


# Per-id channel-name cache. Bounded by workspace size (one entry per
# unique channel ever resolved). ``None`` entries record known-missing
# ids so we don't retry conversations.info for channels the bot can't
# read. No TTL -- channel names rarely change and a stale name is
# better than a 30s rate-limit storm.
_channel_name_cache: dict[str, str | None] = {}


async def _resolve_channel_name(client: SlackClient, channel_id: str) -> str | None:
    """Return a `#channel-name` for a given channel id via conversations.info.

    Per-id cached. We deliberately do NOT enumerate the full workspace
    channel list here (``_list_channels_cached``) -- pagination of
    conversations.list is Tier-2 throttled at ~20 req/min, so a single
    cold lookup in a workspace with thousands of channels can take
    >5 minutes of rate-limit sleep. conversations.info is Tier-3
    (~50 req/min) and resolves one id in one call.

    Returns None when conversations.info fails (private channel the
    bot can't read, archived, wrong workspace); callers should fall
    back to rendering the raw id.
    """
    if channel_id in _channel_name_cache:
        return _channel_name_cache[channel_id]
    try:
        info = await client.get_channel(channel_id)
        name = info.name or None
    except Exception:  # noqa: BLE001
        name = None
    _channel_name_cache[channel_id] = name
    return name


# --- message shaping ------------------------------------------------


def _truncate_text(text: str) -> tuple[str, bool]:
    if len(text) <= _MESSAGE_TRUNCATE_CHARS:
        return text, False
    return text[:_MESSAGE_TRUNCATE_CHARS] + " ...[truncated]", True


_USER_MENTION_RE = re.compile(r"<@(U[A-Z0-9]+)(?:\|([^>]+))?>")
_CHANNEL_MENTION_RE = re.compile(r"<#(C[A-Z0-9]+)(?:\|([^>]+))?>")


async def _rewrite_slack_mentions(client: SlackClient, text: str) -> str:
    """Replace raw Slack mention markers with display-friendly names.

    `<@Uxxx>` -> `@Real Name`, `<#Cxxx>` -> `#channel-name`. Slack stores
    mentions as opaque IDs; the LLM needs the resolved name to produce
    answers a human can read. Best-effort: any resolution failure leaves
    the original marker intact so we don't silently drop information.
    """
    if not text:
        return text
    user_ids = {m.group(1) for m in _USER_MENTION_RE.finditer(text)}
    channel_ids = {m.group(1) for m in _CHANNEL_MENTION_RE.finditer(text)}
    user_map: dict[str, str] = {}
    for uid in user_ids:
        try:
            name = await client.resolve_user(uid)
            if name:
                user_map[uid] = name
        except Exception as exc:  # noqa: BLE001
            _log.debug("rewrite resolve_user(%s) failed: %s", uid, exc)
    channel_map: dict[str, str] = {}
    for cid in channel_ids:
        try:
            name = await _resolve_channel_name(client, cid)
            if name:
                channel_map[cid] = name
        except Exception as exc:  # noqa: BLE001
            _log.debug("rewrite resolve_channel(%s) failed: %s", cid, exc)

    def _u_sub(m: re.Match) -> str:
        uid = m.group(1)
        fallback = m.group(2)  # Slack sometimes inlines `<@Uxxx|name>`.
        name = user_map.get(uid) or fallback
        return f"@{name}" if name else m.group(0)

    def _c_sub(m: re.Match) -> str:
        cid = m.group(1)
        fallback = m.group(2)
        name = channel_map.get(cid) or fallback
        return f"#{name}" if name else m.group(0)

    return _CHANNEL_MENTION_RE.sub(_c_sub, _USER_MENTION_RE.sub(_u_sub, text))


def _format_reactions(raw: list[dict] | None) -> list[dict]:
    if not raw:
        return []
    return [
        {"name": r.get("name"), "count": int(r.get("count", 0))}
        for r in raw
        if r.get("name")
    ]


# Slack `<url|alias>` and `<url>` markup -- preserve URLs as bare text so
# the reasoner can extract them (matches the same regex used in
# parse_slack_url). Without unwrapping, the reasoner sees `<https://...>`
# inside fielded text and may not register it as a URL.
_LINK_RE = re.compile(r"<((?:https?|mailto)?:?//?[^|>]+)(?:\|([^>]+))?>")


def _unwrap_slack_links(text: str) -> str:
    """Replace `<https://x|label>` -> `label (https://x)` and `<https://x>` -> `https://x`.

    Keeps both the URL and any human label so the agent (and downstream
    URL extractors like the reasoner's URL gate) can read either. Idempotent
    on non-Slack-markup text.
    """
    if not text:
        return text

    def _sub(m: re.Match) -> str:
        url = m.group(1)
        alias = m.group(2)
        if alias and alias.strip() and alias.strip() != url:
            return f"{alias.strip()} ({url})"
        return url

    return _LINK_RE.sub(_sub, text)


def _flatten_rich_content(
    attachments: list[dict] | None,
    blocks: list[dict] | None,
) -> str:
    """Render Slack attachments + blocks into a single text blob.

    Why this exists: bot-posted messages (Rootly alerts, Prometheus
    Alertmanager forwarders, GitLab webhook bots, etc.) leave the
    top-level `text` field nearly empty and put the actionable content
    -- alert title, severity, service, runbook URL, Rootly link -- into
    `attachments[]` (legacy Slack API) or `blocks[]` (modern block-kit).
    The reasoner needs that content to chain into the right next tool
    (e.g. `rootly_get_alert` once it sees the Rootly URL).

    Output shape: a newline-separated text block, structured but plain
    text. We preserve URLs as bare strings (via `_unwrap_slack_links`)
    so URL-extracting regexes work on the result.
    """
    parts: list[str] = []

    # -- attachments[] (legacy) --
    for att in attachments or []:
        # Each attachment can carry pretext, title (+title_link), text,
        # fields[], footer, actions[] AND nested blocks[] (modern Rootly
        # / GitHub bots put almost everything into `attachments[].blocks[]`
        # -- the top-level `text`/`fields` are empty). We walk both shapes.
        pretext = (att.get("pretext") or "").strip()
        if pretext:
            parts.append(_unwrap_slack_links(pretext))
        title = (att.get("title") or "").strip()
        title_link = (att.get("title_link") or "").strip()
        if title:
            if title_link:
                parts.append(f"{title} ({title_link})")
            else:
                parts.append(title)
        body = (att.get("text") or "").strip()
        if body:
            parts.append(_unwrap_slack_links(body))
        for field in att.get("fields") or []:
            ft = (field.get("title") or "").strip()
            fv = (field.get("value") or "").strip()
            if ft and fv:
                parts.append(f"{ft}: {_unwrap_slack_links(fv)}")
            elif fv:
                parts.append(_unwrap_slack_links(fv))
        for action in att.get("actions") or []:
            a_text = (action.get("text") or "").strip()
            a_url = (action.get("url") or "").strip()
            if a_url:
                parts.append(f"[Action] {a_text or 'link'}: {a_url}")
        # Nested blocks INSIDE an attachment -- Rootly's standard shape.
        nested = _flatten_rich_content(None, att.get("blocks"))
        if nested:
            parts.append(nested)
        footer = (att.get("footer") or "").strip()
        if footer:
            parts.append(_unwrap_slack_links(footer))

    # -- blocks[] (block kit) --
    for blk in blocks or []:
        btype = blk.get("type") or ""
        if btype == "header":
            t = ((blk.get("text") or {}).get("text") or "").strip()
            if t:
                parts.append(f"## {_unwrap_slack_links(t)}")
        elif btype == "section":
            txt_obj = blk.get("text") or {}
            t = (txt_obj.get("text") or "").strip()
            if t:
                parts.append(_unwrap_slack_links(t))
            for f in blk.get("fields") or []:
                ft = (f.get("text") or "").strip()
                if ft:
                    parts.append(_unwrap_slack_links(ft))
            acc = blk.get("accessory") or {}
            if isinstance(acc, dict):
                acc_url = (acc.get("url") or "").strip()
                acc_text = (((acc.get("text") or {}).get("text")) or "").strip()
                if acc_url:
                    parts.append(f"[Button] {acc_text or 'link'}: {acc_url}")
        elif btype == "context":
            for el in blk.get("elements") or []:
                if isinstance(el, dict):
                    et = (el.get("text") or "").strip()
                    if et:
                        parts.append(_unwrap_slack_links(et))
        elif btype == "actions":
            for el in blk.get("elements") or []:
                if isinstance(el, dict):
                    et = ((el.get("text") or {}).get("text") or "").strip()
                    eu = (el.get("url") or "").strip()
                    if eu:
                        parts.append(f"[Button] {et or 'link'}: {eu}")
        elif btype == "rich_text":
            # rich_text -> elements[] -> elements[] of text/link items.
            for outer in blk.get("elements") or []:
                buf: list[str] = []
                for inner in (outer or {}).get("elements") or []:
                    it = inner.get("type") if isinstance(inner, dict) else ""
                    if it == "text":
                        buf.append(inner.get("text") or "")
                    elif it == "link":
                        url = (inner.get("url") or "").strip()
                        if url:
                            buf.append(url)
                if buf:
                    parts.append("".join(buf))

    # Dedupe consecutive identical lines (Slack alert posts often repeat
    # the title in pretext + title + first block) without reordering.
    out: list[str] = []
    for p in parts:
        if not out or out[-1] != p:
            out.append(p)
    return "\n".join(out).strip()


def _build_permalink(channel: str, ts: str, *, thread_ts: str | None = None) -> str | None:
    """Build a Slack web permalink. Returns ``None`` when no workspace
    URL is configured -- callers MUST handle the absent-link case
    (typically by emitting the channel/ts as plain text)."""
    base = _workspace_url()
    if not base:
        return None
    # Slack permalinks squash the dot: `1714939200.000100` -> `p1714939200000100`.
    ts_squash = "p" + ts.replace(".", "")
    url = f"{base}/archives/{channel}/{ts_squash}"
    if thread_ts:
        url += f"?thread_ts={thread_ts}&cid={channel}"
    return url


async def _shape_message(
    client: SlackClient,
    raw: dict,
    *,
    channel: str,
    channel_name: str | None,
) -> dict:
    """Denormalize a raw Slack message into the dict we return to the agent."""
    user_id = raw.get("user")
    bot_id = raw.get("bot_id")
    user_name: str | None = None
    if user_id:
        try:
            user_name = await client.resolve_user(user_id)
        except Exception as exc:  # noqa: BLE001
            _log.debug("resolve_user(%s) failed: %s", user_id, exc)
            user_name = None
    elif bot_id:
        # Bot messages can have a username field directly.
        user_name = raw.get("username")

    # Merge top-level `text` with anything in `attachments[]` / `blocks[]`.
    # Bot-posted messages (Rootly alerts, Prometheus Alertmanager, GitLab
    # webhooks) leave `text` nearly empty and put the actionable content
    # -- alert title, severity, service, Rootly URL -- into the rich
    # blocks. Without merging, the agent sees an empty body and can't
    # chain (e.g. into rootly_get_alert).
    fallback_text = (raw.get("text") or "").strip()
    rich_text = _flatten_rich_content(raw.get("attachments"), raw.get("blocks"))
    if rich_text and fallback_text and rich_text != fallback_text:
        combined = f"{fallback_text}\n{rich_text}"
    else:
        combined = rich_text or fallback_text
    combined = await _rewrite_slack_mentions(client, combined)
    text, truncated = _truncate_text(combined)
    ts = raw.get("ts", "")
    return {
        "channel": channel,
        "channel_name": channel_name,
        "ts": ts,
        "user": user_id,
        "user_name": user_name,
        "bot_id": bot_id,
        "text": text,
        "truncated": truncated,
        "permalink": _build_permalink(channel, ts, thread_ts=raw.get("thread_ts")),
        "thread_ts": raw.get("thread_ts"),
        "reply_count": int(raw.get("reply_count", 0)),
        "reactions": _format_reactions(raw.get("reactions")),
        "subtype": raw.get("subtype"),
    }


def _wrap_runtime_error(exc: RuntimeError, *, channel: str) -> SlackMCPError:
    """Translate the SlackClient's `RuntimeError("slack ... failed: <err>")`
    into a friendly SlackMCPError with a helpful `reason`."""
    msg = str(exc)
    low = msg.lower()
    if "not_in_channel" in low:
        return SlackMCPError(
            f"Bot is not a member of channel {channel}. Invite it with "
            f"`/invite @<bot-name>` from the Slack channel.",
            reason="not_in_channel",
        )
    if "channel_not_found" in low:
        return SlackMCPError(
            f"Channel {channel} not found (wrong ID or bot lacks visibility).",
            reason="channel_not_found",
        )
    if "missing_scope" in low:
        return SlackMCPError(
            f"Slack token is missing a required scope: {msg}",
            reason="missing_scope",
        )
    return SlackMCPError(f"Slack API error: {msg}", reason="slack_api_error")


# --- handlers -------------------------------------------------------


async def _h_get_message_by_url(_unused, args: dict) -> Any:
    parsed = parse_slack_url(args["url"])
    bot_token = _resolve_bot_token()
    client = SlackClient(bot_token=bot_token)
    try:
        try:
            data = await client._get(  # noqa: SLF001
                "conversations.history",
                {
                    "channel": parsed.channel,
                    "latest": parsed.ts,
                    "oldest": parsed.ts,
                    "inclusive": "true",
                    "limit": 1,
                },
            )
        except RuntimeError as exc:
            raise _wrap_runtime_error(exc, channel=parsed.channel) from exc
        messages = data.get("messages") or []
        if not messages:
            raise SlackMCPError(
                f"No message found at {parsed.original_url}. The message "
                f"may have been deleted, or the bot lacks access to channel "
                f"{parsed.channel}.",
                reason="message_not_found",
            )
        channel_name = await _resolve_channel_name(client, parsed.channel)
        shaped = await _shape_message(
            client, messages[0], channel=parsed.channel, channel_name=channel_name,
        )
        # Preserve the original permalink from the user (includes any
        # query params they pasted, e.g. thread_ts).
        shaped["permalink_original"] = parsed.original_url
        return {"message": shaped}
    finally:
        await client.close()


async def _h_get_thread_by_url(_unused, args: dict) -> Any:
    parsed = parse_slack_url(args["url"])
    bot_token = _resolve_bot_token()
    client = SlackClient(bot_token=bot_token)
    try:
        # Determine the thread root timestamp: prefer the `thread_ts`
        # query param if present (it identifies the root); otherwise use
        # the message ts itself (caller may have linked the root).
        root_ts = parsed.thread_ts or parsed.ts
        try:
            thread = await client.get_thread(parsed.channel, root_ts)
        except RuntimeError as exc:
            raise _wrap_runtime_error(exc, channel=parsed.channel) from exc

        channel_name = await _resolve_channel_name(client, parsed.channel)

        async def _shape_internal_msg(m) -> dict:
            user_id = m.user_id
            user_name: str | None = None
            if user_id:
                try:
                    user_name = await client.resolve_user(user_id)
                except Exception:  # noqa: BLE001
                    user_name = None
            raw_text = await _rewrite_slack_mentions(client, m.text)
            text, truncated = _truncate_text(raw_text)
            return {
                "ts": m.ts,
                "user": user_id,
                "user_name": user_name,
                "bot_id": m.bot_id,
                "text": text,
                "truncated": truncated,
                "subtype": m.subtype,
                "thread_ts": m.thread_ts,
                "permalink": _build_permalink(parsed.channel, m.ts, thread_ts=root_ts),
            }

        root_shaped = await _shape_internal_msg(thread.root)
        replies_shaped = [await _shape_internal_msg(r) for r in thread.replies]
        return {
            "channel": parsed.channel,
            "channel_name": channel_name,
            "thread_ts": root_ts,
            "root": root_shaped,
            "replies": replies_shaped,
            "reply_count": len(replies_shaped),
            "permalink_original": parsed.original_url,
        }
    finally:
        await client.close()


async def _h_search_messages(_unused, args: dict) -> Any:
    """`search.messages` -- requires a USER token (search:read).

    The bot token can't call this endpoint; if `SLACK_USER_TOKEN` isn't
    set we return a structured error rather than a silent 0-result list.
    """
    query = (args.get("query") or "").strip()
    if not query:
        raise SlackMCPError("`query` is required.", reason="bad_args")
    # Strip ASCII + curly quotes. Slack treats ``"foo bar"`` as exact-phrase
    # match, which is almost never what the user wants -- when they say
    # ``search for "sms hung"`` they mean "find messages about sms hangs",
    # not "find exact substring sms<space>hung". Casual punctuation
    # routinely costs 0 results on otherwise-good queries. If we ever
    # need explicit phrase match, add a separate ``phrase: true`` arg.
    query = query.translate(str.maketrans("", "", '"""'''))
    query = " ".join(query.split())  # collapse whitespace after strip
    channel_hint = (args.get("channel") or "").strip()
    limit = max(1, min(int(args.get("limit") or _DEFAULT_SEARCH_LIMIT), _MAX_SEARCH_LIMIT))

    user_token = _resolve_user_token()
    if not user_token:
        return {
            "query": query,
            "messages": [],
            "warning": (
                "search.messages requires a user token with `search:read` scope. "
                "Set SLACK_USER_TOKEN (xoxp-...) or OPSRAG_SLACK_USER_TOKEN to enable search."
            ),
            "reason": "no_user_token",
        }

    # Bot token still used for the cached channel-name resolution.
    bot_token = _resolve_bot_token()
    bot_client = SlackClient(bot_token=bot_token)

    # Channel hint syntax: Slack search supports `in:#channel-name`.
    full_query = query
    if channel_hint:
        # Accept either a Slack channel ID (e.g. `C0XXXXXXXXX`), `#sre-alerts`, or `sre-alerts`.
        if channel_hint.startswith("C") and channel_hint[1:].isalnum() and channel_hint.isupper():
            name = await _resolve_channel_name(bot_client, channel_hint)
            if name:
                full_query = f"in:#{name} {query}"
            else:
                full_query = f"in:<{channel_hint}> {query}"
        else:
            cleaned = channel_hint.lstrip("#")
            full_query = f"in:#{cleaned} {query}"

    async with httpx.AsyncClient(
        base_url="https://slack.com/api",
        headers={"Authorization": f"Bearer {user_token}"},
        timeout=httpx.Timeout(30.0, connect=10.0),
    ) as http:
        resp = await http.get(
            "/search.messages",
            params={"query": full_query, "count": limit, "sort": "timestamp", "sort_dir": "desc"},
        )
    if resp.status_code >= 400:
        await bot_client.close()
        raise SlackMCPError(
            f"search.messages HTTP {resp.status_code}: {resp.text[:300]}",
            reason="slack_api_error",
        )
    data = resp.json()
    if not data.get("ok"):
        await bot_client.close()
        raise SlackMCPError(
            f"search.messages failed: {data.get('error', 'unknown')}",
            reason="slack_api_error",
        )

    matches = ((data.get("messages") or {}).get("matches")) or []
    results: list[dict] = []
    try:
        for m in matches[:limit]:
            ch_obj = m.get("channel") or {}
            ch_id = ch_obj.get("id", "")
            ts = m.get("ts", "")
            raw_text = await _rewrite_slack_mentions(bot_client, m.get("text") or "")
            text, truncated = _truncate_text(raw_text)
            user_id = m.get("user")
            user_name: str | None = None
            if user_id:
                try:
                    user_name = await bot_client.resolve_user(user_id)
                except Exception:  # noqa: BLE001
                    user_name = None
            permalink = m.get("permalink") or _build_permalink(ch_id, ts)
            results.append({
                "ts": ts,
                "channel": ch_id,
                "channel_name": ch_obj.get("name"),
                "user": user_id,
                "user_name": user_name or m.get("username"),
                "text": text,
                "truncated": truncated,
                "permalink": permalink,
            })
    finally:
        await bot_client.close()
    return {
        "query": full_query,
        "total": (data.get("messages") or {}).get("total"),
        "count": len(results),
        "messages": results,
    }


async def _h_list_channels(_unused, args: dict) -> Any:
    bot_token = _resolve_bot_token()
    client = SlackClient(bot_token=bot_token)
    try:
        channels = await _list_channels_cached(client)
    finally:
        await client.close()
    needle = (args.get("name_substring") or "").strip().lstrip("#").lower()
    if needle:
        filtered = [c for c in channels if needle in (c.get("name") or "").lower()]
    else:
        filtered = list(channels)
    # Sort: bot is a member first (most useful for follow-up history calls), then by name.
    filtered.sort(key=lambda c: (not c.get("is_member"), c.get("name") or ""))
    return {
        "count": len(filtered),
        "total": len(channels),
        "channels": filtered[:200],  # hard cap on response size
    }


# --- tool registry --------------------------------------------------


SLACK_TOOLS: tuple[MCPTool, ...] = (
    MCPTool(
        name="slack_get_message_by_url",
        description=(
            "Fetch a single Slack message by its permalink. Use this when "
            "the user pastes a Slack URL like "
            "https://example.slack.com/archives/C0XXXXXXXXX/p1778658682028889 "
            "and wants its content. Returns the message body (truncated to ~2000 chars), "
            "author (with display name), timestamp, reactions, and a thread_ts if "
            "the message is in a thread. If the bot isn't in the channel, returns a "
            "clear 'not_in_channel' error -- do NOT retry, surface it to the user."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Slack permalink. Accepts example.slack.com/archives/<CHAN>/p<TS> with or without scheme; also app.slack.com/client/<TEAM>/<CHAN>/p<TS>; thread query params (?thread_ts=...&cid=...) are extracted automatically.",
                },
            },
            "required": ["url"],
        },
        handler=_h_get_message_by_url,
    ),
    MCPTool(
        name="slack_get_thread_by_url",
        description=(
            "Fetch a Slack thread (root + all replies) by permalink. Use when the user "
            "wants the full discussion around a Slack URL, not just the single message. "
            "If the URL has ?thread_ts=... the thread root is taken from there; otherwise "
            "the message at the URL is treated as the root and all its replies are returned."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Slack permalink (same formats as slack_get_message_by_url).",
                },
            },
            "required": ["url"],
        },
        handler=_h_get_thread_by_url,
    ),
    MCPTool(
        name="slack_search_messages",
        description=(
            "Free-text search across Slack messages the bot/user token can see. "
            "REQUIRES a user token (SLACK_USER_TOKEN with search:read scope) -- if "
            "absent, returns a 'no_user_token' warning. Use to find prior discussion "
            "of a topic across channels. Scope to one channel with `channel` (id like "
            "C0... or name like 'sre-alerts')."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query. Slack's search syntax works (from:@user, has:link, etc.).",
                },
                "channel": {
                    "type": "string",
                    "description": "Optional channel scope: channel id (C0...), `#name`, or bare name.",
                },
                "limit": {
                    "type": "number",
                    "description": f"Max results, default {_DEFAULT_SEARCH_LIMIT}, hard cap {_MAX_SEARCH_LIMIT}.",
                },
            },
            "required": ["query"],
        },
        handler=_h_search_messages,
    ),
    MCPTool(
        name="slack_list_channels",
        description=(
            "List Slack channels visible to the bot. Useful for resolving a channel "
            "name -> id before drilling into history. Filter with `name_substring` "
            "(case-insensitive). Cached for 5 minutes per process."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name_substring": {
                    "type": "string",
                    "description": "Optional substring (case-insensitive). E.g. 'sre' matches sre-alerts, sre-oncall.",
                },
            },
        },
        handler=_h_list_channels,
    ),
)


def get_tool(name: str) -> MCPTool:
    for t in SLACK_TOOLS:
        if t.name == name:
            return t
    raise KeyError(f"unknown slack tool: {name}")


# --- fake backend (FR-012; integration tests) ----------------------
#
# Slack handlers ignore their first (`_unused`) arg and reach back into
# the module: they call `_resolve_bot_token()` then build a real
# `SlackClient`, and `slack_list_channels` reaches the module-level
# channel cache. So the offline fake swaps `_resolve_bot_token` for a
# dummy and `SlackClient` for a canned, shape-faithful stand-in -- no
# network, no token. `build_fake()` returns client=None (handlers
# discard it) plus a teardown that restores the real names and clears
# the per-process caches.


class _FakeSlackClient:
    """Offline stand-in for SlackClient. Mirrors only the surface the
    MCP handlers touch (`_get`, `get_thread`, `get_channel`,
    `resolve_user`, `close`) and returns canned, shape-faithful Slack
    Web API responses. No network."""

    def __init__(self, bot_token: str | None = None, **_kwargs: Any) -> None:
        self.bot_token = bot_token
        self.calls: list[tuple[str, dict]] = []

    async def close(self) -> None:  # parity with SlackClient
        return None

    async def _get(self, method: str, params: dict | None = None) -> dict:
        self.calls.append((method, dict(params or {})))
        if method == "conversations.list":
            return {
                "channels": [
                    {
                        "id": "C0000000001",
                        "name": "sre-alerts",
                        "is_member": True,
                        "is_private": False,
                        "is_archived": False,
                        "num_members": 42,
                        "topic": {"value": "Production alerts"},
                        "purpose": {"value": "SRE on-call signal"},
                    },
                    {
                        "id": "C0000000002",
                        "name": "general",
                        "is_member": False,
                        "is_private": False,
                        "is_archived": False,
                        "num_members": 120,
                        "topic": {"value": ""},
                        "purpose": {"value": ""},
                    },
                ],
                "response_metadata": {"next_cursor": ""},
            }
        if method == "conversations.history":
            return {
                "messages": [
                    {
                        "ts": params.get("latest", "1700000000.000000") if params else "1700000000.000000",
                        "user": "U0000000001",
                        "text": "Deploy of service-a finished, see <@U0000000002>.",
                        "thread_ts": None,
                        "reply_count": 0,
                        "reactions": [{"name": "white_check_mark", "count": 2}],
                    }
                ],
            }
        if method == "users.info":
            uid = (params or {}).get("user", "U0000000000")
            return {
                "user": {
                    "id": uid,
                    "name": "canned-user",
                    "profile": {"display_name": "Canned User", "real_name": "Canned User"},
                }
            }
        if method == "conversations.info":
            cid = (params or {}).get("channel", "C0000000001")
            return {
                "channel": {
                    "id": cid,
                    "name": "sre-alerts",
                    "is_member": True,
                    "is_private": False,
                    "num_members": 42,
                }
            }
        return {"ok": True}

    async def get_channel(self, channel_id: str) -> ChannelInfo:
        return ChannelInfo(
            id=channel_id,
            name="sre-alerts",
            is_member=True,
            is_private=False,
            num_members=42,
        )

    async def resolve_user(self, user_id: str) -> str:
        if not user_id:
            return ""
        return "Canned User"

    async def get_thread(self, channel_id: str, thread_ts: str) -> SlackThread:
        root = SlackMessage(
            ts=thread_ts,
            user_id="U0000000001",
            bot_id=None,
            text="Root message about the incident.",
            subtype=None,
            thread_ts=thread_ts,
            reply_count=1,
            reply_users_count=1,
        )
        reply = SlackMessage(
            ts="1700000000.000200",
            user_id="U0000000002",
            bot_id=None,
            text="Reply: mitigated, watching dashboards.",
            subtype=None,
            thread_ts=thread_ts,
            reply_count=0,
            reply_users_count=0,
        )
        return SlackThread(channel_id=channel_id, root=root, replies=(reply,))


def build_fake():
    """Return a FakeMCP exposing the Slack tools wired to an offline
    backend. Needs NO Slack token / network: `_resolve_bot_token` and
    `SlackClient` are swapped for canned stand-ins and restored by
    `teardown`, which also clears the per-process channel caches.

    Only the registry-declared tools are exposed (the registry omits
    `slack_search_messages`, which requires a user token and an online
    search endpoint), so the fake's tool set matches REGISTRY["slack"]
    exactly per the FR-012 contract test."""
    import opsrag.mcp.slack as _mod
    from opsrag.mcp._fake import FakeMCP
    from opsrag.mcp.registry import REGISTRY
    from opsrag.sources.slack.client import (
        ChannelInfo as _ChannelInfo,
    )
    from opsrag.sources.slack.client import (
        SlackMessage as _SlackMessage,
    )
    from opsrag.sources.slack.client import (
        SlackThread as _SlackThread,
    )

    # Bind the client dataclasses into the module namespace so the fake
    # client (defined above) can reference ChannelInfo / SlackMessage /
    # SlackThread without a separate import line.
    global ChannelInfo, SlackMessage, SlackThread
    ChannelInfo = _ChannelInfo
    SlackMessage = _SlackMessage
    SlackThread = _SlackThread

    _orig_resolve = _mod._resolve_bot_token
    _orig_client = _mod.SlackClient
    _orig_cache = dict(_mod._channels_cache)
    _orig_name_cache = dict(_mod._channel_name_cache)

    _mod._resolve_bot_token = lambda: "xoxb-fake-token"
    _mod.SlackClient = _FakeSlackClient
    # Start from a clean cache so canned channels are always recomputed.
    _mod._channels_cache = {"ts": 0.0, "channels": []}
    _mod._channel_name_cache = {}

    def _restore() -> None:
        _mod._resolve_bot_token = _orig_resolve
        _mod.SlackClient = _orig_client
        _mod._channels_cache = _orig_cache
        _mod._channel_name_cache = _orig_name_cache

    declared = set(REGISTRY["slack"].tool_names)
    tools = [t for t in SLACK_TOOLS if t.name in declared]
    return FakeMCP(tools=tools, client=None, teardown=_restore)
