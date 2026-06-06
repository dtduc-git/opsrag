"""Slack Web API client -- async, retry-aware, paginated.

Scope: read-only. We never call any write endpoint, no chat.postMessage,
no reactions, nothing. Just `auth.test`, `conversations.info`,
`conversations.history`, `conversations.replies`, `users.info`.

Rate limits per Slack docs (Web API tier 3 for the history endpoints):
~50 requests / minute / workspace. We cap concurrency at 3 by default
and back off exponentially on `Retry-After`.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

_log = logging.getLogger("opsrag.slack.client")

_BASE = "https://slack.com/api"


@dataclass(frozen=True)
class ChannelInfo:
    id: str
    name: str
    is_member: bool
    is_private: bool
    num_members: int


@dataclass(frozen=True)
class SlackMessage:
    ts: str                  # canonical message timestamp ("1714939200.000100")
    user_id: str | None      # None for bot messages or system events
    bot_id: str | None
    text: str                # raw text -- may contain Slack formatting (<@U...>, <#C...>)
    subtype: str | None      # e.g. "channel_join", "bot_message"
    thread_ts: str | None    # ts of the thread root, or None if not a thread
    reply_count: int         # 0 if no replies
    reply_users_count: int


@dataclass(frozen=True)
class SlackThread:
    """A thread = root message + ordered replies (oldest first)."""

    channel_id: str
    root: SlackMessage
    replies: tuple[SlackMessage, ...]

    @property
    def thread_ts(self) -> str:
        # Slack uses the root message's ts as the thread identifier.
        return self.root.thread_ts or self.root.ts

    @property
    def latest_ts(self) -> str:
        return self.replies[-1].ts if self.replies else self.root.ts


class SlackClient:
    def __init__(
        self,
        bot_token: str,
        *,
        max_retries: int = 3,
        retry_base_seconds: float = 2.0,
    ):
        if not bot_token or not bot_token.startswith("xoxb-"):
            raise ValueError("SLACK_BOT_TOKEN must be a bot token (xoxb-...)")
        self._token = bot_token
        self._max_retries = max_retries
        self._retry_base = retry_base_seconds
        self._http = httpx.AsyncClient(
            base_url=_BASE,
            headers={"Authorization": f"Bearer {bot_token}"},
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
        # Cached user_id -> display_name resolution. Populated on demand
        # by `_resolve_user`. Slack user IDs are stable across renames
        # so this cache is safe for the lifetime of the process.
        self._user_cache: dict[str, str] = {}

    async def close(self) -> None:
        await self._http.aclose()

    # -- core request with retry on 429 + transient 5xx ----------------
    async def _get(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = await self._http.get(f"/{method}", params=params or {})
            except httpx.RequestError as exc:
                if attempt > self._max_retries:
                    raise
                wait = self._retry_base * (2 ** (attempt - 1))
                _log.warning("slack %s transport error: %s -- retry %d in %.1fs", method, exc, attempt, wait)
                await asyncio.sleep(wait)
                continue

            if resp.status_code == 429:
                # Slack returns Retry-After in seconds.
                wait = float(resp.headers.get("Retry-After", self._retry_base * (2 ** (attempt - 1))))
                _log.warning("slack %s rate-limited -- sleeping %.1fs", method, wait)
                await asyncio.sleep(wait)
                if attempt > self._max_retries:
                    resp.raise_for_status()
                continue

            if resp.status_code >= 500 and attempt <= self._max_retries:
                wait = self._retry_base * (2 ** (attempt - 1))
                _log.warning("slack %s status=%d -- retry %d in %.1fs", method, resp.status_code, attempt, wait)
                await asyncio.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok"):
                err = data.get("error", "unknown")
                # Slack returns ok=false for application-level errors.
                # `not_in_channel` and `channel_not_found` are caller
                # mistakes -- surface them clearly so the operator knows
                # to invite the bot or fix the channel ID.
                raise RuntimeError(f"slack {method} failed: {err}")
            return data

    # -- high-level methods ---------------------------------------------
    async def auth_test(self) -> dict[str, Any]:
        return await self._get("auth.test")

    async def get_channel(self, channel_id: str) -> ChannelInfo:
        data = await self._get("conversations.info", {"channel": channel_id})
        c = data["channel"]
        return ChannelInfo(
            id=c["id"],
            name=c.get("name", ""),
            is_member=bool(c.get("is_member")),
            is_private=bool(c.get("is_private")),
            num_members=int(c.get("num_members", 0)),
        )

    async def list_thread_roots(
        self,
        channel_id: str,
        *,
        oldest: datetime | None = None,
        latest: datetime | None = None,
    ) -> AsyncIterator[SlackMessage]:
        """Yield top-level messages in the channel within the time
        window. Replies live on `conversations.replies` and are fetched
        on-demand when a thread is hydrated.

        `oldest` is inclusive; `latest` is exclusive. Both are converted
        to Slack's float-seconds timestamp format.
        """
        cursor: str | None = None
        params: dict[str, Any] = {"channel": channel_id, "limit": 200}
        if oldest is not None:
            params["oldest"] = f"{oldest.timestamp():.6f}"
        if latest is not None:
            params["latest"] = f"{latest.timestamp():.6f}"
        while True:
            if cursor:
                params["cursor"] = cursor
            data = await self._get("conversations.history", params)
            for raw in data.get("messages", []):
                # Skip thread replies that show up at top level via
                # `thread_broadcast` subtype -- they'll be re-fetched as
                # part of their parent's reply list.
                if raw.get("subtype") == "thread_broadcast":
                    continue
                yield _to_message(raw)
            cursor = data.get("response_metadata", {}).get("next_cursor") or None
            if not cursor:
                break

    async def get_thread(self, channel_id: str, thread_ts: str) -> SlackThread:
        """Fetch the full thread (root + all replies) for `thread_ts`."""
        cursor: str | None = None
        replies: list[SlackMessage] = []
        root: SlackMessage | None = None
        while True:
            params: dict[str, Any] = {
                "channel": channel_id,
                "ts": thread_ts,
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor
            data = await self._get("conversations.replies", params)
            for raw in data.get("messages", []):
                msg = _to_message(raw)
                if msg.ts == thread_ts:
                    root = msg
                else:
                    replies.append(msg)
            cursor = data.get("response_metadata", {}).get("next_cursor") or None
            if not cursor:
                break
        if root is None:
            # Slack occasionally omits the root if it was deleted but
            # replies survive. Synthesize a placeholder so the caller
            # can still produce a coherent doc.
            root = SlackMessage(
                ts=thread_ts, user_id=None, bot_id=None,
                text="(message deleted)", subtype="tombstone",
                thread_ts=thread_ts, reply_count=len(replies),
                reply_users_count=0,
            )
        return SlackThread(channel_id=channel_id, root=root, replies=tuple(replies))

    async def resolve_user(self, user_id: str) -> str:
        """Return a display name for a user_id, cached. Falls back to
        the raw ID if the API rejects the lookup (deactivated user,
        deleted account, scope mismatch)."""
        if not user_id:
            return ""
        cached = self._user_cache.get(user_id)
        if cached is not None:
            return cached
        try:
            data = await self._get("users.info", {"user": user_id})
            u = data.get("user") or {}
            profile = u.get("profile") or {}
            name = (
                profile.get("display_name")
                or profile.get("real_name")
                or u.get("name")
                or user_id
            )
        except Exception as exc:
            _log.debug("slack users.info(%s) failed: %s", user_id, exc)
            name = user_id
        self._user_cache[user_id] = name
        return name


# -- helpers -----------------------------------------------------------
def _to_message(raw: dict[str, Any]) -> SlackMessage:
    return SlackMessage(
        ts=raw.get("ts", ""),
        user_id=raw.get("user"),
        bot_id=raw.get("bot_id"),
        text=raw.get("text") or "",
        subtype=raw.get("subtype"),
        thread_ts=raw.get("thread_ts"),
        reply_count=int(raw.get("reply_count", 0)),
        reply_users_count=int(raw.get("reply_users_count", 0)),
    )


def slack_ts_to_datetime(ts: str) -> datetime:
    """Slack timestamps are 'seconds.microseconds' epoch floats."""
    try:
        return datetime.fromtimestamp(float(ts), tz=UTC)
    except (ValueError, TypeError):
        return datetime.fromtimestamp(0, tz=UTC)
