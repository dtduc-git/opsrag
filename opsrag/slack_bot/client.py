"""Socket Mode wrapper around `slack_sdk` for the OpsRAG chatbot.

Long-lived singleton per process. Owns:
  - the AsyncWebClient used for chat.postMessage / chat.update / etc.
  - the AsyncSocketModeClient that pulls events off the websocket
  - an in-memory user_id -> users.info cache (no TTL; pod lifetime)

`start(dispatcher)` registers a listener that filters out bot
messages and routes `app_mention` / `message.im` events to the
sister-agent's dispatcher. The dispatcher type is intentionally
`Any` -- we MUST NOT import the sister module to avoid the circular
import that would happen during package init.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from slack_sdk.socket_mode.aiohttp import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.web.async_client import AsyncWebClient

_log = logging.getLogger("opsrag.slack_bot.client")

# Cap on in-memory user-info cache. A single pod's lifetime is short
# enough that we won't see millions of distinct users; 5000 is a
# generous ceiling for a typical workspace.
_USER_CACHE_MAX = 5000


class SlackBotClient:
    """Long-lived Socket Mode connection. Single instance per process."""

    def __init__(self, bot_token: str, app_token: str) -> None:
        if not bot_token or not bot_token.startswith("xoxb-"):
            raise ValueError("bot_token must be a Slack bot token (xoxb-...)")
        if not app_token or not app_token.startswith("xapp-"):
            raise ValueError("app_token must be a Slack app-level token (xapp-...)")
        self._bot_token = bot_token
        self._app_token = app_token
        # Lazily constructed in start() so that unit tests can patch the
        # constructors at module level and inject mocks. We expose the
        # attributes on `self` so tests can also inject directly.
        self._web: AsyncWebClient | None = None
        self._socket: SocketModeClient | None = None
        self._dispatcher: Any = None
        # FIFO-evicted user cache. Use a list for the eviction order so
        # we don't need an OrderedDict (dict is insertion-ordered in
        # 3.7+, but we want explicit semantics for testability).
        self._user_cache: dict[str, dict] = {}
        self._user_cache_order: list[str] = []
        self._connected = asyncio.Event()
        # Populated by start() via auth.test. Used by thread_context to
        # filter out the bot's OWN previous replies (so the agent doesn't
        # read its own history as user context -> feedback loop) while
        # KEEPING messages from other bots (Rootly, Datadog, Prometheus
        # alerts), which are exactly the context users want to triage.
        self.self_user_id: str | None = None
        self.self_bot_id: str | None = None

    # -- lifecycle --------------------------------------------------

    async def start(self, dispatcher: Any) -> None:
        """Connect to Slack via Socket Mode and register the event listener.

        `dispatcher` is the sister-agent's SlackEventDispatcher; we type
        it as Any to avoid a circular import at package init.
        Returns once the websocket handshake completes.
        """
        if self._socket is not None:
            _log.warning("SlackBotClient.start() called twice -- ignoring second call")
            return
        self._dispatcher = dispatcher
        self._web = AsyncWebClient(token=self._bot_token)
        self._socket = SocketModeClient(
            app_token=self._app_token,
            web_client=self._web,
        )
        self._socket.socket_mode_request_listeners.append(self._on_request)
        _log.info("Slack bot connecting via Socket Mode")
        await self._socket.connect()
        self._connected.set()
        # Capture our own identity so thread_context can distinguish
        # "our past reply" from "Rootly alert" -- both are technically
        # bot_message, but we only want to filter ours.
        try:
            auth = await self._web.auth_test()
            self.self_user_id = auth.get("user_id")
            self.self_bot_id = auth.get("bot_id")
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "auth.test failed during start -- self-identity unavailable, "
                "thread_context will fall back to permissive filtering: %s",
                exc,
            )
        _log.info(
            "Slack bot connected via Socket Mode (self_user=%s bot_id=%s)",
            self.self_user_id, self.self_bot_id,
        )

    async def stop(self) -> None:
        """Graceful shutdown -- close websocket. Idempotent."""
        if self._socket is None:
            return
        _log.info("Slack bot stopping")
        try:
            await self._socket.close()
        except Exception as exc:  # noqa: BLE001
            _log.warning("error closing socket mode client: %s", exc)
        if self._web is not None:
            # AsyncWebClient uses an aiohttp session that is owned
            # internally; closing it cleans up the connector.
            close = getattr(self._web, "close", None)
            if callable(close):
                try:
                    await close()
                except Exception as exc:  # noqa: BLE001
                    _log.debug("error closing web client: %s", exc)
        self._socket = None
        self._web = None
        self._connected.clear()
        _log.info("Slack bot stopped")

    # -- dispatch ---------------------------------------------------

    async def _on_request(
        self, client: SocketModeClient, req: SocketModeRequest
    ) -> None:
        """Top-level event filter.

        Slack delivers all event types over the same websocket. We must:
          1. Ack EVERY request (even ones we ignore) so Slack stops
             re-delivering.
          2. Skip messages from bots (subtype=bot_message OR bot_id set)
             to avoid feedback loops with our own posts.
          3. Route app_mention -> dispatcher.on_app_mention
             and message.im -> dispatcher.on_message_im
        """
        # 1. Always ack first. Slack retries up to 3x if we don't ack
        # within 3s -- and an unhandled exception elsewhere should never
        # turn into a flood of duplicate requests.
        try:
            await client.send_socket_mode_response(
                SocketModeResponse(envelope_id=req.envelope_id)
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("failed to ack envelope %s: %s", req.envelope_id, exc)

        # Interactive requests (block_actions from 👍 Helpful / 👎 Wrong
        # buttons attached to bot answers) arrive on the SAME websocket
        # but with `req.type == "interactive"` and NO `event` wrapper --
        # the payload IS the action body. Route to the dispatcher's
        # block-action handler. Without this branch the buttons were
        # silently ack'd-and-dropped (real failure mode observed
        # 2026-05-26 -- buttons rendered, clicks recorded nothing).
        if req.type == "interactive":
            payload = req.payload or {}
            if payload.get("type") == "block_actions" and self._dispatcher is not None:
                try:
                    await self._dispatcher.on_block_action(payload)
                except Exception:  # noqa: BLE001
                    _log.exception(
                        "dispatcher raised for block_actions user=%s action=%s",
                        ((payload.get("user") or {}).get("id")),
                        ((payload.get("actions") or [{}])[0].get("action_id")),
                    )
            return

        if req.type != "events_api":
            return
        payload = req.payload or {}
        event = (payload.get("event") or {})
        if not event:
            return

        event_type = event.get("type")

        # 2. Bot-message filter. Always skip our own / other bots'
        # messages to avoid feedback loops.
        if event.get("subtype") == "bot_message" or event.get("bot_id"):
            return

        # 3. Route.
        try:
            if event_type == "app_mention" and self._dispatcher is not None:
                await self._dispatcher.on_app_mention(event)
            elif (
                event_type == "message"
                and event.get("channel_type") == "im"
                and self._dispatcher is not None
            ):
                await self._dispatcher.on_message_im(event)
        except Exception:  # noqa: BLE001
            # The dispatcher owns its own error reporting (posts a
            # friendly message back to Slack). We just log + swallow so
            # one bad event doesn't kill the websocket loop.
            _log.exception(
                "dispatcher raised for event_type=%s channel=%s user=%s",
                event_type,
                event.get("channel"),
                event.get("user"),
            )

    # -- chat APIs --------------------------------------------------

    async def post_message(
        self,
        channel: str,
        text: str,
        thread_ts: str | None = None,
        blocks: list | None = None,
    ) -> str:
        """Post a message. Returns its `ts` (Slack message timestamp ID).

        Caller uses the returned ts for later chat.update calls.
        """
        if self._web is None:
            raise RuntimeError("SlackBotClient.start() must be called before post_message")
        kwargs: dict[str, Any] = {"channel": channel, "text": text}
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        if blocks is not None:
            kwargs["blocks"] = blocks
        _log.debug(
            "chat.postMessage channel=%s thread_ts=%s len=%d",
            channel, thread_ts, len(text or ""),
        )
        resp = await self._web.chat_postMessage(**kwargs)
        ts = resp.get("ts") if isinstance(resp, dict) else getattr(resp, "data", {}).get("ts")
        if not ts:
            raise RuntimeError(f"chat.postMessage returned no ts: {resp!r}")
        return str(ts)

    async def update_message(
        self,
        channel: str,
        ts: str,
        text: str,
        blocks: list | None = None,
    ) -> None:
        """Edit-in-place via chat.update."""
        if self._web is None:
            raise RuntimeError("SlackBotClient.start() must be called before update_message")
        kwargs: dict[str, Any] = {"channel": channel, "ts": ts, "text": text}
        if blocks is not None:
            kwargs["blocks"] = blocks
        _log.debug(
            "chat.update channel=%s ts=%s len=%d", channel, ts, len(text or ""),
        )
        await self._web.chat_update(**kwargs)

    async def add_reaction(self, channel: str, ts: str, emoji: str) -> None:
        """Best-effort reaction. Swallows errors -- reactions are UX
        nice-to-have, not load-bearing."""
        if self._web is None:
            _log.warning("add_reaction called before start(); skipping")
            return
        try:
            await self._web.reactions_add(channel=channel, timestamp=ts, name=emoji)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "reactions.add failed channel=%s ts=%s emoji=%s: %s",
                channel, ts, emoji, exc,
            )

    async def fetch_thread_replies(
        self, channel: str, thread_ts: str, limit: int = 30,
    ) -> list[dict]:
        """conversations.replies -- returns the raw `messages` array."""
        if self._web is None:
            raise RuntimeError(
                "SlackBotClient.start() must be called before fetch_thread_replies"
            )
        _log.debug(
            "conversations.replies channel=%s thread_ts=%s limit=%d",
            channel, thread_ts, limit,
        )
        resp = await self._web.conversations_replies(
            channel=channel, ts=thread_ts, limit=limit,
        )
        # AsyncSlackResponse behaves like a dict; both forms supported
        # so tests can hand back a plain dict mock.
        if isinstance(resp, dict):
            messages = resp.get("messages") or []
        else:
            messages = resp.get("messages") if hasattr(resp, "get") else (resp.data or {}).get("messages") or []
        return list(messages or [])

    async def get_user_info(self, user_id: str) -> dict:
        """users.info -- resolve user_id -> display name + email.

        Cached in-memory for the lifetime of the process. Cache is
        FIFO-evicted at `_USER_CACHE_MAX` entries to bound memory.
        """
        if not user_id:
            return {}
        cached = self._user_cache.get(user_id)
        if cached is not None:
            return cached
        if self._web is None:
            raise RuntimeError(
                "SlackBotClient.start() must be called before get_user_info"
            )
        try:
            resp = await self._web.users_info(user=user_id)
        except Exception as exc:  # noqa: BLE001
            _log.debug("users.info(%s) failed: %s", user_id, exc)
            # Cache the negative result too -- within a pod's lifetime a
            # deleted/deactivated user won't come back.
            empty: dict = {}
            self._cache_user(user_id, empty)
            return empty
        user = resp.get("user") if isinstance(resp, dict) else resp.get("user")
        info: dict = dict(user) if user else {}
        self._cache_user(user_id, info)
        return info

    def _cache_user(self, user_id: str, info: dict) -> None:
        """Insert into the FIFO cache, evicting the oldest entry if at cap."""
        if user_id in self._user_cache:
            # Already present (race) -- leave order untouched.
            self._user_cache[user_id] = info
            return
        if len(self._user_cache_order) >= _USER_CACHE_MAX:
            oldest = self._user_cache_order.pop(0)
            self._user_cache.pop(oldest, None)
        self._user_cache[user_id] = info
        self._user_cache_order.append(user_id)
