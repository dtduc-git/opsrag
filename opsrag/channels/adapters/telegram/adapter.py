"""``TelegramAdapter`` -- the Telegram implementation of the ``ChannelAdapter`` port.

Telegram's **Bot API** is a plain HTTPS JSON API, so -- unlike Slack (Socket
Mode SDK) or Discord (gateway SDK) -- this adapter needs no vendor SDK at all:
it speaks the Bot API directly with ``httpx`` (already a project dependency).
The single network seam is :meth:`TelegramAdapter._api`, an ``async`` helper
that POSTs ``{method, payload}`` to ``https://api.telegram.org/bot<token>/<method>``
and unwraps the ``{"ok": ..., "result": ...}`` envelope. Tests mock that one
seam (``httpx.MockTransport`` or a monkeypatched ``_api``) to assert real
behaviour with no network.

Transport model (design §4.2): there is no inbound webhook. ``connect(sink)``
spawns a background **long-poll loop** that calls
``getUpdates?timeout=50&offset=<last_update_id+1>`` forever; each batch of
updates is normalised to neutral types and pushed into the ``CoreSink``.
``close()`` cancels the loop.

Port mapping (design §4.2):

  ============================  ===================================================
  port method                   Telegram Bot API call
  ============================  ===================================================
  ``connect``                   long-poll ``getUpdates`` loop (+ ``getMe`` for username)
  ``post_placeholder``          ``sendMessage`` -> handle = (chat_id, message_id)
  ``edit``                      ``editMessageText`` (heartbeat tick)
  ``finalize``                  render HTML answer+sources+footer -> ``editMessageText``
                                with an inline keyboard (👍/👎 callback buttons)
  ``react``                     no-op (Bot API has no message reaction in this path)
  ``fetch_thread``              ``[]`` (Bot API exposes no thread-replies fetch)
  ``resolve_identity``          synthetic ``telegram-bot:<chat>:<user>`` (anonymous)
  ``send_denial``               ``sendMessage`` to the user's private chat
  ``confirm_feedback``          ``answerCallbackQuery`` (the toast popup)
  ============================  ===================================================

Inbound normalization (the adapter owns it; the dispatcher owns the flow):
``message`` updates with ``chat.type == 'private'`` are DMs (always triggered);
in groups/supergroups the bot triggers only on an ``@botusername`` mention or a
reply to one of the bot's own messages, and the ``@botusername`` token is
stripped from ``text``. ``callback_query`` updates (inline-button clicks) become
``FeedbackEvent``. Messages from bots (``from.is_bot``) -- our own loop and any
other bot -- are dropped before the sink is touched.
"""
from __future__ import annotations

import asyncio
import html
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from opsrag.auth.pomerium import CurrentUser
from opsrag.channels.base import ChannelAdapter, CoreSink
from opsrag.channels.types import (
    AgentResult,
    FeedbackEvent,
    ImageRef,
    InboundMessage,
    MessageHandle,
    ReactionKind,
    ThreadMessage,
)

_log = logging.getLogger("opsrag.channels.adapters.telegram")

# Telegram's hard per-message text cap. Long answers are paginated across
# multiple messages (see ``_paginate_html``) rather than truncated.
_TELEGRAM_MAX_CHARS = 4096
# How many source bullets to render inline before collapsing to "+N more".
_MAX_SOURCES = 10
# Long-poll timeout (seconds) handed to getUpdates. Telegram holds the
# request open up to this long when there are no updates (server-side long
# poll), so the loop is near-idle between bursts.
_LONGPOLL_TIMEOUT_S = 50


@dataclass(frozen=True)
class _TelegramHandle:
    """Opaque per-message handle: a ``(chat_id, message_id)`` pair.

    The core treats this purely as a token it hands back to ``edit`` /
    ``finalize``; only this adapter inspects it. ``thread_id`` is carried so
    paginated follow-up messages land in the same forum topic as the
    placeholder (it is ``None`` for DMs and non-forum chats).
    """

    chat_id: str
    message_id: str
    thread_id: str | None = None


class TelegramAdapter(ChannelAdapter):
    """Telegram Bot API adapter over ``httpx`` (no vendor SDK)."""

    name = "telegram"

    def __init__(self, config: Any) -> None:
        """Build the adapter from a Telegram channel sub-config.

        ``config`` is a ``TelegramChannelConfig`` (the unified
        ``channels.telegram`` block). The bot token is read from the env var
        it names (Principle VI -- never inline). No ``httpx`` client is built
        here: it is created lazily in :meth:`connect` so importing this module
        never opens a connection.
        """
        self._config = config
        self._web_ui_base_url = (
            getattr(config, "web_ui_base_url", "") or ""
        ).rstrip("/")
        self._token: str = ""
        self._sink: CoreSink | None = None
        self._client: Any = None  # httpx.AsyncClient (lazy)
        self._poll_task: asyncio.Task | None = None
        self._offset: int = 0
        self._closing = False
        # The bot's own @username + numeric id, learned from getMe in
        # connect(); used for mention-trigger + reply-to-bot detection.
        self._bot_username: str = ""
        self._bot_id: str = ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def connect(self, sink: CoreSink) -> None:
        """Build the httpx client and start the long-poll loop wired to ``sink``.

        Reads the bot token from the env var named by the config, calls
        ``getMe`` to learn the bot's ``@username`` (needed for group mention
        triggers), then spawns the background ``getUpdates`` loop.
        ``httpx`` is imported here, not at module top, so a disabled channel
        never imports it.
        """
        self._sink = sink
        token_env = getattr(self._config, "bot_token_env", "") or ""
        token = os.environ.get(token_env, "").strip()
        if not token:
            raise RuntimeError(
                f"Telegram adapter: bot token env unset ({token_env or '?'})"
            )
        self._token = token

        import httpx

        # One pooled client for the lifetime of the worker. The read timeout
        # must exceed the long-poll window or getUpdates would abort early.
        self._client = httpx.AsyncClient(
            base_url=f"https://api.telegram.org/bot{token}",
            timeout=httpx.Timeout(_LONGPOLL_TIMEOUT_S + 15.0, connect=10.0),
        )

        # Learn our own identity so we can recognise @mentions + replies to us
        # in groups. Best-effort -- a missing username simply means we only
        # trigger on reply-to-bot + DMs.
        try:
            me = await self._api("getMe", {})
            self._bot_username = str(me.get("username") or "")
            self._bot_id = str(me.get("id") or "")
        except Exception as exc:  # noqa: BLE001
            _log.warning("telegram getMe failed (mention trigger degraded): %s", exc)

        self._closing = False
        self._poll_task = asyncio.create_task(
            self._poll_loop(), name="telegram-getupdates",
        )
        _log.info(
            "telegram adapter connected (bot=@%s) via getUpdates long-poll",
            self._bot_username or "unknown",
        )

    async def close(self) -> None:
        """Cancel the poll loop and close the httpx client."""
        self._closing = True
        task = self._poll_task
        self._poll_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception as exc:  # noqa: BLE001
                _log.debug("telegram client close failed: %s", exc)
            self._client = None

    # ------------------------------------------------------------------
    # The single network seam (mocked in tests)
    # ------------------------------------------------------------------
    async def _api(self, method: str, payload: dict[str, Any]) -> Any:
        """POST one Bot API ``method`` and return its unwrapped ``result``.

        This is the ONE seam every outbound + inbound call funnels through, so
        a test can mock it (or the underlying ``httpx`` transport) exactly once
        and exercise the whole adapter without a network. Raises
        ``RuntimeError`` when Telegram replies ``{"ok": false}`` so callers see
        a real failure rather than a silent ``None``.
        """
        if self._client is None:
            raise RuntimeError("TelegramAdapter.connect() must run before this call")
        resp = await self._client.post(f"/{method}", json=payload)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or not data.get("ok"):
            desc = (data or {}).get("description") if isinstance(data, dict) else None
            raise RuntimeError(f"telegram {method} failed: {desc or data!r}")
        return data.get("result")

    # ------------------------------------------------------------------
    # Long-poll loop
    # ------------------------------------------------------------------
    async def _poll_loop(self) -> None:
        """Drain ``getUpdates`` forever, normalising each update into the sink.

        ``offset`` advances to ``last_update_id + 1`` after each batch so
        Telegram never re-delivers an acknowledged update. Per-update errors
        are logged and skipped (one malformed update must not kill the loop);
        transport errors back off briefly then retry. ``CancelledError`` (from
        :meth:`close`) is the normal exit.
        """
        while not self._closing:
            try:
                updates = await self._api(
                    "getUpdates",
                    {
                        "timeout": _LONGPOLL_TIMEOUT_S,
                        "offset": self._offset,
                        "allowed_updates": ["message", "callback_query"],
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                if self._closing:
                    return
                _log.warning("telegram getUpdates failed, backing off: %s", exc)
                await asyncio.sleep(3.0)
                continue

            for update in updates or []:
                try:
                    update_id = int(update.get("update_id", 0))
                except (TypeError, ValueError):
                    update_id = 0
                if update_id >= self._offset:
                    self._offset = update_id + 1
                try:
                    await self._handle_update(update)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("telegram update handling failed: %s", exc)

    async def _handle_update(self, update: dict[str, Any]) -> None:
        """Route one raw update to ``on_message`` / ``on_feedback`` (or drop it)."""
        sink = self._sink
        if sink is None:
            return
        if "callback_query" in update:
            fb = _callback_to_feedback(update["callback_query"])
            if fb is not None:
                await sink.on_feedback(fb)
            return
        message = update.get("message")
        if not isinstance(message, dict):
            return  # edited_message / channel_post / etc. -- ignore in v1
        msg = self._message_to_inbound(message)
        if msg is not None:
            await sink.on_message(msg)

    # ------------------------------------------------------------------
    # Inbound normalization
    # ------------------------------------------------------------------
    def _message_to_inbound(self, message: dict[str, Any]) -> InboundMessage | None:
        """Map a Telegram ``message`` to ``InboundMessage`` (or ``None`` to drop).

        Drops bot-authored messages (``from.is_bot`` -- our own loop AND other
        bots), and, in groups, drops messages that neither @mention the bot nor
        reply to one of the bot's own messages. Strips the ``@botusername``
        token from the text the agent sees. ``is_dm`` is ``chat.type ==
        'private'``; ``channel_id``/``workspace`` are the (possibly negative)
        chat id; ``thread_id`` is the forum-topic ``message_thread_id`` when
        present.
        """
        chat = message.get("chat") or {}
        sender = message.get("from") or {}

        # Bot-loop filter: drop our own bot and any other bot before the sink.
        if sender.get("is_bot"):
            return None

        chat_type = chat.get("type") or ""
        is_dm = chat_type == "private"
        raw_text = message.get("text") or message.get("caption") or ""

        if is_dm:
            triggered = True
        else:
            triggered, raw_text = self._group_trigger(message, raw_text)
        if not triggered:
            return None

        text = _strip_mention(raw_text, self._bot_username).strip()

        chat_id = _str_id(chat.get("id"))
        thread_raw = message.get("message_thread_id")
        thread_id = _str_id(thread_raw) if thread_raw is not None else None

        image_refs: list[ImageRef] = []
        photos = message.get("photo") or []
        if photos:
            largest = max(photos, key=lambda p: p.get("file_size", 0) or 0)
            image_refs.append(ImageRef(
                file_id=largest.get("file_id"),
                mime_type="image/jpeg",  # Telegram photos are always JPEG
                size=largest.get("file_size"),
            ))
        doc = message.get("document") or {}
        if doc and str(doc.get("mime_type", "")).startswith("image/"):
            image_refs.append(ImageRef(
                file_id=doc.get("file_id"),
                mime_type=doc.get("mime_type"),
                size=doc.get("file_size"),
            ))

        return InboundMessage(
            channel_id=chat_id,
            user_id=_str_id(sender.get("id")),
            text=text,
            message_id=_str_id(message.get("message_id")),
            thread_id=thread_id,
            is_dm=is_dm,
            workspace=chat_id,
            images=tuple(image_refs),
            raw=dict(message),
        )

    async def fetch_image(self, ref: ImageRef) -> bytes | None:
        """Download a Telegram attachment: getFile(file_id) -> file download.

        Telegram is a two-step fetch -- ``getFile`` resolves the ``file_id`` to a
        ``file_path``, then the bytes are pulled from the file CDN under the bot
        token. Returns ``None`` if there is no ``file_id`` to resolve.
        """
        if not ref.file_id:
            return None
        import httpx
        base = f"https://api.telegram.org/bot{self._token}"
        async with httpx.AsyncClient(timeout=30) as client:
            meta = await client.get(f"{base}/getFile", params={"file_id": ref.file_id})
            meta.raise_for_status()
            file_path = meta.json()["result"]["file_path"]
            dl = await client.get(
                f"https://api.telegram.org/file/bot{self._token}/{file_path}"
            )
            dl.raise_for_status()
            return dl.content

    def _group_trigger(
        self, message: dict[str, Any], raw_text: str,
    ) -> tuple[bool, str]:
        """Decide whether a group message should wake the bot.

        Returns ``(triggered, text)``. A group message triggers iff it
        @mentions the bot's username OR is a reply to one of the bot's own
        messages. Other group chatter is ignored so the bot isn't a firehose.
        """
        # Reply to one of OUR messages -> trigger (no mention needed).
        reply_to = message.get("reply_to_message") or {}
        reply_from = reply_to.get("from") or {}
        if self._bot_id and _str_id(reply_from.get("id")) == self._bot_id:
            return True, raw_text

        # @botusername mention anywhere in the text -> trigger.
        if self._bot_username and _mentions_bot(raw_text, self._bot_username):
            return True, raw_text

        return False, raw_text

    # ------------------------------------------------------------------
    # Outbound primitives
    # ------------------------------------------------------------------
    async def post_placeholder(
        self, channel_id: str, thread_id: str | None, text: str,
    ) -> MessageHandle:
        payload: dict[str, Any] = {"chat_id": channel_id, "text": text}
        if thread_id:
            payload["message_thread_id"] = _maybe_int(thread_id)
        result = await self._api("sendMessage", payload)
        message_id = _str_id((result or {}).get("message_id"))
        return _TelegramHandle(
            chat_id=channel_id, message_id=message_id, thread_id=thread_id,
        )

    async def edit(self, handle: MessageHandle, text: str) -> None:
        h = _as_handle(handle)
        await self._api(
            "editMessageText",
            {
                "chat_id": _maybe_int(h.chat_id),
                "message_id": _maybe_int(h.message_id),
                "text": text,
            },
        )

    async def finalize(self, handle: MessageHandle, result: AgentResult) -> None:
        h = _as_handle(handle)
        # The full answer is paginated across as many messages as it takes --
        # Telegram caps a single message at 4096 chars, so a long answer would
        # otherwise be clipped. The first chunk edits the placeholder in place;
        # the rest are sent as follow-up messages in the same chat/thread. The
        # feedback keyboard is attached to the LAST chunk only.
        messages = render_answer_messages(
            result.answer,
            result.sources,
            diagram_present=result.diagram_present,
        )
        keyboard = _feedback_keyboard(result.investigation_id)
        last = len(messages) - 1
        for i, text in enumerate(messages):
            markup = keyboard if i == last else None
            if i == 0:
                payload: dict[str, Any] = {
                    "chat_id": _maybe_int(h.chat_id),
                    "message_id": _maybe_int(h.message_id),
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                }
                if markup is not None:
                    payload["reply_markup"] = markup
                await self._api("editMessageText", payload)
            else:
                payload = {
                    "chat_id": _maybe_int(h.chat_id),
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                }
                if h.thread_id:
                    payload["message_thread_id"] = _maybe_int(h.thread_id)
                if markup is not None:
                    payload["reply_markup"] = markup
                await self._api("sendMessage", payload)

    async def react(
        self, channel_id: str, message_id: str, kind: ReactionKind,
    ) -> None:
        """No-op: the Bot API path used here has no message-reaction affordance.

        The core already treats reactions as best-effort, so a clean no-op
        satisfies the port without faking an unsupported call.
        """
        return None

    async def fetch_thread(
        self, channel_id: str, thread_id: str, *, cap: int,
    ) -> list[ThreadMessage]:
        """Always ``[]`` -- the Bot API exposes no thread-replies fetch.

        Telegram's Bot API has no endpoint to read prior messages of a chat or
        a forum topic (a bot only sees messages pushed to it via getUpdates).
        With nothing to fetch we return ``[]``; the dispatcher then simply uses
        the user's message as the whole query (no prior-thread block), which is
        the correct behaviour for a platform without a readable thread model.
        """
        return []

    async def resolve_identity(self, msg: InboundMessage) -> CurrentUser:
        """Synthetic anonymous identity ``telegram-bot:<chat>:<user>`` (design §4.2)."""
        return _telegram_user_to_current_user(msg)

    async def send_denial(self, msg: InboundMessage, reason: str) -> None:
        """Privately tell the user why they were denied (DM = their chat id).

        On Telegram the user's private chat id equals their numeric user id, so
        ``sendMessage`` to ``user_id`` reaches them directly rather than
        polluting the group with denial noise.
        """
        target = msg.user_id or msg.channel_id
        if not target:
            return
        await self._api("sendMessage", {"chat_id": _maybe_int(target), "text": reason})

    async def confirm_feedback(self, fb: FeedbackEvent, *, accepted: bool) -> None:
        """Ephemeral confirm via ``answerCallbackQuery`` (the toast popup).

        The inbound parser stashes the originating ``callback_query.id`` in
        ``fb.raw['callback_query_id']`` so we can answer the exact query; an
        unanswered callback leaves a spinner on the user's button.
        """
        callback_query_id = (fb.raw or {}).get("callback_query_id") or ""
        if not callback_query_id:
            return
        confirm_text = (
            "👍 Thanks -- recorded as helpful."
            if fb.thumbs == "up"
            else "👎 Thanks -- recorded as wrong. We'll learn from this."
        )
        await self._api(
            "answerCallbackQuery",
            {"callback_query_id": callback_query_id, "text": confirm_text},
        )


# =====================================================================
# Module-level helpers (free functions so tests can pin them directly)
# =====================================================================
def _as_handle(handle: MessageHandle) -> _TelegramHandle:
    if not isinstance(handle, _TelegramHandle):
        raise TypeError(f"expected _TelegramHandle, got {type(handle)!r}")
    return handle


def _str_id(value: Any) -> str:
    """Coerce a Telegram id (often an int, possibly negative) to ``str``."""
    if value is None:
        return ""
    return str(value)


def _maybe_int(value: str) -> int | str:
    """Return ``value`` as an ``int`` when it is a (possibly negative) integer.

    Telegram accepts numeric ``chat_id`` either as int or numeric string, but
    sending the native int avoids any ambiguity for negative group ids.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _strip_mention(text: str, bot_username: str) -> str:
    """Remove the leading/standalone ``@botusername`` token from ``text``.

    Telegram delivers a group @mention as a literal ``@username`` token in the
    message text; the agent must not see it. Matching is case-insensitive
    (clients lower-case usernames inconsistently). When ``bot_username`` is
    unknown (getMe failed) the text is returned unchanged.
    """
    if not bot_username:
        return text
    token = f"@{bot_username}"
    lowered = text.lower()
    token_lower = token.lower()
    out = text
    idx = lowered.find(token_lower)
    while idx != -1:
        out = out[:idx] + out[idx + len(token):]
        lowered = out.lower()
        idx = lowered.find(token_lower)
    return out


def _mentions_bot(text: str, bot_username: str) -> bool:
    """True iff ``text`` contains the bot's ``@username`` (case-insensitive)."""
    if not bot_username:
        return False
    return f"@{bot_username}".lower() in (text or "").lower()


def _telegram_user_to_current_user(msg: InboundMessage) -> CurrentUser:
    """Map a Telegram inbound message to a synthetic anonymous ``CurrentUser``.

    Phase-1 identity (design §4.2 / D4): a deterministic ``oid`` of the shape
    ``telegram-bot:<chat_id>:<user_id>`` with ``is_anonymous=True`` -- traceable
    (we can group traces by chat + user) but not authenticated, so admin gates
    stay fail-closed.
    """
    from dataclasses import replace

    workspace = msg.workspace or "unknown"
    user = msg.user_id or "unknown-user"
    oid = f"telegram-bot:{workspace}:{user}"
    return replace(CurrentUser.anonymous(), oid=oid)


def _callback_to_feedback(callback: dict[str, Any]) -> FeedbackEvent | None:
    """Parse a Telegram ``callback_query`` (inline-button click) -> ``FeedbackEvent``.

    Our feedback buttons carry ``callback_data`` of the form ``up:<id>`` /
    ``down:<id>`` (see :func:`_feedback_keyboard`). Returns ``None`` for any
    callback that is not one of ours or carries a malformed payload. The
    originating ``callback_query.id`` is stashed in ``raw`` so
    :meth:`TelegramAdapter.confirm_feedback` can answer the exact query.
    """
    if not isinstance(callback, dict):
        return None
    data = (callback.get("data") or "").strip()
    if ":" not in data:
        return None
    thumbs, investigation_id = data.split(":", 1)
    if thumbs not in ("up", "down") or not investigation_id:
        return None

    sender = callback.get("from") or {}
    user_id = _str_id(sender.get("id")) or "telegram-unknown"

    # thread_id: the chat the button lives in, mirrored from the message the
    # button is attached to. callback_query.message is the bot's own answer.
    cb_message = callback.get("message") or {}
    cb_chat = cb_message.get("chat") or {}
    thread_raw = cb_message.get("message_thread_id")
    thread_id = (
        _str_id(thread_raw)
        if thread_raw is not None
        else (_str_id(cb_chat.get("id")) or None)
    )

    return FeedbackEvent(
        thumbs=thumbs,
        investigation_id=investigation_id,
        user_id=user_id,
        thread_id=thread_id,
        raw={"callback_query_id": _str_id(callback.get("id"))},
    )


def _feedback_keyboard(investigation_id: str | None) -> dict[str, Any] | None:
    """Build the inline 👍/👎 keyboard, or ``None`` when there's no anchor.

    Without an ``investigation_id`` there is nowhere to record the vote, so we
    omit the keyboard entirely (matching the Slack render's "no anchor -> no
    buttons" rule). The ``callback_data`` is ``up:<id>`` / ``down:<id>``, parsed
    back by :func:`_callback_to_feedback`.
    """
    if not investigation_id:
        return None
    return {
        "inline_keyboard": [
            [
                {"text": "👍 Helpful", "callback_data": f"up:{investigation_id}"},
                {"text": "👎 Wrong", "callback_data": f"down:{investigation_id}"},
            ],
        ],
    }


# ---------------------------------------------------------------------------
# Markdown -> Telegram-HTML rendering
# ---------------------------------------------------------------------------
# The agent emits Markdown; Telegram's parse_mode=HTML supports only a small
# tag allow-list (<b>/<i>/<s>/<code>/<pre>/<a>) and rejects raw < > &. We
# convert the supported Markdown constructs to those tags and HTML-escape
# everything else, so answers render as rich text instead of raw markup.

# Fenced blocks whose content is machine-only (a diagram spec for the web UI's
# renderer) -- dropped from chat; a plain callout is shown instead.
_DIAGRAM_FENCE_RE = re.compile(
    r"```[ \t]*(?:diagram-json|diagram|mermaid)\b.*?```",
    re.DOTALL | re.IGNORECASE,
)
# Any remaining fenced code block -> <pre> (group 1 = body, lang line dropped).
_CODE_FENCE_RE = re.compile(r"```[ \t]*[^\n`]*\n?(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.*?)[ \t]*#*[ \t]*$")
_BULLET_RE = re.compile(r"^([ \t]*)[-*+][ \t]+(.*)$")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\*)")
_STRIKE_RE = re.compile(r"~~(.+?)~~")
_LINK_RE = re.compile(r"\[([^\]\n]+)\]\(([^)\s]+)\)")

_DIAGRAM_CALLOUT = "📊 <i>Diagram available (rendered in the web UI only).</i>"


def _md_inline(text: str) -> str:
    """Escape one line of prose and apply inline Markdown -> Telegram HTML.

    ``text`` must NOT contain code spans (those are extracted to placeholders
    upstream) so the inline rules never fire inside code. Escaping happens
    first; the Markdown markers (``**``/``*``/``~~``/``[](...)``) carry no
    ``<>&`` so they survive escaping and are then rewritten into real tags.
    Underscore emphasis is intentionally unsupported (collides with snake_case
    / dunder identifiers).
    """
    s = html.escape(text, quote=False)
    s = _BOLD_RE.sub(lambda m: f"<b>{m.group(1)}</b>", s)
    s = _STRIKE_RE.sub(lambda m: f"<s>{m.group(1)}</s>", s)
    s = _ITALIC_RE.sub(lambda m: f"<i>{m.group(1)}</i>", s)
    s = _LINK_RE.sub(
        lambda m: f'<a href="{m.group(2).replace(chr(34), "&quot;")}">{m.group(1)}</a>',
        s,
    )
    return s


def markdown_to_telegram_html(md: str) -> str:
    """Convert the agent's Markdown answer to Telegram-safe HTML.

    Supported: headings (-> bold; Telegram has none), ``**bold**``,
    ``*italic*``, ``~~strike~~``, ``` `inline code` ```, fenced code (-> <pre>),
    ``[text](url)`` links, and ``-/*/+`` bullets (-> ``•``). Machine-only
    diagram fences (``diagram-json`` / ``mermaid``) are dropped.
    """
    if not md:
        return ""
    text = _DIAGRAM_FENCE_RE.sub("", md)

    # Extract code (block, then inline) to placeholders so inline Markdown rules
    # never fire inside code, and code content is escaped verbatim at the end.
    blocks: list[str] = []

    def _stash_block(m: re.Match[str]) -> str:
        blocks.append(m.group(1))
        return f"\x00B{len(blocks) - 1}\x00"

    text = _CODE_FENCE_RE.sub(_stash_block, text)

    spans: list[str] = []

    def _stash_span(m: re.Match[str]) -> str:
        spans.append(m.group(1))
        return f"\x00C{len(spans) - 1}\x00"

    text = _INLINE_CODE_RE.sub(_stash_span, text)

    out: list[str] = []
    for line in text.split("\n"):
        h = _HEADING_RE.match(line)
        if h:
            out.append(f"<b>{_md_inline(h.group(1))}</b>")
            continue
        b = _BULLET_RE.match(line)
        if b:
            out.append(f"{b.group(1)}• {_md_inline(b.group(2))}")
            continue
        out.append(_md_inline(line))
    text = "\n".join(out)

    for i, code in enumerate(spans):
        text = text.replace(
            f"\x00C{i}\x00", f"<code>{html.escape(code, quote=False)}</code>"
        )
    for i, code in enumerate(blocks):
        text = text.replace(
            f"\x00B{i}\x00", f"<pre>{html.escape(code.strip(chr(10)), quote=False)}</pre>"
        )

    # Collapse the blank-line runs left where diagram fences were dropped.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def render_answer_messages(
    answer: str,
    sources: list[dict[str, Any]],
    *,
    diagram_present: bool = False,
    max_chars: int = _TELEGRAM_MAX_CHARS,
) -> list[str]:
    """Render the FULL answer as a list of Telegram-HTML messages (no truncation).

    The answer Markdown is converted to Telegram HTML, then body + trailer
    (diagram callout + sources + footer) is paginated across ``<=max_chars``
    messages. No web-UI deep-link is emitted: channel conversations are
    oid-owned and not shareable, so a ``/#chat`` link would 404 for the
    recipient.
    """
    had_diagram = bool(_DIAGRAM_FENCE_RE.search(answer or ""))
    body = markdown_to_telegram_html(answer or "")

    trailer_parts: list[str] = []
    if diagram_present or had_diagram:
        trailer_parts.append(_DIAGRAM_CALLOUT)
    sources_block = _render_sources_html(sources or [])
    if sources_block:
        trailer_parts.append(sources_block)
    trailer_parts.append(_footer_html())
    trailer = "\n\n".join(trailer_parts)

    full = f"{body}\n\n{trailer}" if body else trailer
    return _paginate_html(full, max_chars)


def _paginate_html(text: str, max_chars: int) -> list[str]:
    """Pack ``text`` into ``<=max_chars`` Telegram-HTML chunks.

    Splits on line, then word, boundaries (inline tags never span a line, so a
    split never cuts one). Multi-line ``<pre>`` blocks are handled specially: a
    split inside a ``<pre>`` closes it at the chunk end and reopens it at the
    start of the next chunk, so every chunk is independently valid HTML.
    """
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    buf = ""
    pre_open = False  # currently inside an unclosed <pre> in buf?
    close = "</pre>"

    def _flush(reopen: bool) -> None:
        nonlocal buf, pre_open
        if not buf:
            return
        chunks.append(buf + close if pre_open else buf)
        if reopen and pre_open:
            buf = "<pre>"
        else:
            buf = ""
            pre_open = False

    for line in text.split("\n"):
        reserve = len(close) if pre_open else 0
        addition = (1 + len(line)) if buf else len(line)
        if buf and len(buf) + addition + reserve > max_chars:
            _flush(reopen=True)
        if not buf and len(line) > max_chars:
            # A single line still overflows -> hard-wrap it (rare; in practice
            # never inside a <pre>).
            chunks.extend(_wrap_long_line(line, max_chars))
            continue
        buf = line if not buf else f"{buf}\n{line}"
        opens, closes = line.count("<pre>"), line.count(close)
        if opens > closes:
            pre_open = True
        elif closes > opens:
            pre_open = False
    _flush(reopen=False)
    return chunks or [""]


def _wrap_long_line(line: str, max_chars: int) -> list[str]:
    """Hard-wrap a single over-long line on spaces (entity-safe last resort)."""
    out: list[str] = []
    cur = ""
    for word in line.split(" "):
        while len(word) > max_chars:
            if cur:
                out.append(cur)
                cur = ""
            cut = _entity_safe_cut(word, max_chars)
            out.append(word[:cut])
            word = word[cut:]
        if cur and len(cur) + 1 + len(word) > max_chars:
            out.append(cur)
            cur = ""
        cur = word if not cur else f"{cur} {word}"
    if cur:
        out.append(cur)
    return out


def _entity_safe_cut(s: str, cap: int) -> int:
    """Return a cut index ``<=cap`` that never lands inside an ``&...;`` entity."""
    cut = min(len(s), cap)
    amp = s.rfind("&", 0, cut)
    if amp > 0 and ";" not in s[amp:cut]:
        return amp
    return cut


def _render_sources_html(sources: list[dict[str, Any]]) -> str:
    """Render up to ``_MAX_SOURCES`` source bullets as escaped Telegram HTML.

    Each source becomes ``• <a href="url">title</a>`` when a URL is present,
    else a bare escaped title/path. Returns ``""`` for an empty list so the
    caller can omit the section entirely.
    """
    if not sources:
        return ""
    bullets: list[str] = []
    for src in sources[:_MAX_SOURCES]:
        if not isinstance(src, dict):
            continue
        title = (src.get("title") or src.get("source") or "source").strip()
        url = (src.get("url") or "").strip()
        path = (src.get("source") or "").strip()
        if url:
            bullets.append(
                f'• <a href="{html.escape(url, quote=True)}">{html.escape(title)}</a>'
            )
        elif path:
            bullets.append(f"• <code>{html.escape(path)}</code>")
        else:
            bullets.append(f"• {html.escape(title)}")
    if not bullets:
        return ""
    extra = len(sources) - _MAX_SOURCES
    if extra > 0:
        bullets.append(f"• <i>+{extra} more</i>")
    return f"<b>Sources</b> ({len(sources)})\n" + "\n".join(bullets)


def _footer_html() -> str:
    """Render the attribution + timestamp footer (no deep link).

    Channel conversations are oid-owned and not shareable, so no ``/#chat``
    deep-link is emitted (it would 404 for the recipient).
    """
    from datetime import UTC, datetime

    return "<i>OpsRAG · " + datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC") + "</i>"
