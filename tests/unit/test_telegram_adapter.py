"""Unit tests for the Telegram ChannelAdapter (P2 -- httpx Bot API, no SDK).

Covers design 8.4 for the Telegram adapter, with NO network: ``httpx`` is a
real dependency, so we drive the adapter's one network seam (``_api`` ->
``httpx.AsyncClient.post``) through an ``httpx.MockTransport`` that records every
Bot API call and returns canned ``{"ok": true, "result": ...}`` envelopes.
Inbound parsing is exercised with duck-typed update dicts (exactly the shape
Telegram delivers).

Assertions are on REAL behaviour, never "it didn't raise":

  * inbound normalization: private chat -> is_dm; group @mention strip + trigger;
    reply-to-bot trigger; non-trigger group chatter dropped; bot-loop drop;
    negative chat ids; forum ``message_thread_id`` -> thread_id;
  * outbound payload shapes: sendMessage (placeholder), editMessageText
    (heartbeat), finalize (HTML render + inline 👍/👎 keyboard with
    ``callback_data='up:<id>'/'down:<id>'``, ``parse_mode=HTML``, escaping);
  * truncation: an over-length answer is clipped + carries the "view in UI" link;
  * feedback parse: callback_query -> FeedbackEvent (+ malformed drop) with the
    callback_query.id stashed in raw;
  * identity oid format ``telegram-bot:<chat>:<user>`` (anonymous);
  * react() no-op; fetch_thread() == [];
  * confirm_feedback -> answerCallbackQuery; send_denial -> sendMessage to the
    user's private chat.
"""
from __future__ import annotations

import httpx
import pytest

from opsrag.channels.adapters.telegram.adapter import (
    TelegramAdapter,
    _callback_to_feedback,
    _feedback_keyboard,
    _strip_mention,
    _TelegramHandle,
    markdown_to_telegram_html,
    render_answer_messages,
)
from opsrag.channels.config import TelegramChannelConfig
from opsrag.channels.types import (
    AgentResult,
    FeedbackEvent,
    InboundMessage,
    ReactionKind,
)


# ---------------------------------------------------------------------------
# Test scaffolding: a MockTransport that records every Bot API call
# ---------------------------------------------------------------------------
class _BotAPIRecorder:
    """Records (method, payload) for each Bot API POST; returns canned results.

    ``results`` maps a Bot API method name to the ``result`` value returned in
    the ``{"ok": true, "result": ...}`` envelope. Methods without an override
    return ``{}``.
    """

    def __init__(self, results: dict | None = None) -> None:
        self.results = results or {}
        self.calls: list[tuple[str, dict]] = []

    def transport(self) -> httpx.MockTransport:
        def _handler(request: httpx.Request) -> httpx.Response:
            method = request.url.path.rsplit("/", 1)[-1]
            import json

            payload = json.loads(request.content.decode() or "{}")
            self.calls.append((method, payload))
            result = self.results.get(method, {})
            return httpx.Response(200, json={"ok": True, "result": result})

        return httpx.MockTransport(_handler)

    def last(self, method: str) -> dict:
        for m, payload in reversed(self.calls):
            if m == method:
                return payload
        raise AssertionError(f"no {method} call recorded; calls={[c[0] for c in self.calls]}")


def _adapter_with_recorder(
    recorder: _BotAPIRecorder, *, web_ui: str = "https://opsrag.example.com",
) -> TelegramAdapter:
    """Build an adapter whose httpx client uses the recorder's MockTransport.

    We bypass ``connect`` (which would real-fetch getMe + spawn the poll loop)
    and inject the client + bot identity directly, mirroring the Slack test's
    ``adapter._client = fake`` injection.
    """
    adapter = TelegramAdapter(TelegramChannelConfig(web_ui_base_url=web_ui))
    adapter._client = httpx.AsyncClient(  # noqa: SLF001
        base_url="https://api.telegram.org/botTEST",
        transport=recorder.transport(),
    )
    adapter._bot_username = "opsrag_bot"  # noqa: SLF001
    adapter._bot_id = "999"  # noqa: SLF001
    return adapter


# ---------------------------------------------------------------------------
# Inbound normalization
# ---------------------------------------------------------------------------
def test_private_message_is_dm() -> None:
    adapter = TelegramAdapter(TelegramChannelConfig())
    msg = adapter._message_to_inbound(  # noqa: SLF001
        {
            "message_id": 12,
            "from": {"id": 555, "is_bot": False},
            "chat": {"id": 555, "type": "private"},
            "text": "why is prod down?",
        }
    )
    assert msg is not None
    assert msg.is_dm is True
    assert msg.channel_id == "555"
    assert msg.user_id == "555"
    assert msg.message_id == "12"
    assert msg.thread_id is None
    assert msg.workspace == "555"
    assert msg.text == "why is prod down?"


def test_group_mention_triggers_and_strips() -> None:
    adapter = TelegramAdapter(TelegramChannelConfig())
    adapter._bot_username = "opsrag_bot"  # noqa: SLF001
    msg = adapter._message_to_inbound(  # noqa: SLF001
        {
            "message_id": 7,
            "from": {"id": 42, "is_bot": False},
            "chat": {"id": -1001234567890, "type": "supergroup"},
            "text": "@opsrag_bot why did the canary roll back?",
        }
    )
    assert msg is not None
    assert msg.is_dm is False
    # Negative supergroup id is preserved verbatim.
    assert msg.channel_id == "-1001234567890"
    assert msg.workspace == "-1001234567890"
    # The @botusername token is stripped from the text the agent sees.
    assert "@opsrag_bot" not in msg.text
    assert msg.text == "why did the canary roll back?"


def test_group_message_without_mention_is_dropped() -> None:
    adapter = TelegramAdapter(TelegramChannelConfig())
    adapter._bot_username = "opsrag_bot"  # noqa: SLF001
    adapter._bot_id = "999"  # noqa: SLF001
    msg = adapter._message_to_inbound(  # noqa: SLF001
        {
            "message_id": 8,
            "from": {"id": 42, "is_bot": False},
            "chat": {"id": -100, "type": "group"},
            "text": "just chatting, not for the bot",
        }
    )
    assert msg is None


def test_group_reply_to_bot_triggers_without_mention() -> None:
    adapter = TelegramAdapter(TelegramChannelConfig())
    adapter._bot_username = "opsrag_bot"  # noqa: SLF001
    adapter._bot_id = "999"  # noqa: SLF001
    msg = adapter._message_to_inbound(  # noqa: SLF001
        {
            "message_id": 9,
            "from": {"id": 42, "is_bot": False},
            "chat": {"id": -100, "type": "group"},
            "text": "and what about the database?",
            "reply_to_message": {"from": {"id": 999, "is_bot": True}},
        }
    )
    assert msg is not None
    assert msg.text == "and what about the database?"


def test_bot_authored_message_is_dropped() -> None:
    adapter = TelegramAdapter(TelegramChannelConfig())
    # Our own loop AND other bots are dropped via from.is_bot before the sink.
    assert (
        adapter._message_to_inbound(  # noqa: SLF001
            {
                "message_id": 10,
                "from": {"id": 1000, "is_bot": True},
                "chat": {"id": 1000, "type": "private"},
                "text": "echo from another bot",
            }
        )
        is None
    )


def test_forum_topic_thread_id_is_captured() -> None:
    adapter = TelegramAdapter(TelegramChannelConfig())
    adapter._bot_username = "opsrag_bot"  # noqa: SLF001
    msg = adapter._message_to_inbound(  # noqa: SLF001
        {
            "message_id": 11,
            "from": {"id": 42, "is_bot": False},
            "chat": {"id": -100, "type": "supergroup"},
            "message_thread_id": 4242,
            "text": "@opsrag_bot status?",
        }
    )
    assert msg is not None
    assert msg.thread_id == "4242"


def test_strip_mention_helper_is_case_insensitive_and_idempotent() -> None:
    assert _strip_mention("@OpsRAG_Bot hello", "opsrag_bot").strip() == "hello"
    assert _strip_mention("no mention here", "opsrag_bot") == "no mention here"
    # Unknown bot username -> unchanged.
    assert _strip_mention("@opsrag_bot hi", "") == "@opsrag_bot hi"


# ---------------------------------------------------------------------------
# Long-poll: getUpdates parsing routes to the sink + advances offset
# ---------------------------------------------------------------------------
class _RecordingSink:
    def __init__(self) -> None:
        self.messages: list[InboundMessage] = []
        self.feedbacks: list[FeedbackEvent] = []

    async def on_message(self, msg: InboundMessage) -> None:
        self.messages.append(msg)

    async def on_feedback(self, fb: FeedbackEvent) -> None:
        self.feedbacks.append(fb)


@pytest.mark.asyncio
async def test_handle_update_routes_message_and_callback_and_advances_offset() -> None:
    recorder = _BotAPIRecorder()
    adapter = _adapter_with_recorder(recorder)
    sink = _RecordingSink()
    adapter._sink = sink  # noqa: SLF001

    # A normal DM message update.
    await adapter._handle_update(  # noqa: SLF001
        {
            "update_id": 100,
            "message": {
                "message_id": 1,
                "from": {"id": 7, "is_bot": False},
                "chat": {"id": 7, "type": "private"},
                "text": "hello",
            },
        }
    )
    # A feedback callback_query update.
    await adapter._handle_update(  # noqa: SLF001
        {
            "update_id": 101,
            "callback_query": {
                "id": "cbq-1",
                "from": {"id": 7},
                "data": "up:inv-77",
                "message": {"chat": {"id": 7}},
            },
        }
    )

    assert len(sink.messages) == 1
    assert sink.messages[0].text == "hello"
    assert len(sink.feedbacks) == 1
    assert sink.feedbacks[0].investigation_id == "inv-77"


# ---------------------------------------------------------------------------
# Outbound primitives: payload shapes
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_post_placeholder_sends_message_and_returns_handle() -> None:
    recorder = _BotAPIRecorder({"sendMessage": {"message_id": 321}})
    adapter = _adapter_with_recorder(recorder)
    handle = await adapter.post_placeholder("-100", "4242", "🤔 Thinking...")
    assert isinstance(handle, _TelegramHandle)
    assert handle.chat_id == "-100"
    assert handle.message_id == "321"
    payload = recorder.last("sendMessage")
    assert payload["chat_id"] == "-100"
    assert payload["text"] == "🤔 Thinking..."
    # Forum topic threading is carried through as an int.
    assert payload["message_thread_id"] == 4242


@pytest.mark.asyncio
async def test_edit_calls_edit_message_text() -> None:
    recorder = _BotAPIRecorder()
    adapter = _adapter_with_recorder(recorder)
    handle = _TelegramHandle(chat_id="-100", message_id="321")
    await adapter.edit(handle, "😅 still fetching...")
    payload = recorder.last("editMessageText")
    # Negative chat id is sent as a native int.
    assert payload["chat_id"] == -100
    assert payload["message_id"] == 321
    assert payload["text"] == "😅 still fetching..."


@pytest.mark.asyncio
async def test_finalize_renders_html_with_feedback_keyboard() -> None:
    recorder = _BotAPIRecorder()
    adapter = _adapter_with_recorder(recorder)
    handle = _TelegramHandle(chat_id="555", message_id="10")
    result = AgentResult(
        answer="Root cause: a <bad> deploy & rollback",
        sources=[{"title": "runbook", "url": "https://example.com/rb"}],
        diagram_present=False,
        session_id="telegram-dm:555",
        investigation_id="inv-9",
    )
    await adapter.finalize(handle, result)
    payload = recorder.last("editMessageText")
    assert payload["parse_mode"] == "HTML"
    # HTML-special chars in the answer are escaped (Telegram rejects raw < > &).
    assert "<bad>" not in payload["text"]
    assert "&lt;bad&gt;" in payload["text"]
    assert "&amp;" in payload["text"]
    # Source link is rendered as an <a> tag.
    assert '<a href="https://example.com/rb">runbook</a>' in payload["text"]
    # Inline keyboard carries the up/down callbacks anchored on the investigation.
    keyboard = payload["reply_markup"]["inline_keyboard"]
    callbacks = [btn["callback_data"] for row in keyboard for btn in row]
    assert "up:inv-9" in callbacks
    assert "down:inv-9" in callbacks


def test_render_no_investigation_means_no_keyboard() -> None:
    assert _feedback_keyboard(None) is None
    kb = _feedback_keyboard("inv-1")
    assert kb is not None
    assert kb["inline_keyboard"][0][0]["callback_data"] == "up:inv-1"
    assert kb["inline_keyboard"][0][1]["callback_data"] == "down:inv-1"


# ---------------------------------------------------------------------------
# Markdown -> Telegram HTML conversion
# ---------------------------------------------------------------------------
def test_markdown_headings_become_bold() -> None:
    out = markdown_to_telegram_html("# Title\n## Sub\n### Deep")
    assert out == "<b>Title</b>\n<b>Sub</b>\n<b>Deep</b>"


def test_markdown_inline_emphasis_and_code_and_links() -> None:
    out = markdown_to_telegram_html(
        "Use **bold**, *italic*, ~~old~~, `code`, and [docs](https://x.io/a)."
    )
    assert "<b>bold</b>" in out
    assert "<i>italic</i>" in out
    assert "<s>old</s>" in out
    assert "<code>code</code>" in out
    assert '<a href="https://x.io/a">docs</a>' in out


def test_markdown_bullets_become_dots() -> None:
    out = markdown_to_telegram_html("- one\n* two\n+ three")
    assert out == "• one\n• two\n• three"


def test_markdown_underscores_are_not_emphasis() -> None:
    # snake_case / dunder identifiers must survive untouched.
    out = markdown_to_telegram_html("call app_module and __init__ now")
    assert "app_module" in out
    assert "__init__" in out
    assert "<i>" not in out


def test_markdown_escapes_html_specials() -> None:
    out = markdown_to_telegram_html("compare a < b && c > d")
    assert "&lt;" in out and "&gt;" in out and "&amp;" in out
    assert "<b>" not in out  # raw markup never leaks


def test_markdown_no_markup_inside_code() -> None:
    # Markdown markers inside a code span/block must NOT be converted.
    out = markdown_to_telegram_html("inline `a**b**c` then\n```\nx = a*b*c\n```")
    assert "<code>a**b**c</code>" in out
    assert "<pre>x = a*b*c</pre>" in out


def test_markdown_drops_diagram_json_block() -> None:
    md = 'before\n\n```diagram-json\n{"nodes": ["a","b"]}\n```\n\nafter'
    out = markdown_to_telegram_html(md)
    assert "diagram-json" not in out
    assert "nodes" not in out
    assert "before" in out and "after" in out


def test_render_messages_diagram_callout_no_ui_link() -> None:
    md = '# Arch\n\n```diagram-json\n{"x":1}\n```\n\ndone'
    parts = render_answer_messages(md, sources=[])
    joined = "".join(parts)
    assert "Diagram available" in joined
    assert "/#chat/" not in joined  # no broken conversation deep-link
    assert "View in OpsRAG UI" not in joined
    assert "OpsRAG ·" in joined  # plain footer attribution


def test_render_messages_paginates_long_answer_without_truncating() -> None:
    # An answer well over a single Telegram message (4096) is split, not clipped.
    long_answer = "\n".join(f"line {i} " + "x" * 200 for i in range(120))
    parts = render_answer_messages(
        long_answer,
        sources=[{"title": "runbook", "url": "https://example.com/rb"}],
    )
    assert len(parts) > 1  # actually paginated
    assert all(len(p) <= 4096 for p in parts)  # every chunk fits the cap
    # Nothing is dropped and no truncation marker is emitted.
    assert "truncated" not in "".join(parts)
    assert "line 0 " in parts[0]
    assert "line 119 " in "".join(parts)
    # The trailer (sources + footer) flows into the final chunk(s).
    assert '<a href="https://example.com/rb">runbook</a>' in parts[-1]
    assert "OpsRAG ·" in parts[-1]


def test_render_messages_short_answer_is_single_chunk() -> None:
    parts = render_answer_messages("just a short answer", sources=[])
    assert len(parts) == 1


def test_paginate_keeps_pre_blocks_valid_across_split() -> None:
    # A <pre> code block larger than the cap must close+reopen across chunks so
    # every chunk has balanced <pre>/</pre>.
    code = "\n".join(f"row{i} = {i}" for i in range(600))
    parts = render_answer_messages(f"```\n{code}\n```", sources=[])
    assert len(parts) > 1
    for p in parts:
        assert p.count("<pre>") == p.count("</pre>")  # balanced in each chunk


@pytest.mark.asyncio
async def test_finalize_paginates_across_multiple_messages() -> None:
    recorder = _BotAPIRecorder({"sendMessage": {"message_id": 2}})
    adapter = _adapter_with_recorder(recorder)
    handle = _TelegramHandle(chat_id="555", message_id="10", thread_id="77")
    result = AgentResult(
        answer="\n".join(f"para {i} " + "y" * 300 for i in range(60)),
        sources=[{"title": "rb", "url": "https://example.com/rb"}],
        diagram_present=False,
        session_id="telegram-dm:555",
        investigation_id="inv-9",
    )
    await adapter.finalize(handle, result)
    methods = [m for m, _ in recorder.calls]
    # First chunk edits the placeholder; the rest are follow-up sends.
    assert methods[0] == "editMessageText"
    assert methods.count("sendMessage") >= 1
    # The placeholder edit carries NO keyboard; the final send does.
    assert "reply_markup" not in recorder.calls[0][1]
    last_send = recorder.last("sendMessage")
    callbacks = [
        btn["callback_data"]
        for row in last_send["reply_markup"]["inline_keyboard"]
        for btn in row
    ]
    assert "up:inv-9" in callbacks and "down:inv-9" in callbacks
    # Follow-up messages stay in the same forum topic as the placeholder.
    assert last_send["message_thread_id"] == 77


# ---------------------------------------------------------------------------
# react no-op + fetch_thread empty
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_react_is_noop_no_network() -> None:
    recorder = _BotAPIRecorder()
    adapter = _adapter_with_recorder(recorder)
    await adapter.react("555", "10", ReactionKind.ACK)
    await adapter.react("555", "10", ReactionKind.DONE)
    await adapter.react("555", "10", ReactionKind.ERROR)
    # react is a clean no-op -- it must not hit the Bot API at all.
    assert recorder.calls == []


@pytest.mark.asyncio
async def test_fetch_thread_returns_empty() -> None:
    recorder = _BotAPIRecorder()
    adapter = _adapter_with_recorder(recorder)
    out = await adapter.fetch_thread("555", "4242", cap=20)
    assert out == []
    assert recorder.calls == []


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resolve_identity_oid_format() -> None:
    adapter = TelegramAdapter(TelegramChannelConfig())
    msg = InboundMessage(
        channel_id="555", user_id="42", text="hi", message_id="1",
        thread_id=None, is_dm=True, workspace="555",
    )
    user = await adapter.resolve_identity(msg)
    assert user.oid == "telegram-bot:555:42"
    assert user.is_anonymous is True


# ---------------------------------------------------------------------------
# Feedback parse
# ---------------------------------------------------------------------------
def test_callback_to_feedback_up() -> None:
    fb = _callback_to_feedback(
        {
            "id": "cbq-9",
            "from": {"id": 42},
            "data": "up:inv-42",
            "message": {"chat": {"id": -100}, "message_thread_id": 7},
        }
    )
    assert fb is not None
    assert fb.thumbs == "up"
    assert fb.investigation_id == "inv-42"
    assert fb.user_id == "42"
    assert fb.thread_id == "7"
    # The callback_query.id is stashed so confirm_feedback can answer it.
    assert fb.raw["callback_query_id"] == "cbq-9"


def test_callback_to_feedback_malformed_returns_none() -> None:
    # No colon in data.
    assert _callback_to_feedback({"id": "x", "from": {"id": 1}, "data": "garbage"}) is None
    # Bad direction.
    assert _callback_to_feedback({"id": "x", "from": {"id": 1}, "data": "maybe:inv-1"}) is None
    # Empty investigation id.
    assert _callback_to_feedback({"id": "x", "from": {"id": 1}, "data": "up:"}) is None


# ---------------------------------------------------------------------------
# confirm_feedback + send_denial
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_confirm_feedback_answers_callback_query() -> None:
    recorder = _BotAPIRecorder()
    adapter = _adapter_with_recorder(recorder)
    fb = FeedbackEvent(
        thumbs="up", investigation_id="inv-1", user_id="42", thread_id="7",
        raw={"callback_query_id": "cbq-77"},
    )
    await adapter.confirm_feedback(fb, accepted=True)
    payload = recorder.last("answerCallbackQuery")
    assert payload["callback_query_id"] == "cbq-77"
    assert "helpful" in payload["text"].lower()


@pytest.mark.asyncio
async def test_confirm_feedback_without_query_id_is_noop() -> None:
    recorder = _BotAPIRecorder()
    adapter = _adapter_with_recorder(recorder)
    fb = FeedbackEvent(
        thumbs="down", investigation_id="inv-1", user_id="42", thread_id=None, raw={},
    )
    await adapter.confirm_feedback(fb, accepted=True)
    assert recorder.calls == []


@pytest.mark.asyncio
async def test_send_denial_dms_user_private_chat() -> None:
    recorder = _BotAPIRecorder()
    adapter = _adapter_with_recorder(recorder)
    msg = InboundMessage(
        channel_id="-100", user_id="42", text="hi", message_id="1",
        thread_id=None, is_dm=False, workspace="-100",
    )
    await adapter.send_denial(msg, "not allowed here")
    payload = recorder.last("sendMessage")
    # The denial goes to the user's private chat (their user id), not the group.
    assert payload["chat_id"] == 42
    assert payload["text"] == "not allowed here"


# ---------------------------------------------------------------------------
# _api seam: ok=false surfaces a real error
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_api_raises_on_not_ok() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": False, "description": "Bad Request"})

    adapter = TelegramAdapter(TelegramChannelConfig())
    adapter._client = httpx.AsyncClient(  # noqa: SLF001
        base_url="https://api.telegram.org/botTEST",
        transport=httpx.MockTransport(_handler),
    )
    with pytest.raises(RuntimeError, match="Bad Request"):
        await adapter._api("sendMessage", {"chat_id": 1, "text": "x"})  # noqa: SLF001
