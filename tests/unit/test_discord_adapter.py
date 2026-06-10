"""Unit tests for the Discord ChannelAdapter (P2).

Covers design 8 for the Discord adapter, with NO network and NO real
``discord.py`` installed: every Discord object is a small duck-typed fake
exposing only the attributes + async methods the adapter touches. The render
path (Embed + Button View) lazily ``import discord``; rather than depend on the
SDK, the relevant tests install a minimal fake ``discord`` module into
``sys.modules`` so the lazy import resolves to our stub and we can assert the
REAL render shape (description truncation, source fields, custom_ids).

Assertions are on REAL behaviour, never just "no exception":

  * inbound normalization: DM vs guild @mention -> InboundMessage (mention
    strip, is_dm, thread_id, workspace=guild/dm); bot-loop + no-trigger drops;
  * feedback parse: button interaction -> FeedbackEvent (custom_id ``up:<id>``
    / ``down:<id>``, interaction stashed in raw) + malformed drop;
  * outbound: post_placeholder wraps the Message; edit; finalize renders an
    Embed (truncated description + source fields) + a View with ``up:<id>`` /
    ``down:<id>`` buttons; react maps ACK/DONE/ERROR -> 👀/✅/❌ and no-ops
    when the inbound Message wasn't retained; send_denial DMs the author;
  * fetch_thread maps Thread history -> ThreadMessage (source_id populated,
    our own past reply -> is_self=True, other bots kept) and returns [] for a
    non-thread channel;
  * identity oid format (discord-bot:<guild_or_dm>:<user>).

The adapter MODULE must import cleanly with ``discord`` NOT installed -- the
import at the top of this file is the proof.
"""
from __future__ import annotations

import sys
import types

import pytest

from opsrag.channels.adapters.discord import adapter as dadapter
from opsrag.channels.adapters.discord.adapter import (
    DiscordAdapter,
    _discord_user_to_current_user,
    _DiscordHandle,
    _history_message_to_thread,
    _interaction_to_feedback,
    _message_to_inbound,
)
from opsrag.channels.config import DiscordChannelConfig
from opsrag.channels.types import (
    AgentResult,
    InboundMessage,
    ReactionKind,
    ThreadMessage,
)


# ---------------------------------------------------------------------------
# Duck-typed Discord fakes (no SDK)
# ---------------------------------------------------------------------------
class _FakeAuthor:
    def __init__(self, uid: int, *, bot: bool = False, name: str = "user",
                 display_name: str | None = None) -> None:
        self.id = uid
        self.bot = bot
        self.name = name
        self.display_name = display_name or name
        self.sent: list[str] = []

    async def send(self, content: str) -> None:
        self.sent.append(content)


class _FakeChannelType:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:  # mirrors discord.ChannelType str
        return self.name


class _FakeDMChannel:
    """isinstance won't match discord.DMChannel (absent), so name-based path."""

    __name__ = "DMChannel"

    def __init__(self, cid: int) -> None:
        self.id = cid
        self.type = _FakeChannelType("private")
        self.sent: list[str] = []

    async def send(self, content):  # noqa: ANN001
        msg = _FakeMessage(mid=999, content="", author=_FakeAuthor(1), channel=self)
        self.sent.append(content)
        return msg


# Give the class the exact name the adapter's duck-typed fallback checks.
_FakeDMChannel.__qualname__ = "DMChannel"


class _FakeTextChannel:
    def __init__(self, cid: int) -> None:
        self.id = cid
        self.type = _FakeChannelType("text")
        self._next_id = 5000
        self.sent: list[object] = []

    async def send(self, content):  # noqa: ANN001
        self._next_id += 1
        msg = _FakeMessage(
            mid=self._next_id, content="", author=_FakeAuthor(1), channel=self,
        )
        self.sent.append(msg)
        return msg


class _FakeThread:
    """Name contains 'Thread' so the adapter's duck-typed _is_thread matches."""

    def __init__(self, cid: int, history: list | None = None) -> None:
        self.id = cid
        self.type = _FakeChannelType("public_thread")
        self._history = history or []
        self._next_id = 8000
        self.sent: list[object] = []

    async def send(self, content):  # noqa: ANN001
        self._next_id += 1
        msg = _FakeMessage(mid=self._next_id, content="", author=_FakeAuthor(1), channel=self)
        self.sent.append(msg)
        return msg

    def history(self, *, limit: int):  # noqa: ANN001 - returns async iterator
        items = self._history[:limit]

        async def _gen():
            for m in items:
                yield m

        return _gen()


class _FakeGuild:
    def __init__(self, gid: int) -> None:
        self.id = gid


class _FakeMessage:
    def __init__(self, *, mid: int, content: str, author: _FakeAuthor,
                 channel, guild: _FakeGuild | None = None,
                 mentions: list | None = None) -> None:
        self.id = mid
        self.content = content
        self.author = author
        self.channel = channel
        self.guild = guild
        self.mentions = mentions or []
        self.reactions: list[str] = []
        self.edits: list[dict] = []

    async def add_reaction(self, emoji: str) -> None:
        self.reactions.append(emoji)

    async def edit(self, **kwargs) -> None:  # noqa: ANN003
        self.edits.append(kwargs)


class _FakeInteractionResponse:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send_message(self, content, *, ephemeral=False):  # noqa: ANN001
        self.messages.append({"content": content, "ephemeral": ephemeral})


class _FakeInteraction:
    def __init__(self, *, custom_id: str, user_id: int, channel=None) -> None:
        self.data = {"custom_id": custom_id}
        self.user = _FakeAuthor(user_id)
        self.channel = channel
        self.response = _FakeInteractionResponse()


_BOT = _FakeAuthor(100, bot=True, name="opsrag-bot")


# ---------------------------------------------------------------------------
# Fake ``discord`` module for the lazy-imported render path
# ---------------------------------------------------------------------------
class _FakeEmbed:
    def __init__(self, *, description: str = "") -> None:
        self.description = description
        self.fields: list[dict] = []
        self.footer: str | None = None

    def add_field(self, *, name, value, inline=False):  # noqa: ANN001
        self.fields.append({"name": name, "value": value, "inline": inline})

    def set_footer(self, *, text):  # noqa: ANN001
        self.footer = text


class _FakeButtonStyle:
    success = "success"
    danger = "danger"


class _FakeButton:
    def __init__(self, *, label, style, custom_id):  # noqa: ANN001
        self.label = label
        self.style = style
        self.custom_id = custom_id


class _FakeView:
    def __init__(self, *, timeout=None) -> None:  # noqa: ANN001
        self.timeout = timeout
        self.items: list = []

    def add_item(self, item) -> None:  # noqa: ANN001
        self.items.append(item)


def _install_fake_discord(monkeypatch) -> None:
    """Inject a minimal fake ``discord`` module for the lazy render imports."""
    mod = types.ModuleType("discord")
    mod.Embed = _FakeEmbed
    mod.ButtonStyle = _FakeButtonStyle
    ui = types.ModuleType("discord.ui")
    ui.View = _FakeView
    ui.Button = _FakeButton
    mod.ui = ui
    # DMChannel / Thread sentinels: real classes the isinstance checks can run
    # against (our fakes won't be instances, so the duck-typed fallback wins).
    mod.DMChannel = type("DMChannel", (), {})
    mod.Thread = type("Thread", (), {})
    monkeypatch.setitem(sys.modules, "discord", mod)
    monkeypatch.setitem(sys.modules, "discord.ui", ui)


def _adapter() -> DiscordAdapter:
    return DiscordAdapter(
        DiscordChannelConfig(web_ui_base_url="https://opsrag.example.com"),
    )


@pytest.fixture(autouse=True)
def _clear_retained():
    dadapter._RETAINED_MESSAGES.clear()
    yield
    dadapter._RETAINED_MESSAGES.clear()


# ---------------------------------------------------------------------------
# Inbound normalization
# ---------------------------------------------------------------------------
def test_guild_mention_strips_mention_and_sets_workspace() -> None:
    channel = _FakeTextChannel(cid=222)
    guild = _FakeGuild(gid=777)
    author = _FakeAuthor(11, name="alice")
    message = _FakeMessage(
        mid=333, content="<@100> why is prod down?", author=author,
        channel=channel, guild=guild, mentions=[_BOT],
    )
    msg = _message_to_inbound(message, _BOT)
    assert msg is not None
    assert msg.channel_id == "222"
    assert msg.user_id == "11"
    assert msg.text == "why is prod down?"  # mention stripped
    assert msg.message_id == "333"
    assert msg.thread_id is None  # plain text channel, not a Thread
    assert msg.is_dm is False
    assert msg.workspace == "777"  # guild id


def test_dm_message_is_dm_workspace_dm() -> None:
    channel = _FakeDMChannel(cid=444)
    author = _FakeAuthor(11)
    message = _FakeMessage(mid=555, content="hello there", author=author, channel=channel)
    msg = _message_to_inbound(message, _BOT)
    assert msg is not None
    assert msg.is_dm is True
    assert msg.thread_id is None
    assert msg.text == "hello there"
    assert msg.workspace == "dm"
    assert msg.channel_id == "444"


def test_thread_message_sets_thread_id() -> None:
    thread = _FakeThread(cid=666)
    guild = _FakeGuild(gid=777)
    author = _FakeAuthor(11)
    message = _FakeMessage(
        mid=888, content="<@100> follow-up question", author=author,
        channel=thread, guild=guild, mentions=[_BOT],
    )
    msg = _message_to_inbound(message, _BOT)
    assert msg is not None
    assert msg.thread_id == "666"  # Thread channel id
    assert msg.is_dm is False
    assert msg.text == "follow-up question"


def test_legacy_nickname_mention_is_stripped() -> None:
    channel = _FakeTextChannel(cid=222)
    guild = _FakeGuild(gid=777)
    # ``<@!100>`` is the legacy nickname-mention form.
    message = _FakeMessage(
        mid=1, content="<@!100> status?", author=_FakeAuthor(11),
        channel=channel, guild=guild, mentions=[_BOT],
    )
    msg = _message_to_inbound(message, _BOT)
    assert msg is not None
    assert msg.text == "status?"


def test_bot_author_is_dropped() -> None:
    channel = _FakeTextChannel(cid=222)
    other_bot = _FakeAuthor(200, bot=True, name="datadog")
    message = _FakeMessage(
        mid=1, content="<@100> alert", author=other_bot, channel=channel,
        guild=_FakeGuild(777), mentions=[_BOT],
    )
    assert _message_to_inbound(message, _BOT) is None


def test_guild_message_without_mention_is_dropped() -> None:
    channel = _FakeTextChannel(cid=222)
    message = _FakeMessage(
        mid=1, content="just chatting, no ping", author=_FakeAuthor(11),
        channel=channel, guild=_FakeGuild(777), mentions=[],
    )
    assert _message_to_inbound(message, _BOT) is None


def test_mention_only_message_with_empty_text_is_dropped() -> None:
    channel = _FakeTextChannel(cid=222)
    message = _FakeMessage(
        mid=1, content="<@100>", author=_FakeAuthor(11), channel=channel,
        guild=_FakeGuild(777), mentions=[_BOT],
    )
    assert _message_to_inbound(message, _BOT) is None


def test_inbound_retains_message_for_react_and_dm() -> None:
    channel = _FakeTextChannel(cid=222)
    message = _FakeMessage(
        mid=42, content="<@100> q", author=_FakeAuthor(11), channel=channel,
        guild=_FakeGuild(777), mentions=[_BOT],
    )
    msg = _message_to_inbound(message, _BOT)
    assert msg is not None
    # The Message is retained under its id so react/send_denial can use it.
    assert dadapter._RETAINED_MESSAGES.get("42") is message


# ---------------------------------------------------------------------------
# Feedback parse
# ---------------------------------------------------------------------------
def test_interaction_to_feedback_up() -> None:
    interaction = _FakeInteraction(custom_id="up:inv-42", user_id=11)
    fb = _interaction_to_feedback(interaction)
    assert fb is not None
    assert fb.thumbs == "up"
    assert fb.investigation_id == "inv-42"
    assert fb.user_id == "11"
    # The interaction is stashed so confirm_feedback can reply ephemerally.
    assert fb.raw["interaction"] is interaction


def test_interaction_to_feedback_down_in_thread_sets_thread_id() -> None:
    thread = _FakeThread(cid=666)
    interaction = _FakeInteraction(custom_id="down:inv-7", user_id=11, channel=thread)
    fb = _interaction_to_feedback(interaction)
    assert fb is not None
    assert fb.thumbs == "down"
    assert fb.thread_id == "666"


def test_interaction_malformed_returns_none() -> None:
    # No colon in custom_id.
    assert _interaction_to_feedback(_FakeInteraction(custom_id="garbage", user_id=1)) is None
    # Wrong direction token.
    assert _interaction_to_feedback(_FakeInteraction(custom_id="maybe:x", user_id=1)) is None
    # Empty investigation id.
    assert _interaction_to_feedback(_FakeInteraction(custom_id="up:", user_id=1)) is None


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
def test_resolve_identity_oid_format_guild() -> None:
    msg = InboundMessage(
        channel_id="222", user_id="11", text="hi", message_id="1",
        thread_id=None, is_dm=False, workspace="777",
    )
    user = _discord_user_to_current_user(msg)
    assert user.oid == "discord-bot:777:11"
    assert user.is_anonymous is True


def test_resolve_identity_oid_format_dm() -> None:
    msg = InboundMessage(
        channel_id="444", user_id="11", text="hi", message_id="1",
        thread_id=None, is_dm=True, workspace="dm",
    )
    user = _discord_user_to_current_user(msg)
    assert user.oid == "discord-bot:dm:11"


@pytest.mark.asyncio
async def test_adapter_resolve_identity_delegates() -> None:
    msg = InboundMessage(
        channel_id="222", user_id="11", text="hi", message_id="1",
        thread_id=None, is_dm=False, workspace="777",
    )
    user = await _adapter().resolve_identity(msg)
    assert user.oid == "discord-bot:777:11"


# ---------------------------------------------------------------------------
# fetch_thread -> ThreadMessage (source_id + is_self)
# ---------------------------------------------------------------------------
def test_history_message_marks_self_and_populates_source_id() -> None:
    bot_id = "100"
    our_reply = _FakeMessage(
        mid=2, content="our past answer", author=_FakeAuthor(100, name="opsrag-bot"),
        channel=_FakeThread(1),
    )
    user_msg = _FakeMessage(
        mid=1, content="the alert fired", author=_FakeAuthor(11, name="alice"),
        channel=_FakeThread(1),
    )
    other_bot = _FakeMessage(
        mid=3, content="INC-1 sev2", author=_FakeAuthor(200, bot=True, name="Rootly"),
        channel=_FakeThread(1),
    )
    empty = _FakeMessage(
        mid=4, content="   ", author=_FakeAuthor(11), channel=_FakeThread(1),
    )

    tm_self = _history_message_to_thread(our_reply, bot_id)
    assert tm_self is not None
    assert tm_self.is_self is True
    assert tm_self.source_id == "2"  # platform message id populated

    tm_user = _history_message_to_thread(user_msg, bot_id)
    assert tm_user is not None
    assert tm_user.is_self is False
    assert tm_user.author == "alice"
    assert tm_user.source_id == "1"

    tm_other = _history_message_to_thread(other_bot, bot_id)
    assert tm_other is not None
    assert tm_other.is_self is False  # other bots are KEPT
    assert tm_other.author == "Rootly"

    assert _history_message_to_thread(empty, bot_id) is None  # empty skipped


@pytest.mark.asyncio
async def test_fetch_thread_maps_history_and_orders_oldest_first() -> None:
    adapter = _adapter()
    thread_channel = _FakeThread(cid=666)
    # discord history yields NEWEST first; the adapter reverses to oldest-first.
    thread_channel._history = [
        _FakeMessage(mid=3, content="newest", author=_FakeAuthor(11, name="alice"),
                     channel=thread_channel),
        _FakeMessage(mid=2, content="middle", author=_FakeAuthor(100, name="bot"),
                     channel=thread_channel),
        _FakeMessage(mid=1, content="oldest", author=_FakeAuthor(11, name="alice"),
                     channel=thread_channel),
    ]

    class _Client:
        user = _FakeAuthor(100, name="bot")

        def get_channel(self, cid):  # noqa: ANN001
            return thread_channel if cid == 666 else None

    adapter._client = _Client()
    msgs = await adapter.fetch_thread("666", "666", cap=20)
    assert [m.text for m in msgs] == ["oldest", "middle", "newest"]
    # The bot's own message (id 100) is flagged is_self.
    assert msgs[1].is_self is True
    assert all(m.source_id for m in msgs)


@pytest.mark.asyncio
async def test_fetch_thread_returns_empty_for_non_thread_channel() -> None:
    adapter = _adapter()
    text_channel = _FakeTextChannel(cid=222)

    class _Client:
        user = _FakeAuthor(100)

        def get_channel(self, cid):  # noqa: ANN001
            return text_channel

    adapter._client = _Client()
    # A plain text channel has no thread-replies model -> [].
    assert await adapter.fetch_thread("222", "222", cap=20) == []


# ---------------------------------------------------------------------------
# Outbound primitives
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_post_placeholder_wraps_message_and_edit() -> None:
    adapter = _adapter()
    text_channel = _FakeTextChannel(cid=222)

    class _Client:
        user = _FakeAuthor(100)

        def get_channel(self, cid):  # noqa: ANN001
            return text_channel if cid == 222 else None

    adapter._client = _Client()
    handle = await adapter.post_placeholder("222", None, "🤔 Thinking...")
    assert isinstance(handle, _DiscordHandle)
    # The placeholder was sent into the channel and the returned Message wrapped.
    assert text_channel.sent and handle.message is text_channel.sent[-1]

    await adapter.edit(handle, "still thinking")
    assert handle.message.edits[-1] == {"content": "still thinking"}


@pytest.mark.asyncio
async def test_react_maps_kinds_and_noop_when_unretained() -> None:
    adapter = _adapter()
    adapter._client = object()  # react only needs a non-None client

    # No retained message -> clean no-op (no error).
    await adapter.react("222", "999", ReactionKind.ACK)

    # Retain a message under id "42", then react.
    msg = _FakeMessage(mid=42, content="q", author=_FakeAuthor(11),
                       channel=_FakeTextChannel(222))
    dadapter._retain_message("42", msg)
    await adapter.react("222", "42", ReactionKind.ACK)
    await adapter.react("222", "42", ReactionKind.DONE)
    await adapter.react("222", "42", ReactionKind.ERROR)
    assert msg.reactions == ["👀", "✅", "❌"]


@pytest.mark.asyncio
async def test_send_denial_dms_author() -> None:
    adapter = _adapter()
    adapter._client = object()
    author = _FakeAuthor(11)
    msg = _FakeMessage(mid=42, content="q", author=author, channel=_FakeTextChannel(222))
    dadapter._retain_message("42", msg)

    inbound = InboundMessage(
        channel_id="222", user_id="11", text="q", message_id="42",
        thread_id=None, is_dm=False, workspace="777",
    )
    await adapter.send_denial(inbound, "not allowed here")
    assert author.sent == ["not allowed here"]


@pytest.mark.asyncio
async def test_confirm_feedback_responds_ephemeral() -> None:
    from opsrag.channels.types import FeedbackEvent

    adapter = _adapter()
    interaction = _FakeInteraction(custom_id="up:inv-1", user_id=11)
    fb = FeedbackEvent(
        thumbs="up", investigation_id="inv-1", user_id="11", thread_id=None,
        raw={"interaction": interaction},
    )
    await adapter.confirm_feedback(fb, accepted=True)
    assert interaction.response.messages
    assert interaction.response.messages[-1]["ephemeral"] is True


# ---------------------------------------------------------------------------
# finalize -> Embed + feedback View (needs the fake ``discord`` module)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_finalize_renders_embed_and_feedback_buttons(monkeypatch) -> None:
    _install_fake_discord(monkeypatch)
    adapter = _adapter()
    placeholder = _FakeMessage(
        mid=900, content="🤔", author=_FakeAuthor(100), channel=_FakeTextChannel(222),
    )
    handle = _DiscordHandle(message=placeholder)
    result = AgentResult(
        answer="**Root cause**: bad deploy",
        sources=[
            {"title": "runbook", "url": "https://example.com/rb"},
            {"source": "svc/api/handler.py"},
        ],
        diagram_present=True,
        session_id="discord-thread:222:333",
        investigation_id="inv-9",
    )
    await adapter.finalize(handle, result)

    edit = placeholder.edits[-1]
    embed = edit["embed"]
    assert isinstance(embed, _FakeEmbed)
    assert "Root cause" in embed.description
    # Sources rendered as embed fields (+ a diagram callout field).
    field_values = [f["value"] for f in embed.fields]
    assert "https://example.com/rb" in field_values
    assert any("svc/api/handler.py" in v for v in field_values)
    # Deep-link footer from the configured web UI base.
    assert embed.footer and "opsrag.example.com" in embed.footer

    # The feedback View carries up:/down: custom_ids anchored on the inv id.
    view = edit["view"]
    custom_ids = {b.custom_id for b in view.items}
    assert custom_ids == {"up:inv-9", "down:inv-9"}


@pytest.mark.asyncio
async def test_finalize_truncates_long_answer(monkeypatch) -> None:
    _install_fake_discord(monkeypatch)
    adapter = _adapter()
    placeholder = _FakeMessage(
        mid=900, content="🤔", author=_FakeAuthor(100), channel=_FakeTextChannel(222),
    )
    handle = _DiscordHandle(message=placeholder)
    long_answer = "x" * 5000  # exceeds the 4096 embed description cap
    result = AgentResult(
        answer=long_answer, sources=[], diagram_present=False,
        session_id=None, investigation_id=None,
    )
    await adapter.finalize(handle, result)
    embed = placeholder.edits[-1]["embed"]
    assert len(embed.description) <= 4096
    assert embed.description.endswith("…answer truncated")
    # No investigation_id -> no feedback View attached.
    assert "view" not in placeholder.edits[-1]


@pytest.mark.asyncio
async def test_finalize_caps_source_fields(monkeypatch) -> None:
    _install_fake_discord(monkeypatch)
    adapter = _adapter()
    placeholder = _FakeMessage(
        mid=900, content="🤔", author=_FakeAuthor(100), channel=_FakeTextChannel(222),
    )
    handle = _DiscordHandle(message=placeholder)
    sources = [{"title": f"src{i}", "url": f"https://e/{i}"} for i in range(15)]
    result = AgentResult(
        answer="ans", sources=sources, diagram_present=False,
        session_id=None, investigation_id="inv-1",
    )
    await adapter.finalize(handle, result)
    embed = placeholder.edits[-1]["embed"]
    # 10 source fields capped + 1 "+N more" overflow field = 11.
    assert len(embed.fields) == 11
    assert any("more" in f["value"] for f in embed.fields)


# ---------------------------------------------------------------------------
# connect() requires the token env (no SDK needed to hit this guard)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_connect_raises_without_token(monkeypatch) -> None:
    monkeypatch.delenv("OPSRAG_DISCORD_BOT_TOKEN", raising=False)

    class _Sink:
        async def on_message(self, msg):  # noqa: ANN001
            ...

        async def on_feedback(self, fb):  # noqa: ANN001
            ...

    with pytest.raises(RuntimeError, match="token env unset"):
        await _adapter().connect(_Sink())


# ---------------------------------------------------------------------------
# SDK-gated smoke test (skipped in unit CI where discord.py is absent)
# ---------------------------------------------------------------------------
def test_real_discord_embed_build() -> None:
    pytest.importorskip("discord")
    # With the real SDK present, _build_embed must produce a real discord.Embed.
    import discord

    from opsrag.channels.adapters.discord.adapter import _build_embed

    embed = _build_embed(
        AgentResult(answer="hi", sources=[{"title": "t", "url": "https://e"}],
                    diagram_present=False, session_id="s", investigation_id="i"),
        web_ui_base_url="https://opsrag.example.com",
    )
    assert isinstance(embed, discord.Embed)
    assert embed.description == "hi"


# ---------------------------------------------------------------------------
# Port conformance
# ---------------------------------------------------------------------------
def test_adapter_satisfies_port() -> None:
    from opsrag.channels.base import ChannelAdapter

    assert isinstance(_adapter(), ChannelAdapter)
    assert _adapter().name == "discord"


def test_thread_message_type_roundtrip() -> None:
    # Guard against drift in the neutral type the adapter constructs.
    tm = ThreadMessage(author="alice", text="hi", is_self=False, source_id="1")
    assert tm.source_id == "1"
