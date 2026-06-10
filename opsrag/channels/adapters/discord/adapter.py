"""``DiscordAdapter`` -- the Discord implementation of the ``ChannelAdapter`` port.

Transport is the ``discord.py`` gateway websocket (an *outbound* connection --
no public ingress, so this adapter runs as the ``discordbot`` worker role, not
on the API role). ``connect(sink)`` builds a ``discord.Client``, registers
``on_message`` + ``on_interaction`` handlers that normalise platform events into
the neutral types and push them into the :class:`CoreSink`, and starts the
client in a background task. ``close()`` closes the client and awaits the task.

Port mapping (design §4.3):

  ============================  ===================================================
  port method                   Discord call
  ============================  ===================================================
  ``post_placeholder``          ``channel.send(text)`` -> wraps the ``Message``
  ``edit``                      ``message.edit(content=...)`` (heartbeat tick)
  ``finalize``                  a ``discord.Embed`` (answer + sources as fields) +
                                a ``discord.ui.View`` with 👍/👎 ``Button``s
  ``react`` ACK/DONE/ERROR      best-effort ``message.add_reaction`` (👀/✅/❌)
  ``fetch_thread``              ``Thread.history`` -> ``[ThreadMessage]`` (else [])
  ``resolve_identity``          synthetic ``discord-bot:<guild_or_dm>:<user>``
  ``send_denial``               ``author.send(reason)`` (DM)
  ``confirm_feedback``          ``interaction.response.send_message(ephemeral=True)``
  ============================  ===================================================

Inbound normalization (the adapter owns it; the dispatcher owns the flow):
``on_message`` builds an ``InboundMessage`` (bot @mention stripped, ``is_dm``
set from ``isinstance(channel, discord.DMChannel)``, ``thread_id`` set when the
channel is a ``Thread``) and pushes it to ``sink.on_message``; ``on_interaction``
parses a feedback ``Button`` click (``custom_id`` ``up:<id>`` / ``down:<id>``)
into a ``FeedbackEvent`` and pushes it to ``sink.on_feedback``. Bot-loop messages
(our own bot AND any other bot, i.e. ``author.bot``) are dropped BEFORE the sink
is called.

Privileged intent: triggering on @mentions / DMs requires reading message
content, which needs the **MESSAGE CONTENT** privileged gateway intent. It must
be enabled in the Discord Developer Portal for the bot application AND requested
here (``intents.message_content = True``); otherwise ``message.content`` arrives
empty and the bot never sees a query.
"""
from __future__ import annotations

import asyncio
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
    InboundMessage,
    MessageHandle,
    ReactionKind,
    ThreadMessage,
)

# NOTE: the ``discord`` SDK is imported LAZILY (inside ``connect`` and the
# render helpers), never at module top -- so this package imports cleanly on
# the ``api`` role and in unit CI where ``discord.py`` is not installed. Every
# Discord object is therefore typed ``Any`` here rather than via a
# ``TYPE_CHECKING`` import of the (possibly-absent) SDK.

_log = logging.getLogger("opsrag.channels.adapters.discord")

# Discord renders a user mention as ``<@123>`` or ``<@!123>`` (the ``!`` form is
# the legacy nickname mention). Strip whichever the gateway delivers so the
# agent never sees the bot ping token.
_MENTION_RE = re.compile(r"<@!?\d+>")

# ReactionKind -> Discord unicode emoji (the glyphs the Slack handler used).
_REACTION_EMOJI: dict[ReactionKind, str] = {
    ReactionKind.ACK: "👀",
    ReactionKind.DONE: "✅",
    ReactionKind.ERROR: "❌",
}

# Discord hard limits we render against.
_EMBED_DESCRIPTION_MAX = 4096  # max chars in an embed ``description``
_EMBED_FIELD_VALUE_MAX = 1024  # max chars in a single embed field value
_MAX_SOURCE_FIELDS = 10        # cap source rows (embeds allow 25 fields total)
_TRUNCATION_SUFFIX = "\n\n…answer truncated"


@dataclass(frozen=True)
class _DiscordHandle:
    """Opaque per-message handle wrapping the placeholder ``discord.Message``.

    The core treats this as a token it hands back to ``edit`` / ``finalize``;
    only this adapter inspects it (to reach ``message.edit``).
    """

    message: Any  # discord.Message (duck-typed; SDK not imported at module top)


class DiscordAdapter(ChannelAdapter):
    """Discord adapter over the ``discord.py`` gateway."""

    name = "discord"

    def __init__(self, config: Any) -> None:
        """Build the adapter from a Discord channel sub-config.

        ``config`` is a ``DiscordChannelConfig``. The bot token is read from the
        env var it names (``bot_token_env``; Principle VI -- never inline). The
        ``discord.Client`` itself is constructed lazily in ``connect`` so
        importing this module never touches ``discord.py``.
        """
        self._config = config
        self._web_ui_base_url = (getattr(config, "web_ui_base_url", "") or "").rstrip("/")
        self._sink: CoreSink | None = None
        self._client: Any = None  # discord.Client
        self._task: asyncio.Task[Any] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def connect(self, sink: CoreSink) -> None:
        """Build the gateway client and start it in a background task.

        Reads the bot token from the env var named by the config, requests the
        privileged ``message_content`` intent (see module docstring), registers
        the ``on_message`` / ``on_interaction`` handlers that normalise events
        and push them into ``sink``, and launches ``client.start(token)`` as a
        background task so ``connect`` returns once the worker is running.
        """
        self._sink = sink
        token = os.environ.get(getattr(self._config, "bot_token_env", ""), "").strip()
        if not token:
            raise RuntimeError(
                "Discord adapter: bot token env unset "
                f"({getattr(self._config, 'bot_token_env', '?')})"
            )

        import discord  # lazy: keep discord.py out of the import graph on other roles

        intents = discord.Intents.default()
        # MESSAGE CONTENT is a privileged intent -- it must ALSO be toggled on
        # for the application in the Discord Developer Portal. Without it,
        # ``message.content`` is empty and we never see the user's query.
        intents.message_content = True
        client = discord.Client(intents=intents)

        @client.event
        async def on_message(message: Any) -> None:  # noqa: ANN401 - SDK type
            inbound = _message_to_inbound(message, client.user)
            if inbound is None:
                return  # bot-loop, no trigger, or empty -- dropped by the mapper
            await sink.on_message(inbound)

        @client.event
        async def on_interaction(interaction: Any) -> None:  # noqa: ANN401 - SDK type
            fb = _interaction_to_feedback(interaction)
            if fb is None:
                return
            await sink.on_feedback(fb)

        self._client = client
        self._task = asyncio.create_task(
            client.start(token), name="discord-gateway",
        )
        _log.info("discord adapter connected via gateway")

    async def close(self) -> None:
        """Close the gateway client and await the background task's exit."""
        client = self._client
        self._client = None
        if client is not None:
            try:
                await client.close()
            except Exception as exc:  # noqa: BLE001
                _log.warning("discord client close failed: %s", exc)
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # Outbound primitives
    # ------------------------------------------------------------------
    async def post_placeholder(
        self, channel_id: str, thread_id: str | None, text: str,
    ) -> MessageHandle:
        """Send the placeholder into the resolved channel and wrap the Message.

        Discord replies stay in the same channel/thread the message arrived in
        (there is no per-message reply anchor like Slack's ``thread_ts``), so we
        resolve the target channel by id and ``send`` into it. The returned
        ``discord.Message`` is the handle ``edit`` / ``finalize`` operate on.
        """
        client = self._require_client()
        # Prefer the thread id when present (a reply inside a Discord Thread),
        # else the originating channel id.
        target_id = thread_id or channel_id
        channel = await self._resolve_channel(target_id)
        if channel is None:
            raise RuntimeError(f"discord: channel {target_id!r} not resolvable")
        message = await channel.send(text)
        return _DiscordHandle(message=message)

    async def edit(self, handle: MessageHandle, text: str) -> None:
        """Heartbeat tick: rewrite the placeholder's plain content."""
        h = _as_handle(handle)
        await h.message.edit(content=text)

    async def finalize(self, handle: MessageHandle, result: AgentResult) -> None:
        """Render the neutral result as an Embed + a feedback Button View.

        The answer prose goes in the embed ``description`` (truncated to
        Discord's 4096-char limit); each source becomes an embed field; the
        👍/👎 controls are two ``discord.ui.Button``s on a ``discord.ui.View``
        carrying ``custom_id`` ``up:<id>`` / ``down:<id>``. The plain-text
        ``content`` is cleared so only the embed shows.
        """
        h = _as_handle(handle)
        embed = _build_embed(
            result,
            web_ui_base_url=self._web_ui_base_url,
        )
        view = _build_feedback_view(result.investigation_id)
        # ``view`` is None when there's no investigation_id to anchor a vote.
        kwargs: dict[str, Any] = {"content": None, "embed": embed}
        if view is not None:
            kwargs["view"] = view
        await h.message.edit(**kwargs)

    async def react(
        self, channel_id: str, message_id: str, kind: ReactionKind,
    ) -> None:
        """Best-effort reaction on the inbound message (👀 / ✅ / ❌).

        Discord reactions are added to a ``discord.Message`` object, not to a
        bare ``(channel_id, message_id)`` pair. We retain the inbound Message in
        an in-memory cache keyed by its id (populated by ``on_message``); if it
        was evicted / never retained, this is a clean no-op.
        """
        client = self._client
        if client is None:
            return
        emoji = _REACTION_EMOJI.get(kind)
        if not emoji:
            return
        message = _RETAINED_MESSAGES.get(message_id)
        if message is None:
            return  # not retained -> clean no-op (best-effort contract)
        try:
            await message.add_reaction(emoji)
        except Exception as exc:  # noqa: BLE001 - best-effort
            _log.debug("discord add_reaction failed err=%s", exc)

    async def fetch_thread(
        self, channel_id: str, thread_id: str, *, cap: int,
    ) -> list[ThreadMessage]:
        """Fetch a Discord ``Thread``'s history and map to ``ThreadMessage``.

        Only Discord ``Thread`` channels have a replies model; a plain guild
        text channel or DM has none, so we return ``[]`` there (the core then
        assembles no prior-thread block). ``source_id`` is the platform message
        id so the core can drop the *triggering* message from the context block;
        ``is_self`` marks our own past replies (our bot user id) so they're
        filtered, while OTHER bots' messages are kept (they may carry an alert
        payload the user is asking us to triage).
        """
        client = self._require_client()
        channel = await self._resolve_channel(thread_id)
        if channel is None or not _is_thread(channel):
            return []
        bot_user_id = _user_id_of(client.user)
        out: list[ThreadMessage] = []
        async for m in channel.history(limit=cap):
            tm = _history_message_to_thread(m, bot_user_id)
            if tm is not None:
                out.append(tm)
        # ``history`` yields newest-first; the core's serializer is
        # order-tolerant (it truncates from newest), but return
        # oldest-first to mirror natural reading order.
        out.reverse()
        return out

    async def resolve_identity(self, msg: InboundMessage) -> CurrentUser:
        """Synthetic, traceable-but-anonymous identity.

        ``discord-bot:<guild_id_or_dm>:<user_id>`` -- ``workspace`` is the guild
        id (or ``"dm"`` for direct messages), filled by the inbound mapper.
        """
        return _discord_user_to_current_user(msg)

    async def send_denial(self, msg: InboundMessage, reason: str) -> None:
        """Privately DM the user the denial reason (no public channel noise).

        We reach the user via the retained inbound Message's author (so we can
        open a DM channel even from a guild context); if the message wasn't
        retained, this is a no-op.
        """
        message = _RETAINED_MESSAGES.get(msg.message_id)
        if message is None:
            return
        author = getattr(message, "author", None)
        if author is None:
            return
        try:
            await author.send(reason)
        except Exception as exc:  # noqa: BLE001 - best-effort
            _log.warning("discord denial DM failed: %s", exc)

    async def confirm_feedback(self, fb: FeedbackEvent, *, accepted: bool) -> None:
        """Ephemeral confirm via the stashed button ``Interaction``.

        ``on_interaction`` stashes the click's ``discord.Interaction`` in
        ``fb.raw["interaction"]`` so this method can respond ephemerally (only
        the clicker sees it).
        """
        interaction = (fb.raw or {}).get("interaction")
        if interaction is None:
            return
        confirm_text = (
            "👍 Thanks -- recorded as helpful."
            if fb.thumbs == "up"
            else "👎 Thanks -- recorded as wrong. We'll learn from this."
        )
        try:
            await interaction.response.send_message(confirm_text, ephemeral=True)
        except Exception as exc:  # noqa: BLE001 - best-effort
            _log.warning("discord feedback confirm failed: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _require_client(self) -> Any:
        if self._client is None:
            raise RuntimeError("DiscordAdapter.connect() must run before this call")
        return self._client

    async def _resolve_channel(self, channel_id: str) -> Any:
        """Resolve a channel/thread id to a sendable Discord channel object.

        Tries the gateway cache first (``get_channel``) and falls back to an
        API fetch (``fetch_channel``). Returns ``None`` when the id can't be
        parsed or resolved.
        """
        client = self._client
        if client is None:
            return None
        try:
            cid = int(channel_id)
        except (TypeError, ValueError):
            return None
        channel = client.get_channel(cid)
        if channel is not None:
            return channel
        fetch = getattr(client, "fetch_channel", None)
        if fetch is None:
            return None
        try:
            return await fetch(cid)
        except Exception as exc:  # noqa: BLE001
            _log.debug("discord fetch_channel failed id=%s err=%s", cid, exc)
            return None


# =====================================================================
# Module-level neutral mappers (SDK-free, duck-typed -- unit-testable)
# =====================================================================

# In-memory retention of inbound Messages so ``react`` / ``send_denial`` can act
# on the real ``discord.Message`` (reactions + DMs need the object, not a bare
# id). Bounded so a busy server can't grow it without limit -- oldest entries
# are evicted FIFO. This is per-process state on the single discordbot worker.
_RETAINED_MESSAGES: dict[str, Any] = {}
_RETAIN_CAP = 512


def _retain_message(message_id: str, message: Any) -> None:
    """Cache the inbound Message for later react/DM, evicting oldest past cap."""
    if not message_id:
        return
    _RETAINED_MESSAGES[message_id] = message
    while len(_RETAINED_MESSAGES) > _RETAIN_CAP:
        # dicts preserve insertion order -> pop the oldest key.
        oldest = next(iter(_RETAINED_MESSAGES))
        _RETAINED_MESSAGES.pop(oldest, None)


def _user_id_of(user: Any) -> str | None:
    """Extract a stable str id from a discord user/member object (or None)."""
    if user is None:
        return None
    uid = getattr(user, "id", None)
    return str(uid) if uid is not None else None


def _is_dm_channel(channel: Any) -> bool:
    """True iff the channel is a Discord DM.

    Uses ``discord.DMChannel`` when the SDK is importable; falls back to a
    duck-typed check (``type`` attr / class name) so the mapper stays testable
    without the SDK installed.
    """
    try:
        import discord
        if isinstance(channel, discord.DMChannel):
            return True
    except Exception:  # noqa: BLE001 - SDK absent in unit CI
        pass
    # Duck-typed fallback: discord.ChannelType.private / class name.
    ctype = getattr(channel, "type", None)
    if ctype is not None and str(getattr(ctype, "name", ctype)) in ("private", "dm"):
        return True
    return type(channel).__name__ == "DMChannel"


def _is_thread(channel: Any) -> bool:
    """True iff the channel is a Discord ``Thread``.

    SDK-aware with a duck-typed fallback (class name / channel type) so the
    mapper is testable without ``discord.py``.
    """
    try:
        import discord
        if isinstance(channel, discord.Thread):
            return True
    except Exception:  # noqa: BLE001 - SDK absent in unit CI
        pass
    ctype = getattr(channel, "type", None)
    if ctype is not None and "thread" in str(getattr(ctype, "name", ctype)).lower():
        return True
    return "Thread" in type(channel).__name__


def _is_bot_mentioned(message: Any, bot_user: Any) -> bool:
    """True iff the bot user is mentioned in ``message``.

    Prefers the structured ``message.mentions`` list (ids), falling back to a
    raw-text scan for the bot's ``<@id>`` token.
    """
    bot_id = _user_id_of(bot_user)
    if bot_id is None:
        return False
    for m in getattr(message, "mentions", None) or []:
        if _user_id_of(m) == bot_id:
            return True
    content = getattr(message, "content", "") or ""
    return f"<@{bot_id}>" in content or f"<@!{bot_id}>" in content


def _message_to_inbound(message: Any, bot_user: Any) -> InboundMessage | None:
    """Map a ``discord.Message`` to ``InboundMessage`` (or ``None`` to drop it).

    Drops, in order:
      * our own + any other bot's messages (``author.bot``) -- bot-loop
        avoidance BEFORE the sink is ever called;
      * messages that are neither a DM nor a bot @mention in a guild channel
        (no trigger);
      * messages with no text left after stripping the mention (nothing to
        answer).

    Sets ``is_dm`` from the channel type, ``thread_id`` to the channel id when
    the channel is a ``Thread`` (else ``None``), and ``workspace`` to the guild
    id (or ``"dm"``). Retains the Message so ``react`` / ``send_denial`` can
    later act on it.
    """
    author = getattr(message, "author", None)
    # Drop bot-loop messages (our own bot AND other bots) before the sink.
    if author is None or getattr(author, "bot", False):
        return None

    channel = getattr(message, "channel", None)
    is_dm = _is_dm_channel(channel)

    # Trigger only on a DM or a bot @mention in a (guild) text channel.
    if not is_dm and not _is_bot_mentioned(message, bot_user):
        return None

    raw_text = getattr(message, "content", "") or ""
    text = _MENTION_RE.sub("", raw_text).strip()
    if not text:
        return None

    channel_id = str(getattr(channel, "id", "") or "")
    user_id = _user_id_of(author) or ""
    message_id = str(getattr(message, "id", "") or "")
    thread_id = channel_id if _is_thread(channel) else None

    guild = getattr(message, "guild", None)
    if is_dm or guild is None:
        workspace = "dm"
    else:
        guild_id = getattr(guild, "id", None)
        workspace = str(guild_id) if guild_id is not None else "dm"

    _retain_message(message_id, message)

    return InboundMessage(
        channel_id=channel_id,
        user_id=user_id,
        text=text,
        message_id=message_id,
        thread_id=thread_id,
        is_dm=is_dm,
        workspace=workspace,
    )


def _interaction_to_feedback(interaction: Any) -> FeedbackEvent | None:
    """Parse a Discord button ``Interaction`` into a ``FeedbackEvent``.

    Our feedback buttons carry ``custom_id`` ``up:<investigation_id>`` /
    ``down:<investigation_id>``. Returns ``None`` for any interaction that is
    not one of our feedback buttons (wrong ``custom_id`` shape / missing data),
    so the sink is only called for real votes. The originating ``Interaction``
    is stashed in ``raw["interaction"]`` so ``confirm_feedback`` can reply
    ephemerally to the clicker.
    """
    data = getattr(interaction, "data", None) or {}
    custom_id = ""
    if isinstance(data, dict):
        custom_id = data.get("custom_id") or ""
    else:  # duck-typed: some shims expose ``data.custom_id``
        custom_id = getattr(data, "custom_id", "") or ""
    if ":" not in custom_id:
        return None
    thumbs, investigation_id = custom_id.split(":", 1)
    if thumbs not in ("up", "down") or not investigation_id:
        return None

    user = getattr(interaction, "user", None)
    user_id = _user_id_of(user) or "discord-unknown"

    channel = getattr(interaction, "channel", None)
    thread_id = str(channel.id) if channel is not None and _is_thread(channel) else None

    return FeedbackEvent(
        thumbs=thumbs,
        investigation_id=investigation_id,
        user_id=user_id,
        thread_id=thread_id,
        raw={"interaction": interaction},
    )


def _history_message_to_thread(message: Any, bot_user_id: str | None) -> ThreadMessage | None:
    """Map one ``Thread.history`` message to a ``ThreadMessage`` (or ``None``).

    Skips empty messages. ``is_self`` is True only for OUR bot's own past
    replies (``author.id == bot_user_id``) so the core filters them; other bots
    (alerting tools) keep ``is_self=False`` so the agent still sees their
    payload. ``source_id`` is the platform message id so the core can drop the
    triggering message from the context block.
    """
    text = (getattr(message, "content", "") or "").strip()
    if not text:
        return None
    author = getattr(message, "author", None)
    author_id = _user_id_of(author)
    is_self = bot_user_id is not None and author_id == bot_user_id
    # display_name (server nickname) when present, else username, else id.
    display = (
        getattr(author, "display_name", None)
        or getattr(author, "name", None)
        or author_id
        or "user"
    )
    return ThreadMessage(
        author=str(display),
        text=text,
        is_self=is_self,
        source_id=str(getattr(message, "id", "") or "") or None,
    )


def _discord_user_to_current_user(msg: InboundMessage) -> CurrentUser:
    """Synthetic ``discord-bot:<guild_or_dm>:<user>`` identity (anonymous).

    Mirrors the Slack ``slack-bot:<workspace>:<user>`` shape: traceable but not
    authenticated (``is_anonymous`` stays True so admin gates fail closed).
    """
    from dataclasses import replace

    workspace = msg.workspace or "dm"
    user = msg.user_id or "unknown-user"
    oid = f"discord-bot:{workspace}:{user}"
    return replace(CurrentUser.anonymous(), oid=oid)


# =====================================================================
# Render (Embed + feedback View)
# =====================================================================

def _truncate(text: str, cap: int) -> str:
    """Clip ``text`` to ``cap`` chars, appending a truncation marker if cut."""
    if len(text) <= cap:
        return text
    head = text[: max(0, cap - len(_TRUNCATION_SUFFIX))]
    return head.rstrip() + _TRUNCATION_SUFFIX


def _source_field(src: dict[str, Any]) -> tuple[str, str] | None:
    """Map one source dict to an ``(name, value)`` embed-field pair (or None)."""
    if not isinstance(src, dict):
        return None
    title = (src.get("title") or src.get("source") or "source").strip()
    url = (src.get("url") or "").strip()
    path = (src.get("source") or "").strip()
    if url:
        value = url
    elif path:
        value = f"`{path}`"
    else:
        value = title
    name = title or "source"
    return name[:256], value[:_EMBED_FIELD_VALUE_MAX]


def _build_embed(result: AgentResult, *, web_ui_base_url: str = "") -> Any:
    """Build a ``discord.Embed`` from the neutral result.

    The answer prose becomes the embed ``description`` (capped at Discord's
    4096-char limit), each source becomes a field (capped at 10), a diagram
    callout + a deep link footer are added when available. The ``discord`` SDK
    is imported lazily here -- the render path is exercised in tests via a
    duck-typed ``discord`` stub.
    """
    import discord  # lazy

    description = _truncate(result.answer or "", _EMBED_DESCRIPTION_MAX)
    embed = discord.Embed(description=description)

    if result.diagram_present:
        embed.add_field(
            name="Diagram",
            value="Diagram available -- open in the OpsRAG UI for the full visual.",
            inline=False,
        )

    for src in (result.sources or [])[:_MAX_SOURCE_FIELDS]:
        field = _source_field(src)
        if field is None:
            continue
        name, value = field
        embed.add_field(name=name, value=value, inline=False)

    extra = len(result.sources or []) - _MAX_SOURCE_FIELDS
    if extra > 0:
        embed.add_field(name="More sources", value=f"+{extra} more", inline=False)

    deep_link = _deep_link(web_ui_base_url, result.session_id)
    footer = "OpsRAG"
    if deep_link:
        footer += f" · {deep_link}"
    embed.set_footer(text=footer)
    return embed


def _deep_link(web_ui_base_url: str, session_id: str | None) -> str | None:
    base = (web_ui_base_url or "").rstrip("/")
    if not base:
        return None
    if session_id:
        return f"{base}/#chat/{session_id}"
    return base


def _build_feedback_view(investigation_id: str | None) -> Any:
    """Build a ``discord.ui.View`` with 👍/👎 buttons, or ``None``.

    Returns ``None`` when there's no ``investigation_id`` to anchor the vote
    (no anchor -> no place to record feedback, so we omit the controls). Each
    button's ``custom_id`` is ``up:<id>`` / ``down:<id>`` -- the same format
    ``_interaction_to_feedback`` parses. The ``discord`` SDK is imported lazily.
    """
    if not investigation_id:
        return None

    import discord  # lazy

    view = discord.ui.View(timeout=None)
    up = discord.ui.Button(
        label="👍 Helpful",
        style=discord.ButtonStyle.success,
        custom_id=f"up:{investigation_id}",
    )
    down = discord.ui.Button(
        label="👎 Wrong",
        style=discord.ButtonStyle.danger,
        custom_id=f"down:{investigation_id}",
    )
    view.add_item(up)
    view.add_item(down)
    return view


def _as_handle(handle: MessageHandle) -> _DiscordHandle:
    if not isinstance(handle, _DiscordHandle):
        raise TypeError(f"expected _DiscordHandle, got {type(handle)!r}")
    return handle
