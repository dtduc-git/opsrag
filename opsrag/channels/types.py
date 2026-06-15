"""Channel-neutral value types shared by the core flow + every adapter.

These types are the lingua franca between the platform-agnostic core
(``dispatcher`` / ``streaming`` / ``permission`` / ``feedback``) and the
per-platform adapters (Slack / Telegram / Discord / Teams). The core
never sees a Slack ``event`` dict or a Telegram ``Update`` -- it only
ever sees the neutral types below. Each adapter is responsible for
normalising its platform payloads into ``InboundMessage`` /
``FeedbackEvent`` on the way in, and for rendering an ``AgentResult``
into its own surface format on the way out.

See design doc ``specs/002-channel-bots/design.md`` section 3.2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ReactionKind(str, Enum):
    """Abstract reaction the core asks the adapter to apply.

    The adapter maps each kind to its platform's emoji/affordance (Slack
    👀/✅/❌). ``react`` is best-effort -- platforms with no
    reaction-on-message (Telegram/Teams) implement these as no-ops.
    """

    ACK = "ack"      # "I picked this up" (Slack 👀)
    DONE = "done"    # success (Slack ✅)
    ERROR = "error"  # failure (Slack ❌)


@dataclass(frozen=True)
class ImageRef:
    """A lightweight, pre-fetch reference to a platform image attachment.

    Adapters emit these without downloading; the dispatcher resolves them to
    bytes (via ``adapter.fetch_image``) only AFTER the permission check passes
    (spec FR-007). At least one of ``file_id`` / ``url`` is set.
    """

    file_id: str | None = None
    url: str | None = None
    mime_type: str = "image/png"
    size: int | None = None


@dataclass(frozen=True)
class InboundMessage:
    """A normalised inbound user message, platform-stripped.

    ``text`` already has the bot-mention removed by the adapter (the core
    must not re-parse platform mention syntax). ``thread_id`` is ``None``
    for flat conversations (DMs). ``workspace`` namespaces the synthetic
    identity oid. ``raw`` is an escape hatch for adapter-specific bits the
    adapter may need when it later renders/reacts.
    """

    channel_id: str                 # platform chat/channel id
    user_id: str                    # platform user id
    text: str                       # message text, bot-mention already stripped
    message_id: str                 # id of the inbound message (reaction / reply anchor)
    thread_id: str | None           # Slack thread_ts / Discord+Teams reply id; None => flat
    is_dm: bool
    workspace: str | None           # team/guild/tenant id; namespaces the synthetic oid
    images: tuple[ImageRef, ...] = field(default_factory=tuple)
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResult:
    """The neutral result the core hands ``adapter.finalize`` to render.

    The adapter -- and only the adapter -- turns this into Block Kit /
    HTML / Embed / Adaptive Card. The core stays format-blind.
    """

    answer: str
    sources: list[dict]
    diagram_present: bool
    session_id: str | None
    investigation_id: str | None


@dataclass(frozen=True)
class FeedbackEvent:
    """A normalised 👍/👎 feedback action on a prior bot answer."""

    thumbs: str                     # "up" | "down"
    investigation_id: str
    user_id: str
    thread_id: str | None
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ThreadMessage:
    """One prior message in a thread, as returned by ``fetch_thread``.

    ``is_self`` marks our own past replies so the core can drop them from
    the assembled "PRIOR THREAD MESSAGES" block (feedback-loop avoidance).
    Other bots' messages (alerting tools etc.) keep ``is_self=False`` so
    the agent still sees the alert payload it's being asked to triage.

    ``source_id`` is the platform id of this thread message (Slack ``ts``,
    etc.). The core uses it to drop the *triggering* message from the block
    -- the thread fetch returns the whole thread INCLUDING the message the
    user just sent, and that text is already appended as the primary query,
    so without this de-dup the question would appear twice in the prompt
    (and eat the thread-context char budget). Adapters that cannot supply an
    id leave it ``None`` (never excluded).
    """

    author: str                     # display name ("Rootly", "Alice", ...)
    text: str
    is_self: bool                   # True => our own past reply (filtered by core)
    source_id: str | None = None    # platform message id; core drops the triggering msg


# MessageHandle is opaque + per-adapter: a Slack ``(channel, ts)`` tuple,
# a Telegram ``message_id``, a Discord ``Message``, a Teams
# ``ConversationReference``. The core treats it purely as a token it hands
# back to ``edit()`` / ``finalize()``; it never inspects it.
MessageHandle = object
