"""The ports: ``ChannelAdapter`` + ``CoreSink`` (typing.Protocol).

This is the hexagonal boundary. The core depends only on these two
Protocols, never on a concrete adapter. An adapter depends only on the
neutral types + ``CoreSink``. Neither side imports a platform SDK at this
layer (SDK imports are lazy, inside each adapter module).

* ``CoreSink`` -- what the dispatcher exposes; the adapter pushes
  normalised inbound events into it (``on_message`` / ``on_feedback``).
* ``ChannelAdapter`` -- what each platform implements; the dispatcher +
  ``ProgressStreamer`` drive the outbound primitives through it.

Both are ``runtime_checkable`` so tests (and ``boot``) can assert an
object satisfies the port via ``isinstance``.

See design doc ``specs/002-channel-bots/design.md`` section 3.3.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from opsrag.auth.pomerium import CurrentUser
from opsrag.channels.types import (
    AgentResult,
    FeedbackEvent,
    ImageRef,
    InboundMessage,
    MessageHandle,
    ReactionKind,
    ThreadMessage,
)


@runtime_checkable
class CoreSink(Protocol):
    """What the dispatcher hands the adapter so inbound events drive the flow."""

    async def on_message(self, msg: InboundMessage) -> None: ...

    async def on_feedback(self, fb: FeedbackEvent) -> None: ...


@runtime_checkable
class ChannelAdapter(Protocol):
    """A per-platform transport + render + identity ring.

    The adapter owns its transport, normalises inbound platform events to
    ``InboundMessage`` / ``FeedbackEvent`` and pushes them into the
    ``CoreSink``, and renders the neutral ``AgentResult`` into its own
    surface format. Rendering NEVER leaves the adapter.
    """

    name: str  # "slack" | "telegram" | "discord" | "teams"

    # --- lifecycle ---------------------------------------------------------
    # ``connect`` wires the transport to the sink. The adapter MUST drop
    # bot-loop messages (its own + other bots per channel policy) and set
    # ``is_dm`` before calling ``sink.on_message``.
    async def connect(self, sink: CoreSink) -> None: ...

    async def close(self) -> None: ...

    # --- outbound primitives (driven by ProgressStreamer + dispatcher) -----
    async def post_placeholder(
        self, channel_id: str, thread_id: str | None, text: str,
    ) -> MessageHandle: ...

    async def edit(self, handle: MessageHandle, text: str) -> None: ...  # heartbeat tick

    async def finalize(
        self, handle: MessageHandle, result: AgentResult,
    ) -> None: ...  # adapter RENDERS here

    async def react(
        self, channel_id: str, message_id: str, kind: ReactionKind,
    ) -> None: ...  # best-effort; may be a no-op

    async def fetch_thread(
        self, channel_id: str, thread_id: str, *, cap: int,
    ) -> list[ThreadMessage]: ...  # [] where the platform has no thread model

    async def fetch_image(self, ref: ImageRef) -> bytes | None:
        """Download the bytes for an inbound image reference, or None on
        failure. Called only after the permission check passes (FR-007)."""
        ...

    async def resolve_identity(self, msg: InboundMessage) -> CurrentUser: ...

    async def send_denial(self, msg: InboundMessage, reason: str) -> None: ...  # private/DM

    async def confirm_feedback(
        self, fb: FeedbackEvent, *, accepted: bool,
    ) -> None: ...  # ephemeral ack
