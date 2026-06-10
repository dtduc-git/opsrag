"""In-memory ``ChannelAdapter`` for testing the core flow.

``FakeAdapter`` implements every port method without any platform SDK or
network. It records each outbound call into a public list so a test can
assert real behaviour (what was posted/edited/finalized, which reactions
fired, denials/confirms sent), feed scripted threads, and verify identity
resolution.

The handle returned by ``post_placeholder`` is a small ``_Handle`` object
so ``edit``/``finalize`` can be correlated back to a placeholder.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
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


@dataclass
class _Handle:
    """Opaque per-message token the fake hands back to edit/finalize."""

    channel_id: str
    thread_id: str | None
    seq: int


@dataclass
class FakeAdapter(ChannelAdapter):
    """A scriptable, fully in-memory adapter.

    Configure inbound behaviour via the ``thread_messages`` /
    ``resolve_identity_*`` knobs; read outbound behaviour off the public
    record lists after driving ``dispatcher.on_message`` / ``on_feedback``.
    """

    name: str = "fake"

    # --- scripted inbound knobs -------------------------------------------
    thread_messages: list[ThreadMessage] = field(default_factory=list)
    identity_oid: str | None = None  # default synthetic oid derived per-msg
    identity_email: str | None = None
    identity_name: str | None = None

    # --- recorded outbound calls (public, asserted by tests) --------------
    sink: CoreSink | None = None
    connected: bool = False
    closed: bool = False
    posted: list[dict[str, Any]] = field(default_factory=list)
    edited: list[dict[str, Any]] = field(default_factory=list)
    finalized: list[AgentResult] = field(default_factory=list)
    reactions: list[tuple[str, str, ReactionKind]] = field(default_factory=list)
    denials: list[tuple[InboundMessage, str]] = field(default_factory=list)
    confirms: list[tuple[FeedbackEvent, bool]] = field(default_factory=list)
    fetched_threads: list[tuple[str, str, int]] = field(default_factory=list)
    resolved: list[InboundMessage] = field(default_factory=list)

    _seq: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def connect(self, sink: CoreSink) -> None:
        self.sink = sink
        self.connected = True

    async def close(self) -> None:
        self.closed = True

    # ------------------------------------------------------------------
    # Outbound primitives
    # ------------------------------------------------------------------
    async def post_placeholder(
        self, channel_id: str, thread_id: str | None, text: str,
    ) -> MessageHandle:
        self._seq += 1
        handle = _Handle(channel_id=channel_id, thread_id=thread_id, seq=self._seq)
        self.posted.append(
            {"channel_id": channel_id, "thread_id": thread_id, "text": text,
             "handle": handle},
        )
        return handle

    async def edit(self, handle: MessageHandle, text: str) -> None:
        self.edited.append({"handle": handle, "text": text})

    async def finalize(self, handle: MessageHandle, result: AgentResult) -> None:
        self.finalized.append(result)

    async def react(
        self, channel_id: str, message_id: str, kind: ReactionKind,
    ) -> None:
        self.reactions.append((channel_id, message_id, kind))

    async def fetch_thread(
        self, channel_id: str, thread_id: str, *, cap: int,
    ) -> list[ThreadMessage]:
        self.fetched_threads.append((channel_id, thread_id, cap))
        return list(self.thread_messages)

    async def resolve_identity(self, msg: InboundMessage) -> CurrentUser:
        self.resolved.append(msg)
        oid = self.identity_oid or self._synthetic_oid(msg)
        base = CurrentUser.anonymous()
        return replace(
            base,
            oid=oid,
            email=self.identity_email,
            name=self.identity_name,
        )

    async def send_denial(self, msg: InboundMessage, reason: str) -> None:
        self.denials.append((msg, reason))

    async def confirm_feedback(self, fb: FeedbackEvent, *, accepted: bool) -> None:
        self.confirms.append((fb, accepted))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _synthetic_oid(self, msg: InboundMessage) -> str:
        workspace = msg.workspace or "unknown"
        return f"{self.name}-bot:{workspace}:{msg.user_id or 'unknown-user'}"
