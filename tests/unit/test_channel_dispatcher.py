"""Unit tests for the channel-neutral ChannelDispatcher (the shared flow).

Covers design 8.2: permission deny, empty text, happy path, agent error /
empty final, thread-context serialization (self-filter + greedy-newest
truncation), feedback happy + malformed, and the _session_thread_id matrix.

The agent is stubbed by monkeypatching
``opsrag.channels.dispatcher.query_with_session_events`` with an async
generator -- no real graph, no network. Assertions are on REAL behaviour
(what was posted/finalized/reacted, whether the agent was called with the
right thread_id + user oid, whether quota moved), not "no exception".
"""
from __future__ import annotations

from typing import Any

import pytest

import opsrag.channels.dispatcher as dispatcher_mod
from opsrag.channels.adapters.fake import FakeAdapter
from opsrag.channels.dispatcher import ChannelDispatcher, _serialize_thread_context
from opsrag.channels.permission import ChannelPermission
from opsrag.channels.types import (
    AgentResult,
    FeedbackEvent,
    InboundMessage,
    ReactionKind,
    ThreadMessage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _msg(
    *,
    channel_id: str = "C123",
    user_id: str = "U1",
    text: str = "why is the deploy red?",
    message_id: str = "m1",
    thread_id: str | None = None,
    is_dm: bool = False,
    workspace: str | None = "W1",
) -> InboundMessage:
    return InboundMessage(
        channel_id=channel_id,
        user_id=user_id,
        text=text,
        message_id=message_id,
        thread_id=thread_id,
        is_dm=is_dm,
        workspace=workspace,
    )


def _stub_final(monkeypatch, final_event: dict[str, Any], *, captured: dict) -> None:
    """Monkeypatch the agent to yield exactly one 'final' event.

    Captures the kwargs the dispatcher called the agent with into
    ``captured`` so tests can assert thread_id / user_id / query.
    """

    async def fake_query(graph, **kwargs):  # noqa: ANN001
        captured.update(kwargs)
        captured["graph"] = graph
        yield {"type": "node_start", "node": "route_query"}
        yield final_event

    monkeypatch.setattr(dispatcher_mod, "query_with_session_events", fake_query)


def _make_dispatcher(
    adapter: FakeAdapter,
    *,
    permission: ChannelPermission | None = None,
    investigation_cache: Any = None,
    feedback_store: Any = None,
) -> ChannelDispatcher:
    return ChannelDispatcher(
        adapter=adapter,
        agent_graph=object(),
        providers=object(),
        permission=permission or ChannelPermission(allowed_channels={"C123"}),
        web_ui_base_url="https://opsrag.example",
        investigation_cache=investigation_cache,
        feedback_store=feedback_store,
    )


class _RecordingCache:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def record_feedback(self, investigation_id, *, thumbs, correction):  # noqa: ANN001
        self.calls.append(
            {"id": investigation_id, "thumbs": thumbs, "correction": correction},
        )


class _RecordingStore:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def record(self, **kwargs):  # noqa: ANN003
        self.calls.append(kwargs)


# ---------------------------------------------------------------------------
# Permission deny
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_permission_deny_sends_denial_and_skips_agent(monkeypatch) -> None:
    adapter = FakeAdapter()
    captured: dict = {}
    _stub_final(monkeypatch, {"type": "final", "answer": "x"}, captured=captured)
    # Channel not allowlisted => deny with a user-facing reason.
    disp = _make_dispatcher(
        adapter, permission=ChannelPermission(allowed_channels={"C999"}),
    )

    await disp.on_message(_msg(channel_id="C123"))

    assert len(adapter.denials) == 1
    _, reason = adapter.denials[0]
    assert isinstance(reason, str) and reason
    # Agent must NOT have run.
    assert captured == {}
    assert adapter.posted == []
    assert adapter.finalized == []


@pytest.mark.asyncio
async def test_silent_deny_no_denial_message(monkeypatch) -> None:
    # Missing channel id on a non-DM message => silent fail-closed: no
    # placeholder, no denial DM.
    adapter = FakeAdapter()
    captured: dict = {}
    _stub_final(monkeypatch, {"type": "final", "answer": "x"}, captured=captured)
    disp = _make_dispatcher(adapter)

    await disp.on_message(_msg(channel_id="", is_dm=False))

    assert adapter.denials == []
    assert adapter.posted == []
    assert captured == {}


# ---------------------------------------------------------------------------
# Empty text
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_empty_text_is_noop(monkeypatch) -> None:
    adapter = FakeAdapter()
    captured: dict = {}
    _stub_final(monkeypatch, {"type": "final", "answer": "x"}, captured=captured)
    disp = _make_dispatcher(adapter)

    await disp.on_message(_msg(text="   "))

    assert adapter.posted == []          # no placeholder
    assert captured == {}                # agent not called
    assert adapter.reactions == []


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_happy_path(monkeypatch) -> None:
    adapter = FakeAdapter()
    captured: dict = {}
    final = {
        "type": "final",
        "answer": "The deploy failed because the migration timed out.",
        "sources": [{"title": "runbook"}],
        "investigation_id": "inv-42",
        "thread_id": "slack-thread:C123:m1",
        "diagram": True,
    }
    _stub_final(monkeypatch, final, captured=captured)
    perm = ChannelPermission(allowed_channels={"C123"}, per_user_daily_quota=5)
    disp = _make_dispatcher(adapter, permission=perm)

    await disp.on_message(_msg(channel_id="C123", user_id="U1", message_id="m1"))

    # placeholder posted
    assert len(adapter.posted) == 1
    assert adapter.posted[0]["channel_id"] == "C123"

    # ACK reaction fired on the inbound message (then DONE on success)
    kinds = [k for (_, _, k) in adapter.reactions]
    assert ReactionKind.ACK in kinds
    assert ReactionKind.DONE in kinds
    assert ReactionKind.ERROR not in kinds

    # identity resolved
    assert len(adapter.resolved) == 1

    # agent called with the channel-prefixed thread id + the resolved oid
    assert captured["thread_id"] == "fake-thread:C123:m1"
    assert captured["user_id"] == "fake-bot:W1:U1"
    assert "why is the deploy red?" in captured["query"]

    # finalize_result carried the answer + diagram flag + ids
    assert len(adapter.finalized) == 1
    result = adapter.finalized[0]
    assert isinstance(result, AgentResult)
    assert result.answer == final["answer"]
    assert result.sources == final["sources"]
    assert result.diagram_present is True
    assert result.investigation_id == "inv-42"
    assert result.session_id == "slack-thread:C123:m1"

    # quota moved exactly once
    assert perm.usage_count("U1") == 1


@pytest.mark.asyncio
async def test_dm_happy_path_no_reactions_or_thread_fetch(monkeypatch) -> None:
    adapter = FakeAdapter()
    captured: dict = {}
    _stub_final(
        monkeypatch,
        {"type": "final", "answer": "hi", "sources": []},
        captured=captured,
    )
    perm = ChannelPermission(
        allowed_channels=set(), per_user_daily_quota=5, allowed_dm_users={"*"}
    )
    disp = _make_dispatcher(adapter, permission=perm)

    await disp.on_message(_msg(channel_id="D1", is_dm=True, thread_id=None))

    # DM => placeholder posted with no thread, no reactions, no thread fetch
    assert len(adapter.posted) == 1
    assert adapter.posted[0]["thread_id"] is None
    assert adapter.reactions == []
    assert adapter.fetched_threads == []
    # DM session id form
    assert captured["thread_id"] == "fake-dm:D1"
    assert len(adapter.finalized) == 1
    assert perm.usage_count("U1") == 1


# ---------------------------------------------------------------------------
# Agent error / empty final
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_agent_error_event_finalizes_text_and_error_reaction(monkeypatch) -> None:
    adapter = FakeAdapter()
    perm = ChannelPermission(allowed_channels={"C123"}, per_user_daily_quota=5)

    async def fake_query(graph, **kwargs):  # noqa: ANN001
        yield {"type": "error", "detail": "boom"}

    monkeypatch.setattr(dispatcher_mod, "query_with_session_events", fake_query)
    disp = _make_dispatcher(adapter, permission=perm)

    await disp.on_message(_msg(channel_id="C123", user_id="U1", message_id="m1"))

    # error finalize_text went through edit (not finalize_result)
    assert adapter.finalized == []
    assert adapter.edited, "expected an error finalize via edit()"
    last_edit = adapter.edited[-1]["text"]
    assert "Sorry" in last_edit  # ERROR_TEXT copy
    assert "(RuntimeError)" in last_edit  # error kind appended

    # ERROR reaction fired, DONE did not
    kinds = [k for (_, _, k) in adapter.reactions]
    assert ReactionKind.ERROR in kinds
    assert ReactionKind.DONE not in kinds

    # quota NOT burned on error
    assert perm.usage_count("U1") == 0


@pytest.mark.asyncio
async def test_empty_final_treated_as_error(monkeypatch) -> None:
    adapter = FakeAdapter()
    perm = ChannelPermission(allowed_channels={"C123"}, per_user_daily_quota=5)

    async def fake_query(graph, **kwargs):  # noqa: ANN001
        # no 'final' event at all
        yield {"type": "node_start", "node": "route_query"}

    monkeypatch.setattr(dispatcher_mod, "query_with_session_events", fake_query)
    disp = _make_dispatcher(adapter, permission=perm)

    await disp.on_message(_msg(channel_id="C123", user_id="U1"))

    assert adapter.finalized == []
    assert adapter.edited, "expected an error finalize via edit()"
    assert "(EmptyResult)" in adapter.edited[-1]["text"]
    kinds = [k for (_, _, k) in adapter.reactions]
    assert ReactionKind.ERROR in kinds
    assert perm.usage_count("U1") == 0  # quota not burned


@pytest.mark.asyncio
async def test_agent_raises_is_caught(monkeypatch) -> None:
    adapter = FakeAdapter()
    perm = ChannelPermission(allowed_channels={"C123"}, per_user_daily_quota=5)

    async def fake_query(graph, **kwargs):  # noqa: ANN001
        raise ValueError("kaboom")
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(dispatcher_mod, "query_with_session_events", fake_query)
    disp = _make_dispatcher(adapter, permission=perm)

    await disp.on_message(_msg(channel_id="C123", user_id="U1"))

    assert adapter.finalized == []
    assert "(ValueError)" in adapter.edited[-1]["text"]
    assert perm.usage_count("U1") == 0


# ---------------------------------------------------------------------------
# Thread-context serialization
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_thread_context_serialized_and_prepended(monkeypatch) -> None:
    adapter = FakeAdapter()
    adapter.thread_messages = [
        ThreadMessage(author="Rootly", text="P1 deploy failed", is_self=False),
        ThreadMessage(author="OpsRAG", text="my earlier reply", is_self=True),
        ThreadMessage(author="Alice", text="any idea why?", is_self=False),
    ]
    captured: dict = {}
    _stub_final(
        monkeypatch,
        {"type": "final", "answer": "ok", "sources": []},
        captured=captured,
    )
    disp = _make_dispatcher(adapter)

    # thread_id != message_id => an existing thread, so context is fetched
    await disp.on_message(
        _msg(channel_id="C123", message_id="m9", thread_id="t1", text="follow up"),
    )

    assert adapter.fetched_threads == [("C123", "t1", 20)]
    combined = captured["query"]
    assert combined.startswith("PRIOR THREAD MESSAGES:")
    # self message dropped, others kept in order
    assert "Rootly: P1 deploy failed" in combined
    assert "Alice: any idea why?" in combined
    assert "my earlier reply" not in combined
    # the actual user question is appended after the context block
    assert combined.rstrip().endswith("follow up")


@pytest.mark.asyncio
async def test_thread_root_mention_skips_fetch(monkeypatch) -> None:
    # When the mention IS the thread root (thread_id == message_id) there
    # is no prior context -- fetch is skipped.
    adapter = FakeAdapter()
    adapter.thread_messages = [
        ThreadMessage(author="Alice", text="hi", is_self=False),
    ]
    captured: dict = {}
    _stub_final(
        monkeypatch,
        {"type": "final", "answer": "ok", "sources": []},
        captured=captured,
    )
    disp = _make_dispatcher(adapter)

    await disp.on_message(_msg(message_id="m1", thread_id="m1", text="root q"))

    assert adapter.fetched_threads == []
    assert captured["query"] == "root q"  # no prefix


def test_serialize_drops_self_and_empties() -> None:
    msgs = [
        ThreadMessage(author="Bot", text="mine", is_self=True),
        ThreadMessage(author="Alice", text="  ", is_self=False),  # empty
        ThreadMessage(author="Bob", text="real msg", is_self=False),
    ]
    out = _serialize_thread_context(msgs)
    assert out == "PRIOR THREAD MESSAGES:\nBob: real msg"


def test_serialize_empty_returns_blank() -> None:
    assert _serialize_thread_context([]) == ""
    assert (
        _serialize_thread_context(
            [ThreadMessage(author="Bot", text="x", is_self=True)],
        )
        == ""
    )


def test_serialize_greedy_newest_truncation_drops_oldest() -> None:
    # Three lines; cap that fits only the two newest -> oldest dropped.
    msgs = [
        ThreadMessage(author="A", text="oldest" + "o" * 40, is_self=False),
        ThreadMessage(author="B", text="middle" + "m" * 40, is_self=False),
        ThreadMessage(author="C", text="newest" + "n" * 40, is_self=False),
    ]
    out = _serialize_thread_context(msgs, max_chars=120)
    assert "C: newest" in out         # newest kept
    assert "A: oldest" not in out     # oldest dropped first
    assert out.startswith("PRIOR THREAD MESSAGES:")


def test_serialize_single_overflowing_line_truncated_with_ellipsis() -> None:
    msgs = [ThreadMessage(author="A", text="x" * 5000, is_self=False)]
    out = _serialize_thread_context(msgs, max_chars=100)
    assert out.startswith("PRIOR THREAD MESSAGES:")
    assert out.endswith("...")
    assert len(out) <= 110  # header + truncated line, roughly bounded


def test_serialize_excludes_triggering_message_by_source_id() -> None:
    # The thread fetch returns the whole thread INCLUDING the message the
    # user just sent (Slack conversations.replies does). The core must drop
    # it (by source_id) so the question isn't duplicated into the prompt.
    msgs = [
        ThreadMessage(author="Bob", text="context", is_self=False, source_id="b1"),
        ThreadMessage(author="Alice", text="the question", is_self=False, source_id="m9"),
    ]
    out = _serialize_thread_context(msgs, exclude_id="m9")
    assert "Alice: the question" not in out   # triggering message dropped
    assert "Bob: context" in out              # real prior context kept
    # Without exclude_id the message is kept (back-compat / default).
    assert "Alice: the question" in _serialize_thread_context(msgs)


@pytest.mark.asyncio
async def test_thread_context_does_not_duplicate_triggering_message(monkeypatch) -> None:
    # Regression: an in-thread mention must not echo the user's own question
    # into the PRIOR THREAD MESSAGES block (it is already the primary query).
    adapter = FakeAdapter()
    adapter.thread_messages = [
        ThreadMessage(author="Rootly", text="P1 deploy failed", is_self=False, source_id="r1"),
        # The triggering message as conversations.replies echoes it back:
        ThreadMessage(author="Alice", text="why did it fail?", is_self=False, source_id="m9"),
    ]
    captured: dict = {}
    _stub_final(
        monkeypatch,
        {"type": "final", "answer": "ok", "sources": []},
        captured=captured,
    )
    disp = _make_dispatcher(adapter)

    await disp.on_message(
        _msg(channel_id="C123", message_id="m9", thread_id="t1", text="why did it fail?"),
    )

    combined = captured["query"]
    # The question appears EXACTLY once (the appended primary query), not
    # also inside the thread block.
    assert combined.count("why did it fail?") == 1
    assert "Alice: why did it fail?" not in combined   # echoed copy dropped
    assert "Rootly: P1 deploy failed" in combined      # real context survives
    assert combined.rstrip().endswith("why did it fail?")


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_feedback_happy_path() -> None:
    adapter = FakeAdapter()
    cache = _RecordingCache()
    store = _RecordingStore()
    disp = _make_dispatcher(
        adapter, investigation_cache=cache, feedback_store=store,
    )

    fb = FeedbackEvent(
        thumbs="up", investigation_id="inv-7", user_id="U9", thread_id="t1",
    )
    await disp.on_feedback(fb)

    # both stores written
    assert cache.calls == [{"id": "inv-7", "thumbs": "up", "correction": None}]
    assert len(store.calls) == 1
    assert store.calls[0]["investigation_id"] == "inv-7"
    assert store.calls[0]["direction"] == 1
    # user namespaced by channel name
    assert store.calls[0]["user_id"] == "fake:U9"
    # ephemeral confirm sent
    assert adapter.confirms == [(fb, True)]


@pytest.mark.asyncio
async def test_feedback_threads_resolved_query_answer_into_store() -> None:
    # Regression: Slack feedback rows landed in Postgres with NULL
    # query/answer snippets, so the Retrieval-Quality "needs attention"
    # cards were blank. When the cache resolves the investigation it now
    # returns query+answer; the feedback path must persist them as snippets.
    from opsrag.agent.cache.investigation_cache import FeedbackResult

    class _CacheWithPayload:
        async def record_feedback(self, investigation_id, *, thumbs, correction):  # noqa: ANN001
            return FeedbackResult(
                ok=True,
                point_id=investigation_id,
                query="why is passport-be 400ing?",
                answer="rs-backend rejects the org sync payload",
            )

    adapter = FakeAdapter()
    store = _RecordingStore()
    disp = _make_dispatcher(
        adapter, investigation_cache=_CacheWithPayload(), feedback_store=store,
    )
    fb = FeedbackEvent(
        thumbs="down", investigation_id="inv-7", user_id="U9", thread_id="t1",
    )
    await disp.on_feedback(fb)

    assert store.calls[0]["query_snippet"] == "why is passport-be 400ing?"
    assert store.calls[0]["answer_snippet"] == "rs-backend rejects the org sync payload"


@pytest.mark.asyncio
async def test_feedback_uses_payload_answer_snippet_when_cache_resolves_nothing() -> None:
    # Ungrounded / LOW-confidence answers aren't cached -> the cache resolves no
    # investigation (record_feedback returns falsy). The answer snippet captured
    # from the click payload must still reach the store so the dashboard card
    # isn't blank.
    adapter = FakeAdapter()
    store = _RecordingStore()
    disp = _make_dispatcher(
        adapter, investigation_cache=_RecordingCache(), feedback_store=store,
    )
    fb = FeedbackEvent(
        thumbs="down", investigation_id="slack-thread:C1:1.1", user_id="U9",
        thread_id="t1", answer_snippet="Unverified: the deploy failed.",
    )
    await disp.on_feedback(fb)
    assert store.calls[0]["answer_snippet"] == "Unverified: the deploy failed."


@pytest.mark.asyncio
async def test_feedback_down_direction() -> None:
    adapter = FakeAdapter()
    store = _RecordingStore()
    disp = _make_dispatcher(adapter, feedback_store=store)
    fb = FeedbackEvent(
        thumbs="down", investigation_id="inv-1", user_id="U1", thread_id=None,
    )
    await disp.on_feedback(fb)
    assert store.calls[0]["direction"] == -1
    assert adapter.confirms == [(fb, True)]


@pytest.mark.asyncio
async def test_feedback_malformed_ignored() -> None:
    adapter = FakeAdapter()
    cache = _RecordingCache()
    store = _RecordingStore()
    disp = _make_dispatcher(
        adapter, investigation_cache=cache, feedback_store=store,
    )

    # bad thumbs
    await disp.on_feedback(
        FeedbackEvent(
            thumbs="sideways", investigation_id="inv-1", user_id="U1",
            thread_id=None,
        ),
    )
    # missing investigation id
    await disp.on_feedback(
        FeedbackEvent(thumbs="up", investigation_id="", user_id="U1", thread_id=None),
    )

    assert cache.calls == []
    assert store.calls == []
    assert adapter.confirms == []  # no confirm for a malformed event


@pytest.mark.asyncio
async def test_feedback_store_failure_does_not_block_confirm() -> None:
    # A best-effort store that raises must not prevent the confirm.
    class _Boom:
        async def record(self, **kwargs):  # noqa: ANN003
            raise RuntimeError("db down")

    adapter = FakeAdapter()
    disp = _make_dispatcher(adapter, feedback_store=_Boom())
    fb = FeedbackEvent(
        thumbs="up", investigation_id="inv-1", user_id="U1", thread_id=None,
    )
    await disp.on_feedback(fb)
    assert adapter.confirms == [(fb, True)]


@pytest.mark.asyncio
async def test_feedback_up_reacts_thumbs_up_on_answer_message() -> None:
    adapter = FakeAdapter()
    disp = _make_dispatcher(adapter)
    fb = FeedbackEvent(
        thumbs="up", investigation_id="inv-7", user_id="U9", thread_id="t1",
        channel_id="C123", message_ts="111.1",
    )
    await disp.on_feedback(fb)
    assert adapter.reactions == [("C123", "111.1", ReactionKind.THUMBS_UP)]
    # ephemeral confirm still fires alongside the public reaction.
    assert adapter.confirms == [(fb, True)]


@pytest.mark.asyncio
async def test_feedback_down_reacts_thumbs_down_on_answer_message() -> None:
    adapter = FakeAdapter()
    disp = _make_dispatcher(adapter)
    fb = FeedbackEvent(
        thumbs="down", investigation_id="inv-8", user_id="U9", thread_id="t1",
        channel_id="C123", message_ts="222.2",
    )
    await disp.on_feedback(fb)
    assert adapter.reactions == [("C123", "222.2", ReactionKind.THUMBS_DOWN)]
    assert adapter.confirms == [(fb, True)]


@pytest.mark.asyncio
async def test_feedback_missing_coords_skips_react_but_still_confirms() -> None:
    adapter = FakeAdapter()
    disp = _make_dispatcher(adapter)
    fb = FeedbackEvent(
        thumbs="up", investigation_id="inv-9", user_id="U9", thread_id="t1",
        # channel_id / message_ts left at their default "" -- no coords.
    )
    await disp.on_feedback(fb)
    assert adapter.reactions == []
    assert adapter.confirms == [(fb, True)]


@pytest.mark.asyncio
async def test_feedback_react_failure_does_not_block_confirm() -> None:
    # A react that raises must not prevent the ephemeral confirm.
    class _BoomAdapter(FakeAdapter):
        async def react(self, channel_id, message_id, kind):  # noqa: ANN001
            raise RuntimeError("slack api down")

    adapter = _BoomAdapter()
    disp = _make_dispatcher(adapter)
    fb = FeedbackEvent(
        thumbs="up", investigation_id="inv-10", user_id="U9", thread_id="t1",
        channel_id="C123", message_ts="333.3",
    )
    await disp.on_feedback(fb)
    assert adapter.confirms == [(fb, True)]


# ---------------------------------------------------------------------------
# _session_thread_id matrix
# ---------------------------------------------------------------------------
def test_session_thread_id_dm() -> None:
    adapter = FakeAdapter()
    disp = _make_dispatcher(adapter)
    sid = disp._session_thread_id(  # noqa: SLF001
        _msg(channel_id="D1", is_dm=True, thread_id=None, message_id="m1"),
    )
    assert sid == "fake-dm:D1"


def test_session_thread_id_new_thread_uses_message_id() -> None:
    adapter = FakeAdapter()
    disp = _make_dispatcher(adapter)
    sid = disp._session_thread_id(  # noqa: SLF001
        _msg(channel_id="C1", is_dm=False, thread_id=None, message_id="m5"),
    )
    assert sid == "fake-thread:C1:m5"


def test_session_thread_id_existing_thread_uses_thread_id() -> None:
    adapter = FakeAdapter()
    disp = _make_dispatcher(adapter)
    sid = disp._session_thread_id(  # noqa: SLF001
        _msg(channel_id="C1", is_dm=False, thread_id="t9", message_id="m5"),
    )
    assert sid == "fake-thread:C1:t9"


def test_session_thread_id_prefix_is_channel_name() -> None:
    # The <ch> prefix keeps sessions disjoint across platforms.
    adapter = FakeAdapter(name="telegram")
    disp = _make_dispatcher(adapter)
    sid = disp._session_thread_id(  # noqa: SLF001
        _msg(channel_id="C1", is_dm=True, message_id="m1"),
    )
    assert sid.startswith("telegram-dm:")
