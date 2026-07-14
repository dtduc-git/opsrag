"""Integration tests for the self-contained FirstResponder pipeline."""
from __future__ import annotations

import pytest

import opsrag.slack_bot.first_responder as fr_mod
from opsrag.channels.config import FirstResponderChannelConfig, FirstResponderConfig
from opsrag.channels.permission import ChannelPermission
from opsrag.slack_bot.first_responder import FirstResponder

pytestmark = pytest.mark.asyncio

CHANNEL = "C0EXAMPLE01"
CONFIG = FirstResponderConfig(
    enabled=True,
    channels={
        CHANNEL: FirstResponderChannelConfig(
            request_app_allowlist=["A0EXAMPLE01"],
            oncall_handle="S0EXAMPLE01",
            include_direct=True,
        )
    },
)


class FakeClient:
    def __init__(self):
        self.self_bot_id = "BSELF"
        self.self_user_id = "USELF"
        self.posted: list[dict] = []
        self.updated: list[dict] = []
        self._ts = 0

    async def post_message(self, channel, text, thread_ts=None, blocks=None):
        self._ts += 1
        ts = f"ts-{self._ts}"
        self.posted.append({"channel": channel, "text": text, "thread_ts": thread_ts, "ts": ts})
        return ts

    async def update_message(self, channel, ts, text, blocks=None):
        self.updated.append({"channel": channel, "ts": ts, "text": text, "blocks": blocks})

    async def get_user_info(self, user_id):
        return {}


def _fake_query(final):
    async def _gen(*args, **kwargs):
        yield {"type": "final", **final}
    return _gen


def _make_fr():
    perm = ChannelPermission(allowed_channels={CHANNEL}, per_user_daily_quota=500)
    fr = FirstResponder(
        graph=object(),
        providers=object(),
        permission=perm,
        config=CONFIG,
    )
    client = FakeClient()
    fr.bind_client(client)
    return fr, client, perm


def _workflow_event(ts="100.1"):
    return {
        "type": "message", "subtype": "bot_message",
        "bot_id": "B0EXAMPLE01", "app_id": "A0EXAMPLE01",
        "channel": CHANNEL, "ts": ts,
        "text": "Environment: prd\nFull Description: pods crashing",
    }


async def test_workflow_post_produces_two_threaded_posts_with_cc(monkeypatch):
    # Phase-1 polish: the ack and the answer are TWO independent posts (not a
    # placeholder + edit). Neither is an update_message call.
    monkeypatch.setattr(
        fr_mod, "query_with_session_events",
        _fake_query({"answer": "Check the CrashLoop.", "sources": [{"title": "r"}], "grounded": True}),
    )
    fr, client, _ = _make_fr()
    await fr.on_channel_message(_workflow_event())

    assert len(client.posted) == 2                    # ack + answer
    ack, answer = client.posted
    assert ack["thread_ts"] == "100.1"                # in-thread
    assert answer["thread_ts"] == "100.1"
    assert client.updated == []                       # never an edit
    assert "<!subteam^S0EXAMPLE01>" in answer["text"]


async def test_duplicate_ts_is_deduped(monkeypatch):
    monkeypatch.setattr(
        fr_mod, "query_with_session_events",
        _fake_query({"answer": "x", "sources": [], "grounded": True}),
    )
    fr, client, _ = _make_fr()
    await fr.on_channel_message(_workflow_event(ts="200.1"))
    await fr.on_channel_message(_workflow_event(ts="200.1"))
    assert len(client.posted) == 2  # ack + answer once; second call short-circuits


async def test_message_changed_edit_is_ignored(monkeypatch):
    monkeypatch.setattr(
        fr_mod, "query_with_session_events",
        _fake_query({"answer": "x", "sources": [], "grounded": True}),
    )
    fr, client, _ = _make_fr()
    # OpsRAG's own finalize edit: message_changed with nested bot_id.
    event = {
        "type": "message", "subtype": "message_changed", "channel": CHANNEL,
        "ts": "300.1",
        "message": {"bot_id": "BSELF", "user": "USELF", "text": "edited"},
    }
    await fr.on_channel_message(event)
    assert client.posted == []
    assert client.updated == []


async def test_self_top_level_post_is_ignored(monkeypatch):
    monkeypatch.setattr(
        fr_mod, "query_with_session_events",
        _fake_query({"answer": "x", "sources": [], "grounded": True}),
    )
    fr, client, _ = _make_fr()
    event = {"type": "message", "bot_id": "BSELF", "channel": CHANNEL,
             "ts": "400.1", "text": "my own answer"}
    await fr.on_channel_message(event)
    assert client.posted == []


async def test_unmapped_channel_is_ignored(monkeypatch):
    monkeypatch.setattr(
        fr_mod, "query_with_session_events",
        _fake_query({"answer": "x", "sources": [], "grounded": True}),
    )
    fr, client, _ = _make_fr()
    event = dict(_workflow_event(), channel="COTHER")
    await fr.on_channel_message(event)
    assert client.posted == []


async def test_thread_reply_is_ignored(monkeypatch):
    monkeypatch.setattr(
        fr_mod, "query_with_session_events",
        _fake_query({"answer": "x", "sources": [], "grounded": True}),
    )
    fr, client, _ = _make_fr()
    event = dict(_workflow_event(ts="500.9"), thread_ts="500.1")
    await fr.on_channel_message(event)
    assert client.posted == []


async def test_quota_exhaustion_blocks_agent_run(monkeypatch):
    monkeypatch.setattr(
        fr_mod, "query_with_session_events",
        _fake_query({"answer": "x", "sources": [], "grounded": True}),
    )
    perm = ChannelPermission(allowed_channels={CHANNEL}, per_user_daily_quota=500)
    cfg = FirstResponderConfig(
        enabled=True,
        channels={CHANNEL: FirstResponderChannelConfig(
            request_app_allowlist=["A0EXAMPLE01"], oncall_handle="S0EXAMPLE01",
            daily_quota=1,
        )},
    )
    fr = FirstResponder(graph=object(), providers=object(), permission=perm, config=cfg)
    client = FakeClient()
    fr.bind_client(client)

    await fr.on_channel_message(_workflow_event(ts="600.1"))
    await fr.on_channel_message(_workflow_event(ts="600.2"))
    assert len(client.posted) == 2  # only the first ran (ack + answer); second hit the cap


async def test_disabled_feature_is_noop(monkeypatch):
    monkeypatch.setattr(
        fr_mod, "query_with_session_events",
        _fake_query({"answer": "x", "sources": [], "grounded": True}),
    )
    perm = ChannelPermission(allowed_channels={CHANNEL}, per_user_daily_quota=500)
    cfg = FirstResponderConfig(enabled=False, channels=CONFIG.channels)
    fr = FirstResponder(graph=object(), providers=object(), permission=perm, config=cfg)
    client = FakeClient()
    fr.bind_client(client)
    await fr.on_channel_message(_workflow_event())
    assert client.posted == []


async def test_agent_failure_finalizes_with_oncall_cc(monkeypatch):
    # Primary path: the agent stream raises mid-iteration. `_run_agent`'s
    # try/except catches it and returns None -- the ack was already posted,
    # and a SEPARATE error post (with on-call cc) follows, never dangling.
    def _raising_query(*args, **kwargs):
        async def _gen(*a, **k):
            raise RuntimeError("boom")
            yield  # pragma: no cover - makes this an async generator
        return _gen
    monkeypatch.setattr(fr_mod, "query_with_session_events", _raising_query())
    fr, client, _ = _make_fr()
    await fr.on_channel_message(_workflow_event(ts="700.1"))
    assert len(client.posted) == 2          # ack + error post
    assert client.updated == []             # never an edit
    assert "<!subteam^S0EXAMPLE01>" in client.posted[1]["text"]

    # Secondary path: the code also treats a *silent* failure the same way --
    # a stream that completes without ever emitting a `final` event raises no
    # exception, but `_run_agent`'s `final` stays None, hitting the same
    # `final is None` error-post-with-cc branch in on_channel_message.
    def _empty_query(*args, **kwargs):
        async def _gen(*a, **k):
            return
            yield  # pragma: no cover - makes this an async generator
        return _gen
    monkeypatch.setattr(fr_mod, "query_with_session_events", _empty_query())
    fr2, client2, _ = _make_fr()
    await fr2.on_channel_message(_workflow_event(ts="700.2"))
    assert len(client2.posted) == 2
    assert client2.updated == []
    assert "<!subteam^S0EXAMPLE01>" in client2.posted[1]["text"]


async def test_nested_self_bot_id_is_ignored(monkeypatch):
    monkeypatch.setattr(
        fr_mod, "query_with_session_events",
        _fake_query({"answer": "x", "sources": [], "grounded": True}),
    )
    fr, client, _ = _make_fr()
    # Non-ignorable subtype (so the subtype filter does NOT catch it), but the
    # nested message carries our own bot_id -> the nested self-guard must drop it.
    # Note: this event carries no top-level bot_id/app_id/user, so classify()
    # would independently IGNORE it too -- but _is_self() runs BEFORE classify()
    # (pipeline step 3 vs step 4) and returns True on the *nested* match, so
    # the self-guard is the step that actually fires and short-circuits here.
    event = {
        "type": "message", "channel": CHANNEL, "ts": "800.1",
        "message": {"bot_id": "BSELF", "text": "our own edit-ish payload"},
    }
    await fr.on_channel_message(event)
    assert client.posted == []


async def test_self_user_direct_post_dropped_via_live_identity(monkeypatch):
    # A human-shaped post whose `user` is OpsRAG's OWN user id. classify() would
    # treat it as DIRECT (has user, include_direct=True) and answer it, so ONLY
    # the self-guard can drop it. Identity is set AFTER bind_client, mirroring
    # the real adapter.connect ordering (bind before client.start populates ids).
    monkeypatch.setattr(
        fr_mod, "query_with_session_events",
        _fake_query({"answer": "x", "sources": [], "grounded": True}),
    )
    fr, client, _ = _make_fr()
    client.self_user_id = None   # not yet known at bind time
    client.self_bot_id = None
    fr.bind_client(client)       # re-bind with ids unknown (as in prod connect order)
    client.self_user_id = "USELF"  # auth.test completes AFTER bind
    event = {"type": "message", "channel": CHANNEL, "ts": "900.1",
             "user": "USELF", "text": "looks like a direct question"}
    await fr.on_channel_message(event)
    assert client.posted == []   # dropped by the LIVE self-guard, not classify
