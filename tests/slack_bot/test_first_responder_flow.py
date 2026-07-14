"""Async behavior tests for the Phase-1 polish FirstResponder two-post flow.

Covers: two independent in-thread posts (ack, then a SEPARATE answer post --
never an edit/update), ack-failure hard-abort, agent-error cc'd error post,
quota-recorded-only-on-success, capped mention lookups, the CLEAN normalized
query reaching the agent, resolved sender name/email, and loop-safety
(OpsRAG's own posts self-drop via `_is_self` / `classify`).

Mirrors the plain fake-object style of ``tests/channels/test_first_responder.py``.
"""
from __future__ import annotations

import pytest

import opsrag.slack_bot.first_responder as fr_mod
from opsrag.channels.config import FirstResponderChannelConfig, FirstResponderConfig
from opsrag.channels.permission import ChannelPermission
from opsrag.slack_bot.first_responder import FirstResponder

pytestmark = pytest.mark.asyncio

CHANNEL = "C0EXAMPLE01"
APP_ID = "A0EXAMPLE01"
CONFIG = FirstResponderConfig(
    enabled=True,
    channels={
        CHANNEL: FirstResponderChannelConfig(
            request_app_allowlist=[APP_ID],
            oncall_handle="S0EXAMPLE01",
            oncall_display="Ops on-call",
            include_direct=True,
        )
    },
)

# Directory the FakeClient consults for get_user_info -- default entries so
# tests that don't care about a particular id still resolve deterministically.
USER_DIR = {
    "U0EXAMPLE01": {"profile": {"display_name": "duc", "email": "duc@example.com"}},
    "U1": {"profile": {"display_name": "alice"}},
}


class FakeClient:
    def __init__(self, *, user_dir=None, fail_first_post=False):
        self.self_bot_id = "BSELF"
        self.self_user_id = "USELF"
        self.posted: list[dict] = []
        self.updated: list[dict] = []
        self.user_info_calls: list[str] = []
        self._user_dir = user_dir if user_dir is not None else USER_DIR
        self._fail_first_post = fail_first_post
        self._ts = 0

    async def post_message(self, channel, text, thread_ts=None, blocks=None):
        if self._fail_first_post and not self.posted:
            raise RuntimeError("ack post boom")
        self._ts += 1
        ts = f"ts-{self._ts}"
        self.posted.append(
            {"channel": channel, "text": text, "thread_ts": thread_ts, "ts": ts, "blocks": blocks}
        )
        return ts

    async def update_message(self, channel, ts, text, blocks=None):  # pragma: no cover - must not be called
        self.updated.append({"channel": channel, "ts": ts, "text": text, "blocks": blocks})

    async def get_user_info(self, user_id):
        self.user_info_calls.append(user_id)
        return self._user_dir.get(user_id, {})


def _fake_query(final, *, captured=None):
    async def _gen(*args, **kwargs):
        if captured is not None:
            captured.append(kwargs)
        yield {"type": "final", **final}
    return _gen


def _make_fr(config=CONFIG, *, permission=None, client=None):
    perm = permission or ChannelPermission(allowed_channels={CHANNEL}, per_user_daily_quota=500)
    fr = FirstResponder(
        graph=object(),
        providers=object(),
        permission=perm,
        config=config,
    )
    client = client or FakeClient()
    fr.bind_client(client)
    return fr, client, perm


def _workflow_event(ts="100.1", text="Requester: <@U0EXAMPLE01>\nFull Description: pods crashing"):
    return {
        "type": "message", "subtype": "bot_message",
        "bot_id": "B0EXAMPLE01", "app_id": APP_ID,
        "channel": CHANNEL, "ts": ts,
        "text": text,
    }


def _direct_event(ts="150.1", user="U0EXAMPLE01", text="how do I restart the pod?"):
    return {"type": "message", "channel": CHANNEL, "ts": ts, "user": user, "text": text}


# ---------------------------------------------------------------------------
# 1. Two posts on success -- ack then a SEPARATE answer post, never an edit.
# ---------------------------------------------------------------------------
async def test_success_produces_two_separate_posts_never_an_edit(monkeypatch):
    monkeypatch.setattr(
        fr_mod, "query_with_session_events",
        _fake_query({"answer": "Check the CrashLoop.", "sources": [{"title": "r"}], "grounded": True}),
    )
    fr, client, _ = _make_fr()
    await fr.on_channel_message(_workflow_event())

    assert len(client.posted) == 2
    ack, answer = client.posted
    assert "I am *OpsRAG*" in ack["text"]
    assert "<@U0EXAMPLE01>" in ack["text"]
    assert ack["thread_ts"] == "100.1"
    assert "Check the CrashLoop." in answer["text"]
    assert answer["thread_ts"] == "100.1"
    assert client.updated == []  # never an update -- answer is a fresh post


# ---------------------------------------------------------------------------
# 2. Ack failure -> hard-abort: no agent run, no quota recorded.
# ---------------------------------------------------------------------------
async def test_ack_post_failure_hard_aborts_no_agent_no_quota(monkeypatch):
    ran_agent = []

    def _tracking_query(*args, **kwargs):
        ran_agent.append(True)
        return _fake_query({"answer": "x", "sources": [], "grounded": True})(*args, **kwargs)

    monkeypatch.setattr(fr_mod, "query_with_session_events", _tracking_query)
    client = FakeClient(fail_first_post=True)
    perm = ChannelPermission(allowed_channels={CHANNEL}, per_user_daily_quota=500)
    fr, client, perm = _make_fr(permission=perm, client=client)

    await fr.on_channel_message(_workflow_event())

    assert client.posted == []
    assert ran_agent == []
    assert perm.usage_count(f"slack-wf:{APP_ID}") == 0


# ---------------------------------------------------------------------------
# 3. Agent error -> second post is the cc'd error text; no quota recorded.
# ---------------------------------------------------------------------------
async def test_agent_error_posts_ccd_error_and_records_no_quota(monkeypatch):
    def _raising_query(*args, **kwargs):
        async def _gen(*a, **k):
            raise RuntimeError("boom")
            yield  # pragma: no cover
        return _gen()
    monkeypatch.setattr(fr_mod, "query_with_session_events", _raising_query)
    fr, client, perm = _make_fr()

    await fr.on_channel_message(_workflow_event())

    assert len(client.posted) == 2
    ack, error_post = client.posted
    assert "hit an error" in error_post["text"]
    assert "<!subteam^S0EXAMPLE01>" in error_post["text"]
    assert client.updated == []
    assert perm.usage_count(f"slack-wf:{APP_ID}") == 0


# ---------------------------------------------------------------------------
# 4. Quota recorded exactly once, only on success, keyed on the unchanged principal.
# ---------------------------------------------------------------------------
async def test_quota_recorded_once_on_success_workflow_principal(monkeypatch):
    monkeypatch.setattr(
        fr_mod, "query_with_session_events",
        _fake_query({"answer": "x", "sources": [], "grounded": True}),
    )
    fr, client, perm = _make_fr()
    await fr.on_channel_message(_workflow_event())
    assert perm.usage_count(f"slack-wf:{APP_ID}") == 1


async def test_quota_recorded_once_on_success_direct_principal(monkeypatch):
    monkeypatch.setattr(
        fr_mod, "query_with_session_events",
        _fake_query({"answer": "x", "sources": [], "grounded": True}),
    )
    fr, client, perm = _make_fr()
    await fr.on_channel_message(_direct_event())
    assert perm.usage_count("U0EXAMPLE01") == 1


# ---------------------------------------------------------------------------
# 5. Mention-lookup cap: >15 distinct mentions -> at most 15 get_user_info calls.
# ---------------------------------------------------------------------------
async def test_mention_lookup_cap_respected(monkeypatch):
    monkeypatch.setattr(
        fr_mod, "query_with_session_events",
        _fake_query({"answer": "x", "sources": [], "grounded": True}),
    )
    many_mentions = " ".join(f"<@U{i:03d}NOPE>" for i in range(20))
    event = _direct_event(text=f"ping everyone: {many_mentions}")
    fr, client, _ = _make_fr()
    await fr.on_channel_message(event)

    # requester lookup (_resolve_sender) + exactly _MAX_MENTION_LOOKUPS (15) mention lookups.
    assert len(client.user_info_calls) == 16
    mention_calls = [c for c in client.user_info_calls if c != "U0EXAMPLE01"]
    assert len(mention_calls) == 15


# ---------------------------------------------------------------------------
# 6. Clean query reaches the agent; identity/thread_id unchanged; user_name/
#    user_email populated from get_user_info.
# ---------------------------------------------------------------------------
async def test_agent_receives_clean_normalized_query_and_unchanged_identity(monkeypatch):
    captured: list[dict] = []
    monkeypatch.setattr(
        fr_mod, "query_with_session_events",
        _fake_query({"answer": "x", "sources": [], "grounded": True}, captured=captured),
    )
    fr, client, _ = _make_fr()
    event = _workflow_event(
        ts="222.1",
        text="Requester: <@U0EXAMPLE01>\nFull Description: ping <@U1> about *deploy*",
    )
    await fr.on_channel_message(event)

    assert len(captured) == 1
    kwargs = captured[0]
    assert kwargs["query"] == "Requester: @duc\nFull Description: ping @alice about **deploy**"
    assert kwargs["user_id"] == f"slack-wf:{APP_ID}"
    assert kwargs["thread_id"] == f"slack-thread:{CHANNEL}:222.1"
    assert kwargs["user_name"] == "duc"
    assert kwargs["user_email"] == "duc@example.com"


# ---------------------------------------------------------------------------
# 6b. Same normalization when the query is SOURCED FROM BLOCKS (text empty --
#     `extract_query`'s `_flatten_blocks` fallback), confirming normalize_slack_text
#     runs on both text and flattened-block sources identically.
# ---------------------------------------------------------------------------
async def test_agent_receives_clean_query_when_sourced_from_blocks(monkeypatch):
    captured: list[dict] = []
    monkeypatch.setattr(
        fr_mod, "query_with_session_events",
        _fake_query({"answer": "x", "sources": [], "grounded": True}, captured=captured),
    )
    fr, client, _ = _make_fr()
    event = _workflow_event(ts="222.1", text="")
    event["blocks"] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Requester: <@U0EXAMPLE01>\nFull Description: ping <@U1> about *deploy*",
            },
        }
    ]
    await fr.on_channel_message(event)

    assert len(captured) == 1
    kwargs = captured[0]
    # Identical normalized text to the `text`-sourced case in test 6 above --
    # confirms normalization runs on the flattened-block path too.
    assert kwargs["query"] == "Requester: @duc\nFull Description: ping @alice about **deploy**"
    assert kwargs["user_id"] == f"slack-wf:{APP_ID}"
    assert kwargs["thread_id"] == f"slack-thread:{CHANNEL}:222.1"


# ---------------------------------------------------------------------------
# 7. Loop-safety: OpsRAG's own posts self-drop.
# ---------------------------------------------------------------------------
async def test_self_bot_id_post_is_dropped_before_any_ack(monkeypatch):
    monkeypatch.setattr(
        fr_mod, "query_with_session_events",
        _fake_query({"answer": "x", "sources": [], "grounded": True}),
    )
    fr, client, _ = _make_fr()
    event = {"type": "message", "bot_id": "BSELF", "channel": CHANNEL,
              "ts": "300.1", "text": "our own answer, echoed back"}
    await fr.on_channel_message(event)
    assert client.posted == []


async def test_non_allowlisted_app_id_is_ignored_by_classify_no_ack(monkeypatch):
    monkeypatch.setattr(
        fr_mod, "query_with_session_events",
        _fake_query({"answer": "x", "sources": [], "grounded": True}),
    )
    fr, client, _ = _make_fr()
    event = _workflow_event(ts="400.1")
    event["app_id"] = "AROOTLY"
    event["bot_id"] = "BROOTLY"
    await fr.on_channel_message(event)
    assert client.posted == []


# ---------------------------------------------------------------------------
# Extras: get_user_info failure degrades gracefully (no crash, no name leak).
# ---------------------------------------------------------------------------
async def test_get_user_info_failure_degrades_gracefully(monkeypatch):
    monkeypatch.setattr(
        fr_mod, "query_with_session_events",
        _fake_query({"answer": "x", "sources": [], "grounded": True}),
    )

    class RaisingUserInfoClient(FakeClient):
        async def get_user_info(self, user_id):
            self.user_info_calls.append(user_id)
            raise RuntimeError("users.info rate limited")

    client = RaisingUserInfoClient()
    fr, client, _ = _make_fr(client=client)
    await fr.on_channel_message(_workflow_event())

    assert len(client.posted) == 2  # still both posts -- no crash
