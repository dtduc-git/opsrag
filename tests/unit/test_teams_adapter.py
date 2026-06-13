"""Unit tests for the Teams ChannelAdapter + webhook router (design 4.4 / 8.4).

These tests run WITHOUT the ``botbuilder`` SDK installed (the unit CI does not
install the ``teams`` extra). That is exactly the deployment shape that matters:
the ``api`` role mounts the Teams router and MUST import it cleanly without the
extra. So everything here is exercised with duck-typed Activity dicts and the
FastAPI ``TestClient``; the only ``botbuilder``-dependent path (real JWT crypto
in ``validate_bot_jwt`` when the SDK *is* present) is guarded with
``pytest.importorskip``.

Coverage:
  * Activity -> InboundMessage normalization: personal (DM) vs group, ``<at>``
    mention strip, tenant -> workspace, conversation reference stashed in raw.
  * Action.Submit -> FeedbackEvent (+ malformed drop).
  * Adaptive Card render shape: answer TextBlock, sources FactSet, 👍/👎
    Action.Submit pair carrying the investigation id, footer, truncation.
  * Identity oid format (``teams-bot:<tenant>:<user>``, anonymous).
  * ``react`` is a clean no-op; ``fetch_thread`` returns ``[]``.
  * ``POST /messages`` returns 401 on missing/invalid auth (validator
    monkeypatched), 200 on a valid message activity (dispatcher stubbed).
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from opsrag.channels.adapters.teams import router as teams_router
from opsrag.channels.adapters.teams.adapter import (
    TeamsAdapter,
    _TeamsHandle,
    render_adaptive_card,
)
from opsrag.channels.adapters.teams.router import (
    TeamsAuthError,
    activity_to_feedback,
    activity_to_inbound,
    build_teams_router,
    validate_bot_jwt,
)
from opsrag.channels.config import TeamsChannelConfig
from opsrag.channels.types import AgentResult, InboundMessage, ReactionKind


# ---------------------------------------------------------------------------
# Activity -> InboundMessage normalization
# ---------------------------------------------------------------------------
def _message_activity(**over: object) -> dict:
    base = {
        "type": "message",
        "id": "act-1",
        "text": "why is prod down?",
        "from": {"id": "29:user-aaa", "name": "Alice"},
        "recipient": {"id": "28:bot-bbb", "name": "OpsRAG"},
        "conversation": {"id": "19:conv-xyz", "conversationType": "channel"},
        "channelId": "msteams",
        "serviceUrl": "https://smba.example/teams/",
        "channelData": {"tenant": {"id": "tenant-123"}},
    }
    base.update(over)  # type: ignore[arg-type]
    return base


def test_activity_to_inbound_group_is_not_dm() -> None:
    msg = activity_to_inbound(_message_activity())
    assert msg.channel_id == "19:conv-xyz"
    assert msg.user_id == "29:user-aaa"
    assert msg.text == "why is prod down?"
    assert msg.message_id == "act-1"
    assert msg.is_dm is False  # conversationType=channel => not a DM
    assert msg.workspace == "tenant-123"  # tenant namespaces the synthetic oid
    # The conversation reference must be stashed for proactive replies/denials.
    ref = msg.raw["conversation_reference"]
    assert ref["conversation"]["id"] == "19:conv-xyz"
    assert ref["service_url"] == "https://smba.example/teams/"


def test_activity_to_inbound_personal_is_dm() -> None:
    msg = activity_to_inbound(
        _message_activity(conversation={"id": "a:1", "conversationType": "personal"}),
    )
    assert msg.is_dm is True


def test_activity_to_inbound_strips_at_mention() -> None:
    msg = activity_to_inbound(
        _message_activity(text="<at>OpsRAG Bot</at> why is prod down?"),
    )
    assert msg.text == "why is prod down?"  # <at>...</at> span removed
    assert "<at>" not in msg.text


# ---------------------------------------------------------------------------
# Action.Submit -> FeedbackEvent
# ---------------------------------------------------------------------------
def test_activity_to_feedback_up() -> None:
    activity = _message_activity(
        text="",
        value={"feedback": "up", "id": "inv-42"},
    )
    fb = activity_to_feedback(activity)
    assert fb is not None
    assert fb.thumbs == "up"
    assert fb.investigation_id == "inv-42"
    assert fb.user_id == "29:user-aaa"
    assert fb.raw["conversation_reference"]["conversation"]["id"] == "19:conv-xyz"


def test_activity_to_feedback_down() -> None:
    fb = activity_to_feedback(
        _message_activity(value={"feedback": "down", "id": "inv-7"}),
    )
    assert fb is not None and fb.thumbs == "down" and fb.investigation_id == "inv-7"


def test_activity_to_feedback_malformed_returns_none() -> None:
    # No value dict at all -> plain message, not feedback.
    assert activity_to_feedback(_message_activity()) is None
    # Bad thumbs.
    assert activity_to_feedback(
        _message_activity(value={"feedback": "sideways", "id": "x"}),
    ) is None
    # Missing investigation id.
    assert activity_to_feedback(
        _message_activity(value={"feedback": "up"}),
    ) is None
    # value present but not a dict.
    assert activity_to_feedback(_message_activity(value="up:inv-1")) is None


# ---------------------------------------------------------------------------
# Adaptive Card render
# ---------------------------------------------------------------------------
def _texts(card: dict) -> list[str]:
    return [b.get("text", "") for b in card["body"] if b.get("type") == "TextBlock"]


def test_render_adaptive_card_shape_with_sources_and_feedback() -> None:
    result = AgentResult(
        answer="**Root cause**: a bad deploy at 14:02 UTC.",
        sources=[
            {"title": "runbook", "url": "https://example.com/rb"},
            {"source": "services/api/deploy.yaml"},
        ],
        diagram_present=True,
        session_id="teams-thread:19:conv:act-1",
        investigation_id="inv-9",
    )
    card = render_adaptive_card(result, web_ui_base_url="https://opsrag.example.com")

    assert card["type"] == "AdaptiveCard"
    # 1. Answer TextBlock present + carries the answer text.
    assert any("bad deploy" in t for t in _texts(card))
    # 2. Sources rendered as a FactSet with both sources.
    factsets = [b for b in card["body"] if b.get("type") == "FactSet"]
    assert factsets and len(factsets[0]["facts"]) == 2
    # The url source becomes an OpenUrl action.
    open_urls = [a for a in card["actions"] if a["type"] == "Action.OpenUrl"]
    assert any(a["url"] == "https://example.com/rb" for a in open_urls)
    # 3. Feedback Action.Submit pair carries {feedback, id:<investigation_id>}.
    submits = [a for a in card["actions"] if a["type"] == "Action.Submit"]
    assert len(submits) == 2
    by_dir = {a["data"]["feedback"]: a["data"]["id"] for a in submits}
    assert by_dir == {"up": "inv-9", "down": "inv-9"}
    # 4. Footer present.
    assert any(t.startswith("OpsRAG ·") for t in _texts(card))


def test_render_adaptive_card_no_feedback_without_investigation_id() -> None:
    result = AgentResult(
        answer="ok", sources=[], diagram_present=False,
        session_id=None, investigation_id=None,
    )
    card = render_adaptive_card(result)
    submits = [a for a in card.get("actions", []) if a["type"] == "Action.Submit"]
    assert submits == []  # no anchor -> no feedback row


def test_render_adaptive_card_truncates_long_answer_with_deep_link() -> None:
    long_answer = "x" * 5000
    card = render_adaptive_card(
        AgentResult(
            answer=long_answer, sources=[], diagram_present=False,
            session_id="sess-1", investigation_id=None,
        ),
        web_ui_base_url="https://opsrag.example.com",
        answer_chars_cap=200,
    )
    answer_block = _texts(card)[0]
    assert len(answer_block) <= 200
    assert "view full in OpsRAG UI" in answer_block
    assert "https://opsrag.example.com/#chat/sess-1" in answer_block


# ---------------------------------------------------------------------------
# Identity + best-effort primitives
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resolve_identity_oid_format() -> None:
    adapter = TeamsAdapter(TeamsChannelConfig())
    msg = InboundMessage(
        channel_id="19:conv", user_id="29:user-aaa", text="hi", message_id="m1",
        thread_id=None, is_dm=False, workspace="tenant-123",
    )
    user = await adapter.resolve_identity(msg)
    assert user.oid == "teams-bot:tenant-123:29:user-aaa"
    assert user.is_anonymous is True


@pytest.mark.asyncio
async def test_resolve_identity_unknown_tenant() -> None:
    adapter = TeamsAdapter(TeamsChannelConfig())
    msg = InboundMessage(
        channel_id="c", user_id="", text="hi", message_id="m1",
        thread_id=None, is_dm=True, workspace=None,
    )
    user = await adapter.resolve_identity(msg)
    assert user.oid == "teams-bot:unknown:unknown-user"


@pytest.mark.asyncio
async def test_react_is_noop() -> None:
    adapter = TeamsAdapter(TeamsChannelConfig())
    # No SDK, no transport -- must simply return without raising.
    assert await adapter.react("c", "m", ReactionKind.ACK) is None
    assert await adapter.react("c", "m", ReactionKind.DONE) is None
    assert await adapter.react("c", "m", ReactionKind.ERROR) is None


@pytest.mark.asyncio
async def test_fetch_thread_returns_empty() -> None:
    adapter = TeamsAdapter(TeamsChannelConfig())
    assert await adapter.fetch_thread("c", "t", cap=20) == []


@pytest.mark.asyncio
async def test_outbound_degrades_when_sdk_absent() -> None:
    """post_placeholder/edit/finalize must not raise without botbuilder."""
    adapter = TeamsAdapter(TeamsChannelConfig())
    # connect() with no SDK leaves _cloud_adapter None (outbound disabled).
    await adapter.connect(_DummySink())
    handle = await adapter.post_placeholder("19:conv", None, "thinking")
    assert isinstance(handle, _TeamsHandle)
    assert handle.activity_id is None  # outbound disabled -> no id captured
    # edit/finalize are clean no-ops in this state.
    await adapter.edit(handle, "still thinking")
    await adapter.finalize(
        handle,
        AgentResult(answer="done", sources=[], diagram_present=False,
                    session_id=None, investigation_id="inv-1"),
    )


class _DummySink:
    async def on_message(self, msg: InboundMessage) -> None:  # noqa: D401
        pass

    async def on_feedback(self, fb: object) -> None:
        pass


# ---------------------------------------------------------------------------
# JWT validation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_validate_bot_jwt_rejects_missing_header() -> None:
    with pytest.raises(TeamsAuthError):
        await validate_bot_jwt(None, {"type": "message"}, app_id="app")
    with pytest.raises(TeamsAuthError):
        await validate_bot_jwt("   ", {"type": "message"}, app_id="app")


@pytest.mark.asyncio
async def test_validate_bot_jwt_fails_closed_without_sdk() -> None:
    """A present-but-unverifiable token must be rejected when no SDK is here.

    (When the ``teams`` extra IS installed this path instead reaches the real
    crypto check -- which also rejects a bogus token -- so the assertion holds
    either way; we only need to confirm it never silently accepts.)
    """
    with pytest.raises(TeamsAuthError):
        await validate_bot_jwt("Bearer not-a-real-token", {"type": "message"}, app_id="app")


# ---------------------------------------------------------------------------
# Router: POST /messages auth + routing
# ---------------------------------------------------------------------------
def _build_app(monkeypatch, *, dispatcher_record: dict) -> TestClient:
    """Build a FastAPI app mounting a real Teams router with a stub dispatcher."""
    router = build_teams_router(
        agent_graph=object(),
        providers=object(),
        caches=object(),
        teams_cfg=TeamsChannelConfig(allowlist=["19:conv-xyz"]),
    )

    # Replace the dispatcher's sink methods with recorders so we assert routing
    # without running the agent. The router closed over this exact instance.
    dispatcher = router.dispatcher  # type: ignore[attr-defined]

    async def _on_message(msg):
        dispatcher_record["messages"].append(msg)

    async def _on_feedback(fb):
        dispatcher_record["feedbacks"].append(fb)

    monkeypatch.setattr(dispatcher, "on_message", _on_message)
    monkeypatch.setattr(dispatcher, "on_feedback", _on_feedback)
    # Avoid building a real CloudAdapter on first request.
    monkeypatch.setattr(
        router.adapter, "connect", _noop_connect,  # type: ignore[attr-defined]
    )

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


async def _noop_connect(sink) -> None:  # noqa: D401
    return None


def test_post_messages_401_on_missing_auth(monkeypatch) -> None:
    record = {"messages": [], "feedbacks": []}
    client = _build_app(monkeypatch, dispatcher_record=record)
    # No Authorization header at all -> 401, dispatcher never called.
    resp = client.post(
        "/api/channels/teams/messages",
        json={"type": "message", "text": "hi", "conversation": {"id": "19:conv-xyz"},
              "from": {"id": "u1"}},
    )
    assert resp.status_code == 401
    assert record["messages"] == []


def test_post_messages_401_on_invalid_auth(monkeypatch) -> None:
    record = {"messages": [], "feedbacks": []}

    async def _reject(auth_header, activity, *, app_id):
        raise TeamsAuthError("bad token")

    # Monkeypatch the module-level validator so the handler hits the 401 path
    # even with a header present (no real signed token available offline).
    monkeypatch.setattr(teams_router, "validate_bot_jwt", _reject)
    client = _build_app(monkeypatch, dispatcher_record=record)
    resp = client.post(
        "/api/channels/teams/messages",
        headers={"Authorization": "Bearer wrong"},
        json={"type": "message", "text": "hi", "conversation": {"id": "19:conv-xyz"},
              "from": {"id": "u1"}},
    )
    assert resp.status_code == 401
    assert record["messages"] == []


def test_post_messages_routes_message_to_dispatcher(monkeypatch) -> None:
    record = {"messages": [], "feedbacks": []}

    async def _accept(auth_header, activity, *, app_id):
        return None

    monkeypatch.setattr(teams_router, "validate_bot_jwt", _accept)
    client = _build_app(monkeypatch, dispatcher_record=record)
    resp = client.post(
        "/api/channels/teams/messages",
        headers={"Authorization": "Bearer ok"},
        json=_message_activity(text="<at>OpsRAG</at> why is prod down?"),
    )
    assert resp.status_code == 200
    assert len(record["messages"]) == 1
    inbound = record["messages"][0]
    assert inbound.text == "why is prod down?"  # mention stripped end-to-end
    assert inbound.user_id == "29:user-aaa"
    assert record["feedbacks"] == []


def test_post_messages_routes_card_submit_to_feedback(monkeypatch) -> None:
    record = {"messages": [], "feedbacks": []}

    async def _accept(auth_header, activity, *, app_id):
        return None

    monkeypatch.setattr(teams_router, "validate_bot_jwt", _accept)
    client = _build_app(monkeypatch, dispatcher_record=record)
    resp = client.post(
        "/api/channels/teams/messages",
        headers={"Authorization": "Bearer ok"},
        json=_message_activity(text="", value={"feedback": "up", "id": "inv-3"}),
    )
    assert resp.status_code == 200
    assert len(record["feedbacks"]) == 1
    assert record["feedbacks"][0].investigation_id == "inv-3"
    assert record["messages"] == []  # routed as feedback, not a query


def test_post_messages_ignores_non_message_activity(monkeypatch) -> None:
    record = {"messages": [], "feedbacks": []}

    async def _accept(auth_header, activity, *, app_id):
        return None

    monkeypatch.setattr(teams_router, "validate_bot_jwt", _accept)
    client = _build_app(monkeypatch, dispatcher_record=record)
    resp = client.post(
        "/api/channels/teams/messages",
        headers={"Authorization": "Bearer ok"},
        json={"type": "conversationUpdate", "conversation": {"id": "19:conv-xyz"}},
    )
    assert resp.status_code == 200
    assert record["messages"] == [] and record["feedbacks"] == []


# ---------------------------------------------------------------------------
# SDK-guarded: only runs when the teams extra is actually installed
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_card_activity_builds_attachment_when_sdk_present() -> None:
    pytest.importorskip("botbuilder")
    adapter = TeamsAdapter(TeamsChannelConfig())
    card = render_adaptive_card(
        AgentResult(answer="hello", sources=[], diagram_present=False,
                    session_id=None, investigation_id="inv-1"),
    )
    activity = adapter._card_activity(card, fallback_text="hello")  # noqa: SLF001
    assert activity.attachments
    att = activity.attachments[0]
    assert att.content_type == "application/vnd.microsoft.card.adaptive"
    assert att.content["type"] == "AdaptiveCard"


# ---------------------------------------------------------------------------
# Single-tenant support (Microsoft deprecated multi-tenant bot creation, 2025)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_connect_reads_single_tenant_app_settings(monkeypatch) -> None:
    monkeypatch.setenv("OPSRAG_TEAMS_APP_ID", "appid-123")
    monkeypatch.setenv("OPSRAG_TEAMS_APP_PASSWORD", "secret-val")
    monkeypatch.setenv("OPSRAG_TEAMS_APP_TYPE", "SingleTenant")
    monkeypatch.setenv("OPSRAG_TEAMS_APP_TENANT_ID", "tenant-abc")
    adapter = TeamsAdapter(TeamsChannelConfig(enabled=True))
    await adapter.connect(object())  # bare sink is enough; connect only stores it
    assert adapter._app_type == "SingleTenant"  # noqa: SLF001
    assert adapter._app_tenant_id == "tenant-abc"  # noqa: SLF001


@pytest.mark.asyncio
async def test_connect_defaults_to_multitenant(monkeypatch) -> None:
    monkeypatch.delenv("OPSRAG_TEAMS_APP_TYPE", raising=False)
    monkeypatch.delenv("OPSRAG_TEAMS_APP_TENANT_ID", raising=False)
    monkeypatch.setenv("OPSRAG_TEAMS_APP_ID", "appid-123")
    monkeypatch.setenv("OPSRAG_TEAMS_APP_PASSWORD", "secret-val")
    adapter = TeamsAdapter(TeamsChannelConfig(enabled=True))
    await adapter.connect(object())
    assert adapter._app_type == "MultiTenant"  # noqa: SLF001 -- back-compat default
    assert adapter._app_tenant_id == ""  # noqa: SLF001
