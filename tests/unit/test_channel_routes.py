"""Route tests for the public-channel-conversation read API
(``opsrag/api/routes_channels.py``), driven through a FastAPI TestClient.

The security contract (the most important part):

  * GET /channels/conversations lists ONLY shared-channel (``-thread:``)
    conversations, each carrying a ``platform`` label and NO synthetic
    ``user_id``.
  * GET /channels/conversations/{tid}/messages returns **404** for a private
    DM (``-dm:``) thread_id AND for a web thread_id -- the store is never even
    asked (no existence oracle / no content leak) -- and **200** for a
    shared-channel (``-thread:``) thread_id.
  * The router is gated on the ``chat`` scope: a caller without it gets 403.

See ``docs/superpowers/specs/2026-06-15-public-channel-conversations-design.md``.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from opsrag.api.routes_channels import channels_router
from opsrag.auth.oidc import CurrentUser
from opsrag.auth.scopes import Scope, current_user_with_authz, scopes_for_roles


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeStore:
    """Records every thread_id whose history is read so the security tests can
    prove the 404 fires BEFORE any store read.

    ``list_sessions_by_prefixes`` returns canned shared-channel summaries (each
    with the synthetic bot ``user_id`` the route is expected to strip).
    """

    def __init__(self) -> None:
        self.read: list[str] = []
        self.prefixes_seen: list[tuple[str, ...]] = []

    async def list_sessions_by_prefixes(self, prefixes):
        self.prefixes_seen.append(tuple(prefixes))
        return [
            {
                "thread_id": "slack-thread:C1:1700000000.0001",
                "user_id": "slack-bot-oid",
                "title": "why is checkout down",
                "preview": "Checkout is degraded because...",
                "created_at": "2026-06-15T10:00:00",
                "updated_at": "2026-06-15T10:05:00",
                "turn_count": 2,
                "checkpoint_count": 4,
            },
            {
                "thread_id": "discord-thread:G3:42",
                "user_id": "discord-bot-oid",
                "title": "deploy rollback steps",
                "preview": "To roll back...",
                "created_at": "2026-06-15T09:00:00",
                "updated_at": "2026-06-15T09:30:00",
                "turn_count": 1,
                "checkpoint_count": 2,
            },
        ]

    async def get_messages(self, thread_id: str):
        self.read.append(thread_id)
        return [
            {"role": "user", "content": f"hello on {thread_id}"},
            {"role": "assistant", "content": "hi", "sources": []},
        ]


def _user(scopes: set[str]) -> CurrentUser:
    return CurrentUser(
        sub="u1",
        email=None,
        name=None,
        picture_url=None,
        groups=(),
        is_anonymous=False,
        roles=frozenset(),
        scopes=frozenset(scopes),
    )


def _build_app(store: _FakeStore, user: CurrentUser) -> FastAPI:
    app = FastAPI()
    app.include_router(channels_router)
    app.state.session_store = store
    # Override the user the scope guard resolves -- same trick the auth-scopes
    # route tests use.
    app.dependency_overrides[current_user_with_authz] = lambda: user
    return app


def _client(store: _FakeStore | None = None, *, scopes=None) -> tuple[TestClient, _FakeStore]:
    store = store or _FakeStore()
    user = _user(scopes if scopes is not None else {Scope.CHAT})
    return TestClient(_build_app(store, user)), store


# ---------------------------------------------------------------------------
# GET /channels/conversations
# ---------------------------------------------------------------------------
def test_list_conversations_returns_shared_channel_items():
    client, store = _client()
    resp = client.get("/channels/conversations")
    assert resp.status_code == 200
    convos = resp.json()["conversations"]
    assert {c["thread_id"] for c in convos} == {
        "slack-thread:C1:1700000000.0001",
        "discord-thread:G3:42",
    }
    # The route asked the store with exactly the public prefixes.
    assert store.prefixes_seen and all(
        p.endswith("-thread:") for p in store.prefixes_seen[0]
    )


def test_list_conversations_attaches_platform():
    client, _ = _client()
    convos = client.get("/channels/conversations").json()["conversations"]
    by_tid = {c["thread_id"]: c for c in convos}
    assert by_tid["slack-thread:C1:1700000000.0001"]["platform"] == "slack"
    assert by_tid["discord-thread:G3:42"]["platform"] == "discord"


def test_list_conversations_strips_user_id():
    # The opaque synthetic bot oid must never leak to a reader.
    client, _ = _client()
    convos = client.get("/channels/conversations").json()["conversations"]
    assert convos  # sanity: we actually checked something
    for c in convos:
        assert "user_id" not in c


def test_list_conversations_newest_first():
    client, _ = _client()
    convos = client.get("/channels/conversations").json()["conversations"]
    updated = [c["updated_at"] for c in convos]
    assert updated == sorted(updated, reverse=True)


# ---------------------------------------------------------------------------
# GET /channels/conversations/{thread_id}/messages -- SECURITY
# ---------------------------------------------------------------------------
def test_messages_for_shared_channel_thread_is_200():
    client, store = _client()
    resp = client.get(
        "/channels/conversations/slack-thread:C1:1700000000.0001/messages"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["thread_id"] == "slack-thread:C1:1700000000.0001"
    assert body["platform"] == "slack"
    assert body["messages"]
    # The store WAS read for the (allowed) public thread.
    assert store.read == ["slack-thread:C1:1700000000.0001"]


def test_messages_for_dm_thread_is_404_and_never_reads_store():
    # Private 1:1 DM -- must 404 and the store must never be touched.
    client, store = _client()
    resp = client.get("/channels/conversations/slack-dm:U999/messages")
    assert resp.status_code == 404
    assert store.read == []


def test_messages_for_web_thread_is_404_and_never_reads_store():
    # A plain web thread id -- no leak: 404 before any read.
    client, store = _client()
    resp = client.get("/channels/conversations/user_abcd/messages")
    assert resp.status_code == 404
    assert store.read == []


# ---------------------------------------------------------------------------
# scope gating
# ---------------------------------------------------------------------------
def test_routes_require_chat_scope():
    # A caller lacking the chat scope (e.g. an MCP-only member) is 403'd on
    # both endpoints.
    store = _FakeStore()
    client, _ = _client(store, scopes=scopes_for_roles(["member_mcp"]))
    assert Scope.CHAT not in scopes_for_roles(["member_mcp"])  # premise
    assert client.get("/channels/conversations").status_code == 403
    assert (
        client.get(
            "/channels/conversations/slack-thread:C1:1700000000.0001/messages"
        ).status_code
        == 403
    )
    # 403 short-circuits before any store work.
    assert store.read == []


def test_routes_allow_chat_scope():
    client, _ = _client(scopes={Scope.CHAT})
    assert client.get("/channels/conversations").status_code == 200
