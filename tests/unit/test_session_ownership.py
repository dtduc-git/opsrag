"""Security unit test: session-ownership IDOR fix.

Three session endpoints previously leaked across users:
  * GET    /sessions/{user_id}            (list_sessions)    -- no auth dep
  * DELETE /sessions/{thread_id}          (delete_session)   -- no owner check
  * GET    /sessions/{thread_id}/messages (session_messages) -- no auth dep

The recorded owner is the checkpoint-metadata ``user_id``. After the fix the
write path binds that owner to the AUTHENTICATED id, and the three read/delete
endpoints enforce per-session ownership:

  * an authenticated user A cannot delete or read a thread owned by B -> 404
    (not 403, to avoid an existence oracle);
  * list_sessions returns ONLY the caller's own sessions (the path user_id is
    overridden with the verified oid);
  * OPEN / anonymous mode is NOT enforced (preserves zero-config dev behavior);
  * legacy ``anonymous``/empty-owned threads are grandfathered (stay accessible).

We drive the real route handlers directly with fake stores and a fake
CurrentUser -- no FastAPI app needed.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from opsrag.api import routes
from opsrag.auth.scopes import Scope

# --- Fakes ----------------------------------------------------------------


@dataclass
class _FakeUser:
    """Minimal stand-in for ``CurrentUser`` -- only the attributes the
    ownership guards read (``oid`` + ``is_anonymous`` + ``scopes``)."""

    oid: str | None
    is_anonymous: bool
    scopes: frozenset = field(default_factory=frozenset)


def _user(oid: str) -> _FakeUser:
    # A plain member: chat scope, NOT admin.
    return _FakeUser(oid=oid, is_anonymous=False, scopes=frozenset({Scope.CHAT}))


def _admin(oid: str) -> _FakeUser:
    return _FakeUser(
        oid=oid, is_anonymous=False, scopes=frozenset({Scope.CHAT, Scope.ADMIN})
    )


_ANON = _FakeUser(oid=None, is_anonymous=True)


class _FakeStore:
    """In-memory session store: thread_id -> owner user_id.

    ``list_sessions`` mirrors the real stores' user-scoping (a thread is in
    the result iff its owner matches the requested user_id). ``delete_session``
    records the deleted thread_id so the test can assert it was (not) hit.
    """

    def __init__(self, owners: dict[str, str]) -> None:
        self._owners = dict(owners)
        self.deleted: list[str] = []
        self.read: list[str] = []

    async def get_session_owner(self, thread_id: str) -> str | None:
        return self._owners.get(thread_id)

    async def list_sessions(
        self, user_id: str, *, include_all: bool = False
    ) -> list[dict]:
        return [
            {"thread_id": tid, "user_id": owner, "checkpoint_count": 1}
            for tid, owner in self._owners.items()
            if include_all or owner == user_id
        ]

    async def delete_session(self, thread_id: str) -> bool:
        self.deleted.append(thread_id)
        return thread_id in self._owners

    async def get_messages(self, thread_id: str) -> list[dict]:
        self.read.append(thread_id)
        return [{"role": "user", "content": f"hello from {thread_id}"}]


class _FakeRequest:
    def __init__(self, store: _FakeStore) -> None:
        self.app = type("_App", (), {})()
        self.app.state = type("_State", (), {})()
        self.app.state.session_store = store


def _store() -> _FakeStore:
    # Owners: A owns t_a, B owns t_b, a legacy thread owned by "anonymous",
    # and one with an empty-string owner.
    return _FakeStore(
        {
            "t_a": "A",
            "t_b": "B",
            "t_legacy": "anonymous",
            "t_empty": "",
        }
    )


# --- delete_session -------------------------------------------------------


def test_delete_other_users_thread_is_404():
    store = _store()
    req = _FakeRequest(store)
    with pytest.raises(routes.HTTPException) as ei:
        asyncio.run(routes.delete_session("t_b", req, current_user=_user("A")))
    assert ei.value.status_code == 404
    # Critical: the store delete must NOT have run for a denied thread.
    assert store.deleted == []


def test_delete_own_thread_succeeds():
    store = _store()
    req = _FakeRequest(store)
    out = asyncio.run(routes.delete_session("t_a", req, current_user=_user("A")))
    assert out["deleted"] is True
    assert store.deleted == ["t_a"]


def test_delete_legacy_anonymous_owned_thread_is_grandfathered():
    store = _store()
    req = _FakeRequest(store)
    out = asyncio.run(routes.delete_session("t_legacy", req, current_user=_user("A")))
    assert out["deleted"] is True
    assert store.deleted == ["t_legacy"]


def test_delete_empty_owner_thread_is_grandfathered():
    store = _store()
    req = _FakeRequest(store)
    out = asyncio.run(routes.delete_session("t_empty", req, current_user=_user("A")))
    assert out["deleted"] is True
    assert store.deleted == ["t_empty"]


def test_delete_in_open_mode_is_not_enforced():
    store = _store()
    req = _FakeRequest(store)
    # Anonymous (open mode) may delete any thread -- dev behavior preserved.
    out = asyncio.run(routes.delete_session("t_b", req, current_user=_ANON))
    assert out["deleted"] is True
    assert store.deleted == ["t_b"]


# --- session_messages -----------------------------------------------------


def test_read_other_users_thread_is_404():
    store = _store()
    req = _FakeRequest(store)
    with pytest.raises(routes.HTTPException) as ei:
        asyncio.run(routes.session_messages("t_b", req, current_user=_user("A")))
    assert ei.value.status_code == 404
    # Critical: history must NOT be read for a denied thread (no oracle, no leak).
    assert store.read == []


def test_read_own_thread_succeeds():
    store = _store()
    req = _FakeRequest(store)
    out = asyncio.run(routes.session_messages("t_a", req, current_user=_user("A")))
    assert out["thread_id"] == "t_a"
    assert store.read == ["t_a"]
    assert out["messages"]


def test_read_legacy_anonymous_owned_thread_is_grandfathered():
    store = _store()
    req = _FakeRequest(store)
    out = asyncio.run(routes.session_messages("t_legacy", req, current_user=_user("A")))
    assert out["thread_id"] == "t_legacy"
    assert store.read == ["t_legacy"]


def test_read_in_open_mode_is_not_enforced():
    store = _store()
    req = _FakeRequest(store)
    out = asyncio.run(routes.session_messages("t_b", req, current_user=_ANON))
    assert store.read == ["t_b"]
    assert out["messages"]


# --- list_sessions --------------------------------------------------------


def test_list_sessions_scopes_to_caller_ignoring_path_user_id():
    store = _store()
    req = _FakeRequest(store)
    # Caller A asks for B's sessions via the path -- must get ONLY A's.
    resp = asyncio.run(routes.list_sessions("B", req, current_user=_user("A")))
    tids = {s.thread_id for s in resp.sessions}
    assert tids == {"t_a"}


def test_list_sessions_open_mode_uses_path_user_id():
    store = _store()
    req = _FakeRequest(store)
    # Open mode keeps the path-supplied id (zero-config dev behavior).
    resp = asyncio.run(routes.list_sessions("B", req, current_user=_ANON))
    tids = {s.thread_id for s in resp.sessions}
    assert tids == {"t_b"}


# --- admin sees + manages everything --------------------------------------


def test_list_sessions_admin_sees_all():
    store = _store()
    req = _FakeRequest(store)
    # Admin asking for any path id gets EVERY thread (team oversight).
    resp = asyncio.run(routes.list_sessions("A", req, current_user=_admin("ADM")))
    tids = {s.thread_id for s in resp.sessions}
    assert tids == {"t_a", "t_b", "t_legacy", "t_empty"}


def test_list_sessions_regular_user_with_no_threads_sees_empty():
    store = _store()
    req = _FakeRequest(store)
    # A signed-in user who owns nothing (the legacy threads belong to
    # default/anonymous, not them) gets an EMPTY list -- not everyone's.
    resp = asyncio.run(routes.list_sessions("anything", req, current_user=_user("Z")))
    assert resp.sessions == []


def test_admin_can_read_any_thread():
    store = _store()
    req = _FakeRequest(store)
    out = asyncio.run(routes.session_messages("t_b", req, current_user=_admin("ADM")))
    assert out["thread_id"] == "t_b"
    assert store.read == ["t_b"]  # admin bypass -> history IS read


def test_admin_can_delete_any_thread():
    store = _store()
    req = _FakeRequest(store)
    out = asyncio.run(routes.delete_session("t_b", req, current_user=_admin("ADM")))
    assert out["deleted"] is True
    assert store.deleted == ["t_b"]


# --- store impls expose get_session_owner + read the right owner ----------


def test_inmemory_store_get_session_owner_round_trips_via_checkpoint():
    from langgraph.checkpoint.base import empty_checkpoint

    from opsrag.sessions.memory import InMemorySessionStore

    store = InMemorySessionStore()
    saver = store.get_checkpointer()
    # Mirror what graph.py persists -- a checkpoint whose configurable carries
    # the owner. ``empty_checkpoint()`` builds a schema-valid checkpoint so we
    # don't hand-roll LangGraph's internal shape.
    cfg = {
        "configurable": {
            "thread_id": "t1",
            "checkpoint_ns": "",
            "user_id": "owner-1",
        }
    }
    saver.put(cfg, empty_checkpoint(), {}, {})
    assert asyncio.run(store.get_session_owner("t1")) == "owner-1"
    # Unknown thread -> None (so the guard treats it as not-a-real-owner).
    assert asyncio.run(store.get_session_owner("nope")) is None


def test_inmemory_list_sessions_filters_by_owner_and_admin_include_all():
    from langgraph.checkpoint.base import empty_checkpoint

    from opsrag.sessions.memory import InMemorySessionStore

    store = InMemorySessionStore()
    saver = store.get_checkpointer()
    for tid, owner in (("tA", "userA"), ("tB", "userB")):
        cfg = {
            "configurable": {"thread_id": tid, "checkpoint_ns": "", "user_id": owner}
        }
        saver.put(cfg, empty_checkpoint(), {}, {})
    # Regular caller sees ONLY their own thread (not the other user's).
    own = asyncio.run(store.list_sessions("userA"))
    assert {s["thread_id"] for s in own} == {"tA"}
    # include_all (admin path) returns every thread.
    every = asyncio.run(store.list_sessions("userA", include_all=True))
    assert {s["thread_id"] for s in every} == {"tA", "tB"}


def test_both_stores_satisfy_session_store_protocol():
    from opsrag.interfaces.session import SessionStore
    from opsrag.sessions.memory import InMemorySessionStore

    mem = InMemorySessionStore()
    assert isinstance(mem, SessionStore)
    assert hasattr(mem, "get_session_owner")
    # Postgres store is constructed lazily (needs a DSN); just assert the
    # method exists on the class so the Protocol contract is satisfied.
    from opsrag.sessions.postgres import PostgresSessionStore

    assert hasattr(PostgresSessionStore, "get_session_owner")


# --- write-path owner binding (_owner_id_for) -----------------------------


def test_owner_id_binds_to_authenticated_oid_not_client_value():
    # Authenticated: the spoofable req.user_id is ignored; owner == oid.
    assert routes._owner_id_for(_user("A"), "pretend-to-be-B") == "A"


def test_owner_id_falls_back_to_req_user_id_in_open_mode():
    assert routes._owner_id_for(_ANON, "team-shared") == "team-shared"


def test_owner_id_defaults_to_anonymous_when_open_and_no_req_user_id():
    assert routes._owner_id_for(_ANON, None) == "anonymous"


# --- /query write+read path (continuing a thread requires owning it) -------
#
# POST /query loads req.thread_id's prior checkpoints (history) and appends
# to them, so an unguarded thread_id is an IDOR (read + inject into another
# user's conversation). The guard runs before any provider/graph work, so a
# cross-user thread_id 404s. NOTE: the fake request deliberately has NO
# `providers` -- if the guard did NOT short-circuit, the handler would raise
# AttributeError reaching for providers, not HTTPException(404). So a 404 is
# proof the guard fired first.


def _query_request(req: _FakeRequest):
    # agent_graph is read at the top of the handler before the guard; stub it.
    req.app.state.agent_graph = object()
    return req


def test_query_continue_other_users_thread_is_404():
    from opsrag.api.models import QueryRequest

    store = _store()
    req = _query_request(_FakeRequest(store))
    qr = QueryRequest(query="summarize our conversation", thread_id="t_b")
    with pytest.raises(routes.HTTPException) as ei:
        asyncio.run(routes.query(qr, req, current_user=_user("A")))
    assert ei.value.status_code == 404


def test_query_legacy_anonymous_thread_is_grandfathered_past_guard():
    # The guard must NOT 404 a legacy anonymous-owned thread. It proceeds past
    # the guard and then fails on the absent providers -- which proves the
    # guard let it through (an AttributeError, not a 404).
    from opsrag.api.models import QueryRequest

    store = _store()
    req = _query_request(_FakeRequest(store))
    qr = QueryRequest(query="hi", thread_id="t_legacy")
    with pytest.raises(Exception) as ei:
        asyncio.run(routes.query(qr, req, current_user=_user("A")))
    assert not isinstance(ei.value, routes.HTTPException)


# --- /usage/{session_id} (usage is keyed by the thread_id namespace) -------


def test_usage_other_users_session_is_404():
    store = _store()
    req = _FakeRequest(store)
    with pytest.raises(routes.HTTPException) as ei:
        asyncio.run(routes.session_usage("t_b", req, current_user=_user("A")))
    assert ei.value.status_code == 404


def test_usage_own_session_is_allowed():
    store = _store()
    req = _FakeRequest(store)
    out = asyncio.run(routes.session_usage("t_a", req, current_user=_user("A")))
    assert out["session_id"] == "t_a"  # no 404; usage may be None (none recorded)


def test_usage_legacy_anonymous_session_is_grandfathered():
    store = _store()
    req = _FakeRequest(store)
    out = asyncio.run(routes.session_usage("t_legacy", req, current_user=_user("A")))
    assert out["session_id"] == "t_legacy"


def test_usage_open_mode_not_enforced():
    store = _store()
    req = _FakeRequest(store)
    out = asyncio.run(routes.session_usage("t_b", req, current_user=_ANON))
    assert out["session_id"] == "t_b"


# --- /investigation/{id}/feedback (bind submitter to authenticated id) -----
#
# Investigations are a shared team resource (scope-gated, no per-row owner),
# so the fix is: gate on the investigate scope (enforced by the route
# dependency, not unit-tested here) AND persist the AUTHENTICATED identity as
# the submitter -- never the spoofable client-supplied req.user_id.


class _FakeFeedbackStore:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def record(self, **kwargs) -> int:
        self.calls.append(kwargs)
        return 1


def _feedback_request(fb: _FakeFeedbackStore) -> _FakeRequest:
    req = _FakeRequest(_store())
    req.app.state.investigation_cache = None  # exercise the Postgres-only path
    req.app.state.feedback_store = fb
    return req


def test_investigation_feedback_binds_submitter_to_authenticated_oid():
    from opsrag.api.models import InvestigationFeedbackRequest

    fb = _FakeFeedbackStore()
    req = _feedback_request(fb)
    body = InvestigationFeedbackRequest(thumbs="up", user_id="pretend-to-be-B")
    asyncio.run(routes.investigation_feedback("inv1", body, req, current_user=_user("A")))
    # The persisted user_id is the verified oid, NOT the spoofed client value.
    assert fb.calls and fb.calls[0]["user_id"] == "A"


def test_investigation_feedback_open_mode_falls_back_to_client_user_id():
    from opsrag.api.models import InvestigationFeedbackRequest

    fb = _FakeFeedbackStore()
    req = _feedback_request(fb)
    body = InvestigationFeedbackRequest(thumbs="down", user_id="team-shared")
    asyncio.run(routes.investigation_feedback("inv2", body, req, current_user=_ANON))
    assert fb.calls and fb.calls[0]["user_id"] == "team-shared"
