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
from dataclasses import dataclass

import pytest

from opsrag.api import routes


# --- Fakes ----------------------------------------------------------------


@dataclass
class _FakeUser:
    """Minimal stand-in for ``CurrentUser`` -- only the attributes the
    ownership guards read (``oid`` + ``is_anonymous``)."""

    oid: str | None
    is_anonymous: bool


def _user(oid: str) -> _FakeUser:
    return _FakeUser(oid=oid, is_anonymous=False)


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

    async def list_sessions(self, user_id: str) -> list[dict]:
        return [
            {"thread_id": tid, "user_id": owner, "checkpoint_count": 1}
            for tid, owner in self._owners.items()
            if owner == user_id
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
