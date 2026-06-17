"""Unit test (H2, route level): the corpus-mutating admin ingestion routes
must be gated by the ``admin`` scope, like their sibling ``POST /index/repo``.

Three handlers were previously ungated -- any authenticated user (even a
chat-only member) could trigger ingestion / re-augmentation:

  * POST /index/source
  * POST /admin/reaugment/confluence
  * POST /admin/index/investigation-history

We mount the real router on a FastAPI app and override the resolved-user
dependency (``current_user_with_authz``, which ``require_scope`` depends on)
to return a non-admin user, mirroring ``tests/test_auth_scopes.py``. A
non-admin must receive 403 with the ``missing_scope`` reason BEFORE any
handler body runs (so app.state need not be populated).
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from opsrag.api.routes import router
from opsrag.auth.oidc import CurrentUser
from opsrag.auth.scopes import (
    Scope,
    current_user_with_authz,
    scopes_for_roles,
)


def _non_admin_user() -> CurrentUser:
    # member_chat has CHAT but NOT ADMIN -- the exact privilege-escalation
    # case the guard must block.
    return CurrentUser(
        sub="u1", email=None, name=None, picture_url=None, groups=(),
        is_anonymous=False, roles=frozenset({"member_chat"}),
        scopes=frozenset(scopes_for_roles(["member_chat"])),
    )


def _build_client(user: CurrentUser) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[current_user_with_authz] = lambda: user
    # raise_server_exceptions=False so that if a guard ever regressed and the
    # body ran against an unpopulated app.state, we'd still see a 5xx (and a
    # failing assertion) rather than the test erroring out opaquely.
    return TestClient(app, raise_server_exceptions=False)


def test_reaugment_confluence_denies_non_admin():
    client = _build_client(_non_admin_user())
    resp = client.post("/admin/reaugment/confluence")
    assert resp.status_code == 403
    detail = resp.json()["detail"]
    assert detail["error"] == "forbidden"
    assert detail["reason"] == "missing_scope"
    assert detail["scope"] == Scope.ADMIN


def test_index_investigation_history_denies_non_admin():
    client = _build_client(_non_admin_user())
    resp = client.post("/admin/index/investigation-history")
    assert resp.status_code == 403
    detail = resp.json()["detail"]
    assert detail["error"] == "forbidden"
    assert detail["reason"] == "missing_scope"
    assert detail["scope"] == Scope.ADMIN


def test_index_source_denies_non_admin():
    # The third newly-guarded route. Requires a JSON body, but the guard
    # runs before the body is touched, so a minimal valid body suffices.
    client = _build_client(_non_admin_user())
    resp = client.post(
        "/index/source", json={"source_type": "confluence", "scope": "OPS"},
    )
    assert resp.status_code == 403
    detail = resp.json()["detail"]
    assert detail["reason"] == "missing_scope"
    assert detail["scope"] == Scope.ADMIN


def test_list_feedback_denies_non_admin():
    # R6: GET /feedback was authenticated but ungated -- any member_chat user
    # could read every other user's query/answer snippets, notes, and user_id.
    # It must require the ``admin`` scope, like its sibling GET /corrections.
    client = _build_client(_non_admin_user())
    resp = client.get("/feedback")
    assert resp.status_code == 403
    detail = resp.json()["detail"]
    assert detail["error"] == "forbidden"
    assert detail["reason"] == "missing_scope"
    assert detail["scope"] == Scope.ADMIN


def test_admin_user_passes_guard_on_admin_route():
    # Positive control: an admin clears the guard. The handler body then runs
    # against an unpopulated app.state and 5xxs (or 404s for the unregistered
    # source) -- the point is only that it is NOT a 403 from the scope guard.
    admin = CurrentUser(
        sub="admin1", email=None, name=None, picture_url=None, groups=(),
        is_anonymous=False, roles=frozenset({"admin"}),
        scopes=frozenset(scopes_for_roles(["admin"])),
    )
    client = _build_client(admin)
    resp = client.post("/admin/index/investigation-history")
    assert resp.status_code != 403
