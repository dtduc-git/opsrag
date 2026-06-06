"""Unit + integration tests for the RBAC scope model (opsrag.auth.scopes).

Covers:
  * resolve_roles: group->role mapping, default role, admin signal.
  * scopes_for_roles / ROLE_SCOPES bundle correctness.
  * CurrentUser RBAC fields + has_scope + open-mode (anonymous) all-scopes.
  * require_scope dependency: allow/deny via a FastAPI TestClient and via
    direct dependency invocation with a fake CurrentUser.
"""
from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from opsrag.auth.oidc import CurrentUser
from opsrag.auth.scopes import (
    ALL_SCOPES,
    DEFAULT_ROLE,
    ROLE_SCOPES,
    Scope,
    attach_authz,
    current_user_with_authz,
    has_scope,
    require_scope,
    resolve_roles,
    scopes_for_roles,
)


# ---------------------------------------------------------------------------
# resolve_roles
# ---------------------------------------------------------------------------
def test_resolve_roles_maps_groups_via_mappings():
    mappings = {"sre-admins": ["admin"], "oncall": ["member_investigate"]}
    assert resolve_roles(["oncall"], mappings) == {"member_investigate"}
    assert resolve_roles(["sre-admins"], mappings) == {"admin"}
    # Multiple groups union their roles.
    assert resolve_roles(["oncall", "sre-admins"], mappings) == {
        "member_investigate",
        "admin",
    }


def test_resolve_roles_default_when_no_match():
    # Authenticated user whose groups match nothing -> default role.
    assert resolve_roles(["nobody"], {"x": ["admin"]}) == {DEFAULT_ROLE}
    # No mappings configured at all.
    assert resolve_roles(["any"], {}) == {DEFAULT_ROLE}
    assert resolve_roles([], None) == {DEFAULT_ROLE}


def test_resolve_roles_admin_signal_adds_admin():
    assert "admin" in resolve_roles([], {}, is_admin=True)
    # Admin signal combines with a group-mapped role.
    roles = resolve_roles(["oncall"], {"oncall": ["member_chat"]}, is_admin=True)
    assert roles == {"member_chat", "admin"}


# ---------------------------------------------------------------------------
# scopes_for_roles / ROLE_SCOPES
# ---------------------------------------------------------------------------
def test_admin_gets_all_scopes():
    assert scopes_for_roles(["admin"]) == set(ALL_SCOPES)


def test_member_chat_lacks_investigate():
    scopes = scopes_for_roles(["member_chat"])
    assert Scope.CHAT in scopes
    assert Scope.INVESTIGATE not in scopes
    assert Scope.MCP not in scopes
    assert Scope.ADMIN not in scopes


def test_member_investigate_implies_chat():
    scopes = scopes_for_roles(["member_investigate"])
    assert {Scope.CHAT, Scope.INVESTIGATE} <= scopes
    assert Scope.ADMIN not in scopes


def test_member_mcp_only_mcp():
    assert scopes_for_roles(["member_mcp"]) == {Scope.MCP}


def test_unknown_role_contributes_no_scopes():
    assert scopes_for_roles(["does-not-exist"]) == set()


def test_role_scopes_map_shape():
    assert set(ROLE_SCOPES) == {
        "admin",
        "member_chat",
        "member_investigate",
        "member_mcp",
    }


# ---------------------------------------------------------------------------
# CurrentUser RBAC fields + open mode
# ---------------------------------------------------------------------------
def test_open_mode_anonymous_has_all_scopes():
    u = CurrentUser.anonymous()
    assert u.is_anonymous
    assert u.scopes == frozenset(ALL_SCOPES)
    for s in (Scope.CHAT, Scope.INVESTIGATE, Scope.MCP, Scope.ADMIN):
        assert u.has_scope(s)


def test_attach_authz_open_mode_grants_all():
    u = CurrentUser(
        sub="u1", email=None, name=None, picture_url=None,
        groups=("member_chat-grp",), is_anonymous=False,
    )
    enriched = attach_authz(u, role_mappings={}, open_mode=True)
    assert enriched.scopes == frozenset(ALL_SCOPES)


def test_attach_authz_oidc_resolves_from_groups():
    u = CurrentUser(
        sub="u1", email=None, name=None, picture_url=None,
        groups=("chat-only",), is_anonymous=False,
    )
    enriched = attach_authz(
        u, role_mappings={"chat-only": ["member_chat"]}, open_mode=False,
    )
    assert enriched.roles == frozenset({"member_chat"})
    assert enriched.scopes == frozenset({Scope.CHAT})
    assert enriched.has_scope(Scope.CHAT)
    assert not enriched.has_scope(Scope.INVESTIGATE)


def test_has_scope_none_user_false():
    assert has_scope(None, Scope.CHAT) is False


# ---------------------------------------------------------------------------
# require_scope dependency -- direct invocation with a fake user
# ---------------------------------------------------------------------------
def _make_user(scopes: set[str], *, anonymous: bool = False) -> CurrentUser:
    return CurrentUser(
        sub=None if anonymous else "u1",
        email=None, name=None, picture_url=None, groups=(),
        is_anonymous=anonymous, roles=frozenset(), scopes=frozenset(scopes),
    )


# ---------------------------------------------------------------------------
# require_scope dependency -- via FastAPI TestClient with overrides
# ---------------------------------------------------------------------------
def _build_app(user: CurrentUser) -> FastAPI:
    app = FastAPI()

    @app.get("/needs-chat")
    async def needs_chat(u: CurrentUser = Depends(require_scope(Scope.CHAT))):
        return {"ok": True, "sub": u.sub}

    @app.get("/needs-admin")
    async def needs_admin(u: CurrentUser = Depends(require_scope(Scope.ADMIN))):
        return {"ok": True}

    # Override the resolved-user dependency that require_scope depends on.
    app.dependency_overrides[current_user_with_authz] = lambda: user
    return app


def test_require_scope_allows_when_present():
    user = _make_user({Scope.CHAT})
    client = TestClient(_build_app(user))
    resp = client.get("/needs-chat")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_require_scope_denies_when_missing():
    user = _make_user({Scope.CHAT})  # no admin scope
    client = TestClient(_build_app(user))
    resp = client.get("/needs-admin")
    assert resp.status_code == 403
    body = resp.json()["detail"]
    assert body["error"] == "forbidden"
    assert body["reason"] == "missing_scope"
    assert body["scope"] == Scope.ADMIN


def test_require_scope_open_mode_anonymous_allows_all():
    # Open-mode anonymous carries ALL scopes -> every guard passes.
    user = CurrentUser.anonymous()
    client = TestClient(_build_app(user))
    assert client.get("/needs-chat").status_code == 200
    assert client.get("/needs-admin").status_code == 200


def test_member_chat_user_denied_admin_route():
    user = _make_user(scopes_for_roles(["member_chat"]))
    client = TestClient(_build_app(user))
    assert client.get("/needs-chat").status_code == 200
    assert client.get("/needs-admin").status_code == 403
