"""Tests for the converged get_current_user_dep + current_user_with_authz.

Verifies there is ONE OIDC-shaped identity dependency (no dual shape) and
that RBAC scopes are resolved off app.state from the user's roles/claims.
Authentication is ALWAYS enforced -- there is no anonymous / "open" mode
that grants scopes; a token-less request on an allowlisted route yields a
SCOPELESS anonymous user.
"""
from __future__ import annotations

import types

import pytest

import opsrag.auth as auth_pkg
import opsrag.auth.middleware as mw
from opsrag.auth.oidc import CurrentUser
from opsrag.auth.scopes import (
    Scope,
    current_user_with_authz,
)


def test_single_converged_dep_is_oidc_shape():
    # The package-level dep is the converged callable (with user_store
    # upsert side-effect) and exposes a single name.
    assert auth_pkg.get_current_user_dep is auth_pkg.get_current_user
    # CurrentUser exported from the package is the OIDC shape (has .sub
    # and the .oid back-compat alias and RBAC fields).
    u = CurrentUser.anonymous()
    assert hasattr(u, "sub")
    assert u.oid == u.sub  # back-compat alias
    assert hasattr(u, "roles") and hasattr(u, "scopes")
    # Anonymous carries NO scopes now (auth is always enforced).
    assert u.scopes == frozenset()


class _FakeState:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeApp:
    def __init__(self, state):
        self.state = state


class _FakeRequest:
    def __init__(self, *, headers=None, app_state=None):
        self.headers = headers or {}
        self.app = _FakeApp(app_state or _FakeState())
        self.state = types.SimpleNamespace()


class _StubVerifier:
    def __init__(self, groups):
        self._groups = tuple(groups)

    def verify_to_user(self, token):
        return CurrentUser(
            sub="user-1", email="u@example.com", name="U",
            picture_url=None, groups=self._groups, is_anonymous=False,
        )


@pytest.mark.asyncio
async def test_tokenless_oidc_request_is_scopeless_anonymous():
    # oidc mode, no verifier wired (allowlisted route reached pre-verifier),
    # no bearer -> scopeless anonymous. There is NO open/all-scopes path.
    state = _FakeState(auth_config=_FakeState(mode="oidc", role_mappings={}))
    req = _FakeRequest(app_state=state)
    user = await current_user_with_authz(req)
    assert user.is_anonymous
    # Auth is always enforced: anonymous carries NO scopes.
    assert user.scopes == frozenset()
    assert not user.has_scope(Scope.CHAT)
    assert not user.has_scope(Scope.ADMIN)


@pytest.mark.asyncio
async def test_oidc_mode_resolves_scopes_from_groups():
    state = _FakeState(
        auth_config=_FakeState(
            mode="oidc", role_mappings={"chat-grp": ["member_chat"]},
        ),
        oidc_verifier=_StubVerifier(groups=["chat-grp"]),
    )
    req = _FakeRequest(
        headers={"authorization": "Bearer anything"}, app_state=state,
    )
    user = await current_user_with_authz(req)
    assert not user.is_anonymous
    assert user.sub == "user-1"
    assert user.roles == frozenset({"member_chat"})
    assert user.scopes == frozenset({Scope.CHAT})
    assert not user.has_scope(Scope.INVESTIGATE)


@pytest.mark.asyncio
async def test_oidc_mode_no_token_is_anonymous():
    # No bearer on a (token-less) request in oidc mode -> anonymous.
    # (The global OIDCAuthMiddleware 401s protected routes earlier.)
    state = _FakeState(
        auth_config=_FakeState(mode="oidc", role_mappings={}),
        oidc_verifier=_StubVerifier(groups=[]),
    )
    req = _FakeRequest(app_state=state)
    user = await current_user_with_authz(req)
    assert user.is_anonymous


@pytest.mark.asyncio
async def test_middleware_dep_tokenless_is_scopeless_anonymous():
    # middleware.get_current_user_dep never 401s itself (the global
    # middleware enforces 401 on protected routes). With no verifier wired
    # and no token, it returns a SCOPELESS anonymous user (no open mode).
    state = _FakeState(auth_config=_FakeState(mode="oidc"))
    req = _FakeRequest(app_state=state)
    user = await mw.get_current_user_dep(req)
    assert user.is_anonymous
    assert user.scopes == frozenset()
