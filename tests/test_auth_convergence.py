"""Tests for the converged get_current_user_dep + current_user_with_authz.

Verifies there is ONE OIDC-shaped identity dependency (no dual shape) and
that RBAC scopes are resolved off app.state in oidc mode while open mode
yields all-scopes anonymous.
"""
from __future__ import annotations

import types

import pytest

import opsrag.auth as auth_pkg
import opsrag.auth.middleware as mw
from opsrag.auth.oidc import CurrentUser
from opsrag.auth.scopes import (
    ALL_SCOPES,
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
async def test_open_mode_returns_all_scopes_anonymous():
    # auth_config.mode == "open" -> open mode regardless of verifier.
    state = _FakeState(auth_config=_FakeState(mode="open", role_mappings={}))
    req = _FakeRequest(app_state=state)
    user = await current_user_with_authz(req)
    assert user.is_anonymous
    assert user.scopes == frozenset(ALL_SCOPES)


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
async def test_middleware_dep_open_mode_no_401():
    # middleware.get_current_user_dep must NOT 401 in open mode.
    state = _FakeState(auth_config=_FakeState(mode="open"))
    req = _FakeRequest(app_state=state)
    user = await mw.get_current_user_dep(req)
    assert user.is_anonymous
