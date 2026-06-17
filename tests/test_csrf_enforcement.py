"""H1: CSRF double-submit enforcement in login/cookie mode (middleware level).

The session cookie is SameSite=Lax, which does NOT protect top-level cross-site
POSTs, so OIDCAuthMiddleware (login mode) requires unsafe methods to echo the
non-HttpOnly ``opsrag_csrf`` cookie in the ``X-CSRF-Token`` header. Safe methods
and the /auth/* allowlist are exempt; bearer/oidc mode never reaches this branch.
"""
from __future__ import annotations

import pytest

pytest.importorskip("itsdangerous")  # login extra; skip in the minimal unit job

from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from opsrag.api.oidc_enforcement import OIDCAuthMiddleware
from opsrag.auth.sessions import SessionManager


class _Cfg:
    mode = "login"


def _build():
    sm = SessionManager(b"a-test-signing-key-32-bytes-long!", cookie_secure=False)

    async def ok(_request):
        return JSONResponse({"ok": True})

    app = Starlette(
        routes=[
            Route("/query", ok, methods=["GET", "POST"]),
            Route("/auth/login", ok, methods=["POST"]),
        ]
    )
    app.add_middleware(OIDCAuthMiddleware)
    app.state.auth_config = _Cfg()
    app.state.session_manager = sm

    client = TestClient(app)
    client.cookies.set(sm.SESSION_COOKIE, sm.mint_session(user_id="u1", email="u1@x", roles=("admin",)))
    csrf = sm.new_csrf_token()
    client.cookies.set(sm.CSRF_COOKIE, csrf)
    return client, sm, csrf


def test_safe_get_needs_no_csrf():
    client, _sm, _csrf = _build()
    assert client.get("/query").status_code == 200


def test_unsafe_post_without_csrf_header_rejected():
    client, _sm, _csrf = _build()
    r = client.post("/query", json={})
    assert r.status_code == 401
    assert r.json()["reason"] == "csrf_failed"


def test_unsafe_post_with_matching_csrf_passes():
    client, sm, csrf = _build()
    r = client.post("/query", json={}, headers={sm.CSRF_HEADER: csrf})
    assert r.status_code == 200


def test_unsafe_post_with_mismatched_csrf_rejected():
    client, sm, _csrf = _build()
    r = client.post("/query", json={}, headers={sm.CSRF_HEADER: "not-the-token"})
    assert r.status_code == 401
    assert r.json()["reason"] == "csrf_failed"


def test_auth_route_bypasses_csrf():
    # Login itself must NOT be CSRF-gated (it issues the cookie); /auth/* is
    # allowlisted before the login branch.
    client, _sm, _csrf = _build()
    assert client.post("/auth/login", json={}).status_code == 200
