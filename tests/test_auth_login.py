"""Tests for first-party login (auth Tier 2): password, sessions, refresh,
rate-limit/lockout, SSO state/nonce + email_verified linking, role/scope
assignment.

All tests use the in-memory AuthUserStore and mock external IdPs -- no
live OAuth apps and no Postgres required.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from opsrag.auth.login import LoginRateLimiter
from opsrag.auth.login import router as login_router
from opsrag.auth.password import hash_password, needs_rehash, verify_password
from opsrag.auth.scopes import (
    ALL_SCOPES,
    DEFAULT_ROLE,
    Scope,
    default_roles,
    scopes_for_roles,
    set_default_roles,
)
from opsrag.auth.sessions import (
    InlineKeyMaterialError,
    SessionManager,
    generate_opaque_token,
    load_signing_key,
)
from opsrag.auth.sso import (
    ExternalIdentity,
    ProviderConfig,
    SSOError,
    build_oauth_registry,
    resolve_or_link_user,
    verify_state,
)
from opsrag.auth.user_store import InMemoryAuthUserStore, hash_token


def _run(coro):
    """Run an async coroutine to completion from a sync test."""
    return asyncio.new_event_loop().run_until_complete(coro)


# ===========================================================================
# password.py
# ===========================================================================
def test_password_hash_verify_roundtrip():
    h = hash_password("hunter2")
    assert h.startswith("$argon2id$")
    ok, new = verify_password("hunter2", h)
    assert ok is True
    assert new is None  # current policy, no upgrade needed


def test_password_wrong_password_rejected():
    h = hash_password("hunter2")
    ok, new = verify_password("wrong", h)
    assert ok is False
    assert new is None


def test_password_empty_hash_is_non_match():
    # SSO-only account (password_hash is None) can never log in by password.
    assert verify_password("anything", None) == (False, None)
    assert verify_password("anything", "") == (False, None)


def test_password_malformed_hash_does_not_raise():
    ok, new = verify_password("x", "not-a-real-hash")
    assert ok is False and new is None


def test_password_hash_is_salted_unique():
    assert hash_password("same") != hash_password("same")


def test_needs_rehash_flags_non_argon2():
    assert needs_rehash(hash_password("x")) is False
    assert needs_rehash("$2b$12$legacybcrypthashvalue") is True
    assert needs_rehash(None) is False


# ===========================================================================
# sessions.py — signing key, mint/verify, tamper, refresh, csrf
# ===========================================================================
def test_load_signing_key_rejects_inline_material():
    with pytest.raises(InlineKeyMaterialError):
        load_signing_key(inline="super-secret-pasted-into-yaml")


def test_load_signing_key_from_env(monkeypatch):
    monkeypatch.setenv("OPSRAG_TEST_SIGNKEY", "k" * 48)
    key = load_signing_key(key_env="OPSRAG_TEST_SIGNKEY")
    assert key == b"k" * 48


def test_load_signing_key_from_path(tmp_path):
    p = tmp_path / "session.key"
    p.write_bytes(b"file-based-signing-key-material")
    assert load_signing_key(key_path=str(p)) == b"file-based-signing-key-material"


def test_load_signing_key_missing_source_raises():
    with pytest.raises(ValueError):
        load_signing_key()
    with pytest.raises(ValueError):
        load_signing_key(key_env="OPSRAG_DEFINITELY_UNSET_KEY")


def _sm() -> SessionManager:
    return SessionManager(b"a-test-signing-key-32-bytes-long!", cookie_secure=False)


def test_session_mint_verify_roundtrip():
    sm = _sm()
    tok = sm.mint_session(user_id="u1", email="a@b.c", roles=("admin",))
    payload = sm.verify_session(tok)
    assert payload is not None
    assert payload.user_id == "u1"
    assert payload.email == "a@b.c"
    assert payload.roles == ("admin",)


def test_session_tamper_rejected():
    sm = _sm()
    tok = sm.mint_session(user_id="u1", email="a@b.c", roles=())
    # Flip a character in the signed token -> signature must fail.
    tampered = tok[:-1] + ("A" if tok[-1] != "A" else "B")
    assert sm.verify_session(tampered) is None


def test_session_wrong_key_rejected():
    sm1 = SessionManager(b"key-one-key-one-key-one-key-one!", cookie_secure=False)
    sm2 = SessionManager(b"key-two-key-two-key-two-key-two!", cookie_secure=False)
    tok = sm1.mint_session(user_id="u1", email=None, roles=())
    assert sm2.verify_session(tok) is None


def test_session_expired_rejected():
    sm = SessionManager(
        b"a-test-signing-key-32-bytes-long!",
        session_ttl_seconds=-1,  # already expired
        cookie_secure=False,
    )
    tok = sm.mint_session(user_id="u1", email=None, roles=())
    assert sm.verify_session(tok) is None


def test_session_none_token():
    assert _sm().verify_session(None) is None
    assert _sm().verify_session("") is None


def test_refresh_token_hashed_at_rest():
    sm = _sm()
    raw, token_hash, expires = sm.new_refresh_token()
    # The stored value is the hash, never the raw token.
    assert token_hash == hash_token(raw)
    assert token_hash != raw
    assert len(token_hash) == 64  # sha256 hex
    assert expires > datetime.now(UTC)


def test_csrf_double_submit():
    sm = _sm()
    tok = sm.new_csrf_token()
    assert sm.verify_csrf(tok, tok) is True
    assert sm.verify_csrf(tok, "other") is False
    assert sm.verify_csrf(None, tok) is False
    assert sm.verify_csrf(tok, None) is False


# ===========================================================================
# user_store.py — refresh rotation + hashed-at-rest + revocation
# ===========================================================================
@pytest.mark.asyncio
async def test_store_user_crud_and_email_index():
    store = InMemoryAuthUserStore()
    u = await store.create_user(email="A@B.com", password_hash="ph", email_verified=True)
    assert u.email == "a@b.com"  # normalized
    assert (await store.get_user_by_email("a@b.com")).id == u.id
    assert (await store.get_user_by_id(u.id)).email == "a@b.com"
    with pytest.raises(ValueError):
        await store.create_user(email="a@b.com", password_hash="x")


@pytest.mark.asyncio
async def test_refresh_session_rotation_and_revoke():
    store = InMemoryAuthUserStore()
    u = await store.create_user(email="x@y.z", password_hash="ph")
    raw = generate_opaque_token()
    h = hash_token(raw)
    exp = datetime.now(UTC) + timedelta(days=1)
    sess = await store.create_refresh_session(user_id=u.id, token_hash=h, expires_at=exp)
    assert sess.is_active
    got = await store.get_refresh_session(h)
    assert got is not None and got.token_hash == h
    # Stored hash, not the raw token.
    assert got.token_hash != raw
    # Revoke -> no longer active.
    await store.revoke_refresh_session(h)
    assert not (await store.get_refresh_session(h)).is_active


@pytest.mark.asyncio
async def test_expired_refresh_session_not_active():
    store = InMemoryAuthUserStore()
    u = await store.create_user(email="x@y.z", password_hash="ph")
    exp = datetime.now(UTC) - timedelta(seconds=1)
    sess = await store.create_refresh_session(
        user_id=u.id, token_hash="h", expires_at=exp
    )
    assert not sess.is_active


@pytest.mark.asyncio
async def test_revoke_all_for_user():
    store = InMemoryAuthUserStore()
    u = await store.create_user(email="x@y.z", password_hash="ph")
    for i in range(3):
        await store.create_refresh_session(
            user_id=u.id,
            token_hash=f"h{i}",
            expires_at=datetime.now(UTC) + timedelta(days=1),
        )
    await store.revoke_all_for_user(u.id)
    for i in range(3):
        assert not (await store.get_refresh_session(f"h{i}")).is_active


# ===========================================================================
# Rate-limit / lockout
# ===========================================================================
def test_rate_limiter_locks_after_max_attempts():
    rl = LoginRateLimiter(max_attempts=3, window_seconds=300, lockout_seconds=900)
    assert rl.record_failure("k") is False
    assert rl.record_failure("k") is False
    assert rl.record_failure("k") is True  # 3rd failure locks
    assert rl.is_locked("k")
    assert rl.retry_after("k") > 0


def test_rate_limiter_success_resets():
    rl = LoginRateLimiter(max_attempts=3)
    rl.record_failure("k")
    rl.record_failure("k")
    rl.record_success("k")
    assert not rl.is_locked("k")
    # Counter reset: two more failures should not lock yet.
    assert rl.record_failure("k") is False


# ===========================================================================
# SSO — state/nonce + email_verified-required linking + role mapping
# ===========================================================================
def test_verify_state_constant_time_match():
    assert verify_state("abc", "abc") is True
    assert verify_state("abc", "xyz") is False
    assert verify_state(None, "abc") is False
    assert verify_state("abc", None) is False


def test_build_oauth_registry_only_registers_enabled():
    providers = {
        "google": ProviderConfig(enabled=True, client_id="cid", client_secret="sec"),
        "github": ProviderConfig(enabled=False, client_id="x", client_secret="y"),
        "microsoft": ProviderConfig(enabled=True),  # missing creds -> skipped
    }
    oauth = build_oauth_registry(providers)
    assert oauth.create_client("google") is not None
    assert oauth.create_client("github") is None
    assert oauth.create_client("microsoft") is None


@pytest.mark.asyncio
async def test_sso_new_account_for_verified_email():
    store = InMemoryAuthUserStore()
    ident = ExternalIdentity(
        provider="google",
        subject="g-123",
        email="new@corp.com",
        email_verified=True,
        name="New User",
    )
    user = await resolve_or_link_user(ident, store=store)
    assert user.email == "new@corp.com"
    assert user.email_verified is True
    assert user.password_hash is None  # SSO-only
    # Link recorded.
    link = await store.get_identity("google", "g-123")
    assert link is not None and link.user_id == user.id
    # Absent an ``auth.default_roles`` override, a new SSO user is provisioned
    # with the built-in default role only.
    assert set(user.roles) == {DEFAULT_ROLE}


@pytest.mark.asyncio
async def test_sso_new_account_uses_configured_default_roles():
    """A newly onboarded SSO user is provisioned from ``auth.default_roles``
    (bound via set_default_roles), NOT the hardcoded DEFAULT_ROLE -- so adding
    e.g. member_mcp to the default applies to future onboardings automatically.
    Regression guard for the 'new engineers miss MCP' bug."""
    set_default_roles(["member_investigate", "member_mcp"])
    try:
        store = InMemoryAuthUserStore()
        ident = ExternalIdentity(
            provider="microsoft",
            subject="ms-77",
            email="onboarded@corp.com",
            email_verified=True,
            name="Onboarded Eng",
        )
        user = await resolve_or_link_user(ident, store=store)
        assert set(user.roles) == {"member_investigate", "member_mcp"}
        # The MCP scope is now effective without any manual admin step.
        assert Scope.MCP in scopes_for_roles(user.roles)
    finally:
        # Restore the process-global default so other tests aren't affected.
        set_default_roles(None)
        assert default_roles() == frozenset({DEFAULT_ROLE})


@pytest.mark.asyncio
async def test_sso_links_verified_email_to_existing_account():
    store = InMemoryAuthUserStore()
    existing = await store.create_user(
        email="dev@corp.com", password_hash=hash_password("pw"), email_verified=True
    )
    ident = ExternalIdentity(
        provider="google", subject="g-9", email="dev@corp.com", email_verified=True
    )
    user = await resolve_or_link_user(ident, store=store)
    # Linked to the SAME existing (password) account, not a duplicate.
    assert user.id == existing.id
    assert (await store.get_identity("google", "g-9")).user_id == existing.id


@pytest.mark.asyncio
async def test_sso_unverified_email_cannot_take_over_existing_account():
    store = InMemoryAuthUserStore()
    await store.create_user(
        email="victim@corp.com", password_hash=hash_password("pw"), email_verified=True
    )
    # Attacker controls a GitHub account whose UNVERIFIED email = victim's.
    ident = ExternalIdentity(
        provider="github", subject="attacker", email="victim@corp.com",
        email_verified=False,
    )
    with pytest.raises(SSOError):
        await resolve_or_link_user(ident, store=store)
    # No link was created.
    assert await store.get_identity("github", "attacker") is None


@pytest.mark.asyncio
async def test_sso_existing_link_returns_same_user():
    store = InMemoryAuthUserStore()
    ident = ExternalIdentity(
        provider="google", subject="g-1", email="a@b.c", email_verified=True
    )
    u1 = await resolve_or_link_user(ident, store=store)
    u2 = await resolve_or_link_user(ident, store=store)
    assert u1.id == u2.id


def test_role_to_scope_assignment_from_groups():
    # The single scope model: admin role -> all scopes; member_chat -> chat.
    assert scopes_for_roles({"admin"}) == set(ALL_SCOPES)
    assert scopes_for_roles({"member_chat"}) == {Scope.CHAT}
    assert scopes_for_roles({"member_investigate"}) == {Scope.CHAT, Scope.INVESTIGATE}


# ===========================================================================
# login.py router — end-to-end with TestClient (in-memory store, mock IdP)
# ===========================================================================
def _make_app() -> tuple[FastAPI, InMemoryAuthUserStore, SessionManager]:
    app = FastAPI()
    store = InMemoryAuthUserStore()
    sm = SessionManager(b"a-test-signing-key-32-bytes-long!", cookie_secure=False)
    app.state.auth_user_store = store
    app.state.session_manager = sm
    app.state.login_rate_limiter = LoginRateLimiter(max_attempts=3)
    app.state.role_mappings = {}
    app.include_router(login_router)
    return app, store, sm


def test_password_login_endpoint_success_sets_cookies():
    app, store, sm = _make_app()
    _run(
        store.create_user(
            email="user@corp.com", password_hash=hash_password("s3cret"),
            email_verified=True,
        )
    )
    client = TestClient(app)
    resp = client.post(
        "/auth/login", data={"email": "user@corp.com", "password": "s3cret"}
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    cookies = resp.cookies
    assert SessionManager.SESSION_COOKIE in cookies
    assert SessionManager.REFRESH_COOKIE in cookies
    assert SessionManager.CSRF_COOKIE in cookies
    # The session cookie verifies. (TestClient may surround the value with
    # quotes per RFC 6265 since the signed token contains '=' / '.'.)
    raw = cookies[SessionManager.SESSION_COOKIE].strip('"')
    payload = sm.verify_session(raw)
    assert payload is not None
    assert payload.email == "user@corp.com"


def test_password_login_bad_password_401_then_lockout_429():
    app, store, _ = _make_app()
    _run(
        store.create_user(email="u@c.com", password_hash=hash_password("right"))
    )
    client = TestClient(app)
    # 2 failures (max_attempts=3): both 401.
    for _ in range(2):
        r = client.post("/auth/login", data={"email": "u@c.com", "password": "x"})
        assert r.status_code == 401
        assert r.json()["detail"]["error"] == "invalid_credentials"
    # 3rd failure trips the lockout -> 429.
    r = client.post("/auth/login", data={"email": "u@c.com", "password": "x"})
    assert r.status_code == 429
    # Subsequent attempts (even with the RIGHT password) stay locked.
    r = client.post("/auth/login", data={"email": "u@c.com", "password": "right"})
    assert r.status_code == 429


def test_password_login_unknown_user_401():
    app, _, _ = _make_app()
    client = TestClient(app)
    r = client.post("/auth/login", data={"email": "ghost@c.com", "password": "x"})
    assert r.status_code == 401


def _set_cookie_lines(resp) -> list[str]:
    return [v for k, v in resp.headers.multi_items() if k.lower() == "set-cookie"]


def _cookie_path(set_cookie_lines: list[str], name: str) -> str | None:
    import re
    line = next((c for c in set_cookie_lines if c.startswith(name + "=")), None)
    if line is None:
        return None
    m = re.search(r"[Pp]ath=([^;]+)", line)
    return m.group(1) if m else None


def test_refresh_cookie_path_is_root_so_it_reaches_the_api_prefix():
    # Regression: the SPA calls /api/auth/refresh (nginx strips /api before the
    # app). A refresh cookie scoped to "/auth/refresh" fails RFC 6265 path-match
    # against the browser-visible "/api/auth/refresh" and is NEVER sent -> silent
    # refresh is impossible. It must be path="/" like session + csrf.
    app, store, _ = _make_app()
    _run(store.create_user(
        email="u@c.com", password_hash=hash_password("pw"), email_verified=True,
    ))
    client = TestClient(app)
    login = client.post("/auth/login", data={"email": "u@c.com", "password": "pw"})
    assert login.status_code == 200
    lines = _set_cookie_lines(login)
    assert _cookie_path(lines, SessionManager.REFRESH_COOKIE) == "/"
    # Sanity: session + csrf were already "/".
    assert _cookie_path(lines, SessionManager.SESSION_COOKIE) == "/"
    assert _cookie_path(lines, SessionManager.CSRF_COOKIE) == "/"


def test_logout_clears_refresh_cookie_at_root_path():
    # The delete_cookie path MUST match the set path, or the cleared cookie is a
    # no-op and the refresh cookie lingers after logout.
    app, store, _ = _make_app()
    _run(store.create_user(
        email="u@c.com", password_hash=hash_password("pw"), email_verified=True,
    ))
    client = TestClient(app)
    client.post("/auth/login", data={"email": "u@c.com", "password": "pw"})
    logout = client.post("/auth/logout")
    assert logout.status_code == 200
    assert _cookie_path(_set_cookie_lines(logout), SessionManager.REFRESH_COOKIE) == "/"


def test_refresh_rotates_token_and_revokes_old():
    app, store, sm = _make_app()
    _run(
        store.create_user(
            email="u@c.com", password_hash=hash_password("pw"), email_verified=True
        )
    )
    client = TestClient(app)
    login = client.post("/auth/login", data={"email": "u@c.com", "password": "pw"})
    old_refresh = login.cookies[SessionManager.REFRESH_COOKIE]

    # Refresh with the cookie present.
    client.cookies.set(SessionManager.REFRESH_COOKIE, old_refresh)
    r = client.post("/auth/refresh")
    assert r.status_code == 200
    new_refresh = r.cookies[SessionManager.REFRESH_COOKIE]
    assert new_refresh != old_refresh
    # Old refresh hash is now revoked.
    old_sess = _run(
        store.get_refresh_session(hash_token(old_refresh))
    )
    assert old_sess is not None and not old_sess.is_active


def test_refresh_with_revoked_token_401s_without_clearing_cookies():
    # Multi-tab safety: a "losing" tab presents an already-rotated (revoked)
    # refresh token. The server must 401 but MUST NOT emit clear-cookie
    # deletions -- with cookies at path="/", clearing would wipe the WINNING
    # tab's freshly minted session/refresh from the shared browser jar.
    app, store, _ = _make_app()
    _run(store.create_user(
        email="u@c.com", password_hash=hash_password("pw"), email_verified=True,
    ))
    client = TestClient(app)
    login = client.post("/auth/login", data={"email": "u@c.com", "password": "pw"})
    r1 = login.cookies[SessionManager.REFRESH_COOKIE]
    # Winner: rotate R1 -> R2 (R1 now revoked).
    client.cookies.set(SessionManager.REFRESH_COOKIE, r1)
    assert client.post("/auth/refresh").status_code == 200
    # Loser: replay the revoked R1.
    client.cookies.set(SessionManager.REFRESH_COOKIE, r1)
    lost = client.post("/auth/refresh")
    assert lost.status_code == 401
    assert lost.json()["detail"]["error"] == "invalid_refresh_token"
    # The 401 must NOT clear the session/refresh cookies.
    lines = _set_cookie_lines(lost)
    cleared = [
        c for c in lines
        if c.startswith(SessionManager.SESSION_COOKIE + "=")
        or c.startswith(SessionManager.REFRESH_COOKIE + "=")
    ]
    assert cleared == [], f"invalid refresh must not clear cookies, got: {cleared}"


def test_refresh_with_revoked_token_401():
    app, store, sm = _make_app()
    _run(
        store.create_user(email="u@c.com", password_hash=hash_password("pw"))
    )
    client = TestClient(app)
    login = client.post("/auth/login", data={"email": "u@c.com", "password": "pw"})
    refresh_tok = login.cookies[SessionManager.REFRESH_COOKIE]
    # Revoke it server-side.
    _run(
        store.revoke_refresh_session(hash_token(refresh_tok))
    )
    client.cookies.set(SessionManager.REFRESH_COOKIE, refresh_tok)
    r = client.post("/auth/refresh")
    assert r.status_code == 401


def test_logout_revokes_refresh():
    app, store, sm = _make_app()
    _run(
        store.create_user(email="u@c.com", password_hash=hash_password("pw"))
    )
    client = TestClient(app)
    login = client.post("/auth/login", data={"email": "u@c.com", "password": "pw"})
    refresh_tok = login.cookies[SessionManager.REFRESH_COOKIE]
    client.cookies.set(SessionManager.REFRESH_COOKIE, refresh_tok)
    r = client.post("/auth/logout")
    assert r.status_code == 200
    sess = _run(
        store.get_refresh_session(hash_token(refresh_tok))
    )
    assert sess is not None and not sess.is_active


def test_login_endpoints_503_when_unconfigured():
    app = FastAPI()
    app.include_router(login_router)
    client = TestClient(app)
    r = client.post("/auth/login", data={"email": "a@b.c", "password": "x"})
    assert r.status_code == 503


def test_sso_callback_invalid_state_rejected():
    app, store, sm = _make_app()
    # Minimal oauth stub so we reach the state check.
    class _StubOAuth:
        def create_client(self, provider):
            return object()
    app.state.sso_oauth = _StubOAuth()
    client = TestClient(app)
    # No state cookie set -> mismatch -> 400 invalid_state.
    r = client.get(
        "/auth/sso/google/callback?state=forged&code=abc", follow_redirects=False
    )
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "invalid_state"


def test_sso_callback_success_mints_session(monkeypatch):
    """End-to-end SSO callback with a mocked Authlib OIDC client: valid
    state + verified-email id_token -> new account + session cookies."""
    from opsrag.auth.login import _SSO_STATE_COOKIE

    app, store, sm = _make_app()

    class _MockOIDCClient:
        async def authorize_access_token(self, request):
            # Authlib stores validated id_token claims under "userinfo".
            return {
                "userinfo": {
                    "sub": "ms-sub-42",
                    "email": "sso.user@corp.com",
                    "email_verified": True,
                    "name": "SSO User",
                    "groups": ["sre-admins"],
                }
            }

    class _MockOAuth:
        def create_client(self, provider):
            return _MockOIDCClient()

    app.state.sso_oauth = _MockOAuth()
    # Map the IdP group to the admin role via the single scope model.
    app.state.role_mappings = {"sre-admins": ["admin"]}

    client = TestClient(app)
    state = "matching-state-value"
    client.cookies.set(_SSO_STATE_COOKIE, state)
    r = client.get(
        f"/auth/sso/microsoft/callback?state={state}&code=authcode",
        follow_redirects=False,
    )
    assert r.status_code == 302
    assert SessionManager.SESSION_COOKIE in r.cookies
    # A new federated account was created + linked.
    link = _run(store.get_identity("microsoft", "ms-sub-42"))
    assert link is not None
    user = _run(store.get_user_by_email("sso.user@corp.com"))
    assert user is not None and user.email_verified is True
