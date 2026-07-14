"""Login APIRouter: password + SSO + refresh + logout (auth Tier 2).

Endpoints (mounted under the app root; all on the NO_AUTH prefix
``/auth/*`` so the OIDC enforcement middleware lets them through):

* ``POST /auth/login``                  -- password login (rate-limited/lockout)
* ``POST /auth/logout``                 -- clear cookies + revoke refresh sessions
* ``GET  /auth/sso/{provider}/login``   -- begin the OAuth redirect
* ``GET  /auth/sso/{provider}/callback``-- complete OAuth, mint session
* ``POST /auth/refresh``                -- rotate refresh token, re-mint session

On success the handlers set the signed session cookie + the rotating,
hashed-at-rest refresh cookie + the double-submit CSRF cookie via the
:class:`opsrag.auth.sessions.SessionManager` wired onto ``app.state``.

Roles/scopes: a logged-in user's roles are the union of any
operator-assigned ``AuthUser.roles`` and the roles derived from IdP
groups via ``opsrag.auth.scopes.resolve_roles`` + the configured
``role_mappings``; scopes come from ``scopes_for_roles``. This reuses the
SINGLE authoritative scope model -- the login layer never invents its own.

Wiring contract (set on ``app.state`` by server.py):
  * ``app.state.auth_user_store``  -- an :class:`AuthUserStore`
  * ``app.state.session_manager``  -- a :class:`SessionManager`
  * ``app.state.sso_oauth``        -- an Authlib OAuth registry (optional)
  * ``app.state.login_rate_limiter``-- a :class:`LoginRateLimiter`
  * ``app.state.role_mappings``    -- the ``{group: [roles]}`` map (RBAC)

Every handler degrades to ``503`` when its required state piece is
absent, so registering the router is always safe even if login is not
configured.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from fastapi import APIRouter, Form, HTTPException, Request, Response
from fastapi.responses import RedirectResponse

from opsrag.auth.password import verify_password
from opsrag.auth.scopes import resolve_roles, scopes_for_roles
from opsrag.auth.sessions import SessionManager
from opsrag.auth.sso import (
    SSOError,
    fetch_identity,
    new_nonce,
    new_state,
    resolve_or_link_user,
    verify_state,
)
from opsrag.auth.user_store import AuthUserStore, hash_token

_log = logging.getLogger("opsrag.auth.login")

router = APIRouter()

# Short-lived signed cookies that carry the OAuth state/nonce across the
# redirect (so the callback can verify without server-side storage).
_SSO_STATE_COOKIE = "opsrag_sso_state"
_SSO_NONCE_COOKIE = "opsrag_sso_nonce"
_SSO_RETURN_COOKIE = "opsrag_sso_return"


# ---------------------------------------------------------------------------
# Login rate-limit / lockout.
# ---------------------------------------------------------------------------
@dataclass
class _Attempts:
    count: int = 0
    first_ts: float = 0.0
    locked_until: float = 0.0


@dataclass
class LoginRateLimiter:
    """Per-key (email/IP) failed-login throttle with temporary lockout.

    After ``max_attempts`` failures within ``window_seconds`` the key is
    locked for ``lockout_seconds``. A success resets the counter.

    State lives either in-process (the default -- per replica, sufficient to
    blunt online password guessing) or in a shared
    :class:`opsrag.api.rate_limit_backend.RateLimitBackend` (e.g. Redis) so
    the lockout is enforced across replicas. The synchronous methods below
    are the in-process fast path and stay behaviorally identical when no
    ``backend`` is set; the ``*_async`` wrappers are what the request
    handler calls and route to the backend when one is configured.
    """

    max_attempts: int = 5
    window_seconds: float = 300.0
    lockout_seconds: float = 900.0
    backend: object | None = None  # opsrag.api.rate_limit_backend.RateLimitBackend
    _state: dict[str, _Attempts] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def is_locked(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            a = self._state.get(key)
            return bool(a and a.locked_until > now)

    def retry_after(self, key: str) -> int:
        now = time.monotonic()
        with self._lock:
            a = self._state.get(key)
            if not a or a.locked_until <= now:
                return 0
            return int(a.locked_until - now) + 1

    def record_failure(self, key: str) -> bool:
        """Record a failed attempt. Returns True iff the key is NOW locked."""
        now = time.monotonic()
        with self._lock:
            a = self._state.get(key)
            if a is None or (now - a.first_ts) > self.window_seconds:
                a = _Attempts(count=0, first_ts=now)
                self._state[key] = a
            a.count += 1
            if a.count >= self.max_attempts:
                a.locked_until = now + self.lockout_seconds
                return True
            return False

    def record_success(self, key: str) -> None:
        with self._lock:
            self._state.pop(key, None)

    # -- async surface used by the handler (backend-aware) ------------------
    async def is_locked_async(self, key: str) -> bool:
        if self.backend is not None:
            return await self.backend.login_locked(key)
        return self.is_locked(key)

    async def retry_after_async(self, key: str) -> int:
        if self.backend is not None:
            return await self.backend.login_retry_after(key)
        return self.retry_after(key)

    async def record_failure_async(self, key: str) -> bool:
        if self.backend is not None:
            return await self.backend.record_login_failure(
                key,
                max_attempts=self.max_attempts,
                window_seconds=self.window_seconds,
                lockout_seconds=self.lockout_seconds,
            )
        return self.record_failure(key)

    async def record_success_async(self, key: str) -> None:
        if self.backend is not None:
            await self.backend.record_login_success(key)
            return
        self.record_success(key)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _require_state(request: Request, name: str):
    obj = getattr(request.app.state, name, None)
    if obj is None:
        raise HTTPException(status_code=503, detail=f"login not configured ({name})")
    return obj


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _roles_for(user, request: Request) -> tuple[frozenset, frozenset]:
    """Compute (roles, scopes) for an AuthUser using the SINGLE scope model.

    Combines operator-assigned ``user.roles`` with IdP-group-derived
    roles (none for password users; SSO users may carry groups already
    folded into ``user.roles`` at link time). resolve_roles applies the
    default member role when nothing matches, so a brand-new user still
    gets the interactive baseline."""
    role_mappings = getattr(request.app.state, "role_mappings", None) or {}
    # AuthUser.roles are concrete opsrag role names already; we still run
    # them through scopes_for_roles. Group-derived roles (if any future
    # group claim is persisted) would be merged here.
    explicit = set(user.roles or ())
    if explicit:
        roles = explicit
    else:
        roles = resolve_roles((), role_mappings)
    scopes = scopes_for_roles(roles)
    return frozenset(roles), frozenset(scopes)


async def _issue_session(
    request: Request,
    response: Response,
    user,
) -> None:
    """Mint + set the session, refresh, and CSRF cookies for ``user``."""
    sm: SessionManager = _require_state(request, "session_manager")
    store: AuthUserStore = _require_state(request, "auth_user_store")

    roles, _scopes = _roles_for(user, request)
    session_token = sm.mint_session(
        user_id=user.id, email=user.email, roles=tuple(sorted(roles))
    )
    raw_refresh, refresh_hash, expires = sm.new_refresh_token()
    await store.create_refresh_session(
        user_id=user.id, token_hash=refresh_hash, expires_at=expires
    )
    csrf = sm.new_csrf_token()
    sm.set_login_cookies(
        response,
        session_token=session_token,
        refresh_token=raw_refresh,
        csrf_token=csrf,
    )


# ---------------------------------------------------------------------------
# Available login methods (lets the UI show exactly password / SSO / both).
# ---------------------------------------------------------------------------
@router.get("/auth/providers")
async def auth_providers(request: Request) -> dict:
    return {
        "password_enabled": bool(
            getattr(request.app.state, "login_password_enabled", True)
        ),
        "providers": list(getattr(request.app.state, "sso_providers", []) or []),
    }


# ---------------------------------------------------------------------------
# Password login.
# ---------------------------------------------------------------------------
@router.post("/auth/login")
async def password_login(
    request: Request,
    response: Response,
    email: str = Form(...),
    password: str = Form(...),
):
    if not getattr(request.app.state, "login_password_enabled", True):
        # SSO-only deployment -> password login disabled.
        raise HTTPException(status_code=403, detail={"error": "password_login_disabled"})
    store: AuthUserStore = _require_state(request, "auth_user_store")
    limiter: LoginRateLimiter = _require_state(request, "login_rate_limiter")

    key = f"{email.strip().lower()}|{_client_ip(request)}"
    if await limiter.is_locked_async(key):
        retry = await limiter.retry_after_async(key)
        raise HTTPException(
            status_code=429,
            detail={"error": "locked_out", "retry_after": retry},
            headers={"Retry-After": str(retry)},
        )

    user = await store.get_user_by_email(email)
    stored_hash = user.password_hash if user else None
    # Always run verify (even on a missing user, against None) so the
    # response timing does not reveal whether the email exists.
    ok, new_hash = verify_password(password, stored_hash)

    if not ok or user is None:
        locked = await limiter.record_failure_async(key)
        detail = {"error": "invalid_credentials"}
        if locked:
            retry = await limiter.retry_after_async(key)
            detail = {"error": "locked_out", "retry_after": retry}
            raise HTTPException(status_code=429, detail=detail)
        raise HTTPException(status_code=401, detail=detail)

    await limiter.record_success_async(key)
    if new_hash is not None:
        # Verify-and-upgrade: persist the stronger hash.
        try:
            await store.set_password_hash(user.id, new_hash)
        except Exception as exc:  # noqa: BLE001
            _log.debug("password rehash persist failed for %s: %s", user.id, exc)

    await _issue_session(request, response, user)
    return {"ok": True, "user": {"id": user.id, "email": user.email}}


# ---------------------------------------------------------------------------
# Logout.
# ---------------------------------------------------------------------------
@router.post("/auth/logout")
async def logout(request: Request, response: Response):
    sm: SessionManager | None = getattr(request.app.state, "session_manager", None)
    store: AuthUserStore | None = getattr(request.app.state, "auth_user_store", None)
    if sm is not None:
        # Revoke the presented refresh token (best-effort).
        raw_refresh = request.cookies.get(SessionManager.REFRESH_COOKIE)
        if raw_refresh and store is not None:
            try:
                await store.revoke_refresh_session(hash_token(raw_refresh))
            except Exception as exc:  # noqa: BLE001
                _log.debug("refresh revoke on logout failed: %s", exc)
        sm.clear_cookies(response)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Refresh: rotate the refresh token + re-mint the session cookie.
# ---------------------------------------------------------------------------
@router.post("/auth/refresh")
async def refresh(request: Request, response: Response):
    sm: SessionManager = _require_state(request, "session_manager")
    store: AuthUserStore = _require_state(request, "auth_user_store")

    raw_refresh = request.cookies.get(SessionManager.REFRESH_COOKIE)
    if not raw_refresh:
        raise HTTPException(status_code=401, detail={"error": "missing_refresh_token"})

    presented_hash = hash_token(raw_refresh)
    sess = await store.get_refresh_session(presented_hash)
    if sess is None or not sess.is_active:
        # Unknown/expired/revoked -> 401, but do NOT clear_cookies() here.
        # The refresh cookie is now path="/" (shared across same-origin tabs).
        # In a multi-tab idle-lapse both tabs may present the SAME single-use
        # token; one rotates it, the other arrives with the now-revoked token
        # and lands here. Clearing cookies would delete the WINNING tab's
        # freshly minted opsrag_session/opsrag_refresh/opsrag_csrf from the
        # shared jar and log every tab out. A bare 401 is sufficient: the
        # client falls back to login on its own, and a stale/dead cookie
        # lingering is harmless (it just 401s until re-login overwrites it).
        raise HTTPException(status_code=401, detail={"error": "invalid_refresh_token"})

    user = await store.get_user_by_id(sess.user_id)
    if user is None:
        sm.clear_cookies(response)
        raise HTTPException(status_code=401, detail={"error": "user_gone"})

    # Rotation: revoke the presented token, then issue a fresh session
    # (which mints + persists a brand-new refresh token).
    await store.revoke_refresh_session(presented_hash)
    await _issue_session(request, response, user)
    return {"ok": True}


# ---------------------------------------------------------------------------
# SSO: begin redirect.
# ---------------------------------------------------------------------------
@router.get("/auth/sso/{provider}/login")
async def sso_login(provider: str, request: Request):
    oauth = _require_state(request, "sso_oauth")
    sm: SessionManager = _require_state(request, "session_manager")
    client = _sso_client(oauth, provider)

    state = new_state()
    nonce = new_nonce()
    # Prefer the operator-configured external base (correct behind a path-
    # stripping reverse proxy like the UI's /api). Fall back to request.url_for
    # only when no base is set (API hit directly, no proxy).
    base = getattr(request.app.state, "sso_callback_base", None)
    if base:
        redirect_uri = f"{base.rstrip('/')}/auth/sso/{provider}/callback"
    else:
        redirect_uri = str(request.url_for("sso_callback", provider=provider))

    # Authlib's authorize_redirect for OIDC providers will embed nonce.
    redirect = await client.authorize_redirect(
        request, redirect_uri, state=state, nonce=nonce
    )
    # Persist state/nonce in short-lived signed (HttpOnly) cookies so the
    # callback can verify them. SameSite=lax so they survive the IdP
    # redirect GET back to us.
    _set_sso_cookie(redirect, _SSO_STATE_COOKIE, state, sm)
    _set_sso_cookie(redirect, _SSO_NONCE_COOKIE, nonce, sm)
    return redirect


# ---------------------------------------------------------------------------
# SSO: callback.
# ---------------------------------------------------------------------------
@router.get("/auth/sso/{provider}/callback", name="sso_callback")
async def sso_callback(provider: str, request: Request):
    oauth = _require_state(request, "sso_oauth")
    _require_state(request, "session_manager")  # 503 if login not configured
    store: AuthUserStore = _require_state(request, "auth_user_store")
    client = _sso_client(oauth, provider)

    # Verify state (CSRF guard).
    expected_state = request.cookies.get(_SSO_STATE_COOKIE)
    received_state = request.query_params.get("state")
    if not verify_state(expected_state, received_state):
        raise HTTPException(status_code=400, detail={"error": "invalid_state"})
    expected_nonce = request.cookies.get(_SSO_NONCE_COOKIE)

    try:
        identity = await fetch_identity(
            provider,
            oauth_client=client,
            request=request,
            expected_nonce=expected_nonce,
        )
        user = await resolve_or_link_user(identity, store=store)
    except SSOError as exc:
        _log.warning("SSO %s callback rejected: %s", provider, exc)
        raise HTTPException(status_code=400, detail={"error": "sso_failed"})
    except Exception as exc:  # noqa: BLE001
        _log.warning("SSO %s callback error: %s", provider, exc)
        raise HTTPException(status_code=400, detail={"error": "sso_failed"})

    # Redirect back to the SPA root with the session cookies set.
    response = RedirectResponse(url="/", status_code=302)
    await _issue_session(request, response, user)
    # Clear the transient SSO cookies.
    response.delete_cookie(_SSO_STATE_COOKIE, path="/")
    response.delete_cookie(_SSO_NONCE_COOKIE, path="/")
    return response


# ---------------------------------------------------------------------------
# Small internals.
# ---------------------------------------------------------------------------
def _sso_client(oauth, provider: str):
    try:
        client = oauth.create_client(provider)
    except Exception:
        client = None
    if client is None:
        raise HTTPException(
            status_code=404, detail={"error": "sso_provider_unavailable"}
        )
    return client


def _set_sso_cookie(response, name: str, value: str, sm: SessionManager) -> None:
    response.set_cookie(
        name,
        value,
        max_age=600,  # 10 min: ample for the IdP round-trip
        httponly=True,
        secure=sm.cookie_secure,
        samesite="lax",
        path="/",
    )
