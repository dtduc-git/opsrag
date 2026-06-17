"""Global OIDC Bearer enforcement middleware (FR-016).

Replaces the upstream ``X-API-Key`` gate with OIDC-only auth: every
request except the health/metadata allowlist must carry a valid
``Authorization: Bearer <token>`` verified against the configured
issuer's JWKS.

Design:

- Authentication is ALWAYS enforced -- there is no anonymous / "open"
  mode. The app factory attaches an ``OIDCVerifier`` to
  ``app.state.oidc_verifier`` (built from ``settings.auth``) in ``oidc``
  mode, or wires a ``SessionManager`` in ``login`` mode. Either way every
  non-allowlisted request must carry a valid identity; a request that
  reaches this middleware with neither a wired verifier nor a login
  session manager is a misconfiguration and is rejected (fail closed).
- On rejection it returns the stable error envelope directly
  (contracts/http-api.md), with a 401 ``error: unauthenticated`` and a
  ``reason`` from the closed set
  ``{missing_bearer, invalid_signature, issuer_mismatch,
  audience_mismatch, expired}``.
- The token is NEVER logged. Verified claims are stashed on
  ``request.state.oidc_claims`` / ``request.state.user_sub`` for handlers
  that want them; the per-route ``get_current_user`` dependency re-derives
  the legacy ``CurrentUser`` shape from the same verifier.
"""
from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from opsrag.auth.middleware import _extract_bearer

# Paths that never require auth: liveness/readiness probes plus the
# schema/docs endpoints (kept open so tooling and the UI can read the
# spec). Everything else is gated when a verifier is configured.
NO_AUTH_PATHS: frozenset[str] = frozenset(
    {
        "/healthz",
        "/readyz",
        "/health",  # legacy liveness path kept for UI back-compat
        "/openapi.json",
        "/docs",
        "/docs/oauth2-redirect",
        "/redoc",
        # Public so the login screen can brand itself before auth (no secrets).
        "/ui-config",
        # SCM webhooks authenticate with a per-provider secret/HMAC, not OIDC
        # (see opsrag.api.routes_webhooks); they must bypass Bearer enforcement.
        "/webhook/gitlab",
        "/webhook/github",
    }
)

# MCP *wire protocol* endpoints. External MCP clients (Claude Code, Cursor)
# authenticate these with their own ``opsrag_`` bearer token, validated by
# ``get_mcp_token_dep`` IN the endpoint -- they never carry a session cookie or
# an OIDC bearer. So they must bypass the global session/OIDC enforcement here
# (the endpoint is self-protecting: no valid MCP token -> 401). NOTE: this is
# only the wire protocol; the token-MANAGEMENT routes (``/mcp/tokens*``) are
# browser/session-authed and intentionally NOT listed, so they stay enforced.
MCP_WIRE_PATHS: frozenset[str] = frozenset({"/mcp/sse", "/mcp/messages"})

# Map the verifier's terse exception detail to a contract reason code.
_DETAIL_TO_REASON: dict[str, str] = {
    "missing token": "missing_bearer",
    "missing bearer token": "missing_bearer",
    "token expired": "expired",
    "invalid audience": "audience_mismatch",
    "invalid issuer": "issuer_mismatch",
}


def _reason_for(detail: str) -> str:
    return _DETAIL_TO_REASON.get(detail, "invalid_signature")


# Read-only methods never mutate state, so they don't need a CSRF token.
_SAFE_METHODS: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})


class OIDCAuthMiddleware(BaseHTTPMiddleware):
    """Enforce OIDC Bearer auth on every non-allowlisted request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Stamp a request id so logs and the error envelope agree.
        rid = uuid.uuid4().hex
        request.state.request_id = rid

        path = request.url.path
        # Exact allowlist + the first-party login surface (login mode). The
        # login/SSO routes (incl. parameterized /auth/sso/{provider}/callback)
        # must bypass bearer auth so users can actually authenticate.
        if path in NO_AUTH_PATHS or path.startswith("/auth/"):
            return await call_next(request)

        # MCP wire protocol: bearer-token (opsrag_*) authed at the endpoint,
        # not via session/OIDC. Bypass global enforcement; the endpoint 401s
        # on a missing/invalid MCP token.
        if path in MCP_WIRE_PATHS:
            return await call_next(request)

        # Login mode: enforce the first-party session COOKIE (no OIDC
        # verifier is wired in this mode). Reject requests without a valid
        # signed session; /auth/* + health are already allowlisted above so
        # users can log in.
        auth_cfg = getattr(request.app.state, "auth_config", None)
        mode = getattr(auth_cfg, "mode", None) if auth_cfg is not None else None
        if mode == "login":
            sm = getattr(request.app.state, "session_manager", None)
            if sm is None:
                # login configured but runtime not ready -> fail closed.
                return self._reject(rid, "login_unavailable")
            cookie = request.cookies.get(sm.SESSION_COOKIE)
            if sm.verify_session(cookie) is None:
                return self._reject(rid, "missing_session")
            # CSRF double-submit on state-changing methods (H1): the session
            # cookie is SameSite=Lax, which does NOT protect top-level cross-site
            # POSTs, so require the SPA to echo the non-HttpOnly opsrag_csrf
            # cookie in the X-CSRF-Token header. Read-only methods are exempt;
            # /auth/* + MCP wire + health are allowlisted above; bearer/oidc mode
            # never reaches this branch.
            if request.method not in _SAFE_METHODS:
                if not sm.verify_csrf(
                    request.cookies.get(sm.CSRF_COOKIE),
                    request.headers.get(sm.CSRF_HEADER),
                ):
                    return self._reject(rid, "csrf_failed")
            return await call_next(request)

        verifier = getattr(request.app.state, "oidc_verifier", None)
        if verifier is None:
            # No verifier wired and not in login mode -> auth is
            # misconfigured. Fail closed: there is no anonymous/open mode.
            return self._reject(rid, "auth_misconfigured")

        token = _extract_bearer(request)
        if not token:
            return self._reject(rid, "missing_bearer")

        try:
            claims = verifier.verify(token)
        except Exception as exc:  # HTTPException(401) from the verifier
            detail = getattr(exc, "detail", "") or ""
            return self._reject(rid, _reason_for(str(detail)))

        request.state.oidc_claims = claims
        request.state.user_sub = claims.get("sub")
        return await call_next(request)

    @staticmethod
    def _reject(request_id: str, reason: str) -> JSONResponse:
        return JSONResponse(
            status_code=401,
            content={
                "error": "unauthenticated",
                "reason": reason,
                "request_id": request_id,
            },
        )
