"""FastAPI Bearer-token middleware backed by ``opsrag.auth.oidc``.

Wiring:

1. The app factory reads ``settings.auth`` and constructs an
   ``OIDCVerifier`` once at startup (failing fast on
   ``AUTH_MISCONFIGURED`` if discovery doesn't resolve), then attaches
   it to ``app.state.oidc_verifier``.
2. Route handlers declare ``user: CurrentUser =
   Depends(require_authenticated_user)`` to enforce auth, or
   ``Depends(optional_user)`` to read identity when present without
   rejecting anonymous requests (e.g. ``/healthz``, ``/readyz``).

Security properties:

- The Bearer token itself is NEVER logged. The verifier's exception
  detail strings are deliberately terse ("invalid token", "token
  expired", ...) and do not echo the token or any of its claims.
- The ``sub`` claim is exposed on ``request.state.user_sub`` and on the
  returned ``CurrentUser`` object so handlers can attribute usage. It
  is NOT logged automatically -- handlers that want per-user telemetry
  must opt in.
- A missing or malformed ``Authorization`` header returns 401 with
  no extra detail; this prevents oracle-style probing.
"""
from __future__ import annotations

import logging

from fastapi import HTTPException, Request

from opsrag.auth.oidc import CurrentUser, OIDCVerifier

_log = logging.getLogger("opsrag.auth.middleware")


def _get_verifier(request: Request) -> OIDCVerifier | None:
    """Return the ``OIDCVerifier`` attached to ``app.state``, or None."""
    return getattr(request.app.state, "oidc_verifier", None)


def _extract_bearer(request: Request) -> str | None:
    """Pull the Bearer token out of the Authorization header.

    Returns ``None`` if the header is absent or doesn't match
    ``Bearer <token>``. Does NOT log the token value."""
    header = request.headers.get("authorization") or request.headers.get(
        "Authorization"
    )
    if not header:
        return None
    parts = header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


async def require_authenticated_user(request: Request) -> CurrentUser:
    """FastAPI dependency that enforces a verified OIDC Bearer token.

    Returns the ``CurrentUser`` constructed from the token's claims and
    attaches ``request.state.user_sub`` for downstream usage attribution.

    Raises ``HTTPException(401)`` when:
      - no verifier is wired (server misconfiguration -- 500 would be
        more honest, but 401 avoids leaking that auth is disabled)
      - the Authorization header is missing or malformed
      - the token fails signature / iss / aud / exp / kid verification
    """
    verifier = _get_verifier(request)
    if verifier is None:
        # Server-side misconfiguration: ``auth`` not wired into Settings,
        # or app factory skipped attaching the verifier. We refuse to
        # serve auth-required routes in this state.
        _log.error(
            "auth required but app.state.oidc_verifier is unset; "
            "refusing request",
        )
        raise HTTPException(status_code=401, detail="auth unavailable")
    token = _extract_bearer(request)
    if token is None:
        raise HTTPException(status_code=401, detail="missing bearer token")
    user = verifier.verify_to_user(token)
    # Surface the sub claim on request.state so downstream code (usage
    # attribution, logging filters that opt in) can read it without
    # re-parsing the token. The token itself is never put on state.
    request.state.user_sub = user.sub
    return user


async def optional_user(request: Request) -> CurrentUser:
    """FastAPI dependency that returns the verified ``CurrentUser`` when
    a valid Bearer token is present, or anonymous otherwise.

    Suitable for endpoints that read identity for personalisation but
    must not 401 anonymous clients (``/healthz``, ``/readyz``). Does
    NOT 401 on a malformed token -- a malformed token is treated as
    "no token". This keeps health probes immune to client-side bugs.
    """
    verifier = _get_verifier(request)
    if verifier is None:
        return CurrentUser.anonymous()
    token = _extract_bearer(request)
    if token is None:
        return CurrentUser.anonymous()
    try:
        user = verifier.verify_to_user(token)
    except HTTPException:
        return CurrentUser.anonymous()
    request.state.user_sub = user.sub
    return user


def _is_open_mode(request: Request) -> bool:
    """True when no auth enforcement applies.

    Open mode is intentional zero-config (``auth is None`` or
    ``auth.mode == "open"``), distinct from a misconfigured auth-required
    route with a missing verifier. We read ``app.state.auth_config.mode``
    when present and otherwise fall back to "no verifier == open" so
    today's local-dev behavior is preserved."""
    auth_cfg = getattr(request.app.state, "auth_config", None)
    if auth_cfg is not None:
        mode = getattr(auth_cfg, "mode", None)
        if mode is not None:
            return mode == "open"
    return _get_verifier(request) is None


async def get_current_user_dep(request: Request) -> CurrentUser:
    """The ONE converged identity dependency (OIDC-shape in all modes).

    This replaces the two same-named ``get_current_user_dep`` that used
    to exist (the ``middleware`` alias to ``require_authenticated_user``
    which 401'd, and the ``__init__`` bridge that produced the legacy
    Pomerium ``oid``-shape). Both are gone; this is the single source.

    Behavior by mode:
      * **open** (``auth is None`` / ``auth.mode == "open"``) -- never
        401; returns ``CurrentUser.anonymous()`` (which carries ALL
        scopes), preserving today's open behavior.
      * **oidc / login** -- verifies the Bearer token and returns the
        OIDC-shape ``CurrentUser``. A missing/invalid token here yields
        anonymous (the global ``OIDCAuthMiddleware`` has already rejected
        unauthenticated requests on protected routes before the handler
        runs, so reaching here token-less means an open route).

    Note: this returns identity WITHOUT resolved RBAC scopes. Routes that
    enforce scopes should depend on
    ``opsrag.auth.scopes.current_user_with_authz`` (or a
    ``require_scope(...)`` guard), which calls this and then attaches
    roles/scopes. ``CurrentUser.anonymous()`` already carries all scopes
    so open-mode handlers that read ``.scopes`` work without the authz
    step.
    """
    if _is_open_mode(request):
        return CurrentUser.anonymous()
    verifier = _get_verifier(request)
    if verifier is None:
        # auth.mode is oidc/login but no verifier wired: misconfiguration.
        # Fail closed -- treat as anonymous (the global middleware will
        # already have 401'd protected routes).
        return CurrentUser.anonymous()
    token = _extract_bearer(request)
    if token is None:
        return CurrentUser.anonymous()
    try:
        user = verifier.verify_to_user(token)
    except HTTPException:
        return CurrentUser.anonymous()
    request.state.user_sub = user.sub
    return user
