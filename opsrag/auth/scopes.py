"""RBAC scope model -- the single source of truth for authorization.

This module is the *only* place that knows the role->scope map and how
to turn an IdP ``groups`` claim into a set of roles/scopes. Both the
``/me`` payload (UI nav gating, cosmetic scope pill) and the
server-side ``require_scope`` route guards read from here, so the UI can
never show navigation the server then 403s on (FINDING 13 drift risk).

Concepts
--------
* **Scope** -- a capability string (``chat``, ``investigate``, ``mcp``,
  ``admin``). Handlers gate on scopes, not roles.
* **Role** -- a named bundle of scopes (``admin``, ``member_chat``,
  ``member_investigate``, ``member_mcp``). Roles come from the IdP via
  the operator-configured ``auth.role_mappings`` (group -> [roles]).
* **resolve_roles** -- maps an authenticated user's groups to roles,
  defaulting to ``member_investigate`` when nothing matches and to
  ``admin`` when the existing ``is_admin`` signal is set.

Modes
-----
Authentication is ALWAYS enforced -- there is no anonymous / "open" mode.
In both modes ``resolve_roles`` + ``scopes_for_roles`` compute the user's
real scopes from their roles/claims; ``require_scope`` 403s on a missing
scope (authenticated-but-unscoped), distinct from the 401 the auth layer
raises for unauthenticated requests.
* **login** (default) -- identity comes from the signed first-party
  session cookie (built-in admin + optional SSO); roles are baked into the
  session at login.
* **oidc** -- identity comes from a verified Bearer JWT; roles are resolved
  from the IdP ``groups`` claim via ``auth.role_mappings``.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from fastapi import Depends, HTTPException, Request

from opsrag.auth.middleware import get_current_user_dep
from opsrag.auth.oidc import CurrentUser


# ---------------------------------------------------------------------------
# Scope + role model.
# ---------------------------------------------------------------------------
class Scope:
    """Capability constants. String values are the wire/contract form
    (they appear in the ``/me`` payload and config ``role_mappings``)."""

    CHAT = "chat"
    INVESTIGATE = "investigate"
    MCP = "mcp"
    ADMIN = "admin"


# Every defined scope. The ``admin`` role bundles this whole set.
ALL_SCOPES: frozenset[str] = frozenset(
    {Scope.CHAT, Scope.INVESTIGATE, Scope.MCP, Scope.ADMIN}
)

# Role -> scopes. The single authoritative bundle map.
#   admin               -> everything
#   member_chat         -> chat only
#   member_investigate  -> chat + investigate (investigate implies chat)
#   member_mcp          -> mcp only
ROLE_SCOPES: dict[str, set[str]] = {
    "admin": {Scope.CHAT, Scope.INVESTIGATE, Scope.MCP, Scope.ADMIN},
    "member_chat": {Scope.CHAT},
    "member_investigate": {Scope.CHAT, Scope.INVESTIGATE},
    "member_mcp": {Scope.MCP},
}

# Role assigned to an authenticated user whose groups match no mapping.
# NOT default-deny-to-nothing: a signed-in user with no explicit mapping
# gets chat+investigate (the common interactive role). Operators tighten
# this by configuring ``role_mappings`` explicitly.
DEFAULT_ROLE = "member_investigate"

# Config-driven override of the unmatched-user fallback. ``auth.default_roles``
# (bound once at startup via ``set_default_roles``) replaces the single
# DEFAULT_ROLE so operators can, e.g., give every authenticated user
# ``[member_investigate, member_mcp]`` without a per-user override. Empty/None
# keeps the built-in DEFAULT_ROLE. Unknown role names contribute no scopes
# (scopes_for_roles is default-deny), so a typo can't over-grant.
_DEFAULT_ROLES: frozenset[str] = frozenset({DEFAULT_ROLE})


def set_default_roles(roles: Iterable[str] | None) -> None:
    """Bind the configured default-role set (from ``auth.default_roles``).

    Called once at startup. ``None``/empty restores the built-in DEFAULT_ROLE."""
    global _DEFAULT_ROLES
    cleaned = frozenset(r.strip() for r in (roles or ()) if r and r.strip())
    _DEFAULT_ROLES = cleaned or frozenset({DEFAULT_ROLE})


def default_roles() -> frozenset[str]:
    """The active unmatched-user fallback role set."""
    return _DEFAULT_ROLES

# Role implied by the existing ``is_admin`` boolean signal (e.g. the
# legacy ``is_admin_for(admin_group_oid)`` check, or a future
# admin-groups config). Always added on top of group-resolved roles.
ADMIN_ROLE = "admin"


# ---------------------------------------------------------------------------
# Resolution.
# ---------------------------------------------------------------------------
def resolve_roles(
    groups: Iterable[str],
    role_mappings: dict[str, list[str]] | None,
    *,
    is_admin: bool = False,
) -> set[str]:
    """Map IdP ``groups`` -> opsrag role names.

    Args:
      groups: the user's IdP group/role claim values.
      role_mappings: config ``auth.role_mappings`` -- ``{group: [roles]}``.
        ``None`` or ``{}`` means "no explicit mapping configured".
      is_admin: the existing admin signal (e.g. legacy
        ``is_admin_for(admin_group_oid)``). When set, ``admin`` is added.

    Returns the resolved set of role names. An authenticated user whose
    groups match nothing falls back to :data:`DEFAULT_ROLE` so a signed-in
    user is never left with zero roles; ``is_admin`` always adds
    :data:`ADMIN_ROLE` on top.
    """
    mappings = role_mappings or {}
    group_list = list(groups or ())
    roles: set[str] = set()
    for g in group_list:
        mapped = mappings.get(g)
        if mapped:
            roles.update(mapped)
    if is_admin:
        roles.add(ADMIN_ROLE)
    # Authenticated but unmatched -> configured default role set (built-in
    # DEFAULT_ROLE unless auth.default_roles overrides). We only apply the
    # default when nothing matched at all (including no admin signal); an
    # explicit mapping wins.
    if not roles:
        roles.update(_DEFAULT_ROLES)
    return roles


def scopes_for_roles(roles: Iterable[str]) -> set[str]:
    """Union the scope bundles of every role in ``roles``.

    Unknown role names contribute no scopes (default-deny on unknown)."""
    out: set[str] = set()
    for role in roles or ():
        out.update(ROLE_SCOPES.get(role, set()))
    return out


def has_scope(user: CurrentUser | None, scope: str) -> bool:
    """True iff ``user`` carries ``scope``.

    A user's scopes come solely from their resolved roles/claims. Anonymous
    users (the scopeless ``CurrentUser.anonymous`` returned on public
    allowlist routes) and a ``None`` user (defensive) carry no scopes, so
    this returns False for them."""
    if user is None:
        return False
    return scope in (user.scopes or frozenset())


# ---------------------------------------------------------------------------
# app.state wiring helpers (read by the auth dependency + server.py).
# ---------------------------------------------------------------------------
def _role_mappings(request: Request) -> dict[str, list[str]]:
    """Read ``auth.role_mappings`` off app.state (wired by server.py).

    Accepts either a bare dict on ``app.state.role_mappings`` or an
    ``auth_config`` object exposing ``.role_mappings``."""
    direct = getattr(request.app.state, "role_mappings", None)
    if direct is not None:
        return direct
    auth_cfg = getattr(request.app.state, "auth_config", None)
    if auth_cfg is not None:
        return getattr(auth_cfg, "role_mappings", None) or {}
    return {}


def attach_authz(
    user: CurrentUser,
    *,
    role_mappings: dict[str, list[str]] | None,
    is_admin: bool = False,
) -> CurrentUser:
    """Resolve roles+scopes for ``user`` and return the enriched copy.

    Scopes come solely from the user's roles, resolved from their IdP
    ``groups`` claim + ``role_mappings`` + the ``is_admin`` signal. An
    anonymous user (only reachable on a public allowlist route) carries no
    roles and therefore no scopes."""
    if user.is_anonymous:
        return user.with_authz(roles=frozenset(), scopes=frozenset())
    roles = resolve_roles(user.groups, role_mappings, is_admin=is_admin)
    scopes = scopes_for_roles(roles)
    return user.with_authz(roles=frozenset(roles), scopes=frozenset(scopes))


def _user_from_session(request: Request) -> CurrentUser | None:
    """Resolve a CurrentUser from the first-party session COOKIE (login
    mode). Returns None when there's no SessionManager or no/invalid cookie."""
    sm = getattr(request.app.state, "session_manager", None)
    if sm is None:
        return None
    token = request.cookies.get(sm.SESSION_COOKIE)
    payload = sm.verify_session(token)
    if payload is None:
        return None
    roles = frozenset(payload.roles or ())
    scopes = frozenset(scopes_for_roles(roles))
    return CurrentUser(
        sub=payload.user_id,
        email=payload.email,
        name=payload.email,
        picture_url=None,
        groups=(),
        is_anonymous=False,
    ).with_authz(roles=roles, scopes=scopes)


async def current_user_with_authz(request: Request) -> CurrentUser:
    """The converged auth dependency, RBAC-aware.

    Authentication is ALWAYS enforced -- there is no anonymous / "open"
    mode. A request that reaches a protected handler has already been
    authenticated by the global middleware; reaching here without a valid
    identity means the route is on the public allowlist, for which we
    return a scopeless anonymous user (passes no ``require_scope`` guard).

    - login mode: identity comes from the signed session COOKIE.
    - oidc mode: delegate to the OIDC ``get_current_user_dep`` and enrich
      with roles/scopes resolved from groups + ``role_mappings``."""
    auth_cfg = getattr(request.app.state, "auth_config", None)
    mode = getattr(auth_cfg, "mode", None) if auth_cfg is not None else None
    if mode == "login":
        cookie_user = _user_from_session(request)
        if cookie_user is not None:
            request.state.user_sub = cookie_user.sub
            return cookie_user
        return CurrentUser.anonymous().with_authz(
            roles=frozenset(), scopes=frozenset()
        )

    user = await get_current_user_dep(request)
    user = attach_authz(
        user,
        role_mappings=_role_mappings(request),
    )
    # Surface for downstream usage attribution without re-parsing.
    if not user.is_anonymous:
        request.state.user_sub = user.sub
    return user


# ---------------------------------------------------------------------------
# require_scope dependency factory.
# ---------------------------------------------------------------------------
def _forbidden(request: Request, scope: str) -> HTTPException:
    """Build the repo's 403 error-envelope HTTPException.

    Mirrors the ``{error, reason, request_id}`` shape the auth
    middleware emits for 401s (oidc_enforcement.py), but with
    ``error: forbidden`` so the UI can tell 401 (re-auth) apart from
    403 (insufficient scope)."""
    rid = getattr(request.state, "request_id", None)
    return HTTPException(
        status_code=403,
        detail={
            "error": "forbidden",
            "reason": "missing_scope",
            "scope": scope,
            "request_id": rid,
        },
    )


def require_scope(scope: str) -> Callable[..., Any]:
    """FastAPI dependency factory enforcing ``scope`` on a route.

    Returns a dependency that resolves the current user (with authz) and
    403s via the repo error envelope when the user lacks ``scope``. The
    global middleware has already 401'd unauthenticated requests, so this
    distinguishes authenticated-but-unscoped (403) from unauthenticated
    (401).

    Usage::

        @router.get("/admin/usage")
        async def admin_usage(
            user: CurrentUser = Depends(require_scope(Scope.ADMIN)),
        ): ...
    """

    async def _dep(
        request: Request,
        user: CurrentUser = Depends(current_user_with_authz),
    ) -> CurrentUser:
        if not has_scope(user, scope):
            raise _forbidden(request, scope)
        return user

    return _dep
