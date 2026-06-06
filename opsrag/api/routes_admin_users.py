"""Admin RBAC API -- list users + assign roles (login-mode user management).

Backs the UI "Users & Roles" page. Every route requires the ``admin`` scope
(``require_scope`` 403s non-admins). Roles are the bundles defined in
``opsrag.auth.scopes`` -- the single authoritative model -- so assigning a
role re-derives the user's scopes from the same map the request guards read.

Only meaningful in **login** mode (users live in the ``AuthUserStore``); in
open/oidc mode there is no local user store and these routes 503.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from opsrag.auth.oidc import CurrentUser
from opsrag.auth.scopes import (
    ROLE_SCOPES,
    Scope,
    require_scope,
    resolve_roles,
    scopes_for_roles,
)

_log = logging.getLogger("opsrag.api.routes_admin_users")

admin_users_router = APIRouter(prefix="/admin", tags=["admin"])

# Display metadata for the role catalog (drives the UI editor + ordering).
# (role, label, description). Scopes come from the authoritative ROLE_SCOPES.
ROLE_LABELS: list[tuple[str, str, str]] = [
    ("admin", "Admin", "Full access — everything, including user management."),
    ("member_investigate", "Investigate", "Chat + agentic investigations."),
    ("member_chat", "Chat", "Ask questions only (no investigations)."),
    ("member_mcp", "MCP", "Use OpsRAG from external editors via MCP tokens."),
]


class AdminUser(BaseModel):
    id: str
    email: str
    name: str | None = None
    roles: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)
    has_password: bool = False
    email_verified: bool = False
    created_at: str | None = None
    updated_at: str | None = None


class AdminUserListResponse(BaseModel):
    users: list[AdminUser]


class RoleInfo(BaseModel):
    role: str
    label: str
    description: str
    scopes: list[str]


class RoleCatalogResponse(BaseModel):
    roles: list[RoleInfo]


class SetRolesRequest(BaseModel):
    roles: list[str]


def _store(request: Request) -> Any:
    store = getattr(request.app.state, "auth_user_store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="user management requires login mode (no auth user store wired)",
        )
    return store


def _iso(d: datetime | None) -> str | None:
    return d.isoformat() if d else None


def _to_admin_user(u: Any) -> AdminUser:
    # Show EFFECTIVE roles: a user with no explicitly-stored roles still gets
    # the default interactive role at login (resolve_roles fallback), so the
    # admin view reflects what the user can actually do -- not a misleading
    # empty set. Saving from the UI then persists it explicitly.
    roles = list(u.roles or ()) or sorted(resolve_roles((), None))
    return AdminUser(
        id=str(u.id),
        email=u.email,
        name=u.name,
        roles=roles,
        scopes=sorted(scopes_for_roles(roles)),
        has_password=bool(u.password_hash),
        email_verified=bool(u.email_verified),
        created_at=_iso(u.created_at),
        updated_at=_iso(u.updated_at),
    )


@admin_users_router.get("/roles", response_model=RoleCatalogResponse)
async def list_roles(
    _user: CurrentUser = Depends(require_scope(Scope.ADMIN)),
) -> RoleCatalogResponse:
    """The role catalog (role -> scopes + UI label). Drives the editor."""
    return RoleCatalogResponse(
        roles=[
            RoleInfo(
                role=r,
                label=lbl,
                description=desc,
                scopes=sorted(ROLE_SCOPES.get(r, set())),
            )
            for r, lbl, desc in ROLE_LABELS
        ]
    )


@admin_users_router.get("/users", response_model=AdminUserListResponse)
async def list_users(
    request: Request,
    limit: int = 200,
    _user: CurrentUser = Depends(require_scope(Scope.ADMIN)),
) -> AdminUserListResponse:
    """List users (newest first) with their roles + derived scopes."""
    store = _store(request)
    users = await store.list_users(limit=limit)
    return AdminUserListResponse(users=[_to_admin_user(u) for u in users])


@admin_users_router.put("/users/{user_id}/roles", response_model=AdminUser)
async def set_user_roles(
    user_id: str,
    body: SetRolesRequest,
    request: Request,
    current_user: CurrentUser = Depends(require_scope(Scope.ADMIN)),
) -> AdminUser:
    """Replace a user's roles. Validates against the catalog, blocks
    self-lockout, and revokes the user's refresh sessions so the change
    can't be extended past the current (15-min) session cookie."""
    store = _store(request)

    # Validate against the authoritative catalog (default-deny unknown).
    requested = list(dict.fromkeys(body.roles))  # dedupe, preserve order
    unknown = [r for r in requested if r not in ROLE_SCOPES]
    if unknown:
        raise HTTPException(
            status_code=400, detail=f"unknown role(s): {', '.join(unknown)}"
        )

    target = await store.get_user_by_id(user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="user not found")

    # Lockout guard: an admin cannot strip their OWN admin role (which would
    # lock them out of user management). Demoting OTHER admins is allowed.
    if current_user.oid == user_id and "admin" not in requested:
        raise HTTPException(
            status_code=400, detail="you cannot remove your own admin role"
        )

    await store.set_roles(user_id, tuple(requested))
    # Roles are baked into the signed session cookie + refresh token. Revoke
    # the user's refresh sessions so they can't extend the old roles; the new
    # roles take effect when their session next refreshes or they sign in.
    try:
        await store.revoke_all_for_user(user_id)
    except Exception as exc:  # noqa: BLE001
        _log.warning("set_roles: revoke_all_for_user(%s) failed: %s", user_id, exc)

    updated = await store.get_user_by_id(user_id)
    _log.info(
        "admin %s set roles for user %s -> %s",
        current_user.email, user_id, requested,
    )
    return _to_admin_user(updated or target)
