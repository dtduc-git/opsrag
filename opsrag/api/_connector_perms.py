"""Request-scoped installation of per-connector RBAC.

Bridges the pure policy (:mod:`opsrag.auth.connector_perms`) to a live request:
resolves the caller's allowed connectors from config + roles + per-user
overrides, then installs the per-request tool filter that both the LLM
tool-spec builder and the tool executor consult (via
:mod:`opsrag.mcp_server.registry_loader`).

Called from the agent-facing handlers (``/query``, ``/investigate``) and the
MCP-token server path, right before the agent graph runs, so the contextvar is
set in the same task the graph streams in (mirrors ``current_user_oid_var``).
"""
from __future__ import annotations

from typing import Any

from opsrag.auth.connector_perms import resolve_allowed_connectors
from opsrag.mcp_server.registry_loader import set_request_connector_perms


def _enabled_and_restricted(cfg: Any) -> tuple[list[str], list[str]]:
    mcp_map = getattr(cfg, "mcp", {}) or {}
    enabled: list[str] = []
    restricted: list[str] = []
    for name, block in mcp_map.items():
        if not getattr(block, "enabled", False):
            continue
        enabled.append(name)
        if getattr(block, "restricted", False):
            restricted.append(name)
    return enabled, restricted


async def install_request_connector_perms(request: Any, user: Any) -> None:
    """Resolve ``user``'s allowed connectors and install them for this request.

    No-op (leaves gating off) when there's no config on ``app.state`` -- then
    only the process-wide enabled-integration gate applies. Per-user overrides
    are read from the auth user store (login mode only); role-based grants apply
    in every mode.
    """
    cfg = getattr(request.app.state, "config", None)
    if cfg is None:
        return

    enabled, restricted = _enabled_and_restricted(cfg)
    if not enabled:
        set_request_connector_perms(frozenset(), ())
        return

    role_connectors = getattr(getattr(cfg, "auth", None), "role_connectors", {}) or {}
    roles = list(getattr(user, "roles", ()) or ())

    # Per-user allow/deny overrides live on the auth user (login mode).
    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()
    store = getattr(request.app.state, "auth_user_store", None)
    uid = getattr(user, "oid", None) or getattr(user, "sub", None)
    if store is not None and uid:
        try:
            au = await store.get_user_by_id(str(uid))
        except Exception:  # noqa: BLE001 -- override lookup must never 500 a query
            au = None
        if au is not None:
            allow = tuple(getattr(au, "connectors_allow", ()) or ())
            deny = tuple(getattr(au, "connectors_deny", ()) or ())

    allowed = resolve_allowed_connectors(
        roles=roles,
        role_connectors=role_connectors,
        restricted=restricted,
        enabled_connectors=enabled,
        user_allow=allow,
        user_deny=deny,
    )
    set_request_connector_perms(allowed, enabled)
