"""Per-connector authorization policy (RBAC over MCP connectors).

This module is the *pure* policy layer that decides which MCP connectors a
user may use. It complements the coarse capability scopes in
:mod:`opsrag.auth.scopes` (``chat``/``investigate``/``mcp``/``admin``) with a
finer, per-connector grant model, WITHOUT changing how scopes gate routes.

Model
-----
* **Default-allow.** Any enabled connector is usable by any authenticated user
  UNLESS the operator flags it ``restricted: true`` in ``mcp.<name>``. This
  keeps upgrades behavior-preserving: nothing is locked down until an operator
  explicitly restricts a connector.
* **Restricted connectors need a grant.** A restricted connector is usable only
  by users granted it -- via ``auth.role_connectors`` (role -> [connector...])
  for one of their roles, OR via a per-user allow override.
* **Admin / wildcard.** The ``admin`` role, and any role whose grant list
  contains ``"*"``, implies every enabled connector.
* **Per-user overrides.** ``connectors_allow`` adds connectors (even restricted
  ones); ``connectors_deny`` removes them and **wins over everything** (role
  grants, default-allow, even admin) -- the escape hatch for "this one person
  must not touch billing".

The result is always a subset of the ENABLED connectors: you can never grant a
connector the deployment hasn't turned on.

Enforcement wiring lives elsewhere (it needs the tool registry + request
context, which would make this module impure):
  * :mod:`opsrag.mcp_server.registry_loader` maps allowed connectors -> allowed
    tool names and applies a per-request filter that both the LLM tool-spec
    builder and the tool executor consult.
  * The API request handlers compute a user's allowed set (via this module) and
    install it for the request.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

# Wildcard grant token in ``role_connectors`` values: grants every enabled
# connector (used for allow-all roles without enumerating connectors).
WILDCARD = "*"

# Role that implicitly grants every enabled connector regardless of
# ``role_connectors`` (kept in sync with opsrag.auth.scopes.ADMIN_ROLE, but
# duplicated here to keep this module import-light and pure).
ADMIN_ROLE = "admin"


def resolve_allowed_connectors(
    *,
    roles: Iterable[str],
    role_connectors: Mapping[str, Iterable[str]] | None,
    restricted: Iterable[str],
    enabled_connectors: Iterable[str],
    user_allow: Iterable[str] = (),
    user_deny: Iterable[str] = (),
) -> frozenset[str]:
    """Compute the connectors ``roles`` + overrides may use.

    Args:
      roles: the user's resolved role names (e.g. from the session cookie).
      role_connectors: config ``auth.role_connectors`` -- ``{role: [connector]}``.
        A value containing :data:`WILDCARD` grants all enabled connectors.
      restricted: connector names flagged ``restricted: true``.
      enabled_connectors: connectors actually enabled on this deployment. The
        result is always a subset of this set.
      user_allow: per-user connector grants (add; may grant a restricted one).
      user_deny: per-user connector denials (remove; wins over everything).

    Returns the frozenset of connector names the user may use.
    """
    enabled = {str(c) for c in enabled_connectors}
    if not enabled:
        return frozenset()

    restricted_set = {str(c) for c in restricted} & enabled
    role_set = {str(r) for r in roles or ()}
    rc = {str(r): {str(c) for c in v} for r, v in (role_connectors or {}).items()}

    # Connectors granted to the user's roles.
    grants: set[str] = set()
    for r in role_set:
        g = rc.get(r, set())
        grants |= enabled if WILDCARD in g else (g & enabled)
    if ADMIN_ROLE in role_set:
        grants |= enabled  # admin implies every enabled connector

    # Default-allow non-restricted; restricted require a grant.
    allowed = {c for c in enabled if c not in restricted_set}
    allowed |= {c for c in restricted_set if c in grants}

    # Per-user allow adds (even a restricted connector); deny wins over all.
    allowed |= ({str(c) for c in user_allow} & enabled)
    allowed -= {str(c) for c in user_deny}

    return frozenset(allowed)
