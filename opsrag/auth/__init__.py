"""OIDC identity package.

Current (v1) surface for new code::

    from opsrag.auth.oidc import CurrentUser, OIDCVerifier
    from opsrag.auth.middleware import (
        require_authenticated_user,
        optional_user,
    )

The dependency-injection contract is:

  1. The app factory reads ``settings.auth`` and builds an
     ``OIDCVerifier`` (failing fast on ``AUTH_MISCONFIGURED`` if
     discovery does not resolve); attaches it to
     ``app.state.oidc_verifier``.
  2. Route handlers declare ``user: CurrentUser =
     Depends(require_authenticated_user)`` to enforce auth, or
     ``Depends(optional_user)`` to read identity when present.

Legacy (M1) surface -- ``PomeriumVerifier``,
``opsrag.auth.pomerium.CurrentUser`` with the ``oid`` /
``is_admin_for`` shape -- is still re-exported because
``opsrag.api.routes`` and friends have not been rewritten yet. Those
modules continue to import ``CurrentUser`` from this module and get the
legacy shape. ``opsrag.api.server`` + ``opsrag.api.routes`` rewrites
land in T057-T060; after those land, ``opsrag.auth.pomerium`` and the
legacy re-exports here disappear.

Note: ``from opsrag.auth import CurrentUser`` resolves to the **legacy**
Pomerium-shape ``CurrentUser`` (attribute ``oid``, method
``is_admin_for``). New code that wants the OIDC-shape ``CurrentUser``
(attribute ``sub``, method ``is_member_of``) MUST import explicitly
from ``opsrag.auth.oidc``.
"""
from __future__ import annotations

from contextvars import ContextVar

from fastapi import Request

from opsrag.auth.middleware import (  # noqa: F401
    get_current_user_dep as _oidc_get_current_user_dep,
)
from opsrag.auth.middleware import (
    optional_user,
    require_authenticated_user,
)

# Canonical identity surface -- the OIDC-shape ``CurrentUser`` (RBAC v1
# convergence). ``opsrag.auth.CurrentUser`` now resolves to the
# OIDC-shape model (attributes ``sub``/``groups``/``roles``/``scopes``,
# methods ``is_member_of``/``has_scope``). For back-compat it also
# exposes ``.oid`` (alias of ``.sub``) so the not-yet-rewritten
# ``opsrag.api.routes`` call sites keep working until T057-T060.
from opsrag.auth.oidc import (  # noqa: F401
    CurrentUser,
    OIDCVerifier,
    build_verifier_from_settings,
)
from opsrag.auth.oidc import CurrentUser as OIDCCurrentUser  # noqa: F401

# Legacy Pomerium surface. The verifier + claims-extraction helper are
# still re-exported for the M1 fallback path used by ``get_current_user``
# below; the Pomerium-shape ``CurrentUser`` is NO LONGER the default
# export (it lives at ``opsrag.auth.pomerium.CurrentUser`` for the few
# remaining callers). Removed after T057-T060.
from opsrag.auth.pomerium import (  # noqa: F401
    PomeriumVerifier,
    extract_current_user,
)
from opsrag.auth.store import UserStore

__all__ = [
    "CurrentUser",
    "OIDCCurrentUser",
    "OIDCVerifier",
    "PomeriumVerifier",
    "UserStore",
    "build_verifier_from_settings",
    "current_user_oid_var",
    "extract_current_user",
    "get_current_user",
    "get_current_user_dep",
    "optional_user",
    "require_authenticated_user",
]


# M2 -- request-scoped user oid. The query handler `set()`s this on the
# way in; the Vertex `on_usage` hook reads it when forwarding to
# UsagePersistence.enqueue. ContextVar (not a global) so concurrent
# requests don't cross-contaminate.
current_user_oid_var: ContextVar[str | None] = ContextVar(
    "opsrag_current_user_oid", default=None,
)


async def get_current_user(request: Request) -> CurrentUser:
    """Converged identity + RBAC dependency (OIDC-shape, all modes).

    Resolution is delegated to the single OIDC identity dependency
    (``opsrag.auth.middleware.get_current_user_dep``) and then enriched
    with resolved RBAC ``roles``/``scopes`` via
    ``opsrag.auth.scopes.current_user_with_authz``:

      * **open** (``auth is None`` / ``auth.mode == "open"``) -> anonymous
        carrying ALL scopes (today's zero-config behavior preserved).
      * **oidc / login** -> verified OIDC user with roles/scopes resolved
        from ``groups`` + ``auth.role_mappings``.

    The OIDC ``CurrentUser`` exposes a back-compat ``.oid`` alias of
    ``.sub`` so the not-yet-migrated ``opsrag.api.routes`` call sites and
    the usage-attribution contextvar keep working (removed at T057-T060).

    Side-effect: if ``app.state.user_store`` is wired AND the user is
    non-anonymous, upsert their row into ``opsrag_user`` (debounced). The
    analytics table is a nice-to-have, never load-bearing.
    """
    from opsrag.auth.scopes import current_user_with_authz

    user = await current_user_with_authz(request)
    # Side-channel upsert -- only when we have a real identity AND the
    # store was wired at startup. Failure of upsert MUST NOT break the
    # request (the analytics table is a nice-to-have, not load-bearing).
    if not user.is_anonymous:
        store = getattr(request.app.state, "user_store", None)
        if store is not None:
            try:
                await store.upsert(user)
            except Exception as exc:  # noqa: BLE001
                # Log at debug -- a noisy warning here would be a
                # foot-gun if the user_store pool degrades.
                import logging
                logging.getLogger("opsrag.auth").debug(
                    "user_store.upsert failed for %s: %s", user.oid, exc,
                )
    return user


# The spec asks for a callable named ``get_current_user_dep``. It is the
# converged dependency above (RBAC-aware + user_store upsert). FastAPI's
# ``Depends`` plumbing gives it request access, so the function itself IS
# the dependency. ``opsrag.api.routes`` imports THIS name.
get_current_user_dep = get_current_user
