"""Integration test for per-connector RBAC wiring (no LLM).

Exercises the real request path used by ``/query``:
``install_request_connector_perms`` -> the ``registry_loader`` contextvar ->
``filter_enabled`` (what the LLM sees / the executor runs) and
``request_denied_connectors`` (the refusal prompt hint). Uses lightweight
fakes for ``request`` + config and the real :class:`InMemoryAuthUserStore`.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from opsrag.api._connector_perms import install_request_connector_perms
from opsrag.auth.user_store import InMemoryAuthUserStore
from opsrag.mcp.registry import REGISTRY
from opsrag.mcp_server.registry_loader import (
    clear_request_connector_perms,
    filter_enabled,
    request_denied_connectors,
    set_active_enabled,
)


def _block(enabled: bool, restricted: bool = False) -> SimpleNamespace:
    return SimpleNamespace(enabled=enabled, restricted=restricted)


def _cfg(*, role_connectors=None) -> SimpleNamespace:
    # datadog is enabled + RESTRICTED; gitlab is enabled + open.
    return SimpleNamespace(
        mcp={
            "datadog": _block(True, restricted=True),
            "gitlab": _block(True),
            "sentry": _block(False),  # disabled -> never in play
        },
        auth=SimpleNamespace(role_connectors=role_connectors or {}),
    )


def _request(store, cfg) -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(config=cfg, auth_user_store=store))
    )


# All tool objects the registry knows about (fakes with a `.name`), so
# filter_enabled has a realistic superset to trim.
_ALL_TOOLS = [
    SimpleNamespace(name=t)
    for integ in REGISTRY.values()
    for t in integ.tool_names
]
_DATADOG_TOOLS = set(REGISTRY["datadog"].tool_names)
_GITLAB_TOOLS = set(REGISTRY["gitlab"].tool_names)


def _visible_tool_names() -> set[str]:
    return {t.name for t in filter_enabled(_ALL_TOOLS)}


@pytest.fixture(autouse=True)
def _isolate_gating():
    # The request layer is what we test; leave process-wide gating off so
    # filter_enabled reflects only the per-request RBAC set.
    set_active_enabled(None)
    clear_request_connector_perms()
    yield
    set_active_enabled(None)
    clear_request_connector_perms()


@pytest.mark.asyncio
async def test_admin_sees_restricted_connector():
    store = InMemoryAuthUserStore()
    admin = await store.create_user(email="admin@x.io", password_hash="h", roles=("admin",))
    user = SimpleNamespace(roles=frozenset({"admin"}), oid=admin.id, sub=admin.id)

    await install_request_connector_perms(_request(store, _cfg()), user)

    visible = _visible_tool_names()
    assert _DATADOG_TOOLS <= visible  # admin gets the restricted connector
    assert _GITLAB_TOOLS <= visible
    assert request_denied_connectors() == frozenset()


@pytest.mark.asyncio
async def test_member_denied_restricted_connector():
    store = InMemoryAuthUserStore()
    member = await store.create_user(
        email="m@x.io", password_hash="h", roles=("member_investigate",)
    )
    user = SimpleNamespace(
        roles=frozenset({"member_investigate"}), oid=member.id, sub=member.id
    )

    await install_request_connector_perms(_request(store, _cfg()), user)

    visible = _visible_tool_names()
    assert not (_DATADOG_TOOLS & visible)          # datadog hidden entirely
    assert _GITLAB_TOOLS <= visible                # open connector still visible
    assert request_denied_connectors() == frozenset({"datadog"})  # drives refusal


@pytest.mark.asyncio
async def test_member_granted_restricted_via_role_connectors():
    store = InMemoryAuthUserStore()
    member = await store.create_user(
        email="fin@x.io", password_hash="h", roles=("finance",)
    )
    user = SimpleNamespace(roles=frozenset({"finance"}), oid=member.id, sub=member.id)

    cfg = _cfg(role_connectors={"finance": ["datadog"]})
    await install_request_connector_perms(_request(store, cfg), user)

    assert _DATADOG_TOOLS <= _visible_tool_names()
    assert request_denied_connectors() == frozenset()


@pytest.mark.asyncio
async def test_per_user_allow_override_grants_restricted():
    store = InMemoryAuthUserStore()
    member = await store.create_user(
        email="m2@x.io", password_hash="h", roles=("member_investigate",)
    )
    await store.set_connector_overrides(member.id, allow=("datadog",), deny=())
    user = SimpleNamespace(
        roles=frozenset({"member_investigate"}), oid=member.id, sub=member.id
    )

    await install_request_connector_perms(_request(store, _cfg()), user)

    assert _DATADOG_TOOLS <= _visible_tool_names()
    assert request_denied_connectors() == frozenset()


@pytest.mark.asyncio
async def test_per_user_deny_override_fences_admin():
    store = InMemoryAuthUserStore()
    admin = await store.create_user(email="a2@x.io", password_hash="h", roles=("admin",))
    await store.set_connector_overrides(admin.id, allow=(), deny=("datadog",))
    user = SimpleNamespace(roles=frozenset({"admin"}), oid=admin.id, sub=admin.id)

    await install_request_connector_perms(_request(store, _cfg()), user)

    visible = _visible_tool_names()
    assert not (_DATADOG_TOOLS & visible)  # deny wins even over admin
    assert _GITLAB_TOOLS <= visible
    assert request_denied_connectors() == frozenset({"datadog"})
