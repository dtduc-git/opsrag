"""Integration test (T082): the Prometheus MCP tools against the fake backend.

Exercises representative tools through build_fake() with no cluster, no
Prometheus, and no network. The fake installs a DeploymentContext with one
configured cluster (so cluster resolution succeeds) and swaps the module's
`_proxy_get` for a canned, shape-faithful Prometheus HTTP API responder.
Asserts the canned responses and the registry's declared tool set (FR-012).
"""
from __future__ import annotations

import pytest

try:
    from opsrag.mcp.prometheus import build_fake
    from opsrag.mcp.registry import REGISTRY
    _IMPORT_ERROR = None
except Exception as exc:  # optional-dep / import guard
    build_fake = None
    REGISTRY = None
    _IMPORT_ERROR = exc

pytestmark = pytest.mark.skipif(
    _IMPORT_ERROR is not None,
    reason=f"prometheus mcp import failed: {_IMPORT_ERROR}",
)


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()


def test_fake_exposes_registry_tool_set(fake) -> None:
    # The fake's tools must match exactly what the registry declares.
    assert set(fake.tool_names()) == set(REGISTRY["prometheus"].tool_names)


@pytest.mark.asyncio
async def test_query(fake) -> None:
    result = await fake.call("prometheus_query", {"query": "up"})
    # Cluster resolved from the installed DeploymentContext.
    assert result["cluster"] == "example-cluster"
    data = result["data"]
    assert data["resultType"] == "vector"
    assert data["result"][0]["metric"]["__name__"] == "up"
    assert data["result"][0]["value"][1] == "1"


@pytest.mark.asyncio
async def test_alerts(fake) -> None:
    result = await fake.call("prometheus_alerts", {})
    assert result["cluster"] == "example-cluster"
    assert result["count"] == 1
    assert result["firing"] == 1
    assert result["alerts"][0]["name"] == "HighCpu"
    assert result["alerts"][0]["state"] == "firing"
    assert result["alerts"][0]["severity"] == "warning"


@pytest.mark.asyncio
async def test_targets(fake) -> None:
    result = await fake.call("prometheus_targets", {})
    assert result["cluster"] == "example-cluster"
    assert result["active_total"] == 1
    assert result["active_healthy"] == 1
    assert result["active_unhealthy"] == 0


@pytest.mark.asyncio
async def test_unknown_tool_raises(fake) -> None:
    with pytest.raises(KeyError):
        await fake.call("prometheus_does_not_exist", {})
