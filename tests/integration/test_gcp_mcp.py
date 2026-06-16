"""Integration test: the GCP MCP tools against the fake backend.

Exercises every tool through build_fake() with no network, no google libs,
and no GCP credentials, asserting shape-faithful responses and the
registry's declared tool set. Follows the AWS/CloudWatch reference
(FR-012).

Cloud Monitoring + Logging moved to the dedicated `stackdriver` connector
(see tests/integration/test_stackdriver_mcp.py).
"""
from __future__ import annotations

import pytest

from opsrag.mcp.gcp import build_fake, get_tool
from opsrag.mcp.registry import REGISTRY


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()  # restore the real module-level _get


def test_fake_exposes_registry_tool_set(fake) -> None:
    # The fake's tools must match exactly what the registry declares.
    assert set(fake.tool_names()) == set(REGISTRY["gcp"].tool_names)


@pytest.mark.asyncio
async def test_gke_list_clusters(fake) -> None:
    res = await fake.call("gcp_gke_list_clusters", {"project": "demo"})
    assert res["count"] == 1
    assert res["clusters"][0]["name"] == "prod-gke"


@pytest.mark.asyncio
async def test_run_list_services(fake) -> None:
    res = await fake.call("gcp_run_list_services", {"project": "demo"})
    assert res["count"] == 1
    assert res["services"][0]["images"] == ["gcr.io/demo/api:1.2.3"]


@pytest.mark.asyncio
async def test_asset_search(fake) -> None:
    res = await fake.call(
        "gcp_asset_search",
        {"project": "demo", "query": "state:RUNNING"},
    )
    assert res["count"] == 1
    assert res["results"][0]["state"] == "RUNNING"


@pytest.mark.asyncio
async def test_handler_direct_invocation_pattern() -> None:
    # Exercise the get_tool + tool.handler(client, args) path explicitly.
    fake = build_fake()
    try:
        tool = get_tool("gcp_gke_list_clusters")
        res = await tool.handler(fake.client, {"project": "demo"})
        assert res["count"] == 1
    finally:
        fake.teardown()


@pytest.mark.asyncio
async def test_unknown_tool_raises(fake) -> None:
    with pytest.raises(KeyError):
        await fake.call("gcp_does_not_exist", {})
