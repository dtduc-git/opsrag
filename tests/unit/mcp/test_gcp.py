"""Unit tests for the GCP MCP tools against the offline fake backend.

Exercises every tool through build_fake() with NO network, NO google libs,
and NO GCP credentials, asserting the parsed shape from the canned data.
asyncio_mode = "auto" (see pyproject.toml) so no decorator is needed.

Cloud Monitoring + Logging moved to the dedicated `stackdriver` connector
(see tests/unit/mcp/test_stackdriver.py); they are no longer part of this
tool set.
"""
from __future__ import annotations

import pytest

from opsrag.mcp.gcp import GCP_TOOLS, build_fake, get_tool

EXPECTED_TOOLS = {
    "gcp_gke_list_clusters",
    "gcp_run_list_services",
    "gcp_asset_search",
}


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()  # restore real module-level _get


def test_tool_set_matches_exactly():
    assert {t.name for t in GCP_TOOLS} == EXPECTED_TOOLS
    assert fake_tool_names() == EXPECTED_TOOLS


def fake_tool_names():
    f = build_fake()
    try:
        return set(f.tool_names())
    finally:
        f.close()


async def test_gke_list_clusters(fake):
    res = await fake.call("gcp_gke_list_clusters", {"project": "demo"})
    assert res["location"] == "-"
    assert res["count"] == 1
    c = res["clusters"][0]
    assert c["name"] == "prod-gke"
    assert c["status"] == "RUNNING"
    assert c["current_node_count"] == 6
    assert c["master_version"] == "1.30.2-gke.100"


async def test_run_list_services(fake):
    res = await fake.call("gcp_run_list_services", {"project": "demo"})
    assert res["location"] == "-"
    assert res["count"] == 1
    s = res["services"][0]
    assert s["uri"] == "https://api-xyz.a.run.app"
    assert s["images"] == ["gcr.io/demo/api:1.2.3"]
    assert s["labels"] == {"team": "platform"}


async def test_asset_search(fake):
    res = await fake.call(
        "gcp_asset_search",
        {
            "project": "demo",
            "query": "state:RUNNING",
            "asset_types": ["compute.googleapis.com/Instance"],
        },
    )
    assert res["query"] == "state:RUNNING"
    assert res["asset_types"] == "compute.googleapis.com/Instance"
    assert res["count"] == 1
    r = res["results"][0]
    assert r["asset_type"] == "compute.googleapis.com/Instance"
    assert r["state"] == "RUNNING"
    assert r["labels"] == {"env": "prod"}


async def test_handler_direct_invocation_pattern():
    # Exercise the get_tool + tool.handler(client, args) path explicitly.
    fake = build_fake()
    try:
        tool = get_tool("gcp_gke_list_clusters")
        res = await tool.handler(fake.client, {"project": "demo"})
        assert res["count"] == 1
    finally:
        fake.teardown()
