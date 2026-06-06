"""Unit tests for the GCP MCP tools against the offline fake backend.

Exercises every tool through build_fake() with NO network, NO google libs,
and NO GCP credentials, asserting the parsed shape from the canned data.
asyncio_mode = "auto" (see pyproject.toml) so no decorator is needed.
"""
from __future__ import annotations

import pytest

from opsrag.mcp.gcp import GCP_TOOLS, build_fake, get_tool

EXPECTED_TOOLS = {
    "gcp_logging_list_entries",
    "gcp_monitoring_list_timeseries",
    "gcp_monitoring_list_alert_policies",
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
        f.close()  # restore real module-level _get / _post


def test_tool_set_matches_exactly():
    assert {t.name for t in GCP_TOOLS} == EXPECTED_TOOLS
    assert fake_tool_names() == EXPECTED_TOOLS


def fake_tool_names():
    f = build_fake()
    try:
        return set(f.tool_names())
    finally:
        f.close()


async def test_logging_list_entries(fake):
    res = await fake.call(
        "gcp_logging_list_entries",
        {"project": "demo", "filter": "severity>=ERROR"},
    )
    assert res["project"] == "demo"
    assert res["filter"] == "severity>=ERROR"
    assert res["count"] == 1
    assert res["next_page_token"] == "next-1"
    entry = res["entries"][0]
    assert entry["insert_id"] == "abc123"
    assert entry["severity"] == "ERROR"
    assert entry["resource_type"] == "cloud_run_revision"
    assert "boom" in entry["payload"]


async def test_monitoring_list_timeseries(fake):
    res = await fake.call(
        "gcp_monitoring_list_timeseries",
        {"project": "demo", "filter": 'metric.type="x"'},
    )
    assert res["project"] == "demo"
    assert res["count"] == 1
    ts = res["time_series"][0]
    assert ts["metric"]["type"].endswith("cpu/utilization")
    assert ts["value_type"] == "DOUBLE"
    assert ts["point_count"] == 1
    assert ts["points"][0]["value"] == {"doubleValue": 0.42}


async def test_monitoring_list_alert_policies(fake):
    res = await fake.call("gcp_monitoring_list_alert_policies", {"project": "demo"})
    assert res["count"] == 1
    pol = res["alert_policies"][0]
    assert pol["display_name"] == "High CPU"
    assert pol["enabled"] is True
    assert pol["combiner"] == "OR"
    assert pol["conditions"][0]["display_name"] == "CPU > 80%"
    assert pol["notification_channels"] == ["projects/demo/notificationChannels/9"]


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
        tool = get_tool("gcp_logging_list_entries")
        res = await tool.handler(fake.client, {"project": "demo"})
        assert res["count"] == 1
    finally:
        fake.teardown()
