"""Unit tests for the Stackdriver MCP tools against the offline fake backend.

Exercises every tool through build_fake() with NO network, NO google libs,
and NO GCP credentials, asserting the parsed shape from the canned data.
asyncio_mode = "auto" (see pyproject.toml) so no decorator is needed.
"""
from __future__ import annotations

import pytest

from opsrag.mcp.stackdriver import STACKDRIVER_TOOLS, build_fake, get_tool

EXPECTED_TOOLS = {
    "stackdriver_list_timeseries",
    "stackdriver_list_alert_policies",
    "stackdriver_list_log_entries",
}


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()  # restore real module-level _get / _post


def test_tool_set_matches_exactly():
    assert {t.name for t in STACKDRIVER_TOOLS} == EXPECTED_TOOLS
    assert fake_tool_names() == EXPECTED_TOOLS


def fake_tool_names():
    f = build_fake()
    try:
        return set(f.tool_names())
    finally:
        f.close()


async def test_list_timeseries(fake):
    res = await fake.call(
        "stackdriver_list_timeseries",
        {"project": "demo", "filter": 'metric.type="x"'},
    )
    assert res["project"] == "demo"
    assert res["count"] == 1
    ts = res["time_series"][0]
    assert ts["metric"]["type"].endswith("cpu/utilization")
    assert ts["value_type"] == "DOUBLE"
    assert ts["point_count"] == 1
    assert ts["points"][0]["value"] == {"doubleValue": 0.42}


async def test_list_alert_policies(fake):
    res = await fake.call("stackdriver_list_alert_policies", {"project": "demo"})
    assert res["count"] == 1
    pol = res["alert_policies"][0]
    assert pol["display_name"] == "High CPU"
    assert pol["enabled"] is True
    assert pol["combiner"] == "OR"
    assert pol["conditions"][0]["display_name"] == "CPU > 80%"
    assert pol["notification_channels"] == ["projects/demo/notificationChannels/9"]


async def test_list_log_entries(fake):
    res = await fake.call(
        "stackdriver_list_log_entries",
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


async def test_handler_direct_invocation_pattern():
    # Exercise the get_tool + tool.handler(client, args) path explicitly.
    fake = build_fake()
    try:
        tool = get_tool("stackdriver_list_log_entries")
        res = await tool.handler(fake.client, {"project": "demo"})
        assert res["count"] == 1
    finally:
        fake.teardown()
