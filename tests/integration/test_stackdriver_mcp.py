"""Integration test: the Stackdriver MCP tools against the fake backend.

Exercises every tool through build_fake() with no network, no google libs,
and no GCP credentials, asserting shape-faithful responses and the
registry's declared tool set. Follows the AWS/GCP/CloudWatch reference
(FR-012).
"""
from __future__ import annotations

import pytest

from opsrag.mcp.registry import REGISTRY
from opsrag.mcp.stackdriver import build_fake, get_tool


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()  # restore the real module-level _get / _post


def test_fake_exposes_registry_tool_set(fake) -> None:
    # The fake's tools must match exactly what the registry declares.
    assert set(fake.tool_names()) == set(REGISTRY["stackdriver"].tool_names)


@pytest.mark.asyncio
async def test_list_timeseries(fake) -> None:
    res = await fake.call(
        "stackdriver_list_timeseries",
        {"project": "demo", "filter": 'metric.type="x"'},
    )
    assert res["count"] == 1
    assert res["time_series"][0]["value_type"] == "DOUBLE"


@pytest.mark.asyncio
async def test_list_alert_policies(fake) -> None:
    res = await fake.call("stackdriver_list_alert_policies", {"project": "demo"})
    assert res["count"] == 1
    assert res["alert_policies"][0]["display_name"] == "High CPU"


@pytest.mark.asyncio
async def test_list_log_entries(fake) -> None:
    res = await fake.call(
        "stackdriver_list_log_entries",
        {"project": "demo", "filter": "severity>=ERROR"},
    )
    assert res["count"] == 1
    assert "boom" in res["entries"][0]["payload"]


@pytest.mark.asyncio
async def test_handler_direct_invocation_pattern() -> None:
    # Exercise the get_tool + tool.handler(client, args) path explicitly.
    fake = build_fake()
    try:
        tool = get_tool("stackdriver_list_alert_policies")
        res = await tool.handler(fake.client, {"project": "demo"})
        assert res["count"] == 1
    finally:
        fake.teardown()


@pytest.mark.asyncio
async def test_unknown_tool_raises(fake) -> None:
    with pytest.raises(KeyError):
        await fake.call("stackdriver_does_not_exist", {})
