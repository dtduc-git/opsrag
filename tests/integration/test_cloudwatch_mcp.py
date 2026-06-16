"""Integration test: the CloudWatch MCP tools against the fake backend.

Exercises every tool through build_fake() with no network, no boto3, and no
AWS credentials, asserting shape-faithful responses and the registry's
declared tool set. Follows the AWS/GCP reference (FR-012).
"""
from __future__ import annotations

import pytest

from opsrag.mcp.cloudwatch import build_fake, get_tool
from opsrag.mcp.registry import REGISTRY


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()  # restore the real module-level _call


def test_fake_exposes_registry_tool_set(fake) -> None:
    # The fake's tools must match exactly what the registry declares.
    assert set(fake.tool_names()) == set(REGISTRY["cloudwatch"].tool_names)


@pytest.mark.asyncio
async def test_get_metric_data(fake) -> None:
    res = await fake.call(
        "cloudwatch_get_metric_data",
        {
            "metric_data_queries": [{"Id": "m1"}],
            "start_time": "2026-05-20T00:00:00Z",
            "end_time": "2026-05-20T01:00:00Z",
        },
    )
    assert res["count"] == 1
    assert res["results"][0]["values"] == [42.0, 47.5]


@pytest.mark.asyncio
async def test_describe_alarms(fake) -> None:
    res = await fake.call("cloudwatch_describe_alarms", {})
    assert res["count"] == 1
    assert res["alarms"][0]["state"] == "ALARM"


@pytest.mark.asyncio
async def test_list_metrics(fake) -> None:
    res = await fake.call("cloudwatch_list_metrics", {"namespace": "AWS/EC2"})
    assert res["count"] == 1
    assert res["metrics"][0]["metric_name"] == "CPUUtilization"


@pytest.mark.asyncio
async def test_logs_filter(fake) -> None:
    res = await fake.call("cloudwatch_logs_filter", {"log_group_name": "/ecs/api"})
    assert res["count"] == 1
    assert "ERROR" in res["events"][0]["message"]


@pytest.mark.asyncio
async def test_logs_describe_groups(fake) -> None:
    res = await fake.call("cloudwatch_logs_describe_groups", {})
    assert res["count"] == 1
    assert res["log_groups"][0]["name"] == "/ecs/api"


@pytest.mark.asyncio
async def test_handler_direct_invocation_pattern() -> None:
    # Exercise the get_tool + tool.handler(client, args) path explicitly.
    fake = build_fake()
    try:
        tool = get_tool("cloudwatch_describe_alarms")
        res = await tool.handler(fake.client, {})
        assert res["count"] == 1
    finally:
        fake.teardown()


@pytest.mark.asyncio
async def test_unknown_tool_raises(fake) -> None:
    with pytest.raises(KeyError):
        await fake.call("cloudwatch_does_not_exist", {})
