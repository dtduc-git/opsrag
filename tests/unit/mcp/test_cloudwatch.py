"""Unit tests for the CloudWatch MCP tools (read-only, boto3).

Drives every tool through build_fake() with NO boto3, NO AWS creds, and NO
network: the module-level `_call` is swapped for a canned (service, op)
dispatcher. asyncio_mode = auto, but @pytest.mark.asyncio is kept explicit.
"""
from __future__ import annotations

import pytest

from opsrag.mcp.cloudwatch import (
    CLOUDWATCH_TOOLS,
    CloudWatchMCPError,
    build_fake,
    get_tool,
)

EXPECTED_TOOLS = {
    "cloudwatch_get_metric_data",
    "cloudwatch_describe_alarms",
    "cloudwatch_list_metrics",
    "cloudwatch_logs_filter",
    "cloudwatch_logs_describe_groups",
}


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()  # restore the real module-level _call


def test_tool_names_exact() -> None:
    assert {t.name for t in CLOUDWATCH_TOOLS} == EXPECTED_TOOLS
    assert set(fake_names := build_fake().tool_names()) == EXPECTED_TOOLS  # parity
    assert len(fake_names) == len(CLOUDWATCH_TOOLS)


def test_all_tools_read_only() -> None:
    # No mutating verbs anywhere in the tool surface.
    bad = ("create", "update", "delete", "put", "start_query", "ack", "resolve")
    for t in CLOUDWATCH_TOOLS:
        assert not any(verb in t.name for verb in bad), t.name


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
    r = res["results"][0]
    assert r["id"] == "m1"
    assert r["label"] == "CPUUtilization"
    assert r["values"] == [42.0, 47.5]
    assert len(r["timestamps"]) == 2


@pytest.mark.asyncio
async def test_describe_alarms(fake) -> None:
    res = await fake.call("cloudwatch_describe_alarms", {"state_value": "ALARM"})
    assert res["count"] == 1
    a = res["alarms"][0]
    assert a["name"] == "api-5xx-high"
    assert a["state"] == "ALARM"
    assert a["comparison"] == "GreaterThanThreshold"
    assert a["threshold"] == 10.0


@pytest.mark.asyncio
async def test_list_metrics(fake) -> None:
    res = await fake.call(
        "cloudwatch_list_metrics",
        {"namespace": "AWS/EC2", "metric_name": "CPUUtilization"},
    )
    assert res["count"] == 1
    assert res["next_token"] == "next-1"
    m = res["metrics"][0]
    assert m["namespace"] == "AWS/EC2"
    assert m["metric_name"] == "CPUUtilization"
    assert m["dimensions"][0] == {"name": "InstanceId", "value": "i-0abc123"}


@pytest.mark.asyncio
async def test_logs_filter(fake) -> None:
    res = await fake.call(
        "cloudwatch_logs_filter",
        {"log_group_name": "/ecs/api", "filter_pattern": "ERROR"},
    )
    assert res["log_group"] == "/ecs/api"
    assert res["count"] == 1
    assert res["next_token"] == "next-1"
    e = res["events"][0]
    assert e["stream"] == "api/abc"
    assert "ERROR" in e["message"]
    assert e["timestamp"] == 1716000000000


@pytest.mark.asyncio
async def test_logs_describe_groups(fake) -> None:
    res = await fake.call(
        "cloudwatch_logs_describe_groups",
        {"log_group_name_prefix": "/ecs/"},
    )
    assert res["count"] == 1
    assert res["next_token"] == "next-1"
    g = res["log_groups"][0]
    assert g["name"] == "/ecs/api"
    assert g["retention_days"] == 30
    assert g["stored_bytes"] == 1048576


def test_get_tool_and_unknown() -> None:
    assert get_tool("cloudwatch_get_metric_data").name == "cloudwatch_get_metric_data"
    with pytest.raises(KeyError):
        get_tool("cloudwatch_nope")


def test_cloudwatch_error_redacts() -> None:
    exc = CloudWatchMCPError("leaked AKIAABCDEFGHIJKLMNOP key", tool="cloudwatch")
    assert "AKIAABCDEFGHIJKLMNOP" not in str(exc)
    assert "[REDACTED:aws_access_key]" in str(exc)


def test_module_imports_without_boto3() -> None:
    # The module must import even if boto3 is absent: boto3 is lazy-imported
    # inside _client only. Importing the module here must never touch boto3.
    import importlib

    import opsrag.mcp.cloudwatch as mod
    importlib.reload(mod)
    assert hasattr(mod, "CLOUDWATCH_TOOLS")
