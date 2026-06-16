"""Integration test: the AWS MCP tools against the fake backend.

Exercises representative tools through build_fake() with no network, no
boto3, and no AWS credentials, asserting shape-faithful responses and the
registry's declared (reduced) tool set. CloudWatch metrics/alarms + Logs
now live in the dedicated `cloudwatch` connector and must NOT appear here.
Follows the Rootly/GCP reference (FR-012).
"""
from __future__ import annotations

import pytest

from opsrag.mcp.aws import build_fake, get_tool
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
    assert set(fake.tool_names()) == set(REGISTRY["aws"].tool_names)


def test_cloudwatch_and_logs_tools_absent(fake) -> None:
    names = set(fake.tool_names())
    for moved in (
        "aws_cloudwatch_get_metric_data",
        "aws_cloudwatch_describe_alarms",
        "aws_logs_filter_events",
        "aws_logs_insights_query",
    ):
        assert moved not in names


@pytest.mark.asyncio
async def test_describe_ec2_instances(fake) -> None:
    res = await fake.call("aws_describe_ec2_instances", {})
    assert res["count"] == 1
    assert res["instances"][0]["id"] == "i-0abc123"


@pytest.mark.asyncio
async def test_list_eks_clusters(fake) -> None:
    res = await fake.call("aws_list_eks_clusters", {})
    assert res["count"] == 2
    assert "prod-eks" in res["clusters"]


@pytest.mark.asyncio
async def test_s3_list_buckets(fake) -> None:
    res = await fake.call("aws_s3_list_buckets", {})
    assert res["count"] == 2
    assert res["owner"] == "acme"


@pytest.mark.asyncio
async def test_read_generic_allows_describe(fake) -> None:
    res = await fake.call(
        "aws_read",
        {"service": "rds", "operation": "DescribeDBInstances", "params": {}},
    )
    assert res["result"]["DBInstances"][0]["DBInstanceIdentifier"] == "prod-pg"


@pytest.mark.asyncio
async def test_handler_direct_invocation_pattern() -> None:
    # Exercise the get_tool + tool.handler(client, args) path explicitly.
    fake = build_fake()
    try:
        tool = get_tool("aws_list_eks_clusters")
        res = await tool.handler(fake.client, {})
        assert res["count"] == 2
    finally:
        fake.teardown()


@pytest.mark.asyncio
async def test_unknown_tool_raises(fake) -> None:
    with pytest.raises(KeyError):
        await fake.call("aws_does_not_exist", {})
