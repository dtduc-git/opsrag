"""Unit tests for the AWS MCP tools (read-only, boto3).

Drives every tool through build_fake() with NO boto3, NO AWS creds, and NO
network: the module-level `_call` is swapped for a canned (service, op)
dispatcher. asyncio_mode = auto, but @pytest.mark.asyncio is kept explicit.
"""
from __future__ import annotations

import pytest

from opsrag.mcp.aws import AWS_TOOLS, AWSMCPError, build_fake, get_tool


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()  # restore the real module-level _call


def test_tool_names_exact() -> None:
    expected = {
        "aws_describe_ec2_instances",
        "aws_list_eks_clusters",
        "aws_describe_eks_cluster",
        "aws_list_ecs_services",
        "aws_cloudwatch_get_metric_data",
        "aws_cloudwatch_describe_alarms",
        "aws_logs_filter_events",
        "aws_logs_insights_query",
        "aws_s3_list_buckets",
        "aws_cost_and_usage",
        "aws_read",
    }
    assert {t.name for t in AWS_TOOLS} == expected
    assert set(fake_names := build_fake().tool_names()) == expected  # build_fake parity
    assert len(fake_names) == len(AWS_TOOLS)


@pytest.mark.asyncio
async def test_describe_ec2_instances(fake) -> None:
    res = await fake.call("aws_describe_ec2_instances", {})
    assert res["count"] == 1
    inst = res["instances"][0]
    assert inst["id"] == "i-0abc123"
    assert inst["type"] == "m5.large"
    assert inst["state"] == "running"
    assert inst["private_ip"] == "10.0.1.5"
    assert inst["name"] == "web-1"
    assert inst["tags"]["env"] == "prod"


@pytest.mark.asyncio
async def test_list_eks_clusters(fake) -> None:
    res = await fake.call("aws_list_eks_clusters", {})
    assert res["count"] == 2
    assert "prod-eks" in res["clusters"]


@pytest.mark.asyncio
async def test_describe_eks_cluster(fake) -> None:
    res = await fake.call("aws_describe_eks_cluster", {"name": "prod-eks"})
    assert res["name"] == "prod-eks"
    assert res["status"] == "ACTIVE"
    assert res["version"] == "1.30"
    assert res["vpc"] == "vpc-0abc"
    assert res["tags"]["env"] == "prod"


@pytest.mark.asyncio
async def test_list_ecs_services(fake) -> None:
    res = await fake.call("aws_list_ecs_services", {"cluster": "prod"})
    assert res["cluster"] == "prod"
    assert res["count"] == 1
    svc = res["services"][0]
    assert svc["name"] == "api"
    assert svc["status"] == "ACTIVE"
    assert svc["running"] == 3
    assert svc["launch_type"] == "FARGATE"


@pytest.mark.asyncio
async def test_cloudwatch_get_metric_data(fake) -> None:
    res = await fake.call(
        "aws_cloudwatch_get_metric_data",
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
async def test_cloudwatch_describe_alarms(fake) -> None:
    res = await fake.call("aws_cloudwatch_describe_alarms", {"state_value": "ALARM"})
    assert res["count"] == 1
    a = res["alarms"][0]
    assert a["name"] == "api-5xx-high"
    assert a["state"] == "ALARM"
    assert a["comparison"] == "GreaterThanThreshold"
    assert a["threshold"] == 10.0


@pytest.mark.asyncio
async def test_logs_filter_events(fake) -> None:
    res = await fake.call(
        "aws_logs_filter_events",
        {"log_group_name": "/ecs/api", "filter_pattern": "ERROR"},
    )
    assert res["log_group"] == "/ecs/api"
    assert res["count"] == 1
    e = res["events"][0]
    assert e["stream"] == "api/abc"
    assert "ERROR" in e["message"]
    assert e["timestamp"] == 1716000000000


@pytest.mark.asyncio
async def test_logs_insights_query(fake) -> None:
    res = await fake.call(
        "aws_logs_insights_query",
        {
            "query_string": "fields @timestamp, @message | limit 20",
            "start_time": 1716000000,
            "end_time": 1716003600,
            "log_group_names": ["/ecs/api"],
        },
    )
    assert res["query_id"] == "q-12345"
    assert res["status"] == "Complete"
    assert res["count"] == 1
    row = res["rows"][0]
    assert row["@message"] == "ERROR boom"
    assert res["statistics"]["records_matched"] == 1.0


@pytest.mark.asyncio
async def test_s3_list_buckets(fake) -> None:
    res = await fake.call("aws_s3_list_buckets", {})
    assert res["count"] == 2
    names = [b["name"] for b in res["buckets"]]
    assert "acme-prod-assets" in names
    assert res["owner"] == "acme"


@pytest.mark.asyncio
async def test_cost_and_usage(fake) -> None:
    res = await fake.call(
        "aws_cost_and_usage",
        {"start": "2026-05-01", "end": "2026-05-02", "granularity": "DAILY"},
    )
    assert res["granularity"] == "DAILY"
    assert res["count"] == 1
    r = res["results"][0]
    assert r["start"] == "2026-05-01"
    assert r["total"]["UnblendedCost"]["amount"] == "123.45"
    assert r["groups"][0]["keys"] == ["AmazonEC2"]


@pytest.mark.asyncio
async def test_read_generic_allows_describe(fake) -> None:
    res = await fake.call(
        "aws_read",
        {"service": "rds", "operation": "DescribeDBInstances", "params": {}},
    )
    assert res["service"] == "rds"
    assert res["operation"] == "DescribeDBInstances"
    # ResponseMetadata stripped from the boto3 response.
    assert "ResponseMetadata" not in res["result"]
    assert res["result"]["DBInstances"][0]["DBInstanceIdentifier"] == "prod-pg"


@pytest.mark.asyncio
async def test_read_generic_rejects_mutating_op(fake) -> None:
    for bad in ("DeleteDBInstance", "CreateBucket", "TerminateInstances", "PutObject"):
        with pytest.raises(AWSMCPError):
            await fake.call("aws_read", {"service": "rds", "operation": bad, "params": {}})


def test_get_tool_and_unknown() -> None:
    assert get_tool("aws_read").name == "aws_read"
    with pytest.raises(KeyError):
        get_tool("aws_nope")


def test_module_imports_without_boto3() -> None:
    # The module must import even if boto3 is absent: boto3 is lazy-imported
    # inside _client only. Importing the module here must never touch boto3.
    import importlib

    import opsrag.mcp.aws as mod
    importlib.reload(mod)
    assert hasattr(mod, "AWS_TOOLS")
