"""AWS MCP-style tools for OpsRAG (read-only, boto3).

Read-only async tools over the AWS API via boto3. boto3 is LAZY-imported
inside ``_client`` so this module imports fine without boto3 installed; the
offline fake swaps a single module-level ``_call`` so tests need no boto3 and
no AWS credentials.

## Auth

Uses the boto3 default credential chain (env vars, shared config/credentials,
IRSA / IAM-role-for-service-account, instance/container metadata). Region is
read from ``AWS_REGION`` / ``AWS_DEFAULT_REGION`` (or per-call ``region``);
``AWS_PROFILE`` selects a named profile.

## Read-only enforcement

Every handler issues a Describe / List / Get-style call (plus the read-only
CloudWatch Logs Insights start/get-query and Cost Explorer query). The generic
``aws_read`` escape hatch ENFORCES read-only by rejecting any operation whose
name does not start with an allow-listed read verb.

## Tool list (11 read-only)

| Tool                              | boto3 call(s)                              |
|-----------------------------------|--------------------------------------------|
| `aws_describe_ec2_instances`      | ec2.describe_instances                     |
| `aws_list_eks_clusters`           | eks.list_clusters                          |
| `aws_describe_eks_cluster`        | eks.describe_cluster                       |
| `aws_list_ecs_services`           | ecs.list_services + describe_services      |
| `aws_cloudwatch_get_metric_data`  | cloudwatch.get_metric_data                 |
| `aws_cloudwatch_describe_alarms`  | cloudwatch.describe_alarms                 |
| `aws_logs_filter_events`          | logs.filter_log_events                     |
| `aws_logs_insights_query`         | logs.start_query + get_query_results       |
| `aws_s3_list_buckets`             | s3.list_buckets                            |
| `aws_cost_and_usage`              | ce.get_cost_and_usage                      |
| `aws_read` (generic escape hatch) | <service>.<read-only op>                   |
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

from opsrag.mcp.gitlab import MCPTool

_log = logging.getLogger("opsrag.mcp.aws")

_DEFAULT_LIMIT = 25
_MAX_LIMIT = 100
_RESULT_TRUNCATE_CHARS = 8000
_INSIGHTS_POLL_TRIES = 5
_INSIGHTS_POLL_SLEEP_S = 1.0

# Read-only verb allow-list for the generic `aws_read` escape hatch. Any
# operation whose PascalCase name does not begin with one of these is rejected.
_READ_ONLY_PREFIXES = (
    "Describe",
    "List",
    "Get",
    "Lookup",
    "Search",
    "Scan",
    "BatchGet",
    "View",
    "Query",
    "Estimate",
)

# Redact obvious secrets that can show up in log events / tags / error text.
_REDACT_PATTERNS = [
    (re.compile(r"\bAKIA[A-Z0-9]{16}\b"), "[REDACTED:aws_access_key]"),
    (re.compile(r"\bASIA[A-Z0-9]{16}\b"), "[REDACTED:aws_session_key]"),
    (re.compile(r"\bxoxb-[A-Za-z0-9-]{20,}"), "[REDACTED:slack_bot_token]"),
    (re.compile(r"\bghp_[A-Za-z0-9]{30,}"), "[REDACTED:github_token]"),
    (re.compile(r"\bglpat-[A-Za-z0-9_]{20,}"), "[REDACTED:gitlab_token]"),
    (re.compile(r"\bAIza[A-Za-z0-9_-]{35}"), "[REDACTED:google_api_key]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]+"), "[REDACTED:jwt]"),
]


class AWSMCPError(Exception):
    """Raised on AWS errors or read-only-policy violations.

    ``body`` is always redaction-cleaned so an exception message can never
    re-expose a credential it tripped over.
    """

    def __init__(self, message: str, *, tool: str | None = None):
        self.tool = tool
        self.body = _redact(str(message))[:500]
        super().__init__(f"[{tool or 'aws'}] {self.body}")


def _redact(text: str) -> str:
    if not text:
        return ""
    for pat, repl in _REDACT_PATTERNS:
        text = pat.sub(repl, text)
    return text


def _truncate(text: str | None, limit: int = _RESULT_TRUNCATE_CHARS) -> str:
    if not text:
        return ""
    text = _redact(str(text))
    if len(text) > limit:
        return text[:limit] + " ...[truncated]"
    return text


def _clamp(n: int | None, *, default: int = _DEFAULT_LIMIT, max: int = _MAX_LIMIT) -> int:
    """Cap a caller-supplied count into [1, max], defaulting when unset."""
    import builtins
    if n is None:
        return default
    try:
        return builtins.max(1, builtins.min(int(n), max))
    except (TypeError, ValueError):
        return default


def _region(args: dict | None = None) -> str | None:
    if args and args.get("region"):
        return str(args["region"])
    return (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or None
    )


def _client(service: str, *, region: str | None = None):
    """LAZY-import boto3 and build a client for ``service``.

    boto3 is imported INSIDE this function so the module imports cleanly
    without boto3 installed. Credentials come from the boto3 default chain.
    """
    try:
        import boto3  # noqa: PLC0415 -- intentional lazy import
    except ImportError as exc:  # pragma: no cover - exercised only without boto3
        raise AWSMCPError(
            "boto3 is not installed. Install boto3 to use the AWS MCP tools.",
            tool=service,
        ) from exc
    kwargs: dict[str, Any] = {}
    reg = region or _region()
    if reg:
        kwargs["region_name"] = reg
    profile = os.environ.get("AWS_PROFILE")
    if profile:
        session = boto3.Session(profile_name=profile)
        return session.client(service, **kwargs)
    return boto3.client(service, **kwargs)


async def _call(service: str, op_name: str, *, region: str | None = None, **params) -> Any:
    """Single choke-point for every boto3 call.

    Runs the (blocking) boto3 client method in a thread so the event loop is
    not blocked. ``build_fake`` swaps THIS function with a canned dispatcher,
    so tests need no boto3 / network. ``op_name`` is the boto3 snake_case
    method name (e.g. ``describe_instances``).
    """
    def _do() -> Any:
        try:
            client = _client(service, region=region)
            method = getattr(client, op_name)
            return method(**params)
        except AWSMCPError:
            raise
        except Exception as exc:  # botocore ClientError, EndpointError, etc.
            raise AWSMCPError(f"{service}.{op_name} failed: {exc}", tool=service) from exc

    return await asyncio.to_thread(_do)


# --- helpers --------------------------------------------------------


def _pascal_to_snake(op: str) -> str:
    """`DescribeInstances` -> `describe_instances`; handles acronyms like
    `ListEC2Instances` reasonably (`list_ec2_instances`)."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", op)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


# --- handlers -------------------------------------------------------


async def _h_describe_ec2_instances(_unused, args: dict) -> Any:
    """ec2.describe_instances -- trimmed to id/type/state/private_ip/tags."""
    params: dict[str, Any] = {"MaxResults": _clamp(args.get("limit"), default=25, max=1000)}
    if args.get("instance_ids"):
        params["InstanceIds"] = list(args["instance_ids"])
        params.pop("MaxResults", None)  # InstanceIds + MaxResults conflict
    if args.get("filters"):
        params["Filters"] = args["filters"]
    resp = await _call("ec2", "describe_instances", region=_region(args), **params)
    out = []
    for res in resp.get("Reservations") or []:
        for inst in res.get("Instances") or []:
            tags = {t.get("Key"): t.get("Value") for t in (inst.get("Tags") or [])}
            out.append({
                "id": inst.get("InstanceId"),
                "type": inst.get("InstanceType"),
                "state": (inst.get("State") or {}).get("Name"),
                "private_ip": inst.get("PrivateIpAddress"),
                "az": (inst.get("Placement") or {}).get("AvailabilityZone"),
                "name": tags.get("Name"),
                "tags": tags,
            })
    return {"count": len(out), "instances": out}


async def _h_list_eks_clusters(_unused, args: dict) -> Any:
    """eks.list_clusters."""
    resp = await _call(
        "eks", "list_clusters",
        region=_region(args),
        maxResults=_clamp(args.get("limit"), default=100, max=100),
    )
    clusters = list(resp.get("clusters") or [])
    return {"count": len(clusters), "clusters": clusters}


async def _h_describe_eks_cluster(_unused, args: dict) -> Any:
    """eks.describe_cluster(name=)."""
    name = args["name"]
    resp = await _call("eks", "describe_cluster", region=_region(args), name=name)
    c = resp.get("cluster") or {}
    return {
        "name": c.get("name"),
        "status": c.get("status"),
        "version": c.get("version"),
        "endpoint": c.get("endpoint"),
        "arn": c.get("arn"),
        "platform_version": c.get("platformVersion"),
        "created_at": str(c.get("createdAt")) if c.get("createdAt") else None,
        "vpc": (c.get("resourcesVpcConfig") or {}).get("vpcId"),
        "tags": c.get("tags") or {},
    }


async def _h_list_ecs_services(_unused, args: dict) -> Any:
    """ecs.list_services(cluster=) + ecs.describe_services."""
    cluster = args["cluster"]
    region = _region(args)
    listed = await _call(
        "ecs", "list_services",
        region=region, cluster=cluster,
        maxResults=_clamp(args.get("limit"), default=10, max=10),
    )
    arns = list(listed.get("serviceArns") or [])
    if not arns:
        return {"cluster": cluster, "count": 0, "services": []}
    desc = await _call(
        "ecs", "describe_services",
        region=region, cluster=cluster, services=arns,
    )
    out = []
    for s in desc.get("services") or []:
        out.append({
            "name": s.get("serviceName"),
            "status": s.get("status"),
            "desired": s.get("desiredCount"),
            "running": s.get("runningCount"),
            "pending": s.get("pendingCount"),
            "launch_type": s.get("launchType"),
            "task_definition": s.get("taskDefinition"),
        })
    return {"cluster": cluster, "count": len(out), "services": out}


async def _h_cloudwatch_get_metric_data(_unused, args: dict) -> Any:
    """cloudwatch.get_metric_data(MetricDataQueries=, StartTime=, EndTime=)."""
    queries = args["metric_data_queries"]
    params: dict[str, Any] = {
        "MetricDataQueries": queries,
        "StartTime": args["start_time"],
        "EndTime": args["end_time"],
    }
    if args.get("scan_by"):
        params["ScanBy"] = args["scan_by"]
    resp = await _call("cloudwatch", "get_metric_data", region=_region(args), **params)
    out = []
    for r in resp.get("MetricDataResults") or []:
        out.append({
            "id": r.get("Id"),
            "label": r.get("Label"),
            "status": r.get("StatusCode"),
            "timestamps": [str(t) for t in (r.get("Timestamps") or [])],
            "values": r.get("Values") or [],
        })
    return {"count": len(out), "results": out}


async def _h_cloudwatch_describe_alarms(_unused, args: dict) -> Any:
    """cloudwatch.describe_alarms(StateValue=)."""
    params: dict[str, Any] = {"MaxRecords": _clamp(args.get("limit"), default=50, max=100)}
    if args.get("state_value"):
        params["StateValue"] = args["state_value"]
    if args.get("alarm_names"):
        params["AlarmNames"] = list(args["alarm_names"])
    if args.get("alarm_name_prefix"):
        params["AlarmNamePrefix"] = args["alarm_name_prefix"]
    resp = await _call("cloudwatch", "describe_alarms", region=_region(args), **params)
    out = []
    for a in resp.get("MetricAlarms") or []:
        out.append({
            "name": a.get("AlarmName"),
            "state": a.get("StateValue"),
            "reason": _truncate(a.get("StateReason"), 500),
            "metric": a.get("MetricName"),
            "namespace": a.get("Namespace"),
            "comparison": a.get("ComparisonOperator"),
            "threshold": a.get("Threshold"),
        })
    return {"count": len(out), "alarms": out}


async def _h_logs_filter_events(_unused, args: dict) -> Any:
    """logs.filter_log_events(logGroupName=, filterPattern=, startTime=, limit=)."""
    params: dict[str, Any] = {
        "logGroupName": args["log_group_name"],
        "limit": _clamp(args.get("limit"), default=25, max=100),
    }
    if args.get("filter_pattern"):
        params["filterPattern"] = args["filter_pattern"]
    if args.get("start_time") is not None:
        params["startTime"] = int(args["start_time"])
    if args.get("end_time") is not None:
        params["endTime"] = int(args["end_time"])
    if args.get("log_stream_names"):
        params["logStreamNames"] = list(args["log_stream_names"])
    resp = await _call("logs", "filter_log_events", region=_region(args), **params)
    out = []
    for e in resp.get("events") or []:
        out.append({
            "timestamp": e.get("timestamp"),
            "ingestion_time": e.get("ingestionTime"),
            "stream": e.get("logStreamName"),
            "message": _truncate(e.get("message"), 2000),
        })
    return {
        "log_group": args["log_group_name"],
        "count": len(out),
        "events": out,
    }


async def _h_logs_insights_query(_unused, args: dict) -> Any:
    """logs.start_query(...) then poll logs.get_query_results(queryId=)."""
    region = _region(args)
    start_params: dict[str, Any] = {
        "queryString": args["query_string"],
        "startTime": int(args["start_time"]),
        "endTime": int(args["end_time"]),
        "limit": _clamp(args.get("limit"), default=50, max=1000),
    }
    groups = args.get("log_group_names")
    if groups:
        start_params["logGroupNames"] = list(groups)
    elif args.get("log_group_name"):
        start_params["logGroupName"] = args["log_group_name"]
    started = await _call("logs", "start_query", region=region, **start_params)
    query_id = started.get("queryId")

    results: dict[str, Any] = {}
    status = "Scheduled"
    for _ in range(_INSIGHTS_POLL_TRIES):
        results = await _call("logs", "get_query_results", region=region, queryId=query_id)
        status = results.get("status") or "Unknown"
        if status in ("Complete", "Failed", "Cancelled", "Timeout"):
            break
        await asyncio.sleep(_INSIGHTS_POLL_SLEEP_S)

    rows = []
    for row in results.get("results") or []:
        rows.append({c.get("field"): _truncate(c.get("value"), 2000) for c in row})
    stats = results.get("statistics") or {}
    return {
        "query_id": query_id,
        "status": status,
        "count": len(rows),
        "rows": rows,
        "statistics": {
            "records_matched": stats.get("recordsMatched"),
            "records_scanned": stats.get("recordsScanned"),
            "bytes_scanned": stats.get("bytesScanned"),
        },
    }


async def _h_s3_list_buckets(_unused, args: dict) -> Any:
    """s3.list_buckets."""
    resp = await _call("s3", "list_buckets", region=_region(args))
    out = [
        {
            "name": b.get("Name"),
            "created": str(b.get("CreationDate")) if b.get("CreationDate") else None,
        }
        for b in (resp.get("Buckets") or [])
    ]
    owner = (resp.get("Owner") or {}).get("DisplayName")
    return {"count": len(out), "owner": owner, "buckets": out}


async def _h_cost_and_usage(_unused, args: dict) -> Any:
    """ce.get_cost_and_usage(TimePeriod=, Granularity=, Metrics=).

    NOTE: Cost Explorer API calls are BILLED ($0.01/request). Default window
    is kept small (caller supplies start/end as YYYY-MM-DD).
    """
    params: dict[str, Any] = {
        "TimePeriod": {"Start": args["start"], "End": args["end"]},
        "Granularity": args.get("granularity") or "DAILY",
        "Metrics": args.get("metrics") or ["UnblendedCost"],
    }
    if args.get("group_by"):
        params["GroupBy"] = args["group_by"]
    if args.get("filter"):
        params["Filter"] = args["filter"]
    resp = await _call("ce", "get_cost_and_usage", region=_region(args), **params)
    out = []
    for r in resp.get("ResultsByTime") or []:
        groups = []
        for g in r.get("Groups") or []:
            groups.append({
                "keys": g.get("Keys"),
                "metrics": {
                    k: {"amount": v.get("Amount"), "unit": v.get("Unit")}
                    for k, v in (g.get("Metrics") or {}).items()
                },
            })
        out.append({
            "start": (r.get("TimePeriod") or {}).get("Start"),
            "end": (r.get("TimePeriod") or {}).get("End"),
            "total": {
                k: {"amount": v.get("Amount"), "unit": v.get("Unit")}
                for k, v in (r.get("Total") or {}).items()
            },
            "groups": groups,
        })
    return {"granularity": params["Granularity"], "count": len(out), "results": out}


async def _h_read(_unused, args: dict) -> Any:
    """Generic read-only escape hatch: {service, operation, params}.

    ENFORCES read-only: the ``operation`` (PascalCase, as in the AWS API
    reference) must start with one of the allow-listed read verbs; anything
    else is rejected with ``AWSMCPError`` and NEVER reaches boto3.
    """
    service = args["service"]
    operation = args["operation"]
    params = args.get("params") or {}
    if not isinstance(params, dict):
        raise AWSMCPError("`params` must be an object/dict", tool="aws_read")
    if not operation or not operation[0].isupper() or not operation.startswith(_READ_ONLY_PREFIXES):
        raise AWSMCPError(
            f"operation {operation!r} is not read-only. Only operations starting "
            f"with one of {_READ_ONLY_PREFIXES} are permitted.",
            tool="aws_read",
        )
    op_name = _pascal_to_snake(operation)
    resp = await _call(service, op_name, region=_region(args), **params)
    # boto3 responses include a ResponseMetadata block we don't need.
    if isinstance(resp, dict):
        resp = {k: v for k, v in resp.items() if k != "ResponseMetadata"}
    return {"service": service, "operation": operation, "result": resp}


# --- tool registry --------------------------------------------------


AWS_TOOLS: list[MCPTool] = [
    MCPTool(
        name="aws_describe_ec2_instances",
        description=(
            "List/describe EC2 instances (trimmed to id, type, state, "
            "private_ip, az, Name tag, tags). Filter with `instance_ids` or "
            "EC2 `filters` (e.g. [{'Name':'tag:env','Values':['prod']}])."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "instance_ids": {"type": "array", "items": {"type": "string"}},
                "filters": {"type": "array", "items": {"type": "object"}},
                "region": {"type": "string"},
                "limit": {"type": "number"},
            },
        },
        handler=_h_describe_ec2_instances,
    ),
    MCPTool(
        name="aws_list_eks_clusters",
        description="List EKS cluster names in the region.",
        input_schema={
            "type": "object",
            "properties": {
                "region": {"type": "string"},
                "limit": {"type": "number"},
            },
        },
        handler=_h_list_eks_clusters,
    ),
    MCPTool(
        name="aws_describe_eks_cluster",
        description="Describe one EKS cluster by `name` (status, version, endpoint, vpc, tags).",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "region": {"type": "string"},
            },
            "required": ["name"],
        },
        handler=_h_describe_eks_cluster,
    ),
    MCPTool(
        name="aws_list_ecs_services",
        description=(
            "List ECS services in a `cluster` and describe them "
            "(status, desired/running/pending counts, launch type, task def)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "cluster": {"type": "string"},
                "region": {"type": "string"},
                "limit": {"type": "number"},
            },
            "required": ["cluster"],
        },
        handler=_h_list_ecs_services,
    ),
    MCPTool(
        name="aws_cloudwatch_get_metric_data",
        description=(
            "Fetch CloudWatch metric data via get_metric_data. Pass "
            "`metric_data_queries` (the MetricDataQueries list), `start_time` "
            "and `end_time` (datetimes). Returns per-query timestamps + values."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "metric_data_queries": {"type": "array", "items": {"type": "object"}},
                "start_time": {"description": "Start datetime"},
                "end_time": {"description": "End datetime"},
                "scan_by": {"type": "string", "enum": ["TimestampDescending", "TimestampAscending"]},
                "region": {"type": "string"},
            },
            "required": ["metric_data_queries", "start_time", "end_time"],
        },
        handler=_h_cloudwatch_get_metric_data,
    ),
    MCPTool(
        name="aws_cloudwatch_describe_alarms",
        description=(
            "List CloudWatch metric alarms. Filter by `state_value` "
            "(OK/ALARM/INSUFFICIENT_DATA), `alarm_names`, or `alarm_name_prefix`."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "state_value": {"type": "string", "enum": ["OK", "ALARM", "INSUFFICIENT_DATA"]},
                "alarm_names": {"type": "array", "items": {"type": "string"}},
                "alarm_name_prefix": {"type": "string"},
                "region": {"type": "string"},
                "limit": {"type": "number"},
            },
        },
        handler=_h_cloudwatch_describe_alarms,
    ),
    MCPTool(
        name="aws_logs_filter_events",
        description=(
            "filter_log_events on a CloudWatch Logs group. Pass "
            "`log_group_name` (required), optional `filter_pattern`, "
            "`start_time`/`end_time` (unix ms), `log_stream_names`."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "log_group_name": {"type": "string"},
                "filter_pattern": {"type": "string"},
                "start_time": {"type": "number", "description": "unix ms"},
                "end_time": {"type": "number", "description": "unix ms"},
                "log_stream_names": {"type": "array", "items": {"type": "string"}},
                "region": {"type": "string"},
                "limit": {"type": "number"},
            },
            "required": ["log_group_name"],
        },
        handler=_h_logs_filter_events,
    ),
    MCPTool(
        name="aws_logs_insights_query",
        description=(
            "Run a CloudWatch Logs Insights query: start_query then poll "
            "get_query_results until complete. Pass `query_string`, "
            "`start_time`/`end_time` (unix seconds), and either "
            "`log_group_names` (list) or `log_group_name`."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query_string": {"type": "string"},
                "start_time": {"type": "number", "description": "unix seconds"},
                "end_time": {"type": "number", "description": "unix seconds"},
                "log_group_names": {"type": "array", "items": {"type": "string"}},
                "log_group_name": {"type": "string"},
                "region": {"type": "string"},
                "limit": {"type": "number"},
            },
            "required": ["query_string", "start_time", "end_time"],
        },
        handler=_h_logs_insights_query,
    ),
    MCPTool(
        name="aws_s3_list_buckets",
        description="List all S3 buckets (name + creation date) for the account.",
        input_schema={
            "type": "object",
            "properties": {"region": {"type": "string"}},
        },
        handler=_h_s3_list_buckets,
    ),
    MCPTool(
        name="aws_cost_and_usage",
        description=(
            "Cost Explorer get_cost_and_usage. BILLED ($0.01/request) -- keep "
            "windows small. Pass `start`/`end` (YYYY-MM-DD), `granularity` "
            "(DAILY/MONTHLY/HOURLY), `metrics` (default ['UnblendedCost']), "
            "optional `group_by` / `filter`."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "start": {"type": "string", "description": "YYYY-MM-DD"},
                "end": {"type": "string", "description": "YYYY-MM-DD"},
                "granularity": {"type": "string", "enum": ["DAILY", "MONTHLY", "HOURLY"]},
                "metrics": {"type": "array", "items": {"type": "string"}},
                "group_by": {"type": "array", "items": {"type": "object"}},
                "filter": {"type": "object"},
                "region": {"type": "string"},
            },
            "required": ["start", "end"],
        },
        handler=_h_cost_and_usage,
    ),
    MCPTool(
        name="aws_read",
        description=(
            "Generic READ-ONLY AWS escape hatch. Call any boto3 read "
            "operation: `service` (e.g. 'rds'), `operation` (PascalCase as in "
            "the AWS API reference, e.g. 'DescribeDBInstances'), and `params` "
            "(the boto3 kwargs). ENFORCED read-only: only operations starting "
            "with Describe/List/Get/Lookup/Search/Scan/BatchGet/View/Query/"
            "Estimate are allowed; mutating calls are rejected."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "boto3 service name, e.g. 'rds', 'elbv2'"},
                "operation": {"type": "string", "description": "PascalCase API operation, e.g. 'DescribeDBInstances'"},
                "params": {"type": "object", "description": "boto3 kwargs for the operation"},
                "region": {"type": "string"},
            },
            "required": ["service", "operation"],
        },
        handler=_h_read,
    ),
]


def get_tool(name: str) -> MCPTool:
    for t in AWS_TOOLS:
        if t.name == name:
            return t
    raise KeyError(f"unknown aws tool: {name}")


# --- fake backend (FR-012; integration tests) ----------------------
#
# Every handler routes through the module-level `_call(service, op, **kw)`,
# which lazy-imports boto3. The offline fake swaps THAT single function for a
# canned dispatcher keyed by (service, op) returning responses shaped like the
# real botocore responses the handlers parse. No boto3, no AWS creds, no
# network. `build_fake()` returns client=None (handlers discard their first
# arg) plus a teardown that restores the real `_call`.


async def _fake_call(service: str, op_name: str, *, region: str | None = None, **params) -> Any:
    """Canned stand-in for `_call`, keyed by (service, op_name). Shapes mirror
    botocore responses for the operations the handlers parse."""
    key = (service, op_name)

    if key == ("ec2", "describe_instances"):
        return {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-0abc123",
                            "InstanceType": "m5.large",
                            "State": {"Name": "running"},
                            "PrivateIpAddress": "10.0.1.5",
                            "Placement": {"AvailabilityZone": "us-east-1a"},
                            "Tags": [
                                {"Key": "Name", "Value": "web-1"},
                                {"Key": "env", "Value": "prod"},
                            ],
                        }
                    ]
                }
            ],
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }
    if key == ("eks", "list_clusters"):
        return {"clusters": ["prod-eks", "staging-eks"]}
    if key == ("eks", "describe_cluster"):
        return {
            "cluster": {
                "name": params.get("name", "prod-eks"),
                "status": "ACTIVE",
                "version": "1.30",
                "endpoint": "https://abc.gr7.us-east-1.eks.amazonaws.com",
                "arn": "arn:aws:eks:us-east-1:123456789012:cluster/prod-eks",
                "platformVersion": "eks.5",
                "createdAt": "2026-01-01T00:00:00Z",
                "resourcesVpcConfig": {"vpcId": "vpc-0abc"},
                "tags": {"env": "prod"},
            }
        }
    if key == ("ecs", "list_services"):
        return {
            "serviceArns": [
                "arn:aws:ecs:us-east-1:123456789012:service/prod/api",
            ]
        }
    if key == ("ecs", "describe_services"):
        return {
            "services": [
                {
                    "serviceName": "api",
                    "status": "ACTIVE",
                    "desiredCount": 3,
                    "runningCount": 3,
                    "pendingCount": 0,
                    "launchType": "FARGATE",
                    "taskDefinition": "arn:aws:ecs:us-east-1:123456789012:task-definition/api:12",
                }
            ]
        }
    if key == ("cloudwatch", "get_metric_data"):
        return {
            "MetricDataResults": [
                {
                    "Id": "m1",
                    "Label": "CPUUtilization",
                    "StatusCode": "Complete",
                    "Timestamps": ["2026-05-20T00:00:00Z", "2026-05-20T00:05:00Z"],
                    "Values": [42.0, 47.5],
                }
            ]
        }
    if key == ("cloudwatch", "describe_alarms"):
        return {
            "MetricAlarms": [
                {
                    "AlarmName": "api-5xx-high",
                    "StateValue": "ALARM",
                    "StateReason": "Threshold crossed: 5 datapoints > 10.0",
                    "MetricName": "HTTPCode_Target_5XX_Count",
                    "Namespace": "AWS/ApplicationELB",
                    "ComparisonOperator": "GreaterThanThreshold",
                    "Threshold": 10.0,
                }
            ]
        }
    if key == ("logs", "filter_log_events"):
        return {
            "events": [
                {
                    "timestamp": 1716000000000,
                    "ingestionTime": 1716000001000,
                    "logStreamName": "api/abc",
                    "message": "ERROR something broke",
                }
            ]
        }
    if key == ("logs", "start_query"):
        return {"queryId": "q-12345"}
    if key == ("logs", "get_query_results"):
        return {
            "status": "Complete",
            "results": [
                [
                    {"field": "@timestamp", "value": "2026-05-20 00:00:00.000"},
                    {"field": "@message", "value": "ERROR boom"},
                ]
            ],
            "statistics": {
                "recordsMatched": 1.0,
                "recordsScanned": 1000.0,
                "bytesScanned": 204800.0,
            },
        }
    if key == ("s3", "list_buckets"):
        return {
            "Buckets": [
                {"Name": "acme-prod-assets", "CreationDate": "2025-01-01T00:00:00Z"},
                {"Name": "acme-logs", "CreationDate": "2025-02-01T00:00:00Z"},
            ],
            "Owner": {"DisplayName": "acme", "ID": "abc"},
        }
    if key == ("ce", "get_cost_and_usage"):
        return {
            "ResultsByTime": [
                {
                    "TimePeriod": {"Start": "2026-05-01", "End": "2026-05-02"},
                    "Total": {"UnblendedCost": {"Amount": "123.45", "Unit": "USD"}},
                    "Groups": [
                        {
                            "Keys": ["AmazonEC2"],
                            "Metrics": {"UnblendedCost": {"Amount": "100.00", "Unit": "USD"}},
                        }
                    ],
                }
            ]
        }
    # Generic aws_read path: return a shape-faithful response for the canned
    # operation the test drives (rds.describe_db_instances).
    if key == ("rds", "describe_db_instances"):
        return {
            "DBInstances": [
                {"DBInstanceIdentifier": "prod-pg", "DBInstanceStatus": "available", "Engine": "postgres"}
            ],
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }
    return {}


def build_fake():
    """Return a FakeMCP exposing the AWS tools wired to an offline backend.

    Needs NO boto3 / AWS creds / network: the module-level `_call` is swapped
    for a canned (service, op) dispatcher and restored by `teardown`."""
    import opsrag.mcp.aws as _mod
    from opsrag.mcp._fake import FakeMCP

    _orig_call = _mod._call
    _mod._call = _fake_call

    def _restore() -> None:
        _mod._call = _orig_call

    return FakeMCP(tools=list(AWS_TOOLS), client=None, teardown=_restore)
