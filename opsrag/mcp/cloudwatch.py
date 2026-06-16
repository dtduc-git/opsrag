"""CloudWatch MCP-style tools for OpsRAG (read-only, boto3).

A dedicated connector for Amazon CloudWatch (metrics + alarms) and
CloudWatch Logs, split out of the broader ``opsrag.mcp.aws`` connector so
observability tooling lives in one Observability-categorised place.

Read-only async tools over the CloudWatch / CloudWatch Logs APIs via
boto3. boto3 is LAZY-imported inside ``_client`` so this module imports
fine without boto3 installed; the offline fake swaps a single module-level
``_call`` so tests need no boto3 and no AWS credentials.

## Auth

Uses the boto3 default credential chain (env vars, shared config/credentials,
IRSA / IAM-role-for-service-account, instance/container metadata). Region is
read from ``AWS_REGION`` / ``AWS_DEFAULT_REGION`` (or per-call ``region``);
``AWS_PROFILE`` selects a named profile. The same credential chain the AWS
connector uses -- no separate config.

## Read-only enforcement

Every handler issues a Describe / List / Get / Filter-style call only. There
is no escape hatch and no mutating operation anywhere in this module.

## Tool list (5 read-only)

| Tool                              | boto3 call(s)                              |
|-----------------------------------|--------------------------------------------|
| `cloudwatch_get_metric_data`      | cloudwatch.get_metric_data                 |
| `cloudwatch_describe_alarms`      | cloudwatch.describe_alarms                 |
| `cloudwatch_list_metrics`         | cloudwatch.list_metrics                    |
| `cloudwatch_logs_filter`          | logs.filter_log_events                     |
| `cloudwatch_logs_describe_groups` | logs.describe_log_groups                   |
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

from opsrag.mcp.gitlab import MCPTool

_log = logging.getLogger("opsrag.mcp.cloudwatch")

_DEFAULT_LIMIT = 25
_MAX_LIMIT = 100
_RESULT_TRUNCATE_CHARS = 8000

# Redact obvious secrets that can show up in log events / alarm reasons / error text.
_REDACT_PATTERNS = [
    (re.compile(r"\bAKIA[A-Z0-9]{16}\b"), "[REDACTED:aws_access_key]"),
    (re.compile(r"\bASIA[A-Z0-9]{16}\b"), "[REDACTED:aws_session_key]"),
    (re.compile(r"\bxoxb-[A-Za-z0-9-]{20,}"), "[REDACTED:slack_bot_token]"),
    (re.compile(r"\bghp_[A-Za-z0-9]{30,}"), "[REDACTED:github_token]"),
    (re.compile(r"\bglpat-[A-Za-z0-9_]{20,}"), "[REDACTED:gitlab_token]"),
    (re.compile(r"\bAIza[A-Za-z0-9_-]{35}"), "[REDACTED:google_api_key]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]+"), "[REDACTED:jwt]"),
]


class CloudWatchMCPError(Exception):
    """Raised on CloudWatch errors.

    ``body`` is always redaction-cleaned so an exception message can never
    re-expose a credential it tripped over.
    """

    def __init__(self, message: str, *, tool: str | None = None):
        self.tool = tool
        self.body = _redact(str(message))[:500]
        super().__init__(f"[{tool or 'cloudwatch'}] {self.body}")


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
    """LAZY-import boto3 and build a client for ``service`` ('cloudwatch' or 'logs').

    boto3 is imported INSIDE this function so the module imports cleanly
    without boto3 installed. Credentials come from the boto3 default chain.
    """
    try:
        import boto3  # noqa: PLC0415 -- intentional lazy import
    except ImportError as exc:  # pragma: no cover - exercised only without boto3
        raise CloudWatchMCPError(
            "boto3 is not installed. Install boto3 to use the CloudWatch MCP tools.",
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
    method name (e.g. ``describe_alarms``). ``service`` is 'cloudwatch' or
    'logs'.
    """
    def _do() -> Any:
        try:
            client = _client(service, region=region)
            method = getattr(client, op_name)
            return method(**params)
        except CloudWatchMCPError:
            raise
        except Exception as exc:  # botocore ClientError, EndpointError, etc.
            raise CloudWatchMCPError(
                f"{service}.{op_name} failed: {exc}", tool=service
            ) from exc

    return await asyncio.to_thread(_do)


# --- handlers -------------------------------------------------------


async def _h_get_metric_data(_unused, args: dict) -> Any:
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


async def _h_describe_alarms(_unused, args: dict) -> Any:
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


async def _h_list_metrics(_unused, args: dict) -> Any:
    """cloudwatch.list_metrics(Namespace=, MetricName=, Dimensions=).

    Enumerates the metrics CloudWatch knows about so the agent can discover
    valid `namespace` / `metric_name` / dimensions before building a
    `get_metric_data` query."""
    params: dict[str, Any] = {}
    if args.get("namespace"):
        params["Namespace"] = args["namespace"]
    if args.get("metric_name"):
        params["MetricName"] = args["metric_name"]
    if args.get("dimensions"):
        params["Dimensions"] = list(args["dimensions"])
    if args.get("recently_active"):
        params["RecentlyActive"] = args["recently_active"]
    resp = await _call("cloudwatch", "list_metrics", region=_region(args), **params)
    limit = _clamp(args.get("limit"), default=_DEFAULT_LIMIT, max=_MAX_LIMIT)
    out = []
    for m in (resp.get("Metrics") or [])[:limit]:
        out.append({
            "namespace": m.get("Namespace"),
            "metric_name": m.get("MetricName"),
            "dimensions": [
                {"name": d.get("Name"), "value": d.get("Value")}
                for d in (m.get("Dimensions") or [])
            ],
        })
    return {
        "count": len(out),
        "next_token": resp.get("NextToken"),
        "metrics": out,
    }


async def _h_logs_filter(_unused, args: dict) -> Any:
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
        "next_token": resp.get("nextToken"),
        "events": out,
    }


async def _h_logs_describe_groups(_unused, args: dict) -> Any:
    """logs.describe_log_groups(logGroupNamePrefix=).

    Discovery for the Logs group namespace so the agent can find a valid
    `log_group_name` before calling `cloudwatch_logs_filter`."""
    params: dict[str, Any] = {"limit": _clamp(args.get("limit"), default=50, max=50)}
    if args.get("log_group_name_prefix"):
        params["logGroupNamePrefix"] = args["log_group_name_prefix"]
    if args.get("log_group_name_pattern"):
        params["logGroupNamePattern"] = args["log_group_name_pattern"]
    resp = await _call("logs", "describe_log_groups", region=_region(args), **params)
    out = []
    for g in resp.get("logGroups") or []:
        out.append({
            "name": g.get("logGroupName"),
            "created": g.get("creationTime"),
            "retention_days": g.get("retentionInDays"),
            "stored_bytes": g.get("storedBytes"),
            "arn": g.get("arn"),
        })
    return {
        "count": len(out),
        "next_token": resp.get("nextToken"),
        "log_groups": out,
    }


# --- tool registry --------------------------------------------------


CLOUDWATCH_TOOLS: list[MCPTool] = [
    MCPTool(
        name="cloudwatch_get_metric_data",
        description=(
            "Fetch CloudWatch metric data via get_metric_data. Pass "
            "`metric_data_queries` (the MetricDataQueries list), `start_time` "
            "and `end_time` (datetimes). Returns per-query timestamps + values. "
            "Use `cloudwatch_list_metrics` first to discover valid namespaces / "
            "metric names / dimensions."
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
        handler=_h_get_metric_data,
    ),
    MCPTool(
        name="cloudwatch_describe_alarms",
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
        handler=_h_describe_alarms,
    ),
    MCPTool(
        name="cloudwatch_list_metrics",
        description=(
            "List the CloudWatch metrics available (list_metrics). Filter by "
            "`namespace` (e.g. 'AWS/EC2'), `metric_name`, or `dimensions` "
            "(list of {Name, Value}). Set `recently_active`='PT3H' to only show "
            "metrics with data in the last 3h. Use to discover valid metric "
            "identifiers before `cloudwatch_get_metric_data`."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "namespace": {"type": "string", "description": "e.g. 'AWS/EC2', 'AWS/ApplicationELB'"},
                "metric_name": {"type": "string"},
                "dimensions": {"type": "array", "items": {"type": "object"}},
                "recently_active": {"type": "string", "enum": ["PT3H"]},
                "region": {"type": "string"},
                "limit": {"type": "number"},
            },
        },
        handler=_h_list_metrics,
    ),
    MCPTool(
        name="cloudwatch_logs_filter",
        description=(
            "filter_log_events on a CloudWatch Logs group. Pass "
            "`log_group_name` (required), optional `filter_pattern`, "
            "`start_time`/`end_time` (unix ms), `log_stream_names`. Use "
            "`cloudwatch_logs_describe_groups` first to find a valid log group."
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
        handler=_h_logs_filter,
    ),
    MCPTool(
        name="cloudwatch_logs_describe_groups",
        description=(
            "List CloudWatch Logs log groups (describe_log_groups). Filter by "
            "`log_group_name_prefix` or `log_group_name_pattern` (substring). "
            "Returns name, retention, stored bytes. Use to discover a valid "
            "`log_group_name` for `cloudwatch_logs_filter`."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "log_group_name_prefix": {"type": "string"},
                "log_group_name_pattern": {"type": "string", "description": "case-sensitive substring match"},
                "region": {"type": "string"},
                "limit": {"type": "number"},
            },
        },
        handler=_h_logs_describe_groups,
    ),
]


def get_tool(name: str) -> MCPTool:
    for t in CLOUDWATCH_TOOLS:
        if t.name == name:
            return t
    raise KeyError(f"unknown cloudwatch tool: {name}")


# --- fake backend (FR-012; integration tests) ----------------------
#
# Every handler routes through the module-level `_call(service, op, **kw)`,
# which lazy-imports boto3. The offline fake swaps THAT single function for a
# canned dispatcher keyed by (service, op) returning responses shaped like the
# real botocore responses the handlers parse. No boto3, no AWS creds, no
# network. `build_fake()` returns client=None (handlers discard their first
# arg) plus a teardown that restores the real `_call`. Services are
# "cloudwatch" and "logs".


async def _fake_call(service: str, op_name: str, *, region: str | None = None, **params) -> Any:
    """Canned stand-in for `_call`, keyed by (service, op_name). Shapes mirror
    botocore responses for the operations the handlers parse."""
    key = (service, op_name)

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
    if key == ("cloudwatch", "list_metrics"):
        return {
            "Metrics": [
                {
                    "Namespace": "AWS/EC2",
                    "MetricName": "CPUUtilization",
                    "Dimensions": [
                        {"Name": "InstanceId", "Value": "i-0abc123"},
                    ],
                }
            ],
            "NextToken": "next-1",
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
            ],
            "nextToken": "next-1",
        }
    if key == ("logs", "describe_log_groups"):
        return {
            "logGroups": [
                {
                    "logGroupName": "/ecs/api",
                    "creationTime": 1716000000000,
                    "retentionInDays": 30,
                    "storedBytes": 1048576,
                    "arn": "arn:aws:logs:us-east-1:123456789012:log-group:/ecs/api:*",
                }
            ],
            "nextToken": "next-1",
        }
    return {}


def build_fake():
    """Return a FakeMCP exposing the CloudWatch tools wired to an offline backend.

    Needs NO boto3 / AWS creds / network: the module-level `_call` is swapped
    for a canned (service, op) dispatcher and restored by `teardown`."""
    import opsrag.mcp.cloudwatch as _mod
    from opsrag.mcp._fake import FakeMCP

    _orig_call = _mod._call
    _mod._call = _fake_call

    def _restore() -> None:
        _mod._call = _orig_call

    return FakeMCP(tools=list(CLOUDWATCH_TOOLS), client=None, teardown=_restore)
