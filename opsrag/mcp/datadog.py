"""Datadog MCP-style tools for OpsRAG (Sub-sprint 4).

Read-only async tools over Datadog v1+v2 REST APIs. Reuses
`DD_API_KEY` + `DD_APP_KEY` + `DD_SITE` from env. The APP key needs
read scopes: `logs_read_data`, `apm_read`, `apm_service_catalog_read`,
`monitors_read`, `events_read`, `slo_read`.

## Read-only enforcement

Every tool issues `httpx.AsyncClient.get` or `.post` to a search-only
v2 endpoint (search APIs are POST in DD's v2 spec). No
`PUT`/`DELETE`/`PATCH` anywhere -- no monitor mutation, no log
deletion, no SLO edit.

## Tool list (8 read-only)

| Tool                       | Endpoint                                  |
|----------------------------|--------------------------------------------|
| `datadog_search_logs`      | GET `/api/v2/logs/events`                  |
| `datadog_search_spans`     | POST `/api/v2/spans/events/search`         |
| `datadog_get_trace`        | GET `/api/v2/spans/events?trace_id=`       |
| `datadog_list_services`    | GET `/api/v2/services/definitions`         |
| `datadog_list_monitors`    | GET `/api/v1/monitor`                      |
| `datadog_get_monitor`      | GET `/api/v1/monitor/<id>`                 |
| `datadog_list_events`      | GET `/api/v1/events`                       |
| `datadog_get_slo`          | GET `/api/v1/slo/<id>/history`             |
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC
from typing import Any

import httpx

from opsrag.mcp.gitlab import MCPTool

_log = logging.getLogger("opsrag.mcp.datadog")

DEFAULT_DD_SITE = "datadoghq.com"
_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_LIMIT = 25
_MAX_LIMIT = 100
_RESULT_TRUNCATE_CHARS = 32000

# Same redaction patterns as K8s pod logs -- log content can leak tokens.
_REDACT_PATTERNS = [
    (re.compile(r"\bxoxb-[A-Za-z0-9-]{20,}"), "[REDACTED:slack_bot_token]"),
    (re.compile(r"\bxoxp-[A-Za-z0-9-]{20,}"), "[REDACTED:slack_user_token]"),
    (re.compile(r"\bghp_[A-Za-z0-9]{30,}"), "[REDACTED:github_token]"),
    (re.compile(r"\bglpat-[A-Za-z0-9_]{20,}"), "[REDACTED:gitlab_token]"),
    (re.compile(r"\bAKIA[A-Z0-9]{16}\b"), "[REDACTED:aws_access_key]"),
    (re.compile(r"\bAIza[A-Za-z0-9_-]{35}"), "[REDACTED:google_api_key]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]+"), "[REDACTED:jwt]"),
    (re.compile(r"\brootly_[A-Za-z0-9_]{30,}"), "[REDACTED:rootly_token]"),
    (re.compile(r"\bddapp_[A-Za-z0-9_]{30,}"), "[REDACTED:dd_app_key]"),
]


def _redact(text: str) -> str:
    for pat, repl in _REDACT_PATTERNS:
        text = pat.sub(repl, text)
    return text


class DatadogMCPError(Exception):
    def __init__(self, status: int, body: str, *, tool: str | None = None):
        self.status = status
        self.body = body
        self.tool = tool
        super().__init__(f"[{tool or 'datadog'}] {status}: {body[:300]}")


@dataclass
class _Config:
    api_key: str
    app_key: str
    api_url: str


def _config() -> _Config:
    api_key = os.environ.get("DD_API_KEY", "").strip()
    app_key = os.environ.get("DD_APP_KEY", "").strip()
    site = (os.environ.get("DD_SITE") or DEFAULT_DD_SITE).strip().lstrip(".")
    if not api_key or not app_key:
        raise RuntimeError(
            "Datadog credentials not set. Need DD_API_KEY + DD_APP_KEY "
            "(APP key needs read scopes: logs_read_data, apm_read, "
            "monitors_read, events_read, slo_read)."
        )
    return _Config(api_key=api_key, app_key=app_key, api_url=f"https://api.{site}")


def _headers() -> dict:
    cfg = _config()
    return {
        "DD-API-KEY": cfg.api_key,
        "DD-APPLICATION-KEY": cfg.app_key,
        "Content-Type": "application/json",
    }


async def _get(path: str, params: dict | None = None, *, tool: str = "datadog") -> Any:
    cfg = _config()
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    async with httpx.AsyncClient(headers=_headers(), timeout=_DEFAULT_TIMEOUT_S) as http:
        resp = await http.get(f"{cfg.api_url}{path}", params=clean)
    if resp.status_code >= 400:
        raise DatadogMCPError(resp.status_code, resp.text, tool=tool)
    return resp.json() if resp.text else {}


async def _post(path: str, body: dict, *, tool: str = "datadog") -> Any:
    cfg = _config()
    async with httpx.AsyncClient(headers=_headers(), timeout=_DEFAULT_TIMEOUT_S) as http:
        resp = await http.post(f"{cfg.api_url}{path}", json=body)
    if resp.status_code >= 400:
        raise DatadogMCPError(resp.status_code, resp.text, tool=tool)
    return resp.json() if resp.text else {}


def _clamp(n: int | None, default: int = _DEFAULT_LIMIT) -> int:
    if n is None:
        return default
    return max(1, min(int(n), _MAX_LIMIT))


def _truncate(text: str, limit: int = _RESULT_TRUNCATE_CHARS) -> str:
    if not text:
        return ""
    text = _redact(text)
    if len(text) > limit:
        return text[:limit] + " ...[truncated]"
    return text


# --- handlers -------------------------------------------------------


async def _h_search_logs(_unused, args: dict) -> Any:
    """`/api/v2/logs/events` -- Datadog v2 logs search.
    Filters: `query` (DD log syntax), `from`/`to` (e.g. `now-1h`),
    `service`, `env`, `host`. Returns up to `limit` events with key
    fields trimmed."""
    query = args.get("query") or "*"
    extra = []
    if args.get("service"):
        extra.append(f"service:{args['service']}")
    if args.get("env"):
        extra.append(f"env:{args['env']}")
    if args.get("host"):
        extra.append(f"host:{args['host']}")
    if args.get("status"):
        extra.append(f"status:{args['status']}")
    full_query = " ".join([query] + extra) if extra else query

    params = {
        "filter[query]": full_query,
        "filter[from]": args.get("from") or "now-15m",
        "filter[to]": args.get("to") or "now",
        "page[limit]": _clamp(args.get("limit")),
        "sort": args.get("sort") or "-timestamp",
    }
    resp = await _get("/api/v2/logs/events", params=params, tool="datadog_search_logs")
    items = resp.get("data") or []
    out = []
    for x in items:
        a = x.get("attributes") or {}
        attrs = a.get("attributes") or {}
        msg = _truncate(a.get("message") or "", 2000)
        out.append({
            "id": x.get("id"),
            "ts": a.get("timestamp"),
            "service": attrs.get("service"),
            "env": attrs.get("env"),
            "host": attrs.get("host"),
            "status": attrs.get("status"),
            "message": msg,
            "tags": (attrs.get("tags") or [])[:20],
        })
    return {
        "query": full_query,
        "from": params["filter[from]"], "to": params["filter[to]"],
        "count": len(out),
        "logs": out,
    }


async def _h_search_spans(_unused, args: dict) -> Any:
    """`/api/v2/spans/events/search` -- APM trace span search.
    Use this to find slow / errored requests; chain with `datadog_get_trace`
    (using one span's `trace_id`) to drill into a full distributed trace."""
    body = {
        "data": {
            "type": "search_request",
            "attributes": {
                "filter": {
                    "query": args.get("query") or "*",
                    "from": args.get("from") or "now-15m",
                    "to": args.get("to") or "now",
                },
                "page": {"limit": _clamp(args.get("limit"))},
                "sort": args.get("sort") or "-timestamp",
            },
        },
    }
    resp = await _post("/api/v2/spans/events/search", body, tool="datadog_search_spans")
    items = resp.get("data") or []
    out = []
    # Datadog response shape (curl-verified 2026-05-15 against /api/v2/spans/events/search):
    #   data[].attributes is FLAT -- has trace_id/span_id/service/resource_name/status/env
    #   directly. Duration is at attributes.custom.duration (nanoseconds). Timestamps are
    #   start_timestamp / end_timestamp (ISO-8601). Rich error info is at
    #   attributes.error (top-level: {type}) and attributes.custom.error
    #   (detailed: {type, message, stack, fingerprint, file, handling}).
    for x in items:
        a = x.get("attributes") or {}
        custom = a.get("custom") or {}
        custom_err = custom.get("error") or {}
        err = a.get("error") or {}
        err_type = custom_err.get("type") or err.get("type")
        err_msg = custom_err.get("message")
        err_file = custom_err.get("file")
        out.append({
            "id": x.get("id"),
            "trace_id": a.get("trace_id"),
            "span_id": a.get("span_id"),
            "parent_id": a.get("parent_id"),
            "ts": a.get("start_timestamp"),
            "end_ts": a.get("end_timestamp"),
            "service": a.get("service"),
            "resource": a.get("resource_name"),
            "operation": a.get("operation_name"),
            "duration_ns": custom.get("duration"),
            "status": a.get("status"),
            "env": a.get("env"),
            "host": a.get("host"),
            "error": {
                "type": err_type,
                "message": err_msg,
                "file": err_file,
            } if (err_type or err_msg) else None,
        })
    return {
        "query": body["data"]["attributes"]["filter"]["query"],
        "count": len(out),
        "spans": out,
    }


async def _h_get_trace(_unused, args: dict) -> Any:
    """Walk all spans of one distributed trace by `trace_id`.

    Datadog has TWO retention layers and earlier code conflated them:

    1. **Live Search** (everything ingested, rolling **15-MINUTE** window) --
       used by the live trace stream UI. Not what this tool queries.
    2. **Indexed spans** (kept by Retention Filters -- incl. the default
       Intelligent Retention Filter + error/high-latency filters) --
       retained **15 DAYS**, fully queryable via
       `/api/v2/spans/events/search` by `trace_id`. THIS is what we use.

    So a trace from this morning is almost certainly still retrievable
    here (error traces typically match a default retention filter).
    The default time window is therefore **`now-15d` -> `now`**, not the
    narrow `now-1h` we used before -- that was the root cause of agents
    wrongly concluding "trace beyond retention" on a 9h-old trace ID.

    When the caller passes `epoch_ms` (e.g. from a parsed Datadog trace
    URL's `timeHint=...`), we narrow to `epoch_ms +/- 1h` for speed --
    same indexed dataset, just a tighter window so DD scans fewer
    shards. If neither `from/to` nor `epoch_ms` is given, default to
    `now-15d`/`now`.
    """
    trace_id = args["trace_id"]
    epoch_ms = args.get("epoch_ms")
    from_arg = args.get("from")
    to_arg = args.get("to")

    if from_arg or to_arg:
        # Caller explicitly set the window -- respect verbatim.
        time_from = from_arg or "now-15d"
        time_to = to_arg or "now"
    elif epoch_ms is not None:
        # Narrow +/-1h around the trace's known timestamp for speed.
        # Datadog v2 spans-search `filter.from`/`filter.to` REJECTS
        # numeric ms with `400 error decoding attribute "filter.from":
        # invalid type number`. Format as ISO 8601 strings instead.
        try:
            from datetime import datetime, timedelta
            anchor_dt = datetime.fromtimestamp(
                int(epoch_ms) / 1000, tz=UTC,
            )
            time_from = (anchor_dt - timedelta(hours=1)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            time_to = (anchor_dt + timedelta(hours=1)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        except (TypeError, ValueError, OverflowError, OSError):
            time_from, time_to = "now-15d", "now"
    else:
        # Trace ID alone -- default to the full indexed-spans retention.
        time_from, time_to = "now-15d", "now"

    body = {
        "data": {
            "type": "search_request",
            "attributes": {
                "filter": {
                    "query": f"trace_id:{trace_id}",
                    "from": time_from,
                    "to": time_to,
                },
                "page": {"limit": _clamp(args.get("limit"), default=100)},
                "sort": "timestamp",  # chronological for trace reconstruction
            },
        },
    }
    resp = await _post("/api/v2/spans/events/search", body, tool="datadog_get_trace")
    items = resp.get("data") or []
    # See _h_search_spans: attributes is FLAT, duration is in attributes.custom.duration,
    # rich error data is in attributes.custom.error (curl-verified 2026-05-15).
    spans = []
    services_seen: set[str] = set()
    total_duration_ns = 0
    errors_seen = []
    for x in items:
        a = x.get("attributes") or {}
        custom = a.get("custom") or {}
        custom_err = custom.get("error") or {}
        err = a.get("error") or {}
        svc = a.get("service")
        if svc:
            services_seen.add(svc)
        d = custom.get("duration") or 0
        if isinstance(d, (int, float)):
            total_duration_ns += int(d)
        err_type = custom_err.get("type") or err.get("type")
        err_msg = custom_err.get("message")
        err_stack = custom_err.get("stack")
        if err_type or err_msg:
            # Bound stack to 4 KiB so a single huge traceback can't blow the payload.
            stack_trimmed = (err_stack or "")[:4096]
            errors_seen.append({
                "span_id": a.get("span_id"),
                "service": svc,
                "resource": a.get("resource_name"),
                "type": err_type,
                "message": err_msg,
                "file": custom_err.get("file"),
                "stack": stack_trimmed if stack_trimmed else None,
            })
        spans.append({
            "span_id": a.get("span_id"),
            "parent_id": a.get("parent_id"),
            "ts": a.get("start_timestamp"),
            "service": svc,
            "resource": a.get("resource_name"),
            "operation": a.get("operation_name"),
            "duration_ns": custom.get("duration"),
            "status": a.get("status"),
        })
    return {
        "trace_id": trace_id,
        "span_count": len(spans),
        "services_seen": sorted(services_seen),
        "total_duration_ns": total_duration_ns,
        "errors": errors_seen,
        "spans": spans,
    }


# Datadog trace URL shapes (all observed at app.datadoghq.com):
#   /apm/trace/<hex>?spanID=<int>&timeHint=<epoch_ms>&...
#   /apm/trace/<hex>?env=prod&service=foo&spanID=...
#   query-string ordering is not stable, so parse via urllib.
_DD_TRACE_PATH_RE = re.compile(r"/apm/trace/([0-9a-fA-F]+)")


async def _h_parse_trace_url(_unused, args: dict) -> dict:
    """Deterministic Datadog APM trace URL parser. No network calls.

    Extracts trace_id, span_id, epoch_ms, ISO timestamp, site, and
    optional env/service hints from any Datadog trace URL. LLM-side
    URL reasoning is unreliable; this handler is the single source of
    truth so the agent can chain `parse_trace_url -> get_trace -> ES
    fallback` mechanically.

    Returns a dict with all fields the agent needs to continue the
    chain. Raises `bad_args` only when the URL is missing the trace_id
    path segment (the one piece without which nothing else is useful).
    """
    from datetime import datetime
    from urllib.parse import parse_qs, urlparse

    raw = (args.get("url") or "").strip()
    if not raw:
        raise RuntimeError(
            "datadog_parse_trace_url: `url` is required (a Datadog APM "
            "trace URL like https://app.datadoghq.com/apm/trace/<hex>?...)"
        )

    parsed = urlparse(raw)
    m = _DD_TRACE_PATH_RE.search(parsed.path or "")
    if not m:
        raise RuntimeError(
            f"datadog_parse_trace_url: not a Datadog trace URL "
            f"(no `/apm/trace/<hex>` in path: {parsed.path!r})"
        )
    trace_id = m.group(1)
    qs = parse_qs(parsed.query or "")
    span_id = (qs.get("spanID") or [None])[0]
    epoch_ms_raw = (qs.get("timeHint") or [None])[0]
    env_hint = (qs.get("env") or [None])[0]
    service_hint = (qs.get("service") or [None])[0]

    # Site: app.datadoghq.com -> datadoghq.com; app.datadoghq.eu -> datadoghq.eu, etc.
    host = (parsed.netloc or "").lower()
    if host.startswith("app."):
        host = host[len("app."):]
    site = host or "datadoghq.com"

    timestamp_iso: str | None = None
    epoch_ms: int | None = None
    if epoch_ms_raw:
        try:
            epoch_ms = int(epoch_ms_raw)
            timestamp_iso = (
                datetime.fromtimestamp(epoch_ms / 1000.0, tz=UTC)
                .isoformat()
                .replace("+00:00", "Z")
            )
        except (TypeError, ValueError):
            pass

    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "epoch_ms": epoch_ms,
        "timestamp_iso": timestamp_iso,
        "site": site,
        "env_hint": env_hint,
        "service_hint": service_hint,
        # Helpful nudge for the agent: tell it what to do next so the
        # chain doesn't depend on prompt memory alone.
        "next_action": (
            "Call `datadog_get_trace(trace_id, epoch_ms=<epoch_ms from "
            "this response>)`. If it returns 0 spans, fall back to "
            "`elasticsearch_search_logs(env=<env_hint or 'prod'>, "
            "service=<service_hint or user-provided>, "
            "time_range='<timestamp_iso minus 5m>/<timestamp_iso plus 5m>')`."
        ),
    }


async def _h_list_services(_unused, args: dict) -> Any:
    """List services with a FORMAL service definition (service.datadog.yaml or
    registered through the Service Catalog UI). This is a CURATED subset --
    most services emit APM data without a definition. For the active set
    (what actually shows up in Datadog Software Catalog), use
    `datadog_list_apm_services` instead.
    """
    params = {
        "page[size]": _clamp(args.get("limit")),
        "page[number]": int(args.get("page") or 0),
    }
    resp = await _get("/api/v2/services/definitions", params=params, tool="datadog_list_services")
    items = resp.get("data") or []
    out = []
    for x in items:
        schema = (x.get("attributes") or {}).get("schema") or {}
        out.append({
            "service": schema.get("dd-service"),
            "team": schema.get("team"),
            "tier": schema.get("tier"),
            "lifecycle": schema.get("lifecycle"),
            "type": schema.get("type"),
            "tags": (schema.get("tags") or [])[:20],
        })
    return {
        "count": len(out),
        "services": out,
        "_note": (
            "These are services with a formal Service Definition. Most "
            "services emit APM without one -- call datadog_list_apm_services "
            "to see every active tracer."
        ),
    }


async def _h_list_apm_services(_unused, args: dict) -> Any:
    """List EVERY service emitting APM traces in the given env + time window.

    Aggregates spans by the `service` facet -- returns the same list of
    services that the Datadog Software Catalog UI shows. Use this when the
    user asks "how many services in prod" or "which services emit APM" --
    `datadog_list_services` only returns formally-registered definitions
    (typically a small fraction).
    """
    env = (args.get("env") or "prod").strip()
    hours = int(args.get("hours") or 24)
    extra_query = (args.get("query") or "").strip()
    limit = int(args.get("limit") or 1000)
    # Build the Datadog query string. `env:` is added unconditionally
    # so the caller can't accidentally pull cross-env spans.
    parts = [f"env:{env}"]
    if extra_query:
        parts.append(extra_query)
    query = " ".join(parts)

    # Datadog spans aggregate API. Confirmed working shape (curl'd from
    # inside the pod 2026-05-15):
    #
    #   POST /api/v2/spans/analytics/aggregate
    #   {
    #     "data": {"type":"aggregate_request","attributes":{
    #       "filter":{"query":"env:prod","from":"now-24h","to":"now"},
    #       "compute":[{"aggregation":"count"}],
    #       "group_by":[{"facet":"service","limit":50}]
    #     }}
    #   }
    #
    # The Datadog API rejects `group_by[].sort` with a confusing
    # `Field 'aggregation' is invalid: Unrecognized parameter` error,
    # even when the sort body matches the published OpenAPI spec. Curl
    # probes A/B/C confirmed: the API only accepts a body WITHOUT
    # `sort`. We sort the results client-side (cheap -- 50 entries).
    #
    # Response shape (also curl-verified):
    #   { data: [
    #       { type: "bucket",
    #         attributes: { by: { service: "..." }, compute: { c0: 123 } } },
    #       ...
    #     ] }
    # Note `attributes.compute` is SINGULAR, not `computes`.
    body = {
        "data": {
            "type": "aggregate_request",
            "attributes": {
                "filter": {
                    "query": query,
                    "from": f"now-{hours}h",
                    "to": "now",
                },
                "compute": [{"aggregation": "count"}],
                "group_by": [{
                    "facet": "service",
                    "limit": max(1, min(limit, 10_000)),
                }],
            },
        }
    }
    resp = await _post(
        "/api/v2/spans/analytics/aggregate",
        body=body, tool="datadog_list_apm_services",
    )
    # Parse buckets. Verified response shape:
    #   data[].attributes.by.service + data[].attributes.compute.c0
    # We default any missing keys to safe values and dedupe-by-service
    # in case the API ever returns duplicate buckets.
    services_map: dict[str, int] = {}
    for bucket in resp.get("data") or []:
        attrs = bucket.get("attributes") or {}
        service = (attrs.get("by") or {}).get("service")
        if not service:
            continue
        # API uses `compute` (singular). Keep `computes` fallback just
        # in case Datadog flips it back -- both are cheap to check.
        computes = attrs.get("compute") or attrs.get("computes") or {}
        if isinstance(computes, dict):
            count = next(iter(computes.values()), 0) or 0
        elif isinstance(computes, list) and computes:
            count = computes[0].get("value", 0) if isinstance(computes[0], dict) else computes[0]
        else:
            count = 0
        services_map[service] = services_map.get(service, 0) + int(count or 0)

    # Sort client-side (Datadog rejects sort in the request -- see body
    # comment above). 50 entries -> sub-millisecond, no concern.
    services = [
        {"service": s, "span_count": c}
        for s, c in sorted(services_map.items(), key=lambda kv: kv[1], reverse=True)
    ]

    return {
        "env": env,
        "hours": hours,
        "query": query,
        "count": len(services),
        "services": services,
    }


async def _h_list_monitors(_unused, args: dict) -> Any:
    params = {
        "name": args.get("name"),
        "tags": args.get("tags"),
        "monitor_tags": args.get("monitor_tags"),
        "group_states": args.get("group_states"),
        "page": int(args.get("page") or 0),
        "page_size": _clamp(args.get("limit")),
    }
    resp = await _get("/api/v1/monitor", params=params, tool="datadog_list_monitors")
    items = resp if isinstance(resp, list) else []
    out = [
        {
            "id": m.get("id"),
            "name": (m.get("name") or "")[:200],
            "type": m.get("type"),
            "state": m.get("overall_state"),
            "query": (m.get("query") or "")[:300],
            "tags": (m.get("tags") or [])[:15],
            "muted": (m.get("options") or {}).get("silenced") or False,
            "modified": m.get("modified"),
        }
        for m in items
    ]
    # Sort alerting first so the LLM sees important rows.
    state_order = {"Alert": 0, "Warn": 1, "No Data": 2, "OK": 3, "Skipped": 4}
    out.sort(key=lambda m: state_order.get(m.get("state") or "", 99))
    return {
        "count": len(out),
        "alerting": sum(1 for m in out if m.get("state") == "Alert"),
        "no_data": sum(1 for m in out if m.get("state") == "No Data"),
        "ok": sum(1 for m in out if m.get("state") == "OK"),
        "monitors": out,
    }


async def _h_get_monitor(_unused, args: dict) -> Any:
    monitor_id = args["monitor_id"]
    resp = await _get(f"/api/v1/monitor/{monitor_id}", tool="datadog_get_monitor")
    state = resp.get("state") or {}
    groups = state.get("groups") or {}
    return {
        "id": resp.get("id"),
        "name": resp.get("name"),
        "type": resp.get("type"),
        "state": resp.get("overall_state"),
        "query": (resp.get("query") or "")[:1000],
        "message": (resp.get("message") or "")[:1500],
        "tags": resp.get("tags") or [],
        "options": {
            k: v for k, v in (resp.get("options") or {}).items()
            if k in ("silenced", "thresholds", "evaluation_delay", "timeout_h", "renotify_interval")
        },
        "creator": (resp.get("creator") or {}).get("email"),
        "created": resp.get("created"),
        "modified": resp.get("modified"),
        "group_states": [
            {
                "name": g.get("name"),
                "status": g.get("status"),
                "last_triggered_ts": g.get("last_triggered_ts"),
                "last_resolved_ts": g.get("last_resolved_ts"),
            }
            for g in list(groups.values())[:20]
        ],
    }


async def _h_list_events(_unused, args: dict) -> Any:
    """Events stream -- deploys, alert state changes, custom events. Uses unix
    timestamps for `start`/`end`."""
    now = int(time.time())
    end = int(args.get("end") or now)
    start = int(args.get("start") or (now - 3600))  # default last 1h
    params = {
        "start": start,
        "end": end,
        "priority": args.get("priority"),
        "sources": args.get("sources"),
        "tags": args.get("tags"),
    }
    if args.get("query"):
        params["unaggregated"] = "true"
    resp = await _get("/api/v1/events", params=params, tool="datadog_list_events")
    events = resp.get("events") or []
    # Apply text filter client-side if `query` provided (DD's GET endpoint
    # doesn't accept query param; tags filtering is the API path).
    q = (args.get("query") or "").lower()
    if q:
        events = [e for e in events if q in (e.get("title") or "").lower()
                  or q in (e.get("text") or "").lower()]
    limit = _clamp(args.get("limit"), default=50)
    events = events[:limit]
    out = [
        {
            "id": e.get("id"),
            "ts": e.get("date_happened"),
            "title": (e.get("title") or "")[:300],
            "source": e.get("source"),
            "priority": e.get("priority"),
            "alert_type": e.get("alert_type"),
            "tags": (e.get("tags") or [])[:15],
            "host": e.get("host"),
            "text": _truncate((e.get("text") or "")[:1500], 1500),
        }
        for e in events
    ]
    return {
        "from": start, "to": end,
        "count": len(out),
        "events": out,
    }


async def _h_get_slo(_unused, args: dict) -> Any:
    slo_id = args["slo_id"]
    now = int(time.time())
    fr = int(args.get("from_ts") or (now - 7 * 86400))  # default last 7 days
    to = int(args.get("to_ts") or now)
    params = {"from_ts": fr, "to_ts": to}
    resp = await _get(f"/api/v1/slo/{slo_id}/history", params=params, tool="datadog_get_slo")
    data = resp.get("data") or {}
    overall = data.get("overall") or {}
    return {
        "slo_id": slo_id,
        "from": fr, "to": to,
        "name": data.get("slo", {}).get("name"),
        "target": (data.get("slo", {}).get("thresholds") or [{}])[0].get("target") if data.get("slo") else None,
        "sli": overall.get("sli_value"),
        "error_budget_remaining_pct": overall.get("error_budget_remaining"),
        "uptime": overall.get("uptime"),
        "history_points": len(data.get("series") or []),
    }


# --- tool registry --------------------------------------------------


DATADOG_TOOLS: list[MCPTool] = [
    # `datadog_search_logs` was DELIBERATELY REMOVED 2026-05-21.
    # The Datadog deployment is provisioned for traces/APM/metrics/SLOs ONLY --
    # application logs route to Elasticsearch (eck-applogs cluster
    # per env). The new `elasticsearch_*` MCP tools (search_logs,
    # log_count, list_services) are the correct entry point for log
    # search. Keeping the Datadog log tool was actively misleading
    # the agent: it kept selecting it instead of the working ES path.
    MCPTool(
        name="datadog_search_spans",
        description=(
            "APM trace-span search. Find slow / errored requests across all "
            "services. Each result includes a `trace_id`; chain with "
            "`datadog_get_trace` to pull the full distributed trace.\n\n"
            "Common queries:\n"
            "- Slow requests: `query='@duration:>5s'`\n"
            "- Errors only: `query='status:error'`\n"
            "- One service: `query='service:acme-notes-be'`"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "from": {"type": "string"},
                "to": {"type": "string"},
                "limit": {"type": "number"},
                "sort": {"type": "string"},
            },
        },
        handler=_h_search_spans,
    ),
    MCPTool(
        name="datadog_get_trace",
        description=(
            "Pull every span of a distributed trace by `trace_id`. Returns "
            "span tree (parent_id chain), services involved, total duration. "
            "Queries Datadog's indexed-spans store (Retention Filter window, "
            "retained for ~15 days) -- NOT the live-stream 15-min window. "
            "Default time window is the full `now-15d` -> `now` so traces "
            "from hours/days ago are still found. Pass `epoch_ms` (e.g. "
            "from `datadog_parse_trace_url`) to narrow to +/-1h for speed."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "trace_id": {"type": "string"},
                "epoch_ms": {
                    "type": "number",
                    "description": (
                        "Optional unix-ms timestamp anchor (e.g. from a "
                        "parsed trace URL's `timeHint`). When set, the "
                        "search window narrows to +/-1h around this point. "
                        "When unset, default is now-15d -> now."
                    ),
                },
                "from": {
                    "type": "string",
                    "description": (
                        "Override start of search window. Default `now-15d` "
                        "(indexed-spans retention). Don't shrink this "
                        "without good reason -- old traces are often the "
                        "interesting ones."
                    ),
                },
                "to": {"type": "string", "description": "default 'now'"},
                "limit": {"type": "number", "description": "max spans (default 100)"},
            },
            "required": ["trace_id"],
        },
        handler=_h_get_trace,
    ),
    MCPTool(
        name="datadog_parse_trace_url",
        description=(
            "Deterministic extractor for any Datadog APM trace URL. Pure "
            "string parsing -- no API calls. Returns `trace_id`, "
            "`span_id`, `epoch_ms`, `timestamp_iso`, plus optional "
            "`env_hint` / `service_hint` if the URL carries them. "
            "Call this FIRST when the user pastes a `datadoghq.com/apm/"
            "trace/...` URL -- then feed `trace_id` + `epoch_ms` into "
            "`datadog_get_trace`. Skips LLM-based URL parsing which is "
            "fragile."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "Full Datadog trace URL, e.g. "
                        "https://app.datadoghq.com/apm/trace/<hex>?"
                        "spanID=<id>&timeHint=<epoch_ms>&env=prod&service=..."
                    ),
                },
            },
            "required": ["url"],
        },
        handler=_h_parse_trace_url,
    ),
    MCPTool(
        name="datadog_list_services",
        description=(
            "Services with a FORMAL Service Definition (registered via "
            "`service.datadog.yaml` or the Service Catalog UI). This is "
            "a CURATED subset -- most services emit APM "
            "without a definition, so this typically returns far fewer "
            "rows than the Software Catalog UI shows. "
            "For 'how many services in env X?' or 'list all services "
            "emitting APM traces' -> use `datadog_list_apm_services` "
            "instead. This tool is best for 'which services have a "
            "registered definition + tier/team metadata?'"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "limit": {"type": "number"},
                "page": {"type": "number"},
            },
        },
        handler=_h_list_services,
    ),
    MCPTool(
        name="datadog_list_apm_services",
        description=(
            "List EVERY service emitting APM traces in a given env "
            "(default `prod`) over the last N hours (default 24). "
            "Aggregates spans by the `service` facet -- matches what "
            "the Datadog Software Catalog UI shows. "
            "Use this for: 'how many services in prod?', 'which services "
            "emit APM?', 'list all services with traffic in the last 6h'. "
            "Returns service name + span count, sorted by traffic desc. "
            "Prefer this over `datadog_list_services` when the user's "
            "question is about ACTIVE services, not curated definitions."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "env": {
                    "type": "string",
                    "description": "Datadog env tag. Default `prod`.",
                },
                "hours": {
                    "type": "number",
                    "description": "Look-back window in hours. Default 24.",
                },
                "query": {
                    "type": "string",
                    "description": "Optional extra DDQL clause AND-combined with `env:<env>` (e.g. `team:platform`).",
                },
                "limit": {
                    "type": "number",
                    "description": "Max services to return. Default 1000, cap 10000.",
                },
            },
        },
        handler=_h_list_apm_services,
    ),
    MCPTool(
        name="datadog_list_monitors",
        description=(
            "List configured monitors with current state. Sorted alerting-first "
            "so org health surfaces immediately. Returns count by state "
            "(alerting / no_data / ok). Filter by `name`, `tags`, "
            "`group_states` ('alert', 'warn', 'no data', 'ok')."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "name substring"},
                "tags": {"type": "string", "description": "comma-separated"},
                "monitor_tags": {"type": "string"},
                "group_states": {"type": "string", "description": "alert|warn|no data|ok"},
                "limit": {"type": "number"},
                "page": {"type": "number"},
            },
        },
        handler=_h_list_monitors,
    ),
    MCPTool(
        name="datadog_get_monitor",
        description="Full detail of one monitor -- query, thresholds, options, per-group state with last-triggered timestamps.",
        input_schema={
            "type": "object",
            "properties": {"monitor_id": {"type": "string"}},
            "required": ["monitor_id"],
        },
        handler=_h_get_monitor,
    ),
    MCPTool(
        name="datadog_list_events",
        description=(
            "Event stream -- deploys, alert state changes, custom events. "
            "`query` filters client-side on title/text. `sources` filters by "
            "integration source (e.g. 'kubernetes', 'github')."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Substring filter on event title/text"},
                "start": {"type": "number", "description": "Unix timestamp; default now-1h"},
                "end": {"type": "number", "description": "Unix timestamp; default now"},
                "priority": {"type": "string", "enum": ["normal", "low"]},
                "sources": {"type": "string", "description": "e.g. 'kubernetes,github'"},
                "tags": {"type": "string"},
                "limit": {"type": "number"},
            },
        },
        handler=_h_list_events,
    ),
    MCPTool(
        name="datadog_get_slo",
        description="SLO history -- current SLI, target, error-budget remaining, uptime over the time window.",
        input_schema={
            "type": "object",
            "properties": {
                "slo_id": {"type": "string"},
                "from_ts": {"type": "number", "description": "Unix; default 7 days ago"},
                "to_ts": {"type": "number", "description": "Unix; default now"},
            },
            "required": ["slo_id"],
        },
        handler=_h_get_slo,
    ),
]


def get_tool(name: str) -> MCPTool:
    for t in DATADOG_TOOLS:
        if t.name == name:
            return t
    raise KeyError(f"unknown datadog tool: {name}")


# --- fake backend (FR-012; integration tests) ----------------------
#
# Datadog handlers ignore their first (`_unused`) arg and reach the
# module-level `_get` / `_post`, which build httpx clients from env via
# `_config()`. So the offline fake replaces those two module functions
# with canned, shape-faithful responders -- no network, no DD keys.
# `datadog_parse_trace_url` is pure string parsing and touches neither,
# so it works unchanged. `build_fake()` returns client=None (handlers
# discard it) plus a teardown that restores the real `_get` / `_post`.


async def _fake_get(path: str, params: dict | None = None, *, tool: str = "datadog") -> Any:
    """Canned stand-in for the module-level GET. Routes by path to a
    response shaped like the real Datadog v1/v2 endpoint the handler
    parses."""
    if path == "/api/v2/services/definitions":
        return {
            "data": [
                {
                    "attributes": {
                        "schema": {
                            "dd-service": "acme-notes-be",
                            "team": "platform",
                            "tier": "1",
                            "lifecycle": "production",
                            "type": "web",
                            "tags": ["env:prod"],
                        }
                    }
                }
            ]
        }
    if path == "/api/v1/monitor":
        return [
            {
                "id": 101,
                "name": "High error rate",
                "type": "metric alert",
                "overall_state": "Alert",
                "query": "avg(last_5m):sum:errors > 100",
                "tags": ["service:acme-notes-be"],
                "options": {"silenced": {}},
                "modified": "2026-05-20T00:00:00Z",
            },
            {
                "id": 102,
                "name": "CPU OK",
                "type": "metric alert",
                "overall_state": "OK",
                "query": "avg(last_5m):cpu < 90",
                "tags": ["service:acme-notes-be"],
                "options": {},
                "modified": "2026-05-19T00:00:00Z",
            },
        ]
    if path.startswith("/api/v1/monitor/"):
        return {
            "id": 101,
            "name": "High error rate",
            "type": "metric alert",
            "overall_state": "Alert",
            "query": "avg(last_5m):sum:errors > 100",
            "message": "Errors are high. Notify @team",
            "tags": ["service:acme-notes-be"],
            "options": {"silenced": {}, "thresholds": {"critical": 100}},
            "creator": {"email": "sre@example.com"},
            "created": "2026-01-01T00:00:00Z",
            "modified": "2026-05-20T00:00:00Z",
            "state": {
                "groups": {
                    "host:web-1": {
                        "name": "host:web-1",
                        "status": "Alert",
                        "last_triggered_ts": 1716000000,
                        "last_resolved_ts": 1715990000,
                    }
                }
            },
        }
    if path == "/api/v1/events":
        return {
            "events": [
                {
                    "id": 555,
                    "date_happened": 1716000000,
                    "title": "Deploy acme-notes-be v1.2.3",
                    "source": "github",
                    "priority": "normal",
                    "alert_type": "info",
                    "tags": ["env:prod"],
                    "host": "web-1",
                    "text": "Deployed by CI",
                }
            ]
        }
    if path.startswith("/api/v1/slo/") and path.endswith("/history"):
        return {
            "data": {
                "slo": {
                    "name": "Availability",
                    "thresholds": [{"target": 99.9}],
                },
                "overall": {
                    "sli_value": 99.95,
                    "error_budget_remaining": 50.0,
                    "uptime": 99.95,
                },
                "series": [{"ts": 1716000000}, {"ts": 1716003600}],
            }
        }
    return {}


async def _fake_post(path: str, body: dict, *, tool: str = "datadog") -> Any:
    """Canned stand-in for the module-level POST (span search + APM
    service aggregate)."""
    if path == "/api/v2/spans/analytics/aggregate":
        return {
            "data": [
                {
                    "type": "bucket",
                    "attributes": {"by": {"service": "acme-notes-be"}, "compute": {"c0": 1200}},
                },
                {
                    "type": "bucket",
                    "attributes": {"by": {"service": "acme-auth"}, "compute": {"c0": 300}},
                },
            ]
        }
    if path == "/api/v2/spans/events/search":
        return {
            "data": [
                {
                    "id": "span-1",
                    "attributes": {
                        "trace_id": "abc123",
                        "span_id": "1",
                        "parent_id": None,
                        "start_timestamp": "2026-05-20T00:00:00Z",
                        "end_timestamp": "2026-05-20T00:00:01Z",
                        "service": "acme-notes-be",
                        "resource_name": "GET /notes",
                        "operation_name": "http.request",
                        "status": "error",
                        "env": "prod",
                        "host": "web-1",
                        "custom": {
                            "duration": 1000000,
                            "error": {
                                "type": "RuntimeError",
                                "message": "boom",
                                "stack": "Traceback...",
                                "file": "app.py",
                            },
                        },
                        "error": {"type": "RuntimeError"},
                    },
                }
            ]
        }
    return {}


def build_fake():
    """Return a FakeMCP exposing the Datadog tools wired to an offline
    backend. Needs NO DD keys / network: the module-level `_get` / `_post`
    are swapped for canned responders and restored by `teardown`."""
    import opsrag.mcp.datadog as _mod
    from opsrag.mcp._fake import FakeMCP

    _orig_get = _mod._get
    _orig_post = _mod._post
    _mod._get = _fake_get
    _mod._post = _fake_post

    def _restore() -> None:
        _mod._get = _orig_get
        _mod._post = _orig_post

    return FakeMCP(tools=list(DATADOG_TOOLS), client=None, teardown=_restore)
