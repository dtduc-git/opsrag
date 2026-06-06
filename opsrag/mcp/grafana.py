"""Grafana MCP-style tools for OpsRAG.

Read-only async tools over the Grafana HTTP API, fronting Prometheus +
Loki via Grafana's datasource proxy. A single service-account token
(`GRAFANA_TOKEN`) authenticates every call with
``Authorization: Bearer <token>``. The base URL (`GRAFANA_URL`) is
configurable so self-hosted / regional Grafana instances work.

## Read-only enforcement

Every tool issues `httpx.AsyncClient.get` -- no POST / PUT / DELETE /
PATCH anywhere. No dashboard mutation, no datasource edits, no alert
silencing. Prometheus / Loki queries go through the datasource proxy's
read-only `/api/v1/query`, `/api/v1/query_range`, `/loki/api/v1/...`
endpoints.

## Tool list (9 read-only)

| Tool                              | Endpoint                                                     |
|-----------------------------------|-------------------------------------------------------------|
| `grafana_search_dashboards`       | GET `/api/search?type=dash-db&query=`                       |
| `grafana_get_dashboard`           | GET `/api/dashboards/uid/{uid}`                             |
| `grafana_list_datasources`        | GET `/api/datasources`                                      |
| `grafana_query_prometheus`        | GET proxy `/api/v1/query` or `/api/v1/query_range`          |
| `grafana_prometheus_label_values` | GET proxy `/api/v1/label/{label}/values`                    |
| `grafana_query_loki`              | GET proxy `/loki/api/v1/query_range`                        |
| `grafana_loki_label_values`       | GET proxy `/loki/api/v1/label/{label}/values`              |
| `grafana_list_alert_rules`        | GET `/api/prometheus/grafana/api/v1/rules`                  |
| `grafana_list_contact_points`     | GET `/api/v1/provisioning/contact-points`                   |
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx

from opsrag.mcp.gitlab import MCPTool

_log = logging.getLogger("opsrag.mcp.grafana")

_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_LIMIT = 25
_MAX_LIMIT = 100
_RESULT_TRUNCATE_CHARS = 32000

# Redact secrets that can leak through log lines / dashboard JSON.
_REDACT_PATTERNS = [
    (re.compile(r"\bxoxb-[A-Za-z0-9-]{20,}"), "[REDACTED:slack_bot_token]"),
    (re.compile(r"\bxoxp-[A-Za-z0-9-]{20,}"), "[REDACTED:slack_user_token]"),
    (re.compile(r"\bghp_[A-Za-z0-9]{30,}"), "[REDACTED:github_token]"),
    (re.compile(r"\bglpat-[A-Za-z0-9_]{20,}"), "[REDACTED:gitlab_token]"),
    (re.compile(r"\bAKIA[A-Z0-9]{16}\b"), "[REDACTED:aws_access_key]"),
    (re.compile(r"\bAIza[A-Za-z0-9_-]{35}"), "[REDACTED:google_api_key]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]+"), "[REDACTED:jwt]"),
    (re.compile(r"\bglsa_[A-Za-z0-9_]{20,}"), "[REDACTED:grafana_sa_token]"),
]


def _redact(text: str) -> str:
    for pat, repl in _REDACT_PATTERNS:
        text = pat.sub(repl, text)
    return text


class GrafanaMCPError(Exception):
    """Raised on Grafana API errors. Wraps upstream status + body (token redacted)."""

    def __init__(self, status: int, body: str, *, tool: str | None = None):
        self.status = status
        self.body = _redact(body or "")
        self.tool = tool
        super().__init__(f"[{tool or 'grafana'}] {status}: {self.body[:300]}")


@dataclass
class _Config:
    url: str
    token: str


def _config() -> _Config:
    url = (os.environ.get("GRAFANA_URL") or "").strip().rstrip("/")
    token = (os.environ.get("GRAFANA_TOKEN") or "").strip().strip('"').strip("'")
    if not url:
        raise GrafanaMCPError(
            0, "GRAFANA_URL not set (e.g. https://grafana.example.com)", tool="grafana"
        )
    if not token:
        raise GrafanaMCPError(
            0,
            "GRAFANA_TOKEN not set (Grafana service-account token; auth is "
            "'Authorization: Bearer <token>')",
            tool="grafana",
        )
    return _Config(url=url, token=token)


def _headers() -> dict:
    cfg = _config()
    return {
        "Authorization": f"Bearer {cfg.token}",
        "Accept": "application/json",
    }


async def _get(path: str, params: dict | None = None, *, tool: str = "grafana") -> Any:
    cfg = _config()
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    async with httpx.AsyncClient(headers=_headers(), timeout=_DEFAULT_TIMEOUT_S) as http:
        resp = await http.get(f"{cfg.url}{path}", params=clean)
    if resp.status_code >= 400:
        raise GrafanaMCPError(resp.status_code, resp.text, tool=tool)
    return resp.json() if resp.text else {}


def _clamp(n: int | None, default: int = _DEFAULT_LIMIT, *, maximum: int = _MAX_LIMIT) -> int:
    if n is None:
        return default
    return max(1, min(int(n), maximum))


def _truncate(text: str, limit: int = _RESULT_TRUNCATE_CHARS) -> str:
    if not text:
        return ""
    text = _redact(text)
    if len(text) > limit:
        return text[:limit] + " ...[truncated]"
    return text


# --- handlers -------------------------------------------------------


async def _h_search_dashboards(_unused, args: dict) -> Any:
    """`GET /api/search?type=dash-db` -- find dashboards by title/tag.

    Returns trimmed records (uid, title, url, tags, folder) so the agent
    can pick a uid and follow up with `grafana_get_dashboard`."""
    params = {
        "type": "dash-db",
        "query": args.get("query"),
        "limit": _clamp(args.get("limit")),
    }
    tag = args.get("tag")
    if tag:
        params["tag"] = tag
    resp = await _get("/api/search", params=params, tool="grafana_search_dashboards")
    items = resp if isinstance(resp, list) else []
    out = [
        {
            "uid": d.get("uid"),
            "title": (d.get("title") or "")[:300],
            "url": d.get("url"),
            "type": d.get("type"),
            "tags": (d.get("tags") or [])[:20],
            "folder": d.get("folderTitle"),
        }
        for d in items
    ]
    return {"count": len(out), "dashboards": out}


async def _h_get_dashboard(_unused, args: dict) -> Any:
    """`GET /api/dashboards/uid/{uid}` -- dashboard title + panel titles +
    each panel's query targets/exprs. Trims the heavy raw JSON to just what
    an SRE needs to understand what a dashboard measures."""
    uid = args["uid"]
    resp = await _get(f"/api/dashboards/uid/{uid}", tool="grafana_get_dashboard")
    dash = resp.get("dashboard") or {}
    meta = resp.get("meta") or {}
    panels_out = []
    for p in dash.get("panels") or []:
        targets = []
        for t in p.get("targets") or []:
            targets.append({
                "datasource": (t.get("datasource") or {}) if isinstance(t.get("datasource"), dict) else t.get("datasource"),
                "expr": _truncate(t.get("expr") or t.get("query") or "", 2000),
                "legend": t.get("legendFormat"),
                "refId": t.get("refId"),
            })
        panels_out.append({
            "id": p.get("id"),
            "title": (p.get("title") or "")[:300],
            "type": p.get("type"),
            "targets": targets,
        })
    return {
        "uid": dash.get("uid") or uid,
        "title": (dash.get("title") or "")[:300],
        "tags": (dash.get("tags") or [])[:20],
        "folder": meta.get("folderTitle"),
        "url": meta.get("url"),
        "panel_count": len(panels_out),
        "panels": panels_out,
    }


async def _h_list_datasources(_unused, args: dict) -> Any:
    """`GET /api/datasources` -- list configured datasources (name, uid,
    type). Use the uid to target `grafana_query_prometheus` /
    `grafana_query_loki`."""
    resp = await _get("/api/datasources", tool="grafana_list_datasources")
    items = resp if isinstance(resp, list) else []
    out = [
        {
            "name": d.get("name"),
            "uid": d.get("uid"),
            "type": d.get("type"),
            "is_default": d.get("isDefault", False),
        }
        for d in items
    ]
    return {"count": len(out), "datasources": out}


def _proxy(uid: str, suffix: str) -> str:
    return f"/api/datasources/proxy/uid/{uid}{suffix}"


async def _h_query_prometheus(_unused, args: dict) -> Any:
    """PromQL instant or range query via the datasource proxy.

    Instant: pass `query` (+ optional `time`). Range: pass `query` + `start`
    + `end` (+ optional `step`). Routes to `/api/v1/query` or
    `/api/v1/query_range` accordingly."""
    uid = args["datasource_uid"]
    query = args["query"]
    start = args.get("start")
    end = args.get("end")
    if start is not None and end is not None:
        params = {
            "query": query,
            "start": start,
            "end": end,
            "step": args.get("step") or "60s",
        }
        path = _proxy(uid, "/api/v1/query_range")
        mode = "range"
    else:
        params = {"query": query}
        if args.get("time") is not None:
            params["time"] = args["time"]
        path = _proxy(uid, "/api/v1/query")
        mode = "instant"
    resp = await _get(path, params=params, tool="grafana_query_prometheus")
    data = resp.get("data") or {}
    result = data.get("result") or []
    limit = _clamp(args.get("limit"), default=_DEFAULT_LIMIT)
    return {
        "status": resp.get("status"),
        "mode": mode,
        "result_type": data.get("resultType"),
        "series_count": len(result),
        "result": result[:limit],
    }


async def _h_prometheus_label_values(_unused, args: dict) -> Any:
    """`GET proxy /api/v1/label/{label}/values` -- enumerate values of a
    Prometheus label (e.g. `job`, `instance`, `namespace`). Handy for
    discovering valid query selectors."""
    uid = args["datasource_uid"]
    label = args["label"]
    resp = await _get(
        _proxy(uid, f"/api/v1/label/{label}/values"),
        tool="grafana_prometheus_label_values",
    )
    values = resp.get("data") or []
    limit = _clamp(args.get("limit"), default=_MAX_LIMIT)
    return {
        "status": resp.get("status"),
        "label": label,
        "count": len(values),
        "values": values[:limit],
    }


async def _h_query_loki(_unused, args: dict) -> Any:
    """LogQL range query via the datasource proxy
    (`/loki/api/v1/query_range`). Requires an explicit `limit` and a time
    bound (`start`/`end`, unix-ns or RFC3339) so an unbounded log scan is
    impossible. Log lines are redacted + length-capped."""
    uid = args["datasource_uid"]
    query = args["query"]
    start = args.get("start")
    end = args.get("end")
    if start is None or end is None:
        raise GrafanaMCPError(
            0,
            "grafana_query_loki requires explicit `start` and `end` time "
            "bounds (no unbounded log scans).",
            tool="grafana_query_loki",
        )
    limit = _clamp(args.get("limit"), default=_DEFAULT_LIMIT)
    params = {
        "query": query,
        "start": start,
        "end": end,
        "limit": limit,
        "direction": args.get("direction") or "backward",
    }
    resp = await _get(
        _proxy(uid, "/loki/api/v1/query_range"),
        params=params, tool="grafana_query_loki",
    )
    data = resp.get("data") or {}
    streams = data.get("result") or []
    out_streams = []
    for s in streams[:limit]:
        entries = []
        for ts, line in (s.get("values") or [])[:limit]:
            entries.append([ts, _truncate(line or "", 2000)])
        out_streams.append({"labels": s.get("stream") or {}, "values": entries})
    return {
        "status": resp.get("status"),
        "result_type": data.get("resultType"),
        "stream_count": len(out_streams),
        "streams": out_streams,
    }


async def _h_loki_label_values(_unused, args: dict) -> Any:
    """`GET proxy /loki/api/v1/label/{label}/values` -- enumerate values of
    a Loki label (e.g. `app`, `namespace`, `pod`)."""
    uid = args["datasource_uid"]
    label = args["label"]
    params = {}
    if args.get("start") is not None:
        params["start"] = args["start"]
    if args.get("end") is not None:
        params["end"] = args["end"]
    resp = await _get(
        _proxy(uid, f"/loki/api/v1/label/{label}/values"),
        params=params or None, tool="grafana_loki_label_values",
    )
    values = resp.get("data") or []
    limit = _clamp(args.get("limit"), default=_MAX_LIMIT)
    return {
        "status": resp.get("status"),
        "label": label,
        "count": len(values),
        "values": values[:limit],
    }


async def _h_list_alert_rules(_unused, args: dict) -> Any:
    """`GET /api/prometheus/grafana/api/v1/rules` -- Grafana-managed alert
    rule groups with current firing state. Returns groups + each rule's
    name / state (firing/pending/inactive) / health."""
    resp = await _get(
        "/api/prometheus/grafana/api/v1/rules",
        tool="grafana_list_alert_rules",
    )
    data = resp.get("data") or {}
    groups_in = data.get("groups") or []
    groups_out = []
    firing = 0
    pending = 0
    for g in groups_in:
        rules_out = []
        for r in g.get("rules") or []:
            state = r.get("state")
            if state == "firing":
                firing += 1
            elif state == "pending":
                pending += 1
            rules_out.append({
                "name": r.get("name"),
                "state": state,
                "health": r.get("health"),
                "query": _truncate(r.get("query") or "", 2000),
                "active_alerts": len(r.get("alerts") or []),
            })
        groups_out.append({
            "name": g.get("name"),
            "file": g.get("file"),
            "rules": rules_out,
        })
    return {
        "status": resp.get("status"),
        "group_count": len(groups_out),
        "firing": firing,
        "pending": pending,
        "groups": groups_out,
    }


async def _h_list_contact_points(_unused, args: dict) -> Any:
    """`GET /api/v1/provisioning/contact-points` -- alerting contact points
    (name + type, e.g. slack/email/pagerduty). Settings are NOT returned --
    they can hold secrets."""
    resp = await _get(
        "/api/v1/provisioning/contact-points",
        tool="grafana_list_contact_points",
    )
    items = resp if isinstance(resp, list) else []
    out = [
        {
            "name": c.get("name"),
            "type": c.get("type"),
            "uid": c.get("uid"),
            "disable_resolve_message": c.get("disableResolveMessage", False),
        }
        for c in items
    ]
    return {"count": len(out), "contact_points": out}


# --- tool registry --------------------------------------------------


GRAFANA_TOOLS: list[MCPTool] = [
    MCPTool(
        name="grafana_search_dashboards",
        description=(
            "Search Grafana dashboards by title substring and/or tag. "
            "Returns uid + title + url + tags. Use the uid with "
            "`grafana_get_dashboard` to inspect a dashboard's panels/queries."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Title substring filter"},
                "tag": {"type": "string", "description": "Filter by dashboard tag"},
                "limit": {"type": "number", "description": f"Max {_MAX_LIMIT}, default {_DEFAULT_LIMIT}"},
            },
        },
        handler=_h_search_dashboards,
    ),
    MCPTool(
        name="grafana_get_dashboard",
        description=(
            "Get one dashboard by uid: title + every panel's title and its "
            "query targets/exprs. Use this to learn what metrics a dashboard "
            "tracks (then run the exprs via `grafana_query_prometheus`)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "uid": {"type": "string", "description": "Dashboard uid (from grafana_search_dashboards)"},
            },
            "required": ["uid"],
        },
        handler=_h_get_dashboard,
    ),
    MCPTool(
        name="grafana_list_datasources",
        description=(
            "List configured datasources (name, uid, type, is_default). The "
            "uid is required by `grafana_query_prometheus` / "
            "`grafana_query_loki` / the label-values tools."
        ),
        input_schema={"type": "object", "properties": {}},
        handler=_h_list_datasources,
    ),
    MCPTool(
        name="grafana_query_prometheus",
        description=(
            "Run a PromQL query against a Prometheus datasource via Grafana's "
            "datasource proxy. Instant query: pass `query` (+ optional "
            "`time`). Range query: pass `query` + `start` + `end` (+ optional "
            "`step`, default 60s). Find the datasource_uid via "
            "`grafana_list_datasources`."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "datasource_uid": {"type": "string"},
                "query": {"type": "string", "description": "PromQL expression"},
                "time": {"type": "string", "description": "Instant-query eval time (RFC3339 or unix)"},
                "start": {"type": "string", "description": "Range-query start (RFC3339 or unix)"},
                "end": {"type": "string", "description": "Range-query end (RFC3339 or unix)"},
                "step": {"type": "string", "description": "Range step, default 60s"},
                "limit": {"type": "number", "description": "Max series returned"},
            },
            "required": ["datasource_uid", "query"],
        },
        handler=_h_query_prometheus,
    ),
    MCPTool(
        name="grafana_prometheus_label_values",
        description=(
            "Enumerate the values of a Prometheus label (e.g. `job`, "
            "`namespace`, `instance`) via the datasource proxy. Use to "
            "discover valid selectors before writing a PromQL query."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "datasource_uid": {"type": "string"},
                "label": {"type": "string", "description": "Label name, e.g. 'job'"},
                "limit": {"type": "number"},
            },
            "required": ["datasource_uid", "label"],
        },
        handler=_h_prometheus_label_values,
    ),
    MCPTool(
        name="grafana_query_loki",
        description=(
            "Run a LogQL range query against a Loki datasource via the "
            "datasource proxy. REQUIRES explicit `start` + `end` time bounds "
            "and a `limit` (no unbounded scans). Returns log streams with "
            "redacted, length-capped lines."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "datasource_uid": {"type": "string"},
                "query": {"type": "string", "description": "LogQL expression, e.g. '{app=\"api\"} |= \"error\"'"},
                "start": {"type": "string", "description": "Start time (unix-ns or RFC3339). REQUIRED."},
                "end": {"type": "string", "description": "End time (unix-ns or RFC3339). REQUIRED."},
                "limit": {"type": "number", "description": f"Max log lines, default {_DEFAULT_LIMIT}, cap {_MAX_LIMIT}"},
                "direction": {"type": "string", "enum": ["backward", "forward"]},
            },
            "required": ["datasource_uid", "query", "start", "end"],
        },
        handler=_h_query_loki,
    ),
    MCPTool(
        name="grafana_loki_label_values",
        description=(
            "Enumerate the values of a Loki label (e.g. `app`, `namespace`, "
            "`pod`) via the datasource proxy. Use to discover valid stream "
            "selectors before a `grafana_query_loki` call."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "datasource_uid": {"type": "string"},
                "label": {"type": "string", "description": "Label name, e.g. 'app'"},
                "start": {"type": "string"},
                "end": {"type": "string"},
                "limit": {"type": "number"},
            },
            "required": ["datasource_uid", "label"],
        },
        handler=_h_loki_label_values,
    ),
    MCPTool(
        name="grafana_list_alert_rules",
        description=(
            "List Grafana-managed alert rule groups with current firing "
            "state. Returns per-rule name / state (firing|pending|inactive) "
            "/ health, plus totals of firing + pending rules. Use to answer "
            "'what's alerting right now?'."
        ),
        input_schema={"type": "object", "properties": {}},
        handler=_h_list_alert_rules,
    ),
    MCPTool(
        name="grafana_list_contact_points",
        description=(
            "List alerting contact points (name + type, e.g. slack/email/"
            "pagerduty). Secret-bearing settings are intentionally omitted. "
            "Use to see where alerts are routed."
        ),
        input_schema={"type": "object", "properties": {}},
        handler=_h_list_contact_points,
    ),
]


def get_tool(name: str) -> MCPTool:
    for t in GRAFANA_TOOLS:
        if t.name == name:
            return t
    raise KeyError(f"unknown grafana tool: {name}")


# --- fake backend (FR-012; integration tests) ----------------------
#
# Grafana handlers ignore their first (`_unused`) arg and reach the
# module-level `_get`, which builds an httpx client from env via
# `_config()`. The offline fake swaps `_get` for a canned, shape-faithful
# responder -- no network, no GRAFANA_URL / GRAFANA_TOKEN. `build_fake()`
# returns client=None plus a teardown that restores the real `_get`.


async def _fake_get(path: str, params: dict | None = None, *, tool: str = "grafana") -> Any:
    """Canned stand-in for the module-level GET. Routes by path to a
    response shaped like the real Grafana / Prometheus / Loki endpoint the
    handler parses."""
    if path == "/api/search":
        return [
            {
                "uid": "dash-abc",
                "title": "API Overview",
                "url": "/d/dash-abc/api-overview",
                "type": "dash-db",
                "tags": ["api", "prod"],
                "folderTitle": "Services",
            }
        ]
    if path.startswith("/api/dashboards/uid/"):
        return {
            "meta": {"folderTitle": "Services", "url": "/d/dash-abc/api-overview"},
            "dashboard": {
                "uid": "dash-abc",
                "title": "API Overview",
                "tags": ["api", "prod"],
                "panels": [
                    {
                        "id": 1,
                        "title": "Request rate",
                        "type": "timeseries",
                        "targets": [
                            {
                                "datasource": {"uid": "prom-uid", "type": "prometheus"},
                                "expr": "sum(rate(http_requests_total[5m]))",
                                "legendFormat": "rps",
                                "refId": "A",
                            }
                        ],
                    },
                    {
                        "id": 2,
                        "title": "Error rate",
                        "type": "timeseries",
                        "targets": [
                            {
                                "datasource": {"uid": "prom-uid", "type": "prometheus"},
                                "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m]))",
                                "refId": "A",
                            }
                        ],
                    },
                ],
            },
        }
    if path == "/api/datasources":
        return [
            {"name": "Prometheus", "uid": "prom-uid", "type": "prometheus", "isDefault": True},
            {"name": "Loki", "uid": "loki-uid", "type": "loki", "isDefault": False},
        ]
    # Loki label values: /loki/api/v1/label/{label}/values
    # (checked before the generic Prometheus matches below, since the Loki
    # proxy paths also end with `.../values` and `.../query_range`).
    if "/loki/api/v1/label/" in path and path.endswith("/values"):
        return {"status": "success", "data": ["api", "auth", "ingress"]}
    # Prometheus label values: /api/v1/label/{label}/values
    if "/api/v1/label/" in path and path.endswith("/values"):
        return {"status": "success", "data": ["api", "auth", "worker"]}
    # Prometheus instant query via proxy.
    if path.endswith("/api/v1/query"):
        return {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {
                        "metric": {"__name__": "up", "job": "api"},
                        "value": [1716000000, "1"],
                    }
                ],
            },
        }
    # Loki range query via proxy (more specific suffix; check before Prom).
    if path.endswith("/loki/api/v1/query_range"):
        return {
            "status": "success",
            "data": {
                "resultType": "streams",
                "result": [
                    {
                        "stream": {"app": "api", "level": "error"},
                        "values": [
                            ["1716000000000000000", "boom: something failed"],
                            ["1716000001000000000", "retrying request"],
                        ],
                    }
                ],
            },
        }
    # Prometheus range query via proxy (checked AFTER the Loki query_range
    # above, whose path also ends with `/api/v1/query_range`).
    if path.endswith("/api/v1/query_range"):
        return {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {"job": "api"},
                        "values": [[1716000000, "1"], [1716000060, "2"]],
                    }
                ],
            },
        }
    if path == "/api/prometheus/grafana/api/v1/rules":
        return {
            "status": "success",
            "data": {
                "groups": [
                    {
                        "name": "api-alerts",
                        "file": "api",
                        "rules": [
                            {
                                "name": "HighErrorRate",
                                "state": "firing",
                                "health": "ok",
                                "query": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) > 1",
                                "alerts": [{"state": "firing"}],
                            },
                            {
                                "name": "HighLatency",
                                "state": "inactive",
                                "health": "ok",
                                "query": "histogram_quantile(0.99, latency) > 2",
                                "alerts": [],
                            },
                        ],
                    }
                ]
            },
        }
    if path == "/api/v1/provisioning/contact-points":
        return [
            {"name": "sre-slack", "type": "slack", "uid": "cp-1", "disableResolveMessage": False},
            {"name": "oncall-pd", "type": "pagerduty", "uid": "cp-2", "disableResolveMessage": True},
        ]
    return {}


def build_fake():
    """Return a FakeMCP exposing the Grafana tools wired to an offline
    backend. Needs NO GRAFANA_URL / GRAFANA_TOKEN / network: the
    module-level `_get` is swapped for a canned responder and restored by
    `teardown`."""
    import opsrag.mcp.grafana as _mod
    from opsrag.mcp._fake import FakeMCP

    _orig_get = _mod._get
    _mod._get = _fake_get

    def _restore() -> None:
        _mod._get = _orig_get

    return FakeMCP(tools=list(GRAFANA_TOOLS), client=None, teardown=_restore)
