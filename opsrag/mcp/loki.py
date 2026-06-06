"""Grafana Loki MCP-style tools for OpsRAG.

Read-only async tools over the Loki HTTP query API (standalone Loki, not
via Grafana datasource proxy). Reuses the canonical httpx pattern from
``opsrag.mcp.datadog``: a module-level ``_config()`` reads env (raising a
clear ``LokiMCPError`` if a required var is missing), ``_get()`` builds an
httpx client from that config, async handlers ``_h_<verb>`` ignore their
first arg and call ``_get``, a module-level ``LOKI_TOOLS`` registry, a
``get_tool(name)`` helper, and a ``build_fake()`` that swaps ``_get`` for a
canned responder (shape-faithful to the real Loki API) and restores it in
teardown.

## Read-only enforcement

Every tool is an HTTP GET against the Loki query API. No POST / PUT /
DELETE / PATCH anywhere -- Loki has no write endpoint here (ingestion is a
separate push API which we never touch).

## Auth / config

- ``LOKI_URL`` (required) -- base URL, e.g. ``https://loki.example.com``.
- ``LOKI_ORG_ID`` (optional) -- multi-tenant org -> ``X-Scope-OrgID`` header.
- ``LOKI_BEARER_TOKEN`` (optional) -> ``Authorization: Bearer ...``.
- ``LOKI_USERNAME`` / ``LOKI_PASSWORD`` (optional) -> HTTP basic auth.

## Tool list (5 read-only)

| Tool                 | Endpoint                                      |
|----------------------|-----------------------------------------------|
| `loki_query_range`   | GET `/loki/api/v1/query_range`                |
| `loki_query`         | GET `/loki/api/v1/query`                       |
| `loki_labels`        | GET `/loki/api/v1/labels`                      |
| `loki_label_values`  | GET `/loki/api/v1/label/{name}/values`         |
| `loki_series`        | GET `/loki/api/v1/series`                      |
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC
from typing import Any
from urllib.parse import quote

import httpx

from opsrag.mcp.gitlab import MCPTool

_log = logging.getLogger("opsrag.mcp.loki")

_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_LIMIT = 100
_MAX_LIMIT = 1000
_RESULT_TRUNCATE_CHARS = 32000
_LINE_TRUNCATE_CHARS = 4000
# Default look-back window when no explicit time bound is supplied.
_DEFAULT_LOOKBACK_S = 3600

# Redact secrets from log lines / error bodies -- log content can leak tokens.
_REDACT_PATTERNS = [
    (re.compile(r"\bxoxb-[A-Za-z0-9-]{20,}"), "[REDACTED:slack_bot_token]"),
    (re.compile(r"\bxoxp-[A-Za-z0-9-]{20,}"), "[REDACTED:slack_user_token]"),
    (re.compile(r"\bghp_[A-Za-z0-9]{30,}"), "[REDACTED:github_token]"),
    (re.compile(r"\bglpat-[A-Za-z0-9_]{20,}"), "[REDACTED:gitlab_token]"),
    (re.compile(r"\bAKIA[A-Z0-9]{16}\b"), "[REDACTED:aws_access_key]"),
    (re.compile(r"\bAIza[A-Za-z0-9_-]{35}"), "[REDACTED:google_api_key]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]+"), "[REDACTED:jwt]"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}"), "Bearer [REDACTED:token]"),
]


def _redact(text: str) -> str:
    for pat, repl in _REDACT_PATTERNS:
        text = pat.sub(repl, text)
    return text


class LokiMCPError(Exception):
    """Raised on Loki API errors. Wraps upstream status + (redacted) body."""

    def __init__(self, status: int, body: str, *, tool: str | None = None):
        self.status = status
        self.body = _redact(body or "")
        self.tool = tool
        super().__init__(f"[{tool or 'loki'}] {status}: {self.body[:300]}")


@dataclass
class _Config:
    base_url: str
    org_id: str | None
    bearer_token: str | None
    username: str | None
    password: str | None


def _config() -> _Config:
    base_url = (os.environ.get("LOKI_URL") or "").strip().rstrip("/")
    if not base_url:
        raise LokiMCPError(
            0,
            "Loki not configured. Set LOKI_URL (base URL, e.g. "
            "https://loki.example.com). Optional: LOKI_ORG_ID, "
            "LOKI_BEARER_TOKEN, or LOKI_USERNAME/LOKI_PASSWORD.",
            tool="loki",
        )
    org_id = (os.environ.get("LOKI_ORG_ID") or "").strip() or None
    bearer = (os.environ.get("LOKI_BEARER_TOKEN") or "").strip() or None
    username = (os.environ.get("LOKI_USERNAME") or "").strip() or None
    password = os.environ.get("LOKI_PASSWORD")
    return _Config(
        base_url=base_url,
        org_id=org_id,
        bearer_token=bearer,
        username=username,
        password=password if password else None,
    )


def _headers(cfg: _Config) -> dict:
    headers = {"Accept": "application/json"}
    if cfg.org_id:
        headers["X-Scope-OrgID"] = cfg.org_id
    if cfg.bearer_token:
        headers["Authorization"] = f"Bearer {cfg.bearer_token}"
    return headers


async def _get(path: str, params: dict | None = None, *, tool: str = "loki") -> Any:
    cfg = _config()
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    auth = None
    if cfg.username is not None:
        auth = (cfg.username, cfg.password or "")
    async with httpx.AsyncClient(
        headers=_headers(cfg), timeout=_DEFAULT_TIMEOUT_S, auth=auth
    ) as http:
        resp = await http.get(f"{cfg.base_url}{path}", params=clean)
    if resp.status_code >= 400:
        raise LokiMCPError(resp.status_code, resp.text, tool=tool)
    return resp.json() if resp.text else {}


def _clamp(n: int | None, default: int = _DEFAULT_LIMIT, max: int = _MAX_LIMIT) -> int:
    """Clamp a caller-supplied count into [1, max]; None -> default."""
    if n is None:
        return default
    try:
        v = int(n)
    except (TypeError, ValueError):
        return default
    if v < 1:
        return 1
    if v > max:
        return max
    return v


def _truncate_line(text: str, limit: int = _LINE_TRUNCATE_CHARS) -> str:
    if not text:
        return ""
    text = _redact(text)
    if len(text) > limit:
        return text[:limit] + " ...[truncated]"
    return text


def _ns_to_iso(ns: str | int | None) -> str | None:
    """Convert a Loki unix-nanosecond timestamp to an ISO-8601 string."""
    if ns is None:
        return None
    try:
        from datetime import datetime

        secs = int(ns) / 1_000_000_000
        return (
            datetime.fromtimestamp(secs, tz=UTC)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _resolve_time_bounds(args: dict) -> tuple[str, str]:
    """Return (start, end) as unix-nanosecond strings.

    Accepts caller-supplied `start`/`end` verbatim (Loki accepts unix
    ns, unix seconds, or RFC3339). When absent, defaults to a
    `now-1h -> now` window so a query_range ALWAYS has a time bound.
    """
    now_ns = int(time.time() * 1_000_000_000)
    end = args.get("end")
    start = args.get("start")
    if end is None:
        end = str(now_ns)
    if start is None:
        start = str(now_ns - _DEFAULT_LOOKBACK_S * 1_000_000_000)
    return str(start), str(end)


def _parse_streams(resp: dict, *, max_lines: int) -> dict:
    """Parse a Loki query/query_range response into trimmed log lines.

    Real Loki response (curl-verified shape):
      {"status":"success","data":{"resultType":"streams","result":[
        {"stream":{"app":"x","level":"error"},
         "values":[["1700000000000000000","log line"], ...]}]}}

    For metric queries `resultType` is `matrix`/`vector` and each result
    carries `metric` + `values`/`value` of `[unix_seconds, "value"]`.
    """
    data = resp.get("data") or {}
    result_type = data.get("resultType")
    result = data.get("result") or []

    if result_type in ("matrix", "vector"):
        series = []
        for r in result[: max_lines]:
            metric = r.get("metric") or {}
            if result_type == "vector":
                val = r.get("value") or []
                samples = [val] if val else []
            else:
                samples = r.get("values") or []
            series.append({
                "labels": metric,
                "samples": [
                    {"ts": s[0], "value": s[1]}
                    for s in samples[: max_lines]
                    if isinstance(s, (list, tuple)) and len(s) >= 2
                ],
            })
        return {"result_type": result_type, "count": len(series), "series": series}

    # streams (the common log case)
    entries = []
    streams_seen = 0
    for r in result:
        labels = r.get("stream") or {}
        streams_seen += 1
        for v in r.get("values") or []:
            if len(entries) >= max_lines:
                break
            if not (isinstance(v, (list, tuple)) and len(v) >= 2):
                continue
            ts_ns = v[0]
            entries.append({
                "ts_ns": ts_ns,
                "ts": _ns_to_iso(ts_ns),
                "labels": labels,
                "line": _truncate_line(str(v[1])),
            })
        if len(entries) >= max_lines:
            break
    return {
        "result_type": result_type or "streams",
        "streams": streams_seen,
        "count": len(entries),
        "entries": entries,
    }


# --- handlers -------------------------------------------------------


async def _h_query_range(_unused, args: dict) -> Any:
    """`/loki/api/v1/query_range` -- LogQL query over a time window.

    ALWAYS sends a capped `limit` (default 100) and a time bound
    (`start`/`end`; defaults to now-1h -> now). `direction` defaults to
    `backward` (newest first)."""
    query = args.get("query")
    if not query:
        raise LokiMCPError(0, "`query` (a LogQL selector) is required", tool="loki_query_range")
    start, end = _resolve_time_bounds(args)
    limit = _clamp(args.get("limit"))
    direction = args.get("direction") or "backward"
    if direction not in ("backward", "forward"):
        direction = "backward"
    params = {
        "query": query,
        "start": start,
        "end": end,
        "limit": limit,
        "direction": direction,
    }
    if args.get("step") is not None:
        params["step"] = args.get("step")
    resp = await _get("/loki/api/v1/query_range", params=params, tool="loki_query_range")
    parsed = _parse_streams(resp, max_lines=limit)
    return {
        "query": query,
        "start": start,
        "end": end,
        "direction": direction,
        "limit": limit,
        **parsed,
    }


async def _h_query(_unused, args: dict) -> Any:
    """`/loki/api/v1/query` -- instant LogQL query at a single `time`."""
    query = args.get("query")
    if not query:
        raise LokiMCPError(0, "`query` (a LogQL selector) is required", tool="loki_query")
    limit = _clamp(args.get("limit"))
    direction = args.get("direction") or "backward"
    if direction not in ("backward", "forward"):
        direction = "backward"
    params = {
        "query": query,
        "limit": limit,
        "direction": direction,
    }
    if args.get("time") is not None:
        params["time"] = str(args.get("time"))
    resp = await _get("/loki/api/v1/query", params=params, tool="loki_query")
    parsed = _parse_streams(resp, max_lines=limit)
    return {
        "query": query,
        "time": params.get("time"),
        "limit": limit,
        **parsed,
    }


async def _h_labels(_unused, args: dict) -> Any:
    """`/loki/api/v1/labels` -- list all label names in a time window."""
    params: dict = {}
    if args.get("start") is not None:
        params["start"] = str(args.get("start"))
    if args.get("end") is not None:
        params["end"] = str(args.get("end"))
    resp = await _get("/loki/api/v1/labels", params=params, tool="loki_labels")
    labels = resp.get("data") or []
    capped = _clamp(args.get("limit"), default=_MAX_LIMIT)
    labels = list(labels)[:capped]
    return {"count": len(labels), "labels": labels}


async def _h_label_values(_unused, args: dict) -> Any:
    """`/loki/api/v1/label/{name}/values` -- values for one label."""
    name = args.get("name")
    if not name:
        raise LokiMCPError(0, "`name` (the label name) is required", tool="loki_label_values")
    params: dict = {}
    if args.get("start") is not None:
        params["start"] = str(args.get("start"))
    if args.get("end") is not None:
        params["end"] = str(args.get("end"))
    if args.get("query") is not None:
        params["query"] = args.get("query")
    resp = await _get(
        f"/loki/api/v1/label/{quote(str(name), safe='')}/values",
        params=params,
        tool="loki_label_values",
    )
    values = resp.get("data") or []
    capped = _clamp(args.get("limit"), default=_MAX_LIMIT)
    values = list(values)[:capped]
    return {"name": name, "count": len(values), "values": values}


async def _h_series(_unused, args: dict) -> Any:
    """`/loki/api/v1/series` -- list label sets (streams) matching selectors.

    Accepts `match` as a single LogQL stream selector string or a list of
    them; sent as repeated `match[]` query params."""
    match = args.get("match")
    if not match:
        raise LokiMCPError(
            0, "`match` (a LogQL stream selector or list of them) is required",
            tool="loki_series",
        )
    if isinstance(match, str):
        match = [match]
    start, end = _resolve_time_bounds(args)
    params = {
        "match[]": list(match),
        "start": start,
        "end": end,
    }
    resp = await _get("/loki/api/v1/series", params=params, tool="loki_series")
    data = resp.get("data") or []
    capped = _clamp(args.get("limit"), default=_MAX_LIMIT)
    data = list(data)[:capped]
    return {"start": start, "end": end, "count": len(data), "series": data}


# --- tool registry --------------------------------------------------


LOKI_TOOLS: list[MCPTool] = [
    MCPTool(
        name="loki_query_range",
        description=(
            "Run a LogQL query over a time window (Grafana Loki). Use this for "
            "'show me error logs for app X in the last hour'. ALWAYS bounded by "
            "`limit` (default 100, cap 1000) and a time range (`start`/`end`; "
            "defaults to now-1h -> now). Returns trimmed log lines with their "
            "stream labels + timestamps. `direction` defaults to `backward` "
            "(newest first). Example query: '{app=\"acme-notes-be\"} |= "
            "\"error\"'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "LogQL query, e.g. '{app=\"x\"} |= \"error\"'"},
                "start": {"type": "string", "description": "Start time: unix ns, unix seconds, or RFC3339. Default now-1h."},
                "end": {"type": "string", "description": "End time. Default now."},
                "limit": {"type": "number", "description": "Max log lines (default 100, cap 1000)."},
                "direction": {"type": "string", "enum": ["backward", "forward"], "description": "Default backward (newest first)."},
                "step": {"type": "string", "description": "Query resolution step for metric queries (e.g. '60s')."},
            },
            "required": ["query"],
        },
        handler=_h_query_range,
    ),
    MCPTool(
        name="loki_query",
        description=(
            "Run an instant LogQL query at a single point in `time` (Grafana "
            "Loki). Best for metric-style LogQL (rates, counts) evaluated at "
            "one instant. For browsing raw log lines over a window, prefer "
            "`loki_query_range`. `limit` capped (default 100)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "LogQL query."},
                "time": {"type": "string", "description": "Evaluation time: unix ns, unix seconds, or RFC3339. Default now."},
                "limit": {"type": "number", "description": "Max log lines (default 100, cap 1000)."},
                "direction": {"type": "string", "enum": ["backward", "forward"]},
            },
            "required": ["query"],
        },
        handler=_h_query,
    ),
    MCPTool(
        name="loki_labels",
        description=(
            "List all label NAMES known to Loki in a time window (e.g. `app`, "
            "`namespace`, `level`). Use this to discover what you can filter "
            "on before building a LogQL query. Optional `start`/`end` narrow "
            "the window."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "start": {"type": "string", "description": "Start time (unix ns/seconds or RFC3339)."},
                "end": {"type": "string", "description": "End time."},
                "limit": {"type": "number", "description": "Max label names returned (cap 1000)."},
            },
        },
        handler=_h_labels,
    ),
    MCPTool(
        name="loki_label_values",
        description=(
            "List the VALUES of one label `name` (e.g. all values of `app` or "
            "`namespace`). Use after `loki_labels` to enumerate concrete "
            "values for a LogQL selector. Optional `query` scopes values to "
            "streams matching a selector."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Label name, e.g. 'app'."},
                "start": {"type": "string", "description": "Start time."},
                "end": {"type": "string", "description": "End time."},
                "query": {"type": "string", "description": "Optional LogQL selector to scope values."},
                "limit": {"type": "number", "description": "Max values (cap 1000)."},
            },
            "required": ["name"],
        },
        handler=_h_label_values,
    ),
    MCPTool(
        name="loki_series",
        description=(
            "List the label sets (streams) that match one or more LogQL stream "
            "selectors. Use this to see which concrete streams exist for a "
            "selector like '{namespace=\"prod\"}' before querying their logs. "
            "`match` is a single selector string or a list of them."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "match": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "LogQL stream selector(s), e.g. '{app=\"x\"}'.",
                },
                "start": {"type": "string", "description": "Start time. Default now-1h."},
                "end": {"type": "string", "description": "End time. Default now."},
                "limit": {"type": "number", "description": "Max series (cap 1000)."},
            },
            "required": ["match"],
        },
        handler=_h_series,
    ),
]


def get_tool(name: str) -> MCPTool:
    for t in LOKI_TOOLS:
        if t.name == name:
            return t
    raise KeyError(f"unknown loki tool: {name}")


# --- fake backend (FR-012; integration tests) ----------------------
#
# Loki handlers ignore their first (`_unused`) arg and reach the
# module-level `_get`, which builds an httpx client from env via
# `_config()`. So the offline fake replaces that one module function with
# a canned, shape-faithful responder -- no network, no LOKI_URL needed.
# `build_fake()` returns client=None (handlers discard it) plus a teardown
# that restores the real `_get`.


async def _fake_get(path: str, params: dict | None = None, *, tool: str = "loki") -> Any:
    """Canned stand-in for the module-level GET. Routes by path to a
    response shaped like the real Loki HTTP API the handler parses."""
    if path == "/loki/api/v1/query_range" or path == "/loki/api/v1/query":
        return {
            "status": "success",
            "data": {
                "resultType": "streams",
                "result": [
                    {
                        "stream": {
                            "app": "acme-notes-be",
                            "namespace": "prod",
                            "level": "error",
                        },
                        "values": [
                            ["1700000000000000000", "boom: unhandled RuntimeError"],
                            ["1699999999000000000", "starting request handler"],
                        ],
                    }
                ],
                "stats": {},
            },
        }
    if path == "/loki/api/v1/labels":
        return {
            "status": "success",
            "data": ["app", "namespace", "level", "pod"],
        }
    if path.startswith("/loki/api/v1/label/") and path.endswith("/values"):
        return {
            "status": "success",
            "data": ["acme-notes-be", "acme-auth", "acme-web"],
        }
    if path == "/loki/api/v1/series":
        return {
            "status": "success",
            "data": [
                {"app": "acme-notes-be", "namespace": "prod", "level": "error"},
                {"app": "acme-notes-be", "namespace": "prod", "level": "info"},
            ],
        }
    return {"status": "success", "data": {}}


def build_fake():
    """Return a FakeMCP exposing the Loki tools wired to an offline backend.
    Needs NO LOKI_URL / network: the module-level `_get` is swapped for a
    canned responder and restored by `teardown`."""
    import opsrag.mcp.loki as _mod
    from opsrag.mcp._fake import FakeMCP

    _orig_get = _mod._get
    _mod._get = _fake_get

    def _restore() -> None:
        _mod._get = _orig_get

    return FakeMCP(tools=list(LOKI_TOOLS), client=None, teardown=_restore)
