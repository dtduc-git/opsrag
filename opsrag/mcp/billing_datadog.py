"""Datadog cost/usage MCP connector — Billing category, read-only.

Read-only FinOps tools over the **Datadog Usage Metering / Cost API**
(`/api/v2/usage/*` + `/api/v2/cost_by_tag/*`). Reuses the SAME Datadog
auth as `opsrag.mcp.datadog`: `DD_API_KEY` + `DD_APP_KEY` + `DD_SITE`
(default `datadoghq.com`), `DD-API-KEY` / `DD-APPLICATION-KEY` headers,
base URL `https://api.{DD_SITE}`.

## Scopes (important)

Every endpoint here is under Datadog Usage Metering. The **APP key must
carry both `usage_read` AND `billing_read` scopes**, and it must belong
to the **parent organization** (multi-org accounts expose cost only at
the parent). A key missing either scope — or a child-org key — gets a
`403`; the `_get` choke-point turns that into a clear, actionable error.

## Estimates & lag

All figures are ESTIMATES. Datadog usage/cost data lags by **up to ~72
hours**; `estimated_cost` and `projected_cost` in particular are
best-effort and shift as data finalizes. Only `historical_cost` returns
finalized past-month numbers. Every tool echoes this in a `note` field
and in its description so the agent never presents them as exact.

## Read-only enforcement

Every tool issues a single `httpx.AsyncClient.get` through the
module-level `_get` seam. No POST / PUT / DELETE / PATCH anywhere — this
connector can only read usage/cost, never mutate anything.

## Tool list (5 read-only)

| Tool                              | Endpoint                                       |
|-----------------------------------|------------------------------------------------|
| `billing_datadog_estimated_cost`  | GET `/api/v2/usage/estimated_cost`             |
| `billing_datadog_historical_cost` | GET `/api/v2/usage/historical_cost`            |
| `billing_datadog_projected_cost`  | GET `/api/v2/usage/projected_cost`             |
| `billing_datadog_hourly_usage`    | GET `/api/v2/usage/hourly_usage`               |
| `billing_datadog_cost_by_tag`     | GET `/api/v2/cost_by_tag/monthly_cost_attribution` |
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from opsrag.mcp.gitlab import MCPTool

_log = logging.getLogger("opsrag.mcp.billing_datadog")

DEFAULT_DD_SITE = "datadoghq.com"
_DEFAULT_TIMEOUT_S = 30.0
_ESTIMATE_NOTE = (
    "Datadog cost/usage figures are ESTIMATES and lag up to ~72h. Only "
    "historical_cost is finalized. Do not present as exact billing."
)

# 'YYYY-MM' month strings, validated/clamped before hitting the API.
_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")

# Bound the hourly-usage window so an agent can't request a giant payload.
_DEFAULT_USAGE_HOURS = 24
_MAX_USAGE_HOURS = 168  # 7 days
_DEFAULT_PRODUCT_FAMILIES = "infra_hosts,apm_hosts"

# Redact secrets that can leak in an upstream error body before it reaches
# the LLM (same pattern family as the other Datadog / log-bearing connectors).
_REDACT_PATTERNS = [
    (re.compile(r"\bxoxb-[A-Za-z0-9-]{20,}"), "[REDACTED:slack_bot_token]"),
    (re.compile(r"\bghp_[A-Za-z0-9]{30,}"), "[REDACTED:github_token]"),
    (re.compile(r"\bglpat-[A-Za-z0-9_]{20,}"), "[REDACTED:gitlab_token]"),
    (re.compile(r"\bAKIA[A-Z0-9]{16}\b"), "[REDACTED:aws_access_key]"),
    (re.compile(r"\bAIza[A-Za-z0-9_-]{35}"), "[REDACTED:google_api_key]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]+"), "[REDACTED:jwt]"),
    (re.compile(r"\bddapp_[A-Za-z0-9_]{30,}"), "[REDACTED:dd_app_key]"),
]


def _redact(text: str) -> str:
    for pat, repl in _REDACT_PATTERNS:
        text = pat.sub(repl, text)
    return text


class DatadogBillingMCPError(Exception):
    """Read-only Datadog cost/usage tool failure. Carries a short ``reason``
    code: ``bad_config`` / ``bad_args`` / ``forbidden`` (scope) /
    ``rate_limited`` / ``http``. Any secrets in the wrapped body are redacted."""

    def __init__(self, message: str, *, reason: str = "error") -> None:
        super().__init__(_redact(message))
        self.reason = reason


# --- config (env; identical vars to opsrag.mcp.datadog) --------------------


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
        raise DatadogBillingMCPError(
            "Datadog credentials not set. Need DD_API_KEY + DD_APP_KEY. The "
            "APP key must carry `usage_read` + `billing_read` scopes and "
            "belong to the parent organization.",
            reason="bad_config",
        )
    return _Config(api_key=api_key, app_key=app_key, api_url=f"https://api.{site}")


def _headers() -> dict:
    cfg = _config()
    return {
        "DD-API-KEY": cfg.api_key,
        "DD-APPLICATION-KEY": cfg.app_key,
        "Content-Type": "application/json",
    }


# --- HTTP choke-point (swapped by build_fake) ------------------------------


async def _get(path: str, params: dict | None = None, *, tool: str = "billing_datadog") -> Any:
    """The single network seam — a read-only GET against the Datadog usage/cost
    API. Turns a 403 into a clear scope error, a 429 into a rate-limit error
    (reading `X-RateLimit-*` headers), and any other >=400 into a generic http
    error. `build_fake` swaps this module-level function."""
    cfg = _config()
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    async with httpx.AsyncClient(headers=_headers(), timeout=_DEFAULT_TIMEOUT_S) as http:
        resp = await http.get(f"{cfg.api_url}{path}", params=clean)

    if resp.status_code == 403:
        raise DatadogBillingMCPError(
            f"[{tool}] 403 Forbidden. The Datadog APP key lacks the "
            "`usage_read`/`billing_read` scopes, or it is not a parent-org "
            "key (usage/cost is exposed only at the parent organization). "
            f"Upstream: {_redact(resp.text or '')[:200]}",
            reason="forbidden",
        )
    if resp.status_code == 429:
        rl = {
            k: v for k, v in resp.headers.items()
            if k.lower().startswith("x-ratelimit")
        }
        raise DatadogBillingMCPError(
            f"[{tool}] 429 rate limited by the Datadog usage API. "
            f"Rate-limit headers: {rl}. Retry after the reset window.",
            reason="rate_limited",
        )
    if resp.status_code >= 400:
        raise DatadogBillingMCPError(
            f"[{tool}] {resp.status_code}: {_redact(resp.text or '')[:300]}",
            reason="http",
        )
    return resp.json() if resp.text else {}


# --- pure helpers ----------------------------------------------------------


def _current_month() -> str:
    return datetime.now(UTC).strftime("%Y-%m")


def _valid_month(m: str | None, *, default: str | None = None) -> str:
    """Validate/clamp a 'YYYY-MM' string; fall back to `default` or this month."""
    if m and _MONTH_RE.match(str(m).strip()):
        return str(m).strip()
    return default or _current_month()


def _summarize_cost(data: list | None) -> list[dict]:
    """Compact a Datadog cost payload (estimated/historical/projected share a
    shape) into per-org totals + a product breakdown. Defensive about the
    two field naming conventions: `cost`/`total_cost` for estimated &
    historical, `projected_cost`/`projected_total_cost` for projected."""
    orgs: list[dict] = []
    for row in data or []:
        a = row.get("attributes") or {}
        charges = []
        for c in a.get("charges") or []:
            cost = c.get("cost")
            if cost is None:
                cost = c.get("projected_cost")
            charges.append({
                "product": c.get("product_name"),
                "charge_type": c.get("charge_type"),
                "cost_usd": round(float(cost or 0), 2),
            })
        charges.sort(key=lambda x: x["cost_usd"], reverse=True)
        total = a.get("total_cost")
        if total is None:
            total = a.get("projected_total_cost")
        orgs.append({
            "org_name": a.get("org_name"),
            "public_id": a.get("public_id"),
            "date": a.get("date") or a.get("month"),
            "region": a.get("region"),
            "total_cost_usd": round(float(total or 0), 2),
            "charges": charges,
        })
    return orgs


# --- handlers --------------------------------------------------------------


async def _h_estimated_cost(_unused, args: dict) -> Any:
    """Current (month-to-date) ESTIMATED cost, broken down by product / sub-org.
    Estimate — lags up to ~72h. `view=summary` collapses per-hour rows."""
    start_month = _valid_month(args.get("start_month"))
    end_month = _valid_month(args.get("end_month"), default=start_month)
    params = {
        "view": "summary",
        "start_month": start_month,
        "end_month": end_month,
    }
    resp = await _get("/api/v2/usage/estimated_cost", params=params,
                      tool="billing_datadog_estimated_cost")
    return {
        "start_month": start_month,
        "end_month": end_month,
        "currency": "USD",
        "orgs": _summarize_cost(resp.get("data")),
        "note": _ESTIMATE_NOTE,
    }


async def _h_historical_cost(_unused, args: dict) -> Any:
    """Finalized past-month cost, broken down by product / sub-org. This is the
    only endpoint here that returns FINALIZED (not estimated) numbers."""
    start_month = _valid_month(args.get("start_month"))
    end_month = _valid_month(args.get("end_month"), default=start_month)
    params = {
        "view": "summary",
        "start_month": start_month,
        "end_month": end_month,
    }
    resp = await _get("/api/v2/usage/historical_cost", params=params,
                      tool="billing_datadog_historical_cost")
    return {
        "start_month": start_month,
        "end_month": end_month,
        "currency": "USD",
        "orgs": _summarize_cost(resp.get("data")),
        "note": "Finalized past-month cost (not an estimate). Net of Datadog credits.",
    }


async def _h_projected_cost(_unused, args: dict) -> Any:
    """End-of-current-month cost PROJECTION, by product / sub-org. Projection
    (straight-line, per Datadog) — shifts as usage data lands (~72h lag)."""
    params = {"view": "summary"}
    resp = await _get("/api/v2/usage/projected_cost", params=params,
                      tool="billing_datadog_projected_cost")
    return {
        "month": _current_month(),
        "currency": "USD",
        "orgs": _summarize_cost(resp.get("data")),
        "note": _ESTIMATE_NOTE,
    }


def _clamp_usage_window(start: str | None, end: str | None) -> tuple[str, str]:
    """Return an ISO8601 (start, end) window for hourly usage, clamped to
    <= _MAX_USAGE_HOURS. Defaults to the last _DEFAULT_USAGE_HOURS."""
    now = datetime.now(UTC)

    def _iso(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _parse(s: str | None) -> datetime | None:
        if not s:
            return None
        try:
            return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(UTC)
        except (TypeError, ValueError):
            return None

    end_dt = _parse(end) or now
    start_dt = _parse(start) or (end_dt - timedelta(hours=_DEFAULT_USAGE_HOURS))
    if start_dt >= end_dt:
        start_dt = end_dt - timedelta(hours=_DEFAULT_USAGE_HOURS)
    # Clamp an over-wide window from the END back, keeping the requested end.
    if (end_dt - start_dt) > timedelta(hours=_MAX_USAGE_HOURS):
        start_dt = end_dt - timedelta(hours=_MAX_USAGE_HOURS)
    return _iso(start_dt), _iso(end_dt)


async def _h_hourly_usage(_unused, args: dict) -> Any:
    """Hourly usage timeseries for given product families (e.g. `infra_hosts`,
    `apm_hosts`). Usage — not cost — and lags up to ~72h. Window clamped to
    <= 7 days to bound payload size."""
    families = (args.get("product_families") or _DEFAULT_PRODUCT_FAMILIES).strip()
    start_iso, end_iso = _clamp_usage_window(args.get("start"), args.get("end"))
    params = {
        "filter[timestamp][start]": start_iso,
        "filter[timestamp][end]": end_iso,
        "filter[product_families]": families,
    }
    resp = await _get("/api/v2/usage/hourly_usage", params=params,
                      tool="billing_datadog_hourly_usage")
    # Response: data[].attributes = {org_name, product_family, region,
    # timestamp, measurements:[{usage_type, value}]}. Roll up per family.
    per_family: dict[str, dict[str, Any]] = {}
    for row in resp.get("data") or []:
        a = row.get("attributes") or {}
        fam = a.get("product_family") or "unknown"
        agg = per_family.setdefault(fam, {"product_family": fam, "points": 0, "total_usage": 0.0})
        agg["points"] += 1
        for m in a.get("measurements") or []:
            v = m.get("value")
            if isinstance(v, int | float):
                agg["total_usage"] += float(v)
    families_out = [
        {**v, "total_usage": round(v["total_usage"], 4)}
        for v in sorted(per_family.values(), key=lambda x: x["product_family"])
    ]
    return {
        "start": start_iso,
        "end": end_iso,
        "product_families": families,
        "families": families_out,
        "note": _ESTIMATE_NOTE,
    }


async def _h_cost_by_tag(_unused, args: dict) -> Any:
    """Monthly cost attribution split by tag values (FinOps showback). Estimate,
    ~72h lag.

    NOTE: `/api/v2/cost_by_tag/monthly_cost_attribution` requires a `fields`
    param naming product-specific columns (e.g. `<product>_percentage_in_org`).
    The exact field names and the response shape are org/product dependent, so
    this handler is implemented DEFENSIVELY: it passes a caller-supplied (or
    defaulted) `fields`, and summarizes whatever numeric attributes come back.
    Verify field names against the live account before relying on it.
    """
    start_month = _valid_month(args.get("start_month"))
    end_month = _valid_month(args.get("end_month"), default=start_month)
    tag_keys = (args.get("tag_keys") or "").strip()
    if not tag_keys:
        raise DatadogBillingMCPError(
            "tag_keys is required (comma-separated tag keys to break cost down "
            "by, e.g. 'team,service').",
            reason="bad_args",
        )
    # `fields` is required by the endpoint but is product-specific; allow an
    # override, otherwise fall back to a broad default and flag it in the note.
    fields = (args.get("fields") or "total_cost,total_percentage_in_org").strip()
    params = {
        "start_month": start_month,
        "end_month": end_month,
        "fields": fields,
        "tag_breakdown_keys": tag_keys,
    }
    resp = await _get("/api/v2/cost_by_tag/monthly_cost_attribution", params=params,
                      tool="billing_datadog_cost_by_tag")
    rows = []
    for row in resp.get("data") or []:
        a = row.get("attributes") or {}
        # tags may be under `tags` (dict) or `tag_breakdown`; numeric cost/pct
        # columns are sprinkled directly on `attributes` and/or under `values`.
        numeric = {k: v for k, v in a.items() if isinstance(v, int | float)}
        rows.append({
            "month": a.get("month"),
            "tags": a.get("tags") or a.get("tag_breakdown") or {},
            "values": a.get("values") or numeric or {},
        })
    return {
        "start_month": start_month,
        "end_month": end_month,
        "tag_keys": tag_keys,
        "fields": fields,
        "rows": rows,
        "note": (
            _ESTIMATE_NOTE + " cost_by_tag `fields` are product-specific; "
            "adjust the `fields` arg if columns look wrong."
        ),
    }


# --- tool specs ------------------------------------------------------------

_MONTH_PROP = {
    "start_month": {"type": "string", "description": "Start month 'YYYY-MM' (default: current month)."},
    "end_month": {"type": "string", "description": "End month 'YYYY-MM' (default: same as start_month)."},
}

BILLING_DATADOG_TOOLS: list[MCPTool] = [
    MCPTool(
        name="billing_datadog_estimated_cost",
        description=(
            "Datadog current-month ESTIMATED cost, broken down by product "
            "(APM, Infra, Logs, ...) and sub-org. ESTIMATE — lags up to ~72h, "
            "not a final bill. Read-only (Datadog Usage Metering API). Args: "
            "optional `start_month` / `end_month` ('YYYY-MM')."
        ),
        input_schema={"type": "object", "properties": {**_MONTH_PROP}},
        handler=_h_estimated_cost,
    ),
    MCPTool(
        name="billing_datadog_historical_cost",
        description=(
            "Datadog FINALIZED past-month cost by product / sub-org — the "
            "closed-book numbers (unlike estimated/projected). Read-only. Args: "
            "`start_month`, `end_month` ('YYYY-MM')."
        ),
        input_schema={"type": "object", "properties": {**_MONTH_PROP}},
        handler=_h_historical_cost,
    ),
    MCPTool(
        name="billing_datadog_projected_cost",
        description=(
            "Datadog end-of-current-month cost PROJECTION by product / sub-org. "
            "Projection — shifts as usage lands (~72h lag). Read-only. No args."
        ),
        input_schema={"type": "object", "properties": {}},
        handler=_h_projected_cost,
    ),
    MCPTool(
        name="billing_datadog_hourly_usage",
        description=(
            "Datadog hourly USAGE timeseries for one or more product families "
            "(e.g. `infra_hosts,apm_hosts`), rolled up per family. Usage (not "
            "cost); lags up to ~72h. Window clamped to <=7 days. Read-only. "
            "Args: `product_families` (comma str), `start`/`end` (ISO8601)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "product_families": {
                    "type": "string",
                    "description": "Comma-separated families, e.g. 'infra_hosts,apm_hosts'. Default 'infra_hosts,apm_hosts'.",
                },
                "start": {"type": "string", "description": "ISO8601 start (default: 24h before end)."},
                "end": {"type": "string", "description": "ISO8601 end (default: now)."},
            },
        },
        handler=_h_hourly_usage,
    ),
    MCPTool(
        name="billing_datadog_cost_by_tag",
        description=(
            "Datadog monthly cost ATTRIBUTION split by tag values (FinOps "
            "showback, e.g. by 'team' or 'service'). ESTIMATE, ~72h lag. "
            "Read-only. Args: `start_month`, `end_month` ('YYYY-MM'), "
            "`tag_keys` (comma str). Field names are product-specific — see "
            "the returned `note`."
        ),
        input_schema={
            "type": "object",
            "properties": {
                **_MONTH_PROP,
                "tag_keys": {
                    "type": "string",
                    "description": "Comma-separated tag keys to break cost down by, e.g. 'team,service'.",
                },
                "fields": {
                    "type": "string",
                    "description": "Optional product-specific cost columns (advanced). Default 'total_cost,total_percentage_in_org'.",
                },
            },
            "required": ["tag_keys"],
        },
        handler=_h_cost_by_tag,
    ),
]


def get_tool(name: str) -> MCPTool:
    for t in BILLING_DATADOG_TOOLS:
        if t.name == name:
            return t
    raise KeyError(f"unknown billing_datadog tool: {name}")


# --- fake backend (FR-012; offline tests) ----------------------------------
#
# Handlers ignore their first (`_unused`) arg and reach the module-level
# `_get`, which builds an httpx client from env via `_config()`. So the
# offline fake swaps `_get` for a canned dispatcher keyed by path fragment —
# no network, no DD keys. `build_fake()` returns client=None (handlers discard
# it) plus a teardown that restores the real `_get`.


async def _fake_get(path: str, params: dict | None = None, *, tool: str = "billing_datadog") -> Any:
    """Canned stand-in for the module-level GET. Routes by path to a response
    shaped like the real Datadog usage/cost endpoint the handler parses."""
    if path == "/api/v2/usage/estimated_cost":
        return {
            "data": [
                {
                    "attributes": {
                        "org_name": "acme",
                        "public_id": "abcdef123",
                        "date": "2026-07-01",
                        "charges": [
                            {"product_name": "apm", "charge_type": "on_demand", "cost": 812.0},
                            {"product_name": "infra_hosts", "charge_type": "committed", "cost": 14556.0},
                        ],
                        "total_cost": 15368.0,
                    }
                }
            ]
        }
    if path == "/api/v2/usage/historical_cost":
        return {
            "data": [
                {
                    "attributes": {
                        "org_name": "acme",
                        "public_id": "abcdef123",
                        "date": "2026-06-01",
                        "charges": [
                            {"product_name": "logs", "charge_type": "on_demand", "cost": 4200.0},
                        ],
                        "total_cost": 14980.0,
                    }
                }
            ]
        }
    if path == "/api/v2/usage/projected_cost":
        return {
            "data": [
                {
                    "attributes": {
                        "org_name": "acme",
                        "public_id": "abcdef123",
                        "charges": [
                            {"product_name": "apm", "charge_type": "on_demand", "projected_cost": 1700.0},
                        ],
                        "projected_total_cost": 31200.0,
                    }
                }
            ]
        }
    if path == "/api/v2/usage/hourly_usage":
        return {
            "data": [
                {
                    "attributes": {
                        "org_name": "acme",
                        "product_family": "infra_hosts",
                        "region": "us",
                        "timestamp": "2026-07-08T00:00:00Z",
                        "measurements": [{"usage_type": "host_count", "value": 120}],
                    }
                },
                {
                    "attributes": {
                        "org_name": "acme",
                        "product_family": "infra_hosts",
                        "region": "us",
                        "timestamp": "2026-07-08T01:00:00Z",
                        "measurements": [{"usage_type": "host_count", "value": 118}],
                    }
                },
                {
                    "attributes": {
                        "org_name": "acme",
                        "product_family": "apm_hosts",
                        "region": "us",
                        "timestamp": "2026-07-08T00:00:00Z",
                        "measurements": [{"usage_type": "host_count", "value": 40}],
                    }
                },
            ]
        }
    if path == "/api/v2/cost_by_tag/monthly_cost_attribution":
        return {
            "data": [
                {
                    "attributes": {
                        "month": "2026-07-01T00:00:00Z",
                        "tags": {"team": "platform"},
                        "total_cost": 9200.0,
                        "total_percentage_in_org": 59.9,
                    }
                },
                {
                    "attributes": {
                        "month": "2026-07-01T00:00:00Z",
                        "tags": {"team": "data"},
                        "total_cost": 6168.0,
                        "total_percentage_in_org": 40.1,
                    }
                },
            ]
        }
    return {}


def build_fake():
    """Return a FakeMCP exposing the Datadog billing tools wired to an offline
    backend. Needs NO DD keys / network: the module-level `_get` is swapped for
    a canned dispatcher and restored by `teardown`."""
    import opsrag.mcp.billing_datadog as _mod
    from opsrag.mcp._fake import FakeMCP

    _orig = _mod._get
    _mod._get = _fake_get

    def _restore() -> None:
        _mod._get = _orig

    return FakeMCP(tools=list(BILLING_DATADOG_TOOLS), client=None, teardown=_restore)
