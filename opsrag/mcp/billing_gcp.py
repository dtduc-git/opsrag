"""GCP billing/cost MCP connector — Billing category, restricted.

Read-only FinOps tools over the **standard Cloud Billing BigQuery export**
(`gcp_billing_export_v1_*`). GCP has no REST "get my charges" endpoint; the
BigQuery export is the official programmatic source of actual spend (the
Cloud Billing REST API only returns SKU catalog / list prices). SQL patterns
follow the standard Cloud Billing export schema (net cost = cost + credits).

Config (env vars; the export location is deployment-specific):
  ``OPSRAG_GCP_BILLING_TABLE``    REQUIRED. Fully-qualified wildcard export
       table, e.g. ``my-proj.billing_dataset.gcp_billing_export_v1_*`` (union
       of all billing accounts; the standard, NOT ``..._resource_v1_*``).
  ``OPSRAG_GCP_BILLING_PROJECT``  BQ project to run query jobs in (billed for
       the scan). Defaults to the table's project.
  ``OPSRAG_GCP_BILLING_MAX_BYTES``  per-query ``maximum_bytes_billed`` guard
       (default 40 GB) — a hard ceiling so an agent query can't run away.
  ``OPSRAG_GCP_BILLING_ENV_MAP``  optional JSON ``{"project-id": "env"}`` for
       tagging the by-project breakdown with an environment label.
Auth: **ADC / Workload Identity** — the runtime SA needs ``roles/bigquery.jobUser``
(on the query project) + ``roles/bigquery.dataViewer`` on the export dataset.

Safety: net spend = ``cost + credits`` (discounts/CUDs are negative). Every
query is **date-bounded, byte-capped, and parameterized** for agent-supplied
strings — the agent never supplies raw SQL, only tool args.
"""
from __future__ import annotations

import calendar
import json
import logging
import os
import re
from datetime import date, timedelta
from typing import Any

from opsrag.mcp.gitlab import MCPTool

_log = logging.getLogger("opsrag.mcp.billing_gcp")

_DEFAULT_MAX_BYTES = 40_000_000_000  # 40 GB scan ceiling per query
_DEFAULT_TOP_N = 8
_MAX_TOP_N = 50
_MAX_TREND_DAYS = 90

# Net cost = list cost plus credits (discounts/CUD are negative amounts).
_NET = "cost + IFNULL((SELECT SUM(c.amount) FROM UNNEST(credits) c), 0)"

# invoice.month is 'YYYYMM'.
_MONTH_RE = re.compile(r"^\d{6}$")
# Operator-supplied table name (trusted; from env, not the agent). Allow the
# BQ FQN charset only, then backtick it.
_TABLE_RE = re.compile(r"^[A-Za-z0-9_.\-*]+$")


class BillingGcpMCPError(Exception):
    """Read-only GCP billing tool failure. Carries a short ``reason`` code
    (``bad_config`` / ``bad_args`` / ``query``)."""

    def __init__(self, message: str, *, reason: str = "error") -> None:
        super().__init__(message)
        self.reason = reason


# --- config (Helm values via the bound config block; env vars are fallback) --
#
# `bind()` is called once at startup with the `mcp.billing_gcp` config block
# (populated from Helm values -> config.yaml), so the table/project/env_map/
# max_bytes are operator-configured, never hardcoded. Env vars only fill in
# when a config field is unset, so a values-free / env-only deploy still works.
_BOUND: Any | None = None


def bind(cfg: Any | None = None) -> None:
    """Register the billing_gcp config block (or None to clear)."""
    global _BOUND
    _BOUND = cfg


def _cfg(field: str) -> Any:
    return getattr(_BOUND, field, None) if _BOUND is not None else None


def _raw_table() -> str:
    return (str(_cfg("table") or "").strip()
            or (os.environ.get("OPSRAG_GCP_BILLING_TABLE") or "").strip())


def _table() -> str:
    raw = _raw_table()
    if not raw:
        raise BillingGcpMCPError(
            "billing_gcp table is not configured. Set `mcp.billing_gcp.table` "
            "in Helm values (or OPSRAG_GCP_BILLING_TABLE) to the standard Cloud "
            "Billing BigQuery export, e.g. `proj.dataset.gcp_billing_export_v1_*`.",
            reason="bad_config",
        )
    if not _TABLE_RE.match(raw.strip("`")):
        raise BillingGcpMCPError(
            f"billing_gcp table {raw!r} has unexpected characters.",
            reason="bad_config",
        )
    return f"`{raw.strip('`')}`"


def _bq_project() -> str | None:
    proj = (str(_cfg("project") or "").strip()
            or (os.environ.get("OPSRAG_GCP_BILLING_PROJECT") or "").strip())
    if proj:
        return proj
    # default: the table's leading project segment
    raw = _raw_table().strip("`")
    return raw.split(".", 1)[0] if "." in raw else None


def _max_bytes() -> int:
    if _cfg("max_bytes"):
        try:
            return int(_cfg("max_bytes"))
        except (TypeError, ValueError):
            pass
    try:
        return int(os.environ.get("OPSRAG_GCP_BILLING_MAX_BYTES") or _DEFAULT_MAX_BYTES)
    except ValueError:
        return _DEFAULT_MAX_BYTES


def _env_map() -> dict[str, str]:
    cfg_map = _cfg("env_map")
    if isinstance(cfg_map, dict) and cfg_map:
        return {str(k): str(v) for k, v in cfg_map.items()}
    raw = (os.environ.get("OPSRAG_GCP_BILLING_ENV_MAP") or "").strip()
    if not raw:
        return {}
    try:
        m = json.loads(raw)
        return {str(k): str(v) for k, v in m.items()} if isinstance(m, dict) else {}
    except (ValueError, TypeError):
        return {}


# --- pure helpers (unit-testable) ------------------------------------------

def _clamp(n: Any, *, default: int, lo: int = 1, hi: int) -> int:
    try:
        v = int(n)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _valid_month(m: str | None) -> str:
    """Return a validated 'YYYYMM' invoice month, or the current month."""
    if m and _MONTH_RE.match(str(m).strip()):
        return str(m).strip()
    return date.today().strftime("%Y%m")


def _prev_month(m: str) -> str:
    y, mo = int(m[:4]), int(m[4:])
    first = date(y, mo, 1)
    prev = first - timedelta(days=1)
    return prev.strftime("%Y%m")

def _partition_floor(month: str) -> str:
    """`_PARTITIONTIME` lower bound (YYYY-MM-DD) for an invoice month: the first
    of the month minus a 2-day buffer. Billing rows for invoice month M are
    ingested during/after M, so this prunes all earlier partitions without
    dropping in-month rows (the exact `invoice.month` filter stays authoritative)."""
    y, mo = int(month[:4]), int(month[4:])
    return (date(y, mo, 1) - timedelta(days=2)).strftime("%Y-%m-%d")


def project_month(mtd_usd: float, *, day_of_month: int, days_in_month: int) -> float:
    """Extrapolate month-to-date spend to a full-month projection."""
    if day_of_month <= 0:
        return mtd_usd
    return mtd_usd / day_of_month * days_in_month


def trend_pct(*, projected: float, prev: float) -> float:
    if prev <= 0:
        return 0.0
    return (projected - prev) / prev * 100.0


# --- SQL builders (pure; agent strings go through @params, ints are clamped) ---

def sql_month_total(table: str) -> str:
    return f"SELECT ROUND(SUM({_NET}), 2) AS cost FROM {table} WHERE _PARTITIONTIME >= TIMESTAMP(@pfloor) AND invoice.month = @month"


def sql_yesterday_total(table: str) -> str:
    return (f"SELECT ROUND(SUM({_NET}), 2) AS cost FROM {table} "
            f"WHERE _PARTITIONTIME >= TIMESTAMP(DATE_SUB(CURRENT_DATE(), INTERVAL 3 DAY)) "
            f"AND DATE(usage_start_time) = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)")


def sql_by_service(table: str, limit: int) -> str:
    return (f"SELECT service.description AS name, ROUND(SUM({_NET}), 2) AS cost "
            f"FROM {table} WHERE _PARTITIONTIME >= TIMESTAMP(@pfloor) AND invoice.month = @month "
            f"GROUP BY name HAVING cost > 0 ORDER BY cost DESC LIMIT {limit}")


def sql_by_project(table: str, limit: int) -> str:
    return (f"SELECT project.id AS project_id, ROUND(SUM({_NET}), 2) AS cost "
            f"FROM {table} WHERE _PARTITIONTIME >= TIMESTAMP(@pfloor) AND invoice.month = @month AND project.id IS NOT NULL "
            f"GROUP BY project_id HAVING cost > 0 ORDER BY cost DESC LIMIT {limit}")


def sql_by_label(table: str, limit: int) -> str:
    return (
        f"SELECT IFNULL((SELECT l.value FROM UNNEST(labels) l WHERE l.key = @label_key), "
        f"'(unlabelled)') AS label_value, ROUND(SUM({_NET}), 2) AS cost "
        f"FROM {table} WHERE _PARTITIONTIME >= TIMESTAMP(@pfloor) AND invoice.month = @month "
        f"GROUP BY label_value HAVING cost > 0 ORDER BY cost DESC LIMIT {limit}"
    )


def sql_daily_trend(table: str, days: int) -> str:
    return (
        f"SELECT DATE(usage_start_time) AS day, ROUND(SUM({_NET}), 2) AS cost "
        f"FROM {table} "
        f"WHERE _PARTITIONTIME >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days + 2} DAY) "
        f"AND usage_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY) "
        f"GROUP BY day ORDER BY day"
    )


def sql_by_month(table: str) -> str:
    """Net spend per invoice month across an inclusive [from,to] month range.
    ``invoice.month`` is a 'YYYYMM' string, so a lexical BETWEEN is a correct
    range for same-century months and lets one query return the whole series."""
    return (
        f"SELECT invoice.month AS month, ROUND(SUM({_NET}), 2) AS cost "
        f"FROM {table} WHERE _PARTITIONTIME >= TIMESTAMP(@pfloor) "
        f"AND invoice.month BETWEEN @from_month AND @to_month "
        f"GROUP BY month ORDER BY month"
    )


def sql_anomalies(table: str, *, threshold_pct: float, min_usd: float, limit: int) -> str:
    factor = 1 + threshold_pct / 100.0
    return (
        f"WITH daily AS (SELECT service.description AS svc, DATE(usage_start_time) AS d, "
        f"SUM({_NET}) AS cost FROM {table} "
        f"WHERE _PARTITIONTIME >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 11 DAY) "
        f"AND usage_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 9 DAY) "
        f"GROUP BY svc, d) "
        f"SELECT svc AS scope, "
        f"ROUND(SUM(IF(d = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY), cost, 0)), 2) AS yesterday, "
        f"ROUND(AVG(IF(d BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY) "
        f"AND DATE_SUB(CURRENT_DATE(), INTERVAL 2 DAY), cost, NULL)), 2) AS baseline "
        f"FROM daily GROUP BY scope "
        f"HAVING baseline > 0 AND yesterday > baseline * {factor} AND yesterday > {min_usd} "
        f"ORDER BY (yesterday - baseline) DESC LIMIT {limit}"
    )


# --- query choke-point (swapped by build_fake) -----------------------------

async def _run_query(sql: str, params: dict | None = None) -> list[dict]:
    """Run a read-only BigQuery SELECT with a hard byte ceiling; return rows as
    dicts. This is THE single network seam — build_fake swaps it. All SDK use
    (client + ScalarQueryParameter) is confined here, so the handlers never
    import google-cloud-bigquery; ``params`` is a plain ``{name: str_value}``
    map bound as STRING query parameters (injection-safe for agent strings)."""
    from google.cloud import bigquery

    client = bigquery.Client(project=_bq_project())
    query_params = [
        bigquery.ScalarQueryParameter(k, "STRING", v) for k, v in (params or {}).items()
    ]
    job_config = bigquery.QueryJobConfig(
        maximum_bytes_billed=_max_bytes(),
        query_parameters=query_params,
    )
    try:
        rows = client.query(sql, job_config=job_config).result()
    except Exception as exc:  # noqa: BLE001 -- surface a clean tool error
        raise BillingGcpMCPError(f"BigQuery query failed: {exc}", reason="query") from exc
    return [dict(r.items()) for r in rows]


async def _scalar_cost(sql: str, params: dict | None = None) -> float:
    rows = await _run_query(sql, params)
    return float((rows[0].get("cost") if rows else 0.0) or 0.0)


# --- handlers --------------------------------------------------------------

async def _h_cost_summary(_unused, args: dict) -> Any:
    """Total spend: month-to-date, yesterday, previous month, and a
    straight-line full-month projection with the MoM trend %. Net of credits."""
    table = _table()
    month = _valid_month(args.get("month"))
    prev = _prev_month(month)
    today = date.today()
    days_in_month = calendar.monthrange(today.year, today.month)[1]

    mtd = await _scalar_cost(sql_month_total(table), {"month": month, "pfloor": _partition_floor(month)})
    prev_cost = await _scalar_cost(sql_month_total(table), {"month": prev, "pfloor": _partition_floor(prev)})
    yesterday = await _scalar_cost(sql_yesterday_total(table))
    projected = project_month(mtd, day_of_month=today.day, days_in_month=days_in_month)
    return {
        "month": month,
        "mtd_usd": round(mtd, 2),
        "yesterday_usd": round(yesterday, 2),
        "prev_month_usd": round(prev_cost, 2),
        "projected_month_usd": round(projected, 2),
        "trend_pct_vs_prev": round(trend_pct(projected=projected, prev=prev_cost), 1),
        "currency": "USD",
        "note": "Net of credits/discounts. BigQuery export lags ~24h.",
    }


async def _h_cost_by_service(_unused, args: dict) -> Any:
    """Top spending GCP services this month (e.g. Compute Engine, Cloud SQL,
    Vertex AI). Net of credits."""
    table = _table()
    month = _valid_month(args.get("month"))
    limit = _clamp(args.get("limit"), default=_DEFAULT_TOP_N, hi=_MAX_TOP_N)
    rows = await _run_query(sql_by_service(table, limit), {"month": month, "pfloor": _partition_floor(month)})
    return {"month": month, "by_service": [
        {"service": r.get("name") or "(unlabelled)", "cost_usd": float(r.get("cost") or 0)}
        for r in rows
    ]}


async def _h_cost_by_project(_unused, args: dict) -> Any:
    """Spend by GCP project this month, optionally tagged with an environment
    label (OPSRAG_GCP_BILLING_ENV_MAP). Net of credits."""
    table = _table()
    month = _valid_month(args.get("month"))
    limit = _clamp(args.get("limit"), default=20, hi=_MAX_TOP_N)
    env_map = _env_map()
    rows = await _run_query(sql_by_project(table, limit), {"month": month, "pfloor": _partition_floor(month)})
    return {"month": month, "by_project": [
        {"project_id": r.get("project_id"),
         "env": env_map.get(r.get("project_id"), None),
         "cost_usd": float(r.get("cost") or 0)}
        for r in rows
    ]}


async def _h_cost_by_label(_unused, args: dict) -> Any:
    """Spend grouped by the values of one resource/label key (e.g. `team`,
    `product`, `app`) — the FinOps showback lens. Net of credits."""
    label_key = (args.get("label_key") or "").strip()
    if not label_key:
        raise BillingGcpMCPError("label_key is required (e.g. 'team' or 'product').",
                                 reason="bad_args")
    table = _table()
    month = _valid_month(args.get("month"))
    limit = _clamp(args.get("limit"), default=_DEFAULT_TOP_N, hi=_MAX_TOP_N)
    rows = await _run_query(sql_by_label(table, limit),
                            {"month": month, "label_key": label_key, "pfloor": _partition_floor(month)})
    return {"month": month, "label_key": label_key, "by_label": [
        {"label_value": r.get("label_value"), "cost_usd": float(r.get("cost") or 0)}
        for r in rows
    ]}


async def _h_cost_trend(_unused, args: dict) -> Any:
    """Daily net spend for the last N days (default 14, max 90) — for spotting
    ramps and week-over-week shape."""
    table = _table()
    days = _clamp(args.get("days"), default=14, lo=2, hi=_MAX_TREND_DAYS)
    rows = await _run_query(sql_daily_trend(table, days))
    return {"days": days, "daily": [
        {"day": str(r.get("day")), "cost_usd": float(r.get("cost") or 0)} for r in rows
    ]}


async def _h_cost_by_month(_unused, args: dict) -> Any:
    """Net GCP spend per invoice month for the last N months (default 3, max 12)
    -- a clean monthly series for cost-over-time charts, in ONE call (vs. calling
    cost_summary once per month). Feed `by_month` straight into `render_chart`
    (type=line, x=month, y=cost_usd). Net of credits. Read-only."""
    table = _table()
    months = _clamp(args.get("months"), default=3, lo=2, hi=12)
    to_month = _valid_month(args.get("month"))  # inclusive latest month
    from_month = to_month
    for _ in range(months - 1):
        from_month = _prev_month(from_month)
    rows = await _run_query(
        sql_by_month(table),
        {"from_month": from_month, "to_month": to_month, "pfloor": _partition_floor(from_month)},
    )
    return {"months": months, "from_month": from_month, "to_month": to_month, "by_month": [
        {"month": str(r.get("month")), "cost_usd": float(r.get("cost") or 0)} for r in rows
    ]}


async def _h_cost_anomalies(_unused, args: dict) -> Any:
    """Services whose yesterday spend jumped above a rolling 7-day baseline
    (default >30% and >$50). The FinOps early-warning signal."""
    table = _table()
    threshold = float(args.get("threshold_pct") or 30.0)
    min_usd = float(args.get("min_usd") or 50.0)
    limit = _clamp(args.get("limit"), default=5, hi=25)
    rows = await _run_query(
        sql_anomalies(table, threshold_pct=threshold, min_usd=min_usd, limit=limit)
    )
    out = []
    for r in rows:
        y, b = float(r.get("yesterday") or 0), float(r.get("baseline") or 0)
        out.append({"scope": r.get("scope"), "yesterday_usd": y, "baseline_usd": b,
                    "pct_change": round((y - b) / b * 100.0, 1) if b else 0.0})
    return {"threshold_pct": threshold, "min_usd": min_usd, "anomalies": out}


# --- tool specs ------------------------------------------------------------

_MONTH_PROP = {"month": {"type": "string", "description": "Invoice month 'YYYYMM' (default: current month)."}}
_LIMIT_PROP = {"limit": {"type": "integer", "description": f"Max rows (default {_DEFAULT_TOP_N}, max {_MAX_TOP_N})."}}

BILLING_GCP_TOOLS: list[MCPTool] = [
    MCPTool(
        name="billing_gcp_cost_summary",
        description="GCP total spend: month-to-date, yesterday, previous month, full-month projection, and MoM trend %. Net of credits/discounts. Read-only (BigQuery billing export).",
        input_schema={"type": "object", "properties": {**_MONTH_PROP}},
        handler=_h_cost_summary,
    ),
    MCPTool(
        name="billing_gcp_cost_by_service",
        description="Top spending GCP services this month (Compute Engine, Cloud SQL, Vertex AI, ...). Net of credits. Read-only.",
        input_schema={"type": "object", "properties": {**_MONTH_PROP, **_LIMIT_PROP}},
        handler=_h_cost_by_service,
    ),
    MCPTool(
        name="billing_gcp_cost_by_project",
        description="GCP spend by project this month (optionally tagged with an environment label). Net of credits. Read-only.",
        input_schema={"type": "object", "properties": {**_MONTH_PROP, **_LIMIT_PROP}},
        handler=_h_cost_by_project,
    ),
    MCPTool(
        name="billing_gcp_cost_by_label",
        description="GCP spend grouped by one resource/label key's values (e.g. 'team', 'product', 'app') — FinOps showback. Net of credits. Read-only.",
        input_schema={
            "type": "object",
            "properties": {
                "label_key": {"type": "string", "description": "Label/tag key to group by, e.g. 'team' or 'product'."},
                **_MONTH_PROP, **_LIMIT_PROP,
            },
            "required": ["label_key"],
        },
        handler=_h_cost_by_label,
    ),
    MCPTool(
        name="billing_gcp_cost_trend",
        description="Daily net GCP spend for the last N days (default 14, max 90) — spot ramps / week-over-week shape. Read-only.",
        input_schema={"type": "object", "properties": {"days": {"type": "integer", "description": "Lookback days (2-90, default 14)."}}},
        handler=_h_cost_trend,
    ),
    MCPTool(
        name="billing_gcp_cost_by_month",
        description="Net GCP spend per invoice month for the last N months (default 3, max 12) — a monthly cost-over-time series in ONE call. Feed the result into render_chart (type=line) for a 3-month billing trend. Net of credits. Read-only.",
        input_schema={"type": "object", "properties": {
            "months": {"type": "integer", "description": "Number of months back, inclusive of the latest (2-12, default 3)."},
            **_MONTH_PROP,
        }},
        handler=_h_cost_by_month,
    ),
    MCPTool(
        name="billing_gcp_cost_anomalies",
        description="GCP services whose yesterday spend jumped above their rolling 7-day baseline (default >30% and >$50) — cost early-warning. Read-only.",
        input_schema={
            "type": "object",
            "properties": {
                "threshold_pct": {"type": "number", "description": "Jump threshold % over baseline (default 30)."},
                "min_usd": {"type": "number", "description": "Ignore scopes under this yesterday spend (default 50)."},
                "limit": {"type": "integer", "description": "Max anomalies (default 5, max 25)."},
            },
        },
        handler=_h_cost_anomalies,
    ),
]


def get_tool(name: str) -> MCPTool:
    for t in BILLING_GCP_TOOLS:
        if t.name == name:
            return t
    raise KeyError(f"unknown billing_gcp tool: {name}")


# --- fake backend (FR-012; offline tests) ----------------------------------

async def _fake_run_query(sql: str, params: list | None = None) -> list[dict]:
    """Canned stand-in for `_run_query`, keyed by fragments of the SQL. Shapes
    mirror the BigQuery rows the handlers parse. No SDK / GCP creds / network."""
    if "DATE(usage_start_time) = DATE_SUB" in sql:  # yesterday total
        return [{"cost": 2248.11}]
    if "invoice.month = @month" in sql and "GROUP BY" not in sql:  # month total
        return [{"cost": 15368.42}]
    if "service.description AS name" in sql:  # by service
        return [{"name": "Compute Engine", "cost": 6275.0},
                {"name": "Cloud SQL", "cost": 6149.0},
                {"name": "Vertex AI", "cost": 812.0}]
    if "project.id AS project_id" in sql:  # by project
        return [{"project_id": "example-prod", "cost": 11435.0},
                {"project_id": "example-sandbox", "cost": 1232.0}]
    if "UNNEST(labels)" in sql:  # by label
        return [{"label_value": "team-a", "cost": 2245.0},
                {"label_value": "(unlabelled)", "cost": 199.0}]
    if "invoice.month AS month" in sql:  # monthly series
        return [{"month": "202605", "cost": 49508.64},
                {"month": "202606", "cost": 51472.20},
                {"month": "202607", "cost": 14979.62}]
    if "AS day" in sql:  # daily trend
        return [{"day": "2026-07-08", "cost": 2303.0}, {"day": "2026-07-09", "cost": 2248.0}]
    if "AS yesterday" in sql:  # anomalies
        return [{"scope": "Cloud SQL", "yesterday": 5457.0, "baseline": 3900.0}]
    return []


def build_fake():
    """Return a FakeMCP exposing the GCP billing tools wired to an offline
    backend. Needs NO bigquery SDK / GCP creds / network: the module-level
    `_run_query` is swapped for a canned dispatcher, restored by `teardown`."""
    import opsrag.mcp.billing_gcp as _mod
    from opsrag.mcp._fake import FakeMCP

    _orig = _mod._run_query
    _mod._run_query = _fake_run_query

    def _restore() -> None:
        _mod._run_query = _orig

    return FakeMCP(tools=list(BILLING_GCP_TOOLS), client=None, teardown=_restore)
