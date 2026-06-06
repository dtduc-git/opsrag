"""Azure MCP-style tools for OpsRAG (read-only).

Read-only async tools over Azure Resource Manager (ARM), Azure Monitor,
and Log Analytics REST APIs. Auth is via
``azure.identity.DefaultAzureCredential`` (env / managed identity / CLI /
workload identity), so no static secrets live in this module. The default
subscription comes from ``AZURE_SUBSCRIPTION_ID``; the standard ``AZURE_*``
env vars (``AZURE_TENANT_ID``, ``AZURE_CLIENT_ID``, ``AZURE_CLIENT_SECRET``,
etc.) are consumed by ``DefaultAzureCredential`` transparently.

## Read-only enforcement

Every tool issues a GET, or a read-only POST against a query/search
endpoint (Resource Graph KQL, Log Analytics KQL). No
``PUT``/``DELETE``/``PATCH`` anywhere -- no resource mutation, no scale,
no exec.

## Lazy SDK import

``azure.identity`` is imported INSIDE ``_token()`` (never at module top),
so this module imports fine without the azure libs installed. Real calls
route through the module-level ``_get`` / ``_post`` httpx helpers, which
``build_fake()`` swaps for canned responders -- tests need NO azure libs
and NO network.

## Tool list (5 read-only)

| Tool                            | Endpoint                                                     |
|---------------------------------|-------------------------------------------------------------|
| `azure_monitor_logs_query`      | POST `https://api.loganalytics.io/v1/workspaces/{ws}/query` |
| `azure_monitor_metrics_query`   | GET  `.../{resourceId}/.../Microsoft.Insights/metrics`      |
| `azure_aks_list_clusters`       | GET  `.../Microsoft.ContainerService/managedClusters`       |
| `azure_list_resource_groups`    | GET  `.../subscriptions/{sub}/resourcegroups`               |
| `azure_resource_graph_query`    | POST `.../Microsoft.ResourceGraph/resources`                |
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx

from opsrag.mcp.gitlab import MCPTool

_log = logging.getLogger("opsrag.mcp.azure")

ARM_BASE = "https://management.azure.com"
LOG_ANALYTICS_BASE = "https://api.loganalytics.io"
ARM_SCOPE = "https://management.azure.com/.default"
LOG_ANALYTICS_SCOPE = "https://api.loganalytics.io/.default"

_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_LIMIT = 25
_MAX_LIMIT = 100
_RESULT_TRUNCATE_CHARS = 16000

# Redact common secret shapes from any error text we surface (ARM error
# bodies and Log Analytics errors can echo back query strings / tokens).
_REDACT_PATTERNS = [
    (re.compile(r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]+"), "[REDACTED:jwt]"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._-]{20,}", re.IGNORECASE), "Bearer [REDACTED:token]"),
    (re.compile(r"\bAKIA[A-Z0-9]{16}\b"), "[REDACTED:aws_access_key]"),
    (re.compile(r"(?i)\b(client_secret|password|secret|api[_-]?key)\b\s*[=:]\s*\S+"), r"\1=[REDACTED]"),
]


def _redact(text: str) -> str:
    for pat, repl in _REDACT_PATTERNS:
        text = pat.sub(repl, text)
    return text


class AzureMCPError(Exception):
    """Raised on Azure API errors. Wraps the upstream status + (redacted) body."""

    def __init__(self, status: int, body: str, *, tool: str | None = None):
        self.status = status
        self.body = _redact(body or "")
        self.tool = tool
        super().__init__(f"[{tool or 'azure'}] {status}: {self.body[:300]}")


@dataclass
class _Config:
    subscription_id: str


def _config() -> _Config:
    sub = (os.environ.get("AZURE_SUBSCRIPTION_ID") or "").strip()
    if not sub:
        raise AzureMCPError(
            0,
            "Azure subscription not set. Set AZURE_SUBSCRIPTION_ID (and the "
            "standard AZURE_* credential env vars consumed by "
            "DefaultAzureCredential: AZURE_TENANT_ID, AZURE_CLIENT_ID, "
            "AZURE_CLIENT_SECRET / managed-identity / az-cli login).",
            tool="azure",
        )
    return _Config(subscription_id=sub)


def _token(scope: str) -> str:
    """Acquire a bearer token via DefaultAzureCredential. Lazy-imports
    azure.identity so the module loads without azure libs installed."""
    try:
        from azure.identity import DefaultAzureCredential  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without SDK
        raise AzureMCPError(
            0,
            "azure-identity not installed. `pip install azure-identity` to "
            "use Azure tools (DefaultAzureCredential).",
            tool="azure",
        ) from exc
    cred = DefaultAzureCredential()
    return cred.get_token(scope).token


def _headers(scope: str) -> dict:
    return {
        "Authorization": f"Bearer {_token(scope)}",
        "Content-Type": "application/json",
    }


async def _get(
    url: str,
    params: dict | None = None,
    *,
    scope: str = ARM_SCOPE,
    tool: str = "azure",
) -> Any:
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    async with httpx.AsyncClient(headers=_headers(scope), timeout=_DEFAULT_TIMEOUT_S) as http:
        resp = await http.get(url, params=clean)
    if resp.status_code >= 400:
        raise AzureMCPError(resp.status_code, resp.text, tool=tool)
    return resp.json() if resp.text else {}


async def _post(
    url: str,
    body: dict,
    *,
    scope: str = ARM_SCOPE,
    tool: str = "azure",
) -> Any:
    async with httpx.AsyncClient(headers=_headers(scope), timeout=_DEFAULT_TIMEOUT_S) as http:
        resp = await http.post(url, json=body)
    if resp.status_code >= 400:
        raise AzureMCPError(resp.status_code, resp.text, tool=tool)
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


def _sub(args: dict) -> str:
    """Per-call subscription override, else the env default."""
    return (args.get("subscription_id") or "").strip() or _config().subscription_id


# --- handlers -------------------------------------------------------


async def _h_resource_graph_query(_unused, args: dict) -> Any:
    """Azure Resource Graph KQL query (the escape hatch).

    POST `/providers/Microsoft.ResourceGraph/resources`. Pass `query`
    (KQL over the `Resources` table family). Optional `subscription_id`
    overrides the default; otherwise scoped to the env default sub.
    Returns the parsed `data` rows (capped) plus `count`/`totalRecords`.
    """
    query = args.get("query")
    if not query:
        raise AzureMCPError(0, "azure_resource_graph_query: `query` (KQL) is required", tool="azure_resource_graph_query")
    sub = _sub(args)
    top = _clamp(args.get("limit"), default=_DEFAULT_LIMIT)
    body = {
        "subscriptions": [sub],
        "query": query,
        "options": {"top": top},
    }
    url = f"{ARM_BASE}/providers/Microsoft.ResourceGraph/resources?api-version=2021-03-01"
    resp = await _post(url, body, tool="azure_resource_graph_query")
    rows = resp.get("data") or []
    if isinstance(rows, list):
        rows = rows[:top]
    return {
        "query": query,
        "subscription_id": sub,
        "count": resp.get("count", len(rows) if isinstance(rows, list) else 0),
        "total_records": resp.get("totalRecords"),
        "data": rows,
    }


async def _h_list_resource_groups(_unused, args: dict) -> Any:
    """List resource groups in a subscription.

    GET `/subscriptions/{sub}/resourcegroups`. Returns trimmed records
    (name, id, location, provisioning state, tags) capped at `limit`.
    """
    sub = _sub(args)
    url = f"{ARM_BASE}/subscriptions/{sub}/resourcegroups?api-version=2021-04-01"
    resp = await _get(url, tool="azure_list_resource_groups")
    items = resp.get("value") or []
    limit = _clamp(args.get("limit"), default=_DEFAULT_LIMIT)
    out = []
    for rg in items[:limit]:
        props = rg.get("properties") or {}
        out.append({
            "name": rg.get("name"),
            "id": rg.get("id"),
            "location": rg.get("location"),
            "provisioning_state": props.get("provisioningState"),
            "tags": rg.get("tags") or {},
        })
    return {
        "subscription_id": sub,
        "count": len(out),
        "resource_groups": out,
    }


async def _h_aks_list_clusters(_unused, args: dict) -> Any:
    """List AKS (managed Kubernetes) clusters in a subscription.

    GET `/subscriptions/{sub}/providers/Microsoft.ContainerService/managedClusters`.
    Returns trimmed cluster records (name, location, k8s version, power
    state, fqdn, node-pool summary) capped at `limit`.
    """
    sub = _sub(args)
    url = (
        f"{ARM_BASE}/subscriptions/{sub}/providers/"
        f"Microsoft.ContainerService/managedClusters?api-version=2024-02-01"
    )
    resp = await _get(url, tool="azure_aks_list_clusters")
    items = resp.get("value") or []
    limit = _clamp(args.get("limit"), default=_DEFAULT_LIMIT)
    out = []
    for c in items[:limit]:
        props = c.get("properties") or {}
        pools = props.get("agentPoolProfiles") or []
        out.append({
            "name": c.get("name"),
            "id": c.get("id"),
            "location": c.get("location"),
            "kubernetes_version": props.get("kubernetesVersion"),
            "provisioning_state": props.get("provisioningState"),
            "power_state": (props.get("powerState") or {}).get("code"),
            "fqdn": props.get("fqdn"),
            "node_resource_group": props.get("nodeResourceGroup"),
            "node_pools": [
                {
                    "name": p.get("name"),
                    "count": p.get("count"),
                    "vm_size": p.get("vmSize"),
                    "mode": p.get("mode"),
                }
                for p in pools[:20]
            ],
        })
    return {
        "subscription_id": sub,
        "count": len(out),
        "clusters": out,
    }


async def _h_monitor_metrics_query(_unused, args: dict) -> Any:
    """Query Azure Monitor metrics for a resource.

    GET `/{resourceId}/providers/Microsoft.Insights/metrics`. Pass the
    full ARM `resource_id` (e.g.
    `/subscriptions/.../resourceGroups/.../providers/Microsoft.Compute/virtualMachines/vm1`),
    comma-separated `metricnames`, and an optional ISO-8601 `timespan`
    (e.g. `2026-06-01T00:00:00Z/2026-06-01T01:00:00Z` or `PT1H`).
    Returns parsed metric series (name, unit, timeseries data points).
    """
    resource_id = args.get("resource_id")
    if not resource_id:
        raise AzureMCPError(
            0,
            "azure_monitor_metrics_query: `resource_id` (full ARM resource id) is required",
            tool="azure_monitor_metrics_query",
        )
    resource_id = str(resource_id).lstrip("/")
    params = {
        "api-version": "2018-01-01",
        "metricnames": args.get("metricnames"),
        "timespan": args.get("timespan"),
        "interval": args.get("interval"),
        "aggregation": args.get("aggregation"),
        "filter": args.get("filter"),
    }
    url = f"{ARM_BASE}/{resource_id}/providers/Microsoft.Insights/metrics"
    resp = await _get(url, params=params, tool="azure_monitor_metrics_query")
    series_out = []
    for m in resp.get("value") or []:
        name = (m.get("name") or {})
        ts_out = []
        for ts in m.get("timeseries") or []:
            points = ts.get("data") or []
            ts_out.append({
                "metadata": [
                    {"name": (mv.get("name") or {}).get("value"), "value": mv.get("value")}
                    for mv in (ts.get("metadatavalues") or [])
                ],
                "data": points[:200],
            })
        series_out.append({
            "name": name.get("value") if isinstance(name, dict) else name,
            "unit": m.get("unit"),
            "type": m.get("type"),
            "timeseries": ts_out,
        })
    return {
        "resource_id": f"/{resource_id}",
        "timespan": resp.get("timespan") or args.get("timespan"),
        "interval": resp.get("interval"),
        "count": len(series_out),
        "metrics": series_out,
    }


async def _h_monitor_logs_query(_unused, args: dict) -> Any:
    """Query a Log Analytics workspace with KQL (Azure Monitor Logs).

    POST `https://api.loganalytics.io/v1/workspaces/{workspaceId}/query`
    with body `{query, timespan}`. The token audience is
    `https://api.loganalytics.io`. Pass `workspace_id` (the workspace
    GUID), `query` (KQL), and optional ISO-8601 duration/range `timespan`
    (e.g. `PT1H`). Returns parsed tables -> rows-as-dicts (capped).
    """
    workspace_id = args.get("workspace_id")
    query = args.get("query")
    if not workspace_id:
        raise AzureMCPError(0, "azure_monitor_logs_query: `workspace_id` is required", tool="azure_monitor_logs_query")
    if not query:
        raise AzureMCPError(0, "azure_monitor_logs_query: `query` (KQL) is required", tool="azure_monitor_logs_query")
    body: dict[str, Any] = {"query": query}
    if args.get("timespan"):
        body["timespan"] = args["timespan"]
    url = f"{LOG_ANALYTICS_BASE}/v1/workspaces/{workspace_id}/query"
    resp = await _post(url, body, scope=LOG_ANALYTICS_SCOPE, tool="azure_monitor_logs_query")
    limit = _clamp(args.get("limit"), default=_DEFAULT_LIMIT)
    tables_out = []
    for table in resp.get("tables") or []:
        cols = [c.get("name") for c in (table.get("columns") or [])]
        rows = table.get("rows") or []
        dict_rows = []
        for row in rows[:limit]:
            rec = {}
            for i, col in enumerate(cols):
                val = row[i] if i < len(row) else None
                if isinstance(val, str):
                    val = _truncate(val, 2000)
                rec[col] = val
            dict_rows.append(rec)
        tables_out.append({
            "name": table.get("name"),
            "columns": cols,
            "row_count": len(dict_rows),
            "rows": dict_rows,
        })
    return {
        "workspace_id": workspace_id,
        "query": query,
        "table_count": len(tables_out),
        "tables": tables_out,
    }


# --- tool registry --------------------------------------------------


AZURE_TOOLS: list[MCPTool] = [
    MCPTool(
        name="azure_monitor_logs_query",
        description=(
            "Run a KQL query against a Log Analytics workspace (Azure "
            "Monitor Logs). Use for app/platform logs, AzureDiagnostics, "
            "KubeEvents, ContainerLog, Heartbeat, etc. Pass `workspace_id` "
            "(workspace GUID), `query` (KQL), and optional `timespan` "
            "(ISO-8601 duration like `PT1H` or a `<start>/<end>` range). "
            "Returns parsed tables with rows as column->value dicts."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "workspace_id": {"type": "string", "description": "Log Analytics workspace GUID"},
                "query": {"type": "string", "description": "KQL query"},
                "timespan": {"type": "string", "description": "ISO-8601 duration (PT1H) or <start>/<end> range"},
                "limit": {"type": "number", "description": f"Max rows per table (default {_DEFAULT_LIMIT}, cap {_MAX_LIMIT})"},
            },
            "required": ["workspace_id", "query"],
        },
        handler=_h_monitor_logs_query,
    ),
    MCPTool(
        name="azure_monitor_metrics_query",
        description=(
            "Query Azure Monitor platform metrics for a single resource. "
            "Pass the full ARM `resource_id`, comma-separated `metricnames` "
            "(e.g. `Percentage CPU,Available Memory Bytes`), and optional "
            "`timespan` / `interval` / `aggregation`. Returns metric series "
            "with timeseries data points."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "resource_id": {
                    "type": "string",
                    "description": (
                        "Full ARM resource id, e.g. "
                        "/subscriptions/<sub>/resourceGroups/<rg>/providers/"
                        "Microsoft.Compute/virtualMachines/<vm>"
                    ),
                },
                "metricnames": {"type": "string", "description": "Comma-separated metric names"},
                "timespan": {"type": "string", "description": "ISO-8601 duration or <start>/<end> range"},
                "interval": {"type": "string", "description": "ISO-8601 interval, e.g. PT5M"},
                "aggregation": {"type": "string", "description": "e.g. Average,Maximum,Total"},
                "filter": {"type": "string", "description": "Optional metric dimension filter"},
            },
            "required": ["resource_id"],
        },
        handler=_h_monitor_metrics_query,
    ),
    MCPTool(
        name="azure_aks_list_clusters",
        description=(
            "List AKS managed-Kubernetes clusters in the subscription. "
            "Returns name, location, kubernetes_version, provisioning/power "
            "state, fqdn, node_resource_group, and node-pool summary. Use "
            "for 'which AKS clusters exist', cluster versions, power state."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "subscription_id": {"type": "string", "description": "Override default AZURE_SUBSCRIPTION_ID"},
                "limit": {"type": "number"},
            },
        },
        handler=_h_aks_list_clusters,
    ),
    MCPTool(
        name="azure_list_resource_groups",
        description=(
            "List resource groups in the subscription. Returns name, id, "
            "location, provisioning state, and tags. Use to discover the "
            "RG layout before drilling into resources."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "subscription_id": {"type": "string", "description": "Override default AZURE_SUBSCRIPTION_ID"},
                "limit": {"type": "number"},
            },
        },
        handler=_h_list_resource_groups,
    ),
    MCPTool(
        name="azure_resource_graph_query",
        description=(
            "Escape hatch: run an Azure Resource Graph KQL query across the "
            "subscription's resources (the `Resources` table family). Use "
            "for ad-hoc inventory questions any other tool can't answer, "
            "e.g. `Resources | where type =~ 'microsoft.compute/virtualmachines' "
            "| project name, location, tags`. Read-only. Returns the parsed "
            "`data` rows."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Azure Resource Graph KQL query"},
                "subscription_id": {"type": "string", "description": "Override default AZURE_SUBSCRIPTION_ID"},
                "limit": {"type": "number", "description": f"Max rows (default {_DEFAULT_LIMIT}, cap {_MAX_LIMIT})"},
            },
            "required": ["query"],
        },
        handler=_h_resource_graph_query,
    ),
]


def get_tool(name: str) -> MCPTool:
    for t in AZURE_TOOLS:
        if t.name == name:
            return t
    raise KeyError(f"unknown azure tool: {name}")


# --- fake backend (FR-012; integration tests) ----------------------
#
# Azure handlers ignore their first (`_unused`) arg and reach the
# module-level `_get` / `_post`, which acquire a bearer token via
# DefaultAzureCredential (lazy-imported) and call ARM / Log Analytics
# over httpx. The offline fake swaps those two module functions for
# canned, shape-faithful responders -- so tests need NO azure libs, NO
# Azure credentials, and NO network. `build_fake()` returns client=None
# (handlers discard it) plus a teardown that restores the real helpers.


async def _fake_get(
    url: str,
    params: dict | None = None,
    *,
    scope: str = ARM_SCOPE,
    tool: str = "azure",
) -> Any:
    """Canned stand-in for the module-level GET. Routes by URL to a
    response shaped like the real ARM / Monitor endpoint the handler parses."""
    if "/resourcegroups" in url:
        return {
            "value": [
                {
                    "id": "/subscriptions/sub-123/resourceGroups/rg-prod",
                    "name": "rg-prod",
                    "location": "eastus",
                    "properties": {"provisioningState": "Succeeded"},
                    "tags": {"env": "prod"},
                },
                {
                    "id": "/subscriptions/sub-123/resourceGroups/rg-dev",
                    "name": "rg-dev",
                    "location": "westus",
                    "properties": {"provisioningState": "Succeeded"},
                    "tags": {},
                },
            ]
        }
    if "Microsoft.ContainerService/managedClusters" in url:
        return {
            "value": [
                {
                    "id": "/subscriptions/sub-123/resourceGroups/rg-prod/providers/Microsoft.ContainerService/managedClusters/aks-prod",
                    "name": "aks-prod",
                    "location": "eastus",
                    "properties": {
                        "kubernetesVersion": "1.29.2",
                        "provisioningState": "Succeeded",
                        "powerState": {"code": "Running"},
                        "fqdn": "aks-prod-abc.hcp.eastus.azmk8s.io",
                        "nodeResourceGroup": "MC_rg-prod_aks-prod_eastus",
                        "agentPoolProfiles": [
                            {"name": "system", "count": 3, "vmSize": "Standard_D4s_v5", "mode": "System"},
                            {"name": "user", "count": 5, "vmSize": "Standard_D8s_v5", "mode": "User"},
                        ],
                    },
                }
            ]
        }
    if "/providers/Microsoft.Insights/metrics" in url:
        return {
            "timespan": "2026-06-01T00:00:00Z/2026-06-01T01:00:00Z",
            "interval": "PT5M",
            "value": [
                {
                    "id": "/subscriptions/sub-123/.../providers/Microsoft.Insights/metrics/Percentage CPU",
                    "type": "Microsoft.Insights/metrics",
                    "name": {"value": "Percentage CPU", "localizedValue": "Percentage CPU"},
                    "unit": "Percent",
                    "timeseries": [
                        {
                            "metadatavalues": [],
                            "data": [
                                {"timeStamp": "2026-06-01T00:00:00Z", "average": 12.5},
                                {"timeStamp": "2026-06-01T00:05:00Z", "average": 18.2},
                            ],
                        }
                    ],
                }
            ],
        }
    return {}


async def _fake_post(
    url: str,
    body: dict,
    *,
    scope: str = ARM_SCOPE,
    tool: str = "azure",
) -> Any:
    """Canned stand-in for the module-level POST (Resource Graph + Log
    Analytics query)."""
    if "Microsoft.ResourceGraph/resources" in url:
        return {
            "totalRecords": 2,
            "count": 2,
            "resultTruncated": "false",
            "data": [
                {"name": "vm-1", "location": "eastus", "type": "microsoft.compute/virtualmachines"},
                {"name": "vm-2", "location": "westus", "type": "microsoft.compute/virtualmachines"},
            ],
        }
    if "/v1/workspaces/" in url and url.endswith("/query"):
        return {
            "tables": [
                {
                    "name": "PrimaryResult",
                    "columns": [
                        {"name": "TimeGenerated", "type": "datetime"},
                        {"name": "Computer", "type": "string"},
                        {"name": "Level", "type": "string"},
                        {"name": "Message", "type": "string"},
                    ],
                    "rows": [
                        ["2026-06-01T00:00:00Z", "aks-node-1", "Error", "OOMKilled container"],
                        ["2026-06-01T00:01:00Z", "aks-node-2", "Warning", "high memory"],
                    ],
                }
            ]
        }
    return {}


def build_fake():
    """Return a FakeMCP exposing the Azure tools wired to an offline
    backend. Needs NO azure libs / credentials / network: the
    module-level `_get` / `_post` are swapped for canned responders and
    restored by `teardown`. AZURE_SUBSCRIPTION_ID is set if absent so
    `_config()` resolves without a real environment."""
    import opsrag.mcp.azure as _mod
    from opsrag.mcp._fake import FakeMCP

    _orig_get = _mod._get
    _orig_post = _mod._post
    _orig_sub = os.environ.get("AZURE_SUBSCRIPTION_ID")
    _mod._get = _fake_get
    _mod._post = _fake_post
    if not _orig_sub:
        os.environ["AZURE_SUBSCRIPTION_ID"] = "sub-123"

    def _restore() -> None:
        _mod._get = _orig_get
        _mod._post = _orig_post
        if _orig_sub is None:
            os.environ.pop("AZURE_SUBSCRIPTION_ID", None)
        else:
            os.environ["AZURE_SUBSCRIPTION_ID"] = _orig_sub

    return FakeMCP(tools=list(AZURE_TOOLS), client=None, teardown=_restore)
