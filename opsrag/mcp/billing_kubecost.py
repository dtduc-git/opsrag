"""Kubecost / OpenCost billing MCP connector — Billing category, read-only.

Read-only FinOps tools over the **in-cluster Kubecost/OpenCost HTTP API**
(`/model/*`). Kubecost is the canonical source of Kubernetes-native cost
allocation — it attributes cluster spend (CPU, RAM, PV, network, load
balancers) down to namespace / controller / label, and — when a cloud
billing integration is configured — reconciles those in-cluster numbers
against the cloud provider's actual invoice. Sibling to `billing_gcp`
(GCP BigQuery export): same category, GCP answers "what did the invoice
say", Kubecost answers "which workload spent it".

The `/model` API is unauthenticated (it runs in-cluster behind the
cost-analyzer frontend / an ingress); OpsRAG reaches it either directly by
its cluster-DNS service name in prod or via a port-forward locally. All
calls are `httpx` GETs — no mutation surface exists on these endpoints.

Config (env vars):
  ``OPSRAG_KUBECOST_URL``     REQUIRED. Base URL of the cost-analyzer, e.g.
       ``http://kubecost-cost-analyzer.kubecost.svc.cluster.local:9090`` in
       prod, or ``http://localhost:9090`` when port-forwarded. Read at call
       time (not import time) so tests / redeploys can rebind it.
  ``OPSRAG_KUBECOST_TIMEOUT`` optional per-request timeout seconds (default 30).

Safety: ``window`` is whitelisted to a safe pattern (``\\d+d`` / ``today`` /
``week`` / ``month`` / ``lastmonth`` / ``\\d+h``) — the agent never supplies
an arbitrary window string. Every handler returns a COMPACT dict (totalCost
plus the cost components per aggregation, top ~25) rather than dumping the
raw Kubecost payload.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

import httpx

from opsrag.mcp.gitlab import MCPTool

_log = logging.getLogger("opsrag.mcp.billing_kubecost")

_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_WINDOW = "7d"
_DEFAULT_AGGREGATE = "namespace"
_TOP_N = 25  # cap rows returned per aggregation so we never dump the whole payload

# Whitelisted `window` shapes. Kubecost accepts durations (`7d`, `24h`),
# named windows (`today`, `week`, `month`, `lastmonth`, `yesterday`), and
# `Nd`/`Nh` offsets — reject anything else so the agent can't inject a raw
# comma-separated RFC3339 range or arbitrary string.
_WINDOW_RE = re.compile(r"^(\d+d|\d+h|today|yesterday|week|month|lastweek|lastmonth)$")
# `aggregate` is a small closed-ish set plus `label:<key>` / `annotation:<key>`.
_AGGREGATE_RE = re.compile(r"^(cluster|node|namespace|controller|controllerKind|"
                           r"pod|service|deployment|daemonset|statefulset|job|"
                           r"container|type|provider|account|project|"
                           r"(label|annotation):[A-Za-z0-9_.\-/]+)$")


class KubecostMCPError(Exception):
    """Read-only Kubecost tool failure. Carries a short ``reason`` code
    (``bad_config`` / ``bad_args`` / ``query``)."""

    def __init__(self, message: str, *, reason: str = "error") -> None:
        super().__init__(message)
        self.reason = reason


# --- config (Helm values via the bound config block; env vars are fallback) --
_BOUND: Any | None = None


def bind(cfg: Any | None = None) -> None:
    """Register the billing_kubecost config block (or None to clear)."""
    global _BOUND
    _BOUND = cfg


def _base_url() -> str:
    raw = (str(getattr(_BOUND, "url", None) or "").strip()
           if _BOUND is not None else "")
    if not raw:
        raw = (os.environ.get("OPSRAG_KUBECOST_URL") or "").strip()
    if not raw:
        raise KubecostMCPError(
            "billing_kubecost url is not configured. Set `mcp.billing_kubecost.url` "
            "in Helm values (or OPSRAG_KUBECOST_URL) to the Kubecost cost-analyzer, "
            "e.g. `http://kubecost-cost-analyzer.kubecost.svc.cluster.local:9090`.",
            reason="bad_config",
        )
    return raw.rstrip("/")


def _timeout() -> float:
    if _BOUND is not None and getattr(_BOUND, "timeout_seconds", None):
        try:
            return float(_BOUND.timeout_seconds)
        except (TypeError, ValueError):
            pass
    try:
        return float(os.environ.get("OPSRAG_KUBECOST_TIMEOUT") or _DEFAULT_TIMEOUT_S)
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_S


# --- pure helpers (unit-testable) ------------------------------------------

def _clamp(n: Any, *, default: int, lo: int = 1, hi: int) -> int:
    try:
        v = int(n)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _valid_window(w: str | None) -> str:
    """Return a whitelisted Kubecost `window`, or the default (`7d`)."""
    if w and _WINDOW_RE.match(str(w).strip()):
        return str(w).strip()
    return _DEFAULT_WINDOW


def _valid_aggregate(a: str | None, *, default: str = _DEFAULT_AGGREGATE) -> str:
    """Return a whitelisted `aggregate` field, or the given default."""
    if a and _AGGREGATE_RE.match(str(a).strip()):
        return str(a).strip()
    return default


def _f(v: Any) -> float:
    """Coerce a possibly-missing Kubecost numeric to a rounded float."""
    try:
        return round(float(v or 0.0), 4)
    except (TypeError, ValueError):
        return 0.0


# --- request choke-point (swapped by build_fake) ---------------------------

async def _get(path: str, params: dict | None = None, *, tool: str = "billing_kubecost") -> Any:
    """Issue one read-only GET to the Kubecost `/model` API and return the
    parsed JSON. This is THE single network seam — build_fake swaps it.

    Returns ``(status_code, json)`` so callers can degrade gracefully on a
    404 (e.g. cloudCost when no cloud integration is configured) instead of
    raising. Raises ``KubecostMCPError`` only on transport failure or a 5xx.
    """
    base = _base_url()
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    try:
        async with httpx.AsyncClient(timeout=_timeout()) as http:
            resp = await http.get(f"{base}{path}", params=clean)
    except httpx.HTTPError as exc:
        raise KubecostMCPError(f"Kubecost request to {path} failed: {exc}", reason="query") from exc
    if resp.status_code >= 500:
        raise KubecostMCPError(
            f"[{tool}] Kubecost {resp.status_code} on {path}: {(resp.text or '')[:300]}",
            reason="query",
        )
    body: Any = {}
    if resp.text:
        try:
            body = resp.json()
        except ValueError:
            body = {}
    return resp.status_code, body


def _alloc_rows(data: Any) -> list[dict]:
    """Normalise the Kubecost `/model/allocation` payload into a flat list of
    per-aggregation rows.

    Kubecost's `data` is either a LIST of window-sets (`accumulate=false`) or a
    single accumulated set; each set is a dict mapping aggregation-name ->
    allocation object (which itself carries `name`, `cpuCost`, `ramCost`,
    `pvCost`, `networkCost`, `totalCost`, `totalEfficiency`, ...). We flatten
    to one row per name and sum across window-sets if there are several.
    """
    sets: list[dict] = []
    if isinstance(data, list):
        sets = [s for s in data if isinstance(s, dict)]
    elif isinstance(data, dict):
        sets = [data]

    merged: dict[str, dict] = {}
    for s in sets:
        for name, alloc in s.items():
            if not isinstance(alloc, dict):
                continue
            key = alloc.get("name") or name
            row = merged.setdefault(key, {
                "name": key, "cpuCost": 0.0, "ramCost": 0.0, "pvCost": 0.0,
                "networkCost": 0.0, "loadBalancerCost": 0.0, "totalCost": 0.0,
                "_eff": [],
            })
            row["cpuCost"] += _f(alloc.get("cpuCost"))
            row["ramCost"] += _f(alloc.get("ramCost"))
            row["pvCost"] += _f(alloc.get("pvCost"))
            row["networkCost"] += _f(alloc.get("networkCost"))
            row["loadBalancerCost"] += _f(alloc.get("loadBalancerCost"))
            row["totalCost"] += _f(alloc.get("totalCost"))
            eff = alloc.get("totalEfficiency")
            if eff is not None:
                row["_eff"].append(_f(eff))

    out = []
    for row in merged.values():
        eff_list = row.pop("_eff")
        row["efficiency"] = round(sum(eff_list) / len(eff_list), 4) if eff_list else None
        for k in ("cpuCost", "ramCost", "pvCost", "networkCost", "loadBalancerCost", "totalCost"):
            row[k] = round(row[k], 4)
        out.append(row)
    out.sort(key=lambda r: r["totalCost"], reverse=True)
    return out


def _asset_rows(data: Any) -> list[dict]:
    """Normalise `/model/assets` into a flat list. Asset objects carry
    `name`/`type`/`totalCost` (plus provider-specific fields). Shape mirrors
    allocation: a list of sets or a single dict keyed by asset name."""
    sets: list[dict] = []
    if isinstance(data, list):
        sets = [s for s in data if isinstance(s, dict)]
    elif isinstance(data, dict):
        sets = [data]

    merged: dict[str, dict] = {}
    for s in sets:
        for name, asset in s.items():
            if not isinstance(asset, dict):
                continue
            props = asset.get("properties") or {}
            key = asset.get("name") or props.get("name") or name
            row = merged.setdefault(key, {
                "name": key,
                "type": asset.get("type") or props.get("category"),
                "totalCost": 0.0,
            })
            row["totalCost"] += _f(asset.get("totalCost"))
    out = [dict(r, totalCost=round(r["totalCost"], 4)) for r in merged.values()]
    out.sort(key=lambda r: r["totalCost"], reverse=True)
    return out


# --- handlers --------------------------------------------------------------

async def _h_allocation(_unused, args: dict) -> Any:
    """`GET /model/allocation` — per-aggregation Kubernetes cost allocation.

    Breaks cluster spend down by namespace (default), controller, or a label
    (`label:app`) over `window` (default 7d). Returns each aggregation's
    cpuCost / ramCost / pvCost / networkCost / totalCost and efficiency,
    top 25 by totalCost, plus a grand total."""
    window = _valid_window(args.get("window"))
    aggregate = _valid_aggregate(args.get("aggregate"))
    accumulate = "true" if args.get("accumulate", True) else "false"
    _status, body = await _get(
        "/model/allocation",
        params={"window": window, "aggregate": aggregate, "accumulate": accumulate},
        tool="billing_kubecost_allocation",
    )
    rows = _alloc_rows(body.get("data") if isinstance(body, dict) else body)
    total = round(sum(r["totalCost"] for r in rows), 4)
    return {
        "window": window,
        "aggregate": aggregate,
        "total_cost": total,
        "currency": "USD",
        "count": len(rows),
        "allocations": rows[:_TOP_N],
        "note": "Kubernetes in-cluster allocation (cpu+ram+pv+network). Idle/unmounted PV may be excluded.",
    }


async def _h_allocation_summary(_unused, args: dict) -> Any:
    """`GET /model/allocation/summary` — lighter allocation summary.

    Same breakdown as `billing_kubecost_allocation` but the summary endpoint
    returns only totals (no per-resource asset detail), so it is cheaper for
    a quick 'who spends the most' question."""
    window = _valid_window(args.get("window"))
    aggregate = _valid_aggregate(args.get("aggregate"))
    _status, body = await _get(
        "/model/allocation/summary",
        params={"window": window, "aggregate": aggregate},
        tool="billing_kubecost_allocation_summary",
    )
    # The summary endpoint nests the sets under data.sets[].allocations, but
    # older builds return the same shape as /model/allocation. Handle both.
    data = body.get("data") if isinstance(body, dict) else body
    if isinstance(data, dict) and "sets" in data:
        sets = [s.get("allocations") or {} for s in (data.get("sets") or []) if isinstance(s, dict)]
        rows = _alloc_rows(sets)
    else:
        rows = _alloc_rows(data)
    total = round(sum(r["totalCost"] for r in rows), 4)
    return {
        "window": window,
        "aggregate": aggregate,
        "total_cost": total,
        "currency": "USD",
        "count": len(rows),
        "summary": [{"name": r["name"], "total_cost": r["totalCost"], "efficiency": r["efficiency"]}
                    for r in rows[:_TOP_N]],
    }


async def _h_assets(_unused, args: dict) -> Any:
    """`GET /model/assets` — cluster ASSET cost (nodes, disks, load balancers).

    Where `allocation` attributes spend to workloads, `assets` reports the
    underlying infrastructure line-items. Aggregate by `type` (default) to
    see node vs disk vs LB spend, or `node` for per-node cost. Read-only."""
    window = _valid_window(args.get("window"))
    aggregate = _valid_aggregate(args.get("aggregate"), default="type")
    _status, body = await _get(
        "/model/assets",
        params={"window": window, "aggregate": aggregate, "accumulate": "true"},
        tool="billing_kubecost_assets",
    )
    rows = _asset_rows(body.get("data") if isinstance(body, dict) else body)
    total = round(sum(r["totalCost"] for r in rows), 4)
    return {
        "window": window,
        "aggregate": aggregate,
        "total_cost": total,
        "currency": "USD",
        "count": len(rows),
        "assets": rows[:_TOP_N],
    }


async def _h_cloud_cost(_unused, args: dict) -> Any:
    """`GET /model/cloudCost` — cloud-provider billing-integration cost.

    Reconciled actual cloud spend (out-of-cluster too) IF a cloud billing
    integration is configured. **Degrades gracefully**: not every Kubecost
    has cloud-cost set up, so a 404 / empty payload returns an empty result
    plus a `note` rather than raising. Read-only."""
    window = _valid_window(args.get("window"))
    aggregate = _valid_aggregate(args.get("aggregate"), default="provider")
    status, body = await _get(
        "/model/cloudCost",
        params={"window": window, "aggregate": aggregate, "accumulate": "true"},
        tool="billing_kubecost_cloud_cost",
    )
    if status == 404:
        return {
            "window": window, "aggregate": aggregate,
            "total_cost": 0.0, "currency": "USD", "count": 0, "cloud_costs": [],
            "note": "No cloud-cost integration configured on this Kubecost (endpoint 404). "
                    "Use billing_gcp_* for the GCP invoice, or billing_kubecost_allocation "
                    "for in-cluster attribution.",
        }
    # cloudCost payload: data.sets[].cloudCosts{ name -> {..., aggregate/amortizedNetCost/listCost} }
    data = body.get("data") if isinstance(body, dict) else body
    merged: dict[str, float] = {}
    sets: list[dict] = []
    if isinstance(data, dict) and "sets" in data:
        sets = [s.get("cloudCosts") or {} for s in (data.get("sets") or []) if isinstance(s, dict)]
    elif isinstance(data, list):
        sets = [s for s in data if isinstance(s, dict)]
    elif isinstance(data, dict):
        sets = [data]
    for s in sets:
        for name, cc in s.items():
            if not isinstance(cc, dict):
                continue
            key = cc.get("name") or name
            amount = cc.get("amortizedNetCost")
            if isinstance(amount, dict):  # some builds nest {cost, kubernetesPercent}
                amount = amount.get("cost")
            if amount is None:
                amount = cc.get("netCost") or cc.get("listCost") or cc.get("cost")
            merged[key] = merged.get(key, 0.0) + _f(amount)
    rows = sorted(
        ({"name": k, "cost_usd": round(v, 4)} for k, v in merged.items()),
        key=lambda r: r["cost_usd"], reverse=True,
    )
    if not rows:
        return {
            "window": window, "aggregate": aggregate,
            "total_cost": 0.0, "currency": "USD", "count": 0, "cloud_costs": [],
            "note": "Cloud-cost endpoint returned no data (integration may be unconfigured or still ingesting).",
        }
    return {
        "window": window,
        "aggregate": aggregate,
        "total_cost": round(sum(r["cost_usd"] for r in rows), 4),
        "currency": "USD",
        "count": len(rows),
        "cloud_costs": rows[:_TOP_N],
    }


# --- tool specs ------------------------------------------------------------

_WINDOW_PROP = {"window": {"type": "string",
                           "description": "Time window: '7d' (default), '24h', 'today', 'week', "
                                          "'month', 'lastmonth'. Arbitrary strings are rejected."}}

BILLING_KUBECOST_TOOLS: list[MCPTool] = [
    MCPTool(
        name="billing_kubecost_allocation",
        description="Kubernetes in-cluster cost allocation from Kubecost/OpenCost: cluster spend broken "
                    "down by namespace (default), controller, or label (e.g. `label:app`) over a window. "
                    "Returns cpuCost/ramCost/pvCost/networkCost/totalCost + efficiency per aggregation, "
                    "top 25. Read-only (`/model/allocation`).",
        input_schema={
            "type": "object",
            "properties": {
                **_WINDOW_PROP,
                "aggregate": {"type": "string",
                              "description": "Group by: 'namespace' (default), 'controller', 'pod', "
                                             "'node', 'cluster', or 'label:<key>' (e.g. 'label:app')."},
                "accumulate": {"type": "boolean",
                               "description": "Sum the whole window into one set (default true)."},
            },
        },
        handler=_h_allocation,
    ),
    MCPTool(
        name="billing_kubecost_allocation_summary",
        description="Lighter Kubecost allocation summary (`/model/allocation/summary`): per-aggregation "
                    "totalCost + efficiency only, no per-resource detail. Use for a quick 'which namespace "
                    "/ controller spends the most' question. Read-only.",
        input_schema={
            "type": "object",
            "properties": {
                **_WINDOW_PROP,
                "aggregate": {"type": "string",
                              "description": "Group by: 'namespace' (default), 'controller', 'label:<key>', ..."},
            },
        },
        handler=_h_allocation_summary,
    ),
    MCPTool(
        name="billing_kubecost_assets",
        description="Kubernetes ASSET cost from Kubecost (`/model/assets`): the underlying infrastructure "
                    "line-items — nodes, disks, load balancers. Aggregate by 'type' (default) for "
                    "node/disk/LB split, or 'node' for per-node cost. Complements allocation (which "
                    "attributes to workloads). Read-only.",
        input_schema={
            "type": "object",
            "properties": {
                **_WINDOW_PROP,
                "aggregate": {"type": "string",
                              "description": "Group by: 'type' (default: Node/Disk/LoadBalancer/...), "
                                             "'node', 'cluster', 'provider'."},
            },
        },
        handler=_h_assets,
    ),
    MCPTool(
        name="billing_kubecost_cloud_cost",
        description="Cloud-provider billing cost reconciled by Kubecost's cloud-cost integration "
                    "(`/model/cloudCost`) — actual out-of-cluster spend too. DEGRADES GRACEFULLY: if no "
                    "cloud integration is configured the endpoint 404s / returns empty and this returns an "
                    "empty result with a note (use billing_gcp_* for the raw GCP invoice instead). Read-only.",
        input_schema={
            "type": "object",
            "properties": {
                **_WINDOW_PROP,
                "aggregate": {"type": "string",
                              "description": "Group by: 'provider' (default), 'account', 'project', 'service'."},
            },
        },
        handler=_h_cloud_cost,
    ),
]


def get_tool(name: str) -> MCPTool:
    for t in BILLING_KUBECOST_TOOLS:
        if t.name == name:
            return t
    raise KeyError(f"unknown billing_kubecost tool: {name}")


# --- fake backend (FR-012; offline tests) ----------------------------------

async def _fake_get(path: str, params: dict | None = None, *, tool: str = "billing_kubecost") -> Any:
    """Canned stand-in for the module-level GET, keyed by `/model` path
    fragment. Shapes mirror the Kubecost API the handlers parse. Returns
    ``(status_code, json)``. No network / cluster / creds."""
    if path == "/model/allocation":
        return 200, {"data": [{
            "kube-system": {"name": "kube-system", "cpuCost": 120.5, "ramCost": 60.0,
                            "pvCost": 0.0, "networkCost": 5.0, "totalCost": 185.5,
                            "totalEfficiency": 0.42},
            "opsrag": {"name": "opsrag", "cpuCost": 40.0, "ramCost": 88.0,
                       "pvCost": 12.0, "networkCost": 2.0, "totalCost": 142.0,
                       "totalEfficiency": 0.71},
            "__idle__": {"name": "__idle__", "cpuCost": 300.0, "ramCost": 150.0,
                         "pvCost": 0.0, "networkCost": 0.0, "totalCost": 450.0,
                         "totalEfficiency": 0.0},
        }]}
    if path == "/model/allocation/summary":
        return 200, {"data": {"sets": [{"allocations": {
            "opsrag": {"name": "opsrag", "totalCost": 142.0, "totalEfficiency": 0.71},
            "kube-system": {"name": "kube-system", "totalCost": 185.5, "totalEfficiency": 0.42},
        }}]}}
    if path == "/model/assets":
        return 200, {"data": [{
            "node-1": {"name": "gke-prod-pool-1", "type": "Node", "totalCost": 640.0},
            "disk-1": {"name": "pvc-abc", "type": "Disk", "totalCost": 44.0},
            "lb-1": {"name": "ingress-lb", "type": "LoadBalancer", "totalCost": 21.5},
        }]}
    if path == "/model/cloudCost":
        return 200, {"data": {"sets": [{"cloudCosts": {
            "Compute Engine": {"name": "Compute Engine", "amortizedNetCost": {"cost": 6275.0}},
            "Cloud SQL": {"name": "Cloud SQL", "netCost": 6149.0},
        }}]}}
    return 404, {}


def build_fake():
    """Return a FakeMCP exposing the Kubecost billing tools wired to an
    offline backend. Needs NO cluster / network / creds: the module-level
    `_get` is swapped for a canned dispatcher, restored by `teardown`."""
    import opsrag.mcp.billing_kubecost as _mod
    from opsrag.mcp._fake import FakeMCP

    _orig = _mod._get
    _mod._get = _fake_get

    def _restore() -> None:
        _mod._get = _orig

    return FakeMCP(tools=list(BILLING_KUBECOST_TOOLS), client=None, teardown=_restore)
