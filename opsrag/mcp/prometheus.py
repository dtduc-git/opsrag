"""Prometheus MCP-style tools for OpsRAG (Sub-sprint 4).

Read-only PromQL access to every registered cluster's Prometheus.
Reaches the in-cluster `monitoring-main-prometheus` (or `monitoring-istio-prometheus`)
service via the Kubernetes API server's service-proxy:

    https://<api-server>/api/v1/namespaces/monitoring/services/<svc>:9090/proxy/api/v1/query

This means:
- Same auth as K8s MCP (patched-kubeconfig with ADC token locally,
  in-cluster service account in production)
- Multi-cluster works automatically per kube-context
- No port-forward, no separate Prometheus auth, no broker-network reachability concerns

## Tools (6 read-only)

| Tool                       | Endpoint                          |
|----------------------------|-----------------------------------|
| `prometheus_query`         | `/api/v1/query`                   |
| `prometheus_query_range`   | `/api/v1/query_range`             |
| `prometheus_series`        | `/api/v1/series`                  |
| `prometheus_label_values`  | `/api/v1/label/{name}/values`     |
| `prometheus_alerts`        | `/api/v1/alerts`                  |
| `prometheus_targets`       | `/api/v1/targets?state=active`    |

## Use cases unlocked

- "Who consumes topic X?" -> `prometheus_query('count by (consumergroup) (kafka_consumergroup_lag{topic="X"})')`
- "Anyone lagging?" -> `prometheus_query('topk(5, sum by (consumergroup,topic) (kafka_consumergroup_lag) > 1000)')`
- "<service> CPU?" -> `prometheus_query('rate(container_cpu_usage_seconds_total{namespace="<namespace>"}[5m])')`
- "Pods restarting?" -> `prometheus_query('changes(kube_pod_container_status_restarts_total[1h]) > 0')`
- "Alerts firing?" -> `prometheus_alerts()`
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any
from urllib.parse import quote

_RELATIVE_TIME_RE = re.compile(r"^\s*now\s*(?:([+-])\s*(\d+)\s*([smhd]))?\s*$", re.IGNORECASE)
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def _resolve_relative_time(value: Any) -> Any:
    """Translate Grafana-style `now`, `now-10m`, `now+1h` shorthands into
    unix seconds. Pass-through for anything Prometheus already accepts
    (numeric strings, RFC3339, ints/floats). The agent reaches for these
    shorthands by reflex -- accepting them server-side beats teaching it
    to compute timestamps with arithmetic-on-strings.
    """
    if value is None or isinstance(value, (int, float)):
        return value
    if not isinstance(value, str):
        return value
    m = _RELATIVE_TIME_RE.match(value)
    if not m:
        return value
    now = time.time()
    sign, amount, unit = m.group(1), m.group(2), m.group(3)
    if not sign:
        return str(int(now))
    delta = int(amount) * _UNIT_SECONDS[unit.lower()]
    target = now + delta if sign == "+" else now - delta
    return str(int(target))

from opsrag.mcp import kubernetes as _k8s
from opsrag.mcp.gitlab import MCPTool
from opsrag.mcp.kubernetes import (
    _default_cluster,
    _ensure_imports,
    _ensure_kubeconfig,
    refresh_adc_token,
)

_log = logging.getLogger("opsrag.mcp.prometheus")

# In-cluster Prometheus service. Two instances exist in every
# cluster: `monitoring-main-prometheus` (app + infra metrics, default)
# and `monitoring-istio-prometheus` (istio mesh metrics).
DEFAULT_PROMETHEUS_SERVICE = "monitoring-main-prometheus"
ISTIO_PROMETHEUS_SERVICE = "monitoring-istio-prometheus"
PROMETHEUS_NAMESPACE = "monitoring"
PROMETHEUS_PORT = 9090

# Per-cluster client cache (we reuse kubernetes_asyncio's API client).
_clients: dict[str, Any] = {}
_RESULT_TRUNCATE_SERIES = 200  # cap the number of series in a response


def _default_prometheus_cluster() -> str:
    """Default cluster for Prometheus queries. Honors a Prometheus-specific
    env override, else the shared K8s default (env or first configured
    cluster). Returns "" when nothing is configured."""
    return os.environ.get("OPSRAG_PROMETHEUS_DEFAULT_CLUSTER") or _default_cluster()


def _resolve_prometheus_cluster(args: dict) -> str:
    """Explicit `cluster` arg wins; else the configured default. Raises a
    clear error when neither is available (empty deployment context)."""
    cluster = args.get("cluster") or _default_prometheus_cluster()
    if not cluster:
        raise RuntimeError(
            "prometheus mcp: no cluster specified and none configured. Pass "
            "a `cluster` argument, or set kubernetes.clusters in the "
            "deployment context (or OPSRAG_PROMETHEUS_DEFAULT_CLUSTER / "
            "OPSRAG_K8S_DEFAULT_CLUSTER env)."
        )
    return cluster


async def _get_proxy_config(cluster: str) -> dict:
    """Returns {host, token, verify} for a cluster -- direct httpx GET
    to the API server's service-proxy URL with bearer token. We bypass
    kubernetes_asyncio's `connect_..._with_path` because the K8s API
    server's proxy doesn't preserve query strings appended to the
    service-proxy path; httpx with explicit `params=` works correctly.

    Two auth paths mirror the K8s MCP:
      - GKE-API mode (preferred): when clusters are registered via
        config.k8s.clusters, use ADC + Container API -> endpoint/CA/token.
      - Legacy kubeconfig mode (local dev fallback).
    """
    if cluster in _clients:
        return _clients[cluster]
    await _ensure_kubeconfig()
    _ensure_imports()
    from kubernetes_asyncio import config as k8s_config

    if _k8s._cluster_coords:
        coords = _k8s._cluster_coords.get(cluster)
        if not coords:
            raise RuntimeError(
                f"prometheus mcp: cluster {cluster!r} not registered. "
                f"Known: {sorted(_k8s._cluster_coords)}. Add to config.k8s.clusters."
            )
        info = await _k8s._gke_get_cluster(
            project=coords["project"],
            location=coords["location"],
            name=coords["name"],
        )
        # Build the CA path the same way the K8s API client does so the
        # cert is reused across clusters' lifetimes.
        import base64
        import tempfile
        if _k8s._ca_cert_dir is None:
            _k8s._ca_cert_dir = tempfile.mkdtemp(prefix="opsrag-ca-")
        ca_path = os.path.join(_k8s._ca_cert_dir, f"{cluster}.crt")
        if not os.path.exists(ca_path):
            with open(ca_path, "wb") as f:
                f.write(base64.b64decode(info["ca_cert_b64"]))
        token = await _k8s._get_adc_token()
        out = {
            "host": f"https://{info['endpoint']}",
            "token": token,
            "verify": ca_path,
        }
        _clients[cluster] = out
        return out

    cfg_path = os.environ.get("KUBECONFIG") or "/root/.kube/config"
    api_client = await k8s_config.new_client_from_config(
        config_file=cfg_path, context=cluster,
    )
    cfg = api_client.configuration
    # Bearer token from the configuration's api_key dict.
    token = ""
    api_key = cfg.api_key or {}
    auth_val = api_key.get("authorization") or api_key.get("BearerToken") or ""
    if auth_val.startswith("Bearer "):
        token = auth_val[len("Bearer "):]
    elif auth_val:
        token = auth_val
    out = {
        "host": cfg.host.rstrip("/"),
        "token": token,
        "verify": cfg.ssl_ca_cert if cfg.ssl_ca_cert else False,
    }
    await api_client.close()
    _clients[cluster] = out
    return out


async def _proxy_get(
    cluster: str,
    service: str,
    path: str,
    params: dict | None = None,
    *,
    _retried: bool = False,
) -> Any:
    """GET via the K8s API server service-proxy. Returns parsed JSON.
    Uses httpx so query strings reach Prometheus intact.

    Auto-refreshes the ADC token on 401 (token lifetime ~1h; the
    container can outlive multiple token windows). Retries once after
    refresh; if the second attempt also 401s, surfaces the error."""
    import httpx
    cfg = await _get_proxy_config(cluster)
    url = (
        f"{cfg['host']}/api/v1/namespaces/{PROMETHEUS_NAMESPACE}"
        f"/services/{service}:{PROMETHEUS_PORT}/proxy{path}"
    )
    headers = {}
    if cfg["token"]:
        headers["Authorization"] = f"Bearer {cfg['token']}"
    async with httpx.AsyncClient(verify=cfg["verify"], timeout=30.0) as http:
        resp = await http.get(url, params=params or {}, headers=headers)
        if resp.status_code == 401 and not _retried:
            # Token expired. Refresh + invalidate cached configs + retry once.
            _log.warning(
                "prometheus proxy 401 on %s -- refreshing ADC token + retry", cluster,
            )
            await refresh_adc_token()
            _clients.pop(cluster, None)
            return await _proxy_get(cluster, service, path, params=params, _retried=True)
        if resp.status_code >= 400:
            return {
                "status": "error",
                "errorType": f"http_{resp.status_code}",
                "error": resp.text[:500],
            }
        try:
            return resp.json()
        except Exception:
            return {
                "status": "error",
                "errorType": "parse",
                "error": resp.text[:500],
            }


_MAX_POINTS_PER_SERIES = 240  # 4h at 60s step, 24h at 6m step -- keeps matrix JSON under ~40KB
_MAX_MATRIX_SERIES = 16       # tighter than the generic vector cap; matrix bytes scale with Nxpointsxprecision
_VALUE_DECIMALS = 4           # `0.4316` instead of `0.4316126996072033` -- 4 decimals is plenty for CPU/lag


def _round_value(v: object) -> object:
    """Trim float precision on Prometheus value strings. Prometheus
    returns each value as a string like "0.4316126996072033"; the
    full ~17-digit precision is meaningless for chart/LLM context and
    triples the JSON byte size. Returns input unchanged on parse fail
    so we never silently mutate non-numeric payloads.
    """
    if isinstance(v, (int, float)):
        if not isinstance(v, bool) and isinstance(v, float):
            return round(v, _VALUE_DECIMALS)
        return v
    if isinstance(v, str):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return v
        # Preserve int-typed values (e.g. lag counts) as ints.
        if f.is_integer():
            return str(int(f))
        return f"{f:.{_VALUE_DECIMALS}f}"
    return v


def _trim_series(result: dict, max_series: int = _RESULT_TRUNCATE_SERIES) -> dict:
    """Cap a Prometheus response payload to the LLM-relevant slice.

    Matrix results get THREE caps:
      1. series count (<= _MAX_MATRIX_SERIES, tighter than the vector cap)
      2. points-per-series (<= _MAX_POINTS_PER_SERIES, keeps recent end)
      3. float precision (<= _VALUE_DECIMALS, 4dp is plenty for charts/LLM)

    Without these caps a 1h x 60s step x 12-pod query JSON-serializes
    to ~80KB, blowing the downstream 64KB truncation in `_safe_json`
    and breaking both the chart extractor's JSON parse AND the
    generator LLM (which silently times out on bloated context).
    """
    if result.get("resultType") not in ("vector", "matrix"):
        return result
    is_matrix = result.get("resultType") == "matrix"
    effective_max = min(max_series, _MAX_MATRIX_SERIES) if is_matrix else max_series
    series = result.get("result", []) or []
    total = len(series)
    sliced = series[:effective_max]
    pts_truncated = False
    if is_matrix:
        new_series = []
        for s in sliced:
            if not isinstance(s, dict):
                new_series.append(s)
                continue
            values = s.get("values") or []
            if len(values) > _MAX_POINTS_PER_SERIES:
                pts_truncated = True
                values = values[-_MAX_POINTS_PER_SERIES:]
            # Round float precision on every (ts, val) pair.
            new_values = []
            for pair in values:
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    new_values.append([pair[0], _round_value(pair[1])])
                else:
                    new_values.append(pair)
            new_series.append({**s, "values": new_values, **({"_points_truncated": len(s.get("values") or [])} if pts_truncated and len(s.get("values") or []) > _MAX_POINTS_PER_SERIES else {})})
        sliced = new_series
    else:
        # Vector -- round each instant value too.
        new_series = []
        for s in sliced:
            if isinstance(s, dict) and isinstance(s.get("value"), (list, tuple)) and len(s["value"]) >= 2:
                new_series.append({**s, "value": [s["value"][0], _round_value(s["value"][1])]})
            else:
                new_series.append(s)
        sliced = new_series
    out = {**result, "result": sliced}
    if total > effective_max:
        out["_truncated"] = True
        out["_total_series"] = total
        out["_returned_series"] = len(sliced)
    if pts_truncated:
        out["_points_per_series_cap"] = _MAX_POINTS_PER_SERIES
    return out


# --- handlers -------------------------------------------------------


async def _h_query(_unused, args: dict) -> Any:
    cluster = _resolve_prometheus_cluster(args)
    service = ISTIO_PROMETHEUS_SERVICE if args.get("istio") else DEFAULT_PROMETHEUS_SERVICE
    params = {"query": args["query"]}
    if args.get("time"):
        params["time"] = _resolve_relative_time(args["time"])
    resp = await _proxy_get(cluster, service, "/api/v1/query", params=params)
    if resp.get("status") != "success":
        return {"cluster": cluster, "error": resp.get("error", "unknown"), "errorType": resp.get("errorType")}
    data = _trim_series(resp.get("data", {}))
    return {"cluster": cluster, "service": service, "data": data}


async def _h_query_range(_unused, args: dict) -> Any:
    cluster = _resolve_prometheus_cluster(args)
    service = ISTIO_PROMETHEUS_SERVICE if args.get("istio") else DEFAULT_PROMETHEUS_SERVICE
    params = {
        "query": args["query"],
        "start": _resolve_relative_time(args["start"]),
        "end": _resolve_relative_time(args["end"]),
        "step": args.get("step", "60s"),
    }
    resp = await _proxy_get(cluster, service, "/api/v1/query_range", params=params)
    if resp.get("status") != "success":
        return {"cluster": cluster, "error": resp.get("error", "unknown"), "errorType": resp.get("errorType")}
    data = _trim_series(resp.get("data", {}))
    return {"cluster": cluster, "service": service, "data": data}


async def _h_series(_unused, args: dict) -> Any:
    cluster = _resolve_prometheus_cluster(args)
    service = ISTIO_PROMETHEUS_SERVICE if args.get("istio") else DEFAULT_PROMETHEUS_SERVICE
    match_list = args["match"]
    if isinstance(match_list, str):
        match_list = [match_list]
    # httpx supports list-valued params for the repeating `match[]` key.
    params: dict = {"match[]": match_list, "limit": int(args.get("limit") or 200)}
    if args.get("start"):
        params["start"] = _resolve_relative_time(args["start"])
    if args.get("end"):
        params["end"] = _resolve_relative_time(args["end"])
    resp = await _proxy_get(cluster, service, "/api/v1/series", params=params)
    if resp.get("status") != "success":
        return {"cluster": cluster, "error": resp.get("error", "unknown")}
    return {
        "cluster": cluster,
        "service": service,
        "match": match_list,
        "count": len(resp.get("data") or []),
        "series": (resp.get("data") or [])[:_RESULT_TRUNCATE_SERIES],
    }


async def _h_label_values(_unused, args: dict) -> Any:
    cluster = _resolve_prometheus_cluster(args)
    service = ISTIO_PROMETHEUS_SERVICE if args.get("istio") else DEFAULT_PROMETHEUS_SERVICE
    label = args["label"]
    params = {}
    if args.get("match"):
        # Single match[] for label_values (repeats not as common here)
        params["match[]"] = args["match"]
    resp = await _proxy_get(
        cluster, service, f"/api/v1/label/{quote(label, safe='')}/values", params=params,
    )
    return {
        "cluster": cluster, "service": service, "label": label,
        "count": len(resp.get("data") or []),
        "values": (resp.get("data") or [])[:500],
    }


async def _h_alerts(_unused, args: dict) -> Any:
    cluster = _resolve_prometheus_cluster(args)
    service = ISTIO_PROMETHEUS_SERVICE if args.get("istio") else DEFAULT_PROMETHEUS_SERVICE
    resp = await _proxy_get(cluster, service, "/api/v1/alerts")
    if resp.get("status") != "success":
        return {"cluster": cluster, "error": resp.get("error", "unknown")}
    alerts = (resp.get("data") or {}).get("alerts") or []
    # Trim alert payloads -- keep only what's useful
    trimmed = []
    for a in alerts:
        trimmed.append({
            "name": a.get("labels", {}).get("alertname"),
            "state": a.get("state"),
            "severity": a.get("labels", {}).get("severity"),
            "active_at": a.get("activeAt"),
            "value": a.get("value"),
            "labels": a.get("labels", {}),
            "annotations": a.get("annotations", {}),
        })
    # Most-recent first by activeAt.
    trimmed.sort(key=lambda x: x.get("active_at") or "", reverse=True)
    return {
        "cluster": cluster, "service": service,
        "count": len(trimmed),
        "firing": sum(1 for a in trimmed if a.get("state") == "firing"),
        "pending": sum(1 for a in trimmed if a.get("state") == "pending"),
        "alerts": trimmed[:_RESULT_TRUNCATE_SERIES],
    }


async def _h_targets(_unused, args: dict) -> Any:
    cluster = _resolve_prometheus_cluster(args)
    service = ISTIO_PROMETHEUS_SERVICE if args.get("istio") else DEFAULT_PROMETHEUS_SERVICE
    state = args.get("state") or "active"
    resp = await _proxy_get(cluster, service, "/api/v1/targets", params={"state": state})
    if resp.get("status") != "success":
        return {"cluster": cluster, "error": resp.get("error", "unknown")}
    data = resp.get("data") or {}
    active = data.get("activeTargets") or []
    # Just return health summary + unhealthy details
    healthy = sum(1 for t in active if t.get("health") == "up")
    unhealthy = [
        {
            "scrapePool": t.get("scrapePool"),
            "labels": t.get("labels", {}),
            "lastError": t.get("lastError"),
            "lastScrape": t.get("lastScrape"),
        }
        for t in active if t.get("health") != "up"
    ]
    return {
        "cluster": cluster, "service": service,
        "active_total": len(active),
        "active_healthy": healthy,
        "active_unhealthy": len(unhealthy),
        "unhealthy_targets": unhealthy[:50],
    }


# --- tool registry --------------------------------------------------


_CLUSTER_PROP = {
    "type": "string",
    "description": (
        "Kube-context name (Prometheus is in-cluster per environment). "
        "Taken from the deployment's configured clusters. Defaults to "
        "OPSRAG_PROMETHEUS_DEFAULT_CLUSTER / OPSRAG_K8S_DEFAULT_CLUSTER "
        "env, else the first configured cluster. Required when no clusters "
        "are configured."
    ),
}

_ISTIO_PROP = {
    "type": "boolean",
    "description": (
        "Set true to query the istio Prometheus instance "
        f"({ISTIO_PROMETHEUS_SERVICE}) for mesh / sidecar metrics. "
        f"Default false -> app/infra Prometheus ({DEFAULT_PROMETHEUS_SERVICE})."
    ),
}


PROMETHEUS_TOOLS: list[MCPTool] = [
    MCPTool(
        name="prometheus_query",
        description=(
            "Run an instant PromQL query against a registered Prometheus. "
            "Returns a vector or scalar. Best for single-point-in-time questions. "
            "Use `prometheus_query_range` for graphs over time.\n\n"
            "## Common SRE PromQL recipes (use these EXACT metric names -- kube-state-metrics v2+ schema):\n\n"
            "**Pod CPU usage** (cores): `sum by (pod) (rate(container_cpu_usage_seconds_total{namespace=\"<ns>\",container!=\"\",container!=\"POD\"}[5m]))`\n"
            "**Pod CPU %-of-limit**: same as above divided by `sum by (pod) (kube_pod_container_resource_limits{namespace=\"<ns>\",resource=\"cpu\"}) * 100`\n"
            "**Pod memory usage** (bytes): `sum by (pod) (container_memory_working_set_bytes{namespace=\"<ns>\",container!=\"\",container!=\"POD\"})`\n"
            "**Pod memory %-of-limit**: same divided by `sum by (pod) (kube_pod_container_resource_limits{namespace=\"<ns>\",resource=\"memory\"}) * 100`\n"
            "**Pod restart count last hour**: `changes(kube_pod_container_status_restarts_total{namespace=\"<ns>\"}[1h])`\n"
            "**CrashLoopBackOff pods**: `kube_pod_container_status_waiting_reason{reason=\"CrashLoopBackOff\"} > 0`\n"
            "**Kafka consumer lag**: `kafka_consumergroup_lag{topic=\"<topic>\"}` -- labels: `consumergroup`, `topic`, `partition`\n"
            "**Lagging groups (>1k msgs)**: `topk(10, sum by (consumergroup, topic) (kafka_consumergroup_lag) > 1000)`\n"
            "**Topic -> consumers**: `count by (consumergroup) (kafka_consumergroup_lag{topic=\"<topic>\"})` -- single hit per group\n"
            "**Deployment replicas mismatch**: `kube_deployment_status_replicas_available{namespace=\"<ns>\"} != kube_deployment_spec_replicas{namespace=\"<ns>\"}`\n"
            "**Active firing alerts**: `ALERTS{alertstate=\"firing\"}` (use `prometheus_alerts` for richer detail)\n"
            "**HTTP 5xx rate** (Istio): use `istio: true` flag, query `sum by (destination_service_name) (rate(istio_requests_total{response_code=~\"5..\"}[5m]))`\n\n"
            "When a query returns empty, the metric name might differ from these recipes -- use `prometheus_label_values(label='__name__')` to discover available metrics, OR `prometheus_series` to inspect labels on a known metric."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "cluster": _CLUSTER_PROP,
                "istio": _ISTIO_PROP,
                "query": {"type": "string", "description": "PromQL expression"},
                "time": {"type": "string", "description": "RFC3339 or unix seconds; default now"},
            },
            "required": ["query"],
        },
        handler=_h_query,
    ),
    MCPTool(
        name="prometheus_query_range",
        description=(
            "Run a range PromQL query (returns a matrix of [timestamp,value] "
            "pairs per series). Use for trending -- CPU over the last hour, "
            "lag over an incident window, request-rate before/after a deploy.\n\n"
            "**Time inputs**: pass `now`, `now-10m`, `now-1h`, `now-2d` etc. "
            "directly as strings -- the server resolves them to unix seconds. "
            "Unix seconds or RFC3339 also accepted. Do NOT try to compute "
            "timestamps client-side or call any `now`/`timedelta` helper "
            "(those do not exist as tools)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "cluster": _CLUSTER_PROP,
                "istio": _ISTIO_PROP,
                "query": {"type": "string"},
                "start": {"type": "string", "description": "'now-10m', RFC3339, or unix seconds"},
                "end": {"type": "string", "description": "'now', RFC3339, or unix seconds"},
                "step": {"type": "string", "description": "Resolution e.g. '60s', '5m'. Default 60s."},
            },
            "required": ["query", "start", "end"],
        },
        handler=_h_query_range,
    ),
    MCPTool(
        name="prometheus_series",
        description=(
            "Find series matching a label selector (no aggregation). Useful "
            "for discovery -- 'what topics does kafka_consumergroup_lag have?' "
            "-> match=[`kafka_consumergroup_lag`]."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "cluster": _CLUSTER_PROP,
                "istio": _ISTIO_PROP,
                "match": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "PromQL series-selectors, e.g. ['up{job=\"foo\"}', 'kafka_consumergroup_lag']",
                },
                "start": {"type": "string"},
                "end": {"type": "string"},
                "limit": {"type": "number", "description": "Max series; default 200"},
            },
            "required": ["match"],
        },
        handler=_h_series,
    ),
    MCPTool(
        name="prometheus_label_values",
        description=(
            "Return the distinct values of a label. Optional `match` selector "
            "narrows scope. E.g. label='topic', match='kafka_consumergroup_lag' "
            "-> every Kafka topic with a consumer group."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "cluster": _CLUSTER_PROP,
                "istio": _ISTIO_PROP,
                "label": {"type": "string"},
                "match": {"type": "string", "description": "Optional series-selector to narrow scope"},
            },
            "required": ["label"],
        },
        handler=_h_label_values,
    ),
    MCPTool(
        name="prometheus_alerts",
        description=(
            "List currently active Prometheus alerts (firing or pending). "
            "Sorted most-recent first by activeAt. Use this for 'is anything "
            "wrong right now in cluster X?'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "cluster": _CLUSTER_PROP,
                "istio": _ISTIO_PROP,
            },
        },
        handler=_h_alerts,
    ),
    MCPTool(
        name="prometheus_targets",
        description=(
            "Scrape-target health summary. Surfaces unhealthy targets with "
            "lastError messages -- useful when metrics for a service look stale."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "cluster": _CLUSTER_PROP,
                "istio": _ISTIO_PROP,
                "state": {"type": "string", "enum": ["active", "dropped", "any"]},
            },
        },
        handler=_h_targets,
    ),
]


def get_tool(name: str) -> MCPTool:
    for t in PROMETHEUS_TOOLS:
        if t.name == name:
            return t
    raise KeyError(f"unknown prometheus tool: {name}")


# --- fake backend (FR-012; integration tests) ----------------------


async def _fake_proxy_get(
    cluster: str,
    service: str,
    path: str,
    params: dict | None = None,
    *,
    _retried: bool = False,
) -> Any:
    """Offline stand-in for `_proxy_get`. Returns canned, shape-faithful
    Prometheus HTTP API JSON keyed by the endpoint path. No cluster, no
    Prometheus, no network. Mirrors the real `_proxy_get` signature so the
    handlers run unchanged."""
    if path == "/api/v1/query":
        return {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {
                        "metric": {"__name__": "up", "job": "node", "instance": "node-0"},
                        "value": [1700000000, "1"],
                    }
                ],
            },
        }
    if path == "/api/v1/query_range":
        return {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {"__name__": "up", "job": "node", "instance": "node-0"},
                        "values": [[1700000000, "1"], [1700000060, "1"]],
                    }
                ],
            },
        }
    if path == "/api/v1/series":
        return {
            "status": "success",
            "data": [
                {"__name__": "up", "job": "node", "instance": "node-0"},
            ],
        }
    if path.startswith("/api/v1/label/") and path.endswith("/values"):
        return {"status": "success", "data": ["node", "kubelet"]}
    if path == "/api/v1/alerts":
        return {
            "status": "success",
            "data": {
                "alerts": [
                    {
                        "labels": {"alertname": "HighCpu", "severity": "warning"},
                        "annotations": {"summary": "cpu high"},
                        "state": "firing",
                        "activeAt": "2026-01-01T00:00:00Z",
                        "value": "0.9",
                    }
                ]
            },
        }
    if path == "/api/v1/targets":
        return {
            "status": "success",
            "data": {
                "activeTargets": [
                    {
                        "scrapePool": "node",
                        "labels": {"job": "node", "instance": "node-0"},
                        "health": "up",
                        "lastError": "",
                        "lastScrape": "2026-01-01T00:00:00Z",
                    }
                ]
            },
        }
    return {"status": "success", "data": {}}


def build_fake():
    """Return a FakeMCP exposing the Prometheus tools wired to an offline
    backend. Needs NO cluster / Prometheus / network: a DeploymentContext
    with one configured cluster is installed so cluster resolution
    succeeds, and the module-level `_proxy_get` is swapped for a canned
    responder. `teardown` restores both."""
    import opsrag.mcp.prometheus as _mod
    from opsrag.agent.prompt_render import active_deployment, set_active_deployment
    from opsrag.context import DeploymentContext, KubernetesContext
    from opsrag.mcp._fake import FakeMCP

    _orig_deployment = active_deployment()
    set_active_deployment(
        DeploymentContext(
            kubernetes=KubernetesContext(clusters={"prod": "example-cluster"}),
        )
    )

    _orig_proxy_get = _mod._proxy_get
    _orig_clients = dict(_mod._clients)
    _mod._proxy_get = _fake_proxy_get
    _mod._clients.clear()

    def _restore() -> None:
        set_active_deployment(_orig_deployment)
        _mod._proxy_get = _orig_proxy_get
        _mod._clients.clear()
        _mod._clients.update(_orig_clients)

    return FakeMCP(
        tools=list(PROMETHEUS_TOOLS), client=None, teardown=_restore,
    )
