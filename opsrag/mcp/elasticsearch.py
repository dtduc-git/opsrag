"""Read-only Elasticsearch / OpenSearch MCP integration (per-environment).

Resolves the endpoint + credentials + field mapping PER ENVIRONMENT through
the unified ``environments:`` registry (Approach A,
``opsrag.environments.resolve_environment``): every handler accepts an optional
``env`` arg (back-compat alias ``cluster``); when omitted, the registry default
env is used. ``_config(env)`` returns ``resolve_environment(env).elasticsearch``
(an ``EsTarget``) turned into a concrete request bundle; ``_get`` / ``_post``
issue read-only httpx requests against it, and each tool is an async
``_h_<verb>`` handler. ``build_fake()`` swaps the network helpers for offline
canned responders.

One OpsRAG instance can target N environments, each with its own URL / index /
field schema / backend / TLS / reach mode -- no single global endpoint, and no
org-specific field names baked into code (the ``EsTarget.fields`` map
de-hardcodes one org's ECK schema).

Reach modes (``EsTarget.reach``):
- ``direct``       -> hit ``target.url`` over HTTPS (API key or HTTP basic).
- ``port_forward`` -> reach the in-cluster ES at
  ``target.service.target.namespace:target.port`` THROUGH the env's Kubernetes
  API server (``kubernetes.cluster_api_access(env)``), preserving the
  ``Authorization`` header for API-key auth (the bare service-proxy strips it).
- ``proxy``        -> the same, but via the API server's service-proxy URL.

Five read-only tools (all GET, or a read-only ``_search`` / ``_query`` POST):

| tool                            | endpoint                                  |
|---------------------------------|-------------------------------------------|
| ``elasticsearch_list_indices``  | ``GET /_cat/indices?format=json``         |
| ``elasticsearch_get_mappings``  | ``GET /<index>/_mapping``                 |
| ``elasticsearch_search``        | ``POST /<index>/_search``                 |
| ``elasticsearch_esql_query``    | ``POST /_query`` (Elasticsearch only)     |
| ``elasticsearch_cluster_health``| ``GET /_cluster/health``                  |

Works against Elasticsearch and OpenSearch (same wire API for the first three +
health); ``esql_query`` is gated to the ``elasticsearch`` backend. Auth is an
API key (``Authorization: ApiKey <b64>``) or HTTP basic. The credential's role
should be read-only (``read`` / ``view_index_metadata`` / ``monitor``) -- the
API itself does not distinguish reads from writes, so least-privilege creds are
the real read-only boundary (these tools never expose ``_delete_by_query`` /
``_update`` / ``_bulk``).
"""
from __future__ import annotations

import os
import re
from typing import Any

import httpx

from opsrag.mcp.gitlab import MCPTool

_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_LIMIT = 20
_MAX_LIMIT = 200
_RESULT_TRUNCATE_CHARS = 4000


class MCPElasticsearchError(Exception):
    """Read-only ES/OpenSearch tool failure. Carries a short ``reason``
    code (e.g. ``bad_env``, ``bad_args``, ``backend``, ``reach``, ``http``)."""

    def __init__(self, message: str, *, reason: str = "error") -> None:
        super().__init__(message)
        self.reason = reason


def bind(es_config: Any, **_ignored: Any) -> None:
    """BACK-COMPAT SHIM (no-op). Historically this captured a SINGLE global ES
    endpoint into the module-level ``_BOUND``. Endpoint/credential/field
    resolution now flows PER ENV through the unified ``environments:`` registry
    (``bind_environments(cfg)`` synthesizes a single ES env from the legacy
    ``cfg.elasticsearch`` block when no explicit ``environments:`` is set), so
    there is nothing to capture here. Kept so old callers / tests importing the
    symbol still work."""
    return None


# --- config / client -------------------------------------------------------

def _resolve_env(args: dict) -> str | None:
    """Pick the env name for a handler call: explicit ``env`` arg wins, then
    the back-compat ``cluster`` alias, else ``None`` (-> registry default)."""
    return args.get("env") or args.get("cluster") or None


def _config(env: str | None = None) -> dict:
    """Resolve the request bundle for an environment's Elasticsearch target.

    Looks up ``resolve_environment(env).elasticsearch`` (an ``EsTarget``; ``env``
    None -> registry default) and turns it into a concrete bundle:
    ``{reach, env, url, headers, auth, backend, default_index, verify_ssl,
    fields, target}``. Credentials are read from env vars (``api_key_env`` /
    ``username_env`` + ``password_env``) at call time -- never captured at bind.

    Raises ``MCPElasticsearchError`` when the env has no ES target, or (for
    ``reach=direct``) when no URL is configured. ``EnvironmentResolutionError``
    (unknown / empty registry) propagates for a clear caller-facing error."""
    from opsrag.environments import resolve_environment

    target = resolve_environment(env).elasticsearch
    if target is None:
        name = env or "(default)"
        raise MCPElasticsearchError(
            f"elasticsearch not configured for env {name!r} -- add an "
            f"`elasticsearch:` block under environments.targets.{name}.",
            reason="bad_env",
        )

    backend = (getattr(target, "backend", "elasticsearch") or "elasticsearch").lower()
    if backend not in ("elasticsearch", "opensearch"):
        backend = "elasticsearch"

    # Credentials (env-only; prefer API key). Headers carry the API key so it
    # survives a k8s port-forward tunnel (the service-proxy strips Authorization,
    # which is exactly why port_forward exists).
    headers = {"Accept": "application/json"}
    auth = None
    api_key_env = getattr(target, "api_key_env", None)
    username_env = getattr(target, "username_env", None)
    password_env = getattr(target, "password_env", None)
    api_key = (os.environ.get(api_key_env) or "").strip() if api_key_env else ""
    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"
    elif username_env:
        username = (os.environ.get(username_env) or "").strip()
        password = os.environ.get(password_env) if password_env else ""
        if username:
            auth = (username, password or "")

    return {
        "reach": getattr(target, "reach", "direct") or "direct",
        "env": env,
        "url": (getattr(target, "url", None) or "").strip().rstrip("/") or None,
        "headers": headers,
        "auth": auth,
        "backend": backend,
        "default_index": getattr(target, "index_pattern", "*") or "*",
        "verify_ssl": bool(getattr(target, "verify_ssl", True)),
        "fields": dict(getattr(target, "fields", {}) or {}),
        "target": target,
    }


async def _base(cfg: dict) -> tuple[str, dict, Any, Any]:
    """Resolve ``(base_url, headers, auth, verify)`` for the configured reach.

    - ``direct``       -> ``cfg["url"]`` + API-key/basic auth + ``verify_ssl``.
    - ``port_forward`` -> the in-cluster service reached through the env's k8s
      API server (``cluster_api_access``); the ES ``Authorization`` header is
      preserved on top of the cluster bearer token (so API-key auth survives).
    - ``proxy``        -> the API server's service-proxy URL for the service.

    The ``port_forward`` / ``proxy`` paths build the URL from
    ``target.service`` / ``target.namespace`` / ``target.port`` and use the
    cluster's CA for TLS verification."""
    reach = cfg["reach"]
    if reach == "direct":
        if not cfg["url"]:
            raise MCPElasticsearchError(
                "elasticsearch mcp: reach=direct but no `url` configured for "
                "this environment's elasticsearch target.",
                reason="bad_env",
            )
        return cfg["url"], dict(cfg["headers"]), cfg["auth"], cfg["verify_ssl"]

    if reach in ("port_forward", "proxy"):
        target = cfg["target"]
        service = getattr(target, "service", None)
        namespace = getattr(target, "namespace", None)
        port = getattr(target, "port", 9200) or 9200
        if not (service and namespace):
            raise MCPElasticsearchError(
                f"elasticsearch mcp: reach={reach} requires `service` and "
                f"`namespace` on the environment's elasticsearch target.",
                reason="bad_env",
            )
        from opsrag.mcp import kubernetes as _k8s

        access = await _k8s.cluster_api_access(cfg["env"])
        host = access["host"].rstrip("/")
        # Service-proxy path on the API server. For reach=proxy this is the
        # whole story; for port_forward we ALSO keep the ES Authorization
        # header (the proxy forwards the path but the cluster bearer token
        # authenticates to the API server -- the ES API key rides through in
        # the preserved Authorization header to authenticate to ES itself).
        scheme = "https" if int(port) in (443, 9243) else "http"
        base = (
            f"{host}/api/v1/namespaces/{namespace}/services/"
            f"{scheme}:{service}:{port}/proxy"
        )
        headers = dict(cfg["headers"])
        if access.get("token"):
            # Cluster bearer token authenticates to the API server. When the
            # ES target itself wants API-key/basic auth we MUST not clobber the
            # ES Authorization header; port_forward preserves it, so the
            # cluster token goes through only when ES has no own auth.
            if reach == "proxy" or "Authorization" not in headers:
                headers["Authorization"] = f"Bearer {access['token']}"
        return base, headers, None, access.get("verify", False)

    raise MCPElasticsearchError(
        f"elasticsearch mcp: unknown reach mode {reach!r} "
        f"(expected direct | port_forward | proxy).",
        reason="reach",
    )


def _redact(text: str) -> str:
    if not text:
        return text
    return re.sub(r"(?i)(ApiKey|Bearer|Basic)\s+\S+", r"\1 [redacted]", str(text))


def _truncate(text: str, limit: int = _RESULT_TRUNCATE_CHARS) -> str:
    if not text:
        return ""
    text = _redact(str(text))
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"


def _clamp(n: Any, *, default: int = _DEFAULT_LIMIT, maximum: int = _MAX_LIMIT) -> int:
    try:
        v = int(n)
    except (TypeError, ValueError):
        return default
    return max(1, min(v, maximum))


async def _refresh_cluster_access(cfg: dict) -> None:
    """On a 401 from a cluster-tunneled reach (port_forward / proxy), re-mint
    the env's API-server token + drop the cached access bundle so the retry
    rebuilds. No-op for ``reach=direct`` (ES creds are static env vars)."""
    if cfg["reach"] in ("port_forward", "proxy"):
        from opsrag.mcp import kubernetes as _k8s

        await _k8s.refresh_adc_token()
        _k8s.invalidate_env_api_cache(cfg["env"])


async def _get(
    cfg: dict, path: str, params: dict | None = None, *,
    tool: str = "elasticsearch", _retried: bool = False,
) -> Any:
    base, headers, auth, verify = await _base(cfg)
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    async with httpx.AsyncClient(
        headers=headers, auth=auth, timeout=_DEFAULT_TIMEOUT_S, verify=verify,
    ) as http:
        resp = await http.get(f"{base}{path}", params=clean)
    if resp.status_code == 401 and not _retried:
        await _refresh_cluster_access(cfg)
        return await _get(cfg, path, params, tool=tool, _retried=True)
    if resp.status_code >= 400:
        raise MCPElasticsearchError(
            f"{tool}: HTTP {resp.status_code} {_truncate(resp.text, 500)}", reason="http",
        )
    return resp.json() if resp.text else {}


async def _post(
    cfg: dict, path: str, body: dict, *,
    tool: str = "elasticsearch", _retried: bool = False,
) -> Any:
    base, headers, auth, verify = await _base(cfg)
    async with httpx.AsyncClient(
        headers={**headers, "Content-Type": "application/json"}, auth=auth,
        timeout=_DEFAULT_TIMEOUT_S, verify=verify,
    ) as http:
        resp = await http.post(f"{base}{path}", json=body)
    if resp.status_code == 401 and not _retried:
        await _refresh_cluster_access(cfg)
        return await _post(cfg, path, body, tool=tool, _retried=True)
    if resp.status_code >= 400:
        raise MCPElasticsearchError(
            f"{tool}: HTTP {resp.status_code} {_truncate(resp.text, 500)}", reason="http",
        )
    return resp.json() if resp.text else {}


def _index(args: dict, cfg: dict) -> str:
    idx = (args.get("index") or "").strip()
    return idx or cfg["default_index"]


def _map_field(cfg: dict, logical: str) -> str:
    """Map a LOGICAL field name (e.g. ``timestamp`` / ``service``) to the
    physical ES field for this environment via ``EsTarget.fields``. Returns the
    logical name unchanged when unmapped (so generic schemas just work)."""
    return cfg["fields"].get(logical, logical)


# --- handlers --------------------------------------------------------------

async def _h_list_indices(_unused, args: dict) -> Any:
    """`GET /_cat/indices?format=json` -- list indices with health + doc
    count. Optional `index` filters by pattern; system indices (`.`) hidden."""
    cfg = _config(_resolve_env(args))
    pattern = (args.get("index") or "").strip()
    path = f"/_cat/indices/{pattern}" if pattern else "/_cat/indices"
    rows = await _get(
        cfg, path, {"format": "json", "h": "index,health,status,docs.count,store.size"},
        tool="elasticsearch_list_indices",
    )
    out = [
        {
            "index": r.get("index"),
            "health": r.get("health"),
            "status": r.get("status"),
            "docs": r.get("docs.count"),
            "size": r.get("store.size"),
        }
        for r in (rows or [])
        if not str(r.get("index", "")).startswith(".")
    ][:_MAX_LIMIT]
    return {"env": cfg["env"], "count": len(out), "indices": out}


async def _h_get_mappings(_unused, args: dict) -> Any:
    """`GET /<index>/_mapping` -- field mappings (schema discovery)."""
    cfg = _config(_resolve_env(args))
    index = _index(args, cfg)
    resp = await _get(cfg, f"/{index}/_mapping", tool="elasticsearch_get_mappings")
    out: dict[str, dict] = {}
    for idx, body in (resp or {}).items():
        props = ((body or {}).get("mappings") or {}).get("properties") or {}
        out[idx] = {f: (v.get("type") or "object") for f, v in props.items()}
    return {"env": cfg["env"], "index": index, "mappings": out}


async def _h_search(_unused, args: dict) -> Any:
    """`POST /<index>/_search` -- read-only Query DSL search. Pass `query`
    (a full DSL object) or `q` (a Lucene query string). Returns up to `size`
    hits (capped).

    `service` (logical) filters via the env's `fields.service` mapping (so a
    caller need not know the physical field name); `sort` defaults to most-
    recent-first on the env's `fields.timestamp` field when set."""
    cfg = _config(_resolve_env(args))
    index = _index(args, cfg)
    size = _clamp(args.get("size"))
    body: dict[str, Any] = {"size": size, "track_total_hits": False}
    if isinstance(args.get("query"), dict):
        query = args["query"]
    elif args.get("q"):
        query = {"query_string": {"query": str(args["q"]), "default_operator": "AND"}}
    else:
        query = {"match_all": {}}
    # Optional logical `service` filter -> physical field via target.fields.
    if args.get("service"):
        svc_field = _map_field(cfg, "service")
        query = {"bool": {"must": [query, {"term": {svc_field: str(args["service"])}}]}}
    body["query"] = query
    if args.get("sort"):
        body["sort"] = args["sort"]
    elif cfg["fields"].get("timestamp"):
        # Default to most-recent-first on the env's timestamp field.
        body["sort"] = [{_map_field(cfg, "timestamp"): {"order": "desc"}}]
    resp = await _post(cfg, f"/{index}/_search", body, tool="elasticsearch_search")
    hits = ((resp or {}).get("hits") or {}).get("hits") or []
    out = []
    for h in hits:
        src = h.get("_source") or {}
        out.append({
            "id": h.get("_id"),
            "index": h.get("_index"),
            "score": h.get("_score"),
            "source": src if len(str(src)) <= _RESULT_TRUNCATE_CHARS else {"_truncated": _truncate(str(src))},
        })
    return {"env": cfg["env"], "index": index, "size": size, "count": len(out), "hits": out}


async def _h_esql_query(_unused, args: dict) -> Any:
    """`POST /_query` -- run an ES|QL query (Elasticsearch only; OpenSearch
    does not implement ES|QL). `query` is the ES|QL string."""
    cfg = _config(_resolve_env(args))
    if cfg["backend"] != "elasticsearch":
        raise MCPElasticsearchError(
            "elasticsearch_esql_query: ES|QL is Elasticsearch-only "
            "(backend is 'opensearch'); use elasticsearch_search instead.",
            reason="backend",
        )
    query = (args.get("query") or "").strip()
    if not query:
        raise MCPElasticsearchError("elasticsearch_esql_query: 'query' is required", reason="bad_args")
    resp = await _post(cfg, "/_query", {"query": query}, tool="elasticsearch_esql_query")
    cols = [c.get("name") for c in (resp.get("columns") or [])]
    rows = (resp.get("values") or [])[:_MAX_LIMIT]
    return {"env": cfg["env"], "columns": cols, "row_count": len(rows), "rows": rows}


async def _h_cluster_health(_unused, args: dict) -> Any:
    """`GET /_cluster/health` -- cluster status (green/yellow/red) + shard
    counts."""
    cfg = _config(_resolve_env(args))
    resp = await _get(cfg, "/_cluster/health", tool="elasticsearch_cluster_health")
    keys = ("cluster_name", "status", "number_of_nodes", "active_shards",
            "relocating_shards", "initializing_shards", "unassigned_shards",
            "active_shards_percent_as_number")
    out = {k: resp.get(k) for k in keys if k in (resp or {})}
    out["env"] = cfg["env"]
    return out


# --- tool specs ------------------------------------------------------------

_INDEX_PROP = {"index": {"type": "string", "description": "Index or index pattern (default: configured index_pattern)."}}
# Optional env selector on EVERY tool. Omitted -> the registry default env, so
# single-env callers are unaffected. `cluster` is a back-compat alias.
_ENV_PROP = {
    "env": {"type": "string", "description": "Target environment name from the environments: registry (default: the configured default env). Back-compat alias: 'cluster'."},
    "cluster": {"type": "string", "description": "Back-compat alias for 'env'."},
}

ES_TOOLS: list[MCPTool] = [
    MCPTool(
        name="elasticsearch_list_indices",
        description="List Elasticsearch/OpenSearch indices with health, status, and document counts. Optional 'index' filters by pattern. Optional 'env' selects the target environment. Read-only.",
        input_schema={"type": "object", "properties": {**_INDEX_PROP, **_ENV_PROP}},
        handler=_h_list_indices,
    ),
    MCPTool(
        name="elasticsearch_get_mappings",
        description="Get the field mappings (schema) for an index — use to discover which fields exist before searching. Optional 'env' selects the target environment. Read-only.",
        input_schema={"type": "object", "properties": {**_INDEX_PROP, **_ENV_PROP}, "required": []},
        handler=_h_get_mappings,
    ),
    MCPTool(
        name="elasticsearch_search",
        description="Run a read-only search. Pass 'q' (Lucene query string, e.g. 'level:error AND service:payments') OR 'query' (a full Query DSL object). Optional 'service' filters by the env's logical service field. 'index', 'size', 'sort', 'env' optional.",
        input_schema={
            "type": "object",
            "properties": {
                **_INDEX_PROP,
                "q": {"type": "string", "description": "Lucene query string."},
                "query": {"type": "object", "description": "Full Elasticsearch Query DSL query object (alternative to 'q')."},
                "service": {"type": "string", "description": "Filter by service name; mapped to the env's physical service field (fields.service)."},
                "size": {"type": "integer", "description": f"Max hits (default {_DEFAULT_LIMIT}, max {_MAX_LIMIT})."},
                "sort": {"description": "Optional sort clause (ES sort syntax). Defaults to most-recent-first on the env's timestamp field when configured."},
                **_ENV_PROP,
            },
        },
        handler=_h_search,
    ),
    MCPTool(
        name="elasticsearch_esql_query",
        description="Run an ES|QL query (Elasticsearch only; not OpenSearch). 'query' is the ES|QL string, e.g. 'FROM logs-* | WHERE level==\"error\" | STATS count() BY service'. Optional 'env' selects the target environment. Read-only.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "ES|QL query string."}, **_ENV_PROP},
            "required": ["query"],
        },
        handler=_h_esql_query,
    ),
    MCPTool(
        name="elasticsearch_cluster_health",
        description="Cluster health: status (green/yellow/red), node count, and shard counts. Optional 'env' selects the target environment. Read-only.",
        input_schema={"type": "object", "properties": {**_ENV_PROP}},
        handler=_h_cluster_health,
    ),
]


def get_tool(name: str) -> MCPTool:
    for t in ES_TOOLS:
        if t.name == name:
            return t
    raise KeyError(f"unknown elasticsearch tool: {name}")


# --- fake backend (FR-012; offline tests) ----------------------------------

async def _fake_get(cfg: dict, path: str, params: dict | None = None, *, tool: str = "elasticsearch", _retried: bool = False) -> Any:
    if path.startswith("/_cat/indices"):
        return [
            {"index": "app-logs-000001", "health": "green", "status": "open",
             "docs.count": "12345", "store.size": "1.2gb"},
            {"index": "metrics-000002", "health": "yellow", "status": "open",
             "docs.count": "678", "store.size": "44mb"},
            {"index": ".kibana_1", "health": "green", "status": "open",
             "docs.count": "5", "store.size": "20kb"},
        ]
    if path.endswith("/_mapping"):
        return {
            "app-logs-000001": {
                "mappings": {"properties": {
                    "@timestamp": {"type": "date"},
                    "level": {"type": "keyword"},
                    "service": {"type": "keyword"},
                    "message": {"type": "text"},
                }}
            }
        }
    if path == "/_cluster/health":
        return {
            "cluster_name": "opsrag-demo", "status": "green", "number_of_nodes": 3,
            "active_shards": 12, "relocating_shards": 0, "initializing_shards": 0,
            "unassigned_shards": 0, "active_shards_percent_as_number": 100.0,
        }
    return {}


async def _fake_post(cfg: dict, path: str, body: dict, *, tool: str = "elasticsearch", _retried: bool = False) -> Any:
    if path.endswith("/_search"):
        return {
            "hits": {"hits": [
                {"_id": "1", "_index": "app-logs-000001", "_score": 1.0,
                 "_source": {"@timestamp": "2026-06-06T00:00:00Z", "level": "error",
                             "service": "payments", "message": "connection refused"}},
                {"_id": "2", "_index": "app-logs-000001", "_score": 0.9,
                 "_source": {"@timestamp": "2026-06-06T00:00:01Z", "level": "error",
                             "service": "payments", "message": "retry exhausted"}},
            ]}
        }
    if path == "/_query":
        return {"columns": [{"name": "service"}, {"name": "count()"}],
                "values": [["payments", 2], ["orders", 1]]}
    return {}


def build_fake():
    """Offline FakeMCP: swaps the module-level `_get`/`_post` for canned
    responders (no network, no ES creds) and binds a single-env registry so
    `_config(env)` resolves to a `direct` ES target. Restores both on teardown."""
    import opsrag.mcp.elasticsearch as _mod
    from opsrag.config import (
        EnvironmentsConfig,
        EnvironmentTarget,
        EsTarget,
    )
    from opsrag.environments import bind_environments, reset_environments
    from opsrag.mcp._fake import FakeMCP

    _orig_get, _orig_post = _mod._get, _mod._post
    _mod._get = _fake_get
    _mod._post = _fake_post

    # Bind a one-env registry ('default') so the env-driven `_config` resolves.
    class _Cfg:
        environments = EnvironmentsConfig(
            default="default",
            targets={
                "default": EnvironmentTarget(
                    elasticsearch=EsTarget(
                        reach="direct", url="http://es.local:9200",
                        backend="elasticsearch", index_pattern="*", verify_ssl=True,
                    ),
                ),
            },
        )

    bind_environments(_Cfg())

    def _restore() -> None:
        _mod._get, _mod._post = _orig_get, _orig_post
        reset_environments()

    return FakeMCP(tools=list(ES_TOOLS), client=None, teardown=_restore)
