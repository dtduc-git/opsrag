"""Read-only Elasticsearch / OpenSearch MCP integration (direct, generic).

Replaces the former GKE-coupled, multi-env, K8s-port-forward implementation
with a simple direct-over-HTTPS client (the canonical ``opsrag/mcp/datadog.py``
pattern): a module-level ``_config()`` resolves the endpoint + credentials from
config/env, ``_get`` / ``_post`` issue read-only httpx requests, and each tool is
an async ``_h_<verb>`` handler. ``build_fake()`` swaps the network helpers for
offline canned responders.

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

# --- module state (bound at startup; falls back to pure-env) ---------------
_BOUND: dict | None = None

_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_LIMIT = 20
_MAX_LIMIT = 200
_RESULT_TRUNCATE_CHARS = 4000


class MCPElasticsearchError(Exception):
    """Read-only ES/OpenSearch tool failure. Carries a short ``reason``
    code (e.g. ``bad_env``, ``bad_args``, ``backend``, ``http``)."""

    def __init__(self, message: str, *, reason: str = "error") -> None:
        super().__init__(message)
        self.reason = reason


def bind(es_config: Any, **_ignored: Any) -> None:
    """Capture the operator's ES config (URL/backend/index/verify_ssl) at
    startup. Credentials are NOT captured here -- they are read from env at
    call time. Extra kwargs (e.g. the legacy ``k8s_clusters=``) are ignored
    for backward compatibility with the previous bind signature."""
    global _BOUND
    _BOUND = {
        "url": (getattr(es_config, "url", "") or "").strip(),
        "url_env": getattr(es_config, "url_env", "ES_URL") or "ES_URL",
        "api_key_env": getattr(es_config, "api_key_env", "ES_API_KEY") or "ES_API_KEY",
        "username_env": getattr(es_config, "username_env", "ES_USERNAME") or "ES_USERNAME",
        "password_env": getattr(es_config, "password_env", "ES_PASSWORD") or "ES_PASSWORD",
        "backend": (getattr(es_config, "backend", "elasticsearch") or "elasticsearch").lower(),
        "default_index": getattr(es_config, "default_index", "*") or "*",
        "verify_ssl": bool(getattr(es_config, "verify_ssl", True)),
    }


# --- config / client -------------------------------------------------------

def _config() -> dict:
    """Resolve the endpoint + auth from the bound config (or pure env when
    unbound). Raises ``MCPElasticsearchError`` when the URL is missing."""
    b = _BOUND or {}
    url_env = b.get("url_env", "ES_URL")
    url = (b.get("url") or "").strip() or (os.environ.get(url_env) or "").strip()
    if not url:
        raise MCPElasticsearchError(
            f"elasticsearch mcp: no endpoint URL (set elasticsearch.url or ${url_env})",
            reason="bad_env",
        )
    api_key = (os.environ.get(b.get("api_key_env", "ES_API_KEY")) or "").strip()
    username = (os.environ.get(b.get("username_env", "ES_USERNAME")) or "").strip()
    password = os.environ.get(b.get("password_env", "ES_PASSWORD")) or ""
    headers = {"Accept": "application/json"}
    auth = None
    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"
    elif username:
        auth = (username, password)
    return {
        "url": url.rstrip("/"),
        "headers": headers,
        "auth": auth,
        "backend": b.get("backend", "elasticsearch"),
        "default_index": b.get("default_index", "*"),
        "verify_ssl": b.get("verify_ssl", True),
    }


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


async def _get(path: str, params: dict | None = None, *, tool: str = "elasticsearch") -> Any:
    cfg = _config()
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    async with httpx.AsyncClient(
        headers=cfg["headers"], auth=cfg["auth"], timeout=_DEFAULT_TIMEOUT_S,
        verify=cfg["verify_ssl"],
    ) as http:
        resp = await http.get(f"{cfg['url']}{path}", params=clean)
    if resp.status_code >= 400:
        raise MCPElasticsearchError(
            f"{tool}: HTTP {resp.status_code} {_truncate(resp.text, 500)}", reason="http",
        )
    return resp.json() if resp.text else {}


async def _post(path: str, body: dict, *, tool: str = "elasticsearch") -> Any:
    cfg = _config()
    async with httpx.AsyncClient(
        headers={**cfg["headers"], "Content-Type": "application/json"}, auth=cfg["auth"],
        timeout=_DEFAULT_TIMEOUT_S, verify=cfg["verify_ssl"],
    ) as http:
        resp = await http.post(f"{cfg['url']}{path}", json=body)
    if resp.status_code >= 400:
        raise MCPElasticsearchError(
            f"{tool}: HTTP {resp.status_code} {_truncate(resp.text, 500)}", reason="http",
        )
    return resp.json() if resp.text else {}


def _index(args: dict) -> str:
    idx = (args.get("index") or "").strip()
    return idx or _config()["default_index"]


# --- handlers --------------------------------------------------------------

async def _h_list_indices(_unused, args: dict) -> Any:
    """`GET /_cat/indices?format=json` -- list indices with health + doc
    count. Optional `index` filters by pattern; system indices (`.`) hidden."""
    pattern = (args.get("index") or "").strip()
    path = f"/_cat/indices/{pattern}" if pattern else "/_cat/indices"
    rows = await _get(
        path, {"format": "json", "h": "index,health,status,docs.count,store.size"},
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
    return {"count": len(out), "indices": out}


async def _h_get_mappings(_unused, args: dict) -> Any:
    """`GET /<index>/_mapping` -- field mappings (schema discovery)."""
    index = _index(args)
    resp = await _get(f"/{index}/_mapping", tool="elasticsearch_get_mappings")
    out: dict[str, dict] = {}
    for idx, body in (resp or {}).items():
        props = ((body or {}).get("mappings") or {}).get("properties") or {}
        out[idx] = {f: (v.get("type") or "object") for f, v in props.items()}
    return {"index": index, "mappings": out}


async def _h_search(_unused, args: dict) -> Any:
    """`POST /<index>/_search` -- read-only Query DSL search. Pass `query`
    (a full DSL object) or `q` (a Lucene query string). Returns up to `size`
    hits (capped)."""
    index = _index(args)
    size = _clamp(args.get("size"))
    body: dict[str, Any] = {"size": size, "track_total_hits": False}
    if isinstance(args.get("query"), dict):
        body["query"] = args["query"]
    elif args.get("q"):
        body["query"] = {"query_string": {"query": str(args["q"]), "default_operator": "AND"}}
    else:
        body["query"] = {"match_all": {}}
    if args.get("sort"):
        body["sort"] = args["sort"]
    resp = await _post(f"/{index}/_search", body, tool="elasticsearch_search")
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
    return {"index": index, "size": size, "count": len(out), "hits": out}


async def _h_esql_query(_unused, args: dict) -> Any:
    """`POST /_query` -- run an ES|QL query (Elasticsearch only; OpenSearch
    does not implement ES|QL). `query` is the ES|QL string."""
    cfg = _config()
    if cfg["backend"] != "elasticsearch":
        raise MCPElasticsearchError(
            "elasticsearch_esql_query: ES|QL is Elasticsearch-only "
            "(backend is 'opensearch'); use elasticsearch_search instead.",
            reason="backend",
        )
    query = (args.get("query") or "").strip()
    if not query:
        raise MCPElasticsearchError("elasticsearch_esql_query: 'query' is required", reason="bad_args")
    resp = await _post("/_query", {"query": query}, tool="elasticsearch_esql_query")
    cols = [c.get("name") for c in (resp.get("columns") or [])]
    rows = (resp.get("values") or [])[:_MAX_LIMIT]
    return {"columns": cols, "row_count": len(rows), "rows": rows}


async def _h_cluster_health(_unused, args: dict) -> Any:
    """`GET /_cluster/health` -- cluster status (green/yellow/red) + shard
    counts."""
    resp = await _get("/_cluster/health", tool="elasticsearch_cluster_health")
    keys = ("cluster_name", "status", "number_of_nodes", "active_shards",
            "relocating_shards", "initializing_shards", "unassigned_shards",
            "active_shards_percent_as_number")
    return {k: resp.get(k) for k in keys if k in (resp or {})}


# --- tool specs ------------------------------------------------------------

_INDEX_PROP = {"index": {"type": "string", "description": "Index or index pattern (default: configured default_index)."}}

ES_TOOLS: list[MCPTool] = [
    MCPTool(
        name="elasticsearch_list_indices",
        description="List Elasticsearch/OpenSearch indices with health, status, and document counts. Optional 'index' filters by pattern. Read-only.",
        input_schema={"type": "object", "properties": _INDEX_PROP},
        handler=_h_list_indices,
    ),
    MCPTool(
        name="elasticsearch_get_mappings",
        description="Get the field mappings (schema) for an index — use to discover which fields exist before searching. Read-only.",
        input_schema={"type": "object", "properties": _INDEX_PROP, "required": []},
        handler=_h_get_mappings,
    ),
    MCPTool(
        name="elasticsearch_search",
        description="Run a read-only search. Pass 'q' (Lucene query string, e.g. 'level:error AND service:payments') OR 'query' (a full Query DSL object). 'index', 'size', 'sort' optional.",
        input_schema={
            "type": "object",
            "properties": {
                **_INDEX_PROP,
                "q": {"type": "string", "description": "Lucene query string."},
                "query": {"type": "object", "description": "Full Elasticsearch Query DSL query object (alternative to 'q')."},
                "size": {"type": "integer", "description": f"Max hits (default {_DEFAULT_LIMIT}, max {_MAX_LIMIT})."},
                "sort": {"description": "Optional sort clause (ES sort syntax)."},
            },
        },
        handler=_h_search,
    ),
    MCPTool(
        name="elasticsearch_esql_query",
        description="Run an ES|QL query (Elasticsearch only; not OpenSearch). 'query' is the ES|QL string, e.g. 'FROM logs-* | WHERE level==\"error\" | STATS count() BY service'. Read-only.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "ES|QL query string."}},
            "required": ["query"],
        },
        handler=_h_esql_query,
    ),
    MCPTool(
        name="elasticsearch_cluster_health",
        description="Cluster health: status (green/yellow/red), node count, and shard counts. Read-only.",
        input_schema={"type": "object", "properties": {}},
        handler=_h_cluster_health,
    ),
]


def get_tool(name: str) -> MCPTool:
    for t in ES_TOOLS:
        if t.name == name:
            return t
    raise KeyError(f"unknown elasticsearch tool: {name}")


# --- fake backend (FR-012; offline tests) ----------------------------------

async def _fake_get(path: str, params: dict | None = None, *, tool: str = "elasticsearch") -> Any:
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


async def _fake_post(path: str, body: dict, *, tool: str = "elasticsearch") -> Any:
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
    responders (no network, no ES creds) and restores them on teardown."""
    import opsrag.mcp.elasticsearch as _mod
    from opsrag.mcp._fake import FakeMCP

    _orig_get, _orig_post, _orig_bound = _mod._get, _mod._post, _mod._BOUND
    _mod._get = _fake_get
    _mod._post = _fake_post
    _mod._BOUND = {
        "url": "http://es.local:9200", "url_env": "ES_URL", "api_key_env": "ES_API_KEY",
        "username_env": "ES_USERNAME", "password_env": "ES_PASSWORD",
        "backend": "elasticsearch", "default_index": "*", "verify_ssl": True,
    }

    def _restore() -> None:
        _mod._get, _mod._post, _mod._BOUND = _orig_get, _orig_post, _orig_bound

    return FakeMCP(tools=list(ES_TOOLS), client=None, teardown=_restore)
