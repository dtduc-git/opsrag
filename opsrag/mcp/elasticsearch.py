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
import random
import re
import time as _time
from typing import Any

import httpx

from opsrag.mcp import kubernetes as _k8s
from opsrag.mcp.gitlab import MCPTool

_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_LIMIT = 20
_MAX_LIMIT = 200
_RESULT_TRUNCATE_CHARS = 4000

# Per-(env, cluster) cached pod pick for the port_forward reach. We resolve
# the ES Service -> a Running+Ready pod on first use and stick to it for a
# short TTL (pods rotate on rollouts/evictions, so we soft-expire even on the
# happy path; a WS/connection error also invalidates + re-resolves).
_pod_cache: dict[tuple[str | None, str | None], dict[str, Any]] = {}
_POD_CACHE_TTL_SECONDS = 300.0


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
    """The env name for a handler call: explicit ``env`` arg, else ``None``
    (-> registry default)."""
    return args.get("env") or None


def _resolve_cluster(args: dict) -> str | None:
    """The Elasticsearch cluster name within the env (multi-cluster envs). The
    ``cluster`` arg selects a named cluster from the env's ``EsClusters`` map;
    ``None`` -> the env's default cluster."""
    return args.get("cluster") or None


def _es_clusters(env: str | None) -> tuple[dict[str, Any], str | None]:
    """Return ``({cluster_name: EsTarget}, default_name)`` for an env.

    Normalizes both config shapes: a single ``EsTarget`` becomes a one-entry map
    named ``"default"``; an ``EsClusters`` map is returned as-is with its
    configured (or first) default. Raises ``MCPElasticsearchError`` when the env
    has no ES configured at all."""
    from opsrag.config import EsClusters
    from opsrag.environments import resolve_environment

    es = resolve_environment(env).elasticsearch
    if es is None:
        name = env or "(default)"
        raise MCPElasticsearchError(
            f"elasticsearch not configured for env {name!r} -- add an "
            f"`elasticsearch:` block under environments.targets.{name}.",
            reason="bad_env",
        )
    if isinstance(es, EsClusters):
        clusters = dict(es.clusters or {})
        default = es.default or (next(iter(clusters), None) if clusters else None)
        return clusters, default
    # single EsTarget -> a one-cluster map
    return {"default": es}, "default"


def _config(env: str | None = None, cluster: str | None = None) -> dict:
    """Resolve the request bundle for an environment's Elasticsearch target.

    Looks up ``resolve_environment(env).elasticsearch`` and selects the named
    ``cluster`` (multi-cluster envs; omitted -> the env's default cluster), then
    turns the chosen ``EsTarget`` into a concrete bundle: ``{reach, env, cluster,
    url, headers, auth, backend, default_index, verify_ssl, fields, target}``.
    Credentials are read from env vars (``api_key_env`` / ``username_env`` +
    ``password_env``) at call time -- never captured at bind.

    Raises ``MCPElasticsearchError`` when the env has no ES target, an unknown
    ``cluster`` is requested, or (for ``reach=direct``) no URL is configured.
    ``EnvironmentResolutionError`` (unknown / empty registry) propagates."""
    clusters, default_name = _es_clusters(env)
    name = cluster or default_name
    if name is None or name not in clusters:
        env_label = env or "(default)"
        raise MCPElasticsearchError(
            f"unknown elasticsearch cluster {cluster!r} for env {env_label!r}. "
            f"Configured: {sorted(clusters)}. Call `elasticsearch_list_clusters` "
            "to see what each holds.",
            reason="bad_cluster",
        )
    target = clusters[name]

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
        "cluster": name,
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


# --- port_forward transport (real pods/portforward WebSocket tunnel) -------
#
# For ``reach == "port_forward"`` we do NOT use the httpx service-proxy path in
# ``_base`` (the API server strips/consumes the ``Authorization`` header on the
# service-proxy, so the ES ApiKey never reaches ES -> 401). Instead we open a
# TCP-level ``pods/portforward`` WebSocket tunnel to a backing ES pod through
# the env's k8s API server and ride the raw HTTP/1.1 request (carrying
# ``Authorization: ApiKey <es-key>``) end-to-end into ES. The cluster bearer
# token authenticates only the WebSocket handshake to the API server.
#
# The wire-protocol helpers below (``_portforward_request`` / ``_build_http_request``
# / ``_parse_http_response`` / ``_dechunk`` + the ``_PF_*`` constants) implement the
# v4.channel.k8s.io SPDY framing, which is subtle -- keep them together and treat
# them as a self-contained, well-tested unit (the only module-specific piece is the
# ``MCPElasticsearchError`` exception class).

# v4.channel.k8s.io port-forward framing constants.
_PF_DATA_CHANNEL = 0   # bytes flowing TO/FROM the forwarded port
_PF_ERROR_CHANNEL = 1  # error stream for the forwarded port
# Each forwarded port creates two channels (data, error). For 1 port we
# expect to see 2 initial "port" frames after handshake -- these announce
# which port each channel maps to and should be consumed silently.
_PF_INITIAL_PORT_FRAMES = 2


async def _portforward_request(
    cfg: dict,
    *,
    namespace: str,
    pod: str,
    port: int,
    raw_request: bytes,
    read_timeout: float = 30.0,
) -> bytes:
    """Open a port-forward WebSocket to ``pod:port`` and exchange a raw
    HTTP/1.1 request/response pair through it. Returns the raw HTTP/1.1
    response bytes (status line + headers + body).

    Protocol notes (v4.channel.k8s.io):
      * Each WS binary frame is `[channel_byte][payload]`. We write
        outgoing bytes on channel 0 (data for port[0]); read incoming
        bytes from the same channel. Channel 1 carries server-side
        errors -- we collect any payload there into the exception text.
      * The server sends two initial frames acknowledging the requested
        port (one per channel). They show up as 2-byte payloads
        containing the port number; we discard them.
      * ES speaks HTTP/1.1 keep-alive by default; we ask for
        ``Connection: close`` so the upstream closes the data channel
        once the response is fully sent. That's our EOF signal.
    """
    import ssl

    import aiohttp

    # API server expects a ws:// or wss:// URL. Build wss:// from the
    # https://<host> we cached.
    base = cfg["host"].rstrip("/")
    if base.startswith("https://"):
        ws_base = "wss://" + base[len("https://"):]
    elif base.startswith("http://"):
        ws_base = "ws://" + base[len("http://"):]
    else:
        ws_base = base
    url = (
        f"{ws_base}/api/v1/namespaces/{namespace}/pods/{pod}"
        f"/portforward?ports={port}"
    )

    headers = {}
    if cfg.get("token"):
        headers["Authorization"] = f"Bearer {cfg['token']}"

    # SSLContext from CA cert path (verify=True). When verify is False
    # (legacy kubeconfig with no CA), disable verification.
    verify = cfg.get("verify")
    ssl_ctx: Any
    if verify is False or verify is None:
        ssl_ctx = False
    elif isinstance(verify, str):
        ssl_ctx = ssl.create_default_context(cafile=verify)
    else:
        ssl_ctx = True

    timeout = aiohttp.ClientTimeout(total=read_timeout + 5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # K8s requires the v4.channel.k8s.io subprotocol on the WS
        # upgrade -- without it the server falls back to v1 framing
        # which is incompatible with what we read/write below.
        async with session.ws_connect(
            url,
            headers=headers,
            protocols=("v4.channel.k8s.io",),
            ssl=ssl_ctx,
            max_msg_size=0,  # 0 = unlimited; ES responses can be large
            heartbeat=None,
        ) as ws:
            # Consume the two initial "port announcement" frames.
            consumed = 0
            err_buf = bytearray()
            while consumed < _PF_INITIAL_PORT_FRAMES:
                msg = await ws.receive(timeout=read_timeout)
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.ERROR,
                ):
                    raise MCPElasticsearchError(
                        f"port-forward WS closed before initial port "
                        f"frames (got {msg.type!r}, data={msg.data!r})",
                        reason="upstream_error",
                    )
                if msg.type != aiohttp.WSMsgType.BINARY:
                    continue
                data = msg.data
                if not data:
                    continue
                ch = data[0]
                if ch == _PF_ERROR_CHANNEL:
                    err_buf.extend(data[1:])
                consumed += 1

            # Send the HTTP/1.1 request as a single binary frame on
            # channel 0. ES accepts the entire request as one TCP write
            # (request line + headers + body); we batch them into one
            # WS frame for the same reason.
            await ws.send_bytes(bytes([_PF_DATA_CHANNEL]) + raw_request)

            # Read response bytes from channel 0 until the server
            # closes the data side (Connection: close).
            response_buf = bytearray()
            while True:
                try:
                    msg = await ws.receive(timeout=read_timeout)
                except TimeoutError as exc:  # asyncio.TimeoutError is this alias on py311+
                    raise MCPElasticsearchError(
                        f"port-forward read timeout after "
                        f"{read_timeout}s; partial response "
                        f"{len(response_buf)} bytes",
                        reason="upstream_error",
                    ) from exc
                if msg.type == aiohttp.WSMsgType.BINARY:
                    if not msg.data:
                        continue
                    ch = msg.data[0]
                    payload = msg.data[1:]
                    if ch == _PF_DATA_CHANNEL:
                        response_buf.extend(payload)
                    elif ch == _PF_ERROR_CHANNEL:
                        err_buf.extend(payload)
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise MCPElasticsearchError(
                        f"port-forward WS error: {ws.exception()!r}",
                        reason="upstream_error",
                    )

            if err_buf and not response_buf:
                # Server sent an error but no data -- surface the error
                # rather than returning an empty (and thus unparseable)
                # response. Common cause: pod refused the connection.
                raise MCPElasticsearchError(
                    f"port-forward error stream: {bytes(err_buf).decode('utf-8', 'replace')[:300]}",
                    reason="upstream_error",
                )
            return bytes(response_buf)


def _build_http_request(
    *,
    method: str,
    path: str,
    host: str,
    headers: dict[str, str],
    body: bytes,
) -> bytes:
    """Hand-roll an HTTP/1.1 request for the port-forward tunnel.

    ES accepts JSON bodies on GET (the official client does this too),
    so we ALWAYS pass the body slot through -- if `body` is empty we
    just don't emit `Content-Length`. ``Connection: close`` ensures the
    server hangs up after the response so we can detect EOF on the
    tunnel without needing chunked-decoding logic.
    """
    lines = [f"{method} {path} HTTP/1.1", f"Host: {host}"]
    base_headers = {
        "Connection": "close",
        "Accept": "application/json",
    }
    # Caller-provided headers override defaults.
    for k, v in headers.items():
        if v is None:
            continue
        base_headers[k] = v
    if body:
        base_headers.setdefault("Content-Length", str(len(body)))
    for k, v in base_headers.items():
        lines.append(f"{k}: {v}")
    head = ("\r\n".join(lines) + "\r\n\r\n").encode("ascii")
    return head + body


def _parse_http_response(raw: bytes) -> tuple[int, dict[str, str], bytes]:
    """Parse a raw HTTP/1.1 response: (status, headers, body).

    Handles ``Transfer-Encoding: chunked`` (defensive -- most ES versions
    use Content-Length, but a proxy in front of ES could re-chunk). We
    do NOT handle gzip/deflate -- we set ``Accept-Encoding: identity``
    implicitly by NOT advertising compression, so the server won't
    compress.
    """
    if not raw:
        raise MCPElasticsearchError(
            "empty HTTP response from port-forward tunnel "
            "(pod closed the data channel without writing anything)",
            reason="upstream_error",
        )
    sep = raw.find(b"\r\n\r\n")
    if sep < 0:
        raise MCPElasticsearchError(
            f"malformed HTTP response (no header/body separator); "
            f"first 200 bytes={raw[:200]!r}",
            reason="upstream_error",
        )
    head = raw[:sep].decode("iso-8859-1", "replace")
    body = raw[sep + 4:]
    lines = head.split("\r\n")
    if not lines:
        raise MCPElasticsearchError(
            "malformed HTTP response (empty head)", reason="upstream_error",
        )
    status_line = lines[0]
    parts = status_line.split(" ", 2)
    if len(parts) < 2 or not parts[0].startswith("HTTP/1."):
        raise MCPElasticsearchError(
            f"malformed HTTP status line: {status_line!r}",
            reason="upstream_error",
        )
    try:
        status = int(parts[1])
    except ValueError as exc:
        raise MCPElasticsearchError(
            f"non-integer HTTP status: {parts[1]!r}",
            reason="upstream_error",
        ) from exc
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        headers[k.strip().lower()] = v.strip()

    # Handle chunked transfer encoding if present.
    if headers.get("transfer-encoding", "").lower() == "chunked":
        body = _dechunk(body)
    return status, headers, body


def _dechunk(buf: bytes) -> bytes:
    """Decode HTTP/1.1 chunked transfer-encoding into a flat body."""
    out = bytearray()
    i = 0
    n = len(buf)
    while i < n:
        eol = buf.find(b"\r\n", i)
        if eol < 0:
            break
        size_str = buf[i:eol].split(b";", 1)[0].strip()
        try:
            size = int(size_str, 16)
        except ValueError:
            break
        i = eol + 2
        if size == 0:
            break
        out.extend(buf[i : i + size])
        i += size + 2  # skip chunk + trailing \r\n
    return bytes(out)


def _pf_pod_selector(target: Any) -> str:
    """Label selector used to pick a backing pod for the ES Service.

    Prefers an explicit ``EsTarget.pod_label_selector``; otherwise derives the
    ECK convention from the service name by stripping the ``-es-http`` suffix
    (ECK names the HTTP Service ``<cr>-es-http`` and labels its pods with
    ``elasticsearch.k8s.elastic.co/cluster-name=<cr>``), e.g. service
    ``eck-infra-logs-es-http`` -> ``...cluster-name=eck-infra-logs``."""
    selector = (getattr(target, "pod_label_selector", None) or "").strip()
    if selector:
        return selector
    service = getattr(target, "service", None) or ""
    cr = service[: -len("-es-http")] if service.endswith("-es-http") else service
    return f"elasticsearch.k8s.elastic.co/cluster-name={cr}"


async def _resolve_es_pod(
    cfg: dict, access: dict, *, force_refresh: bool = False,
) -> str:
    """Pick a Running+Ready ES pod for ``cfg``'s target Service.

    Lists pods in ``target.namespace`` matching ``_pf_pod_selector(target)``
    via a plain httpx GET to the env's k8s API server (Bearer token + CA from
    ``cluster_api_access``), filters to phase=Running with a True Ready
    condition, random-picks one (read-only traffic; ES fans the search out
    across shards internally anyway) and caches it per ``(env, cluster)`` for
    ``_POD_CACHE_TTL_SECONDS``. ``force_refresh`` skips the cache (used after a
    tunnel connection error -- the pod likely rotated)."""
    target = cfg["target"]
    namespace = getattr(target, "namespace", None)
    selector = _pf_pod_selector(target)
    cache_key = (cfg["env"], cfg["cluster"])
    now = _time.monotonic()
    cached = _pod_cache.get(cache_key)
    if cached and not force_refresh and (now - cached["picked_at"]) < _POD_CACHE_TTL_SECONDS:
        return cached["pod_name"]

    host = access["host"].rstrip("/")
    headers = {"Accept": "application/json"}
    if access.get("token"):
        headers["Authorization"] = f"Bearer {access['token']}"
    async with httpx.AsyncClient(
        timeout=_DEFAULT_TIMEOUT_S, verify=access.get("verify", False),
    ) as http:
        resp = await http.get(
            f"{host}/api/v1/namespaces/{namespace}/pods",
            params={"labelSelector": selector}, headers=headers,
        )
    if resp.status_code >= 400:
        raise MCPElasticsearchError(
            f"elasticsearch mcp: listing pods for selector {selector!r} in "
            f"namespace {namespace!r} failed: HTTP {resp.status_code} "
            f"{_truncate(resp.text, 300)}",
            reason="http",
        )
    items = (resp.json() or {}).get("items") or []
    ready_pods: list[str] = []
    for p in items:
        status = p.get("status") or {}
        if status.get("phase") != "Running":
            continue
        conds = status.get("conditions") or []
        if not any(
            c.get("type") == "Ready" and c.get("status") == "True" for c in conds
        ):
            continue
        name = (p.get("metadata") or {}).get("name")
        if name:
            ready_pods.append(name)
    if not ready_pods:
        raise MCPElasticsearchError(
            f"elasticsearch mcp: no Running+Ready pods matched selector "
            f"{selector!r} in namespace {namespace!r} for env {cfg['env']!r}. "
            f"Either the ECK pods are down, the selector is wrong (override via "
            f"the target's `pod_label_selector`), or the GSA lacks `pods` list "
            f"permission in that namespace.",
            reason="no_pod",
        )
    picked = random.choice(ready_pods)
    _pod_cache[cache_key] = {"pod_name": picked, "picked_at": now}
    return picked


async def _pf_request(
    cfg: dict, method: str, path: str, params: dict | None, body: dict | None,
    *, tool: str = "elasticsearch", _retried: bool = False,
) -> Any:
    """Issue an ES request through a pods/portforward WebSocket tunnel and
    return the parsed JSON (same shape ``_get``/``_post`` return from httpx).

    The ES ``Authorization: ApiKey`` header (from ``cfg['headers']``, built by
    ``_config``) rides INSIDE the tunnel straight to ES; the cluster bearer
    token from ``cluster_api_access`` authenticates only the WebSocket handshake
    to the API server (set in ``_portforward_request``). Mirrors the httpx path:
    a 401 refreshes the API-server token + re-resolves the pod and retries once,
    and non-2xx raises ``MCPElasticsearchError(reason="http")``."""
    import json as _json

    target = cfg["target"]
    service = getattr(target, "service", None)
    namespace = getattr(target, "namespace", None)
    port = getattr(target, "port", 9200) or 9200
    if not (service and namespace):
        raise MCPElasticsearchError(
            "elasticsearch mcp: reach=port_forward requires `service` and "
            "`namespace` on the environment's elasticsearch target.",
            reason="bad_env",
        )

    access = await _k8s.cluster_api_access(cfg["env"])
    pod = await _resolve_es_pod(cfg, access)

    # Compose the request-target (path + query string). ES accepts a JSON body
    # on GET, so both verbs go through the same builder.
    req_path = path if path.startswith("/") else "/" + path
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    if clean:
        from urllib.parse import urlencode
        req_path = f"{req_path}?{urlencode(clean)}"

    # ES ApiKey (or basic) Authorization rides end-to-end through the tunnel;
    # `_config` already put it on cfg["headers"].
    req_headers: dict[str, str] = {}
    es_auth = cfg["headers"].get("Authorization")
    if es_auth:
        req_headers["Authorization"] = es_auth
    raw_body = b""
    if body is not None and body != {}:
        raw_body = _json.dumps(body, separators=(",", ":")).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    raw_request = _build_http_request(
        method=method,
        path=req_path,
        # Host header is informational to ES (it only cares about the ApiKey +
        # path); use the Service DNS name for traceability in ES access logs.
        host=f"{service}.{namespace}.svc:{port}",
        headers=req_headers,
        body=raw_body,
    )

    try:
        raw_response = await _portforward_request(
            access, namespace=namespace, pod=pod, port=int(port),
            raw_request=raw_request,
        )
    except MCPElasticsearchError:
        raise
    except Exception as exc:
        # A 401 here means the API server rejected the WS upgrade -- i.e. the
        # ADC/API-server token expired (the inner 401 retry below never sees
        # this because the handshake fails before any data flows). Refresh the
        # token + re-resolve the pod and retry once; otherwise treat it as a
        # connection-level failure (pod rotated / network blip) and bubble up.
        import aiohttp as _aiohttp
        _pod_cache.pop((cfg["env"], cfg["cluster"]), None)
        is_k8s_401 = (
            isinstance(exc, _aiohttp.WSServerHandshakeError)
            and getattr(exc, "status", None) == 401
        )
        if is_k8s_401 and not _retried:
            await _refresh_cluster_access(cfg)
            return await _pf_request(cfg, method, path, params, body, tool=tool, _retried=True)
        raise MCPElasticsearchError(
            f"{tool}: port-forward tunnel failed: {exc!r}", reason="http",
        ) from exc

    status, _headers, body_bytes = _parse_http_response(raw_response)
    if status == 401 and not _retried:
        await _refresh_cluster_access(cfg)
        _pod_cache.pop((cfg["env"], cfg["cluster"]), None)
        return await _pf_request(cfg, method, path, params, body, tool=tool, _retried=True)
    if status >= 400:
        snippet = body_bytes[:500].decode("utf-8", "replace")
        raise MCPElasticsearchError(
            f"{tool}: HTTP {status} {_truncate(snippet, 500)}", reason="http",
        )
    if not body_bytes:
        return {}
    try:
        return _json.loads(body_bytes.decode("utf-8"))
    except Exception as exc:
        snippet = body_bytes[:300].decode("utf-8", "replace")
        raise MCPElasticsearchError(
            f"{tool}: failed to parse ES response: {exc}; body={_truncate(snippet, 300)}",
            reason="http",
        ) from exc


async def _get(
    cfg: dict, path: str, params: dict | None = None, *,
    tool: str = "elasticsearch", _retried: bool = False,
) -> Any:
    if cfg["reach"] == "port_forward":
        return await _pf_request(cfg, "GET", path, params, None, tool=tool, _retried=_retried)
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
    if cfg["reach"] == "port_forward":
        return await _pf_request(cfg, "POST", path, None, body, tool=tool, _retried=_retried)
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
    cfg = _config(_resolve_env(args), _resolve_cluster(args))
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
    cfg = _config(_resolve_env(args), _resolve_cluster(args))
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
    cfg = _config(_resolve_env(args), _resolve_cluster(args))
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
    cfg = _config(_resolve_env(args), _resolve_cluster(args))
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
    cfg = _config(_resolve_env(args), _resolve_cluster(args))
    resp = await _get(cfg, "/_cluster/health", tool="elasticsearch_cluster_health")
    keys = ("cluster_name", "status", "number_of_nodes", "active_shards",
            "relocating_shards", "initializing_shards", "unassigned_shards",
            "active_shards_percent_as_number")
    out = {k: resp.get(k) for k in keys if k in (resp or {})}
    out["env"] = cfg["env"]
    out["cluster"] = cfg["cluster"]
    return out


async def _h_list_clusters(_unused, args: dict) -> Any:
    """List the ES clusters configured for an env + each cluster's `usage_note`
    (what it holds), so the agent can pick the right one. No network call --
    reads the environments registry."""
    env = _resolve_env(args)
    clusters, default_name = _es_clusters(env)
    out = [
        {
            "cluster": cname,
            "default": cname == default_name,
            "backend": getattr(t, "backend", "elasticsearch"),
            "reach": getattr(t, "reach", "direct"),
            "index_pattern": getattr(t, "index_pattern", "*"),
            "usage_note": getattr(t, "usage_note", None),
        }
        for cname, t in sorted(clusters.items())
    ]
    return {"env": env, "default": default_name, "count": len(out), "clusters": out}


# --- tool specs ------------------------------------------------------------

_INDEX_PROP = {"index": {"type": "string", "description": "Index or index pattern (default: configured index_pattern)."}}
# Optional env + cluster selectors on the query tools. Both omitted -> the
# registry default env and that env's default ES cluster, so single-env /
# single-cluster callers are unaffected.
_ENV_PROP = {
    "env": {"type": "string", "description": "Target environment name from the environments: registry (default: the configured default env)."},
    "cluster": {"type": "string", "description": "Elasticsearch cluster name within the env (multi-cluster envs; see elasticsearch_list_clusters). Omitted -> the env's default cluster."},
}
_ENV_ONLY = {"env": _ENV_PROP["env"]}

ES_TOOLS: list[MCPTool] = [
    MCPTool(
        name="elasticsearch_list_clusters",
        description="List the Elasticsearch clusters available for an environment, each with a usage note describing what it holds (e.g. container logs vs integration logs vs product data). Call this FIRST when a query could target more than one cluster, or when a search finds nothing. Optional 'env'. Read-only.",
        input_schema={"type": "object", "properties": {**_ENV_ONLY}},
        handler=_h_list_clusters,
    ),
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
