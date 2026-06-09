"""Kubernetes MCP-style tools for OpsRAG (Sub-sprint 4).

Read-only async tools over the Kubernetes API. Multi-environment: each
tool accepts an `env` arg (back-compat alias `cluster`) naming an entry
in the unified `environments:` registry. The set of environments and the
default come from that registry (`opsrag.environments`): the resolver
returns an `EnvironmentTarget` whose `.kubernetes` (a `K8sTarget`) says
how to reach that env's API server -- `mode=gke` (Workload Identity +
GCP Container API) or `mode=kubeconfig` (a KUBECONFIG context / current
context / in-cluster ServiceAccount). When no environment is configured
the caller must pass `env`/`cluster` explicitly, else the tool returns a
clear "configure the environments: block" error. `OPSRAG_K8S_DEFAULT_CLUSTER`
still overrides the default env name for back-compat.

This module also exposes a SHARED helper -- `cluster_api_access(env) ->
{host, token, verify}` -- that the prometheus + elasticsearch MCPs reuse
to reach an env's API server (service-proxy / port-forward) without
depending on kubernetes internals.

## Read-only enforcement

Every tool calls a `*read*` / `list*` / `get*` API method. No
`create_*` / `delete_*` / `patch_*` / `exec_*` anywhere. The user
running OpsRAG is expected to have `view` / `cluster-reader` RBAC at
most. Production deployment uses ADC + the GCP Container API (Workload
Identity in-cluster, gcloud ADC locally) -- no baked-in cluster list.

## Tool list (9 read-only)

| Tool                    | API                                  |
|-------------------------|--------------------------------------|
| `k8s_get_pod`           | CoreV1.read_namespaced_pod           |
| `k8s_list_pods`         | CoreV1.list_namespaced_pod           |
| `k8s_get_pod_logs`      | CoreV1.read_namespaced_pod_log       |
| `k8s_list_events`       | CoreV1.list_namespaced_event         |
| `k8s_get_deployment`    | AppsV1.read_namespaced_deployment    |
| `k8s_list_deployments`  | AppsV1.list_namespaced_deployment    |
| `k8s_get_service`       | CoreV1.read_namespaced_service       |
| `k8s_get_role_bindings` | RbacV1.list_namespaced_role_binding  |
| `k8s_top_pod`           | metrics.k8s.io PodMetrics (raw API)  |
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

# kubernetes_asyncio import is deferred to first call so the module
# can be imported even before the package is installed (e.g. during
# the brief window after pyproject.toml is updated but before the
# image is rebuilt). At runtime, _ensure_imports raises a clear error
# if the package is still missing.
k8s_client: Any = None
k8s_config: Any = None

from opsrag.mcp.gitlab import MCPTool


def _ensure_imports() -> None:
    """Lazy-load kubernetes_asyncio. Raises a clear ImportError with
    install hint if the package isn't available yet."""
    global k8s_client, k8s_config
    if k8s_client is not None:
        return
    try:
        from kubernetes_asyncio import client as _client
        from kubernetes_asyncio import config as _config
        k8s_client = _client
        k8s_config = _config
    except ImportError as exc:
        raise ImportError(
            "kubernetes_asyncio package not installed. Run "
            "`pip install kubernetes_asyncio>=32.0` inside the container "
            "or rebuild the image after the pyproject.toml change."
        ) from exc

_log = logging.getLogger("opsrag.mcp.kubernetes")

DEFAULT_CLUSTER_ENV = "OPSRAG_K8S_DEFAULT_CLUSTER"
_DEFAULT_TAIL_LINES = 200
_LOG_TRUNCATE_CHARS = 32000  # generator gets the trace tail; bound it


def _known_clusters() -> list[str]:
    """Environment names exposed by the unified `environments:` registry.
    Empty list when nothing is configured -- callers must then supply an
    explicit `env` (or back-compat `cluster`) argument."""
    from opsrag.environments import available_environments
    return available_environments()


def _default_cluster() -> str:
    """Resolve the default environment name. Precedence:
    OPSRAG_K8S_DEFAULT_CLUSTER env var, then the registry's default
    environment. Returns "" when nothing is configured; handlers turn an
    empty value into a clear "env required" error rather than falling back
    to a baked-in org default."""
    env = os.environ.get(DEFAULT_CLUSTER_ENV)
    if env:
        return env
    from opsrag.environments import default_environment
    return default_environment() or ""


def _resolve_cluster(args: dict) -> str:
    """Pick the environment for a handler call: explicit `env` arg wins,
    then the back-compat `cluster` alias, then the configured default.
    Raises a clear error when none is available so behavior degrades
    gracefully on an empty registry.

    The returned string is the ENV NAME -- it keys the per-env API-client
    cache and the resolver (`resolve_environment(env)`); handlers echo it
    back as ``result["cluster"]`` for caller continuity.

    Precedence: explicit `env` arg > back-compat `cluster` alias >
    `_default_cluster()` (OPSRAG_K8S_DEFAULT_CLUSTER env, then the
    registry's default environment)."""
    cluster = args.get("env") or args.get("cluster") or _default_cluster()
    if not cluster:
        raise RuntimeError(
            "k8s mcp: no environment specified and none configured. Pass an "
            "`env` argument (back-compat alias `cluster`), or define the "
            "`environments:` block in config (legacy k8s.clusters / "
            "OPSRAG_K8S_DEFAULT_CLUSTER env also accepted)."
        )
    return cluster


# Token / credential redaction patterns. Pod logs and event messages
# are well-known leaks; redact before handing to the LLM.
_REDACT_PATTERNS = [
    (re.compile(r"\bxoxb-[A-Za-z0-9-]{20,}"), "[REDACTED:slack_bot_token]"),
    (re.compile(r"\bxoxp-[A-Za-z0-9-]{20,}"), "[REDACTED:slack_user_token]"),
    (re.compile(r"\bghp_[A-Za-z0-9]{30,}"), "[REDACTED:github_token]"),
    (re.compile(r"\bglpat-[A-Za-z0-9_]{20,}"), "[REDACTED:gitlab_token]"),
    (re.compile(r"\bAKIA[A-Z0-9]{16}\b"), "[REDACTED:aws_access_key]"),
    (re.compile(r"\bAIza[A-Za-z0-9_-]{35}"), "[REDACTED:google_api_key]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]+"), "[REDACTED:jwt]"),
    (re.compile(r"\brootly_[A-Za-z0-9_]{30,}"), "[REDACTED:rootly_token]"),
]


def _redact(text: str) -> str:
    for pat, repl in _REDACT_PATTERNS:
        text = pat.sub(repl, text)
    return text


@dataclass
class _Connection:
    """Per-cluster API client bundle. We construct these on demand and
    cache by context name to avoid reloading kubeconfig on every call."""
    api_client: Any
    core: Any
    apps: Any
    batch: Any  # BatchV1Api for Jobs / CronJobs
    rbac: Any
    custom: Any  # CustomObjectsApi for metrics.k8s.io


_clients: dict[str, _Connection] = {}
# Per-env API-access bundle cache: { env: {host, token, verify} }. Built by
# `cluster_api_access(env)` and reused by prometheus/elasticsearch. Cleared
# on 401 (refresh_adc_token) or via `invalidate_env_api_cache`.
_api_access: dict[str, dict] = {}
# Cluster coordinates registry: { name: {project, location, name} }.
# Populated from OPSRAG_K8S_CLUSTERS env (JSON) or config.yaml's
# `k8s.clusters` block. If empty, falls back to legacy kubeconfig mode.
_cluster_coords: dict[str, dict] = {}
_cluster_coords_loaded = False
# Temp dir for per-cluster CA cert files. The K8s client requires a
# filesystem path for ssl_ca_cert, not raw bytes. Created lazily.
_ca_cert_dir: str | None = None
# Cached ADC creds object -- refreshable. Same creds used across clusters.
_adc_creds: Any = None
# Whether we've successfully resolved a default (non-GKE) connection mode
# (in-cluster ServiceAccount or a standard kubeconfig). Used only when no
# GKE clusters are registered.
_kubeconfig_loaded = False
# True when running inside a pod and using the mounted ServiceAccount
# (KUBERNETES_SERVICE_HOST present). Selected in `_ensure_kubeconfig`.
_incluster_mode = False


def _load_cluster_coords_from_env() -> dict[str, dict]:
    """Parse OPSRAG_K8S_CLUSTERS (JSON) if set. Shape:

        {"<context-name>": {"project": "...", "location": "...",
                            "name": "..."}, ...}

    Returns empty dict if unset/invalid. Falls back to kubeconfig path.
    """
    raw = os.environ.get("OPSRAG_K8S_CLUSTERS") or ""
    if not raw.strip():
        return {}
    try:
        import json
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {}
        return parsed
    except Exception as exc:
        _log.warning("OPSRAG_K8S_CLUSTERS env: invalid JSON (%s)", exc)
        return {}


def register_clusters(clusters: dict[str, dict]) -> None:
    """BACK-COMPAT SHIM. Historically this registered the cluster ->
    (project, location, name) GKE-coords mapping consumed directly by the
    K8s MCP. The MCP is now ENV-DRIVEN through the unified `environments:`
    registry (`opsrag.environments`): `register_clusters` no longer feeds
    the resolution path -- `bind_environments(cfg)` does (it synthesizes a
    GKE env per `k8s.clusters` entry when no explicit `environments:` block
    is set). We keep populating the legacy `_cluster_coords` dict so any
    old caller / test importing this symbol still works; it is otherwise
    unused by the env-driven `cluster_api_access` / `_get_connection`.
    """
    global _cluster_coords, _cluster_coords_loaded
    _cluster_coords = {k: dict(v) for k, v in (clusters or {}).items()}
    _cluster_coords_loaded = True
    _log.info(
        "k8s mcp: register_clusters() back-compat shim recorded %d cluster(s) "
        "%s (env resolution now flows through the environments: registry)",
        len(_cluster_coords), sorted(_cluster_coords),
    )


def invalidate_env_api_cache(env: str | None = None) -> None:
    """Drop the cached API-access bundle(s) so the next call rebuilds with
    a fresh endpoint/CA/token. `env=None` clears all. Used by the 401
    auto-refresh path (prometheus/es reuse `cluster_api_access`)."""
    if env is None:
        _api_access.clear()
        _clients.clear()
        return
    _api_access.pop(env, None)
    _clients.pop(env, None)


async def refresh_adc_token() -> None:
    """Force a re-mint of the ADC token. Called on 401 from any env's API.
    Clears the per-env API-access + client caches so the next call rebuilds
    with a fresh token. Env-agnostic: works whether the bound envs are GKE
    (re-mint ADC) or kubeconfig (re-derive token from the context)."""
    global _adc_creds
    _adc_creds = None  # forces re-mint on next _get_adc_token()
    _api_access.clear()
    _clients.clear()
    _log.info("k8s mcp: ADC creds reset + api-access cache cleared")


async def _get_adc_token() -> str:
    """Mint (or reuse) an ADC OAuth token. Works identically in pod
    (Workload Identity) and local dev (gcloud ADC)."""
    global _adc_creds
    from google.auth import default as gauth_default
    from google.auth.transport.requests import Request as GAuthRequest
    if _adc_creds is None or not _adc_creds.valid:
        if _adc_creds is None:
            _adc_creds, _ = gauth_default(scopes=[
                "https://www.googleapis.com/auth/cloud-platform",
            ])
        # google-auth's refresh is sync; run in a thread.
        await asyncio.to_thread(_adc_creds.refresh, GAuthRequest())
    return _adc_creds.token


async def _gke_get_cluster(project: str, location: str, name: str) -> dict:
    """Call the GCP Container API to fetch a cluster's endpoint + CA
    cert. Returns {endpoint, ca_cert_b64}. Uses ADC for auth -- the
    caller must have roles/container.viewer (or stronger) on the
    target project."""
    import httpx
    token = await _get_adc_token()
    url = (
        f"https://container.googleapis.com/v1/projects/{project}"
        f"/locations/{location}/clusters/{name}"
    )
    async with httpx.AsyncClient(timeout=15.0) as http:
        resp = await http.get(url, headers={"Authorization": f"Bearer {token}"})
        if resp.status_code != 200:
            raise RuntimeError(
                f"GKE Container API GET {url} returned {resp.status_code}: "
                f"{resp.text[:300]}"
            )
        data = resp.json()
    return {
        "endpoint": data["endpoint"],
        "ca_cert_b64": data["masterAuth"]["clusterCaCertificate"],
    }


async def _ensure_kubeconfig() -> None:
    """Either resolve cluster coords (GKE-API mode) or load a kubeconfig
    file (legacy local-dev mode). GKE-API mode is preferred and
    auto-selected when `_cluster_coords` is non-empty (populated via
    `register_clusters()` from config OR via OPSRAG_K8S_CLUSTERS env).
    """
    global _kubeconfig_loaded, _cluster_coords, _cluster_coords_loaded
    _ensure_imports()

    # Lazy load from env if config-driven registration hasn't run yet.
    if not _cluster_coords_loaded:
        _cluster_coords = _load_cluster_coords_from_env()
        _cluster_coords_loaded = True
        if _cluster_coords:
            _log.info(
                "k8s mcp: cluster coords loaded from env: %s",
                sorted(_cluster_coords),
            )

    # GKE-API mode: clusters explicitly registered -> use ADC + Container API.
    # Nothing to pre-load; `_get_connection` builds the client on demand.
    if _cluster_coords:
        return

    # Default (vendor-neutral) mode: no GKE clusters registered. Use the
    # in-cluster ServiceAccount when running in a pod, otherwise a standard
    # kubeconfig -- honouring whatever auth the kubeconfig declares (client
    # certs, tokens, or exec plugins like `aws eks get-token` /
    # `gke-gcloud-auth-plugin`). No cloud-specific token minting. GKE
    # Workload-Identity users configure `k8s.clusters` instead (handled above).
    global _incluster_mode
    if _kubeconfig_loaded:
        return
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        k8s_config.load_incluster_config()
        _incluster_mode = True
        _log.info("k8s mcp: using in-cluster service-account config")
        _kubeconfig_loaded = True
        return
    src_path = os.environ.get("KUBECONFIG") or os.path.expanduser("~/.kube/config")
    if not os.path.exists(src_path):
        raise RuntimeError(
            f"k8s mcp: not running in-cluster and no kubeconfig at {src_path}. "
            f"Set KUBECONFIG, mount ~/.kube/config, or run inside a cluster. "
            f"(For GKE Workload Identity instead, configure k8s.clusters.)"
        )
    _log.info("k8s mcp: using kubeconfig at %s", src_path)
    _kubeconfig_loaded = True


async def _materialize_kubeconfig_with_adc_token(src_path: str) -> None:
    """Read the source kubeconfig, mint an OAuth token via google-auth
    ADC, and write a patched kubeconfig to /tmp/opsrag-kubeconfig where
    every user with `gcp` auth-provider or `exec` is replaced by a
    static `token:` entry. Then point KUBECONFIG at the patched file."""
    import yaml
    from google.auth import default as gauth_default
    from google.auth.transport.requests import Request as GAuthRequest

    # Mint an access token (~1h lifetime).
    creds, _ = gauth_default(
        scopes=[
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/userinfo.email",
        ],
    )
    creds.refresh(GAuthRequest())
    token = creds.token

    with open(src_path) as f:
        kc = yaml.safe_load(f)

    # Replace gke-gcloud-auth-plugin / gcp auth-provider entries with token.
    patched = 0
    for user_entry in kc.get("users", []) or []:
        u = user_entry.get("user", {})
        if "exec" in u or "auth-provider" in u:
            u.pop("exec", None)
            u.pop("auth-provider", None)
            u["token"] = token
            patched += 1
    _log.info("k8s mcp: patched %d kubeconfig user entries with ADC token", patched)

    dst_path = "/tmp/opsrag-kubeconfig"
    with open(dst_path, "w") as f:
        yaml.safe_dump(kc, f)
    os.environ["KUBECONFIG"] = dst_path
    await k8s_config.load_kube_config(config_file=dst_path)


# NOTE: K8s MCP handlers don't auto-retry on 401 yet -- if you see
# Unauthorized errors >1h after container start, restart for a fresh
# ADC token. The Prometheus MCP DOES auto-refresh on 401 (handlers go
# through a single `_proxy_get` chokepoint, easy to wrap). K8s
# handlers each call a different API method; wrapping every one is a
# follow-up TODO.


def _materialize_ca_cert(key: str, ca_cert_b64: str) -> str:
    """Write a base64 CA cert to /tmp/opsrag-ca-<key>.crt and return the
    path. The K8s client + httpx both want ssl_ca_cert as a filesystem
    PATH, not bytes; we cache one file per key (env/cluster) for the
    process lifetime. Shared by the K8s API client builder and
    `cluster_api_access` (prometheus/es reuse the exact same CA file)."""
    global _ca_cert_dir
    import base64
    import tempfile
    if _ca_cert_dir is None:
        _ca_cert_dir = tempfile.mkdtemp(prefix="opsrag-ca-")
    ca_path = os.path.join(_ca_cert_dir, f"{key}.crt")
    if not os.path.exists(ca_path):
        with open(ca_path, "wb") as f:
            f.write(base64.b64decode(ca_cert_b64))
    return ca_path


async def cluster_api_access(env: str | None) -> dict:
    """SHARED helper -- return ``{host, token, verify}`` for reaching an
    environment's Kubernetes API server. Prometheus (k8s_proxy reach) and
    Elasticsearch (port_forward / proxy reach) call this so they depend on
    this foundation, not on K8s internals.

    - ``host``   : ``https://<endpoint>`` (no trailing slash).
    - ``token``  : bearer token (may be "" for cert-only kubeconfigs).
    - ``verify`` : a CA-file PATH (str) for httpx ``verify=``, or False to
      disable verification (cert-less kubeconfig host).

    Resolves ``resolve_environment(env).kubernetes`` (a ``K8sTarget``) and
    branches on ``mode``:
      - ``gke``        -> ADC + GCP Container API endpoint/CA + ADC token
                          (same as the old prometheus ``_get_proxy_config``).
      - ``kubeconfig`` -> ``new_client_from_config(context=target.context)``
                          (None -> current-context / in-cluster), then read
                          host/token/ssl_ca_cert off the Configuration.

    Cached by ``env``; the cache is invalidated on 401 via
    ``refresh_adc_token`` / ``invalidate_env_api_cache``. Raises
    ``EnvironmentResolutionError`` for an unknown/empty env and a clear
    ``RuntimeError`` when the resolved env has no kubernetes target."""
    from opsrag.environments import resolve_environment

    env = _resolve_cluster({"env": env}) if env is None else env
    if env in _api_access:
        return _api_access[env]

    _ensure_imports()
    target = resolve_environment(env).kubernetes
    if target is None:
        raise RuntimeError(
            f"k8s mcp: environment {env!r} has no `kubernetes` target "
            f"configured -- cannot reach its API server."
        )

    if target.mode == "gke":
        if not (target.project and target.location and target.name):
            raise RuntimeError(
                f"k8s mcp: env {env!r} kubernetes mode=gke requires "
                f"project/location/name."
            )
        info = await _gke_get_cluster(
            project=target.project, location=target.location, name=target.name,
        )
        ca_path = _materialize_ca_cert(env, info["ca_cert_b64"])
        token = await _get_adc_token()
        out = {
            "host": f"https://{info['endpoint']}".rstrip("/"),
            "token": token,
            "verify": ca_path,
        }
    else:
        # kubeconfig mode: build a client for the target context (None ->
        # current-context / in-cluster), then extract host/token/CA off the
        # parsed Configuration. Mirrors the old prometheus kubeconfig branch.
        api_client = await k8s_config.new_client_from_config(
            config_file=(os.environ.get("KUBECONFIG") or None),
            context=(target.context or None),
        )
        cfg = api_client.configuration
        token = ""
        api_key = cfg.api_key or {}
        auth_val = api_key.get("authorization") or api_key.get("BearerToken") or ""
        if auth_val.startswith("Bearer "):
            token = auth_val[len("Bearer "):]
        elif auth_val:
            token = auth_val
        out = {
            "host": (cfg.host or "").rstrip("/"),
            "token": token,
            "verify": cfg.ssl_ca_cert if cfg.ssl_ca_cert else False,
        }
        await api_client.close()

    _api_access[env] = out
    return out


async def _build_api_client_from_cluster_info(
    cluster: str, cluster_info: dict,
) -> Any:
    """Build a kubernetes_asyncio ApiClient from a GKE-API-fetched
    {endpoint, ca_cert_b64} pair + a freshly-minted ADC token.

    The bearer token is set on the Configuration directly; the CA cert is
    materialized to a per-env file (see `_materialize_ca_cert`).
    """
    ca_path = _materialize_ca_cert(cluster, cluster_info["ca_cert_b64"])
    token = await _get_adc_token()
    configuration = k8s_client.Configuration()
    configuration.host = f"https://{cluster_info['endpoint']}"
    configuration.ssl_ca_cert = ca_path
    # K8s OpenAPI security scheme is named "BearerToken" -- the client
    # looks up the value via that key (NOT the literal header name
    # "authorization"). The full header value goes in api_key with the
    # "Bearer " prefix included; api_key_prefix is only used if we want
    # the client to prepend it for us. Either form works; using the
    # combined value matches how kubernetes_asyncio's load_kube_config
    # populates Configuration after parsing a kubeconfig user.token.
    configuration.api_key = {"BearerToken": f"Bearer {token}"}
    return k8s_client.ApiClient(configuration=configuration)


async def _get_connection(env: str) -> _Connection:
    """Build (or reuse) the kubernetes_asyncio client bundle for an
    environment. ENV-DRIVEN: resolves ``resolve_environment(env).kubernetes``
    (a ``K8sTarget``) and builds the client for BOTH modes:
      - ``gke``        -> ADC + GCP Container API coords (project/location/name).
      - ``kubeconfig`` -> ``new_client_from_config(context=target.context)``
                          (None -> current-context / in-cluster SA).
    Cached by env name. (`build_fake()` pre-seeds this cache so the offline
    tests never reach the resolver.)"""
    if env in _clients:
        return _clients[env]

    _ensure_imports()
    from opsrag.environments import resolve_environment

    target = resolve_environment(env).kubernetes
    if target is None:
        raise RuntimeError(
            f"k8s mcp: environment {env!r} has no `kubernetes` target "
            f"configured -- add a `kubernetes:` block under "
            f"environments.targets.{env}."
        )

    if target.mode == "gke":
        # GKE-API mode: ADC token + GCP Container API fetches the endpoint +
        # CA cert. Same flow in local dev (gcloud ADC) and pod (Workload
        # Identity).
        if not (target.project and target.location and target.name):
            raise RuntimeError(
                f"k8s mcp: env {env!r} kubernetes mode=gke requires "
                f"project/location/name."
            )
        cluster_info = await _gke_get_cluster(
            project=target.project,
            location=target.location,
            name=target.name,
        )
        api_client = await _build_api_client_from_cluster_info(
            cluster=env, cluster_info=cluster_info,
        )
    elif target.context is None and os.environ.get("KUBERNETES_SERVICE_HOST"):
        # In-cluster ServiceAccount: kubeconfig mode with no explicit
        # context, running inside a pod -> use the mounted SA config.
        k8s_config.load_incluster_config()
        api_client = k8s_client.ApiClient()
    else:
        # Standard kubeconfig: select the target context (or the
        # kubeconfig's current-context when context is None).
        cfg_path = os.environ.get("KUBECONFIG") or os.path.expanduser("~/.kube/config")
        api_client = await k8s_config.new_client_from_config(
            config_file=cfg_path, context=(target.context or None),
        )

    conn = _Connection(
        api_client=api_client,
        core=k8s_client.CoreV1Api(api_client=api_client),
        apps=k8s_client.AppsV1Api(api_client=api_client),
        batch=k8s_client.BatchV1Api(api_client=api_client),
        rbac=k8s_client.RbacAuthorizationV1Api(api_client=api_client),
        custom=k8s_client.CustomObjectsApi(api_client=api_client),
    )
    _clients[env] = conn
    return conn


# POSIX/k8s convention: container exit codes 128+N where N is the signal
# number that killed the process. Translating to human-readable summaries
# saves the LLM from "what does 137 mean" lookups and surfaces the cheap
# signal (OOMKilled) without needing to fetch logs.
_EXIT_CODE_SUMMARIES = {
    0:   "Success",
    1:   "GenericAppError",
    125: "ContainerCannotRun",
    126: "ContainerCommandNotExecutable",
    127: "ContainerCommandNotFound",
    130: "SIGINT (Ctrl+C)",
    134: "SIGABRT (assert / abort)",
    137: "OOMKilled (SIGKILL -- out of memory)",
    139: "SIGSEGV (segmentation fault)",
    143: "SIGTERM (graceful stop)",
    255: "GenericExitCodeMinus1",
}


def _decode_exit_code(code: Any) -> str | None:
    """Best-effort summary string for a container exit code. Returns None
    when code is None; never raises."""
    if code is None:
        return None
    try:
        return _EXIT_CODE_SUMMARIES.get(int(code)) or f"ExitCode{code}"
    except Exception:  # noqa: BLE001
        return None


def _summarize_pod(pod: Any) -> dict:
    """Trim a V1Pod down to fields the LLM actually needs.

    Aggregated top-level fields (`pod_ready`, `max_restart_count`,
    `worst_terminated_*`) save the LLM from per-container reasoning --
    one glance tells it whether this pod is healthy or which signal
    killed it (e.g. `worst_termination_summary="OOMKilled"`).
    """
    status = pod.status or k8s_client.V1PodStatus()
    containers = []
    for cs in (status.container_statuses or []):
        last_state = cs.last_state or k8s_client.V1ContainerState()
        terminated = last_state.terminated
        containers.append({
            "name": cs.name,
            "ready": cs.ready,
            "restart_count": cs.restart_count,
            "image": cs.image,
            "state": (
                "running" if (cs.state and cs.state.running) else
                "waiting" if (cs.state and cs.state.waiting) else
                "terminated" if (cs.state and cs.state.terminated) else
                "unknown"
            ),
            "waiting_reason": (cs.state.waiting.reason if cs.state and cs.state.waiting else None),
            "last_terminated_reason": (terminated.reason if terminated else None),
            "last_terminated_exit_code": (terminated.exit_code if terminated else None),
            "last_terminated_summary": _decode_exit_code(terminated.exit_code) if terminated else None,
        })

    # Aggregate per-container state into pod-level fields.
    if containers:
        pod_ready = all(c.get("ready") for c in containers)
        max_restart_count = max((c.get("restart_count") or 0) for c in containers)
        # "Worst" termination = container with highest restart count + a
        # non-zero exit. Falls back to the first one with any termination
        # data so e.g. a CrashLoopBackOff pod with one restart still
        # surfaces its signal.
        worst = max(
            containers,
            key=lambda c: (
                (c.get("restart_count") or 0),
                1 if (c.get("last_terminated_exit_code") not in (None, 0)) else 0,
            ),
            default={},
        )
        worst_terminated_reason = worst.get("last_terminated_reason")
        worst_terminated_exit_code = worst.get("last_terminated_exit_code")
        worst_termination_summary = worst.get("last_terminated_summary")
        worst_waiting_reason = next(
            (c.get("waiting_reason") for c in containers if c.get("waiting_reason")),
            None,
        )
    else:
        pod_ready = False
        max_restart_count = 0
        worst_terminated_reason = None
        worst_terminated_exit_code = None
        worst_termination_summary = None
        worst_waiting_reason = None

    return {
        "name": pod.metadata.name,
        "namespace": pod.metadata.namespace,
        "node": (pod.spec.node_name if pod.spec else None),
        "phase": status.phase,
        "reason": status.reason,
        "message": status.message,
        "start_time": str(status.start_time) if status.start_time else None,
        "ip": status.pod_ip,
        # Aggregated signals -- read these BEFORE diving into `containers`.
        "pod_ready": pod_ready,
        "max_restart_count": max_restart_count,
        "worst_terminated_reason": worst_terminated_reason,
        "worst_terminated_exit_code": worst_terminated_exit_code,
        "worst_termination_summary": worst_termination_summary,
        "worst_waiting_reason": worst_waiting_reason,
        "containers": containers,
        "labels": (pod.metadata.labels or {}),
        "owner": [
            {"kind": o.kind, "name": o.name}
            for o in (pod.metadata.owner_references or [])
        ],
        "conditions": [
            {"type": c.type, "status": c.status, "reason": c.reason}
            for c in (status.conditions or [])
            if c.status != "True"  # only surface non-True conditions to keep payload small
        ],
    }


def _summarize_event(ev: Any) -> dict:
    return {
        "type": ev.type,
        "reason": ev.reason,
        "message": _redact(ev.message or ""),
        "involved_object": {
            "kind": ev.involved_object.kind if ev.involved_object else None,
            "name": ev.involved_object.name if ev.involved_object else None,
        },
        "count": ev.count,
        "first_seen": str(ev.first_timestamp) if ev.first_timestamp else None,
        "last_seen": str(ev.last_timestamp) if ev.last_timestamp else None,
    }


def _summarize_deployment(dep: Any) -> dict:
    spec = dep.spec or k8s_client.V1DeploymentSpec()
    status = dep.status or k8s_client.V1DeploymentStatus()
    containers = []
    for c in (spec.template.spec.containers if spec.template and spec.template.spec else []):
        containers.append({
            "name": c.name,
            "image": c.image,
            "resources": (c.resources.to_dict() if c.resources else {}),
        })
    return {
        "name": dep.metadata.name,
        "namespace": dep.metadata.namespace,
        "replicas_desired": spec.replicas,
        "replicas_ready": status.ready_replicas or 0,
        "replicas_available": status.available_replicas or 0,
        "replicas_unavailable": status.unavailable_replicas or 0,
        "strategy": (spec.strategy.type if spec.strategy else None),
        "containers": containers,
        "labels": (dep.metadata.labels or {}),
        "annotations": (dep.metadata.annotations or {}),
        "conditions": [
            {"type": c.type, "status": c.status, "reason": c.reason, "message": c.message}
            for c in (status.conditions or [])
        ],
    }


def _summarize_service(svc: Any) -> dict:
    spec = svc.spec or k8s_client.V1ServiceSpec()
    return {
        "name": svc.metadata.name,
        "namespace": svc.metadata.namespace,
        "type": spec.type,
        "cluster_ip": spec.cluster_ip,
        "external_ips": spec.external_i_ps or [],
        "selector": spec.selector or {},
        "ports": [
            {"name": p.name, "port": p.port, "target_port": str(p.target_port), "protocol": p.protocol}
            for p in (spec.ports or [])
        ],
    }


def _summarize_daemonset(ds: Any) -> dict:
    spec = ds.spec or k8s_client.V1DaemonSetSpec()
    status = ds.status or k8s_client.V1DaemonSetStatus()
    containers = []
    for c in (spec.template.spec.containers if spec.template and spec.template.spec else []):
        containers.append({
            "name": c.name,
            "image": c.image,
            "resources": (c.resources.to_dict() if c.resources else {}),
        })
    return {
        "name": ds.metadata.name,
        "namespace": ds.metadata.namespace,
        "desired_number_scheduled": status.desired_number_scheduled or 0,
        "current_number_scheduled": status.current_number_scheduled or 0,
        "number_ready": status.number_ready or 0,
        "number_available": status.number_available or 0,
        "number_unavailable": status.number_unavailable or 0,
        "number_misscheduled": status.number_misscheduled or 0,
        "updated_number_scheduled": status.updated_number_scheduled or 0,
        "selector_match_labels": (spec.selector.match_labels if spec.selector else {}),
        "containers": containers,
        "labels": (ds.metadata.labels or {}),
        "conditions": [
            {"type": c.type, "status": c.status, "reason": c.reason, "message": c.message}
            for c in (status.conditions or [])
        ],
    }


def _summarize_statefulset(sts: Any) -> dict:
    spec = sts.spec or k8s_client.V1StatefulSetSpec()
    status = sts.status or k8s_client.V1StatefulSetStatus()
    containers = []
    for c in (spec.template.spec.containers if spec.template and spec.template.spec else []):
        containers.append({
            "name": c.name,
            "image": c.image,
            "resources": (c.resources.to_dict() if c.resources else {}),
        })
    return {
        "name": sts.metadata.name,
        "namespace": sts.metadata.namespace,
        "replicas_desired": spec.replicas,
        "replicas_ready": status.ready_replicas or 0,
        "replicas_current": status.current_replicas or 0,
        "replicas_updated": status.updated_replicas or 0,
        "service_name": spec.service_name,
        "update_strategy": (spec.update_strategy.type if spec.update_strategy else None),
        "containers": containers,
        "labels": (sts.metadata.labels or {}),
        "conditions": [
            {"type": c.type, "status": c.status, "reason": c.reason, "message": c.message}
            for c in (status.conditions or [])
        ],
    }


def _summarize_job(job: Any) -> dict:
    spec = job.spec or k8s_client.V1JobSpec()
    status = job.status or k8s_client.V1JobStatus()
    containers = []
    for c in (spec.template.spec.containers if spec.template and spec.template.spec else []):
        containers.append({"name": c.name, "image": c.image})
    return {
        "name": job.metadata.name,
        "namespace": job.metadata.namespace,
        "parallelism": spec.parallelism,
        "completions": spec.completions,
        "backoff_limit": spec.backoff_limit,
        "active": status.active or 0,
        "succeeded": status.succeeded or 0,
        "failed": status.failed or 0,
        "start_time": str(status.start_time) if status.start_time else None,
        "completion_time": str(status.completion_time) if status.completion_time else None,
        "containers": containers,
        "labels": (job.metadata.labels or {}),
        "owner": [
            {"kind": o.kind, "name": o.name}
            for o in (job.metadata.owner_references or [])
        ],
        "conditions": [
            {"type": c.type, "status": c.status, "reason": c.reason, "message": c.message}
            for c in (status.conditions or [])
        ],
    }


def _summarize_cronjob(cj: Any) -> dict:
    spec = cj.spec or k8s_client.V1CronJobSpec()
    status = cj.status or k8s_client.V1CronJobStatus()
    job_template = spec.job_template.spec if spec.job_template else None
    containers = []
    if job_template and job_template.template and job_template.template.spec:
        for c in job_template.template.spec.containers:
            containers.append({"name": c.name, "image": c.image})
    return {
        "name": cj.metadata.name,
        "namespace": cj.metadata.namespace,
        "schedule": spec.schedule,
        "suspend": bool(spec.suspend),
        "concurrency_policy": spec.concurrency_policy,
        "successful_jobs_history_limit": spec.successful_jobs_history_limit,
        "failed_jobs_history_limit": spec.failed_jobs_history_limit,
        "starting_deadline_seconds": spec.starting_deadline_seconds,
        "last_schedule_time": str(status.last_schedule_time) if status.last_schedule_time else None,
        "last_successful_time": str(status.last_successful_time) if getattr(status, "last_successful_time", None) else None,
        "active_jobs": [
            {"name": o.name, "namespace": getattr(o, "namespace", None)}
            for o in (status.active or [])
        ],
        "containers": containers,
        "labels": (cj.metadata.labels or {}),
    }


def _summarize_role_binding(rb: Any) -> dict:
    return {
        "name": rb.metadata.name,
        "namespace": rb.metadata.namespace,
        "role_ref": {
            "kind": rb.role_ref.kind if rb.role_ref else None,
            "name": rb.role_ref.name if rb.role_ref else None,
        },
        "subjects": [
            {"kind": s.kind, "name": s.name, "namespace": getattr(s, "namespace", None)}
            for s in (rb.subjects or [])
        ],
    }


# --- handlers -------------------------------------------------------


async def _h_get_pod(_unused, args: dict) -> Any:
    cluster = _resolve_cluster(args)
    conn = await _get_connection(cluster)
    pod = await conn.core.read_namespaced_pod(
        name=args["name"], namespace=args["namespace"],
    )
    return {"cluster": cluster, "pod": _summarize_pod(pod)}


async def _h_list_pods(_unused, args: dict) -> Any:
    cluster = _resolve_cluster(args)
    conn = await _get_connection(cluster)
    namespace = args.get("namespace") or None
    kwargs: dict[str, Any] = {}
    if args.get("label_selector"):
        kwargs["label_selector"] = args["label_selector"]
    if args.get("field_selector"):
        kwargs["field_selector"] = args["field_selector"]
    # Cap a single-page list at 500. Cluster-wide unfiltered listing on
    # a large cluster (>500 pods) won't see everything -- caller should
    # use `k8s_find_workloads` for cluster-wide service discovery
    # instead, which scans Deployments/StatefulSets/DaemonSets
    # (far scarcer than pods).
    kwargs["limit"] = min(int(args.get("limit") or 50), 500)
    # When no namespace is given, list cluster-wide. This is the
    # "discover before assume" path: caller doesn't know which
    # namespace the workload lives in, so we ask the API to scan all
    # of them. The result's `metadata.namespace` per pod then grounds
    # any follow-up scope-narrowed call.
    if namespace:
        pod_list = await conn.core.list_namespaced_pod(
            namespace=namespace, **kwargs,
        )
    else:
        pod_list = await conn.core.list_pod_for_all_namespaces(**kwargs)
    return {
        "cluster": cluster,
        "namespace": namespace or "(all)",
        "count": len(pod_list.items),
        "pods": [_summarize_pod(p) for p in pod_list.items],
    }


async def _h_get_pod_logs(_unused, args: dict) -> Any:
    cluster = _resolve_cluster(args)
    conn = await _get_connection(cluster)
    kwargs = {
        "name": args["name"],
        "namespace": args["namespace"],
        "tail_lines": int(args.get("tail_lines") or _DEFAULT_TAIL_LINES),
    }
    if args.get("container"):
        kwargs["container"] = args["container"]
    if args.get("previous"):
        kwargs["previous"] = bool(args["previous"])
    try:
        log_text = await conn.core.read_namespaced_pod_log(**kwargs)
    except k8s_client.ApiException as exc:
        return {"cluster": cluster, "error": f"k8s {exc.status}: {exc.reason}"}
    redacted = _redact(log_text or "")
    if len(redacted) > _LOG_TRUNCATE_CHARS:
        redacted = redacted[-_LOG_TRUNCATE_CHARS:]
    return {
        "cluster": cluster,
        "name": args["name"],
        "namespace": args["namespace"],
        "container": args.get("container"),
        "previous": bool(args.get("previous", False)),
        "tail_lines_requested": kwargs["tail_lines"],
        "log_chars": len(redacted),
        "log": redacted,
    }


async def _h_list_events(_unused, args: dict) -> Any:
    cluster = _resolve_cluster(args)
    conn = await _get_connection(cluster)
    namespace = args.get("namespace") or None
    kwargs: dict[str, Any] = {}
    if args.get("field_selector"):
        kwargs["field_selector"] = args["field_selector"]
    kwargs["limit"] = int(args.get("limit") or 50)
    # Cluster-wide listing when no namespace is supplied -- supports the
    # "discover before assume" pattern. Caller can use `field_selector`
    # like `involvedObject.name=<workload>` to scope without knowing the
    # namespace.
    if namespace:
        ev_list = await conn.core.list_namespaced_event(
            namespace=namespace, **kwargs,
        )
    else:
        ev_list = await conn.core.list_event_for_all_namespaces(**kwargs)
    items = [_summarize_event(e) for e in ev_list.items]
    # Most-recent first by last_seen.
    items.sort(key=lambda e: e.get("last_seen") or "", reverse=True)
    return {
        "cluster": cluster, "namespace": namespace or "(all)",
        "count": len(items), "events": items,
    }


async def _h_get_deployment(_unused, args: dict) -> Any:
    cluster = _resolve_cluster(args)
    conn = await _get_connection(cluster)
    dep = await conn.apps.read_namespaced_deployment(
        name=args["name"], namespace=args["namespace"],
    )
    return {"cluster": cluster, "deployment": _summarize_deployment(dep)}


async def _h_list_deployments(_unused, args: dict) -> Any:
    cluster = _resolve_cluster(args)
    conn = await _get_connection(cluster)
    namespace = args.get("namespace") or None
    kwargs: dict[str, Any] = {}
    if args.get("label_selector"):
        kwargs["label_selector"] = args["label_selector"]
    kwargs["limit"] = int(args.get("limit") or 50)
    if namespace:
        dep_list = await conn.apps.list_namespaced_deployment(
            namespace=namespace, **kwargs,
        )
    else:
        dep_list = await conn.apps.list_deployment_for_all_namespaces(**kwargs)
    return {
        "cluster": cluster,
        "namespace": namespace or "(all)",
        "count": len(dep_list.items),
        "deployments": [_summarize_deployment(d) for d in dep_list.items],
    }


async def _h_get_service(_unused, args: dict) -> Any:
    cluster = _resolve_cluster(args)
    conn = await _get_connection(cluster)
    svc = await conn.core.read_namespaced_service(
        name=args["name"], namespace=args["namespace"],
    )
    return {"cluster": cluster, "service": _summarize_service(svc)}


async def _h_list_services(_unused, args: dict) -> Any:
    cluster = _resolve_cluster(args)
    conn = await _get_connection(cluster)
    namespace = args.get("namespace") or None
    kwargs: dict[str, Any] = {}
    if args.get("label_selector"):
        kwargs["label_selector"] = args["label_selector"]
    kwargs["limit"] = int(args.get("limit") or 50)
    if namespace:
        svc_list = await conn.core.list_namespaced_service(
            namespace=namespace, **kwargs,
        )
    else:
        svc_list = await conn.core.list_service_for_all_namespaces(**kwargs)
    return {
        "cluster": cluster, "namespace": namespace or "(all)",
        "count": len(svc_list.items),
        "services": [_summarize_service(s) for s in svc_list.items],
    }


async def _h_get_daemonset(_unused, args: dict) -> Any:
    cluster = _resolve_cluster(args)
    conn = await _get_connection(cluster)
    ds = await conn.apps.read_namespaced_daemon_set(
        name=args["name"], namespace=args["namespace"],
    )
    return {"cluster": cluster, "daemonset": _summarize_daemonset(ds)}


async def _h_find_workloads(_unused, args: dict) -> Any:
    """Substring-match a workload name across ALL kinds cluster-wide,
    then drill into each match's pods to surface POD-LEVEL health
    (restart counts, OOMKilled, CrashLoopBackOff) -- not just
    deployment-level ready/desired.

    Why pod-level signals matter: a Deployment with `ready=4/4 desired=4`
    looks healthy at the controller level even when its pods are being
    OOM-killed every few minutes and restart_count is climbing. Reading
    only `ready/desired` led both chat and investigate paths to declare
    the service healthy on alert TFFFR1 (2026-05-27 manual investigation)
    and then hallucinate a Kong config / dependency error elsewhere.

    For each matched workload we fetch the namespaced pods that match
    its `selector_match_labels` and aggregate:
      - `pod_count`
      - `unhealthy_pod_count`  (pod_ready=False or max_restart>0)
      - `max_pod_restart_count`
      - `worst_pod_termination_summary`  (e.g. "OOMKilled (SIGKILL -- out of memory)")
      - `worst_pod_waiting_reason`       (e.g. "CrashLoopBackOff")
    A workload is marked `unhealthy: True` if the controller is short
    on replicas OR any pod signal indicates trouble.
    """
    cluster = _resolve_cluster(args)
    conn = await _get_connection(cluster)
    needle = (args.get("name_contains") or "").lower().strip()
    if not needle:
        return {"error": "name_contains is required (substring to match against workload names)"}

    # Fetch the three workload kinds in parallel.
    dep_task = conn.apps.list_deployment_for_all_namespaces(limit=500)
    sts_task = conn.apps.list_stateful_set_for_all_namespaces(limit=500)
    ds_task = conn.apps.list_daemon_set_for_all_namespaces(limit=500)
    deps, stss, dss = await asyncio.gather(dep_task, sts_task, ds_task, return_exceptions=True)

    matches: list[dict[str, Any]] = []

    def _push(kind: str, items: Any, ready_attr: str, desired_attr: str) -> None:
        if isinstance(items, Exception):
            return
        for w in (items.items or []):
            name = (w.metadata.name or "")
            if needle not in name.lower():
                continue
            status = w.status or type(w.status)()
            spec = w.spec or type(w.spec)()
            ready = getattr(status, ready_attr, None) or 0
            desired = getattr(spec, desired_attr, None)
            if desired is None and kind == "DaemonSet":
                desired = getattr(status, "desired_number_scheduled", None) or 0
            sel = (
                (spec.selector.match_labels if (spec.selector and hasattr(spec.selector, "match_labels")) else {})
                if hasattr(spec, "selector") else {}
            )
            matches.append({
                "kind": kind,
                "namespace": w.metadata.namespace,
                "name": name,
                "ready": ready,
                "desired": desired,
                "controller_short_on_replicas": (ready or 0) < (desired or 0),
                "selector_match_labels": sel or {},
                "creation_timestamp": str(w.metadata.creation_timestamp) if w.metadata.creation_timestamp else None,
            })

    _push("Deployment",  deps, "ready_replicas",     "replicas")
    _push("StatefulSet", stss, "ready_replicas",     "replicas")
    _push("DaemonSet",   dss,  "number_ready",       "desired_number_scheduled")

    # -- Drill into pods per match, in parallel --
    async def _pod_signals(m: dict[str, Any]) -> dict[str, Any]:
        sel = m.get("selector_match_labels") or {}
        if not sel:
            return {"pod_count": 0, "unhealthy_pod_count": 0,
                    "max_pod_restart_count": 0,
                    "worst_pod_termination_summary": None,
                    "worst_pod_waiting_reason": None,
                    "worst_pod_terminated_reason": None,
                    "unhealthy_pod_names": []}
        sel_str = ",".join(f"{k}={v}" for k, v in sel.items())
        try:
            pod_list = await conn.core.list_namespaced_pod(
                namespace=m["namespace"], label_selector=sel_str, limit=200,
            )
        except Exception as exc:  # noqa: BLE001
            return {"pod_signals_error": str(exc)}
        pods = [_summarize_pod(p) for p in (pod_list.items or [])]
        if not pods:
            return {"pod_count": 0, "unhealthy_pod_count": 0,
                    "max_pod_restart_count": 0,
                    "worst_pod_termination_summary": None,
                    "worst_pod_waiting_reason": None,
                    "worst_pod_terminated_reason": None,
                    "unhealthy_pod_names": []}
        max_restart = max((p.get("max_restart_count") or 0) for p in pods)
        unhealthy = [
            p for p in pods
            if (not p.get("pod_ready"))
            or ((p.get("max_restart_count") or 0) > 0)
            or p.get("worst_waiting_reason")
        ]
        worst_pod = max(
            pods,
            key=lambda p: (
                (p.get("max_restart_count") or 0),
                1 if p.get("worst_termination_summary") else 0,
            ),
            default={},
        )
        return {
            "pod_count": len(pods),
            "unhealthy_pod_count": len(unhealthy),
            "max_pod_restart_count": max_restart,
            "worst_pod_termination_summary": worst_pod.get("worst_termination_summary"),
            "worst_pod_terminated_reason": worst_pod.get("worst_terminated_reason"),
            "worst_pod_waiting_reason": worst_pod.get("worst_waiting_reason"),
            "unhealthy_pod_names": [p.get("name") for p in unhealthy[:10]],
        }

    if matches:
        signals = await asyncio.gather(*[_pod_signals(m) for m in matches], return_exceptions=False)
        for m, s in zip(matches, signals):
            m.update(s)
            # Final unhealthy flag rolls up controller-level + pod-level signals.
            m["unhealthy"] = (
                m.get("controller_short_on_replicas")
                or (m.get("unhealthy_pod_count") or 0) > 0
                or (m.get("max_pod_restart_count") or 0) > 0
                or bool(m.get("worst_pod_termination_summary"))
                or bool(m.get("worst_pod_waiting_reason"))
            )

    # Sort: unhealthy first (with the worst restart count first within that), then by kind/name.
    matches.sort(key=lambda m: (
        not m.get("unhealthy"),
        -(m.get("max_pod_restart_count") or 0),
        m.get("kind"),
        m.get("name"),
    ))
    return {
        "cluster": cluster,
        "name_contains": needle,
        "count": len(matches),
        "workloads": matches,
    }


async def _h_list_daemonsets(_unused, args: dict) -> Any:
    cluster = _resolve_cluster(args)
    conn = await _get_connection(cluster)
    namespace = args.get("namespace") or None
    kwargs: dict[str, Any] = {}
    if args.get("label_selector"):
        kwargs["label_selector"] = args["label_selector"]
    kwargs["limit"] = int(args.get("limit") or 50)
    if namespace:
        ds_list = await conn.apps.list_namespaced_daemon_set(
            namespace=namespace, **kwargs,
        )
    else:
        ds_list = await conn.apps.list_daemon_set_for_all_namespaces(**kwargs)
    return {
        "cluster": cluster, "namespace": namespace or "(all)",
        "count": len(ds_list.items),
        "daemonsets": [_summarize_daemonset(d) for d in ds_list.items],
    }


async def _h_get_statefulset(_unused, args: dict) -> Any:
    cluster = _resolve_cluster(args)
    conn = await _get_connection(cluster)
    sts = await conn.apps.read_namespaced_stateful_set(
        name=args["name"], namespace=args["namespace"],
    )
    return {"cluster": cluster, "statefulset": _summarize_statefulset(sts)}


async def _h_list_statefulsets(_unused, args: dict) -> Any:
    cluster = _resolve_cluster(args)
    conn = await _get_connection(cluster)
    kwargs = {"namespace": args["namespace"]}
    if args.get("label_selector"):
        kwargs["label_selector"] = args["label_selector"]
    kwargs["limit"] = int(args.get("limit") or 50)
    sts_list = await conn.apps.list_namespaced_stateful_set(**kwargs)
    return {
        "cluster": cluster, "namespace": args["namespace"],
        "count": len(sts_list.items),
        "statefulsets": [_summarize_statefulset(s) for s in sts_list.items],
    }


async def _h_get_job(_unused, args: dict) -> Any:
    cluster = _resolve_cluster(args)
    conn = await _get_connection(cluster)
    job = await conn.batch.read_namespaced_job(
        name=args["name"], namespace=args["namespace"],
    )
    return {"cluster": cluster, "job": _summarize_job(job)}


async def _h_list_jobs(_unused, args: dict) -> Any:
    cluster = _resolve_cluster(args)
    conn = await _get_connection(cluster)
    kwargs = {"namespace": args["namespace"]}
    if args.get("label_selector"):
        kwargs["label_selector"] = args["label_selector"]
    kwargs["limit"] = int(args.get("limit") or 50)
    job_list = await conn.batch.list_namespaced_job(**kwargs)
    return {
        "cluster": cluster, "namespace": args["namespace"],
        "count": len(job_list.items),
        "jobs": [_summarize_job(j) for j in job_list.items],
    }


async def _h_get_cronjob(_unused, args: dict) -> Any:
    cluster = _resolve_cluster(args)
    conn = await _get_connection(cluster)
    cj = await conn.batch.read_namespaced_cron_job(
        name=args["name"], namespace=args["namespace"],
    )
    return {"cluster": cluster, "cronjob": _summarize_cronjob(cj)}


async def _h_list_cronjobs(_unused, args: dict) -> Any:
    cluster = _resolve_cluster(args)
    conn = await _get_connection(cluster)
    kwargs = {"namespace": args["namespace"]}
    if args.get("label_selector"):
        kwargs["label_selector"] = args["label_selector"]
    kwargs["limit"] = int(args.get("limit") or 50)
    cj_list = await conn.batch.list_namespaced_cron_job(**kwargs)
    return {
        "cluster": cluster, "namespace": args["namespace"],
        "count": len(cj_list.items),
        "cronjobs": [_summarize_cronjob(c) for c in cj_list.items],
    }


async def _h_get_role_bindings(_unused, args: dict) -> Any:
    cluster = _resolve_cluster(args)
    conn = await _get_connection(cluster)
    rb_list = await conn.rbac.list_namespaced_role_binding(
        namespace=args["namespace"],
        limit=int(args.get("limit") or 50),
    )
    return {
        "cluster": cluster, "namespace": args["namespace"],
        "count": len(rb_list.items),
        "role_bindings": [_summarize_role_binding(rb) for rb in rb_list.items],
    }


async def _h_top_pod(_unused, args: dict) -> Any:
    """metrics.k8s.io PodMetrics -- needs metrics-server installed in
    the target cluster. Returns 404 with a clear message if not."""
    cluster = _resolve_cluster(args)
    conn = await _get_connection(cluster)
    try:
        obj = await conn.custom.get_namespaced_custom_object(
            group="metrics.k8s.io",
            version="v1beta1",
            namespace=args["namespace"],
            plural="pods",
            name=args["name"],
        )
    except k8s_client.ApiException as exc:
        return {
            "cluster": cluster,
            "error": f"metrics.k8s.io {exc.status}: {exc.reason}",
            "hint": "metrics-server may not be installed; use k8s_get_pod for static info instead",
        }
    containers = []
    for c in (obj.get("containers") or []):
        usage = c.get("usage") or {}
        containers.append({
            "name": c.get("name"),
            "cpu": usage.get("cpu"),       # e.g. "12m"
            "memory": usage.get("memory"),  # e.g. "23Mi"
        })
    return {
        "cluster": cluster, "namespace": args["namespace"],
        "name": args["name"],
        "timestamp": obj.get("timestamp"),
        "window": obj.get("window"),
        "containers": containers,
    }


# --- tool registry --------------------------------------------------


_ENV_PROP = {
    "type": "string",
    "description": (
        "Environment name from the unified `environments:` registry "
        "(e.g. 'prod', 'staging'). Selects which env's Kubernetes API to "
        "query. Defaults to the registry's default env (or "
        "OPSRAG_K8S_DEFAULT_CLUSTER env). Required when no environment is "
        "configured."
    ),
}

# Back-compat alias: older callers/alerts pass `cluster`. Accepted as a
# synonym for `env` (the handler reads `env` first, then `cluster`).
_CLUSTER_PROP = {
    "type": "string",
    "description": (
        "DEPRECATED alias for `env`. Environment name from the "
        "`environments:` registry (kept for back-compat; prefer `env`)."
    ),
}


KUBERNETES_TOOLS: list[MCPTool] = [
    MCPTool(
        name="k8s_get_pod",
        description="Get a single pod's status (phase, conditions, container restart counts, image).",
        input_schema={
            "type": "object",
            "properties": {
                "cluster": _CLUSTER_PROP,
                "namespace": {"type": "string"},
                "name": {"type": "string"},
            },
            "required": ["namespace", "name"],
        },
        handler=_h_get_pod,
    ),
    MCPTool(
        name="k8s_list_pods",
        description="List pods (filter by label_selector / field_selector). OMIT `namespace` to list cluster-wide -- useful for the discovery step when the alert didn't specify the namespace. Default limit 50.",
        input_schema={
            "type": "object",
            "properties": {
                "cluster": _CLUSTER_PROP,
                "namespace": {"type": "string", "description": "Optional -- omit to list across all namespaces."},
                "label_selector": {"type": "string", "description": "e.g. 'app=<service>,env=<env>'"},
                "field_selector": {"type": "string"},
                "limit": {"type": "number"},
            },
            "required": [],
        },
        handler=_h_list_pods,
    ),
    MCPTool(
        name="k8s_get_pod_logs",
        description="Tail recent log lines from a pod's container. `previous=true` reads the logs from the last terminated container (useful for CrashLoopBackOff). Tokens are redacted.",
        input_schema={
            "type": "object",
            "properties": {
                "cluster": _CLUSTER_PROP,
                "namespace": {"type": "string"},
                "name": {"type": "string"},
                "container": {"type": "string", "description": "Required if pod has multiple containers"},
                "tail_lines": {"type": "number", "description": f"Default {_DEFAULT_TAIL_LINES}"},
                "previous": {"type": "boolean", "description": "Read logs of the previous (terminated) container"},
            },
            "required": ["namespace", "name"],
        },
        handler=_h_get_pod_logs,
    ),
    MCPTool(
        name="k8s_list_events",
        description="Recent Events, sorted most-recent first. OMIT `namespace` to scan all namespaces (use this for discovery when you don't yet know which namespace the workload lives in). Filter by `field_selector` (e.g. `involvedObject.name=<pod>`).",
        input_schema={
            "type": "object",
            "properties": {
                "cluster": _CLUSTER_PROP,
                "namespace": {"type": "string", "description": "Optional -- omit to list across all namespaces."},
                "field_selector": {"type": "string"},
                "limit": {"type": "number"},
            },
            "required": [],
        },
        handler=_h_list_events,
    ),
    MCPTool(
        name="k8s_get_deployment",
        description="Get a Deployment -- desired vs ready replicas, container images, conditions, strategy.",
        input_schema={
            "type": "object",
            "properties": {
                "cluster": _CLUSTER_PROP,
                "namespace": {"type": "string"},
                "name": {"type": "string"},
            },
            "required": ["namespace", "name"],
        },
        handler=_h_get_deployment,
    ),
    MCPTool(
        name="k8s_list_deployments",
        description="List Deployments. OMIT `namespace` to list cluster-wide.",
        input_schema={
            "type": "object",
            "properties": {
                "cluster": _CLUSTER_PROP,
                "namespace": {"type": "string", "description": "Optional -- omit for all namespaces."},
                "label_selector": {"type": "string"},
                "limit": {"type": "number"},
            },
            "required": [],
        },
        handler=_h_list_deployments,
    ),
    MCPTool(
        name="k8s_get_service",
        description="Get a Service -- type, cluster IP, ports, selector.",
        input_schema={
            "type": "object",
            "properties": {
                "cluster": _CLUSTER_PROP,
                "namespace": {"type": "string"},
                "name": {"type": "string"},
            },
            "required": ["namespace", "name"],
        },
        handler=_h_get_service,
    ),
    MCPTool(
        name="k8s_list_services",
        description="List Services (filter by label_selector). OMIT `namespace` to list cluster-wide. Default limit 50.",
        input_schema={
            "type": "object",
            "properties": {
                "cluster": _CLUSTER_PROP,
                "namespace": {"type": "string", "description": "Optional -- omit for all namespaces."},
                "label_selector": {"type": "string"},
                "limit": {"type": "number"},
            },
            "required": [],
        },
        handler=_h_list_services,
    ),
    MCPTool(
        name="k8s_get_daemonset",
        description="Get a DaemonSet -- desired/ready/unavailable counts, pod selector, update strategy. Useful when the alert names a DaemonSet (fluentd, node-exporter, csi-driver, etc.).",
        input_schema={
            "type": "object",
            "properties": {
                "cluster": _CLUSTER_PROP,
                "namespace": {"type": "string"},
                "name": {"type": "string"},
            },
            "required": ["namespace", "name"],
        },
        handler=_h_get_daemonset,
    ),
    MCPTool(
        name="k8s_list_daemonsets",
        description="List DaemonSets -- desired/ready/unavailable counts + selector match_labels per DS. OMIT `namespace` to list cluster-wide -- this is the fastest way to discover where a named DaemonSet (e.g. fluentd) actually lives and what selector its pods use.",
        input_schema={
            "type": "object",
            "properties": {
                "cluster": _CLUSTER_PROP,
                "namespace": {"type": "string", "description": "Optional -- omit for all namespaces."},
                "label_selector": {"type": "string"},
                "limit": {"type": "number"},
            },
            "required": [],
        },
        handler=_h_list_daemonsets,
    ),
    MCPTool(
        name="k8s_find_workloads",
        description=(
            "PREFERRED FIRST CALL when an alert names a service. Substring-matches "
            "`name_contains` against the names of every Deployment / StatefulSet / DaemonSet "
            "cluster-wide AND drills into each match's pods to surface POD-LEVEL health. "
            "Each returned workload includes: `kind`, `namespace`, `name`, controller "
            "counts (`ready`/`desired`), AND pod-aggregated signals -- "
            "`pod_count`, `unhealthy_pod_count`, `max_pod_restart_count`, "
            "`worst_pod_termination_summary` (e.g. \"OOMKilled (SIGKILL -- out of memory)\"), "
            "`worst_pod_waiting_reason` (e.g. \"CrashLoopBackOff\"), and "
            "`unhealthy_pod_names`. The `unhealthy: true` flag is set if EITHER the "
            "controller is short on replicas OR any pod has restarted / OOM-killed / "
            "is in a waiting state -- so a Deployment can be `ready=4/4` and still "
            "`unhealthy: true` when its pods are being OOM-killed. Sorted "
            "unhealthy-first, then by max_pod_restart_count desc."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "cluster": _CLUSTER_PROP,
                "name_contains": {"type": "string", "description": "Lowercase substring to find in workload names (e.g. a service name, or 'fluentd')."},
            },
            "required": ["name_contains"],
        },
        handler=_h_find_workloads,
    ),
    MCPTool(
        name="k8s_get_statefulset",
        description="Get a StatefulSet -- desired vs ready replicas, container images, update strategy, service name.",
        input_schema={
            "type": "object",
            "properties": {
                "cluster": _CLUSTER_PROP,
                "namespace": {"type": "string"},
                "name": {"type": "string"},
            },
            "required": ["namespace", "name"],
        },
        handler=_h_get_statefulset,
    ),
    MCPTool(
        name="k8s_list_statefulsets",
        description="List StatefulSets in a namespace.",
        input_schema={
            "type": "object",
            "properties": {
                "cluster": _CLUSTER_PROP,
                "namespace": {"type": "string"},
                "label_selector": {"type": "string"},
                "limit": {"type": "number"},
            },
            "required": ["namespace"],
        },
        handler=_h_list_statefulsets,
    ),
    MCPTool(
        name="k8s_get_job",
        description="Get a Job -- parallelism, completions, active/succeeded/failed counts, start/completion times, conditions.",
        input_schema={
            "type": "object",
            "properties": {
                "cluster": _CLUSTER_PROP,
                "namespace": {"type": "string"},
                "name": {"type": "string"},
            },
            "required": ["namespace", "name"],
        },
        handler=_h_get_job,
    ),
    MCPTool(
        name="k8s_list_jobs",
        description="List Jobs in a namespace (filter by label_selector -- e.g. 'app=migration'). Useful for inspecting recent Job runs spawned by a CronJob.",
        input_schema={
            "type": "object",
            "properties": {
                "cluster": _CLUSTER_PROP,
                "namespace": {"type": "string"},
                "label_selector": {"type": "string"},
                "limit": {"type": "number"},
            },
            "required": ["namespace"],
        },
        handler=_h_list_jobs,
    ),
    MCPTool(
        name="k8s_get_cronjob",
        description="Get a CronJob -- schedule, suspend flag, concurrency policy, last-schedule / last-successful timestamps, active child jobs.",
        input_schema={
            "type": "object",
            "properties": {
                "cluster": _CLUSTER_PROP,
                "namespace": {"type": "string"},
                "name": {"type": "string"},
            },
            "required": ["namespace", "name"],
        },
        handler=_h_get_cronjob,
    ),
    MCPTool(
        name="k8s_list_cronjobs",
        description="List CronJobs in a namespace.",
        input_schema={
            "type": "object",
            "properties": {
                "cluster": _CLUSTER_PROP,
                "namespace": {"type": "string"},
                "label_selector": {"type": "string"},
                "limit": {"type": "number"},
            },
            "required": ["namespace"],
        },
        handler=_h_list_cronjobs,
    ),
    MCPTool(
        name="k8s_get_role_bindings",
        description="List RoleBindings in a namespace -- who has which Role / ClusterRole. Useful for RBAC verification.",
        input_schema={
            "type": "object",
            "properties": {
                "cluster": _CLUSTER_PROP,
                "namespace": {"type": "string"},
                "limit": {"type": "number"},
            },
            "required": ["namespace"],
        },
        handler=_h_get_role_bindings,
    ),
    MCPTool(
        name="k8s_top_pod",
        description="Current CPU + memory usage of a pod (requires metrics-server in the target cluster).",
        input_schema={
            "type": "object",
            "properties": {
                "cluster": _CLUSTER_PROP,
                "namespace": {"type": "string"},
                "name": {"type": "string"},
            },
            "required": ["namespace", "name"],
        },
        handler=_h_top_pod,
    ),
]


# Expose the unified `env` arg on every tool alongside the back-compat
# `cluster` alias (handlers read `env` first, then `cluster`). Done once
# here so each tool schema stays a single `_CLUSTER_PROP`/`env` source.
for _t in KUBERNETES_TOOLS:
    _props = _t.input_schema.setdefault("properties", {})
    _props.setdefault("env", _ENV_PROP)


def get_tool(name: str) -> MCPTool:
    for t in KUBERNETES_TOOLS:
        if t.name == name:
            return t
    raise KeyError(f"unknown k8s tool: {name}")


# --- fake backend (FR-012; integration tests) ----------------------
#
# Data path (b): handlers ignore the `client` arg and read MODULE-INTERNAL
# state. Every handler calls `_resolve_cluster(args)` then
# `await _get_connection(cluster)`, which returns a `_Connection` bundling
# the per-resource API objects (`core`, `apps`, `batch`, `rbac`, `custom`)
# the handlers then call (e.g. `conn.core.list_namespaced_pod(...)`,
# `conn.apps.read_namespaced_deployment(...)`).
#
# The upstream K8s MCP carried a hand-rolled `FakeApiClient` pattern (rich,
# per-resource fakes returning canned objects) -- see research.md section 5.
# That upstream fake lived in `tests/unit/test_*_mcp.py` and was not carried
# into this repo, so we reuse the SAME approach here: a small set of fake
# API objects whose async methods return canned, attribute-faithful objects
# (so `_summarize_*` reads them exactly as it would live V1 objects). The
# canned objects are lightweight attribute bags (`_Obj`) rather than the
# kubernetes_asyncio V1 models, because the V1 models carry strict
# constructor validators and version-specific attribute spellings; the bag
# returns None for any attribute the handlers probe but the fake didn't set,
# matching how the handlers already guard `getattr` / `... or []` / `... or
# {}`. No kube-apiserver, no kubeconfig, no network.
#
# `build_fake()`:
#   (i)  installs a DeploymentContext with kubernetes.clusters={"prod":
#        "example-cluster"} so `_resolve_cluster` finds a default cluster;
#   (ii) pre-seeds `_clients["example-cluster"]` with a `_Connection` whose
#        API objects are the fakes, and flips the module load flags so
#        `_ensure_kubeconfig()` no-ops (no kubeconfig read).
# The teardown restores the prior deployment context and module state.

_FAKE_CLUSTER = "example-cluster"
_FAKE_ENV = "prod"
_FAKE_NAMESPACE = "default"
_FAKE_IMAGE = "registry.example.com/app:1.0"


class _Obj:
    """Lightweight attribute bag standing in for a kubernetes_asyncio V1
    model. Any attribute the handlers probe but the fake didn't set reads
    as None, matching the handlers' existing `getattr` / `... or []` / `...
    or {}` guards across client-library versions."""

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)

    def __getattr__(self, _name: str) -> Any:  # only for unset attrs
        return None


def _container(name: str = "app", image: str = _FAKE_IMAGE) -> _Obj:
    return _Obj(name=name, image=image, resources=None)


def _pod_template(containers: list | None = None) -> _Obj:
    return _Obj(spec=_Obj(containers=containers or [_container()]))


def _fake_pod(name: str = "example-pod") -> _Obj:
    """A pod with one container whose previous state terminated (OOMKilled)
    so the aggregated fields in `_summarize_pod` exercise their non-trivial
    path while the current state stays Running/ready."""
    cs = _Obj(
        name="app",
        ready=True,
        restart_count=2,
        image=_FAKE_IMAGE,
        state=_Obj(running=_Obj()),
        last_state=_Obj(terminated=_Obj(exit_code=137, reason="OOMKilled")),
    )
    return _Obj(
        metadata=_Obj(name=name, namespace=_FAKE_NAMESPACE, labels={"app": "example"}),
        spec=_Obj(node_name="node-1", containers=[_container()]),
        status=_Obj(
            phase="Running",
            pod_ip="10.0.0.1",
            container_statuses=[cs],
            conditions=[_Obj(type="Ready", status="True", reason=None)],
        ),
    )


def _fake_deployment(name: str = "example-deployment") -> _Obj:
    return _Obj(
        metadata=_Obj(name=name, namespace=_FAKE_NAMESPACE, labels={"app": "example"}, annotations={}),
        spec=_Obj(
            replicas=3,
            selector=_Obj(match_labels={"app": "example"}),
            strategy=_Obj(type="RollingUpdate"),
            template=_pod_template(),
        ),
        status=_Obj(ready_replicas=3, available_replicas=3, unavailable_replicas=0),
    )


def _fake_service(name: str = "example-service") -> _Obj:
    return _Obj(
        metadata=_Obj(name=name, namespace=_FAKE_NAMESPACE),
        spec=_Obj(
            type="ClusterIP",
            cluster_ip="10.0.1.1",
            external_i_ps=[],
            selector={"app": "example"},
            ports=[_Obj(name="http", port=80, target_port=8080, protocol="TCP")],
        ),
    )


def _fake_daemonset(name: str = "example-daemonset") -> _Obj:
    return _Obj(
        metadata=_Obj(name=name, namespace=_FAKE_NAMESPACE, labels={"app": "example"}),
        spec=_Obj(selector=_Obj(match_labels={"app": "example"}), template=_pod_template()),
        status=_Obj(
            desired_number_scheduled=1, current_number_scheduled=1,
            number_ready=1, number_available=1, number_unavailable=0,
            number_misscheduled=0, updated_number_scheduled=1,
        ),
    )


def _fake_statefulset(name: str = "example-statefulset") -> _Obj:
    return _Obj(
        metadata=_Obj(name=name, namespace=_FAKE_NAMESPACE, labels={"app": "example"}),
        spec=_Obj(
            replicas=1, service_name="example",
            selector=_Obj(match_labels={"app": "example"}),
            update_strategy=_Obj(type="RollingUpdate"),
            template=_pod_template(),
        ),
        status=_Obj(ready_replicas=1, current_replicas=1, updated_replicas=1),
    )


def _fake_job(name: str = "example-job") -> _Obj:
    return _Obj(
        metadata=_Obj(name=name, namespace=_FAKE_NAMESPACE, labels={"app": "example"}),
        spec=_Obj(parallelism=1, completions=1, backoff_limit=3, template=_pod_template()),
        status=_Obj(active=0, succeeded=1, failed=0),
    )


def _fake_cronjob(name: str = "example-cronjob") -> _Obj:
    return _Obj(
        metadata=_Obj(name=name, namespace=_FAKE_NAMESPACE, labels={"app": "example"}),
        spec=_Obj(
            schedule="*/5 * * * *", suspend=False,
            concurrency_policy="Allow",
            successful_jobs_history_limit=3, failed_jobs_history_limit=1,
            job_template=_Obj(spec=_Obj(template=_pod_template())),
        ),
        status=_Obj(),
    )


def _fake_role_binding(name: str = "example-binding") -> _Obj:
    return _Obj(
        metadata=_Obj(name=name, namespace=_FAKE_NAMESPACE),
        role_ref=_Obj(kind="Role", name="view"),
        subjects=[_Obj(kind="ServiceAccount", name="example-sa", namespace=_FAKE_NAMESPACE)],
    )


def _fake_event() -> _Obj:
    return _Obj(
        type="Warning", reason="BackOff",
        message="Back-off restarting failed container",
        involved_object=_Obj(kind="Pod", name="example-pod"),
        count=3, first_timestamp=None, last_timestamp=None,
    )


def _fake_list(items: list) -> _Obj:
    """Wrap items in an object exposing `.items` (matches V1*List shape)."""
    return _Obj(items=items)


class _FakeCoreV1:
    async def read_namespaced_pod(self, name, namespace, **_):  # noqa: ANN001
        return _fake_pod(name)

    async def list_namespaced_pod(self, namespace=None, **_):  # noqa: ANN001
        return _fake_list([_fake_pod()])

    async def list_pod_for_all_namespaces(self, **_):
        return _fake_list([_fake_pod()])

    async def read_namespaced_pod_log(self, name, namespace, **_):  # noqa: ANN001
        return "line 1\nline 2\nJob done\n"

    async def list_namespaced_event(self, namespace=None, **_):  # noqa: ANN001
        return _fake_list([_fake_event()])

    async def list_event_for_all_namespaces(self, **_):
        return _fake_list([_fake_event()])

    async def read_namespaced_service(self, name, namespace, **_):  # noqa: ANN001
        return _fake_service(name)

    async def list_namespaced_service(self, namespace=None, **_):  # noqa: ANN001
        return _fake_list([_fake_service()])

    async def list_service_for_all_namespaces(self, **_):
        return _fake_list([_fake_service()])


class _FakeAppsV1:
    async def read_namespaced_deployment(self, name, namespace, **_):  # noqa: ANN001
        return _fake_deployment(name)

    async def list_namespaced_deployment(self, namespace=None, **_):  # noqa: ANN001
        return _fake_list([_fake_deployment()])

    async def list_deployment_for_all_namespaces(self, **_):
        return _fake_list([_fake_deployment()])

    async def read_namespaced_daemon_set(self, name, namespace, **_):  # noqa: ANN001
        return _fake_daemonset(name)

    async def list_namespaced_daemon_set(self, namespace=None, **_):  # noqa: ANN001
        return _fake_list([_fake_daemonset()])

    async def list_daemon_set_for_all_namespaces(self, **_):
        return _fake_list([_fake_daemonset()])

    async def read_namespaced_stateful_set(self, name, namespace, **_):  # noqa: ANN001
        return _fake_statefulset(name)

    async def list_namespaced_stateful_set(self, namespace=None, **_):  # noqa: ANN001
        return _fake_list([_fake_statefulset()])

    async def list_stateful_set_for_all_namespaces(self, **_):
        return _fake_list([_fake_statefulset()])


class _FakeBatchV1:
    async def read_namespaced_job(self, name, namespace, **_):  # noqa: ANN001
        return _fake_job(name)

    async def list_namespaced_job(self, namespace=None, **_):  # noqa: ANN001
        return _fake_list([_fake_job()])

    async def read_namespaced_cron_job(self, name, namespace, **_):  # noqa: ANN001
        return _fake_cronjob(name)

    async def list_namespaced_cron_job(self, namespace=None, **_):  # noqa: ANN001
        return _fake_list([_fake_cronjob()])


class _FakeRbacV1:
    async def list_namespaced_role_binding(self, namespace=None, **_):  # noqa: ANN001
        return _fake_list([_fake_role_binding()])


class _FakeCustomObjects:
    async def get_namespaced_custom_object(self, group, version, namespace, plural, name, **_):  # noqa: ANN001
        return {
            "timestamp": "2026-01-01T00:00:00Z",
            "window": "30s",
            "containers": [{"name": "app", "usage": {"cpu": "12m", "memory": "23Mi"}}],
        }


class _FakeApiClient:
    """Offline stand-in for kubernetes_asyncio.client.ApiClient. Only the
    aclose() lifecycle hook is exercised by the module's teardown."""

    async def close(self) -> None:  # parity with the real ApiClient
        return None


def build_fake():
    """Return a FakeMCP exposing the Kubernetes tools wired to an offline
    per-resource fake API bundle. Needs NO kube-apiserver, kubeconfig, or
    network.

    Reuses the upstream-style hand-rolled `FakeApiClient` pattern (per
    research.md section 5): fake `core`/`apps`/`batch`/`rbac`/`custom`
    objects whose async methods return real `kubernetes_asyncio` V1 models,
    so `_summarize_*` reads them exactly as it would live objects.

    Data path (b): handlers reach module-internal state (`_clients`,
    `_resolve_cluster`), not the `client` arg -- so `client=None`. We
    install a DeploymentContext (so a cluster resolves) and pre-seed the
    per-cluster `_Connection`. The teardown restores the prior deployment
    context and module state.
    """
    _ensure_imports()  # need the real V1 models to build shape-faithful objects
    from opsrag.config import (
        EnvironmentsConfig,
        EnvironmentTarget,
        K8sTarget,
        OpsRAGConfig,
    )
    from opsrag.environments import bind_environments, reset_environments
    from opsrag.mcp._fake import FakeMCP

    # (i) bind a one-env `environments:` registry so `_resolve_cluster`
    # resolves the default env name -- the cache key + echoed `cluster`.
    # The kubeconfig target is never reached: the per-env connection is
    # pre-seeded below, so `_get_connection` short-circuits on the cache.
    _cfg = OpsRAGConfig()
    _cfg.environments = EnvironmentsConfig(
        default=_FAKE_CLUSTER,
        targets={
            _FAKE_CLUSTER: EnvironmentTarget(
                kubernetes=K8sTarget(mode="kubeconfig", context=_FAKE_CLUSTER),
            ),
        },
    )
    bind_environments(_cfg)

    # (ii) pre-seed the per-env connection + flip load flags so the offline
    # path never touches a kubeconfig / network.
    global _clients, _kubeconfig_loaded, _cluster_coords, _cluster_coords_loaded
    prev_clients = dict(_clients)
    prev_kubeconfig_loaded = _kubeconfig_loaded
    prev_cluster_coords = dict(_cluster_coords)
    prev_cluster_coords_loaded = _cluster_coords_loaded

    _clients[_FAKE_CLUSTER] = _Connection(
        api_client=_FakeApiClient(),
        core=_FakeCoreV1(),
        apps=_FakeAppsV1(),
        batch=_FakeBatchV1(),
        rbac=_FakeRbacV1(),
        custom=_FakeCustomObjects(),
    )
    # No real clusters registered, and pretend kubeconfig already loaded so
    # `_ensure_kubeconfig` returns immediately on the legacy path.
    _cluster_coords = {}
    _cluster_coords_loaded = True
    _kubeconfig_loaded = True

    def _restore() -> None:
        global _clients, _kubeconfig_loaded, _cluster_coords, _cluster_coords_loaded
        _clients = prev_clients
        _kubeconfig_loaded = prev_kubeconfig_loaded
        _cluster_coords = prev_cluster_coords
        _cluster_coords_loaded = prev_cluster_coords_loaded
        _api_access.clear()
        reset_environments()

    return FakeMCP(tools=list(KUBERNETES_TOOLS), client=None, teardown=_restore)
