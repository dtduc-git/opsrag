# OpsRAG multi-env: unified `environments:` registry (Approach A)

**Date:** 2026-06-09
**Status:** Approved (Section 1 schema approved by operator; sections 2–4 recorded here)
**Goal:** Replace the hardcoded / fragmented multi-environment wiring for kubernetes / prometheus / elasticsearch with a single, config-driven `environments:` registry. One OpsRAG instance can target N environments; each environment bundles how to reach its k8s + prometheus + ES. No org-specific names baked into code (Constitution Principle VI).

## Problem (grounded in current code)

Today there are **three overlapping representations** of "environment → target":
- `Settings.k8s.clusters: dict[str, K8sClusterCoords]` — GKE Workload-Identity coords (`config.py:705`, `421-444`).
- `DeploymentContext.kubernetes.clusters: dict[str, str]` — env → kubectl context name (`context.py:43-61`), read by `mcp/kubernetes.py:_known_clusters` (`:78-84`).
- `DeploymentContext.cloud.gcp_projects: dict[str, str]` — env → GCP project (`context.py:63-72`).

And two hard gaps:
- **Prometheus** hardcodes service/namespace/port: `DEFAULT_PROMETHEUS_SERVICE="monitoring-main-prometheus"`, `PROMETHEUS_NAMESPACE="monitoring"`, `PROMETHEUS_PORT=9090` (`mcp/prometheus.py:83-86`); reached via the k8s API proxy only.
- **Elasticsearch** is single-endpoint only (`ElasticsearchConfig`, `config.py:447-472`; `mcp/elasticsearch.py:_config` single `_BOUND`). No per-env.

The private cluster build (`internal-image-repo/images/opsrag/backend`) hardcodes `_KNOWN_CLUSTERS=("acme-prd","acme-pre","acme-stg","acme-ant")`, a canonical-env enum `["ant","stg","pre","prd"]`, the ECK ES schema (`time`, `kubernetes_metadata.labels_name`), etc.

## Section 1 — Config schema (APPROVED)

New top-level `environments:` block. Env names in `targets` are the canonical env list (derive `DeploymentContext.environments` from them — kills the hardcoded enum).

```yaml
environments:
  default: prod
  targets:
    prod:
      kubernetes:
        mode: gke                # gke (Workload Identity) | kubeconfig (vendor-neutral, EKS/any)
        project: my-prod
        location: us-east1
        name: my-prod-gke
        # mode: kubeconfig
        # context: my-prod-ctx
        default_namespace: null
        pod_label_selector: null
      prometheus:
        reach: k8s_proxy         # k8s_proxy (via cluster API) | direct (URL)
        namespace: monitoring
        service: kube-prometheus-stack-prometheus   # generic default, NOT monitoring-main-prometheus
        port: 9090
        extra_services: {}       # e.g. { istio: monitoring-istio-prometheus }
        # reach: direct
        # url: https://prom.prod.example.com
        # bearer_token_env: null
      elasticsearch:
        reach: direct            # direct (URL) | port_forward (k8s API) | proxy
        url: https://es.prod.example.com:9200
        api_key_env: OPSRAG_ES_PROD_API_KEY
        index_pattern: "app-logs-*"
        backend: elasticsearch
        verify_ssl: true
        fields:                  # logical -> ES field; de-hardcodes acme ECK schema
          timestamp: "@timestamp"
          service: "kubernetes.labels.app"
          stream: "stream"
        # reach: port_forward (in-cluster ECK):
        # service: eck-infra-logs-es-http
        # namespace: eck-infra
        # port: 9200
```

### Pydantic models (in `config.py`)
- `K8sTarget`: `mode: Literal["gke","kubeconfig"]="kubeconfig"`; gke → `project/location/name: str|None`; kubeconfig → `context: str|None`; `default_namespace: str|None`, `pod_label_selector: str|None`. Validator: gke requires project+location+name; kubeconfig requires nothing (context optional → current-context).
- `PrometheusTarget`: `reach: Literal["k8s_proxy","direct"]="k8s_proxy"`; k8s_proxy → `namespace:str="monitoring"`, `service:str="kube-prometheus-stack-prometheus"`, `port:int=9090`, `extra_services: dict[str,str]={}`; direct → `url:str|None`, `bearer_token_env:str|None`.
- `EsTarget`: `reach: Literal["direct","port_forward","proxy"]="direct"`; direct → `url:str|None`; in-cluster → `service/namespace:str|None`, `port:int=9200`; common → `api_key_env/username_env/password_env:str|None`, `index_pattern:str="*"`, `backend:Literal["elasticsearch","opensearch"]="elasticsearch"`, `verify_ssl:bool=True`, `fields: dict[str,str]={}`.
- `EnvironmentTarget`: `kubernetes: K8sTarget|None`, `prometheus: PrometheusTarget|None`, `elasticsearch: EsTarget|None`.
- `EnvironmentsConfig`: `default: str|None`, `targets: dict[str, EnvironmentTarget]={}`.
- `Settings.environments: EnvironmentsConfig` (new field, `config.py:~705`).

## Section 2 — Resolver + env-hint flow

New module `opsrag/environments.py` — a process-global bound registry (mirrors `prompt_render.active_deployment` / `registry_loader._active_enabled`):
- `bind_environments(cfg) -> None` — called once at startup. If `cfg.environments.targets` non-empty → use it. ELSE **legacy synthesis** (Section 3) from `cfg.k8s` + `cfg.elasticsearch` + `cfg.deployment` with a one-line deprecation log.
- `resolve_environment(name: str | None) -> EnvironmentTarget` — `name` or the bound default; on miss raise a structured error listing `available_environments()` (never substitute a default silently — matches the existing "lookup misses surface as structured errors" rule in `context.py`).
- `available_environments() -> list[str]`, `default_environment() -> str | None`.
- Default-gating: `None` (unbound) → empty registry → MCP tools return a clean "no environments configured" error.

**Env-hint flow:** alert/Rootly `Environment:` → an env name → the agent passes `env=<name>` to k8s/prometheus/es tools → each resolves `resolve_environment(env)` → one name drives all three consistently. Tool arg name unified to `env` (keep `cluster` as a back-compat alias on the k8s/prom tools).

## Section 3 — Per-MCP changes + migration

**`mcp/kubernetes.py` (shared dependency — done first):**
- `_resolve_cluster`/`_cluster_coords`/`_known_clusters`/`_default_cluster` → read `resolve_environment(env).kubernetes` + `available_environments()`/`default_environment()`.
- Support both `mode=gke` (existing GCP Container API path) and `mode=kubeconfig` (existing `_ensure_kubeconfig` path, select `context`).
- Expose a stable helper `async def cluster_api_access(env) -> {host, token, verify}` that prometheus + ES(port_forward) reuse — so they depend on the foundation, not on kubernetes internals.
- Keep `register_clusters(...)` as a thin back-compat shim that feeds the resolver (so legacy callers/tests don't break).

**`mcp/prometheus.py` (fan-out agent):**
- Delete the hardcoded `DEFAULT_PROMETHEUS_SERVICE/NAMESPACE/PORT` as REQUIRED values; keep them only as the model defaults. Read `resolve_environment(env).prometheus`.
- `reach=k8s_proxy` → build the proxy URL from `target.namespace/service/port` via `kubernetes.cluster_api_access(env)`; `istio` arg → `target.extra_services["istio"]`.
- `reach=direct` → hit `target.url` directly (optional bearer from `bearer_token_env`).

**`mcp/elasticsearch.py` (fan-out agent):**
- Replace single `_BOUND`/`_config()` with `resolve_environment(env).elasticsearch`. Handlers gain optional `env` arg (default = default env).
- `reach=direct` → `target.url`; `reach=port_forward`/`proxy` → via `kubernetes.cluster_api_access(env)` to `service.namespace:port`.
- Apply `target.fields` mapping (replace hardcoded `time`/`kubernetes_metadata.labels_name`); `index_pattern`, `backend`, `verify_ssl` from target.

**`api/server.py` (wiring — done after MCPs):**
- Call `bind_environments(cfg)` at startup (near `set_active_deployment`, `:1402`). Replace the `register_clusters(cfg.k8s.clusters)` + `es_mcp.bind(cfg.elasticsearch)` calls (`:245-261`) with the registry; keep them behind the legacy-synthesis path.

**Migration (backward-compatible):** new `environments:` wins. If absent: synthesize one env per `cfg.k8s.clusters` key (gke target) + merge `cfg.deployment.kubernetes.clusters` (kubeconfig target) + a single ES env from `cfg.elasticsearch` (under the ES default or `"default"`), prometheus default target per env. The demo (`config-local.yaml`, no `environments:` block, k8s/prometheus enabled but no cluster) keeps starting and tools keep erroring gracefully.

## Section 4 — Testing

- `tests/unit/test_environments_resolver.py`: bind from explicit registry; default selection; miss → structured error; legacy synthesis from `k8s.clusters`+`elasticsearch`; `available_environments()`.
- Per-MCP unit tests: target resolution drives the right host/service/url/fields; `env` arg + default; `reach` mode branches.
- Keep existing investigation + contract tests green; the new ES `env` arg must default so existing single-env callers don't break.

## Out of scope (this pass)
- Loki / Grafana / Splunk / Sentry multi-env (same pattern, later).
- The private `backend` convergence (separate repo).
