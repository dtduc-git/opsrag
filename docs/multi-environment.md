# Multi-environment

One OpsRAG instance can target N environments (prod, staging, preprod, …), each with its own Kubernetes, Prometheus, and Elasticsearch reachability and schema. A single `environments:` registry maps an env name to an `EnvironmentTarget`; the k8s / prometheus / elasticsearch MCP tools take an `env` argument that selects which target to use.

The config models live in `opsrag/config.py`; the process-global resolver is `opsrag/environments.py`; the tools consume it in `opsrag/mcp/kubernetes.py`, `prometheus.py`, and `elasticsearch.py`.

## The registry

```yaml
environments:
  default: prod
  targets:
    prod: { ... }       # EnvironmentTarget
    staging: { ... }
```

`EnvironmentsConfig` (`opsrag/config.py`) holds a `default` env name and a `targets` map. Each `EnvironmentTarget` bundles three optional integrations:

```python
class EnvironmentTarget(BaseModel):
    kubernetes: K8sTarget | None = None
    prometheus: PrometheusTarget | None = None
    elasticsearch: EsTarget | None = None
```

Any integration may be `None` — that integration is simply unavailable for that env, and using it returns a clear, env-named error rather than a silent default. All four models use `extra="forbid"`, so a typo'd key fails validation at boot.

At startup `bind_environments(cfg)` (`opsrag/environments.py`) installs the registry as a process-global, computes the default env, and thereafter all lookups are pure. Lookup misses raise a structured `EnvironmentResolutionError` — never a silent fallback.

### Per-env Kubernetes (`K8sTarget`)

```python
class K8sTarget(BaseModel):
    mode: Literal["gke", "kubeconfig"] = "kubeconfig"
    # gke mode:
    project: str | None        # GCP project
    location: str | None       # region/zone
    name: str | None           # cluster name
    # kubeconfig mode (None -> current-context / in-cluster SA):
    context: str | None
    default_namespace: str | None
    pod_label_selector: str | None
```

- `mode: gke` reaches the cluster via Workload Identity (ADC) + the GCP Container API, addressed by `project` / `location` / `name`.
- `mode: kubeconfig` is the vendor-neutral path: a named KUBECONFIG `context` (EKS via `aws-eks-get-token`, client certs, etc.), or `None` to use the current context / the in-cluster ServiceAccount.

The k8s MCP exposes a shared `cluster_api_access(env)` helper that returns the host/token/CA for an env's API server. Prometheus (`k8s_proxy` mode) and Elasticsearch (`port_forward`/`proxy` modes) reuse it, so all three integrations authenticate the same way per env.

### Per-env Prometheus (`PrometheusTarget`)

```python
class PrometheusTarget(BaseModel):
    reach: Literal["k8s_proxy", "direct"] = "k8s_proxy"
    # k8s_proxy mode:
    namespace: str = "monitoring"
    service: str = "kube-prometheus-stack-prometheus"
    port: int = 9090
    extra_services: dict[str, str]    # e.g. {"istio": "..."}
    # direct mode:
    url: str | None
    bearer_token_env: str | None
```

- `reach: k8s_proxy` (default) reaches the in-cluster Prometheus service through the cluster's API-server service-proxy, riding the env's `kubernetes` target for auth. Note the default service name is the vendor-neutral `kube-prometheus-stack-prometheus` (not any org-specific name).
- `reach: direct` does an `httpx GET` against `url`, with an optional bearer token from `bearer_token_env`.

`extra_services` lets one cluster expose additional named Prometheis (e.g. an Istio one); the tool selects the `"istio"` service when asked, else the main `service`.

### Per-env Elasticsearch (`EsTarget`)

```python
class EsTarget(BaseModel):
    reach: Literal["direct", "port_forward", "proxy"] = "direct"
    url: str | None                 # direct mode
    service: str | None             # in-cluster (port_forward / proxy)
    namespace: str | None
    port: int = 9200
    pod_label_selector: str | None
    api_key_env: str | None         # auth: env-only, prefer API key
    username_env: str | None
    password_env: str | None
    index_pattern: str = "*"
    backend: Literal["elasticsearch", "opensearch"] = "elasticsearch"
    verify_ssl: bool = True
    fields: dict[str, str]          # logical -> physical field mapping
```

Three reach modes:

- `direct` — HTTPS to a reachable `url`.
- `port_forward` — tunnel the pod port over the k8s API for the env (ECK-style; the API proxy strips `Authorization`, so port-forward preserves API-key auth).
- `proxy` — through the k8s service-proxy.

Credentials are read from env vars at call time (never captured at bind), preferring an API key over basic auth.

The **`fields` map** de-hardcodes one org's schema. ES handlers translate logical field names to physical ones per env via `_map_field`, so generic schemas just work while a bespoke schema is configured, not coded:

```yaml
fields:
  timestamp: "@timestamp"
  service:   "kubernetes.labels.app_name"
```

A logical `service` filter on `elasticsearch_search` is mapped to `fields.service`, and recent-first sort uses `fields.timestamp` — both fall back to the logical name when unmapped.

## How a tool selects an environment

Every k8s / prometheus / elasticsearch tool accepts an optional `env` argument (back-compat alias: `cluster`). Resolution precedence is identical across the three modules:

1. explicit `env` argument
2. back-compat `cluster` alias
3. the per-integration default env (`OPSRAG_K8S_DEFAULT_CLUSTER` / `OPSRAG_PROMETHEUS_DEFAULT_CLUSTER` env var), then
4. the registry's `default`

```python
# opsrag/mcp/kubernetes.py
cluster = args.get("env") or args.get("cluster") or _default_cluster()
```

If nothing resolves (empty registry, no arg), the handler raises a clear "no environment specified and none configured — pass an `env` argument, or define the `environments:` block" error rather than guessing. The resolved string is the **env name**: it keys the per-env API-client cache and the resolver, and handlers echo it back (e.g. `result["cluster"]`) for caller continuity. The available env names surface to callers via `available_environments()` (which reads the bound registry), so an agent can discover the valid `env` values.

## Legacy back-compat synthesis

If no `environments:` block is set, `bind_environments` **synthesizes** a registry from the legacy `k8s` / `elasticsearch` / `deployment` blocks (`_synthesize_legacy`) so existing deployments keep working unchanged:

- Each `k8s.clusters` entry becomes a `gke` `K8sTarget` (project/location/name); each `deployment.kubernetes.clusters` entry becomes a `kubeconfig` `K8sTarget` (context).
- Every synthesized env gets a `k8s_proxy` `PrometheusTarget` pointed at the historical `monitoring-main-prometheus` service (with `monitoring-istio-prometheus` as the `istio` extra), preserving old behavior exactly.
- The legacy single global `elasticsearch` block becomes one `direct` `EsTarget` attached to every env.
- The default env is `k8s.default_cluster` (or the first synthesized name).

A `WARNING` is logged when synthesis happens, nudging operators to migrate to the explicit `environments:` block. When no env source exists at all, the tools report "no environment configured" until one is set.

## Validation at boot (`validate_enabled_mcps`)

`validate_enabled_mcps` (`opsrag/mcp/registry.py`) fails fast for every enabled MCP whose required config is unresolved. For the env-driven integrations a **custom validator** replaces the flat `required_env` / `required_config` checks because the requirement is an OR the tuples can't express: the integration is satisfied by *either* the `environments:` registry *or* legacy cluster config.

For example, Prometheus is considered reachable if any env declares a `prometheus` target, **or** any env declares a `kubernetes` target (a `k8s_proxy` Prometheus can ride any k8s target), **or** a legacy cluster source exists (`KUBECONFIG`, an in-cluster ServiceAccount, `k8s.clusters`, or `deployment.kubernetes.clusters`). Only the truly-empty case fails, with the canonical `MCP_MISCONFIGURED:<name>:<missing>` message naming exactly which config paths would satisfy it.

## Full example

```yaml
environments:
  default: prod
  targets:
    prod:
      kubernetes:
        mode: gke
        project: acme-prod
        location: us-central1
        name: acme-prod-gke
        default_namespace: payments
      prometheus:
        reach: k8s_proxy
        namespace: monitoring
        service: kube-prometheus-stack-prometheus
        port: 9090
        extra_services:
          istio: istio-prometheus
      elasticsearch:
        reach: direct
        url: https://es.prod.acme.internal:9200
        api_key_env: ES_PROD_API_KEY
        index_pattern: "logs-*"
        backend: elasticsearch
        verify_ssl: true
        fields:
          timestamp: "@timestamp"
          service: "kubernetes.labels.app_name"

    staging:
      kubernetes:
        mode: kubeconfig
        context: acme-staging
      prometheus:
        reach: direct
        url: https://prometheus.staging.acme.internal
        bearer_token_env: PROM_STAGING_TOKEN
      elasticsearch:
        reach: port_forward
        service: elasticsearch-es-http
        namespace: elastic-system
        port: 9200
        api_key_env: ES_STAGING_API_KEY
        index_pattern: "logs-*"
        backend: opensearch
        fields:
          timestamp: "@timestamp"
          service: "service.name"
```

With this config a tool call selects the env explicitly:

- `k8s_list_pods(env="staging", namespace="payments")` -> the `acme-staging` kubeconfig context.
- `prometheus_query(env="prod", query="up")` -> the in-cluster prod Prometheus via service-proxy.
- `elasticsearch_search(q="level:error", service="payments")` (no `env`) -> the `default` env (`prod`), with `service` mapped to `kubernetes.labels.app_name`.

Secrets are never in YAML — only the `*_env` keys name the env vars that hold them (`ES_PROD_API_KEY`, `PROM_STAGING_TOKEN`, …). See [./configuration.md](./configuration.md) for the env-var precedence rules.

## See also

- [./configuration.md](./configuration.md) — config precedence (env > YAML > bundle) and secrets handling.
- [./mcp-integrations.md](./mcp-integrations.md) — the k8s / prometheus / elasticsearch tools that take an `env` arg.
- [./investigations.md](./investigations.md) — how the investigation reasoner passes `env` to tools.
