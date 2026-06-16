# Multi-cluster / multi-environment

A single OpsRAG instance can target many environments at once -- prod, staging,
preprod, dr, an EU region, and so on -- each with its own way of reaching its
Kubernetes API, its Prometheus, and its Elasticsearch/OpenSearch, and even its
own log-index schema. One `environments:` registry maps an environment name to
an `EnvironmentTarget`; at query time the agent (or a human) selects which
target a tool call runs against via an `env` argument.

This guide covers what "multi-cluster" means in OpsRAG, which integrations
honour it, a complete copy-pasteable two-cluster config, how an environment is
selected at query time, the `OPSRAG_*_DEFAULT_CLUSTER` overrides, the legacy
back-compat synthesis, and how to troubleshoot `EnvironmentResolutionError`.

The models live in `opsrag/config.py`; the process-global resolver is
`opsrag/environments.py`; the tools that consume it are
`opsrag/mcp/kubernetes.py`, `opsrag/mcp/prometheus.py`, and
`opsrag/mcp/elasticsearch.py`.

## What multi-cluster means here, and which integrations support it

"Multi-cluster" in OpsRAG means one running instance can be pointed at N
independent clusters/environments, and each *cluster-coupled* tool call carries
an `env` (or back-compat `cluster`) argument that selects which environment it
runs against. The selection is per call -- one investigation can pull pod
status from `prod` and compare a metric against `staging` in the same turn.

Support by integration:

| Integration              | Multi-cluster?              | How it is selected                                              |
| ------------------------ | --------------------------- | -------------------------------------------------------------- |
| Kubernetes               | YES                         | `env` / `cluster` arg -> per-env `K8sTarget`                    |
| Prometheus               | YES                         | `env` / `cluster` arg -> per-env `PrometheusTarget`            |
| Elasticsearch/OpenSearch | YES                         | `env` / `cluster` arg -> per-env `EsTarget`                     |
| aws / cloudwatch         | Multi-REGION (not per-env)  | per-call `region` arg (defaults to `AWS_REGION`)                |
| gcp / stackdriver        | Single (project via config) | not env-scoped                                                  |
| All other connectors     | Single                      | one endpoint/credential set per connector                      |

Only `kubernetes`, `prometheus`, and `elasticsearch` participate in the
`environments:` registry. The AWS-family connectors (`aws`, `cloudwatch`) are
*multi-region* -- they are not env-scoped, but each tool call accepts an
optional `region` argument and otherwise falls back to `AWS_REGION` /
`AWS_DEFAULT_REGION`. Every other connector (`gcp`, `stackdriver`,
`pagerduty`, GitHub, Sentry, Grafana, Loki, Splunk, etc.) targets a single
endpoint configured once for the instance.

## The registry

```yaml
environments:
  default: prod
  targets:
    prod: { ... }       # EnvironmentTarget
    staging: { ... }
```

`EnvironmentsConfig` (`opsrag/config.py`) holds:

- `default: str | None` -- the environment used when a tool call omits `env`.
  When unset, the first target in `targets` is used.
- `targets: dict[str, EnvironmentTarget]` -- the canonical environment list.
  The keys are the environment names callers pass as `env`.

Each `EnvironmentTarget` bundles three optional integrations:

```python
class EnvironmentTarget(BaseModel):
    kubernetes: K8sTarget | None = None
    prometheus: PrometheusTarget | None = None
    elasticsearch: EsTarget | None = None
```

Any integration may be `None` -- that integration is simply unavailable for
that environment, and using it returns a clear, env-named error rather than a
silent default (e.g. "prometheus not configured for env 'staging'"). All of
`EnvironmentsConfig`, `EnvironmentTarget`, `K8sTarget`, `PrometheusTarget`, and
`EsTarget` use `extra="forbid"`, so a mistyped key fails validation at boot
rather than being silently ignored.

At startup `bind_environments(cfg)` (`opsrag/environments.py`) installs the
registry as a process-global, computes the default environment, and thereafter
all lookups are pure. Lookup misses raise a structured
`EnvironmentResolutionError` -- never a silent fallback.

### Per-env Kubernetes (`K8sTarget`)

```python
class K8sTarget(BaseModel):
    mode: Literal["gke", "kubeconfig"] = "kubeconfig"
    # gke mode:
    project: str | None = None        # GCP project
    location: str | None = None       # region/zone
    name: str | None = None           # cluster name
    # kubeconfig mode (None -> current-context / in-cluster SA):
    context: str | None = None
    # shared:
    default_namespace: str | None = None
    pod_label_selector: str | None = None
```

- `mode: gke` reaches the cluster via Workload Identity (ADC) + the GCP
  Container API, addressed by `project` / `location` / `name`.
- `mode: kubeconfig` (the default) is the vendor-neutral path: a named
  KUBECONFIG `context` (EKS via `aws eks get-token`, client certs, exec
  plugins, etc.), or `context: null` to use the current context / the
  in-cluster ServiceAccount.

The Kubernetes MCP exposes a shared `cluster_api_access(env)` helper that
returns the `{host, token, verify}` for an environment's API server. Prometheus
(`reach: k8s_proxy`) and Elasticsearch (`reach: port_forward` / `proxy`) reuse
it, so all three integrations authenticate the same way per environment.

### Per-env Prometheus (`PrometheusTarget`)

```python
class PrometheusTarget(BaseModel):
    reach: Literal["k8s_proxy", "direct"] = "k8s_proxy"
    # k8s_proxy mode:
    namespace: str = "monitoring"
    service: str = "kube-prometheus-stack-prometheus"
    port: int = 9090
    extra_services: dict[str, str] = {}    # e.g. {"istio": "..."}
    # direct mode:
    url: str | None = None
    bearer_token_env: str | None = None
```

- `reach: k8s_proxy` (the default) reaches the in-cluster Prometheus service
  through the cluster API server's service-proxy, riding the environment's
  `kubernetes` target for auth. The default service name is the vendor-neutral
  `kube-prometheus-stack-prometheus` (not any org-specific name). The 401 path
  refreshes the ADC token and retries once.
- `reach: direct` does an `httpx GET` against `url`, with an optional bearer
  token read at call time from the env var named by `bearer_token_env`.

`extra_services` lets one cluster expose additional named Prometheis (e.g. an
Istio one); a tool call with `istio: true` selects `extra_services["istio"]`,
and falls back to the main `service` when no istio service is configured.

### Per-env Elasticsearch / OpenSearch (`EsTarget`)

```python
class EsTarget(BaseModel):
    reach: Literal["direct", "port_forward", "proxy"] = "direct"
    url: str | None = None                 # direct mode
    service: str | None = None             # in-cluster (port_forward / proxy)
    namespace: str | None = None
    port: int = 9200
    pod_label_selector: str | None = None
    api_key_env: str | None = None         # auth: env-only, prefer API key
    username_env: str | None = None
    password_env: str | None = None
    index_pattern: str = "*"
    backend: Literal["elasticsearch", "opensearch"] = "elasticsearch"
    verify_ssl: bool = True
    fields: dict[str, str] = {}            # logical -> physical field mapping
```

Three reach modes:

- `direct` -- HTTPS to a reachable `url`.
- `port_forward` -- tunnel the pod port over the k8s API for the environment
  (ECK-style; the API proxy strips `Authorization`, so port-forward preserves
  the ES API-key auth instead of clobbering it with the cluster bearer).
- `proxy` -- through the k8s service-proxy; when no ES credentials are set, the
  cluster bearer token authenticates.

Credentials are read from env vars at call time (never captured at bind),
preferring an API key (`api_key_env`) over basic auth
(`username_env` / `password_env`).

The `fields` map de-hardcodes one org's log schema. ES handlers translate
logical field names to physical ones per environment via `_map_field`, so
generic schemas just work while a bespoke schema is configured, not coded:

```yaml
fields:
  timestamp: "@timestamp"
  service:   "kubernetes.labels.app_name"
```

A logical `service` filter on `elasticsearch_search` is mapped to
`fields.service`, and recent-first sort uses `fields.timestamp` -- both fall
back to the logical name when unmapped, and `index_pattern` supplies the
default index for a search.

## How a tool selects an environment at query time

Every Kubernetes / Prometheus / Elasticsearch tool accepts an optional `env`
argument, with `cluster` retained as a back-compat alias. The resolution
precedence is identical across the three modules:

1. explicit `env` argument
2. back-compat `cluster` alias
3. the per-integration default-cluster env var
   (`OPSRAG_K8S_DEFAULT_CLUSTER` / `OPSRAG_PROMETHEUS_DEFAULT_CLUSTER`)
4. the registry's `default`

```python
# opsrag/mcp/kubernetes.py
cluster = args.get("env") or args.get("cluster") or _default_cluster()
```

Note the precedence carefully: the `OPSRAG_*_DEFAULT_CLUSTER` env var is only
consulted when neither `env` nor `cluster` is passed -- it changes the
*default*, it never overrides an explicit argument.

If nothing resolves (empty registry and no arg), the handler raises a clear
"no environment specified and none configured -- pass an `env` argument, or
define the `environments:` block" error rather than guessing. The resolved
string is the environment *name*: it keys the per-env API-client cache and the
resolver (`resolve_environment(env)`), and handlers echo it back to the caller
(e.g. `result["cluster"]` for k8s/prometheus, `result["env"]` for
elasticsearch) for continuity. The available environment names surface to
callers via `available_environments()` (which reads the bound registry), so the
agent can discover the valid `env` values.

An agent (or a human via the chat UI) selects a cluster simply by passing the
argument:

- `k8s_list_pods(env="staging", namespace="payments")` -> the `staging`
  environment's `K8sTarget`.
- `prometheus_query(env="prod", query="up")` -> the in-cluster `prod`
  Prometheus via service-proxy.
- `prometheus_query(env="prod", query="...", istio=true)` -> the `prod`
  environment's `extra_services["istio"]` Prometheus.
- `elasticsearch_search(q="level:error", service="payments")` (no `env`) ->
  the `default` environment, with `service` mapped through that env's `fields`.

## The OPSRAG_*_DEFAULT_CLUSTER overrides

Two env vars override the registry's `default` for callers that pass no `env`:

- `OPSRAG_K8S_DEFAULT_CLUSTER` -- the default environment for Kubernetes tool
  calls (`opsrag/mcp/kubernetes.py`, `_default_cluster()`).
- `OPSRAG_PROMETHEUS_DEFAULT_CLUSTER` -- the default environment for Prometheus
  tool calls (`opsrag/mcp/prometheus.py`, `_default_prometheus_cluster()`).

These are useful to repoint a default at deploy time without editing YAML --
e.g. a per-replica or per-namespace deployment that should default to a
particular cluster. They do not affect calls that pass an explicit `env` /
`cluster`. Elasticsearch has no dedicated default-cluster env var; when no
`env`/`cluster` is passed it uses the registry `default` directly.

## Legacy back-compat synthesis

If no `environments:` block is set (an empty `targets`), `bind_environments`
*synthesizes* a registry from the legacy `k8s` / `elasticsearch` / `deployment`
blocks (`_synthesize_legacy` in `opsrag/environments.py`) so existing
deployments keep working unchanged:

- Each `k8s.clusters` entry (a `K8sClusterCoords` of project/location/name)
  becomes a `gke` `K8sTarget`.
- Each `deployment.kubernetes.clusters` entry (env-name -> kubeconfig context)
  becomes a `kubeconfig` `K8sTarget`.
- Every synthesized environment gets a `k8s_proxy` `PrometheusTarget` pointed
  at the historical service `monitoring-main-prometheus` (with
  `monitoring-istio-prometheus` as the `istio` extra), preserving the old
  behaviour exactly. Note this differs from the NEW `PrometheusTarget`
  default service `kube-prometheus-stack-prometheus` -- synthesis intentionally
  keeps the old constant so legacy deployments behave identically.
- The legacy single global `elasticsearch` block (when `enabled`) becomes one
  `direct` `EsTarget` attached to every synthesized environment. If there are
  no clusters at all but ES is enabled, a single environment named `default`
  is synthesized carrying only that `EsTarget`.
- The default environment is `k8s.default_cluster` (or the first synthesized
  name). An explicit `environments.default` still wins if set.

A `WARNING` is logged when synthesis happens, nudging operators to migrate to
the explicit `environments:` block. When no environment source exists at all,
the tools report "no environment configured" until one is set.

## Full example: two clusters (prod via GKE + staging via kubeconfig)

A complete, copy-pasteable registry. `prod` is a GKE cluster reached by
Workload Identity, with an in-cluster Prometheus via service-proxy and a
directly-reachable Elasticsearch. `staging` is reached by a KUBECONFIG context,
with a directly-reachable Prometheus (bearer token) and an ECK Elasticsearch
tunnelled through the cluster API (`port_forward`). No secrets appear in YAML --
only the `*_env` keys name the env vars that hold them.

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
        default_namespace: payments
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
        verify_ssl: true
        fields:
          timestamp: "@timestamp"
          service: "service.name"
```

With this config a tool call selects the environment explicitly:

- `k8s_list_pods(env="staging", namespace="payments")` -> the `acme-staging`
  kubeconfig context.
- `prometheus_query(env="prod", query="up")` -> the in-cluster prod Prometheus
  via service-proxy.
- `elasticsearch_search(q="level:error", service="payments")` (no `env`) ->
  the `default` environment (`prod`), with `service` mapped to
  `kubernetes.labels.app_name`.
- `elasticsearch_search(env="staging", q="level:error", service="payments")`
  -> the `staging` OpenSearch via `port_forward`, with `service` mapped to
  `service.name`.

The env vars referenced above (`ES_PROD_API_KEY`, `PROM_STAGING_TOKEN`,
`ES_STAGING_API_KEY`) must be present in the process environment. See
[./configuration.md](./configuration.md) for the env-var precedence rules and
secrets handling.

## Troubleshooting (EnvironmentResolutionError)

`EnvironmentResolutionError` (`opsrag/environments.py`) is raised when an
environment name cannot be resolved. The common cases:

- **`no environments configured. Set the environments: block ...`** -- the
  registry is empty (no `environments.targets` and no legacy
  `k8s.clusters` / `elasticsearch` to synthesize from). Define an
  `environments:` block, or set legacy `k8s.clusters` /
  `deployment.kubernetes.clusters` / enable `elasticsearch`. Check the boot
  log: synthesis logs a `WARNING` naming how many environments it built, and
  an empty source logs "no environments configured".

- **`unknown environment 'X'. Configured: [...]`** -- a tool was called with
  `env="X"` (or `cluster="X"`) but `X` is not a key in `environments.targets`.
  The error lists the configured names; call `available_environments()` or
  inspect the registry. Note that `_resolve_cluster` only picks the *name*;
  an unknown name surfaces as `EnvironmentResolutionError` later, when the
  connection layer resolves it.

- **`no environment specified and none configured` (RuntimeError)** -- a
  handler was called with no `env`/`cluster` arg AND no default could be
  resolved (empty registry, no `OPSRAG_*_DEFAULT_CLUSTER`). Pass an `env`
  argument or configure a `default`.

- **`prometheus not configured for env 'X'`** /
  **`elasticsearch not configured for env 'X'`** -- the environment exists but
  that specific integration's target is `None`. Add the missing
  `prometheus:` / `elasticsearch:` block under `environments.targets.X`. This
  is intentional: a `None` integration fails with an env-named message instead
  of silently falling back to another environment.

- **`reach=direct but no url set`** / **`port_forward requires service and
  namespace`** -- the integration target is present but incompletely
  configured for its reach mode. For `direct` provide `url`; for
  `port_forward` / `proxy` provide both `service` and `namespace`.

- **`extra=forbid` validation error at boot** -- a mistyped key under any of
  `environments`, `*Target`. Because all the env models forbid extra keys,
  a typo (e.g. `servce:` instead of `service:`) fails fast at config load with
  a Pydantic error naming the offending field. Fix the key name.

For per-integration validation behaviour at boot (whether an MCP is considered
reachable given the registry vs. legacy config), see `validate_enabled_mcps`
in `opsrag/mcp/registry.py`: an env-driven integration is satisfied by *either*
the `environments:` registry *or* a legacy cluster source, and only the
truly-empty case fails with `MCP_MISCONFIGURED:<name>:<missing>`.

## See also

- [./configuration.md](./configuration.md) -- config precedence (env > YAML >
  bundle) and secrets handling.
- [./mcp-integrations.md](./mcp-integrations.md) -- the connectors (and their
  `category`) including the k8s / prometheus / elasticsearch tools that take an
  `env` arg.
- [./investigations.md](./investigations.md) -- how the investigation reasoner
  passes `env` to tools.
</content>
</invoke>
