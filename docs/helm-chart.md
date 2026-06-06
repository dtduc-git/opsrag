# Helm Chart Reference

This is the long-form values reference and upgrade guide for the opsrag Helm
chart. The chart lives at `deploy/helm/opsrag/`. A shorter quick-start lives in
the chart-local `deploy/helm/opsrag/README.md`; this document is the
authoritative per-key reference and the place for operational notes (validation
behavior, what gets created, upgrade semantics, and the release test).

The chart deploys the opsrag API workload and, optionally, the UI and a Slack
bot worker, to a Kubernetes cluster (requires Kubernetes >= 1.25). All 14 MCP
integrations are present in the chart and disabled by default; each
`mcp.<name>.enabled` flag is wired onto the api container as the environment
variable `OPSRAG_MCP_<NAME>_ENABLED`.

## Contents

- [Install / upgrade / uninstall](#install--upgrade--uninstall)
- [Values reference](#values-reference)
- [MCP integrations](#mcp-integrations)
- [Supplying credentials](#supplying-credentials)
- [Schema validation](#schema-validation)
- [Resources created](#resources-created)
- [Upgrade notes](#upgrade-notes)
- [Release test](#release-test)

## Install / upgrade / uninstall

Install a release named `opsrag` into a dedicated namespace, overriding defaults
from your own values file:

```sh
helm install opsrag deploy/helm/opsrag \
  --namespace opsrag --create-namespace \
  -f my-values.yaml
```

A minimal `my-values.yaml` only needs the image, OIDC settings, and a Secret
holding runtime credentials:

```yaml
image:
  repository: ghcr.io/OWNER/opsrag
  tag: "0.1.0"
auth:
  issuer: https://your-idp.example.com
  audience: opsrag
api:
  envFromSecret: opsrag-secrets   # existing Secret with LLM key, POSTGRES_DSN, MCP creds
mcp:
  prometheus:
    enabled: true
```

Upgrade an existing release in place (re-applies templates with the merged
values):

```sh
helm upgrade opsrag deploy/helm/opsrag \
  --namespace opsrag \
  -f my-values.yaml
```

Render templates locally without contacting the cluster (useful for diffing and
for confirming the schema accepts your values):

```sh
helm template opsrag deploy/helm/opsrag -f my-values.yaml
```

Uninstall the release (removes all chart-created resources; it does not remove
the namespace, external Secrets you referenced, or any PersistentVolumes owned
by dependencies such as Postgres or Qdrant):

```sh
helm uninstall opsrag --namespace opsrag
```

## Values reference

Every top-level key in `values.yaml` is listed below with its default and
purpose. For object/array defaults, see `deploy/helm/opsrag/values.yaml` for the
exact shape; `values.schema.json` is the authoritative validation contract.

### Naming and image

| Key | Default | Purpose |
|---|---|---|
| `nameOverride` | `""` | Override the chart name component used in resource names and labels. |
| `fullnameOverride` | `""` | Override the full release name used as the base for all resource names. |
| `image.repository` | `ghcr.io/OWNER/opsrag` | API (and Slack bot) container image repository. Required. |
| `image.tag` | `"0.1.0"` | API image tag. Required. |
| `image.pullPolicy` | `IfNotPresent` | Image pull policy. One of `Always`, `IfNotPresent`, `Never`. |
| `imagePullSecrets` | `[]` | List of `imagePullSecrets` references for pulling private images. |

### auth (OIDC)

Rendered into the ConfigMap (`config.yaml`) for the API and also exposed to the
UI as `OPSRAG_OIDC_ISSUER`.

| Key | Default | Purpose |
|---|---|---|
| `auth.issuer` | `https://your-idp.example.com` | OIDC discovery base URL. Required; must be a real value. |
| `auth.audience` | `opsrag` | Expected token audience. Required. |
| `auth.jwksCacheSeconds` | `300` | How long to cache the IdP JWKS, in seconds. Minimum 1. |

### api

The primary workload (Deployment + Service).

| Key | Default | Purpose |
|---|---|---|
| `api.replicaCount` | `2` | API replica count. Ignored when `autoscaling.enabled=true`. Minimum 1. |
| `api.port` | `8080` | Container port the API listens on (also `OPSRAG_PORT`). |
| `api.resources` | requests `250m`/`512Mi`, limits `1`/`1Gi` | CPU/memory requests and limits for the api container. |
| `api.extraEnv` | `[]` | Extra `name`/`value` environment variables appended to the api container. |
| `api.envFromSecret` | `""` | Name of an existing Secret whose keys are injected as env vars (LLM keys, MCP credentials, `POSTGRES_DSN`). Empty = none. Also applied to the Slack bot container. |
| `api.service.type` | `ClusterIP` | Service type for the API Service. |
| `api.service.port` | `80` | Service port for the API Service (target port is the named `http` port). |

### ui

Opt-out React UI workload (Deployment + Service). Enabled by default.

| Key | Default | Purpose |
|---|---|---|
| `ui.enabled` | `true` | Deploy the UI Deployment and Service. |
| `ui.replicaCount` | `1` | UI replica count. |
| `ui.image.repository` | `ghcr.io/OWNER/opsrag-ui` | UI container image repository. |
| `ui.image.tag` | `"0.1.0"` | UI image tag. |
| `ui.image.pullPolicy` | `IfNotPresent` | UI image pull policy. |
| `ui.port` | `5173` | UI container port. |
| `ui.service.type` | `ClusterIP` | UI Service type. |
| `ui.service.port` | `80` | UI Service port. |
| `ui.resources` | requests `50m`/`128Mi`, limits `250m`/`256Mi` | CPU/memory requests and limits for the ui container. |

### slackBot

Opt-in Slack bot worker (Socket Mode; no Service is created for it).

| Key | Default | Purpose |
|---|---|---|
| `slackBot.enabled` | `false` | Deploy the Slack bot Deployment. |
| `slackBot.replicaCount` | `1` | Slack bot replica count. |
| `slackBot.resources` | requests `50m`/`128Mi`, limits `250m`/`256Mi` | CPU/memory requests and limits for the slackbot container. |

The Slack bot uses the same image as the API, mounts the same ConfigMap, and
receives the same `api.envFromSecret` Secret. Slack tokens are supplied via that
Secret, never inline.

### serviceAccount

| Key | Default | Purpose |
|---|---|---|
| `serviceAccount.create` | `true` | Create a ServiceAccount for the workloads. |
| `serviceAccount.name` | `""` | Name to use; defaults to the full release name when `create=true`, or `default` when `create=false`. |
| `serviceAccount.annotations` | `{}` | Annotations to add to the ServiceAccount (for example, cloud IAM bindings). |

### ingress

Opt-in Ingress for the API Service.

| Key | Default | Purpose |
|---|---|---|
| `ingress.enabled` | `false` | Create an Ingress for the API. |
| `ingress.className` | `""` | `ingressClassName` to set. Omitted when empty. |
| `ingress.annotations` | `{}` | Annotations for the Ingress. |
| `ingress.hosts` | one host `opsrag.example.com`, path `/` (Prefix) | List of host/path rules; each path routes to the API Service port. |
| `ingress.tls` | `[]` | TLS blocks passed through verbatim to the Ingress spec. |

### config

Rendered into a ConfigMap and mounted at `/etc/opsrag/config.yaml`
(`OPSRAG_CONFIG`). These mirror the application `config.yaml`.

| Key | Default | Purpose |
|---|---|---|
| `config.llm.provider` | `anthropic` | LLM provider name. |
| `config.llm.model` | `claude-sonnet-4-20250514` | LLM model identifier. |
| `config.llm.apiKeyEnv` | `ANTHROPIC_API_KEY` | Env var name the app reads the LLM API key from (the value comes from your Secret). |
| `config.embedding.provider` | `fastembed` | Embedding provider name. |
| `config.embedding.model` | `BAAI/bge-small-en-v1.5` | Embedding model identifier. |
| `config.vectorStore.provider` | `qdrant` | Vector store provider name. |
| `config.vectorStore.url` | `http://qdrant:6333` | Vector store endpoint URL. |
| `config.vectorStore.collection` | `opsrag` | Vector store collection name. |
| `config.knowledgeGraph.provider` | `none` | Knowledge graph provider; `none` disables the graph. |
| `config.session.provider` | `postgres` | Session store provider name. |
| `config.session.dsnEnv` | `POSTGRES_DSN` | Env var name the app reads the session DSN from (value from your Secret). |
| `config.observability.provider` | `console` | Observability provider name. |
| `config.observability.projectName` | `opsrag` | Project name reported to the observability backend. |

### mcp

The MCP integration map. The key set must equal the application's MCPIntegration
registry exactly (CI-enforced) and is fixed by the schema. Each entry:

| Key | Default | Purpose |
|---|---|---|
| `mcp.<name>.enabled` | `false` | Enable the integration. Rendered as `OPSRAG_MCP_<NAME>_ENABLED` on the api (and Slack bot) container and into the ConfigMap. |
| `mcp.<name>.secretRef` | `""` | Optional name of a Secret carrying that integration's credentials. |

The 14 integrations: `cartography`, `cloudflare`, `cloudsql`, `code`,
`datadog`, `elasticsearch`, `gitlab`, `knowledge`, `kubernetes`, `prometheus`,
`rootly`, `runbooks`, `slack`, `tool_cache`. See [MCP integrations](#mcp-integrations).

### secret

Opt-in chart-managed Secret. Operators may instead reference an existing Secret
via `api.envFromSecret`.

| Key | Default | Purpose |
|---|---|---|
| `secret.create` | `false` | Create a chart-managed Opaque Secret. |
| `secret.name` | `""` | Secret name; defaults to `<fullname>-secrets` when `create=true`. |
| `secret.data` | `{}` | Key/value pairs rendered as `stringData`. |

Note: creating this Secret does not by itself inject it into a container. To
consume it, set `api.envFromSecret` to the resulting Secret name.

### networkPolicy

Opt-in egress allowlist.

| Key | Default | Purpose |
|---|---|---|
| `networkPolicy.enabled` | `false` | Create a NetworkPolicy restricting egress. |
| `networkPolicy.egress` | `[]` | Extra egress rules appended to the built-in defaults (DNS on UDP/TCP 53 plus intra-namespace traffic). |

### podDisruptionBudget

Opt-in PDB for the API pods.

| Key | Default | Purpose |
|---|---|---|
| `podDisruptionBudget.enabled` | `false` | Create a PodDisruptionBudget for the API. |
| `podDisruptionBudget.minAvailable` | `1` | Minimum available API pods during voluntary disruption. |

### autoscaling

Opt-in Horizontal Pod Autoscaler for the API. When enabled, `api.replicaCount`
is not set on the Deployment (the HPA owns the replica count).

| Key | Default | Purpose |
|---|---|---|
| `autoscaling.enabled` | `false` | Create a HorizontalPodAutoscaler for the API. |
| `autoscaling.minReplicas` | `2` | Minimum replicas. |
| `autoscaling.maxReplicas` | `6` | Maximum replicas. |
| `autoscaling.targetCPUUtilizationPercentage` | `70` | Target average CPU utilization percentage. |

### Pod-level settings

Applied to the API, UI, and Slack bot pod/container specs.

| Key | Default | Purpose |
|---|---|---|
| `podAnnotations` | `{}` | Annotations added to the API pod template. |
| `podSecurityContext` | `runAsNonRoot: true`, `runAsUser: 1000`, `fsGroup: 1000` | Pod-level security context. |
| `securityContext` | `allowPrivilegeEscalation: false`, `readOnlyRootFilesystem: true`, `capabilities.drop: ["ALL"]` | Container-level security context. |
| `nodeSelector` | `{}` | Node selector for scheduling the API pods. |
| `tolerations` | `[]` | Tolerations for the API pods. |
| `affinity` | `{}` | Affinity rules for the API pods. |

## MCP integrations

Each integration is a key under `mcp`. Setting `mcp.<name>.enabled=true` does two
things:

1. It is rendered into the ConfigMap (`config.yaml`) under `mcp.<name>.enabled`.
2. It is rendered as an environment variable on the api container (and the Slack
   bot container) named `OPSRAG_MCP_<NAME>_ENABLED`, with the name uppercased.

For example, `mcp.prometheus.enabled=true` produces:

```yaml
- name: OPSRAG_MCP_PROMETHEUS_ENABLED
  value: "true"
```

and `mcp.tool_cache.enabled=true` produces `OPSRAG_MCP_TOOL_CACHE_ENABLED`. The
env var is always rendered for every integration (value `"true"` or `"false"`),
so the running pod sees the exact set of enable flags chosen by the operator.

With all integrations disabled (the default), the agent answers from the indexed
corpus only.

## Supplying credentials

Credentials are never placed in `values.yaml` inline. There are two supported
mechanisms:

- `api.envFromSecret`: name an existing Kubernetes Secret. All of its keys are
  injected as environment variables on the api container (and on the Slack bot
  container). This is the recommended way to supply the LLM API key
  (`ANTHROPIC_API_KEY` by default), the session DSN (`POSTGRES_DSN` by default),
  Slack tokens, and any MCP credentials.
- `mcp.<name>.secretRef`: name a Secret that carries a single integration's
  credentials. Use this when you prefer to scope credentials per integration.

You may also let the chart create a Secret for you with `secret.create=true` and
`secret.data`, then point `api.envFromSecret` at the resulting name (defaults to
`<fullname>-secrets`). Note that putting secret values in a values file commits
them to your release configuration; referencing an externally managed Secret is
generally preferred.

Enabling an MCP integration without supplying its required credentials makes the
pod fail fast at startup with an error of the form
`MCP_MISCONFIGURED:<name>:<env>`.

## Schema validation

The chart ships `values.schema.json` (JSON Schema draft 2020-12). Helm validates
your merged values against it at `helm install`, `helm upgrade`, and
`helm template` time. Key rules enforced by the schema:

- `additionalProperties: false` at the top level: any unknown top-level key
  fails the operation. A typo such as `imag:` (instead of `image:`) is rejected
  rather than silently ignored.
- Required top-level keys: `image`, `auth`, `api`, `mcp`.
- `image` requires non-empty `repository` and `tag`; `image.pullPolicy` is
  constrained to `Always`, `IfNotPresent`, or `Never`.
- `auth` requires non-empty `issuer` and `audience`; `jwksCacheSeconds` must be
  an integer >= 1.
- `mcp` has `additionalProperties: false` and requires all 14 integration keys
  to be present. Each integration object also has `additionalProperties: false`,
  requires `enabled`, and allows only `enabled` (boolean) and `secretRef`
  (string). Adding an unknown MCP integration name, or an unknown sub-key under
  an integration, fails validation.

Because of these rules, a malformed values file is rejected before any resource
is applied to the cluster.

## Resources created

Always created:

- API `Deployment` (`<fullname>`)
- API `Service` (`<fullname>`)
- `ConfigMap` (`<fullname>-config`) mounted at `/etc/opsrag/config.yaml`

Created by default but gated:

- `ServiceAccount` (`<fullname>`) when `serviceAccount.create=true` (default
  `true`)
- UI `Deployment` and `Service` (`<fullname>-ui`) when `ui.enabled=true`
  (default `true`)

Opt-in (off by default):

- Slack bot `Deployment` (`<fullname>-slackbot`) when `slackBot.enabled=true`
- `Ingress` (`<fullname>`) when `ingress.enabled=true`
- `Secret` when `secret.create=true`
- `NetworkPolicy` (`<fullname>`) when `networkPolicy.enabled=true`
- `PodDisruptionBudget` (`<fullname>`) when `podDisruptionBudget.enabled=true`
- `HorizontalPodAutoscaler` (`<fullname>`) when `autoscaling.enabled=true`

A Helm test Pod (`<fullname>-test-connection`) is defined as a `helm.sh/hook:
test` and is only created when you run `helm test` (see below).

## Upgrade notes

- Chart version vs appVersion. `Chart.yaml` carries two independent versions.
  `version` is the chart version; bump it whenever the chart templates, values,
  or schema change. `appVersion` tracks the opsrag application release and is
  shown in install notes and in the `app.kubernetes.io/version` label. They may
  differ: a chart-only fix bumps `version` while `appVersion` stays the same.
  The container image tag is controlled separately by `image.tag` (and
  `ui.image.tag`), so to roll out a new application build you typically bump
  `image.tag` rather than relying on `appVersion`.
- Upgrades run the same schema validation as installs, so a values change that
  introduces an unknown key is rejected before anything is applied.
- Switching `autoscaling.enabled` from `false` to `true` removes the static
  `replicas` field from the API Deployment and hands replica control to the HPA;
  switching it back restores `api.replicaCount`.
- Enabling a new MCP integration on upgrade requires that its credentials are
  available (via `api.envFromSecret` or `mcp.<name>.secretRef`) before the new
  pods roll out, or they will fail fast with `MCP_MISCONFIGURED:<name>:<env>`.

## Release test

The chart includes a Helm test that verifies the API is serving health checks.
After install or upgrade, run:

```sh
helm test opsrag --namespace opsrag
```

This launches a short-lived Pod that runs `curl -sf` against
`http://<fullname>:<api.service.port>/healthz` through the API Service. The test
passes when the endpoint returns a success status and fails otherwise, giving a
quick post-deploy smoke check of API reachability and health.
