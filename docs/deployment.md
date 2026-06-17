# Deployment Guide

How to run opsrag in three ways — a local `docker-compose` evaluation stack, a
Helm release on Kubernetes, and a hardened production deployment — with the
exact files, config keys, and trade-offs for each.

opsrag is a single container image driven by a Pydantic-validated `config.yaml`
plus environment variables for secrets. The same image runs every role
(`api`, `ui`, `slackbot`, `job-indexer`); the role is selected by the
`OPSRAG_ROLE` env var. Everything below is just different ways of wiring that
image to its dependencies (Qdrant, Postgres, an LLM provider, and your OIDC
issuer).

## Contents

- [1. Local docker-compose](#1-local-docker-compose)
- [2. Helm on Kubernetes](#2-helm-on-kubernetes)
- [3. Production hardening](#3-production-hardening)
  - [Secrets (envFromSecret)](#secrets-envfromsecret)
  - [Scaling and HA](#scaling-and-ha)
  - [Redis-backed rate limiting (multi-replica)](#redis-backed-rate-limiting-multi-replica)
  - [Network policy](#network-policy)
  - [Indexing: CronJob vs ephemeral Job](#indexing-cronjob-vs-ephemeral-job)
- [Scenario values files](#scenario-values-files)
- [See also](#see-also)

## 1. Local docker-compose

The compose stack at `deploy/compose/` brings up the whole system for local
evaluation: the API (`:8080`), the React UI (`:5173`), Qdrant (`:6333`),
Postgres (`:5432`), a bundled Dex OIDC issuer (`:5556`), and Phoenix for trace
viewing (`:6006`). It is the fastest way to see cited answers end to end.

```sh
# from the repo root
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY=...   (every other value can stay default)

# Bring up backend, UI, Qdrant, Postgres, Dex, Phoenix
docker compose -f deploy/compose/docker-compose.yaml up -d

# Verify health (readyz flips to 200 once Postgres + Qdrant are up)
curl -sf http://localhost:8080/healthz
curl -sf http://localhost:8080/readyz

# Index the bundled sample corpus (the fictional "Acme Notes" product)
docker compose -f deploy/compose/docker-compose.yaml exec opsrag-api \
  scripts/seed-sample-corpus.sh
```

The compose API mounts `deploy/compose/config.yaml` over the image's default at
`/app/config.yaml`. That demo profile runs **`login` mode** — auth is always
enforced; there is no anonymous / "open" mode. The API seeds a bootstrap admin
from `OPSRAG_ADMIN_EMAIL` / `OPSRAG_ADMIN_PASSWORD` and signs session cookies
with `OPSRAG_SESSION_SIGNING_KEY` (all defaulted in `docker-compose.yaml`), so a
fresh clone boots a working sign-in with no extra config. Change those env
defaults — and set `cookie_secure: true` behind HTTPS — before exposing it on a
network. To use an external IdP instead, switch `auth.mode` to `oidc` (the
bundled Dex issuer illustrates the path). The full walkthrough (sign-in or a Dex
bearer token, querying, inspecting the Phoenix trace) is in
[`../specs/001-port-opsrag-opensource/quickstart.md`](../specs/001-port-opsrag-opensource/quickstart.md).

Notable compose services and config:

- **Connection strings** to sibling services (`QDRANT_URL`, `POSTGRES_DSN`,
  `PHOENIX_COLLECTOR_ENDPOINT`) are injected via `environment:` in
  `docker-compose.yaml` and take precedence over same-named keys in `.env`.
- **Secrets** (your LLM key, MCP credentials) come from the git-ignored
  `deploy/compose/.env` loaded via `env_file:`.
- **Knowledge graph** is OFF by default. A `neo4j` service (with the APOC
  plugin) is included so you can test the graph lane locally: set
  `knowledge_graph.provider: neo4j` in `config.yaml` and point
  `knowledge_graph.url` at `bolt://neo4j:7687`.
- **Indexing** runs as a run-to-completion Job behind the `jobs` profile (it is
  not started by `up`), mirroring the production k8s Job:

  ```sh
  docker compose -f deploy/compose/docker-compose.yaml \
    run --rm opsrag-indexer-job --repo devops/foo --branch master
  # or a full reindex
  docker compose -f deploy/compose/docker-compose.yaml \
    run --rm opsrag-indexer-job --all
  ```

  Progress lands in the same durable Postgres job-state the API reads, so
  `/indexing/status` is consistent.

Tear down with `docker compose -f deploy/compose/docker-compose.yaml down`
(add `-v` to also drop the Qdrant/Postgres/Phoenix/Neo4j volumes).

## 2. Helm on Kubernetes

The first-class chart at `deploy/helm/opsrag/` deploys the API (Deployment +
Service), the UI, an indexing CronJob, and — opt-in — a Slack bot, Ingress,
NetworkPolicy, PDB, and HPA. It requires Kubernetes `>= 1.25`.

A minimal install needs only the image, the OIDC issuer/audience, and a Secret
carrying your runtime credentials:

```yaml
# my-values.yaml
image:
  repository: ghcr.io/dtduc-git/opsrag
  tag: "0.1.0"
auth:
  issuer: https://your-idp.example.com   # OIDC discovery base URL
  audience: opsrag
api:
  envFromSecret: opsrag-secrets           # existing Secret: LLM key, POSTGRES_DSN, MCP creds
mcp:
  prometheus:
    enabled: true                         # provide its creds via the secret above
```

```sh
helm install opsrag deploy/helm/opsrag \
  --namespace opsrag --create-namespace \
  -f my-values.yaml
```

The chart validates your merged values against `values.schema.json` at install,
upgrade, and `helm template` time — unknown top-level keys (a typo like `imag:`)
and unknown `mcp:` integration names are rejected before anything is applied.
Enabling an MCP integration without its required credentials makes the pod fail
fast at startup with `MCP_MISCONFIGURED:<name>:<env>`.

This guide covers deployment concerns; for the per-key values reference,
schema rules, and upgrade semantics see [`./helm-chart.md`](./helm-chart.md).

After install, smoke-test reachability:

```sh
helm test opsrag --namespace opsrag   # curls /healthz through the API Service
```

You provide the **data dependencies** (Qdrant and Postgres). The chart does not
deploy them — point `config.vectorStore.url` at your Qdrant and supply
`POSTGRES_DSN` in your Secret (default key, see `config.session.dsnEnv`). Run
them in-cluster (a StatefulSet/operator) or use managed services.

## 3. Production hardening

### Secrets (envFromSecret)

Never put secret values in `values.yaml`. Create a Kubernetes Secret
out-of-band and reference it with `api.envFromSecret`; every key in that Secret
is injected as an environment variable on the api container (and on the Slack
bot and indexing Job, which reuse the same Secret):

```sh
kubectl create secret generic opsrag-secrets --namespace opsrag \
  --from-literal=POSTGRES_DSN='postgres://opsrag:...@postgres:5432/opsrag' \
  --from-literal=ANTHROPIC_API_KEY='sk-ant-...' \
  --from-literal=OPSRAG_SESSION_SIGNING_KEY="$(openssl rand -hex 32)" \
  --from-literal=OPSRAG_ADMIN_PASSWORD='<bootstrap-admin>'
# add MCP credentials (GITLAB_TOKEN, DD_API_KEY, ...) and SSO client secrets as needed
```

```yaml
api:
  envFromSecret: opsrag-secrets
```

Two complementary mechanisms exist for credentials:

- `api.envFromSecret` — the recommended path: one existing Secret injected
  wholesale (LLM key, `POSTGRES_DSN`, Slack tokens, MCP creds, SSO secrets).
- `mcp.<name>.secretRef` — scope a single integration's credentials to a
  per-integration Secret instead.

You may instead let the chart create the Secret (`secret.create: true` +
`secret.data`), then point `api.envFromSecret` at the resulting name — but that
commits values to your release config, so an externally managed Secret is
preferred.

On the cloud overlays, the LLM/embedding provider credentials are *not* in a
Secret at all: they come from the workload identity bound to the
ServiceAccount (IRSA on EKS, Workload Identity on GKE — see
[Scenario values files](#scenario-values-files)). Only `POSTGRES_DSN`, the
session signing key, MCP tokens, and SSO secrets live in the Secret.

> **Auth mode note.** The chart renders `auth.issuer` / `auth.audience` /
> `auth.jwksCacheSeconds` into the ConfigMap but does **not** template
> `auth.mode` — so it defaults to `login` (auth is always enforced; there is no
> anonymous / "open" mode). To run `oidc`, or to configure first-party `login`
> SSO + `role_mappings`, supply your own `config.yaml` with the desired
> `auth.mode` and blocks. See [`./auth.md`](./auth.md) for both modes and SSO
> provider setup.

### Scaling and HA

The API is stateless (session/usage/memory state lives in Postgres, the vector
index in Qdrant), so it scales horizontally.

- **Static replicas:** `api.replicaCount` (default `2`). Ignored when
  autoscaling is enabled.
- **Autoscaling:** enable the HPA to let it own the replica count:

  ```yaml
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 6
    targetCPUUtilizationPercentage: 70
  ```

  When `autoscaling.enabled: true`, the chart omits the static `replicas` field
  from the Deployment and the HPA drives scaling on CPU.

- **PodDisruptionBudget:** keep a minimum available during voluntary
  disruptions (node drains, rollouts):

  ```yaml
  podDisruptionBudget:
    enabled: true
    minAvailable: 1
  ```

- **Spreading:** use `nodeSelector`, `tolerations`, and `affinity` (top-level
  pod settings applied to the API) to spread replicas across nodes/zones.

The chart ships sane defaults for production posture: a non-root pod
(`podSecurityContext.runAsNonRoot: true`, uid/gid `1000`), a hardened container
(`securityContext` with `readOnlyRootFilesystem: true`,
`allowPrivilegeEscalation: false`, all capabilities dropped), and
readiness/liveness probes on `/readyz` and `/healthz`.

### Redis-backed rate limiting (multi-replica)

opsrag rate-limits requests (the `RateLimitMiddleware`) and login attempts (the
`LoginRateLimiter`). The default backend is in-process **memory**, which is
correct for a single replica but **diverges across replicas** — each pod tracks
its own window, so the effective limit is `N x rate_limit_rpm` and login
lockouts are per-pod. For any multi-replica deployment, switch to the shared
**redis** backend.

This is an application `config.yaml` setting (`api.rate_limit_backend`), read
only from the config file — there is no env override for the backend selection.
Because the chart's ConfigMap templates a fixed subset of config keys and does
**not** currently render the `api:` block, you enable Redis by supplying your
own `config.yaml` (e.g. baked into your image, or mounted via your own
ConfigMap) containing:

```yaml
api:
  rate_limit_backend: redis     # default "memory"
  rate_limit_rpm: 60
  redis_url_env: OPSRAG_REDIS_URL   # env var the app reads the Redis URL from
```

Then make Redis reachable and required:

1. Build/run the image with the optional `redis` extra (the client is lazily
   imported; `pip install 'opsrag[redis]'` / `uv sync --extra redis`).
2. Supply the Redis URL via the env var named by `redis_url_env` — put
   `OPSRAG_REDIS_URL` in your `api.envFromSecret` Secret:

   ```sh
   kubectl create secret generic opsrag-secrets --namespace opsrag \
     --from-literal=OPSRAG_REDIS_URL='redis://opsrag-redis:6379/0' \
     # ...other keys
   ```

When `rate_limit_backend: redis`, **Redis is required**: the API pings Redis at
startup and fails fast if it is unset or unreachable
(`rate_limit_backend=redis requires a Redis URL ...`). This is intentional — a
silently-degraded limiter is worse than a refused boot. Run Redis in-cluster or
use a managed instance; remember to allow egress to it if you enable the
NetworkPolicy.

### Network policy

`networkPolicy.enabled: true` creates an egress allowlist for the API pods. The
built-in defaults permit DNS (UDP/TCP 53) and intra-namespace traffic (so the
pod reaches Postgres/Qdrant/Redis running in the same namespace). Append rules
for everything the pod must reach *outside* the namespace — your LLM provider
or Bedrock/Vertex endpoints, your OIDC issuer, and any MCP targets:

```yaml
networkPolicy:
  enabled: true
  egress:
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
      ports:
        - protocol: TCP
          port: 443        # LLM/MCP APIs, OIDC issuer over HTTPS
```

Tighten the `0.0.0.0/0` CIDR to your provider's ranges in regulated
environments.

### Indexing: CronJob vs ephemeral Job

Indexing does **not** run in a long-lived pod. The chart creates a single
indexing **CronJob** (`<fullname>-indexer`) that serves two purposes:

1. **Scheduled full reindex** — on `indexJob.schedule` (default `0 3 * * *`) it
   runs `job-indexer --all` to completion.
2. **Template for ad-hoc Jobs** — its `jobTemplate` is the byte-for-byte source
   the API clones when `POST /index/repo` triggers an on-demand index. The API
   only overrides the container args, so ad-hoc Jobs match the scheduled pod
   spec (image, config volume, Secret env, ServiceAccount, TTL).

```yaml
indexJob:
  enabled: true
  mode: auto              # auto | k8s | inprocess
  schedule: "0 3 * * *"
  suspendSchedule: false  # true = keep the template for ad-hoc use, no timer
  ttlSecondsAfterFinished: 3600
  rbac:
    create: true          # Role + RoleBinding so the API SA can clone the CronJob + manage Jobs
  resources:
    requests: { cpu: 500m, memory: 1Gi }
    limits:   { memory: 4Gi }
```

`indexJob.mode` controls how the API triggers indexing: `auto` spawns a k8s Job
when running in-cluster (else indexes in-process for dev), `k8s` always spawns a
Job (errors if RBAC/CronJob are missing), and `inprocess` keeps legacy
in-process indexing. The ad-hoc path needs the narrow RBAC the chart creates
when `indexJob.rbac.create: true` (read the template CronJob, create/manage Jobs
in this namespace only).

Keeping indexing off the serving pods means the API replicas stay pure-serving
and a heavy reindex can't steal their CPU/memory; because progress is tracked in
durable Postgres job-state, all replicas report a consistent
`/indexing/status`. Set `indexJob.suspendSchedule: true` if you want only
on-demand indexing with no nightly run.

## Scenario values files

The chart ships ready-made production overlays. Start from the closest one and
edit the placeholders (`<PROJECT_ID>`, `<ACCOUNT_ID>`, region, host, cert/IP
names, Qdrant/Postgres URLs):

| File | Scenario |
|---|---|
| [`deploy/helm/opsrag/values-aws.yaml`](../deploy/helm/opsrag/values-aws.yaml) | EKS + Amazon Bedrock. Sets `config.cloudProvider: aws` (Sonnet 4.6 reason/pro, Haiku 4.5 tools, Cohere Embed v4 = 1536-dim, Cohere Rerank 3.5), an ALB Ingress, autoscaling, a PDB, the index Job. Credentials via **IRSA** — no static AWS keys in the pod. |
| [`deploy/helm/opsrag/values-gcp.yaml`](../deploy/helm/opsrag/values-gcp.yaml) | GKE + Vertex AI. Sets `config.cloudProvider: gcp` (Gemini 2.5 Flash reason/tools, Gemini 2.5 Pro escalation, `gemini-embedding-001` = 3072-dim, `semantic-ranker-default-004`), a GKE managed Ingress, autoscaling, a PDB, the index Job. Credentials via **Workload Identity** — no JSON keys in the pod. |

Apply an overlay, overriding only the image:

```sh
helm upgrade --install opsrag deploy/helm/opsrag \
  --namespace opsrag --create-namespace \
  -f deploy/helm/opsrag/values-aws.yaml \
  --set image.repository=<your-ecr>/opsrag --set image.tag=<tag>
```

For other scenarios there are ready-made values files in `deploy/helm/opsrag/`:

- **`values-minimal.yaml`** — the smallest real install (Anthropic + fastembed +
  Qdrant, no MCP, 1 api replica).
- **`values-mcp.yaml`** — a curated set of live-telemetry MCP integrations
  (datadog/rootly/kubernetes/prometheus/code/knowledge) so the Investigate
  feature lights up, plus a single-env `environments` registry.
- **`values-multi-env.yaml`** — the `environments` registry across prod/staging/dev
  (per-env kubernetes / prometheus / elasticsearch targets).

MCP toggles are `mcp.<name>.enabled` (creds via the shared Secret or
`mcp.<name>.secretRef`); the multi-environment registry **is** templated via
`config.environments`. See [`./helm-chart.md`](./helm-chart.md) for the full
values reference, [`./multi-environment.md`](./multi-environment.md) for the
registry model, and [`./mcp-integrations.md`](./mcp-integrations.md) for
per-integration credentials.

> **Embedding dimension guard.** `config.embedding.dimension` MUST match both
> the embedding model and the Qdrant collection. Switching embedding models
> against an existing collection requires an intentional reindex with
> `config.vectorStore.allowDimensionChange: true` — otherwise a silent
> dimension mismatch corrupts the shared Qdrant index and caches.

## See also

- [`./helm-chart.md`](./helm-chart.md) — full chart values reference, schema, and upgrade notes
- [`./mcp-integrations.md`](./mcp-integrations.md) — the 20 MCP integrations and their credentials
- [`./auth.md`](./auth.md) — the two auth modes (`login` default / `oidc`) and SSO providers
- [`./architecture.md`](./architecture.md) — system architecture and the indexing/serving split
- [`../specs/001-port-opsrag-opensource/quickstart.md`](../specs/001-port-opsrag-opensource/quickstart.md) — full local walkthrough (sign-in or token, then query)
