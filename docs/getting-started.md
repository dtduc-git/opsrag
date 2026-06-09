# Getting Started

The fastest path from `git clone` to a running stack, an indexed corpus, a
cited answer, authenticated access, and a live investigation — all on your
laptop with the bundled `docker-compose` stack.

## Prerequisites

- Docker with Compose v2 (`docker compose version`)
- `curl` and `jq` (for the API walkthrough)
- One LLM API key. The default config uses Anthropic (`ANTHROPIC_API_KEY`).

You do **not** need Kubernetes, a cloud account, or an external identity
provider: the default config uses the on-device `fastembed` embedder, the
null knowledge-graph backend, zero MCP integrations, and a bundled local
OIDC issuer (Dex).

## 1. Clone and set your LLM key

```sh
git clone https://github.com/OWNER/opsrag.git
cd opsrag
cp .env.example .env
# Edit deploy/compose/.env (or the root .env you copied) and set:
#   ANTHROPIC_API_KEY=sk-ant-...
# Every other value can stay at its placeholder for the quickstart.
```

Secrets live only in `.env` / environment variables — never in `config.yaml`.
Every config key ending in `_env` (e.g. `api_key_env: ANTHROPIC_API_KEY`)
names the env var that carries the secret. See
[`configuration.md`](./configuration.md) for the full model.

## 2. Bring up the stack

```sh
docker compose -f deploy/compose/docker-compose.yaml up -d
```

This starts six services (see
[`deploy/compose/docker-compose.yaml`](../deploy/compose/docker-compose.yaml)):

| Service       | URL                          | Role                                   |
|---------------|------------------------------|----------------------------------------|
| `opsrag-api`  | <http://localhost:8080>      | FastAPI backend (HTTP + SSE)           |
| `opsrag-ui`   | <http://localhost:5173>      | React single-page UI                   |
| `qdrant`      | <http://localhost:6333>      | Vector store                           |
| `postgres`    | `localhost:5432`             | Sessions / memory / investigation ledger |
| `dex`         | <http://localhost:5556>      | Bundled local OIDC issuer              |
| `phoenix`     | <http://localhost:6006>      | LangGraph trace viewer                 |

A seventh service, `neo4j`, is defined but opt-in (the graph lane stays off
unless you set `knowledge_graph.provider: neo4j`).

The API container loads the demo config mounted at `/app/config.yaml`
([`deploy/compose/config.yaml`](../deploy/compose/config.yaml)), which boots
the Anthropic LLM, the `fastembed` embedder, Qdrant, Postgres sessions, and
all MCP integrations disabled.

### Verify health

```sh
curl -sf http://localhost:8080/healthz   # liveness — always 200 once up
curl -sf http://localhost:8080/readyz    # readiness — 200 once Postgres + Qdrant are reachable
```

`/healthz` and `/readyz` are the only unauthenticated endpoints (plus the
schema routes `/openapi.json`, `/docs`, `/redoc`).

## 3. First index

The bundled `samples/` directory holds a synthetic corpus (the fictional
"Acme Notes" product: runbooks, Helm values, manifests, a postmortem). Index
it with the seed script, which writes directly to Qdrant via the indexer —
no auth token required:

```sh
docker compose -f deploy/compose/docker-compose.yaml exec opsrag-api \
  scripts/seed-sample-corpus.sh
```

To index your own Git repositories instead, configure the `scm` block
(provider, `base_url`, `token_env`, `repos`) and run the indexer Job (the
local stand-in for the production Kubernetes Job):

```sh
docker compose -f deploy/compose/docker-compose.yaml --profile jobs \
  run --rm opsrag-indexer-job --repo your-org/your-repo --branch main
# or index every configured repo:
docker compose -f deploy/compose/docker-compose.yaml --profile jobs \
  run --rm opsrag-indexer-job --all
```

Indexing progress is written to durable Postgres job-state, so the API can
report it even after the Job exits. See
[`configuration.md`](./configuration.md) for the `scm` block and
[`deployment.md`](./deployment.md) for the production Job/CronJob model.

## 4. First query

The default demo config configures `auth` against the bundled Dex issuer, so
every endpoint except health needs a Bearer token. Mint one from Dex (static
evaluator user, resource-owner password grant — no browser needed):

```sh
TOKEN=$(curl -sf -X POST \
  -d 'grant_type=password' \
  -d 'username=evaluator@example.com' -d 'password=evaluator' \
  -d 'client_id=opsrag-local' -d 'client_secret=local-secret' \
  -d 'scope=openid profile email' \
  http://localhost:5556/dex/token | jq -r .access_token)

curl -sf -X POST http://localhost:8080/query \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"query":"How do I roll back an Acme Notes deployment?"}' | jq
```

You get a cited English answer drawn from the indexed `samples/` corpus,
plus a `session_id` and a `trace_id`. Open the Phoenix UI at
<http://localhost:6006> to inspect the full LangGraph trace, or the web UI at
<http://localhost:5173> (it runs its own OIDC handshake against Dex).

Requests without a valid token are rejected with a stable envelope:

```json
{"error": "unauthenticated", "reason": "missing_bearer", "request_id": "..."}
```

> **Local Dex gotcha.** Dex advertises its issuer as
> `http://localhost:5556/dex` (the browser-facing URL) while the API reaches
> it in-cluster at `http://dex:5556/dex`. The token `iss`, the Dex `issuer`,
> and `auth.issuer` must be byte-for-byte identical. If you hit an
> `issuer_mismatch`, align all three — see [`auth.md`](./auth.md).

## 5. Enabling auth and SSO

opsrag has three auth modes, selected by `auth.mode` in `config.yaml`
(`opsrag/config.py` → `AuthConfig`):

| Mode    | Behavior                                                                 |
|---------|--------------------------------------------------------------------------|
| `open`  | No enforcement (same as omitting the `auth` block). Local-dev only.      |
| `oidc`  | Verify incoming Bearer JWTs against `issuer` + `audience` (the demo path).|
| `login` | First-party login: cookie sessions + optional SSO providers.            |

**OIDC** (the quickstart default) points at any OIDC issuer — Dex, Keycloak,
Okta, Auth0, Azure AD / Entra:

```yaml
auth:
  mode: oidc
  issuer: https://your-idp.example.com   # OIDC discovery base URL (required)
  audience: opsrag                        # expected token "aud" (required)
  jwks_cache_seconds: 300
```

**Login** mode runs a first-party login page with password and/or SSO. The
session signing key is sourced from a path or env only (never inline):

```yaml
auth:
  mode: login
  login:
    password_enabled: true
    signing_key_env: OPSRAG_SESSION_SIGNING_KEY
    sso_callback_base: https://opsrag.example.com/api
  sso:
    google:    { enabled: true, client_id: "...", client_secret_env: OPSRAG_SSO_GOOGLE_SECRET }
    github:    { enabled: true, client_id: "...", client_secret_env: OPSRAG_SSO_GITHUB_SECRET }
    microsoft: { enabled: true, client_id: "...", client_secret_env: OPSRAG_SSO_MICROSOFT_SECRET,
                 server_metadata_url: "https://login.microsoftonline.com/<tenant>/v2.0/.well-known/openid-configuration" }
```

The three SSO providers are `google`, `github`, and `microsoft` (Azure AD /
Entra). Per-session ownership is enforced: a session is bound to its owner,
and cross-user access returns 404. Map IdP groups to opsrag roles with
`auth.role_mappings`. Full provider-by-provider setup (issuer/audience for
Keycloak, Okta, Auth0, Entra) is in [`auth.md`](./auth.md).

## 6. First investigation

The **Investigate** feature runs an event-driven, multi-lane root-cause
engine: three parallel lanes (runbook / historical / live) feed a Pro
hypothesizer, then a reasoner loop calls read-only MCP tools and a Flash
evaluator returns per-hypothesis verdicts (`confirmed` / `ruled_out` /
`untested` / `open`) with citations, under a hard budget (240s wall-clock,
40 tool-calls, 45s/tool).

It only surfaces when at least one **live-telemetry** MCP integration is
enabled (`opsrag/investigations/feature_gate.py`): `datadog`, `prometheus`,
`kubernetes`, `loki`, `grafana`, `splunk`, `sentry`, or `rootly`. Enable one
in `config.yaml` and supply its credentials via env:

```yaml
mcp:
  prometheus: { enabled: true }   # required env: see .env.example / docs/mcp-integrations.md
```

Enabling an MCP without its required env fails fast at startup with
`MCP_MISCONFIGURED:<name>:<env>`. Restart the API, then launch an
investigation (the surface is gated on the `investigate` scope — in `open`
mode every user holds it):

```sh
# Launch — returns immediately with an investigation_id
INV=$(curl -sf -X POST http://localhost:8080/investigations \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"alert_text":"checkout-api 5xx spiking in prod"}' | jq -r .investigation_id)

# Snapshot (lifecycle row + all events so far)
curl -sf http://localhost:8080/investigations/$INV \
  -H "Authorization: Bearer $TOKEN" | jq

# Resumable SSE event stream (reconnect with the last seen sequence)
curl -sN "http://localhost:8080/investigations/$INV/events?since=0" \
  -H "Authorization: Bearer $TOKEN"
```

The runner streams events to a Postgres ledger, so a tab refresh or network
blip never loses progress. In the web UI, the **Investigations** tab appears
once a telemetry MCP is enabled.

## Next steps

- [`configuration.md`](./configuration.md) — the full config model, every
  top-level block, env precedence, and `cloud_provider` bundles.
- [`auth.md`](./auth.md) — OIDC verification and per-provider SSO setup.
- [`mcp-integrations.md`](./mcp-integrations.md) — the 20 read-only MCP
  integrations and their required credentials.
- [`deployment.md`](./deployment.md) — production deployment, roles, and the
  indexing Job/CronJob model.
- [`multi-environment.md`](./multi-environment.md) — targeting many
  Kubernetes / Prometheus / Elasticsearch environments from one instance.

## See also

- [`architecture.md`](./architecture.md)
- [`helm-chart.md`](./helm-chart.md)
- [`config-example.yaml`](../config-example.yaml) — exhaustive annotated config reference
