# opsrag

> Agentic GraphRAG for DevOps and SRE — query your runbooks, Terraform
> modules, Helm charts, Kubernetes manifests, and incident postmortems.
> Every answer cited, every source linked.

opsrag is an open-source, vendor-neutral, Apache-2.0-licensed agentic
retrieval system designed for SRE workflows. It bundles a LangGraph agent,
a pluggable retrieval pipeline (vector + optional knowledge graph + 14
opt-in MCP integrations), a FastAPI surface with OIDC auth, a React UI, a
Slack bot, an evaluation harness, a Helm chart, and a local compose stack.

A new evaluator should reach a cited answer in **under fifteen minutes**.

## Status

> :warning: **Pre-release.** The first public release is being assembled
> on the `001-port-opsrag-opensource` branch. Tasks and design artefacts
> live under [`specs/001-port-opsrag-opensource/`](specs/001-port-opsrag-opensource/).
> Interfaces and configuration keys may change without notice until
> `v0.1.0-alpha`.

## Quickstart

A new evaluator can go from clone to a cited answer in under fifteen minutes.
The full reference is
[`specs/001-port-opsrag-opensource/quickstart.md`](specs/001-port-opsrag-opensource/quickstart.md);
this is the same flow, end to end.

**Prerequisites:** Docker (Compose v2), `curl`, `jq`, and one LLM API key
(default: Anthropic). You do **not** need Kubernetes, an external OIDC
provider (Dex is bundled), or any cloud account (the default config uses the
null graph backend and zero MCP integrations).

```sh
# 1. Clone and set your LLM key
git clone https://github.com/OWNER/opsrag.git
cd opsrag
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY=...   (every other value can stay default)

# 2. Bring up backend (:8080), UI (:5173), Qdrant, Postgres, Dex (OIDC :5556), Phoenix (:6006)
docker compose -f deploy/compose/docker-compose.yaml up -d

# 3. Verify health (readiness flips to 200 once Postgres + Qdrant are up)
curl -sf http://localhost:8080/healthz
curl -sf http://localhost:8080/readyz

# 4. Index the bundled sample corpus (the fictional "Acme Notes" product)
docker compose -f deploy/compose/docker-compose.yaml exec opsrag-api \
  scripts/seed-sample-corpus.sh

# 5. Get an OIDC token from the bundled Dex (static evaluator user)
TOKEN=$(curl -sf -X POST \
  -d 'grant_type=password' \
  -d 'username=evaluator@example.com' -d 'password=evaluator' \
  -d 'client_id=opsrag-local' -d 'client_secret=local-secret' \
  -d 'scope=openid profile email' \
  http://localhost:5556/dex/token | jq -r .access_token)

# 6. Ask a question — every endpoint except /healthz and /readyz needs the Bearer token
curl -sf -X POST http://localhost:8080/query \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"query":"How do I roll back an Acme Notes deployment?"}' | jq
```

You get back a cited English answer drawn from the indexed `samples/` corpus
(e.g. the Acme Notes deploy/rollback runbook), plus a `session_id` and a
`trace_id`. Open the Phoenix UI at <http://localhost:6006> to inspect the full
LangGraph trace, or the web UI at <http://localhost:5173> (it performs its own
OIDC handshake against Dex).

Requests without a valid Bearer token are rejected with a stable error
envelope: `{"error":"unauthenticated","reason":"missing_bearer","request_id":"..."}`.

> **Local Dex note:** Dex advertises its issuer as `http://localhost:5556/dex`
> while the API reaches it in-cluster at `http://dex:5556/dex`. If you hit an
> issuer-mismatch error, align both sides on the same host — see the auth docs.

## Configuration

opsrag is configured by a single `config.yaml` (Pydantic-v2 validated) plus
environment variables for secrets. The default `config.yaml` boots a
healthy service with **zero MCP integrations enabled** and a built-in
**null knowledge-graph backend** — only an LLM API key and the bundled
local OIDC issuer are required.

Each of the fourteen MCP integrations is opt-in:

```yaml
mcp:
  prometheus:
    enabled: true   # required env: PROMETHEUS_URL, PROMETHEUS_BEARER_TOKEN
  datadog:
    enabled: false
  # ...
```

Missing required credentials for an enabled integration cause a named,
fail-fast startup error (`MCP_MISCONFIGURED:<name>:<env>`).

The full schema is in
[`specs/001-port-opsrag-opensource/contracts/config-schema.md`](specs/001-port-opsrag-opensource/contracts/config-schema.md).

## Deploying with Helm

```sh
helm install opsrag deploy/helm/opsrag \
  -f my-values.yaml \
  --namespace opsrag --create-namespace
```

Ready-made scenario values live in `deploy/helm/opsrag/`: `values-gcp.yaml`,
`values-aws.yaml`, `values-mcp.yaml`, `values-multi-env.yaml`,
`values-minimal.yaml`. See the [Helm chart reference](docs/helm-chart.md) and
the [Deployment guide](docs/deployment.md).

## Project layout

```text
opsrag/                  Python backend package
  agent/                 Core LangGraph RAG agent (4 topologies)
  investigations/        Event-driven incident-investigation engine
  api/                   FastAPI surface (HTTP + SSE + webhooks)
  auth/                  OIDC / SSO / first-party login
  mcp/                   20 MCP integrations, each opt-in
  environments.py        Multi-environment registry resolver
  ...
ui/                      React single-page UI (Vite)
deploy/
  helm/opsrag/           First-class Helm chart
  compose/               Local docker-compose stack (incl. Dex)
samples/                 Synthetic corpus for the quickstart
scripts/                 Audit, seed, helper scripts
tests/                   contract / integration / unit
```

## Documentation

Full documentation is in [`docs/`](docs/README.md). Highlights:

- [Getting started](docs/getting-started.md) | [Configuration](docs/configuration.md) | [Deployment](docs/deployment.md)
- [Architecture](docs/architecture.md) | [RAG pipeline](docs/rag-pipeline.md) | [Investigations](docs/investigations.md)
- [MCP integrations](docs/mcp-integrations.md) | [Multi-environment](docs/multi-environment.md)
- [Authentication](docs/auth.md) | [Operations](docs/operations.md) | [API reference](docs/api-reference.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the PR workflow and the
mandatory checks (lint, types, tests, vendor-neutrality audit, helm-lint,
eval-regression).

## Security

Vulnerability reports go through the process in [SECURITY.md](SECURITY.md).
Please do not file public issues for security findings.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
