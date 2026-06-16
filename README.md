<div align="center">

# OpsRAG

### Agentic GraphRAG for DevOps & SRE

Turn your runbooks, Terraform, Helm charts, Kubernetes manifests, and incident
postmortems into **cited, trustworthy answers** — and let an agent run
**autonomous incident investigations** against your live telemetry.

<br/>

[![CI](https://github.com/dtduc-git/opsrag/actions/workflows/ci.yml/badge.svg)](https://github.com/dtduc-git/opsrag/actions/workflows/ci.yml)
[![Security](https://github.com/dtduc-git/opsrag/actions/workflows/security.yml/badge.svg)](https://github.com/dtduc-git/opsrag/actions/workflows/security.yml)
[![Eval: Recall@5 1.0](https://img.shields.io/badge/eval%20Recall%405-1.0-brightgreen.svg)](docs/evaluation.md)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#status)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

<br/>

**[Quickstart](#-quickstart)** &nbsp;·&nbsp;
**[Documentation](docs/README.md)** &nbsp;·&nbsp;
**[Architecture](docs/architecture.md)** &nbsp;·&nbsp;
**[Configuration](docs/configuration.md)** &nbsp;·&nbsp;
**[Deploy](docs/deployment.md)**

</div>

---

OpsRAG is an open-source, vendor-neutral retrieval system built for SRE and
platform teams. It bundles a LangGraph agent, a pluggable retrieval pipeline
(vector + optional knowledge graph + 20 opt-in MCP integrations), a FastAPI
surface with OIDC/SSO auth, a React UI, chat bots for Slack/Telegram/Discord/Teams
(with image understanding), an evaluation harness, a first-class Helm chart, and a
one-command local stack.

Every answer is **grounded and cited**. A new evaluator goes from `git clone`
to a cited answer in **under fifteen minutes** — no Kubernetes, no cloud
account, and no external identity provider required.

## ✨ Features

|  |  |
|---|---|
| 🔎 **Hybrid retrieval** | Dense + BM25 + a code-aware lane, fused with Reciprocal Rank Fusion and diversified with MMR. |
| 🧠 **Agentic RAG** | LangGraph agent with CRAG / Self-RAG, anti-hallucination grounding gates, and a semantic answer cache. |
| 🕵️ **Incident investigations** | An event-driven engine that forms hypotheses and verifies them against live telemetry, with resumable SSE and a hard budget. |
| 💬 **Chat bots** | Talk to OpsRAG from Slack, Telegram, Discord, and Teams — same agent, allowlist/scoped, with a read-only web view of shared-channel conversations. |
| 🖼️ **Image understanding** | Attach a screenshot or diagram on the web UI or any channel bot — ephemeral, vision pass-through with provider-aware auto-routing to a vision model when the active one can't see (bytes are never persisted). |
| 🔌 **20 MCP integrations** | Read-only, opt-in connectors — Datadog, Prometheus, Kubernetes, Elasticsearch, GitHub, GitLab, Sentry, Grafana, Loki, Splunk, Rootly, Slack, and more. |
| 🌍 **Multi-environment** | One instance targeting many environments' Kubernetes / Prometheus / Elasticsearch via a single `environments:` registry. |
| 🔐 **Auth built in** | `open` / `oidc` / first-party `login` modes, SSO (Google · GitHub · Microsoft), per-session ownership, optional Redis rate limiting. |
| 🧩 **Pluggable everything** | Vector store, knowledge graph, and LLM / embedding / reranker providers are all swappable from config — no rebuild. |
| 📊 **Observability** | Per-request token + cost telemetry, Phoenix / OTLP traces, and an evaluation harness wired into CI gates. |
| 📏 **Runnable eval** | A two-tier golden eval over the shipped `samples/` corpus — an offline retrieval gate (`pytest tests/integration/test_eval_samples_retrieval.py`, no secrets) plus a full answer-quality judge (`python -m opsrag.eval run`). |

## 🏗️ Architecture

```mermaid
flowchart LR
    subgraph SRC ["Knowledge sources"]
        direction TB
        A[Git repos<br/>runbooks · IaC · manifests]
        B[Confluence · Slack<br/>Rootly · postmortems]
    end

    SRC --> ING[Ingestion<br/>parent-child chunking<br/>contextual embeddings]
    ING --> VS[(Vector store<br/>Qdrant · pgvector)]
    ING --> KG[(Knowledge graph<br/>Neo4j · optional)]

    VS --> AGENT
    KG --> AGENT
    MCP[20 read-only<br/>MCP integrations] --> AGENT

    AGENT{{LangGraph agent<br/>hybrid retrieval · RRF · MMR<br/>CRAG / Self-RAG · grounding gates}}

    AGENT --> API[FastAPI<br/>HTTP · SSE · webhooks]
    API --> UI[React UI]
    API --> SLACK[Slack bot]
    API --> INV[Incident<br/>investigations]
```

See [docs/architecture.md](docs/architecture.md) for the request flow, the
provider seams, and the investigation engine.

## 🚀 Quickstart

**Prerequisites:** Docker (Compose v2), `curl`, `jq`, and one LLM API key
(default: Anthropic). You do **not** need Kubernetes, an external OIDC provider
(Dex is bundled), or any cloud account.

```sh
# 1. Clone and set your LLM key
git clone https://github.com/dtduc-git/opsrag.git
cd opsrag
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY=...   (every other value can stay default)

# 2. Bring up the stack:
#    API :8080 · UI :5173 · Qdrant · Postgres · Dex (OIDC :5556) · Phoenix :6006
docker compose -f deploy/compose/docker-compose.yaml up -d --build

# 3. Verify health (readiness flips to 200 once Postgres + Qdrant are up)
curl -sf http://localhost:8080/healthz
curl -sf http://localhost:8080/readyz

# 4. Index the bundled sample corpus (the fictional "Acme Notes" product)
docker compose -f deploy/compose/docker-compose.yaml exec opsrag-api \
  scripts/seed-sample-corpus.sh

# 5. (Illustrates OIDC) Get a token from the bundled Dex — the stand-in for a
#    real IdP. NOTE: the compose demo runs in OPEN mode (see "Authentication"
#    below), so this token is ACCEPTED but not REQUIRED. It shows the `oidc`
#    path; there is no first-party "admin" user in this mode.
TOKEN=$(curl -sf -X POST \
  -d 'grant_type=password' \
  -d 'username=evaluator@example.com' -d 'password=evaluator' \
  -d 'client_id=opsrag-local' -d 'client_secret=local-secret' \
  -d 'scope=openid profile email' \
  http://localhost:5556/dex/token | jq -r .access_token)

# 6. Ask a question. In the open-mode demo the Authorization header is OPTIONAL
#    (the same call works without it); in `oidc` mode it would be required.
curl -sf -X POST http://localhost:8080/query \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"query":"How do I roll back an Acme Notes deployment?"}' | jq
```

You get back a cited answer drawn from the indexed `samples/` corpus, plus a
`session_id` and a `trace_id`. Open the web UI at <http://localhost:5173>, or
inspect the full LangGraph trace in Phoenix at <http://localhost:6006>.

> **Heads-up:** the bundled Dex advertises its issuer as
> `http://localhost:5556/dex` while the API reaches it in-cluster at
> `http://dex:5556/dex`. On an issuer-mismatch error, align both sides — see
> [docs/auth.md](docs/auth.md).

The full walkthrough lives in
[`docs/getting-started.md`](docs/getting-started.md).

## 🔑 Authentication: the Dex user vs. the admin user

The Dex token above and the first-party `admin@opsrag.local` account belong to
**two different auth modes** — a common point of confusion. OpsRAG has three:

| Mode | Who proves identity | First-party users? | Used by |
|---|---|---|---|
| `open` | nobody — every request is anonymous with all scopes | — | the **compose demo** (so the cookie-less UI works with no login screen) |
| `oidc` | an **external IdP** issues a Bearer token; OpsRAG only *validates* it | **No** — no user database; identity lives entirely in the token | the **Dex** flow above; in production, your Okta / Entra / Google |
| `login` | OpsRAG's own login: password + SSO (Google/GitHub/Microsoft) + cookies | **Yes** — including the bootstrap **`admin@opsrag.local`** | first-party deployments |

**The key point: by default there is no admin account.** In `oidc` mode the
"user" is whoever the token says — the Dex `evaluator@example.com` above — and
OpsRAG keeps **no user records at all**. The `admin@opsrag.local` account only
exists when you run **`login` mode**. So the "Dex user" and the "admin user"
are not the same thing, and you can't swap one for the other within a single
mode.

**Getting the admin user.** There is no pre-baked password to retrieve — you
create it. Switch to `login` mode and set your own credentials via env (never
inline in config):

```sh
# config.yaml ->  auth: { mode: login }
OPSRAG_SESSION_SIGNING_KEY=<32+ random bytes>      # signs session cookies
OPSRAG_ADMIN_EMAIL=admin@opsrag.local
OPSRAG_ADMIN_PASSWORD=<choose-a-strong-password>
```

On boot OpsRAG **seeds** that admin (role `admin`, idempotent). Log in via the
web UI's login screen, or `POST /auth/login` to get a session cookie. Full
steps (SSO, refresh tokens, RBAC) are in [docs/auth.md](docs/auth.md), which
covers all three modes end to end.

## 🧭 Documentation

Full docs are in **[`docs/`](docs/README.md)**. Start here:

| Guide | What it covers |
|---|---|
| [Getting started](docs/getting-started.md) | Clone → run → index → first query → enable auth → first investigation. |
| [Configuration](docs/configuration.md) | The config model, `env > YAML > bundle` precedence, secrets, and every config block. |
| [Architecture](docs/architecture.md) | Component map, the `/query` request flow, and the provider seams. |
| [RAG pipeline](docs/rag-pipeline.md) | Ingestion, chunking, hybrid retrieval, reranking, CRAG/Self-RAG, and the answer cache. |
| [Evaluation](docs/evaluation.md) | The two-tier golden eval over `samples/`: the offline retrieval gate (no secrets) and the answer-quality judge. |
| [Investigations](docs/investigations.md) | The event-driven incident-investigation engine. |
| [MCP integrations](docs/mcp-integrations.md) | The 20 read-only integrations, their env vars, and the safety model. |
| [Multi-environment](docs/multi-environment.md) | One instance, many environments' Kubernetes / Prometheus / Elasticsearch. |
| [Authentication](docs/auth.md) · [Operations](docs/operations.md) · [API reference](docs/api-reference.md) | Auth modes + SSO, day-2 ops, and the HTTP/SSE surface. |

## ⚙️ Configuration

OpsRAG is driven by a single `config.yaml` (Pydantic-v2 validated) plus
environment variables for secrets. The default config boots a healthy service
with **zero MCP integrations** and a **null knowledge graph** — only an LLM API
key and the bundled local OIDC issuer are required. Each integration is opt-in:

```yaml
mcp:
  prometheus:
    enabled: true   # required env: PROMETHEUS_URL, PROMETHEUS_BEARER_TOKEN
  datadog:
    enabled: false
```

Missing credentials for an enabled integration produce a named, fail-fast
startup error (`MCP_MISCONFIGURED:<name>:<env>`). See
[docs/configuration.md](docs/configuration.md).

## ☸️ Deploying with Helm

```sh
helm install opsrag deploy/helm/opsrag \
  -f my-values.yaml --namespace opsrag --create-namespace
```

Ready-made scenario values live in `deploy/helm/opsrag/`: `values-gcp.yaml`,
`values-aws.yaml`, `values-mcp.yaml`, `values-multi-env.yaml`,
`values-minimal.yaml`. See the [Helm chart reference](docs/helm-chart.md) and
the [Deployment guide](docs/deployment.md).

## 🗂️ Project layout

```text
opsrag/                  Python backend package
  agent/                 Core LangGraph RAG agent (4 topologies)
  investigations/        Event-driven incident-investigation engine
  api/                   FastAPI surface (HTTP + SSE + webhooks)
  auth/                  OIDC / SSO / first-party login
  mcp/                   20 MCP integrations, each opt-in
  environments.py        Multi-environment registry resolver
ui/                      React single-page UI (Vite)
deploy/
  helm/opsrag/           First-class Helm chart
  compose/               Local docker-compose stack (incl. Dex)
samples/                 Synthetic corpus for the quickstart
scripts/                 Audit, seed, and helper scripts
tests/                   contract / integration / unit
```

## Status

> **Pre-release (`0.1.0a0`).** The first public release is being assembled on
> `master`. Interfaces and configuration keys may change without notice until
> `v0.1.0`.

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the PR workflow and the mandatory
checks (lint, types, tests, vendor-neutrality audit, helm-lint, and the
always-on, no-secrets `eval-offline` retrieval gate). `master` is protected:
every PR must pass the required checks before it can merge.

## 🔒 Security

Report vulnerabilities through the process in [SECURITY.md](SECURITY.md).
Please do not file public issues for security findings.

## 📄 License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
