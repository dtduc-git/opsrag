# OpsRAG Documentation

OpsRAG is a production-grade RAG system for SRE/DevOps: it ingests your
knowledge (Git, Confluence, Slack, Rootly, runbooks), answers questions with a
hybrid-retrieval + multi-stage agent, and runs event-driven incident
investigations against 20 read-only MCP integrations.

Start with **Getting started**, then branch by what you need.

## Get started

| Doc | What it covers |
|-----|----------------|
| [Getting started](./getting-started.md) | Clone → run (docker-compose) → index → first query → enable auth → first investigation. |
| [Configuration](./configuration.md) | The config model, `env > YAML > bundle` precedence, the `_env` secret convention, `cloud_provider` bundles, and a tour of every config block. |
| [Deployment](./deployment.md) | docker-compose, Helm on Kubernetes, and production hardening (secrets, HA, the Redis rate limiter, network policy, the indexer Job). |
| [Helm chart reference](./helm-chart.md) | The chart values reference + the scenario values files (GCP / AWS / MCP / multi-env / minimal). |

## Architecture & internals

| Doc | What it covers |
|-----|----------------|
| [Architecture](./architecture.md) | Component map, the `/query` request flow, pluggable providers, DeploymentContext. |
| [RAG pipeline](./rag-pipeline.md) | Ingestion → chunking → embedding → hybrid retrieval (RRF) → reranking (+MMR) → CRAG/Self-RAG → anti-hallucination gates → semantic QA cache → corrections. |
| [Investigations](./investigations.md) | The event-driven incident-investigation engine: lanes, hypotheses, the MCP reasoner<->evaluator loop, verdicts + citations, the hard budget, resumable SSE. |

## Integrations & environments

| Doc | What it covers |
|-----|----------------|
| [MCP integrations](./mcp-integrations.md) | The 20 read-only integrations, their required env vars and tools, the deny-by-verb safety model, and how to enable one. |
| [Channel bots](./channels.md) | Slack / Telegram / Discord / Teams chat bots — the shared adapter core, per-platform setup, the `channels:` config, run model (role-gated workers + the Teams webhook), identity/quota, and read-only browsing of shared-channel conversations in the web UI. |
| [Multi-environment](./multi-environment.md) | The `environments:` registry — one instance targeting N environments' Kubernetes / Prometheus / Elasticsearch. |

## Operations & reference

| Doc | What it covers |
|-----|----------------|
| [Authentication](./auth.md) | The two auth modes (`login` default, or `oidc` with an external IdP), SSO (Google/GitHub/Microsoft), per-session ownership, and rate limiting. |
| [Memory](./memory.md) | Mem0 per-user/service conversational memory. |
| [Evaluation](./evaluation.md) | The golden-set eval harness, the 9 metrics + gate thresholds, and CI gating. |
| [Operations](./operations.md) | Day-2 ops: indexing/reindexing (and the ratio reindex caveat), observability + cost, scaling, security hardening, troubleshooting. |
| [API reference](./api-reference.md) | The HTTP/SSE endpoints, grouped, with auth scopes. |

## Contributing

- [Vendor-neutrality audit](./contributing-audit.md) — the audit that keeps the engine free of org-specific values.
- See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the PR workflow and required checks.
