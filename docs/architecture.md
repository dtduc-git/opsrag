# Architecture

opsrag is an open-source, vendor-neutral, agentic GraphRAG system for
DevOps and SRE. It indexes operational knowledge (runbooks, postmortems,
Terraform, Helm charts, Kubernetes manifests, source code) into a vector
store, then answers questions with a LangGraph agent that retrieves,
optionally calls live infrastructure tools, generates an answer, and
cites every source it used.

This document gives a high-level map of the system, the request flow for
a `/query`, the pluggable-provider model, the config-gated MCP tool
surface, the `DeploymentContext` boundary, and the deployment options.

## Component map

```text
                    +--------------------------------------------+
   Browser  ---->   |  React SPA (ui/)                           |
   (OIDC)           |  - does its own OIDC handshake             |
                    +----------------------+---------------------+
                                           |  Bearer token
                                           v
  +----------------------------------------------------------------------+
  |  FastAPI app  (opsrag/api/server.py :: create_app)                    |
  |                                                                       |
  |   RateLimit middleware  ->  OIDC middleware  ->  routers              |
  |   (per-minute cap)          (Bearer enforce)     /query /healthz ...  |
  |                                                                       |
  |   lifespan: build_providers() + wire agent graph, ingestion,         |
  |             stores, MCP server, optional Slack bot + scheduler        |
  +----+--------------------+-------------------+-----------------+-------+
       |                    |                   |                 |
       v                    v                   v                 v
  +----------+      +---------------+    +-------------+   +---------------+
  | Agent    |      | Ingestion     |    | Providers   |   | MCP registry  |
  | graph    |      | pipeline      |    | (factory)   |   | (14, gated)   |
  | (LangGr) |      | SCM->parse->  |    | LLM/embed/  |   | config-gated  |
  |          |      | chunk->embed  |    | vector/graph|   | tool exposure |
  +----+-----+      +-------+-------+    +------+------+   +-------+-------+
       |                    |                   |                  |
       |   query embed +    | upsert chunks     | concrete         | enabled
       |   retrieve         v                   v impls            v tools
       |            +---------------+    +-------------+   +---------------+
       +----------> | Vector store  |    | LLM / embed |   | live infra:   |
                    | (Qdrant /     |    | provider    |   | k8s, metrics, |
                    | pgvector)     |    | (per config)|   | logs, SCM,    |
                    +---------------+    +-------------+   | incidents ... |
                    +---------------+                      +---------------+
                    | Graph store   |    +-------------+
                    | (null default |    | Postgres:   |
                    |  / neo4j)     |    | sessions,   |
                    +---------------+    | checkpoints,|
                                         | memory,     |
                                         | usage, etc. |
                                         +-------------+
```

Top-level packages (under `opsrag/`):

- `api/` - FastAPI surface: HTTP routes, SSE streaming, webhooks, the
  OIDC enforcement middleware, the rate-limit middleware, and the app
  lifespan that wires everything together.
- `agent/` - the core LangGraph agent (graph builders, nodes, state).
- `agents/investigation/` - a separate hypothesis-driven investigation
  subgraph for alert root-cause work; deliberately not wired into the
  live-query graph.
- `ingestion/` - the indexing pipeline (SCM -> parse -> chunk -> embed
  -> vector store).
- `factory.py` + `config.py` + `context.py` - provider wiring, the
  Pydantic settings root, and the operator-supplied `DeploymentContext`.
- `mcp/` + `mcp_server/` - the 14 MCP integrations, their registry, and
  the gating that decides which tools the agent may see.
- `interfaces/` - the Protocol interfaces every provider implements
  (the plugin contracts).
- Provider families: `llms/`, `embedders/`, `vectorstores/`,
  `graphstores/`, `rerankers/`, `parsers/`, `chunkers/`, `scm/`,
  `sessions/`, `memory/`, `observability/`, `sources/`.
- `ui/` - the React single-page app (separate workload).

## Request flow for `/query`

```text
  client request
       |
       v
  RateLimit middleware            reject 429 if over per-minute cap
       |
       v
  OIDC middleware                 allowlist: /healthz /readyz /docs ...
       |                          else require valid Bearer; on failure
       |                          return {error: unauthenticated,
       |                                  reason: ..., request_id: ...}
       v
  /query route handler
       |
       v
  query_with_session(_events)     (opsrag/agent/graph.py)
       |
       +-- load prior turns from session store; expand "retry" meta;
       |   optional follow-up query rewrite
       |
       +-- embed query (cached); optional Q&A semantic-cache lookup
       |   -> on hit, return cited cached answer immediately
       |
       v
  compiled LangGraph (agent.mode selects topology):
       route/triage -> retrieve (vector) -> rerank -> grade
                    -> (rewrite & retry | generate | insufficient_info)
                    -> verify_answer -> hallucination_check
       (tool / multi-agent modes can also call gated MCP tools first,
        then synthesize; a casual lane short-circuits to a quick reply)
       |
       v
  build response: answer + sources + source_urls + sources_content
                  + grounded flag + thread_id (+ trace via observability)
       |
       v
  store fresh answer in Q&A cache (unless tool-path / not grounded)
```

Every non-cached answer carries the sources the LLM actually saw, and
deep-link URLs are built only from `DeploymentContext.source_urls` (see
below) so the engine never invents a host.

## Pluggable providers

`opsrag/factory.py :: build_providers(config)` is the single place that
turns configuration into concrete implementations. Each subsystem is
selected by a `provider` string in `config.yaml` and constructed against
a Protocol interface in `opsrag/interfaces/`, so call sites stay
agnostic:

| Subsystem    | Config key                 | Built-in choices               |
|--------------|----------------------------|--------------------------------|
| LLM          | `llm.provider`             | anthropic, openai, vertex, bedrock |
| Embedder     | `embedding.provider`       | openai, fastembed, vertex, bedrock |
| Vector store | `vector_store.provider`    | qdrant (default), pgvector     |
| Graph store  | `knowledge_graph.provider` | null/none (default), neo4j     |
| Reranker     | `reranker.provider`        | noop, cohere, fastembed, vertex|
| Sessions     | `session.provider`         | inmemory, postgres             |
| Memory       | `memory.provider`          | inmemory, postgres             |
| Observability| `observability.provider`   | console, phoenix               |
| SCM          | `scm.provider`             | gitlab, github (clone or API)  |

Optional dependencies are imported lazily inside the relevant branch, so
a minimal deployment never needs the heavier providers installed.
Adding a provider means implementing the interface and adding one branch
to the factory; nothing else has to change.

### Vector store: Qdrant vs pgvector (retrieval-quality trade-off)

The two vector-store backends are not at full retrieval parity, and the
choice has a measurable quality impact on symbol-heavy queries:

- **Qdrant** runs full hybrid retrieval: dense ANN plus a *true BM25*
  sparse lane (FastEmbed with an IDF modifier, fed identifier-subtoken-
  augmented text), fused via RRF. It is also the only backend that supports
  the optional code lane (`code_embedding` + `code_vector_store`). This is
  the recommended default.
- **pgvector** provides strong dense ANN but *not* lexical parity. Its
  sparse lane is Postgres full-text search (`ts_rank_cd` over `simple`
  FTS) plus a best-effort `pg_trgm` trigram lane that recovers exact-symbol
  *recall* — not true BM25/IDF *ranking*. The code lane does not apply
  (the retriever feature-detects and skips it on pgvector). If `pg_trgm`
  is not grantable on the target Postgres (`CREATE EXTENSION pg_trgm`), the
  trigram lane is disabled and exact-symbol recall falls back to dense +
  FTS only (a warning is logged at startup).

Choose pgvector to avoid operating Qdrant (e.g. reuse an existing
RDS/CloudSQL/AlloyDB instance), accepting somewhat weaker lexical-relevance
ranking on identifier/symbol queries. Choose Qdrant when retrieval quality
on code/symbol lookups matters most.

## Null graph backend (default)

The knowledge graph is provider-selected, not feature-flagged. The
default is the null backend (`opsrag/graphstores/null.py`), which
satisfies the `KnowledgeGraphStore` interface with empty results. This
keeps the agent graph topology stable while letting a minimal
deployment run with no graph database at all - the default config needs
only an LLM key, a vector store, and the local OIDC issuer.

Selecting `knowledge_graph.provider: neo4j` wires the Neo4j driver
instead. The legacy graph-anchored retrieval lane was removed; the Neo4j
driver remains available for graph-oriented MCP queries rather than
ingest-time entity writes.

## Config-gated MCP tools

opsrag ships 14 named MCP integrations (live infrastructure surfaces
such as Kubernetes, metrics, logs, SCM, incident trackers, cloud
inventory, and a read-through tool cache). The registry at
`opsrag/mcp/registry.py` is the single source of truth: each entry
declares its name, config-block subclass, required env vars, required
config keys, the tool names it exposes, and its real and fake factories.

Gating happens in two layers:

1. Fail-fast validation. At startup `validate_enabled_mcps()` checks
   every integration with `enabled: true` against its `required_env`
   and `required_config`. The first missing item raises
   `MCP_MISCONFIGURED:<name>:<missing>` before the app serves traffic.
2. Tool exposure. `opsrag/mcp_server/registry_loader.py` computes the
   set of tool names contributed by the enabled integrations and
   installs it as a process-level filter. The agent filters the
   always-present tool superset through this set, so the LLM only ever
   sees tools for integrations the operator turned on.

Every integration defaults to `enabled: false`. With the default
config, the agent sees zero MCP tools and answers from the indexed
corpus only. An operator enables an integration by flipping its
`mcp.<name>.enabled` flag (and supplying the required env). Each
integration also ships a fake factory so per-integration tests run with
no live network.

## DeploymentContext (Principle VI)

The engine carries no organization-specific knowledge. Everything the
agent needs to know about a particular deployment - service names,
environments, key repos, cluster identifiers, cloud projects, ticket
prefix, and source-system base URLs - is supplied at runtime through the
`DeploymentContext` model in `opsrag/context.py`, mounted at
`Settings.deployment`.

Key properties:

- The engine reads from this model; it never invents values for it.
  Absent fields disable the corresponding behavior (for example, a
  missing source URL disables deep-linking for that source rather than
  substituting a placeholder host).
- Operator facts drive concrete engine behavior: `services` seed the
  semantic-router anchors, `environments` build the env-discriminator
  regex, `source_urls` build citation deep-links, and `tracker.prefix`
  is used to recognize and format ticket references.
- At app creation, `set_active_deployment(cfg.deployment)` installs the
  context so prompt templates render against operator facts. The empty
  default yields generic, org-free prompts.

This boundary is what makes opsrag genuinely reusable: the same image
serves any organization, configured entirely from the outside. A
vendor-neutrality audit (see `scripts/audit-vendor-neutrality.sh`) gates
merges to keep proprietary names, non-English text, and hardcoded hosts
out of the shipped engine and its docs.

## Lifespan wiring

The FastAPI lifespan (`opsrag/api/server.py`) runs once at startup:
expand the thread pool, apply DB migrations (idempotent, opt-out),
`build_providers(cfg)`, build the agent graph for the configured
`agent.mode`, construct the ingestion pipeline, open Postgres-backed
stores (sessions, checkpoints, memory, usage, feedback, runbooks), bind
enabled MCP integrations, and optionally start the Slack Socket-Mode bot
and the daily indexing scheduler. A role gate (`OPSRAG_ROLE`) lets a
serving deployment suppress the auto-index loop and scheduler so those
run on a dedicated indexer workload.

## Deployment options

Local development (`deploy/compose/`):

```text
docker compose -f deploy/compose/docker-compose.yaml up -d
```

The compose stack brings up the backend API, the React UI, Qdrant,
Postgres, a bundled Dex OIDC issuer (so no external IdP is needed), and
Phoenix for tracing. This is the fifteen-minute clone-to-cited-answer
path; the default config uses the null graph backend and zero MCP
integrations.

Production (`deploy/helm/opsrag/`):

```text
helm install opsrag deploy/helm/opsrag -f my-values.yaml \
  --namespace opsrag --create-namespace
```

The Helm chart deploys the API, UI, and (optionally) the Slack bot as
separate workloads, with a non-root ServiceAccount, NetworkPolicy
egress allowlist, PodDisruptionBudget, optional HPA, and a `helm test`
connection hook. Every `mcp.<name>.enabled` flag is exposed through
`values.yaml`, validated by `values.schema.json`, so the production
tool surface is configured the same way as the local one.
