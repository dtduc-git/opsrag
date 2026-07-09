# Architecture

opsrag is an open-source, vendor-neutral, agentic GraphRAG system for
DevOps and SRE. It indexes operational knowledge (runbooks, postmortems,
Terraform, Helm charts, Kubernetes manifests, source code) into a vector
store, then answers questions with a LangGraph agent that retrieves,
optionally calls live infrastructure tools, generates an answer, and
cites every source it used.

"GraphRAG" here means vector-RAG retrieval -- dense + BM25 + a
code-aware lane, fused with RRF and diversified with MMR -- which is what
actually answers a `/query`, sitting beside (not on top of) two distinct
graphs:

1. The Neo4j knowledge graph, which captures relationships between
   services, libraries, features, and dependencies. It is built at
   ingest by a hybrid rule + LLM extractor (the default; set
   `entity_extraction.method: rule_based` for a fully deterministic,
   no-LLM build). It powers the topology viewer and is NOT consulted in
   the vector-retrieval lane that answers a `/query`.
2. An optional light entity-graph (Postgres adjacency, zero-LLM
   rule-based), which adds a 1-hop entity expansion to retrieval when
   `light_graph.enabled: true`. It is off by default.

So by default vector retrieval consults no graph; with the light
entity-graph enabled it adds 1-hop entity expansion; the Neo4j
relationship/topology graph is never used during retrieval.

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
  | graph    |      | pipeline      |    | (factory)   |   | (27, gated)   |
  | (LangGr) |      | SCM->parse->  |    | LLM/embed/  |   | config-gated  |
  |          |      | chunk->embed  |    | vector/graph|   | tool exposure |
  +----+-----+      +-------+-------+    +------+------+   +-------+-------+
       |
       |  Investigate mode -> opsrag/investigations/ (InvestigationRunner):
       |  event-ledger runner; 3 lanes -> hypothesizer -> reasoner (MCP
       |  tools) -> evaluator; events tailed over SSE. Reaches each env via
       |  the `environments:` registry (opsrag/environments.py).
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
- `agent/` - the core LangGraph agent (graph builders, nodes, state)
  that answers `/query`.
- `investigations/` - the event-driven Investigate engine
  (`InvestigationRunner`) for alert root-cause work: a Postgres event
  ledger backs a resumable SSE feed, parallel evidence lanes feed a
  hypothesizer + reasoner loop (calling MCP tools) and a structured
  evaluator. Separate from the `/query` agent graph and feature-gated.
- `ingestion/` - the indexing pipeline (SCM -> parse -> chunk -> embed
  -> vector store).
- `factory.py` + `config.py` + `context.py` + `environments.py` -
  provider wiring, the Pydantic settings root, the operator-supplied
  `DeploymentContext`, and the multi-environment registry resolver.
- `mcp/` + `mcp_server/` - the 27 MCP integrations, their registry, and
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

## Investigate engine (event ledger)

Alert root-cause work runs through a separate engine in
`opsrag/investigations/` (`InvestigationRunner`), not the `/query` agent
graph. It is event-driven: every step writes a row to the
`opsrag_investigation_events` Postgres ledger, and the SSE endpoint is a
thin tail-cursor over that table, so a closed tab or network blip
recovers by replaying from the last sequence rather than losing a live
stream.

```text
  POST /investigations                       (routes_investigations.py)
       |   create lifecycle row, kick off run_one()
       v
  InvestigationRunner.run_one()
       |
       +-- 3 evidence lanes (Flash):
       |     LANE_A runbook search   LANE_B historical similar
       |     LANE_C live probe
       |
       +-- INSIGHT_READY            (Pro fuses the lanes into a card)
       +-- HYPOTHESES_GENERATED     (Pro, structured Pydantic output)
       +-- reasoner loop (Pro, MCP tool-calling)
       |     TOOL_CALLED / TOOL_RESULT per call
       +-- evaluator pass (Flash, structured output)
       |     HYPOTHESIS_EVALUATED -> confirmed | ruled_out
       |                            | untested  | open  (+ citations)
       +-- CONCLUSION_READY -> INVESTIGATION_COMPLETED

  GET /investigations/{id}/events?since=N     SSE tail of the ledger
      (browser EventSource reconnects with since=<lastSeenSeq>)
```

The runner reaches each environment's live tools (Kubernetes,
Prometheus, Elasticsearch, logs) through the same gated MCP surface as
the agent, scoped by the `environments:` registry below. A hard
per-run budget bounds cost and latency: a 240s wall-clock stop, a 40
cumulative-tool-call cap, and a 45s per-tool timeout. The feature is
surfaced only when the operator enabled a live-telemetry integration
(`opsrag/investigations/feature_gate.py`), so a corpus-only deployment
never sees the Investigate tab.

## Multi-environment registry

A single opsrag instance can target N environments (for example
`staging` and `production`). The top-level `environments:` config block
(`EnvironmentsConfig` / `EnvironmentTarget` in `opsrag/config.py`) maps
each env name to how to reach its Kubernetes, Prometheus, and
Elasticsearch:

- `kubernetes` - `gke` mode (Workload Identity + the GCP Container API)
  or vendor-neutral `kubeconfig` mode (an EKS/cert/in-cluster context).
- `prometheus` - `k8s_proxy` (through the cluster API server's service
  proxy) or `direct` (a reachable base URL).
- `elasticsearch` - `direct`, `port_forward`, or `proxy`, with a
  logical-to-physical `fields` map that de-hardcodes one org's index
  schema.

The resolver in `opsrag/environments.py` binds this registry once at
startup (`bind_environments`); lookups are pure and a miss raises a
structured error rather than falling back to a default. The k8s,
Prometheus, and Elasticsearch MCP tools each take an `env` argument and
resolve through it. When `environments.targets` is empty, the registry
is synthesized from the legacy `k8s` / `elasticsearch` / `deployment`
blocks, so existing single-env deployments keep working unchanged.

## Pluggable providers

`opsrag/factory.py :: build_providers(config)` is the single place that
turns configuration into concrete implementations. Each subsystem is
selected by a `provider` string in `config.yaml` and constructed against
a Protocol interface in `opsrag/interfaces/`, so call sites stay
agnostic:

| Subsystem    | Config key                 | Built-in choices               |
|--------------|----------------------------|--------------------------------|
| LLM          | `llm.provider`             | anthropic, openai, vertex, bedrock, litellm |
| Embedder     | `embedding.provider`       | fastembed (default), openai, vertex, bedrock, litellm |
| Vector store | `vector_store.provider`    | qdrant (default), pgvector     |
| Graph store  | `knowledge_graph.provider` | none/null (default), neo4j     |
| Reranker     | `reranker.provider`        | fastembed, cohere, bedrock, vertex, noop (+ MMR diversity) |
| Sessions     | `session.provider`         | memory, postgres               |
| Memory       | `memory.provider`          | memory, postgres, mem0         |
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
instead. When a graph backend is bound, ingestion populates it via the
`HybridExtractor`, which by default (`entity_extraction.method: hybrid`)
uses an LLM for prose documents on top of deterministic rule/metadata
lanes; set `entity_extraction.method: rule_based` for a fully
deterministic, no-LLM build. It captures relationships between services,
libraries, features, and dependencies. This graph is a separate,
on-demand relationship/topology source: it is NOT consulted in the
vector-retrieval lane that answers `/query`. The legacy graph-anchored
retrieval lane was removed. Today the graph powers the topology viewer;
on-demand relationship/topology queries are on the roadmap.

The light entity-graph is a different, optional component: a Postgres
adjacency table built at ingest by the zero-LLM `RuleBasedExtractor`.
When `light_graph.enabled: true`, a 1-hop `entity_expand` node augments
vector retrieval with related chunks; it is off by default and works
even with `knowledge_graph.provider: none`.

## Config-gated MCP tools

opsrag ships 27 named MCP integrations (live infrastructure surfaces
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
