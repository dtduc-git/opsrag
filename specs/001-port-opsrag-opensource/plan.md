# Implementation Plan: Port opsrag as a vendor-neutral opensource project

**Branch**: `001-port-opsrag-opensource` | **Date**: 2026-05-28 | **Spec**: [`spec.md`](./spec.md)

**Input**: Feature specification from `specs/001-port-opsrag-opensource/spec.md`

## Summary

Port the existing internal opsrag agentic-GraphRAG codebase (Python backend +
React UI + Slack bot + LangGraph agent + 14 MCP integrations + Phoenix/DeepEval
harness + Helm chart) from `acme/infra/opsrag` to this repository as a
vendor-neutral opensource project under Apache-2.0. Every MCP integration
becomes opt-in via an explicit `mcp.<name>.enabled` flag with fail-fast
behaviour when enabled-but-misconfigured. The HTTP API moves to OIDC-only
authentication with a bundled local IdP (Dex) in the development compose stack
so the User Story 1 fifteen-minute timeline holds. The graph store becomes
provider-selectable with a built-in null backend as the default so a minimal
deployment requires only an LLM key, a vector store, and the local IdP. An
automated vendor-neutrality audit (proprietary-name + non-English-text +
hardcoded-host scanner) gates merges. The existing Helm chart is reworked to
follow standard chart conventions and to expose every MCP flag through
`values.yaml`. The technical approach is *port and sanitize, not rebuild* —
the upstream test suite (~225 tests) moves over and continues to pass after
each sanitization pass.

## Technical Context

**Language/Version**: Python 3.11+ for backend (`opsrag/` package, MCP servers,
investigation agent, Slack bot, eval harness, audit script); TypeScript ≥ 5 on
Node ≥ 20 for the React UI; Bash 5+ for the audit script and compose-stack
wrapper.

**Primary Dependencies** (carried forward from upstream `pyproject.toml`):
- **Agent / orchestration**: `langgraph>=1.1`,
  `langgraph-checkpoint-postgres>=2.0`
- **HTTP**: `fastapi>=0.115`, `uvicorn[standard]>=0.34`, `httpx>=0.28`
- **Config / validation**: `pydantic>=2.10`, `pydantic-settings>=2.6`,
  `pyyaml>=6.0`
- **Auth**: `pyjwt[crypto]>=2.8` (OIDC JWT verification — generalised from
  upstream's Pomerium-specific path)
- **LLM providers (optional extras)**: `anthropic>=0.45`, `openai>=1.60`,
  `google-cloud-aiplatform>=1.80` (`vertex` extra), `boto3>=1.35` (`bedrock`
  extra)
- **Stores**: `qdrant-client>=1.13`, `neo4j>=5.25` (only when `neo4j` selected
  as graph provider), `psycopg[binary,pool]>=3.2`, `asyncpg>=0.30`
  (`pgvector` extra)
- **Embeddings (optional)**: `fastembed>=0.5` (`fastembed` extra),
  `sentencepiece>=0.2` (`vertex` extra)
- **Live infra (MCPs)**: `kubernetes_asyncio>=32.0`, `slack-sdk>=3.27`,
  `apscheduler>=3.10`
- **Observability**: `arize-phoenix-otel>=0.10`,
  `arize-phoenix-evals>=0.20`, OpenInference instrumentations for LangChain,
  Anthropic, VertexAI
- **Dev / eval (optional)**: `pytest>=8`, `pytest-asyncio>=0.24`, `ruff>=0.8`,
  `mypy>=1.13`, `deepeval>=2.0`

**Storage**:
- **Vectors**: Qdrant by default (`vector_store.provider: qdrant`); pgvector
  alternative selectable via provider switch.
- **Graph**: Null backend by default (`knowledge_graph.provider: null` —
  satisfies the interface with empty results, per FR-019); Neo4j optional via
  `knowledge_graph.provider: neo4j`.
- **Sessions / LangGraph checkpoints / agent memory**: PostgreSQL (via
  `langgraph-checkpoint-postgres` + custom session/memory tables).
- **Indexed files / Q&A cache / corrections / feedback / usage**: Postgres
  tables (carried forward from upstream).

**Testing**:
- **Unit**: `pytest` + `pytest-asyncio` (`asyncio_mode = auto`); existing 225
  cases migrate wholesale and must pass after each sanitization pass.
- **Integration**: Per-MCP integration tests against in-process fake
  backends — no live network from CI (FR-012). Existing upstream MCP fakes
  (e.g. `tests/unit/test_cartography_mcp.py`, `test_slack_mcp.py`) move over.
- **Contract**: New contract tests for the OpenAPI surface, `config.yaml`
  schema (Pydantic validation), `values.yaml` schema (`helm lint` +
  `helm template`), audit-script CLI behaviour.
- **Eval (regression)**: Phoenix / DeepEval golden-set runs gated by
  numeric thresholds; CI step on every PR that touches prompts, retrievers,
  or agent graphs.
- **UI**: Vitest unit + Playwright smoke (existing upstream config preserved
  where present; otherwise added).
- **Lint / type**: `ruff` + `mypy` (strict on changed files).

**Target Platform**:
- **Local development / evaluation**: Docker Compose on macOS / Linux
  (developer laptops). Bundled compose stack runs: backend, UI, Qdrant,
  Postgres, Dex (OIDC), and Phoenix.
- **Production**: Kubernetes 1.27+ via the Helm chart at
  `deploy/helm/opsrag/`. Tested against kind 0.22 and a recent managed
  control plane (GKE/EKS-equivalent).

**Project Type**: Multi-component application — Python backend + React
single-page UI + Slack-bot worker + Helm chart + sample corpus, all in one
repository.

**Performance Goals**:
- `/healthz` ready within 30 s of `docker compose up` from a warm cache
  (SC-005).
- Non-investigation `/query` p95 < 3 s on a developer laptop with default
  configuration (vector-only retrieval, no graph, no MCPs).
- Investigation-agent end-to-end p95 ≤ 60 s for a five-hop investigation
  against fakes.
- `helm lint` + `helm template deploy/helm/opsrag` complete in < 5 s in CI.

**Constraints**:
- Container MUST run as non-root (FR-010).
- Zero live external-network access in CI — every MCP integration test uses a
  fake backend.
- No proprietary identifiers, internal hostnames, or non-English text in
  shipped artefacts (FR-001 / FR-002 / FR-007 / SC-002 / SC-003).
- Fifteen-minute clone-to-first-answer for a new evaluator (FR-011 / SC-001).
- Default `config.yaml` boots a healthy service with zero MCPs enabled
  (FR-009).

**Scale/Scope**:
- ~80 backend Python source files ported from upstream.
- 14 MCP integrations placed behind feature flags.
- ~225 existing tests migrate; new tests added for OIDC, audit script, Helm
  values wiring, null graph backend.
- 1 React SPA (~30 components in upstream — exact count subject to
  inventory).
- Initial expected deployment scale: 30–50 concurrent SRE users per
  installation.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Constitution v1.1.0 (`.specify/memory/constitution.md`) defines six
principles. Evaluation:

| # | Principle | Compliance | Notes |
|---|---|---|---|
| I  | Vendor-Neutral by Default (NON-NEGOTIABLE) | **PASS** | FR-001 / FR-002 / FR-007 capture the substantive obligations; FR-014 mandates an automated audit; SC-002 / SC-003 quantify success. No exceptions requested. |
| II | Pluggable Integrations via Feature Flags | **PASS (with documented clarification)** | All 14 MCPs gated by `mcp.<name>.enabled`, default `false` (FR-003). Graph store is **provider-selected**, not MCP-flagged (FR-019, clarified 2026-05-27 / 28) — this preserves the spirit of "minimal default footprint" while keeping the agent graph topology stable. Vector store and session store follow the same provider-selection pattern; this convention is documented in `research.md`. |
| III | Cloud-Native, Container-First | **PASS** | Existing `Dockerfile` is multi-stage; existing Helm chart at `deploy/helm/opsrag/` will be brought into conformance with chart conventions and re-linted under FR-005 / FR-006. |
| IV  | Test & Eval Discipline | **PASS** | Upstream's ~225-test suite ports wholesale; FR-012 makes per-MCP integration tests against fakes mandatory; Phoenix/DeepEval eval harness is included in the port (Q2 clarification). |
| V   | Observability & Secret Hygiene | **PASS** | OIDC-only auth (FR-016 — supersedes any built-in API-key concept), env-var secret loading (FR-008), Phoenix + OpenInference instrumentations carry forward, `/healthz` + `/readyz` required by constitution. |
| VI | Context-Driven Engine (NON-NEGOTIABLE) | **PASS (in progress)** | Added in constitution v1.1.0 (2026-05-28) after the Phase 2 sanitization sweep revealed the "fake-abstract" failure mode (agents substituted one set of org names for another, leaving the engine coupled to the original architecture). The contract for `DeploymentContext` lives at `contracts/deployment-context.md`; the inventory of remaining defects is at `inventory.md`. T029 ships the schema (this PR); a separate prompt-abstraction task carves the remaining work out of T055-T056. The cross-org test gates merges through the audit; full structural enforcement (Check 4) is deferred to Phase 6 final-sweep. |

**Initial gate result**: PASS. No principle violations or exceptions to
record in *Complexity Tracking* below.

## Project Structure

### Documentation (this feature)

```text
specs/001-port-opsrag-opensource/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   ├── http-api.md
│   ├── config-schema.md
│   ├── helm-values-schema.md
│   └── audit-cli.md
├── checklists/
│   └── requirements.md  # Created by /speckit-specify
└── tasks.md             # Phase 2 — created by /speckit-tasks
```

### Source Code (repository root)

```text
opsrag/                          # Python package
├── agent/                       # Core agent graph (LangGraph)
├── agents/                      # Investigation agent and other specialised graphs
│   └── investigation/
├── api/                         # FastAPI surface (HTTP endpoints, SSE, webhooks)
├── auth/                        # OIDC Bearer-token verification (JWKS, claims)
├── chunkers/                    # FixedSize, ParentChild chunking strategies
├── db/                          # Postgres pool, migrations
├── embedders/                   # OpenAI, FastEmbed, Vertex, Bedrock providers
├── eval/                        # Phoenix + DeepEval golden harness
├── extractors/                  # Entity extractors (LLM, rule-based, hybrid)
├── graphstores/                 # Null (default), Neo4j providers — FR-019
├── ingestion/                   # Indexing pipeline + progress tracking
├── interfaces/                  # Protocol interfaces (plugin contracts)
├── llms/                        # Anthropic, OpenAI, Vertex, Bedrock providers
├── mcp/                         # 14 MCP integrations, each behind enabled flag
│   ├── cartography.py
│   ├── cloudflare.py
│   ├── cloudsql.py
│   ├── code.py
│   ├── datadog.py
│   ├── elasticsearch.py
│   ├── gitlab.py
│   ├── knowledge.py
│   ├── kubernetes.py
│   ├── prometheus.py
│   ├── rootly.py
│   ├── runbooks.py
│   ├── slack.py
│   └── tool_cache.py
├── mcp_server/                  # MCP server façade exposing the registry
├── memory/                      # InMemory + Postgres memory stores
├── observability/               # Console + Phoenix providers, OTel wrapper
├── parsers/                     # Markdown, Runbook, Postmortem, Terraform, Helm, K8s, Alert
├── rerankers/                   # NoOp, Cohere, FastEmbed cross-encoder
├── runbooks/                    # Runbook loader + store
├── scheduler/                   # APScheduler-driven indexing jobs
├── scm/                         # GitLab, GitHub, GitClone, LocalFS source backends
├── sessions/                    # InMemory + Postgres session stores
├── slack_bot/                   # Socket-mode Slack chatbot
├── sources/                     # Source-of-truth registry
├── vectorstores/                # Qdrant (default), pgvector
├── config.py                    # Pydantic v2 settings; MCP enabled flags live here
├── factory.py                   # Provider DI factory (sanitised — no Vietnamese)
└── ... (correction_store, feedback_store, indexing_tracker, qa_cache, tokenization, usage)

tests/
├── contract/                    # OpenAPI, config schema, values.yaml, audit CLI
├── integration/                 # Per-MCP tests against in-process fakes
└── unit/

ui/                              # React SPA (Vite)
├── src/
└── public/

deploy/
├── helm/
│   └── opsrag/                  # First-class Helm chart — FR-005 / FR-006
│       ├── Chart.yaml
│       ├── values.yaml          # Includes mcp.<name>.enabled for all 14 MCPs
│       ├── values.schema.json   # NEW — schema validation
│       ├── templates/
│       │   ├── _helpers.tpl
│       │   ├── deployment.yaml
│       │   ├── ui-deployment.yaml
│       │   ├── slackbot-deployment.yaml   # NEW — separate workload
│       │   ├── service.yaml
│       │   ├── ingress.yaml
│       │   ├── configmap.yaml
│       │   ├── secret.yaml
│       │   ├── serviceaccount.yaml         # NEW — non-root SA
│       │   ├── networkpolicy.yaml          # NEW — egress allowlist
│       │   ├── poddisruptionbudget.yaml    # NEW
│       │   ├── hpa.yaml                    # NEW (optional)
│       │   ├── NOTES.txt                   # NEW
│       │   └── tests/
│       │       └── test-connection.yaml    # `helm test` hook
│       └── README.md                       # NEW — chart docs
└── compose/
    ├── docker-compose.yaml      # Backend + UI + Qdrant + Postgres + Dex + Phoenix
    └── dex/
        └── config.yaml          # Bundled local OIDC for FR-017

scripts/
├── audit-vendor-neutrality.sh   # FR-014 audit (proprietary-name + non-English + hardcoded-host)
├── seed-sample-corpus.sh        # Index bundled samples for User Story 1
└── helm-lint.sh                 # CI helper

samples/                         # Synthetic sample corpus for the 15-min walkthrough
├── runbooks/
├── postmortems/
└── manifests/

docs/                            # Adopter-facing docs (no proprietary content)
├── architecture.md
├── mcp-integrations.md
├── helm-chart.md
└── auth.md

.env.example                     # Lists every env var (FR-008)
config.yaml                      # Default — zero MCPs, null graph backend
config-example.yaml              # Annotated example with every flag visible
Dockerfile                       # Multi-stage; non-root runtime user
docker-entrypoint.sh
pyproject.toml                   # SPDX: Apache-2.0
uv.lock
LICENSE                          # Apache-2.0 full text (FR-013)
NOTICE                           # Required by Apache-2.0
CHANGELOG.md                     # Fork entry (FR-015)
README.md                        # Bring-up walkthrough satisfying FR-011
CONTRIBUTING.md
SECURITY.md
CODE_OF_CONDUCT.md
```

**Structure Decision**: Single-repo, multi-component layout — Python backend
(`opsrag/`), React UI (`ui/`), Helm chart (`deploy/helm/opsrag/`), local
compose stack (`deploy/compose/`), and synthetic sample corpus
(`samples/`) — chosen because the upstream project is already organised this
way, the components share configuration and contracts, and a monorepo keeps
the audit script's blast radius (constitutional Principle I) trivially
defined as "everything under HEAD that ships". Workload-level separation
(backend / UI / slack-bot) lives at the Helm chart level rather than at the
source-tree level: three Deployments, one chart, one image
(role-multiplexed by entrypoint) or three images (final choice deferred to
`research.md`).

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No principle violations to record. The graph-store provider-selection model
(FR-019) differs in mechanism from the MCP enabled-flag model (FR-003) but is
explicitly compatible with Principle II — both achieve "zero external
dependencies by default" — and the distinction is documented in the
`Integration` entity note and in `research.md`. Treating it as a violation
would be a category error.

## Post-design re-check

Walked each constitutional principle a second time against
`research.md`, `data-model.md`, `contracts/`, and `quickstart.md`:

| # | Principle | Re-check result | Evidence |
|---|---|---|---|
| I | Vendor-Neutral by Default | **PASS** | Audit script and exception list defined in `contracts/audit-cli.md`. Allowed exceptions: `CHANGELOG.md` (one historical entry), `samples/` (synthetic Acme-Notes placeholders), `specs/` and `.specify/` (Spec-Kit project-management artefacts that necessarily describe what is being removed; not runtime artefacts). Quickstart uses only generic placeholders (`example.com`, Dex local issuer). |
| II | Pluggable Integrations | **PASS** | `config-schema.md` enforces every MCP via `MCPConfigBlock`, `helm-values-schema.md` mandates the 14 keys exactly, contract test `test_helm_values_covers_all_mcps.py` fails the build on drift. Fail-fast error code `MCP_MISCONFIGURED:<name>:<env>` defined. Graph-store null backend is a first-class shipped implementation, not a special case. |
| III | Cloud-Native, Container-First | **PASS** | Chart at `deploy/helm/opsrag/` documented end-to-end including `values.schema.json`, NetworkPolicy, PDB, NOTES.txt, `helm test` hook. Multi-stage Dockerfile + distroless runtime + UID 1000 confirmed in `research.md` §8. |
| IV | Test & Eval Discipline | **PASS** | 14 named contract tests across the four contract documents. Per-MCP fake-backend strategy decided in `research.md` §5. Eval harness explicitly in scope (Q2 clarification). |
| V | Observability & Secret Hygiene | **PASS** | OIDC-only auth (`contracts/http-api.md`), token never logged, `sub` claim used only for usage attribution. Secrets resolved via env vars wrapped in `SecretStr` (`contracts/config-schema.md`). Phoenix + OpenInference instrumentations carry forward. `/healthz` + `/readyz` documented in HTTP contract. |
| VI | Context-Driven Engine | **PASS (in progress)** | Added in constitution v1.1.0 after Phase 2 surfaced the fake-abstract failure mode. `contracts/deployment-context.md` defines the `DeploymentContext` schema; `inventory.md` records the Categories A–D defects the cleanup left and the migration plan. T029 ships the schema; the four-tier model (config / prompt / corpus / skill) is encoded in the principle text. Full structural-audit enforcement (Check 4 in `audit-vendor-neutrality.sh`) is deferred to Phase 6 per the contract. |

**Result**: PASS, no new exceptions, no Complexity Tracking entries.
Principle VI was added post-plan-ratification; its compliance row is
maintained alongside I–V going forward.
