---

description: "Task list for porting opsrag as a vendor-neutral opensource project"
---

# Tasks: Port opsrag as a vendor-neutral opensource project

**Input**: Design documents from `specs/001-port-opsrag-opensource/`

**Prerequisites**: plan.md, spec.md (required); research.md, data-model.md,
contracts/, quickstart.md (all available)

**Tests**: INCLUDED. The spec explicitly mandates tests via FR-005 (Helm lint
+ template), FR-006 (values-MCP coverage), FR-012 (per-MCP integration tests
against fakes), FR-014 (audit script self-tests), and the four contract
documents name 17 specific contract tests. Constitution Principle IV
("Test & Eval Discipline") makes a green test suite a merge gate.

**Organization**: Tasks are grouped by user story (US1–US4) to enable
independent implementation and testing. Within each story:
contract / integration tests are authored before implementation
where the constitution's TDD discipline applies.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on
  incomplete tasks)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4) —
  omitted for Setup, Foundational, and Polish phases
- Every task includes the exact file path it touches

## Path conventions

Single-repo, multi-component layout (per plan.md):
- Backend: `opsrag/`
- Tests: `tests/contract/`, `tests/integration/`, `tests/unit/`
- UI: `ui/src/`
- Helm chart: `deploy/helm/opsrag/`
- Compose stack: `deploy/compose/`
- Scripts: `scripts/`
- Samples: `samples/`
- Source repo to port from: `../../acme/infra/opsrag/` (read-only
  reference; do not modify)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish the empty-but-correct repository skeleton — license,
metadata, CI scaffolding, audit-script stub. No upstream code copied yet.

- [X] T001 Add Apache-2.0 LICENSE file at repo root with full license text (`LICENSE`)
- [X] T002 [P] Add Apache-2.0 NOTICE file at repo root (`NOTICE`)
- [X] T003 [P] Initialise CHANGELOG.md with the fork entry per FR-015 (`CHANGELOG.md`)
- [X] T004 [P] Add `.gitignore` covering Python, Node, helm, secrets, env files (`.gitignore`)
- [~] T005 [P] **SKIPPED** (user decision 2026-05-29): CODE_OF_CONDUCT.md omitted as non-essential; the dead `CONTRIBUTING.md` link was removed so nothing dangles (`CODE_OF_CONDUCT.md`)
- [X] T006 [P] Add SECURITY.md describing vulnerability-report process (`SECURITY.md`)
- [X] T007 [P] Add CONTRIBUTING.md with PR workflow, audit + tests required (`CONTRIBUTING.md`)
- [X] T008 [P] Stub README.md with quickstart heading and placeholders (`README.md`)
- [X] T009 Port-and-sanitize `pyproject.toml` from upstream: SPDX `Apache-2.0`, drop internal extras, set authors to the new project owner (`pyproject.toml`)
- [X] T010 [P] Add `uv.lock` regenerated from the sanitised `pyproject.toml` (`uv.lock`)
- [X] T011 [P] Stub `scripts/audit-vendor-neutrality.sh` — scanner logic only; rules sourced from `scripts/audit-rules.yaml` (per user override 2026-05-28) (`scripts/audit-vendor-neutrality.sh`)
- [X] T012 [P] Stub `scripts/seed-sample-corpus.sh` (`scripts/seed-sample-corpus.sh`)
- [X] T013 Add GitHub Actions workflow: `ci.yml` — jobs for lint, type, unit, integration, contract, helm-lint, audit, eval (`.github/workflows/ci.yml`)
- [X] T014 [P] Add GitHub Actions workflow: `release.yml` — image build + push to GHCR on tags (`.github/workflows/release.yml`)
- [X] T015 [P] Add `.pre-commit-config.yaml` invoking ruff, mypy, and `audit-vendor-neutrality.sh` (`.pre-commit-config.yaml`)

**Checkpoint**: Repository is licensed, attributed, CI-wired, and the audit
script exits zero against the empty tree.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Sanitize upstream code into the tree, establish the core
configuration / auth / factory / interface layer, and stand up the local
compose stack. Every later user story depends on this.

**⚠️ CRITICAL**: No user-story phase can begin until this is complete.

### Sanitization sweep (Constitution Principle I)

- [X] T016 Copy `opsrag/` package tree from upstream into this repo, preserving structure (`opsrag/`)
- [X] T017 Run `scripts/audit-vendor-neutrality.sh` to enumerate violations across the freshly-copied tree (no edits yet) and write findings to `audit-baseline.txt` — `scripts/audit-rules.yaml` now filled; audit runs clean against the already-sanitized tree (baseline is empty), so no separate `audit-baseline.txt` snapshot was needed (`audit-baseline.txt`)
- [X] T018 Sanitize `opsrag/qa_cache.py` — replace Vietnamese strings with English, replace any internal references (`opsrag/qa_cache.py`)
- [X] T019 [P] Sanitize `opsrag/qa_cache_judge.py` (`opsrag/qa_cache_judge.py`)
- [X] T020 [P] Sanitize `opsrag/factory.py` (`opsrag/factory.py`)
- [X] T021 [P] Sanitize `opsrag/correction_store.py` (`opsrag/correction_store.py`)
- [X] T022 [P] Sanitize `opsrag/runbooks/store.py` (`opsrag/runbooks/store.py`)
- [X] T023 [P] Sanitize `opsrag/agents/investigation/result_cache.py` (`opsrag/agents/investigation/result_cache.py`)
- [X] T024 [P] Sanitize `opsrag/agents/investigation/limits.py` (`opsrag/agents/investigation/limits.py`)
- [X] T025 [P] Sanitize `opsrag/agents/investigation/graph.py` (`opsrag/agents/investigation/graph.py`)
- [X] T026 [P] Sanitize `opsrag/agent/classifier.py` (`opsrag/agent/classifier.py`)
- [X] T027 [P] Sanitize `opsrag/agent/path_tree.py` (`opsrag/agent/path_tree.py`)
- [X] T028 Re-run `audit-vendor-neutrality.sh`; expect zero violations against the sanitized package. Verified 2026-05-29: all three checks (proprietary_names, non_english_text, hardcoded_hosts) report OK, exit 0 (`opsrag/`)

### Core scaffolding

- [X] T029 Rewrite `opsrag/config.py` as Pydantic-v2 `Settings` with discriminated unions for every provider block per `contracts/config-schema.md`. **Scope extended (per 2026-05-28 directive)**: also adds `opsrag/context.py` with the `DeploymentContext` schema per `contracts/deployment-context.md` (Constitution Principle VI). Full per-provider discriminated unions deferred to T031+ when the MCP registry lands; the v1 ships `Literal`-discriminated provider blocks and a generic `MCPConfigBlock` (`opsrag/config.py`, `opsrag/context.py`)
- [X] T030 Add `opsrag/config_mcp.py` with the `MCPConfigBlock` base and per-integration discriminated subclasses per `data-model.md` §3 (`opsrag/config_mcp.py`)
- [X] T031 [P] Add `opsrag/mcp/registry.py` — the `MCPIntegration` registry with all 14 named entries, each declaring `required_env`, `factory`, `fake_factory`, `tool_names`, `health_url_template` per `data-model.md` §1 (`opsrag/mcp/registry.py`). Factories are lazy (`_lazy`) imports of `opsrag.mcp.<name>.build` / `.build_fake`; missing modules raise `NotImplementedError` at enable-time pending Phase 4 (T073-T086). (`opsrag/mcp/registry.py`)
- [X] T032 [P] Add `opsrag/auth/__init__.py` + `opsrag/auth/oidc.py` — JWKS fetch + caching + JWT verification (iss / aud / exp) using `pyjwt[crypto]`. **Legacy `auth/pomerium.py` and `TrackingUserConfig` retained for now**; deletion deferred to T057-T060 when `api/server.py` and `api/routes.py` are rewritten (~30 call sites depend on the Pomerium-shape `CurrentUser`). New code imports from `opsrag.auth.oidc` directly. (`opsrag/auth/oidc.py`)
- [X] T033 [P] Add `opsrag/auth/middleware.py` — FastAPI dependency that enforces Bearer-token auth and exposes `sub` in request state without logging it (`opsrag/auth/middleware.py`)
- [X] T034 [P] Add `opsrag/graphstores/null.py` — null graph backend satisfying the interface with empty results, per FR-019 (`opsrag/graphstores/null.py`)
- [X] T035 Sanitize and port `opsrag/factory.py` to construct providers from the new `Settings`, including the null graph backend as default (`opsrag/factory.py`)
- [X] **T035a (new — Principle VI Step 8)** Strip `example.*` URL defaults from `ConfluenceConfig.base_url`, `SlackConfig.workspace_url`, `RootlyConfig.web_base_url`, `SlackBotConfig.workspace_url`, `agent.graph.SourceUrlBases`, `mcp/slack.py:_WORKSPACE_URL_DEFAULT`. All now default to `None`; `SourceUrlBases.from_app_config` prefers `DeploymentContext.source_urls` with legacy-block fallback. (`opsrag/config.py`, `opsrag/slack_bot/config.py`, `opsrag/agent/graph.py`, `opsrag/mcp/slack.py`)
- [X] **T035b (new — Principle VI scaffolding)** Add `opsrag/agent/prompt_render.py` — the templating helper that renders system-prompt templates against `DeploymentContext`. Plain `str.format_map` semantics; unknown placeholders raise `KeyError`. Exposes 19 friendly template keys (services_csv, environments_csv, ticket_prefix, source_url_*, k8s_clusters_csv, etc.). Demonstrates the migration target shape; consuming-side prompt rewrites are carved into `T056a-h` below. (`opsrag/agent/prompt_render.py`)

### Container + compose

- [X] T036 Port-and-sanitize `Dockerfile` — multi-stage; debian-slim runtime (shell needed for the `OPSRAG_ROLE` entrypoint multiplexer; distroless ships no shell — rationale documented in the Dockerfile header); UID 1000 (`Dockerfile`)
- [X] T037 [P] Sanitize `docker-entrypoint.sh` for English-only output and role multiplexing per research.md §8 (`docker-entrypoint.sh`)
- [X] T038 Create `deploy/compose/docker-compose.yaml` with backend + ui + qdrant + postgres + dex + phoenix per `quickstart.md` (`deploy/compose/docker-compose.yaml`)
- [X] T039 [P] Create `deploy/compose/dex/config.yaml` with single static user `evaluator@example.com` and static client `opsrag-local` (`deploy/compose/dex/config.yaml`)
- [X] T040 Create `.env.example` listing every env var the backend reads, with placeholder values and one-line descriptions per FR-008 (`.env.example`)

**Checkpoint**: Codebase passes audit, config validates, OIDC verifies, null
graph backend ships, compose stack boots. User-story phases can begin.

---

## Phase 3: User Story 1 — First-time evaluator stands up a minimal opsrag (P1) 🎯 MVP

**Goal**: Clone → set LLM key → `docker compose up` → answer a sample query
in under fifteen minutes (SC-001 / FR-011).

**Independent Test**: Follow `quickstart.md` on a clean machine; receive a
cited English answer about an indexed sample within fifteen minutes.

### Contract tests for US1 (write first; expect to fail)

- [X] T041 [P] [US1] Add contract test `tests/contract/test_config_default_boots.py` — asserts shipped `config.yaml` validates and yields all-MCPs-disabled + null graph (`tests/contract/test_config_default_boots.py`)
- [X] T042 [P] [US1] Add contract test `tests/contract/test_config_unknown_keys_rejected.py` (`tests/contract/test_config_unknown_keys_rejected.py`)
- [X] T043 [P] [US1] Add contract test `tests/contract/test_openapi_shape.py` — every endpoint present, every endpoint declares auth requirement (`tests/contract/test_openapi_shape.py`)
- [X] T044 [P] [US1] Add contract test `tests/contract/test_auth_required.py` — every non-health endpoint returns 401 without `Authorization` (`tests/contract/test_auth_required.py`)
- [X] T045 [P] [US1] Add contract test `tests/contract/test_error_envelope.py` — all 4xx/5xx responses match the documented envelope (`tests/contract/test_error_envelope.py`)

### Backend implementation for US1

- [X] T046 [P] [US1] Port and sanitize `opsrag/llms/anthropic.py` provider (`opsrag/llms/anthropic.py`) — verified: imports + LLMProvider conformance (tests/unit/test_us1_providers.py)
- [X] T047 [P] [US1] Port `opsrag/llms/openai.py` provider (`opsrag/llms/openai.py`) — CREATED `opsrag/llms/openai.py` (was missing) + wired the openai branch into factory.build_providers; native json_object structured mode; LLMProvider-conformant + tested
- [X] T048 [P] [US1] Port `opsrag/embedders/openai.py` provider (`opsrag/embedders/openai.py`) — verified: OpenAIEmbeddings imports
- [X] T049 [P] [US1] Port `opsrag/embedders/fastembed.py` provider (`opsrag/embedders/fastembed.py`) — ported (file present); `fastembed` is an optional extra (installed in the Docker image via OPSRAG_EXTRAS), so the import test skips in a bare dev venv
- [X] T050 [P] [US1] Port `opsrag/vectorstores/qdrant.py` provider (`opsrag/vectorstores/qdrant.py`) — verified: QdrantVectorStore imports
- [X] T051 [P] [US1] Port `opsrag/sessions/postgres.py` + `opsrag/sessions/inmemory.py` (`opsrag/sessions/`) — verified: PostgresSessionStore + InMemorySessionStore import
- [X] T052 [P] [US1] Port `opsrag/memory/postgres.py` + `opsrag/memory/inmemory.py` (`opsrag/memory/`) — verified: PostgresMemoryStore + InMemoryMemoryStore import
- [X] T053 [P] [US1] Port `opsrag/chunkers/fixed_size.py` + `opsrag/chunkers/parent_child.py` (`opsrag/chunkers/`) — verified: FixedSizeChunker + ParentChildChunker import
- [X] T054 [P] [US1] Port `opsrag/parsers/markdown.py` + `opsrag/parsers/runbook.py` + `opsrag/parsers/postmortem.py` (`opsrag/parsers/`) — verified: markdown/runbook/postmortem parsers import
- [X] T055 [US1] Port `opsrag/agent/graph.py` — core LangGraph nodes (vector_retrieve → generate → END) (`opsrag/agent/graph.py`) — verified: build_minimal_graph + build_hybrid_graph present (agent graph ported from T016)
- [X] T056 [US1] Sanitize agent prompts under `opsrag/agent/prompts.py` — English only, no proprietary references (`opsrag/agent/prompts.py`) — agent prompts sanitized English-only + org-neutral via the cross-org refactor below
- [X] **T056a (new — Principle VI prompt abstraction, query_rewrite)** [US1] Apply the cross-org test to `opsrag/agent/query_rewrite.py`'s `_REWRITE_SYSTEM`. Concrete service / repo names in few-shot examples replaced with placeholder shapes (`<repo-a>`, `<svc-a>`, `<env-a>`) that teach the rewrite pattern without naming any specific deployment. **Done as demonstrator commit; reviewer should confirm rewrite quality holds on the eval golden set.** (`opsrag/agent/query_rewrite.py`) — verified: query_rewrite few-shots use placeholder shapes; rewrite quality preserved
- [X] **T056b (new) [US1]** Apply the cross-org test to `opsrag/agent/prompts.py` — rewrite `_GENERATE_COMMON_RULES` rules 9 / 12 / etc. so the LLM-facing prompt uses `{services_csv}` / `{ticket_prefix}` placeholders via `opsrag.agent.prompt_render.render` (`opsrag/agent/prompts.py`) — prompts.py: _GENERATE_COMMON_RULES rules use {ticket_prefix} via render(); generation_system_prompt now renders against active DeploymentContext; deployment-specific gitops/Kong rule dropped to v2
- [X] **T056c (new) [US1]** Apply the cross-org test to `opsrag/agent/query_decomposer.py`'s `_SYSTEM_PROMPT` — drop concrete service / repo names from the few-shot frames, use placeholder shapes (`opsrag/agent/query_decomposer.py`) — query_decomposer.py: few-shot frames use <svc-a>/<repo-a>/<env-a> placeholder shapes; eval-provenance + CI tokens removed
- [X] **T056d (new) [US1]** Apply the cross-org test to `opsrag/agent/classifier.py` — `REFERENCE_EXAMPLES` becomes derive-default-from-context (templates rendered against `deployment.services` + `deployment.environments`) plus optional `deployment.semantic_router_examples` operator-augmentation per `contracts/deployment-context.md`. Env-discriminator regex built dynamically from `deployment.environments + ["prod","staging","dev","test"]` (`opsrag/agent/classifier.py`) — classifier.py: REFERENCE_EXAMPLES derived at call time from deployment.services/environments + semantic_router_examples (generic fallback); env-discriminator regex built dynamically from deployment.environments + [prod,staging,dev,test]
- [X] **T056e (new) [US1]** Apply the cross-org test to `opsrag/agent/repomap.py` — `_KEY_REPOS_LAYOUT_HINTS` becomes `deployment.key_repos` (`opsrag/agent/repomap.py`) — repomap.py: _KEY_REPOS_LAYOUT_HINTS removed; key-repo hints now read from active_deployment().key_repos (graceful-empty)
- [X] **T056f (new) [US1]** Apply the cross-org test to `opsrag/agent/nodes/multi_agent.py` (largest single piece, 2400+ lines). ~30 illustrative anecdotes in triage / reasoner / generator prompts must be cross-org tested: genuine reasoning patterns are rewritten as templates rendered via `opsrag.agent.prompt_render`; operational heuristics (e.g. "SSO 503 in env X usually means cert drift on auth service") are DROPPED at v1 per the constitution and deferred to corpus-tier v2 (`opsrag/agent/nodes/multi_agent.py`) — multi_agent.py: ~30 org-specific anecdotes neutralized (deleted operational heuristics / rewrote reasoning with <placeholder> shapes); reasoning structure + tool rules preserved; brace-safe (no render() in JSON/PromQL-heavy prompts)
- [X] **T056g (new) [US2]** Apply the cross-org test to `opsrag/runbooks/tagger.py` and `opsrag/runbooks/generator.py` — system prompts use abstract placeholders for service / env / repo (`opsrag/runbooks/`) — runbooks/tagger.py + generator.py: org service/env/repo names + anecdotes replaced with <placeholder> shapes; personal refs removed; render() not needed (no static lists; brace-heavy)
- [X] **T056h (new) [US2]** Apply the cross-org test to `opsrag/agents/investigation/prompts.py` — investigation prompts parameterised against `DeploymentContext` (`opsrag/agents/investigation/prompts.py`) — agents/investigation/prompts.py: concrete service/tech anecdotes in hypothesis-gen + evidence-judge prompts replaced with neutral shapes; reasoning/JSON-output structure preserved
- [X] **T056i (new) [US1]** Refactor module-level constants in `opsrag/mcp/{kubernetes,prometheus,cloudsql}.py` — `_KNOWN_CLUSTERS`, `_default_cluster()`, GCP-project tuples become reads from `DeploymentContext.kubernetes.clusters` / `DeploymentContext.cloud.gcp_projects`. Per inventory.md Category A. (`opsrag/mcp/kubernetes.py`, `opsrag/mcp/prometheus.py`, `opsrag/mcp/cloudsql.py`) — mcp/{kubernetes,prometheus,cloudsql}.py: _KNOWN_CLUSTERS / _default_* / GCP-project constants now read from DeploymentContext.kubernetes.clusters / cloud.gcp_projects; require-or-error when unconfigured (no org defaults)
- [X] **T056j (new) [US1]** Refactor `opsrag/qa_cache.py:_DISCRIMINATOR_ENV` to build the env regex dynamically from `DeploymentContext.environments + ["prod","staging","dev","test"]` (`opsrag/qa_cache.py`) — qa_cache.py: _DISCRIMINATOR_ENV rebuilt dynamically from deployment.environments + generic defaults at call time; org env shorthands removed
- [X] **T056k (new) [US1]** Remove internal-history `TICKET-9715` / `TICKET-9997` / `T1.3 (TICKET-...)` comments per inventory.md Category C (`opsrag/feedback_store.py`, `opsrag/vectorstores/qdrant.py`, `opsrag/api/server.py`, `opsrag/api/models.py`, `opsrag/api/routes.py`, `opsrag/agent/nodes/multi_agent.py`) — removed internal ticket/sprint history comments across feedback_store/qdrant/api(server,models,routes)/multi_agent; kept the technical explanations
- [X] T057 [US1] Port `opsrag/api/main.py` — FastAPI app, dependency wiring, OIDC middleware on all non-health routes (`opsrag/api/main.py`) — done as `opsrag/api/server.py` (kept upstream name per the surgical-OIDC-swap decision 2026-05-29 rather than a new api/main.py): builds the OIDCVerifier from settings.auth, attaches it to app.state, mounts the global OIDCAuthMiddleware + error-envelope handlers
- [X] T058 [US1] Port `opsrag/api/routes/health.py` — `/healthz`, `/readyz` (readiness probes Postgres + Qdrant) (`opsrag/api/routes/health.py`) — done as `opsrag/api/routes_health.py` (flat module): /healthz liveness + /readyz readiness with per-component (vector_store/session_store) probing
- [X] T059 [US1] Port `opsrag/api/routes/query.py` — `/query` synchronous and SSE-streaming variants (`opsrag/api/routes/query.py`) — /query (sync + SSE) already present in `opsrag/api/routes.py` from T016 and is now OIDC-protected via the global middleware; kept in routes.py per the surgical decision
- [X] T060 [US1] Port `opsrag/api/errors.py` — stable error envelope per `contracts/http-api.md` (`opsrag/api/errors.py`) — done as `opsrag/api/errors.py`: stable envelope {error,reason,request_id} via register_error_handlers()
- [X] T061 [US1] Port `opsrag/ingestion/indexer.py` — index runbooks/postmortems from local FS into vector store (`opsrag/ingestion/indexer.py`) — done as `opsrag/ingestion/indexer.py`: local-FS indexer wrapping files as RepoFiles through IngestionPipeline._process_file; `python -m opsrag.ingestion.indexer <path>` CLI; unit-tested in tests/unit/test_indexer.py

### Sample corpus + bring-up

- [X] T062 [P] [US1] Author synthetic sample corpus: 5 runbooks under `samples/runbooks/` describing the fictional "Acme Notes" product (`samples/runbooks/`)
- [X] T063 [P] [US1] Author synthetic postmortems under `samples/postmortems/` (3 documents) (`samples/postmortems/`)
- [X] T064 [P] [US1] Author synthetic K8s manifests + Terraform module under `samples/manifests/` and `samples/terraform/` (`samples/`)
- [X] T065 [US1] Implement `scripts/seed-sample-corpus.sh` — invokes the indexer against `samples/` (`scripts/seed-sample-corpus.sh`) — seed-sample-corpus.sh now indexes directly via `python -m opsrag.ingestion.indexer` (operator action, no OIDC token needed); resolves samples dir for repo or /app container
- [X] T066 [US1] Ship default `config.yaml` per `contracts/config-schema.md` — auth pointing at `http://dex:5556/dex`, all MCPs disabled, null graph (`config.yaml`)
- [X] T067 [P] [US1] Ship annotated `config-example.yaml` showing every available flag with comments (`config-example.yaml`)
- [X] T068 [US1] Write the README walkthrough section matching `quickstart.md` end to end (`README.md`) — README Quickstart rewritten to match quickstart.md end to end (health -> seed -> Dex token -> /query), correct ports (8080/5173/5556/6006)

### Integration test for US1

- [X] T069 [US1] Add integration test `tests/integration/test_quickstart_happy_path.py` — boot the stack via testcontainers / docker-compose, acquire a Dex token, POST `/query` against indexed samples, assert cited answer in English (`tests/integration/test_quickstart_happy_path.py`) — authored as a stack-gated e2e test (skipped unless OPSRAG_E2E=1); runs in CI after compose up + seed. Not executed in this dev environment (no running stack).
- [X] T070 [US1] Add unit test `tests/unit/test_null_graph_backend.py` — assert empty-result behaviour and interface conformance (`tests/unit/test_null_graph_backend.py`)

**Checkpoint**: User Story 1 fully functional and independently testable.
A new evaluator can complete the walkthrough.

---

## Phase 4: User Story 2 — Operator enables specific integrations via configuration (P2)

**Goal**: Operator flips `mcp.<name>.enabled` to true, supplies credentials,
restarts; the agent uses that integration. Missing credentials cause a
named, fail-fast startup error.

**Independent Test**: Starting from the US1 baseline, enable two MCPs in
config, restart, observe both active in `/readyz`; then remove a required
credential, restart, observe the named startup error.

### Contract tests for US2

- [X] T071 [P] [US2] Add contract test `tests/contract/test_config_unknown_mcp_rejected.py` (`tests/contract/test_config_unknown_mcp_rejected.py`) — tests/contract/test_config_unknown_mcp_rejected.py: unknown mcp.<name> key rejected by the Settings validator
- [X] T072 [P] [US2] Add contract test `tests/contract/test_config_failfast_on_missing_env.py` parametrised over all 14 MCPs (`tests/contract/test_config_failfast_on_missing_env.py`) — tests/contract/test_config_failfast_on_missing_env.py: parametrized over all 14 MCPs; enabling without required env/config raises MCP_MISCONFIGURED

### MCP integrations — port + fake-backend test (one task per integration, parallelisable)

- [X] T073 [P] [US2] Port + sanitize `opsrag/mcp/cartography.py` and add `tests/integration/test_cartography_mcp.py` against fake (`opsrag/mcp/cartography.py`) — build_fake() + tests/integration/test_cartography_mcp.py vs offline fake (FR-012); fake tool set == registry tool_names
- [X] T074 [P] [US2] Port + sanitize `opsrag/mcp/cloudflare.py` + integration test (`opsrag/mcp/cloudflare.py`) — build_fake() + tests/integration/test_cloudflare_mcp.py vs offline fake (FR-012); fake tool set == registry tool_names
- [X] T075 [P] [US2] Port + sanitize `opsrag/mcp/cloudsql.py` + integration test (gated by `requires_cloudsql` marker per research §5) (`opsrag/mcp/cloudsql.py`) — build_fake() + tests/integration/test_cloudsql_mcp.py vs offline fake (FR-012); fake tool set == registry tool_names
- [X] T076 [P] [US2] Port + sanitize `opsrag/mcp/code.py` + integration test (`opsrag/mcp/code.py`) — build_fake() + tests/integration/test_code_mcp.py vs offline fake (FR-012); fake tool set == registry tool_names
- [X] T077 [P] [US2] Port + sanitize `opsrag/mcp/datadog.py` + integration test (`opsrag/mcp/datadog.py`) — build_fake() + tests/integration/test_datadog_mcp.py vs offline fake (FR-012); fake tool set == registry tool_names
- [X] T078 [P] [US2] Port + sanitize `opsrag/mcp/elasticsearch.py` + integration test (`opsrag/mcp/elasticsearch.py`) — build_fake() + tests/integration/test_elasticsearch_mcp.py vs offline fake (FR-012); fake tool set == registry tool_names
- [X] T079 [P] [US2] Port + sanitize `opsrag/mcp/gitlab.py` + integration test (`opsrag/mcp/gitlab.py`) — gitlab MCP: build_fake() -> FakeMCP + tests/integration/test_gitlab_mcp.py (reference pattern for FR-012); shared FakeMCP contract in opsrag/mcp/_fake.py
- [X] T080 [P] [US2] Port + sanitize `opsrag/mcp/knowledge.py` + integration test (`opsrag/mcp/knowledge.py`) — build_fake() + tests/integration/test_knowledge_mcp.py vs offline fake (FR-012); fake tool set == registry tool_names
- [X] T081 [P] [US2] Port + sanitize `opsrag/mcp/kubernetes.py` + integration test (preserves upstream `FakeApiClient` pattern) (`opsrag/mcp/kubernetes.py`) — build_fake() + tests/integration/test_kubernetes_mcp.py vs offline fake (FR-012); fake tool set == registry tool_names
- [X] T082 [P] [US2] Port + sanitize `opsrag/mcp/prometheus.py` + integration test (`opsrag/mcp/prometheus.py`) — build_fake() + tests/integration/test_prometheus_mcp.py vs offline fake (FR-012); fake tool set == registry tool_names
- [X] T083 [P] [US2] Port + sanitize `opsrag/mcp/rootly.py` + integration test (`opsrag/mcp/rootly.py`) — build_fake() + tests/integration/test_rootly_mcp.py vs offline fake (FR-012); fake tool set == registry tool_names
- [X] T084 [P] [US2] Port + sanitize `opsrag/mcp/runbooks.py` + integration test (`opsrag/mcp/runbooks.py`) — build_fake() + tests/integration/test_runbooks_mcp.py vs offline fake (FR-012); fake tool set == registry tool_names
- [X] T085 [P] [US2] Port + sanitize `opsrag/mcp/slack.py` + integration test (`opsrag/mcp/slack.py`) — build_fake() + tests/integration/test_slack_mcp.py vs offline fake (FR-012); fake tool set == registry tool_names
- [X] T086 [P] [US2] Port + sanitize `opsrag/mcp/tool_cache.py` + integration test (`opsrag/mcp/tool_cache.py`) — build_fake() + tests/integration/test_tool_cache_mcp.py vs offline fake (FR-012); fake tool set == registry tool_names

### Wiring + readiness

- [X] T087 [US2] Implement `opsrag/mcp_server/registry_loader.py` — iterates `MCPIntegration.registry`, instantiates enabled integrations, registers tools (`opsrag/mcp_server/registry_loader.py`) — opsrag/mcp_server/registry_loader.py: enabled_integration_names/enabled_tool_names + process-level active-enabled gating (set_active_enabled/filter_enabled); wired into create_app
- [X] T088 [US2] Extend `/readyz` to probe every enabled integration's `health_url_template` and report per-MCP status (`opsrag/api/routes/health.py`) — /readyz now probes enabled MCPs: reports per-integration status and best-effort GETs concrete health_url_template URLs (degrading readiness on failure); disabled MCPs omitted
- [X] T089 [US2] Implement startup fail-fast for `MCP_MISCONFIGURED:<name>:<env>` per FR-004 (`opsrag/config.py`) — implemented validate_enabled_mcps() in mcp/registry.py (checks required_env + required_config dotted paths -> MCPMisconfigured); wired into create_app fail-fast
- [X] T090 [US2] Implement startup fail-fast for unknown `mcp.<name>` keys (Pydantic `extra="forbid"` on the MCP map) (`opsrag/config.py`) — unknown mcp.<name> keys already rejected by the Settings mcp-map validator (extra names forbidden); covered by T071

### Agent integration

- [X] T091 [US2] Extend agent graph to route to MCP tools when enabled; preserve null-graph behaviour (no special-casing) (`opsrag/agent/graph.py`) — agent _registry()/_tool_specs_for_llm() filter ALL_MCP_TOOLS through registry_loader.filter_enabled -> only enabled integrations' tools reach the LLM (null-graph/no-MCP path unchanged when nothing enabled)
- [X] T092 [P] [US2] Port + sanitize the Investigation agent (`opsrag/agents/investigation/`) — graph, tools, prompts, result_cache, limits (`opsrag/agents/investigation/`) — investigation agent verified: build_investigation_graph runs end-to-end with fakes (tests/integration/test_investigation_agent_with_fakes.py)
- [X] T093 [P] [US2] Sanitize investigation-agent prompts to English only (`opsrag/agents/investigation/prompts/`) — investigation prompts sanitized in T056h (verified English-only/org-neutral)
- [X] T094 [US2] Add API route `opsrag/api/routes/investigation.py` and wire `POST /query` `agent: "investigation"` dispatch (`opsrag/api/routes/investigation.py`) — investigation route verified: routes_investigations router wired + /query agent='investigation' dispatch (api/routes.py)
- [X] T095 [P] [US2] Add `tests/integration/test_investigation_agent_with_fakes.py` — runs the investigation graph end-to-end with all-fake MCPs (`tests/integration/test_investigation_agent_with_fakes.py`) — tests/integration/test_investigation_agent_with_fakes.py: scripted fake LLM (routes on purpose) + fake retrieve drive the multi-hop graph to termination

### Provider extensions for US2 (optional providers carried forward)

- [X] T096 [P] [US2] Port `opsrag/llms/vertex.py` and `opsrag/llms/bedrock.py` providers (`opsrag/llms/`) — verified: opsrag/llms/vertex.py + bedrock.py import; factory wires vertex/bedrock LLM branches (tests/unit/test_us2_providers.py)
- [X] T097 [P] [US2] Port `opsrag/embedders/vertex.py` and `opsrag/embedders/bedrock.py` providers (`opsrag/embedders/`) — verified: opsrag/embedders/vertex.py + bedrock.py import; factory wires them
- [X] T098 [P] [US2] Port `opsrag/vectorstores/pgvector.py` provider (`opsrag/vectorstores/pgvector.py`) — verified: opsrag/vectorstores/pgvector.py imports; factory wires pgvector branch
- [X] T099 [P] [US2] Port `opsrag/graphstores/neo4j.py` provider (`opsrag/graphstores/neo4j.py`) — verified: opsrag/graphstores/neo4j.py imports; factory wires neo4j branch
- [X] T100 [P] [US2] Port `opsrag/rerankers/cohere.py`, `opsrag/rerankers/fastembed.py`, `opsrag/rerankers/noop.py` (`opsrag/rerankers/`) — verified: rerankers cohere/noop/fastembed_reranker import; factory wires cohere/fastembed/vertex reranker branches

**Checkpoint**: User Story 2 fully functional. Operators can enable
individual MCPs and the Investigation agent against any combination of
backends.

---

## Phase 5: User Story 3 — Platform team deploys opsrag via Helm (P2)

**Goal**: `helm install opsrag deploy/helm/opsrag -f my-values.yaml` produces
a ready, observable deployment with the operator's chosen MCP set.

**Independent Test**: On a kind / local k8s cluster, install the chart with a
minimal values file (LLM key, OIDC issuer, two MCPs enabled); deployment
reaches readiness; `kubectl exec` into the pod and curl `/healthz` succeeds.

### Contract tests for US3 (Helm)

- [X] T101 [P] [US3] Add `tests/contract/test_helm_lint.sh` — runs `helm lint deploy/helm/opsrag` (`tests/contract/test_helm_lint.sh`) — tests/contract/test_helm_lint.sh
- [X] T102 [P] [US3] Add `tests/contract/test_helm_template_default.sh` — runs `helm template` and yaml-parses output (`tests/contract/test_helm_template_default.sh`) — tests/contract/test_helm_template_default.sh
- [X] T103 [P] [US3] Add `tests/contract/test_helm_values_covers_all_mcps.py` — asserts `values.yaml` `mcp:` key set equals `MCPIntegration.registry` exactly (`tests/contract/test_helm_values_covers_all_mcps.py`) — tests/contract/test_helm_values_covers_all_mcps.py (values mcp keys == registry)
- [X] T104 [P] [US3] Add `tests/contract/test_helm_env_propagation.py` — renders chart with one MCP enabled, asserts `OPSRAG_MCP_<NAME>_ENABLED=true` in Deployment spec (`tests/contract/test_helm_env_propagation.py`) — tests/contract/test_helm_env_propagation.py (OPSRAG_MCP_<NAME>_ENABLED wiring)
- [X] T105 [P] [US3] Add `tests/contract/test_helm_schema_rejects_unknown.sh` — `helm install --dry-run` against values with unknown MCP key; expect failure (`tests/contract/test_helm_schema_rejects_unknown.sh`) — tests/contract/test_helm_schema_rejects_unknown.sh (schema enumerates 14 + additionalProperties:false)

### Helm chart authoring

- [X] T106 [US3] Author `deploy/helm/opsrag/Chart.yaml` — `apiVersion: v2`, `type: application`, `license: Apache-2.0`, icon, versioned per project (`deploy/helm/opsrag/Chart.yaml`) — deploy/helm/opsrag/Chart.yaml (apiVersion v2, Apache-2.0)
- [X] T107 [US3] Author `deploy/helm/opsrag/values.yaml` — every key documented inline; all 14 `mcp.<name>.enabled: false` blocks present (`deploy/helm/opsrag/values.yaml`) — deploy/helm/opsrag/values.yaml (all 14 mcp flags, documented)
- [X] T108 [US3] Author `deploy/helm/opsrag/values.schema.json` per `contracts/helm-values-schema.md` (`deploy/helm/opsrag/values.schema.json`) — deploy/helm/opsrag/values.schema.json
- [X] T109 [P] [US3] Author `deploy/helm/opsrag/templates/_helpers.tpl` with `opsrag.fullname`, `opsrag.labels`, `opsrag.selectorLabels`, `opsrag.serviceAccountName`, `opsrag.image` (`deploy/helm/opsrag/templates/_helpers.tpl`) — templates/_helpers.tpl (fullname/labels/selectorLabels/serviceAccountName/image/mcpEnv)
- [X] T110 [P] [US3] Author `deploy/helm/opsrag/templates/deployment.yaml` for the api workload; env wiring for every MCP flag (`deploy/helm/opsrag/templates/deployment.yaml`) — templates/deployment.yaml (api; OPSRAG_MCP_* env wiring)
- [X] T111 [P] [US3] Author `deploy/helm/opsrag/templates/ui-deployment.yaml` (`deploy/helm/opsrag/templates/ui-deployment.yaml`) — templates/ui-deployment.yaml (+ UI service)
- [X] T112 [P] [US3] Author `deploy/helm/opsrag/templates/slackbot-deployment.yaml` gated by `slackBot.enabled` (`deploy/helm/opsrag/templates/slackbot-deployment.yaml`) — templates/slackbot-deployment.yaml (gated)
- [X] T113 [P] [US3] Author `deploy/helm/opsrag/templates/service.yaml` (`deploy/helm/opsrag/templates/service.yaml`) — templates/service.yaml
- [X] T114 [P] [US3] Author `deploy/helm/opsrag/templates/ingress.yaml` gated by `ingress.enabled` (`deploy/helm/opsrag/templates/ingress.yaml`) — templates/ingress.yaml (gated)
- [X] T115 [P] [US3] Author `deploy/helm/opsrag/templates/configmap.yaml` — rendered config.yaml content (`deploy/helm/opsrag/templates/configmap.yaml`) — templates/configmap.yaml
- [X] T116 [P] [US3] Author `deploy/helm/opsrag/templates/secret.yaml` — opt-in default; supports external SecretRef per MCP (`deploy/helm/opsrag/templates/secret.yaml`) — templates/secret.yaml (opt-in)
- [X] T117 [P] [US3] Author `deploy/helm/opsrag/templates/serviceaccount.yaml` (`deploy/helm/opsrag/templates/serviceaccount.yaml`) — templates/serviceaccount.yaml
- [X] T118 [P] [US3] Author `deploy/helm/opsrag/templates/networkpolicy.yaml` with egress allowlist driven from values (`deploy/helm/opsrag/templates/networkpolicy.yaml`) — templates/networkpolicy.yaml (gated egress allowlist)
- [X] T119 [P] [US3] Author `deploy/helm/opsrag/templates/poddisruptionbudget.yaml` (`deploy/helm/opsrag/templates/poddisruptionbudget.yaml`) — templates/poddisruptionbudget.yaml (gated)
- [X] T120 [P] [US3] Author `deploy/helm/opsrag/templates/hpa.yaml` gated by `autoscaling.enabled` (`deploy/helm/opsrag/templates/hpa.yaml`) — templates/hpa.yaml (gated)
- [X] T121 [P] [US3] Author `deploy/helm/opsrag/templates/NOTES.txt` with post-install hints and MCP-enable reminders (`deploy/helm/opsrag/templates/NOTES.txt`) — templates/NOTES.txt
- [X] T122 [P] [US3] Author `deploy/helm/opsrag/templates/tests/test-connection.yaml` — `helm test` curls `/healthz` (`deploy/helm/opsrag/templates/tests/test-connection.yaml`) — templates/tests/test-connection.yaml (helm test)
- [X] T123 [US3] Author `deploy/helm/opsrag/README.md` — chart docs, values table, installation, upgrade notes (`deploy/helm/opsrag/README.md`) — deploy/helm/opsrag/README.md (values table + install/upgrade)

### CI wiring for chart

- [X] T124 [US3] Add a `helm-lint` job in `.github/workflows/ci.yml` that runs T101 + T102 + T103 + T104 + T105 on every PR touching the chart (`.github/workflows/ci.yml`) — ci.yml helm-lint job extended to run shell + python helm contract tests
- [X] T125 [US3] Add a kind-based end-to-end test `tests/integration/test_helm_install_kind.sh` that boots kind, `helm install`s the chart, waits for ready, curls `/healthz` (`tests/integration/test_helm_install_kind.sh`) — tests/integration/test_helm_install_kind.sh (gated on OPSRAG_KIND_E2E=1 + kind)

**Checkpoint**: User Story 3 fully functional. The chart installs cleanly,
mounts every MCP flag, fails fast on misconfigured values, and passes
`helm test`.

---

## Phase 6: User Story 4 — Maintainer audits the fork for organization-specific content (P3)

**Goal**: A scripted audit confirms zero proprietary names, zero non-English
text, and zero hardcoded hosts across the working tree.

**Independent Test**: `scripts/audit-vendor-neutrality.sh` exits 0 on the
default branch; a synthetic violation in a PR causes the audit to exit
non-zero and CI to fail.

### Contract tests for US4

- [X] T126 [P] [US4] Add `tests/contract/test_audit_clean_tree.sh` — runs the audit on HEAD and expects exit 0 (`tests/contract/test_audit_clean_tree.sh`) — tests/contract/test_audit_clean_tree.sh
- [X] T127 [P] [US4] Add `tests/contract/test_audit_catches_proprietary.sh` — writes a temp file containing a denylist token; expects non-zero (`tests/contract/test_audit_catches_proprietary.sh`) — tests/contract/test_audit_catches_proprietary.sh (assembles token at runtime so the test stays clean)
- [X] T128 [P] [US4] Add `tests/contract/test_audit_catches_non_english.sh` (`tests/contract/test_audit_catches_non_english.sh`) — tests/contract/test_audit_catches_non_english.sh
- [X] T129 [P] [US4] Add `tests/contract/test_audit_catches_hardcoded_host.sh` (`tests/contract/test_audit_catches_hardcoded_host.sh`) — tests/contract/test_audit_catches_hardcoded_host.sh
- [X] T130 [P] [US4] Add `tests/contract/test_audit_json_shape.py` — runs `--json` and validates against `AuditReport` schema from `data-model.md` §5 (`tests/contract/test_audit_json_shape.py`) — tests/contract/test_audit_json_shape.py (AuditReport shape)

### Audit script implementation

- [X] T131 [US4] Fill in the proprietary-names check in `scripts/audit-vendor-neutrality.sh` with denylist (`acme`, internal hostnames) and allowlist (`CHANGELOG.md`, `samples/`, `specs/`, `.specify/`) (`scripts/audit-vendor-neutrality.sh`) — proprietary-names check implemented in audit-vendor-neutrality.sh (denylist from audit-rules.yaml)
- [X] T132 [US4] Fill in the non-English-text check (extended-ASCII + CJK + Cyrillic + Hangul ranges) with `tests/fixtures/i18n/` exception (`scripts/audit-vendor-neutrality.sh`) — non-English-text check implemented (structural non-ASCII; exempt paths from rules)
- [X] T133 [US4] Fill in the hardcoded-host check with placeholder + docs-host allowlist (`scripts/audit-vendor-neutrality.sh`) — hardcoded-host check implemented (regex + allowlist incl. json-schema.org for the chart schema)
- [X] T134 [US4] Implement `--json` output emitting the `AuditReport` shape (`scripts/audit-vendor-neutrality.sh`) — --json AuditReport output implemented
- [X] T135 [US4] Implement `--fix-suggestions` output (`scripts/audit-vendor-neutrality.sh`) — --fix-suggestions output implemented
- [X] T136 [US4] Add deterministic violation sorting `(check, file, line)` (`scripts/audit-vendor-neutrality.sh`) — deterministic (check,file,line) sort implemented

### CI integration

- [X] T137 [US4] Add an `audit` job to `.github/workflows/ci.yml` invoking the script with `--json`; annotate PRs on failure (`.github/workflows/ci.yml`) — audit CI job present in ci.yml (from T013)
- [X] T138 [US4] Wire the audit into `.pre-commit-config.yaml` as a local hook (`.pre-commit-config.yaml`) — audit wired into .pre-commit-config.yaml (from T015)

### Final sweep

- [X] T139 [US4] Run the full audit on HEAD; fix any residual violations introduced during phases 3–5; assert clean (`opsrag/`, `ui/`, `deploy/`, `docs/`) — final sweep: audit clean on the full tree (clean-tree contract test green)
- [X] T140 [US4] Document the audit in `docs/auth.md`'s sibling `docs/contributing-audit.md` (`docs/contributing-audit.md`) — docs/contributing-audit.md authored

**Checkpoint**: User Story 4 fully functional. The audit gates merges and
the tree is clean.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Carry over remaining subsystems (UI, Slack bot, eval, webhooks),
write adopter-facing docs, and verify the end-to-end performance promises.

### React UI

- [X] T141 [P] Copy `ui/` from upstream; remove all branding assets and replace with project-neutral defaults (`ui/`) — ported upstream ui/ into the repo (React 19 + Vite + TS); deleted the dead 'Classic' variant; brand is runtime-configurable via /api/ui-config
- [X] T142 [P] Sanitize `ui/src/**` for non-English copy and proprietary references (`ui/src/`) — sanitized ui/src/**: removed internal hosts/MCP URL/help-text, org service names in placeholders, and internal ticket comments; audit clean
- [~] T143 [P] Implement OIDC PKCE flow in the UI against the configured issuer (`ui/src/auth/`) — SSO temporarily SKIPPED (user decision 2026-05-30): the UI uses proxy-level auth, not an embedded OIDC/PKCE client. Removed the organization-specific Pomerium sign-out (now VITE_SIGN_OUT_URL, default off); wiring UI->OIDC-Bearer backend is a follow-up. Demo runs the backend open (deploy/compose/config.yaml has no auth block)
- [X] T144 [P] Update `ui/vite.config.ts` and `ui/package.json` SPDX to `Apache-2.0` (`ui/package.json`) — ui/package.json license: Apache-2.0; VITE_* env vars typed in vite-env.d.ts
- [~] T145 [P] Add Vitest unit tests covering the chat, usage, and indexing components (`ui/src/__tests__/`) — DEFERRED: Vitest component tests need the JS toolchain (npm install) to author/run; not part of the current UI-port plan

### Slack bot

- [X] T146 [P] Sanitize and port `opsrag/slack_bot/` — Socket Mode entrypoint, channel mappings driven entirely from config (`opsrag/slack_bot/`) — slack_bot/ ported + sanitized (audit clean); verified
- [X] T147 [P] Add `tests/unit/test_slack_bot_channel_resolution.py` against a fake Slack client (`tests/unit/test_slack_bot_channel_resolution.py`) — tests/unit/test_slack_bot_channel_resolution.py: allowlist/DM/bot-loop/quota via SlackBotPermission

### Evaluation harness

- [X] T148 [P] Port `opsrag/eval/` — Phoenix + DeepEval scaffolding (`opsrag/eval/`) — opsrag/eval/ ported (runner/loaders/metrics/adapters); deepeval is an optional `eval` extra so the eval test dir skips cleanly when absent
- [X] T149 [P] Author a synthetic golden eval set covering retrieval + investigation flows; assert numeric thresholds (`opsrag/eval/datasets/`) — 11 synthetic golden datasets present under opsrag/eval/golden/ (audit-clean)
- [X] T150 [P] Add CI job `eval-regression` triggered on changes under `opsrag/agent/`, `opsrag/agents/`, `opsrag/mcp/`, `opsrag/eval/` (`.github/workflows/ci.yml`) — eval-regression CI job present in ci.yml (from T013)

### Webhooks

- [X] T151 [P] Port `opsrag/api/routes/webhooks.py` — `/webhook/gitlab` and `/webhook/github` with HMAC validation (`opsrag/api/routes/webhooks.py`) — opsrag/api/routes_webhooks.py: /webhook/gitlab (X-Gitlab-Token) + /webhook/github (HMAC-SHA256); secret-authed, in NO_AUTH_PATHS; best-effort reindex
- [X] T152 [P] Add `tests/integration/test_webhooks.py` — validates signing + dispatching (`tests/integration/test_webhooks.py`) — tests/integration/test_webhooks.py: valid/invalid/missing signature + OIDC-bypass

### Documentation

- [X] T153 [P] Author `docs/architecture.md` (`docs/architecture.md`) — docs/architecture.md
- [X] T154 [P] Author `docs/mcp-integrations.md` listing every integration, its flag, its envs, its capability (`docs/mcp-integrations.md`) — docs/mcp-integrations.md (all 14 from the registry)
- [X] T155 [P] Author `docs/helm-chart.md` with values reference and upgrade paths (`docs/helm-chart.md`) — docs/helm-chart.md
- [X] T156 [P] Author `docs/auth.md` describing OIDC setup for common IdPs (Dex, Keycloak, Okta, Auth0, Azure AD) (`docs/auth.md`) — docs/auth.md (Dex/Keycloak/Okta/Auth0/Azure AD)
- [X] T157 [P] Update `CHANGELOG.md` with the full fork entry per FR-015 — motivation, sanitized categories, flag list (`CHANGELOG.md`) — CHANGELOG.md fork entry extended (Principle VI, sample corpus, MCP fakes, docs)

### Verification

- [X] T158 Run the full test suite (`pytest`, `helm lint`, audit) and confirm zero failures on `master` (`tests/`) — full suite green: 184 passed, 10 skipped (NOTE: full default-testpaths run is slow ~41min; `pytest tests/` ~3s -- a slow-test perf follow-up is flagged)
- [~] T159 Time the User Story 1 walkthrough end-to-end on a clean machine; confirm SC-001 (<15 min) and SC-005 (<30 s healthy boot) (`quickstart.md`) — timing walkthrough (SC-001 <15min, SC-005 <30s) requires a live compose stack; not run in this environment
- [X] T160 Final pass: re-render Helm chart against `values.yaml`, confirm SC-004 (every MCP has a flag in both `config.yaml` and `values.yaml`) (`deploy/helm/opsrag/values.yaml`, `config.yaml`) — SC-004 verified: all 14 MCP flags present in registry == config.yaml == values.yaml; helm renders

---

## Dependencies & execution order

### Phase dependencies

- **Phase 1 (Setup)**: No dependencies. Start immediately.
- **Phase 2 (Foundational)**: Depends on Phase 1. **BLOCKS** all user-story phases.
- **Phase 3 (US1 — P1)**: Depends on Phase 2.
- **Phase 4 (US2 — P2)**: Depends on Phase 2 only (can run alongside Phase 3 if staffed).
- **Phase 5 (US3 — P2)**: Depends on Phase 2 only (can run alongside Phases 3–4).
- **Phase 6 (US4 — P3)**: Depends on Phase 2 only (audit script can be implemented in parallel; final sweep at T139 requires Phases 3–5 complete).
- **Phase 7 (Polish)**: Depends on Phases 3–6 complete (T158–T160 are full-suite verifications).

### Story dependencies

- US1 establishes the bring-up baseline; US2/US3/US4 inherit it.
- US2 (enable MCPs) and US3 (Helm install) are independent of each other.
- US4 (audit) is mechanically independent but its final sweep (T139) needs
  the work of US1–US3 settled.

### Within each story

- Contract tests written before implementation (TDD per Principle IV).
- Models/configs before services; services before endpoints.
- Tests of one integration parallelisable across integrations (different
  files).

### Parallel opportunities

- **Phase 1**: T002–T008 (license, code-of-conduct, security, contributing,
  readme, gitignore) and T010–T015 (CI workflows, lockfile, pre-commit) all
  parallelisable.
- **Phase 2 (sanitization sweep)**: T019–T027 (one task per Vietnamese file)
  all parallelisable; T031–T034 (registry, auth, null backend) parallelisable.
- **Phase 3 (US1)**: T041–T045 (contract tests), T046–T054 (provider ports),
  T062–T064 (sample corpus) all parallelisable.
- **Phase 4 (US2)**: T073–T086 — fourteen MCP ports — all parallelisable
  (different files). T096–T100 provider extensions parallelisable.
- **Phase 5 (US3)**: T109–T122 — fourteen Helm template files —
  parallelisable.
- **Phase 6 (US4)**: T126–T130 (contract tests) parallelisable.
- **Phase 7**: T141–T156 — UI, Slack bot, eval, webhooks, docs — broadly
  parallelisable.

---

## Parallel example: Phase 4 (US2)

The fourteen MCP ports can run completely in parallel:

```bash
# In parallel:
Task: "Port + sanitize opsrag/mcp/cartography.py + integration test"
Task: "Port + sanitize opsrag/mcp/cloudflare.py + integration test"
Task: "Port + sanitize opsrag/mcp/cloudsql.py + integration test"
Task: "Port + sanitize opsrag/mcp/code.py + integration test"
Task: "Port + sanitize opsrag/mcp/datadog.py + integration test"
Task: "Port + sanitize opsrag/mcp/elasticsearch.py + integration test"
Task: "Port + sanitize opsrag/mcp/gitlab.py + integration test"
Task: "Port + sanitize opsrag/mcp/knowledge.py + integration test"
Task: "Port + sanitize opsrag/mcp/kubernetes.py + integration test"
Task: "Port + sanitize opsrag/mcp/prometheus.py + integration test"
Task: "Port + sanitize opsrag/mcp/rootly.py + integration test"
Task: "Port + sanitize opsrag/mcp/runbooks.py + integration test"
Task: "Port + sanitize opsrag/mcp/slack.py + integration test"
Task: "Port + sanitize opsrag/mcp/tool_cache.py + integration test"
```

After all fourteen complete, T087–T090 wire them through the registry and
fail-fast logic (these must run sequentially since they touch shared files).

---

## Implementation strategy

### MVP First (User Story 1 only)

1. Complete Phase 1 (Setup) — repository skeleton + CI scaffolding.
2. Complete Phase 2 (Foundational) — sanitization, core scaffolding,
   compose stack.
3. Complete Phase 3 (US1) — bring-up walkthrough works end-to-end.
4. **STOP and VALIDATE**: run the quickstart on a clean machine; if SC-001
   and SC-005 hold, the MVP is shippable.
5. Tag `v0.1.0-alpha` and announce.

### Incremental delivery

1. Setup + Foundational + US1 → MVP (`v0.1.0-alpha`)
2. Add US2 → MCPs are usable (`v0.2.0`)
3. Add US3 → Production-deployable via Helm (`v0.3.0`)
4. Add US4 → Audit-gated; safe-to-contribute (`v0.4.0`)
5. Add Phase 7 (UI, Slack bot, eval, docs) → Feature-complete `v1.0.0`

### Parallel-team strategy

With three or more developers, after Phase 2 completes:

- Dev A: US1 (Phase 3) — bring-up baseline
- Dev B: US2 (Phase 4) — MCP integrations (parallel by integration)
- Dev C: US3 (Phase 5) — Helm chart
- Dev D (or Dev A returning): US4 (Phase 6) — audit script
- Polish (Phase 7) parcelled out by subsystem (UI / bot / eval / docs).

---

## Notes

- `[P]` tasks touch different files and have no order dependency among
  themselves.
- `[Story]` label maps every story-phase task to its user story for
  traceability.
- Tests are mandatory by FR-005 / FR-006 / FR-012 / FR-014 and Constitution
  Principle IV.
- Verify contract tests fail before writing the matching implementation
  (TDD).
- Commit after each logical group (typically a few related `[P]` tasks plus
  the sequential task they unblock).
- Stop at any phase checkpoint to validate the story independently — the
  checkpoints map to demo-able increments.
- Avoid: vague tasks, two parallel tasks touching the same file,
  cross-story coupling that breaks independent testability.
