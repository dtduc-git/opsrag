<!--
Sync Impact Report
==================
Version change: 1.0.0 -> 1.1.0
Bump rationale: Added Principle VI (Context-Driven Engine). The original
Principle I (Vendor-Neutral by Default) prohibits proprietary identifiers
from being shipped; Principle VI goes further and specifies *where*
deployment knowledge belongs — config, prompt, corpus, or skill. Informed
by the engine-context inventory produced during the Phase 2 sanitization
sweep (specs/001-port-opsrag-opensource/inventory.md). The inventory
revealed that simply substituting one set of organisation names for
another (the "fake-abstract" failure mode) leaves the engine coupled to
the original organisation's architecture even after audit checks pass.
Principle VI codifies the cross-org test that prevents this.

Modified principles: I (cross-references the new context model — no
                       obligations changed, only clarified)
Added sections:
  - Principle VI (Context-Driven Engine)

Removed sections: n/a

Templates audited:
  - .specify/templates/plan-template.md       (Constitution Check table will need to grow a row for Principle VI when plans are re-generated)
  - .specify/templates/spec-template.md       (no edits needed - principle-agnostic)
  - .specify/templates/tasks-template.md      (no edits needed - principle-agnostic)
  - .specify/templates/commands/*             (no edits needed)

Runtime guidance audited:
  - README.md                                  (no edits needed - principles not yet referenced)
  - CLAUDE.md                                  (no edits needed)
  - specs/001-port-opsrag-opensource/plan.md   (Constitution Check table at lines 132-138 should be re-evaluated against Principle VI during plan revision; current evaluation predates VI)

Follow-up TODOs:
  - When the plan-template's Constitution Check is next regenerated for
    new feature work, add a row for Principle VI.
  - The active plan (specs/001-port-opsrag-opensource/plan.md) was
    written before Principle VI; its Constitution Check should be
    re-evaluated against VI when T029 lands.
-->

# opsrag Constitution

opsrag is an opensource, agentic GraphRAG platform for DevOps/SRE workflows. It
is the public, vendor-neutral evolution of an internal toolchain previously
maintained at `acme/infra/opsrag`. This constitution governs how the
project is designed, built, and shipped so that it remains useful to any
operator, in any organization, on any cloud.

## Core Principles

### I. Vendor-Neutral by Default (NON-NEGOTIABLE)

Source, prompts, docs, and configuration MUST be free of organization-specific
names, hostnames, API keys, account IDs, Slack channel IDs, runbook URLs, and
non-English content. Anything that varies between deployments MUST be exposed
via configuration (`config.yaml`) or environment variables. Example values in
shipped files (`.env.example`, `values.yaml`, sample configs) MUST be obvious
placeholders (e.g. `your-cluster.example.com`, `CHANGE_ME`).

**Rationale**: A fork is only meaningfully opensource if a stranger can clone
it and stand it up without reading or rewriting the source. Hardcoded internal
references — and prompts written in any single human language — silently
exclude users and create maintenance debt.

### II. Pluggable Integrations via Feature Flags

Every external integration — each MCP server (Kubernetes, Datadog, GitLab,
Cloudflare, Rootly, Slack, Elasticsearch, CloudSQL, Prometheus, Cartography,
runbooks, code search, knowledge base, etc.) — MUST be independently
enable/disable-able via an explicit feature flag in `config.yaml`. The default
shipped configuration MUST enable zero external integrations. When an
integration is enabled but its credentials/configuration are missing or
invalid, startup MUST fail fast with a clear error naming the offending
flag and the missing setting.

**Rationale**: Different teams use different observability and infrastructure
stacks; nobody should pay (in dependencies, attack surface, or startup cost)
for an integration they do not use. Explicit flags make the active surface
auditable.

### III. Cloud-Native, Container-First

The primary distribution artifact is an OCI container image. A Helm chart under
`deploy/helm/opsrag/` is a first-class deliverable: it MUST follow standard
Helm conventions (`Chart.yaml`, `values.yaml` with documented keys, templates
under `templates/`, `_helpers.tpl`, `NOTES.txt`, optional `tests/`) and MUST
pass `helm lint` and `helm template` on every commit that modifies it. The
project MUST NOT depend on host-specific install paths, OS-level package
managers at runtime, or compiled-in absolute filesystem paths.

**Rationale**: This is operations tooling; the path from `git clone` to a
running pod in a foreign cluster has to be short and uneventful.

### IV. Test & Eval Discipline

All non-trivial code paths require automated tests. MCP adapters require
integration tests that exercise the adapter against a fake/recorded backend.
LLM-driven flows (agent loops, hypothesizers, RAG retrieval) MUST be guarded
by a golden eval set (Phoenix / DeepEval) with quantitative regression
thresholds. CI MUST run lint, type-check, unit, and integration suites on
every pull request; a red build MUST NOT be merged. Eval suites SHOULD run on
every PR that touches prompts, retrievers, or agent graphs.

**Rationale**: Agentic systems regress silently — a prompt tweak can
collapse retrieval quality without any test failing. Numeric eval guardrails
are the only honest verification.

### V. Observability & Secret Hygiene

Every long-running component MUST emit structured logs (JSON) and OpenTelemetry
traces by default, expose `/healthz` and `/readyz` endpoints, and surface
basic Prometheus metrics. Secrets (API keys, tokens, DSNs) MUST be loaded from
environment variables or secret-store references — never read from files
committed to the repository, never logged, never embedded in prompts. A
`.env.example` MUST ship with placeholders so the required environment
contract is discoverable without reading source.

**Rationale**: Operators using this tool are themselves on-call; the tool
must be debuggable in production. Leaked credentials in an opensource repo
are catastrophic and irreversible.

### VI. Context-Driven Engine (NON-NEGOTIABLE)

The engine carries no organization knowledge. Anything that varies between
deployments — service names, environments, namespaces, cluster identifiers,
URL bases, ticket prefixes, repo paths — MUST be supplied at runtime via
the operator's `DeploymentContext` (formal schema:
`specs/.../contracts/deployment-context.md`). Knowledge belongs to exactly
one of four tiers:

- **config** — Deployment facts (services, envs, namespaces, URLs, ticket
  prefix). Lives in `config.yaml`, validated by `DeploymentContext`. The
  engine reads, never invents.
- **prompt** — Org-agnostic reasoning patterns. System prompts are
  templates that the engine renders against `DeploymentContext` at
  startup or query time. Prompts MUST NOT contain hardcoded service
  names, cluster identifiers, or environment labels — not even fictional
  placeholders. The operator's facts are injected, not pre-cooked.
- **corpus** — Domain knowledge (runbooks, postmortems, ticket history,
  alert shapes, failure-mode catalogs). Lives in the indexed vector and
  graph stores. The agent retrieves at query time via RAG; the engine
  never embeds this knowledge in source.
- **skill** — Engine capabilities, tool definitions, static logic.
  Org-agnostic by construction. Tool docstrings MAY teach call format
  with placeholder examples (`<service>`, `<env>`, `<cluster>`); they
  MUST NOT teach deployment topology.

**The cross-org test (gate)**: for every retained pattern in engine code
or prompts, ask: *"would this still be useful and correct if this engine
were deployed at a completely unrelated organisation with different
services, environments, and conventions?"*

- Yes → keep, possibly abstracted with context placeholders.
- No → the pattern is organisation knowledge. Move it: facts → config,
  knowledge → corpus, capability → skill. Or drop it.

**A fake-abstract is a defect.** Substituting one set of organisation
names for another (e.g. renaming an internal service to a fictional name
inside a hardcoded list or a prompt example) leaves the engine coupled to
the original organisation's architecture. The right moves are:

- A module-level constant that names deployments → field on
  `DeploymentContext`. Example: `_KNOWN_CLUSTERS = (...)` becomes
  `deployment.kubernetes.clusters`.
- A prompt anecdote ("SSO 503 in prod usually means cert drift on
  auth-service-X") → corpus document indexed at deployment time.
- A hardcoded URL default in code → `Optional[str] = None` field on
  `DeploymentContext.source_urls`, with no default value.

**Audit obligation**: `scripts/audit-vendor-neutrality.sh` MUST gate
merges by detecting both (a) the explicit denylist tokens in
`audit-rules.yaml` and (b) — by Phase 6 final-sweep — structural
violations of this principle (hardcoded service-shape literals in
non-allowlisted paths).

**Consistency invariant**: any synthetic identifier used in `samples/`
MUST match the identifier set referenced from `opsrag/eval/golden/*.yaml`.
A contract test enforces this; the audit gate enforces no engine code
under `opsrag/` mentions any of those identifiers outside the
allowlisted demo / test paths.

**Rationale**: Vendor-neutrality (Principle I) prevents specific
identifiers from being shipped; Principle VI prevents the *shape of
deployment knowledge itself* from being baked into the engine. The
distinction matters because the latter failure mode passes audit-by-name
yet still produces an engine that only works for one architecture.
The four-tier model gives every kind of deployment knowledge exactly one
correct home, which makes its absence from engine code testable.

## Technical Constraints

- **Language**: Python 3.11+ for the agent, MCP servers, and CLI surface.
- **HTTP surface**: FastAPI with `uvicorn[standard]`.
- **Agent orchestration**: LangGraph (`langgraph>=1.1`).
- **Configuration**: Pydantic v2 (`pydantic-settings`) parsing `config.yaml`
  merged with environment variables. Config schemas MUST validate at boot.
- **Default stores**: Qdrant (vectors), Neo4j (graph), PostgreSQL
  (checkpoints via `langgraph-checkpoint-postgres`). Alternatives MAY be
  added as optional adapters but MUST NOT become the default.
- **LLM providers**: Pluggable; the codebase MUST NOT require any single
  vendor (Anthropic, OpenAI, Vertex, Bedrock, etc.) at import time.
- **Container**: A single multi-stage `Dockerfile` at repository root produces
  the runtime image. The image MUST run as a non-root user.

## Development Workflow & Quality Gates

- **Branching**: Feature work proceeds on branches created via
  `/speckit-git-feature`, then merged via PR to `master`.
- **CI gates (all required to merge)**:
  - `ruff` lint clean
  - `mypy` type-check clean for files in changed paths
  - `pytest` unit + integration suites green
  - `helm lint deploy/helm/opsrag` clean when chart files change
  - Eval regression check green when prompts/retrievers/agents change
- **New integration checklist**: Any new MCP integration MUST (a) add an
  `enabled` flag under `mcp.<name>.enabled` in `config.yaml`, (b) ship an
  integration test against a fake backend, (c) document required env vars in
  `.env.example`, and (d) be referenced in the Helm chart's `values.yaml`
  with the flag wired through.
- **Review checklist**: Reviewers MUST confirm no proprietary names,
  hostnames, account identifiers, or non-English prompts have been
  reintroduced.
- **Spec-driven flow**: Non-trivial features follow the Spec Kit workflow —
  `/speckit-specify` → `/speckit-clarify` (when ambiguous) → `/speckit-plan` →
  `/speckit-tasks` → `/speckit-implement`. The `Constitution Check` gate in
  `plan-template.md` MUST be filled in against the principles above.

## Governance

This constitution supersedes ad-hoc practices and the conventions of the
internal codebase from which opsrag was forked. Where this document and any
older runbook or design doc conflict, this document wins.

- **Amendments** are made via pull request that modifies this file and bumps
  the version line below. Amendments require approval from at least one
  project maintainer.
- **Versioning** of this constitution follows semantic versioning:
  - **MAJOR**: A principle is removed or redefined in a backward-incompatible
    way (e.g. dropping the vendor-neutrality requirement).
  - **MINOR**: A new principle or a materially expanded section is added.
  - **PATCH**: Wording fixes, typos, clarifications that do not change
    obligations.
- **Compliance review**: Every PR description MUST either (a) state that the
  change is constitution-neutral, or (b) cite which principle(s) it relates
  to and how it complies. Reviewers MUST verify.
- **Complexity justification**: Any deviation from a principle (e.g. a new
  hardcoded value, a non-optional integration) MUST be recorded in the
  `Complexity Tracking` section of the corresponding plan with a written
  justification and a removal plan.

**Version**: 1.1.0 | **Ratified**: 2026-05-27 | **Last Amended**: 2026-05-28
