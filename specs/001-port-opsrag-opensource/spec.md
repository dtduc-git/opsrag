# Feature Specification: Port opsrag as a vendor-neutral opensource project

**Feature Branch**: `001-port-opsrag-opensource`

**Created**: 2026-05-27

**Status**: Draft

**Input**: User description: "port opsrag from acme/infra/opsrag as a vendor-neutral opensource project with feature-flagged MCPs and a Helm chart"

## Clarifications

### Session 2026-05-27

- Q: Which opensource license should the project publish under? → A: Apache License 2.0
- Q: Which subsystems ship in the initial port? → A: Everything — core agent + MCPs + React UI + Slack bot + Investigation agent + Eval harness
- Q: How does the HTTP API authenticate callers? → A: OIDC / OAuth2 only — Bearer tokens from an external identity provider; no built-in API keys
- Q: Is the graph store feature-flagged like an MCP, or pluggable like the vector store? → A: Pluggable via provider selection with a built-in null backend as default — no separate "enabled" flag; the agent's graph nodes execute unconditionally and the null backend returns empty results

## User Scenarios & Testing *(mandatory)*

### User Story 1 - First-time evaluator stands up a minimal opsrag (Priority: P1)

A site reliability engineer who just discovered opsrag wants to evaluate it
on their laptop. They clone the repository, run a single bring-up command
(documented in the README), and within minutes they have a running service
that can answer a question about a sample runbook they index. No third-party
integrations are involved — only their LLM provider key and the local vector
and graph stores brought up by the bundled compose file.

**Why this priority**: This is the front door. If a new evaluator cannot get
from clone to "answer my first question" in a single short session, the
project fails as an opensource offering regardless of feature richness.
Every other story depends on the project being trivially adoptable.

**Independent Test**: Following only the README, a contributor with no prior
knowledge of opsrag clones the repository on a clean machine, exports an LLM
API key, runs the documented bring-up command, and successfully receives an
answer to a sample query against an indexed sample runbook. Total elapsed
time: under fifteen minutes.

**Acceptance Scenarios**:

1. **Given** a fresh clone of the repository and a valid LLM API key,
   **When** the evaluator runs the documented bring-up command,
   **Then** the service reports healthy within thirty seconds and
   accepts queries authenticated by a token issued by the bundled
   local identity provider, against indexed sample documents.
2. **Given** the bring-up has completed,
   **When** the evaluator asks a question whose answer is contained in an
   indexed sample document,
   **Then** the answer cites the source document and is in English.
3. **Given** no third-party integrations are configured,
   **When** the service starts,
   **Then** it boots cleanly with zero external integrations enabled by
   default.

---

### User Story 2 - Operator enables specific integrations via configuration (Priority: P2)

An operator already running opsrag wants to plug it into their existing
observability stack. They edit a single configuration file to flip on,
say, the Kubernetes and incident-tracker integrations, supply the relevant
credentials via environment variables, and restart the service. The agent
now uses those integrations during investigation flows; nothing else changes.

**Why this priority**: Configurability is the whole reason opsrag is being
opensourced — different teams use different infrastructure stacks. If
enabling an integration requires editing source code or maintaining a
patched fork, the project has failed at its core promise.

**Independent Test**: Starting from the baseline of Story 1, the operator
edits the configuration file to enable two integrations, restarts the
service, and confirms via the running service's status surface that those
two integrations are active and the rest remain inactive.

**Acceptance Scenarios**:

1. **Given** the default configuration with zero integrations enabled,
   **When** the operator sets the enabled flag for an integration to true
   and supplies its required credentials,
   **Then** restarting the service activates that integration and no
   others.
2. **Given** an integration is marked enabled but a required credential
   is missing,
   **When** the service starts,
   **Then** it refuses to start and reports an error whose first line
   names both the enabled flag and the missing credential.
3. **Given** an integration is currently enabled,
   **When** the operator sets its enabled flag back to false and restarts,
   **Then** the service starts cleanly without the integration's
   credentials being required.

---

### User Story 3 - Platform team deploys opsrag to their cluster via Helm (Priority: P2)

A platform team wants to run opsrag in their existing Kubernetes cluster
alongside other internal tooling. They install the project's Helm chart
with their own values file — pointing at their secret store for credentials
and toggling the set of integrations appropriate to their org — and the
result is a running, observable deployment fronted by their standard
ingress controller.

**Why this priority**: Kubernetes is the canonical deployment target for
SRE tooling. A first-class chart removes a large class of integration work
that every adopter would otherwise repeat.

**Independent Test**: A user with a Kubernetes cluster (real or local
single-node) and the chart sources, supplying only their own values file
with credentials and integration toggles, can `helm install` and reach a
ready, healthy deployment serving queries.

**Acceptance Scenarios**:

1. **Given** the Helm chart and a values file with credentials,
   **When** the user runs the install command,
   **Then** the chart's pre-install lint and template validation pass and
   the deployment reaches readiness without manual intervention.
2. **Given** the deployment is running,
   **When** the user disables an integration in values and upgrades,
   **Then** the integration is deactivated in the running service without
   any other behavior changing.
3. **Given** a values file that enables an integration without supplying
   its credentials,
   **When** the user attempts to install,
   **Then** the rendered manifests still validate but the pod fails its
   readiness probe and the logs show the same fail-fast error described
   in Story 2.

---

### User Story 4 - Maintainer audits the fork for organization-specific content (Priority: P3)

A maintainer reviewing a contribution — or onboarding to the project — runs
a scripted audit over the working tree and confirms that no
organization-specific names, hostnames, account identifiers, or non-English
prompts have leaked into shipped code or configuration. The audit succeeds.

**Why this priority**: Vendor-neutrality is the project's defining
constitutional commitment. Without an automated check, regressions are
inevitable as contributions arrive.

**Independent Test**: A documented audit script (or CI step) runs over the
working tree and exits with status zero, listing zero violations across
the categories: prior-owner brand name, internal hostname patterns,
non-English text in shipped artefacts.

**Acceptance Scenarios**:

1. **Given** the working tree at any commit on the default branch,
   **When** the maintainer runs the audit script,
   **Then** the audit exits successfully with no violations reported.
2. **Given** a contributor's pull request that introduces an internal
   hostname,
   **When** CI runs the audit,
   **Then** the audit fails and points to the offending line.

---

### Edge Cases

- **Partial credentials**: An integration is enabled and some of its
  required environment variables are set but not all. The service must
  treat this the same as fully-missing credentials and refuse to start
  with a precise error.
- **Stale flags**: A user upgrades to a new release that has renamed or
  removed an integration. The service must report an explicit error
  describing the unknown flag rather than silently ignoring it.
- **Default-vs-override clash**: A user supplies an override that
  contradicts the constitutional default of "all integrations off".
  This is permitted, but the running service's status surface must make
  the active set discoverable.
- **Mixed-language inheritance**: A future contributor copies a prompt
  or log message from upstream notes that contains non-English text.
  The audit must catch this before merge.
- **Missing or unreachable identity provider**: The OIDC issuer URL is
  unset, the audience is unset, or the issuer's JWKS endpoint is not
  reachable at startup. The service must refuse to start with a
  single, precise error.
- **Expired or invalid Bearer token**: The API receives a request with
  no `Authorization` header, an unparseable token, a token signed by
  an untrusted issuer, or an expired token. The service must reject
  with HTTP 401 and a stable error code suitable for clients to act
  on, and must not leak claim contents in the response.
- **Helm chart drift**: A new integration is added to the codebase but
  the corresponding flag is not exposed in `values.yaml`. The integration
  exists but is unreachable via the canonical deployment path. CI must
  catch this.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The repository MUST NOT contain the previous owner's
  organization name in any shipped code, configuration, prompt, log
  message, container image, or chart artefact. A single historical
  reference in the project changelog (documenting the fork) is permitted.
- **FR-002**: All prompts, log messages, code comments, configuration
  comments, user-facing strings, and documentation in shipped artefacts
  MUST be written in English. Test fixtures explicitly verifying
  multilingual behaviour are exempt and MUST be marked as such.
- **FR-003**: Every external integration that exists in the repository
  MUST be controllable by an explicit enable/disable flag in the project
  configuration file. The default value of every such flag MUST be
  "disabled".
- **FR-004**: When an integration is marked enabled but its required
  credentials or configuration values are missing or invalid, the
  service MUST refuse to start and MUST emit a single, clear error
  identifying both the offending flag and the missing setting.
- **FR-005**: A Helm chart packaging the service MUST live under
  `deploy/helm/<chart-name>/` and MUST pass standard chart validation
  (lint and template rendering against the default values file) without
  errors.
- **FR-006**: The chart's default values file MUST expose an
  enable/disable toggle for every integration that exists in the
  codebase, and the wiring from values to running container MUST be
  verified by an automated test.
- **FR-007**: Every hostname, account identifier, repository slug,
  channel identifier, dashboard URL, or other deployment-specific value
  appearing in shipped configuration, examples, or documentation MUST
  be either a recognised generic placeholder (e.g. `example.com`,
  `CHANGE_ME`) or sourced from configuration at runtime.
- **FR-008**: A canonical example environment file MUST exist at the
  repository root, listing every environment variable the application
  reads, with placeholder values and a one-line description per variable.
- **FR-009**: The default configuration file MUST result in a service
  that starts successfully and responds to health checks using only
  the minimum core surface — LLM, vector store, the null graph
  backend (see FR-019), and the bundled local OIDC issuer (see
  FR-017) — with zero MCP integrations enabled.
- **FR-010**: The container image MUST run as a non-root user and MUST
  declare the application port as a configurable value rather than a
  hardcoded constant.
- **FR-011**: A new evaluator following the project's README MUST be
  able to bring up a working service and answer a query against an
  indexed sample document within fifteen minutes of cloning the
  repository, assuming they already have an LLM API key.
- **FR-012**: Every external integration MUST ship with an automated
  test that exercises its primary code path against a fake or recorded
  backend (no live network access required in CI).
- **FR-013**: The project MUST publish under the Apache License 2.0,
  with the full license text present at the repository root as
  `LICENSE` and the `Apache-2.0` SPDX identifier referenced in package
  metadata.
- **FR-014**: An audit script (runnable locally and in CI) MUST exist
  that, when invoked on the working tree, exits with status zero only if
  the constraints in FR-001, FR-002, and FR-007 hold.
- **FR-015**: The project changelog MUST contain an entry describing
  this port — its motivation (opensourcing a previously internal
  project), the categories of content removed (proprietary identifiers,
  non-English prompts), and a list of integrations newly placed behind
  feature flags.
- **FR-016**: All HTTP API endpoints other than the unauthenticated
  health surface (`/healthz`, `/readyz`) MUST require a valid OIDC
  Bearer token in the `Authorization` header. The service MUST verify
  token signatures against the configured issuer's JWKS endpoint and
  validate the standard `iss`, `aud`, and `exp` claims; the service
  MUST NOT ship its own user database, password store, or first-party
  API-key issuance. Token acquisition is the responsibility of the
  caller and the deployment's chosen identity provider.
- **FR-017**: The bundled local-development environment (the project's
  compose file) MUST include a pre-configured OIDC issuer suitable for
  evaluation — a local identity provider with at least one known test
  user — so the User Story 1 fifteen-minute timeline is achievable
  without an external identity provider.
- **FR-018**: When the OIDC issuer URL or audience is unset, or the
  configured issuer's JWKS endpoint is unreachable at startup, the
  service MUST fail to start with a single, clear error naming the
  missing setting — mirroring the integration fail-fast behaviour of
  FR-004.
- **FR-019**: The graph store is pluggable via configuration provider
  selection, NOT via the MCP-style enable/disable flag described in
  FR-003. The project MUST ship a first-class **null graph backend**
  as the default `knowledge_graph.provider` value; the null backend
  MUST satisfy the same interface contract as real backends and MUST
  return empty result sets for all queries without raising. The
  agent's graph-retrieval and graph-merge nodes MUST execute
  unconditionally — they MUST NOT branch on graph availability — and
  the merge step MUST tolerate empty graph results by relying solely
  on the other retrievers. Real graph backends (e.g. Neo4j) are
  selected by setting `knowledge_graph.provider` to the corresponding
  provider name and supplying that provider's required credentials,
  at which point FR-004's fail-fast rules apply.

### Key Entities *(include if feature involves data)*

- **Integration (a.k.a. MCP)**: A modular component providing the agent
  access to an external system — e.g. Kubernetes, an APM, an
  incident-tracker, a code-search backend, a graph snapshot. Attributes:
  canonical name, enabled flag, set of required environment variables,
  fake-backend fixture used in tests, status reported by the running
  service. The initial set inherited from upstream comprises fourteen
  named integrations. **Note**: backing stores (vector store, graph
  store, session checkpoint store) are NOT integrations in this
  sense — they are pluggable via provider selection (`vector_store.provider`,
  `knowledge_graph.provider`, `session.provider`) and do not use the
  on/off enabled flag pattern.
- **Configuration**: The runtime contract describing which providers,
  integrations, and parameters are active. Surfaces as a YAML file
  merged with environment variables. Validates at startup.
- **Helm Chart**: The packaged Kubernetes deployment artefact.
  Attributes: chart version, app version, documented values schema,
  manifests for the application workload plus any optional sidecars or
  companion resources.
- **Audit Report**: The output of the vendor-neutrality audit script,
  enumerating any violations of FR-001 / FR-002 / FR-007 by file and
  line.
- **Sample Corpus**: A small bundled set of synthetic runbooks /
  postmortems / configuration files used by README walkthroughs and by
  integration tests, sufficient to demonstrate non-trivial agent
  behaviour without exposing real operational content.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A new evaluator can go from clone to a successful sample
  query in under fifteen minutes following only the README.
- **SC-002**: Zero occurrences of the previous owner's organization
  name, internal hostnames, or proprietary account identifiers remain
  in shipped artefacts (verified by the automated audit). A single
  historical reference in the changelog is permitted and is the only
  exception.
- **SC-003**: Zero non-English characters appear in shipped prompts,
  log messages, comments, configuration, or documentation (verified by
  the automated audit). Multilingual test fixtures are explicitly
  excluded and labelled.
- **SC-004**: One hundred percent of integrations present in the
  codebase have an enable/disable flag exposed in both the project
  configuration file and the Helm chart's default values file.
- **SC-005**: The default configuration boots a healthy service with
  zero integrations enabled in under thirty seconds on a developer
  laptop.
- **SC-006**: The Helm chart passes lint and template validation in
  every continuous-integration run that touches chart files.
- **SC-007**: A startup attempt with an enabled-but-misconfigured
  integration produces an error message whose first line names both
  the offending flag and the missing setting.
- **SC-008**: The project's README and example configuration files
  contain only generic placeholder values for hostnames, identifiers,
  and credentials.
- **SC-009**: Every integration has an automated test against a fake
  backend that runs in continuous integration without external network
  access.

## Assumptions

- **Functional scope is preserved, not redesigned** (clarified
  2026-05-27 — full upstream scope): The initial port brings across
  the upstream project's complete functional surface. Concretely this
  comprises (1) the core agent and HTTP API, (2) all fourteen MCP
  integrations, (3) the autonomous Investigation agent for incident
  triage, (4) the React web UI for chat / usage / indexing, (5) the
  Slack bot (Socket Mode — outbound websocket, no public ingress),
  and (6) the Phoenix / DeepEval evaluation harness. Every subsystem
  is subject to the same vendor-neutrality, English-only, and
  feature-flag obligations as the core — the UI must ship with
  generic branding, the Slack bot's channel mappings must come from
  configuration, the Investigation agent's prompts and tool list must
  be in English, and the eval harness must reference only synthetic
  test data.
- **Integration set inherits from upstream**: The fourteen integrations
  currently in the source repository (covering Kubernetes, an APM,
  logs, infrastructure graph, DNS / edge networking, source-control,
  incident-tracker, chat, metrics, managed-SQL, knowledge base, code
  search, runbooks, and one shared cache) are the initial set placed
  behind feature flags. Future additions follow the same pattern.
- **License confirmed as Apache 2.0** (clarified 2026-05-27): The full
  license text ships at the repository root as `LICENSE` and the
  `Apache-2.0` SPDX identifier is referenced in package metadata.
  Contributions are accepted on the implicit terms of the Apache 2.0
  inbound-equals-outbound model unless a separate Contributor License
  Agreement is introduced later.
- **Internal session notes are not ported**: Working documents from the
  upstream repository — session logs, resume notes, dated status
  files — are intentionally left behind. Only durable design and
  reference documentation moves over, and only after sanitization.
- **English is the project's only first-class language**: The project
  does not adopt a localisation framework; existing non-English content
  is translated rather than wrapped.
- **Container runtime and Kubernetes are the canonical targets**: A
  bare-metal Python install remains possible for development, but
  shipped guidance assumes Docker for local evaluation and Kubernetes
  for production.
- **Adopters bring their own LLM key**: The project does not ship with
  an embedded model or a hosted endpoint. An evaluator supplies their
  own credential for at least one supported LLM provider.
- **Test suite migrates wholesale**: The upstream test suite moves over
  with the code; failures introduced by sanitization (e.g. tests that
  hardcoded internal identifiers) are fixed in-place during the port.
