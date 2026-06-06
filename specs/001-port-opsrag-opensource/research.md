# Phase 0 Research: Port opsrag as a vendor-neutral opensource project

**Branch**: `001-port-opsrag-opensource` | **Date**: 2026-05-28 |
**Status**: Resolved (no remaining NEEDS CLARIFICATION)

Scope: resolve every planning-phase unknown introduced by the spec or surfaced
during `/speckit-clarify`. Each section follows the **Decision /
Rationale / Alternatives** structure.

---

## 1. Sanitization audit (FR-014)

**Decision**: Ship a single bash script at `scripts/audit-vendor-neutrality.sh`
that runs three independent checks and exits non-zero if any check fails:

1. **Proprietary-name scan**: case-insensitive `grep -rE` for a curated
   denylist (e.g. `acme`, internal hostnames like `*.acme.com`,
   internal account IDs) across tracked files, excluding `CHANGELOG.md` and
   `samples/` test fixtures explicitly marked.
2. **Non-English text scan**: `LC_ALL=C grep -rlE '[À-ỹ]'` (and
   broader CJK / Cyrillic / Hangul ranges) across `*.py`, `*.tsx`, `*.ts`,
   `*.md`, `*.yaml`, `*.yml`, `*.sh`, `*.tpl`. Files under
   `tests/fixtures/i18n/` are exempt if explicitly tagged.
3. **Hardcoded-host scan**: regex for non-placeholder hostnames (`.com`,
   `.net`, `.io`, `.dev`, etc.) outside an allowlist of recognised
   placeholders (`example.com`, `example.org`, `localhost`,
   `*.example.com`, `*.invalid`, `127.0.0.1`).

Each check is its own function for unit-test isolation. The script supports
`--json` for CI machine-readable output and `--fix-suggestions` for human use.

**Rationale**: Bash plus `grep` is universal, has no extra build dependency,
and the script's correctness can itself be unit-tested with `bats`. Three
distinct checks let CI report which rule was violated. Putting the denylist
into the script (vs. a config file) keeps the audit immutable for any given
commit and makes regressions visible in `git diff`.

**Alternatives considered**:
- Python implementation: more flexible but adds a dependency just for an
  audit tool; rejected because `grep` is already universal and the regex
  ranges are stable.
- `pre-commit` hook only: catches local regressions but misses CI gating;
  the script doubles as a `pre-commit` hook and a CI step.
- Embedded as a `ruff`/`mypy` plugin: tightly couples sanitization to
  Python tooling, missing chart / docs / UI files; rejected.

---

## 2. Bundled local OIDC provider (FR-017)

**Decision**: **Dex** (`dexidp/dex`), pinned to an image tag, configured via
`deploy/compose/dex/config.yaml` with a single static user (`evaluator@example.com`
/ password documented in the bring-up walkthrough) and a single static client
(`opsrag-local`). The backend's `auth.issuer` value defaults to the Dex
service URL on the compose network; outside compose, the operator overrides
to their real IdP.

**Rationale**:
- Dex is itself Apache-2.0 and is the canonical "small OIDC server" in the
  Kubernetes ecosystem (used by Argo CD, Tekton, etc.) — familiar to the
  target audience.
- Static config from a single YAML file is enough for the local-evaluator
  story; no separate database.
- Image size ~30 MB, startup < 5 s — does not threaten SC-005's 30-second
  boot budget.

**Alternatives considered**:
- **Keycloak**: more powerful but ~10× the resource footprint and a heavy
  bootstrap; not justified for an evaluator path.
- **Ory Hydra**: requires a separate user-management surface; adds friction.
- **Self-signed JWT issuance script**: trivially insecure pattern; rejected.
- **`auth.mode: open` opt-out**: superseded by the Q3 clarification — the
  project chose OIDC-only, so even the dev path uses real OIDC.

---

## 3. Sample-corpus origin

**Decision**: Ship a small, **wholly synthetic** sample corpus under
`samples/` — five runbooks, three postmortems, two K8s manifests, and one
Terraform module — written from scratch for the project, describing a
plausible-but-fictional SaaS ("Acme Notes"). The samples are checked into
the repo and indexed automatically by `scripts/seed-sample-corpus.sh` during
local bring-up. The corpus is referenced in the README walkthrough.

**Rationale**:
- Sanitised excerpts of real internal documents always carry residual
  organisational context (and risk leaking it through the audit's tail).
  Synthetic is safer and clearer.
- A fictional product name ("Acme Notes") gives every sample a consistent
  voice without naming any real organisation.
- Small footprint (≤ 20 documents) keeps repo size sane and indexing
  time inside the 30-second boot target.

**Alternatives considered**:
- Pull from public SRE postmortem corpora (Google SRE Workbook excerpts,
  Github incident reports): licensing complexity, attribution overhead,
  and inconsistent style.
- Generate via LLM at build time: introduces a network dependency and
  non-determinism in tests.
- Ship no samples; require users to bring their own: kills SC-001 (the
  fifteen-minute first-answer story).

---

## 4. Default observability backend

**Decision**: Ship **Phoenix in the local compose stack** as the default
observability target during local evaluation; ship **console-only** as the
default in `config.yaml` and `values.yaml` for production deployments.
Operators opt into Phoenix in production by switching
`observability.provider` to `phoenix` and setting the OTLP endpoint.

**Rationale**:
- Local evaluators benefit from a graphical trace view immediately — and
  Phoenix is the upstream's native instrumentation, so traces already render
  well.
- Production deployments often have an existing observability stack (Datadog,
  Honeycomb, vendor-of-the-day); defaulting to console keeps the chart
  side-effect-free and lets adopters wire to their own backend via the
  standard OTel exporter envs.

**Alternatives considered**:
- Phoenix in production by default: assumes Phoenix infrastructure the
  adopter may not run; violates Principle II in spirit.
- Console-only across the board: deprives evaluators of the strongest
  feature of the upstream agent (rich trace visualisation).
- Jaeger: less aligned with the LangGraph/LangChain ecosystem instrumentation
  already in the codebase.

---

## 5. MCP integration test-fake strategy

**Decision**: Adopt a **hybrid** strategy:
- For HTTP-based MCPs (Datadog, GitLab, Cloudflare, Rootly, Slack,
  Elasticsearch, Prometheus, Knowledge): use `pytest-httpx` (or `respx`)
  to record responses inline in the test file. No live network.
- For Kubernetes (`kubernetes_asyncio`): use the upstream's existing
  hand-rolled `FakeApiClient` pattern (carried forward from
  `tests/unit/test_*_mcp.py`), since the K8s client API is rich and
  per-resource.
- For Cartography (Neo4j Cypher) and CloudSQL (gRPC-backed): boot a
  one-off Neo4j 5 testcontainer and a CloudSQL emulator only inside their
  respective integration tests, gated by a `requires_neo4j` marker so CI
  can skip them in fast tiers if needed.

**Rationale**:
- `pytest-httpx` is well-maintained, lightweight, and produces failure
  messages that show which request fired — easier to debug than recorded
  cassettes (vcr.py).
- Hand-rolled fakes for K8s match the upstream pattern and the existing
  ~25 K8s-MCP unit tests already use them; rewriting them would be churn.
- Testcontainers for Neo4j is a one-line setup and exercises the actual
  Cypher path; the alternative (mocking the driver) is brittle.

**Alternatives considered**:
- `vcr.py` cassettes everywhere: cassettes drift silently when upstream
  APIs change and produce inscrutable diffs when re-recording.
- `wiremock` standalone: requires JVM in CI; out of proportion for a
  Python project.

---

## 6. Helm chart conventions

**Decision**: Align with the **Bitnami / CNCF chart style**:
- `Chart.yaml` declares `apiVersion: v2`, `type: application`, an icon URL
  to a project-hosted asset (no third-party CDNs), and the `Apache-2.0`
  license SPDX.
- `values.yaml` is fully commented, every key has a default, and the file
  doubles as the values documentation. Every `mcp.<name>` block lives
  under a top-level `mcp:` map; every block has at minimum `enabled: false`
  and `secretRef:` placeholder.
- `values.schema.json` constrains shape and rejects unknown top-level
  keys (catches the "stale flag" edge case from the spec).
- `templates/_helpers.tpl` defines `opsrag.fullname`, `opsrag.labels`,
  `opsrag.selectorLabels`, `opsrag.serviceAccountName`,
  `opsrag.image` consistent with the Helm helpers idiom.
- `templates/tests/test-connection.yaml` provides a `helm test` hook that
  curls `/healthz` from inside the cluster.
- `NOTES.txt` prints the post-install URL and reminds the operator to
  enable any MCPs they need.

**Rationale**: Bitnami chart conventions are the *de facto* opensource Helm
norm and produce charts that work cleanly with Argo CD, FluxCD, and Helm CLI
alike. `values.schema.json` is the only way to catch unknown-flag drift
mechanically — a direct response to the spec's "stale flags" edge case.

**Alternatives considered**:
- Replicate upstream's existing chart as-is: it lacks `values.schema.json`,
  PDB, NetworkPolicy, NOTES.txt — falls short of FR-005's "world-class
  opensource chart" intent.
- Generate via Kustomize instead: loses Helm-ecosystem tool support
  (Argo CD, Helmfile, etc.) without compensating gain.

---

## 7. Slack bot deployment shape

**Decision**: **Separate Kubernetes Deployment** in the same Helm release
(`templates/slackbot-deployment.yaml`), gated by
`slack_bot.enabled = false` in `values.yaml`. The bot uses Socket Mode
(outbound websocket to Slack); it needs no Service, Ingress, or public
hostname. It shares the backend container image but with a different
entrypoint command.

**Rationale**:
- The bot's lifecycle is independent of the HTTP API (it can crash-restart
  without disturbing user requests).
- A single image keeps build / scan pipelines simple; the entrypoint
  switches role via env var (`OPSRAG_ROLE=slackbot`).
- Slack Socket Mode requires no inbound ingress, so a public hostname is
  neither needed nor wanted.

**Alternatives considered**:
- Run the bot as a sidecar in the API pod: couples lifecycle, complicates
  HPA, and double-counts CPU/RAM in resource requests.
- Separate image: doubles the Docker build/test/scan surface for marginal
  benefit.

---

## 8. Image strategy

**Decision**: **One Docker image, multiple roles** selected at container
start via the `OPSRAG_ROLE` env var (`api`, `slackbot`, `scheduler`,
`investigator`). The Dockerfile is multi-stage (Python builder → distroless
runtime), runs as user `1000:1000`, and the entrypoint script dispatches
into the appropriate uvicorn / worker / bot main. UI image is separate
(Vite-built static assets served by `nginx-unprivileged`).

**Rationale**:
- One image to scan, sign, and version. Simpler SBOM.
- Role-multiplexing via env var matches upstream's existing
  `docker-entrypoint.sh` pattern; minimal porting work.
- UI is genuinely a different stack (static assets, no Python) and merits
  its own image.

**Alternatives considered**:
- Three images (api / slackbot / investigator): 3× the registry footprint,
  three CVE-scan runs, and any shared code bug requires three updates.
- Single image including UI assets: bloats backend image with frontend
  artefacts; rebuilds backend on UI changes.

---

## 9. License header policy

**Decision**: **No per-file license header.** Apache-2.0's official guidance
permits omitting per-file boilerplate when the `LICENSE` file is present and
the package metadata identifies the license. A short `NOTICE` file at the
repository root carries the optional Apache-2.0 attribution notice and any
third-party notices required by vendored dependencies.

**Rationale**:
- Per-file headers add ~12 lines × 200+ files of boilerplate noise and
  contribute nothing the `LICENSE` file doesn't already cover under
  Apache-2.0 §4.
- Mainstream opensource projects in this space (Kubernetes, Helm, Argo CD)
  omit per-file headers.

**Alternatives considered**:
- Full Apache header on every file: vendor-default for some legal teams
  but not legally required; rejected for noise/value ratio.
- SPDX-only one-liner per file (`# SPDX-License-Identifier: Apache-2.0`):
  considered, but adds maintenance burden without legal upside given that
  `pyproject.toml` already declares the SPDX identifier.

---

## 10. Multi-tenant model

**Decision**: **Single-tenant per install.** A given opsrag deployment
serves one organisation; user identity comes from the configured OIDC IdP
and is used for audit/usage attribution, not for data isolation. Adopters
who need multi-tenancy run multiple deployments.

**Rationale**:
- Multi-tenant data isolation in a RAG + graph + agent system is a deep
  feature (per-tenant vector namespaces, graph filters, retrieval policies)
  and not implied by anything in the spec.
- Single-tenant matches upstream's existing data model — porting it as-is
  preserves the ~225-test suite without redesign.

**Alternatives considered**:
- Bolt-on multi-tenancy via row-level security: adds significant scope
  (new tests, new RBAC contracts) for a feature no FR demands.
- Per-namespace tenancy at the Kubernetes level: already supported simply
  by running multiple Helm releases.

---

## Resolved NEEDS CLARIFICATION

None outstanding. The plan and downstream artefacts (data-model, contracts,
quickstart) can proceed against the decisions above.
