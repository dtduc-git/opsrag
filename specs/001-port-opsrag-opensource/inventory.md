# Engine Context Inventory

**Status**: Phase 2 working document. Produced 2026-05-28 after the Phase 2
sanitization sweep (T016-T028). Captures every defect that violates the
project principle *"engine carries no organization knowledge; deployment
facts live in config; domain knowledge lives in the indexed corpus;
system prompts contain only org-agnostic reasoning patterns parameterized
via context."*

**Scope**: `opsrag/` source tree only. Excludes `samples/`,
`opsrag/eval/golden/`, `tests/`, `specs/`, `.specify/`, `.claude/` per the
allowlist convention.

## How to read this

Findings are bucketed into six categories. Categories A-D are *defects*
that must be remediated before the engine can claim vendor-neutrality.
Categories E-F are *legitimate references* that need rules-side
acknowledgement (allowlist entries), not code changes.

Each category names a target *tier* — where the content properly belongs:

| Tier | Definition |
|---|---|
| **config** | Deployment facts (services, environments, namespaces, URLs, tracker prefix, etc.) supplied at runtime by the operator. |
| **prompt** | Org-agnostic reasoning patterns; concrete identifiers come from rendering against the live `DeploymentContext`. |
| **corpus** | Domain knowledge (runbooks, postmortems, ticket history) retrieved at query time. Not pre-cooked into engine code or prompts. |
| **skill** | Engine capabilities, tool definitions, static logic with no per-deployment customization. |

---

## Category A — Module-level constants (16 items) → `config`

All clean fits. Each constant becomes a field on `DeploymentContext`.

| File | Line | Current | → Context field |
|---|---:|---|---|
| `opsrag/agent/repomap.py` | ~42-50 | `_KEY_REPOS_LAYOUT_HINTS = ["saas/acme-notes-be", "saas/acme-integration-api", "devops/gitops", ...]` | `deployment.key_repos` |
| `opsrag/agent/classifier.py` | ~223+ | `REFERENCE_EXAMPLES: dict[QueryCategory, list[str]]` | derived from `deployment.services` + augmented by optional `deployment.semantic_router_examples` (see Open Question 2) |
| `opsrag/agent/classifier.py` | 99 | comment `# acme-notes-be-prod-1, kafka-2` | drop or replace with `<service>-<env>-<pod>` placeholder |
| `opsrag/mcp/kubernetes.py` | 78 | `_KNOWN_CLUSTERS = (...)` | `deployment.kubernetes.clusters` |
| `opsrag/mcp/kubernetes.py` | (default fn) | `_default_cluster()` returns `acme-prod` | takes `DeploymentContext`, resolves env → cluster id |
| `opsrag/mcp/prometheus.py` | ~72 | `_default_cluster`, `_KNOWN_CLUSTERS` | same as kubernetes.py |
| `opsrag/mcp/cloudsql.py` | (top-of-module) | GCP project tuple (`acme-production`, `acme-preprod`, `acme-saas-shared`) | `deployment.cloud.gcp_projects` |
| `opsrag/mcp/slack.py` | 49 | `_WORKSPACE_URL_DEFAULT = "https://example.slack.com"` | `deployment.source_urls.slack` (required-or-None, no default) |
| `opsrag/config.py` | 155 | `base_url: str = "https://example.atlassian.net"` (Confluence) | `deployment.source_urls.confluence` |
| `opsrag/config.py` | 196 | `workspace_url: str = "https://example.slack.com"` | `deployment.source_urls.slack` |
| `opsrag/config.py` | 241 | `web_base_url: str = "https://example.rootly.com"` | `deployment.source_urls.rootly` |
| `opsrag/config.py` | 350-351 | `service_name = "eck-applogs-es-http"`, `service_namespace = "eck-system"` | `deployment.kubernetes.eck_elasticsearch_service`, `deployment.kubernetes.eck_elasticsearch_namespace` |
| `opsrag/config.py` | 394 | `default_env: str = "prod"` | derived from `deployment.environments[0]` (or explicit field on context) |
| `opsrag/mcp/elasticsearch.py` | 6, 130-132 | Module docstring + `_DEFAULT_POD_SELECTOR` referencing ECK label | `deployment.kubernetes.pod_label_selector` |
| `opsrag/slack_bot/config.py` | 31 | `workspace_url: str = "https://example.slack.com"` | `deployment.source_urls.slack` (same field as above) |
| `opsrag/agent/graph.py` | 476-479 | `SourceUrlBases` Pydantic model with `example.*` defaults | move into `DeploymentContext.source_urls`; this model should not exist independently |

---

## Category B — System-prompt examples (~256 hits across 17 files) → split between `prompt` and `corpus`

The cleanup-batch agents substituted a raw internal service name for a generic
placeholder (e.g. `<service> → acme-notes-be`) inside system prompts. Per the
directive those substitutions are themselves defects — the prompts should be
abstract patterns parameterized via context, not different hardcoded examples.

### B.1 — Structural reasoning patterns → `prompt`

Reasoning patterns and discipline rules. Stay in code as templates; concrete
identifiers come from `DeploymentContext` at render time.

| File | Notes |
|---|---|
| `opsrag/agent/nodes/multi_agent.py` | Triage / reasoner / generator system-prompt scaffolding (rules 1-12, routing logic, citation discipline). Becomes a template with `{services}` / `{tracker_prefix}` / `{environments}` / `{source_urls.*}` placeholders. |
| `opsrag/agent/prompts.py` | `_GENERATE_COMMON_RULES`, system templates. |
| `opsrag/agent/query_decomposer.py` | Few-shot frames. |
| `opsrag/agent/query_rewrite.py` | Coreference-resolution examples. |
| `opsrag/agent/classifier.py` | LLM-fallback prompt body (the abstract instructions; the *anchors* are a separate concern — see Category A). |
| `opsrag/runbooks/tagger.py` | Tagger system prompt. |
| `opsrag/runbooks/generator.py` | Generator system prompt. |
| `opsrag/agents/investigation/prompts.py` | Investigation system prompts. |

### B.2 — Operational heuristics ("the failure-mode catalog") → `corpus` at v2, **drop entirely at v1**

The ~30 illustrative anecdotes in `multi_agent.py` (SSO failure on the
auth service, the cartography drift trap, the Pomerium SSO architecture
explainer, the off-topic cap example, the runbook-grounded example, etc.)
are *substantive operational knowledge that assumes a specific
architecture*. They are not org-agnostic reasoning patterns.

Per the directive — *"Domain knowledge lives in the indexed corpus and is
retrieved at query time"* — they belong in the corpus, not the prompt.

**v1 rule** (the guardrail confirmed 2026-05-28):

> Apply a cross-org test to each retained pattern. Genuine org-agnostic
> reasoning → keep, abstracted. Operational heuristic that assumes
> specific architecture → DROP at v1; do not fake-abstract it. Move to
> corpus at v2. Do not de-name a heuristic and keep it.

**Outcome at v1**: most of the ~30 anecdotes get deleted from prompts.
The retained content is the structural lessons that pass the cross-org
test (e.g. "ground every cited fact in retrieved evidence" — yes;
"when SSO breaks in env X, suspect cert drift on auth service" — no,
that's an operational heuristic for a specific architecture).

**v2 follow-up**: the dropped anecdotes get rewritten as indexed runbooks
in `samples/runbooks/` (for the demo) and operator-provided indexed
runbooks (for production). Agent retrieves relevant ones at query time
via RAG.

### B.3 — MCP tool docstrings → `skill` (generic phrasing, no operator examples)

`opsrag/mcp/*.py` tool descriptions become LLM-facing capability
descriptions with abstract placeholders.

> *Skill-layer generic. Keep format-teaching examples with placeholder
> values (`<service>`, `<env>`, `<cluster>`). Drop topology examples that
> assume a specific deployment shape.*

Affected files: all 14 `opsrag/mcp/*.py` files. Tool descriptions
must teach the LLM the tool's call format and parameter semantics
without revealing any specific deployment's services/clusters/envs.

---

## Category C — Internal history tokens → **delete** (no tier)

| File | Match |
|---|---|
| `opsrag/feedback_store.py:3` | `Tier-1 enhancement T1.3 (TICKET-9715)` |
| `opsrag/vectorstores/qdrant.py:89` | `T1.6 (TICKET-9715)` |
| `opsrag/api/server.py:286` | `T1.3 (TICKET-9715)` |
| `opsrag/api/models.py:114` | `T1.3 (TICKET-9715)` |
| `opsrag/api/routes.py:978` | `Sub-sprint 3 V1 + T1.3 (TICKET-9715)` |
| `opsrag/agent/nodes/multi_agent.py:796` | `TICKET-9997` in a prompt example |

These are tech-debt comments referring to upstream sprint tasks. Delete
the references; keep the surrounding intent in plain prose.

---

## Category D — Env-discriminator regex → hybrid: `skill` (pattern) + `config` (env labels)

| File | Line | Current |
|---|---:|---|
| `opsrag/qa_cache.py` | 91-94 | `_DISCRIMINATOR_ENV = re.compile(r"\b(prod\|production\|preprod\|pre-prod\|staging\|dev\|test\|sandbox)\b", re.IGNORECASE)` (org-specific abbreviations come from `deployment.environments`) |
| `opsrag/agent/classifier.py` | similar pattern | same regex shape |

Implementation: at startup, build the regex dynamically from the union of
industry-standard tokens (`prod`, `staging`, `dev`, `test`) and the
operator-supplied `deployment.environments`. The regex engine (logic) is a
skill; the valid labels come from config.

Operator-specific env labels (e.g. `shared`, `preprod`, `prod`, or any custom
abbreviation an org uses) must come from `deployment.environments` — not
hardcoded.

---

## Category E — Public-API references → `skill` (legitimate dependencies, allowlist in `audit-rules.yaml`)

These are *not* organization knowledge. They are real public APIs the
engine talks to.

| File | Host/value | Why it's there |
|---|---|---|
| `opsrag/mcp/datadog.py` | `datadoghq.com`, `datadoghq.eu` | DD site defaults / URL-parse logic |
| `opsrag/mcp/slack.py`, `opsrag/sources/slack/client.py` | `slack.com/api` | Slack API base URL |
| `opsrag/mcp/elasticsearch.py` | `v4.channel.k8s.io` | K8s portforward subprotocol identifier (not a DNS host) |
| `opsrag/mcp/kubernetes.py` | `metrics.k8s.io` | K8s API group name |
| `opsrag/mcp/cartography.py` | `iam.gke.io/...` annotation | GCP Workload Identity annotation namespace |
| `opsrag/investigations/evaluator.py` | `app.kubernetes.io` | K8s recommended-labels namespace |
| `opsrag/agents/investigation/runbook_grounded.py` | `atlassian.net` | Code does `if "atlassian.net" in url` — functional string match |
| `opsrag/rerankers/cohere.py` | `api.cohere.com` | Cohere reranker endpoint |
| `opsrag/api/mcp_routes.py` | `html.spec.whatwg.org` | SSE spec docs link |
| `opsrag/agent/nodes/multi_agent.py:537` | `*.gserviceaccount.com` | Comment about GCP SA naming |
| `opsrag/agent/nodes/multi_agent.py:663` | `gateway.networking.k8s.io` | K8s CRD name in comment |
| `.github/workflows/release.yml` | `ghcr.io` | Image registry env var |
| `uv.lock` | `files.pythonhosted.org` (3,013) | Dependency download URLs |

**Action**: extend `scripts/audit-rules.yaml hardcoded_hosts_allowlist`
(already done as part of the rules-side fixes pass).

---

## Category F — Slack UX emojis → `skill` (UX strings, path-exempted)

23 emojis (👍 👎 👀 ✅ 🤔 😅 🙏 🐢 😔) in:

- `opsrag/slack_bot/streaming.py`
- `opsrag/slack_bot/handler.py`
- `opsrag/slack_bot/render.py`
- `opsrag/slack_bot/client.py`

These are **load-bearing Slack Block Kit UI strings and reaction emojis**.
Removing them degrades the bot UX. Path-exempted via `non_english_exempt_paths`
in audit-rules.yaml (already done).

---

## `DeploymentContext` schema (proposed)

The contract lives at
[`contracts/deployment-context.md`](contracts/deployment-context.md).
This file describes the v1 schema; field semantics, defaults, validation
rules, and the consistency invariant test specification are in the
contract.

```python
# opsrag/context.py (new module — created by T029)
from typing import Optional
from pydantic import BaseModel, Field


class SourceUrlBases(BaseModel):
    """URL bases for source systems — used to construct deep-links from
    retrieved chunks back to where they came from. All fields are
    optional; absent fields disable deep-linking for that source.
    No defaults; the engine carries no example URLs."""
    confluence: Optional[str] = None     # e.g. https://myorg.atlassian.net
    slack: Optional[str] = None          # e.g. https://myorg.slack.com
    gitlab: Optional[str] = None         # e.g. https://gitlab.myorg.com
    rootly: Optional[str] = None         # e.g. https://myorg.rootly.com
    github_org: Optional[str] = None     # e.g. https://github.com/my-org


class KubernetesContext(BaseModel):
    """K8s topology the agent can reason about."""
    clusters: dict[str, str] = Field(default_factory=dict)
    namespaces: list[str] = Field(default_factory=list)
    pod_label_selector: Optional[str] = None
    eck_elasticsearch_service: Optional[str] = None
    eck_elasticsearch_namespace: Optional[str] = None


class CloudContext(BaseModel):
    gcp_projects: dict[str, str] = Field(default_factory=dict)


class TrackerContext(BaseModel):
    prefix: str = "TICKET"
    web_base_url: Optional[str] = None


class DeploymentContext(BaseModel):
    """Operator-supplied deployment facts.

    The engine itself carries NO organization-specific knowledge. Anything
    the agent needs to know about the operator's deployment — services,
    environments, namespaces, URL bases, tracker prefix — is supplied here
    at runtime via the application's config (config.yaml or env vars)."""

    organization_label: Optional[str] = None   # display-only label for prompts; never logged
    services: list[str] = Field(default_factory=list)
    environments: list[str] = Field(default_factory=list)
    key_repos: list[str] = Field(default_factory=list)

    kubernetes: KubernetesContext = Field(default_factory=KubernetesContext)
    cloud: CloudContext = Field(default_factory=CloudContext)
    tracker: TrackerContext = Field(default_factory=TrackerContext)
    source_urls: SourceUrlBases = Field(default_factory=SourceUrlBases)

    # Optional operator augmentation for semantic-router anchors. When
    # absent, the engine derives anchors from `services` + per-category
    # templates at startup. When present, additional examples are merged
    # in (operator adds query shapes the templates wouldn't produce).
    semantic_router_examples: dict[str, list[str]] = Field(default_factory=dict)
```

### Confirmed design decisions (2026-05-28)

| # | Decision | Rationale |
|---|---|---|
| 1 | **v1 anecdote strategy = cross-org test, then drop or abstract** | Don't de-name and keep an operational heuristic. If a pattern doesn't pass the cross-org test (i.e. it assumes a specific architecture), drop it at v1. Re-introduce via the corpus at v2 — not via templated prompt examples. |
| 2 | **REFERENCE_EXAMPLES strategy = derive-default + operator-augment** | Default: synthesize from `deployment.services` + per-category templates. Operators augment via `deployment.semantic_router_examples`. Pure-derived would force engine fork to fix routing → violates directive. |
| 3 | **MCP tool docstrings = skill-layer generic** | Keep format-teaching examples with `<service>` / `<env>` placeholders. Drop topology examples that assume a specific deployment shape. |
| 4 | **`_KEY_REPOS_LAYOUT_HINTS` = config in v1** | Simple `list[str]` on `DeploymentContext.key_repos`. Corpus-derivation (from repo-touch frequency in the index) is a later optimization. |

### Consistency invariant

> Golden eval fixture names (in `opsrag/eval/golden/*.yaml`) MUST match
> services that exist in the `samples/` corpus. The contract specifies
> a contract test that loads both sides and asserts the intersection
> is exact. This protects against the failure mode where someone
> updates `samples/` and the eval harness silently passes against
> stale fixture names.

The test specification lives in the contract document; implementation
arrives with the eval-regression CI job (T150).

---

## Migration plan

The migration is sequenced as a series of tasks. Each task is small and
independently reviewable.

| Step | Scope | Status | Notes |
|---|---|---|---|
| 1. Update constitution principle | `.specify/memory/constitution.md` | not started | Encode the "engine carries no org knowledge" rule as a principle. Existing Principle I (Vendor-Neutral by Default) gets extended or a new principle added. |
| 2. Spec the contract | `contracts/deployment-context.md` | not started | This file gets a sibling contract that formalises the schema, field semantics, derivation rules, consistency invariant test. |
| 3. T029 (re-scoped): config + context schema | `opsrag/config.py`, `opsrag/context.py` | not started | T029 in tasks.md becomes "rewrite config.py as Pydantic-v2 Settings AND introduce DeploymentContext". |
| 4. Prompt-templating helper | `opsrag/agent/prompt_render.py` (new) | not started | New task carved out of T055-T056. Takes a prompt template and renders against `DeploymentContext`. |
| 5. Refactor agent prompts | `opsrag/agent/nodes/multi_agent.py`, `opsrag/agent/prompts.py`, `opsrag/agent/query_decomposer.py`, `opsrag/agent/query_rewrite.py`, `opsrag/runbooks/tagger.py`, `opsrag/runbooks/generator.py`, `opsrag/agents/investigation/prompts.py` | not started | Apply cross-org test per the v1 rule above. Drop or abstract. |
| 6. Move module-level constants → context | Files in Category A | not started | Mechanical refactor once the context schema lands. |
| 7. Refactor env-discriminator regex | `opsrag/qa_cache.py`, `opsrag/agent/classifier.py` | not started | Build regex dynamically from `DeploymentContext.environments`. |
| 8. Strip `example.*` defaults from config | `opsrag/config.py`, `opsrag/slack_bot/config.py`, `opsrag/agent/graph.py`, `opsrag/mcp/slack.py` | not started | Make required-or-None. |
| 9. Remove internal-history tokens | Files in Category C | not started | Find/replace. |
| 10. Extend audit script with engine-purity check | `scripts/audit-vendor-neutrality.sh` | not started | Optional new check: any `acme-`/`eck-`/etc. literal outside allowlisted paths is a violation. |
| 11. Consistency-invariant test | `tests/contract/test_eval_fixtures_match_samples.py` (new) | not started | Implements the invariant described above. |
| 12. Re-audit | — | not started | Expect clean. |

### What's NOT touched here

- `opsrag/auth/pomerium.py` — scheduled for full replacement in T032; will be deleted when `auth/oidc.py` ships.
- `opsrag/eval/golden/*.yaml` — these legitimately contain concrete `acme-*` names because they validate behaviour against the synthetic `samples/` corpus. They are corpus tier (test fixtures); the consistency invariant covers them.
- `samples/runbooks/`, `samples/postmortems/`, etc. — synthetic demo content; allowed concrete names.
