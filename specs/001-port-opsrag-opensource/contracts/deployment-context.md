# Contract: `DeploymentContext`

**Status**: New for this feature. Implements Constitution Principle VI
(Context-Driven Engine). Companion contract to
[`config-schema.md`](./config-schema.md).

## Purpose

`DeploymentContext` is the single Pydantic v2 model that captures every
piece of organisation-specific knowledge the opsrag engine needs at
runtime. It is the *only* place in shipped code where deployment-specific
identifiers may appear (as field values supplied by the operator;
the schema itself contains no defaults that name any organisation).

The engine reads from this model. The engine never invents values for it.

## Schema (Python)

The canonical source lives at `opsrag/context.py` (created by T029). The
schema below is the v1 contract.

```python
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class SourceUrlBases(BaseModel):
    """Base URLs for the source-of-truth systems the operator runs.

    Used to construct deep-links from retrieved chunks back to the
    source system (Confluence page, Slack message, GitLab MR, Rootly
    incident, GitHub repo). All fields are optional. Absent fields
    disable deep-linking for the corresponding source.

    No defaults: the engine carries no example URLs."""

    model_config = ConfigDict(extra="forbid")

    confluence: Optional[str] = None   # e.g. https://myorg.atlassian.net
    slack: Optional[str] = None        # e.g. https://myorg.slack.com
    gitlab: Optional[str] = None       # e.g. https://gitlab.myorg.example.com
    rootly: Optional[str] = None       # e.g. https://myorg.rootly.com
    github_org: Optional[str] = None   # e.g. https://github.com/my-org


class KubernetesContext(BaseModel):
    """K8s topology the agent can reason about.

    `clusters`: env name -> cluster identifier (the value the operator
    uses with their kubectl context).
    `namespaces`: known application namespaces; used to scope queries
    that don't name a namespace explicitly.
    `pod_label_selector`: the recommended-labels selector convention the
    operator's workloads use (e.g. "app.kubernetes.io/name");
    drives default pod discovery in the K8s MCP.
    `eck_elasticsearch_*`: the operator's Elastic Cloud on K8s deployment
    coordinates, when present; absent disables the ECK shortcut in the
    Elasticsearch MCP."""

    model_config = ConfigDict(extra="forbid")

    clusters: dict[str, str] = Field(default_factory=dict)
    namespaces: list[str] = Field(default_factory=list)
    pod_label_selector: Optional[str] = None
    eck_elasticsearch_service: Optional[str] = None
    eck_elasticsearch_namespace: Optional[str] = None


class CloudContext(BaseModel):
    """Cloud-platform deployment facts.

    `gcp_projects`: env name -> GCP project id. Empty dict if the
    operator doesn't use GCP or the relevant MCPs are disabled."""

    model_config = ConfigDict(extra="forbid")

    gcp_projects: dict[str, str] = Field(default_factory=dict)


class TrackerContext(BaseModel):
    """Ticket / incident tracker.

    `prefix`: the prefix the operator's tracker emits (e.g. "JIRA",
    "INC", "OPS", or a custom value). Used both to recognise ticket
    references in user queries and to format ticket links in agent
    output. The engine has no default opinion about which tracker
    prefix is correct; "TICKET" is a deliberate placeholder shape and
    the operator is expected to override.
    `web_base_url`: deep-link base for ticket URLs (e.g.
    https://myorg.atlassian.net/browse). Absent disables deep-linking."""

    model_config = ConfigDict(extra="forbid")

    prefix: str = "TICKET"
    web_base_url: Optional[str] = None


class DeploymentContext(BaseModel):
    """Operator-supplied deployment facts.

    The engine itself carries NO organisation-specific knowledge.
    Anything the agent needs to know about the operator's deployment is
    supplied here at runtime via the application's config (config.yaml
    or env vars). See Constitution Principle VI."""

    model_config = ConfigDict(extra="forbid")

    organization_label: Optional[str] = None
    services: list[str] = Field(default_factory=list)
    environments: list[str] = Field(default_factory=list)
    key_repos: list[str] = Field(default_factory=list)

    kubernetes: KubernetesContext = Field(default_factory=KubernetesContext)
    cloud: CloudContext = Field(default_factory=CloudContext)
    tracker: TrackerContext = Field(default_factory=TrackerContext)
    source_urls: SourceUrlBases = Field(default_factory=SourceUrlBases)

    semantic_router_examples: dict[str, list[str]] = Field(
        default_factory=dict
    )
```

## Field semantics

### Root

| Field | Type | Default | Required for | Notes |
|---|---|---|---|---|
| `organization_label` | `Optional[str]` | `None` | None | Display-only label used in some prompt templates. **MUST never be logged.** When `None`, prompts use generic phrasing ("the engineer using this system"). When present, prompts may say "an engineer at {organization_label}". |
| `services` | `list[str]` | `[]` | Live-query routing, classifier anchors | Service names operators want the agent to recognise in user queries. Engine derives semantic-router anchors from this list. |
| `environments` | `list[str]` | `[]` | Env classifier, alert parsing | Free-form labels (e.g. `["prod", "staging"]` or `["us-east-1-prod", "us-east-1-staging"]`). The engine combines these with industry-standard tokens (`prod`, `staging`, `dev`, `test`) to build the env discriminator regex. |
| `key_repos` | `list[str]` | `[]` | Repo-map agent hints | Most-touched repos. Drives the agent's repo-map cache priority. |
| `kubernetes` | `KubernetesContext` | empty | K8s MCP, Elasticsearch MCP, Prometheus MCP | Required when the corresponding MCP is enabled. |
| `cloud` | `CloudContext` | empty | CloudSQL MCP, GCP cartography paths | Required when those MCPs are enabled. |
| `tracker` | `TrackerContext` | `prefix="TICKET"` | Ticket recognition / deep-linking | `prefix="TICKET"` is the engine's neutral default; operators override. |
| `source_urls` | `SourceUrlBases` | all `None` | Deep-link construction | When a field is `None`, the agent omits the corresponding deep-link section in answers. |
| `semantic_router_examples` | `dict[str, list[str]]` | `{}` | Classifier augmentation | Optional operator augmentation; merged with engine-derived anchors at startup. See "Derivation rules" below. |

### `SourceUrlBases`

All fields are `Optional[str]`. **The engine ships with no defaults for any
of these.** Any `example.com`-style placeholder in engine code is a
defect under Principle VI.

### `KubernetesContext`

`clusters` is keyed by env. The engine asks "give me the prod cluster id"
by `ctx.kubernetes.clusters.get(ctx.environments[primary_env_index])`. If
the lookup misses, the agent surfaces a structured error ("env X is in
`environments` but has no cluster mapping in `kubernetes.clusters`")
rather than substituting a default.

`pod_label_selector` defaults to `None`. When `None`, the K8s MCP requires
the user to specify selectors per-call. When present (e.g.
`"app.kubernetes.io/name"`), the MCP infers selectors from named services.

### `TrackerContext`

`prefix` defaults to `"TICKET"`. This is deliberately a *non-functional*
placeholder — it is recognisable as a placeholder shape and signals to
the operator that override is expected. The engine recognises the
specific string `"TICKET"` for ticket parsing without claiming any
particular meaning for it.

## Derivation rules

### Semantic-router anchors

Two-step composition at startup:

1. **Engine-derived baseline**. For each `QueryCategory` (LIVE, FORENSIC,
   PROCEDURAL, etc.), the engine renders a small set of anchor templates
   (in code) against `deployment.services` and `deployment.environments`.

   Example LIVE templates: `"is {svc} down right now?"`,
   `"how is {svc} in {env}?"`. With
   `services=["api-gateway","payments"]` and
   `environments=["prod","staging"]`, the engine produces 4
   anchor strings per template.

2. **Operator augmentation**. If `deployment.semantic_router_examples`
   contains entries for that category, they are appended to the
   derived list.

Operators add to `semantic_router_examples` only when the templates do
not capture a query shape they care about (e.g. domain-specific phrasing
the engine's templates would not produce). Pure-derived would force
operators to fork the engine to fix routing — a Principle VI violation.

### Env-discriminator regex

```
valid_env_tokens = ["prod", "production", "staging", "dev", "test"] + deployment.environments
_DISCRIMINATOR_ENV = re.compile(r"\b(" + "|".join(re.escape(t) for t in valid_env_tokens) + r")\b", re.IGNORECASE)
```

Built dynamically at startup. Industry-standard tokens are baked into
the engine (they pass the cross-org test). Org-specific tokens come from
config.

### Tracker prefix recognition

```
pattern = rf"\b({deployment.tracker.prefix})-\d{{2,}}\b"
```

Built dynamically. The engine does not enumerate possible prefixes (no
`INC`, `OPS`, `JIRA` list).

## Validation rules

Schema validation runs at config-load (Pydantic v2 `model_validate`).
Additional cross-field checks:

1. Every key in `kubernetes.clusters` SHOULD appear in `environments`.
   Engine emits a warning at startup; not a fatal error.
2. Every key in `cloud.gcp_projects` SHOULD appear in `environments`.
   Same — warning, not fatal.
3. `services` and `environments` MUST NOT contain values listed in
   `audit-rules.yaml`'s `proprietary_names_denylist`. Engine raises
   `ConfigError` at startup. (Operator's own service names are
   unlikely to collide; this guards against accidentally pasting in
   demo / sample-corpus identifiers.)
4. `tracker.prefix` MUST be `[A-Z][A-Z0-9_-]{1,15}`. Engine raises
   `ConfigError` otherwise.

## Consistency invariant (CI gate)

> Synthetic identifiers used in `samples/` MUST match the identifier set
> referenced from `opsrag/eval/golden/*.yaml`. Golden fixtures may
> reference only identifiers that exist in `samples/`; no other
> identifier is permitted.

### Why

The eval golden fixtures validate agent behaviour against the synthetic
demo corpus. If `samples/runbooks/` says the demo product is
`acme-notes-be` and a golden fixture asks "what does
`acme-notes-be-v2` do?", the eval silently passes (no retrieval match,
agent says it doesn't know) when it should fail (the fixture references
something that doesn't exist).

### Test specification

A new contract test
`tests/contract/test_eval_fixtures_match_samples.py` MUST exist and pass
in CI. The test:

1. Walks `samples/` and extracts the set of synthetic identifiers it
   defines. Identifier extraction follows a documented convention (e.g.
   each runbook front-matter declares a `service:` field and a list of
   `aliases:`; the parser collects these).
2. Walks `opsrag/eval/golden/*.yaml` and extracts every identifier
   referenced (from `must_contain` lists, expected-answer text, query
   text — wherever the fixture format puts them).
3. Asserts `fixture_identifiers <= sample_identifiers`. Set difference
   in either direction surfaces as a test failure with the offending
   identifiers named.

The test is added in the same PR that introduces the consistency
convention (`service:` / `aliases:` front-matter in `samples/`). Until
that PR lands, the test is `@pytest.mark.skip(reason="TBD: convention")`.

## Migration impact

When this contract is implemented (T029 + the prompt-abstraction task):

- **Removed**: every `acme-*` literal in `opsrag/` (Category B), every
  module-level hardcoded constant (Category A), every internal-history
  ticket reference (Category C).
- **Added**: `opsrag/context.py` (the schema), `opsrag/agent/prompt_render.py`
  (the templating helper), `opsrag/api/middleware.py` enrichment that
  attaches `DeploymentContext` to every agent request.
- **Changed**: `opsrag/config.py` grows a `deployment: DeploymentContext`
  field on `Settings`. `config.yaml`'s top-level grows a `deployment:`
  section. The example file `config-example.yaml` documents every
  context field.

## Audit alignment

`audit-vendor-neutrality.sh` SHOULD eventually gate Principle VI directly
(beyond the existing denylist-token detection). A proposed Check 4:

- **Check 4 — Engine purity.** No literal string in `opsrag/` outside
  the allowlisted paths (`samples/`, `tests/`, `opsrag/eval/golden/`)
  may match the structural shape of a hardcoded deployment identifier:
  - Hostnames or path segments that look like
    `[a-z][a-z0-9-]+-(be|fe|svc|api|service|worker|consumer|backend|frontend)`
  - Module-level constants whose name is `[A-Z_]+_(CLUSTERS|HOSTS|SERVICES|REPOS|PROJECTS)` and whose value is a non-empty list/tuple/dict of string literals.

  Implementation is deferred to a follow-up; the v1 audit catches these
  via the `audit-rules.yaml proprietary_names_denylist` for known tokens
  only.
