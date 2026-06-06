"""DeploymentContext -- operator-supplied deployment facts.

The opsrag engine carries no organization-specific knowledge. Anything the
agent needs to know about the operator's deployment -- services,
environments, namespaces, cluster identifiers, URL bases, ticket prefix,
key repos -- is supplied at runtime via the ``DeploymentContext`` model
defined here.

See:

- Constitution Principle VI (Context-Driven Engine) in
  ``.specify/memory/constitution.md``.
- The full schema contract in
  ``specs/001-port-opsrag-opensource/contracts/deployment-context.md``.

The engine reads from this model. The engine never invents values for it.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourceUrlBases(BaseModel):
    """Base URLs for source-of-truth systems the operator runs.

    Used to construct deep-links from retrieved chunks back to the source
    system (Confluence page, Slack message, GitLab MR, Rootly incident,
    GitHub repo). All fields are optional. An absent field disables deep-
    linking for the corresponding source.

    No defaults: the engine carries no example URLs.
    """

    model_config = ConfigDict(extra="forbid")

    confluence: str | None = None
    slack: str | None = None
    gitlab: str | None = None
    rootly: str | None = None
    github_org: str | None = None


class KubernetesContext(BaseModel):
    """Kubernetes topology the agent can reason about.

    ``clusters`` is keyed by env name (the value the operator uses with
    their kubectl context). Lookup misses surface as structured errors
    rather than substituted defaults.

    ``pod_label_selector`` is the recommended-labels selector convention
    the operator's workloads use (for example, ``app.kubernetes.io/name``);
    drives default pod discovery in the K8s MCP. Absent means the MCP
    requires explicit selectors per-call.
    """

    model_config = ConfigDict(extra="forbid")

    clusters: dict[str, str] = Field(default_factory=dict)
    namespaces: list[str] = Field(default_factory=list)
    pod_label_selector: str | None = None


class CloudContext(BaseModel):
    """Cloud-platform deployment facts.

    ``gcp_projects`` is keyed by env name. Empty dict if the operator does
    not use GCP or the relevant MCPs are disabled.
    """

    model_config = ConfigDict(extra="forbid")

    gcp_projects: dict[str, str] = Field(default_factory=dict)


class TrackerContext(BaseModel):
    """Ticket / incident tracker.

    ``prefix`` is the prefix the operator's tracker emits (e.g. ``JIRA``,
    ``INC``, ``OPS``). Used both to recognize ticket references in user
    queries and to format ticket links in agent output. The default
    ``TICKET`` is a deliberate placeholder shape: it is recognizable as a
    placeholder and signals to the operator that an override is expected.
    """

    model_config = ConfigDict(extra="forbid")

    prefix: str = "TICKET"
    web_base_url: str | None = None

    @field_validator("prefix")
    @classmethod
    def _validate_prefix(cls, v: str) -> str:
        import re
        if not re.fullmatch(r"[A-Z][A-Z0-9_-]{1,15}", v):
            raise ValueError(
                "tracker.prefix must match [A-Z][A-Z0-9_-]{1,15} "
                f"(got {v!r})"
            )
        return v


class DeploymentContext(BaseModel):
    """Operator-supplied deployment facts.

    The engine itself carries NO organization-specific knowledge. Anything
    the agent needs to know about the operator's deployment is supplied
    here at runtime via the application's config (config.yaml or env
    vars). See Constitution Principle VI.

    Field semantics, defaults, validation rules, and the consistency
    invariant CI gate are specified in
    ``specs/001-port-opsrag-opensource/contracts/deployment-context.md``.
    """

    model_config = ConfigDict(extra="forbid")

    organization_label: str | None = None
    """Display-only label used in some prompt templates. MUST never be
    logged. When ``None``, prompts use generic phrasing."""

    services: list[str] = Field(default_factory=list)
    """Service names the operator wants the agent to recognize in user
    queries. The engine derives semantic-router anchors from this list at
    startup."""

    environments: list[str] = Field(default_factory=list)
    """Free-form env labels (e.g. ``["prod", "staging"]`` or
    ``["us-east-1-prod", "us-east-1-staging"]``). The engine combines these
    with industry-standard tokens (``prod``, ``staging``, ``dev``,
    ``test``) to build the env-discriminator regex."""

    key_repos: list[str] = Field(default_factory=list)
    """Most-touched repos. Drives the agent's repo-map cache priority."""

    kubernetes: KubernetesContext = Field(default_factory=KubernetesContext)
    cloud: CloudContext = Field(default_factory=CloudContext)
    tracker: TrackerContext = Field(default_factory=TrackerContext)
    source_urls: SourceUrlBases = Field(default_factory=SourceUrlBases)

    semantic_router_examples: dict[str, list[str]] = Field(default_factory=dict)
    """Optional operator augmentation for semantic-router anchors. The
    engine produces a default baseline by rendering per-category templates
    against ``services``; these entries are merged on top so operators can
    add query shapes the templates would not produce. Keys are
    ``QueryCategory`` names; values are example query strings."""

    custom_instructions: str = ""
    """Free-text, deployment-wide operator guidance injected into the agent's
    answer-generation AND conversational (chat) system prompts -- ALWAYS
    applied, on top of retrieval. Use it for org-specific edge cases,
    conventions, escalation / on-call policy, preferred tone, or "always
    mention/check X" rules. Empty -> no addendum. (Distinct from runbooks,
    which are retrieved only when relevant, and from per-user memory.)"""

    @field_validator("services", "environments", "key_repos")
    @classmethod
    def _no_blank_entries(cls, v: list[str]) -> list[str]:
        for item in v:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    "context lists must contain non-empty strings only"
                )
        return v
