"""Per-integration ``MCPConfigBlock`` subclasses + the canonical 14-name
list.

Each MCP integration registers a thin Pydantic subclass of
``MCPConfigBlock`` here, carrying a ``Literal[name]`` discriminator and
any integration-specific configuration fields. ``Settings.mcp`` validates
each ``mcp.<name>`` entry through the subclass keyed by the entry's name
in the registry (``opsrag/mcp/registry.py``).

Why this lives in its own module: keeping the per-integration shapes
together with the base class makes the "every integration has a flag"
guarantee (Constitution Principle II) mechanically obvious -- adding a
new MCP requires adding a subclass here, an entry in
``KNOWN_MCP_NAMES``, and an entry in the registry; the
``test_helm_values_covers_all_mcps`` and
``test_config_unknown_mcp_rejected`` contract tests then fail until all
three are in sync.

See ``specs/001-port-opsrag-opensource/data-model.md`` section 3 and
``specs/001-port-opsrag-opensource/contracts/config-schema.md`` for the
contract this implements.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Canonical name set.
# ---------------------------------------------------------------------------
# Keep this tuple aligned with ``KNOWN_MCP_NAMES`` re-exported from
# ``opsrag.config``, with the registry in ``opsrag.mcp.registry``, and with
# the ``mcp:`` keys in the Helm chart's ``values.yaml``. The contract test
# ``test_helm_values_covers_all_mcps`` enforces drift detection.
KNOWN_MCP_NAMES: tuple[str, ...] = (
    "aws",
    "azure",
    "cloudflare",
    "code",
    "datadog",
    "elasticsearch",
    "gcp",
    "github",
    "gitlab",
    "grafana",
    "knowledge",
    "kubernetes",
    "loki",
    "prometheus",
    "rootly",
    "runbooks",
    "sentry",
    "slack",
    "splunk",
    "tool_cache",
)


# ---------------------------------------------------------------------------
# Base class.
# ---------------------------------------------------------------------------
class MCPConfigBlock(BaseModel):
    """Generic per-integration config block.

    Subclasses pin ``name`` to a ``Literal[<integration_name>]`` so the
    union over all subclasses is discriminated by ``name``. ``extra``
    accepts integration-specific options the base class hasn't surfaced
    yet (e.g. while a new MCP is being ported); over time each
    integration moves those options into typed fields on its own
    subclass.
    """

    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    secret_ref: str | None = None
    endpoint: str | None = None
    api_key_env: str | None = None
    extra: dict[str, object] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-integration subclasses.
# ---------------------------------------------------------------------------
class AWSMCPConfig(MCPConfigBlock):
    """AWS read-only tools (EC2/EKS/ECS inventory, CloudWatch metrics +
    logs + alarms, S3, Cost Explorer, + a read-only generic `aws_read`).
    Auth via the boto3 credential chain (AWS_PROFILE / AWS_REGION / IRSA)."""

    name: Literal["aws"] = "aws"


class AzureMCPConfig(MCPConfigBlock):
    """Azure read-only tools (Monitor logs/metrics, AKS, resource groups,
    + a read-only Resource Graph KQL query). Auth via DefaultAzureCredential."""

    name: Literal["azure"] = "azure"


class CloudflareMCPConfig(MCPConfigBlock):
    """Cloudflare REST API (zones, DNS, Zero-Trust apps, firewall / page
    rules)."""

    name: Literal["cloudflare"] = "cloudflare"


class GCPMCPConfig(MCPConfigBlock):
    """Google Cloud read-only tools (Cloud Logging, Monitoring, GKE, Cloud
    Run, + Asset Inventory search). Auth via ADC / Workload Identity."""

    name: Literal["gcp"] = "gcp"


class GitHubMCPConfig(MCPConfigBlock):
    """GitHub read-only tools (files/tree, code search, commits, PRs,
    issues, Actions runs/logs, releases). Env GITHUB_TOKEN + optional
    GITHUB_API_URL for GitHub Enterprise Server."""

    name: Literal["github"] = "github"


class GrafanaMCPConfig(MCPConfigBlock):
    """Grafana read-only tools (dashboards, datasource-proxied Prometheus +
    Loki queries, alert rules, contact points). Env GRAFANA_URL +
    GRAFANA_TOKEN (read-only service account)."""

    name: Literal["grafana"] = "grafana"


class LokiMCPConfig(MCPConfigBlock):
    """Grafana Loki read-only log queries (LogQL range/instant, labels,
    series). Env LOKI_URL (+ optional LOKI_ORG_ID / basic / bearer)."""

    name: Literal["loki"] = "loki"


class SentryMCPConfig(MCPConfigBlock):
    """Sentry read-only tools (projects, issues, events + stacktraces,
    releases, trace). Env SENTRY_TOKEN + SENTRY_HOST + SENTRY_ORG."""

    name: Literal["sentry"] = "sentry"


class SplunkMCPConfig(MCPConfigBlock):
    """Splunk read-only search (SPL oneshot/export, saved searches, indexes,
    fired alerts) with a mutating-SPL guard. Env SPLUNK_URL + SPLUNK_TOKEN."""

    name: Literal["splunk"] = "splunk"


class CodeMCPConfig(MCPConfigBlock):
    """Local code search across cached repos (grep / glob / symbol-find /
    list-repos / read-file)."""

    name: Literal["code"] = "code"


class DatadogMCPConfig(MCPConfigBlock):
    """Datadog APM (traces, spans, monitors, SLOs, events)."""

    name: Literal["datadog"] = "datadog"


class ElasticsearchMCPConfig(MCPConfigBlock):
    """Application logs in Elasticsearch (per-env API keys; emits Kibana
    deep-links)."""

    name: Literal["elasticsearch"] = "elasticsearch"


class GitLabMCPConfig(MCPConfigBlock):
    """GitLab pipelines, merge requests, commits, branches, deployments."""

    name: Literal["gitlab"] = "gitlab"


class KnowledgeMCPConfig(MCPConfigBlock):
    """Internal knowledge-base retrieval (Confluence / wiki search)."""

    name: Literal["knowledge"] = "knowledge"


class KubernetesMCPConfig(MCPConfigBlock):
    """Multi-cluster K8s state (pods / services / deployments, logs,
    metrics, recent events)."""

    name: Literal["kubernetes"] = "kubernetes"


class PrometheusMCPConfig(MCPConfigBlock):
    """Prometheus queries (instant / range), alerts, targets, series,
    label values."""

    name: Literal["prometheus"] = "prometheus"


class RootlyMCPConfig(MCPConfigBlock):
    """Rootly incidents, alerts, post-mortems."""

    name: Literal["rootly"] = "rootly"


class RunbooksMCPConfig(MCPConfigBlock):
    """Local runbook catalog (list / load)."""

    name: Literal["runbooks"] = "runbooks"


class SlackMCPConfig(MCPConfigBlock):
    """Slack message + thread fetch by permalink."""

    name: Literal["slack"] = "slack"


class ToolCacheMCPConfig(MCPConfigBlock):
    """Read-through cache for idempotent tool calls across other MCPs.
    Does not register tools of its own; wraps the others."""

    name: Literal["tool_cache"] = "tool_cache"


# ---------------------------------------------------------------------------
# Name -> subclass map.
# ---------------------------------------------------------------------------
MCP_CONFIG_TYPES: dict[str, type[MCPConfigBlock]] = {
    "aws": AWSMCPConfig,
    "azure": AzureMCPConfig,
    "cloudflare": CloudflareMCPConfig,
    "code": CodeMCPConfig,
    "datadog": DatadogMCPConfig,
    "elasticsearch": ElasticsearchMCPConfig,
    "gcp": GCPMCPConfig,
    "github": GitHubMCPConfig,
    "gitlab": GitLabMCPConfig,
    "grafana": GrafanaMCPConfig,
    "knowledge": KnowledgeMCPConfig,
    "kubernetes": KubernetesMCPConfig,
    "loki": LokiMCPConfig,
    "prometheus": PrometheusMCPConfig,
    "rootly": RootlyMCPConfig,
    "runbooks": RunbooksMCPConfig,
    "sentry": SentryMCPConfig,
    "slack": SlackMCPConfig,
    "splunk": SplunkMCPConfig,
    "tool_cache": ToolCacheMCPConfig,
}

# Drift-guard: the subclass map and the canonical name set must agree.
assert set(MCP_CONFIG_TYPES) == set(KNOWN_MCP_NAMES), (
    "MCP_CONFIG_TYPES keys must equal KNOWN_MCP_NAMES exactly"
)


def default_mcp_map() -> dict[str, MCPConfigBlock]:
    """Default value for ``Settings.mcp``: every known integration
    present but disabled. Each entry is the integration's own subclass
    (so the discriminator round-trips through serialise / deserialise)."""
    return {name: cls() for name, cls in MCP_CONFIG_TYPES.items()}


def config_type_for(name: str) -> type[MCPConfigBlock]:
    """Look up the per-integration subclass for ``name``.

    Raises ``KeyError`` if ``name`` is not a known integration."""
    return MCP_CONFIG_TYPES[name]
