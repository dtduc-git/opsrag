"""Registry of all known MCP integrations.

The registry is the single source of truth that ties together each
integration's name, its config-block subclass, the environment variables
it needs when enabled, the tool names it exposes, its health-probe URL
template, and its factories (real + fake). Wiring code iterates this
registry to:

- Build the discriminated ``mcp:`` map on ``Settings``
  (see ``opsrag/config_mcp.py``).
- Resolve credentials at startup and fail fast on missing env vars
  (FR-004; ``MCP_MISCONFIGURED:<name>:<env>``).
- Register tools with the MCP-server facade
  (``opsrag/mcp_server/registry_loader.py``; arrives in T087).
- Probe ``/readyz`` per enabled integration (T088).

See ``specs/001-port-opsrag-opensource/data-model.md`` section 1 for
the contract.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from opsrag.config_mcp import (
    KNOWN_MCP_NAMES,
    MCP_CONFIG_TYPES,
    MCPConfigBlock,
)


# ---------------------------------------------------------------------------
# Lazy factories.
# ---------------------------------------------------------------------------
#
# Factories live in each MCP module (``opsrag/mcp/<name>.py``). They are
# imported lazily so that:
#
# - An MCP whose Python dependencies aren't installed (e.g. ``kubernetes``
#   when running a CI image without it) doesn't crash registry import.
# - The registry can be loaded at startup before per-integration deps
#   are resolved.
#
# Real factories use the conventional entry point name ``build``; fakes
# use ``build_fake`` per FR-012. Integrations that haven't been ported
# yet (Phase 4 work; T073-T086) raise ``NotImplementedError`` when
# called.
def _lazy(module_path: str, attr: str) -> Callable[..., Any]:
    """Return a callable that imports ``module_path`` on first call and
    invokes ``attr`` on it. Defers import errors and per-integration
    optional-dependency errors to enable-time, not registry-load-time."""

    def _build(*args: Any, **kwargs: Any) -> Any:
        import importlib
        mod = importlib.import_module(module_path)
        fn = getattr(mod, attr, None)
        if fn is None:
            raise NotImplementedError(
                f"{module_path}.{attr} is not yet implemented. "
                "This integration's factory is scheduled to be ported "
                "in Phase 4 (see specs/.../tasks.md T073-T086)."
            )
        return fn(*args, **kwargs)

    _build.__qualname__ = f"_lazy::{module_path}.{attr}"
    return _build


# ---------------------------------------------------------------------------
# Registry entry shape.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MCPIntegration:
    """Single registry entry per data-model.md section 1.

    Fields with ``= field(default_factory=tuple)`` use empty tuples
    instead of mutable defaults so the dataclass stays hashable / frozen
    -safe."""

    name: str
    display_name: str
    config_type: type[MCPConfigBlock]
    required_env: tuple[str, ...] = field(default_factory=tuple)
    required_config: tuple[str, ...] = field(default_factory=tuple)
    tool_names: tuple[str, ...] = field(default_factory=tuple)
    health_url_template: str | None = None
    factory: Callable[..., Any] | None = None
    fake_factory: Callable[..., Any] | None = None
    # Optional custom validator: (settings, env) -> missing-item str | None.
    # When set it REPLACES the required_env / required_config checks, so an
    # integration can accept EITHER legacy config OR the new `environments:`
    # registry (an OR the flat required_* tuples can't express).
    validate: Callable[[Any, Any], str | None] | None = None


# ---------------------------------------------------------------------------
# The 20 entries.
# ---------------------------------------------------------------------------
#
# tool_names: harvested from each MCP's tool definitions (alphabetised
# for stable diffs). Adding a tool requires adding it here; the
# ``test_openapi_shape`` and ``test_helm_values_covers_all_mcps``
# contract tests fail otherwise.
#
# required_env: the canonical env-var names this integration consumes
# when enabled. Empty tuple is permitted; an integration with no
# required env (e.g. ``tool_cache``, ``runbooks`` which reads from
# local FS) still needs an entry so the registry stays exhaustive.
#
# health_url_template: when set, ``/readyz`` substitutes
# ``{endpoint}`` / ``{api_key_env}`` / ``$ENV_NAME`` style tokens at
# probe time. Absent => no upstream probe.
# --- Custom validators (accept the new `environments:` registry) ----------


def _env_targets(settings: Any) -> dict:
    environments = getattr(settings, "environments", None)
    return getattr(environments, "targets", {}) or {}


def _has_legacy_cluster_source(settings: Any, env: Any) -> bool:
    """A cluster is reachable via a legacy (pre-`environments:`) mechanism."""
    if env.get("KUBECONFIG") or env.get("KUBERNETES_SERVICE_HOST"):
        return True  # explicit kubeconfig, or an in-cluster ServiceAccount
    if getattr(getattr(settings, "k8s", None), "clusters", None):
        return True
    depk = getattr(getattr(settings, "deployment", None), "kubernetes", None)
    if getattr(depk, "clusters", None):
        return True
    return False


def _validate_prometheus(settings: Any, env: Any) -> str | None:
    """Prometheus is reachable if any environment declares a prometheus
    target (direct URL, or k8s_proxy backed by a kubernetes target), OR a
    legacy cluster source exists (which is synthesized into a prometheus
    target at startup). Only the truly-empty case fails fast."""
    targets = _env_targets(settings)
    if any(getattr(t, "prometheus", None) for t in targets.values()):
        return None
    if any(getattr(t, "kubernetes", None) for t in targets.values()):
        return None  # a k8s_proxy prometheus can ride any kubernetes target
    if _has_legacy_cluster_source(settings, env):
        return None
    return ("environments[*].prometheus / environments[*].kubernetes / "
            "k8s.clusters / deployment.kubernetes.clusters / KUBECONFIG")


REGISTRY: dict[str, MCPIntegration] = {
    "aws": MCPIntegration(
        name="aws",
        display_name="AWS (read-only)",
        config_type=MCP_CONFIG_TYPES["aws"],
        required_env=(),  # boto3 credential chain (AWS_PROFILE / AWS_REGION / IRSA)
        tool_names=(
            "aws_cloudwatch_describe_alarms",
            "aws_cloudwatch_get_metric_data",
            "aws_cost_and_usage",
            "aws_describe_ec2_instances",
            "aws_describe_eks_cluster",
            "aws_list_ecs_services",
            "aws_list_eks_clusters",
            "aws_logs_filter_events",
            "aws_logs_insights_query",
            "aws_read",
            "aws_s3_list_buckets",
        ),
        factory=_lazy("opsrag.mcp.aws", "build"),
        fake_factory=_lazy("opsrag.mcp.aws", "build_fake"),
    ),
    "azure": MCPIntegration(
        name="azure",
        display_name="Azure (read-only)",
        config_type=MCP_CONFIG_TYPES["azure"],
        required_env=(),  # DefaultAzureCredential (AZURE_* / managed identity)
        tool_names=(
            "azure_aks_list_clusters",
            "azure_list_resource_groups",
            "azure_monitor_logs_query",
            "azure_monitor_metrics_query",
            "azure_resource_graph_query",
        ),
        factory=_lazy("opsrag.mcp.azure", "build"),
        fake_factory=_lazy("opsrag.mcp.azure", "build_fake"),
    ),
    "cloudflare": MCPIntegration(
        name="cloudflare",
        display_name="Cloudflare (zones, DNS, Zero-Trust)",
        config_type=MCP_CONFIG_TYPES["cloudflare"],
        required_env=("CLOUDFLARE_API_TOKEN",),
        tool_names=(
            "cloudflare_get_access_app_policies",
            "cloudflare_list_access_apps",
            "cloudflare_list_dns_records",
            "cloudflare_list_firewall_rules",
            "cloudflare_list_page_rules",
            "cloudflare_list_zones",
        ),
        health_url_template="https://api.cloudflare.com/client/v4/user/tokens/verify",
        factory=_lazy("opsrag.mcp.cloudflare", "build"),
        fake_factory=_lazy("opsrag.mcp.cloudflare", "build_fake"),
    ),
    "code": MCPIntegration(
        name="code",
        display_name="Local code search",
        config_type=MCP_CONFIG_TYPES["code"],
        required_env=(),
        tool_names=(
            "code_dependency_lookup",
            "code_find_symbol",
            "code_glob",
            "code_grep",
            "code_list_repos",
            "code_read_file",
        ),
        factory=_lazy("opsrag.mcp.code", "build"),
        fake_factory=_lazy("opsrag.mcp.code", "build_fake"),
    ),
    "datadog": MCPIntegration(
        name="datadog",
        display_name="Datadog APM",
        config_type=MCP_CONFIG_TYPES["datadog"],
        required_env=("DD_API_KEY", "DD_APP_KEY"),
        tool_names=(
            "datadog_get_monitor",
            "datadog_get_slo",
            "datadog_get_trace",
            "datadog_list_apm_services",
            "datadog_list_events",
            "datadog_list_monitors",
            "datadog_list_services",
            "datadog_parse_trace_url",
            "datadog_search_spans",
        ),
        health_url_template="https://api.{dd_site}/api/v1/validate",
        factory=_lazy("opsrag.mcp.datadog", "build"),
        fake_factory=_lazy("opsrag.mcp.datadog", "build_fake"),
    ),
    "elasticsearch": MCPIntegration(
        name="elasticsearch",
        display_name="Elasticsearch / OpenSearch",
        config_type=MCP_CONFIG_TYPES["elasticsearch"],
        required_env=(),  # ES_URL + ES_API_KEY/basic via env, or elasticsearch.url in config
        tool_names=(
            "elasticsearch_cluster_health",
            "elasticsearch_esql_query",
            "elasticsearch_get_mappings",
            "elasticsearch_list_indices",
            "elasticsearch_search",
        ),
        factory=_lazy("opsrag.mcp.elasticsearch", "build"),
        fake_factory=_lazy("opsrag.mcp.elasticsearch", "build_fake"),
    ),
    "gcp": MCPIntegration(
        name="gcp",
        display_name="Google Cloud (read-only)",
        config_type=MCP_CONFIG_TYPES["gcp"],
        required_env=(),  # ADC / Workload Identity (GOOGLE_CLOUD_PROJECT optional)
        tool_names=(
            "gcp_asset_search",
            "gcp_gke_list_clusters",
            "gcp_logging_list_entries",
            "gcp_monitoring_list_alert_policies",
            "gcp_monitoring_list_timeseries",
            "gcp_run_list_services",
        ),
        factory=_lazy("opsrag.mcp.gcp", "build"),
        fake_factory=_lazy("opsrag.mcp.gcp", "build_fake"),
    ),
    "github": MCPIntegration(
        name="github",
        display_name="GitHub",
        config_type=MCP_CONFIG_TYPES["github"],
        required_env=("GITHUB_TOKEN",),
        tool_names=(
            "github_get_commit",
            "github_get_file_contents",
            "github_get_job_logs",
            "github_get_pull_request",
            "github_get_repository_tree",
            "github_get_workflow_run",
            "github_list_commits",
            "github_list_issues",
            "github_list_pull_requests",
            "github_list_releases",
            "github_list_workflow_runs",
            "github_search_code",
            "github_search_issues",
        ),
        factory=_lazy("opsrag.mcp.github", "build"),
        fake_factory=_lazy("opsrag.mcp.github", "build_fake"),
    ),
    "gitlab": MCPIntegration(
        name="gitlab",
        display_name="GitLab",
        config_type=MCP_CONFIG_TYPES["gitlab"],
        required_env=("GITLAB_TOKEN",),
        tool_names=(
            "gitlab_get_commit",
            "gitlab_get_merge_request",
            "gitlab_get_pipeline",
            "gitlab_get_pipeline_job",
            "gitlab_get_project",
            "gitlab_grep_job_trace",
            "gitlab_list_branches",
            "gitlab_list_commits",
            "gitlab_list_deployments",
            "gitlab_list_merge_requests",
            "gitlab_list_pipeline_jobs",
            "gitlab_list_pipelines",
            "gitlab_list_tags",
            "gitlab_search_projects",
        ),
        factory=_lazy("opsrag.mcp.gitlab", "build"),
        fake_factory=_lazy("opsrag.mcp.gitlab", "build_fake"),
    ),
    "grafana": MCPIntegration(
        name="grafana",
        display_name="Grafana (dashboards/metrics/logs)",
        config_type=MCP_CONFIG_TYPES["grafana"],
        required_env=("GRAFANA_URL", "GRAFANA_TOKEN"),
        tool_names=(
            "grafana_get_dashboard",
            "grafana_list_alert_rules",
            "grafana_list_contact_points",
            "grafana_list_datasources",
            "grafana_loki_label_values",
            "grafana_prometheus_label_values",
            "grafana_query_loki",
            "grafana_query_prometheus",
            "grafana_search_dashboards",
        ),
        factory=_lazy("opsrag.mcp.grafana", "build"),
        fake_factory=_lazy("opsrag.mcp.grafana", "build_fake"),
    ),
    "knowledge": MCPIntegration(
        name="knowledge",
        display_name="Knowledge-base retrieval",
        config_type=MCP_CONFIG_TYPES["knowledge"],
        required_env=(),
        tool_names=(
            "knowledge_search",
        ),
        factory=_lazy("opsrag.mcp.knowledge", "build"),
        fake_factory=_lazy("opsrag.mcp.knowledge", "build_fake"),
    ),
    "kubernetes": MCPIntegration(
        name="kubernetes",
        display_name="Kubernetes (multi-cluster)",
        config_type=MCP_CONFIG_TYPES["kubernetes"],
        required_env=(),  # in-cluster SA, or KUBECONFIG, or opt-in k8s.clusters (GKE WI)
        tool_names=(
            "k8s_find_workloads",
            "k8s_get_cronjob",
            "k8s_get_daemonset",
            "k8s_get_deployment",
            "k8s_get_job",
            "k8s_get_pod",
            "k8s_get_pod_logs",
            "k8s_get_role_bindings",
            "k8s_get_service",
            "k8s_get_statefulset",
            "k8s_list_cronjobs",
            "k8s_list_daemonsets",
            "k8s_list_deployments",
            "k8s_list_events",
            "k8s_list_jobs",
            "k8s_list_pods",
            "k8s_list_services",
            "k8s_list_statefulsets",
            "k8s_top_pod",
        ),
        factory=_lazy("opsrag.mcp.kubernetes", "build"),
        fake_factory=_lazy("opsrag.mcp.kubernetes", "build_fake"),
    ),
    "loki": MCPIntegration(
        name="loki",
        display_name="Grafana Loki (logs)",
        config_type=MCP_CONFIG_TYPES["loki"],
        required_env=("LOKI_URL",),
        tool_names=(
            "loki_label_values",
            "loki_labels",
            "loki_query",
            "loki_query_range",
            "loki_series",
        ),
        factory=_lazy("opsrag.mcp.loki", "build"),
        fake_factory=_lazy("opsrag.mcp.loki", "build_fake"),
    ),
    "prometheus": MCPIntegration(
        name="prometheus",
        display_name="Prometheus (multi-cluster)",
        config_type=MCP_CONFIG_TYPES["prometheus"],
        # `validate` supersedes these: prometheus is valid via the
        # `environments:` registry OR any legacy cluster source (KUBECONFIG /
        # k8s.clusters / deployment.kubernetes.clusters / in-cluster).
        required_env=("KUBECONFIG",),
        required_config=("deployment.kubernetes.clusters",),
        validate=_validate_prometheus,
        tool_names=(
            "prometheus_alerts",
            "prometheus_label_values",
            "prometheus_query",
            "prometheus_query_range",
            "prometheus_series",
            "prometheus_targets",
        ),
        factory=_lazy("opsrag.mcp.prometheus", "build"),
        fake_factory=_lazy("opsrag.mcp.prometheus", "build_fake"),
    ),
    "rootly": MCPIntegration(
        name="rootly",
        display_name="Rootly incidents",
        config_type=MCP_CONFIG_TYPES["rootly"],
        required_env=("ROOTLY_API_TOKEN",),
        tool_names=(
            "rootly_get_alert",
            "rootly_get_incident",
            "rootly_get_incident_timeline",
            "rootly_get_post_mortem",
            "rootly_list_alerts",
            "rootly_list_incidents",
            "rootly_list_post_mortems",
            "rootly_list_services",
            "rootly_search",
        ),
        factory=_lazy("opsrag.mcp.rootly", "build"),
        fake_factory=_lazy("opsrag.mcp.rootly", "build_fake"),
    ),
    "runbooks": MCPIntegration(
        name="runbooks",
        display_name="Local runbooks",
        config_type=MCP_CONFIG_TYPES["runbooks"],
        required_env=(),
        tool_names=(
            "runbook_list",
            "runbook_load",
        ),
        factory=_lazy("opsrag.mcp.runbooks", "build"),
        fake_factory=_lazy("opsrag.mcp.runbooks", "build_fake"),
    ),
    "sentry": MCPIntegration(
        name="sentry",
        display_name="Sentry (errors)",
        config_type=MCP_CONFIG_TYPES["sentry"],
        required_env=("SENTRY_TOKEN",),
        tool_names=(
            "sentry_get_event",
            "sentry_get_issue",
            "sentry_get_latest_event",
            "sentry_get_trace",
            "sentry_list_projects",
            "sentry_list_releases",
            "sentry_search_events",
            "sentry_search_issues",
        ),
        factory=_lazy("opsrag.mcp.sentry", "build"),
        fake_factory=_lazy("opsrag.mcp.sentry", "build_fake"),
    ),
    "slack": MCPIntegration(
        name="slack",
        display_name="Slack (message / thread fetch)",
        config_type=MCP_CONFIG_TYPES["slack"],
        required_env=("SLACK_BOT_TOKEN",),
        tool_names=(
            "slack_get_message_by_url",
            "slack_get_thread_by_url",
            "slack_list_channels",
        ),
        factory=_lazy("opsrag.mcp.slack", "build"),
        fake_factory=_lazy("opsrag.mcp.slack", "build_fake"),
    ),
    "splunk": MCPIntegration(
        name="splunk",
        display_name="Splunk (logs/search)",
        config_type=MCP_CONFIG_TYPES["splunk"],
        required_env=("SPLUNK_URL", "SPLUNK_TOKEN"),
        tool_names=(
            "splunk_export_search",
            "splunk_fired_alerts",
            "splunk_list_indexes",
            "splunk_list_saved_searches",
            "splunk_run_saved_search",
            "splunk_run_search",
        ),
        factory=_lazy("opsrag.mcp.splunk", "build"),
        fake_factory=_lazy("opsrag.mcp.splunk", "build_fake"),
    ),
    "tool_cache": MCPIntegration(
        name="tool_cache",
        display_name="Read-through cache (wraps other MCPs)",
        config_type=MCP_CONFIG_TYPES["tool_cache"],
        required_env=(),
        # No tools of its own; wraps idempotent calls on other MCPs.
        tool_names=(),
        factory=_lazy("opsrag.mcp.tool_cache", "build"),
        fake_factory=_lazy("opsrag.mcp.tool_cache", "build_fake"),
    ),
}


# Drift-guard: registry keys must equal KNOWN_MCP_NAMES exactly.
assert set(REGISTRY) == set(KNOWN_MCP_NAMES), (
    "MCP registry keys must equal KNOWN_MCP_NAMES exactly"
)


def get_integration(name: str) -> MCPIntegration:
    """Look up a registered integration by name. Raises ``KeyError`` for
    unknown names."""
    return REGISTRY[name]


def integration_names() -> tuple[str, ...]:
    """Stable, sorted tuple of registered integration names. Useful for
    deterministic iteration in tests."""
    return tuple(sorted(REGISTRY))


def resolve_required_envs(name: str) -> tuple[str, ...]:
    """Return the env-var names that MUST be set for ``name`` to be
    enabled."""
    return REGISTRY[name].required_env


class MCPMisconfigured(RuntimeError):
    """Raised when an integration is enabled but its required env vars
    or required-config keys are unresolved. The error message is the
    canonical ``MCP_MISCONFIGURED:<name>:<missing>`` shape from FR-004."""

    def __init__(self, integration: str, missing: str) -> None:
        super().__init__(f"MCP_MISCONFIGURED:{integration}:{missing}")
        self.integration = integration
        self.missing = missing


def _resolve_dotted(root: Any, path: str) -> Any:
    """Walk a dotted attribute path (e.g. ``deployment.kubernetes.clusters``)
    on ``root``. Returns None if any segment is missing."""
    cur = root
    for segment in path.split("."):
        cur = getattr(cur, segment, None)
        if cur is None:
            return None
    return cur


def validate_enabled_mcps(settings: Any, env: dict[str, str] | None = None) -> None:
    """Fail fast (FR-004) for every enabled MCP whose required env vars or
    required-config keys are unresolved.

    Iterates ``settings.mcp``; for each block with ``enabled is True`` it
    checks the registry's ``required_env`` against ``env`` (defaulting to
    ``os.environ``) and ``required_config`` dotted paths against ``settings``.
    The FIRST missing item raises ``MCPMisconfigured`` with the canonical
    ``MCP_MISCONFIGURED:<name>:<missing>`` message. Missing values are: env
    var unset/empty, or a config path that resolves to None / an empty
    collection.
    """
    import os

    resolved_env = os.environ if env is None else env
    mcp_map = getattr(settings, "mcp", {}) or {}
    for name, block in mcp_map.items():
        if not getattr(block, "enabled", False):
            continue
        integration = REGISTRY.get(name)
        if integration is None:
            # Unknown names are rejected earlier by the Settings validator;
            # be defensive here regardless.
            raise MCPMisconfigured(name, "unknown_integration")
        # A custom validator (when present) is authoritative -- it expresses
        # OR semantics the flat required_* tuples can't (e.g. accept either
        # the `environments:` registry OR legacy cluster config).
        if integration.validate is not None:
            missing = integration.validate(settings, resolved_env)
            if missing:
                raise MCPMisconfigured(name, missing)
            continue
        for var in integration.required_env:
            if not resolved_env.get(var):
                raise MCPMisconfigured(name, var)
        for path in integration.required_config:
            value = _resolve_dotted(settings, path)
            # Treat None and empty collections (no clusters/projects) as missing.
            if value is None or (hasattr(value, "__len__") and len(value) == 0):
                raise MCPMisconfigured(name, path)
