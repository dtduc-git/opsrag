"""Filter + wrap the in-house MCP tool registry for safe external exposure.

The internal `opsrag.mcp.ALL_MCP_TOOLS` registry includes some tools we do
NOT want to ship over a public bearer-token-authed endpoint (write paths,
unbounded queries, anything we haven't audited). This module is the
single source of truth for "what does an external MCP client see?".

The allow-list `SAFE_FOR_EXTERNAL_TOOLS` is an explicit set of tool
NAMES. We also accept a few prefixes (read-only families like
`runbook_`, `rootly_`, `knowledge_`) but the discipline is: a tool is
external iff its name is in the explicit set OR matches one of the
read-only prefix rules below.

Each exposed tool is wrapped:

  - For all tools: a top-level `try/except Exception` so handler errors
    don't propagate out of the MCP JSON-RPC envelope (they become
    `{"error": ...}` payloads instead, matching the existing in-house
    error contract). Crucially, `GitLabMCPError` carries an HTTP
    `.status` field -- preserve that.
  - For `prometheus_query_range`: clamp `start..end` to <= 6h and
    enforce <= 500 datapoints by adjusting `step` upward if needed.
    This is the only tool where unbounded input is a real cost concern
    (large matrices both stress Prometheus and bloat the response).
"""
from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from opsrag.mcp import ALL_MCP_TOOLS, MCPTool
from opsrag.mcp.gitlab import GitLabMCPError

_log = logging.getLogger("opsrag.mcp_server.registry")


# Explicit external-facing tool name allow-list.
#
# Rule of thumb: a tool is safe iff it ONLY READS -- no `_create_`,
# `_update_`, `_delete_`, `_post_`, `_patch_`, `_apply_`, `_exec_`,
# `_restart_`, `_scale_` patterns. The existing MCP registry is already
# read-only by construction (gitlab.py docstring: "All HTTP GET"), but
# we re-state the intent here so future additions are gated by an
# explicit list edit.
SAFE_FOR_EXTERNAL_TOOLS: set[str] = {
    # Runbooks (markdown reader, pure file IO)
    "runbook_list",
    "runbook_load",
    # Rootly (read-only REST)
    "rootly_list_incidents",
    "rootly_get_incident",
    "rootly_get_incident_timeline",
    "rootly_list_post_mortems",
    "rootly_get_post_mortem",
    "rootly_get_alert",
    # New (2026-05-15) -- was a gap: only get-by-id existed, no enumeration.
    "rootly_list_alerts",
    "rootly_search",
    "rootly_list_services",
    # PagerDuty (read-only REST)
    "pagerduty_list_incidents",
    "pagerduty_get_incident",
    "pagerduty_list_services",
    "pagerduty_list_oncalls",
    "pagerduty_get_incident_log_entries",
    # GitLab -- only get_/list_ tools (all are HTTP GET upstream)
    "gitlab_list_pipelines",
    "gitlab_get_pipeline",
    "gitlab_list_pipeline_jobs",
    "gitlab_get_pipeline_job",
    "gitlab_list_commits",
    "gitlab_get_commit",
    "gitlab_list_merge_requests",
    "gitlab_get_merge_request",
    "gitlab_get_project",
    # New: lets the agent resolve `acme-notes-be` -> `saas/acme-notes-be` without
    # guessing namespace prefixes. Dev feedback 2026-05-15.
    "gitlab_search_projects",
    # New: direct GET /repository/tags + /repository/branches. The agent
    # was deriving these from pipeline ref-listing, which silently misses
    # any tag/branch without a pipeline run. Dev feedback 2026-05-15.
    "gitlab_list_tags",
    "gitlab_list_branches",
    "gitlab_list_deployments",
    # Greps a job's plaintext trace (CI log) for a regex. Read-only
    # against GitLab's `/projects/:id/jobs/:id/trace` endpoint --
    # paginates server-side and returns matching lines + line numbers.
    # No side effects.
    "gitlab_grep_job_trace",
    # Kubernetes -- read-only describe/list/logs. NB: actual handler
    # names are `k8s_get_pod_logs` and `k8s_top_pod` (singular), not
    # `k8s_get_logs` / `k8s_top_pods` as the spec referred to. The
    # closest equivalent of `k8s_describe_pod` in our codebase is
    # `k8s_get_pod` (returns the spec/status dict).
    "k8s_get_pod",
    "k8s_list_pods",
    "k8s_get_pod_logs",
    "k8s_list_events",
    "k8s_get_deployment",
    "k8s_list_deployments",
    "k8s_get_service",
    "k8s_list_services",
    "k8s_get_statefulset",
    "k8s_list_statefulsets",
    "k8s_get_daemonset",
    "k8s_list_daemonsets",
    "k8s_find_workloads",
    "k8s_get_job",
    "k8s_list_jobs",
    "k8s_get_cronjob",
    "k8s_list_cronjobs",
    "k8s_top_pod",
    # Read-only RBAC inspector -- list/get RoleBindings and
    # ClusterRoleBindings. Useful for permission-debugging questions
    # (which GSA can do X on namespace Y).
    "k8s_get_role_bindings",
    # Code grep over the pod's local clone cache. Read-only -- files
    # are mounted read-only into the backend pod from
    # /tmp/opsrag-repos/. No SCM writes.
    "code_list_repos",
    "code_glob",
    "code_grep",
    "code_read_file",
    "code_find_symbol",
    "code_dependency_lookup",
    # Cloudflare -- LIVE API (Zero-Trust Access + Page/Cache/Firewall Rules).
    "cloudflare_list_zones",
    "cloudflare_list_dns_records",
    "cloudflare_list_firewall_rules",
    "cloudflare_list_page_rules",
    "cloudflare_list_access_apps",
    "cloudflare_get_access_app_policies",
    # Prometheus -- read-only. `query_range` is additionally wrapped to
    # clamp window/step (see _wrap_query_range_clamp).
    "prometheus_query",
    "prometheus_query_range",
    "prometheus_series",
    "prometheus_label_values",
    "prometheus_alerts",
    "prometheus_targets",
    # Vector search over the indexed corpus
    "knowledge_search",
    # Slack -- read-only message/thread access via known URLs / channel lists.
    # `slack_search_messages` is DELIBERATELY EXCLUDED: Slack's
    # `search.messages` API requires `search:read` scope which is
    # USER-token-only (xoxp-). Our cluster ships only a bot token
    # (xoxb-), so the tool always 401s. Exposing it just confuses the
    # agent (dev feedback 2026-05-15: "can't search in slack"). Add it
    # back once a user token is provisioned.
    "slack_get_message_by_url",
    "slack_get_thread_by_url",
    "slack_list_channels",
    # Datadog -- APM traces, monitors, SLOs, metrics. Reads against DD
    # API using DD_API_KEY + DD_APP_KEY (already wired into the cluster
    # opsrag-secrets External Secret). Log search lives in the
    # `elasticsearch_*` tools below (this deployment routes logs to ES,
    # not Datadog -- `datadog_search_logs` was removed 2026-05-21).
    "datadog_search_spans",
    "datadog_get_trace",
    "datadog_list_services",
    # `datadog_list_apm_services` is the right answer for "how many
    # services in env X?" -- `list_services` only returns formally
    # registered Service Definitions (a small fraction).
    "datadog_list_apm_services",
    "datadog_list_monitors",
    "datadog_get_monitor",
    "datadog_list_events",
    "datadog_get_slo",
    # Pure stateless URL parser -- no API call, no auth, no side effect.
    # Extracts trace_id / span_id / epoch_ms / site / env / service from
    # a Datadog Discover URL the user pasted. Safe to expose.
    "datadog_parse_trace_url",
    # Elasticsearch -- cross-cluster application-log search (per-env
    # ECK clusters reached via K8s pod port-forward over the K8s API
    # server). 3 read-only tools, all GET-against-_search-shape -- no
    # writes, no _delete_by_query, no _update. ApiKey auth + 5-layer
    # access control (authorized_networks -> TLS -> WI -> K8s RBAC ->
    # ES role). See opsrag/mcp/elasticsearch.py module docstring for
    # the full security argument.
    "elasticsearch_list_indices",
    "elasticsearch_get_mappings",
    "elasticsearch_search",
    "elasticsearch_esql_query",
    "elasticsearch_cluster_health",
    # CloudWatch -- read-only metrics + alarms + Logs over the boto3
    # credential chain (AWS_PROFILE / AWS_REGION / IRSA). All Get/Describe/
    # List/Filter calls -- no mutating ops, no escape hatch.
    "cloudwatch_get_metric_data",
    "cloudwatch_describe_alarms",
    "cloudwatch_list_metrics",
    "cloudwatch_logs_filter",
    "cloudwatch_logs_describe_groups",
    # Stackdriver -- read-only GCP Cloud Monitoring (metric time series +
    # alert policies) and Cloud Logging (entries) over ADC / Workload
    # Identity. All list verbs -- the entries:list POST is read-only per the
    # Cloud Logging v2 spec. No mutating ops.
    "stackdriver_list_timeseries",
    "stackdriver_list_alert_policies",
    "stackdriver_list_log_entries",
}

# Verb-based deny-list -- defence in depth. Anything whose name matches
# one of these write verbs is dropped from the external registry,
# regardless of whether it appears in SAFE_FOR_EXTERNAL_TOOLS.
#
# We match the verb as a whole _underscore-delimited_ token so that
# legitimately read-only tools like `rootly_get_post_mortem` (the noun
# "post_mortem" contains the substring "post") are NOT swept up.
# Concretely: a name is denied iff one of these tokens appears between
# underscores, anywhere in the name. `_create_`, `create_`, or
# `_create` as a suffix all qualify; `create` as a substring inside
# another word does not.
_WRITE_VERBS = (
    "create", "update", "delete", "remove", "set", "put", "patch",
    "apply", "exec", "restart", "scale", "drain", "cordon", "uncordon",
    "kill", "terminate", "write", "edit", "rename", "move", "copy",
)
_WRITE_VERB_PATTERN = re.compile(
    r"(?:^|_)(" + "|".join(_WRITE_VERBS) + r")(?:_|$)"
)


# Prometheus query_range clamps. Window cap: 6h. Datapoint cap: 500.
# Tuned so a default `step=60s` over 1h (60 points) sails through, but
# a 24h-at-15s pull (~5760 points) gets coerced to a 60s step (~360 pts)
# and a 7d-at-60s pull (~10080 pts) gets coerced to a 6h window.
_PROM_RANGE_MAX_WINDOW_S = 6 * 3600  # 6 hours
_PROM_RANGE_MAX_POINTS = 500


def _parse_step_to_seconds(step: Any) -> int:
    """Lenient parser for Prometheus step strings.

    Accepts ``60``, ``"60"``, ``"60s"``, ``"5m"``, ``"1h"``, ``"1d"``.
    Returns seconds. Unparseable input -> 60s (safe default).
    """
    if isinstance(step, (int, float)):
        return max(1, int(step))
    if not isinstance(step, str):
        return 60
    s = step.strip().lower()
    if not s:
        return 60
    if s.isdigit():
        return max(1, int(s))
    m = re.fullmatch(r"(\d+)([smhd])", s)
    if not m:
        return 60
    n = int(m.group(1))
    unit = m.group(2)
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return max(1, n * mult)


def _resolve_window_seconds(start: Any, end: Any) -> int | None:
    """Best-effort `end - start` as seconds for window clamping.

    Returns ``None`` if either side is a relative string we can't parse
    here (e.g. ``"now-1h"``) -- in that case we don't clamp the window
    and rely only on the step-based datapoint cap. The upstream handler
    resolves relative times to unix seconds anyway.
    """
    def _to_unix(v: Any) -> float | None:
        if isinstance(v, (int, float)):
            return float(v)
        if not isinstance(v, str):
            return None
        s = v.strip()
        if not s:
            return None
        # Plain unix seconds as a string
        try:
            return float(s)
        except ValueError:
            pass
        # Relative shorthand (`now`, `now-1h`) -- let upstream resolve.
        return None

    a = _to_unix(start)
    b = _to_unix(end)
    if a is None or b is None:
        return None
    return max(0, int(b - a))


def _wrap_query_range_clamp(
    inner: Callable[[Any, dict], Awaitable[Any]],
) -> Callable[[Any, dict], Awaitable[Any]]:
    """Wrap `prometheus_query_range`'s handler with safety clamps.

    Mutates ``args`` to keep the upstream call within budget. Returns a
    metadata note in the result dict (``"_clamped"``) so the client
    knows the request was reshaped -- visible but non-fatal.
    """

    async def _wrapped(client: Any, args: dict) -> Any:
        # Always operate on a shallow copy -- never mutate the caller's
        # dict, since the request envelope is logged before / after.
        a = dict(args or {})
        notes: list[str] = []

        # Window clamp (only when both sides are absolute).
        window = _resolve_window_seconds(a.get("start"), a.get("end"))
        if window is not None and window > _PROM_RANGE_MAX_WINDOW_S:
            # Shrink to the most recent _PROM_RANGE_MAX_WINDOW_S worth
            # of data: end stays, start moves up.
            try:
                end_v = float(a["end"]) if not isinstance(a["end"], (int, float)) else float(a["end"])
                a["start"] = end_v - _PROM_RANGE_MAX_WINDOW_S
                notes.append(
                    f"window clamped to {_PROM_RANGE_MAX_WINDOW_S // 3600}h"
                )
                window = _PROM_RANGE_MAX_WINDOW_S
            except (TypeError, ValueError):
                # Couldn't reshape -- fall through to step-only enforcement.
                pass

        # Datapoint clamp. If window is known and step would produce
        # more than _PROM_RANGE_MAX_POINTS, widen step.
        step_s = _parse_step_to_seconds(a.get("step", "60s"))
        if window is not None:
            min_step = max(1, (window + _PROM_RANGE_MAX_POINTS - 1) // _PROM_RANGE_MAX_POINTS)
            if step_s < min_step:
                a["step"] = f"{min_step}s"
                notes.append(
                    f"step widened to {min_step}s to cap at {_PROM_RANGE_MAX_POINTS} datapoints"
                )

        result = await inner(client, a)
        if notes and isinstance(result, dict):
            result["_clamped"] = notes
        return result

    return _wrapped


def _wrap_safe_handler(
    name: str,
    inner: Callable[[Any, dict], Awaitable[Any]],
) -> Callable[[Any, dict], Awaitable[Any]]:
    """Wrap any handler with an outer try/except.

    Errors are returned as `{"error": ...}` instead of propagating to
    the JSON-RPC envelope (where they'd become MCP protocol errors).
    The in-house agent already tolerates `{"error": ...}` payloads on
    tool_result entries, so external clients see the same contract.

    GitLabMCPError keeps its HTTP status in the response so clients
    can distinguish 401/403/404 from a generic upstream failure.
    """

    async def _wrapped(client: Any, args: dict) -> Any:
        try:
            return await inner(client, args)
        except GitLabMCPError as exc:
            return {"error": str(exc), "status": exc.status, "tool": name}
        except Exception as exc:  # noqa: BLE001 -- last-line defence
            _log.warning("mcp_server tool %s raised: %s", name, exc)
            return {"error": f"{type(exc).__name__}: {exc}", "tool": name}

    return _wrapped


def build_external_registry() -> list[MCPTool]:
    """Return the filtered + wrapped external-facing tool list.

    Built lazily so test code can monkey-patch `ALL_MCP_TOOLS` without
    hitting a cached snapshot.
    """
    external: list[MCPTool] = []
    seen: set[str] = set()
    for tool in ALL_MCP_TOOLS:
        if tool.name in seen:
            continue
        if tool.name not in SAFE_FOR_EXTERNAL_TOOLS:
            continue
        # Defence-in-depth: drop anything with a write-verb in the name
        # even if it slipped into the allow-list by accident.
        if _WRITE_VERB_PATTERN.search(tool.name):
            _log.warning("mcp_server dropping write-verb tool: %s", tool.name)
            continue
        seen.add(tool.name)

        # Compose wrappers. Order: clamp first (input shaping), then
        # safe-handler (output shaping).
        inner = tool.handler
        if tool.name == "prometheus_query_range":
            inner = _wrap_query_range_clamp(inner)
        wrapped = _wrap_safe_handler(tool.name, inner)
        external.append(replace(tool, handler=wrapped))
    return external
