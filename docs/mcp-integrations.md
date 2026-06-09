# MCP Integrations

OpsRAG ships twenty read-only MCP integrations that give the agent live access to clouds, clusters, observability backends, source control, incident tooling, and the local corpus. This is the authoritative per-integration reference; it mirrors `opsrag/mcp/registry.py` (the `REGISTRY` dict, which is the single source of truth) and the handlers in `opsrag/mcp/*.py`.

## The read-only model

Every integration is **read-only by construction**. There is no config flag that loosens this; it is enforced by what each handler is allowed to call:

- **Deny-by-verb / GET-only.** Each tool issues a read-style operation only — HTTP `GET` (GitHub, GitLab, Cloudflare, Grafana, Loki, Sentry, Rootly, Splunk, Prometheus, Elasticsearch search), or a cloud SDK `Describe*` / `List*` / `Get*` call (AWS via boto3, Azure, GCP), or a Kubernetes `read_*` / `list_*` client method. No handler calls a create/update/delete/apply operation.
- **Allow-listed escape hatches.** The two generic pass-throughs are clamped to read verbs. `aws_read` (in `opsrag/mcp/aws.py`) rejects any operation whose PascalCase name does not start with one of `Describe`, `List`, `Get`, `Lookup`, `Search`, `Scan`, `BatchGet`, `View`, `Query`, `Estimate` — anything else raises `AWSMCPError`. `elasticsearch_esql_query` runs ES|QL via `POST /_query`, a read-only query endpoint, and never exposes `_delete_by_query` / write APIs.
- **Defense in depth.** The intended deployment posture is that the credential itself is read-only (an AWS read-only IAM policy, a `view` / `cluster-reader` Kubernetes RBAC binding, a read-scoped API token). The verb allow-list is the application-level guard on top of that.
- **Output clamps.** Handlers bound their result size so a degenerate query can't blow the LLM context or the SSE payload. The notable clamps:
  - **Prometheus** (`_trim_series` in `opsrag/mcp/prometheus.py`): matrix results are capped at `_MAX_MATRIX_SERIES = 16` series and `_MAX_POINTS_PER_SERIES = 240` points/series (keeps the recent end; ~40 KB JSON ceiling). `prometheus_label_values` returns at most 500 values; `prometheus_series` is bounded by its `limit` (default 200). `start`/`end`/`time` accept Grafana-style relative shorthands (`now`, `now-10m`, `now+1h`).
  - **Loki** (`_clamp` in `opsrag/mcp/loki.py`): `limit` defaults to 100 and is clamped to `_MAX_LIMIT = 1000`; every query carries a time bound; lines are truncated.
  - **Datadog** (`_clamp` in `opsrag/mcp/datadog.py`): list/search `limit` clamped to `_MAX_LIMIT = 100`.
  - **Cloudflare** / others apply per-call `max_rows` / `per_page` caps from config.

## How integrations are turned on

- **Opt-in, disabled by default.** Enable one with `mcp.<name>.enabled: true` in your `config.yaml` (see the `mcp:` block in `config-example.yaml`). Only the tools of enabled integrations are registered with the agent; a disabled integration contributes no tools and resolves no credentials.
- **Fail-fast credential check.** Enabling an integration without its required environment variables (or required config keys) fails at startup with the canonical error `MCP_MISCONFIGURED:<name>:<missing>` (FR-004), where `<missing>` is the first unresolved env var or dotted config path. See `validate_enabled_mcps` in `opsrag/mcp/registry.py`. Secrets are never read from YAML — only from the environment.

```yaml
mcp:
  github:
    enabled: true
  prometheus:
    enabled: true
```

```sh
# Provide the matching secret in the environment (see .env.example).
GITHUB_TOKEN=ghp_REPLACE_ME
```

On startup, each enabled integration's `required_env` / `required_config` (or its custom `validate`) is checked; the first missing item raises e.g. `MCP_MISCONFIGURED:github:GITHUB_TOKEN`. Once requirements resolve, its tools become available and (when the registry declares a health URL) `/readyz` probes the upstream — see `opsrag/api/routes_health.py`.

## The `env` argument (multi-environment tools)

The Kubernetes, Prometheus, and Elasticsearch tools are **multi-environment**. They resolve their target from the top-level `environments:` registry (see `opsrag/environments.py` and `EnvironmentsConfig` in `opsrag/config.py`), where each `EnvironmentTarget` declares per-env `kubernetes`, `prometheus`, and `elasticsearch` settings.

- These tools accept an optional **`env`** argument naming the target environment (Prometheus also accepts the legacy alias `cluster`). When omitted, the resolver falls back to the configured default environment (`default_environment()`), which can be overridden per subsystem (e.g. `OPSRAG_K8S_DEFAULT_CLUSTER`, the Prometheus-specific default).
- If no environment is configured at all, the handler returns a clear "configure the `environments:` block" error rather than guessing.
- Legacy `k8s.clusters` / flat `elasticsearch` config is auto-synthesized into the registry at startup for back-compat, so older configs keep working.

## Integration reference

For each integration below: its config name (`mcp.<name>`), display name, required env vars (`required_env`), required config keys if any, whether it has an upstream health probe, and its tools. "None" means the integration needs no env vars to enable (it reads the local filesystem, uses an ambient credential chain, or wraps other integrations). Tool lists are reproduced verbatim from `REGISTRY`.

### aws

- Config name: `mcp.aws`
- Display name: AWS (read-only)
- Required env: none — uses the boto3 credential chain (`AWS_PROFILE` / `AWS_REGION` / IRSA / instance role)
- Required config: none
- Health probe: no
- Read-only enforcement: per-handler `Describe`/`List`/`Get` calls; the generic `aws_read` escape hatch rejects any operation not starting with an allow-listed read verb.
- Tools:
  - `aws_cloudwatch_describe_alarms`
  - `aws_cloudwatch_get_metric_data`
  - `aws_cost_and_usage`
  - `aws_describe_ec2_instances`
  - `aws_describe_eks_cluster`
  - `aws_list_ecs_services`
  - `aws_list_eks_clusters`
  - `aws_logs_filter_events`
  - `aws_logs_insights_query`
  - `aws_read` (generic read-only escape hatch)
  - `aws_s3_list_buckets`

### azure

- Config name: `mcp.azure`
- Display name: Azure (read-only)
- Required env: none — uses `DefaultAzureCredential` (`AZURE_*` env / managed identity)
- Required config: none
- Health probe: no
- Tools:
  - `azure_aks_list_clusters`
  - `azure_list_resource_groups`
  - `azure_monitor_logs_query`
  - `azure_monitor_metrics_query`
  - `azure_resource_graph_query`

### cloudflare

- Config name: `mcp.cloudflare`
- Display name: Cloudflare (zones, DNS, Zero-Trust)
- Required env: `CLOUDFLARE_API_TOKEN`
- Required config: none
- Health probe: yes (`GET https://api.cloudflare.com/client/v4/user/tokens/verify`)
- Tools:
  - `cloudflare_get_access_app_policies`
  - `cloudflare_list_access_apps`
  - `cloudflare_list_dns_records`
  - `cloudflare_list_firewall_rules`
  - `cloudflare_list_page_rules`
  - `cloudflare_list_zones`

### code

- Config name: `mcp.code`
- Display name: Local code search
- Required env: none
- Required config: none
- Health probe: no
- Tools:
  - `code_dependency_lookup`
  - `code_find_symbol`
  - `code_glob`
  - `code_grep`
  - `code_list_repos`
  - `code_read_file`

### datadog

- Config name: `mcp.datadog`
- Display name: Datadog APM
- Required env: `DD_API_KEY`, `DD_APP_KEY`
- Required config: none
- Health probe: yes (`GET https://api.{dd_site}/api/v1/validate`; `{dd_site}` substituted at probe time)
- Tools:
  - `datadog_get_monitor`
  - `datadog_get_slo`
  - `datadog_get_trace`
  - `datadog_list_apm_services`
  - `datadog_list_events`
  - `datadog_list_monitors`
  - `datadog_list_services`
  - `datadog_parse_trace_url`
  - `datadog_search_spans`

### elasticsearch

- Config name: `mcp.elasticsearch`
- Display name: Elasticsearch / OpenSearch
- Required env: none — `ES_URL` + `ES_API_KEY` (or basic auth) via env, or `elasticsearch.url` in config; resolved per environment from the `environments:` registry
- Required config: none
- Health probe: no
- Multi-environment: yes — tools take an optional `env` arg; per-env field mapping (e.g. the logical `service` field) comes from the environment target.
- Note: `elasticsearch_esql_query` is Elasticsearch-only (not OpenSearch) and read-only (`POST /_query`).
- Tools:
  - `elasticsearch_cluster_health`
  - `elasticsearch_esql_query`
  - `elasticsearch_get_mappings`
  - `elasticsearch_list_indices`
  - `elasticsearch_search`

### gcp

- Config name: `mcp.gcp`
- Display name: Google Cloud (read-only)
- Required env: none — uses Application Default Credentials / Workload Identity (`GOOGLE_CLOUD_PROJECT` optional)
- Required config: none
- Health probe: no
- Tools:
  - `gcp_asset_search`
  - `gcp_gke_list_clusters`
  - `gcp_logging_list_entries`
  - `gcp_monitoring_list_alert_policies`
  - `gcp_monitoring_list_timeseries`
  - `gcp_run_list_services`

### github

- Config name: `mcp.github`
- Display name: GitHub
- Required env: `GITHUB_TOKEN`
- Required config: none
- Health probe: no
- Read-only enforcement: every handler issues an HTTP `GET` against the GitHub REST API (v3 / 2022-11-28).
- Tools:
  - `github_get_commit`
  - `github_get_file_contents`
  - `github_get_job_logs`
  - `github_get_pull_request`
  - `github_get_repository_tree`
  - `github_get_workflow_run`
  - `github_list_commits`
  - `github_list_issues`
  - `github_list_pull_requests`
  - `github_list_releases`
  - `github_list_workflow_runs`
  - `github_search_code`
  - `github_search_issues`

### gitlab

- Config name: `mcp.gitlab`
- Display name: GitLab
- Required env: `GITLAB_TOKEN`
- Required config: none
- Health probe: no
- Tools:
  - `gitlab_get_commit`
  - `gitlab_get_merge_request`
  - `gitlab_get_pipeline`
  - `gitlab_get_pipeline_job`
  - `gitlab_get_project`
  - `gitlab_grep_job_trace`
  - `gitlab_list_branches`
  - `gitlab_list_commits`
  - `gitlab_list_deployments`
  - `gitlab_list_merge_requests`
  - `gitlab_list_pipeline_jobs`
  - `gitlab_list_pipelines`
  - `gitlab_list_tags`
  - `gitlab_search_projects`

### grafana

- Config name: `mcp.grafana`
- Display name: Grafana (dashboards/metrics/logs)
- Required env: `GRAFANA_URL`, `GRAFANA_TOKEN`
- Required config: none
- Health probe: no
- Tools:
  - `grafana_get_dashboard`
  - `grafana_list_alert_rules`
  - `grafana_list_contact_points`
  - `grafana_list_datasources`
  - `grafana_loki_label_values`
  - `grafana_prometheus_label_values`
  - `grafana_query_loki`
  - `grafana_query_prometheus`
  - `grafana_search_dashboards`

### knowledge

- Config name: `mcp.knowledge`
- Display name: Knowledge-base retrieval
- Required env: none
- Required config: none
- Health probe: no
- Note: searches the indexed corpus (the same retrieval stack the chat agent uses).
- Tools:
  - `knowledge_search`

### kubernetes

- Config name: `mcp.kubernetes`
- Display name: Kubernetes (multi-cluster)
- Required env: none — in-cluster ServiceAccount, or `KUBECONFIG`, or opt-in `k8s.clusters` (GKE Workload Identity)
- Required config: none
- Health probe: no
- Multi-environment: yes — tools take an optional `env` arg selecting a Kubernetes target from the `environments:` registry (`gke` or `kubeconfig` mode). Expects `view` / `cluster-reader` RBAC; every tool calls a `read_*` / `list_*` client method.
- Tools:
  - `k8s_find_workloads`
  - `k8s_get_cronjob`
  - `k8s_get_daemonset`
  - `k8s_get_deployment`
  - `k8s_get_job`
  - `k8s_get_pod`
  - `k8s_get_pod_logs`
  - `k8s_get_role_bindings`
  - `k8s_get_service`
  - `k8s_get_statefulset`
  - `k8s_list_cronjobs`
  - `k8s_list_daemonsets`
  - `k8s_list_deployments`
  - `k8s_list_events`
  - `k8s_list_jobs`
  - `k8s_list_pods`
  - `k8s_list_services`
  - `k8s_list_statefulsets`
  - `k8s_top_pod`

### loki

- Config name: `mcp.loki`
- Display name: Grafana Loki (logs)
- Required env: `LOKI_URL`
- Required config: none
- Health probe: no
- Clamps: `limit` default 100, capped at 1000; every query is time-bounded; lines truncated.
- Tools:
  - `loki_label_values`
  - `loki_labels`
  - `loki_query`
  - `loki_query_range`
  - `loki_series`

### prometheus

- Config name: `mcp.prometheus`
- Display name: Prometheus (multi-cluster)
- Required env / config: validated by a **custom validator** (`_validate_prometheus`) — valid if any environment declares a `prometheus` target (direct URL) OR a `kubernetes` target (a `k8s_proxy` Prometheus rides any kubernetes target) OR a legacy cluster source exists (`KUBECONFIG`, `k8s.clusters`, `deployment.kubernetes.clusters`, or in-cluster). Only the truly-empty case fails fast with `MCP_MISCONFIGURED:prometheus:...`.
- Health probe: no
- Multi-environment: yes — tools take an optional `env` arg (legacy alias `cluster`). `start`/`end`/`time` accept relative shorthands (`now-1h`). Results are clamped (<=16 matrix series, <=240 points/series; label values <=500).
- Tools:
  - `prometheus_alerts`
  - `prometheus_label_values`
  - `prometheus_query`
  - `prometheus_query_range`
  - `prometheus_series`
  - `prometheus_targets`

### rootly

- Config name: `mcp.rootly`
- Display name: Rootly incidents
- Required env: `ROOTLY_API_TOKEN`
- Required config: none
- Health probe: no
- Tools:
  - `rootly_get_alert`
  - `rootly_get_incident`
  - `rootly_get_incident_timeline`
  - `rootly_get_post_mortem`
  - `rootly_list_alerts`
  - `rootly_list_incidents`
  - `rootly_list_post_mortems`
  - `rootly_list_services`
  - `rootly_search`

### runbooks

- Config name: `mcp.runbooks`
- Display name: Local runbooks
- Required env: none (reads from the local filesystem)
- Required config: none
- Health probe: no
- Tools:
  - `runbook_list`
  - `runbook_load`

### sentry

- Config name: `mcp.sentry`
- Display name: Sentry (errors)
- Required env: `SENTRY_TOKEN`
- Required config: none
- Health probe: no
- Tools:
  - `sentry_get_event`
  - `sentry_get_issue`
  - `sentry_get_latest_event`
  - `sentry_get_trace`
  - `sentry_list_projects`
  - `sentry_list_releases`
  - `sentry_search_events`
  - `sentry_search_issues`

### slack

- Config name: `mcp.slack`
- Display name: Slack (message / thread fetch)
- Required env: `SLACK_BOT_TOKEN`
- Required config: none
- Health probe: no
- Tools:
  - `slack_get_message_by_url`
  - `slack_get_thread_by_url`
  - `slack_list_channels`

### splunk

- Config name: `mcp.splunk`
- Display name: Splunk (logs/search)
- Required env: `SPLUNK_URL`, `SPLUNK_TOKEN`
- Required config: none
- Health probe: no
- Tools:
  - `splunk_export_search`
  - `splunk_fired_alerts`
  - `splunk_list_indexes`
  - `splunk_list_saved_searches`
  - `splunk_run_saved_search`
  - `splunk_run_search`

### tool_cache

- Config name: `mcp.tool_cache`
- Display name: Read-through cache (wraps other MCPs)
- Required env: none
- Required config: none
- Health probe: no
- Tools: none. This integration exposes no tools of its own; it wraps idempotent calls made by other enabled integrations with a read-through cache. Purge / inspect it via the `/api/cache/*` endpoints (see `./api-reference.md`).

## Enabling an integration (worked example)

The steps below enable Slack; apply the same pattern to any integration by substituting its config name and `required_env`.

1. Flip it on in `config.yaml`:

   ```yaml
   mcp:
     slack:
       enabled: true
   ```

2. Provide its required env var as a secret. For local dev, in `.env`:

   ```sh
   SLACK_BOT_TOKEN=xoxb-REPLACE_ME
   ```

   In a deployment, supply it as a secret reference (e.g. a Kubernetes Secret mounted as an environment variable) rather than committing it.

3. Start OpsRAG. The required env vars and config keys are validated at startup; if any are missing, startup fails fast with `MCP_MISCONFIGURED:slack:SLACK_BOT_TOKEN` (or the first missing item). Once all requirements resolve, the integration's tools (`slack_get_message_by_url`, `slack_get_thread_by_url`, `slack_list_channels`) become available to the agent, and the `/api/integrations` endpoint reports it as enabled.

## See also

- [API reference](./api-reference.md) — the HTTP surface, including `/integrations`, `/readyz` per-MCP probes, and the MCP-server proxy (`/api/mcp/*`).
- [Authentication & RBAC](./auth.md) — the `mcp` scope gates the MCP token endpoints that let external clients reach these tools.
- [Investigations](./investigations.md) — the investigation engine drives these MCP tools during a live incident triage.
- `opsrag/mcp/registry.py` — the authoritative `REGISTRY` this document mirrors.
