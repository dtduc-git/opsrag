# MCP Integrations

OpsRAG ships 23 read-only MCP connectors that give the agent live access to clouds, clusters, observability backends, source control, incident tooling, and the local corpus. This is the authoritative per-connector reference; it mirrors `opsrag/mcp/registry.py` (the `REGISTRY` dict, which is the single source of truth) and the handlers in `opsrag/mcp/*.py`.

Every connector declares a **category** (the `category` field on each `MCPIntegration`). The categories group the connectors by what they talk to and are surfaced verbatim on the `GET /integrations` endpoint as `category` (see `opsrag/api/routes.py`). There are six functional categories plus one `Internal` bucket:

| Category | Connectors |
|---|---|
| Cloud | aws, azure, cloudflare, gcp |
| Observability | cloudwatch, datadog, elasticsearch, grafana, loki, prometheus, sentry, splunk, stackdriver |
| Source & Code | code, github, gitlab |
| Incident Management | pagerduty, rootly, slack |
| Kubernetes & Infra | kubernetes |
| Knowledge | knowledge, runbooks |
| Internal | tool_cache |

This reference is organized by those categories. To regenerate the authoritative list and counts:

```sh
.venv/bin/python -c "from opsrag.mcp.registry import REGISTRY; print(sorted(REGISTRY)); print(len(REGISTRY))"
```

## The read-only model

Every connector is **read-only by construction**. There is no config flag that loosens this; it is enforced by what each handler is allowed to call:

- **Deny-by-verb / GET-only.** Each tool issues a read-style operation only -- HTTP `GET` (GitHub, GitLab, Cloudflare, Grafana, Loki, Sentry, Rootly, PagerDuty, Splunk, Stackdriver, Prometheus, Elasticsearch search), or a cloud SDK `Describe*` / `List*` / `Get*` call (AWS and CloudWatch via boto3, Azure, GCP), or a Kubernetes `read_*` / `list_*` client method. No handler calls a create/update/delete/apply operation.
- **Allow-listed escape hatches.** The two generic pass-throughs are clamped to read verbs. `aws_read` (in `opsrag/mcp/aws.py`) rejects any operation whose PascalCase name does not start with one of `Describe`, `List`, `Get`, `Lookup`, `Search`, `Scan`, `BatchGet`, `View`, `Query`, `Estimate` -- anything else raises `AWSMCPError`. `elasticsearch_esql_query` runs ES|QL via `POST /_query`, a read-only query endpoint, and never exposes `_delete_by_query` / write APIs. (`stackdriver_list_log_entries` is also a POST, but only because Cloud Logging's `entries:list` read verb is defined as POST in the v2 spec; it never writes.)
- **Defense in depth.** The intended deployment posture is that the credential itself is read-only (an AWS read-only IAM policy, a `view` / `cluster-reader` Kubernetes RBAC binding, a read-scoped API token). The verb allow-list is the application-level guard on top of that.
- **Output clamps.** Handlers bound their result size so a degenerate query can't blow the LLM context or the SSE payload. The notable clamps:
  - **Prometheus** (`_trim_series` in `opsrag/mcp/prometheus.py`): matrix results are capped at `_MAX_MATRIX_SERIES = 16` series and `_MAX_POINTS_PER_SERIES = 240` points/series (keeps the recent end; ~40 KB JSON ceiling). `prometheus_label_values` returns at most 500 values; `prometheus_series` is bounded by its `limit` (default 200). `start`/`end`/`time` accept Grafana-style relative shorthands (`now`, `now-10m`, `now+1h`).
  - **Loki** (`_clamp` in `opsrag/mcp/loki.py`): `limit` defaults to 100 and is clamped to `_MAX_LIMIT = 1000`; every query carries a time bound; lines are truncated.
  - **Datadog** (`_clamp` in `opsrag/mcp/datadog.py`): list/search `limit` clamped to `_MAX_LIMIT = 100`.
  - **CloudWatch** (`_clamp` in `opsrag/mcp/cloudwatch.py`): per-tool `limit` capped at 100 (alarms 100, metrics 100, log events 100, log groups 50); log messages truncated; secrets in log lines / alarm reasons are redacted.
  - **PagerDuty** (`_clamp_limit` in `opsrag/mcp/pagerduty.py`): list `limit` defaults to 20, capped at 100.
  - **Stackdriver** (`_clamp` in `opsrag/mcp/stackdriver.py`): page size defaults to 25, capped at 100; per-series points trimmed; secrets in log payloads redacted.
  - **Cloudflare** / others apply per-call `max_rows` / `per_page` caps from config.

## How connectors are turned on

- **Opt-in, disabled by default.** Enable one with `mcp.<name>.enabled: true` in your `config.yaml` (see the `mcp:` block in `config-example.yaml`). Only the tools of enabled connectors are registered with the agent; a disabled connector contributes no tools and resolves no credentials.
- **Fail-fast credential check.** Enabling a connector without its required environment variables (or required config keys) fails at startup with the canonical error `MCP_MISCONFIGURED:<name>:<missing>` (FR-004), where `<missing>` is the first unresolved env var or dotted config path. See `validate_enabled_mcps` in `opsrag/mcp/registry.py`. Secrets are never read from YAML -- only from the environment.

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

On startup, each enabled connector's `required_env` / `required_config` (or its custom `validate`) is checked; the first missing item raises e.g. `MCP_MISCONFIGURED:github:GITHUB_TOKEN`. Once requirements resolve, its tools become available and (when the registry declares a health URL) `/readyz` probes the upstream -- see `opsrag/api/routes_health.py`.

## The `env` argument (multi-environment tools)

The Kubernetes, Prometheus, and Elasticsearch tools are **multi-environment**. They resolve their target from the top-level `environments:` registry (see `opsrag/environments.py` and `EnvironmentsConfig` in `opsrag/config.py`), where each `EnvironmentTarget` declares per-env `kubernetes`, `prometheus`, and `elasticsearch` settings. The full model and worked examples live in [Multi-environment](./multi-environment.md).

- These tools accept an optional **`env`** argument naming the target environment (Prometheus also accepts the legacy alias `cluster`). When omitted, the resolver falls back to the configured default environment (`default_environment()`), which can be overridden per subsystem (`OPSRAG_K8S_DEFAULT_CLUSTER`, and the Prometheus-specific default).
- If no environment is configured at all, the handler returns a clear "configure the `environments:` block" error rather than guessing.
- Legacy `k8s.clusters` / flat `elasticsearch` config is auto-synthesized into the registry at startup for back-compat, so older configs keep working.

The other connectors (the clouds, CloudWatch, Stackdriver, the SaaS observability and incident tools, source control) are **not** multi-environment: they reach a single upstream resolved from their own credentials. CloudWatch and Stackdriver additionally accept a per-call `region` / `project` override (see their entries below) but do not use the `environments:` registry.

## Connector reference

For each connector below: its config name (`mcp.<name>`), display name, required env vars (`required_env`), required config keys if any, whether it has an upstream health probe, and its tools. "None" means the connector needs no env vars to enable (it reads the local filesystem, uses an ambient credential chain, or wraps other connectors). Tool lists are reproduced verbatim from `REGISTRY`.

---

## Cloud

Inventory and configuration reads against the major cloud control planes. CloudWatch and Stackdriver, the cloud-native metrics/logs backends, live under [Observability](#observability) (their metrics/alarms/logs tools were split out of `aws` and `gcp` -- see those entries).

### aws

- Config name: `mcp.aws`
- Category: Cloud
- Display name: AWS (read-only)
- Required env: none -- uses the boto3 credential chain (`AWS_PROFILE` / `AWS_REGION` / IRSA / instance role)
- Required config: none
- Health probe: no
- Read-only enforcement: per-handler `Describe`/`List`/`Get` calls; the generic `aws_read` escape hatch rejects any operation not starting with an allow-listed read verb.
- Note: CloudWatch metrics/alarms and CloudWatch Logs are NOT here -- they moved to the dedicated [`cloudwatch`](#cloudwatch) connector (Observability). The tools below are inventory/cost only.
- Tools:
  - `aws_cost_and_usage`
  - `aws_describe_ec2_instances`
  - `aws_describe_eks_cluster`
  - `aws_list_ecs_services`
  - `aws_list_eks_clusters`
  - `aws_read` (generic read-only escape hatch)
  - `aws_s3_list_buckets`

### azure

- Config name: `mcp.azure`
- Category: Cloud
- Display name: Azure (read-only)
- Required env: none -- uses `DefaultAzureCredential` (`AZURE_*` env / managed identity)
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
- Category: Cloud
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

### gcp

- Config name: `mcp.gcp`
- Category: Cloud
- Display name: Google Cloud (read-only)
- Required env: none -- uses Application Default Credentials / Workload Identity (`GOOGLE_CLOUD_PROJECT` optional)
- Required config: none
- Health probe: no
- Note: Cloud Monitoring (timeseries, alert policies) and Cloud Logging are NOT here -- they moved to the dedicated [`stackdriver`](#stackdriver) connector (Observability). The tools below are inventory only (GKE, Cloud Run, Asset Inventory).
- Tools:
  - `gcp_asset_search`
  - `gcp_gke_list_clusters`
  - `gcp_run_list_services`

---

## Observability

Metrics, logs, traces, alerts, and APM. This is the largest category. Prometheus and Elasticsearch are also multi-environment (see [the `env` argument](#the-env-argument-multi-environment-tools)).

### cloudwatch

A dedicated read-only connector for Amazon CloudWatch (metrics + alarms) and CloudWatch Logs. These tools were split out of the broader [`aws`](#aws) connector so observability tooling lives in the Observability category and can be enabled independently of the AWS inventory tools.

- Config name: `mcp.cloudwatch`
- Config flag: `mcp.cloudwatch.enabled: true`
- Category: Observability
- Display name: Amazon CloudWatch (metrics/alarms/logs)
- Required env: none -- uses the boto3 default credential chain (`AWS_PROFILE` / `AWS_REGION` / `AWS_DEFAULT_REGION` / IRSA / instance role), the same chain the `aws` connector uses. boto3 is lazy-imported, so the module loads even without boto3 installed (the offline fake needs no boto3 and no AWS creds).
- Required config: none
- Health probe: no
- Region: each tool accepts an optional **`region`** argument; when omitted it falls back to `AWS_REGION` / `AWS_DEFAULT_REGION` (or the SDK default). It is NOT a multi-environment connector -- it does not use the `environments:` registry.
- Read-only enforcement: every handler issues a `Get` / `Describe` / `List` / `Filter`-style call only (`cloudwatch.get_metric_data`, `cloudwatch.describe_alarms`, `cloudwatch.list_metrics`, `logs.filter_log_events`, `logs.describe_log_groups`). There is no escape hatch and no mutating operation. Secrets that appear in log lines / alarm reasons / errors are redacted, and results are clamped (alarms <=100, metrics <=100, log events <=100, log groups <=50).
- Tools:
  - `cloudwatch_get_metric_data` -- fetch metric data (`get_metric_data`); pass `metric_data_queries`, `start_time`, `end_time`.
  - `cloudwatch_describe_alarms` -- list metric alarms; filter by `state_value` (OK/ALARM/INSUFFICIENT_DATA), `alarm_names`, or `alarm_name_prefix`.
  - `cloudwatch_list_metrics` -- discover available metrics; filter by `namespace`, `metric_name`, `dimensions`, `recently_active`.
  - `cloudwatch_logs_filter` -- `filter_log_events` on a log group; pass `log_group_name` (required), optional `filter_pattern`, `start_time`/`end_time` (unix ms), `log_stream_names`.
  - `cloudwatch_logs_describe_groups` -- list log groups (`describe_log_groups`); filter by `log_group_name_prefix` or `log_group_name_pattern`.

### datadog

- Config name: `mcp.datadog`
- Category: Observability
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
- Category: Observability
- Display name: Elasticsearch / OpenSearch
- Required env: none -- `ES_URL` + `ES_API_KEY` (or basic auth) via env, or `elasticsearch.url` in config; resolved per environment from the `environments:` registry
- Required config: none
- Health probe: no
- Multi-environment: yes -- tools take an optional `env` arg; per-env field mapping (e.g. the logical `service` field) comes from the environment target. See [Multi-environment](./multi-environment.md).
- Note: `elasticsearch_esql_query` is Elasticsearch-only (not OpenSearch) and read-only (`POST /_query`).
- Tools:
  - `elasticsearch_cluster_health`
  - `elasticsearch_esql_query`
  - `elasticsearch_get_mappings`
  - `elasticsearch_list_indices`
  - `elasticsearch_search`

### grafana

- Config name: `mcp.grafana`
- Category: Observability
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

### loki

- Config name: `mcp.loki`
- Category: Observability
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
- Category: Observability
- Display name: Prometheus (multi-cluster)
- Required env / config: validated by a **custom validator** (`_validate_prometheus`) -- valid if any environment declares a `prometheus` target (direct URL) OR a `kubernetes` target (a `k8s_proxy` Prometheus rides any kubernetes target) OR a legacy cluster source exists (`KUBECONFIG`, `k8s.clusters`, `deployment.kubernetes.clusters`, or in-cluster). Only the truly-empty case fails fast with `MCP_MISCONFIGURED:prometheus:...`.
- Health probe: no
- Multi-environment: yes -- tools take an optional `env` arg (legacy alias `cluster`). `start`/`end`/`time` accept relative shorthands (`now-1h`). Results are clamped (<=16 matrix series, <=240 points/series; label values <=500). See [Multi-environment](./multi-environment.md).
- Tools:
  - `prometheus_alerts`
  - `prometheus_label_values`
  - `prometheus_query`
  - `prometheus_query_range`
  - `prometheus_series`
  - `prometheus_targets`

### sentry

- Config name: `mcp.sentry`
- Category: Observability
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

### splunk

- Config name: `mcp.splunk`
- Category: Observability
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

### stackdriver

A dedicated read-only connector for the GCP Operations Suite (Cloud Monitoring v3 + Cloud Logging v2, formerly Stackdriver). These tools were split out of the broader [`gcp`](#gcp) connector so an operator can enable metrics/alerts/logs independently of the GKE / Cloud Run / Asset Inventory tools.

- Config name: `mcp.stackdriver`
- Config flag: `mcp.stackdriver.enabled: true`
- Category: Observability
- Display name: Stackdriver (GCP Monitoring + Logging)
- Required env: none -- uses Application Default Credentials (ADC) via `google.auth.default()` to mint a short-lived bearer token, then issues plain HTTP calls (`GOOGLE_CLOUD_PROJECT` optional; the ADC project is used when unset). The google libraries are lazy-imported, so the module loads without the SDK installed (the offline fake needs no google libs).
- Required config: none
- Health probe: no
- Project: each tool accepts an optional `project` argument; when omitted it falls back to `GOOGLE_CLOUD_PROJECT` or the ADC project. It is NOT a multi-environment connector.
- Read-only enforcement: every tool is an HTTP `GET`, except `stackdriver_list_log_entries`, which is the read-only `entries:list` POST (Cloud Logging's list verb is defined as POST in the v2 spec). No create / update / delete / patch / set anywhere. Secrets in log payloads / resource metadata are redacted; results are clamped (page size <=100, per-series points trimmed).
- Tools:
  - `stackdriver_list_timeseries` -- Cloud Monitoring metric data (`v3/projects/{p}/timeSeries`); pass a `filter` (e.g. `metric.type="compute.googleapis.com/instance/cpu/utilization"`), optional time window / aggregation.
  - `stackdriver_list_alert_policies` -- list Cloud Monitoring alert policies (`v3/projects/{p}/alertPolicies`).
  - `stackdriver_list_log_entries` -- read Cloud Logging entries (`entries:list`); pass a logging `filter`, optional time window / order.

---

## Source & Code

The local indexed repositories plus the hosted source-control providers.

### code

- Config name: `mcp.code`
- Category: Source & Code
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

### github

- Config name: `mcp.github`
- Category: Source & Code
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
- Category: Source & Code
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

---

## Incident Management

On-call, incidents, post-mortems, and the incident war-room chat surface.

### pagerduty

A dedicated read-only connector for PagerDuty incidents and on-call schedules, over the PagerDuty REST API v2 (`https://api.pagerduty.com`).

- Config name: `mcp.pagerduty`
- Config flag: `mcp.pagerduty.enabled: true`
- Category: Incident Management
- Display name: PagerDuty incidents / on-call
- Required env: `PAGERDUTY_API_TOKEN` (the validator-enforced required env). The token is also accepted from the alias `OPSRAG_PAGERDUTY_TOKEN`; either is resolved at call time. Auth is the classic API token header (`Authorization: Token token=<token>`). The token's role should be read-only per the deploying organization's PagerDuty config. (The API base URL can be overridden with `OPSRAG_PAGERDUTY_API_URL`.)
- Required config: none
- Health probe: yes (`GET https://api.pagerduty.com/abilities`)
- Read-only enforcement: every tool issues an HTTP `GET`. No POST/PUT/PATCH/DELETE anywhere -- no acknowledge, no resolve, no create/update/delete. List `limit` defaults to 20, capped at 100.
- Tools:
  - `pagerduty_list_incidents` -- list incidents (`GET /incidents`); filter by `statuses` (triggered/acknowledged/resolved), `urgency` (high|low), `since`/`until` (ISO-8601).
  - `pagerduty_get_incident` -- one incident's full details (`GET /incidents/<id>`).
  - `pagerduty_list_services` -- list technical services (`GET /services`); free-text `query` filter.
  - `pagerduty_list_oncalls` -- who is on call now or in a window (`GET /oncalls`); filter by `escalation_policy_ids` / `schedule_ids` / `since` / `until`.
  - `pagerduty_get_incident_log_entries` -- chronological incident timeline (`GET /incidents/<id>/log_entries`).

### rootly

- Config name: `mcp.rootly`
- Category: Incident Management
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

### slack

- Config name: `mcp.slack`
- Category: Incident Management
- Display name: Slack (message / thread fetch)
- Required env: `SLACK_BOT_TOKEN`
- Required config: none
- Health probe: no
- Tools:
  - `slack_get_message_by_url`
  - `slack_get_thread_by_url`
  - `slack_list_channels`

---

## Kubernetes & Infra

The Kubernetes connector. It is multi-cluster / multi-environment (see [the `env` argument](#the-env-argument-multi-environment-tools) and [Multi-environment](./multi-environment.md)).

### kubernetes

- Config name: `mcp.kubernetes`
- Category: Kubernetes & Infra
- Display name: Kubernetes (multi-cluster)
- Required env: none -- in-cluster ServiceAccount, or `KUBECONFIG`, or opt-in `k8s.clusters` (GKE Workload Identity)
- Required config: none
- Health probe: no
- Multi-environment: yes -- tools take an optional `env` arg selecting a Kubernetes target from the `environments:` registry (`gke` or `kubeconfig` mode). Expects `view` / `cluster-reader` RBAC; every tool calls a `read_*` / `list_*` client method.
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

---

## Knowledge

The local corpus and runbooks the agent grounds answers on.

### knowledge

- Config name: `mcp.knowledge`
- Category: Knowledge
- Display name: Knowledge-base retrieval
- Required env: none
- Required config: none
- Health probe: no
- Note: searches the indexed corpus (the same retrieval stack the chat agent uses).
- Tools:
  - `knowledge_search`

### runbooks

- Config name: `mcp.runbooks`
- Category: Knowledge
- Display name: Local runbooks
- Required env: none (reads from the local filesystem)
- Required config: none
- Health probe: no
- Tools:
  - `runbook_list`
  - `runbook_load`

---

## Internal

Infrastructure that supports the other connectors; not a data source itself.

### tool_cache

- Config name: `mcp.tool_cache`
- Category: Internal
- Display name: Read-through cache (wraps other MCPs)
- Required env: none
- Required config: none
- Health probe: no
- Tools: none. This connector exposes no tools of its own; it wraps idempotent calls made by other enabled connectors with a read-through cache. Purge / inspect it via the `/api/cache/*` endpoints (see `./api-reference.md`).

---

## Enabling a connector (worked example)

The steps below enable Slack; apply the same pattern to any connector by substituting its config name and `required_env`.

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

3. Start OpsRAG. The required env vars and config keys are validated at startup; if any are missing, startup fails fast with `MCP_MISCONFIGURED:slack:SLACK_BOT_TOKEN` (or the first missing item). Once all requirements resolve, the connector's tools (`slack_get_message_by_url`, `slack_get_thread_by_url`, `slack_list_channels`) become available to the agent, and the `/api/integrations` endpoint reports it as enabled (with its `category`).

## See also

- [Multi-environment](./multi-environment.md) -- the `environments:` registry that backs the multi-cluster Kubernetes / Prometheus / Elasticsearch tools and the `env` argument.
- [API reference](./api-reference.md) -- the HTTP surface, including `/integrations` (which reports each connector's `category`), `/readyz` per-MCP probes, and the MCP-server proxy (`/api/mcp/*`).
- [Authentication & RBAC](./auth.md) -- the `mcp` scope gates the MCP token endpoints that let external clients reach these tools.
- [Investigations](./investigations.md) -- the investigation engine drives these MCP tools during a live incident triage.
- `opsrag/mcp/registry.py` -- the authoritative `REGISTRY` this document mirrors.
