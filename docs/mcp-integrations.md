# MCP Integrations

OpsRAG ships fourteen MCP integrations. Each one exposes a set of tools to the
agent (for example, querying logs, listing incidents, or reading code). The
authoritative definition of every integration -- its config name, the
environment variables it requires, the config keys it requires, and the tools it
provides -- lives in `opsrag/mcp/registry.py` (the `REGISTRY` dict). This
document mirrors that registry.

## How integrations are turned on

- Integrations are opt-in. Every integration is disabled by default.
- You enable one by setting `mcp.<name>.enabled: true` in your `config.yaml`
  (see the `mcp:` block in `config-example.yaml`).
- Enabling an integration without its required environment variables fails fast
  at startup with the canonical error
  `MCP_MISCONFIGURED:<name>:<env>` (FR-004). The same applies to any required
  config key, which produces `MCP_MISCONFIGURED:<name>:<config-path>`.
- Only the tools of enabled integrations are registered with the agent. A
  disabled integration contributes no tools and resolves no credentials.

Set the matching environment variables only when you flip an integration on.
The per-integration env vars are documented in `.env.example` and summarized
below.

## Integration reference

The table lists, for each integration, its config name (`mcp.<name>`), display
name, required environment variables, required config keys (if any), and the
tools it provides. "None" means the integration needs no environment variables
to enable (for example, it reads from the local filesystem or wraps other
integrations).

### cartography

- Config name: `mcp.cartography`
- Display name: Cartography (graph queries)
- Required env: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- Required config: none
- Tools:
  - `cartography_cypher`
  - `cartography_dns_for_value`
  - `cartography_gcp_assets_in_project`
  - `cartography_pod_blast_radius`
  - `cartography_resource_search`
  - `cartography_who_holds_role`
  - `cartography_workload_identity_chain`

### cloudflare

- Config name: `mcp.cloudflare`
- Display name: Cloudflare (zones, DNS, Zero-Trust)
- Required env: `CLOUDFLARE_API_TOKEN`
- Required config: none
- Tools:
  - `cloudflare_get_access_app_policies`
  - `cloudflare_list_access_apps`
  - `cloudflare_list_dns_records`
  - `cloudflare_list_firewall_rules`
  - `cloudflare_list_page_rules`
  - `cloudflare_list_zones`

### cloudsql

- Config name: `mcp.cloudsql`
- Display name: GCP CloudSQL
- Required env: `GOOGLE_APPLICATION_CREDENTIALS`
- Required config: `deployment.cloud.gcp_projects`
- Tools:
  - `cloudsql_get_instance`
  - `cloudsql_get_metrics`
  - `cloudsql_list_backups`
  - `cloudsql_list_databases`
  - `cloudsql_list_instances`
  - `cloudsql_list_operations`
  - `cloudsql_lock_waits`
  - `cloudsql_oldest_transaction_age`
  - `cloudsql_query_insights`

### code

- Config name: `mcp.code`
- Display name: Local code search
- Required env: none
- Required config: none
- Tools:
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
- Display name: Elasticsearch (application logs)
- Required env: `KUBECONFIG`
- Required config: `deployment.kubernetes.eck_elasticsearch_service`,
  `deployment.kubernetes.eck_elasticsearch_namespace`
- Tools:
  - `elasticsearch_list_services`
  - `elasticsearch_log_count`
  - `elasticsearch_search_logs`

### gitlab

- Config name: `mcp.gitlab`
- Display name: GitLab
- Required env: `GITLAB_TOKEN`
- Required config: none
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

### knowledge

- Config name: `mcp.knowledge`
- Display name: Knowledge-base retrieval
- Required env: none
- Required config: none
- Tools:
  - `knowledge_search`

### kubernetes

- Config name: `mcp.kubernetes`
- Display name: Kubernetes (multi-cluster)
- Required env: `KUBECONFIG`
- Required config: `deployment.kubernetes.clusters`
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

### prometheus

- Config name: `mcp.prometheus`
- Display name: Prometheus (multi-cluster)
- Required env: `KUBECONFIG`
- Required config: `deployment.kubernetes.clusters`
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
- Required env: none
- Required config: none
- Tools:
  - `runbook_list`
  - `runbook_load`

### slack

- Config name: `mcp.slack`
- Display name: Slack (message / thread fetch)
- Required env: `SLACK_BOT_TOKEN`
- Required config: none
- Tools:
  - `slack_get_message_by_url`
  - `slack_get_thread_by_url`
  - `slack_list_channels`

### tool_cache

- Config name: `mcp.tool_cache`
- Display name: Read-through cache (wraps other MCPs)
- Required env: none
- Required config: none
- Tools: none. This integration exposes no tools of its own; it wraps
  idempotent calls made by other enabled integrations.

## Enabling an integration

The steps below enable the Slack integration as an example. Apply the same
pattern to any other integration by substituting its config name and required
env vars from the table above.

1. Flip the integration on in `config.yaml`:

   ```yaml
   mcp:
     slack:
       enabled: true
   ```

2. Provide its required env var as a secret. For local development you can put
   it in your `.env` file:

   ```sh
   SLACK_BOT_TOKEN=xoxb-REPLACE_ME
   ```

   In a deployment, supply it as a secret reference (for example, a Kubernetes
   Secret mounted as an environment variable) rather than committing it.

3. Start OpsRAG. On startup, the integration's required env vars and required
   config keys are validated. If any are missing, startup fails fast with
   `MCP_MISCONFIGURED:slack:SLACK_BOT_TOKEN` (or the first missing item). Once
   all requirements resolve, the integration's tools (`slack_get_message_by_url`,
   `slack_get_thread_by_url`, `slack_list_channels`) become available to the
   agent.
