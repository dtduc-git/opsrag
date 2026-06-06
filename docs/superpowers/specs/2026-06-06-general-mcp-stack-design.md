# Design: General-purpose, read-only MCP integration stack

**Date:** 2026-06-06
**Status:** Draft for review
**Owner:** Duke
**Related:** `opsrag/mcp/registry.py`, `opsrag/config_mcp.py`, `opsrag/mcp_server/`, `specs/001-port-opsrag-opensource/data-model.md` §1

---

## 1. Goal & motivation

OpsRAG's current 14 MCP integrations were curated for a specific enterprise stack (GKE-coupled Kubernetes, multi-env Elasticsearch tied to a cluster map, Cartography, CloudSQL, Cloudflare). For the open-source release we want a **vendor-neutral, broadly useful integration catalog** that works for any DevOps/SRE team out of the box.

**This design defines that general catalog and the work to get there.** Hard constraint: **every tool is read-only** — no create/update/delete/exec/scale/apply anywhere, enforced structurally (only read tools are registered) and backstopped by least-privilege credentials.

### Non-goals
- **No write/mutating tools.** Ever. Read-only is the product promise.
- **No proxying of external MCP servers.** We keep OpsRAG's in-process pattern (httpx → vendor REST API) and only *reference* existing MCP servers' tool surfaces + endpoints.
- **Deferred (not in this pass):** New Relic, Honeycomb, PagerDuty, Opsgenie, Dynatrace. The registry pattern makes these trivial to add later; they're out of scope now.

---

## 2. Architecture (unchanged — confirms existing pattern)

Each integration is a self-contained unit, exactly as today:

- **Module** `opsrag/mcp/<name>.py` exposing `build()` (real) and `build_fake()` (deterministic offline fake, FR-012). Tools are async functions that issue read-only `httpx` requests to the vendor REST API (or use the vendor SDK, e.g. `kubernetes`, `boto3`).
- **Config block** `<Name>MCPConfig(MCPConfigBlock)` in `opsrag/config_mcp.py` with `enabled: bool = False` + per-integration fields (base URL, env-var names, optional region/site).
- **Registry entry** in `opsrag/mcp/registry.py`: `name`, `display_name`, `config_type`, `required_env`, `tool_names`, optional `health_url_template`, `factory`, `fake_factory`.
- **Invariants** (already enforced by tests): `KNOWN_MCP_NAMES` == `MCP_CONFIG_TYPES` keys == `REGISTRY` keys; Helm values + `config.yaml` list every integration; `/readyz` probes each enabled one and fails fast on missing env (`MCP_MISCONFIGURED:<name>:<env>`).

Why keep this: no external MCP process dependencies, we control scopes/pagination/rate-limits/read-only enforcement directly, and the fake-factory pattern keeps everything testable offline.

### Common tool shape per category (the abstraction we build to)

| Category | Canonical read tools |
|---|---|
| SCM | file/tree read · code search · commits · PR/MR list+detail+diff · issues · CI runs+jobs+logs · releases/tags |
| Metrics/APM | metric query (instant/range) · monitors/alerts · SLOs · traces/spans search · events |
| Logs | log query (SPL/LogQL/Query-DSL/Logs-Insights) · list indexes/labels |
| Errors | issues · event+stacktrace · releases · trace |
| Kubernetes | list/get/describe (pods/deploys/svc/ingress/nodes/events) · logs · top · contexts (secrets/configmaps = names only) |
| Cloud | curated reads (metrics/logs/alarms, compute/cluster inventory, cost) + 1 generic read-query escape hatch |

---

## 3. The catalog

Legend: **NEW** = build from scratch · **KEEP** = already general, verify/keep · **GENERALIZE** = exists but needs de-coupling · **REMOVE** = delete · **KEEP-DISABLED** = leave in registry, off by default.

| # | Integration | Status | Category | Auth / key env |
|---|---|---|---|---|
| 1 | `github` | **NEW** | SCM | `GITHUB_TOKEN` (fine-grained PAT, read perms), `GITHUB_API_URL` (GHES) |
| 2 | `gitlab` | KEEP | SCM | `GITLAB_TOKEN` (`read_api`+`read_repository`), `GITLAB_API_URL` |
| 3 | `datadog` | KEEP | Observability | `DD_API_KEY`, `DD_APP_KEY` (`*_read` scopes), `DD_SITE` |
| 4 | `sentry` | **NEW** | Errors | `SENTRY_TOKEN` (`org:read project:read team:read event:read`), `SENTRY_HOST`, `SENTRY_ORG` |
| 5 | `grafana` | **NEW** | Observability (fronts Prometheus/Loki/Tempo/alerting/incidents/oncall) | `GRAFANA_URL`, `GRAFANA_TOKEN` (read-only service account) |
| 6 | `prometheus` | KEEP | Metrics | `PROMETHEUS_URL`, optional bearer/basic |
| 7 | `loki` | **NEW** | Logs | `LOKI_URL`, optional `LOKI_ORG_ID` (X-Scope-OrgID), optional basic/bearer |
| 8 | `elasticsearch` | **GENERALIZE** | Logs/search (+ OpenSearch) | `ES_URL`, `ES_API_KEY` or `ES_USERNAME`/`ES_PASSWORD` |
| 9 | `splunk` | **NEW** | Logs | `SPLUNK_URL` (mgmt :8089), `SPLUNK_TOKEN` (bearer), `SPLUNK_VERIFY_SSL` |
| 10 | `kubernetes` | **GENERALIZE** | Kubernetes | `KUBECONFIG` (+ contexts) / in-cluster; drop GKE-only coupling |
| 11 | `aws` | **NEW** | Cloud | boto3 chain: `AWS_PROFILE`/`AWS_REGION`/keys/IRSA |
| 12 | `gcp` | **NEW** | Cloud | ADC / `GOOGLE_APPLICATION_CREDENTIALS` / `GOOGLE_CLOUD_PROJECT` |
| 13 | `azure` | **NEW** | Cloud | `DefaultAzureCredential` / `AZURE_*` env / `AZURE_SUBSCRIPTION_ID` |
| 14 | `rootly` | KEEP-DISABLED | Incidents | `ROOTLY_API_TOKEN` |
| 15 | `cloudflare` | KEEP-DISABLED | CDN/edge | `CLOUDFLARE_API_TOKEN` |
| — | `cartography` | **REMOVE** | (internal infra-graph) | — |
| — | `cloudsql` | **REMOVE** | (GCP CloudSQL — subsumed by `gcp`) | — |
| — | `slack`, `code`, `knowledge`, `runbooks`, `tool_cache` | KEEP | OpsRAG internal | unchanged |

**Resulting `KNOWN_MCP_NAMES`** (alphabetical): `aws, azure, cloudflare, code, datadog, elasticsearch, gcp, github, gitlab, grafana, knowledge, kubernetes, loki, prometheus, rootly, runbooks, sentry, slack, splunk, tool_cache`.

---

## 4. Read-only enforcement posture

Defense in depth — all four layers:

1. **Allowlist in code.** Only `get/list/describe/search/query`-shaped tools are ever written or registered. No mutating method is implemented. (For generic escape hatches, see §6.)
2. **Least-privilege credentials, documented.** Ship a per-integration "minimum read scope" recipe: GitHub fine-grained PAT (Contents/PRs/Issues/Actions = Read); Datadog app key with only `*_read` scopes; GCP `*.viewer` roles; AWS `ReadOnlyAccess` (or `AmazonEKSMCPReadOnlyAccess`); Azure `Reader` + `Monitoring/Log Analytics Reader`; Grafana read-only service account.
3. **Kubernetes hardening** (special-cased): bind the agent ServiceAccount to a ClusterRole limited to `get/list/watch`; **strip Secret/ConfigMap values** server-side (return names/keys/type only); never expose `exec`/`attach`/`port-forward`/`scale`/`cordon`/`drain`/`delete`.
4. **Query guardrails.** Every query tool caps time-range, result count, and timeout before calling the API (protects both the backend and the RAG context window). Use POST for long query bodies (PromQL/LogQL/ES-DSL/GraphQL).

---

## 5. Per-integration tool specs (curated read sets)

Endpoints/scopes below are drawn from the research pass (official MCP servers + vendor REST docs). Tool names are OpsRAG-namespaced `<integration>_<verb_object>`.

### 5.1 `github` (NEW)
`get_file_contents` (`GET /repos/{o}/{r}/contents/{path}`) · `get_repository_tree` (`/git/trees/{sha}?recursive=1`) · `search_code` (`/search/code`) · `list_commits` / `get_commit` · `list_branches` / `list_tags` · `list_pull_requests` / `get_pull_request` (+ files/diff/reviews) · `list_issues` / `search_issues` · `list_workflow_runs` / `get_workflow_run` / `get_job_logs` (`/actions/...`) · `list_releases` / `get_latest_release`. Bonus: `list_code_scanning_alerts`, `list_dependabot_alerts`. **Auth:** fine-grained PAT, read-only repo permissions; base URL configurable for GHES (`/api/v3`).

### 5.2 `gitlab` (KEEP — verify generality)
Already present. Confirm tools cover: `get_file_contents` (`/projects/{id}/repository/files/{path}`), `get_repository_tree`, `list_commits`/`get_commit`, `list_merge_requests`/`get_merge_request`(+diffs), `list_issues`/`get_issue`, `list_pipelines`/`get_pipeline`/`list_pipeline_jobs`/`get_job_log`, `search_code` (`/search?scope=blobs`). **Auth:** `read_api` + `read_repository`; `GITLAB_API_URL` for self-hosted.

### 5.3 `datadog` (KEEP)
Already present (8 read tools). Optionally extend with `list_incidents`/`get_incident` (`/api/v2/incidents`) and `list_hosts` (`/api/v1/hosts`).

### 5.4 `sentry` (NEW)
`find_projects` · `search_issues` (`/organizations/{org}/issues/`) · `get_issue_details` · `get_latest_event` (full stacktrace) · `search_events` (`/organizations/{org}/events/`) · `get_event_details` · `get_trace_details` (`/events-trace/{trace_id}/`) · `find_releases`. **Auth:** bearer token, `event:read project:read org:read team:read`; `SENTRY_HOST` for region/self-hosted.

### 5.5 `grafana` (NEW) — high leverage, fronts many backends
`search_dashboards` (`/api/search`) · `get_dashboard_panel_queries` (`/api/dashboards/uid/{uid}`) · `list_datasources` · `query_prometheus` (instant/range via `/api/ds/query` or datasource proxy) · `list_prometheus_label_values` · `query_loki_logs` (`/loki/api/v1/query_range` via proxy) · `list_loki_label_values` · `list_alert_rules` (`/api/prometheus/grafana/api/v1/rules`) · `list_contact_points` · `get_current_oncall` / `list_oncall_schedules` (if OnCall present) · `list_incidents`/`get_incident` (if Incident present). **Auth:** Grafana service-account token; datasource queries inherit Grafana RBAC (one token, no per-datasource creds). OnCall/Incident tools degrade gracefully if the plugins aren't installed.

### 5.6 `prometheus` (KEEP)
Standalone PromQL: `instant_query`, `range_query`, `list_metric_names` (`/api/v1/label/__name__/values`), `list_label_values`, `find_series`, `get_targets`. Optional bearer/basic; SigV4 mode noted for AWS Managed Prometheus (future).

### 5.7 `loki` (NEW)
`logql_range_query` (`/loki/api/v1/query_range`) · `logql_instant_query` · `list_labels` · `list_label_values` · `find_series`. Multi-tenant via `X-Scope-OrgID`. Always require an explicit time bound + `limit`.

### 5.8 `elasticsearch` (GENERALIZE → ES + OpenSearch)
**Drop** the multi-env `endpoints{}` map tied to the GKE cluster registry and per-env `OPSRAG_ES_*_API_KEY`. **New shape:** single endpoint (`ES_URL` + `ES_API_KEY` or basic), optional `backend: elasticsearch|opensearch`. Tools: `list_indices` (`/_cat/indices`), `get_mappings`, `search` (`POST /{index}/_search`), `esql_query` (`POST /_query`, ES only — gated off for OpenSearch), `cluster_health`. (OpenSearch equivalents: PPL `/_plugins/_ppl`, SQL `/_plugins/_sql` — optional later.)

### 5.9 `splunk` (NEW)
`run_search` (`POST /services/search/v2/jobs` `exec_mode=oneshot`, `output_mode=json`) · `export_search` · `list_saved_searches` · `run_saved_search` (dispatch + results) · `list_indexes` (`/services/data/indexes`) · `fired_alerts`. **Guardrails:** reject mutating SPL (`| delete`, `| collect`, `| outputlookup`), cap time-range/result-count/timeout. **Auth:** bearer token (HEC tokens are write-only — not used).

### 5.10 `kubernetes` (GENERALIZE → de-GKE-couple)
Tools already read-only. **Work:** add connection strategy `kubeconfig | in-cluster | gke` (currently GKE/ADC only). Default to `KUBECONFIG`/contexts/in-cluster (manusa model); keep GKE/ADC as an optional provider. Every tool takes an optional `context` param (multi-cluster); add `list_contexts`. Confirm/extend read set: `list_namespaces`, `list_pods`/`get_pod`/`describe_pod`, `get_pod_logs` (since/tail/previous/container), `list_events`, `list_deployments`/`get_deployment`, `list_replicasets`, `list_statefulsets`/`list_daemonsets`, `list_services`/`get_endpoints`, `list_ingresses`, `list_nodes`/`get_node`, `list_configmaps` (names+keys), `list_secrets` (names+keys+type, **never values**), `pod_metrics`/`node_metrics` (top, degrade if no metrics-server), `get_resource` (generic GVK read), `list_api_resources`.

### 5.11 `aws` (NEW)
Curated: `describe_ec2_instances`, `list_eks_clusters`/`describe_eks_cluster`, `list_ecs_services`/`describe_ecs_tasks`, `cloudwatch_get_metric_data`, `cloudwatch_describe_alarms`, `logs_filter_events` / `logs_insights_query`+`get_results`, `s3_list_buckets`, `iam_get_role_policies`, `cost_get_cost_and_usage` (note: billed ~$0.01/call — cache). **Escape hatch:** `aws_read` = `call_aws` with `READ_OPERATIONS_ONLY` (validate op against AWS Service Authorization Reference access-level ≠ Write). **Auth:** boto3 chain (`AWS_PROFILE`/`AWS_REGION`/IRSA); recommend `ReadOnlyAccess`.

### 5.12 `gcp` (NEW)
Curated: `logging_list_entries` (`logging.entries.list`), `monitoring_list_timeseries`, `monitoring_list_alert_policies`, `gke_list_clusters`/`get_cluster`, `run_list_services`. **Escape hatch:** `asset_search` (Cloud Asset Inventory `searchAllResources` / `searchAllIamPolicies`) for arbitrary inventory. **Auth:** ADC / Workload Identity; `*.viewer` roles only.

### 5.13 `azure` (NEW)
Curated: `monitor_logs_query` (KQL via `LogsQueryClient`), `monitor_metrics_query`, `aks_list_clusters`/`get`, `list_resource_groups`. **Escape hatch:** `resource_graph_query` (Azure Resource Graph KQL — cross-subscription inventory with just Reader). **Auth:** `DefaultAzureCredential`; `Reader` + `Monitoring/Log Analytics Reader`.

### 5.14 `rootly` / `cloudflare` (KEEP-DISABLED)
Unchanged code; `enabled: false` default; remain in registry as general SaaS options.

---

## 6. Generic read escape hatches (where safe)

Per the approved decision, add one generic read-query tool to the high-cardinality providers, each strictly read-validated:
- **AWS** `aws_read` — `call_aws` style, `READ_OPERATIONS_ONLY`, reject any op classified `Write`.
- **GCP** `asset_search` — Asset Inventory search (inherently read).
- **Azure** `resource_graph_query` — Resource Graph KQL (read-only API).
- **Observability raw query** is already the primary tool (PromQL/LogQL/ES-DSL/SPL/NRQL) — these are read by nature; SPL and ES-DSL get a mutating-clause validator.

No generic escape hatch for SCM or Kubernetes (curated read tools + `get_resource` generic-GVK read are sufficient and safer).

---

## 7. Removals & cleanup

- **Delete** `opsrag/mcp/cartography.py`, `opsrag/mcp/cloudsql.py`, their config classes (`CartographyMCPConfig`, `CloudSQLMCPConfig`), registry entries, Helm/`config.yaml`/`.env.example` lines, and any GCP CloudSQL / Cartography-specific config on `Settings`.
- Remove the **ElasticsearchConfig ↔ K8sConfig cluster-map coupling** and per-env ES key convention.
- Remove **GKE-only assumptions** from `kubernetes.py` (keep as optional provider, not the default).
- Re-run `scripts/audit-vendor-neutrality.sh` (now scans untracked files) — must stay green; add any new public vendor hosts (e.g. `*.datadoghq.com`, `sentry.io`, `*.honeycomb.io`, `api.rootly.com`, cloud endpoints) to the host allowlist as needed.

---

## 8. Wiring changes (cross-cutting)

For each new integration, update in lockstep (contract tests enforce this):
1. `opsrag/config_mcp.py`: new `<Name>MCPConfig`, add to `KNOWN_MCP_NAMES` + `MCP_CONFIG_TYPES`.
2. `opsrag/mcp/<name>.py`: `build()` + `build_fake()` + tool functions.
3. `opsrag/mcp/registry.py`: registry entry (name, display_name, config_type, required_env, tool_names, factories).
4. `deploy/helm/opsrag/values*.yaml` + `deploy/compose/config.yaml`: `mcp.<name>: { enabled: false }`.
5. `.env.example`: documented env vars (read-scope comments).
6. `docs/`: per-integration least-privilege credential recipe.

---

## 9. Testing

- **Per integration:** `build_fake()` returns deterministic canned data; unit tests assert each tool's request shaping (URL/params/headers) against a mocked httpx/SDK and parse a representative response. No live network.
- **Read-only assertions:** a test that asserts no registered tool name matches a denylist of mutating verbs (`create|update|delete|put|patch|exec|scale|apply|drain|cordon|run|install|uninstall`); a K8s test asserting Secret/ConfigMap tools never return `.data`.
- **Escape-hatch validators:** unit tests that mutating AWS ops / SPL clauses / ES write calls are rejected.
- **Contract tests (existing, must stay green):** `KNOWN_MCP_NAMES == MCP_CONFIG_TYPES == REGISTRY`; Helm values + `config.yaml` cover all MCPs; `/readyz` shape; OpenAPI shape.

---

## 10. Phasing (delivery, not architecture — each integration is independent)

1. **Cleanup:** remove cartography + cloudsql; generalize elasticsearch + kubernetes config. (Unblocks a clean catalog.)
2. **SCM:** github. **Errors:** sentry.
3. **Observability:** grafana (highest leverage), loki, splunk.
4. **Cloud:** aws, gcp, azure (curated + escape hatch).
5. **Docs + Helm + audit:** credential recipes, values, vendor-neutrality audit green.

Each milestone is shippable on its own (integrations are isolated registry entries).

---

## 11. Open questions / risks

- **SDK weight:** `aws` (boto3), `gcp` (google-cloud-*), `azure` (azure-*) pull large deps. Mitigation: lazy imports (already the registry pattern) + optional extras in packaging so a user installing only the SCM/observability tools doesn't pay for cloud SDKs.
- **Grafana plugin variance:** OnCall/Incident/Sift tools depend on Grafana Cloud plugins; gate + degrade gracefully.
- **Auth diversity:** cloud keyless identity (IRSA/WI/Managed Identity) vs static keys — support both; prefer keyless on-cluster.
