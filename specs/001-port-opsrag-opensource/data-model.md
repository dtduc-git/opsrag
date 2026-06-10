# Phase 1 Data Model: Port opsrag as a vendor-neutral opensource project

**Branch**: `001-port-opsrag-opensource` | **Date**: 2026-05-28

Scope: the durable "shapes" introduced or refined by this port. Pure
implementation-detail tables (e.g. existing `qa_cache` rows from upstream)
are preserved unchanged and not re-documented here.

---

## 1. `MCPIntegration` (in-memory registry entry, not persisted)

Identifies a single named external integration the agent can invoke.

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | string (`^[a-z][a-z0-9_]*$`) | yes | Canonical identifier; matches the key under `mcp.<name>` in `config.yaml` and the directory key under `values.yaml`. |
| `display_name` | string | yes | Human label for UI / logs. English. |
| `enabled` | bool | yes | Default `false`. Sourced from `mcp.<name>.enabled`. |
| `required_env` | list[string] | yes | Env-var names that MUST be set when `enabled: true`. Empty list permitted. |
| `required_config` | list[string] | no | Config keys under `mcp.<name>` that MUST be present when `enabled: true`. |
| `tool_names` | list[string] | yes | Names of MCP tools this integration registers when enabled. |
| `factory` | callable | yes | Returns an instance implementing the MCP protocol when `enabled: true`. |
| `fake_factory` | callable | yes | Returns a fake instance for integration tests (FR-012). |
| `health_url_template` | string \| null | no | If set, a URL template used by `/readyz` to probe upstream availability. |

**Validation rules**:

- When `enabled` is `true`, every name in `required_env` MUST resolve to a
  non-empty value in the process environment. Otherwise startup fails per
  FR-004 with error `MCP_MISCONFIGURED:<name>:<missing-env-var>`.
- The 20 names in the registry:
  `aws`, `azure`, `cloudflare`, `code`, `datadog`, `elasticsearch`, `gcp`,
  `github`, `gitlab`, `grafana`, `knowledge`, `kubernetes`, `loki`,
  `prometheus`, `rootly`, `runbooks`, `sentry`, `slack`, `splunk`,
  `tool_cache`.
- Adding a new integration requires a new entry in this registry plus a
  corresponding default-`false` entry in `values.yaml` (FR-006); the
  Helm-values contract test (see `contracts/helm-values-schema.md`) fails
  the build if these drift.

---

## 2. `Configuration` (Pydantic v2 settings root)

Top-level shape of `config.yaml`, validated at startup.

```yaml
auth:
  issuer: <required>            # OIDC issuer URL, no trailing slash
  audience: <required>          # OIDC client/audience identifier
  jwks_cache_seconds: 300       # Default
llm:
  provider: <required>          # anthropic | openai | vertex | bedrock
  model: <required>             # Provider-specific
  api_key_env: <required>       # Env var containing the key
  max_tokens: 4096
embedding:
  provider: <required>          # openai | fastembed | vertex | bedrock
  model: <required>
  api_key_env: <required if needed>
vector_store:
  provider: qdrant              # qdrant | pgvector
  url: <required>
  collection: opsrag
knowledge_graph:
  provider: null                # null | neo4j   (FR-019: null is the default)
  # When provider != null, additional keys required (see graph-provider section).
session:
  provider: postgres            # inmemory | postgres
  dsn_env: POSTGRES_DSN
memory:
  provider: postgres            # inmemory | postgres
  dsn_env: POSTGRES_DSN
observability:
  provider: console             # console | phoenix
  project_name: opsrag
agent:
  mode: hybrid
  top_k: 10
  rerank_top_k: 5
  max_retries: 3
slack_bot:
  enabled: false
  app_token_env: SLACK_APP_TOKEN
  bot_token_env: SLACK_BOT_TOKEN
mcp:
  aws:            { enabled: false, ... }
  cloudflare:     { enabled: false, ... }
  gcp:            { enabled: false, ... }
  code:           { enabled: false, ... }
  datadog:        { enabled: false, ... }
  elasticsearch:  { enabled: false, ... }
  gitlab:         { enabled: false, ... }
  knowledge:      { enabled: false, ... }
  kubernetes:     { enabled: false, ... }
  prometheus:     { enabled: false, ... }
  rootly:         { enabled: false, ... }
  runbooks:       { enabled: false, ... }
  slack:          { enabled: false, ... }
  tool_cache:     { enabled: false, ... }
```

**Validation rules**:

- Unknown top-level keys raise a clear error (Pydantic v2
  `model_config = ConfigDict(extra="forbid")`) — addresses the "stale
  flags" edge case.
- `auth.issuer` MUST be a valid URL with `https://` scheme **unless** the
  literal string starts with `http://dex:` (compose-only escape hatch); the
  validator emits a warning when the escape is in use.
- `vector_store.provider in {qdrant, pgvector}`; provider-specific keys
  validated by a discriminated union.
- `knowledge_graph.provider in {null, neo4j}`; when `neo4j`, additional
  keys `url`, `username`, `password_env`, `database` are required.
- Every `mcp.<name>` block matches the `MCPConfigBlock` shape (next
  section). Unknown `mcp.<name>` names are rejected.

---

## 3. `MCPConfigBlock`

The shape of each per-integration block under `mcp:` in `config.yaml`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `enabled` | bool | yes | Default `false`. |
| `secret_ref` | string \| null | no | Optional Kubernetes Secret name when running under Helm. |
| `endpoint` | string \| null | no | Override default endpoint (e.g. self-hosted GitLab). |
| `api_key_env` | string \| null | no | Env-var name for the integration's API key, if applicable. |
| `extra` | map[string, scalar] | no | Free-form per-integration options validated by the integration's own model. |

Each integration registers its own subclass of `MCPConfigBlock` (Pydantic
discriminated union by integration name) with the fields it actually
requires; the union is built by iterating the `MCPIntegration` registry at
startup. This is what makes the "every integration has a flag" guarantee
mechanical: no integration can exist without registering its config block.

---

## 4. `HelmValues`

Top-level shape of the chart's `values.yaml`. The schema is fully expressed
in `deploy/helm/opsrag/values.schema.json`; this section summarises.

```yaml
image:
  repository: ghcr.io/<org>/opsrag
  tag: <appVersion>
  pullPolicy: IfNotPresent
nameOverride: ""
fullnameOverride: ""
serviceAccount:
  create: true
  name: ""
auth:
  issuer: https://idp.example.com
  audience: opsrag
ingress:
  enabled: false
  className: nginx
  hosts: []
service:
  type: ClusterIP
  port: 8080
api:
  replicaCount: 2
  resources: {}
  livenessProbe: { httpGet: { path: /healthz, port: http } }
  readinessProbe: { httpGet: { path: /readyz, port: http } }
ui:
  enabled: true
  replicaCount: 1
  image: { repository: ghcr.io/<org>/opsrag-ui, tag: <appVersion> }
slackBot:
  enabled: false
  replicaCount: 1
mcp:
  aws:           { enabled: false, secretRef: "" }
  cloudflare:    { enabled: false, secretRef: "" }
  gcp:           { enabled: false, secretRef: "" }
  code:          { enabled: false, secretRef: "" }
  datadog:       { enabled: false, secretRef: "" }
  elasticsearch: { enabled: false, secretRef: "" }
  gitlab:        { enabled: false, secretRef: "" }
  knowledge:     { enabled: false, secretRef: "" }
  kubernetes:    { enabled: false, secretRef: "" }
  prometheus:    { enabled: false, secretRef: "" }
  rootly:        { enabled: false, secretRef: "" }
  runbooks:      { enabled: false, secretRef: "" }
  slack:         { enabled: false, secretRef: "" }
  tool_cache:    { enabled: false, secretRef: "" }
networkPolicy:
  enabled: true
  egressAllowlist: []
podDisruptionBudget:
  enabled: true
  minAvailable: 1
autoscaling:
  enabled: false
```

**Validation rules** (enforced by `values.schema.json`):

- `mcp` keys are constrained to the 20 known names; unknown keys fail
  `helm lint`.
- Every `mcp.<name>.enabled` MUST be a boolean (no string coercion).
- `auth.issuer` MUST be a URL.
- `image.tag` MUST be present (no implicit `latest`).

---

## 5. `AuditReport`

Output shape of `scripts/audit-vendor-neutrality.sh --json`.

```json
{
  "exit_code": 0,
  "checks": [
    {
      "name": "proprietary-names",
      "passed": true,
      "violations": []
    },
    {
      "name": "non-english-text",
      "passed": true,
      "violations": [
        {
          "file": "opsrag/factory.py",
          "line": 42,
          "snippet": "...",
          "rule": "non-ascii-vietnamese"
        }
      ]
    },
    {
      "name": "hardcoded-hosts",
      "passed": true,
      "violations": []
    }
  ],
  "scanned_files": 1247
}
```

**Validation rules**:

- `exit_code == 0` iff every `checks[*].passed == true`.
- `violations[*].file` is a repository-relative path.
- Output is stable enough to diff between commits (sorted by `file`, then
  `line`).

---

## 6. `SampleCorpus`

Static, on-disk artefact under `samples/`. Documented for completeness; no
runtime mutation.

```
samples/
├── runbooks/
│   ├── 001-acme-notes-checkout-503.md
│   ├── 002-acme-notes-db-failover.md
│   ├── 003-acme-notes-image-build-failure.md
│   ├── 004-acme-notes-tls-expiry.md
│   └── 005-acme-notes-pagerduty-noise.md
├── postmortems/
│   ├── INC-001-checkout-503-2026-03-12.md
│   ├── INC-002-redis-evictions-2026-04-02.md
│   └── INC-003-cron-overlap-2026-04-19.md
├── manifests/
│   ├── checkout-deployment.yaml
│   └── notes-api-service.yaml
└── terraform/
    └── acme-notes-vpc.tf
```

**Constraints**:

- Total size ≤ 200 KB.
- All content is wholly synthetic; the audit script's
  `proprietary-names` rule treats `acme-notes` / `acme.example.com` as
  recognised placeholders.
- Re-indexing the corpus at compose-up MUST complete in under 20 s on a
  developer laptop (component of the SC-005 30-s boot budget).

---

## 7. `ChunkMetadata` (generalized chunk-metadata schema)

Schema of record for the dict carried on `Chunk.metadata`
(`opsrag/interfaces/chunker.py`). The field stays a plain `dict` for
backward-compat; the typed view + controlled vocabularies live in
`opsrag/ingestion/metadata.py` (`ChunkMetadata` TypedDict, `total=False`).
**All fields are optional — absence MUST NOT break retrieval.** New facets
are additive only; embedding/vector dimensions, `parent_chunk_id` linkage,
and the contextual prefix are unchanged.

Populated in three tiers:

| Tier | Fields | Where set |
|---|---|---|
| Parser | `repo`, `path`, `branch`, `sha`, `title`, `source_system`, `url`, `author` (+ `author_hashed`), `created_at`, `updated_at`, `owner_team`, `version`, `service`, `section_heading`, `section_type`, `section_level`, `helm_file_type`, `runbook`, `postmortem` | `opsrag/parsers/*` (+ `ingestion/metadata.apply_provenance`/`apply_author`) |
| Chunker | `chunk_index`, `chunk_count`, `child_index`, `heading_path`, `content_hash`, `token_count` | `opsrag/chunkers/parent_child.py` |
| Enricher | `doc_type`, `environment`, `tier`, `criticality`, `language`, `tags`, `valid`, `superseded_by`, `services` | `opsrag/ingestion/enrich.py` (deterministic, no LLM) |

### Controlled vocabularies

- `doc_type`: `runbook | postmortem | incident | terraform | helm | helm_values | kubernetes | dockerfile | alert_definition | architecture | adr | wiki | chat | yaml_config | code | generic_markdown`
- `source_system`: `git | confluence | slack | rootly | pagerduty | jira | notion | unknown`
- `environment`: `prod | staging | preprod | dev | test | qa` (canonical names; deployment-specific aliases come from `deployment.environments`)
- `tier` / `criticality`: `tier0 | tier1 | tier2 | tier3`
- `language`: `en | yaml | hcl | json | dockerfile | python | javascript | typescript | go | java | shell | markdown`

Enforcement is **soft** — facets are optional/soft-boost filters, never
hard rejects, to avoid empty-result over-filtering. Unknown values are
tolerated but should be rare.

### PII handling

`author` is an editor/committer identifier (often an email). Stored
**hashed by default** (`anon:<sha256[:16]>`, with `author_hashed: true`);
set `OPSRAG_STORE_AUTHOR_PLAINTEXT=1` to keep the raw value.

### Promoted (indexed) Qdrant payload facets

The high-leverage hard-filter facets are promoted to first-class indexed
payload keys (single source of truth: `metadata.INDEXED_FACETS`):
`doc_type`, `environment`, `tier`, `service`, `updated_at`,
`source_system`, `valid`. Everything else stays inside the nested
`metadata` payload dict. Promoting these requires a reindex/migration
(old vectors lack the indexed facets until re-ingested).

---

## 8. Entity relationships

```
Configuration ──── has-many ──→ MCPConfigBlock
        │
        └── references ────→ MCPIntegration (by name)

HelmValues ──── must-cover ──→ every MCPIntegration name
                                   (enforced by contract test)

AuditReport ──── derived-from ──→ working tree
                                   (no FK — pure scan)

SampleCorpus ──── indexed-by ──→ scripts/seed-sample-corpus.sh
                                   (at compose-up)
```

The single load-bearing relationship for this feature is
**HelmValues must cover every MCPIntegration**. The contract test that
enforces it is named in `contracts/helm-values-schema.md` and runs in CI
on every change to either `opsrag/mcp/` or `deploy/helm/opsrag/`.
