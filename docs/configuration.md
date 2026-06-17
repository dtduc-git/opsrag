# Configuration

How opsrag is configured: a single Pydantic-v2 `config.yaml`, env-var
overrides for secrets and per-deployment facts, and optional cloud-provider
model bundles. The authoritative schema is `opsrag/config.py` (root model
`Settings`); the exhaustive annotated reference is
[`config-example.yaml`](../config-example.yaml).

## The config model

A single YAML file is loaded into the `Settings` root model at startup
(`Settings.load`, `opsrag/config.py`). The path comes from the `OPSRAG_CONFIG`
env var (default `config.yaml`). The model is strict: unknown top-level keys
are rejected (`extra="forbid"`), so a typo'd block name fails fast at boot
rather than being silently ignored. Provider/strategy fields are validated
against closed `Literal` sets, so an unbuildable value (e.g. an embedding
provider the factory can't construct) also fails at load, not at first use.

```sh
export OPSRAG_CONFIG=/app/config.yaml   # default: config.yaml in the CWD
```

### Env-var precedence: env > YAML > bundle

Three layers contribute to the final config, in increasing priority:

1. **Cloud bundle** (lowest) — when `cloud_provider` is set, sane per-purpose
   model defaults fill any slot you didn't specify.
2. **YAML** — values written in `config.yaml`.
3. **Env vars** (highest) — a narrow set of overrides applied after YAML load.

Concretely, `Settings.load` parses the YAML, applies env overrides
(`_apply_env_overrides`), then resolves the cloud bundle into still-unset
slots — so an env-set value always counts as "explicitly set" and the bundle
never overwrites it. This lets the **same image** switch provider, bump a
deprecated model id, or rebrand per tenant without editing YAML or rebuilding.

The env overrides fall into two groups (see `_ENV_OVERRIDES` and
`_MODEL_ENV_OVERRIDES` in `opsrag/config.py`):

| Env var                        | Overrides                          |
|--------------------------------|------------------------------------|
| `OPSRAG_CLOUD_PROVIDER`        | `cloud_provider` (`aws`/`gcp`)     |
| `OPSRAG_LLM_PROVIDER` / `_MODEL` | `llm.provider` / `llm.model`     |
| `OPSRAG_PRO_MODEL`             | `agent.pro_model`                  |
| `OPSRAG_EMBEDDING_PROVIDER` / `_MODEL` / `_DIMENSION` | `embedding.*` |
| `OPSRAG_RERANKER_PROVIDER` / `_MODEL` | `reranker.*`                |
| `OPSRAG_VISION_ENABLED` / `_PROVIDER` / `_MODEL` | `vision.*`        |
| `OPSRAG_SCM_BASE_URL`          | `scm.base_url`                     |
| `OPSRAG_CONFLUENCE_BASE_URL`, `OPSRAG_SLACK_WORKSPACE_URL`, `OPSRAG_ROOTLY_WEB_URL` | source-link base URLs |
| `OPSRAG_BRAND_*`               | `brand.*` (white-label, surfaced to the UI) |

These are kept deliberately narrow: only fields that legitimately differ per
deployment (URLs, model selection, branding). Secrets are **not** in this
list — they flow through the `_env` convention below.

### The `_env` secret convention

Any config key ending in `_env` names the **environment variable** that
carries a secret; the secret value itself never appears in YAML. Examples
from across the schema:

```yaml
llm:        { api_key_env: ANTHROPIC_API_KEY }
embedding:  { api_key_env: OPENAI_API_KEY }
vector_store: { dsn_env: PGVECTOR_DSN }
scm:        { token_env: GITLAB_TOKEN }
knowledge_graph: { password_env: NEO4J_PASSWORD }
reranker:   { api_key_env: COHERE_API_KEY }
```

This keeps `config.yaml` safe to commit and lets the same config target
different credential stores (env, mounted secret files, IRSA/Workload
Identity for cloud providers). The full mapping of env vars to features is in
[`.env.example`](../.env.example).

### Cloud-provider bundles

Setting `cloud_provider` to `aws` or `gcp` auto-fills per-purpose model slots
(reason / tool_call / embed / rerank / summarize / extract) so you don't have
to spell out every provider block:

```yaml
cloud_provider: aws   # Bedrock: Sonnet (reason/pro), Haiku (tools), Cohere Embed v4 + Rerank 3.5
# cloud_provider: gcp # Vertex: Gemini Flash/Pro + Gemini embeddings
```

Resolution runs in `Settings.load` via `opsrag.model_bundles.resolve_cloud_bundle`,
**after** env overrides — so anything you set explicitly (a provider block, a
`models` slot, or an env var) always wins over the bundle. `cloud_provider:
null` (the default) reproduces today's behavior: classic provider blocks
exactly as written. For finer control, the optional top-level `models` block
overrides individual purposes:

```yaml
models:
  reason:    { provider: bedrock, model: us.anthropic.claude-opus-4-..., effort: high }
  tool_call: { provider: bedrock, model: us.anthropic.claude-haiku-... }
  embed:     { provider: bedrock, model: us.cohere.embed-v4:0 }
```

## Guided tour of the top-level blocks

Every block below is a field on `Settings`. All have safe defaults; you only
write the ones you need. The shipped default
([`config.yaml`](../config.yaml)) is a single-key local stack.

### `llm` — language model

`provider` (`anthropic` | `openai` | `vertex` | `bedrock` | `litellm`),
`model`, `api_key_env`, `max_tokens`. Vertex adds
`project`/`location`; Bedrock adds `aws_region`/`aws_profile`; `litellm` adds
`api_base` for self-hosted OpenAI-compatible endpoints (vLLM/TGI, or an
Ollama server).

```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key_env: ANTHROPIC_API_KEY
  max_tokens: 4096
```

### `vision` — image understanding

Lets a user attach images to a chat turn (web UI **and** every channel bot —
Slack/Telegram/Discord/Teams). The image rides along with the question to a
vision-capable model for that turn only; the bytes are **ephemeral** — never
written to the LangGraph checkpoint, the session store, or any durable store
(history keeps just a `[attached image: <name>]` marker).

`model`/`provider` are an auto-route **fallback**, used only when the active
`llm.model` can't see. Left `null`, a provider-aware default is resolved at
startup: **Bedrock/Anthropic → `claude-sonnet-4-6`**, **Vertex →
`gemini-3-flash-preview`**. An explicit `vision.model` always wins. If no
vision model is available the image is dropped and the answer says so. Limits
(`max_images`, `max_bytes`, `allowed_mime`) are enforced on both the web and
channel paths.

```yaml
vision:
  enabled: true
  model: null            # null -> provider-aware default; or pin an id
  provider: null         # null -> llm.provider
  max_images: 4
  max_bytes: 5242880     # 5 MB per image
  allowed_mime: ["image/png", "image/jpeg", "image/gif", "image/webp"]
```

Override at deploy time without a rebuild via `OPSRAG_VISION_ENABLED`,
`OPSRAG_VISION_PROVIDER`, `OPSRAG_VISION_MODEL`.

### `embedding` — index embedder

Five providers: `fastembed` (on-device, 384-dim, no key — the default for the
quickstart), `openai`, `vertex`, `bedrock`, `litellm`. Set `dimension` when
the model supports a choice; it **must** match the vector-store collection's
dimension. Changing model or dimension requires a re-index into a fresh
collection (vectors of different dims aren't comparable).

### `vector_store` — retrieval backend

`provider` is `qdrant` or `pgvector`:

- **`qdrant`** (recommended) — full hybrid retrieval: dense ANN + a true BM25
  sparse lane (FastEmbed IDF over identifier-augmented text), fused with RRF.
  Supports the optional code lane.
- **`pgvector`** — strong dense ANN, but lexical ranking is Postgres FTS
  (`ts_rank_cd`) plus a best-effort `pg_trgm` trigram lane — **not** true
  BM25, and the code lane does **not** apply. Pick it to avoid running Qdrant
  (reuse RDS/CloudSQL/AlloyDB) and accept weaker symbol-heavy ranking.

`allow_dimension_change` (default `false`) is a fail-closed guard: the factory
refuses to start if the embedder's dimension differs from an existing
collection's. Set `true` only for an intentional re-index after an
embed-model switch.

An **optional code lane** (`code_embedding` + `code_vector_store`, Qdrant
only) dual-writes code chunks to a separate collection embedded with a
code-specialized model, adding a 4th RRF lane for "where is function X"
queries. Set both blocks and re-index to enable.

### `reranker` — post-retrieval reorder

Five providers: `fastembed` (local cross-encoder, the default — reranking is
the highest-ROI retrieval lever and runs with no API key), `cohere`,
`bedrock` (Cohere Rerank on Bedrock, no Cohere key), `vertex`, and `noop`.
The `agent.rerank_diversity` knob layers optional MMR diversity on top.

### `agent` — retrieval + answer tuning

`mode` (`minimal` | `full` | `hybrid` | `tool_calling` | `multi_agent`),
`top_k`, `rerank_top_k`, `max_retries`, `rerank_diversity` (MMR penalty,
`0.0` = disabled), and `pro_model` (escalation model for complex
root-cause/cross-source queries; `null` keeps everything on the base LLM).

### `mcp` — read-only integrations

A map of integration name → `{ enabled, ... }`. There are **20** read-only
integrations: `aws`, `azure`, `cloudflare`, `code`, `datadog`,
`elasticsearch`, `gcp`, `github`, `gitlab`, `grafana`, `knowledge`,
`kubernetes`, `loki`, `prometheus`, `rootly`, `runbooks`, `sentry`, `slack`,
`splunk`, `tool_cache`. All disabled by default. An unknown name is rejected
at load. Enabling one without its required env fails fast with
`MCP_MISCONFIGURED:<name>:<env>`. Per-integration knobs live under each
block's own model — see [`mcp-integrations.md`](./mcp-integrations.md).

```yaml
mcp:
  prometheus: { enabled: true }
  datadog:    { enabled: false }
```

### `environments` — multi-environment registry

A top-level registry so one opsrag instance can target N environments. Each
named target bundles how to reach that environment's `kubernetes` (`gke`
Workload-Identity or `kubeconfig` context), `prometheus` (`k8s_proxy` or
`direct`), and `elasticsearch` (`direct` / `port_forward` / `proxy`, with a
logical→physical field mapping). The k8s/prometheus/elasticsearch MCP tools
accept an `env` argument resolved against this registry
(`opsrag/environments.py`).

```yaml
environments:
  default: prod
  targets:
    prod:
      kubernetes:    { mode: kubeconfig, context: prod-eks }
      prometheus:    { reach: k8s_proxy, namespace: monitoring }
      elasticsearch: { reach: direct, url: https://es.example.com:9200, api_key_env: ES_PRD_API_KEY,
                       fields: { timestamp: "@timestamp", service: "kubernetes.labels.app" } }
    staging:
      kubernetes: { mode: gke, project: my-proj, location: us-central1, name: stg-cluster }
```

When `targets` is empty, the engine synthesizes a registry from the legacy
`k8s` / `elasticsearch` / `deployment` blocks for back-compat. See
[`multi-environment.md`](./multi-environment.md).

### `auth` — authentication

Two modes via `auth.mode`: `login` (the default — first-party cookie sessions +
SSO) and `oidc` (verify Bearer JWTs against `issuer` + `audience` — both
required). Authentication is always enforced; there is no anonymous / "open"
mode. SSO providers are `google`, `github`, `microsoft` (Azure AD / Entra), each
with `client_id` + `client_secret_env`. Session signing keys come from a path or
env only. `role_mappings` maps an IdP groups/roles claim to opsrag roles (e.g. an
`admin` group → the `admin` role, which bundles every scope). Full setup is in
[`auth.md`](./auth.md).

```yaml
auth:
  mode: oidc
  issuer: https://your-idp.example.com
  audience: opsrag
```

### `memory` — conversational memory

`provider` is `memory` (in-process), `postgres`, `none`, or `mem0`. `mem0`
adds per-user cross-session operational memory backed by the main Qdrant
client; it can use a dedicated embedder (`mem0_embed_provider` /
`mem0_embed_model` / `mem0_embed_dimension`) when the main embedder isn't
mem0-compatible. The related `session` block (`memory` | `postgres`) backs
chat sessions; Postgres lets `/readyz` verify the database.

### `observability` — tracing

`provider` is `console` (default, boot-safe) or `phoenix` (stream LangGraph
traces to a Phoenix collector via `endpoint`). `project_name` labels traces.

### `eval` and `scheduler` — background jobs

The `scheduler` block (APScheduler) drives daily re-indexing (`enabled`,
`timezone`, `cron_hour`/`cron_minute`, `jitter_seconds`, `parallel_limit`).
The `investigation_history` block promotes settled past investigations into
the RAG corpus as historical-reference docs. Evaluation artifacts live under
`samples/` and the eval harness; see [`deployment.md`](./deployment.md) for
how roles (`api`, `job-indexer`, `scheduler`, `slackbot`) are selected via
`OPSRAG_ROLE`.

### `scm` — source indexing

What to index from Git: `provider` (`gitlab` | `github` | `gitea` | `local`),
`base_url`, `token_env`, `repos` (paths or `{name, branch}` objects),
`default_branch`, `auto_index`, `clone_mode` (git clone vs API-per-file), and
SSH options (`use_ssh`, `ssh_host`, `ssh_user`) to bypass HTTPS proxies.

```yaml
scm:
  provider: gitlab
  base_url: https://gitlab.example.com
  token_env: GITLAB_TOKEN
  repos:
    - devops/infra
    - { name: platform/api, branch: develop }
```

### `sources` — non-Git knowledge connectors

Opt-in connectors that ingest non-code knowledge into the same RAG corpus,
each disabled by default and activated with its credentials:

- **`confluence`** — wiki pages, with space allow/deny lists and a label
  denylist (skip HR/personal pages).
- **`slack`** — channel archive; each thread becomes one document, gated by
  `channels_allowlist` and `min_replies_per_thread`.
- **`rootly`** — incidents + published post-mortems as a single document per
  incident.

(The `slack_bot` block is separate — it's the Socket-Mode worker that answers
in Slack, not a knowledge source.)

### `knowledge_graph` and `light_graph` — graph lanes

`knowledge_graph.provider` is `none` (default null backend), `neo4j`, or
`networkx`. The graph is populated at ingestion but only joins retrieval when
`use_in_retrieval: true`. `light_graph` is a lighter alternative: a Postgres
entity-adjacency table that adds a fail-safe 1-hop expansion lane after
vector search, with no graph engine. `entity_extraction.method` (`hybrid`
default) controls how entities are pulled from chunks.

### `api` — HTTP surface

Static `api_keys` (dev only; prefer OIDC), `rate_limit_rpm`,
`rate_limit_enabled`, and `rate_limit_backend` (`memory` default, or `redis`
to share limit state across replicas). With `redis`, Redis is **required**:
the API pings `redis_url_env` (`OPSRAG_REDIS_URL`) and fails fast at boot if
it can't connect. `redis` is an optional install extra.

### `brand` — UI white-label

`name`, `subtitle`, `assistant_name`, `favicon_url`, `accent_color` — all
overridable per tenant via `OPSRAG_BRAND_*` env, surfaced to the React UI
through `/ui-config`.

## The exhaustive reference

Every block above, with inline comments on each option, lives in
[`config-example.yaml`](../config-example.yaml). Copy the blocks you need into
your own `config.yaml`; it is a reference, not a ready-to-run file. The
shipped default at [`config.yaml`](../config.yaml) is the minimal
single-key stack the compose quickstart loads.

## See also

- [`getting-started.md`](./getting-started.md) — clone to first query
- [`auth.md`](./auth.md) — OIDC + SSO provider setup
- [`multi-environment.md`](./multi-environment.md) — the `environments` registry
- [`deployment.md`](./deployment.md) — roles, Helm, and the indexing Job
- [`mcp-integrations.md`](./mcp-integrations.md) — the 20 MCP integrations
- [`config-example.yaml`](../config-example.yaml) — annotated, exhaustive
