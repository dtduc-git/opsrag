# Contract: `config.yaml` schema

**Status**: New for this feature. Pydantic v2 root model
`opsrag.config.Settings` is the single source of truth; this document
mirrors it.

## Top-level keys

| Key | Required | Type | Notes |
|---|---|---|---|
| `auth` | yes | object | OIDC settings — see below |
| `llm` | yes | object | LLM provider settings |
| `embedding` | yes | object | Embedding provider settings |
| `vector_store` | yes | object | Default `qdrant` |
| `knowledge_graph` | yes | object | **Default `provider: null`** (FR-019) |
| `session` | yes | object | Session-store provider |
| `memory` | yes | object | Memory-store provider |
| `observability` | yes | object | `console` (default) or `phoenix` |
| `agent` | yes | object | Agent tuning |
| `slack_bot` | no | object | Bot-worker config; `enabled: false` default |
| `mcp` | yes | object | Map of 14 known MCP keys; unknown keys rejected |
| `entity_extraction` | yes | object | Method: `rule_based` / `llm` / `hybrid` (default `hybrid`) |
| `cloud_provider` | no | string\|null | `aws` / `gcp` / null (default). Fills unset model slots from a per-purpose bundle; explicit blocks + `models` + env always win. |
| `models` | no | object\|null | Per-purpose overrides (`reason`/`tool_call`/`embed`/`rerank`/`summarize`/`extract`), each a `{provider?, model?, effort?}` `ModelSpec`. |

## `auth` block

```yaml
auth:
  mode: oidc                    # open | oidc | login (default oidc when block present)
  issuer: <url>                 # required when mode=oidc; OIDC discovery base
  audience: <string>            # required when mode=oidc
  jwks_cache_seconds: 300       # default
  role_mappings: {}             # IdP group/claim value -> [opsrag role names]
  # mode=login only — first-party password + SSO (cookie sessions):
  login:
    signing_key_env: OPSRAG_SESSION_SIGNING_KEY   # path/env only; inline refused
    session_ttl_seconds: 900
    refresh_ttl_seconds: 1209600
    cookie_secure: true
    cookie_samesite: lax
  sso:
    google:    { enabled: false, client_id: null, client_secret_env: null }
    microsoft: { enabled: false, client_id: null, client_secret_env: null }
    github:    { enabled: false, client_id: null, client_secret_env: null }
```

RBAC roles → scopes (`opsrag.auth.scopes`): `admin` (all) · `member_chat`
(chat) · `member_investigate` (chat+investigate) · `member_mcp` (mcp). Scopes
gate `/query` (chat), the investigate branch (investigate), `/admin/usage`
(admin). Open mode (no `auth`) grants all scopes.

`knowledge_graph.use_in_retrieval` (default `false`): when true and
`provider != none`, the agent augments vector retrieval with a graph-traversal
lane; the graph is always populated at ingestion (low-trust soft-boost).

Behaviour: `auth` absent ⇒ open mode (no enforcement). `mode: oidc` (default
when the block is present) validates `issuer` reachability at startup via the
OIDC `/.well-known/openid-configuration` endpoint and requires both `issuer`
and `audience`; failure ⇒ refuse to start with `AUTH_MISCONFIGURED:<reason>`.
`mode: login` enables first-party login (password + SSO) — built in the auth
feature track. `role_mappings` drives RBAC role assignment.

## `memory` block (Mem0 operational memory)

```yaml
memory:
  provider: memory              # postgres | memory | none | mem0
  mem0_collection: opsrag_mem0_ops   # only used when provider=mem0
  mem0_infer: true              # distil facts via the cheap summarize model
```

`provider: mem0` reuses the main Qdrant client + the project LLM/embedder (no
separate API key); stores per-service operational memory only (the semantic
Q&A cache is unchanged).

## `vector_store` dimension guard

`vector_store.allow_dimension_change` (default `false`): the factory refuses to
start if the embedder's dimension differs from an existing collection's
dimension. Set `true` only for an intentional reindex after switching embedding
models (a silent mismatch corrupts the main index, the QA cache, and
investigations, which share the Qdrant client).

## `mcp` block

Exactly the 14 keys listed in `data-model.md` §1. Each value follows the
`MCPConfigBlock` shape:

```yaml
mcp:
  <name>:
    enabled: false              # required boolean
    secret_ref: null            # optional, Helm hint
    endpoint: null              # optional override
    api_key_env: null           # optional
    extra: {}                   # validated by the integration's own model
```

Validation rules:

- `extra={"forbid"}` is *not* applied to `extra` (it is intentionally
  free-form), but the integration's own model validates whatever it
  reads from there.
- `enabled: true` triggers credential resolution: every name in the
  integration's `required_env` MUST resolve to a non-empty env value.
  Failure ⇒ `MCP_MISCONFIGURED:<name>:<missing-env-var>`.

## Discriminated-union validators

- `llm.provider` ∈ `{anthropic, openai, vertex, bedrock, litellm}` —
  discriminates the keys allowed in the block.
- `vector_store.provider` ∈ `{qdrant, pgvector}`.
- `knowledge_graph.provider` ∈ `{null, neo4j}`.
- `session.provider` ∈ `{inmemory, postgres}`.
- `memory.provider` ∈ `{memory, postgres, none, mem0}`.
- `observability.provider` ∈ `{console, phoenix}`.
- `embedding.provider` ∈ `{openai, fastembed, vertex, bedrock, litellm}`.
- `llm.api_base` / `embedding.api_base` — optional base URL for the
  `litellm` provider, used for self-hosted / OpenAI-compatible endpoints
  (e.g. a Qwen TEI/vLLM server). Ignored by the other providers.

## Environment-variable resolution

Any key suffixed `_env` is interpreted as the *name* of the env var that
carries the actual secret. The settings loader:

1. Reads the env var.
2. Stores it in a `pydantic.SecretStr`.
3. Never logs the resolved value.
4. Refuses to start if the env var is unset when the containing block is
   enabled.

## Contract tests (CI gates)

- `tests/contract/test_config_default_boots.py` — load
  `config.yaml` (the shipped default), assert it validates and yields
  `mcp.<name>.enabled == false` for all 14, and
  `knowledge_graph.provider == "null"`.
- `tests/contract/test_config_unknown_keys_rejected.py` — load a synthetic
  config with an unknown top-level key; assert validation error.
- `tests/contract/test_config_unknown_mcp_rejected.py` — add `mcp.bogus:`
  block; assert validation error names the unknown key.
- `tests/contract/test_config_failfast_on_missing_env.py` — set
  `mcp.kubernetes.enabled: true` without `KUBECONFIG`; assert startup
  fails with `MCP_MISCONFIGURED:kubernetes:KUBECONFIG`.
