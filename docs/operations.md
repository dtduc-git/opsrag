# Operations

Day-2 operational guide for running opsrag: indexing and reindexing the
corpus, observability and cost tracking, rate limiting across replicas,
scaling, security hardening, and common troubleshooting. For first-time
setup see [`getting-started.md`](./getting-started.md); for the deployment
topology (roles, the indexing Job/CronJob) see
[`deployment.md`](./deployment.md).

## Indexing and reindexing

### Triggering an index

Indexing is fire-and-forget. `POST /index/repo` (admin scope) registers the
repo in the durable Postgres job-state immediately and returns; progress is
watched via `GET /indexing/status` and `GET /indexing/jobs`. In a Kubernetes
deployment the request spawns an ephemeral indexing Job; with no cluster it
falls back to an in-process background task. A daily CronJob picks up deltas.
`POST /index/source` does the same for non-git sources (Confluence, etc.).

### Incremental skip is content-hash based

The indexer skips a file when its content hash already matches the recorded
hash for that `(repo, branch, path)` row in the `indexed_files` table
(`opsrag/indexed_files/postgres.py:should_skip`). Files unseen in a run are
swept as deletions. This makes re-runs cheap — only changed files are
re-chunked and re-embedded.

### The RATIOS_VERSION reindex caveat

The chunker sizes chunks using **chars-per-token ratios** keyed by content
type (`opsrag/tokenization.py`): config ~2.5, code ~3.5, prose ~4.0, with a
flat fallback of 3. These ratios decide chunk char-budgets → chunk boundaries
→ **chunk IDs**.

Editing any ratio (or adding/removing a doc type) reshapes every chunk
boundary. But the incremental skip keys on **file content hash, not chunker
config** — a ratio change does not change file content, so `should_skip`
will happily skip every unchanged file and leave the old chunk boundaries
and IDs in place. The corpus then silently mis-ranks against the new sizing
until re-ingested.

The discipline (enforced by convention, not a hard guard):

1. **Bump `RATIOS_VERSION`** whenever you change a ratio. It is a single
   greppable constant, logged at index start by `log_active_ratios` (a
   WARNING naming the active ratios + version) so the change is visible in
   the indexing logs.
2. **Force a full reindex.** Because skipping is content-hash based, clear
   the `indexed_files` rows for the affected repos/branches so every file is
   re-chunked under the new sizing (or re-create the collection — see below).

### Embedding model / dimension changes

`vector_store.dimension` is pinned to the embedder's dimension and shared
across the main index, the QA cache, and investigations. Switching embed
models across dimensions (e.g. Titan 1024 <-> Vertex 768 <-> OpenAI 3072)
corrupts an existing Qdrant collection. opsrag **fails closed**: at startup
`assert_dimension_compatible` (`opsrag/vectorstore_guard.py`) compares the
existing collection's dense-vector size to the embedder's dimension and
raises `DimensionMismatchError` unless `vector_store.allow_dimension_change`
is `true`.

To change embed models intentionally:

```yaml
vector_store:
  allow_dimension_change: true   # set ONLY for the reindex window
```

Then clear `indexed_files` and use **fresh collections** (the old vectors are
the wrong dimension and the QA cache must be rebuilt too). A re-create of the
shared collection destroys all three consumers' data, so plan a clean
reindex rather than an in-place flip. Revert `allow_dimension_change` to
`false` afterward so the guard protects you again.

## Observability and cost

### Usage telemetry

LLM and embedder/reranker calls are instrumented into a thread-safe
`UsageTracker` (`opsrag/usage.py`), surfaced live at `GET /usage` (and
`/usage/weekly`, `/me/usage`, `/admin/usage`). When Postgres is configured,
`UsagePersistence` (`opsrag/usage_persistence.py`) buffers events and flushes
batches every ~2 s into the **`opsrag_usage_events`** table
(`model`, `purpose`, `session_id`, `input_tokens`, `output_tokens`,
`latency_ms`). On restart, `seed_tracker` rolls historical rows back into the
in-memory tracker so `/usage` reflects all-time totals immediately instead of
resetting to 0.

The cost field is intentionally **not** stored — cost is recomputed on read
using the current pricing table, so a price-sheet update applies
retroactively.

### Two pricing tables to update per new model

When you add or re-price a model, there are **two** price tables to keep in
sync:

1. `opsrag/usage.py` — `_PRICING` (USD per 1M input/output tokens) plus
   `_PRICING_PER_CALL` for per-request-priced models (Vertex semantic ranker,
   Bedrock Rerank API). Region-prefixed inference profiles like
   `us.anthropic.*` / `us.cohere.*` are normalized to their base id by
   `_pricing_for` before lookup.
2. `opsrag/llms/pricing.py` — the same prices encoded as integer **micro-cents
   per 1M tokens** (1 micro-cent = 1/100,000,000 USD), used for lossless
   integer cost arithmetic.

A new model added to only one table will mis-cost. Update both.

### Tracing (Phoenix)

`observability.provider` selects the tracer: `console` (default), `phoenix`,
or `datadog`. With `phoenix`, opsrag sends OTLP traces to a Phoenix collector
(`opsrag/observability/phoenix.py`) — spans for the agent graph, LLM calls,
and retrieval. Point it with `observability.endpoint` (or
`PHOENIX_COLLECTOR_ENDPOINT`); install the `arize-phoenix-otel` /
`openinference-instrumentation-langchain` extras. If the Phoenix libraries
aren't installed, opsrag logs a warning and continues with no-op tracing —
tracing is never load-bearing.

## Rate limiting across replicas

The default `api.rate_limit_backend: memory` keeps throttle state in-process:
each replica enforces its own limit, so the effective aggregate is
`rate_limit_rpm x replicas` and a login lockout on one pod is invisible to
the others. For any horizontally-scaled deployment, switch to the shared
Redis backend:

```yaml
api:
  rate_limit_backend: redis
  redis_url_env: OPSRAG_REDIS_URL
```

**Redis is required when selected** — `redis` is an optional extra, and
opsrag `PING`s the server at startup and **fails fast** if it is unreachable.
There is no silent fallback to in-memory. See
[`auth.md`](./auth.md#rate-limiting) for the request-limiter and login-lockout
semantics.

## Scaling

- **Stateless API replicas.** Move all shared state off-process before scaling
  out: `sessions.provider: postgres`, `memory.provider: postgres` or `mem0`,
  `api.rate_limit_backend: redis`, and Postgres usage persistence. With those,
  API pods are stateless and horizontally scalable.
- **Indexing is decoupled.** Indexing runs as ephemeral Jobs (or a CronJob for
  deltas), not in the API request path, so heavy ingestion doesn't compete
  with query traffic. See [`deployment.md`](./deployment.md).
- **Embedder choice affects footprint.** The default `fastembed` embedder runs
  on-device (384d) with no external call; API-backed embedders (openai,
  vertex, bedrock, litellm) trade local CPU for network latency and per-token
  cost. Rerankers (fastembed, cohere, bedrock, vertex, noop) trade quality for
  latency/cost similarly.
- **Vector store.** Qdrant supports the hybrid dense+BM25+code-lane RRF path;
  pgvector is simpler (FTS + trigram, no code lane / no true BM25). See
  [`rag-pipeline.md`](./rag-pipeline.md).

## Security hardening checklist

- **Enforce auth.** Set `auth.mode: oidc` (or `login`) — never run `open` on a
  shared/production deployment. Per-session ownership only protects threads
  once callers are authenticated. See [`auth.md`](./auth.md).
- **Secrets via env, never YAML.** Every secret is sourced from an env var
  named by a `*_env` key (API keys, Redis URL, session signing key, SSO
  client secrets, Neo4j password). Inline session signing-key material is
  refused at load time.
- **Set the session signing key explicitly** (`auth.login.signing_key_env` /
  `signing_key_path`) for `login` mode; keep `cookie_secure: true` behind TLS.
- **Shared rate limiting.** Use the Redis backend across replicas so the login
  brute-force lockout is actually enforced cluster-wide.
- **Fail-closed dimension guard.** Leave `vector_store.allow_dimension_change:
  false` except during a deliberate reindex window.
- **PII in memory.** mem0 memory redacts emails/tokens/secrets before storage,
  but treat the memory collection as sensitive and scope access accordingly.
- **CI scanning.** The pipeline runs Trivy, gitleaks, CodeQL, and pip-audit;
  keep these green before publishing images.
- **MCP integrations are read-only**, but still scope each integration's
  credentials to least privilege — the agent can only do what the token can.

## Common troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| Startup aborts with `DimensionMismatchError` | Embed model changed without a reindex | Set `allow_dimension_change: true`, reindex into fresh collections, revert. |
| Retrieval quality dropped after a chunker tweak | Ratios changed but files skipped (content-hash unchanged) | Bump `RATIOS_VERSION`, clear `indexed_files`, full reindex. |
| Startup fails right after `rate_limit_backend: redis` | Redis unreachable (fail-fast PING) | Fix `OPSRAG_REDIS_URL` / network; Redis is required for this backend. |
| `/usage` resets to 0 after a restart | Usage persistence not configured | Configure Postgres so `seed_tracker` reloads all-time totals. |
| New model shows `$0.00` cost | Model missing from a pricing table | Add it to **both** `opsrag/usage.py` and `opsrag/llms/pricing.py`. |
| Cross-user gets 404 on a valid thread | Per-session ownership (by design) | The thread is owned by another verified identity; 404 avoids an existence oracle. |
| Login lockout not shared across pods | `rate_limit_backend: memory` with >1 replica | Switch to the Redis backend. |
| Eval gate passes despite a regression | QA cache served stored sources | Run the eval target with `OPSRAG_DISABLE_QA_CACHE=1`. |
| Memory recall silently empty | Best-effort mem0 write/read failed | Check logs for `mem0 … failed`; verify the embedder dimension and Qdrant collection. |
| `MCP_MISCONFIGURED:<name>:<VAR>` at boot | An enabled MCP integration's required env/config is missing | Supply the named secret/key; startup fails fast by design. |

## See also

- [`deployment.md`](./deployment.md) — roles, the indexing Job/CronJob model,
  and production topology.
- [`auth.md`](./auth.md) — auth modes, per-session ownership, rate limiting.
- [`evaluation.md`](./evaluation.md) — the regression gate and cache bypass.
- [`memory.md`](./memory.md) — long-term memory backends and PII redaction.
- [`configuration.md`](./configuration.md) — every config block and env
  precedence.
