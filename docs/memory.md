# Long-term memory

opsrag can persist durable, per-user/per-service memory across sessions so
follow-up turns feel like a continuous conversation and so operational facts
accumulate over time. Memory is **best-effort** — a write or read failure
never breaks a request — and is fronted by a single `MemoryStore` interface
with three backends.

## Backends

`memory.provider` selects the backend (`opsrag/config.py:MemoryConfig`):

| `provider` | Store | Persistence | Use case |
|---|---|---|---|
| `memory` (default) | `InMemoryMemoryStore` | Lost on restart | Local dev / tests |
| `postgres` | `PostgresMemoryStore` | Durable (`opsrag_memory` table) | Simple key/value recall |
| `mem0` | `Mem0ServiceMemory` | Durable (Qdrant collection) | Semantic, LLM-distilled recall |
| `none` | (disabled) | — | Turn memory off entirely |

All backends implement the same Protocol (`opsrag/interfaces/memory.py`):
`put` / `get` / `search` / `delete` over a `namespace: tuple[str, ...]` and a
`key`. The agent never special-cases the backend — it talks to the interface,
so switching providers is a config change.

```yaml
memory:
  provider: mem0           # memory | postgres | mem0 | none
  dsn_env: POSTGRES_DSN    # used by the postgres backend
```

## Namespaces

Memory is addressed by a namespace tuple plus a key. opsrag uses a small set
of namespaces, all keyed off the request's `user_id`:

| Namespace | Key | Written by | Purpose |
|---|---|---|---|
| `("user", user_id)` | `thread_id` | `save_memory_node` | **Conversational memory** — the durable facts a follow-up turn recalls. |
| `("user", user_id, "preferences")` | `default` | (preferences) | Per-user preference blob loaded into context. |
| `("user", user_id, "topics")` | service name | `save_memory_node` | Frequent-topic counters (per service). |
| `("user", user_id, "query_types")` | query type | `save_memory_node` | Per-query-type counters. |

The `memory_loader` node (`opsrag/agent/nodes/memory_loader.py`) runs before
generation: it loads preferences, the top frequent topics, and — via
`search(("user", user_id), query=<current query>)` — semantically relevant
durable facts about this user, injecting them so the answer is personalized.

The `memory_saver` node (`opsrag/agent/nodes/memory_saver.py`) runs after
generation, **fire-and-forget** off the response critical path. The
recall-critical write happens first: it stores only the user's own message
(`{"user_message": query}`) under `("user", user_id)`. Storing the
assistant's answer was deliberately removed — with mem0's fact extraction it
caused facts to be distilled *from the answer* as if the user had stated
them. Durable user facts come from what the user says. Topic and query-type
counters are written last (low value when routed through fact extraction).

## Mem0 service memory

The `mem0` backend (`opsrag/memory/mem0_store.py`) gives semantic,
LLM-distilled recall by **reusing opsrag's existing infrastructure** — no
separate API key or second client path:

- **Vector store:** the project's Qdrant, in a dedicated collection
  (`memory.mem0_collection`, default `opsrag_mem0_ops`). The live Qdrant
  client is injected so the existing connection is reused.
- **LLM:** the project's configured LLM, mapped to mem0's provider name
  (anthropic / openai / vertex / aws_bedrock / litellm). Run an Ollama model
  through the `litellm` provider.
- **Embedder:** the project's embedder, with an optional override (see
  below). The mem0 **graph store stays OFF**.

### Infer mode

`memory.mem0_infer` (default `true`) routes each write through mem0's
fact-extraction LLM call, distilling a turn into durable facts. Set it
`false` to store raw turns verbatim (cheaper, no LLM call, but no
distillation). Because an infer write is a blocking ~1–2 s LLM call, mem0's
synchronous client is run off the event loop via `asyncio.to_thread`, and
the memory_saver schedules writes in the background so the user never waits
on them.

### Embedder override

The main retrieval embedder may be a code-tuned model mem0 cannot drive — for
example Cohere Embed v4 on Bedrock, where mem0's `aws_bedrock` embedder sends
Titan-style payloads and Cohere v4 rejects them ("Malformed request").
Memory facts are short natural-language strings, so a simpler embedder is
fine. Point mem0 at a compatible one with:

```yaml
memory:
  provider: mem0
  mem0_embed_provider: bedrock                       # e.g. bedrock
  mem0_embed_model: amazon.titan-embed-text-v2:0     # a model mem0 can drive
  mem0_embed_dimension: 1024                          # MUST match the model
```

Leave these unset to reuse the main embedder (works for OpenAI / Vertex).
`mem0_embed_dimension` must equal the model's true dimension (Titan v2 =
1024) — it is passed to mem0's Qdrant collection as `embedding_model_dims`.

## Service-scope safety (no global bucket)

`Mem0ServiceMemory` refuses to write to a global / shared bucket. A namespace
must have a non-empty trailing "service" segment (`_has_service_segment`); an
empty tuple or one whose last segment is empty/whitespace is treated as "no
service" and the write is **skipped and logged**, never raised. The namespace
tuple is mapped to a mem0 `user_id` string by joining segments with `:` (for
example `("ops", "acme-notes-be")` → `"ops:acme-notes-be"`).

## PII redaction

Before any text reaches mem0's fact extraction or storage, a conservative,
dependency-free redaction pass runs (`_redact_pii` in
`opsrag/memory/mem0_store.py`). It is intentionally narrow — operational
memory should never accumulate raw credentials or personal email addresses:

- **Emails** → `[redacted-email]`.
- **Bearer / API tokens** — `bearer …`, `token: …`, `api_key=…` → `[redacted-token]`.
- **Opaque secrets** — GitHub PATs (`ghp_…`), OpenAI keys (`sk-…`), AWS access
  keys (`AKIA…`), and long hex/base64-ish runs → `[redacted-token]`.

Redaction is applied recursively to nested dict/list/scalar values, and again
on the rendered text after values are joined, to catch PII that only appears
once flattened into a single string.

## Best-effort guarantee

Every mem0 method swallows and logs exceptions and **never propagates into
the agent path**. The handlers catch `BaseException` (not just `Exception`),
because mem0 can fail in non-`Exception` ways — for example a `SystemExit`
from a spaCy model-download attempt in a slim image. The one exception is
`asyncio.CancelledError`, which is always re-raised so cancellation
propagates correctly. The practical contract: memory makes answers better
when it works and is invisible when it does not.

> **Conversational-memory write window:** because the recall-critical write
> is fire-and-forget and an infer write takes ~1–2 s, a follow-up sent within
> ~1–2 s of the previous answer might not yet see the just-stored fact. In
> practice the user spends longer than that reading before typing again, so
> the window is negligible.

## See also

- [`configuration.md`](./configuration.md) — the full `memory` config block.
- [`architecture.md`](./architecture.md) — the agent graph and where the
  memory loader/saver nodes run.
- [`operations.md`](./operations.md) — observability and cost (memory writes
  consume LLM/embedder tokens when `mem0_infer` is on).
