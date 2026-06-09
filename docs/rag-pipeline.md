# RAG Pipeline

The end-to-end retrieval and answer pipeline: how a source file becomes searchable chunks, and how a question becomes a grounded, cited answer. This covers ingestion (parse → chunk → enrich → embed → index), hybrid retrieval with RRF, reranking and MMR diversity, the CRAG/Self-RAG correction loops and anti-hallucination gates, the semantic Q&A cache, and the corrections/feedback loop.

## Overview

There are two halves, joined by the vector store:

```
INGESTION (offline)                          RETRIEVAL (per query)
  SCM / source connector                       classify + cache lookup
    -> parse (per doc-type)                       -> route -> HyDE
    -> parent/child chunk                         -> hybrid retrieve (RRF lanes)
    -> enrich metadata                            -> entity_expand (1-hop)
    -> contextual prefix (optional)               -> rerank + MMR
    -> embed (+ code lane)                        -> grade (CRAG)
    -> upsert to vector store  <---------- search -> generate (parent substitution)
    -> light-graph / Neo4j edges                  -> verify + hallucination gate
                                                  -> cache write-back
```

Ingestion lives in `opsrag/ingestion/`, `opsrag/parsers/`, `opsrag/chunkers/`, `opsrag/embedders/`. Retrieval is a LangGraph agent in `opsrag/agent/graph.py` + `opsrag/agent/nodes/`, fed by `opsrag/vectorstores/` and `opsrag/rerankers/`. Provider selection (embedder, reranker, vector store, models) is all configuration — see `./configuration.md`.

## Ingestion

`IngestionPipeline` (`opsrag/ingestion/pipeline.py`) drives `SCM -> parse -> chunk -> embed -> vector store` for git repos (`index_repo`) and non-git connectors (`index_source`: Confluence, Rootly, Slack, …). It runs a bounded producer/consumer pipeline — a single `asyncio.Queue` with capacity `OPSRAG_FILE_PARALLEL * 2` (default parallelism 4) backpressures the SCM streamer so a 50K-file repo never parks 50K `RepoFile` bodies on the event loop.

### Per-file flow

`_process_file` runs this sequence (CPU-bound parse/chunk are offloaded to a thread pool so HTTP handlers stay responsive):

1. **Parser selection** — first parser whose `supports(path, content)` matches.
2. **Content-hash dedup** — sha256 of the file; if `IndexedFilesTracker.should_skip` says the exact bytes were already indexed for `(repo, branch, path)`, the whole embed/parse/upsert chain is skipped (only `last_seen_at` is bumped). No-op without Postgres.
3. **Parse** → `ParsedDocument` (sections + metadata).
4. **Provenance** — `apply_provenance` centrally stamps `updated_at`/`service`/`url` onto `doc.metadata` so every doc type (not just markdown) is recency-rankable.
5. **Chunk** → parent + child chunks (below).
6. **Enrich** — `enrich_metadata` adds deterministic facets (`doc_type`, `environment`, `tier`, `tags`, `language`, …) to each chunk's payload. Additive only; non-fatal.
7. **Light-graph** — `_attach_entity_ids` stamps deterministic `entity_ids` onto each chunk (powers the `entity_expand` retrieval lane) and collects Postgres adjacency edges. Zero-LLM, independent of Neo4j.
8. **Contextual prefix** (optional) — see below.
9. **Orphan sweep** — `delete_by_filter({repo, source_path})` before upsert, because a chunk ID hashes `content[:64]` (so any edit to a chunk's first 64 chars mints a new ID and the old vector would otherwise linger forever).
10. **Embed + upsert** — only *searchable* chunks (parents are excluded from search) are embedded; parents go in as BM25-only / NULL-embedding points.
11. **Code dual-write** — code-typed chunks are additionally embedded with the code embedder into the `opsrag_code` collection (below).
12. **Graph lane** (optional Neo4j) and `indexed_files.record(content_hash)`.

A repo-level **deletion sweep** runs only after a *fully completed* `index_repo` pass: files whose `last_seen_at` predates the run start were removed from source, so their chunks (and code-collection / graph edges) are purged. A partial/aborted run never purges.

### Parsers and chunking

Parsers (`opsrag/parsers/`) are per-doc-type: `markdown`, `runbook`, `postmortem`, `helm`, `k8s`, `terraform`, `alert`, `code_structure` (AST), and a `generic` fallback. Each emits a `ParsedDocument` with `sections` (heading + body + breadcrumb) and a `DocType`. The code parser emits one section per function/class via AST so a `def` stays whole.

`ParentChildChunker` (`opsrag/chunkers/parent_child.py`) creates **two layers**:

- **Parent chunks** — one per document section (or per split when a section overflows the parent budget). Stored for generation context; *never searched*.
- **Child chunks** — sliding windows over each parent (`child_size=256` tokens, `child_overlap=32`). These are what get embedded and searched. Searching children gives precise hits; substituting their parent at generation time (below) gives the LLM full surrounding context.

Sizing is **per-doc-type**. Targets are expressed in tokens, then converted to a char budget via `chars_per_token_for(doc_type)` (`opsrag/tokenization.py`): config ~ 2.5, code ~ 3.5, prose ~ 4.0. A flat ratio mis-sizes both ends (a `replicas: 3` YAML line tokenizes to many short tokens; English packs ~4 chars/token), so a 256-token child is actually ~256 tokens *of that content type*.

Boundary snapping is content-aware:

- **Code + structured config** (`_LINE_AWARE_DOC_TYPES`: code, k8s, terraform, helm, yaml, Dockerfile, alerts) split on **line boundaries** — a child never cuts through an identifier (`handle_web|hook`) or a `key: value` pair, which would poison the BM25 lexical lane.
- **Prose** snaps the window end back to a paragraph break, then a sentence end (only if it doesn't shrink the window below ~60%).
- **Code parents** get a larger budget (`code_parent_max_tokens=2048` vs `1024`) so most functions fit whole.

> **Re-index caveat.** The chars/token ratios decide chunk char-budgets → boundaries → chunk IDs. Editing any value in `_CHARS_PER_TOKEN_BY_TYPE` (or adding a doc type) silently invalidates every previously-indexed chunk until a **full re-index** — old chunks keep their stale boundaries. Bump `RATIOS_VERSION` when you change them; the pipeline logs the active ratios + version at index start (`log_active_ratios`) as a reminder.

### Contextual chunking (optional)

`opsrag/ingestion/contextual.py` prepends a one-sentence context so a 256-token window "knows" its doc-level scope. Toggle with `OPSRAG_CONTEXTUAL_CHUNKING=1`. Two paths:

- **Prose** (`RUNBOOK`, `POSTMORTEM`, `GENERIC_MARKDOWN`, `ARCHITECTURE`, `ADR`) — one LLM call per document (full doc + numbered child list in, JSON array of contexts out; ~$0.001/doc on Flash). Children only.
- **Structured** (`HELM`, `TERRAFORM`, `YAML_CONFIG`, `KUBERNETES`, `DOCKERFILE`, `ALERT_DEFINITION`, code) — a **deterministic, free** template: `[Context: <type label> in <repo>/<path> [env <env>] [section/key '<...>']]`. This matters because `values.yaml` repeats per env (staging/prod) — same content, different scope — and a Terraform module splits across `apps.tf` / `apps_variables.tf`. The env is detected from the path (`values-prod.yaml` → `production`); the leading config key disambiguates sibling chunks with no heading.

Crucially, the context goes into a separate `embed_content` field (the **dense lane only**), not `content`. BM25/FTS and the stored payload always index the clean `content`, so injecting "Context"/path tokens doesn't dilute IDF or double-count path slugs.

### Embedding and the code lane

Embedders (`opsrag/embedders/`): `fastembed` (default, on-device 384-d), `openai`, `vertex`, `bedrock`, `litellm`. A `cached` wrapper memoizes embeddings. The chunker embeds `embed_content` when set (contextual prefix), else `content`.

The **code lane** (P3) is a dual-write: chunks of code doc-types (`PYTHON`, `JAVASCRIPT`, `TYPESCRIPT`, `GO`, `JAVA`, `SHELL`) are *additionally* embedded with a code-specific embedder and upserted into the `opsrag_code` collection. Prose/config stays in the main collection only. Both `code_embedder` and `code_vector_store` must be wired; otherwise this is a no-op and behavior is identical to pre-P3. The code lane is non-fatal — a failure leaves the main collection intact.

## Retrieval and answer generation

Each query is driven by `query_with_session` / `query_with_session_events` (streaming SSE) in `opsrag/agent/graph.py`, which wraps a compiled LangGraph. Before the graph runs, the request does: retry-meta expansion ("retry"/"again" → prior question), coreference rewrite (follow-ups like "what about its config"), embedding, classification, and a **Q&A cache lookup**. On a cache miss the graph runs and the result is written back to the cache.

### Graph topologies

Four builders, selected by `agent.mode` config:

| Mode | Builder | Flow |
| --- | --- | --- |
| `minimal` | `build_minimal_graph` | hyde → vector_retrieve → [rerank] → generate → verify → END |
| `full` | `build_full_graph` | route → hyde → retrieve → [entity_expand] → rerank → grade (CRAG) → generate → verify → hallucination-check (+ memory load/save) |
| `tool_calling` | `build_tool_calling_graph` | tool_decide <-> tool_execute (loop) → tool_synthesize, with retrieval fall-through |
| `multi_agent` | `build_multi_agent_graph` | entry_route → triage → tool_caller <-> reasoner → generator, with retrieval fall-through + a CASUAL friendly lane |

The full RAG flow (`build_full_graph`) is the canonical pipeline. `entry_route` fast-paths CASUAL chitchat ("hi", "what can you do") straight to a `friendly_generator` (no retrieval). Everything else flows through the nodes below. (`build_hybrid_graph` is a removed stub — the old Neo4j graph-anchored lane raises `NotImplementedError`; never wire it.)

### Route → HyDE

`route_query` (`nodes/router.py`) classifies into a `query_type` (`incident`, `howto`, `architecture`, `config_lookup`, `blast_radius`, …). It's a plain edge into HyDE — `query_type` is consumed downstream (HyDE skip, generator system-prompt selection, memory saver) rather than for branching.

`hyde_expansion` (`nodes/hyde_expansion.py`) writes a short *hypothetical answer* (HyDE, Gao et al. 2022) and embeds **that** instead of the raw query, so the dense vector lands closer to real document text. It is a no-op (embeds the raw query) when:

- the classifier verdict is `live`/`mixed` (paraphrasing erases temporal/ID signal),
- `query_type == config_lookup` or the query carries exact identifiers/anchors (HyDE invents plausible-but-wrong YAML keys),
- the query is shorter than 4 words, or
- Flash errors/returns empty.

Only the hypothetical is embedded — and via `embed_texts` (document space), not `embed_query`, since it's now a doc-to-doc match. Lexical anchoring on the user's literal terms comes free via the BM25 lane (always run on the raw query).

### Hybrid retrieval and RRF

`vector_retrieve` (`nodes/vector_retriever.py`) calls the store's `hybrid_search`. Both vector stores fuse lanes with **Reciprocal Rank Fusion** (RRF, Cormack et al. 2009, `k=60`, parameter-free): each chunk's score is `sum(lane_weight / (k + rank))` across the lanes it appears in. Documents in multiple lanes accumulate score (consensus); single-lane hits still contribute. Each lane fetches a deep candidate pool (`candidate_k = max(top_k*8, 50)`).

**Qdrant** (`vectorstores/qdrant.py`) — the full hybrid store:

- **Dense** lane (semantic, the configured embedder).
- **Sparse BM25** lane — a named sparse vector with the IDF modifier, fed identifier-subtoken-augmented text (`bm25_sparse`).
- **Code** lane (optional) — the code embedding against the `opsrag_code` collection.

**pgvector** (`vectorstores/pgvector.py`) — the dependency-light alternative:

- **Dense** ANN (HNSW).
- **Lexical FTS** via `ts_rank_cd` + `websearch_to_tsquery('simple', ...)` over a GIN index (this is *not* true BM25).
- **Trigram** substring lane (best-effort) — fires only on identifier queries when `pg_trgm` is available, recovering exact-symbol recall the whole-token FTS lexer misses. **No code lane.**

Both stores apply **identifier-aware lane weights** (`vectorstores/lane_weights.py`): an identifier-heavy query (snake_case, dotted paths, kebab service names, backticked tokens, routes, globs) biases the lexical lane to `1.5x` and the code lane to `1.25x` (gentler — it's a semantic lane), while dense/graph stay `1.0` so the boost is additive, not zero-sum. Prose queries get all `1.0` (identical to vanilla RRF).

Both stores then apply an **authoritative-content priority boost** (`vectorstores/priority.py`). Authoritative content (SRE-KB `docs/architecture/*` → `architecture-canonical`, operator-approved corrections → `user-correction`, other SRE-KB → `high`) gets a bounded **additive** RRF bonus (a *fraction* of one RRF unit `1/(k+1)`), not a multiplier. A multiplier would steamroll the compressed RRF band (~0.010–0.016) and vault a weakly-ranked chunk past a genuine single-lane #1; the additive form wins close calls without leaping past a strong multi-lane consensus hit (~0.033+). The dense-only `search()` path uses the multiplier form instead, since cosine lives on a clean `[0,1]` band.

When the query names specific repos/files/slugs, `vector_retrieve` also runs **slug/filename fanout** sub-queries and fuses the pools one level up with `rrf_merge_pools` (`vectorstores/rrf.py`) — same math, applied across whole ranked `SearchResult` lists from parallel sub-queries rather than raw points from one query.

### Entity expansion (1-hop)

When a light graph is wired, `entity_expand` (`nodes/entity_expand.py`) runs between retrieve and rerank. It seeds `entity_ids` from the top vector chunks, gets 1-hop neighbors from the Postgres adjacency, and fetches a few extra chunks whose `entity_ids` intersect the neighbors (Qdrant `MatchAny` filter + vector score). It is **augment-only and fail-safe** — empty graph or any error → no change; the vector result stays the authoritative line.

### Reranking and MMR diversity

`rerank` (`nodes/reranker.py`) scores the full candidate pool with a cross-encoder reranker. Providers (`opsrag/rerankers/`): `fastembed` (default), `cohere`, `bedrock`, `vertex`, `noop`. The node:

- Applies a **path-anchor boost** — an additive, capped `+0.15` when a chunk's `source_path`/`repo` literally contains a query anchor token. Additive (not the old uncapped `1.5x`) so it only overturns *close* calls, not large quality gaps.
- Surfaces per-reranker calibration floors into state (`min_rerank_score`, `rerank_trust_score`) so downstream gates use the right thresholds for *this* provider (FastEmbed's sigmoid `[0,1]` vs Cohere's compressed-low scale).
- Widens for `listing_intent`/`plural_repo_intent` (broad coverage, skip narrow rerank) and `synthesis_intent` (`top_k → 10` so both compared docs survive).
- Falls back to bi-encoder order on a reranker outage (with a mid-band neutral score so the outage neither fakes "insufficient info" nor suppresses CRAG).

Optional **MMR** diversity (`rerankers/mmr.py`, default OFF via `rerank_diversity=0`) re-orders the reranked list to break up near-duplicate config variants the cross-encoder stacks at the top:
`score(c) = lambda*relevance(c) - (1-lambda)*max sim(c, selected)`. Relevance is min-max normalized so the diversity weight is provider-agnostic; similarity defaults to token-set Jaccard over chunk content (no embeddings needed). `diversity in {None, 0}` is a byte-for-byte pass-through.

`rerank_decision` is a path-aware gate: if the query named anchors but no kept chunk's path/repo matches any anchor **and** the best cross-encoder score is below the noise floor, it routes straight to `insufficient_info` instead of fabricating from adjacent chunks.

### CRAG grading + Self-RAG loops

`grade_documents` (`nodes/grader.py`) is the CRAG corrective step (Yan et al. 2024). It fans out one binary relevance LLM call per candidate (bounded at `_GRADE_CONCURRENCY=6` to avoid 429 cascades; fail-open per chunk). `grade_decision` then routes:

- **relevant chunks present** → `generate`,
- **none relevant + retries left** → `rewrite_query` → re-retrieve (skipping HyDE, since the rewrite is already aggressive),
- **none relevant + retries exhausted** → `insufficient_info` (honest fallback, no fabrication).

The grader leaves `graded_chunks` empty on early attempts so the rewrite loop actually fires (unconditional flooring would silently disable CRAG). It only **floors** to the top candidates as a last resort — when the cross-encoder was confident about its #1 (`best_rerank_score >= rerank_trust_score`, so re-running CRAG would waste budget on already-good retrieval) or the retry budget is spent. The floor count scales with distinct anchors / sub-queries (capped at the generation budget) so multi-fact answers don't ship a 1-chunk context.

### Generation with parent substitution

`generate` (`nodes/generator.py`) swaps each retrieved **child** for its **parent** (`_substitute_parents`, deduped by parent id) so the LLM sees full surrounding context (~1024-token parents) instead of the 256-token slice that matched. Falls back transparently if the store has no parent lookup or chunks are synthetic. It also assembles a path-tree summary (when chunks share a pivot directory), an anchor hint (when the query named entities none of the sources contain — "lead with the gap"), recent conversation turns, and per-user Mem0 memories into the system prompt. The final answer uses the stronger "answer" model (the router's `pro_llm`) when wired; cheap nodes stay on the base LLM. The post-substitution `final_chunks` are stashed so the API surfaces exactly what the LLM saw.

A **regenerate loop** (Self-RAG style) re-runs generation when grounding fails (below), warming the temperature each retry and telling the model the prior attempt failed grounding, so it actually changes rather than re-emitting the same answer at temperature 0.

### Anti-hallucination gates

Two independent gates run after generation:

1. **`verify_answer`** (`nodes/answer_verifier.py`) — Flash extracts every file path / YAML-key / CRD name the answer *claims* and decides whether each appears in the cited evidence (recalled per-user memories count as valid evidence). Unverifiable claims get a single-line `Warning: Some claims could not be verified…` hedge **prepended** (lines are never silently stripped — the engineer sees what the agent doubted). Fail-open on any LLM/JSON error.

2. **`check_hallucination`** (`nodes/hallucination.py`) — Flash judges whether *every* factual claim is supported by the context. It grounds against the **same** evidence the generator saw (`final_chunks`, the parents) — checking children produced spurious "not grounded" verdicts. `hallucination_decision` routes: `grounded` → end, `not_grounded` → regenerate, `max_retries_hit` → end (ship best-effort). The regenerate budget (`regen_count`/`max_regens`) is **separate** from the CRAG rewrite budget (`retry_count`/`max_retries`) so the two loops don't cannibalize one budget.

Every per-turn budget and scratch field is explicitly reset in `query_with_session` — the LangGraph Postgres checkpointer is keyed by `thread_id` and last-write-wins, so any field not reset leaks last turn's chunks/budgets.

## Semantic Q&A cache

`QAVectorCache` (`opsrag/qa_cache.py`) short-circuits the whole graph when a sufficiently similar question was answered before. It's a separate Qdrant collection (`opsrag_qa_cache`) keyed by the **embedding of the question**. Toggle via `OPSRAG_QA_CACHE` (default ON); the eval harness disables it with `OPSRAG_DISABLE_QA_CACHE` so retrieval regressions aren't masked by cache hits.

**Lookup** (cosine >= threshold, default `0.93`) is guarded by three additional checks, because cosine alone has a high false-positive rate at 0.93 (a year/verb/env flip passes cosine but changes meaning):

- **Discriminator tokens** — the meaning-shifting token *set* of the current and cached questions must match **exactly**, or it's a forced miss. A two-layer ensemble: a precision-first regex layer (multi-digit numbers, years, semver, ticket IDs like `OPS-7890`, environment names, commit SHAs, kebab service names, GitLab path leaf IDs) tagged with type prefixes (`year:2026`, `verb`/`env`/`semver:3.7`), plus an opt-in spaCy NER recall layer (`OPSRAG_QA_NER_SPACY=1`). Environment names are built *dynamically* from the active deployment so the engine carries no org-specific vocabulary.
- **Flash judge** (`qa_cache_judge.py`) — for the borderline cosine band `[0.93, 0.97]`, a 1-shot Flash prompt decides SAME vs DIFFERENT (catches paraphrase + intent/verb/env swaps the regex misses). Above `0.97` it skips the judge (trust the score); identical strings short-circuit; default-allow on error or when disabled (`OPSRAG_QA_JUDGE=0`).
- **Degenerate-answer backstop** — refuses to serve (or store) answers that are too short, low-alpha, single-glyph-dominant, etc. (the "2592 dashes" incident).

**TTLs are category-aware** (`agent/classifier.py:CATEGORY_POLICY`), keyed by the query classifier (forensic / live / procedural / mixed / infra_graph / casual / unknown):

| Category | TTL | Cache? | QA threshold |
| --- | --- | --- | --- |
| forensic (frozen past event) | 90 d | yes | 0.92 |
| procedural (how X works) | 30 d | yes | 0.93 |
| infra_graph | 4 h | yes | 0.94 |
| mixed (live + forensic) | 5 min | yes | 0.96 |
| live ("now", "right now") | — | **skip** | 0.99 |
| casual | — | **skip** | 0.99 |
| unknown (default) | 14 d | yes | 0.93 |

The cache is also bypassed for user-scoped queries (`my`/`our`/`your`) and for **tool-path answers** (live state) — those go to a separate `opsrag_investigations` collection instead. Cache writes are skipped when the hallucination gate explicitly failed (gated on the durable `grounding_checked` flag, not the routing function). User-memory-influenced answers are `user_scope`-namespaced so a recalled personal fact can't leak to another user on a high cosine match.

**Stale-while-revalidate** (`OPSRAG_QA_CACHE_SWR`, default on): an entry just past its TTL is served instantly tagged `is_stale`, and a background `_swr_revalidate` re-runs the agent (on a throwaway thread id, SWR disabled) to replace the entry for the next caller.

**Invalidation**: `invalidate_repo` (hooked into `/index/repo`) drops every cache entry whose `source_repos` contains the reindexed repo, so stale answers don't survive a reindex.

## Corrections and feedback loop

Two operator signals feed back into retrieval quality:

- **Corrections** (`opsrag/correction_store.py`). An operator-approved correction (question + correct answer) is injected as a synthetic chunk into the main collection with `repo = priority = "user-correction"`. It's embedded with the same provider the retriever uses (so cosine geometry lines up) and upserted with both dense and BM25 sparse vectors, `wait=True` so a `/query` right after sees it. The `user-correction` priority tag lifts its fused RRF score at search time (the `1.8x`-tier additive bonus), so the corrected answer reliably out-ranks the original sources on the next ask. The generator/reasoner prompts treat `user-correction` chunks as ground truth ("if chunks disagree, user-correction wins") and cite them inline.

- **Feedback** (`opsrag/feedback_store.py`). Thumbs-up/down (and submitted corrections) from the chat UI are written to a Postgres `opsrag_feedback` table (`direction` in `{1, -1, 2}`), with a partial index on `direction = -1` for a cheap "what went wrong this week" view SREs use to author corrections into the SRE-KB. A thumbs-down also flags the matching Q&A cache entry low-quality (`quality_flag = "low"`) so it's skipped on future lookups. Writes are graceful — a broken pool logs and returns without breaking the UX.

## See also

- `./configuration.md` — selecting embedders, rerankers, the vector store, models, and the agent mode; all the `OPSRAG_*` env toggles named here.
- `./evaluation.md` — the golden eval harness, the cache-disable flag, and the retrieval/faithfulness metrics this pipeline is tuned against.
- `./mcp-integrations.md` — the live read-only tool integrations the `tool_calling` / `multi_agent` topologies call.
