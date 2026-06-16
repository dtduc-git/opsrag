# Evaluation harness

opsrag ships a golden-set evaluation harness that scores retrieval and
answer quality against a versioned corpus of expected questions, and a CI
regression gate that fails a build when any gated metric drops below
threshold. It lives in `opsrag/eval/` with the golden YAML under
`opsrag/eval/golden/`.

The goldens grade against the **synthetic `samples/` corpus shipped in this
repo** (a fictional "Acme Notes" SaaS: runbooks, postmortems, a k8s manifest,
and a Terraform module), so the eval is runnable publicly with no private
corpus.

## Two tiers

| Tier | Command | Needs | When |
|---|---|---|---|
| **Offline retrieval** | `pytest tests/integration/test_eval_samples_retrieval.py` | nothing but the one-time FastEmbed model fetch | always-on CI gate (`eval-offline`) |
| **Answer quality** | `python -m opsrag.eval run` | a live opsrag + LLM + judge credentials | the live-server `eval-regression` gate |

The offline tier is the no-secrets proof that retrieval works; the answer
tier adds the LLM judge for faithfulness/answer-quality. The two share the
golden set and the path-matching rules.

## Tier 1 -- offline retrieval eval (no secrets)

`tests/integration/test_eval_samples_retrieval.py` indexes `samples/` into an
**in-process Qdrant** (`url=":memory:"`) using a **local FastEmbed ONNX
embedder** (`BAAI/bge-small-en-v1.5`, 384-dim -- no API key), loads the public
goldens, runs retrieval for every scored golden, and asserts aggregate
**Recall@5 >= 0.85** (observed ~1.0 on the validated recipe). Negative goldens
(empty relevant set) are skipped for ranking. Run it:

```sh
uv sync --extra dev --extra eval --extra fastembed
pytest tests/integration/test_eval_samples_retrieval.py -v
```

The reusable harness is `opsrag/eval/retrieval_offline.py`:
`build_offline_index(samples_dir) -> (embedder, vector_store)` and
`retrieval_scores(embedder, vector_store, goldens, k=5)` (returns per-golden
Recall@K + MRR and the aggregate means). The `eval-offline` CI job runs this
on every push/PR -- it's the always-on, secret-free gate.

## Tier 2 -- answer-quality eval (live server + judge)

The rest of this document covers the live-server tier driven by
`python -m opsrag.eval run`.

## What it measures

Each golden query is sent to a running opsrag instance (`POST /query`,
`stream: false`), and the response -- answer, retrieved source paths, and
chunk content -- is scored by a suite of metrics
(`opsrag/eval/runner.py:_run_metrics`). Metrics are computed cheap-first
(deterministic set/ranking metrics) then expensive-last (the LLM judge):

| Metric | Kind | What it checks |
|---|---|---|
| `SourceRecall` | Deterministic, set | Were the expected source paths retrieved (AND-semantics; OR via `acceptable_sources`)? |
| `Precision@5` | Deterministic, ranking | Fraction of the top-5 that are relevant. |
| `Recall@5` | Deterministic, ranking | Are the expected sources in the top-5 (top-of-list recall the generator actually reads)? |
| `Recall@10` | Deterministic, ranking | Are they anywhere in the top-10? |
| `NDCG@5` | Deterministic, ranking | Position-aware relevance over the top-5. |
| `MRR` | Deterministic, ranking | Reciprocal rank of the first relevant source. |
| `RetrievalRestraint` | Deterministic, retrieval-side | For negative/fabricated-entity goldens, did the system surface at most `max_retrieved_sources`? |
| `MustContain` / `MustNotContain` | Deterministic, answer text | Required / forbidden substrings in the answer (hallucination guards). |
| `Faithfulness` | LLM judge | Is the answer grounded in the retrieved context (Vertex Gemini Pro judge)? |

`Recall@5` and `Recall@10` are gated **together** on purpose: a document
slipping from rank 4 to rank 8 tanks Recall@5 while Recall@10 stays flat,
which localizes a re-ranking regression.

Path matching (`opsrag/eval/loaders.py:match_path`) is forgiving across
chunker versions: canonical-form equality, suffix match on a `/` or `:`
boundary, and a `<page_id>:<slug>` stem-only fallback for slug drift. Authoring
rules -- including why `expected_sources` must be path-qualified and must never
contain chunk IDs -- live in [`opsrag/eval/golden/README.md`](../opsrag/eval/golden/README.md).

## Running it

```sh
# List the golden categories and their counts.
python -m opsrag.eval list

# Run the full golden set against a target, write a tagged report.
OPSRAG_URL=http://localhost:8000 \
  python -m opsrag.eval run --tag baseline

# Restrict to one category, or to the smoke set.
python -m opsrag.eval run --tag step1 --category factual_lookup
python -m opsrag.eval run --tag dry-run --golden _smoke

# CI regression gate: exit 1 if any gated metric's mean is below threshold.
python -m opsrag.eval run --tag ci --gate
```

Key environment:

- `OPSRAG_URL` -- the opsrag instance to query (default `http://localhost:8000`).
- `OPSRAG_JUDGE_MODEL` -- the faithfulness judge model (default `gemini-2.5-pro`,
  via `VertexGeminiJudge`; needs Vertex credentials).
- `OPSRAG_DISABLE_QA_CACHE=1` -- **must be set on the target server** (see below).

Every run writes a diff-able markdown report and a JSON sibling to
`tests/eval/reports/<tag>.{md,json}` (`opsrag/eval/report.py:write_report`).
The report aggregates per-metric means, per-category and per-prompt-variant
breakdowns, and full per-query detail.

## The gate and its thresholds

`--gate` turns the run into a pass/fail regression check. It compares each
gated metric's corpus mean against a floor
(`opsrag/eval/report.py:DEFAULT_GATE_THRESHOLDS`) and exits non-zero if any
is below:

| Metric | Threshold |
|---|---|
| `SourceRecall` | 0.80 |
| `Precision@5` | 0.40 |
| `Recall@5` | 0.60 |
| `Recall@10` | 0.70 |
| `NDCG@5` | 0.50 |
| `MRR` | 0.50 |
| `RetrievalRestraint` | 1.00 |
| `Faithfulness` | 0.70 |

`MustContain` / `MustNotContain` are reported per query but are not part of
the aggregate floor -- they assert on individual goldens, not a corpus mean.
Thresholds mirror the per-test-case thresholds the metric classes define and
are tuned as the golden set matures.

### Cache-bypass pre-flight

`POST /query` runs through the QA cache, classifier, and generation. A cache
hit serves a **stored** answer and sources, so the ranking metrics would
measure the cache, not retrieval -- masking a regression. Before a gated run,
the CLI probes the target (`_assert_cache_bypassed`): it sends a cacheable
query twice, and if the second call returns `cache_hit: true` it **aborts the
gate** (exit 3) with a message telling you to launch the eval server with
`OPSRAG_DISABLE_QA_CACHE=1`. Transport errors during the probe don't block
the gate.

## Vacuous-case handling

Not every golden has expected/acceptable sources -- `negative` goldens and
meta-pattern goldens may have none. For those, a ranking metric is **vacuous**
and marks itself `skipped=True`. Vacuous evaluations are **excluded** from
the aggregate means rather than scored a free 1.0 (or a penalizing 0.0):

- Counting them as free 1.0s used to inflate Recall/MRR/Precision for exactly
  the `cross_source` category meant to test source-blending.
- The report surfaces a `Skipped` column and a note prompting reviewers to
  label those goldens with `expected_sources` so the ranking metrics actually
  measure them.
- The gate guards the other direction too: a metric with **zero** scored
  cases (all vacuous or the metric absent entirely) is reported as a
  **failure**, so a silently-empty category can't pass the gate green.

Adversarial goldens are a related but distinct case: they are *designed* to
partially fail, and carry an `expected_baseline_faith` field recording the
design-intended Faithfulness score. That field is **documentation-only** -- the
framework does not enforce or compare against it -- so reviewers don't misread
a stable-by-design 0.30 as a defect. See the golden README's "Adversarial
baselines" section.

## CI gating

The `eval-regression` job in `.github/workflows/ci.yml` runs the gate, but
only when it can be meaningful:

- **Path-scoped trigger.** The job diffs the PR and runs only when
  eval-relevant paths changed -- `opsrag/{agent,agents,mcp,eval,ingestion,parsers,chunkers,vectorstores,embedders}/`.
  These cover both the agent/MCP/eval code and the retrieval pipeline, since a
  change to chunking or embedding changes what gets retrieved and so must be
  able to trip the gate. Unrelated changes skip it.
- **Needs a live target.** The gate queries a running opsrag and needs judge
  credentials. It runs only when `OPSRAG_EVAL_URL` is configured; the
  instance and corpus are the operator's to provision (the gate can't assert
  quality against an empty index). The target must run with
  `OPSRAG_DISABLE_QA_CACHE=1`.
- **Fail-closed on a dropped secret.** Set `OPSRAG_EVAL_REQUIRED=1` to turn
  the "no URL -> skip" silent no-op into a hard failure, so a missing secret
  can't make the gate pass green.

When all conditions hold, CI runs `python -m opsrag.eval run --tag ci --gate`
and the build fails if any gated metric is below threshold.

## See also

- [`opsrag/eval/golden/README.md`](../opsrag/eval/golden/README.md) -- golden
  authoring policy, the matching contract, and anti-patterns.
- [`architecture.md`](./architecture.md) -- the `/query` path the eval drives.
- [`operations.md`](./operations.md) -- cost/observability of eval runs (the
  judge consumes Vertex Pro tokens) and indexing the corpus under eval.
