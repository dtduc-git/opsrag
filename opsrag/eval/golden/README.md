# OpsRAG golden eval -- authoring policy

> **Source of truth** for how to write a golden YAML entry that
> survives chunker swaps, prompt rewrites, and judge-version drift.
> If a rule here trips at load time, you'll see a `ValueError` from
> `opsrag/eval/loaders.py:_validate_golden` pointing at the offending
> file + golden id.

This policy was locked in Sprint 0 Spec 4 (`tests/eval/specs/sprint0-04-goldens-migration-plan.md`) on 2026-05-07.

---

## YAML schema

Each file under this directory is a **category** (filename = category id, e.g. `confluence.yaml`, `multi_doc_synthesis.yaml`). Inside, a list of goldens:

```yaml
- id: factual_001
  category: factual_lookup
  query: "What stages are defined in generic-pipeline.yaml?"
  expected_sources:
    - devops/gitops-pipeline-templates/generic-pipeline.yaml
  must_contain:
    - "utils"
    - "build"
    - "delivery"
  must_not_contain:
    - "I don't have access"
  notes: |
    Sprint 0 expansion -- anchor on stable factual content.
    Verified anchor exclusivity in Qdrant on 2026-05-07.
```

| Field | Required | Notes |
|---|---|---|
| `id` | yes | Unique within the file. Convention: `<category>_NNN` zero-padded. |
| `category` | yes | Should match the filename stem. |
| `query` | yes | The user-facing question, verbatim. |
| `expected_sources` | optional | List of `source_path` strings, **AND-semantics** (must find ALL). Empty list is valid (used by `negative`, meta-pattern, and `acceptable_sources`-style goldens). |
| `acceptable_sources` | optional | Alternative sources with **OR-semantics** (any one found = satisfied). Use when a question has multiple valid groundings, e.g. a YAML file AND the prose doc that describes it. See "OR-semantics goldens" below. |
| `must_contain` | optional | Substrings the answer MUST include. Anchor on facts, not phrasing. |
| `must_not_contain` | optional | Substrings the answer MUST NOT include. Useful for hallucination guards. |
| `max_retrieved_sources` | optional | Integer cap on how many sources the system may surface. Adds a **retrieval-side** assertion (`RetrievalRestraintMetric`) on top of the answer-side `must_not_contain` — for a query naming a purely fabricated entity the weak-retrieval gate should surface (near) nothing. Set it only on cases whose query has no real-tech term that would legitimately retrieve, and keep it permissive (tolerate a floored top-1 + rewrite residue). Pair it with `must_not_contain` (the cap checks source count, not that the answer refused) — the loader warns otherwise. Omit = no retrieval-side assertion. |

### Path qualification (avoid recall inflation)

`expected_sources` / `acceptable_sources` MUST be **qualified** — include the repo/dir path (`apps/auth/values.yaml`), not a bare filename (`values.yaml`). A bare leaf matches *any* retrieved path ending in it, which silently over-credits recall in a corpus with many duplicate filenames (e.g. vendored chart copies). `match_path` no longer honours the `/`-suffix arm for bare leaves, so an unqualified entry simply won't match; the loader emits a warning at load time. The `<page_id>:<slug>` Confluence/Rootly form is exempt — its page-id is discriminating.

### Running the gate against retrieval, not the cache

The CI gate hits `POST /query`, which runs the **QA cache + classifier + generation**. Launch the eval target server with `OPSRAG_DISABLE_QA_CACHE=1` so a cache hit can't serve a stored answer/sources for a golden — otherwise a retrieval regression is masked and the ranking metrics measure the cache, not retrieval.
| `notes` | optional | Provenance + rationale. **Required for adversarial / multi-doc goldens.** |
| `expected_baseline_faith` | optional | Float in `[0, 1]`. The *design-intended* Faithfulness baseline for this golden. **Documentation-only** -- the eval framework does not enforce or compare against this; it exists so future reviewers don't misread a low-but-stable score as a defect. **Required on adversarial-category goldens.** See "Adversarial baselines" below. |

---

## How matching works (`canonical_path` + `match_path`)

`SourceRecallMetric` calls `loaders.match_path(expected, retrieved)` for each `(expected_sources entry, retrieved sources entry)` pair. A match is recorded if **any** of three rules holds:

1. **Canonical-form equality** -- both sides lowercased, `.md` suffix stripped, surrounding slashes/whitespace trimmed. Catches "same path written two ways".
2. **Suffix on `/` or `:` boundary** -- lets you write a bare path like `12345:slug.md` and match a retrieved repo-prefixed `confluence:SRE/12345:slug.md`. This is the user-friendly authoring form.
3. **Stem-only fallback for `<page_id>:<slug>` paths** -- if both sides have the same digit-only `page_id` prefix, they refer to the same doc even if slugs got truncated to different lengths by chunker version. Cross-namespace collisions are blocked by comparing the prefix where both sides specify one.

See `tests/eval/test_canonical_path.py` for the canonical contract -- 24 tests covering positive cases, partial-substring rejection, namespace collisions, and slug-drift fallbacks.

---

## OR-semantics goldens (`acceptable_sources`)

Some questions have multiple valid groundings -- the literal source file
the user named AND a prose doc that describes the same fact both answer
the question correctly. Strict AND-semantics on `expected_sources`
punishes the bot for picking the more semantically-rich option when
both are factually equivalent.

`acceptable_sources` exists for these cases. Authoring rules:

```yaml
- id: factual_002
  query: "What chart name and version is declared in base-charts/generic-application's Chart.yaml?"
  expected_sources: []                                                       # nothing strictly required
  acceptable_sources:                                                        # any one of these = satisfied
    - devops/base-charts/generic-application/Chart.yaml
    - devops/sre/sre-knowledge-base/docs/platform/generic-application.md
```

How the metrics handle the new field:

| Metric | `expected` non-empty | `expected` empty + `acceptable` non-empty | both empty |
|---|---|---|---|
| `SourceRecallMetric` | AND over expected (current) | OR: 1.0 if any acceptable found, else 0.0 | 1.0 (vacuous) |
| `RankRecallAtKMetric` | AND over expected (in top-K) | OR: 1.0 if any acceptable in top-K | 1.0 |
| `RankPrecisionAtKMetric` | Numerator = retrieved matching `expected OR acceptable` | Same -- union as relevant set | 1.0 if retrieved empty |
| `MRRMetric` | First retrieved matching `expected OR acceptable` | Same | 1.0 |

**Don't use `acceptable_sources` to weaken a real requirement.** It's
for "the bot can answer this from either of these docs", not for "this
golden is too hard, let me list more options until something matches".
Strong filter: if you wouldn't accept the alternative source as a
correct citation in a real user-facing answer, it shouldn't be in
`acceptable_sources` either.

**Don't mix expected + acceptable casually.** When `expected_sources`
is non-empty, `SourceRecallMetric` and `RankRecallAtKMetric` use strict
AND over expected and **ignore** `acceptable_sources`. (Ranking metrics
P@K and MRR still use the union for "relevant" -- that's intentional, to
reward surfacing alternative valid sources high in the list.) If you
want OR-semantics, set `expected_sources: []` explicitly.

---

## Anti-patterns (refused at load time)

The validator exists to catch the recurring failure modes that have bitten this project across iterations. Each rule below corresponds to a real-world bug we don't want to repeat.

### 1. [X] Don't put chunk identifiers in `expected_sources`

```yaml
expected_sources:
  - foo/bar.md::child-3            # [X] chunk-id-shaped suffix
  - chunk_id=abc123                # [X] explicit chunk_id reference
  - parent_chunk_id=xyz            # [X] parent_chunk_id reference
```

**Why:** chunk IDs regenerate on every re-chunk. The same content under a new chunker gets a fresh ID; the assertion silently fails despite retrieval working correctly.

**Do this instead:**

```yaml
expected_sources:
  - foo/bar.md                     # OK source_path only
```

### 2. [X] Don't anchor `must_contain` on contextual-chunking prefixes

```yaml
must_contain:
  - "<context>"                    # [X] contextual-chunk wrapper tag
  - "This chunk is from"           # [X] LLM-generated context preamble
  - "This section describes"       # [X] same family
```

**Why:** these strings exist only when `OPSRAG_CONTEXTUAL_CHUNKING=1`. Toggling that env var (or fixing the augmenter to skip non-prose chunks) breaks the assertion for no real reason.

**Do this instead:** anchor on **stable facts** that exist in the source content regardless of how it was chunked:

```yaml
must_contain:
  - "Cloud SQL"                    # OK a noun phrase from the page itself
  - "MERGE_REQUEST"                # OK a literal value in the YAML
  - "ALL_BROKERS_DOWN"             # OK an alert name from a runbook
```

### 3. [X] Don't depend on chunk count or order

Goldens shouldn't care whether retrieval returned 3 chunks or 7, or in what order. The metrics evaluate the **answer**, not the retrieval shape.

If you find yourself writing a `must_contain` like `"3 chunks"`, you've coupled to retrieval implementation. Refactor.

---

## Adversarial baselines

Adversarial goldens are *designed* to fail or partially fail -- they exercise known RAG failure modes (casual-query degradation, negation, multi-intent, etc.). A score of 1.00 on every adversarial would mean the test set is too easy; a score of 0.00 would mean the system is broken. We expect a stable middle.

The convention (locked Sprint 0, 2026-05-07):

- **Every adversarial golden carries an `expected_baseline_faith` field** that records the design-intended Faith score, measured at authoring time across n=3 identical-system runs.
- Subsequent eval runs are interpreted *relative to that baseline*, not against an absolute 1.0 target. A Pattern 2 (ill-formed/casual) golden that consistently grades 0.30 with sigma=0 is *operating as designed* -- the baseline reflects "judge thinks the bot's casual-query answer is weakly grounded, but stably so." Sprint 1 work goal is to **move the baseline up** (e.g. 0.30 -> 0.50+ via query-rewriting, HyDE, etc.), not to flip it to a pass.
- A score *below* the baseline on a future run indicates a regression in the system's handling of that adversarial pattern.
- A score *significantly above* the baseline (>2sigma_per_golden) without a matching system change indicates a baseline drift to investigate (often: judge prompt change, judge model swap, or measurement error).

**This is documentation-only metadata.** The eval framework does not enforce, compare, or report against `expected_baseline_faith` -- it exists so future reviewers reading category aggregates know which goldens are "stably 0.30 by design" vs "0.30 indicating a real bug."

When authoring a new adversarial golden:
1. Run **n=6** sigma validation (see `scripts/adv_batch_sigma.py`). At n=3 the sigma estimator itself is too noisy: ~17 % false-fail rate observed on 2026-05-07 (2/12 prior-passing goldens flipped at fresh n=3 sampling, both recovered at n>=6).
2. Discard if sigma > 0.15 at n>=6.
3. If sigma <= 0.15, record `expected_baseline_faith` as mu from the n>=6 runs (round to 2 decimal places).
4. If your n=3 result lands close to threshold (e.g. sigma in [0.10, 0.20]), always extend to n=6 before deciding -- it's the boundary cases where measurement noise dominates.

## Adversarial categories

`adversarial.yaml` (Sprint 0 Spec 1) uses a sensitivity tagging system per pattern -- see `tests/eval/specs/sprint0-01-goldens-expansion.md`:

| Sensitivity | Meaning |
|---|---|
| [GREEN] stable | Anchored on facts that survive any retrieval shift. ~0% expected rework after chunker swaps. |
| [YELLOW] neutral | Anchored on stable facts but at a retrieval boundary. ~10-15% touch-up rate after chunker swaps; flag for re-spot-check. |
| [RED] fix-sensitive | Probes retrieval boundary cases on purpose. ~30% likely to need rework after Confluence re-augmentation. **Defer authoring** until after Spec 3's chunking decision point. |

When authoring a new adversarial golden, set the sensitivity in the `notes:` field so future re-eval triage knows where to look.

---

## When in doubt

- **"Is this anchor stable?"** -> grep your `must_contain` strings across the index. If they appear only in the expected source's content (not in chunker artifacts, not in template prefixes), they're stable.
- **"Is this `expected_sources` going to drift?"** -> check if it has `:` followed by a slug. If yes, the stem-only fallback covers slug drift; you're safe. If no, exact path match must hold.
- **"Should I write this as a new category or add to existing?"** -> if 3+ goldens share a failure mode (e.g. "all about contradiction across sources"), make it its own category. Smaller, focused categories give cleaner per-category eval signal.

---

## References

- `tests/eval/specs/sprint0-04-goldens-migration-plan.md` -- full policy rationale
- `tests/eval/test_canonical_path.py` -- the matching contract as code
- `opsrag/eval/loaders.py` -- `canonical_path`, `match_path`, `_validate_golden`
- `opsrag/eval/metrics/source_recall.py` -- the only metric that consumes path matches today
