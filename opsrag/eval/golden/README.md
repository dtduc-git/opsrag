# OpsRAG golden eval -- public `samples/` corpus

> **Source of truth** for the golden YAML schema and the path-matching
> rules. If a rule trips at load time you'll see a `ValueError` from
> `opsrag/eval/loaders.py:_validate_golden` naming the offending file + id.

These goldens grade retrieval and answer quality against the **synthetic
`samples/` corpus shipped in this repo** -- so the eval is runnable publicly
with no private corpus and no secrets. The corpus is a small, fictional
"Acme Notes" SaaS:

```
samples/
  runbooks/      001 deploy, 002 db-failover, 003 scaling, 004 incident, 005 cache-flush
  postmortems/   2026-01-15 db-outage, 2026-02-03 api-latency, 2026-03-20 auth-outage
  manifests/     acme-notes-api.yaml (Namespace/Deployment/Service/HPA)
  terraform/     main.tf (db primary + replica + bucket)
```

---

## Path convention -- relative to `samples/`

`expected_sources` / `acceptable_sources` are paths **relative to the
`samples/` directory**, exactly as the indexer records `source_path` when it
indexes the corpus with `repo="samples"`. Examples:

```
runbooks/001-acme-notes-deploy.md
manifests/acme-notes-api.yaml
postmortems/2026-03-20-acme-notes-auth-outage.md
terraform/main.tf
```

Never reference a path outside `samples/` -- the public eval must be
self-contained. (The previous goldens pointed at an unshipped private corpus
and were not runnable; this set replaces them.)

---

## YAML schema

Each file under this directory is a **category** (filename = category id).
Inside, a list of goldens:

```yaml
- id: factual_002
  category: factual_lookup
  query: "How do I roll back an acme-notes deploy?"
  expected_sources:                 # AND-semantics: must find ALL (relative to samples/)
    - runbooks/001-acme-notes-deploy.md
  must_contain: ["rollout undo"]    # literal substrings the answer MUST include
  must_not_contain: ["I don't know"]
  notes: "..."
```

| Field | Required | Notes |
|---|---|---|
| `id` | yes | Unique within the file. Convention: `<category>_NNN`. |
| `category` | yes | Matches the filename stem. |
| `query` | yes | The user-facing question, verbatim. |
| `expected_sources` | optional | `source_path` strings, **AND-semantics** (must find ALL). Empty for negatives + `acceptable_sources`-only goldens. |
| `acceptable_sources` | optional | Alternatives, **OR-semantics** (any one found = satisfied). Use when a fact is grounded in more than one doc (e.g. the HPA target lives in both the manifest and the scaling runbook). Set `expected_sources: []` to get OR-semantics on recall. |
| `must_contain` | optional | Substrings the answer MUST include. **Anchor on facts, not phrasing**, and copy the literal verbatim from the source file. |
| `must_not_contain` | optional | Substrings the answer MUST NOT include (hallucination guards). |
| `max_retrieved_sources` | optional | Integer cap; adds a retrieval-side restraint assertion for negatives. Pair with `must_not_contain`. |
| `notes` | optional | Provenance + rationale. |

**Every `must_contain` literal must appear verbatim in the referenced file.**
A wrong literal makes the eval lie. Grep the `samples/` file before adding one.

---

## Categories

| File | Meaning |
|---|---|
| `factual_lookup.yaml` | Single-fact lookups (a command, a number, a config value). |
| `runbook_howto.yaml` | Procedure how-tos -- "how do I do X?" against a runbook. |
| `listing.yaml` | Enumerations (e.g. the four golden signals). |
| `multi_doc_synthesis.yaml` | Questions whose full answer needs 2+ docs (AND-semantics). |
| `negative.yaml` | Facts NOT in the corpus (Kafka, per-key rate limits). The answer must hedge/refuse. **Empty relevant set -> the ranking/recall metric SKIPS these**; they exist for the answer-quality tier (`must_not_contain` guards against fabrication). |

---

## How matching works (`canonical_path` + `match_path`)

The ranking metrics + the offline harness call
`loaders.match_path(expected, retrieved)`. A match holds if **any** of:

1. **Canonical-form equality** -- both sides lowercased, `.md` stripped,
   slashes/whitespace trimmed.
2. **Suffix on `/` or `:` boundary** -- lets a bare path match a
   repo-prefixed retrieved path.
3. **Stem-only fallback for `<page_id>:<slug>` paths** -- same digit-only
   page-id prefix = same doc even if the slug drifted by chunker version.

Qualify paths (`runbooks/001-...md`, not a bare `001-...md`); a bare leaf is
warned at load time and won't match.

---

## Running the eval

**Tier 1 -- offline retrieval gate (no secrets, runs in CI):**

```bash
uv sync --extra dev --extra eval --extra fastembed
pytest tests/integration/test_eval_samples_retrieval.py -v
```

Indexes `samples/` into an in-process Qdrant with a local FastEmbed ONNX
embedder, scores every golden's Recall@5 / MRR, and asserts aggregate
Recall@5 stays above threshold. The reusable harness is
`opsrag/eval/retrieval_offline.py` (`build_offline_index` + `retrieval_scores`).

**Tier 2 -- full answer-quality eval (needs an LLM + judge):**

```bash
python -m opsrag.eval run --tag local
```

Hits a live OpsRAG instance, runs generation + the DeepEval judge for
faithfulness / source-recall / must-contain. Needs model + judge credentials.

---

## Anti-patterns (refused at load time)

1. **No chunk identifiers in `expected_sources`** (`...::child-3`,
   `chunk_id=...`) -- they regenerate every re-chunk.
2. **No contextual-chunking prefixes in `must_contain`** (`<context>`,
   "This chunk is from") -- they exist only when contextual chunking is on.
3. **No chunk count/order dependence** -- evaluate the answer, not the
   retrieval shape.

---

## References

- `opsrag/eval/loaders.py` -- `canonical_path`, `match_path`, `_validate_golden`
- `opsrag/eval/metrics/ranking.py` -- Recall@K / Precision@K / MRR / NDCG@K
- `opsrag/eval/retrieval_offline.py` -- the offline harness
- `tests/integration/test_eval_samples_retrieval.py` -- the CI gate
