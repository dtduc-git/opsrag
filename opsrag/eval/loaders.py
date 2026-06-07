"""YAML golden-set loader -> DeepEval LLMTestCase.

Each YAML file under opsrag/eval/golden/ is a category (factual_lookup,
runbook_howto, multi_doc_synthesis, listing, negative, ...). Schema:

    - id: factual_001
      category: factual_lookup
      query: "What stages are defined in generic-pipeline.yaml?"
      expected_sources:
        - devops/gitops-pipeline-templates/generic-pipeline.yaml
      must_contain: ["utils", "build", "delivery"]
      must_not_contain: ["I don't have access"]
      notes: "..."

See `opsrag/eval/golden/README.md` for the canonical-path matching
policy and the anti-pattern guard rules enforced at load time.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from deepeval.test_case import LLMTestCase

_log = logging.getLogger("opsrag.eval.loaders")

GOLDEN_DIR = Path(__file__).parent / "golden"


# -- canonical_path: chunker-stable path matching ---------------------
#
# Goldens reference docs by `source_path`. The retrieval API returns
# repo-prefixed paths (`confluence:SRE/12345:slug.md`,
# `rootly:acme/incidents/...`). A golden written as the bare path
# `12345:slug.md` must still match. Additionally, when the chunker is
# swapped (Sprint 3), slugs get truncated to different lengths -- we
# need a stem-only fallback so a golden referencing
# `12345:long-slug.md` still matches a retrieved
# `12345:long-sl.md` if the underlying page_id (`12345`) is the same.
#
# All matching logic lives here so SourceRecall, the load-time
# validator, and any future metric share one source of truth.

def canonical_path(s: str) -> str:
    """Lowercase, drop trailing `.md`, strip surrounding whitespace and
    leading/trailing slashes. The form used for equality checks."""
    if not s:
        return ""
    out = s.strip().lower()
    if out.endswith(".md"):
        out = out[:-3]
    return out.strip("/")


# Confluence-style id-prefixed slug: `<page_id>:<slug>` (page_id is
# digits-only by convention). When the slug differs but the page_id
# matches, treat as the same doc.
_PAGE_ID_PREFIX = re.compile(r"^(\d+):")


def _page_id(canon: str) -> str | None:
    """Extract the digit-only page-id prefix from a canonical path
    (e.g. `12345:long-slug` -> `12345`). Returns None when the path
    isn't id-prefixed (most non-Confluence paths)."""
    # The page_id appears AFTER the repo prefix (e.g.
    # `confluence:sre/12345:slug`) -- find the last colon-separated
    # segment that starts with digits.
    parts = canon.rsplit("/", 1)
    leaf = parts[-1] if parts else canon
    m = _PAGE_ID_PREFIX.match(leaf)
    return m.group(1) if m else None


def match_path(expected: str, retrieved: str) -> bool:
    """Does `retrieved` represent the same document as `expected`?

    Three matchers, ordered most-strict to most-loose:

    1. Canonical-form equality (post `canonical_path`). Catches the
       common case of golden written as repo-prefixed path matching
       retrieval's repo-prefixed path.

    2. Suffix on `/` or `:` boundary. Lets a golden written as a bare
       `12345:slug` match a retrieved `confluence:sre/12345:slug`. This
       is the SourceRecall behaviour we shipped on 2026-05-07.

    3. Stem-only fallback for `<page_id>:<slug>` paths. If both sides
       carry the same digit-only page_id prefix, they refer to the
       same doc even if the slug got truncated to different lengths
       by chunker-version differences. Stops `12345:foo` from matching
       `99999:foo` (different page) while letting `12345:foo` match
       `12345:fooo` (same page, slug drift).
    """
    e = canonical_path(expected)
    r = canonical_path(retrieved)
    if not e or not r:
        return False
    if e == r:
        return True
    # Skip the "/"-suffix arm for a BARE filename (no path separator AND no
    # `:`-page-id): `config.yaml` / `values.yaml` would otherwise match ANY
    # retrieved path ending in it -- silently inflating recall/precision in a
    # corpus with many duplicate filenames (e.g. 41K vendored chart copies, all
    # `values.yaml`). A qualified path (`apps/auth/values.yaml`) or a
    # discriminating `<page_id>:<slug>` form still uses both arms.
    e_is_bare_leaf = "/" not in e and ":" not in e
    if (not e_is_bare_leaf and r.endswith("/" + e)) or r.endswith(":" + e):
        return True

    # Stem-only fallback. Both sides must have the same page_id
    # prefix; AND the colon-segments either side of the page_id must
    # match too (so `confluence:sre/12345:foo` matches
    # `confluence:sre/12345:fo` but NOT `rootly:inc/12345:foo`).
    e_pid = _page_id(e)
    r_pid = _page_id(r)
    if e_pid and r_pid and e_pid == r_pid:
        # Compare repo prefix (everything before the leaf segment)
        # to avoid cross-namespace collisions.
        e_prefix = e.rsplit("/", 1)[0] if "/" in e else ""
        r_prefix = r.rsplit("/", 1)[0] if "/" in r else ""
        # Empty expected prefix means the golden was bare `<id>:<slug>`
        # -- that's the user-friendly form, accept any retrieved prefix.
        if not e_prefix or r_prefix.endswith(e_prefix):
            return True

    return False


@dataclass
class GoldenQuery:
    id: str
    category: str
    query: str
    expected_sources: list[str] = field(default_factory=list)
    # Alternative sources that ALSO satisfy the question (OR-semantics).
    # Use when a query has multiple valid groundings -- e.g., the literal
    # YAML file the user named AND a prose doc that describes its contents
    # both answer "what's in this Chart.yaml?". `expected_sources` retains
    # AND-semantics (must find ALL); `acceptable_sources` is OR (any one
    # found = satisfied). Ranking metrics (Precision@K, MRR) treat the
    # union as the "relevant set" -- surfacing any acceptable doc in top-K
    # counts as a relevant hit. See `golden/README.md` for the full schema.
    acceptable_sources: list[str] = field(default_factory=list)
    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    # Retrieval-side restraint cap for negative goldens: the max number of
    # sources the system may surface for a "doesn't exist in the corpus" query.
    # When set, RetrievalRestraintMetric fails if more are returned -- catches a
    # weak-retrieval-gate regression that pulls in junk even when the answer
    # text still happens to refuse. None = no retrieval-side assertion.
    max_retrieved_sources: int | None = None
    notes: str = ""
    # Optional design-intent baseline for the Faithfulness metric, expressed
    # as a value in [0, 1]. Used for adversarial goldens where the bot is
    # *expected* to score below 1.0 (e.g. casual-query Pattern 2 cases that
    # consistently grade ~0.30). Sprint 1 work tracks improvement *relative
    # to this baseline* (e.g. 0.30 -> 0.50+) rather than absolute pass.
    # Documentation-only -- does not affect eval pass/fail.
    expected_baseline_faith: float | None = None

    @classmethod
    def from_dict(cls, d: dict) -> GoldenQuery:
        return cls(
            id=d["id"],
            category=d.get("category", "general"),
            query=d["query"],
            expected_sources=list(d.get("expected_sources") or []),
            acceptable_sources=list(d.get("acceptable_sources") or []),
            must_contain=list(d.get("must_contain") or []),
            must_not_contain=list(d.get("must_not_contain") or []),
            max_retrieved_sources=d.get("max_retrieved_sources"),
            notes=d.get("notes", ""),
            expected_baseline_faith=d.get("expected_baseline_faith"),
        )


# -- Anti-pattern guards (Spec 4 policy) -----------------------------
#
# These run at load time and refuse goldens that couple to
# implementation details that don't survive a chunker swap.
# See opsrag/eval/golden/README.md for the rationale.

# Chunk identifiers shouldn't appear in expected_sources -- they
# regenerate every re-chunk. Generic patterns: "chunk_id=", a UUID-
# shaped suffix, or a "::child-N" / "::parent-N" tail.
_CHUNK_ID_PATTERNS = [
    re.compile(r"::(?:child|parent)-\d+"),  # opsrag chunk-id structure
    re.compile(r"chunk_id\s*[=:]"),          # explicit "chunk_id=" form
    re.compile(r"parent_chunk_id\s*[=:]"),
]

# Contextual-chunk prefix anchors. These exist only when contextual
# chunking is on; toggling it breaks the assertion for no real reason.
_CONTEXTUAL_PREFIX_ANCHORS = [
    "<context>",
    "</context>",
    "this chunk is from",     # case-insensitive
    "this section describes", # generic LLM-generated context preamble
]


def _validate_golden(g: GoldenQuery, source_file: Path) -> None:
    """Refuse goldens that couple to chunker internals. Raises ValueError
    with a descriptive message naming the offending field + the rule
    that caught it.
    """
    where = f"{source_file.name}::{g.id}"

    # Rule 1: expected_sources must not reference chunk_id / parent_chunk_id.
    for src in g.expected_sources:
        for pat in _CHUNK_ID_PATTERNS:
            if pat.search(src):
                raise ValueError(
                    f"{where}: expected_sources entry {src!r} contains a "
                    f"chunk-id-shaped pattern. Goldens must reference docs "
                    f"by `source_path` only -- chunk IDs regenerate on "
                    f"every re-chunk and break the assertion. "
                    f"See opsrag/eval/golden/README.md."
                )

    # Rule 2: must_contain anchors must not match contextual-prefix patterns.
    for anchor in g.must_contain:
        low = anchor.lower()
        for prefix in _CONTEXTUAL_PREFIX_ANCHORS:
            if prefix in low:
                raise ValueError(
                    f"{where}: must_contain anchor {anchor!r} matches a "
                    f"contextual-chunking prefix pattern ({prefix!r}). "
                    f"These exist only when OPSRAG_CONTEXTUAL_CHUNKING=1; "
                    f"toggling it breaks the assertion. Anchor on stable "
                    f"factual content instead. "
                    f"See opsrag/eval/golden/README.md."
                )

    # Soft hygiene warnings (don't fail the load -- just surface a risk):
    # (a) Bare-leaf expected_sources (no path separator, no `:`-page-id) match
    #     ANY retrieved path ending in that filename, so in a duplicate-filename
    #     corpus they over-credit recall. match_path now ignores the "/"-arm for
    #     them, so a bare leaf will simply never match -- qualify the path.
    for src in g.expected_sources:
        if "/" not in src and ":" not in src:
            _log.warning(
                "%s: expected_sources entry %r is a bare filename -- qualify it "
                "with its repo/dir path (config.yaml -> apps/foo/config.yaml), "
                "else it won't match (and previously over-matched duplicates).",
                where, src,
            )
    # (b) A negative golden with a retrieval cap but no answer-side guard only
    #     checks source COUNT, not that the answer refused -- pair them.
    if g.max_retrieved_sources is not None and not g.must_not_contain:
        _log.warning(
            "%s: max_retrieved_sources set without must_not_contain -- the "
            "retrieval cap alone doesn't assert the answer refused; add an "
            "answer-side guard for belt-and-suspenders.",
            where,
        )


def load_golden(category: str | None = None) -> list[GoldenQuery]:
    """Load all golden queries, optionally filtered to one category file.

    Validates each entry against Spec 4 anti-patterns at load time;
    raises ValueError with a pointer to the offending file + golden id
    if any rule trips. Fail-loud is intentional -- bad goldens silently
    pollute aggregates and waste eval budget.
    """
    if category:
        files = [GOLDEN_DIR / f"{category}.yaml"]
    else:
        files = sorted(GOLDEN_DIR.glob("*.yaml"))

    out: list[GoldenQuery] = []
    for path in files:
        if not path.exists():
            continue
        data = yaml.safe_load(path.read_text()) or []
        for entry in data:
            g = GoldenQuery.from_dict(entry)
            _validate_golden(g, path)
            out.append(g)
    return out


def to_llm_test_case(
    g: GoldenQuery,
    actual_output: str,
    retrieval_context: list[str],
    retrieved_sources: list[str] | None = None,
) -> LLMTestCase:
    """Convert a golden query + actual OpsRAG run into a DeepEval test case.

    `retrieval_context` is the chunk content the LLM judge will see for
    faithfulness scoring (file content, not just paths).
    `retrieved_sources` is the raw list of source paths used for SourceRecall
    set-intersection. If omitted, falls back to retrieval_context.
    """
    return LLMTestCase(
        input=g.query,
        actual_output=actual_output,
        expected_output=None,  # we don't assert exact-string equality
        retrieval_context=retrieval_context,
        metadata={
            "id": g.id,
            "category": g.category,
            "expected_sources": g.expected_sources,
            "acceptable_sources": g.acceptable_sources,
            "retrieved_sources": retrieved_sources if retrieved_sources is not None else retrieval_context,
            "must_contain": g.must_contain,
            "must_not_contain": g.must_not_contain,
            "max_retrieved_sources": g.max_retrieved_sources,
        },
    )
