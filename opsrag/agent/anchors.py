"""Anchor entity extraction from a query.

An "anchor" is a token in the user's question that very likely names a
specific repo, module, file, service, or product the answer must be
*about* (not just adjacent to). Examples:

- `acme-tf-state`         -- repo slug (hyphenated)
- `acme-notes-be`         -- service name
- `acme-qc-be`            -- service name
- `variables.tf`          -- specific filename
- `sre-knowledge-base`    -- repo slug
- `kustomize.yaml`        -- specific filename
- `TICKET-7890`           -- Jira ticket

Heuristic -- a token is an anchor when it matches any of:

1. A hyphenated identifier with at least one hyphen AND at least one
   segment of length >= 3  (e.g. `acme-tf-state`, `acme-notes-be`,
   but NOT `to-do`).
2. A dotted filename with a known infra extension
   (`.tf`, `.yaml`, `.yml`, `.json`, `.sh`, `.py`, `.md`, `.toml`).
3. A Jira-style ticket id  (`[A-Z]{2,}-\\d{3,}`).
4. A path-like token containing `/` (e.g. `modules/pagerduty/env/main.tf`).

We DELIBERATELY do not extract single-word tokens -- those are too
common to be specific. The retrieval pipeline already handles those via
dense vector search.

Used by the rerank node to apply a multiplicative boost to chunks whose
`source_path` or `repo` literally contains an anchor token (case-
insensitive), and by the grader/decision to force `insufficient_info`
when the query has anchors but NO chunk matches any of them.
"""
from __future__ import annotations

import re

_HYPHEN_RE = re.compile(r"\b[a-z0-9]+(?:-[a-z0-9]+){1,}\b", re.IGNORECASE)
_DOT_FILE_RE = re.compile(
    r"\b[a-z0-9_.-]+\.(?:tf|tfvars|yaml|yml|json|sh|py|md|toml|hcl|conf|env)\b",
    re.IGNORECASE,
)
_TICKET_RE = re.compile(r"\b[A-Z]{2,}-\d{2,}\b")
_PATH_RE = re.compile(r"\b[a-z0-9_.-]+(?:/[a-z0-9_.-]+){1,}\b", re.IGNORECASE)
_STOPWORDS = frozenset({
    # Hyphenated tokens that are too generic to anchor on.
    "step-by-step", "well-known", "single-replica", "multi-replica",
    "read-only", "read-write", "service-account", "service-name",
    "key-value", "to-do",
})


def extract_anchors(query: str) -> list[str]:
    """Return a deduplicated lowercase list of anchor tokens from the query.

    Order preserved by first-occurrence. Length >= 3 segments per hyphenated
    token. Stopwords removed.
    """
    seen: dict[str, None] = {}
    for pat in (_TICKET_RE, _PATH_RE, _DOT_FILE_RE, _HYPHEN_RE):
        for m in pat.finditer(query):
            tok = m.group(0).lower()
            if tok in _STOPWORDS:
                continue
            # Hyphenated token must have >= 1 segment of length >= 3 to qualify
            # as anchor -- `to-do` won't, `acme-notes-be` will (segment "acme").
            if "-" in tok and "/" not in tok and "." not in tok:
                segs = tok.split("-")
                if not any(len(s) >= 3 for s in segs):
                    continue
            seen[tok] = None
    return list(seen.keys())


# Single-word tokens that are TOO generic to be useful repo anchors on
# their own. Common English + product nouns. We still emit them as weak
# anchors (see `weak_repo_anchors`) but downstream code should treat
# them as a fallback ONLY, not strong matches.
_STRONG_STOPWORDS = frozenset({
    "the", "and", "for", "with", "what", "where", "when", "which", "how",
    "this", "that", "these", "those", "from", "into", "over", "your",
    "list", "show", "tell", "give", "find", "want", "need", "please",
    "module", "modules", "chart", "charts", "service", "services",
    "project", "projects", "value", "values", "playbook", "playbooks",
    "environment", "environments", "ansible", "config", "configuration",
    "file", "files", "directory", "directories", "subdir", "subdirs",
    "all", "any", "every", "some", "one", "two", "three",
    "repo", "repository", "repositories", "code", "codebase", "source",
})


def weak_repo_anchors(query: str) -> list[str]:
    """Single-word tokens (length >= 4) from the query that COULD name a
    repo by domain keyword (e.g. "terraform", "gitops", "monorepo",
    "kafka"). Returned lowercase, deduplicated, in first-occurrence
    order. Strong stopwords are filtered out so we don't end up
    matching every repo whose name contains "service" or "config".

    Used as a fallback when `extract_anchors` returned nothing -- common
    when the user says "the terraform repo" instead of "acme-tf-state".
    Caller should only consume the result if find_repo_by_substring
    resolves to exactly one repo.
    """
    import re as _re
    seen: dict[str, None] = {}
    for tok in _re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", query):
        low = tok.lower()
        if low in _STRONG_STOPWORDS:
            continue
        seen[low] = None
    return list(seen.keys())


def path_matches_any_anchor(path: str, repo: str, anchors: list[str]) -> bool:
    """Case-insensitive: does the chunk's source_path or repo literally
    contain any of the anchor tokens?"""
    if not anchors:
        return False
    haystack = f"{path or ''}|{repo or ''}".lower()
    return any(a in haystack for a in anchors)
