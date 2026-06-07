"""Shared per-query RRF lane weights (identifier-aware).

Single source of truth for the dynamic lane weighting used by BOTH the Qdrant
and pgvector hybrid paths. Identifier-heavy queries (function names, dotted
paths, kebab service names, backticked tokens, routes) bias the lexical (BM25)
lane up -- pure dense embeddings underperform on exact-symbol retrieval. Kept
dependency-free so any store can import it.
"""
from __future__ import annotations

import re

_IDENT_PATTERN = re.compile(
    r"(`[^`]+`"                          # backticked tokens
    r"|\b[a-z][a-z0-9_]*\.[a-zA-Z_]"      # dotted paths: foo.bar / api.v1
    r"|\b[a-z]+_[a-z]+(?:_[a-z]+)*\b"     # snake_case (>=2 segments)
    r"|\b[A-Z][a-z]+[A-Z][a-zA-Z]+\b"     # CamelCase (>=2 capitalised words)
    r"|\b[a-z]+-[a-z]+(?:-[a-z]+)+\b"     # kebab-case >=3 segments (acme-notes-be-api)
    r"|(?<![A-Za-z0-9])/[a-zA-Z][a-zA-Z0-9_/.-]+"  # absolute paths or routes
    r"|\*\.[a-z]+\b"                       # file globs (*.py)
    r"|\b[a-z]{2,}[0-9]+\b"                # lowercase-with-digit suffix: auth2, kafka1
    r")"
)

# Boost for the BM25/lexical lane on identifier-heavy queries.
_IDENT_BM25_BOOST = 1.5
# The code lane is a SEMANTIC lane (code-specific embedder), so it gets a gentler
# boost than BM25 -- a full 1.5x ran symbol queries ~3:1 lexical:semantic.
_CODE_LANE_BOOST = 1.25


def compute_lane_weights(query_text: str | None) -> dict[str, float]:
    """RRF lane multipliers by query shape. Prose -> all 1.0 (identical to the
    non-identifier-aware behaviour). Identifier-shaped -> lexical 1.5x, code
    1.25x, dense/graph stay 1.0 so the boost is additive, not zero-sum."""
    if not query_text or not query_text.strip():
        return {"dense": 1.0, "sparse": 1.0, "graph": 1.0, "code": 1.0}
    if _IDENT_PATTERN.search(query_text):
        return {"dense": 1.0, "sparse": _IDENT_BM25_BOOST, "graph": 1.0, "code": _CODE_LANE_BOOST}
    return {"dense": 1.0, "sparse": 1.0, "graph": 1.0, "code": 1.0}
