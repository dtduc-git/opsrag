"""Shared authoritative-content priority boost.

Single source of truth for the priority tier -> score-multiplier mapping used by
the Qdrant store, the pgvector store, AND the agent's fanout re-ranking (so the
boost isn't silently discarded when a slug/filename fanout RRF-merge runs). Kept
dependency-free (no qdrant_client / asyncpg) so any module can import it.
"""
from __future__ import annotations

# SRE-KB content gets a tier tag at index time; tags map to these multipliers.
# Tuned so authoritative content wins ties without steamrolling on-topic hits.
_PRIORITY_BOOSTS: dict[str, float] = {
    "architecture-canonical": 2.0,  # SRE-KB docs/architecture/*
    "user-correction": 1.8,         # operator-APPROVED correction
    "high": 1.5,                    # other SRE-KB canonical content
}

# Repo/path rules for deriving the tier when no stored tag is available
# (e.g. the pgvector store has no `priority` payload field).
_HIGH_PRIORITY_REPO_SUBSTR = ("sre-knowledge-base",)
_ARCHITECTURE_PATH_PREFIX = "docs/architecture/"


def priority_multiplier(tag: str | None) -> float:
    """Score multiplier for a priority tag; 1.0 (no boost) for None/unknown."""
    if not tag:
        return 1.0
    return _PRIORITY_BOOSTS.get(tag.lower(), 1.0)


def chunk_priority(repo: str | None, source_path: str | None) -> str | None:
    """Derive a priority tier from repo/source_path (SRE-KB + architecture
    tiers only; user-correction is stamped at index time, not derivable here)."""
    if not repo or not any(s in repo.lower() for s in _HIGH_PRIORITY_REPO_SUBSTR):
        return None
    if source_path and source_path.startswith(_ARCHITECTURE_PATH_PREFIX):
        return "architecture-canonical"
    return "high"
