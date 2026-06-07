"""Shared authoritative-content priority boost.

Single source of truth for the priority tier -> boost mapping used by the Qdrant
store, the pgvector store, AND the agent's fanout re-ranking (so the boost isn't
silently discarded when a slug/filename fanout RRF-merge runs). Kept
dependency-free (no qdrant_client / asyncpg) so any module can import it.

Two boost forms, because the two score spaces are different:

* ``priority_multiplier`` -- a MULTIPLIER, correct only on the raw COSINE band
  ([0, 1]) used by the dense-only ``search()`` path.
* ``priority_rrf_bonus`` -- an ADDITIVE bonus in RRF units, for the fused
  hybrid/​fanout path. RRF scores live in a tiny compressed band (rank 1 of one
  lane = 1/(60+1) ~= 0.0164, rank 40 ~= 0.0100), so a x2.0 multiplier there is a
  landslide: it lifts a weakly-ranked SRE-KB chunk at rank 40 (0.0100 -> 0.0200)
  above a genuine single-lane #1 (0.0164). An additive bonus sized to a fraction
  of one RRF unit lets authoritative content jump a few ranks / win close calls
  without leaping past a strong multi-lane consensus hit (which scores ~0.033+).
"""
from __future__ import annotations

# SRE-KB content gets a tier tag at index time; tags map to these multipliers.
# Multiplier form -- ONLY valid on the cosine [0, 1] band (dense-only search()).
_PRIORITY_BOOSTS: dict[str, float] = {
    "architecture-canonical": 2.0,  # SRE-KB docs/architecture/*
    "user-correction": 1.8,         # operator-APPROVED correction
    "high": 1.5,                    # other SRE-KB canonical content
}

# RRF unit = the maximum single-lane RRF contribution, 1/(k+1) with k=60 (matches
# rrf._RRF_K and the hybrid_search constant). Additive bonuses are expressed as
# fractions of this unit so they reorder WITHIN the single-lane tier and win
# close calls, but don't out-score a strong multi-lane hit (~2-3 RRF units).
_RRF_UNIT = 1.0 / (60 + 1)
# Bonuses are a FRACTION of one unit, capped below 1.0 so even the top tier can't
# out-score a strong two-lane consensus (~2 units, e.g. 1/61 + 1/62 = 0.033). At
# the prior 1.0*unit, an architecture-canonical single-lane #1 (1/61 + 1/61 =
# 0.0328) edged past that consensus (0.0325) -- violating this module's own
# "won't leap past a strong multi-lane hit" guarantee. The fetched pool is deep
# (candidate_k = max(top_k*8, 50) ~ 50-80), so the bonus must stay a tie-breaker
# among close single-lane hits, not a tier-jumper. 0.75 keeps arch #1 at 0.0287 <
# 0.0325 while still beating every non-priority single-lane hit.
_PRIORITY_RRF_BONUS: dict[str, float] = {
    "architecture-canonical": 0.75 * _RRF_UNIT,
    "user-correction": 0.6 * _RRF_UNIT,
    "high": 0.4 * _RRF_UNIT,
}

# Repo/path rules for deriving the tier when no stored tag is available
# (e.g. the pgvector store has no `priority` payload field).
_HIGH_PRIORITY_REPO_SUBSTR = ("sre-knowledge-base",)
_ARCHITECTURE_PATH_PREFIX = "docs/architecture/"


def priority_multiplier(tag: str | None) -> float:
    """Score MULTIPLIER for a priority tag; 1.0 (no boost) for None/unknown.

    Use ONLY on the raw cosine band. On fused RRF scores use
    ``priority_rrf_bonus`` instead -- a multiplier there steamrolls (see module
    docstring)."""
    if not tag:
        return 1.0
    return _PRIORITY_BOOSTS.get(tag.lower(), 1.0)


def priority_rrf_bonus(tag: str | None) -> float:
    """Additive priority bonus in RRF units; 0.0 for None/unknown. Add this to a
    fused RRF score -- bounded so authoritative content wins close calls without
    leaping past a strong multi-lane consensus hit."""
    if not tag:
        return 0.0
    return _PRIORITY_RRF_BONUS.get(tag.lower(), 0.0)


def chunk_priority(repo: str | None, source_path: str | None) -> str | None:
    """Derive a priority tier from repo/source_path (SRE-KB + architecture
    tiers only; user-correction is stamped at index time, not derivable here)."""
    if not repo or not any(s in repo.lower() for s in _HIGH_PRIORITY_REPO_SUBSTR):
        return None
    if source_path and source_path.startswith(_ARCHITECTURE_PATH_PREFIX):
        return "architecture-canonical"
    return "high"
