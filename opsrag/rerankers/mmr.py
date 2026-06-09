"""Maximal Marginal Relevance (MMR) diversity re-ordering.

Applied AFTER the cross-encoder rerank to break up near-duplicate
candidates (e.g. five config variants of the same values file) that the
reranker happily scores almost identically and stacks at the top. MMR
trades a little relevance for diversity so the top-k spans distinct
documents instead of clones.

Greedy MMR selection: at each step pick the candidate that maximises

    score(c) = lambda * relevance(c) - (1 - lambda) * max_{s in selected} sim(c, s)

`lambda == 1.0` reduces to pure relevance order (no diversity penalty);
`lambda == 0.0` maximises diversity. We deliberately treat
``diversity in (None, 0.0)`` and ``diversity >= 1.0`` as DISABLED and
return the input order untouched, so the default (flag off) is a
byte-for-byte pass-through and existing behaviour is unchanged.

The similarity term defaults to token-set Jaccard over chunk content --
zero extra dependencies and no embeddings required at the rerank stage,
which is exactly where near-duplicate config files cluster. Callers that
have embeddings handy can inject a cosine ``similarity_fn`` instead.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _normalize_relevance(relevance: Sequence[float]) -> list[float]:
    """Sanitize + min-max normalize relevance into [0, 1].

    Two jobs, both required for a provider-agnostic ``lambda``:

    * **Sanitize** NaN/-inf to the observed floor and +inf to the ceiling.
      A raw NaN poisons the MMR objective (``NaN > best_score`` is always
      False), so ``best_idx`` would stay ``None`` and ``items[None]`` would
      raise. Treating a non-finite score as the floor keeps it a valid (if
      worst-ranked) candidate instead of crashing the whole rerank.
    * **Min-max normalize** to [0, 1] so the diversity penalty weight means
      the same thing regardless of the reranker's native scale (Cohere's
      compressed ~0.02-0.4 vs FastEmbed's sigmoid 0-1). Without this, the
      same ``diversity`` value barely perturbs Cohere order but dominates
      FastEmbed order.
    """
    finite = [r for r in relevance if math.isfinite(r)]
    if finite:
        lo, hi = min(finite), max(finite)
    else:
        # Every score is non-finite -- normalize to a flat 0.0 so MMR
        # degrades to pure-diversity selection rather than exploding.
        lo = hi = 0.0
    span = hi - lo

    out: list[float] = []
    for r in relevance:
        if math.isnan(r) or r == float("-inf"):
            r = lo  # treat as the worst observed relevance, not a crash
        elif r == float("inf"):
            r = hi  # treat as the best observed relevance
        out.append(0.0 if span == 0.0 else (r - lo) / span)
    return out


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall((text or "").lower()))


def jaccard_similarity(a: str, b: str) -> float:
    """Token-set Jaccard in [0, 1]. 1.0 == identical token sets."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def mmr_reorder(
    items: Sequence[T],
    relevance: Sequence[float],
    *,
    diversity: float | None,
    text_of: Callable[[T], str],
    similarity_fn: Callable[[str, str], float] = jaccard_similarity,
    top_k: int | None = None,
) -> list[T]:
    """Re-order ``items`` by Maximal Marginal Relevance.

    Args:
        items: candidates, already in relevance (rerank) order.
        relevance: relevance score per item, parallel to ``items``.
        diversity: the MMR penalty weight ``(1 - lambda)`` in [0, 1].
            ``0.0`` / ``None`` -> DISABLED, returns ``items[:top_k]``
            unchanged. Values >= 1.0 are clamped to 1.0 (pure diversity).
        text_of: extracts the text used for the similarity term.
        similarity_fn: pairwise similarity in [0, 1]; defaults to
            token-set Jaccard over the extracted text.
        top_k: how many items to return. None -> all of them.

    Returns:
        A re-ordered list. When disabled, this is ``list(items)[:top_k]``
        (a pass-through), so default callers see no behavioural change.
    """
    n = len(items)
    limit = n if top_k is None else min(top_k, n)

    # Disabled / no-op fast path -- pass-through, identical to no MMR.
    if not diversity or diversity <= 0.0 or n <= 1:
        return list(items)[:limit]

    # lambda = 1 - diversity (the relevance weight). Clamp diversity to
    # [0, 1] so a stray config value can't invert the objective.
    div = min(1.0, float(diversity))
    lam = 1.0 - div

    # Sanitize non-finite scores + min-max normalize to [0, 1] so the
    # diversity weight is provider-agnostic and a NaN can never poison the
    # MMR objective (which would leave best_idx == None -> items[None]).
    rel = _normalize_relevance(relevance)

    texts = [text_of(it) for it in items]
    remaining = list(range(n))
    # Seed with the single most relevant item -- MMR's first pick is
    # always pure relevance (no `selected` set to diversify against yet).
    # Use normalized (finite) scores so a NaN can't make max() pick garbage.
    first = max(remaining, key=lambda i: rel[i])
    selected: list[int] = [first]
    remaining.remove(first)

    while remaining and len(selected) < limit:
        # Seed with the first remaining index (not None) so best_idx is
        # always a valid index even in degenerate score landscapes.
        best_idx = remaining[0]
        best_score = float("-inf")
        for i in remaining:
            max_sim = max(similarity_fn(texts[i], texts[s]) for s in selected)
            score = lam * rel[i] - div * max_sim
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [items[i] for i in selected]
