"""Offline retrieval eval gate over the shipped `samples/` corpus.

This is the always-on, NO-SECRETS proof that opsrag retrieval works: it
indexes `samples/` into an in-process Qdrant with a local FastEmbed ONNX
embedder, loads the public golden set, runs retrieval for every scored
golden, and asserts aggregate Recall@5 stays above threshold.

The only network it needs is the one-time FastEmbed model fetch (cached
thereafter). No API keys, no running services.

The full answer-quality eval (LLM + judge) lives in
`python -m opsrag.eval run` and needs credentials -- that's the other tier.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# FastEmbed is an optional extra (`uv sync --extra fastembed`). Skip cleanly
# rather than error when it (or qdrant-client) isn't installed.
pytest.importorskip("fastembed")
pytest.importorskip("qdrant_client")

from opsrag.eval.loaders import load_golden  # noqa: E402
from opsrag.eval.retrieval_offline import (  # noqa: E402
    build_offline_index,
    retrieval_scores,
)

_SAMPLES_DIR = Path(__file__).resolve().parents[2] / "samples"

# Observed aggregate Recall@5 over the public golden set was 1.0 on the
# validated recipe (BAAI/bge-small-en-v1.5 @384, in-process Qdrant). Threshold
# set well below that so embedder/chunker noise can't flake the gate while a
# real retrieval regression (a golden's doc dropping out of top-5) still trips.
_RECALL_THRESHOLD = 0.85


@pytest.mark.asyncio
async def test_samples_retrieval_recall_at_5():
    assert _SAMPLES_DIR.is_dir(), f"samples corpus missing: {_SAMPLES_DIR}"

    goldens = load_golden()
    assert goldens, "no goldens loaded"

    embedder, vector_store = await build_offline_index(_SAMPLES_DIR)
    agg = await retrieval_scores(embedder, vector_store, goldens, k=5)

    assert agg["num_scored"] > 0, "no scored goldens (all skipped?)"

    # Surface per-golden detail on failure so a regression names the offender.
    below = [
        p for p in agg["per_golden"] if p["recall_at_k"] < 1.0
    ]
    detail = "\n".join(
        f"  {p['id']} R@5={p['recall_at_k']:.2f} retrieved={p['retrieved']}"
        for p in below
    )
    recall = agg["mean_recall_at_k"]
    print(
        f"\nOffline retrieval eval over samples/: "
        f"mean Recall@5={recall:.4f}, mean MRR={agg['mean_mrr']:.4f}, "
        f"scored={agg['num_scored']}, skipped={agg['num_skipped']}"
    )
    assert recall >= _RECALL_THRESHOLD, (
        f"aggregate Recall@5 {recall:.4f} < threshold {_RECALL_THRESHOLD}\n"
        f"goldens below 1.0:\n{detail}"
    )
