"""Day 1 smoke test -- validates the full eval pipeline end-to-end.

Verifies:
1. VertexGeminiJudge structured-output adapter actually parses JSON via
   Vertex's response_schema mode (not regex fallback).
2. Loading golden YAMLs into LLMTestCase works.
3. Hitting OpsRAG /query, building retrieval context, running a metric.
4. Determinism: 3 reruns of the same query -> faithfulness sigma < 0.05.

Real golden queries (27 across 5 categories) come Day 3 after manual
curation against actual indexed content.
"""
from __future__ import annotations

import logging
import statistics
from typing import Any

import httpx
import pytest

# The eval harness depends on the optional `eval` extra (deepeval). Skip the
# whole module cleanly when it is absent so the default test run and CI
# collection do not error on import.
pytest.importorskip("deepeval", reason="eval extra (deepeval) not installed")

from opsrag.eval.adapters.vertex_judge import VertexGeminiJudge
from opsrag.eval.loaders import GoldenQuery, load_golden, to_llm_test_case
from opsrag.eval.metrics import (
    MustContainMetric,
    MustNotContainMetric,
    SourceRecallMetric,
)

_log = logging.getLogger("opsrag.eval.smoke")


def _query_opsrag(opsrag_url: str, query: str) -> dict[str, Any]:
    """Call OpsRAG /query and return parsed JSON."""
    resp = httpx.post(
        f"{opsrag_url}/query",
        json={"query": query, "stream": False},
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()


def test_judge_structured_output(judge: VertexGeminiJudge) -> None:
    """Verify VertexGeminiJudge.generate_schema returns a parsed Pydantic instance.

    This is the highest-risk part of Day 1 -- if Vertex SDK's response_schema
    field name or shape changed across versions, this fails fast.
    """
    from pydantic import BaseModel, Field

    class JudgeResponse(BaseModel):
        score: float = Field(..., ge=0.0, le=1.0)
        reasoning: str

    prompt = (
        "Rate the relevance of this answer to the question on a 0.0-1.0 scale.\n"
        "Question: What is the capital of France?\n"
        "Answer: Paris is the capital of France.\n"
        "Return JSON {score, reasoning}."
    )
    result = judge.generate_schema(prompt, JudgeResponse)
    assert isinstance(result, JudgeResponse)
    assert 0.0 <= result.score <= 1.0
    assert result.score > 0.7, f"expected high score, got {result.score}"
    _log.info("judge structured output OK: score=%.2f", result.score)


def test_golden_loader() -> None:
    """Smoke YAML loads into GoldenQuery instances."""
    queries = load_golden(category="_smoke")
    assert len(queries) == 3
    ids = {q.id for q in queries}
    assert ids == {"smoke_001", "smoke_002", "smoke_003"}
    # Spot-check a known field
    smoke_001 = next(q for q in queries if q.id == "smoke_001")
    assert "gitlab-ci" in " ".join(smoke_001.must_contain).lower()


def test_opsrag_query_roundtrip(opsrag_url: str) -> None:
    """OpsRAG /query reachable and returns the expected shape."""
    result = _query_opsrag(opsrag_url, "What is in gitops-pipeline-templates?")
    assert "answer" in result
    assert "sources" in result
    assert isinstance(result["sources"], list)
    assert len(result["answer"]) > 0
    _log.info(
        "opsrag query OK: answer_len=%d, sources=%d",
        len(result["answer"]), len(result["sources"]),
    )


def test_smoke_test_case_construction(opsrag_url: str) -> None:
    """End-to-end: golden query -> opsrag -> LLMTestCase ready for metrics."""
    g: GoldenQuery = next(q for q in load_golden(category="_smoke") if q.id == "smoke_001")
    result = _query_opsrag(opsrag_url, g.query)
    test_case = to_llm_test_case(
        g,
        actual_output=result["answer"],
        retrieval_context=result.get("sources", []),
    )
    assert test_case.input == g.query
    assert test_case.actual_output == result["answer"]
    assert test_case.retrieval_context is not None
    _log.info("LLMTestCase constructed OK for %s", g.id)


def test_metrics_deterministic_run(opsrag_url: str) -> None:
    """End-to-end: run smoke goldens through deterministic metrics (no judge).

    Verifies SourceRecall + MustContain + MustNotContain wiring without
    paying for judge calls. Faithfulness is tested separately because it
    costs Pro tokens -- the runner.py path is what eval CLI uses.
    """
    queries = load_golden(category="_smoke")
    assert len(queries) == 3
    for g in queries:
        result = _query_opsrag(opsrag_url, g.query)
        tc = to_llm_test_case(
            g, actual_output=result["answer"], retrieval_context=result.get("sources", []),
        )
        from opsrag.eval.metrics import (
            MRRMetric,
            RankPrecisionAtKMetric,
            RankRecallAtKMetric,
        )

        for metric_cls in (
            SourceRecallMetric,
            RankPrecisionAtKMetric,
            RankRecallAtKMetric,
            MRRMetric,
            MustContainMetric,
            MustNotContainMetric,
        ):
            m = metric_cls()
            score = m.measure(tc)
            assert 0.0 <= score <= 1.0, f"{m.__name__} returned out-of-range score for {g.id}"
            _log.info("  %s [%s] %s = %.3f (%s)", g.id, g.category, m.__name__, score, m.reason)


def test_determinism_judge_score(judge: VertexGeminiJudge) -> None:
    """Day 1 A1: run the same judge prompt 3x and measure sigma.

    If sigma > 0.05, faithfulness scores are too noisy for single-run baselines --
    we'd need to run each baseline query multiple times and average.
    Result documented in docs/eval-methodology.md.
    """
    from pydantic import BaseModel, Field

    class JudgeResponse(BaseModel):
        score: float = Field(..., ge=0.0, le=1.0)
        reasoning: str

    prompt = (
        "Rate how well this answer is grounded in the provided context.\n"
        "Context: 'The pipeline file generic-pipeline.yaml defines stages: "
        "utils, build, test, delivery, cleanup.'\n"
        "Answer: 'The pipeline includes utils, build, test, delivery, and "
        "cleanup stages.'\n"
        "Return JSON {score (0=hallucinated, 1=fully grounded), reasoning}."
    )
    scores = []
    for i in range(3):
        result = judge.generate_schema(prompt, JudgeResponse)
        scores.append(result.score)
        _log.info("rerun %d/3: score=%.3f", i + 1, result.score)

    sigma = statistics.stdev(scores) if len(scores) > 1 else 0.0
    mean = statistics.mean(scores)
    _log.info("determinism: mean=%.3f sigma=%.3f scores=%s", mean, sigma, scores)
    # Soft assertion -- high sigma is a signal, not necessarily a failure. Log
    # to docs/eval-methodology.md after the run if exceeded.
    if sigma > 0.05:
        pytest.fail(
            f"judge non-deterministic: sigma={sigma:.3f} > 0.05 -- "
            f"baseline runs need n=3 averaging. Scores: {scores}"
        )
