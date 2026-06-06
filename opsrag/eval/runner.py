"""Eval runner -- load golden set, query OpsRAG, score, return structured results.

Used both by:
  - CLI (`python -m opsrag.eval run --tag baseline`) -> tagged baseline reports.
  - Pytest (`pytest opsrag/eval/`) -> CI assertions per metric.

The runner returns a list of `EvalRunResult` dicts; report.py knows how
to render those into markdown.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field

import httpx

from opsrag.eval.adapters.vertex_judge import VertexGeminiJudge
from opsrag.eval.loaders import GoldenQuery, load_golden, to_llm_test_case
from opsrag.eval.metrics import (
    FaithfulnessJudgeMetric,
    MRRMetric,
    MustContainMetric,
    MustNotContainMetric,
    RankPrecisionAtKMetric,
    RankRecallAtKMetric,
    SourceRecallMetric,
)
from opsrag.eval.usage_hook import get_usage_total

_log = logging.getLogger("opsrag.eval.runner")


@dataclass
class MetricResult:
    name: str
    score: float
    success: bool
    reason: str = ""
    error: str | None = None
    # True when the metric was vacuous for this golden (e.g. a ranking metric
    # on a golden with no expected/acceptable sources). Excluded from aggregate
    # means so free 1.0s / penalising 0.0s don't distort the headline numbers.
    skipped: bool = False


@dataclass
class EvalRunResult:
    id: str
    category: str
    query: str
    answer: str
    sources: list[str] = field(default_factory=list)
    metrics: list[MetricResult] = field(default_factory=list)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None
    # Which generation prompt variant the router picked for this query.
    # The agent's router picks query_type in {incident, howto, general,
    # ...} which maps 1:1 to one of GENERATE_SYSTEM_INCIDENT /
    # GENERATE_SYSTEM_HOWTO / GENERATE_SYSTEM_DEFAULT in agent/prompts.py.
    # Logging this lets us debug regressions: when a golden fails, we
    # know whether the router picked the wrong variant vs the prompt
    # itself was insufficient. None when the API didn't return one
    # (older builds, or the query bypassed the router).
    prompt_variant: str | None = None

    def metric_score(self, name: str) -> float | None:
        for m in self.metrics:
            if m.name == name:
                return m.score
        return None


def _query_opsrag(opsrag_url: str, query: str) -> dict:
    resp = httpx.post(
        f"{opsrag_url}/query",
        json={"query": query, "stream": False},
        timeout=180.0,
    )
    resp.raise_for_status()
    return resp.json()


def _run_metrics(test_case, judge: VertexGeminiJudge) -> list[MetricResult]:
    """Run the standard OpsRAG metric suite on one test case.

    Order matters only for human-readable report grouping:
      1. Cheap deterministic metrics first (recall, must-contain).
      2. LLM-judged faithfulness last (slowest, costs Pro tokens).
    """
    metrics = [
        # Deterministic, set-based -- keep at top so the cheap signal
        # comes first in the report.
        SourceRecallMetric(),
        RankPrecisionAtKMetric(k=5),
        RankRecallAtKMetric(k=10),
        MRRMetric(),
        MustContainMetric(),
        MustNotContainMetric(),
        # LLM-judged, slowest, costs Pro tokens -- last.
        FaithfulnessJudgeMetric(judge=judge),
    ]
    out: list[MetricResult] = []
    for m in metrics:
        try:
            score = m.measure(test_case)
            out.append(MetricResult(
                name=m.__name__,
                score=float(score),
                success=bool(m.is_successful()),
                reason=m.reason,
                error=getattr(m, "error", None),
                skipped=bool(getattr(m, "skipped", False)),
            ))
        except Exception as exc:
            _log.warning("metric %s raised: %s", getattr(m, "__name__", "?"), exc)
            out.append(MetricResult(
                name=getattr(m, "__name__", "unknown"),
                score=0.0,
                success=False,
                reason="exception during measure",
                error=str(exc),
            ))
    return out


def run_golden(
    opsrag_url: str,
    judge: VertexGeminiJudge | None = None,
    category: str | None = None,
    queries: list[GoldenQuery] | None = None,
) -> list[EvalRunResult]:
    """Execute the golden-set against a running OpsRAG instance.

    Each golden query -> 1 OpsRAG /query call + N metric calls (1-3 of those
    hit Vertex Gemini Pro for faithfulness). Returns full results so the
    caller can either persist or assert.
    """
    judge = judge or VertexGeminiJudge()
    golden = queries if queries is not None else load_golden(category=category)
    out: list[EvalRunResult] = []

    for g in golden:
        cost_before = get_usage_total()
        t0 = time.perf_counter()
        try:
            resp = _query_opsrag(opsrag_url, g.query)
        except Exception as exc:
            _log.exception("query failed for %s", g.id)
            out.append(EvalRunResult(
                id=g.id,
                category=g.category,
                query=g.query,
                answer="",
                sources=[],
                error=f"opsrag query failed: {exc}",
            ))
            continue
        latency_ms = (time.perf_counter() - t0) * 1000
        cost_delta = max(0.0, get_usage_total() - cost_before)

        # Prefer chunk content over bare paths for faithfulness scoring --
        # the judge can verify factual claims only against actual content.
        # Pass raw source paths separately for SourceRecall set-intersection.
        sources_paths: list[str] = resp.get("sources", [])
        sources_content = resp.get("sources_content") or []
        if sources_content:
            retrieval_context = [
                f"[{c.get('source','?')}]\n{c.get('content','')}"
                for c in sources_content
            ]
        else:
            retrieval_context = sources_paths
        test_case = to_llm_test_case(
            g,
            actual_output=resp.get("answer", ""),
            retrieval_context=retrieval_context,
            retrieved_sources=sources_paths,
        )
        metrics = _run_metrics(test_case, judge)

        out.append(EvalRunResult(
            id=g.id,
            category=g.category,
            query=g.query,
            answer=resp.get("answer", ""),
            sources=resp.get("sources", []),
            metrics=metrics,
            cost_usd=cost_delta,
            latency_ms=latency_ms,
            # `query_type` is the router's classification; the prompt
            # selector (agent/prompts.py:generation_system_prompt)
            # consumes it to pick incident/howto/default. Capturing it
            # here surfaces routing+prompt mismatches in the report.
            prompt_variant=resp.get("query_type"),
        ))
        _log.info(
            "%s [%s variant=%s] cost=$%.4f latency=%.0fms %s",
            g.id, g.category, resp.get("query_type") or "-",
            cost_delta, latency_ms,
            " ".join(f"{m.name}={m.score:.2f}" for m in metrics),
        )

    return out


def to_dicts(results: list[EvalRunResult]) -> list[dict]:
    return [asdict(r) for r in results]
