"""Markdown report writer for eval runs.

Aggregates per-category and overall metrics, plus per-query detail. Output
lives at tests/eval/reports/<tag>.md so changes are diff-able across runs.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from opsrag.eval.runner import EvalRunResult

REPORT_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "eval" / "reports"

# Minimum acceptable aggregate mean per metric for the CI regression gate.
# These mirror the per-test-case thresholds the metric classes already define,
# applied to the (vacuous-excluded) corpus mean. Metrics absent here aren't
# gated. Tune as the golden set + corpus mature.
DEFAULT_GATE_THRESHOLDS: dict[str, float] = {
    "SourceRecall": 0.80,
    "Precision@5": 0.40,
    # Recall@5 (top-of-list recall the generator actually reads) gates alongside
    # Recall@10: a doc slipping from rank 4->8 tanks Recall@5 while Recall@10
    # stays flat. Without this the exact regression these metrics exist to catch
    # passed green (the metrics were computed + reported but never gated).
    "Recall@5": 0.60,
    "Recall@10": 0.70,
    "NDCG@5": 0.50,
    "MRR": 0.50,
    # Negative-set retrieval restraint: any golden whose cap is exceeded fails.
    "RetrievalRestraint": 1.0,
    "Faithfulness": 0.70,
}


def check_thresholds(
    results: list[EvalRunResult],
    thresholds: dict[str, float] | None = None,
) -> list[str]:
    """Regression gate: return a list of human-readable failure strings for
    metrics whose vacuous-excluded mean falls below its threshold. Empty list
    == pass. A metric with zero scored cases (all vacuous / absent) is reported
    as a failure so a silently-empty category can't pass the gate green."""
    thresholds = thresholds if thresholds is not None else DEFAULT_GATE_THRESHOLDS
    failures: list[str] = []
    for name, floor in thresholds.items():
        scored = [
            m.score
            for r in results
            for m in r.metrics
            if m.name == name and not getattr(m, "skipped", False)
        ]
        if not scored:
            failures.append(f"{name}: no scored cases (all vacuous or metric absent)")
            continue
        mean = statistics.mean(scored)
        if mean < floor:
            failures.append(f"{name}: mean {mean:.3f} < threshold {floor:.2f} (n={len(scored)})")
    return failures


def _agg_metric(results: list[EvalRunResult], name: str) -> tuple[float, float]:
    """Return (mean, pass_rate) for a metric across results.

    Vacuous cases (metric.skipped -- e.g. a ranking metric on a golden with no
    expected/acceptable sources) are EXCLUDED. Counting them as free 1.0s used
    to inflate Recall/MRR/Precision for the cross_source category, which is
    exactly the category meant to test source-blending."""
    scores: list[float] = []
    passes = 0
    for r in results:
        for m in r.metrics:
            if m.name == name and not getattr(m, "skipped", False):
                scores.append(m.score)
                if m.success:
                    passes += 1
    if not scores:
        return 0.0, 0.0
    return statistics.mean(scores), passes / len(scores)


def _skipped_count(results: list[EvalRunResult], name: str) -> int:
    """How many goldens had this metric skipped as vacuous -- surfaced in the
    report so reviewers know to LABEL those goldens with expected_sources."""
    return sum(
        1
        for r in results
        for m in r.metrics
        if m.name == name and getattr(m, "skipped", False)
    )


def _by_category(results: list[EvalRunResult]) -> dict[str, list[EvalRunResult]]:
    out: dict[str, list[EvalRunResult]] = {}
    for r in results:
        out.setdefault(r.category, []).append(r)
    return out


def _by_prompt_variant(results: list[EvalRunResult]) -> dict[str, list[EvalRunResult]]:
    """Group by the system prompt variant the router picked for each
    query. None / missing maps to '(none)' so old runs render cleanly."""
    out: dict[str, list[EvalRunResult]] = {}
    for r in results:
        key = r.prompt_variant or "(none)"
        out.setdefault(key, []).append(r)
    return out


def render(
    results: list[EvalRunResult],
    tag: str,
    config_summary: dict | None = None,
) -> str:
    """Render eval results to a markdown report string."""
    if not results:
        return f"# Eval report -- {tag}\n\nNo results.\n"

    metric_names = sorted({m.name for r in results for m in r.metrics})
    by_cat = _by_category(results)

    total_cost = sum(r.cost_usd for r in results)
    avg_latency = statistics.mean(r.latency_ms for r in results if r.latency_ms > 0) if any(
        r.latency_ms > 0 for r in results
    ) else 0.0

    lines: list[str] = []
    lines.append(f"# Eval report -- `{tag}`")
    lines.append("")
    lines.append(f"**Run at:** {datetime.now(UTC).isoformat(timespec='seconds')}")
    lines.append(f"**Total queries:** {len(results)}")
    lines.append(f"**Total OpsRAG cost:** ${total_cost:.4f}")
    lines.append(f"**Avg latency:** {avg_latency:.0f} ms")
    lines.append("")

    if config_summary:
        lines.append("## Pipeline config")
        lines.append("")
        for k, v in config_summary.items():
            lines.append(f"- **{k}:** `{v}`")
        lines.append("")

    # -- Aggregate per metric --
    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| Metric | Mean score | Pass rate | Scored | Skipped |")
    lines.append("|---|---:|---:|---:|---:|")
    for name in metric_names:
        mean, pass_rate = _agg_metric(results, name)
        skipped = _skipped_count(results, name)
        scored = sum(
            1 for r in results for m in r.metrics
            if m.name == name and not getattr(m, "skipped", False)
        )
        lines.append(
            f"| {name} | {mean:.3f} | {pass_rate * 100:.1f}% | {scored} | {skipped} |"
        )
    lines.append("")
    total_skipped = sum(_skipped_count(results, n) for n in metric_names)
    if total_skipped:
        lines.append(
            f"> {total_skipped} metric evaluations skipped as vacuous (golden "
            f"has no expected/acceptable sources). Label those goldens so the "
            f"ranking metrics actually measure them -- until then they are "
            f"excluded from the means above rather than scored a free 1.0."
        )
        lines.append("")

    # -- Per-category --
    lines.append("## Per-category")
    lines.append("")
    header = ["Category", "Count"] + [f"{n} mean" for n in metric_names]
    align = ["---"] + ["---:"] * (len(header) - 1)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(align) + " |")
    for cat, rows in sorted(by_cat.items()):
        means = [_agg_metric(rows, n)[0] for n in metric_names]
        cells = [cat, str(len(rows))] + [f"{m:.3f}" for m in means]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # -- Per prompt variant --
    # Lets us debug regressions by isolating "is the prompt insufficient
    # for this query class?" from "did the router pick the wrong
    # variant?". A query landing in `general` when it should have been
    # `incident` shows up here as a low-faith outlier in the wrong row.
    by_variant = _by_prompt_variant(results)
    if any(r.prompt_variant for r in results):
        lines.append("## Per prompt variant")
        lines.append("")
        lines.append(
            "Which `generation_system_prompt(query_type)` variant the "
            "router selected for each query. Helps separate "
            "*prompt-not-good-enough* from *router-picked-wrong-prompt*."
        )
        lines.append("")
        v_header = ["Variant", "Count"] + [f"{n} mean" for n in metric_names]
        v_align = ["---"] + ["---:"] * (len(v_header) - 1)
        lines.append("| " + " | ".join(v_header) + " |")
        lines.append("| " + " | ".join(v_align) + " |")
        for variant, rows in sorted(by_variant.items()):
            means = [_agg_metric(rows, n)[0] for n in metric_names]
            cells = [variant, str(len(rows))] + [f"{m:.3f}" for m in means]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # -- Per-query detail --
    lines.append("## Per-query detail")
    lines.append("")
    for r in results:
        lines.append(f"### `{r.id}` -- {r.category}")
        lines.append("")
        lines.append(f"**Query:** {r.query}")
        if r.error:
            lines.append("")
            lines.append(f"**ERROR:** `{r.error}`")
            lines.append("")
            continue
        lines.append("")
        variant_str = f" | **Prompt variant:** `{r.prompt_variant}`" if r.prompt_variant else ""
        lines.append(
            f"**Cost:** ${r.cost_usd:.4f} | **Latency:** {r.latency_ms:.0f} ms | "
            f"**Sources retrieved:** {len(r.sources)}{variant_str}"
        )
        lines.append("")
        lines.append("| Metric | Score | Success | Reason |")
        lines.append("|---|---:|:---:|---|")
        for m in r.metrics:
            mark = "PASS" if m.success else "FAIL"
            reason = (m.reason or "")[:120].replace("|", "\\|")
            lines.append(f"| {m.name} | {m.score:.3f} | {mark} | {reason} |")
        lines.append("")
        lines.append("<details><summary>Answer (truncated)</summary>")
        lines.append("")
        lines.append("```")
        lines.append((r.answer or "(empty)")[:1500])
        lines.append("```")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines)


def write_report(
    results: list[EvalRunResult],
    tag: str,
    config_summary: dict | None = None,
    out_dir: Path | None = None,
) -> Path:
    out_dir = out_dir or REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{tag}.md"
    md_path.write_text(render(results, tag, config_summary))
    # Also dump raw JSON alongside for diff-friendly machine consumption.
    json_path = out_dir / f"{tag}.json"
    json_path.write_text(
        json.dumps(
            {"tag": tag, "results": [asdict(r) for r in results], "config": config_summary or {}},
            indent=2,
            default=str,
        )
    )
    return md_path
