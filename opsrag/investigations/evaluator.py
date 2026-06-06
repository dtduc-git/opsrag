"""Per-hypothesis evaluator -- Pydantic structured output (Flash).

This is the **core fix** of the Option B refactor. Previously hypothesis
verdicts came from regex-parsing the generator's final markdown answer
-- fragile against hedge words, nested parens, multi-line descriptions,
and anti-hallucination refusals.

Now: after each tool round in the reasoner loop, we call this evaluator.
It gets the current hypothesis board + the new evidence pool (tool
results) and returns a structured `HypothesisVerdictBatch` -- one verdict
per hypothesis id, with status from a fixed Literal and an evidence
sentence sourced from the tool results.

Generator's responsibility shrinks: write prose for HUMAN. UI cards no
longer depend on what the generator wrote.

Cost: 1 Flash call per tool round. Typical investigation ~5 rounds ~=
~5 Flash calls ~= ~$0.005. Latency ~2-3s per call (parallel with the
reasoner's next step is possible but not yet implemented).
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

_log = logging.getLogger("opsrag.investigations.evaluator")


HypothesisStatus = Literal["confirmed", "ruled_out", "untested", "open"]


class HypothesisVerdict(BaseModel):
    """One LLM-produced verdict for a single hypothesis id."""

    hypothesis_id: str = Field(
        description="ID of the hypothesis being evaluated, e.g. 'h1', 'h2'. "
        "Must match an id from the input board exactly."
    )
    status: HypothesisStatus = Field(
        description=(
            "'confirmed' -- the evidence directly supports this hypothesis. "
            "'ruled_out' -- the evidence definitively contradicts it. "
            "'untested' -- no tool result speaks to this hypothesis yet. "
            "'open' -- partial evidence, still investigating."
        ),
    )
    evidence: str = Field(
        default="",
        description=(
            "1-2 sentence summary of which tool result supports the status. "
            "Quote a value (pod name, error msg, query latency) when available. "
            "Empty string when status='untested'."
        ),
        max_length=400,
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="0.0 = no signal, 0.5 = ambiguous, 1.0 = ironclad.",
    )


class HypothesisVerdictBatch(BaseModel):
    """One verdict per hypothesis. Length must equal the input board."""

    verdicts: list[HypothesisVerdict]


_SYSTEM_EVALUATOR = """\
You are an SRE evidence-grader. Given a list of competing root-cause
hypotheses and a pool of tool results gathered so far, return a verdict
per hypothesis.

Rules:
  - One verdict per hypothesis id, in the same order as the input board.
  - 'confirmed' requires DIRECT POSITIVE evidence: a log line, metric
    value, or k8s state whose CONTENT names the hypothesis's failure
    mode (an error message, a non-zero counter, a specific exception).
  - 'ruled_out' requires evidence that CONTRADICTS the hypothesis.
  - 'untested' = no tool result speaks to this hypothesis YET. Prefer
    untested over open when uncertain -- the user reads 'untested' as
    "still need to call a tool for this one".
  - 'open' is the initial state; only emit it if the evidence is
    partial (some support, no conclusion).

**HARD RULE -- ABSENCE IS NOT CONFIRMATION**
A tool returning 0 results / empty list / "not found" / "no pods
matched" is NOT evidence that something is broken. It usually means
the tool's filter args were wrong (wrong label selector, wrong
namespace, wrong project, wrong time window). Examples of what
ABSENCE EVIDENCE CANNOT CONFIRM:
  - k8s_list_pods with label_selector="app=foo" returning 0 pods does
    NOT confirm "foo is not running" -- the label may be wrong. The
    real labels in this deployment are often
    `app.kubernetes.io/name=<svc>` or the namespace itself is named
    after the workload.
  - prometheus_query returning [] does NOT confirm "metric never
    fires" -- the query may be using the wrong metric name or filter.
  - gitlab_list_deployments returning [] does NOT confirm "no deploys
    happened" -- the project_id may be wrong.
When you see absence evidence, mark the hypothesis 'untested' and put
the recommendation in `evidence`: "k8s_list_pods returned 0 with
label_selector=<x>; rerun without label or with namespace=<y>".

**HARD RULE -- REQUIRE TWO INDEPENDENT SIGNALS BEFORE CONFIRMING**
Pick 'confirmed' only when EITHER:
  (a) you see a specific error/value from at least TWO different
      tools (logs + metrics, or k8s state + logs), OR
  (b) one tool returned a smoking-gun log line that names the exact
      failure mode (e.g. "violates not-null constraint" for a DB
      schema bug, "OOMKilled" for memory pressure).

Evidence must be a SENTENCE QUOTING (or paraphrasing) the tool result.
Do NOT invent. confidence in [0,1] reflects strength: absence-only or
single-source = at most 0.4; corroborated signals = 0.7+.

Output: a JSON object matching the HypothesisVerdictBatch schema. No prose."""


async def evaluate_hypotheses(
    *,
    llm,  # Vertex Flash LLM
    hypotheses: list[dict[str, Any]],
    evidence_pool: str,
    incident_target: str | None = None,
    prior_verdicts: list[dict[str, Any]] | None = None,
) -> HypothesisVerdictBatch:
    """Single LLM call. Returns a verdict per hypothesis.

    Args:
        llm: Vertex LLM client with `generate_structured` method.
        hypotheses: List of {id, text, discriminating_tools} dicts (the board).
        evidence_pool: Concatenated tool results, formatted for context.
            Caller is responsible for keeping this under ~8K tokens.
        incident_target: Service/component the alert names (for context).
        prior_verdicts: Previous round's verdicts so the LLM can keep
            confirmed/ruled_out stable instead of re-evaluating each round.

    Returns:
        HypothesisVerdictBatch with `len(verdicts) == len(hypotheses)`.
        Falls back to all-untested if the LLM call fails -- never raises.
    """
    if not hypotheses:
        return HypothesisVerdictBatch(verdicts=[])

    # -- Build the user message --------------------------------------
    lines: list[str] = []
    lines.append(f"INCIDENT TARGET: {incident_target or '(unspecified)'}")
    lines.append("")
    lines.append("=== HYPOTHESIS BOARD ===")
    for h in hypotheses:
        hid = h.get("id") or "?"
        text = h.get("text") or ""
        tools = h.get("discriminating_tools") or []
        tools_str = ", ".join(tools) if tools else "(none specified)"
        lines.append(f"{hid}: {text}")
        lines.append(f"    Discriminating tools: {tools_str}")
    lines.append("")

    if prior_verdicts:
        lines.append("=== PRIOR ROUND VERDICTS (keep stable when evidence unchanged) ===")
        for pv in prior_verdicts:
            lines.append(
                f"  {pv.get('hypothesis_id','?')}: {pv.get('status','open')} "
                f"-- {pv.get('evidence','')[:200]}"
            )
        lines.append("")

    lines.append("=== EVIDENCE POOL (tool results so far) ===")
    lines.append(evidence_pool or "(no tool results yet)")
    lines.append("")
    lines.append(
        "Return one verdict per hypothesis id, in the same order. "
        "Statuses: confirmed | ruled_out | untested | open."
    )

    user_msg = "\n".join(lines)
    t0 = time.perf_counter()
    try:
        batch = await llm.generate_structured(
            messages=[{"role": "user", "content": user_msg}],
            schema=HypothesisVerdictBatch,
            system_prompt=_SYSTEM_EVALUATOR,
            purpose="hypothesis_evaluator",
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning(
            "evaluator LLM call failed (%s) -- returning all-untested fallback", exc,
        )
        return HypothesisVerdictBatch(
            verdicts=[
                HypothesisVerdict(
                    hypothesis_id=str(h.get("id") or f"h{i+1}"),
                    status="untested",
                    evidence="",
                    confidence=0.0,
                )
                for i, h in enumerate(hypotheses)
            ],
        )

    elapsed = int((time.perf_counter() - t0) * 1000)
    _log.info(
        "evaluator: %d verdicts in %dms (target=%s)",
        len(batch.verdicts), elapsed, incident_target,
    )

    # Defensive: if the LLM omitted a hypothesis, pad with untested so
    # the UI always sees one verdict per board slot.
    seen_ids = {v.hypothesis_id for v in batch.verdicts}
    for h in hypotheses:
        hid = str(h.get("id") or "")
        if hid and hid not in seen_ids:
            batch.verdicts.append(
                HypothesisVerdict(
                    hypothesis_id=hid, status="untested",
                    evidence="", confidence=0.0,
                )
            )
    return batch


def format_evidence_pool(tool_history: list[dict[str, Any]]) -> str:
    """Squash the tool-call history into a readable block for the
    evaluator's user message. Truncates each tool result to ~800 chars
    so a busy investigation stays under the Flash token budget."""
    if not tool_history:
        return ""
    blocks: list[str] = []
    for i, item in enumerate(tool_history, 1):
        name = item.get("name") or "?"
        args = item.get("args") or {}
        result = item.get("result") or item.get("response") or ""
        if isinstance(result, dict):
            result = json.dumps(result)
        result_str = str(result)
        if len(result_str) > 800:
            result_str = result_str[:800] + " ...[truncated]"
        args_str = json.dumps(args) if args else ""
        if len(args_str) > 200:
            args_str = args_str[:200] + " ..."
        blocks.append(
            f"#{i} [{name}] args={args_str}\n    result: {result_str}"
        )
    return "\n\n".join(blocks)
