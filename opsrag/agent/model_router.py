"""Phase 03 Pillar 3 -- hybrid Flash/Pro reasoning routing.

A `ModelRouter` holds the two configured Vertex Gemini clients
(`flash` for default traffic, `pro` for escalation) and a cheap
heuristic that picks per-query. Queries that need multi-step
reasoning, cross-source synthesis, or root-cause analysis route to
Pro; everything else stays on Flash.

Heuristic-only: no extra LLM round-trip. We accept some misclassification
in exchange for zero added latency / cost. Real-user feedback in
Phase 05 will tell us where the heuristic falls short.

Estimated steady-state cost at SRE-team scale:
  - 80% Flash @ ~$0.003/query  -> $1-3/mo
  - 20% Pro   @ ~$0.025/query  -> $5-15/mo
  Total: $6-18/mo (within the $1-15 roadmap target).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

from opsrag.interfaces.llm import LLMProvider

_log = logging.getLogger("opsrag.agent.model_router")

ModelTier = Literal["flash", "pro"]


# Hint patterns that indicate multi-step / cross-source / root-cause work.
# Word-boundary regex; case-insensitive. Order matters only for log readability.
_PRO_PATTERNS = [
    (re.compile(r"\bwhy\b", re.I),                    "why"),
    (re.compile(r"\broot[\s-]?cause\b", re.I),        "root_cause"),
    (re.compile(r"\bwhat\s+caused?\b", re.I),         "what_caused"),
    (re.compile(r"\bcompare\b", re.I),                "compare"),
    (re.compile(r"\bdifference[s]?\s+between\b", re.I), "difference_between"),
    (re.compile(r"\b(across|throughout)\s+(all\s+)?(repos?|services?|projects?|environments?)", re.I), "across_X"),
    (re.compile(r"\b(troubleshoot|debug|investigate)\b", re.I), "investigate"),
    (re.compile(r"\b(synthesi[sz]e|correlate)\b", re.I), "synthesize"),
    (re.compile(r"\bhow\s+(does|do)\b.*\bwork\b", re.I), "how_does_X_work"),
    (re.compile(r"\b(impact|blast\s*radius)\b", re.I),  "impact"),
    # Diagram / visualization requests -- Pro is much more reliable at
    # emitting structured `diagram-json` blocks. Flash often falls back
    # to prose with numbered lists even when the prompt demands JSON.
    # Observed 2026-05-23 with an ingress query on Flash: reasoner
    # looped 10x on code_read_file(prod.yaml) and the generator
    # produced an empty answer. Routing to Pro converges in 2-3 hops
    # and emits a valid diagram block.
    (re.compile(r"\b(diagram|architecture)\b", re.I), "diagram_request"),
    (re.compile(r"\b(draw|sketch|visuali[sz]e)\s+", re.I), "draw_visualize"),
    (re.compile(r"\bshow\s+(me\s+)?(a\s+|the\s+)?(flow|diagram|architecture|components?|topology)\b", re.I), "show_diagram"),
    # Chart / graph / plot requests -- Pro is likewise more reliable at
    # producing a clean `render_chart` spec (correct type + series) than Flash,
    # which tends to fall back to a markdown table.
    (re.compile(r"\b(chart|graph|plot|trend)\b", re.I), "chart_request"),
]

_LONG_QUERY_TOKEN_THRESHOLD = 30  # >30 tokens -> likely multi-step
_MULTI_QUESTION_THRESHOLD = 2     # 2+ "?" marks -> multi-question


@dataclass
class RoutingDecision:
    tier: ModelTier
    reason: str
    matched_patterns: list[str]


def classify(query: str) -> RoutingDecision:
    """Return Flash / Pro routing for a query. Pure function -- no I/O."""
    if not query or not query.strip():
        return RoutingDecision(tier="flash", reason="empty_query", matched_patterns=[])

    matched: list[str] = []
    for pat, label in _PRO_PATTERNS:
        if pat.search(query):
            matched.append(label)

    # Multi-question -- count standalone '?' separators (rough proxy).
    q_count = query.count("?")
    if q_count >= _MULTI_QUESTION_THRESHOLD:
        matched.append(f"multi_question_{q_count}")

    # Token-length heuristic.
    token_count = len(re.findall(r"\w+", query))
    if token_count > _LONG_QUERY_TOKEN_THRESHOLD:
        matched.append(f"long_query_{token_count}t")

    if matched:
        return RoutingDecision(
            tier="pro", reason="patterns_matched", matched_patterns=matched,
        )
    return RoutingDecision(
        tier="flash", reason="default_simple_lookup", matched_patterns=[],
    )


class ModelRouter:
    """Holds Flash and Pro LLM clients and routes per-query.

    When `pro_llm` is None, all routes return `flash_llm` -- a degenerate
    setup that still emits classification audit data, useful for shadow
    testing before enabling Pro.
    """

    def __init__(self, flash_llm: LLMProvider, pro_llm: LLMProvider | None = None):
        self.flash_llm = flash_llm
        self.pro_llm = pro_llm

    @property
    def has_pro(self) -> bool:
        return self.pro_llm is not None

    def pick(self, query: str) -> tuple[LLMProvider, RoutingDecision]:
        decision = classify(query)
        if decision.tier == "pro" and self.pro_llm is not None:
            chosen = self.pro_llm
        else:
            # Pro requested but unconfigured -- fall back to Flash and note it.
            if decision.tier == "pro":
                decision = RoutingDecision(
                    tier="flash",
                    reason=f"pro_unavailable|{decision.reason}",
                    matched_patterns=decision.matched_patterns,
                )
            chosen = self.flash_llm
        _log.info(
            "model_router pick=%s reason=%s matched=%s model=%s",
            decision.tier, decision.reason, decision.matched_patterns, chosen.model_name,
        )
        return chosen, decision
