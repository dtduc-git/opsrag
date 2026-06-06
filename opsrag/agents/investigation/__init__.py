"""Hypothesis-driven investigation subgraph.

A LangGraph subgraph that decomposes an alert/question into a tree of
hypotheses, tests each one against targeted retrieval, and synthesizes
the deepest validated chain into a root-cause answer.

Inspired by Datadog Bits AI SRE
(https://www.datadoghq.com/blog/building-bits-ai-sre/) -- the core idea
is to AVOID dumping every tool response into a single summarization
prompt. Each hypothesis fetches ONLY the evidence on its causal path.
"""
from opsrag.agents.investigation.budget import (
    BudgetExceeded,
    check_budget,
    is_duplicate_ancestor,
)
from opsrag.agents.investigation.graph import build_investigation_graph
from opsrag.agents.investigation.state import (
    Citation,
    HypothesisNode,
    InvestigationState,
    NodeStatus,
)

__all__ = [
    "BudgetExceeded",
    "Citation",
    "HypothesisNode",
    "InvestigationState",
    "NodeStatus",
    "build_investigation_graph",
    "check_budget",
    "is_duplicate_ancestor",
]
