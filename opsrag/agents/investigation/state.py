"""Pydantic state schema for the hypothesis-driven investigation agent.

State is intentionally one big Pydantic model so the LangGraph
checkpointer can serialize/restore the whole tree mid-investigation.

Tree shape rationale: hypotheses form a tree (root cause -> sub-causes
-> deeper sub-causes). We store the tree by id-indexed flat dict and
keep child pointers as id lists -- this avoids LangGraph reducer
ambiguity around merging recursive Pydantic models and makes the DFS
cursor (`current_node_id`) cheap to advance.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

NodeStatus = Literal["pending", "validated", "invalidated", "inconclusive"]
TerminationReason = Literal[
    "circuit_breaker_max_nodes",
    "circuit_breaker_max_tool_calls",
    "circuit_breaker_max_duration",
    "circuit_breaker_max_tokens",
    "duplicate_ancestor",
    "duplicate_sibling",
    "max_depth_reached",
    "below_recurse_threshold",
]


class Citation(BaseModel):
    """A single piece of evidence attached to a hypothesis node.

    Mirrors the (source_id, chunk_id) tuple required by the eval harness
    so faithfulness can be measured per-hypothesis, not only on the
    final answer.
    """

    source_id: str = Field(
        description="Origin system + path. E.g. 'confluence:DEVOPS/runbook-kafka'."
    )
    chunk_id: str = Field(description="Vector-store chunk identifier.")
    snippet: str = Field(default="", description="Up to ~280 chars of the cited text.")
    score: float = Field(default=0.0, description="Retrieval similarity score.")
    repo: str = Field(default="", description="Repo or space the chunk came from.")


class HypothesisNode(BaseModel):
    """A single hypothesis under investigation.

    The whole tree is stored flat in `InvestigationState.nodes_by_id`;
    `children` here holds child IDs only, not nested objects, to keep
    Pydantic serialization predictable inside LangGraph checkpoints.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    statement: str = Field(description="One-sentence hypothesis.")
    status: NodeStatus = "pending"
    evidence: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    depth: int = 0
    parent_id: str | None = None
    children: list[str] = Field(default_factory=list)
    # Best-first bookkeeping: set once decide_next has chosen this
    # validated node for expansion (or finalized it as non-recursable),
    # so the frontier never re-picks the same node. A validated node that
    # produced zero (all-duplicate) children would otherwise be selected
    # forever.
    expanded: bool = False
    termination_reason: TerminationReason | None = None
    judge_rationale: str = Field(
        default="",
        description="LLM judge's one-line explanation for the status decision.",
    )
    # Provenance of this hypothesis. The investigation UI shows a
    # runbook badge on "runbook"-sourced nodes so operators can see
    # when the agent's hypothesis came straight from their own runbook.
    hypothesis_source: Literal["llm", "runbook", "past_investigation"] = "llm"


class BudgetState(BaseModel):
    """Running counters checked before every new node is spawned/tested."""

    model_config = ConfigDict(extra="forbid")

    total_nodes: int = 0
    total_tool_calls: int = 0
    total_llm_tokens: int = 0
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    circuit_breakers_hit: list[str] = Field(default_factory=list)

    # Per-purpose tool-call breakdown for cost observability dashboards.
    retrieval_calls: int = 0
    llm_query_gen_calls: int = 0
    llm_judge_calls: int = 0
    llm_synth_calls: int = 0
    # P0-B: live-telemetry tool dispatch (datadog/rootly/code/...) and the
    # LLM call that selects which tool(s) test a hypothesis.
    tool_dispatch_calls: int = 0
    llm_tool_select_calls: int = 0
    # Embeddings done purely for sibling/ancestor dedup. Tracked here for
    # observability but deliberately NOT counted in total_tool_calls --
    # they're cheap, bounded by the node cap, and must not consume the
    # retrieval/LLM tool-call budget (the 300-call breaker).
    embed_dedup_calls: int = 0

    # Per-purpose token breakdown so we can attribute spend.
    input_tokens: int = 0
    output_tokens: int = 0

    def elapsed_seconds(self) -> float:
        return (datetime.now(UTC) - self.started_at).total_seconds()


class AlertContext(BaseModel):
    """Input to the investigation: an alert/question + optional anchors."""

    model_config = ConfigDict(extra="forbid")

    investigation_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    alert_text: str
    runbook_urls: list[str] = Field(default_factory=list)
    service_hint: str | None = None
    namespace_hint: str | None = None
    env_hint: str | None = None


class TraceEvent(BaseModel):
    """One entry in the agent_trace stream -- what the agent did and why."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    event_type: str  # bootstrap | hypothesis_gen | judge | recurse | terminate | synth
    node_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class InvestigationState(BaseModel):
    """Full subgraph state -- single Pydantic model so LangGraph
    checkpointing round-trips cleanly."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # -- Inputs -----------------------------------------------------
    alert_context: AlertContext

    # -- Bootstrap phase --------------------------------------------
    bootstrap_findings: list[str] = Field(default_factory=list)
    bootstrap_citations: list[Citation] = Field(default_factory=list)

    # Past similar investigations injected into the hypothesis-gen
    # prompt for consistency + speed. Pre-populated by the route
    # handler BEFORE graph.astream() so generate_hypotheses_node sees
    # them on its first invocation. Each entry:
    #   {alert_text, final_root_cause, similarity, age_days,
    #    tool_calls_used: [str], outcome: str}
    past_investigations: list[dict] = Field(default_factory=list)

    # -- Hypothesis tree --------------------------------------------
    # Flat id-keyed map so deep trees don't blow up Pydantic recursion.
    nodes_by_id: dict[str, HypothesisNode] = Field(default_factory=dict)
    root_ids: list[str] = Field(default_factory=list)
    current_node_id: str | None = None

    # Embeddings of the hypothesis statements -- used for the
    # duplicate-ancestor check without re-embedding each iteration.
    statement_embeddings: dict[str, list[float]] = Field(default_factory=dict)

    # -- Final output -----------------------------------------------
    final_root_cause: str | None = None
    final_chain_node_ids: list[str] = Field(default_factory=list)
    outcome: Literal[
        "pending",
        "validated_root_cause",
        "inconclusive",
        "circuit_breaker_terminated",
    ] = "pending"

    # -- Budget + trace ---------------------------------------------
    budget_state: BudgetState = Field(default_factory=BudgetState)
    agent_trace: list[TraceEvent] = Field(default_factory=list)

    # -- Helpers ----------------------------------------------------
    def get(self, node_id: str) -> HypothesisNode:
        return self.nodes_by_id[node_id]

    def ancestors(self, node_id: str) -> list[HypothesisNode]:
        chain: list[HypothesisNode] = []
        cursor = self.nodes_by_id.get(node_id)
        while cursor and cursor.parent_id:
            parent = self.nodes_by_id.get(cursor.parent_id)
            if parent is None:
                break
            chain.append(parent)
            cursor = parent
        return chain

    def add_node(self, node: HypothesisNode) -> HypothesisNode:
        """Insert a node and wire the parent's children list."""
        self.nodes_by_id[node.id] = node
        if node.parent_id is None:
            self.root_ids.append(node.id)
        else:
            parent = self.nodes_by_id.get(node.parent_id)
            if parent is not None and node.id not in parent.children:
                parent.children.append(node.id)
        self.budget_state.total_nodes += 1
        return node

    def next_pending_id(self) -> str | None:
        """DFS over the tree, returning the first pending node found.

        Order: process every pending child of the current chain before
        moving back up to siblings. Roots are scanned left-to-right.
        """
        stack: list[str] = list(reversed(self.root_ids))
        while stack:
            nid = stack.pop()
            node = self.nodes_by_id.get(nid)
            if node is None:
                continue
            if node.status == "pending":
                return nid
            for child in reversed(node.children):
                stack.append(child)
        return None
