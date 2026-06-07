"""LangGraph agent state definition."""
from __future__ import annotations

from typing import Literal, TypedDict

from opsrag.interfaces.chunker import Chunk
from opsrag.interfaces.graphstore import GraphSearchResult

QueryType = Literal[
    "incident",
    "howto",
    "architecture",
    "config_lookup",
    "postmortem_search",
    "blast_radius",
    "dependency_map",
    "general",
]


class OpsRAGState(TypedDict, total=False):
    # Input
    query: str
    conversation_history: list[dict]
    user_id: str
    thread_id: str

    # Routing
    query_type: QueryType
    intent_confidence: float
    requires_graph: bool

    # Retrieval
    retrieved_chunks: list[Chunk]
    graph_context: GraphSearchResult | None
    keyword_results: list[Chunk]
    merged_results: list[Chunk]
    graded_chunks: list[Chunk]
    sources_searched: list[str]
    scoped_repo: str | None        # Repo path the retriever scoped to, if any
    scoped_repo_mode: str | None   # "hard" (Qdrant filter) | "soft" (boost only) | None
    plural_repo_intent: bool       # User asked for cross-repo coverage ("all repos with X")
    fanout_slugs: list[str]        # Service slugs the retriever fanned-out on via text-match
    listing_intent: bool           # User asked for "structure / list of files / what's in"
    scoped_path: str | None        # Sub-directory path within the scoped repo, if any

    # Generation
    generation: str
    generation_grounded: bool
    final_chunks: list[Chunk]      # Post parent-substitution; what the LLM saw

    # T1.2 -- code-grounded answer verification verdict
    # {verified: [str], unverifiable: [str], skipped?: bool, reason?: str}
    verification_result: dict

    # T1.5 -- HyDE hypothetical answer used for retrieval embedding
    hyde_text: str | None

    # Memory
    user_preferences: dict
    session_context: str
    # Per-user durable memories (Mem0), semantically recalled for THIS query and
    # injected into generation so answers feel personalized + continuous.
    user_memories: list

    # Control flow
    retry_count: int
    max_retries: int
    current_step: str
    error: str | None
    # Classifier verdict ("forensic"/"live"/"procedural"/"mixed"/"casual"/...)
    # piped in before invoke; read by hyde_expansion (live-skip) and entry_route
    # (casual -> friendly lane).
    query_category: str
    # Best (un-boosted) cross-encoder score from the reranker; the grader trusts
    # the rerank top-1 over a CRAG rewrite when this clears the trust floor.
    best_rerank_score: float
    # Per-reranker calibration surfaced by the rerank node (weak-retrieval floor
    # + grader trust floor), so scale-dependent thresholds match the active
    # provider (FastEmbed sigmoid vs Cohere compressed-low).
    min_rerank_score: float
    rerank_trust_score: float
    # True when the reranker errored and we fell back to vector order -- carries a
    # mid-band score so it bypasses the weak gate WITHOUT signalling max trust.
    reranker_outage: bool
    # Generation budget (caps the grader's synthesis floor); decomposed sub-query
    # texts when query decomposition is enabled (also sizes the floor).
    top_k: int
    sub_queries: list[str]
    # Queries already tried by the rewriter this turn -- fed back into the
    # rewrite prompt so each CRAG retry explores a different reformulation.
    rewrite_history: list[str]

    # Tool use
    tool_calls: list[dict]
    tool_results: list[dict]

    # Phase 03 Pillar 2 -- agentic tool-calling loop state
    tool_call_count: int                 # total tools executed this query (loop bound)
    tool_message_history: list[dict]     # role/content + tool_call/tool_result entries for the LLM
    tool_path_active: bool               # set when tool_decide chose tools over retrieval
    tool_call_audit: list[dict]          # per-call: name, args, latency_ms, error?, timestamp
    plan: list[dict]                     # rec #3 -- externalized reasoner plan (items: id/hypothesis/status/next_tool/evidence_so_far/confidence). Mutated via the `update_plan` tool; rendered to the operator via SSE `render_component: InvestigationPlan`.

    # Chunks retrieved by tool_caller's retrieval tools (currently
    # `knowledge_search`; extensible to `runbook_load`, `confluence_search`,
    # etc. via the registry in `tool_caller.py`). Accumulated across all
    # iterations of the tool loop, deduped by (repo, source_path), and
    # copied to `final_chunks` by `tool_synthesize_node` so the API
    # response carries proper sources/sources_content/source_urls -- same
    # shape as the standard retrieval-graph path. Empty when the tool
    # path didn't run any retrieval tools (e.g., GitLab-only queries).
    tool_retrieved_chunks: list[Chunk]

    # Phase 03 Pillar 3 -- hybrid Flash/Pro routing decision audit
    model_route_decision: dict           # {tier, reason, matched_patterns, model}

    # Sub-sprint 1 -- multi-agent emitted event (one per node)
    agent_event: dict                    # {type:agent_status, agent, status, message, metadata}

    # Sub-sprint 3 V1 -- past investigations matching the current query.
    # Pre-fetched in query_with_session* via cosine similarity. The
    # reasoner reads these as additional context on follow-up queries.
    past_investigations: list[dict]      # [{question, answer, tool_calls, similarity, age_seconds}]
