"""LangGraph graph builders.

``build_minimal_graph``  -- Phase 1: vector_retrieve -> generate.
``build_full_graph``     -- Phase 2: route -> retrieve -> rerank -> grade ->
                           (rewrite | generate) -> hallucination_check.
``build_hybrid_graph``   -- Phase 3 hybrid (vector + Neo4j + keyword fan-out).
                           REMOVED 2026-05-23; the Neo4j graph lane was
                           dead in prod for ~3 months. Stub kept so
                           imports don't break until api/server.py is
                           rewired.
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from uuid import uuid4

_log = logging.getLogger("opsrag.graph")


def _qa_cache_globally_disabled() -> bool:
    """True when OPSRAG_DISABLE_QA_CACHE is set. The eval harness runs the target
    server with this on so the QA cache can't serve a stored answer (and stored
    sources) for a golden -- otherwise a retrieval regression is masked by a cache
    hit and the ranking metrics measure the cache, not retrieval. See
    eval/golden/README.md."""
    import os
    return os.environ.get("OPSRAG_DISABLE_QA_CACHE", "").lower() in (
        "1", "true", "yes", "on",
    )

# Retry meta-command: one-word user inputs that mean "re-run my previous
# question" rather than literal investigation topics. Anchored to whole-
# query so "retry the acme-analytics-v3 pipeline" still routes to investigation.
_RETRY_META_RE = re.compile(
    r"^\s*(retry|again|redo|do\s+(it|that)\s+again|once\s+more|"
    r"try\s+again|go\s+again|repeat\s*(that|it)?|same\s+thing)\s*[?!.\s]*$",
    re.IGNORECASE,
)


def _expand_retry_meta(query: str, prior: list[dict]) -> str:
    """If `query` is a retry meta-command, substitute the user's previous
    real question. Otherwise return `query` unchanged.

    `prior` is the chronological replay from session_store (oldest first);
    user messages alternate with assistant. We walk backwards looking for
    the most recent non-retry user message.

    No-op when the thread has no prior user turn or when `query` isn't a
    retry pattern. Substitution is logged so the SSE thinking-trace can
    show why the agent is investigating something the user didn't type.
    """
    if not _RETRY_META_RE.match(query or ""):
        return query
    for m in reversed(prior or []):
        if not isinstance(m, dict): continue
        if m.get("role") != "user": continue
        content = (m.get("content") or "").strip()
        if not content: continue
        if _RETRY_META_RE.match(content): continue  # don't recurse on prior retries
        _log.info(
            "retry meta-command: substituting %r -> previous user query (len=%d)",
            query, len(content),
        )
        return content
    _log.info("retry meta-command: %r received but no prior user query found", query)
    return query


_swr_log = logging.getLogger("opsrag.cache.swr")

from langgraph.graph import END, START, StateGraph

# Phase 02.7 -- user-facing labels per LangGraph node, used by
# `query_with_session_events()` to emit human-readable status lines
# alongside raw node names. Kept colocated with graph.py so additions
# stay in sync with `build_*_graph()` node registration.
_NODE_LABELS: dict[str, str] = {
    "load_memory": "Loading conversation history...",
    "route_query": "Routing query...",
    "hyde_expansion": "Expanding query (HyDE)...",
    "vector_retrieve": "Searching documentation...",
    "keyword_retrieve": "Scanning keywords...",
    "merge_results": "Merging results...",
    "rerank": "Ranking by relevance...",
    "grade_documents": "Verifying retrieval coverage...",
    "rewrite_query": "Refining search...",
    "generate": "Generating answer...",
    "verify_answer": "Verifying claims against the corpus...",
    "check_hallucination": "Checking grounding...",
    "insufficient_info": "No relevant information found...",
    "save_memory": "Saving conversation...",
    # Phase 03 Pillar 2 -- agentic tool-calling labels
    "tool_decide": "Deciding whether to call live tools...",
    "tool_execute": "Calling live GitLab tools...",
    "tool_synthesize": "Synthesizing tool results...",
    # Sub-sprint 1 -- multi-agent named labels
    "triage": "Triaging the query...",
    "tool_caller": "Calling live tools...",
    "reasoner": "Reasoning over results...",
    "generator": "Writing the answer...",
    # Third lane -- friendly chitchat, bypasses triage/RAG/tools.
    "friendly_generator": "Replying...",
}

from opsrag.agent.nodes import (
    check_hallucination_node,
    entry_route,
    friendly_generator_node,
    generate_node,
    generator_node,
    grade_decision,
    grade_documents_node,
    hallucination_decision,
    insufficient_info_node,
    load_memory_node,
    reasoner_node,
    reasoner_route,
    rerank_node,
    rewrite_query_node,
    route_query_node,
    save_memory_node,
    tool_caller_node,
    tool_decide_node,
    tool_decide_route,
    tool_execute_node,
    tool_synthesize_node,
    triage_node,
    triage_route,
    vector_retrieve_node,
    verify_answer_node,
)

# T1.5 -- HyDE retrieval expansion node. Imported at module level so all
# 3 graph builders can wire it before vector_retrieve.
from opsrag.agent.nodes.hyde_expansion import hyde_expansion_node
from opsrag.agent.nodes.reranker import rerank_decision
from opsrag.agent.state import OpsRAGState
from opsrag.interfaces.embedder import EmbeddingProvider
from opsrag.interfaces.graphstore import KnowledgeGraphStore
from opsrag.interfaces.llm import LLMProvider
from opsrag.interfaces.memory import MemoryStore
from opsrag.llms.content import ImagePart
from opsrag.interfaces.observability import ObservabilityProvider
from opsrag.interfaces.reranker import Reranker
from opsrag.interfaces.vectorstore import VectorStore


def build_minimal_graph(
    llm: LLMProvider,
    vector_store: VectorStore,
    embedder: EmbeddingProvider,
    observability: ObservabilityProvider,
    reranker: Reranker | None = None,
    checkpointer=None,
    top_k: int = 5,
    # 5, not 3: the reranker writes only the top rerank_top_k to merged_results,
    # and the CRAG grader can only judge that truncated set -- so this is both
    # the generation budget AND the entire universe the grader sees. 3 caps
    # recall below where multi-fact / synthesis answers break down; align with
    # the other graphs' default of 5.
    rerank_top_k: int = 5,
    rerank_diversity: float = 0.0,
    known_repos: list[str] | None = None,
    code_embedder: EmbeddingProvider | None = None,
    code_store: VectorStore | None = None,
):
    """Minimal agent: hyde -> vector_retrieve -> [rerank] -> generate -> verify -> END."""
    graph = StateGraph(OpsRAGState)
    graph.add_node("hyde_expansion", hyde_expansion_node(llm, observability))
    graph.add_node(
        "vector_retrieve",
        vector_retrieve_node(
            vector_store, embedder, observability,
            top_k=top_k, known_repos=known_repos,
            code_embedder=code_embedder, code_store=code_store,
        ),
    )
    if reranker:
        graph.add_node("rerank", rerank_node(reranker, observability, top_k=rerank_top_k, diversity=rerank_diversity))
    graph.add_node("generate", generate_node(llm, observability, vector_store=vector_store))
    graph.add_node("verify_answer", verify_answer_node(llm, vector_store, observability))

    graph.add_edge(START, "hyde_expansion")
    graph.add_edge("hyde_expansion", "vector_retrieve")
    if reranker:
        graph.add_edge("vector_retrieve", "rerank")
        graph.add_edge("rerank", "generate")
    else:
        graph.add_edge("vector_retrieve", "generate")
    graph.add_edge("generate", "verify_answer")
    graph.add_edge("verify_answer", END)

    return graph.compile(checkpointer=checkpointer)


def build_full_graph(
    llm: LLMProvider,
    vector_store: VectorStore,
    embedder: EmbeddingProvider,
    reranker: Reranker,
    observability: ObservabilityProvider,
    memory_store: MemoryStore | None = None,
    checkpointer=None,
    top_k: int = 10,
    rerank_top_k: int = 5,
    rerank_diversity: float = 0.0,
    known_repos: list[str] | None = None,
    light_graph=None,
    model_router=None,
    code_embedder: EmbeddingProvider | None = None,
    code_store: VectorStore | None = None,
):
    """Phase 2+ agent with routing, grading, rewriting, hallucination check, and memory.

    When ``light_graph`` is wired, a 1-hop ``entity_expand`` node runs between
    vector_retrieve and rerank -- augment-only, never the main retrieval line."""
    graph = StateGraph(OpsRAGState)

    has_memory = memory_store is not None
    if has_memory:
        graph.add_node("load_memory", load_memory_node(memory_store, observability))
    graph.add_node("route_query", route_query_node(llm, observability))
    graph.add_node("hyde_expansion", hyde_expansion_node(llm, observability))
    graph.add_node(
        "vector_retrieve",
        vector_retrieve_node(
            vector_store, embedder, observability,
            top_k=top_k, known_repos=known_repos,
            code_embedder=code_embedder, code_store=code_store,
        ),
    )
    graph.add_node("rerank", rerank_node(reranker, observability, top_k=rerank_top_k, diversity=rerank_diversity))
    if light_graph is not None:
        from opsrag.agent.nodes.entity_expand import entity_expand_node
        graph.add_node(
            "entity_expand",
            entity_expand_node(vector_store, light_graph, embedder),
        )
    graph.add_node("grade_documents", grade_documents_node(llm, observability))
    graph.add_node("rewrite_query", rewrite_query_node(llm, observability))
    # Final answer uses the stronger "answer" model (Sonnet 4.6 pro_llm) when a
    # model_router is wired, so replies feel like Claude; cheap nodes stay on
    # the base llm. Falls back to base llm when no router.
    _answer_llm = getattr(model_router, "pro_llm", None) if model_router else None
    graph.add_node(
        "generate",
        generate_node(llm, observability, vector_store=vector_store, answer_llm=_answer_llm),
    )
    graph.add_node("verify_answer", verify_answer_node(llm, vector_store, observability))
    graph.add_node("check_hallucination", check_hallucination_node(llm, observability))
    # Phase 2 Step 3 (ADR-004 CRAG): honest "insufficient information" fallback.
    graph.add_node("insufficient_info", insufficient_info_node(observability))
    if has_memory:
        graph.add_node("save_memory", save_memory_node(memory_store, observability))

    # Casual / greeting lane. Without this, chitchat ("hi", "what can you do")
    # fell through the full RAG pipeline -> "I cannot find relevant
    # information" with terraform sources, which feels broken. entry_route
    # reads state["query_category"] (set by the classifier before invoke):
    # casual -> a natural conversational reply (Sonnet, no retrieval); anything
    # else -> the normal RAG flow. Uses the pro/answer model so chat feels like
    # talking to a real agent, not a search box.
    # Casual / greeting lane. entry_route reads state["query_category"] (set by
    # the classifier before invoke): casual -> friendly_generator (natural,
    # memory-aware reply, no retrieval); else -> the RAG flow. entry_route stays
    # on START (routing here is reliable; attaching it after load_memory caused
    # casual queries to fall through to RAG). The friendly node reads per-user
    # memory ITSELF (memory_store passed in) so the chat lane still answers
    # identity/ownership/preference questions from recall without depending on
    # load_memory running first.
    graph.add_node(
        "friendly_generator",
        friendly_generator_node(
            _answer_llm or llm, observability,
            model_router=model_router, memory_store=memory_store,
        ),
    )
    graph.add_edge("friendly_generator", END)

    _rag_entry = "load_memory" if has_memory else "route_query"
    graph.add_conditional_edges(
        START,
        entry_route,
        {"friendly": "friendly_generator", "triage": _rag_entry},
    )
    if has_memory:
        graph.add_edge("load_memory", "route_query")

    # route_query unconditionally flows into hyde_expansion -- this was a
    # conditional edge with a hard-coded `lambda s: "hyde_expansion"`, which
    # looked like it branched but never did. It's a plain edge. The node still
    # earns its keep: its `query_type` output is consumed by hyde_expansion
    # (skips HyDE for live queries), the generator (system-prompt selection),
    # and memory_saver -- just not for routing.
    graph.add_edge("route_query", "hyde_expansion")
    graph.add_edge("hyde_expansion", "vector_retrieve")
    if light_graph is not None:
        # 1-hop entity augmentation between retrieve and rerank (augment-only).
        graph.add_edge("vector_retrieve", "entity_expand")
        graph.add_edge("entity_expand", "rerank")
    else:
        graph.add_edge("vector_retrieve", "rerank")
    # Path-aware gate: if the query named specific entities (anchors) but
    # NO retrieved chunk's source_path/repo contains any anchor, AND the
    # cross-encoder's best score is below the noise floor, retrieval has
    # nothing about the asked-about entity. Rewriting won't help -- emit
    # insufficient_info directly instead of fabricating from adjacent
    # chunks (the failure mode this gate is here to prevent).
    graph.add_conditional_edges(
        "rerank",
        rerank_decision,
        {"ok": "grade_documents", "weak_retrieval": "insufficient_info"},
    )

    graph.add_conditional_edges(
        "grade_documents",
        grade_decision,
        {
            "has_relevant": "generate",
            "needs_rewrite": "rewrite_query",
            "insufficient_info": "insufficient_info",
        },
    )
    # On rewrite, skip hyde for the retry -- the rewritten query is
    # already an aggressive rewrite. Re-running hyde would just stack
    # two layers of paraphrase and slow retries.
    graph.add_edge("rewrite_query", "vector_retrieve")

    graph.add_edge("generate", "verify_answer")
    graph.add_edge("verify_answer", "check_hallucination")

    end_after_check = "save_memory" if has_memory else END
    graph.add_conditional_edges(
        "check_hallucination",
        hallucination_decision,
        {
            "grounded": end_after_check,
            "not_grounded": "generate",
            "max_retries_hit": end_after_check,
        },
    )
    # Insufficient-info fallback skips generation entirely -- straight to save_memory or END.
    graph.add_edge("insufficient_info", end_after_check)
    if has_memory:
        graph.add_edge("save_memory", END)

    return graph.compile(checkpointer=checkpointer)


def build_hybrid_graph(
    llm: LLMProvider,
    vector_store: VectorStore,
    embedder: EmbeddingProvider,
    graph_store: KnowledgeGraphStore,
    reranker: Reranker,
    observability: ObservabilityProvider,
    memory_store: MemoryStore | None = None,
    checkpointer=None,
    top_k: int = 10,
    rerank_top_k: int = 5,
    known_repos: list[str] | None = None,
):
    """Hybrid graph (vector + Neo4j graph + keyword fan-out) -- REMOVED 2026-05-23.

    The Neo4j graph-anchored retrieval lane was dead in prod for ~3
    months (Community edition without APOC plugin -> silent failure).
    Effort was redirected to a Cartography-pointed graph backend.

    Callers should use `build_full_graph` instead (same retrieval
    pipeline minus the graph fan-out). Kept as a stub so existing
    `from opsrag.agent import build_hybrid_graph` imports keep working
    until the sibling Cartography agent rewires api/server.py.
    """
    raise NotImplementedError(
        "build_hybrid_graph was removed 2026-05-23 along with the Neo4j "
        "graph-anchored retrieval lane. Configure agent.mode='full' "
        "(or 'minimal') in config-local.yaml; a Cartography-backed "
        "hybrid mode may return in a future revision."
    )


def build_tool_calling_graph(
    llm: LLMProvider,
    vector_store: VectorStore,
    embedder: EmbeddingProvider,
    observability: ObservabilityProvider,
    reranker: Reranker | None = None,
    checkpointer=None,
    top_k: int = 10,
    rerank_top_k: int = 5,
    rerank_diversity: float = 0.0,
    known_repos: list[str] | None = None,
    model_router=None,
    code_embedder: EmbeddingProvider | None = None,
    code_store: VectorStore | None = None,
    light_graph=None,
):
    """Phase 03 Pillar 2 -- agent with live MCP tool-calling.

    The graph runs `tool_decide` first. If the LLM emits function
    calls, the agent loops `tool_decide <-> tool_execute` up to
    `MAX_TOOL_CALLS` times, then synthesizes via `tool_synthesize`.
    If the LLM declines to call any function, the graph falls through
    to the standard retrieval path (vector_retrieve -> [rerank] ->
    generate).

        START -> tool_decide
                  |
                  +-- tools chosen -> tool_execute -> tool_decide (loop) -> tool_synthesize -> END
                  |
                  +-- retrieval -> vector_retrieve -> [rerank] -> generate -> END
    """
    graph = StateGraph(OpsRAGState)
    graph.add_node("tool_decide", tool_decide_node(llm, observability))
    graph.add_node("tool_execute", tool_execute_node(observability))
    graph.add_node("tool_synthesize", tool_synthesize_node(llm, observability, model_router=model_router))
    graph.add_node(
        "vector_retrieve",
        vector_retrieve_node(
            vector_store, embedder, observability,
            top_k=top_k, known_repos=known_repos,
            code_embedder=code_embedder, code_store=code_store,
        ),
    )
    if reranker:
        graph.add_node("rerank", rerank_node(reranker, observability, top_k=rerank_top_k, diversity=rerank_diversity))
    graph.add_node("generate", generate_node(llm, observability, vector_store=vector_store))
    if light_graph is not None:
        from opsrag.agent.nodes.entity_expand import entity_expand_node
        graph.add_node(
            "entity_expand",
            entity_expand_node(vector_store, light_graph, embedder),
        )

    graph.add_edge(START, "tool_decide")

    # tool_decide branches: tools -> execute, no tools -> retrieval, cap -> synthesize
    graph.add_conditional_edges(
        "tool_decide",
        tool_decide_route,
        {
            "tool_execute": "tool_execute",
            "tool_synthesize": "tool_synthesize",
            "retrieval": "vector_retrieve",
        },
    )
    # After execute, loop back to decide (which will either issue more
    # calls or pivot to synthesize once the LLM is done / cap is hit).
    graph.add_edge("tool_execute", "tool_decide")

    # Tool path terminates at synthesize.
    graph.add_edge("tool_synthesize", END)

    # Retrieval path mirrors build_minimal_graph (no NLI, no CRAG --
    # tool-calling is opt-in and we keep the retrieval branch simple).
    # Retrieval branch, with optional entity_expand between retrieve and rerank.
    _post_retrieve = "entity_expand" if light_graph is not None else None
    if _post_retrieve:
        graph.add_edge("vector_retrieve", "entity_expand")
    if reranker:
        graph.add_edge(_post_retrieve or "vector_retrieve", "rerank")
        graph.add_edge("rerank", "generate")
    else:
        graph.add_edge(_post_retrieve or "vector_retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=checkpointer)


def build_multi_agent_graph(
    llm: LLMProvider,
    vector_store: VectorStore,
    embedder: EmbeddingProvider,
    observability: ObservabilityProvider,
    reranker: Reranker | None = None,
    checkpointer=None,
    top_k: int = 10,
    rerank_top_k: int = 5,
    rerank_diversity: float = 0.0,
    known_repos: list[str] | None = None,
    model_router=None,
    code_embedder: EmbeddingProvider | None = None,
    code_store: VectorStore | None = None,
    light_graph=None,
):
    """Sub-sprint 1 -- multi-agent graph: triage -> tool_caller <-> reasoner -> generator,
    with retrieval fall-through.

        START -> entry_route
                  |
                  +-- CASUAL -> friendly_generator -> END        (third lane, <=2s)
                  |
                  +-- everything else -> triage
                                          |
                                          +-- tool_path -> tool_caller <-> reasoner -> generator -> END
                                          |
                                          +-- retrieval -> vector_retrieve -> [rerank] -> generate -> END
    """
    graph = StateGraph(OpsRAGState)
    graph.add_node("friendly_generator", friendly_generator_node(llm, observability, model_router=model_router))
    graph.add_node("triage", triage_node(llm, observability, model_router=model_router))
    graph.add_node("tool_caller", tool_caller_node(observability, llm_for_compaction=llm))
    graph.add_node("reasoner", reasoner_node(llm, observability, model_router=model_router))
    graph.add_node("generator", generator_node(llm, observability, model_router=model_router, vector_store=vector_store))
    graph.add_node(
        "vector_retrieve",
        vector_retrieve_node(
            vector_store, embedder, observability,
            top_k=top_k, known_repos=known_repos,
            code_embedder=code_embedder, code_store=code_store,
        ),
    )
    if reranker:
        graph.add_node("rerank", rerank_node(reranker, observability, top_k=rerank_top_k, diversity=rerank_diversity))
    graph.add_node("generate", generate_node(llm, observability, vector_store=vector_store))
    # Light-graph 1-hop entity augmentation on the retrieval branch -- was wired
    # ONLY into build_full_graph, so in multi_agent mode (config-local default)
    # the lane was dead: edges still computed at index time + entity_ids still
    # bloated every payload, but nothing read them. Mirror the full-graph wiring.
    if light_graph is not None:
        from opsrag.agent.nodes.entity_expand import entity_expand_node
        graph.add_node(
            "entity_expand",
            entity_expand_node(vector_store, light_graph, embedder),
        )

    # Pre-triage branch: CASUAL -> friendly_generator (terminal); else -> triage.
    graph.add_conditional_edges(
        START,
        entry_route,
        {
            "friendly": "friendly_generator",
            "triage": "triage",
        },
    )
    graph.add_edge("friendly_generator", END)
    graph.add_conditional_edges(
        "triage",
        triage_route,
        {
            "tool_caller": "tool_caller",
            "generator": "generator",   # triage emitted no tools but flagged tool path
            "retrieval": "vector_retrieve",
        },
    )
    graph.add_edge("tool_caller", "reasoner")
    graph.add_conditional_edges(
        "reasoner",
        reasoner_route,
        {"tool_caller": "tool_caller", "generator": "generator"},
    )
    graph.add_edge("generator", END)

    # Retrieval branch, with optional entity_expand between retrieve and rerank.
    _post_retrieve = "entity_expand" if light_graph is not None else None
    if _post_retrieve:
        graph.add_edge("vector_retrieve", "entity_expand")
    if reranker:
        graph.add_edge(_post_retrieve or "vector_retrieve", "rerank")
        graph.add_edge("rerank", "generate")
    else:
        graph.add_edge(_post_retrieve or "vector_retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=checkpointer)


# V1 UX -- module-level so both `query_with_session` (non-streaming) and
# `query_with_session_events` (streaming) can use it. Compute a
# clickable URL for a source chunk where possible. GitLab repos ->
# blob URL on the configured base. Confluence -> space/page URL. Slack
# -> channel URL. Rootly -> incident URL. None for unknown source types
# (UI falls back to plain-text rendering).
@dataclass(frozen=True)
class SourceUrlBases:
    """Web base URLs used to render clickable source links.

    Built from ``DeploymentContext.source_urls`` at request time via
    :meth:`from_app_config`. All fields default to ``None``; per
    Constitution Principle VI, the engine carries no example URLs.
    When a field is ``None``, callers that build deep-links MUST skip
    the corresponding link rather than substituting a placeholder.

    Tools and tests that don't pass an ``app_config`` get the empty
    bases (all None) and produce no deep-links -- which is correct:
    a deployment without a Confluence URL configured shouldn't render
    Confluence deep-links anyway.
    """

    confluence: str | None = None
    slack: str | None = None
    rootly: str | None = None
    gitlab: str | None = None

    @classmethod
    def from_app_config(cls, cfg) -> SourceUrlBases:
        # Prefer ``deployment.source_urls`` when populated. Fall back to
        # the legacy per-provider config blocks for the call sites that
        # haven't been migrated yet -- they will be retired alongside
        # the legacy provider blocks themselves.
        deployment_sources = getattr(getattr(cfg, "deployment", None), "source_urls", None)

        def _pick(deployment_attr: str, legacy_block: str, legacy_attr: str) -> str | None:
            if deployment_sources is not None:
                v = getattr(deployment_sources, deployment_attr, None)
                if v:
                    return v.rstrip("/")
            block = getattr(cfg, legacy_block, None)
            if block is None:
                return None
            v = getattr(block, legacy_attr, None)
            return v.rstrip("/") if v else None

        return cls(
            confluence=_pick("confluence", "confluence", "base_url"),
            slack=_pick("slack", "slack", "workspace_url"),
            rootly=_pick("rootly", "rootly", "web_base_url"),
            gitlab=_pick("gitlab", "scm", "base_url"),
        )


_DEFAULT_BASES = SourceUrlBases()


def _src(c) -> str:
    repo = getattr(c, "repo", "") or ""
    return f"{repo}/{c.source_path}" if repo else c.source_path


def _src_url(c, bases: SourceUrlBases | None = None) -> str | None:
    bases = bases or _DEFAULT_BASES
    repo = getattr(c, "repo", "") or ""
    path = getattr(c, "source_path", "") or ""
    branch = getattr(c, "branch", "") or "master"
    if not repo:
        return None
    if repo.startswith("confluence:"):
        page_id = path.split(":", 1)[0] if ":" in path else ""
        if page_id:
            return f"{bases.confluence}/wiki/pages/viewpage.action?pageId={page_id}"
        return None
    if repo.startswith("slack:"):
        channel = repo.split(":", 1)[1].lstrip("#")
        return f"{bases.slack}/archives/{channel}"
    if repo.startswith("rootly:"):
        inc_id = path.split(":", 1)[0] if ":" in path else ""
        if inc_id:
            return f"{bases.rootly}/incidents/{inc_id}"
        return None
    return f"{bases.gitlab}/{repo}/-/blob/{branch}/{path}"


def _src_url_from_string(s: str, bases: SourceUrlBases | None = None) -> str | None:
    """Same logic as `_src_url` but derived from the rendered source string
    `<repo>/<path>` (or just `<path>` when no repo). Used on cache hits
    where the original Chunk objects are gone -- the cache only persists
    the rendered source strings."""
    bases = bases or _DEFAULT_BASES
    if not s:
        return None
    if s.startswith("confluence:"):
        rest = s.split("/", 1)[1] if "/" in s else ""
        page_id = rest.split(":", 1)[0] if ":" in rest else ""
        if page_id:
            return f"{bases.confluence}/wiki/pages/viewpage.action?pageId={page_id}"
        return None
    if s.startswith("slack:"):
        repo, _, _path = s.partition("/")
        channel = repo.split(":", 1)[1].lstrip("#")
        return f"{bases.slack}/archives/{channel}"
    if s.startswith("rootly:"):
        rest = s.split("/", 1)[1] if "/" in s else ""
        inc_id = rest.split(":", 1)[0] if ":" in rest else ""
        if inc_id:
            return f"{bases.rootly}/incidents/{inc_id}"
        return None
    parts = s.split("/", 2)
    if len(parts) < 3:
        return None
    owner, repo, path = parts
    return f"{bases.gitlab}/{owner}/{repo}/-/blob/master/{path}"


async def _swr_revalidate(
    *,
    compiled_graph,
    query: str,
    user_id: str,
    thread_id: str,
    embedder,
    qa_cache,
    llm,
    session_store,
    investigation_cache,
    source_url_bases,
    semantic_router,
) -> None:
    """Fire-and-forget background re-run of the agent for a stale cache
    hit. Calls `query_with_session` with SWR disabled (to avoid serving
    stale to itself) and lets its normal write-back path replace the
    cached entry. Errors are swallowed -- best-effort refresh."""
    import os
    prev = os.environ.get("OPSRAG_QA_CACHE_SWR")
    try:
        # Temporarily disable SWR for the recursive call so the freshly
        # spawned task can't loop on its own stale hit. Per-task scope:
        # we restore in the finally block.
        os.environ["OPSRAG_QA_CACHE_SWR"] = "0"
        await query_with_session(
            compiled_graph,
            query=query,
            user_id=user_id,
            thread_id=f"{thread_id}__swr",  # separate thread to not pollute the user's history
            embedder=embedder,
            qa_cache=qa_cache,
            llm=llm,
            session_store=session_store,
            investigation_cache=investigation_cache,
            source_url_bases=source_url_bases,
            semantic_router=semantic_router,
        )
        _swr_log.info("revalidation completed for query=%r", query[:80])
    except Exception as exc:
        _swr_log.warning("revalidation failed for query=%r: %s", query[:80], exc)
    finally:
        if prev is None:
            os.environ.pop("OPSRAG_QA_CACHE_SWR", None)
        else:
            os.environ["OPSRAG_QA_CACHE_SWR"] = prev


async def query_with_session(
    compiled_graph,
    query: str,
    user_id: str = "anonymous",
    thread_id: str | None = None,
    embedder=None,
    qa_cache=None,
    llm=None,
    session_store=None,
    investigation_cache=None,
    source_url_bases: SourceUrlBases | None = None,
    semantic_router=None,
    user_email: str | None = None,
    user_name: str | None = None,
    images: list[ImagePart] | None = None,
    vision_llm=None,
) -> dict:
    if thread_id is None:
        thread_id = f"{user_id}_{uuid4().hex[:8]}"
    bases = source_url_bases or _DEFAULT_BASES
    # Sub-sprint 5 classification -- drives cache TTL + skip decisions.
    # `_classification` is set after the embedding is computed so we can
    # use the semantic-router layer; falls back to regex-only otherwise.
    _classification = None

    # Coreference rewrite: if the user query looks like a follow-up
    # ("tell me more about this repo", "what about its config"), expand
    # it using the last 2 turns from the session before doing anything
    # else. Skipped silently when no prior turns / no LLM / not a
    # follow-up. The rewritten query is what gets cache-looked-up,
    # embedded, and fed to the agent -- so cache hits work on follow-ups
    # and retrieval has named entities to anchor on.
    original_query = query
    prior: list[dict] = []
    if session_store is not None and thread_id:
        try:
            prior = await session_store.get_messages(thread_id)
        except Exception:
            prior = []
        # Same retry-expansion as the streaming path.
        query = _expand_retry_meta(query, prior)
        if prior and llm is not None:
            from opsrag.agent.query_rewrite import maybe_rewrite_query
            query = await maybe_rewrite_query(
                query=query, prior_messages=prior, llm=llm,
            )

    # Classify FIRST, unconditionally (when an embedder is available): the
    # query_category must be set even for live / user-scoped / no-cache queries,
    # because HyDE + lane decisions read it. Nesting it under `not
    # should_skip_cache` (or a present qa_cache) left it None for exactly the
    # live queries HyDE must skip -> HyDE ran on "is X slow right now".
    cached_embedding: list[float] | None = None
    from opsrag.agent.classifier import classify_query, policy_for
    if embedder is not None:
        try:
            cached_embedding = await embedder.embed_query(query)
            _classification = await classify_query(
                query,
                query_embedding=cached_embedding,
                semantic_router=semantic_router,
                llm=llm,
            )
        except Exception:
            _classification = None

    # Step 4: Q&A semantic cache. Short-circuit on a similar prior answer. Skip
    # on the classifier's own LIVE verdict, the regex user-scoped guard
    # (`should_skip_cache`: my/our/now/...), or a globally-disabled cache.
    if qa_cache is not None and cached_embedding is not None:
        from opsrag.qa_cache import should_skip_cache
        _skip_cache = (
            should_skip_cache(query)
            or _qa_cache_globally_disabled()
            or (
                _classification is not None
                and policy_for(_classification.category)["skip_cache"]
            )
        )
        hit = None
        if not _skip_cache:
            try:
                # SWR: serve a recently-expired entry tagged stale so the user
                # gets an instant response; a background task revalidates.
                import os
                swr_enabled = os.environ.get("OPSRAG_QA_CACHE_SWR", "1").lower() in ("1", "true", "yes", "on")
                hit = await qa_cache.lookup(
                    cached_embedding, current_query=query,
                    user_id=user_id,
                    serve_stale=swr_enabled,
                )
                # Flash judge for borderline cosine band [0.93, 0.97].
                if hit is not None:
                    from opsrag.qa_cache_judge import judge_match
                    if not await judge_match(
                        current_query=query,
                        cached_question=hit.question,
                        cosine=float(hit.similarity),
                        llm=llm,
                    ):
                        hit = None
            except Exception:
                hit = None
        if hit is not None:
            # SWR -- kick off background revalidation so the next
            # caller gets a fresh answer. Best-effort, swallowed.
            if hit.is_stale:
                asyncio.create_task(_swr_revalidate(
                    compiled_graph=compiled_graph,
                    query=query,
                    user_id=user_id,
                    thread_id=thread_id,
                    embedder=embedder,
                    qa_cache=qa_cache,
                    llm=llm,
                    session_store=session_store,
                    investigation_cache=investigation_cache,
                    source_url_bases=bases,
                    semantic_router=semantic_router,
                ))
            return {
                "answer": hit.answer,
                "sources": hit.sources,
                "source_urls": list(hit.source_urls) if hit.source_urls else [_src_url_from_string(s, bases) for s in hit.sources],
                "sources_content": hit.sources_content,
                "graph_paths": [],
                "grounded": True,  # cached answer already passed prior grounding check
                "query_type": None,
                "thread_id": thread_id,
                "session_resumable": True,
                "cache_hit": True,
                "cache_similarity": hit.similarity,
                "cache_age_seconds": hit.age_seconds,
                "cache_is_stale": bool(hit.is_stale),
                "query_category": _classification.category.value if _classification else None,
            }

    # `configurable` is persisted alongside every checkpoint by langgraph
    # postgres saver. We embed the author identity here so a session
    # replayed by ANOTHER user can show the original author's name
    # instead of a generic "You". user_email/user_name are None for
    # legacy / pre-Pomerium threads -- UI falls back to "You".
    # EPHEMERAL vision side-channel: image bytes + the vision-fallback LLM ride
    # in the runnable `config` ONLY (spec FR-003). LangGraph persists `state`
    # (the `initial` dict below) to the checkpointer per thread_id, so bytes
    # there would survive the turn. `configurable` is only PARTLY persisted:
    # LangGraph promotes a configurable value into the durable checkpoint
    # metadata ONLY when it is a scalar (str/int/float/bool) -- see langgraph
    # `_internal._config._exclude_as_metadata`. That's how the identity strings
    # below (user_email/user_name) survive for cross-user replay. `turn_images`
    # (a list[ImagePart]) and `vision_llm` (an object) are NON-scalar, so they
    # are never promoted, never serialized, and never persisted -- the generator
    # reads them from config at run time only. (Footgun: never put an image as a
    # base64 STRING here; a scalar WOULD be persisted.) The persisted query
    # (in `initial`) records only a text marker noting an image was attached.
    config = {"configurable": {
        "thread_id": thread_id,
        "user_id": user_id,
        "user_email": user_email,
        "user_name": user_name,
        "turn_images": images or [],
        "vision_llm": vision_llm,
    }}
    # History/checkpoint must record THAT an image was attached (so follow-up
    # turns + the session transcript make sense) but NEVER the bytes.
    query_for_state = query
    if images:
        names = ", ".join(img.name or "image" for img in images)
        query_for_state = f"{query} [attached image: {names}]".strip()
    # Per-turn state. CRITICAL: explicitly reset every retrieval/generation
    # field so prior-turn chunks don't leak through the LangGraph checkpointer.
    # Without these resets, rerank_node reads `merged_results` from the
    # previous turn (because it's truthy) before the current turn's
    # `retrieved_chunks` is checked -> the LLM answers using last turn's
    # context regardless of what we just retrieved.
    initial: dict = {
        "query": query_for_state,
        "user_id": user_id,
        "thread_id": thread_id,
        # Loop budgets + per-turn scratch. ALL must reset every turn -- the
        # Postgres checkpointer is keyed by thread_id and TypedDict is
        # last-write-wins, so any field not reset here leaks from the prior turn:
        # spent regen/rewrite budgets (ungrounded answers ship with no
        # correction), rewrite_history[0] re-injecting last turn's anchors, a
        # stale best_rerank_score mis-driving the grader's trust gate, etc.
        "retry_count": 0,
        "max_retries": 3,
        "regen_count": 0,
        "max_regens": 3,
        "rewrite_history": [],
        "best_rerank_score": 0.0,
        "verification_result": {},
        "anchors": [],
        "anchors_matched_in_results": False,
        "hyde_text": None,
        "sub_queries": [],
        # Retrieval / chunks
        "retrieved_chunks": [],
        "merged_results": [],
        "graded_chunks": [],
        "final_chunks": [],
        "keyword_results": [],
        "graph_context": None,
        "sources_searched": [],
        # Routing / control flow
        "query_type": None,
        "intent_confidence": 0.0,
        "requires_graph": False,
        "scoped_repo": None,
        "scoped_repo_mode": None,
        "plural_repo_intent": False,
        "fanout_slugs": [],
        "listing_intent": False,
        "scoped_path": None,
        "current_step": "",
        "error": None,
        # Generation
        "generation": "",
        "generation_grounded": False,
        # Tool use
        "tool_calls": [],
        "tool_results": [],
        # Phase 03 Pillar 2 -- reset every turn so tool-call state from
        # the prior turn doesn't leak via the LangGraph checkpointer.
        # Without these resets, follow-up turns reuse stale
        # tool_message_history and skip tool_execute, regurgitating the
        # previous answer.
        "tool_call_count": 0,
        "tool_message_history": [],
        "tool_path_active": False,
        "tool_call_audit": [],
        # Phase 03 Pillar 3 -- reset routing decision per turn.
        "model_route_decision": {},
        # Sub-sprint 3 V1 -- past-investigations slot, populated below.
        "past_investigations": [],
        # Multi-turn context: prior session turns so triage can resolve
        # follow-up queries against the in-progress investigation. Kept
        # separate from tool_message_history (which stays scoped to the
        # current turn's tool-calling chain).
        "conversation_history": prior,
        # Classifier result piped into state so `entry_route` (the
        # pre-triage branch added 2026-05-24) can fast-path CASUAL
        # queries to the friendly generator without invoking triage +
        # retrieval. None when classifier is disabled.
        "query_category": _classification.category.value if _classification else None,
    }

    # Sub-sprint 3 V1 -- pre-fetch top-K past investigations cosine-near
    # the current query so the reasoner can reference them. We piggyback
    # on the cache_embedding already computed above; no extra embed call.
    if investigation_cache is not None and cached_embedding is not None:
        try:
            past = await investigation_cache.search(cached_embedding, top_k=3)
            initial["past_investigations"] = [
                {
                    "question": h.question,
                    "answer": h.answer,
                    "tool_calls": [a.get("name") for a in h.tool_call_audit if a.get("name")],
                    "similarity": h.similarity,
                    "age_seconds": h.age_seconds,
                    "thread_id": h.thread_id,
                }
                for h in past
            ]
        except Exception:
            pass

    result = await compiled_graph.ainvoke(initial, config=config)

    # Prefer post-parent-substitution chunks (what the LLM actually saw)
    # so the API response reflects the real context. Falls back through
    # earlier pipeline stages for hybrid/full graphs that may not always
    # populate final_chunks.
    graded = (
        result.get("final_chunks")
        or result.get("graded_chunks")
        or result.get("merged_results")
        or result.get("retrieved_chunks")
        or []
    )
    graph_paths = []
    if gc := result.get("graph_context"):
        graph_paths = getattr(gc, "paths", []) if gc else []

    # _src + _src_url are module-level (defined above query_with_session)
    # so the streaming variant can also use them.

    # Per-source chunk content -- used by eval for accurate faithfulness
    # scoring (judge needs actual chunk text, not just file paths).
    # Deduped by source path; truncated to keep response payload bounded.
    seen_paths: set[str] = set()
    sources_content: list[dict] = []
    for c in graded:
        src = _src(c)
        if src in seen_paths:
            continue
        seen_paths.add(src)
        content = (getattr(c, "content", "") or "")[:4000]
        sources_content.append({"source": src, "content": content})

    answer = result.get("generation", "")
    sources_list = list(dict.fromkeys(_src(c) for c in graded))
    # Build a parallel source_urls list -- same order, None where not derivable.
    _src_to_url: dict[str, str | None] = {}
    for c in graded:
        s = _src(c)
        if s not in _src_to_url:
            _src_to_url[s] = _src_url(c, bases)
    source_urls_list: list[str | None] = [_src_to_url.get(s) for s in sources_list]

    # Store fresh result in cache for next time. Skip caching only when
    # hallucination_check explicitly flagged the answer as not grounded.
    # In minimal mode there's no hallucination check, so we treat absent
    # field as "no objection raised" -> cache it.
    # `hallucination_decision` is a ROUTING function, never a state key -- the
    # old `"hallucination_decision" in result` was always False, so an ungrounded
    # answer that shipped via max_retries_hit got cached as grounded (forensic TTL
    # = 90 days) and re-served. Gate on the node's actual outputs instead.
    # NOTE: do NOT key on current_step == "hallucination_checked" -- the terminal
    # save_memory node overwrites current_step to "memory_saved", so with memory
    # enabled that guard is ~always False and ungrounded (max-retries) answers
    # leaked back into the cache. `grounding_checked` is a DURABLE flag set by the
    # hallucination node and never clobbered downstream; it also stays False in
    # minimal mode (no hallucination check) so absent-check answers still cache.
    grounded_explicitly_failed = (
        result.get("generation_grounded") is False
        and result.get("grounding_checked") is True
    )
    tool_path_answer = bool(result.get("tool_path_active"))
    if (
        qa_cache is not None
        and cached_embedding is not None
        and answer
        and not grounded_explicitly_failed
        and not tool_path_answer
    ):
        try:
            # TTL per category -- forensic 90d, procedural 30d, mixed 5min,
            # unknown legacy default. Live answers don't reach this branch
            # (skipped at lookup time).
            _ttl = None
            if _classification is not None:
                from opsrag.agent.classifier import policy_for
                _ttl = policy_for(_classification.category).get("ttl_seconds")
            # Scope to this user ONLY when the answer wove in per-user memories
            # (Mem0) -- otherwise a recalled personal fact could leak to another
            # user on a high cosine match. Shared knowledge answers stay global.
            _user_scope = user_id if result.get("user_memories") else None
            await qa_cache.store(
                question=query,
                embedding=cached_embedding,
                answer=answer,
                sources=sources_list,
                sources_content=sources_content,
                source_urls=source_urls_list,
                ttl_seconds=_ttl,
                user_scope=_user_scope,
            )
        except Exception:
            pass

    # Sub-sprint 3 V1 -- store the investigation outcome for tool-path answers.
    investigation_id: str | None = None
    if (
        investigation_cache is not None
        and tool_path_answer
        and answer
        and cached_embedding is not None
    ):
        try:
            investigation_id = await investigation_cache.store(
                question=query,
                embedding=cached_embedding,
                answer=answer,
                tool_call_audit=result.get("tool_call_audit") or [],
                model_route_decision=result.get("model_route_decision") or {},
                thread_id=thread_id,
                user_id=user_id,
            )
        except Exception:
            pass

    return {
        "answer": answer,
        "sources": sources_list,
        "source_urls": source_urls_list,
        "sources_content": sources_content,
        "graph_paths": graph_paths,
        "grounded": result.get("generation_grounded", False),
        "query_type": result.get("query_type"),
        "thread_id": thread_id,
        "session_resumable": True,
        "cache_hit": False,
        "investigation_id": investigation_id,
        "query_category": _classification.category.value if _classification else None,
        "plan": result.get("plan") or [],
    }


# Phase 02.7 -- streaming progress events. The function below mirrors
# `query_with_session()` but yields per-node progress events as the
# LangGraph executes, so the SSE handler can render a real-time
# "thinking timeline" in the UI. Backward-compat: `query_with_session`
# stays unchanged for non-streaming callers.
async def query_with_session_events(
    compiled_graph,
    query: str,
    user_id: str = "anonymous",
    thread_id: str | None = None,
    embedder=None,
    qa_cache=None,
    llm=None,
    session_store=None,
    investigation_cache=None,
    source_url_bases: SourceUrlBases | None = None,
    semantic_router=None,
    user_email: str | None = None,
    user_name: str | None = None,
    images: list[ImagePart] | None = None,
    vision_llm=None,
) -> AsyncIterator[dict]:
    """Yields progress events as the agent runs.

    Event shapes::

        {"type": "node_start",  "node": <name>, "label": <human text>}
        {"type": "node_end",    "node": <name>, "label": <human text>}
        {"type": "cache_hit",   "similarity": float, "age_seconds": int}
        {"type": "final",       <full result dict -- same keys as query_with_session>}
        {"type": "error",       "detail": str}
    """
    if thread_id is None:
        thread_id = f"{user_id}_{uuid4().hex[:8]}"
    bases = source_url_bases or _DEFAULT_BASES

    original_query = query
    prior: list[dict] = []
    if session_store is not None and thread_id:
        try:
            prior = await session_store.get_messages(thread_id)
        except Exception:
            prior = []
        # Deterministic retry / "do it again" handling -- must run BEFORE
        # the LLM query rewriter and BEFORE classification, because the
        # whole point is to substitute the literal "retry" with the
        # prior user question so routing makes sense.
        query = _expand_retry_meta(query, prior)
        if prior and llm is not None:
            from opsrag.agent.query_rewrite import maybe_rewrite_query
            query = await maybe_rewrite_query(
                query=query, prior_messages=prior, llm=llm,
            )

    cached_embedding: list[float] | None = None
    _classification = None
    # Classify unconditionally (see the non-streaming path) so query_category is
    # set for live / user-scoped / no-cache queries -- HyDE + lane gates read it.
    from opsrag.agent.classifier import classify_query, policy_for
    if embedder is not None:
        try:
            cached_embedding = await embedder.embed_query(query)
            _classification = await classify_query(
                query,
                query_embedding=cached_embedding,
                semantic_router=semantic_router,
                llm=llm,
            )
        except Exception:
            _classification = None

    if qa_cache is not None and cached_embedding is not None:
        from opsrag.qa_cache import should_skip_cache
        _skip_cache = (
            should_skip_cache(query)
            or _qa_cache_globally_disabled()
            or (
                _classification is not None
                and policy_for(_classification.category)["skip_cache"]
            )
        )
        hit = None
        if not _skip_cache:
            try:
                import os
                swr_enabled = os.environ.get("OPSRAG_QA_CACHE_SWR", "1").lower() in ("1", "true", "yes", "on")
                hit = await qa_cache.lookup(
                    cached_embedding, current_query=query,
                    user_id=user_id,
                    serve_stale=swr_enabled,
                )
                if hit is not None:
                    from opsrag.qa_cache_judge import judge_match
                    if not await judge_match(
                        current_query=query,
                        cached_question=hit.question,
                        cosine=float(hit.similarity),
                        llm=llm,
                    ):
                        hit = None
            except Exception:
                hit = None
        if hit is not None:
            if hit.is_stale:
                asyncio.create_task(_swr_revalidate(
                    compiled_graph=compiled_graph,
                    query=query,
                    user_id=user_id,
                    thread_id=thread_id,
                    embedder=embedder,
                    qa_cache=qa_cache,
                    llm=llm,
                    session_store=session_store,
                    investigation_cache=investigation_cache,
                    source_url_bases=bases,
                    semantic_router=semantic_router,
                ))
            yield {
                "type": "cache_hit",
                "similarity": float(hit.similarity),
                "age_seconds": int(hit.age_seconds),
                "is_stale": bool(hit.is_stale),
            }
            yield {
                "type": "final",
                "answer": hit.answer,
                "sources": hit.sources,
                "source_urls": list(hit.source_urls) if hit.source_urls else [_src_url_from_string(s, bases) for s in hit.sources],
                "sources_content": hit.sources_content,
                "graph_paths": [],
                "grounded": True,
                "query_type": None,
                "thread_id": thread_id,
                "session_resumable": True,
                "cache_hit": True,
                "cache_similarity": float(hit.similarity),
                "cache_age_seconds": int(hit.age_seconds),
                "cache_is_stale": bool(hit.is_stale),
                "query_category": _classification.category.value if _classification else None,
            }
            return

    # Scalar `configurable` values are promoted into the durable checkpoint
    # metadata by langgraph (see `_exclude_as_metadata`). We embed the author
    # identity here so a session replayed by ANOTHER user can show the original
    # author's name instead of a generic "You". user_email/user_name are None
    # for legacy / pre-Pomerium threads -- UI falls back to "You".
    # EPHEMERAL vision side-channel (see query_with_session for the full
    # rationale): image bytes + vision-fallback LLM ride in the runnable
    # `config` only. They are NON-scalar, so langgraph never promotes them to
    # checkpoint metadata and never writes them to the graph `state` below --
    # nothing durable holds the bytes. (Footgun: never pass an image as a
    # base64 STRING via configurable; a scalar WOULD be persisted.)
    config = {"configurable": {
        "thread_id": thread_id,
        "user_id": user_id,
        "user_email": user_email,
        "user_name": user_name,
        "turn_images": images or [],
        "vision_llm": vision_llm,
    }}
    # History/checkpoint records THAT an image was attached, never the bytes.
    query_for_state = query
    if images:
        names = ", ".join(img.name or "image" for img in images)
        query_for_state = f"{query} [attached image: {names}]".strip()
    initial: dict = {
        "query": query_for_state,
        "user_id": user_id,
        "thread_id": thread_id,
        # Per-turn budgets + scratch -- ALL reset (checkpointer leaks otherwise;
        # see the non-streaming path).
        "retry_count": 0,
        "max_retries": 3,
        "regen_count": 0,
        "max_regens": 3,
        "rewrite_history": [],
        "best_rerank_score": 0.0,
        "verification_result": {},
        "anchors": [],
        "anchors_matched_in_results": False,
        "hyde_text": None,
        "sub_queries": [],
        "retrieved_chunks": [],
        "merged_results": [],
        "graded_chunks": [],
        "final_chunks": [],
        "keyword_results": [],
        "graph_context": None,
        "sources_searched": [],
        "query_type": None,
        "intent_confidence": 0.0,
        "requires_graph": False,
        "scoped_repo": None,
        "scoped_repo_mode": None,
        "plural_repo_intent": False,
        "fanout_slugs": [],
        "listing_intent": False,
        "scoped_path": None,
        "current_step": "",
        "error": None,
        "generation": "",
        "generation_grounded": False,
        "tool_calls": [],
        "tool_results": [],
        # Per-turn tool budget + scratch -- MUST reset, else the checkpointer
        # leaks them across turns on the same thread_id. Omitting these caused
        # `tool_call_count` to accumulate across a long-lived chat thread (e.g.
        # a Telegram DM) until every new turn hit the loop cap (10) on its first
        # tool call -> the agent could call NO tools -> ungrounded/hallucinated
        # answers. Mirrors the non-streaming `query_with_session` reset above.
        "tool_call_count": 0,
        "tool_message_history": [],
        "tool_path_active": False,
        "tool_call_audit": [],
        "model_route_decision": {},
        "past_investigations": [],
        # Multi-turn context -- see query_with_session for rationale.
        "conversation_history": prior,
        # Classifier result piped into state so `entry_route` (the
        # pre-triage branch added 2026-05-24) can fast-path CASUAL
        # queries to the friendly generator. Same wiring as the
        # non-streaming `query_with_session` above.
        "query_category": _classification.category.value if _classification else None,
    }

    result: dict | None = None
    try:
        # astream_events v2 emits on_chain_start / on_chain_end for each
        # graph node by name. We filter to known nodes and emit
        # progress events as user-facing status lines.
        async for ev in compiled_graph.astream_events(
            initial, config=config, version="v2",
        ):
            kind = ev.get("event")
            name = ev.get("name") or ""
            if kind == "on_chain_start" and name in _NODE_LABELS:
                # Per-MCP dynamic label for tool_caller -- peek at the
                # pending tool_calls in the node's input state and emit
                # something like "Calling Cloud SQL tools..." instead of
                # the generic "Calling live tools...".
                label = _NODE_LABELS[name]
                if name == "tool_caller":
                    from opsrag.agent.nodes.multi_agent import _tool_caller_label
                    state_in = (ev.get("data") or {}).get("input") or {}
                    pending = state_in.get("tool_calls") or []
                    if pending:
                        label = _tool_caller_label(pending)
                yield {
                    "type": "node_start",
                    "node": name,
                    "label": label,
                }
            elif kind == "on_chain_end" and name in _NODE_LABELS:
                yield {
                    "type": "node_end",
                    "node": name,
                    "label": _NODE_LABELS[name],
                }
            elif kind == "on_custom_event" and name == "reasoner_token":
                # Live "thinking out loud" -- each reasoner LLM text
                # delta is dispatched by `_reason_streaming` in
                # multi_agent.py. The FE appends to the current
                # "Reasoning over results..." step's body.
                data = ev.get("data") or {}
                delta = data.get("delta") if isinstance(data, dict) else None
                if delta:
                    yield {"type": "reasoner_token", "delta": delta}
            # The graph itself emits on_chain_end with name == "LangGraph"
            # (or the compiled graph's name) and data.output == final state.
            if (
                kind == "on_chain_end"
                and name not in _NODE_LABELS
                and ev.get("data", {}).get("output") is not None
                and isinstance(ev["data"]["output"], dict)
                and "query" in ev["data"]["output"]
            ):
                # Heuristic: outermost graph end carries the full state.
                result = ev["data"]["output"]
    except Exception as exc:
        yield {"type": "error", "detail": str(exc)}
        return

    if result is None:
        yield {"type": "error", "detail": "graph completed without producing a result"}
        return

    graded = (
        result.get("final_chunks")
        or result.get("graded_chunks")
        or result.get("merged_results")
        or result.get("retrieved_chunks")
        or []
    )
    graph_paths = []
    if gc := result.get("graph_context"):
        graph_paths = getattr(gc, "paths", []) if gc else []

    # _src + _src_url defined at module level (above query_with_session).
    seen_paths: set[str] = set()
    sources_content: list[dict] = []
    for c in graded:
        src = _src(c)
        if src in seen_paths:
            continue
        seen_paths.add(src)
        content = (getattr(c, "content", "") or "")[:4000]
        sources_content.append({"source": src, "content": content})

    answer = result.get("generation", "")
    sources_list = list(dict.fromkeys(_src(c) for c in graded))
    # Build a parallel source_urls list -- same order, None where not derivable.
    _src_to_url: dict[str, str | None] = {}
    for c in graded:
        s = _src(c)
        if s not in _src_to_url:
            _src_to_url[s] = _src_url(c, bases)
    source_urls_list: list[str | None] = [_src_to_url.get(s) for s in sources_list]

    # See the non-streaming path: `hallucination_decision` is a routing fn, not a
    # state key, and current_step is clobbered to "memory_saved" by the terminal
    # save_memory node -- so gate on the DURABLE `grounding_checked` flag instead,
    # otherwise ungrounded (max-retries) answers leak into the cache as grounded.
    grounded_explicitly_failed = (
        result.get("generation_grounded") is False
        and result.get("grounding_checked") is True
    )
    tool_path_answer = bool(result.get("tool_path_active"))
    if (
        qa_cache is not None
        and cached_embedding is not None
        and answer
        and not grounded_explicitly_failed
        and not tool_path_answer
    ):
        try:
            # TTL per category -- forensic 90d, procedural 30d, mixed 5min,
            # unknown legacy default. Live answers don't reach this branch
            # (skipped at lookup time).
            _ttl = None
            if _classification is not None:
                from opsrag.agent.classifier import policy_for
                _ttl = policy_for(_classification.category).get("ttl_seconds")
            # Scope to this user ONLY when the answer wove in per-user memories
            # (Mem0) -- otherwise a recalled personal fact could leak to another
            # user on a high cosine match. Shared knowledge answers stay global.
            _user_scope = user_id if result.get("user_memories") else None
            await qa_cache.store(
                question=query,
                embedding=cached_embedding,
                answer=answer,
                sources=sources_list,
                sources_content=sources_content,
                source_urls=source_urls_list,
                ttl_seconds=_ttl,
                user_scope=_user_scope,
            )
        except Exception:
            pass

    # Sub-sprint 3 V1 -- store tool-path investigation outcome
    investigation_id: str | None = None
    if (
        investigation_cache is not None
        and tool_path_answer
        and answer
        and cached_embedding is not None
    ):
        try:
            investigation_id = await investigation_cache.store(
                question=query,
                embedding=cached_embedding,
                answer=answer,
                tool_call_audit=result.get("tool_call_audit") or [],
                model_route_decision=result.get("model_route_decision") or {},
                thread_id=thread_id,
                user_id=user_id,
            )
        except Exception:
            pass

    yield {
        "type": "final",
        "investigation_id": investigation_id,
        "answer": answer,
        "sources": sources_list,
        "source_urls": source_urls_list,
        "sources_content": sources_content,
        "graph_paths": graph_paths,
        "grounded": result.get("generation_grounded", False),
        "query_type": result.get("query_type"),
        "thread_id": thread_id,
        "session_resumable": True,
        "cache_hit": False,
        # Carry the tool conversation so the SSE handler can detect
        # renderable tool outputs (e.g. prometheus timeseries -> inline
        # chart). Kept in the final event rather than streamed so it
        # arrives only AFTER the agent commits to its answer -- avoids
        # racing a chart event ahead of an aborted tool path.
        "tool_message_history": result.get("tool_message_history") or [],
        "plan": result.get("plan") or [],
    }
