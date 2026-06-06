"""Corpus retrieval as an MCP tool -- `knowledge_search`.

The multi-agent path is a tool-only loop and does not run the standard
RAG retrieval graph. When the user pastes a Slack URL that turns out to
be an SRE Support Request, or an alert that maps to a known runbook,
the agent needs corpus access to answer "what does our policy say
about this?" / "what's the runbook for X?" without falling back to
parametric memory.

This tool wraps the same embedder + Qdrant store the chat agent uses
during normal retrieval, so the boost-weighting (SRE-KB 1.5x,
user-correction 2.5x) and BM25 hybrid lane both work the same way as
the standard `vector_retrieve_node` lane.

Wiring lives in `opsrag.api.server.lifespan` -- at startup the providers
are bound here via `bind(embedder, vector_store)`. Before binding,
calling the tool returns a structured error rather than crashing.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from opsrag.mcp.gitlab import MCPTool

_log = logging.getLogger("opsrag.mcp.knowledge")

# Listing intent -- when the user asks for "all X" / "every X" / "list of
# X" / "each X", the answer needs to enumerate every bucket. Top-K=15
# routinely crowds the earliest buckets out (e.g. "all SRE cycles in
# 2025" returned 5 chunks, all 2026 cycles, because 2026 prose ranked
# higher than 2025 prose under semantic similarity). Bumping K to 30
# gives the reasoner enough surface area to spot the missing buckets
# and issue per-bucket follow-up searches (see _SYSTEM_REASONER_BASE).
_LISTING_INTENT_RE = re.compile(
    r"\b(all|every|each|list of|across all|for all|of all)\s+\w+",
    re.IGNORECASE,
)
_LISTING_INTENT_TOP_K = 30


def _is_listing_intent(query: str) -> bool:
    """Cheap regex check: does the query ask to enumerate a bucket set?

    Conservative -- only matches well-formed listing patterns so we don't
    over-fetch on ordinary single-target queries.
    """
    return bool(_LISTING_INTENT_RE.search(query or ""))

# Lazily-bound state set by server.lifespan. We avoid a class so the
# tool list at import time still has the right shape.
_embedder: Any | None = None
_vector_store: Any | None = None
# P3 -- optional code-specific embedder + code collection. When set,
# `_h_knowledge_search` embeds the query with the code embedder
# (CODE_RETRIEVAL_QUERY task type) and passes both signals into
# `hybrid_search` so the code lane joins the RRF fusion.
_code_embedder: Any | None = None
_code_vector_store: Any | None = None


def bind(
    embedder: Any,
    vector_store: Any,
    code_embedder: Any | None = None,
    code_vector_store: Any | None = None,
) -> None:
    """Inject providers so the handler can serve queries. Called once
    from server lifespan after providers are constructed.

    Optional providers:
      - `code_embedder` + `code_vector_store`: enable the P3 code lane
        (dual-write `opsrag_code` collection, embedded with
        gemini-embedding-001 + CODE_RETRIEVAL_QUERY)

    Missing providers degrade silently -- the handler always falls back
    to the lanes it has.

    Note: the graph-anchored retrieval lane (P1 #4) was removed
    2026-05-23. The wiring is gone but the Neo4j driver remains for
    Cartography integration.
    """
    global _embedder, _vector_store, _code_embedder, _code_vector_store
    _embedder = embedder
    _vector_store = vector_store
    _code_embedder = code_embedder
    _code_vector_store = code_vector_store
    _log.info(
        "knowledge_search bound: embedder=%s vector_store=%s code_embedder=%s code_vector_store=%s",
        type(embedder).__name__,
        type(vector_store).__name__,
        type(code_embedder).__name__ if code_embedder else "None",
        type(code_vector_store).__name__ if code_vector_store else "None",
    )


async def _retrieve_one_pool(query_str: str, k: int) -> list:
    """Run a single full hybrid_search for one (sub-)query. Returns a
    ranked list of SearchResult -- same shape regardless of whether
    code lane, etc. fired. Errors propagate to caller."""
    emb = await _embedder.embed_query(query_str)

    code_emb: list[float] | None = None
    if _code_embedder is not None and _code_vector_store is not None:
        try:
            code_emb = await _code_embedder.embed_query(query_str)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "code query embedding failed for %r (%s) -- proceeding without code lane",
                query_str[:60], exc,
            )
            code_emb = None

    if hasattr(_vector_store, "hybrid_search"):
        return await _vector_store.hybrid_search(
            emb, query_str, top_k=k,
            code_embedding=code_emb,
            code_store=_code_vector_store,
        )
    return await _vector_store.search(emb, top_k=k)


async def _h_knowledge_search(_unused, args: dict) -> Any:
    if _embedder is None or _vector_store is None:
        return {"error": "knowledge_search not configured -- embedder or vector_store unbound"}

    query = (args.get("query") or "").strip()
    if not query:
        return {"error": "query is required"}
    k = int(args.get("k") or 5)
    # Cap so a runaway agent can't grab 100 hits, but allow a wider
    # surface when the user's query has explicit listing intent
    # ("all X in 2025", "every cycle", "list of services"). At k=15
    # bucketed enumerations like "all SRE cycles in 2025" silently
    # drop entire years from the answer.
    upper_cap = _LISTING_INTENT_TOP_K if _is_listing_intent(query) else 15
    k = max(1, min(k, upper_cap))

    # -- T1.1 -- Multi-query decomposition ----------------------------
    # Decide whether to fan out into 2-4 parallel sub-query retrievals.
    # Feature-flagged via OPSRAG_DECOMPOSE_QUERIES; default off so the
    # behavior matches pre-T1.1 exactly when disabled. When enabled,
    # the decomposer skips trivially-single-target queries via a regex
    # heuristic, so the LLM cost only fires on actual multi-target asks
    # ("compare staging vs prod", "X and how it integrates with Y").
    decomposition_meta: dict[str, Any] = {}
    try:
        from opsrag.agent.query_decomposer import decompose_query
        # Graph-lane LLM wiring removed 2026-05-23; decomposer falls back
        # to the regex heuristic when llm is None, so the common case is
        # unaffected. If we want LLM-driven decomposition back, callers
        # can pass an LLM via `bind()` again.
        decomp = await decompose_query(query, None)
        sub_queries = decomp.sub_queries
        decomposition_meta = {
            "n_sub_queries": len(sub_queries),
            "reason": decomp.reason,
            "used_llm": decomp.used_llm,
        }
    except Exception as exc:  # noqa: BLE001
        _log.warning("query decomposition failed (%s) -- using original query only", exc)
        sub_queries = [query]
        decomposition_meta = {"n_sub_queries": 1, "reason": f"decomp error: {exc}"}

    # -- Retrieve ----------------------------------------------------
    # When sub_queries == [query] (the common case), this is a single
    # hybrid_search call -- same cost / latency as pre-T1.1. When the
    # decomposer fanned out, asyncio.gather runs the parallel retrievals
    # concurrently and cross-pool RRF merges them.
    pool_k = max(k * 2, 10) if len(sub_queries) > 1 else k
    try:
        if len(sub_queries) == 1:
            results = await _retrieve_one_pool(sub_queries[0], pool_k)
        else:
            pools = await asyncio.gather(
                *(_retrieve_one_pool(sq, pool_k) for sq in sub_queries),
                return_exceptions=True,
            )
            # Drop pools that errored; if all errored, fall back to a
            # single retrieval of the original query so we never return
            # empty just because one sub-query failed.
            good_pools: list[list] = []
            for sq, pool in zip(sub_queries, pools):
                if isinstance(pool, Exception):
                    _log.warning("sub-query pool failed for %r: %s", sq[:60], pool)
                    continue
                good_pools.append(pool)
            if not good_pools:
                _log.warning("all sub-query pools failed -- single-query fallback")
                results = await _retrieve_one_pool(query, pool_k)
            else:
                from opsrag.vectorstores.rrf import rrf_merge_pools
                results = rrf_merge_pools(good_pools, top_k=k)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"retrieval failed: {exc}"}

    hits: list[dict] = []
    for r in results or []:
        chunk = getattr(r, "chunk", None) or r
        content = (getattr(chunk, "content", "") or "")[:1200]
        source = getattr(chunk, "source_path", None) or getattr(chunk, "id", "")
        repo = getattr(chunk, "repo", None)
        meta = getattr(chunk, "metadata", {}) or {}
        # Source-type-specific URL: Confluence sets `page_url`, Slack sets
        # `permalink`, GitLab chunks may have `web_url`. Surface whichever
        # is present so the agent can cite a clickable link instead of
        # asking the user to look up the source manually. Dev feedback
        # 2026-05-15.
        url = (
            meta.get("page_url")
            or meta.get("permalink")
            or meta.get("permalink_url")
            or meta.get("web_url")
            or meta.get("url")
        )
        # Human-readable title (Confluence page title / Slack channel name /
        # GitLab file name). Improves the agent's ability to format citations.
        title = (
            meta.get("page_title")
            or meta.get("title")
            or meta.get("name")
        )
        hit: dict = {
            "source": source,
            "repo": repo,
            "priority": meta.get("priority"),
            "score": float(getattr(r, "score", 0.0)),
            "content": content,
        }
        # Only include url/title when set -- keeps the result tight for
        # sources that don't have them (e.g. raw GitLab Markdown).
        if url:
            hit["url"] = url
        if title:
            hit["title"] = title
        hits.append(hit)

    out: dict[str, Any] = {
        "query": query,
        "count": len(hits),
        "results": hits,
    }
    # T1.1 -- surface decomposition meta only when it actually fired
    # (used_llm=True). Skipping the field on no-op decomposition keeps
    # the response shape stable for the common single-target case.
    if decomposition_meta.get("used_llm"):
        out["decomposition"] = decomposition_meta
        out["sub_queries"] = sub_queries
    return out


# --- fake backend (FR-012; integration tests) ----------------------
#
# Data path (b): the handler reads module-level `_embedder` /
# `_vector_store` bound at startup via `bind()`. There is no client
# object -- `_h_knowledge_search` ignores its first arg. So the fake
# installs a fake embedder + fake vector store via `bind()` and restores
# the previous module state on teardown. No embeddings, no Qdrant, no
# network.


class _FakeChunk:
    """Shape-faithful stand-in for a corpus chunk. The handler reads
    `content`, `source_path` / `id`, `repo`, and `metadata`."""

    def __init__(
        self,
        content: str,
        source_path: str,
        repo: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.content = content
        self.source_path = source_path
        self.id = source_path
        self.repo = repo
        self.metadata = metadata or {}


class _FakeResult:
    """Shape-faithful stand-in for a SearchResult. The handler reads
    `chunk` and `score`."""

    def __init__(self, chunk: _FakeChunk, score: float) -> None:
        self.chunk = chunk
        self.score = score


# A few canned, shape-faithful hits with text / source / score.
_FAKE_HITS: list[_FakeResult] = [
    _FakeResult(
        _FakeChunk(
            content="SRE access requests are handled via the support queue; "
            "an on-call engineer grants time-boxed access.",
            source_path="kb/sre/access-requests.md",
            repo="sre-kb",
            metadata={
                "priority": "user-correction",
                "page_title": "Access Requests",
                "page_url": "https://docs.example.com/access-requests",
            },
        ),
        score=0.92,
    ),
    _FakeResult(
        _FakeChunk(
            content="Runbook: on a HighErrorRate alert, check the deploy log "
            "then roll back to the last known-good revision.",
            source_path="kb/runbooks/high-error-rate.md",
            repo="sre-kb",
            metadata={"title": "HighErrorRate runbook"},
        ),
        score=0.81,
    ),
    _FakeResult(
        _FakeChunk(
            content="Production change policy requires a peer-reviewed merge "
            "request and a green pipeline before deploy.",
            source_path="kb/policy/prod-change.md",
            repo=None,
            metadata={},
        ),
        score=0.67,
    ),
]


class _FakeEmbedder:
    """Offline embedder. Returns a fixed-width vector, no model load."""

    async def embed_query(self, query: str) -> list[float]:
        return [0.0, 0.0, 0.0]


class _FakeVectorStore:
    """Offline vector store. Ignores the embedding and returns canned,
    ranked hits (top-k slice), mirroring the `search` surface the handler
    falls back to when `hybrid_search` is absent."""

    async def search(self, embedding: list[float], top_k: int = 5) -> list:
        return list(_FAKE_HITS[: max(1, top_k)])


def build_fake():
    """Return a FakeMCP exposing knowledge_search bound to an offline
    embedder + vector store. Restores prior module state on teardown."""
    from opsrag.mcp._fake import FakeMCP

    prev = (_embedder, _vector_store, _code_embedder, _code_vector_store)

    bind(_FakeEmbedder(), _FakeVectorStore())

    def _restore() -> None:
        global _embedder, _vector_store, _code_embedder, _code_vector_store
        _embedder, _vector_store, _code_embedder, _code_vector_store = prev

    return FakeMCP(
        tools=list(KNOWLEDGE_TOOLS), client=None, teardown=_restore
    )


KNOWLEDGE_TOOLS: list[MCPTool] = [
    MCPTool(
        name="knowledge_search",
        description=(
            "Search the OpsRAG corpus (SRE knowledge base, Confluence, "
            "Slack threads, runbooks, Terraform/Helm docs, postmortems) "
            "for documentation relevant to the user's question. Use this "
            "whenever you need to know:\n"
            "  - 'How do we handle <X>?' (e.g. SRE access requests, "
            "Ack patterns, on-call escalation, Production change policy)\n"
            "  - 'What's the runbook for <alert>?'\n"
            "  - 'What does the Confluence doc for <topic> say?'\n"
            "  - 'How is service <Y> deployed / configured?'\n\n"
            "Especially relevant AFTER a Slack URL fetch reveals an SRE "
            "support request or an alert -- chain into this tool to find "
            "the policy/runbook that explains the next steps, rather "
            "than guessing.\n\n"
            "Returns the top-k chunks with their source path, repo, and "
            "(when present) `priority` tag -- 'user-correction' chunks "
            "are operator-authored ground truth and should be cited first."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language question or keyword phrase. Be specific; the embedder is paraphrase-tolerant but topic-narrow.",
                },
                "k": {
                    "type": "integer",
                    "description": "Top-k chunks to return (default 5, max 15).",
                },
            },
            "required": ["query"],
        },
        handler=_h_knowledge_search,
    ),
]
