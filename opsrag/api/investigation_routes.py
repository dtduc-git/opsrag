"""FastAPI router for the hypothesis-driven investigation agent.

Two endpoints:
  - POST /investigate          -- run to completion, return the full state
  - POST /investigate/stream   -- SSE; emit a JSON line per TraceEvent so
                                  the FE can animate the tree as it grows

The investigation subgraph itself is in `opsrag/agents/investigation/`.
This module is the thin glue that adapts the project's vector store +
embedder + LLM router to the subgraph's contract.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from opsrag.agents.investigation import build_investigation_graph
from opsrag.agents.investigation.alert_extractor import extract_alert_context
from opsrag.agents.investigation.observability import build_investigation_summary
from opsrag.agents.investigation.result_cache import InvestigationResultCache
from opsrag.agents.investigation.rootly_resolver import (
    AlertEnrichment,
)
from opsrag.agents.investigation.rootly_resolver import (
    resolve_alert as rootly_resolve_alert,
)
from opsrag.agents.investigation.runbook_grounded import generate_runbook_hypotheses
from opsrag.agents.investigation.state import (
    AlertContext,
    HypothesisNode,
    InvestigationState,
)

_log = logging.getLogger("opsrag.api.investigation")
router = APIRouter()


# --- request / response shapes ------------------------------------


class InvestigateRequest(BaseModel):
    alert_text: str = Field(..., min_length=1, max_length=4000)
    service_hint: str | None = None
    namespace_hint: str | None = None
    env_hint: str | None = None
    runbook_urls: list[str] = Field(default_factory=list)


class InvestigateNode(BaseModel):
    """Serialized HypothesisNode -- drops the embedding (heavy + not
    needed in the UI). Adds `evidence_count` for cheap list views."""

    id: str
    statement: str
    status: str
    depth: int
    parent_id: str | None
    children: list[str]
    confidence: float
    judge_rationale: str
    termination_reason: str | None
    evidence_count: int
    evidence: list[dict[str, Any]]
    hypothesis_source: str = "llm"  # "llm" | "runbook" | "past_investigation"


class InvestigateResponse(BaseModel):
    investigation_id: str
    alert_text: str
    service_hint: str | None
    namespace_hint: str | None
    env_hint: str | None
    bootstrap_findings: list[str]
    nodes: list[InvestigateNode]
    root_ids: list[str]
    final_chain_node_ids: list[str]
    final_root_cause: str | None
    outcome: str
    summary: dict[str, Any]


# --- adapter helpers ----------------------------------------------


_RETRIEVE_CACHE_TTL_SEC = 120


def _build_retrieve_fn(vector_store, embedder):
    """Wrap the project vector store as a (query, top_k) -> [dict]
    callable matching the subgraph's `RetrieveFn` contract.

    Phase A -- within-investigation LRU. The same investigation often
    re-queries similar phrases as it generates parent + child
    hypotheses (cosine-near, sometimes identical after normalization).
    Cache key is (lowercased-trimmed query, top_k) with a 120s TTL
    so results don't go stale during a typical 2-5 min run.
    """
    # Per-RetrieveFn-instance cache -> one cache per request, no
    # cross-request pollution. Cleared automatically when the request
    # finishes and the closure goes out of scope.
    cache: dict[tuple[str, int], tuple[float, list[dict[str, Any]]]] = {}

    async def _retrieve(query: str, top_k: int) -> list[dict[str, Any]]:
        key = (query.lower().strip(), top_k)
        now = time.monotonic()
        if key in cache:
            ts, hit = cache[key]
            if now - ts < _RETRIEVE_CACHE_TTL_SEC:
                return hit
        try:
            emb = await embedder.embed_query(query)
            results = await vector_store.search(emb, top_k=top_k)
        except Exception as exc:
            _log.warning("retrieve failed for %r: %s", query[:60], exc)
            return []
        out: list[dict[str, Any]] = []
        for r in results:
            chunk = r.chunk
            content = (chunk.content or "")[:400]
            source_path = chunk.source_path or ""
            repo = chunk.repo or ""
            source_id = f"{repo}:{source_path}" if repo else source_path
            out.append({
                "chunk_id": chunk.id,
                "source_id": source_id,
                "snippet": content,
                "score": float(r.score),
                "repo": repo,
            })
        cache[key] = (now, out)
        return out

    return _retrieve


# --- Phase B: result cache (past investigations as context) -------


def _get_result_cache(app_state) -> InvestigationResultCache | None:
    """Lazy-init the result cache on app.state so it's shared across
    requests. Returns None if Qdrant isn't available -- the agent then
    runs without past-investigation context (graceful degradation)."""
    existing = getattr(app_state, "agent_investigation_cache", None)
    if existing is not None:
        return existing
    try:
        from qdrant_client import AsyncQdrantClient
        cfg = getattr(app_state, "config", None)
        url = (
            getattr(getattr(cfg, "vector_store", None), "url", None)
            or os.environ.get("QDRANT_URL", "http://qdrant:6333")
        )
        client = AsyncQdrantClient(url=url)
        cache = InvestigationResultCache(qdrant=client)
        app_state.agent_investigation_cache = cache
        _log.info("agent investigation result cache wired (collection=opsrag_agent_investigations)")
        return cache
    except Exception as exc:
        _log.warning("result cache init failed: %s -- continuing without past context", exc)
        app_state.agent_investigation_cache = None
        return None


async def _resolve_alert_hints(
    app_state, body: InvestigateRequest,
) -> tuple[InvestigateRequest, AlertEnrichment | None]:
    """Resolve service/namespace/env (and runbook URL) for the alert.

    Precedence:
      1. Rootly -- paste URL or matching free-text title. Gives the
         richest payload: structured fields + PromQL expression + runbook URL.
      2. Hybrid heuristic + Flash LLM extractor (fallback when Rootly
         has no matching alert OR is unavailable).
      3. Defaults from the LLM extractor (env=prod, ns=service).

    Returns the updated request AND the Rootly enrichment (None when we
    fell through to the LLM extractor) so the caller can also seed
    runbook hypotheses from the enrichment's `runbook_url`.
    """
    providers = getattr(app_state, "providers", None)
    llm = getattr(providers, "llm", None) if providers else None
    enrichment: AlertEnrichment | None = None

    # Step 1: Try Rootly (URL or title match).
    try:
        enrichment = await rootly_resolve_alert(body.alert_text)
    except Exception as exc:  # noqa: BLE001
        _log.warning("rootly resolve failed (%s) -- falling back", exc)
        enrichment = None

    if enrichment and enrichment.is_useful():
        _log.info(
            "alert resolved via %s service=%s ns=%s env=%s runbook=%s",
            enrichment.match_source, enrichment.service, enrichment.namespace,
            enrichment.env, bool(enrichment.runbook_url),
        )
        new_runbook_urls = list(body.runbook_urls or [])
        if enrichment.runbook_url and enrichment.runbook_url not in new_runbook_urls:
            new_runbook_urls.append(enrichment.runbook_url)
        updated = body.model_copy(update={
            "service_hint": body.service_hint or enrichment.service,
            "namespace_hint": body.namespace_hint or enrichment.namespace or enrichment.service,
            "env_hint": body.env_hint or enrichment.env or "prod",
            "runbook_urls": new_runbook_urls,
        })
        return updated, enrichment

    # Step 2: Hybrid extractor (current behaviour).
    if body.service_hint and body.namespace_hint and body.env_hint:
        return body, None  # caller fully specified
    try:
        ctx = await extract_alert_context(
            body.alert_text,
            llm=llm,
            explicit={
                "service": body.service_hint,
                "namespace": body.namespace_hint,
                "env": body.env_hint,
            },
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning("alert auto-extract failed (%s) -- using request as-is", exc)
        return body, None
    _log.info(
        "alert auto-extract source=%s service=%s namespace=%s env=%s",
        ctx.source, ctx.service, ctx.namespace, ctx.env,
    )
    updated = body.model_copy(update={
        "service_hint": body.service_hint or ctx.service,
        "namespace_hint": body.namespace_hint or ctx.namespace,
        "env_hint": body.env_hint or ctx.env,
    })
    return updated, None


async def _seed_runbook_hypotheses(
    app_state, state: InvestigationState, enrichment: AlertEnrichment | None,
) -> None:
    """When Rootly resolution surfaced a runbook URL, fetch the
    Confluence page, ask Flash to enumerate candidate causes, and
    attach each as a root-level HypothesisNode with `hypothesis_source="runbook"`.

    No-op when:
      - no enrichment (LLM-extract fallback path),
      - no `runbook_url` in the enrichment,
      - Confluence creds missing,
      - page resolver fails / page too short,
      - relevance gate rejects,
      - 0 causes extracted.
    All silent -- investigation proceeds with LLM-only hypotheses.
    """
    if enrichment is None or not enrichment.runbook_url:
        return
    providers = getattr(app_state, "providers", None)
    llm = getattr(providers, "llm", None) if providers else None
    if llm is None:
        return
    try:
        nodes = await generate_runbook_hypotheses(
            state.alert_context.alert_text, enrichment.runbook_url, llm,
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning("runbook hypothesis injection failed (non-fatal): %s", exc)
        return
    for node in nodes:
        state.add_node(node)
    _log.info(
        "seeded %d runbook hypothesis node(s) for investigation_id=%s",
        len(nodes), state.alert_context.investigation_id,
    )


async def _lookup_past_investigations(app_state, alert_ctx: AlertContext) -> list[dict[str, Any]]:
    """Pre-fetch top-K similar past investigations to seed the
    hypothesis-gen prompt. Returns a list shaped for the prompt
    renderer in `graph._render_past_investigations()`."""
    cache = _get_result_cache(app_state)
    if cache is None:
        return []
    providers = getattr(app_state, "providers", None)
    if providers is None or providers.embedder is None:
        return []
    query_text = InvestigationResultCache.compose_query(
        alert_ctx.alert_text,
        alert_ctx.service_hint,
        alert_ctx.namespace_hint,
        alert_ctx.env_hint,
    )
    try:
        emb = await providers.embedder.embed_query(query_text)
    except Exception as exc:
        _log.warning("past investigations embed failed: %s", exc)
        return []
    try:
        hits = await cache.search(emb, top_k=3)
    except Exception as exc:
        _log.warning("past investigations search failed: %s", exc)
        return []
    return [
        {
            "investigation_id": h.investigation_id,
            "alert_text": h.alert_text,
            "final_root_cause": h.final_root_cause,
            "outcome": h.outcome,
            "validated_chain_summary": h.validated_chain_summary,
            "tool_calls_used": h.tool_calls_used,
            "similarity": h.adjusted_similarity,
            "age_days": h.age_days,
        }
        for h in hits
    ]


async def _store_investigation_result(app_state, state: InvestigationState) -> None:
    """Persist the completed investigation so future runs can reference
    it AND so the history sidebar can replay it. We store every
    outcome (validated / invalidated / inconclusive) -- the history UI
    needs all of them; the bootstrap-context retrieval ranks by cosine
    + decay so validated runs naturally outrank dead-ends.
    """
    cache = _get_result_cache(app_state)
    if cache is None:
        return
    providers = getattr(app_state, "providers", None)
    if providers is None or providers.embedder is None:
        return
    ac = state.alert_context
    query_text = InvestigationResultCache.compose_query(
        ac.alert_text, ac.service_hint, ac.namespace_hint, ac.env_hint,
    )
    try:
        emb = await providers.embedder.embed_query(query_text)
    except Exception as exc:
        _log.warning("result-store embed failed: %s", exc)
        return
    # Validated-chain summary: one short line per chain node.
    chain_summary: list[str] = []
    for nid in state.final_chain_node_ids:
        node = state.nodes_by_id.get(nid)
        if node is not None:
            chain_summary.append(node.statement[:200])
    # Tool-calls used: derive from the citation `source_id` prefixes
    # (e.g. "confluence:...", "rootly:...") so we know which retrievers
    # contributed to the validated chain.
    tools_used: set[str] = set()
    for nid in state.final_chain_node_ids:
        node = state.nodes_by_id.get(nid)
        if node is None:
            continue
        for c in node.evidence:
            prefix = (c.source_id or "").split(":", 1)[0].strip()
            if prefix:
                tools_used.add(prefix)
    # Serialize the full node tree (without embeddings) so history
    # replay can rehydrate every hypothesis card, not just the chain.
    nodes_full = [_serialize_node(n).model_dump() for n in state.nodes_by_id.values()]
    summary = build_investigation_summary(state)
    await cache.store(
        investigation_id=ac.investigation_id,
        alert_text=ac.alert_text,
        service_hint=ac.service_hint,
        namespace_hint=ac.namespace_hint,
        env_hint=ac.env_hint,
        embedding=emb,
        final_root_cause=state.final_root_cause or "",
        outcome=state.outcome,
        validated_chain_summary=chain_summary,
        tool_calls_used=sorted(tools_used),
        # Tier-B replay payload.
        nodes_full=nodes_full,
        root_ids=list(state.root_ids),
        final_chain_node_ids=list(state.final_chain_node_ids),
        bootstrap_findings=list(state.bootstrap_findings),
        summary=summary,
    )


def _serialize_node(node: HypothesisNode) -> InvestigateNode:
    return InvestigateNode(
        id=node.id,
        statement=node.statement,
        status=node.status,
        depth=node.depth,
        parent_id=node.parent_id,
        children=list(node.children),
        confidence=float(node.confidence),
        judge_rationale=node.judge_rationale,
        termination_reason=node.termination_reason,
        evidence_count=len(node.evidence),
        evidence=[c.model_dump() for c in node.evidence],
        hypothesis_source=node.hypothesis_source,
    )


def _serialize_state(state: InvestigationState) -> InvestigateResponse:
    nodes = [_serialize_node(n) for n in state.nodes_by_id.values()]
    summary = build_investigation_summary(state)
    return InvestigateResponse(
        investigation_id=state.alert_context.investigation_id,
        alert_text=state.alert_context.alert_text,
        service_hint=state.alert_context.service_hint,
        namespace_hint=state.alert_context.namespace_hint,
        env_hint=state.alert_context.env_hint,
        bootstrap_findings=state.bootstrap_findings,
        nodes=nodes,
        root_ids=list(state.root_ids),
        final_chain_node_ids=list(state.final_chain_node_ids),
        final_root_cause=state.final_root_cause,
        outcome=state.outcome,
        summary=summary,
    )


def _build_graph_for_request(app_state, *, embed_query_enabled: bool = True):
    """Compose the investigation subgraph against the live app state.

    `app_state.providers` is the `OpsRAGProviders` populated in
    server.create_app(). `app_state.model_router` has both Flash and
    optional Pro LLMs.
    """
    providers = getattr(app_state, "providers", None)
    if providers is None:
        raise HTTPException(503, "providers not initialized")
    model_router = getattr(app_state, "model_router", None)
    llm_flash = providers.llm  # always present, the cost-optimal default
    llm_pro = getattr(model_router, "pro_llm", None) if model_router else None

    retrieve = _build_retrieve_fn(providers.vector_store, providers.embedder)
    embed_query = providers.embedder.embed_query if embed_query_enabled else None

    return build_investigation_graph(
        retrieve=retrieve,
        llm_flash=llm_flash,
        llm_pro=llm_pro,
        embed_query=embed_query,
    )


# --- endpoints ----------------------------------------------------


@router.post("/investigate", response_model=InvestigateResponse)
async def investigate(request: Request, body: InvestigateRequest) -> InvestigateResponse:
    """Run the full investigation to completion, return the tree.

    For long alerts this may take 2-5 minutes wall-clock. Browsers/
    proxies with shorter idle timeouts should prefer `/investigate/stream`.
    """
    graph = _build_graph_for_request(request.app.state)
    body, enrichment = await _resolve_alert_hints(request.app.state, body)
    alert_ctx = AlertContext(
        alert_text=body.alert_text,
        runbook_urls=body.runbook_urls,
        service_hint=body.service_hint,
        namespace_hint=body.namespace_hint,
        env_hint=body.env_hint,
    )
    # Phase B: prime the state with past similar investigations so the
    # hypothesis-gen prompt sees them on its first call.
    past = await _lookup_past_investigations(request.app.state, alert_ctx)
    initial = InvestigationState(alert_context=alert_ctx, past_investigations=past)
    # Tier B: pre-seed root-level runbook-grounded hypotheses if the
    # alert had a runbook URL and Confluence could fetch + parse it.
    await _seed_runbook_hypotheses(request.app.state, initial, enrichment)
    try:
        result = await graph.ainvoke(initial)
    except Exception as exc:
        _log.exception("investigate failed")
        raise HTTPException(500, f"investigation failed: {exc}") from exc

    final_state = InvestigationState.model_validate(result)
    # Persist the result so the next investigation can reference it.
    try:
        await _store_investigation_result(request.app.state, final_state)
    except Exception as exc:
        _log.warning("result-store failed (non-fatal): %s", exc)
    return _serialize_state(final_state)


# -- streaming variant ----------------------------------------------


async def _stream_investigation(
    app_state, body: InvestigateRequest,
) -> AsyncIterator[str]:
    """Drive the graph via `astream(stream_mode="values")` and emit
    SSE events as each node completes.

    LangGraph's `ainvoke` returns only the final state; it does NOT
    mutate the input state object in-place, so the old polling-the-
    input approach saw nothing until completion (~3-5 min). Using
    `astream(stream_mode="values")` yields the merged state dict
    after every node -- we diff against the last emit and ship new/
    updated nodes + bootstrap findings as they land.
    """
    graph = _build_graph_for_request(app_state)
    body, enrichment = await _resolve_alert_hints(app_state, body)
    alert_ctx = AlertContext(
        alert_text=body.alert_text,
        runbook_urls=body.runbook_urls,
        service_hint=body.service_hint,
        namespace_hint=body.namespace_hint,
        env_hint=body.env_hint,
    )
    # Phase B: prime past-investigation context BEFORE the graph runs.
    past = await _lookup_past_investigations(app_state, alert_ctx)
    initial = InvestigationState(alert_context=alert_ctx, past_investigations=past)
    # Tier B: pre-seed runbook-grounded root hypotheses (silent skip on miss).
    await _seed_runbook_hypotheses(app_state, initial, enrichment)

    yield _sse("start", {
        "investigation_id": initial.alert_context.investigation_id,
        "alert_text": initial.alert_context.alert_text,
        "service_hint": initial.alert_context.service_hint,
        "namespace_hint": initial.alert_context.namespace_hint,
        "env_hint": initial.alert_context.env_hint,
        "past_investigations_count": len(past),
        # Enrichment provenance for the FE chip row.
        "match_source": enrichment.match_source if enrichment else "",
        "match_score": enrichment.match_score if enrichment else None,
        "runbook_url": (enrichment.runbook_url if enrichment else None),
        "runbook_seeded_count": sum(
            1 for n in initial.nodes_by_id.values() if n.hypothesis_source == "runbook"
        ),
        "promql_expression": (enrichment.promql_expression if enrichment else None),
    })

    emitted_event_idx = 0
    emitted_node_signature: dict[str, str] = {}  # node_id -> status:conf:evidence_count
    bootstrap_emitted = False
    final_state: InvestigationState | None = None

    def _signature(n: HypothesisNode) -> str:
        return f"{n.status}:{round(n.confidence, 3)}:{len(n.evidence)}:{n.termination_reason or ''}"

    try:
        async for snapshot in graph.astream(initial, stream_mode="values"):
            # `snapshot` is a dict matching InvestigationState schema.
            # Normalize via model_validate so downstream serialisers
            # work the same as the synchronous endpoint.
            try:
                state = InvestigationState.model_validate(snapshot)
            except Exception as exc:
                _log.warning("snapshot validate failed: %s", exc)
                continue
            final_state = state

            # Bootstrap findings -- emit once when they first appear.
            if not bootstrap_emitted and state.bootstrap_findings:
                yield _sse("bootstrap", {
                    "findings": state.bootstrap_findings,
                    "citations_count": len(state.bootstrap_citations),
                })
                bootstrap_emitted = True

            # Trace events -- emit any new ones.
            if len(state.agent_trace) > emitted_event_idx:
                for ev in state.agent_trace[emitted_event_idx:]:
                    yield _sse("trace", {
                        "event_type": ev.event_type,
                        "node_id": ev.node_id,
                        "timestamp": ev.timestamp.isoformat(),
                        "payload": ev.payload,
                    })
                emitted_event_idx = len(state.agent_trace)

            # Node deltas -- emit any new node OR node whose status/
            # confidence/evidence changed since last emit.
            for nid, node in state.nodes_by_id.items():
                sig = _signature(node)
                if emitted_node_signature.get(nid) == sig:
                    continue
                emitted_node_signature[nid] = sig
                yield _sse("node", _serialize_node(node).model_dump())
    except Exception as exc:
        _log.exception("investigate stream failed")
        yield _sse("error", {"detail": str(exc)})
        return

    if final_state is None:
        yield _sse("error", {"detail": "investigation produced no state"})
        return

    yield _sse("complete", _serialize_state(final_state).model_dump())

    # Phase B: persist after the SSE stream closes. Fires only for
    # validated outcomes (filtered inside _store_investigation_result).
    try:
        await _store_investigation_result(app_state, final_state)
    except Exception as exc:
        _log.warning("result-store failed (non-fatal): %s", exc)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/investigate/stream")
async def investigate_stream(request: Request, body: InvestigateRequest):
    """Server-Sent Events variant -- recommended for the UI."""
    return StreamingResponse(
        _stream_investigation(request.app.state, body),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# --- history endpoints (Tier B replay) ----------------------------


class HistoryItem(BaseModel):
    """Lightweight history-list entry -- no full tree. Use
    `/investigation/{id}` to fetch full payload for replay."""
    investigation_id: str
    alert_text: str
    service_hint: str | None
    namespace_hint: str | None
    env_hint: str | None
    outcome: str
    final_root_cause: str
    created_at: float
    age_seconds: float


class HistoryResponse(BaseModel):
    investigations: list[HistoryItem]


@router.get("/investigations", response_model=HistoryResponse)
async def list_investigations(request: Request, limit: int = 50) -> HistoryResponse:
    """Recent investigations, newest first. Drives the history sidebar."""
    cache = _get_result_cache(request.app.state)
    if cache is None:
        return HistoryResponse(investigations=[])
    items = await cache.list_recent(limit=max(1, min(limit, 200)))
    now = time.time()
    out = [
        HistoryItem(
            investigation_id=i["investigation_id"],
            alert_text=i["alert_text"],
            service_hint=i.get("service_hint") or None,
            namespace_hint=i.get("namespace_hint") or None,
            env_hint=i.get("env_hint") or None,
            outcome=i.get("outcome") or "",
            final_root_cause=i.get("final_root_cause") or "",
            created_at=float(i.get("created_at") or 0.0),
            age_seconds=max(0.0, now - float(i.get("created_at") or now)),
        )
        for i in items
    ]
    return HistoryResponse(investigations=out)


@router.get("/investigation/{investigation_id}", response_model=InvestigateResponse)
async def get_investigation(request: Request, investigation_id: str) -> InvestigateResponse:
    """Fetch a stored investigation's full state for tree-replay.

    Returns the same shape as `POST /investigate` -- `nodes`,
    `root_ids`, `final_chain_node_ids`, `bootstrap_findings`,
    `final_root_cause`, `summary`. The UI can drive the existing
    react-flow renderer off this without distinguishing live vs.
    replayed.
    """
    cache = _get_result_cache(request.app.state)
    if cache is None:
        raise HTTPException(503, "investigation cache not configured")
    payload = await cache.get_full(investigation_id)
    if not payload:
        raise HTTPException(404, f"investigation {investigation_id!r} not found")
    # Rehydrate nodes -- they were serialized via _serialize_node which
    # produced the same shape as InvestigateNode.
    raw_nodes = payload.get("nodes_full") or []
    nodes = [InvestigateNode.model_validate(n) for n in raw_nodes if isinstance(n, dict)]
    return InvestigateResponse(
        investigation_id=investigation_id,
        alert_text=str(payload.get("alert_text", "")),
        service_hint=payload.get("service_hint") or None,
        namespace_hint=payload.get("namespace_hint") or None,
        env_hint=payload.get("env_hint") or None,
        bootstrap_findings=list(payload.get("bootstrap_findings") or []),
        nodes=nodes,
        root_ids=list(payload.get("root_ids") or []),
        final_chain_node_ids=list(payload.get("final_chain_node_ids") or []),
        final_root_cause=payload.get("final_root_cause") or None,
        outcome=str(payload.get("outcome") or ""),
        summary=dict(payload.get("summary") or {}),
    )
