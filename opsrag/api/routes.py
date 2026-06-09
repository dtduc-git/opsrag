"""FastAPI route handlers."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from opsrag import __version__
from opsrag.agent.graph import (
    SourceUrlBases,
    query_with_session,
    query_with_session_events,
)
from opsrag.api.models import (
    AgentGuidanceRequest,
    AgentGuidanceResponse,
    CachePurgeRequest,
    CachePurgeResponse,
    CacheSummaryResponse,
    CorrectionListItem,
    CorrectionListResponse,
    CorrectionRequest,
    CorrectionResponse,
    CorrectionReviewResponse,
    FeedbackListItem,
    FeedbackListResponse,
    HealthResponse,
    IndexRepoRequest,
    IndexRepoResponse,
    IndexSourceRequest,
    IndexSourceResponse,
    InvestigationCacheSummary,
    InvestigationFeedbackRequest,
    InvestigationFeedbackResponse,
    PendingCorrectionItem,
    PendingCorrectionListResponse,
    QueryRequest,
    QueryResponse,
    SessionListResponse,
    SessionSummary,
    UIConfigResponse,
)
from opsrag.auth import (
    CurrentUser,
    current_user_oid_var,
    get_current_user_dep,
)
from opsrag.investigations.feature_gate import investigation_live_telemetry_enabled
from opsrag.auth.scopes import Scope, require_scope
from opsrag.indexing_tracker import indexing_tracker
from opsrag.usage import tracker as usage_tracker

_log = logging.getLogger("opsrag.routes")

router = APIRouter()


def _owner_id_for(current_user: "CurrentUser", req_user_id: str | None) -> str:
    """Resolve the OWNER id to persist on a session's checkpoints.

    Security: the persisted owner MUST be the authenticated identity, never
    the client-supplied ``req.user_id`` (which is spoofable). When the user
    is authenticated and carries an oid, bind the owner to it. In open /
    anonymous mode we fall back to the client value (or "anonymous") to
    preserve zero-config dev behavior. ``req.user_id`` may still feed
    memory/personalization, but the OWNER binding uses the verified id.
    """
    if not current_user.is_anonymous and current_user.oid:
        return current_user.oid
    return req_user_id or "anonymous"


def _is_real_owner(owner: str | None) -> bool:
    """True iff ``owner`` is a real, lockable identity (not a legacy /
    pre-auth placeholder). MIGRATION/grandfather: threads whose owner is
    empty or "anonymous" predate owner binding and stay accessible -- we
    cannot retroactively assign them an owner, so only threads with a real
    authenticated owner are guarded."""
    return bool(owner) and owner != "anonymous"


def _deny_if_not_owner(current_user: "CurrentUser", owner: str | None) -> None:
    """Enforce per-session ownership for a single-thread read/delete.

    Raises 404 (NOT 403 -- avoids an existence oracle) when the caller is
    authenticated AND the thread has a real owner that isn't them. Open /
    anonymous mode does not enforce (preserves dev behavior); legacy
    anonymous-owned threads (see _is_real_owner) stay accessible.
    """
    if current_user.is_anonymous or not current_user.oid:
        return
    # Grandfather: only real authenticated owners are locked down.
    if _is_real_owner(owner) and owner != current_user.oid:
        raise HTTPException(status_code=404, detail="session not found")


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@router.get("/ui-config", response_model=UIConfigResponse)
async def ui_config(request: Request) -> UIConfigResponse:
    """White-label / runtime UI config. Reads `OpsRAGConfig.brand` plus
    the source-link bases. The React UI fetches this on boot so the same
    image works for any tenant by changing env vars at deploy time."""
    cfg = getattr(request.app.state, "config", None)
    if cfg is None:
        # No app config bound (test fixture, dev mode) -- return safe defaults.
        from opsrag.config import OpsRAGConfig
        cfg = OpsRAGConfig()
    return UIConfigResponse(
        brand_name=cfg.brand.name,
        brand_subtitle=cfg.brand.subtitle,
        assistant_name=cfg.brand.assistant_name,
        favicon_url=cfg.brand.favicon_url,
        accent_color=cfg.brand.accent_color,
        confluence_base_url=cfg.confluence.base_url,
        slack_workspace_url=cfg.slack.workspace_url,
        rootly_web_url=cfg.rootly.web_base_url,
        gitlab_base_url=cfg.scm.base_url,
        # Show the strongest reasoning model (pro/escalation) when set, else
        # the default llm -- so the footer reflects what answers complex queries.
        model_name=(cfg.agent.pro_model or cfg.llm.model),
        # Config-driven feature gate: only surface the Investigate tab when a
        # live-telemetry MCP integration is enabled (the operator's pick).
        investigation_enabled=investigation_live_telemetry_enabled(cfg),
    )


@router.get("/usage")
async def usage(request: Request) -> dict:
    """Token usage summary -- per-model breakdown with cost estimates.

    Reads directly from the shared Postgres event table so the answer is
    POD-AGNOSTIC: backend + indexer both record into the same table, but
    each pod's in-memory tracker only sees its own post-startup work.
    Indexer-side work (embed-index, contextual-chunk) would otherwise be
    invisible when this endpoint is served by the backend pod, even
    though the FE expects a unified view. Falls back to the in-memory
    tracker if persistence isn't available (e.g. local dev without DB).
    """
    persistence = getattr(request.app.state, "usage_persistence", None)
    if persistence is not None:
        try:
            summary = await persistence.get_summary()
            if summary is not None:
                return summary
        except Exception as exc:
            import logging
            logging.getLogger("opsrag.routes").warning(
                "usage Postgres read failed, falling back to in-memory tracker: %s",
                exc,
            )
    return usage_tracker.get_summary()


@router.get("/usage/weekly")
async def usage_weekly(request: Request) -> dict:
    """Per-week token + cost buckets for the Home dashboard mini chart.

    Reads from the shared Postgres event table (pod-agnostic, same as
    ``/usage``). Returns ``{"weeks": [...]}`` oldest-first with the
    current week last; each entry carries ``week_start`` (ISO date),
    ``tokens``, ``input_tokens``, ``output_tokens``, ``call_count`` and
    ``cost_usd``. Empty list when persistence isn't configured (local
    dev without DB) so the UI renders a graceful empty state.
    """
    persistence = getattr(request.app.state, "usage_persistence", None)
    if persistence is not None:
        try:
            weeks = await persistence.weekly_series(weeks=6)
            if weeks is not None:
                return {"weeks": weeks}
        except Exception as exc:
            import logging
            logging.getLogger("opsrag.routes").warning(
                "usage weekly read failed: %s", exc
            )
    return {"weeks": []}


@router.get("/indexing/status")
async def indexing_status(request: Request) -> dict:
    """Indexing progress -- per-repo file/chunk counts and status.

    Reads the durable Postgres job-state when available so every backend
    replica returns the SAME view (the in-memory tracker was per-pod -> the
    UI flickered between inconsistent states). Falls back to the in-memory
    tracker for dev / no-Postgres setups or on a DB hiccup."""
    store = getattr(request.app.state, "index_store", None)
    if store is not None:
        try:
            return await store.read_summary()
        except Exception as exc:
            logging.getLogger("opsrag.routes").warning(
                "index-state read failed, falling back to in-memory tracker: %s", exc
            )
    return indexing_tracker.get_summary()


@router.get("/indexing/jobs")
async def indexing_jobs(request: Request) -> dict:
    """Indexing run *history* -- newest-first list of jobs with start time,
    duration, success/failed status, and the error on failure. Powers the
    Operations -> Indexing Jobs page (distinct from /indexing/status, which
    is the current per-source catalog behind the Sources page)."""
    store = getattr(request.app.state, "index_store", None)
    if store is not None:
        try:
            return await store.read_jobs()
        except Exception as exc:
            logging.getLogger("opsrag.routes").warning(
                "index-state jobs read failed, falling back to in-memory tracker: %s", exc
            )
    return indexing_tracker.get_jobs()


@router.get("/graph/stats")
async def graph_stats(request: Request) -> dict:
    """Knowledge-graph backend status + schema.

    Powers the UI "Knowledge Graph" page. The graph store is
    provider-selected (config.knowledge_graph.provider); the default is
    ``none`` -> NullGraphStore, in which case ``enabled`` is False and the
    page renders an honest "no graph backend configured" state. When a
    real backend (e.g. neo4j) is wired, ``schema`` carries its node labels
    and relationship types so the page can render the live graph shape.
    """
    cfg = getattr(request.app.state, "config", None)
    providers = getattr(request.app.state, "providers", None)
    provider = "none"
    if cfg is not None:
        provider = getattr(getattr(cfg, "knowledge_graph", None), "provider", "none") or "none"
    enabled = provider != "none"
    schema: dict[str, Any] = {}
    store = getattr(providers, "graph_store", None) if providers is not None else None
    if enabled and store is not None:
        try:
            schema = await store.get_schema()
        except Exception as exc:  # never 500 the status page
            schema = {"error": str(exc)}
    else:
        # Neo4j off -> report the active lightweight entity-graph instead of a
        # bare "off", so the status header reflects what's actually running.
        lg = getattr(providers, "light_graph", None) if providers is not None else None
        if lg is not None:
            try:
                st = await lg.stats()
                return {
                    "provider": "entity-graph",
                    "enabled": st.get("edge_count", 0) > 0,
                    "edge_count": st.get("edge_count", 0),
                    "label_count": len(st.get("labels", [])),
                    "relationship_type_count": len(st.get("relationship_types", [])),
                    "labels": st.get("labels", []),
                    "relationship_types": st.get("relationship_types", []),
                }
            except Exception as exc:  # never 500 the status page
                _log.warning("light-graph stats failed: %s", exc)
    labels = schema.get("labels", []) if isinstance(schema, dict) else []
    rel_types = schema.get("relationship_types", []) if isinstance(schema, dict) else []
    return {
        "provider": provider,
        "enabled": enabled,
        "label_count": len(labels),
        "relationship_type_count": len(rel_types),
        "labels": labels,
        "relationship_types": rel_types,
    }


_GRAPH_VIEWS: dict[str, dict[str, list[str]]] = {
    "business": {
        "labels": ["Service", "Team", "Database", "Dependency"],
        "rels": ["OWNED_BY", "DEPENDS_ON", "USES_DATABASE"],
    },
    "public": {
        "labels": ["Gateway", "Route", "Host", "Middleware", "Service"],
        "rels": ["HAS_ROUTE", "ROUTES_TO", "HAS_HOST", "USES_MIDDLEWARE"],
    },
    "private": {
        "labels": [
            "Service", "Namespace", "Cluster", "Config", "Infra",
            "Repository", "Database",
        ],
        "rels": [
            "DEPENDS_ON", "IN_NAMESPACE", "IN_CLUSTER", "CONFIGURED_BY",
            "USES_DATABASE", "DEFINED_IN", "LIVES_IN", "RUNS_IN", "HOSTED_ON",
        ],
    },
}


@router.post("/admin/light-graph/backfill")
async def light_graph_backfill(
    request: Request,
    repo: str | None = None,
    _user: CurrentUser = Depends(require_scope(Scope.ADMIN)),
) -> dict:
    """Activate the entity-expansion lane on EXISTING chunks WITHOUT re-embedding.

    Scrolls the vector store, derives deterministic entity ids from each chunk's
    metadata (the cheap metadata lane), writes them to the chunk's `entity_ids`
    payload (set_payload -- no re-embed), and upserts the structured edges to the
    Postgres light graph. Optional ``repo`` scopes the backfill to one repo."""
    providers = getattr(request.app.state, "providers", None)
    lg = getattr(providers, "light_graph", None) if providers else None
    if lg is None:
        raise HTTPException(status_code=400, detail="light_graph disabled (set light_graph.enabled + restart)")
    vs = providers.vector_store
    client = getattr(vs, "_client", None)
    coll = getattr(vs, "_collection", None)
    if client is None or coll is None:
        raise HTTPException(status_code=503, detail="vector store is not Qdrant-backed")

    import qdrant_client.models as qm

    from opsrag.extractors.hybrid import entities_from_metadata

    flt = None
    if repo:
        flt = qm.Filter(must=[qm.FieldCondition(key="repo", match=qm.MatchValue(value=repo))])

    scanned = updated = 0
    edges: dict = {}
    offset = None
    while True:
        points, offset = await client.scroll(
            collection_name=coll, scroll_filter=flt, limit=500,
            offset=offset, with_payload=True, with_vectors=False,
        )
        if not points:
            break
        for p in points:
            scanned += 1
            pl = p.payload or {}
            meta = pl.get("metadata") or {}
            if not isinstance(meta, dict):
                continue
            meta = dict(meta)
            meta.setdefault("repo", pl.get("repo", ""))
            try:
                res = entities_from_metadata(meta, source_chunk_id=pl.get("chunk_id"))
            except Exception:
                continue
            ids = sorted({e.id for e in (getattr(res, "entities", []) or [])})
            for r in getattr(res, "relationships", []) or []:
                edges[(r.source_id, r.target_id, r.rel_type)] = r
            if ids:
                try:
                    await client.set_payload(collection_name=coll, payload={"entity_ids": ids}, points=[p.id])
                    updated += 1
                except Exception:
                    pass
        if offset is None:
            break
    # Full repo rebuild: wipe this repo's prior edges first so a removed/renamed
    # entity can't survive the rebuild. These aggregated edges aren't attributed
    # to a single file (source_path=''), which is fine -- delete_by_repo is the
    # matching refresh path for a whole-repo rebuild.
    if repo:
        await lg.delete_by_repo(repo)
    n_edges = await lg.upsert_edges(list(edges.values()), repo=repo or "")
    return {"repo": repo, "scanned": scanned, "points_updated": updated, "edges_upserted": n_edges}


@router.get("/admin/agent-guidance", response_model=AgentGuidanceResponse)
async def get_agent_guidance(
    request: Request,
    _user: CurrentUser = Depends(require_scope(Scope.ADMIN)),
) -> AgentGuidanceResponse:
    """Current deployment-wide custom instructions (the live, UI-editable value
    if an admin has saved one, else the `deployment.custom_instructions` config
    seed). Injected into the agent's answer + chat system prompts."""
    from opsrag.agent_settings import CUSTOM_INSTRUCTIONS_KEY
    store = getattr(request.app.state, "agent_settings", None)
    if store is not None:
        meta = await store.get_meta(CUSTOM_INSTRUCTIONS_KEY)
        if meta is not None:
            return AgentGuidanceResponse(
                custom_instructions=meta.get("value") or "",
                updated_at=meta.get("updated_at"),
                updated_by=meta.get("updated_by"),
                source="db",
            )
    # Fall back to the config seed.
    cfg = getattr(request.app.state, "config", None)
    seed = ((getattr(cfg, "deployment", None) and cfg.deployment.custom_instructions) or "").strip()
    return AgentGuidanceResponse(
        custom_instructions=seed, source="config" if seed else "none",
    )


@router.put("/admin/agent-guidance", response_model=AgentGuidanceResponse)
async def put_agent_guidance(
    req: AgentGuidanceRequest,
    request: Request,
    current_user: CurrentUser = Depends(require_scope(Scope.ADMIN)),
) -> AgentGuidanceResponse:
    """Save deployment-wide custom instructions. Persists to Postgres + updates
    the in-process prompt value immediately, so it takes effect on the NEXT
    query (no restart). Requires a Postgres-backed deployment."""
    from opsrag.agent.prompt_render import set_custom_instructions_live
    from opsrag.agent_settings import CUSTOM_INSTRUCTIONS_KEY
    store = getattr(request.app.state, "agent_settings", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "agent_settings_unavailable",
                    "reason": "agent guidance needs a Postgres-backed deployment"},
        )
    value = (req.custom_instructions or "").strip()
    await store.set(CUSTOM_INSTRUCTIONS_KEY, value, updated_by=current_user.email or current_user.oid)
    set_custom_instructions_live(value)   # effective on the next query, no restart
    meta = await store.get_meta(CUSTOM_INSTRUCTIONS_KEY)
    return AgentGuidanceResponse(
        custom_instructions=value,
        updated_at=(meta or {}).get("updated_at"),
        updated_by=(meta or {}).get("updated_by"),
        source="db",
    )


@router.get("/graph/view")
async def graph_view(
    request: Request,
    view: str = "business",
    limit: int = 300,
) -> dict:
    """Filtered subgraph for the Knowledge Graph UI's three views.

    ``view`` selects a fixed label/relationship-type filter
    (business | public | private); an unknown view is a 400. When the graph
    store is the Null/disabled backend, returns an empty subgraph with
    ``provider="disabled"``. Otherwise the provider mirrors
    ``config.knowledge_graph.provider`` (same source as /graph/stats).

    Like /graph/stats, this route carries NO per-route auth dependency: it
    relies on the global session/OIDC middleware so the (intentionally
    ungated) Knowledge Graph nav works for any authenticated user.
    """
    spec = _GRAPH_VIEWS.get(view)
    if spec is None:
        raise HTTPException(
            status_code=400,
            detail=f"unknown view {view!r}; expected one of "
            f"{sorted(_GRAPH_VIEWS)}",
        )

    cfg = getattr(request.app.state, "config", None)
    providers = getattr(request.app.state, "providers", None)
    provider = "none"
    if cfg is not None:
        provider = getattr(getattr(cfg, "knowledge_graph", None), "provider", "none") or "none"
    enabled = provider != "none"
    store = getattr(providers, "graph_store", None) if providers is not None else None

    if not enabled or store is None:
        # Neo4j off -> fall back to the always-on lightweight entity-graph
        # (opsrag_entity_edges) that actually powers entity-expansion. Its
        # labels/relations don't match the Neo4j-schema views, so we render
        # the whole entity graph (capped) under a single "entity-graph"
        # provider rather than the three rigid views.
        lg = getattr(providers, "light_graph", None) if providers is not None else None
        if lg is None:
            return {
                "provider": "disabled",
                "view": view,
                "truncated": False,
                "nodes": [],
                "edges": [],
            }
        try:
            nodes, edges, truncated = await lg.subgraph(limit=limit)
        except Exception as exc:  # never 500 the status page
            _log.warning("light-graph subgraph failed: %s", exc)
            nodes, edges, truncated = [], [], False
        return {
            "provider": "entity-graph",
            "view": view,
            "truncated": truncated,
            "nodes": nodes,
            "edges": edges,
        }

    nodes, edges = await store.view_subgraph(
        node_labels=spec["labels"],
        rel_types=spec["rels"],
        limit=limit,
    )
    return {
        "provider": provider,
        "view": view,
        "truncated": len(edges) == limit,
        "nodes": nodes,
        "edges": edges,
    }


@router.get("/integrations")
async def integrations(request: Request) -> dict:
    """Enumerate every MCP integration in the registry with its
    enabled/disabled state, tool count, and whether it exposes a health
    probe. Powers the UI "Integrations" page (Operations section).

    The set of *enabled* integrations is derived from the active config
    (same source ``/readyz`` uses for per-MCP probing); the full set comes
    from the static registry so disabled integrations are still listed as
    available-to-enable.
    """
    from opsrag.mcp.registry import REGISTRY
    from opsrag.mcp_server.registry_loader import enabled_integration_names

    cfg = getattr(request.app.state, "config", None)
    enabled_names = set(enabled_integration_names(cfg)) if cfg is not None else set()
    items: list[dict[str, Any]] = []
    for name, integ in sorted(REGISTRY.items()):
        items.append({
            "name": integ.name,
            "display_name": integ.display_name,
            "enabled": name in enabled_names,
            "tool_count": len(integ.tool_names),
            "tool_names": list(integ.tool_names),
            "has_health_probe": integ.health_url_template is not None,
            "required_env": list(integ.required_env),
        })
    return {
        "integrations": items,
        "enabled_count": len(enabled_names),
        "total": len(items),
    }


@router.get("/usage/{session_id}")
async def session_usage(session_id: str) -> dict:
    """Token usage for a specific session."""
    data = usage_tracker.get_session_usage(session_id)
    if data is None:
        return {"session_id": session_id, "usage": None}
    return {"session_id": session_id, "usage": data}


@router.post("/query")
async def query(
    req: QueryRequest,
    request: Request,
    current_user: CurrentUser = Depends(require_scope(Scope.CHAT)),
):
    graph = request.app.state.agent_graph

    cfg = getattr(request.app.state, "config", None)
    source_url_bases = SourceUrlBases.from_app_config(cfg) if cfg else None
    semantic_router = getattr(request.app.state, "semantic_router", None)

    # Bind the request's user_oid into a contextvar so the Vertex
    # `on_usage` hook (wired by the factory) can attribute the cost
    # without needing to thread user_oid through every node call site.
    # Anonymous / tracking-disabled requests set None, which the
    # persistence layer treats as "leave the column NULL".
    token = current_user_oid_var.set(current_user.oid)
    # Owner binding: persist the AUTHENTICATED identity as the session owner
    # (never the spoofable client-supplied req.user_id). Falls back to the
    # client value / "anonymous" in open mode. See _owner_id_for.
    owner_id = _owner_id_for(current_user, req.user_id)
    try:
        if req.stream:
            providers = request.app.state.providers
            qa_cache = getattr(request.app.state, "qa_cache", None)
            investigation_cache = getattr(request.app.state, "investigation_cache", None)
            return StreamingResponse(
                _stream_query(graph, req, providers, qa_cache, investigation_cache, source_url_bases, semantic_router, current_user.oid, current_user.email, current_user.name, owner_id),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        providers = request.app.state.providers
        qa_cache = getattr(request.app.state, "qa_cache", None)
        investigation_cache = getattr(request.app.state, "investigation_cache", None)

        try:
            result = await query_with_session(
                graph,
                query=req.query,
                user_id=owner_id,
                thread_id=req.thread_id,
                embedder=providers.embedder,
                qa_cache=qa_cache,
                llm=providers.llm,
                session_store=providers.session_store,
                investigation_cache=investigation_cache,
                source_url_bases=source_url_bases,
                semantic_router=semantic_router,
                user_email=current_user.email,
                user_name=current_user.name,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"query failed: {exc}") from exc
        # Drop fields QueryResponse doesn't declare (cache_hit, similarity, etc.)
        # -- they're for internal observability, not part of the public contract.
        public = {k: v for k, v in result.items() if k in QueryResponse.model_fields}
        return QueryResponse(**public)
    finally:
        # Streaming responses outlive this scope -- but StreamingResponse
        # runs in the same task so the contextvar stays valid until the
        # stream finishes. For non-stream we reset eagerly.
        if not req.stream:
            current_user_oid_var.reset(token)


# -- Inline chart extraction -----------------------------------------
#
# When the agent calls a Prometheus tool whose result is a matrix
# (range query) or a non-trivial vector (instant with multiple series),
# we want the UI to render an inline chart rather than have the LLM
# verbalize a hundred (ts, value) pairs. The agent stores tool results
# as JSON-stringified payloads under `response.text`; this helper
# unwraps them and emits the FE-friendly props.
#
# Returns a list of {"component", "props"} dicts. Empty list when there
# is nothing chartable.
#
# Friendly metric label heuristic: collapses "container_cpu_usage_seconds_total"
# to "container cpu usage" so the chart header reads naturally. Picks a
# unit from the metric name when obvious (bytes/seconds/percent).
_BYTES_HINTS = ("bytes", "memory")
_SECONDS_HINTS = ("seconds", "duration")
_PERCENT_HINTS = ("percent", "ratio")


import re as _re

# Pattern -> friendly verb-phrase. First match wins; ordered most-specific first
# (e.g. memory % must beat memory bytes). The matcher runs on the full query
# string (lowercased) so it sees both the metric name AND the wrapping rate()
# / sum() / division -- letting us tell "CPU usage" apart from "CPU % of limit".
_METRIC_PHRASE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"container_cpu_usage_seconds_total.*kube_pod_container_resource_limits.*resource=\"?cpu", "CPU %-of-limit"),
    (r"rate\([^)]*container_cpu_usage_seconds_total", "CPU usage (cores)"),
    (r"container_cpu_usage_seconds_total", "CPU usage"),
    (r"container_memory_working_set_bytes.*kube_pod_container_resource_limits.*memory", "Memory %-of-limit"),
    (r"container_memory_working_set_bytes", "Memory working set"),
    (r"container_memory_rss", "Memory RSS"),
    (r"kafka_consumergroup_lag", "Kafka consumer lag"),
    (r"kube_pod_container_status_restarts_total", "Pod restarts"),
    (r"kube_pod_container_status_waiting_reason.*crashloop", "CrashLoopBackOff pods"),
    (r"kube_deployment_status_replicas", "Deployment replicas"),
    (r"istio_requests_total.*response_code=~\"5", "HTTP 5xx rate"),
    (r"istio_requests_total", "Request rate"),
    (r"alerts\{alertstate=\"firing", "Firing alerts"),
)


def _query_scope_hint(query: str) -> str:
    """Extract a short scope hint from PromQL label selectors. Returns
    something like "acme-notes-be / acme-notes-be-appservice-main.*" so two charts
    of the same metric with different filters get distinguishable titles.
    Picks the most specific dimensions: namespace, then pod/topic/etc.
    """
    bits: list[str] = []
    for key in ("namespace", "pod", "topic", "consumergroup", "container", "service", "destination_service_name", "instance"):
        # Match  key="value"  OR  key=~"value-glob"
        m = _re.search(rf'{key}\s*=~?\s*"([^"]+)"', query or "")
        if m:
            bits.append(m.group(1))
    return " / ".join(bits[:2])  # at most 2 -- keep title short


def _metric_friendly_label(metric_name: str, query: str) -> str:
    """Pretty label from a Prometheus metric name + the original query.

    Strategy:
      1. Try a pattern map (CPU/memory/lag/restarts/...) on the full
         lowercased query -- recognises common SRE recipes by shape, not
         by relying on `__name__` which `sum by (...)` strips out.
      2. If the agent issued a custom query, fall back to the raw metric
         name (or a query token).
      3. Append a scope hint like `(acme-notes-be / acme-notes-be-appservice-main.*)`
         so two charts of the same metric with different filters don't
         collide on title.
    """
    q_low = (query or "").lower()
    base = ""
    for pat, phrase in _METRIC_PHRASE_PATTERNS:
        if _re.search(pat, q_low):
            base = phrase
            break
    if not base:
        if metric_name:
            base = metric_name.replace("_", " ").strip()
        else:
            m = _re.search(r"([a-zA-Z_][a-zA-Z0-9_]{4,})", query or "")
            base = (m.group(1).replace("_", " ") if m else "metric").strip()
    scope = _query_scope_hint(query or "")
    return f"{base} -- {scope}" if scope else base


def _metric_unit(metric_name: str, query: str) -> str | None:
    blob = f"{metric_name} {query}".lower()
    if any(h in blob for h in _BYTES_HINTS):
        return "bytes"
    if any(h in blob for h in _SECONDS_HINTS):
        return "seconds"
    if any(h in blob for h in _PERCENT_HINTS):
        return "percent"
    return None


def _series_label(metric_labels: dict) -> str:
    """Pick the most informative label for one series. Prefers the
    obvious dimension (pod, instance, container, consumergroup, topic)
    before falling back to a stringified metric dict."""
    if not isinstance(metric_labels, dict) or not metric_labels:
        return "series"
    for key in ("pod", "instance", "container", "consumergroup", "topic", "service", "destination_service_name", "node", "namespace"):
        if key in metric_labels:
            return str(metric_labels[key])
    # Pick the first non-`__name__` label so we don't return "metric=foo".
    for k, v in metric_labels.items():
        if k.startswith("__"):
            continue
        return f"{k}={v}"
    return str(metric_labels.get("__name__") or "series")


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        # Prometheus emits "NaN" as a JSON string in some clients; coerce
        # to None so the chart doesn't try to plot a NaN point.
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def _salvage_truncated_matrix_json(text: str) -> dict | None:
    """Best-effort recovery of a prom matrix-result tool payload that
    was cut mid-JSON by the downstream char-truncation. Strategy: find
    the last full `]}` (end of a complete series dict), then close out
    the matrix array + outer dict structure with literal closers. If
    the trim point is plausibly before the first complete series we
    bail. Cheap fallback; if it ever misbehaves the chart just won't
    show, which is the current behavior anyway.
    """
    if "\"resultType\": \"matrix\"" not in text:
        return None
    last_end = text.rfind("]}")
    if last_end <= 0:
        return None
    # Series objects look like `{"metric": {...}, "values": [[ts, "val"], ...]}`.
    # Truncating at last_end+2 yields `...]}` -- append `]}}` to close
    # result-array + data-dict + outer-dict.
    candidate = text[: last_end + 2] + "]}}"
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _extract_chart_components(tool_message_history: list[dict]) -> list[dict]:
    """Scan tool history for Prometheus results we can render inline.
    Only `prometheus_query_range` (matrix) is rendered today -- that's
    the data shape with enough points to justify a chart. Instant
    `prometheus_query` results are skipped because a single-timestamp
    bar of N labels is rarely more informative than the agent's prose.
    """
    components: list[dict] = []
    if not tool_message_history:
        return components

    for msg in tool_message_history:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "tool_result":
            continue
        name = msg.get("name") or ""
        if name not in ("prometheus_query_range", "prometheus_query"):
            continue
        resp = msg.get("response") or {}
        text = resp.get("text") if isinstance(resp, dict) else None
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            payload = json.loads(text)
        except Exception:
            # Result was truncated mid-JSON (large matrix tool outputs).
            # Try to salvage by trimming back to the last complete
            # series-end `]}` boundary before re-closing the matrix.
            payload = _salvage_truncated_matrix_json(text)
            if payload is None:
                continue
        if not isinstance(payload, dict) or payload.get("error"):
            continue
        data = payload.get("data") or {}
        result_type = data.get("resultType")
        results = data.get("result") or []
        if result_type != "matrix" or not results:
            # Skip instant queries (vector) -- see docstring.
            continue

        # Find the original query string from a prior tool_call entry
        # with the same name, so we can derive a friendly label/unit.
        # Walk backwards from the current msg index for the most recent
        # matching tool_call.
        query_str = ""
        idx = tool_message_history.index(msg)
        for i in range(idx - 1, -1, -1):
            prev = tool_message_history[i]
            if (
                isinstance(prev, dict)
                and prev.get("role") == "tool_call"
                and prev.get("name") == name
            ):
                args = prev.get("args") or {}
                query_str = args.get("query") or ""
                break

        # Pick the most informative metric_name from the first series
        # for the label heuristic.
        first_metric = (results[0].get("metric") or {}) if isinstance(results[0], dict) else {}
        metric_name = first_metric.get("__name__") or ""
        metric_label = _metric_friendly_label(metric_name, query_str)
        unit = _metric_unit(metric_name, query_str)

        # Flatten each series into points the chart component can plot.
        # Cap at 8 series + 500 points/series so a degenerate query
        # doesn't blow the SSE payload.
        max_series = 8
        max_points = 500
        series_out: list[dict] = []
        for s in results[:max_series]:
            if not isinstance(s, dict):
                continue
            labels = s.get("metric") or {}
            values = s.get("values") or []
            points = []
            for pair in values[:max_points]:
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                ts = _safe_float(pair[0])
                val = _safe_float(pair[1])
                if ts is None or val is None:
                    continue
                points.append({"ts": ts, "value": val})
            if points:
                series_out.append({
                    "label": _series_label(labels),
                    "labels": labels,
                    "points": points,
                })

        if not series_out:
            continue

        components.append({
            "component": "TimeseriesChart",
            "props": {
                "metric_label": metric_label,
                "unit": unit,
                "query": query_str,
                "series": series_out,
                "source": name,
            },
        })

    return components


async def _stream_query(graph, req: QueryRequest, providers, qa_cache, investigation_cache=None, source_url_bases: SourceUrlBases | None = None, semantic_router=None, user_oid: str | None = None, user_email: str | None = None, user_name: str | None = None, owner_id: str | None = None):
    """SSE generator. Emits node-level progress events from
    LangGraph's `astream_events` plus chunked answer for progressive
    rendering. Event taxonomy::

        event: status        -- overall lifecycle (started)
        event: node_start    -- agent entered a graph node
        event: node_end      -- agent finished a graph node
        event: cache_hit     -- short-circuit before graph (synthetic)
        event: chunk         -- final-answer chunk (existing client compat)
        event: done          -- full result payload
        event: error         -- fatal exception, stream closes
        event: close         -- clean stream-close marker
    """
    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    # Re-bind the contextvar inside the generator task so the
    # Vertex `on_usage` hook (which runs in this same task) can see the
    # current user_oid. The handler's `set()` doesn't carry into the
    # StreamingResponse generator because Starlette runs it on a fresh
    # task with a forked context.
    _uo_token = current_user_oid_var.set(user_oid)
    yield _sse("status", {"step": "started", "query": req.query})

    final: dict | None = None
    try:
        async for ev in query_with_session_events(
            graph,
            query=req.query,
            # Owner binding: prefer the authenticated owner_id computed in the
            # handler over the spoofable req.user_id (None only on legacy call
            # paths that don't pass it). See _owner_id_for.
            user_id=owner_id if owner_id is not None else req.user_id,
            thread_id=req.thread_id,
            embedder=providers.embedder,
            qa_cache=qa_cache,
            llm=providers.llm,
            session_store=providers.session_store,
            investigation_cache=investigation_cache,
            source_url_bases=source_url_bases,
            semantic_router=semantic_router,
            user_email=user_email,
            user_name=user_name,
        ):
            kind = ev.get("type")
            if kind == "node_start":
                yield _sse("node_start", {
                    "node": ev["node"], "label": ev["label"],
                })
            elif kind == "node_end":
                yield _sse("node_end", {
                    "node": ev["node"], "label": ev["label"],
                })
            elif kind == "reasoner_token":
                # Live token stream from the reasoner LLM -- UI appends
                # to the current step's body so users see thinking
                # unfold in real time.
                yield _sse("reasoner_token", {"delta": ev["delta"]})
            elif kind == "cache_hit":
                yield _sse("cache_hit", {
                    "similarity": ev["similarity"],
                    "age_seconds": ev["age_seconds"],
                })
            elif kind == "error":
                yield _sse("error", {"detail": ev["detail"]})
                return
            elif kind == "final":
                final = ev
    except Exception as exc:
        yield _sse("error", {"detail": str(exc)})
        return

    if final is None:
        yield _sse("error", {"detail": "no final result emitted"})
        return

    # Backward-compat: chunk the answer for clients that already render
    # progressive text via `chunk` events. UI v2 (ThinkingProgress) can
    # ignore these and render `final.answer` once `done` arrives.
    answer = final.get("answer", "") or ""
    chunk_size = 80
    for i in range(0, len(answer), chunk_size):
        yield _sse("chunk", {"text": answer[i : i + chunk_size]})

    # Renderable components -- inspect the final tool history for
    # Prometheus matrix results and emit a `render_component` event per
    # chartable tool call. Older clients ignore the unknown event type;
    # newer clients attach charts under the
    # current assistant message. Errors here MUST NOT break the stream;
    # the chart is a nice-to-have.
    try:
        components = _extract_chart_components(final.get("tool_message_history") or [])
        if components:
            _log.info("chart-extract: %d components produced", len(components))
        for comp in components:
            yield _sse("render_component", comp)
    except Exception as exc:
        _log.warning("chart extraction failed (non-fatal): %s", exc)

    # Emit the investigation plan as a renderable component, if
    # the reasoner used `update_plan` at any point during the turn.
    try:
        plan = final.get("plan") or []
        if plan:
            from opsrag.agent.services.plan_tool import to_sse_event
            yield _sse("render_component", to_sse_event(plan))
    except Exception as exc:
        _log.warning("plan render failed (non-fatal): %s", exc)

    yield _sse("done", {
        "answer": answer,
        "sources": final.get("sources", []),
        "source_urls": final.get("source_urls", []),
        "grounded": final.get("grounded", False),
        "query_type": final.get("query_type"),
        "thread_id": final.get("thread_id", ""),
        "cache_hit": final.get("cache_hit", False),
        "investigation_id": final.get("investigation_id"),
    })
    yield "event: close\ndata: {}\n\n"


@router.post("/index/repo", response_model=IndexRepoResponse)
async def index_repo(
    req: IndexRepoRequest, request: Request,
    _user: CurrentUser = Depends(require_scope(Scope.ADMIN)),
) -> IndexRepoResponse:
    """Queue a repo for indexing -- fire-and-forget so the UI can submit
    a form and watch /indexing/status for progress without holding the
    request open for the full duration (often minutes for large repos).

    Requires the ``admin`` scope (was previously only listed in the
    unregistered APIKeyAuthMiddleware ADMIN_ROUTES, i.e. ungated at runtime)."""
    pipeline = request.app.state.ingestion_pipeline
    qa_cache = getattr(request.app.state, "qa_cache", None)

    # Resolve the branch: explicit request value wins, else the configured
    # scm.default_branch (e.g. `master`), else "main". Prevents the
    # "Remote branch main not found" clone failure for master-default repos.
    cfg = getattr(request.app.state, "config", None)
    default_branch = getattr(getattr(cfg, "scm", None), "default_branch", None) or "main"
    branch = req.branch or default_branch

    # Register the repo in the tracker NOW so the UI sees it appear
    # immediately. The Job (or background task) moves it through listing ->
    # indexing -> done; progress lands in the durable Postgres job-state.
    indexing_tracker.queue_repo(req.repo, branch)
    # Mirror the queued state into Postgres so backend pods (which don't run
    # the writer flush loop) see the repo appear immediately, even when a
    # separate Job does the actual indexing.
    store = getattr(request.app.state, "index_store", None)
    if store is not None:
        try:
            await store.flush(indexing_tracker.get_summary(), indexing_tracker.get_jobs())
        except Exception:
            pass

    # Production: spawn an ephemeral k8s Job. Dev / no-cluster: run in-process.
    launcher = getattr(request.app.state, "job_launcher", None)
    if launcher is not None:
        try:
            job_name = await launcher.launch_repo(req.repo, branch)
            logging.getLogger("opsrag.routes").info(
                "queued indexing Job %s for repo=%s@%s", job_name, req.repo, branch
            )
            return IndexRepoResponse(repo=req.repo, branch=branch, chunks_indexed=0)
        except Exception as exc:
            logging.getLogger("opsrag.routes").warning(
                "Job launch failed for repo=%s (%s); falling back to in-process",
                req.repo, exc,
            )

    async def _run() -> None:
        try:
            await pipeline.index_repo(req.repo, branch, req.patterns)
            if qa_cache is not None:
                try:
                    await qa_cache.invalidate_repo(req.repo)
                except Exception:
                    pass
        except Exception as exc:
            _log = logging.getLogger("opsrag.routes")
            _log.warning("background index failed repo=%s: %s", req.repo, exc)
            indexing_tracker.repo_failed(req.repo, branch, str(exc))

    asyncio.create_task(_run())
    return IndexRepoResponse(repo=req.repo, branch=branch, chunks_indexed=0)


@router.post("/index/source", response_model=IndexSourceResponse)
async def index_source(req: IndexSourceRequest, request: Request) -> IndexSourceResponse:
    """Trigger ingestion of a non-git source (Confluence space etc).

    Same fire-and-forget pattern as `/index/repo`: the tracker entry
    appears immediately and the UI watches `/indexing/status` for
    progress under the source's group.
    """
    pipeline = request.app.state.ingestion_pipeline
    qa_cache = getattr(request.app.state, "qa_cache", None)

    if req.source_type not in (pipeline.sources or {}):
        raise HTTPException(
            status_code=400,
            detail=f"unknown source_type {req.source_type!r}; "
                   f"registered: {sorted(pipeline.sources or {})}",
        )

    # Pre-register so the UI sees the entry under the right source_type
    # group as soon as the POST returns.
    repo_key = f"{req.source_type}:{req.scope}"
    indexing_tracker.queue_repo(repo_key, req.source_type, source_type=req.source_type)
    store = getattr(request.app.state, "index_store", None)
    if store is not None:
        try:
            await store.flush(indexing_tracker.get_summary(), indexing_tracker.get_jobs())
        except Exception:
            pass

    # Production: spawn an ephemeral k8s Job. Dev / no-cluster: run in-process.
    launcher = getattr(request.app.state, "job_launcher", None)
    if launcher is not None:
        try:
            job_name = await launcher.launch_source(req.source_type, req.scope)
            logging.getLogger("opsrag.routes").info(
                "queued indexing Job %s for source=%s scope=%s",
                job_name, req.source_type, req.scope,
            )
            return IndexSourceResponse(
                source_type=req.source_type, scope=req.scope, chunks_indexed=0,
            )
        except Exception as exc:
            logging.getLogger("opsrag.routes").warning(
                "Job launch failed for source=%s (%s); falling back to in-process",
                req.source_type, exc,
            )

    async def _run() -> None:
        try:
            await pipeline.index_source(req.source_type, req.scope)
            if qa_cache is not None:
                try:
                    await qa_cache.invalidate_repo(repo_key)
                except Exception:
                    pass
        except Exception as exc:
            _log = logging.getLogger("opsrag.routes")
            _log.warning(
                "background index_source failed source=%s scope=%s: %s",
                req.source_type, req.scope, exc,
            )
            indexing_tracker.repo_failed(repo_key, req.source_type, str(exc))

    asyncio.create_task(_run())
    return IndexSourceResponse(
        source_type=req.source_type, scope=req.scope, chunks_indexed=0,
    )


@router.post("/admin/reaugment/confluence")
async def admin_reaugment_confluence(
    request: Request,
    dry_run: bool = False,
    scope: str = "SRE",
    max_docs: int = 0,
) -> dict:
    """Re-run contextual chunking on Confluence docs whose children
    lack the `[Context: ...]` prefix.

    A prior audit found that some Confluence children were left
    un-augmented during initial
    indexing -- likely from silent failure paths in `augment_chunks`
    (transient LLM errors, JSON parse failures, etc.). This endpoint
    targets only the affected docs (idempotent on re-runs).

    Algorithm:
      1. Scroll Qdrant for `repo=confluence:<scope>` child chunks.
      2. Collect distinct `source_path` values where >=1 child does NOT
         start with `[Context:`.
      3. For each affected doc: delete ALL its chunks (parent +
         children -- chunk IDs hash content, so old un-augmented chunks
         would coexist with new augmented ones if not deleted), then
         re-fetch via `ConfluenceSource.fetch_document` and run the
         existing `pipeline._process_file` which augments + re-embeds
         + upserts.

    Query params:
      - `dry_run=true` -- return count of affected docs without
        modifying anything. Use this first to size the work.
      - `scope` -- Confluence space key (default `SRE`).
      - `max_docs` -- cap docs processed (0 = unlimited). Useful to
        smoke-test the path on a few docs before kicking the full job.

    Returns a JSON summary. Long-running; the request blocks until
    done (no fire-and-forget -- caller wants to know the outcome).
    """
    pipeline = request.app.state.ingestion_pipeline
    sources = pipeline.sources or {}
    if "confluence" not in sources:
        raise HTTPException(
            status_code=400,
            detail="Confluence source not registered; "
                   "set CONFLUENCE_API_TOKEN + cfg.confluence.enabled=true",
        )
    source = sources["confluence"]
    vector_store = pipeline.vector_store
    qdrant = vector_store._client
    repo_key = f"confluence:{scope}"

    from qdrant_client import models as qm

    # Step 1+2: scroll all child chunks for this Confluence repo, collect
    # source_paths where at least one child lacks the contextual prefix.
    affected: set[str] = set()
    total_children = 0
    augmented_children = 0
    cursor = None
    for _ in range(200):  # safety bound -- 200 * 2000 = 400k chunks
        points, cursor = await qdrant.scroll(
            collection_name="opsrag",
            scroll_filter=qm.Filter(must=[
                qm.FieldCondition(key="repo", match=qm.MatchValue(value=repo_key)),
                qm.FieldCondition(key="chunk_type", match=qm.MatchValue(value="child")),
            ]),
            limit=2000,
            offset=cursor,
            with_payload=["content", "source_path"],
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            sp = payload.get("source_path") or ""
            content = (payload.get("content") or "").lstrip()
            if not sp:
                continue
            total_children += 1
            if content.startswith("[Context:"):
                augmented_children += 1
            else:
                affected.add(sp)
        if not cursor:
            break

    affected_list = sorted(affected)
    if max_docs > 0:
        affected_list = affected_list[:max_docs]

    summary = {
        "scope": scope,
        "dry_run": dry_run,
        "total_children_scanned": total_children,
        "augmented_children": augmented_children,
        "augmentation_coverage": (
            round(augmented_children / total_children, 4) if total_children else 0.0
        ),
        "affected_docs_count": len(affected_list),
        "affected_docs_sample": affected_list[:10],
        "processed": 0,
        "failed": 0,
        "failures": [],
    }
    if dry_run or not affected_list:
        return summary

    # Fire-and-forget for full runs (>5 docs) -- the request would
    # otherwise exceed reasonable HTTP timeouts at ~10s/doc. Smoke
    # tests with max_docs <= 5 stay synchronous so the caller gets the
    # result inline. Operators can poll /admin/reaugment/confluence?dry_run=true
    # to track progress (affected_docs_count drops as work completes).
    fire_and_forget = len(affected_list) > 5

    if fire_and_forget:
        async def _run_background() -> None:
            await _reaugment_docs(
                affected_list, scope, repo_key, source, qdrant, pipeline,
                summary,
            )
            _log = logging.getLogger("opsrag.routes")
            _log.info(
                "reaugment confluence done: scope=%s processed=%d failed=%d",
                scope, summary["processed"], summary["failed"],
            )

        asyncio.create_task(_run_background())
        summary["fire_and_forget"] = True
        summary["note"] = (
            f"Re-augmenting {len(affected_list)} docs in background. "
            f"Poll the same endpoint with dry_run=true to track progress; "
            f"affected_docs_count drops as work completes. "
            f"Container logs show per-doc progress."
        )
        return summary

    # Synchronous path (smoke test) -- block until done.
    await _reaugment_docs(
        affected_list, scope, repo_key, source, qdrant, pipeline, summary,
    )
    summary["failures"] = summary["failures"][:50]
    return summary


async def _reaugment_docs(
    affected_list: list[str],
    scope: str,
    repo_key: str,
    source,
    qdrant,
    pipeline,
    summary: dict,
) -> None:
    """Per-doc delete-then-reprocess loop. Pulled out so both the
    synchronous (smoke-test) and fire-and-forget paths share the same
    body. Mutates `summary` in place -- caller exposes counters."""
    from qdrant_client import models as qm

    from opsrag.interfaces.source import DocRef

    for sp in affected_list:
        # Reconstruct DocRef from source_path. Confluence convention:
        # source_path == f"{page_id}:{slug}.md" -- strip .md suffix to
        # get the doc_id the connector expects.
        doc_id = sp[:-3] if sp.endswith(".md") else sp
        ref = DocRef(source_type="confluence", scope=scope, doc_id=doc_id)

        # Delete BOTH parent and children for this source_path so the
        # new augmented chunks don't coexist with stale un-augmented ones.
        try:
            await qdrant.delete(
                collection_name="opsrag",
                points_selector=qm.Filter(must=[
                    qm.FieldCondition(key="repo", match=qm.MatchValue(value=repo_key)),
                    qm.FieldCondition(key="source_path", match=qm.MatchValue(value=sp)),
                ]),
            )
        except Exception as exc:
            summary["failed"] += 1
            summary["failures"].append({"path": sp, "stage": "delete", "error": str(exc)[:200]})
            continue

        # Re-fetch + re-process via the existing pipeline path so we
        # get parse -> chunk -> augment -> embed -> upsert with the same
        # config the production path uses. This is THE point of the
        # admin endpoint: don't reimplement the pipeline.
        try:
            doc = await source.fetch_document(ref)
            chunks_added = await pipeline._process_file(doc)
            summary["processed"] += 1
            if chunks_added == 0:
                # Empty doc or parser refusal -- surface in failures so
                # operators can decide whether to investigate.
                summary["failures"].append({
                    "path": sp, "stage": "process", "error": "0 chunks produced",
                })
        except Exception as exc:
            summary["failed"] += 1
            summary["failures"].append({"path": sp, "stage": "fetch+process", "error": str(exc)[:200]})

    # Cap the failures list payload so a 500-failure run doesn't blow
    # the response size. Full list is in container logs.
    summary["failures"] = summary["failures"][:50]
    return summary



@router.get("/sessions/{user_id}", response_model=SessionListResponse)
async def list_sessions(
    user_id: str, request: Request,
    current_user: CurrentUser = Depends(get_current_user_dep),
) -> SessionListResponse:
    store = request.app.state.session_store
    # IDOR fix: an authenticated caller may only list THEIR OWN sessions --
    # override the path user_id with the verified oid so a spoofed/other id
    # in the URL can't enumerate another user's threads. Open / anonymous
    # mode keeps the path-supplied id (preserves zero-config dev behavior).
    if not current_user.is_anonymous and current_user.oid:
        user_id = current_user.oid
    sessions = await store.list_sessions(user_id)
    return SessionListResponse(
        sessions=[SessionSummary(**s) for s in sessions]
    )


@router.delete("/sessions/{thread_id}")
async def delete_session(
    thread_id: str, request: Request,
    current_user: CurrentUser = Depends(require_scope(Scope.CHAT)),
) -> dict:
    """Delete a chat session. Requires authentication (``chat`` scope) and
    per-session OWNERSHIP: an authenticated caller may only delete a thread
    they own. We deny with 404 (not 403) so a non-owner can't probe whether
    a thread_id exists. Open / anonymous mode does not enforce (dev). Legacy
    anonymous-owned threads are grandfathered (still deletable)."""
    store = request.app.state.session_store
    owner = await store.get_session_owner(thread_id)
    _deny_if_not_owner(current_user, owner)
    deleted = await store.delete_session(thread_id)
    return {"thread_id": thread_id, "deleted": deleted}


# -- Slack interactivity (up/down buttons from the Slack bot answer) -----
# Slack POSTs the button click as `application/x-www-form-urlencoded`
# with a single `payload` field whose value is JSON. The handler:
#   1. Verifies the request signature using the bot's signing secret
#      (HMAC-SHA256 over `v0:{ts}:{raw_body}`, compared timing-safe).
#   2. Parses the action_id + value to extract direction +
#      investigation_id.
#   3. Writes to both `investigation_cache.record_feedback` and
#      `feedback_store.record` -- same dual-write the UI endpoint uses.
#   4. Returns an empty 200 (Slack just needs the OK + an inline
#      ephemeral confirmation, which we POST async).
#
# Slack App configuration required:
#   - In api.slack.com -> your app -> Interactivity & Shortcuts:
#     Request URL = https://opsrag.example.com/api/slack/interactivity
#   - Bot scope `chat:write` (already granted)
#   - Bot scope `commands` not needed (we use buttons, not slash cmds)
#
# Signature is mandatory -- Slack signs every interactive request, and
# we refuse to write feedback on a request we can't verify.
@router.post("/slack/interactivity")
async def slack_interactivity(request: Request) -> Response:
    import hashlib
    import hmac
    import json as _json
    import time
    from urllib.parse import parse_qs

    raw_body = await request.body()
    body_text = raw_body.decode("utf-8", errors="replace")

    # 1. Signature verification -- refuse anything that's not from Slack.
    signing_secret = os.environ.get("OPSRAG_SLACK_SIGNING_SECRET") or os.environ.get("SLACK_SIGNING_SECRET") or ""
    if not signing_secret:
        _log.warning("slack interactivity: no signing secret configured -- refusing")
        raise HTTPException(status_code=503, detail="slack signing secret not configured")
    slack_ts = request.headers.get("x-slack-request-timestamp") or ""
    slack_sig = request.headers.get("x-slack-signature") or ""
    # 5-minute replay window per Slack docs.
    try:
        if abs(time.time() - int(slack_ts)) > 300:
            raise HTTPException(status_code=403, detail="stale slack request")
    except ValueError:
        raise HTTPException(status_code=403, detail="bad slack timestamp")
    base = f"v0:{slack_ts}:{body_text}".encode()
    expected = "v0=" + hmac.new(signing_secret.encode(), base, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, slack_sig):
        _log.warning("slack interactivity: signature mismatch")
        raise HTTPException(status_code=403, detail="bad slack signature")

    # 2. Parse the payload -- Slack posts form-encoded `payload=<json>`.
    form = parse_qs(body_text)
    payload_raw = (form.get("payload") or [""])[0]
    try:
        payload = _json.loads(payload_raw)
    except Exception:
        raise HTTPException(status_code=400, detail="bad payload json")
    if payload.get("type") != "block_actions":
        return Response(status_code=200)  # ignore non-button events
    actions = payload.get("actions") or []
    if not actions:
        return Response(status_code=200)
    action = actions[0]
    action_id = action.get("action_id") or ""
    if not action_id.startswith("opsrag_feedback_"):
        return Response(status_code=200)
    value = (action.get("value") or "").strip()
    # value shape: "up:<investigation_id>" or "down:<investigation_id>"
    if ":" not in value:
        return Response(status_code=200)
    thumbs, investigation_id = value.split(":", 1)
    if thumbs not in ("up", "down") or not investigation_id:
        return Response(status_code=200)

    # 3. Dual-write -- same shape as `/api/investigation/<id>/feedback`.
    direction = 1 if thumbs == "up" else -1
    slack_user = ((payload.get("user") or {}).get("id")) or "slack-unknown"
    cache = getattr(request.app.state, "investigation_cache", None)
    if cache is not None:
        try:
            await cache.record_feedback(investigation_id, thumbs=thumbs, correction=None)
        except Exception as exc:
            _log.warning("slack interactivity: investigation cache write failed: %s", exc)
    feedback_store = getattr(request.app.state, "feedback_store", None)
    if feedback_store is not None:
        try:
            await feedback_store.record(
                investigation_id=investigation_id,
                direction=direction,
                thread_id=(payload.get("container") or {}).get("thread_ts"),
                user_id=f"slack:{slack_user}",
                note=None,
                query_snippet=None,
                answer_snippet=None,
            )
        except Exception as exc:
            _log.warning("slack interactivity: feedback_store write failed: %s", exc)

    _log.info(
        "slack feedback recorded: thumbs=%s investigation=%s slack_user=%s",
        thumbs, investigation_id, slack_user,
    )
    # Slack accepts a JSON body to update the original message. For now
    # just 200 OK -- the user already sees the button click animation.
    # Future: post an ephemeral "thanks, feedback recorded" via
    # response_url.
    return Response(status_code=200)


@router.post(
    "/investigation/{investigation_id}/feedback",
    response_model=InvestigationFeedbackResponse,
)
async def investigation_feedback(
    investigation_id: str,
    req: InvestigationFeedbackRequest,
    request: Request,
) -> InvestigationFeedbackResponse:
    """Attach up/down feedback (and
    optional free-text correction) to a past investigation cache entry.

    Dual-write: the Qdrant-side investigation cache gets the
    legacy flag (so cache-audit + thumbs-down purge keep working) AND we
    insert a normalized row into Postgres ``opsrag_feedback`` so SREs can
    query for low-scored answers and author corrections into the SRE-KB.

    Failure modes are GRACEFUL:
      - if investigation cache isn't configured, we still try the
        Postgres write (UI feedback shouldn't depend on the Qdrant cache
        being available)
      - if Postgres feedback store isn't configured / fails, the route
        still returns 200 with ``recorded`` reflecting whether ANY sink
        accepted the write
    """
    if req.thumbs not in ("up", "down"):
        raise HTTPException(status_code=400, detail="thumbs must be 'up' or 'down'")

    direction = 1 if req.thumbs == "up" else -1

    cache = getattr(request.app.state, "investigation_cache", None)
    cache_ok = False
    if cache is not None:
        try:
            cache_ok = await cache.record_feedback(
                investigation_id, thumbs=req.thumbs, correction=req.correction,
            )
        except Exception as exc:
            _log.warning("investigation cache feedback write failed: %s", exc)
            cache_ok = False

    feedback_store = getattr(request.app.state, "feedback_store", None)
    feedback_id: int | None = None
    if feedback_store is not None:
        try:
            feedback_id = await feedback_store.record(
                investigation_id=investigation_id,
                direction=direction,
                thread_id=req.thread_id,
                user_id=req.user_id,
                note=req.correction,
                query_snippet=req.query_snippet,
                answer_snippet=req.answer_snippet,
            )
        except Exception as exc:
            # Defense in depth -- record() already swallows Exceptions, but
            # if a programming bug leaks one out we MUST still return 200
            # so the FE doesn't surface a feedback error to the user.
            _log.warning("feedback_store.record raised unexpectedly: %s", exc)
            feedback_id = None

    recorded = cache_ok or feedback_id is not None
    detail: str | None = None
    if not recorded:
        detail = "no feedback sink available"
    elif cache is None and feedback_id is not None:
        detail = "persisted to opsrag_feedback (investigation cache disabled)"
    return InvestigationFeedbackResponse(
        investigation_id=investigation_id,
        recorded=recorded,
        detail=detail,
        feedback_id=feedback_id,
    )


@router.get("/feedback", response_model=FeedbackListResponse)
async def list_feedback(
    request: Request,
    direction: int | None = None,
    limit: int = 50,
) -> FeedbackListResponse:
    """List recent feedback rows for SRE triage.

    Query ``?direction=-1&limit=50`` returns the most-recent thumbs-down
    answers, which is the primary SRE workflow ("show me what went wrong
    this week, author corrections into the SRE-KB").

    Auth: same DELETE/POST-admin gate via :mod:`opsrag.api.middleware`
    when API keys are configured -- see ADMIN_ROUTES.
    """
    store = getattr(request.app.state, "feedback_store", None)
    if store is None:
        return FeedbackListResponse(items=[], direction_filter=direction, limit=limit)
    rows = await store.list_recent(direction=direction, limit=limit)
    return FeedbackListResponse(
        items=[FeedbackListItem(**r) for r in rows],
        direction_filter=direction,
        limit=limit,
    )


# -- Feedback-as-correction with highest vector boost --
# User submits "no, the correct answer is X" -> we store a synthetic Q+A
# chunk in the SAME `opsrag_v2` retrieval collection with
# `priority: user-correction`, lifting its score 2.5x at search time so a
# single corrective click reliably dominates corpus content. Also writes
# an audit row to Postgres `opsrag_feedback` (direction=2) so the
# correction survives a Qdrant rebuild via replay.


@router.post("/correction", response_model=CorrectionResponse)
async def submit_correction(
    req: CorrectionRequest,
    request: Request,
    current_user: CurrentUser = Depends(require_scope(Scope.CHAT)),
) -> CorrectionResponse:
    """Enqueue a user-authored correction for operator review.

    The correction is NOT injected into retrieval here. It lands in the
    Postgres moderation queue (``opsrag_pending_corrections``) as ``pending``
    and is invisible to search until an operator approves it via
    ``POST /corrections/{id}/approve`` -- only then is the boosted chunk
    written to Qdrant. This closes the prior vector where any caller could
    poison answers for everyone with a single un-reviewed POST.

    Requires the ``chat`` scope (any authenticated user may *submit*; only an
    admin may *approve*). The submitter identity is taken from the
    authenticated principal, NOT the request body -- a caller can't attribute a
    correction to someone else. Also writes an audit row to ``opsrag_feedback``
    (direction=2) for durable history.
    """
    queue = getattr(request.app.state, "pending_correction_store", None)
    if queue is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "moderation queue not available -- backend started without a "
                "Postgres DSN. Check lifespan init logs."
            ),
        )

    # Identity from the authenticated principal, not the spoofable body.
    # Open mode -> oid is None -> falls back to body (no auth to enforce).
    submitter = current_user.oid or req.user_id

    try:
        pending_id = await queue.submit(
            question=req.question,
            correct_answer=req.correct_answer,
            wrong_answer=req.wrong_answer,
            evidence_url=req.evidence_url,
            user_id=submitter,
            thread_id=req.thread_id,
        )
    except Exception as exc:
        _log.warning("correction enqueue failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"correction enqueue failed: {exc}") from exc

    # Audit log -- best-effort. direction=2 == "correction"; the note field
    # carries the corrected answer, query_snippet the original question.
    feedback_id: int | None = None
    feedback_store = getattr(request.app.state, "feedback_store", None)
    if feedback_store is not None:
        try:
            feedback_id = await feedback_store.record(
                investigation_id=f"pending-{pending_id}",
                direction=2,
                thread_id=req.thread_id,
                user_id=submitter,
                note=req.correct_answer[:2000],
                query_snippet=req.question[:400],
                answer_snippet=(req.wrong_answer or "")[:400],
            )
        except Exception as exc:
            _log.warning("correction audit-log write failed (graceful): %s", exc)

    return CorrectionResponse(
        ok=True,
        pending_id=pending_id,
        status="pending",
        message="Correction queued for operator review. It is not live until approved.",
        feedback_id=feedback_id,
    )


@router.get("/corrections/pending", response_model=PendingCorrectionListResponse)
async def list_pending_corrections(
    request: Request, limit: int = 50,
    _user: CurrentUser = Depends(require_scope(Scope.ADMIN)),
) -> PendingCorrectionListResponse:
    """List corrections awaiting operator review. Requires the ``admin`` scope."""
    queue = getattr(request.app.state, "pending_correction_store", None)
    if queue is None:
        return PendingCorrectionListResponse(items=[], limit=limit)
    rows = await queue.list_pending(limit=limit)
    return PendingCorrectionListResponse(
        items=[PendingCorrectionItem(**r) for r in rows],
        limit=limit,
    )


@router.post("/corrections/{pending_id}/approve", response_model=CorrectionReviewResponse)
async def approve_correction(
    pending_id: int, request: Request,
    current_user: CurrentUser = Depends(require_scope(Scope.ADMIN)),
) -> CorrectionReviewResponse:
    """Approve a pending correction: inject the boosted chunk into retrieval
    and mark the queue row ``approved``. Requires the ``admin`` scope --
    approval is the privileged step that makes a correction live. Idempotent --
    a row already resolved is reported back without re-injecting."""
    queue = getattr(request.app.state, "pending_correction_store", None)
    store = getattr(request.app.state, "correction_store", None)
    if queue is None or store is None:
        raise HTTPException(status_code=503, detail="correction stores not available")

    row = await queue.get(pending_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"pending correction {pending_id} not found")
    if row["status"] != "pending":
        return CorrectionReviewResponse(
            ok=True, id=pending_id, status=row["status"], chunk_id=row.get("chunk_id"),
            message=f"already {row['status']}",
        )

    reviewer = current_user.email or current_user.oid or "operator"
    try:
        chunk_id = await store.store_correction(
            question=row["question"],
            wrong_answer=row["wrong_answer"] or "",
            correct_answer=row["correct_answer"],
            user_id=row["user_id"] or "anonymous",
            evidence_url=row["evidence_url"],
            reviewed_by=reviewer,
        )
    except Exception as exc:
        _log.warning("correction approve/inject failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"inject failed: {exc}") from exc

    await queue.resolve(pending_id, status="approved", chunk_id=chunk_id, reviewed_by=reviewer)
    return CorrectionReviewResponse(
        ok=True, id=pending_id, status="approved", chunk_id=chunk_id,
        message="Correction approved and live (1.8x boost).",
    )


@router.post("/corrections/{pending_id}/reject", response_model=CorrectionReviewResponse)
async def reject_correction(
    pending_id: int, request: Request,
    current_user: CurrentUser = Depends(require_scope(Scope.ADMIN)),
) -> CorrectionReviewResponse:
    """Reject a pending correction -- never reaches retrieval. Requires the
    ``admin`` scope."""
    queue = getattr(request.app.state, "pending_correction_store", None)
    if queue is None:
        raise HTTPException(status_code=503, detail="moderation queue not available")
    row = await queue.get(pending_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"pending correction {pending_id} not found")
    reviewer = current_user.email or current_user.oid or "operator"
    changed = await queue.resolve(pending_id, status="rejected", reviewed_by=reviewer)
    return CorrectionReviewResponse(
        ok=True, id=pending_id, status="rejected" if changed else row["status"],
        message="Correction rejected." if changed else f"already {row['status']}",
    )


@router.get("/corrections", response_model=CorrectionListResponse)
async def list_corrections(
    request: Request, limit: int = 50,
    _user: CurrentUser = Depends(require_scope(Scope.ADMIN)),
) -> CorrectionListResponse:
    """List recent APPROVED user-corrections (live in retrieval). Ordered
    newest first by ``metadata.created_at``. Requires the ``admin`` scope."""
    store = getattr(request.app.state, "correction_store", None)
    if store is None:
        return CorrectionListResponse(items=[], limit=limit)
    rows = await store.list_recent_corrections(limit=limit)
    return CorrectionListResponse(
        items=[CorrectionListItem(**r) for r in rows],
        limit=limit,
    )


@router.delete("/corrections/{chunk_id}")
async def delete_correction(
    chunk_id: str, request: Request,
    _user: CurrentUser = Depends(require_scope(Scope.ADMIN)),
) -> dict:
    """Remove a single live user-correction by chunk_id (in case the
    correction itself was wrong). Requires the ``admin`` scope."""
    store = getattr(request.app.state, "correction_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="correction_store not available")
    ok = await store.delete_correction(chunk_id)
    return {"chunk_id": chunk_id, "deleted": ok}


@router.get(
    "/investigation/cache/summary",
    response_model=InvestigationCacheSummary,
)
async def investigation_cache_summary(request: Request) -> InvestigationCacheSummary:
    """Audit summary for the investigation cache -- total points, count
    of stale entries (older than 6 months) + count flagged low-quality
    by feedback."""
    cache = getattr(request.app.state, "investigation_cache", None)
    if cache is None:
        return InvestigationCacheSummary(total=0, available=False)
    from opsrag.agent.cache.audit import cache_summary
    out = await cache_summary(cache._qdrant, collection=cache._collection)
    return InvestigationCacheSummary(**out)


@router.post("/admin/index/investigation-history")
async def index_investigation_history(request: Request) -> dict:
    """Trigger an on-demand index pass of the investigation-history
    source. Same code path as the daily scheduler runs -- useful for
    smoke-testing after wiring the source or after pruning the cache.

    Returns the count of investigations promoted into the corpus."""
    if "investigation-history" not in (request.app.state.ingestion_pipeline.sources or {}):
        raise HTTPException(404, "investigation-history source not registered (enable in config)")
    pipeline = request.app.state.ingestion_pipeline
    try:
        count = await pipeline.index_source("investigation-history", "opsrag")
        return {"source": "investigation-history", "indexed": count}
    except Exception as exc:
        raise HTTPException(500, f"index failed: {exc}") from exc


@router.get("/cache/summary", response_model=CacheSummaryResponse)
async def cache_summary(request: Request) -> CacheSummaryResponse:
    """Unified summary of all 3 caches: Q&A semantic, investigation,
    and the in-process tool-output micro-cache."""
    qa_cache = getattr(request.app.state, "qa_cache", None)
    investigation_cache = getattr(request.app.state, "investigation_cache", None)
    qa_stats = await qa_cache.stats() if qa_cache is not None else {"available": False}
    # Flash judge counters live in `qa_cache_judge` module.
    # Discriminator regex/spaCy ensemble counters.
    from opsrag.qa_cache import discriminator_stats as _disc_stats
    from opsrag.qa_cache_judge import is_enabled as _judge_enabled
    from opsrag.qa_cache_judge import stats as _judge_stats
    from opsrag.qa_cache_ner import stats as _ner_stats
    if isinstance(qa_stats, dict):
        qa_stats["judge_enabled"] = _judge_enabled()
        qa_stats["judge"] = _judge_stats()
        qa_stats["discriminator"] = _disc_stats()
        qa_stats["ner_spacy"] = _ner_stats()
    inv_total = await investigation_cache.count() if investigation_cache is not None else 0
    inv_stats = {"available": investigation_cache is not None, "total": inv_total}
    from opsrag.mcp.tool_cache import get_default_cache as _get_tc
    tool_stats = await _get_tc().stats()
    return CacheSummaryResponse(qa=qa_stats, investigation=inv_stats, tool=tool_stats)


@router.post("/cache/purge", response_model=CachePurgeResponse)
async def cache_purge(
    req: CachePurgeRequest, request: Request,
    _user: CurrentUser = Depends(require_scope(Scope.ADMIN)),
) -> CachePurgeResponse:
    """Multi-strategy purge across all caches. See `CachePurgeRequest`
    for body shape. Returns counts purged per cache; -1 means the
    backend doesn't return a count (Qdrant). Requires the ``admin`` scope
    (was previously gated only by the unregistered ADMIN_ROUTES)."""
    qa_cache = getattr(request.app.state, "qa_cache", None)
    inv_cache = getattr(request.app.state, "investigation_cache", None)
    from opsrag.mcp.tool_cache import get_default_cache as _get_tc
    tool_cache = _get_tc()

    target = req.target.lower()
    strategy = req.strategy.lower()
    older_secs = (req.older_than_hours or 0) * 3600 if req.older_than_hours else None
    pq = pi = pt = 0
    detail = None

    # -- ALL CACHES NUKE --------------------------------------------
    if target == "all" and strategy == "all":
        if qa_cache is not None:
            pq = await qa_cache.purge(all=True)
        if inv_cache is not None:
            pi = await inv_cache.purge(all=True)
        pt = await tool_cache.purge(all=True)
        return CachePurgeResponse(
            target=target, strategy=strategy,
            purged_qa=pq, purged_investigation=pi, purged_tool=pt,
            detail="entire cache nuked",
        )

    # -- Q&A CACHE --------------------------------------------------
    if target == "qa":
        if qa_cache is None:
            raise HTTPException(404, "qa_cache not available")
        if strategy == "all":
            pq = await qa_cache.purge(all=True)
        elif strategy == "older_than":
            if older_secs is None:
                raise HTTPException(400, "older_than_hours required for strategy=older_than")
            pq = await qa_cache.purge(older_than_seconds=older_secs)
        elif strategy == "repo":
            if not req.repo:
                raise HTTPException(400, "repo required for strategy=repo")
            pq = await qa_cache.purge(repo=req.repo)
        elif strategy == "quality_low":
            pq = await qa_cache.purge(quality="low")
        elif strategy == "question_contains":
            if not req.question_substring:
                raise HTTPException(400, "question_substring required")
            pq = await qa_cache.purge(question_substring=req.question_substring)
        else:
            raise HTTPException(400, f"unknown strategy {strategy!r} for target=qa")
        return CachePurgeResponse(target=target, strategy=strategy, purged_qa=pq)

    # -- INVESTIGATION CACHE ----------------------------------------
    if target == "investigation":
        if inv_cache is None:
            raise HTTPException(404, "investigation_cache not available")
        if strategy == "all":
            pi = await inv_cache.purge(all=True)
        elif strategy == "older_than":
            if older_secs is None:
                raise HTTPException(400, "older_than_hours required")
            pi = await inv_cache.purge(older_than_seconds=older_secs)
        elif strategy == "thumbs_down":
            pi = await inv_cache.purge(thumbs_down_only=True)
        elif strategy == "question_contains":
            if not req.question_substring:
                raise HTTPException(400, "question_substring required")
            pi = await inv_cache.purge(question_substring=req.question_substring)
        else:
            raise HTTPException(400, f"unknown strategy {strategy!r} for target=investigation")
        return CachePurgeResponse(target=target, strategy=strategy, purged_investigation=pi)

    # -- TOOL OUTPUT CACHE -----------------------------------------
    if target == "tool":
        if strategy == "all":
            pt = await tool_cache.purge(all=True)
        elif strategy == "tool_name":
            if not req.tool_name:
                raise HTTPException(400, "tool_name required")
            pt = await tool_cache.purge(tool_name=req.tool_name)
        else:
            raise HTTPException(400, f"unknown strategy {strategy!r} for target=tool")
        return CachePurgeResponse(target=target, strategy=strategy, purged_tool=pt)

    raise HTTPException(400, f"unknown target {target!r}")


@router.get("/sessions/{thread_id}/messages")
async def session_messages(
    thread_id: str, request: Request,
    current_user: CurrentUser = Depends(get_current_user_dep),
) -> dict:
    """Replay the message history of a thread by walking its LangGraph
    checkpoints. Returns chronological [{role, content, ...}] pairs.

    IDOR fix: an authenticated caller may only read a thread they own; we
    deny with 404 (not 403) so a non-owner can't probe thread existence.
    Open / anonymous mode does not enforce; legacy anonymous-owned threads
    are grandfathered (still readable)."""
    store = request.app.state.session_store
    owner = await store.get_session_owner(thread_id)
    _deny_if_not_owner(current_user, owner)
    messages = await store.get_messages(thread_id)
    return {"thread_id": thread_id, "messages": messages}


# Webhook endpoints removed: the daily APScheduler
# now drives reindexing on a cron; webhooks were a leftover from the
# pre-cache, pre-idempotency design and produced unbounded Vertex spend
# on busy repos. Manual reindex via POST /index/repo remains available
# for admin-triggered runs.


# -- Identity + per-user attribution endpoints -------------


@router.get("/me")
async def api_me(
    request: Request,
    current_user: CurrentUser = Depends(get_current_user_dep),
) -> dict[str, Any]:
    """Identity of the requesting user, as the backend sees it.

    The UI fetches this on boot:
      * `tracking_enabled=false` -> render anonymous mode, hide the
        user chip in the header.
      * `tracking_enabled=true` + `is_anonymous=true` -> user reached
        the backend without going through Pomerium (local dev, or a
        misconfigured ingress). UI shows "anonymous" but exposes
        `/api/usage/mine` as no-ops.
      * `tracking_enabled=true` + identity -> render the user chip.

    Deliberately a flat dict -- no Pydantic model, no extra coupling.
    """
    tracking_cfg = getattr(request.app.state, "tracking_user_config", None)
    enabled = bool(getattr(tracking_cfg, "enabled", False))
    return {
        "tracking_enabled": enabled,
        "is_anonymous": current_user.is_anonymous,
        "oid": current_user.oid,          # == sub (back-compat alias)
        "sub": current_user.sub,
        "email": current_user.email,
        "name": current_user.name,
        "picture_url": current_user.picture_url,
        "groups": list(current_user.groups),
        # RBAC: roles + scopes drive UI nav gating + the scope pill.
        "roles": sorted(current_user.roles),
        "scopes": sorted(current_user.scopes),
        "is_admin": current_user.has_scope(Scope.ADMIN),
    }


@router.get("/me/usage")
async def api_usage_mine(
    request: Request,
    current_user: CurrentUser = Depends(get_current_user_dep),
) -> dict[str, Any]:
    """Per-user usage roll-up for the requesting user. Always 200 -- an
    anonymous caller just gets `{"anonymous": true, "events": []}` so
    the UI doesn't need to branch on auth state before rendering.
    """
    if current_user.is_anonymous or current_user.oid is None:
        return {"anonymous": True, "events": []}
    persistence = getattr(request.app.state, "usage_persistence", None)
    if persistence is None:
        # Persistence not wired (test fixture, local-dev with no DB).
        # Return a zeroed row so the UI's chart-loading path doesn't
        # have to special-case "endpoint exists but no data".
        return {
            "anonymous": False,
            "user_oid": current_user.oid,
            "email": current_user.email,
            "display_name": current_user.name,
            "query_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd_micros": 0,
            "last_active_at": None,
        }
    data = await persistence.aggregate_for_user(current_user.oid)
    return {"anonymous": False, **data}


@router.get("/admin/usage")
async def api_usage_by_user(
    request: Request,
    current_user: CurrentUser = Depends(require_scope(Scope.ADMIN)),
) -> dict[str, Any]:
    """Admin-only -- per-user usage leaderboard, ordered by cost desc.

    Gated by the `admin` scope (`require_scope` 403s non-admins; open mode
    grants all scopes). Meaningless without identity attribution, so it also
    requires usage persistence to be configured.
    """
    persistence = getattr(request.app.state, "usage_persistence", None)
    if persistence is None:
        return {"users": []}
    rows = await persistence.aggregate_by_user()
    return {"users": rows}
