"""Runbook CRUD + versions + promote-from-investigation endpoints.

Mounted at `/api/runbooks` on the main FastAPI app. The store is
attached at `app.state.runbook_store` during lifespan startup.

Auth: same dependency as the rest of the API (`get_current_user_dep`).
We record the editor's email on every write so version history shows
who changed what.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from opsrag.auth import CurrentUser, get_current_user_dep
from opsrag.runbooks.models import (
    Runbook,
    RunbookCreate,
    RunbookUpdate,
    RunbookVersion,
)

_log = logging.getLogger("opsrag.api.runbooks")

# NOTE: no `/api` prefix here -- the UI's nginx proxy strips `/api/` before
# forwarding to the backend (see `ui/nginx.conf`). curl'ing `/api/runbooks`
# directly works because uvicorn doesn't care about the prefix the proxy
# adds; but the FastAPI router must mount at `/runbooks` to match what
# the backend actually receives over the proxy.
runbooks_router = APIRouter(prefix="/runbooks", tags=["runbooks"])


# -- List / search ---------------------------------------------------


class RunbookListResponse(BaseModel):
    count: int
    runbooks: list[Runbook]


@runbooks_router.get("", response_model=RunbookListResponse)
async def list_runbooks(
    request: Request,
    service: str | None = Query(default=None, description="Filter by service"),
    issue_kind: str | None = Query(default=None, description="Filter by failure_class"),
    enabled_only: bool = Query(default=True, description="Hide soft-deleted runbooks"),
    limit: int = Query(default=100, ge=1, le=500),
    current_user: CurrentUser = Depends(get_current_user_dep),
):
    store = _store_or_503(request)
    runbooks = await store.list(
        service=service, issue_kind=issue_kind,
        enabled_only=enabled_only, limit=limit,
    )
    return RunbookListResponse(count=len(runbooks), runbooks=runbooks)


# -- Get one ---------------------------------------------------------


@runbooks_router.get("/{runbook_id}", response_model=Runbook)
async def get_runbook(
    runbook_id: str,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user_dep),
):
    store = _store_or_503(request)
    try:
        return await store.get(runbook_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"runbook not found: {runbook_id}")


# -- Create ----------------------------------------------------------


@runbooks_router.post("", response_model=Runbook, status_code=201)
async def create_runbook(
    body: RunbookCreate,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user_dep),
):
    store = _store_or_503(request)
    try:
        return await store.create(
            body,
            author_email=current_user.email,
            source="hand",
        )
    except Exception as exc:
        _log.exception("runbook create failed")
        raise HTTPException(status_code=500, detail=f"create failed: {exc}") from exc


# -- Update ----------------------------------------------------------


@runbooks_router.put("/{runbook_id}", response_model=Runbook)
async def update_runbook(
    runbook_id: str,
    body: RunbookUpdate,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user_dep),
):
    store = _store_or_503(request)
    try:
        return await store.update(
            runbook_id, body, editor_email=current_user.email,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"runbook not found: {runbook_id}")
    except Exception as exc:
        _log.exception("runbook update failed")
        raise HTTPException(status_code=500, detail=f"update failed: {exc}") from exc


# -- Delete (soft by default, hard via query param) -----------------


@runbooks_router.delete("/{runbook_id}", status_code=204)
async def delete_runbook(
    runbook_id: str,
    request: Request,
    hard: bool = Query(default=False, description="Permanent delete + remove embedding"),
    current_user: CurrentUser = Depends(get_current_user_dep),
):
    store = _store_or_503(request)
    deleted = await store.delete(runbook_id, hard=hard)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"runbook not found: {runbook_id}")


# -- Versions --------------------------------------------------------


class RunbookVersionsResponse(BaseModel):
    runbook_id: str
    count: int
    versions: list[RunbookVersion]


@runbooks_router.get("/{runbook_id}/versions", response_model=RunbookVersionsResponse)
async def list_versions(
    runbook_id: str,
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    current_user: CurrentUser = Depends(get_current_user_dep),
):
    store = _store_or_503(request)
    versions = await store.versions(runbook_id, limit=limit)
    return RunbookVersionsResponse(
        runbook_id=runbook_id, count=len(versions), versions=versions,
    )


# -- Promote from investigation (Pro-generated draft) --------------


class PromoteFromInvestigationBody(BaseModel):
    """Optional overrides. When omitted, the generator + tagger
    autopopulate. The user reviews the draft markdown in the UI editor
    and clicks Save to persist it."""
    title: str | None = None
    service: str | None = None
    issue_kind: str | None = None
    severity_min: str | None = None


class PromoteFromInvestigationResponse(BaseModel):
    """Pre-save draft. Front-end opens the editor with this body
    pre-filled; user can edit before clicking Save -> POST /api/runbooks.
    NOT yet persisted (that's the user's choice in the UI)."""
    draft_markdown: str = Field(..., description="Pro-generated runbook markdown")
    suggested_title: str | None = None
    suggested_service: str | None = None
    suggested_issue_kind: str | None = None
    source_investigation_id: str


@runbooks_router.post(
    "/from-investigation/{investigation_id}",
    response_model=PromoteFromInvestigationResponse,
)
async def promote_from_investigation(
    investigation_id: str,
    body: PromoteFromInvestigationBody,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user_dep),
):
    """Pro LLM converts a closed investigation into a runbook draft.
    Returns the draft markdown for the user to review + edit + save.
    """
    store = _store_or_503(request)
    cache = getattr(request.app.state, "investigation_cache", None)
    if cache is None:
        raise HTTPException(
            status_code=503,
            detail="investigation_cache not configured -- cannot read source investigation",
        )
    # Pull the investigation from Qdrant directly via the cache's
    # private client (no public get-by-id method exists yet).
    try:
        points = await cache._qdrant.retrieve(
            collection_name=cache._collection,
            ids=[investigation_id],
            with_payload=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"qdrant lookup failed: {exc}")
    if not points:
        raise HTTPException(status_code=404, detail=f"investigation not found: {investigation_id}")
    p = points[0].payload or {}

    providers = getattr(request.app.state, "providers", None)
    # Pro LLM via model_router (same routing as Insight synthesizer).
    router = getattr(request.app.state, "model_router", None) or getattr(providers, "model_router", None)
    pro_llm = None
    if router is not None and getattr(router, "has_pro", False):
        pro_llm = getattr(router, "pro_llm", None)
    if pro_llm is None:
        pro_llm = getattr(providers, "llm", None) if providers else None
    if pro_llm is None:
        raise HTTPException(status_code=503, detail="no LLM configured for runbook generation")

    from opsrag.runbooks.generator import generate_runbook_draft
    tags = p.get("tags") or {}
    incident_target = body.service or tags.get("service")
    draft = await generate_runbook_draft(
        llm=pro_llm,
        investigation_question=p.get("question") or "",
        investigation_answer=p.get("answer") or "",
        tool_call_audit=p.get("tool_call_audit") or [],
        incident_target=incident_target,
    )
    if not draft:
        raise HTTPException(
            status_code=502,
            detail="Pro generator returned empty -- try again or fall back to manual",
        )

    # Suggest title from first heading or first line of the draft.
    suggested_title = body.title or _suggest_title(draft, incident_target)
    suggested_issue_kind = body.issue_kind or tags.get("failure_class")

    return PromoteFromInvestigationResponse(
        draft_markdown=draft,
        suggested_title=suggested_title,
        suggested_service=incident_target,
        suggested_issue_kind=suggested_issue_kind,
        source_investigation_id=investigation_id,
    )


# -- Helpers ---------------------------------------------------------


def _store_or_503(request: Request):
    store = getattr(request.app.state, "runbook_store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="runbook_store not configured (set OPSRAG_POSTGRES_DSN)",
        )
    return store


def _suggest_title(markdown: str, target: str | None) -> str:
    """Pull an H1 if present, else fall back to '<target> Runbook'."""
    for line in (markdown or "").splitlines():
        line = line.strip()
        if line.startswith("# ") and not line.startswith("##"):
            t = line.lstrip("# ").strip()
            if t and len(t) <= 120:
                return t
    if target:
        return f"{target} runbook"
    return "Untitled runbook"
