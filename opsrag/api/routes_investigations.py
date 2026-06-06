"""Investigate-mode API routes (Option B refactor 2026-05-27).

Three endpoints:

  POST /investigations
      Body: {alert_text: str}
      Creates an opsrag_investigations row, kicks off the runner in a
      background asyncio task, returns {investigation_id} immediately.

  GET  /investigations/{id}
      Returns the latest snapshot -- lifecycle row + ALL events to date.
      The UI calls this on mount / refresh so the page is renderable
      WITHOUT waiting for the EventSource to catch up.

  GET  /investigations/{id}/events?since=N
      SSE stream. Tails opsrag_investigation_events with sequence > N.
      ~30s window then closes; the FE EventSource reconnects with the
      latest seen sequence, so a tab refresh / network blip never loses
      events.

  GET  /investigations?limit=N
      Sidebar listing -- most-recent-first lifecycle rows.

Nginx strips /api/ before forwarding, so the FastAPI router uses
prefix='/investigations' (NOT '/api/investigations').
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from opsrag.auth.scopes import Scope, require_scope
from opsrag.investigations import InvestigationEventStore
from opsrag.investigations.runner import InvestigationRunner

_log = logging.getLogger("opsrag.api.routes_investigations")

# RBAC: the entire investigate surface (launch + read snapshots + SSE) is
# gated on the `investigate` scope. In open mode every user carries all
# scopes so this is transparent; in login/oidc mode a chat-only member 403s
# here -- matching the UI, which only shows the Investigations nav to users
# who hold the scope. EventSource sends the session cookie, so the SSE
# stream resolves the user the same way as the JSON endpoints.
investigations_router = APIRouter(
    prefix="/investigations",
    tags=["investigations"],
    dependencies=[Depends(require_scope(Scope.INVESTIGATE))],
)


class CreateInvestigationRequest(BaseModel):
    alert_text: str


class CreateInvestigationResponse(BaseModel):
    investigation_id: str


def _get_store(request: Request) -> InvestigationEventStore:
    store = getattr(request.app.state, "investigation_event_store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "investigation_event_store not initialized -- "
                "check Postgres connectivity at startup"
            ),
        )
    return store


def _get_runner(request: Request) -> InvestigationRunner:
    runner = getattr(request.app.state, "investigation_runner", None)
    if runner is None:
        raise HTTPException(
            status_code=503,
            detail="investigation_runner not initialized at startup",
        )
    return runner


@investigations_router.post("", response_model=CreateInvestigationResponse)
async def create_investigation(
    req: CreateInvestigationRequest,
    request: Request,
) -> CreateInvestigationResponse:
    """Kick off a new investigation. The runner executes in the
    background; the response returns as soon as the lifecycle row is
    written so the UI can subscribe to the event stream right away."""
    alert = (req.alert_text or "").strip()
    if not alert:
        raise HTTPException(status_code=400, detail="alert_text required")

    store = _get_store(request)
    runner = _get_runner(request)
    # Persist incident_target on the lifecycle row at create-time so
    # the sidebar/listing can show the target chip without waiting
    # for the runner to emit hypotheses_generated.
    from opsrag.investigations.runner import _extract_target
    inv_id = await store.create_investigation(
        alert_text=alert,
        incident_target=_extract_target(alert),
    )

    # Detached background task -- completes asynchronously. The runner
    # catches its own errors and emits INVESTIGATION_FAILED so the UI
    # always sees a terminal event.
    asyncio.create_task(
        runner.run_one(inv_id, alert),
        name=f"investigate:{inv_id[:8]}",
    )
    _log.info("investigation %s kicked off", inv_id)
    return CreateInvestigationResponse(investigation_id=inv_id)


@investigations_router.get("")
async def list_investigations(
    request: Request,
    limit: int = 50,
) -> dict:
    """Sidebar listing -- most-recent-first."""
    store = _get_store(request)
    rows = await store.list_investigations(limit=limit)
    return {"investigations": rows}


@investigations_router.get("/{inv_id}")
async def get_investigation_snapshot(
    inv_id: str, request: Request,
) -> dict:
    """Full snapshot -- lifecycle row + every event. The UI uses this on
    page mount so initial render doesn't depend on the SSE stream."""
    store = _get_store(request)
    inv = await store.get_investigation(inv_id)
    if inv is None:
        raise HTTPException(status_code=404, detail=f"investigation {inv_id} not found")
    events = await store.list_all_events(investigation_id=inv_id)
    return {"investigation": inv, "events": events}


@investigations_router.get("/{inv_id}/events")
async def stream_investigation_events(
    inv_id: str,
    request: Request,
    since: int = 0,
):
    """SSE tail-cursor stream. Yields one frame per row in
    opsrag_investigation_events with sequence > `since`. The browser
    reconnects with `since=<lastSeenSeq>` so a network blip never
    drops events.

    We deliberately do NOT set the SSE `event:` field -- per the WHATWG
    spec a typed event is only delivered to a listener registered for
    that exact name, but our backend emits 10+ event types and growing.
    Keeping every frame on the default channel means a single
    `onmessage` handler catches everything; the type lives in the JSON
    payload as `type` for downstream dispatch.
    """
    store = _get_store(request)

    # Verify the investigation exists so unknown ids 404 instead of
    # silently hanging the client.
    inv = await store.get_investigation(inv_id)
    if inv is None:
        raise HTTPException(status_code=404, detail=f"investigation {inv_id} not found")

    async def _gen() -> AsyncIterator[str]:
        cursor = since
        # ~30s window: 60 x 0.5s polls. The browser reconnects with
        # `since=<lastSeenSeq>` so this loop being short isn't
        # user-visible -- keeps each TCP connection short-lived so
        # load balancers don't trip idle timeouts.
        for _ in range(60):
            rows = await store.list_events_since(
                investigation_id=inv_id, since=cursor, limit=200,
            )
            for row in rows:
                cursor = int(row["sequence"])
                payload = json.dumps({
                    "sequence": row["sequence"],
                    "type": row["type"],
                    "payload": row["payload"],
                    "tags": row["tags"],
                    "ts": row["ts"],
                })
                # SSE frame: id + data on the default (unnamed) channel
                # so a single client `onmessage` listener catches all
                # types. See module docstring for the WHATWG rationale.
                yield f"id: {row['sequence']}\ndata: {payload}\n\n"
            # If the investigation has terminated, drain any remaining
            # rows and close -- no point holding the stream open for
            # 30s after the last event.
            inv_now = await store.get_investigation(inv_id)
            if inv_now and inv_now.get("status") in (
                "completed", "failed", "cancelled",
            ) and not rows:
                yield "event: close\ndata: {}\n\n"
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
