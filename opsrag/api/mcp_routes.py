"""HTTP transport for OpsRAG's MCP-server-as-proxy.

Two surfaces under `/api/mcp`:

  1. **Token management** -- Pomerium-authed. Logged-in users can mint
     and revoke their own MCP tokens via the UI.

       POST   /api/mcp/tokens             -- mint
       GET    /api/mcp/tokens             -- list (no plaintext)
       DELETE /api/mcp/tokens/{id}        -- revoke

  2. **MCP wire protocol** -- bearer-token authed via `Authorization:
     Bearer opsrag_...`. This is the surface external clients (Claude
     Code's `mcp-remote`, etc.) speak to:

       GET    /api/mcp/sse                -- server->client SSE stream
       POST   /api/mcp/messages           -- client->server JSON-RPC inbox

The SSE stream carries server-initiated events (today: just `endpoint`
on connect, then keep-alive heartbeats; future: tool result push if we
ever need streaming tools). The `messages` endpoint carries the
JSON-RPC requests for `initialize` / `tools/list` / `tools/call`. This
two-channel split is exactly what `mcp-remote` and the official
TypeScript SDK expect.

The integration agent mounts this router via `app.include_router(...)`
in `opsrag.api.server`. Prerequisites stashed on `app.state`:

  - `mcp_token_store` :: MCPTokenStore
  - `mcp_audit`       :: AuditLogger | None
  - `mcp_rate_limiter`:: TokenRateLimiter
  - `mcp_server`      :: MCPServer
  - `pomerium_verifier`, `tracking_user_config` (from M1, shared)
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from opsrag.auth import CurrentUser
from opsrag.auth.scopes import Scope, require_scope
from opsrag.mcp_server.server import MCPServer, _MCPCallCtx  # type: ignore[attr-defined]
from opsrag.mcp_server.token_store import MCPTokenStore

_log = logging.getLogger("opsrag.api.mcp_routes")

router = APIRouter(prefix="/mcp", tags=["mcp"])


# --- pydantic shapes ------------------------------------------------


class TokenCreateRequest(BaseModel):
    """Body for `POST /api/mcp/tokens`."""

    name: str = Field(
        ...,
        description="Human label for the token (shown in the UI's token list).",
        min_length=1,
        max_length=120,
    )
    expires_in_days: int | None = Field(
        default=None,
        ge=1,
        le=365,
        description=(
            "Days until the token auto-expires. `null` = never expires; "
            "capped at 365 to keep the long tail manageable."
        ),
    )


class TokenCreateResponse(BaseModel):
    """Response for `POST /api/mcp/tokens` -- plaintext shown ONCE."""

    id: str
    name: str
    token: str = Field(
        ...,
        description=(
            "The bearer token, prefixed `opsrag_`. Store this client-side "
            "immediately -- it is NOT retrievable again."
        ),
    )
    created_at: str
    expires_at: str | None


class TokenListItem(BaseModel):
    """Metadata for one token. No plaintext or hash."""

    id: str
    name: str
    created_at: str | None
    expires_at: str | None
    revoked_at: str | None
    last_used_at: str | None


class TokenListResponse(BaseModel):
    tokens: list[TokenListItem]


# --- helpers / dependencies -----------------------------------------


def _require_identified_user(user: CurrentUser) -> str:
    """Token-management routes attribute each token to a concrete user.

    The caller is resolved by ``require_scope(Scope.MCP)`` (above), which
    works across all auth modes: in **login** mode the user comes from the
    signed session COOKIE, in **oidc** mode from the verified bearer, in
    **open** mode it's the all-scopes anonymous user. A token needs a stable
    owner id to attribute + later scope listing/revocation to, so we 401 if
    there's no identity (``oid`` aliases the OIDC ``sub`` / session user id).
    Open-mode anonymous has no id -> minting is (correctly) refused.
    """
    if user.is_anonymous or not user.oid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="MCP token management requires an authenticated user",
        )
    return user.oid


async def get_mcp_token_dep(
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """FastAPI dependency for bearer-token MCP endpoints.

    Reads `Authorization: Bearer <opsrag_...>` and validates against
    `MCPTokenStore`. Returns the token row dict (with `id`,
    `user_oid`, `name`, ...). 401s on missing / invalid / revoked /
    expired tokens.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": 'Bearer realm="opsrag-mcp"'},
        )
    plaintext = authorization.split(" ", 1)[1].strip()
    store: MCPTokenStore | None = getattr(request.app.state, "mcp_token_store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="mcp token store not configured",
        )
    row = await store.validate(plaintext)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or expired token",
            headers={"WWW-Authenticate": 'Bearer realm="opsrag-mcp"'},
        )
    return row


# --- token management endpoints (Pomerium-authed) -------------------


@router.post(
    "/tokens",
    response_model=TokenCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_token(
    request: Request,
    body: TokenCreateRequest,
    current_user: CurrentUser = Depends(require_scope(Scope.MCP)),
) -> TokenCreateResponse:
    """Mint a new MCP bearer token for the calling user."""
    user_oid = _require_identified_user(current_user)
    store: MCPTokenStore | None = getattr(request.app.state, "mcp_token_store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="mcp token store not configured",
        )
    expires_at: datetime | None = None
    if body.expires_in_days is not None:
        expires_at = datetime.now(UTC) + timedelta(days=body.expires_in_days)
    plaintext, meta = await store.create(
        user_oid=user_oid, name=body.name, expires_at=expires_at,
    )
    _log.info(
        "mcp token minted user=%s name=%r expires_at=%s id=%s",
        user_oid, body.name, expires_at, meta["id"],
    )
    return TokenCreateResponse(
        id=meta["id"],
        name=meta["name"],
        token=plaintext,
        created_at=meta["created_at"] or "",
        expires_at=meta["expires_at"],
    )


@router.get("/tokens", response_model=TokenListResponse)
async def list_tokens(
    request: Request,
    current_user: CurrentUser = Depends(require_scope(Scope.MCP)),
) -> TokenListResponse:
    """List the calling user's MCP tokens (no plaintext)."""
    user_oid = _require_identified_user(current_user)
    store: MCPTokenStore | None = getattr(request.app.state, "mcp_token_store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="mcp token store not configured",
        )
    rows = await store.list_for_user(user_oid)
    return TokenListResponse(
        tokens=[TokenListItem(**r) for r in rows],
    )


@router.delete("/tokens/{token_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_token(
    request: Request,
    token_id: str,
    current_user: CurrentUser = Depends(require_scope(Scope.MCP)),
) -> None:
    """Revoke one of the calling user's MCP tokens. 404 if not found
    or already revoked, 401 if not the owner."""
    user_oid = _require_identified_user(current_user)
    store: MCPTokenStore | None = getattr(request.app.state, "mcp_token_store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="mcp token store not configured",
        )
    # Reject obviously bogus IDs early so we don't waste a DB round-trip.
    try:
        uuid.UUID(token_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="token_id must be a UUID")
    ok = await store.revoke(token_id=token_id, user_oid=user_oid)
    if not ok:
        raise HTTPException(status_code=404, detail="token not found or already revoked")
    _log.info("mcp token revoked user=%s id=%s", user_oid, token_id)


# --- Audit log (admin-only view of the centralized MCP) -------------


class AuditRow(BaseModel):
    occurred_at: str | None = None
    user_oid: str | None = None
    token_id: str | None = None
    tool_name: str
    args_hash: str | None = None
    latency_ms: int | None = None
    status: str
    error: str | None = None


class AuditListResponse(BaseModel):
    rows: list[AuditRow]
    total: int
    limit: int
    offset: int


class AuditTopTool(BaseModel):
    tool_name: str
    calls: int


class AuditSummaryResponse(BaseModel):
    total_calls: int
    error_count: int
    denied_count: int
    distinct_users: int
    distinct_tools: int
    top_tools: list[AuditTopTool]


def _require_audit(request: Request):
    audit = getattr(request.app.state, "mcp_audit", None)
    if audit is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="mcp audit log not configured (MCP server disabled)",
        )
    return audit


@router.get("/audit", response_model=AuditListResponse)
async def list_audit(
    request: Request,
    user: str | None = Query(None, description="filter by user_oid"),
    tool: str | None = Query(None, description="filter by tool_name"),
    status_filter: str | None = Query(None, alias="status", description="ok | denied | error"),
    since_minutes: int | None = Query(None, ge=1, description="only rows in the last N minutes"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _admin: CurrentUser = Depends(require_scope(Scope.ADMIN)),
) -> AuditListResponse:
    """Admin-only: read the centralized-MCP audit log -- who called which
    read-only tool, when, and the result. Args are NEVER stored literally;
    only ``args_hash`` (sha256 of canonical args) so secrets cannot leak."""
    audit = _require_audit(request)
    since = datetime.now(UTC) - timedelta(minutes=since_minutes) if since_minutes else None
    rows, total = await audit.query(
        user_oid=user, tool_name=tool, status=status_filter,
        since=since, limit=limit, offset=offset,
    )
    return AuditListResponse(rows=rows, total=total, limit=limit, offset=offset)


@router.get("/audit/summary", response_model=AuditSummaryResponse)
async def audit_summary(
    request: Request,
    since_minutes: int | None = Query(None, ge=1, description="window in minutes (default: all time)"),
    _admin: CurrentUser = Depends(require_scope(Scope.ADMIN)),
) -> AuditSummaryResponse:
    """Admin-only: aggregate stats for the audit dashboard strip."""
    audit = _require_audit(request)
    since = datetime.now(UTC) - timedelta(minutes=since_minutes) if since_minutes else None
    return AuditSummaryResponse(**await audit.summary(since=since))


# --- MCP wire protocol (bearer-token authed) ------------------------


def _sse_event(event: str, data: Any) -> bytes:
    """Format one SSE frame. `data` is JSON-encoded onto one line.

    Per the SSE spec (https://html.spec.whatwg.org/multipage/server-sent-events.html)
    each frame ends with a blank line (`\\n\\n`). Multi-line data must
    use repeated `data: ` lines; we keep payloads on a single line so
    the encoder is trivial.
    """
    body = data if isinstance(data, str) else json.dumps(data, default=str)
    return f"event: {event}\ndata: {body}\n\n".encode()


@router.get("/sse")
async def mcp_sse(
    request: Request,
    token_row: dict = Depends(get_mcp_token_dep),
) -> StreamingResponse:
    """Open the MCP server->client SSE event stream.

    Per the MCP HTTP+SSE transport:

      1. On open, server emits an `endpoint` event whose `data` field
         is the absolute URL the client should POST messages to. Most
         clients (including `mcp-remote`) parse this and stop hard-
         coding the messages URL.
      2. Server sends periodic keep-alive comments (`: ping`) so the
         connection isn't reaped by load balancers.

    The stream lives until the client disconnects. We don't push tool
    results here -- the JSON-RPC response for `tools/call` comes back
    on the POST response, not the SSE channel.
    """
    user_oid = token_row.get("user_oid")
    token_id = token_row.get("id")
    _log.info("mcp sse open token_id=%s user=%s", token_id, user_oid)

    # The "endpoint" the client should POST to. We CANNOT use
    # `request.url_for(...)` here because that reflects the
    # cluster-internal URL (http://, internal path) -- the backend lives
    # behind a TLS-terminating ingress + an nginx frontend that strips
    # the `/api/` prefix. The public-facing URL must be reconstructed
    # from forwarded headers, with the well-known public path prefix
    # (`/api/mcp/messages`) hardcoded since it's an architectural
    # constant (set by the frontend nginx proxy chain), not a
    # tenant value.
    scheme = (
        request.headers.get("x-forwarded-proto")
        or request.headers.get("x-forwarded-protocol")
        or request.url.scheme
        or "http"
    ).split(",")[0].strip()
    host = (
        request.headers.get("x-forwarded-host")
        or request.headers.get("host")
        or request.url.netloc
    ).split(",")[0].strip()
    messages_url = f"{scheme}://{host}/api/mcp/messages"

    async def _gen() -> AsyncIterator[bytes]:
        # 1) endpoint hint
        yield _sse_event("endpoint", messages_url)
        # 2) keep-alive loop. 15s heartbeat is well below the typical
        # 60s idle-timeout on cloud load balancers.
        while True:
            if await request.is_disconnected():
                _log.info("mcp sse client disconnected token_id=%s", token_id)
                return
            try:
                await asyncio.sleep(15.0)
            except asyncio.CancelledError:
                return
            # Comment-line keep-alive -- invisible to event consumers.
            yield b": ping\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # disable nginx response buffering
            "Connection": "keep-alive",
        },
    )


@router.post("/messages", name="mcp_messages")
async def mcp_messages(
    request: Request,
    token_row: dict = Depends(get_mcp_token_dep),
) -> Any:
    """Inbox for client->server JSON-RPC requests.

    Body is a JSON-RPC 2.0 envelope (request, notification, or batch).
    The MCPServer dispatcher (`opsrag.mcp_server.server.MCPServer`)
    handles routing to `initialize` / `tools/list` / `tools/call`,
    rate-limiting, audit-logging, and the wire shape.
    """
    server: MCPServer | None = getattr(request.app.state, "mcp_server", None)
    if server is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="mcp server not configured",
        )
    try:
        envelope = await request.json()
    except json.JSONDecodeError as exc:
        # Log detail server-side; return a generic message to the client
        # (the raw exc can leak request internals -- py/stack-trace-exposure).
        _log.warning("MCP parse error: %s", exc)
        return {
            "jsonrpc": "2.0", "id": None,
            "error": {"code": -32700, "message": "parse error"},
        }
    except Exception as exc:  # noqa: BLE001
        _log.warning("MCP parse error: %s", exc)
        return {
            "jsonrpc": "2.0", "id": None,
            "error": {"code": -32700, "message": "parse error"},
        }

    ctx = _MCPCallCtx(
        user_oid=token_row.get("user_oid"),
        token_id=token_row.get("id"),
    )
    response = await server.dispatch(envelope, ctx)
    # Notifications produce no response -- return 204 so the HTTP client
    # doesn't sit waiting for a body.
    if response is None:
        from fastapi import Response as _Response
        return _Response(status_code=status.HTTP_204_NO_CONTENT)
    return response
