"""Stable HTTP error envelope (contracts/http-api.md).

Every non-2xx response carries the same JSON shape::

    {"error": "<machine_code>", "reason": "<human readable>", "request_id": "<uuid>"}

Machine codes (closed set): ``unauthenticated``, ``forbidden``,
``bad_request``, ``not_found``, ``conflict``, ``mcp_misconfigured``,
``mcp_upstream_failure``, ``agent_timeout``, ``internal``.

Call :func:`register_error_handlers` once on the FastAPI app. It installs
handlers for ``HTTPException`` (incl. the 404 router default), request
validation errors (422 -> ``bad_request``), and any uncaught exception
(-> ``internal``, with the reason scrubbed so internals never leak).
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

_log = logging.getLogger("opsrag.api.errors")

# Closed set of machine codes the API may emit.
ERROR_CODES: frozenset[str] = frozenset(
    {
        "unauthenticated",
        "forbidden",
        "bad_request",
        "not_found",
        "conflict",
        "mcp_misconfigured",
        "mcp_upstream_failure",
        "agent_timeout",
        "internal",
    }
)

# Default status -> machine code mapping.
_STATUS_TO_CODE: dict[int, str] = {
    400: "bad_request",
    401: "unauthenticated",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    422: "bad_request",
    424: "mcp_misconfigured",
    502: "mcp_upstream_failure",
    504: "agent_timeout",
}


def code_for_status(status_code: int) -> str:
    """Machine code for an HTTP status; 5xx (except mapped) -> ``internal``."""
    if status_code in _STATUS_TO_CODE:
        return _STATUS_TO_CODE[status_code]
    if 400 <= status_code < 500:
        return "bad_request"
    return "internal"


def request_id_for(request: Request) -> str:
    """Stable per-request id. Reuses one stamped by upstream middleware
    (``request.state.request_id``) so the same id appears in logs and the
    response body; falls back to a fresh uuid4."""
    rid = getattr(request.state, "request_id", None)
    if rid:
        return str(rid)
    return uuid.uuid4().hex


def envelope(code: str, reason: str, request_id: str) -> dict[str, Any]:
    if code not in ERROR_CODES:
        code = "internal"
    return {"error": code, "reason": reason, "request_id": request_id}


def error_response(
    status_code: int,
    reason: str,
    request: Request,
    *,
    code: str | None = None,
) -> JSONResponse:
    rid = request_id_for(request)
    body = envelope(code or code_for_status(status_code), reason, rid)
    return JSONResponse(status_code=status_code, content=body)


def register_error_handlers(app: FastAPI) -> None:
    """Install the envelope handlers on ``app``."""

    @app.exception_handler(StarletteHTTPException)
    async def _http_exc(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        reason = exc.detail if isinstance(exc.detail, str) else "request failed"
        return error_response(exc.status_code, reason, request)

    @app.exception_handler(RequestValidationError)
    async def _validation_exc(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Summarise the first error; the full list is verbose and can echo
        # request content, so keep the reason short and structural.
        errors = exc.errors()
        if errors:
            first = errors[0]
            loc = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
            reason = f"invalid request: {loc or 'body'}: {first.get('msg', 'invalid')}"
        else:
            reason = "invalid request"
        return error_response(422, reason, request, code="bad_request")

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Never leak internals. Log the real error with the request id so it
        # is correlatable; return a scrubbed envelope.
        rid = request_id_for(request)
        _log.exception("unhandled error (request_id=%s): %s", rid, exc)
        return JSONResponse(
            status_code=500,
            content=envelope("internal", "internal server error", rid),
        )
