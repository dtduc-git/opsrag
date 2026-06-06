"""Health endpoints (contracts/http-api.md, T058).

- ``GET /healthz`` - liveness. 200 as long as the process serves.
- ``GET /readyz``  - readiness. 200 only once the app has initialised its
  providers and the backing stores (vector store, session DB) are
  reachable. Returns 503 with a per-component breakdown otherwise.

Both bypass auth (see ``opsrag.api.oidc_enforcement.NO_AUTH_PATHS``).

Per-MCP readiness probing (every enabled integration's health URL) is
layered on in Phase 4 (T088); this module already returns a ``components``
map that that work extends.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from opsrag import __version__

_log = logging.getLogger("opsrag.api.health")

health_router = APIRouter(tags=["health"])


@health_router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


async def _probe_store(store: Any) -> tuple[bool, str]:
    """Best-effort health probe for a provider store. Calls the first of
    ``health_check`` / ``ping`` / ``healthz`` it exposes (awaiting if
    async); falls back to "present" when no probe method exists. Never
    raises."""
    if store is None:
        return False, "not initialised"
    for meth in ("health_check", "ping", "healthz"):
        fn = getattr(store, meth, None)
        if callable(fn):
            try:
                result = fn()
                if hasattr(result, "__await__"):
                    await result
                return True, "ok"
            except Exception as exc:  # noqa: BLE001
                return False, f"{type(exc).__name__}: {exc}"
    return True, "present"


@health_router.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    providers = getattr(request.app.state, "providers", None)

    components: dict[str, dict[str, str]] = {}

    if providers is None:
        # App constructed but lifespan startup not finished (or running
        # without a stack). Not ready by definition.
        components["providers"] = {"status": "down", "detail": "not initialised"}
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "components": components},
        )

    ready = True
    for name, store in (
        ("vector_store", getattr(providers, "vector_store", None)),
        ("session_store", getattr(providers, "session_store", None)),
    ):
        ok, detail = await _probe_store(store)
        components[name] = {"status": "up" if ok else "down", "detail": detail}
        ready = ready and ok

    # Per-MCP readiness (T088): for each enabled integration, report its status
    # and, where the registry declares a fully-formed health URL, best-effort
    # probe it. Disabled integrations are omitted. A failed MCP probe degrades
    # readiness so an operator who enabled an integration sees it is unreachable.
    mcp_components, mcp_ready = await _probe_enabled_mcps(request)
    components.update(mcp_components)
    ready = ready and mcp_ready

    status_code = 200 if ready else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if ready else "not_ready", "components": components},
    )


async def _probe_enabled_mcps(request: Request) -> tuple[dict[str, dict[str, str]], bool]:
    """Probe each enabled MCP integration. Returns ({mcp:<name>: {...}}, ready)."""
    cfg = getattr(request.app.state, "config", None)
    if cfg is None:
        return {}, True
    try:
        from opsrag.mcp.registry import REGISTRY
        from opsrag.mcp_server.registry_loader import enabled_integration_names
    except Exception:  # noqa: BLE001
        return {}, True

    out: dict[str, dict[str, str]] = {}
    ready = True
    for name in enabled_integration_names(cfg):
        integration = REGISTRY.get(name)
        template = getattr(integration, "health_url_template", None) if integration else None
        # Only probe a concrete URL (no unsubstituted {tokens} / $ENV refs).
        if template and "{" not in template and "$" not in template:
            ok, detail = await _probe_url(template)
            out[f"mcp:{name}"] = {"status": "up" if ok else "down", "detail": detail}
            ready = ready and ok
        else:
            # Enabled but no probeable URL -> report enabled, don't gate readiness.
            out[f"mcp:{name}"] = {"status": "enabled", "detail": "no health probe"}
    return out, ready


async def _probe_url(url: str) -> tuple[bool, str]:
    """Best-effort GET with a short timeout; never raises."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(url)
        # Any HTTP response (even 401 from an unauthenticated probe) means the
        # endpoint is reachable; connection errors mean it is not.
        return True, f"reachable (HTTP {resp.status_code})"
    except Exception as exc:  # noqa: BLE001
        return False, f"unreachable: {type(exc).__name__}"
