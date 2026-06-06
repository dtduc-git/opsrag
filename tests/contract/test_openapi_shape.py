"""Contract test (T043): the OpenAPI schema exposes the documented
endpoints, and the auth posture is correct - the health endpoints are
reachable without a token while protected endpoints are not.

Mirrors contracts/http-api.md. Auth is enforced by a global middleware
(opsrag.api.oidc_enforcement) rather than per-route security schemes, so
"declares auth requirement" is asserted behaviourally (a probe request)
rather than by reading a securitySchemes block.

Webhook endpoints (/webhook/gitlab, /webhook/github) are deferred to
Phase 7 (T151) and are intentionally not asserted here yet.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

# Endpoints implemented as of US1 that the contract documents.
DOCUMENTED_PATHS = {
    "/healthz",
    "/readyz",
    "/query",
    "/usage",
    "/usage/{session_id}",
    "/sessions/{user_id}",
    "/sessions/{thread_id}",
    "/indexing/status",
    "/index/repo",
}

NO_AUTH_PATHS = {"/healthz", "/readyz"}


def _normalise(path: str) -> str:
    # FastAPI records path params with the handler's parameter name; the
    # contract uses its own names. Compare by structure (segment count +
    # literal segments), treating any {param} as a wildcard.
    return "/".join("{}" if seg.startswith("{") else seg for seg in path.split("/"))


def test_documented_endpoints_present(api_app) -> None:
    schema_paths = set(api_app.openapi()["paths"])
    have = {_normalise(p) for p in schema_paths}
    missing = {p for p in DOCUMENTED_PATHS if _normalise(p) not in have}
    assert not missing, f"OpenAPI schema is missing documented endpoints: {missing}"


def test_health_endpoints_need_no_auth(api_app) -> None:
    client = TestClient(api_app)
    # 200 for liveness; readiness may be 503 without a running stack, but
    # crucially it is NOT 401 - i.e. it is not auth-gated.
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code in (200, 503)


def test_protected_endpoint_requires_auth(api_app) -> None:
    client = TestClient(api_app)
    # A representative protected endpoint must reject the anonymous caller.
    assert client.get("/usage").status_code == 401
    assert client.post("/query", json={"query": "hi"}).status_code == 401
