"""Contract test (T044): every non-health endpoint returns 401 when no
Authorization header is present (FR-016, contracts/http-api.md).

The global OIDCAuthMiddleware rejects unauthenticated requests before they
reach a handler, so this needs no running providers - the app is built
without its lifespan.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# (method, concrete-path) for each protected endpoint implemented in US1.
PROTECTED_ENDPOINTS = [
    ("GET", "/usage"),
    ("GET", "/usage/some-session"),
    ("GET", "/sessions/some-user"),
    ("DELETE", "/sessions/some-thread"),
    ("GET", "/indexing/status"),
    ("POST", "/query"),
    ("POST", "/index/repo"),
]

HEALTH_ENDPOINTS = [("GET", "/healthz"), ("GET", "/readyz")]


@pytest.fixture
def client(api_app) -> TestClient:
    return TestClient(api_app)


@pytest.mark.parametrize("method,path", PROTECTED_ENDPOINTS)
def test_protected_endpoint_401_without_auth(client: TestClient, method: str, path: str) -> None:
    resp = client.request(method, path, json={} if method == "POST" else None)
    assert resp.status_code == 401, f"{method} {path} should be 401 without auth"
    body = resp.json()
    assert body["error"] == "unauthenticated"
    assert "reason" in body and "request_id" in body


@pytest.mark.parametrize("method,path", HEALTH_ENDPOINTS)
def test_health_endpoint_not_401(client: TestClient, method: str, path: str) -> None:
    resp = client.request(method, path)
    assert resp.status_code != 401, f"{method} {path} must not be auth-gated"
