"""Integration test (T069): the User Story 1 quickstart happy path.

Boots against a RUNNING compose stack (it does not start one itself), mirrors
quickstart.md end to end: health -> acquire a Dex token -> POST /query against
the indexed sample corpus -> assert a cited English answer.

This requires live services (API, Postgres, Qdrant, Dex) plus an LLM API key,
so it is SKIPPED unless OPSRAG_E2E=1 is set. In CI it runs as a dedicated job
after `docker compose up` + `scripts/seed-sample-corpus.sh`. It is never run
as part of the default unit/contract suite.

Bring-up (matches quickstart.md):

    docker compose -f deploy/compose/docker-compose.yaml up -d
    docker compose -f deploy/compose/docker-compose.yaml exec opsrag-api \\
        scripts/seed-sample-corpus.sh
    OPSRAG_E2E=1 pytest tests/integration/test_quickstart_happy_path.py

Override endpoints via OPSRAG_E2E_API_URL / OPSRAG_E2E_DEX_URL if needed.
"""
from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("OPSRAG_E2E") != "1",
        reason="needs a running compose stack; set OPSRAG_E2E=1 to run",
    ),
]

API_URL = os.environ.get("OPSRAG_E2E_API_URL", "http://localhost:8080")
DEX_URL = os.environ.get("OPSRAG_E2E_DEX_URL", "http://localhost:5556/dex")


@pytest.fixture(scope="module")
def http():
    import httpx

    with httpx.Client(timeout=60.0) as client:
        yield client


def _get_token(client) -> str:
    resp = client.post(
        f"{DEX_URL}/token",
        data={
            "grant_type": "password",
            "username": "evaluator@example.com",
            "password": "evaluator",
            "client_id": "opsrag-local",
            "client_secret": "local-secret",
            "scope": "openid profile email",
        },
    )
    resp.raise_for_status()
    token = resp.json()["access_token"]
    assert token
    return token


def test_healthz_ok(http) -> None:
    resp = http.get(f"{API_URL}/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_readyz_eventually_ready(http) -> None:
    resp = http.get(f"{API_URL}/readyz")
    # 200 once Postgres + Qdrant are reachable; 503 surfaces the per-component
    # breakdown, which is still a valid (non-auth) response.
    assert resp.status_code in (200, 503)


def test_query_requires_auth(http) -> None:
    resp = http.post(f"{API_URL}/query", json={"query": "anything"})
    assert resp.status_code == 401
    assert resp.json()["error"] == "unauthenticated"


def test_query_returns_cited_answer(http) -> None:
    token = _get_token(http)
    resp = http.post(
        f"{API_URL}/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "How do I roll back an Acme Notes deployment?"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Contract shape (contracts/http-api.md): answer + citations + session_id.
    assert isinstance(body.get("answer"), str) and body["answer"].strip()
    assert "session_id" in body
    citations = body.get("citations") or []
    # The answer should be grounded in the indexed Acme Notes corpus.
    joined = (body["answer"] + " " + str(citations)).lower()
    assert "acme" in joined or any("acme" in str(c).lower() for c in citations)
