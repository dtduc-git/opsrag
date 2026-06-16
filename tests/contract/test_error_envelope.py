"""Contract test (T045): every 4xx/5xx response uses the stable error
envelope (contracts/http-api.md)::

    {"error": "<machine_code>", "reason": "<human readable>", "request_id": "<uuid>"}

with ``error`` drawn from the closed code set. Covers the 401 (auth
middleware), 404 (router default), and 422 (request validation) paths.
Authenticated cases use the offline stub verifier from conftest.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from opsrag.api.errors import ERROR_CODES
from tests.conftest import VALID_TEST_TOKEN

ENVELOPE_KEYS = {"error", "reason", "request_id"}


def _assert_envelope(body: dict, expected_code: str) -> None:
    assert ENVELOPE_KEYS <= set(body), f"missing envelope keys: {body}"
    assert body["error"] in ERROR_CODES
    assert body["error"] == expected_code
    assert isinstance(body["reason"], str) and body["reason"]
    assert isinstance(body["request_id"], str) and body["request_id"]


def test_401_envelope(api_app) -> None:
    client = TestClient(api_app)
    resp = client.post("/query", json={"query": "hi"})  # no auth
    assert resp.status_code == 401
    _assert_envelope(resp.json(), "unauthenticated")


def test_404_envelope(stub_app, auth_headers) -> None:
    client = TestClient(stub_app)
    resp = client.get("/definitely-not-a-real-route", headers=auth_headers)
    assert resp.status_code == 404
    _assert_envelope(resp.json(), "not_found")


def test_422_envelope(stub_app, auth_headers) -> None:
    client = TestClient(stub_app)
    # Authenticated but schema-invalid body: `query` must be a string (an empty
    # query is now valid for bare-image turns, so we trigger the 422 with a
    # type error rather than an omitted field).
    resp = client.post("/query", json={"query": ["not", "a", "string"]}, headers=auth_headers)
    assert resp.status_code == 422
    _assert_envelope(resp.json(), "bad_request")


def test_request_ids_are_unique_per_request(api_app) -> None:
    client = TestClient(api_app)
    r1 = client.get("/usage")
    r2 = client.get("/usage")
    assert r1.json()["request_id"] != r2.json()["request_id"]


def test_invalid_token_is_unauthenticated(api_app) -> None:
    # A malformed/garbage bearer is rejected with the envelope too. The real
    # dex verifier rejects this without any network round-trip (bad JWT shape).
    client = TestClient(api_app)
    resp = client.get("/usage", headers={"Authorization": f"Bearer {VALID_TEST_TOKEN}"})
    assert resp.status_code == 401
    _assert_envelope(resp.json(), "unauthenticated")
