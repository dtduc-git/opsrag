"""Integration test (T152): SCM webhook signing + dispatch.

Builds the app (no lifespan) and drives /webhook/gitlab and /webhook/github
through their secret/HMAC checks with the webhook secrets set via env.
"""
from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from opsrag.api.routes_webhooks import GITHUB_SECRET_ENV, GITLAB_SECRET_ENV


@pytest.fixture
def client(api_app):
    return TestClient(api_app)


# --- GitLab (X-Gitlab-Token) ------------------------------------------------
def test_gitlab_disabled_without_secret(client, monkeypatch) -> None:
    monkeypatch.delenv(GITLAB_SECRET_ENV, raising=False)
    r = client.post("/webhook/gitlab", json={})
    assert r.status_code == 503


def test_gitlab_rejects_bad_token(client, monkeypatch) -> None:
    monkeypatch.setenv(GITLAB_SECRET_ENV, "s3cret")
    r = client.post("/webhook/gitlab", json={}, headers={"X-Gitlab-Token": "wrong"})
    assert r.status_code == 401
    assert r.json()["error"] == "unauthenticated" or "error" in r.json()


def test_gitlab_accepts_valid_token(client, monkeypatch) -> None:
    monkeypatch.setenv(GITLAB_SECRET_ENV, "s3cret")
    body = {"project": {"path_with_namespace": "group/project"}, "ref": "refs/heads/main"}
    r = client.post("/webhook/gitlab", json=body, headers={"X-Gitlab-Token": "s3cret"})
    assert r.status_code == 200
    assert r.json()["status"] == "accepted"
    assert r.json()["repo"] == "group/project"


# --- GitHub (X-Hub-Signature-256) -------------------------------------------
def test_github_disabled_without_secret(client, monkeypatch) -> None:
    monkeypatch.delenv(GITHUB_SECRET_ENV, raising=False)
    r = client.post("/webhook/github", json={})
    assert r.status_code == 503


def test_github_rejects_bad_signature(client, monkeypatch) -> None:
    monkeypatch.setenv(GITHUB_SECRET_ENV, "s3cret")
    r = client.post(
        "/webhook/github", json={}, headers={"X-Hub-Signature-256": "sha256=deadbeef"}
    )
    assert r.status_code == 401


def test_github_accepts_valid_signature(client, monkeypatch) -> None:
    monkeypatch.setenv(GITHUB_SECRET_ENV, "s3cret")
    body = {"repository": {"full_name": "org/repo"}, "ref": "refs/heads/main"}
    raw = json.dumps(body).encode()
    sig = "sha256=" + hmac.new(b"s3cret", raw, hashlib.sha256).hexdigest()
    r = client.post(
        "/webhook/github",
        content=raw,
        headers={"X-Hub-Signature-256": sig, "Content-Type": "application/json"},
    )
    assert r.status_code == 200
    assert r.json()["repo"] == "org/repo"


def test_webhooks_bypass_oidc(api_app) -> None:
    # Even with the OIDC verifier set, webhook paths are not Bearer-gated
    # (they self-authenticate). Without a secret configured -> 503, never 401.
    from tests.conftest import StubOIDCVerifier

    api_app.state.oidc_verifier = StubOIDCVerifier()
    client = TestClient(api_app)
    r = client.post("/webhook/gitlab", json={})
    assert r.status_code == 503  # not 401: OIDC middleware did not gate it
