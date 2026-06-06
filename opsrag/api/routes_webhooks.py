"""SCM push webhooks (T151): /webhook/gitlab and /webhook/github.

These endpoints authenticate with a per-provider webhook SECRET (not OIDC), so
they are listed in opsrag.api.oidc_enforcement.NO_AUTH_PATHS and do their own
signature check here:

- GitLab sends ``X-Gitlab-Token: <secret>`` (compared timing-safe).
- GitHub sends ``X-Hub-Signature-256: sha256=<hmac>`` (HMAC-SHA256 of the raw
  body keyed by the secret).

Secrets are read from env (never inline): ``OPSRAG_GITLAB_WEBHOOK_SECRET`` /
``OPSRAG_GITHUB_WEBHOOK_SECRET``. When a provider's secret is unset the endpoint
returns 503 (that webhook is disabled) rather than accepting unauthenticated
calls. A verified push best-effort triggers a reindex of the affected repo via
the ingestion pipeline if one is wired; otherwise it is acknowledged (the
scheduler reindexes on its own cadence).
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os

from fastapi import APIRouter, HTTPException, Request

_log = logging.getLogger("opsrag.api.webhooks")

webhooks_router = APIRouter(tags=["webhooks"])

GITLAB_SECRET_ENV = "OPSRAG_GITLAB_WEBHOOK_SECRET"
GITHUB_SECRET_ENV = "OPSRAG_GITHUB_WEBHOOK_SECRET"


def _secret(env_name: str) -> str | None:
    v = os.environ.get(env_name)
    return v.strip() if v and v.strip() else None


async def _maybe_reindex(request: Request, repo: str | None, branch: str | None) -> bool:
    """Best-effort: trigger a repo reindex if the pipeline is wired. Never
    raises; returns whether a reindex was kicked off."""
    pipeline = getattr(request.app.state, "ingestion_pipeline", None)
    if pipeline is None or not repo:
        return False
    try:
        import asyncio

        asyncio.create_task(pipeline.index_repo(repo, branch or "main"))
        return True
    except Exception as exc:  # noqa: BLE001 -- webhook must still 202
        _log.warning("webhook reindex dispatch failed for %s: %s", repo, exc)
        return False


@webhooks_router.post("/webhook/gitlab")
async def gitlab_webhook(request: Request) -> dict:
    secret = _secret(GITLAB_SECRET_ENV)
    if secret is None:
        raise HTTPException(status_code=503, detail="gitlab webhook not configured")
    token = request.headers.get("x-gitlab-token", "")
    if not hmac.compare_digest(token, secret):
        raise HTTPException(status_code=401, detail="invalid gitlab webhook token")
    body = await request.json()
    repo = (
        (body.get("project") or {}).get("path_with_namespace")
        if isinstance(body, dict)
        else None
    )
    branch = None
    if isinstance(body, dict) and isinstance(body.get("ref"), str):
        branch = body["ref"].rsplit("/", 1)[-1]
    reindexing = await _maybe_reindex(request, repo, branch)
    return {"status": "accepted", "repo": repo, "reindexing": reindexing}


@webhooks_router.post("/webhook/github")
async def github_webhook(request: Request) -> dict:
    secret = _secret(GITHUB_SECRET_ENV)
    if secret is None:
        raise HTTPException(status_code=503, detail="github webhook not configured")
    raw = await request.body()
    sent = request.headers.get("x-hub-signature-256", "")
    expected = "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sent):
        raise HTTPException(status_code=401, detail="invalid github webhook signature")
    body = await request.json()
    repo = (body.get("repository") or {}).get("full_name") if isinstance(body, dict) else None
    branch = None
    if isinstance(body, dict) and isinstance(body.get("ref"), str):
        branch = body["ref"].rsplit("/", 1)[-1]
    reindexing = await _maybe_reindex(request, repo, branch)
    return {"status": "accepted", "repo": repo, "reindexing": reindexing}
