"""Unit test (F1, route level): POST /index/repo and POST /index/source must
flush the tracker to Postgres with ``guarded=True`` whenever a Job launcher is
active.

Why this matters: on the serving pod, boot backfill restores every Qdrant repo
as ``done`` in the in-memory tracker. An *unguarded* admin-triggered flush then
re-UPSERTs those ``done`` rows, stomping a live ephemeral Job's ``indexing`` row
back to ``done``. The handlers must compute
``guarded = request.app.state.job_launcher is not None`` and forward it.

We call the route handlers directly (bypassing the auth ``Depends`` default) with
a fake ``Request`` whose ``app.state`` carries a recording fake store.
"""
from __future__ import annotations

import asyncio
import types

import pytest

from opsrag.api.models import IndexRepoRequest, IndexSourceRequest
from opsrag.api.routes import index_repo, index_source


class _RecordingStore:
    def __init__(self) -> None:
        self.flush_guarded: list[bool] = []

    async def flush(self, summary, jobs, *, guarded: bool = False) -> None:
        self.flush_guarded.append(guarded)


class _FakeLauncher:
    async def launch_repo(self, repo, branch) -> str:
        return "job-abc"

    async def launch_source(self, source_type, scope) -> str:
        return "job-def"


class _FakePipeline:
    # Present only so the no-launcher fallback path can construct _run(); never
    # exercised in the launcher-present cases.
    sources = {"confluence": object()}

    async def index_repo(self, repo, branch, patterns=None) -> int:
        return 0

    async def index_source(self, source_type, scope) -> int:
        return 0


def _make_request(*, launcher, store) -> types.SimpleNamespace:
    state = types.SimpleNamespace(
        ingestion_pipeline=_FakePipeline(),
        qa_cache=None,
        config=None,
        index_store=store,
        job_launcher=launcher,
    )
    app = types.SimpleNamespace(state=state)
    return types.SimpleNamespace(app=app)


@pytest.mark.asyncio
async def test_index_repo_flush_guarded_when_launcher_active():
    store = _RecordingStore()
    req = _make_request(launcher=_FakeLauncher(), store=store)
    await index_repo(IndexRepoRequest(repo="acme/app", branch="main"), req)  # type: ignore[arg-type]
    assert store.flush_guarded == [True]


@pytest.mark.asyncio
async def test_index_repo_flush_unguarded_when_no_launcher():
    store = _RecordingStore()
    req = _make_request(launcher=None, store=store)
    await index_repo(IndexRepoRequest(repo="acme/app", branch="main"), req)  # type: ignore[arg-type]
    # let the fire-and-forget _run() task settle (it uses the fake pipeline)
    await asyncio.sleep(0)
    assert store.flush_guarded == [False]


@pytest.mark.asyncio
async def test_index_source_flush_guarded_when_launcher_active():
    store = _RecordingStore()
    req = _make_request(launcher=_FakeLauncher(), store=store)
    await index_source(
        IndexSourceRequest(source_type="confluence", scope="OPS"), req,  # type: ignore[arg-type]
    )
    assert store.flush_guarded == [True]


@pytest.mark.asyncio
async def test_index_source_flush_unguarded_when_no_launcher():
    store = _RecordingStore()
    req = _make_request(launcher=None, store=store)
    await index_source(
        IndexSourceRequest(source_type="confluence", scope="OPS"), req,  # type: ignore[arg-type]
    )
    await asyncio.sleep(0)
    assert store.flush_guarded == [False]
