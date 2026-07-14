"""Unit tests: Runbook-tab (RunbookStore) runbooks reach the CHAT path.

The chat agent's `runbook_list` / `runbook_load` tools used to read ONLY the
file catalog under OPSRAG_SRE_KB_PATH — hand-authored runbooks created in the
UI Runbooks tab (RunbookStore, Postgres+Qdrant) were consumed exclusively by
the Investigations Lane A, so a freshly authored tab runbook showed
USED=0 forever and chat answers fell back to stale RAG chunks.

Fix under test: the MCP handlers merge BOTH sources — tab-store entries first
(operator-curated, priority-ordered), file-catalog entries after; store
runbooks are addressable as `rb-<id>` and loading one bumps its used_count
(the UI's USED column).
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from opsrag.mcp import runbooks as mcp_runbooks
from opsrag.runbooks.models import Runbook, RunbookHit

NOW = datetime(2026, 7, 12, tzinfo=UTC)


def _rb(rb_id="11111111-1111-1111-1111-111111111111", **over):
    base = dict(
        id=rb_id,
        title="Kafka consumer lag",
        body_markdown="# Runbook: Kafka Consumer Lag\n\nStep 0 ...",
        service="kafka",
        issue_kind="dependency-outage",
        tags=["kafka", "kafka-connect"],
        priority=100,
        created_at=NOW,
        updated_at=NOW,
    )
    base.update(over)
    return Runbook(**base)


class _FakeStore:
    def __init__(self, runbooks):
        self.runbooks = {r.id: r for r in runbooks}
        self.search_calls: list[str] = []
        self.record_use_calls: list[str] = []

    async def list(self, **kwargs):
        return list(self.runbooks.values())

    async def search(self, query, **kwargs):
        self.search_calls.append(query)
        return [RunbookHit(runbook=r, score=0.9) for r in self.runbooks.values()]

    async def get(self, runbook_id):
        if runbook_id not in self.runbooks:
            raise KeyError(runbook_id)
        return self.runbooks[runbook_id]

    async def record_use(self, runbook_id, *, thumbs=None):
        self.record_use_calls.append(runbook_id)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Point the file catalog at an empty tmp dir and reset its cache, so
    each test controls both sources explicitly."""
    monkeypatch.setenv("OPSRAG_SRE_KB_PATH", str(tmp_path))
    monkeypatch.setattr(mcp_runbooks, "_catalog", {})
    monkeypatch.setattr(mcp_runbooks, "_catalog_built_at", 0.0)
    yield
    mcp_runbooks.set_runbook_store(None)


def _write_file_runbook(tmp_path, name="runbook-file-one"):
    d = tmp_path / "docs" / "runbooks" / "kafka"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.md").write_text(
        f"---\nid: {name}\nlast_reviewed: 2026-01-01\n---\n"
        f"# File runbook one\n\n## Summary\nFile-based runbook for tests.\n"
    )


async def test_list_merges_store_first_then_files(tmp_path):
    _write_file_runbook(tmp_path)
    store = _FakeStore([_rb()])
    mcp_runbooks.set_runbook_store(store)

    out = await mcp_runbooks._h_list_runbooks(None, {})

    names = [r["name"] for r in out["runbooks"]]
    assert names[0] == "rb-11111111-1111-1111-1111-111111111111"
    assert "runbook-file-one" in names
    assert names.index(names[0]) < names.index("runbook-file-one")
    assert out["count"] == len(names)


async def test_list_with_topic_searches_store(tmp_path):
    store = _FakeStore([_rb()])
    mcp_runbooks.set_runbook_store(store)

    out = await mcp_runbooks._h_list_runbooks(None, {"topic": "kafka consumer lag"})

    assert store.search_calls == ["kafka consumer lag"]
    assert any(r["name"].startswith("rb-") for r in out["runbooks"])


async def test_load_rb_id_returns_markdown_and_records_use():
    store = _FakeStore([_rb()])
    mcp_runbooks.set_runbook_store(store)

    out = await mcp_runbooks._h_load_runbook(
        None, {"name": "rb-11111111-1111-1111-1111-111111111111"}
    )

    assert "Kafka Consumer Lag" in out["markdown"]
    assert store.record_use_calls == ["11111111-1111-1111-1111-111111111111"]
    assert "error" not in out


async def test_load_unknown_rb_id_lists_available():
    store = _FakeStore([_rb()])
    mcp_runbooks.set_runbook_store(store)

    out = await mcp_runbooks._h_load_runbook(None, {"name": "rb-does-not-exist"})

    assert out["error"] == "runbook not found"
    assert any(n.startswith("rb-") for n in out["available_names_first_5"])


async def test_no_store_keeps_file_only_behavior(tmp_path):
    mcp_runbooks.set_runbook_store(None)
    _write_file_runbook(tmp_path)

    out = await mcp_runbooks._h_list_runbooks(None, {})
    assert [r["name"] for r in out["runbooks"]] == ["runbook-file-one"]

    loaded = await mcp_runbooks._h_load_runbook(None, {"name": "runbook-file-one"})
    assert "File runbook one" in loaded["markdown"]


async def test_both_sources_empty_returns_actionable_error():
    mcp_runbooks.set_runbook_store(None)

    out = await mcp_runbooks._h_list_runbooks(None, {})

    assert out["count"] == 0
    assert "error" in out
