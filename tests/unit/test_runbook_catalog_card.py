"""Unit tests: dynamic runbook-catalog card in the reasoner prompt.

Instead of hardcoding per-domain dispatch rules into the triage prompt
("kafka questions -> runbook_list"), the reasoner's system prompt carries a
LIVE catalog card built from the Runbook tab store + file catalog: every
operator-authored runbook advertises its own scope (title · service ·
issue_kind · tags), so authoring a runbook in the UI is ALL it takes for the
planner to start routing matching questions to it. Same inject-when-available
pattern as the repomap card.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from opsrag.mcp import runbooks as mcp_runbooks
from opsrag.runbooks.models import Runbook

NOW = datetime(2026, 7, 12, tzinfo=UTC)


def _rb(rb_id="22222222-2222-2222-2222-222222222222", **over):
    base = dict(
        id=rb_id,
        title="Kafka consumer lag",
        body_markdown="# body",
        service="kafka",
        issue_kind="dependency_outage",
        tags=["kafka", "kafka-connect"],
        created_at=NOW,
        updated_at=NOW,
    )
    base.update(over)
    return Runbook(**base)


class _FakeStore:
    def __init__(self, runbooks):
        self.runbooks = runbooks

    async def list(self, **kwargs):
        return self.runbooks

    async def search(self, query, **kwargs):  # pragma: no cover - not used here
        return []


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv("OPSRAG_SRE_KB_PATH", str(tmp_path))
    monkeypatch.setattr(mcp_runbooks, "_catalog", {})
    monkeypatch.setattr(mcp_runbooks, "_catalog_built_at", 0.0)
    monkeypatch.setattr(mcp_runbooks, "_catalog_card", "")
    monkeypatch.setattr(mcp_runbooks, "_catalog_card_built_at", 0.0)
    yield
    mcp_runbooks.set_runbook_store(None)


async def test_card_advertises_store_runbooks_with_scope():
    mcp_runbooks.set_runbook_store(_FakeStore([_rb()]))

    await mcp_runbooks.refresh_runbook_catalog_card(force=True)
    card = mcp_runbooks.runbook_catalog_card()

    assert "rb-22222222-2222-2222-2222-222222222222" in card
    assert "Kafka consumer lag" in card
    assert "kafka-connect" in card           # tags advertise the scope
    assert "runbook_load" in card            # tells the planner HOW to use it


async def test_card_empty_when_no_runbooks_anywhere():
    mcp_runbooks.set_runbook_store(None)

    await mcp_runbooks.refresh_runbook_catalog_card(force=True)

    assert mcp_runbooks.runbook_catalog_card() == ""


async def test_card_cached_until_ttl_or_force():
    store = _FakeStore([_rb()])
    mcp_runbooks.set_runbook_store(store)
    await mcp_runbooks.refresh_runbook_catalog_card(force=True)

    store.runbooks = [_rb(title="Something else entirely")]
    await mcp_runbooks.refresh_runbook_catalog_card()          # within TTL -> no-op
    assert "Kafka consumer lag" in mcp_runbooks.runbook_catalog_card()

    await mcp_runbooks.refresh_runbook_catalog_card(force=True)
    assert "Something else entirely" in mcp_runbooks.runbook_catalog_card()


async def test_reasoner_prompt_carries_the_card_before_the_base():
    from opsrag.agent.nodes.multi_agent import (
        _SYSTEM_REASONER_BASE,
        _build_reasoner_prompt,
    )

    mcp_runbooks.set_runbook_store(_FakeStore([_rb()]))
    await mcp_runbooks.refresh_runbook_catalog_card(force=True)

    prompt = _build_reasoner_prompt({})

    assert "Kafka consumer lag" in prompt
    assert "runbook_load" in prompt
    # Routing-level info must lead, not trail: buried after the (huge) base
    # prompt + repomap, the reasoner ignored it in prod while drilling
    # prometheus to the loop cap.
    assert prompt.index("Curated runbooks") < prompt.index(_SYSTEM_REASONER_BASE[:40])


async def test_triage_prompt_carries_the_card_too():
    """TRIAGE picks the first tool and seeds the plan -- in prod the card
    only lived in the reasoner prompt, so triage locked the investigation
    onto prometheus before the reasoner ever saw the catalog."""
    from opsrag.agent.nodes.multi_agent import _triage_system_prompt

    mcp_runbooks.set_runbook_store(_FakeStore([_rb()]))
    await mcp_runbooks.refresh_runbook_catalog_card(force=True)

    prompt = _triage_system_prompt()

    assert "Today's date (UTC)" in prompt
    assert "Kafka consumer lag" in prompt
    assert "runbook_load" in prompt
