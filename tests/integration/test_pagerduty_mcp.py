"""Integration test: the PagerDuty MCP tools against the fake backend.

Exercises representative tools through build_fake() with no network and no
PagerDuty token, asserting shape-faithful responses and the registry's
declared tool set. Follows the Rootly reference (FR-012).
"""
from __future__ import annotations

import pytest

from opsrag.mcp.pagerduty import build_fake
from opsrag.mcp.registry import REGISTRY


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()  # restore the real module-level _get


def test_fake_exposes_registry_tool_set(fake) -> None:
    # The fake's tools must match exactly what the registry declares.
    assert set(fake.tool_names()) == set(REGISTRY["pagerduty"].tool_names)


@pytest.mark.asyncio
async def test_list_incidents(fake) -> None:
    result = await fake.call(
        "pagerduty_list_incidents", {"statuses": ["triggered", "acknowledged"]}
    )
    assert result["count"] == 1
    assert result["more"] is False
    inc = result["incidents"][0]
    assert inc["id"] == "PT4KHLK"
    assert inc["incident_number"] == 1234
    assert inc["status"] == "resolved"
    assert inc["urgency"] == "high"
    assert inc["priority"] == "P1"
    assert inc["service"] == "acme-notes-be"
    assert inc["assignees"] == ["On-call SRE"]


@pytest.mark.asyncio
async def test_get_incident(fake) -> None:
    result = await fake.call("pagerduty_get_incident", {"incident_id": "PT4KHLK"})
    inc = result["incident"]
    assert inc["title"] == "Database connection pool exhausted"
    assert inc["escalation_policy"] == "Platform on-call"
    assert inc["url"].endswith("/incidents/PT4KHLK")


@pytest.mark.asyncio
async def test_list_services(fake) -> None:
    result = await fake.call("pagerduty_list_services", {"query": "notes"})
    assert result["count"] == 1
    svc = result["services"][0]
    assert svc["id"] == "PSVC001"
    assert svc["name"] == "acme-notes-be"
    assert svc["status"] == "active"


@pytest.mark.asyncio
async def test_list_oncalls(fake) -> None:
    result = await fake.call(
        "pagerduty_list_oncalls", {"escalation_policy_ids": ["PEP001"]}
    )
    assert result["count"] == 1
    oc = result["oncalls"][0]
    assert oc["user"] == "On-call SRE"
    assert oc["escalation_policy"] == "Platform on-call"
    assert oc["escalation_level"] == 1
    assert oc["schedule"] == "Primary rotation"


@pytest.mark.asyncio
async def test_get_incident_log_entries(fake) -> None:
    result = await fake.call(
        "pagerduty_get_incident_log_entries", {"incident_id": "PT4KHLK"}
    )
    assert result["incident_id"] == "PT4KHLK"
    assert result["count"] == 1
    entry = result["log_entries"][0]
    assert entry["type"] == "resolve_log_entry"
    assert entry["agent"] == "On-call SRE"


@pytest.mark.asyncio
async def test_unknown_tool_raises(fake) -> None:
    with pytest.raises(KeyError):
        await fake.call("pagerduty_does_not_exist", {})
