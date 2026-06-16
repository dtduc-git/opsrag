"""Unit tests for the PagerDuty MCP tools against the offline fake backend.

Exercises EVERY tool through build_fake() with no network and no PagerDuty
credentials, asserting the exact tool set and shape-faithful parsed
responses. Follows the Sentry/Rootly reference pattern (FR-012).
"""
from __future__ import annotations

import pytest

from opsrag.mcp.pagerduty import PAGERDUTY_TOOLS, build_fake, get_tool

_EXPECTED_TOOLS = {
    "pagerduty_list_incidents",
    "pagerduty_get_incident",
    "pagerduty_list_services",
    "pagerduty_list_oncalls",
    "pagerduty_get_incident_log_entries",
}


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()  # restore the real module-level _get


def test_tool_set_is_exactly_the_contract() -> None:
    assert {t.name for t in PAGERDUTY_TOOLS} == _EXPECTED_TOOLS
    assert len(PAGERDUTY_TOOLS) == len(_EXPECTED_TOOLS)


def test_fake_exposes_full_tool_set(fake) -> None:
    assert set(fake.tool_names()) == _EXPECTED_TOOLS


async def test_list_incidents(fake) -> None:
    res = await fake.call("pagerduty_list_incidents", {})
    assert res["count"] == 1
    inc = res["incidents"][0]
    assert inc["id"] == "PT4KHLK"
    assert inc["status"] == "resolved"
    assert inc["service"] == "acme-notes-be"


async def test_get_incident(fake) -> None:
    res = await fake.call("pagerduty_get_incident", {"incident_id": "PT4KHLK"})
    assert res["incident"]["incident_number"] == 1234
    assert res["incident"]["urgency"] == "high"


async def test_list_services(fake) -> None:
    res = await fake.call("pagerduty_list_services", {})
    svc = res["services"][0]
    assert svc["name"] == "acme-notes-be"


async def test_list_oncalls(fake) -> None:
    res = await fake.call("pagerduty_list_oncalls", {})
    assert res["oncalls"][0]["user"] == "On-call SRE"


async def test_get_incident_log_entries(fake) -> None:
    res = await fake.call(
        "pagerduty_get_incident_log_entries", {"incident_id": "PT4KHLK"}
    )
    assert res["count"] == 1
    assert res["log_entries"][0]["type"] == "resolve_log_entry"


def test_get_tool_lookup_and_unknown() -> None:
    assert get_tool("pagerduty_get_incident").name == "pagerduty_get_incident"
    with pytest.raises(KeyError):
        get_tool("pagerduty_nope")
