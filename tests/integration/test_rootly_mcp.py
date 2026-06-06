"""Integration test (T083): the Rootly MCP tools against the fake backend.

Exercises representative tools through build_fake() with no network and no
Rootly token, asserting shape-faithful responses and the registry's
declared tool set. Follows the GitLab/Datadog reference (FR-012).
"""
from __future__ import annotations

import pytest

from opsrag.mcp.registry import REGISTRY
from opsrag.mcp.rootly import build_fake


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()  # restore the real module-level _get


def test_fake_exposes_registry_tool_set(fake) -> None:
    # The fake's tools must match exactly what the registry declares.
    assert set(fake.tool_names()) == set(REGISTRY["rootly"].tool_names)


@pytest.mark.asyncio
async def test_list_incidents(fake) -> None:
    result = await fake.call("rootly_list_incidents", {})
    assert result["count"] == 1
    assert result["total"] == 1
    inc = result["incidents"][0]
    assert inc["sequential_id"] == 292
    assert inc["status"] == "resolved"
    assert inc["severity"] == "sev1"
    assert inc["service_ids"] == ["svc-1"]
    assert inc["url"].endswith("/292-database-connection-pool-exhausted")


@pytest.mark.asyncio
async def test_get_incident_by_sequential_id(fake) -> None:
    # Sequential id "292" is resolved to a UUID, then fetched.
    result = await fake.call("rootly_get_incident", {"incident_id": "292"})
    inc = result["incident"]
    assert inc["title"] == "Database connection pool exhausted"
    assert inc["sequential_id"] == 292
    assert inc["duration_in_minutes"] == 12


@pytest.mark.asyncio
async def test_list_services(fake) -> None:
    result = await fake.call("rootly_list_services", {})
    assert result["count"] == 1
    svc = result["services"][0]
    assert svc["name"] == "acme-notes-be"
    assert svc["service_tier"] == "1"


@pytest.mark.asyncio
async def test_unknown_tool_raises(fake) -> None:
    with pytest.raises(KeyError):
        await fake.call("rootly_does_not_exist", {})
