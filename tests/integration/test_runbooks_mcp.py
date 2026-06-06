"""Integration test (T084): the runbooks MCP tools against the fake backend.

Exercises the runbook tools through build_fake() with no filesystem or
network access, asserting shape-faithful responses and the registry's
declared tool set. Follows the GitLab reference test (FR-012).
"""
from __future__ import annotations

import pytest

from opsrag.mcp.registry import REGISTRY
from opsrag.mcp.runbooks import build_fake


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()


def test_fake_exposes_registry_tool_set(fake) -> None:
    # The fake's tools must match exactly what the registry declares.
    assert set(fake.tool_names()) == set(REGISTRY["runbooks"].tool_names)


@pytest.mark.asyncio
async def test_runbook_list_returns_catalog(fake) -> None:
    result = await fake.call("runbook_list", {})
    assert result["count"] >= 1
    assert isinstance(result["runbooks"], list) and result["runbooks"]
    first = result["runbooks"][0]
    assert {"name", "title", "when_to_use", "source"} <= set(first)


@pytest.mark.asyncio
async def test_runbook_list_topic_filter(fake) -> None:
    result = await fake.call("runbook_list", {"topic": "disk pressure"})
    assert result["count"] >= 1
    names = [rb["name"] for rb in result["runbooks"]]
    assert "runbook-disk-pressure" in names


@pytest.mark.asyncio
async def test_runbook_load_returns_content(fake) -> None:
    listed = await fake.call("runbook_list", {})
    name = listed["runbooks"][0]["name"]
    result = await fake.call("runbook_load", {"name": name})
    assert result["name"] == name
    assert result["markdown"]
    assert "# " in result["markdown"]


@pytest.mark.asyncio
async def test_runbook_load_unknown_name(fake) -> None:
    result = await fake.call("runbook_load", {"name": "does-not-exist"})
    assert result["error"] == "runbook not found"
    assert "available_names_first_5" in result
