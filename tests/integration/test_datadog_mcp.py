"""Integration test (T077): the Datadog MCP tools against the fake backend.

Exercises representative tools through build_fake() with no network and no
DD credentials, asserting shape-faithful responses and the registry's
declared tool set. Follows the GitLab reference (FR-012).
"""
from __future__ import annotations

import pytest

from opsrag.mcp.datadog import build_fake
from opsrag.mcp.registry import REGISTRY


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()  # restore the real module-level _get / _post


def test_fake_exposes_registry_tool_set(fake) -> None:
    # The fake's tools must match exactly what the registry declares.
    assert set(fake.tool_names()) == set(REGISTRY["datadog"].tool_names)


@pytest.mark.asyncio
async def test_parse_trace_url(fake) -> None:
    # Pure string parsing -- no network, no _get / _post.
    result = await fake.call(
        "datadog_parse_trace_url",
        {
            "url": (
                "https://app.datadoghq.com/apm/trace/abc123"
                "?spanID=42&timeHint=1716000000000&env=prod&service=acme-notes-be"
            )
        },
    )
    assert result["trace_id"] == "abc123"
    assert result["span_id"] == "42"
    assert result["epoch_ms"] == 1716000000000
    assert result["site"] == "datadoghq.com"
    assert result["env_hint"] == "prod"
    assert result["service_hint"] == "acme-notes-be"


@pytest.mark.asyncio
async def test_list_monitors(fake) -> None:
    result = await fake.call("datadog_list_monitors", {})
    assert result["count"] == 2
    assert result["alerting"] == 1
    # Alerting monitors sort first.
    assert result["monitors"][0]["state"] == "Alert"
    assert result["monitors"][0]["id"] == 101


@pytest.mark.asyncio
async def test_get_trace(fake) -> None:
    result = await fake.call("datadog_get_trace", {"trace_id": "abc123"})
    assert result["trace_id"] == "abc123"
    assert result["span_count"] == 1
    assert result["services_seen"] == ["acme-notes-be"]
    assert result["errors"] and result["errors"][0]["type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_unknown_tool_raises(fake) -> None:
    with pytest.raises(KeyError):
        await fake.call("datadog_does_not_exist", {})
