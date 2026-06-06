"""Unit test: the Splunk read-only MCP tools against the fake backend.

Exercises every tool through build_fake() with no network and no Splunk
credentials, asserting shape-faithful parsed responses and the read-only
SPL guardrail. Follows the Datadog/GitLab reference (FR-012).
"""
from __future__ import annotations

import pytest

from opsrag.mcp.splunk import SPLUNK_TOOLS, SplunkMCPError, build_fake, get_tool


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()  # restore the real module-level _get / _post


def test_tool_set_is_exactly_the_six_declared() -> None:
    assert set(fake_names()) == {
        "splunk_run_search",
        "splunk_export_search",
        "splunk_list_saved_searches",
        "splunk_run_saved_search",
        "splunk_list_indexes",
        "splunk_fired_alerts",
    }


def fake_names() -> list[str]:
    return [t.name for t in SPLUNK_TOOLS]


@pytest.mark.asyncio
async def test_run_search(fake) -> None:
    result = await fake.call(
        "splunk_run_search",
        {"search": "index=web status=500", "count": 10},
    )
    # `search ` prepended automatically.
    assert result["search"] == "search index=web status=500"
    assert result["count"] == 2
    assert result["results"][0]["status"] == "500"
    assert result["results"][0]["host"] == "web-1"


@pytest.mark.asyncio
async def test_run_search_leaves_leading_search_intact(fake) -> None:
    result = await fake.call("splunk_run_search", {"search": "search index=web"})
    assert result["search"] == "search index=web"


@pytest.mark.asyncio
async def test_run_search_rejects_mutating_spl(fake) -> None:
    with pytest.raises(SplunkMCPError):
        await fake.call(
            "splunk_run_search",
            {"search": "index=web | delete"},
        )
    with pytest.raises(SplunkMCPError):
        await fake.call(
            "splunk_run_search",
            {"search": "index=web | outputlookup evil.csv"},
        )


@pytest.mark.asyncio
async def test_export_search(fake) -> None:
    result = await fake.call(
        "splunk_export_search",
        {"search": "index=web status>=500"},
    )
    assert result["search"] == "search index=web status>=500"
    assert result["count"] == 1
    assert result["results"][0]["status"] == "503"


@pytest.mark.asyncio
async def test_export_search_rejects_mutating_spl(fake) -> None:
    with pytest.raises(SplunkMCPError):
        await fake.call("splunk_export_search", {"search": "index=x | collect index=y"})


@pytest.mark.asyncio
async def test_list_saved_searches(fake) -> None:
    result = await fake.call("splunk_list_saved_searches", {})
    assert result["count"] == 1
    ss = result["saved_searches"][0]
    assert ss["name"] == "High 5xx rate"
    assert ss["is_scheduled"] is True
    assert ss["owner"] == "sre"
    assert "stats count" in ss["search"]


@pytest.mark.asyncio
async def test_run_saved_search(fake) -> None:
    result = await fake.call(
        "splunk_run_saved_search",
        {"name": "High 5xx rate"},
    )
    assert result["name"] == "High 5xx rate"
    assert result["sid"] == "scheduler__sre__search__dispatched1"
    assert result["count"] == 1
    assert result["results"][0]["status"] == "500"


@pytest.mark.asyncio
async def test_run_saved_search_requires_name(fake) -> None:
    with pytest.raises(SplunkMCPError):
        await fake.call("splunk_run_saved_search", {})


@pytest.mark.asyncio
async def test_list_indexes(fake) -> None:
    result = await fake.call("splunk_list_indexes", {})
    assert result["count"] == 1
    idx = result["indexes"][0]
    assert idx["name"] == "main"
    assert idx["total_event_count"] == 123456
    assert idx["current_db_size_mb"] == 2048


@pytest.mark.asyncio
async def test_fired_alerts(fake) -> None:
    result = await fake.call("splunk_fired_alerts", {})
    assert result["count"] == 1
    a = result["fired_alerts"][0]
    assert a["savedsearch_name"] == "High 5xx rate"
    assert a["severity"] == "5"
    assert a["triggered_alerts"] == 3


@pytest.mark.asyncio
async def test_unknown_tool_raises(fake) -> None:
    with pytest.raises(KeyError):
        await fake.call("splunk_does_not_exist", {})


def test_get_tool_lookup() -> None:
    assert get_tool("splunk_run_search").name == "splunk_run_search"
    with pytest.raises(KeyError):
        get_tool("nope")
