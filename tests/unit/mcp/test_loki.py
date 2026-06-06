"""Unit tests for the Grafana Loki MCP tools against the offline fake backend.

Each tool is exercised via build_fake() with no network and no LOKI_URL,
asserting the parsed shape (keys/values from the canned, API-faithful data).
asyncio_mode = "auto" in pyproject, so no @pytest.mark.asyncio needed.
"""
from __future__ import annotations

import pytest

from opsrag.mcp.loki import LOKI_TOOLS, LokiMCPError, build_fake, get_tool

_EXPECTED_TOOLS = {
    "loki_query_range",
    "loki_query",
    "loki_labels",
    "loki_label_values",
    "loki_series",
}


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()  # restore the real module-level _get


def test_tool_set_matches_exactly(fake) -> None:
    assert set(fake.tool_names()) == _EXPECTED_TOOLS
    assert {t.name for t in LOKI_TOOLS} == _EXPECTED_TOOLS


async def test_query_range(fake) -> None:
    result = await fake.call(
        "loki_query_range", {"query": '{app="acme-notes-be"} |= "error"'}
    )
    assert result["query"] == '{app="acme-notes-be"} |= "error"'
    assert result["direction"] == "backward"
    assert result["limit"] == 100
    # A default time bound is always present.
    assert result["start"] and result["end"]
    assert result["result_type"] == "streams"
    assert result["count"] == 2
    first = result["entries"][0]
    assert first["line"] == "boom: unhandled RuntimeError"
    assert first["labels"]["app"] == "acme-notes-be"
    assert first["ts_ns"] == "1700000000000000000"
    assert first["ts"] is not None  # ns -> ISO conversion


async def test_query_range_requires_query(fake) -> None:
    with pytest.raises(LokiMCPError):
        await fake.call("loki_query_range", {})


async def test_query_range_clamps_limit(fake) -> None:
    result = await fake.call(
        "loki_query_range", {"query": "{app=\"x\"}", "limit": 999999}
    )
    assert result["limit"] == 1000  # capped at _MAX_LIMIT


async def test_query_instant(fake) -> None:
    result = await fake.call("loki_query", {"query": "count_over_time({app=\"x\"}[5m])"})
    assert result["query"] == "count_over_time({app=\"x\"}[5m])"
    assert result["limit"] == 100
    assert result["count"] == 2
    assert result["entries"][0]["labels"]["level"] == "error"


async def test_labels(fake) -> None:
    result = await fake.call("loki_labels", {})
    assert result["count"] == 4
    assert "namespace" in result["labels"]
    assert "app" in result["labels"]


async def test_label_values(fake) -> None:
    result = await fake.call("loki_label_values", {"name": "app"})
    assert result["name"] == "app"
    assert result["count"] == 3
    assert "acme-notes-be" in result["values"]


async def test_label_values_requires_name(fake) -> None:
    with pytest.raises(LokiMCPError):
        await fake.call("loki_label_values", {})


async def test_series(fake) -> None:
    result = await fake.call("loki_series", {"match": '{namespace="prod"}'})
    assert result["count"] == 2
    assert result["start"] and result["end"]
    assert result["series"][0]["app"] == "acme-notes-be"


async def test_series_accepts_list(fake) -> None:
    result = await fake.call(
        "loki_series", {"match": ['{namespace="prod"}', '{app="x"}']}
    )
    assert result["count"] == 2


async def test_series_requires_match(fake) -> None:
    with pytest.raises(LokiMCPError):
        await fake.call("loki_series", {})


async def test_get_tool_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_tool("loki_does_not_exist")


def test_config_missing_url_raises(monkeypatch) -> None:
    # _config raises a clear LokiMCPError when LOKI_URL is unset.
    import opsrag.mcp.loki as _mod

    monkeypatch.delenv("LOKI_URL", raising=False)
    with pytest.raises(LokiMCPError):
        _mod._config()
