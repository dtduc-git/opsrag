"""Integration test (T085): the Slack MCP tools against the fake backend.

Exercises the Slack tools through build_fake() with no network and no
Slack token, asserting shape-faithful responses and that the fake's tool
set matches the registry's declared set (FR-012). Follows the GitLab
reference test (tests/integration/test_gitlab_mcp.py).
"""
from __future__ import annotations

import pytest

from opsrag.mcp.registry import REGISTRY
from opsrag.mcp.slack import build_fake


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()


def test_fake_exposes_registry_tool_set(fake) -> None:
    # The fake's tools must match exactly what the registry declares.
    assert set(fake.tool_names()) == set(REGISTRY["slack"].tool_names)


@pytest.mark.asyncio
async def test_list_channels(fake) -> None:
    result = await fake.call("slack_list_channels", {})
    assert isinstance(result, dict)
    assert result["count"] >= 1
    names = {c["name"] for c in result["channels"]}
    assert "sre-alerts" in names
    # Member channels sort first.
    assert result["channels"][0]["is_member"] is True


@pytest.mark.asyncio
async def test_list_channels_filter(fake) -> None:
    result = await fake.call("slack_list_channels", {"name_substring": "sre"})
    assert result["count"] == 1
    assert result["channels"][0]["name"] == "sre-alerts"


@pytest.mark.asyncio
async def test_get_message_by_url(fake) -> None:
    url = "https://example.slack.com/archives/C0000000001/p1700000000000000"
    result = await fake.call("slack_get_message_by_url", {"url": url})
    msg = result["message"]
    assert msg["channel"] == "C0000000001"
    assert msg["ts"] == "1700000000.000000"
    # Raw <@U...> mention rewritten to a display name (canned resolver).
    assert "Canned User" in msg["text"]
    assert msg["permalink_original"] == url


@pytest.mark.asyncio
async def test_get_thread_by_url(fake) -> None:
    url = "https://example.slack.com/archives/C0000000001/p1700000000000100"
    result = await fake.call("slack_get_thread_by_url", {"url": url})
    assert result["channel"] == "C0000000001"
    assert result["root"]["text"].startswith("Root message")
    assert result["reply_count"] == 1
    assert result["replies"][0]["text"].startswith("Reply:")


@pytest.mark.asyncio
async def test_unknown_tool_raises(fake) -> None:
    with pytest.raises(KeyError):
        await fake.call("slack_does_not_exist", {})
