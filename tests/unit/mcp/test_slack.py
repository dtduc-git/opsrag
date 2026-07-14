"""Unit tests for the Slack MCP tools.

Happy-path tools run through build_fake() with NO network / token. The
`slack_list_channels` pagination is bounded (time budget + page cap) so a
large, rate-limited workspace returns a PARTIAL list instead of hanging past
the MCP client timeout -- those bounds are exercised with a tiny stub client
that drives `_list_channels_cached` directly. asyncio_mode = "auto".
"""
from __future__ import annotations

import asyncio
import time

import pytest

import opsrag.mcp.slack as slack
from opsrag.mcp.registry import REGISTRY
from opsrag.mcp.slack import SLACK_TOOLS, SlackMCPError, build_fake, get_tool


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()


@pytest.fixture(autouse=True)
def _clear_channels_cache():
    """Isolate the module-level channel caches for direct-call tests."""
    slack._channels_cache = {"ts": 0.0, "channels": []}
    slack._channel_name_cache = {}
    yield
    slack._channels_cache = {"ts": 0.0, "channels": []}
    slack._channel_name_cache = {}


class _StubClient:
    """Minimal SlackClient surface for the bounded-pagination tests. `_get` is
    method-aware (serves conversations.list AND conversations.history pages).
    `infinite=True` always hands back a cursor; `sleep` simulates a Retry-After
    stall; `raise_runtime`+`error` simulate an ok=false app error."""

    def __init__(self, *, infinite: bool = False, per_page: int = 1,
                 sleep: float = 0.0, raise_runtime: bool = False,
                 error: str = "invalid_auth"):
        self.calls = 0
        self._infinite = infinite
        self._per_page = per_page
        self._sleep = sleep
        self._raise = raise_runtime
        self._error = error

    async def _get(self, method: str, params: dict | None = None) -> dict:
        self.calls += 1
        if self._raise:
            raise RuntimeError(f"slack {method} failed: {self._error}")
        if self._sleep:
            await asyncio.sleep(self._sleep)
        cursor = "more" if self._infinite else ""
        if method == "conversations.history":
            msgs = [
                {"ts": f"17000{self.calls:05d}.0", "user": "U1",
                 "text": "noise line", "thread_ts": None, "reply_count": 0}
            ]
            return {"messages": msgs, "response_metadata": {"next_cursor": cursor}}
        chans = [
            {
                "id": f"C{self.calls:03d}{i}",
                "name": f"chan-{self.calls}-{i}",
                "is_member": False,
                "is_private": False,
                "is_archived": False,
                "num_members": 1,
                "topic": {"value": ""},
                "purpose": {"value": ""},
            }
            for i in range(self._per_page)
        ]
        return {"channels": chans, "response_metadata": {"next_cursor": cursor}}

    async def get_channel(self, channel_id: str):
        class _C:
            name = "stub-chan"
        return _C()

    async def resolve_user(self, user_id: str) -> str:
        return "Stub User"

    async def close(self) -> None:
        return None


# --- tool surface ---------------------------------------------------

def test_tool_names_match_registry():
    # The fake exposes exactly the registry-declared tools (omits
    # slack_search_messages, which needs a user token + live search).
    assert set(build_fake().tool_names()) == set(REGISTRY["slack"].tool_names)
    assert {t.name for t in SLACK_TOOLS} >= set(REGISTRY["slack"].tool_names)
    assert get_tool("slack_list_channels").name == "slack_list_channels"


# --- happy path (offline fake) --------------------------------------

async def test_list_channels_complete(fake):
    res = await fake.call("slack_list_channels", {})
    assert res["complete"] is True
    assert "warning" not in res
    assert res["count"] == 2 and res["total"] == 2
    # member-first ordering: sre-alerts (is_member) before general.
    assert res["channels"][0]["name"] == "sre-alerts"


async def test_list_channels_name_substring(fake):
    res = await fake.call("slack_list_channels", {"name_substring": "sre"})
    assert res["complete"] is True
    assert res["count"] == 1
    assert res["channels"][0]["name"] == "sre-alerts"


async def test_get_message_by_url_still_works(fake):
    # Guard: the list-channels change didn't disturb the URL tools.
    res = await fake.call(
        "slack_get_message_by_url",
        {"url": "https://x.slack.com/archives/C0000000001/p1700000000000000"},
    )
    assert res["message"]  # canned message payload came back


# --- bounded pagination (the fix) -----------------------------------

async def test_partial_on_page_cap(monkeypatch):
    monkeypatch.setattr(slack, "_LIST_CHANNELS_MAX_PAGES", 3)
    monkeypatch.setattr(slack, "_LIST_CHANNELS_BUDGET_S", 60.0)  # cap, not budget, must bind
    stub = _StubClient(infinite=True, per_page=2)
    channels, complete = await slack._list_channels_cached(stub)
    assert complete is False
    assert stub.calls == 3              # stopped exactly at the page cap
    assert len(channels) == 6           # 3 pages x 2 channels
    assert slack._channels_cache["channels"] == []  # partial NOT cached


async def test_partial_on_time_budget(monkeypatch):
    monkeypatch.setattr(slack, "_LIST_CHANNELS_BUDGET_S", 0.1)
    monkeypatch.setattr(slack, "_LIST_CHANNELS_MAX_PAGES", 999)  # budget, not cap, must bind
    stub = _StubClient(infinite=True, sleep=0.5)  # each page stalls > budget
    started = time.perf_counter()
    channels, complete = await slack._list_channels_cached(stub)
    elapsed = time.perf_counter() - started
    assert complete is False
    assert elapsed < 0.4                # wait_for cancelled the 0.5s stall
    assert slack._channels_cache["channels"] == []


async def test_app_level_error_surfaces(monkeypatch):
    monkeypatch.setattr(slack, "_LIST_CHANNELS_MAX_PAGES", 5)
    stub = _StubClient(raise_runtime=True)
    with pytest.raises(SlackMCPError):
        await slack._list_channels_cached(stub)


async def test_complete_walk_is_cached(monkeypatch):
    stub = _StubClient(infinite=False, per_page=2)  # single page -> complete
    channels, complete = await slack._list_channels_cached(stub)
    assert complete is True
    assert stub.calls == 1
    assert slack._channels_cache["channels"] == channels  # cached
    # Second call within TTL is served from cache -- client not hit again.
    channels2, complete2 = await slack._list_channels_cached(stub)
    assert complete2 is True and stub.calls == 1
    assert channels2 == channels


async def test_handler_surfaces_partial_warning(monkeypatch):
    monkeypatch.setattr(slack, "_LIST_CHANNELS_MAX_PAGES", 2)
    monkeypatch.setattr(slack, "_resolve_bot_token", lambda: "xoxb-stub")
    monkeypatch.setattr(slack, "SlackClient", lambda **_kw: _StubClient(infinite=True, per_page=1))
    res = await slack._h_list_channels(None, {})
    assert res["complete"] is False
    assert "warning" in res and "PARTIAL" in res["warning"]


# --- channel history (bot-token conversations.history scan) ---------

async def test_channel_history_returns_all(fake):
    res = await fake.call("slack_channel_history", {"channel": "C0000000001"})
    assert res["window_complete"] is True
    assert "warning" not in res
    assert res["count"] == 3  # deploy + sms-hang + bot-alert
    assert res["channel"] == "C0000000001"


async def test_channel_history_query_all_words(fake):
    # "sms hang" -> matches "sms service is hanging" (contains 'sms' AND 'hang').
    res = await fake.call("slack_channel_history", {"channel": "C0000000001", "query": "sms hang"})
    assert res["count"] == 1
    assert "hang" in res["messages"][0]["text"].lower()


async def test_channel_history_query_hits_bot_attachment(fake):
    # "sms" alone -> the sms-hang message AND the flattened bot alert (sms-celery...).
    res = await fake.call("slack_channel_history", {"channel": "C0000000001", "query": "sms"})
    assert res["count"] == 2


async def test_channel_history_resolves_channel_name(fake):
    res = await fake.call("slack_channel_history", {"channel": "#sre-alerts"})
    assert res["channel"] == "C0000000001"
    assert res["count"] == 3


async def test_channel_history_name_not_found(fake):
    with pytest.raises(SlackMCPError):
        await fake.call("slack_channel_history", {"channel": "no-such-channel"})


async def test_channel_history_requires_channel(fake):
    with pytest.raises(SlackMCPError):
        await fake.call("slack_channel_history", {})


async def test_channel_history_not_in_channel(monkeypatch):
    monkeypatch.setattr(slack, "_resolve_bot_token", lambda: "xoxb-stub")
    stub = _StubClient(raise_runtime=True, error="not_in_channel")
    monkeypatch.setattr(slack, "SlackClient", lambda **_kw: stub)
    with pytest.raises(SlackMCPError):
        await slack._h_channel_history(None, {"channel": "C0EXAMPLE03"})


async def test_channel_history_partial_on_budget(monkeypatch):
    monkeypatch.setattr(slack, "_HISTORY_BUDGET_S", 0.1)
    monkeypatch.setattr(slack, "_HISTORY_MAX_PAGES", 999)
    monkeypatch.setattr(slack, "_resolve_bot_token", lambda: "xoxb-stub")
    stub = _StubClient(infinite=True, sleep=0.5)  # each page stalls > budget
    monkeypatch.setattr(slack, "SlackClient", lambda **_kw: stub)
    started = time.perf_counter()
    res = await slack._h_channel_history(None, {"channel": "C0EXAMPLE03", "since": "30d"})
    elapsed = time.perf_counter() - started
    assert res["window_complete"] is False
    assert "warning" in res
    assert elapsed < 0.4


async def test_channel_history_limit_reached(monkeypatch):
    monkeypatch.setattr(slack, "_resolve_bot_token", lambda: "xoxb-stub")
    stub = _StubClient(infinite=True)  # endless matchable "noise line" messages
    monkeypatch.setattr(slack, "SlackClient", lambda **_kw: stub)
    res = await slack._h_channel_history(
        None, {"channel": "C0EXAMPLE03", "query": "noise", "limit": 2}
    )
    assert res["limit_reached"] is True
    assert res["count"] == 2
    assert "warning" in res
