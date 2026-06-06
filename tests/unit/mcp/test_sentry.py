"""Unit tests for the Sentry MCP tools against the offline fake backend.

Exercises EVERY tool through build_fake() with no network and no Sentry
credentials, asserting shape-faithful parsed responses. Follows the
Datadog/GitLab reference pattern (FR-012).
"""
from __future__ import annotations

import pytest

from opsrag.mcp.sentry import SENTRY_TOOLS, build_fake, get_tool

_EXPECTED_TOOLS = {
    "sentry_list_projects",
    "sentry_search_issues",
    "sentry_get_issue",
    "sentry_get_latest_event",
    "sentry_search_events",
    "sentry_get_event",
    "sentry_get_trace",
    "sentry_list_releases",
}


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()  # restore the real module-level _get / _config


def test_tool_set_is_exactly_the_contract() -> None:
    assert {t.name for t in SENTRY_TOOLS} == _EXPECTED_TOOLS
    assert len(SENTRY_TOOLS) == len(_EXPECTED_TOOLS)


def test_fake_exposes_full_tool_set(fake) -> None:
    assert set(fake.tool_names()) == _EXPECTED_TOOLS


async def test_list_projects(fake) -> None:
    res = await fake.call("sentry_list_projects", {"org": "acme"})
    assert res["org"] == "acme"
    assert res["count"] == 1
    p = res["projects"][0]
    assert p["slug"] == "acme-notes-be"
    assert p["platform"] == "python"
    assert p["team"] == "platform"


async def test_search_issues_default_query(fake) -> None:
    res = await fake.call("sentry_search_issues", {"org": "acme"})
    assert res["query"] == "is:unresolved"
    assert res["count"] == 1
    issue = res["issues"][0]
    assert issue["id"] == "1001"
    assert issue["title"] == "RuntimeError: boom"
    assert issue["culprit"] == "app.views in get_notes"
    assert issue["level"] == "error"
    assert issue["count"] == "57"
    assert issue["project"] == "acme-notes-be"


async def test_get_issue(fake) -> None:
    res = await fake.call("sentry_get_issue", {"org": "acme", "issue_id": "1001"})
    assert res["id"] == "1001"
    assert res["short_id"] == "ACME-1"
    assert res["type"] == "RuntimeError"
    assert res["value"] == "boom"
    assert res["status"] == "unresolved"


async def test_get_latest_event_has_stacktrace(fake) -> None:
    res = await fake.call("sentry_get_latest_event", {"org": "acme", "issue_id": "1001"})
    assert res["event_id"] == "ev-deadbeef"
    assert res["culprit"] == "app.views in get_notes"
    assert res["level"] == "error"
    frames = res["stacktrace"]
    assert len(frames) == 1
    assert frames[0]["function"] == "get_notes"
    assert frames[0]["filename"] == "app/views.py"
    assert frames[0]["lineno"] == 42
    assert frames[0]["in_app"] is True
    # tags trimmed to {key,value}
    assert {"key": "level", "value": "error"} in res["tags"]


async def test_search_events(fake) -> None:
    res = await fake.call(
        "sentry_search_events",
        {"org": "acme", "query": "level:error", "field": ["title", "project"]},
    )
    assert res["query"] == "level:error"
    assert res["fields"] == ["title", "project"]
    assert res["count"] == 1
    row = res["events"][0]
    assert row["title"] == "RuntimeError: boom"
    assert row["project"] == "acme-notes-be"


async def test_get_event(fake) -> None:
    res = await fake.call(
        "sentry_get_event",
        {"org": "acme", "project": "acme-notes-be", "event_id": "ev-deadbeef"},
    )
    assert res["event_id"] == "ev-deadbeef"
    assert res["issue_id"] == "1001"
    assert res["project"] == "acme-notes-be"
    assert res["stacktrace"][0]["function"] == "get_notes"


async def test_get_trace(fake) -> None:
    res = await fake.call("sentry_get_trace", {"org": "acme", "trace_id": "trace-xyz"})
    assert res["trace_id"] == "trace-xyz"
    assert res["span_count"] == 1
    assert res["services_seen"] == ["acme-notes-be"]
    assert res["errors"] == 1
    span = res["spans"][0]
    assert span["transaction"] == "GET /notes"
    assert span["op"] == "http.server"
    assert span["project"] == "acme-notes-be"


async def test_list_releases(fake) -> None:
    res = await fake.call("sentry_list_releases", {"org": "acme"})
    assert res["count"] == 1
    rel = res["releases"][0]
    assert rel["version"] == "acme-notes-be@1.2.3"
    assert rel["short_version"] == "1.2.3"
    assert rel["new_groups"] == 3
    assert rel["projects"] == ["acme-notes-be"]


async def test_default_org_from_config(fake) -> None:
    # No org arg -- the fake _config() supplies default org 'acme'.
    res = await fake.call("sentry_list_projects", {})
    assert res["org"] == "acme"


def test_get_tool_lookup_and_unknown() -> None:
    assert get_tool("sentry_get_issue").name == "sentry_get_issue"
    with pytest.raises(KeyError):
        get_tool("sentry_nope")
