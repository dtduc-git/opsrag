"""Unit tests for the Grafana MCP tools against the offline fake backend.

Every tool is exercised through build_fake() with NO network and NO
GRAFANA_URL / GRAFANA_TOKEN. Asserts the parsed shape (keys/values from the
canned, API-faithful data). asyncio_mode = "auto" so no marker is needed.
"""
from __future__ import annotations

import pytest

from opsrag.mcp.grafana import GRAFANA_TOOLS, build_fake, get_tool


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()  # restore the real module-level _get


def test_tool_names_exact():
    expected = {
        "grafana_search_dashboards",
        "grafana_get_dashboard",
        "grafana_list_datasources",
        "grafana_query_prometheus",
        "grafana_prometheus_label_values",
        "grafana_query_loki",
        "grafana_loki_label_values",
        "grafana_list_alert_rules",
        "grafana_list_contact_points",
    }
    assert {t.name for t in GRAFANA_TOOLS} == expected
    assert set(fake_names()) == expected


def fake_names():
    f = build_fake()
    try:
        return f.tool_names()
    finally:
        f.close()


async def test_search_dashboards(fake):
    res = await fake.call("grafana_search_dashboards", {"query": "api"})
    assert res["count"] == 1
    d = res["dashboards"][0]
    assert d["uid"] == "dash-abc"
    assert d["title"] == "API Overview"
    assert d["folder"] == "Services"
    assert "prod" in d["tags"]


async def test_get_dashboard(fake):
    res = await fake.call("grafana_get_dashboard", {"uid": "dash-abc"})
    assert res["uid"] == "dash-abc"
    assert res["title"] == "API Overview"
    assert res["folder"] == "Services"
    assert res["panel_count"] == 2
    titles = [p["title"] for p in res["panels"]]
    assert titles == ["Request rate", "Error rate"]
    expr = res["panels"][0]["targets"][0]["expr"]
    assert expr == "sum(rate(http_requests_total[5m]))"


async def test_list_datasources(fake):
    res = await fake.call("grafana_list_datasources", {})
    assert res["count"] == 2
    by_uid = {d["uid"]: d for d in res["datasources"]}
    assert by_uid["prom-uid"]["type"] == "prometheus"
    assert by_uid["prom-uid"]["is_default"] is True
    assert by_uid["loki-uid"]["type"] == "loki"


async def test_query_prometheus_instant(fake):
    res = await fake.call(
        "grafana_query_prometheus",
        {"datasource_uid": "prom-uid", "query": "up"},
    )
    assert res["status"] == "success"
    assert res["mode"] == "instant"
    assert res["result_type"] == "vector"
    assert res["series_count"] == 1
    assert res["result"][0]["metric"]["job"] == "api"


async def test_query_prometheus_range(fake):
    res = await fake.call(
        "grafana_query_prometheus",
        {
            "datasource_uid": "prom-uid",
            "query": "rate(x[5m])",
            "start": "1716000000",
            "end": "1716000600",
            "step": "60s",
        },
    )
    assert res["mode"] == "range"
    assert res["result_type"] == "matrix"
    assert res["series_count"] == 1
    assert len(res["result"][0]["values"]) == 2


async def test_prometheus_label_values(fake):
    res = await fake.call(
        "grafana_prometheus_label_values",
        {"datasource_uid": "prom-uid", "label": "job"},
    )
    assert res["status"] == "success"
    assert res["label"] == "job"
    assert res["count"] == 3
    assert "api" in res["values"]


async def test_query_loki(fake):
    res = await fake.call(
        "grafana_query_loki",
        {
            "datasource_uid": "loki-uid",
            "query": '{app="api"} |= "error"',
            "start": "1716000000000000000",
            "end": "1716000600000000000",
            "limit": 10,
        },
    )
    assert res["status"] == "success"
    assert res["result_type"] == "streams"
    assert res["stream_count"] == 1
    s = res["streams"][0]
    assert s["labels"]["app"] == "api"
    assert s["values"][0][1] == "boom: something failed"


async def test_query_loki_requires_time_bounds(fake):
    from opsrag.mcp.grafana import GrafanaMCPError

    with pytest.raises(GrafanaMCPError):
        await fake.call(
            "grafana_query_loki",
            {"datasource_uid": "loki-uid", "query": "{app=\"api\"}", "limit": 5},
        )


async def test_loki_label_values(fake):
    res = await fake.call(
        "grafana_loki_label_values",
        {"datasource_uid": "loki-uid", "label": "app"},
    )
    assert res["status"] == "success"
    assert res["label"] == "app"
    assert res["count"] == 3
    assert "ingress" in res["values"]


async def test_list_alert_rules(fake):
    res = await fake.call("grafana_list_alert_rules", {})
    assert res["status"] == "success"
    assert res["group_count"] == 1
    assert res["firing"] == 1
    assert res["pending"] == 0
    rules = res["groups"][0]["rules"]
    firing = [r for r in rules if r["state"] == "firing"]
    assert firing[0]["name"] == "HighErrorRate"
    assert firing[0]["active_alerts"] == 1


async def test_list_contact_points(fake):
    res = await fake.call("grafana_list_contact_points", {})
    assert res["count"] == 2
    by_name = {c["name"]: c for c in res["contact_points"]}
    assert by_name["sre-slack"]["type"] == "slack"
    assert by_name["oncall-pd"]["type"] == "pagerduty"
    # Secret-bearing settings must NOT be present.
    assert "settings" not in by_name["sre-slack"]


async def test_get_tool_lookup_and_unknown():
    assert get_tool("grafana_get_dashboard").name == "grafana_get_dashboard"
    with pytest.raises(KeyError):
        get_tool("grafana_nope")
