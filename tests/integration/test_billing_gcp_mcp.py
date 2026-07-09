"""Offline tests for the billing_gcp MCP connector (FR-012 fake backend).

`build_fake()` swaps the single BigQuery seam (`_run_query`) for a canned
dispatcher, so the whole tool surface runs with no google-cloud-bigquery, no
GCP creds, and no network.
"""
from __future__ import annotations

import pytest

from opsrag.mcp import billing_gcp as bg
from opsrag.mcp.registry import REGISTRY

_EXPECTED = {
    "billing_gcp_cost_anomalies",
    "billing_gcp_cost_by_label",
    "billing_gcp_cost_by_project",
    "billing_gcp_cost_by_service",
    "billing_gcp_cost_summary",
    "billing_gcp_cost_trend",
}


def test_fake_exposes_registry_tool_set():
    fake = bg.build_fake()
    try:
        assert set(fake.tool_names()) == _EXPECTED
        assert set(REGISTRY["billing_gcp"].tool_names) == _EXPECTED
    finally:
        fake.close()


def test_registry_category_and_restricted_default():
    assert REGISTRY["billing_gcp"].category == "Billing"
    from opsrag.config_mcp import BillingGcpMCPConfig
    assert BillingGcpMCPConfig().restricted is True  # ships restricted


# --- pure helpers ---------------------------------------------------------
def test_projection_and_trend():
    # 10 USD over 5 days of a 30-day month -> 60 USD projected.
    assert bg.project_month(10.0, day_of_month=5, days_in_month=30) == 60.0
    assert bg.trend_pct(projected=110.0, prev=100.0) == 10.0
    assert bg.trend_pct(projected=1.0, prev=0.0) == 0.0


def test_month_validation():
    assert bg._valid_month("202607") == "202607"
    assert bg._valid_month("bad") == bg._valid_month(None)  # falls back to current
    assert bg._prev_month("202601") == "202512"


def test_table_requires_env(monkeypatch):
    monkeypatch.delenv("OPSRAG_GCP_BILLING_TABLE", raising=False)
    with pytest.raises(bg.BillingGcpMCPError) as ei:
        bg._table()
    assert ei.value.reason == "bad_config"


# --- per-tool fake calls --------------------------------------------------
@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("OPSRAG_GCP_BILLING_TABLE", "proj.ds.gcp_billing_export_v1_*")
    f = bg.build_fake()
    yield f
    f.close()


@pytest.mark.asyncio
async def test_cost_summary(fake):
    r = await fake.call("billing_gcp_cost_summary", {})
    assert r["mtd_usd"] == 15368.42
    assert r["yesterday_usd"] == 2248.11
    assert "projected_month_usd" in r and r["currency"] == "USD"


@pytest.mark.asyncio
async def test_cost_by_service(fake):
    r = await fake.call("billing_gcp_cost_by_service", {"limit": 3})
    names = [s["service"] for s in r["by_service"]]
    assert "Vertex AI" in names and "Compute Engine" in names


@pytest.mark.asyncio
async def test_cost_by_project(fake):
    r = await fake.call("billing_gcp_cost_by_project", {})
    assert any(p["project_id"] == "example-prod" for p in r["by_project"])


@pytest.mark.asyncio
async def test_cost_by_label_requires_key(fake):
    with pytest.raises(bg.BillingGcpMCPError):
        await fake.call("billing_gcp_cost_by_label", {})
    r = await fake.call("billing_gcp_cost_by_label", {"label_key": "team"})
    assert r["label_key"] == "team" and r["by_label"]


@pytest.mark.asyncio
async def test_cost_trend(fake):
    r = await fake.call("billing_gcp_cost_trend", {"days": 7})
    assert r["days"] == 7 and len(r["daily"]) == 2


@pytest.mark.asyncio
async def test_cost_anomalies(fake):
    r = await fake.call("billing_gcp_cost_anomalies", {})
    assert r["anomalies"][0]["scope"] == "Cloud SQL"
    assert r["anomalies"][0]["pct_change"] > 0
