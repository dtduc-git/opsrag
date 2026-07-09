"""Integration test: the MongoDB Atlas billing MCP tools against the fake backend.

Exercises every tool through build_fake() with no network and no Atlas
credentials, asserting the declared tool set and — critically — that Atlas'
cents amounts are converted to dollars everywhere. Follows the Datadog/GCP
reference (FR-012).
"""
from __future__ import annotations

import pytest

from opsrag.mcp.billing_mongodb_atlas import build_fake, get_tool

_EXPECTED_TOOLS = {
    "billing_atlas_list_invoices",
    "billing_atlas_get_invoice",
    "billing_atlas_pending_invoice",
    "billing_atlas_cost_per_project",
}


@pytest.fixture(autouse=True)
def _atlas_env(monkeypatch):
    # Handlers build the URL path from OPSRAG_ATLAS_ORG_ID (required). The fake
    # backend needs no real creds, but the org id must resolve. Set creds too so
    # the request seam's bad_config guard is never the thing under test here.
    monkeypatch.setenv("OPSRAG_ATLAS_ORG_ID", "org-test-123")
    monkeypatch.setenv("OPSRAG_ATLAS_CLIENT_ID", "fake-client-id")
    monkeypatch.setenv("OPSRAG_ATLAS_CLIENT_SECRET", "fake-client-secret")


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()  # restore the real module-level _get / _request / token fetch


def test_fake_exposes_expected_tool_set(fake) -> None:
    assert set(fake.tool_names()) == _EXPECTED_TOOLS


@pytest.mark.asyncio
async def test_list_invoices(fake) -> None:
    res = await fake.call("billing_atlas_list_invoices", {"limit": 10})
    assert res["orgId"] == "org-test-123"
    assert res["count"] == 1
    inv = res["invoices"][0]
    assert inv["id"] == "inv1"
    assert inv["statusName"] == "PENDING"
    # 166000 cents → $1660.00 (cents → dollars conversion).
    assert inv["amountBilledDollars"] == 1660.0


@pytest.mark.asyncio
async def test_get_invoice(fake) -> None:
    res = await fake.call("billing_atlas_get_invoice", {"invoice_id": "inv1"})
    assert res["id"] == "inv1"
    assert res["amountBilledDollars"] == 1660.0
    assert res["lineItemCount"] == 2
    li = res["lineItems"][0]
    assert li["sku"] == "ATLAS_AWS_INSTANCE_M30"
    assert li["groupName"] == "prod"
    assert li["clusterName"] == "prod-cluster-0"
    # 120000 cents → $1200.00.
    assert li["totalPriceDollars"] == 1200.0
    assert li["quantity"] == 720
    assert li["unit"] == "hours"


@pytest.mark.asyncio
async def test_get_invoice_requires_id(fake) -> None:
    from opsrag.mcp.billing_mongodb_atlas import AtlasBillingMCPError

    with pytest.raises(AtlasBillingMCPError) as exc:
        await fake.call("billing_atlas_get_invoice", {})
    assert exc.value.reason == "bad_args"


@pytest.mark.asyncio
async def test_pending_invoice(fake) -> None:
    res = await fake.call("billing_atlas_pending_invoice", {})
    assert res["statusName"] == "PENDING"
    # 166000 cents → $1660.00 running cost this period.
    assert res["amountBilledDollars"] == 1660.0
    assert res["lineItemCount"] == 2
    # Both line-item prices are surfaced in dollars.
    assert {li["totalPriceDollars"] for li in res["lineItems"]} == {1200.0, 460.0}


@pytest.mark.asyncio
async def test_cost_per_project(fake) -> None:
    res = await fake.call("billing_atlas_cost_per_project", {})
    assert res["invoiceId"] == "inv1"
    assert res["count"] == 2
    # Grouped by groupName, summed in dollars, sorted desc by cost.
    by = {r["groupName"]: r["cost_dollars"] for r in res["by_project"]}
    assert by == {"prod": 1200.0, "staging": 460.0}
    assert res["by_project"][0]["groupName"] == "prod"  # 1200 > 460


@pytest.mark.asyncio
async def test_cost_per_project_with_explicit_invoice(fake) -> None:
    res = await fake.call("billing_atlas_cost_per_project", {"invoice_id": "inv1"})
    assert res["invoiceId"] == "inv1"
    assert res["by_project"][0]["cost_dollars"] == 1200.0


@pytest.mark.asyncio
async def test_handler_direct_invocation_pattern() -> None:
    fake = build_fake()
    try:
        tool = get_tool("billing_atlas_pending_invoice")
        res = await tool.handler(fake.client, {})
        assert res["amountBilledDollars"] == 1660.0
    finally:
        fake.teardown()


@pytest.mark.asyncio
async def test_unknown_tool_raises(fake) -> None:
    with pytest.raises(KeyError):
        await fake.call("billing_atlas_does_not_exist", {})
