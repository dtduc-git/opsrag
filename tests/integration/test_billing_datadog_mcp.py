"""Integration test: the Datadog billing (cost/usage) MCP tools against the
fake backend.

Exercises every tool through build_fake() with no network and no DD
credentials, asserting shape-faithful responses and the module's declared
tool set. Follows the Datadog/AWS/GCP reference (FR-012).
"""
from __future__ import annotations

import pytest

from opsrag.mcp.billing_datadog import BILLING_DATADOG_TOOLS, build_fake, get_tool

_EXPECTED_TOOLS = {
    "billing_datadog_estimated_cost",
    "billing_datadog_historical_cost",
    "billing_datadog_projected_cost",
    "billing_datadog_hourly_usage",
    "billing_datadog_cost_by_tag",
}


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()  # restore the real module-level _get


def test_fake_exposes_expected_tool_set(fake) -> None:
    # The fake's tools must match exactly the 5 declared billing tools.
    assert set(fake.tool_names()) == _EXPECTED_TOOLS
    assert {t.name for t in BILLING_DATADOG_TOOLS} == _EXPECTED_TOOLS


@pytest.mark.asyncio
async def test_estimated_cost(fake) -> None:
    res = await fake.call("billing_datadog_estimated_cost", {})
    org = res["orgs"][0]
    assert org["total_cost_usd"] == 15368.0
    # Charges are sorted by cost desc, so infra_hosts leads apm.
    assert org["charges"][0]["product"] == "infra_hosts"
    assert any(c["product"] == "apm" and c["cost_usd"] == 812.0 for c in org["charges"])
    assert "estimate" in res["note"].lower()


@pytest.mark.asyncio
async def test_historical_cost(fake) -> None:
    res = await fake.call(
        "billing_datadog_historical_cost",
        {"start_month": "2026-06", "end_month": "2026-06"},
    )
    assert res["start_month"] == "2026-06"
    assert res["orgs"][0]["total_cost_usd"] == 14980.0
    assert res["orgs"][0]["charges"][0]["product"] == "logs"


@pytest.mark.asyncio
async def test_projected_cost(fake) -> None:
    res = await fake.call("billing_datadog_projected_cost", {})
    org = res["orgs"][0]
    # Projected uses `projected_total_cost` / `projected_cost` — handler must
    # fall back to those field names.
    assert org["total_cost_usd"] == 31200.0
    assert org["charges"][0]["product"] == "apm"
    assert org["charges"][0]["cost_usd"] == 1700.0


@pytest.mark.asyncio
async def test_hourly_usage(fake) -> None:
    res = await fake.call(
        "billing_datadog_hourly_usage",
        {"product_families": "infra_hosts,apm_hosts"},
    )
    fams = {f["product_family"]: f for f in res["families"]}
    assert fams["infra_hosts"]["points"] == 2
    assert fams["infra_hosts"]["total_usage"] == 238.0  # 120 + 118
    assert fams["apm_hosts"]["total_usage"] == 40.0


@pytest.mark.asyncio
async def test_hourly_usage_clamps_window(fake) -> None:
    # An over-wide window is clamped to <= 7 days; still returns canned data.
    res = await fake.call(
        "billing_datadog_hourly_usage",
        {"start": "2020-01-01T00:00:00Z", "end": "2026-07-08T00:00:00Z"},
    )
    from datetime import datetime

    start = datetime.fromisoformat(res["start"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(res["end"].replace("Z", "+00:00"))
    assert (end - start).days <= 7


@pytest.mark.asyncio
async def test_cost_by_tag(fake) -> None:
    res = await fake.call(
        "billing_datadog_cost_by_tag",
        {"start_month": "2026-07", "tag_keys": "team"},
    )
    assert res["tag_keys"] == "team"
    tags_seen = {r["tags"].get("team") for r in res["rows"]}
    assert tags_seen == {"platform", "data"}
    assert res["rows"][0]["values"]["total_cost"] == 9200.0


@pytest.mark.asyncio
async def test_cost_by_tag_requires_tag_keys(fake) -> None:
    from opsrag.mcp.billing_datadog import DatadogBillingMCPError

    with pytest.raises(DatadogBillingMCPError) as exc:
        await fake.call("billing_datadog_cost_by_tag", {"start_month": "2026-07"})
    assert exc.value.reason == "bad_args"


@pytest.mark.asyncio
async def test_handler_direct_invocation_pattern() -> None:
    # Exercise the get_tool + tool.handler(client, args) path explicitly.
    fake = build_fake()
    try:
        tool = get_tool("billing_datadog_estimated_cost")
        res = await tool.handler(fake.client, {})
        assert res["orgs"][0]["total_cost_usd"] == 15368.0
    finally:
        fake.teardown()


@pytest.mark.asyncio
async def test_unknown_tool_raises(fake) -> None:
    with pytest.raises(KeyError):
        await fake.call("billing_datadog_does_not_exist", {})
