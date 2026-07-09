"""Integration test: the Kubecost billing MCP tools against the fake backend.

Exercises every tool through build_fake() with no network, no cluster, and no
Kubecost URL, asserting shape-faithful responses and the declared tool set.
Follows the AWS/GCP reference (FR-012).
"""
from __future__ import annotations

import pytest

from opsrag.mcp.billing_kubecost import build_fake, get_tool

_EXPECTED_TOOLS = {
    "billing_kubecost_allocation",
    "billing_kubecost_allocation_summary",
    "billing_kubecost_assets",
    "billing_kubecost_cloud_cost",
}


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()  # restore the real module-level _get


def test_fake_exposes_expected_tool_set(fake) -> None:
    assert set(fake.tool_names()) == _EXPECTED_TOOLS


@pytest.mark.asyncio
async def test_allocation(fake) -> None:
    res = await fake.call("billing_kubecost_allocation", {"window": "7d", "aggregate": "namespace"})
    assert res["window"] == "7d"
    assert res["aggregate"] == "namespace"
    # __idle__ is the biggest, so it sorts first; total sums all rows.
    assert res["allocations"][0]["name"] == "__idle__"
    assert res["count"] == 3
    top = res["allocations"][0]
    assert {"cpuCost", "ramCost", "pvCost", "totalCost", "efficiency"} <= set(top)
    assert res["total_cost"] == pytest.approx(185.5 + 142.0 + 450.0)


@pytest.mark.asyncio
async def test_allocation_summary(fake) -> None:
    res = await fake.call("billing_kubecost_allocation_summary", {"aggregate": "namespace"})
    assert res["count"] == 2
    names = {r["name"] for r in res["summary"]}
    assert names == {"opsrag", "kube-system"}
    # sorted by totalCost desc: kube-system (185.5) before opsrag (142.0)
    assert res["summary"][0]["name"] == "kube-system"
    assert res["summary"][0]["efficiency"] == pytest.approx(0.42)


@pytest.mark.asyncio
async def test_assets(fake) -> None:
    res = await fake.call("billing_kubecost_assets", {"aggregate": "type"})
    assert res["count"] == 3
    assert res["assets"][0]["name"] == "gke-prod-pool-1"
    assert res["assets"][0]["type"] == "Node"
    assert res["total_cost"] == pytest.approx(640.0 + 44.0 + 21.5)


@pytest.mark.asyncio
async def test_cloud_cost(fake) -> None:
    res = await fake.call("billing_kubecost_cloud_cost", {"aggregate": "provider"})
    assert res["count"] == 2
    names = {r["name"] for r in res["cloud_costs"]}
    assert names == {"Compute Engine", "Cloud SQL"}
    assert res["cloud_costs"][0]["name"] == "Compute Engine"  # 6275 > 6149
    assert res["total_cost"] == pytest.approx(6275.0 + 6149.0)


@pytest.mark.asyncio
async def test_cloud_cost_degrades_gracefully_on_404() -> None:
    # An unknown /model path falls through the fake to a 404; the handler must
    # return an empty result with a note rather than raising.
    import opsrag.mcp.billing_kubecost as mod

    fake = build_fake()
    try:
        async def _404_get(path, params=None, *, tool="billing_kubecost"):
            return 404, {}
        mod._get = _404_get
        res = await mod._h_cloud_cost(None, {"window": "month"})
        assert res["count"] == 0
        assert res["cloud_costs"] == []
        assert res["total_cost"] == 0.0
        assert "note" in res
    finally:
        fake.close()


@pytest.mark.asyncio
async def test_window_is_whitelisted(fake) -> None:
    # An arbitrary/injected window string is rejected and falls back to 7d.
    res = await fake.call("billing_kubecost_allocation", {"window": "2026-01-01T00:00:00Z,2026-02-01T00:00:00Z"})
    assert res["window"] == "7d"


@pytest.mark.asyncio
async def test_handler_direct_invocation_pattern() -> None:
    fake = build_fake()
    try:
        tool = get_tool("billing_kubecost_assets")
        res = await tool.handler(fake.client, {})
        assert res["count"] == 3
    finally:
        fake.teardown()


@pytest.mark.asyncio
async def test_unknown_tool_raises(fake) -> None:
    with pytest.raises(KeyError):
        await fake.call("billing_kubecost_does_not_exist", {})
