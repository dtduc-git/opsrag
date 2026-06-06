"""Unit tests for the read-only Azure MCP tools.

Exercises every tool through build_fake() with NO azure libs, NO Azure
credentials, and NO network: build_fake() swaps the module-level
_get / _post for canned, shape-faithful responders and restores them on
teardown. Asserts the parsed shape the handlers produce.
"""
from __future__ import annotations

import pytest

from opsrag.mcp.azure import AZURE_TOOLS, build_fake, get_tool


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()  # restore the real module-level _get / _post


def test_fake_exposes_exact_tool_set(fake) -> None:
    expected = {
        "azure_monitor_logs_query",
        "azure_monitor_metrics_query",
        "azure_aks_list_clusters",
        "azure_list_resource_groups",
        "azure_resource_graph_query",
    }
    assert set(fake.tool_names()) == expected
    assert len(AZURE_TOOLS) == 5


@pytest.mark.asyncio
async def test_list_resource_groups(fake) -> None:
    result = await fake.call("azure_list_resource_groups", {})
    assert result["subscription_id"] == "sub-123"
    assert result["count"] == 2
    rg = result["resource_groups"][0]
    assert rg["name"] == "rg-prod"
    assert rg["location"] == "eastus"
    assert rg["provisioning_state"] == "Succeeded"
    assert rg["tags"] == {"env": "prod"}


@pytest.mark.asyncio
async def test_aks_list_clusters(fake) -> None:
    result = await fake.call("azure_aks_list_clusters", {})
    assert result["count"] == 1
    c = result["clusters"][0]
    assert c["name"] == "aks-prod"
    assert c["kubernetes_version"] == "1.29.2"
    assert c["power_state"] == "Running"
    assert c["fqdn"].endswith("azmk8s.io")
    assert len(c["node_pools"]) == 2
    assert c["node_pools"][0]["name"] == "system"
    assert c["node_pools"][1]["vm_size"] == "Standard_D8s_v5"


@pytest.mark.asyncio
async def test_resource_graph_query(fake) -> None:
    result = await fake.call(
        "azure_resource_graph_query",
        {"query": "Resources | project name, location, type"},
    )
    assert result["subscription_id"] == "sub-123"
    assert result["count"] == 2
    assert result["total_records"] == 2
    assert result["data"][0]["name"] == "vm-1"
    assert result["data"][1]["type"] == "microsoft.compute/virtualmachines"


@pytest.mark.asyncio
async def test_resource_graph_query_requires_query(fake) -> None:
    from opsrag.mcp.azure import AzureMCPError

    with pytest.raises(AzureMCPError):
        await fake.call("azure_resource_graph_query", {})


@pytest.mark.asyncio
async def test_monitor_metrics_query(fake) -> None:
    result = await fake.call(
        "azure_monitor_metrics_query",
        {
            "resource_id": (
                "/subscriptions/sub-123/resourceGroups/rg-prod/providers/"
                "Microsoft.Compute/virtualMachines/vm-1"
            ),
            "metricnames": "Percentage CPU",
            "timespan": "PT1H",
        },
    )
    assert result["count"] == 1
    m = result["metrics"][0]
    assert m["name"] == "Percentage CPU"
    assert m["unit"] == "Percent"
    assert m["timeseries"][0]["data"][0]["average"] == 12.5


@pytest.mark.asyncio
async def test_monitor_metrics_query_requires_resource_id(fake) -> None:
    from opsrag.mcp.azure import AzureMCPError

    with pytest.raises(AzureMCPError):
        await fake.call("azure_monitor_metrics_query", {"metricnames": "Percentage CPU"})


@pytest.mark.asyncio
async def test_monitor_logs_query(fake) -> None:
    result = await fake.call(
        "azure_monitor_logs_query",
        {
            "workspace_id": "ws-guid-123",
            "query": "ContainerLog | take 10",
            "timespan": "PT1H",
        },
    )
    assert result["workspace_id"] == "ws-guid-123"
    assert result["table_count"] == 1
    table = result["tables"][0]
    assert table["name"] == "PrimaryResult"
    assert table["columns"] == ["TimeGenerated", "Computer", "Level", "Message"]
    assert table["row_count"] == 2
    row0 = table["rows"][0]
    assert row0["Computer"] == "aks-node-1"
    assert row0["Level"] == "Error"
    assert row0["Message"] == "OOMKilled container"


@pytest.mark.asyncio
async def test_monitor_logs_query_requires_workspace(fake) -> None:
    from opsrag.mcp.azure import AzureMCPError

    with pytest.raises(AzureMCPError):
        await fake.call("azure_monitor_logs_query", {"query": "Heartbeat"})


@pytest.mark.asyncio
async def test_logs_query_via_get_tool_handler(fake) -> None:
    # Exercise the get_tool(name).handler(client, args) path explicitly.
    tool = get_tool("azure_monitor_logs_query")
    res = await tool.handler(
        fake.client,
        {"workspace_id": "ws-guid-123", "query": "Heartbeat | take 1"},
    )
    assert res["tables"][0]["row_count"] == 2


@pytest.mark.asyncio
async def test_unknown_tool_raises(fake) -> None:
    with pytest.raises(KeyError):
        await fake.call("azure_does_not_exist", {})
