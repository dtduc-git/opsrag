"""Integration test (T074): the Cloudflare MCP tools against the fake backend.

Exercises representative tools through build_fake() with NO network and NO
token, asserting shape-faithful responses and the registry's declared tool
set. Follows the GitLab reference test for the per-MCP fake pattern (FR-012).

The Cloudflare module is data path (b): handlers ignore the client arg and
reach module-internal state (bound config + httpx helpers). build_fake()
installs a synthetic bound config and monkeypatches the HTTP helpers, then
restores them via FakeMCP.teardown.
"""
from __future__ import annotations

import pytest

# Optional-dependency skip guard: cloudflare.py imports httpx at module load.
httpx = pytest.importorskip("httpx")

from opsrag.mcp.cloudflare import build_fake  # noqa: E402
from opsrag.mcp.registry import REGISTRY  # noqa: E402


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()


def test_fake_exposes_registry_tool_set(fake) -> None:
    # The fake's tools must match exactly what the registry declares.
    assert set(fake.tool_names()) == set(REGISTRY["cloudflare"].tool_names)


@pytest.mark.asyncio
async def test_list_zones(fake) -> None:
    result = await fake.call("cloudflare_list_zones", {})
    assert result["count"] == 1
    zone = result["zones"][0]
    assert zone["name"] == "example.com"
    assert zone["status"] == "active"
    assert zone["plan"] == "Pro"


@pytest.mark.asyncio
async def test_list_dns_records(fake) -> None:
    # Pass a zone NAME -> _resolve_zone_id resolves it via the fake /zones.
    result = await fake.call("cloudflare_list_dns_records", {"zone": "example.com"})
    assert result["zone"] == "example.com"
    assert result["count"] == 2
    names = {r["name"] for r in result["records"]}
    assert "www.example.com" in names
    a_record = next(r for r in result["records"] if r["type"] == "A")
    assert a_record["content"] == "192.0.2.10"
    assert a_record["proxied"] is True


@pytest.mark.asyncio
async def test_get_access_app_policies(fake) -> None:
    # Account is auto-resolved by the fake; only app_id is required.
    result = await fake.call(
        "cloudflare_get_access_app_policies",
        {"app_id": "app0000000000000000000000000000a"},
    )
    assert result["app_id"] == "app0000000000000000000000000000a"
    assert result["count"] == 1
    policy = result["policies"][0]
    assert policy["decision"] == "allow"
    assert policy["name"] == "Allow staff"


@pytest.mark.asyncio
async def test_unknown_tool_raises(fake) -> None:
    with pytest.raises(KeyError):
        await fake.call("cloudflare_does_not_exist", {})
