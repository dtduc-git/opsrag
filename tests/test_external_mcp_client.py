import json

import httpx
import pytest

from opsrag.mcp.external.client import ExternalMCPClient


def _sse(obj: dict) -> httpx.Response:
    body = "event: message\ndata: " + json.dumps(obj) + "\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)


def _handler(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content)
    method = payload.get("method")
    assert request.headers.get("authorization") == "Sentry-Bearer T0KEN"
    if method == "initialize":
        return _sse({"jsonrpc": "2.0", "id": payload["id"],
                     "result": {"protocolVersion": "2025-06-18", "serverInfo": {"name": "Fake"}}})
    if method == "notifications/initialized":
        return httpx.Response(202)
    if method == "tools/list":
        return _sse({"jsonrpc": "2.0", "id": payload["id"],
                     "result": {"tools": [{"name": "find_projects", "description": "d",
                                           "inputSchema": {"type": "object", "properties": {}}}]}})
    if method == "tools/call":
        return _sse({"jsonrpc": "2.0", "id": payload["id"],
                     "result": {"content": [{"type": "text", "text": "ok:" + payload["params"]["name"]}]}})
    return httpx.Response(400)


@pytest.mark.asyncio
async def test_client_discovers_and_calls():
    transport = httpx.MockTransport(_handler)
    c = ExternalMCPClient("https://fake/mcp", auth_header=("Authorization", "Sentry-Bearer T0KEN"),
                          _transport=transport)
    try:
        await c.initialize()
        tools = await c.list_tools()
        assert [t["name"] for t in tools] == ["find_projects"]
        res = await c.call_tool("find_projects", {"organizationSlug": "x"})
        assert res["content"][0]["text"] == "ok:find_projects"
    finally:
        await c.aclose()


@pytest.mark.asyncio
async def test_call_tool_never_raises_on_http_error():
    transport = httpx.MockTransport(lambda r: httpx.Response(500, text="boom"))
    c = ExternalMCPClient("https://fake/mcp", auth_header=("Authorization", "Sentry-Bearer T0KEN"),
                          _transport=transport)
    try:
        res = await c.call_tool("find_projects", {})
        assert "error" in res
    finally:
        await c.aclose()


@pytest.mark.parametrize("body", ["null", "[]", '"a string"', "42"])
@pytest.mark.asyncio
async def test_call_tool_returns_dict_on_non_object_json(body):
    def _h(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, text=body)

    c = ExternalMCPClient("https://fake/mcp", auth_header=("Authorization", "Sentry-Bearer T0KEN"),
                          _transport=httpx.MockTransport(_h))
    try:
        res = await c.call_tool("find_projects", {})
        assert isinstance(res, dict)  # never raised, always a dict
    finally:
        await c.aclose()
