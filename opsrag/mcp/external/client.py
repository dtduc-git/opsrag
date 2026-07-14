"""Minimal MCP client over streamable-HTTP (JSON-RPC 2.0), no mcp SDK.

Speaks the subset the adapter needs: initialize -> notifications/initialized ->
tools/list -> tools/call. Responses may be application/json OR text/event-stream
(SSE `data:` frames) -- both handled. Auth is a single injected header
(e.g. Authorization: Sentry-Bearer <token>). Never raises out of the JSON-RPC
envelope: call_tool returns {"error": ...} on any failure.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

_log = logging.getLogger("opsrag.mcp.external.client")

_PROTOCOL_VERSION = "2025-06-18"
_ACCEPT = "application/json, text/event-stream"


class ExternalMCPClient:
    def __init__(
        self,
        url: str,
        *,
        auth_header: tuple[str, str] | None = None,
        timeout: float = 20.0,
        _transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = url
        headers = {"Content-Type": "application/json", "Accept": _ACCEPT}
        if auth_header:
            headers[auth_header[0]] = auth_header[1]
        self._client = httpx.AsyncClient(
            headers=headers, timeout=timeout, transport=_transport
        )
        self._session_id: str | None = None
        self._next_id = 0

    def _rpc(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        body: dict[str, Any] = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            body["params"] = params
        return body

    @staticmethod
    def _parse(resp: httpx.Response) -> dict:
        ctype = resp.headers.get("content-type", "")
        text = resp.text
        if "text/event-stream" in ctype:
            data_lines = [ln[5:].strip() for ln in text.splitlines() if ln.startswith("data:")]
            if not data_lines:
                return {}
            return json.loads(data_lines[-1])
        return json.loads(text) if text.strip() else {}

    async def _send(self, method: str, params: dict | None = None) -> dict:
        headers = {}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        resp = await self._client.post(self._url, json=self._rpc(method, params), headers=headers)
        sid = resp.headers.get("mcp-session-id")
        if sid:
            self._session_id = sid
        resp.raise_for_status()
        parsed = self._parse(resp)
        return parsed if isinstance(parsed, dict) else {}

    async def initialize(self) -> dict:
        params = {
            "protocolVersion": _PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "opsrag-external-adapter", "version": "0.1"},
        }
        result = await self._send("initialize", params)
        # notifications/initialized is a notification (no id, no result expected).
        try:
            headers = {"Content-Type": "application/json", "Accept": _ACCEPT}
            if self._session_id:
                headers["Mcp-Session-Id"] = self._session_id
            await self._client.post(
                self._url,
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                headers=headers,
            )
        except httpx.HTTPError as exc:  # notification is best-effort
            _log.debug("initialized notification failed (non-fatal): %s", exc)
        return result.get("result", {})

    async def list_tools(self) -> list[dict]:
        result = await self._send("tools/list", {})
        tools = (result.get("result") or {}).get("tools") or []
        return [t for t in tools if isinstance(t, dict) and t.get("name")]

    async def call_tool(self, name: str, args: dict | None) -> dict:
        try:
            result = await self._send("tools/call", {"name": name, "arguments": args or {}})
        except (httpx.HTTPError, ValueError) as exc:
            # httpx.HTTPError: transport/status failures. ValueError: malformed
            # JSON in the response body. Either way, never break the JSON-RPC
            # envelope -- surface it as a call_tool error result instead.
            _log.warning("external call_tool %s failed: %s", name, exc)
            return {"error": f"external MCP call failed: {exc}"}
        if "error" in result:
            return {"error": result["error"]}
        return result.get("result", {})

    async def aclose(self) -> None:
        await self._client.aclose()
