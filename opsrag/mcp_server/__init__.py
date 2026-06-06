"""MCP-server-as-proxy: expose OpsRAG's existing tool surface to external
MCP clients (Claude Code, mcp-remote shims) over HTTP+SSE with bearer-token
auth. Pure proxy -- no LLM invocations are issued on this code path; the
client supplies its own model.

Two pieces:

1. **Token management** (`token_store`) -- Postgres-backed table of API
   tokens. Tokens are random 32-byte URL-safe base64 strings, prefixed
   `opsrag_`. Plaintext is shown to the user exactly once at creation;
   the DB stores only `sha256(plaintext)`.

2. **MCP server** (`server`) -- JSON-RPC 2.0 dispatcher implementing the
   three core MCP methods (`initialize`, `tools/list`, `tools/call`) plus
   `notifications/initialized` and `ping`. Backed by an allow-listed
   subset of `opsrag.mcp.ALL_MCP_TOOLS` (see `registry`), enforced by an
   in-process sliding-window rate limiter (`rate_limit`), and audit-
   logged into `opsrag_mcp_audit` (`audit`).

The HTTP wiring (SSE + JSON-RPC inbox + token-management endpoints) is
in `opsrag.api.mcp_routes`. The integration agent mounts that router in
`opsrag.api.server` and stashes `MCPTokenStore`, `TokenRateLimiter`,
`AuditLogger`, plus a singleton `MCPServer` on `app.state`.
"""
from opsrag.mcp_server.audit import AuditLogger
from opsrag.mcp_server.rate_limit import TokenRateLimiter
from opsrag.mcp_server.registry import (
    SAFE_FOR_EXTERNAL_TOOLS,
    build_external_registry,
)
from opsrag.mcp_server.server import MCPProtocolError, MCPServer
from opsrag.mcp_server.token_store import TOKEN_PREFIX, MCPTokenStore

__all__ = [
    "AuditLogger",
    "MCPProtocolError",
    "MCPServer",
    "MCPTokenStore",
    "SAFE_FOR_EXTERNAL_TOOLS",
    "TOKEN_PREFIX",
    "TokenRateLimiter",
    "build_external_registry",
]
