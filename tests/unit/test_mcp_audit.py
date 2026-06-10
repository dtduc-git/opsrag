"""MCP audit read API (admin view).

These cover the wiring without a live Postgres: the route helper returns 503
when the audit logger is not configured, and AuditLogger's read methods
degrade to empty when the pool was never opened. The admin scope gate rides
the shared ``require_scope(Scope.ADMIN)`` dependency (tested elsewhere).
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from opsrag.api.mcp_routes import _require_audit
from opsrag.mcp_server.audit import AuditLogger


class _FakeRequest:
    def __init__(self, audit: object | None) -> None:
        self.app = type("_App", (), {})()
        self.app.state = type("_State", (), {})()
        self.app.state.mcp_audit = audit


def test_require_audit_503_when_not_configured():
    with pytest.raises(HTTPException) as ei:
        _require_audit(_FakeRequest(None))
    assert ei.value.status_code == 503


def test_require_audit_returns_logger_when_present():
    sentinel = object()
    assert _require_audit(_FakeRequest(sentinel)) is sentinel


def test_audit_query_empty_when_pool_not_opened():
    logger = AuditLogger("postgresql://localhost/opsrag")  # never opened
    rows, total = asyncio.run(logger.query(limit=50))
    assert rows == []
    assert total == 0


def test_audit_summary_empty_when_pool_not_opened():
    logger = AuditLogger("postgresql://localhost/opsrag")
    summary = asyncio.run(logger.summary())
    assert summary["total_calls"] == 0
    assert summary["error_count"] == 0
    assert summary["top_tools"] == []
