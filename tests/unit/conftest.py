"""Shared fixtures for the unit-test suite.

The MCP "active-enabled" gate (``opsrag.mcp_server.registry_loader``) is a
process-global frozenset. Several unit tests legitimately set it -- directly
(``test_registry_loader.py``) or as a side effect of constructing the app /
server (``set_active_enabled(enabled_integration_names(cfg))``). A test that
sets it without restoring leaks a gitlab-only / empty gate into whatever runs
next, which (depending on collection order) collapses the tool_caller registry
to GitLab-only and fails ``test_tool_caller_registry.py``.

Rather than rely on every such test to clean up, this autouse fixture
snapshots the gate before each unit test and restores it afterwards, so no
test can leak gate state into another regardless of run order.
"""
from __future__ import annotations

import pytest

from opsrag.mcp_server import registry_loader as _rl


@pytest.fixture(autouse=True)
def _isolate_active_enabled_gate():
    # Force the gate OFF (None = "all tools") before every unit test, so a
    # value leaked by a prior test (e.g. an empty/gitlab-only gate set while
    # constructing the app) can't collapse this test's registry. Tests that
    # need a specific gate set it themselves in their body. Reset again on
    # teardown so we never leak into the next test either.
    _rl.set_active_enabled(None)
    try:
        yield
    finally:
        _rl.set_active_enabled(None)
