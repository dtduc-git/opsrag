"""Enabled-integration gating for MCP tools (T087).

The per-module tool lists in ``opsrag.mcp`` aggregate into the always-present
superset ``ALL_MCP_TOOLS``. US2 requires that only the integrations the
operator ENABLED in ``config.yaml`` are offered to the agent. This module
computes the enabled tool set from ``Settings`` and exposes a process-level
"active enabled" filter the agent consults (mirroring
``opsrag.agent.prompt_render``'s active-deployment global).

Wiring:

- ``create_app`` calls ``set_active_enabled(enabled_integration_names(cfg))``
  once at startup.
- ``opsrag.agent.nodes.multi_agent`` filters ``ALL_MCP_TOOLS`` through
  :func:`filter_enabled` so the LLM only sees enabled tools.

Default (no setter called) is ``None`` -> "no gating" so existing tests and
tools see the full superset; once the setter runs with the (all-disabled)
default config, the agent correctly sees zero MCP tools.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from opsrag.mcp.registry import REGISTRY


def enabled_integration_names(settings: Any) -> tuple[str, ...]:
    """Names of integrations with ``enabled is True`` in ``settings.mcp``."""
    mcp_map = getattr(settings, "mcp", {}) or {}
    return tuple(
        name for name, block in mcp_map.items() if getattr(block, "enabled", False)
    )


def enabled_tool_names(settings: Any) -> frozenset[str]:
    """Tool names contributed by all enabled integrations (per the registry)."""
    names: set[str] = set()
    for name in enabled_integration_names(settings):
        integration = REGISTRY.get(name)
        if integration is not None:
            names.update(integration.tool_names)
    return frozenset(names)


# ---------------------------------------------------------------------------
# Process-level active enabled set.
# ---------------------------------------------------------------------------
_active_enabled: frozenset[str] | None = None


def set_active_enabled(integration_names: Iterable[str] | None) -> None:
    """Install the active set of enabled integration NAMES. ``None`` disables
    gating (full superset visible). Call once at app startup."""
    global _active_enabled
    if integration_names is None:
        _active_enabled = None
        return
    names: set[str] = set()
    for name in integration_names:
        integration = REGISTRY.get(name)
        if integration is not None:
            names.update(integration.tool_names)
    _active_enabled = frozenset(names)


def active_enabled_tool_names() -> frozenset[str] | None:
    """Active enabled tool-name set, or ``None`` when gating is off."""
    return _active_enabled


def filter_enabled(tools: Iterable[Any]) -> list[Any]:
    """Return only the tools whose ``name`` is in the active enabled set.

    When gating is off (no setter called) returns the tools unchanged, so
    callers can use this unconditionally."""
    allowed = _active_enabled
    if allowed is None:
        return list(tools)
    return [t for t in tools if getattr(t, "name", None) in allowed]
