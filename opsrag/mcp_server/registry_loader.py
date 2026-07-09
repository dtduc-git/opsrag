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

import contextvars
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
    """Return only the tools whose ``name`` is in the active enabled set AND
    (when a per-request connector permission set is installed) in the current
    request's allowed tool set.

    Two layers stack here:
      * ``_active_enabled`` -- the process-wide "which integrations did the
        operator enable" gate (``set_active_enabled`` at startup).
      * the per-request contextvar (``set_request_connector_perms``) -- the
        RBAC "which connectors may THIS user use" gate.

    When BOTH are off (no setter called) returns the tools unchanged, so
    callers can use this unconditionally."""
    allowed = _active_enabled
    req_allowed = _request_allowed_tools.get()
    if allowed is None and req_allowed is None:
        return list(tools)
    out: list[Any] = []
    for t in tools:
        name = getattr(t, "name", None)
        if allowed is not None and name not in allowed:
            continue
        if req_allowed is not None and name not in req_allowed:
            continue
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# Per-request connector RBAC (installed per API request from the caller's
# effective connector permissions; consulted by ``filter_enabled`` above and
# by the tool executor's permission gate).
# ---------------------------------------------------------------------------
#
# ``None`` means "no per-request gating" (unauthenticated internal callers,
# tests, or callers who opt out) -> only the process-wide ``_active_enabled``
# applies. A frozenset (possibly empty) means "restrict this request to exactly
# these tool names".
_request_allowed_tools: contextvars.ContextVar[frozenset[str] | None] = (
    contextvars.ContextVar("opsrag_request_allowed_tools", default=None)
)
# Connectors that ARE enabled on the deployment but the current caller may NOT
# use. Surfaced to the agent prompt so it can honestly refuse ("you don't have
# permission for X") instead of claiming the capability doesn't exist.
_request_denied_connectors: contextvars.ContextVar[frozenset[str]] = (
    contextvars.ContextVar("opsrag_request_denied_connectors", default=frozenset())
)

# tool name -> integration (connector) name. Built once from the registry.
_TOOL_TO_CONNECTOR: dict[str, str] = {
    tool: name
    for name, integ in REGISTRY.items()
    for tool in integ.tool_names
}


def tool_names_for_connectors(connectors: Iterable[str]) -> frozenset[str]:
    """Union the tool names contributed by ``connectors`` (per the registry)."""
    names: set[str] = set()
    for c in connectors:
        integ = REGISTRY.get(c)
        if integ is not None:
            names.update(integ.tool_names)
    return frozenset(names)


def connector_for_tool(tool_name: str | None) -> str | None:
    """The connector (integration) a tool belongs to, or ``None`` if unknown
    (e.g. a retrieval-only tool like ``knowledge_search`` maps to ``knowledge``;
    a made-up tool name maps to ``None``)."""
    if tool_name is None:
        return None
    return _TOOL_TO_CONNECTOR.get(tool_name)


# Per-connector operator system-prompts (config.mcp.<name>.system_prompt),
# bound once at startup. The reasoner appends a connector's note to every one
# of its tools' descriptions so tool SELECTION honors deployment routing (e.g.
# "Datadog = tracing only; logs in Elasticsearch") -- configurable, not hardcoded.
_CONNECTOR_SYSTEM_PROMPTS: dict[str, str] = {}


def set_connector_system_prompts(settings: Any) -> None:
    """Populate the per-connector system-prompt map from ``config.mcp.<name>.
    system_prompt``. Idempotent; safe to call at every startup/reload."""
    global _CONNECTOR_SYSTEM_PROMPTS
    out: dict[str, str] = {}
    mcp = getattr(settings, "mcp", {}) or {}
    for name, block in mcp.items():
        note = (getattr(block, "system_prompt", None) or "").strip()
        if note:
            out[name] = note
    _CONNECTOR_SYSTEM_PROMPTS = out


def connector_system_prompt(connector: str | None) -> str | None:
    """The operator system-prompt configured for ``connector``, or ``None``."""
    if not connector:
        return None
    return _CONNECTOR_SYSTEM_PROMPTS.get(connector)


def set_request_connector_perms(
    allowed_connectors: Iterable[str] | None,
    enabled_connectors: Iterable[str] = (),
) -> contextvars.Token | None:
    """Install the current request's allowed connectors.

    ``allowed_connectors=None`` disables per-request gating for this context
    (process-wide gating still applies). Otherwise the allowed tool-name set is
    derived from the connectors and stored; the difference between
    ``enabled_connectors`` and ``allowed_connectors`` is recorded as the denied
    set for the prompt hint.

    Returns the ContextVar token for ``_request_allowed_tools`` so a caller can
    ``reset`` it, or ``None`` when gating was disabled."""
    if allowed_connectors is None:
        _request_denied_connectors.set(frozenset())
        return _request_allowed_tools.set(None)
    allowed = {str(c) for c in allowed_connectors}
    enabled = {str(c) for c in enabled_connectors} or allowed
    _request_denied_connectors.set(frozenset(enabled - allowed))
    return _request_allowed_tools.set(tool_names_for_connectors(allowed))


def clear_request_connector_perms() -> None:
    """Reset per-request connector gating to "off" (used in finally blocks)."""
    _request_allowed_tools.set(None)
    _request_denied_connectors.set(frozenset())


def request_allowed_tool_names() -> frozenset[str] | None:
    """The current request's allowed tool-name set, or ``None`` when off."""
    return _request_allowed_tools.get()


def request_denied_connectors() -> frozenset[str]:
    """Enabled-but-forbidden connectors for the current request (for the
    agent's refusal prompt). Empty when gating is off or nothing is denied."""
    return _request_denied_connectors.get()


def connector_permission_prompt_block() -> str:
    """A system-prompt note listing enabled-but-forbidden connectors for the
    current request (per-connector RBAC). Empty when the caller may use every
    enabled connector (the common case) or when gating is off.

    Injected into EVERY answer-writing node (reasoner, multi-agent generator,
    and the retrieval-path generator) so that whichever route the triage picks,
    the model refuses with an honest permission message instead of silently
    substituting tangential documentation for the forbidden live data. The
    forbidden connectors' tools are already absent from the tool list; this just
    makes the refusal accurate."""
    denied = _request_denied_connectors.get()
    if not denied:
        return ""
    names = ", ".join(
        f"{REGISTRY[c].display_name if c in REGISTRY else c} (`{c}`)"
        for c in sorted(denied)
    )
    return (
        "\nCONNECTOR PERMISSIONS (per-user RBAC): the current user is NOT "
        f"permitted to use these connectors: {names}. Their live tools are "
        "intentionally unavailable to this user. If answering the question "
        "would require data from one of them, you MUST LEAD with the permission "
        "limitation -- state plainly and explicitly that the user does NOT have "
        "permission to access that data (e.g. \"I can't get that information "
        "because you don't have permission to use Datadog\"). Do NOT claim the "
        "capability doesn't exist, do NOT fabricate an answer, and do NOT "
        "quietly substitute unrelated documentation as if it were the requested "
        "live data. You may add that they can request access from an OpsRAG "
        "admin.\n"
    )


def is_tool_allowed(tool_name: str | None) -> bool:
    """Whether ``tool_name`` is callable in the current request context.

    ``True`` when per-request gating is off; otherwise membership in the
    request's allowed set. This is the authoritative server-side gate the tool
    executor consults before dispatching a call."""
    req_allowed = _request_allowed_tools.get()
    if req_allowed is None:
        return True
    return tool_name in req_allowed
