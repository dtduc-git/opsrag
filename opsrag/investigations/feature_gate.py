"""Config-driven gate for whether the Investigate feature is surfaced.

The Investigate tab/feature is only meaningful when the operator enabled a
live-telemetry MCP integration. The UI reads `investigation_enabled` from
/ui-config (driven by this helper); a vendor-neutral deployment with no
telemetry enabled never sees it. Lives in the investigations package (not a
route module) so it survives engine refactors.
"""
from __future__ import annotations

from typing import Any

# Integration NAMES (config keys under `mcp:`) whose LIVE signals make the
# investigation feature meaningfully more than RAG-only. `code` is excluded
# on purpose: code search alone isn't live telemetry and isn't reason enough
# to surface the Investigate tab.
_INVESTIGATION_TELEMETRY_INTEGRATIONS = (
    "datadog", "prometheus", "kubernetes", "loki",
    "grafana", "splunk", "sentry", "rootly",
)


def investigation_live_telemetry_enabled(cfg: Any) -> bool:
    """True iff the operator enabled at least one live-telemetry MCP
    integration -- the condition under which the Investigate feature is
    worth surfacing in the UI. Purely config-driven: reads the same
    `cfg.mcp` enables the investigation engine honors, so the feature
    appears/disappears with the operator's integration picks."""
    mcp = getattr(cfg, "mcp", None) or {}
    for name in _INVESTIGATION_TELEMETRY_INTEGRATIONS:
        block = mcp.get(name) if isinstance(mcp, dict) else getattr(mcp, name, None)
        if block is not None and getattr(block, "enabled", False):
            return True
    return False
