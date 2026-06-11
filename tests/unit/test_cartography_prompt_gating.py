"""The triage prompt must not advertise the cartography_* tool family when it
isn't bound.

cartography was removed from the catalog, but the triage system prompt still
told the model "TAP CARTOGRAPHY FIRST" and showed an all-cartography few-shot
section -> the model planned `cartography_*` calls that failed as unknown tools
(wasted a tool round + risked fabricated output). `_triage_prompt()` strips
that guidance when no cartography_* tool is bound.
"""
from __future__ import annotations

import opsrag.agent.nodes.multi_agent as ma


def test_triage_prompt_strips_cartography_when_unbound() -> None:
    # In the unit context no cartography_* tool is registered.
    assert ma._cartography_enabled() is False
    p = ma._triage_prompt()

    # The "tap cartography first" DIRECTIVES + the all-cartography few-shot
    # examples are gone (the spans that made the model plan cartography calls).
    assert "TAP CARTOGRAPHY FIRST" not in p
    assert "INFRASTRUCTURE GRAPH -- `cartography_*`" not in p
    assert "PER-HOSTNAME DIAGRAM" not in p
    assert "FEW-SHOT TOOL-PICK EXAMPLES" not in p
    # Tools that ONLY appear in the stripped directive/example spans are gone.
    assert "cartography_who_holds_role" not in p
    assert "cartography_pod_blast_radius" not in p
    # (A couple of cautionary cross-refs in the kept SECRET/Cloudflare sections
    # intentionally remain -- they warn against fabrication / cross-zone DNS and
    # don't instruct "use cartography first".)

    # Non-cartography sections are preserved.
    assert "TRIAGE agent" in p
    assert "CLOUDFLARE LIVE QUERIES" in p
    assert "ARCHITECTURE / TOPOLOGY FALLBACK" in p

    assert len(p) < len(ma._SYSTEM_TRIAGE)


def test_triage_prompt_unchanged_when_cartography_bound(monkeypatch) -> None:
    # If cartography is ever re-added (any cartography_* tool bound), the full
    # prompt is returned verbatim -- the guidance reappears automatically.
    monkeypatch.setattr(ma, "_cartography_enabled", lambda: True)
    assert ma._triage_prompt() == ma._SYSTEM_TRIAGE
