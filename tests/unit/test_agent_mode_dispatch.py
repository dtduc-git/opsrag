"""L15: explicit agent.mode dispatch (server._build_agent_graph).

The bare ``else -> build_full_graph`` was replaced with an EXPLICIT mapping
that (a) still builds the full graph for ``full``/``hybrid`` (so which graph
builds is UNCHANGED -- pure-equivalence requirement), (b) emits a one-time
warning for the removed ``hybrid`` legacy alias, and (c) raises on an
unknown/typo'd mode so a misconfigured knob fails fast at startup instead of
silently degrading to the full graph.

These tests spy the four builders so no real LangGraph / providers are needed;
they assert WHICH builder ran (and the warning / raise behaviour).
"""
from __future__ import annotations

import logging

import pytest

import opsrag.api.server as server
from opsrag.config import Settings


class _Fake:
    """Any attribute access returns a harmless no-op -- the spied builders
    ignore their args, so providers never get exercised."""

    def __getattr__(self, name):  # noqa: ANN001
        def _noop(*a, **k):  # noqa: ANN002, ANN003
            return None
        return _noop


def _spy_builders(monkeypatch):
    """Replace the four graph builders with sentinels recording which ran."""
    seen: list[str] = []

    def _mk(tag):
        def _builder(*a, **k):  # noqa: ANN002, ANN003
            seen.append(tag)
            return f"graph::{tag}"
        return _builder

    monkeypatch.setattr(server, "build_multi_agent_graph", _mk("multi_agent"))
    monkeypatch.setattr(server, "build_tool_calling_graph", _mk("tool_calling"))
    monkeypatch.setattr(server, "build_minimal_graph", _mk("minimal"))
    monkeypatch.setattr(server, "build_full_graph", _mk("full"))
    return seen


def _call(mode: str):
    cfg = Settings()
    cfg.agent.mode = mode
    return server._build_agent_graph(
        cfg, _Fake(), checkpointer=None, known_repos=[], model_router=_Fake(),
    )


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("multi_agent", "multi_agent"),
        ("tool_calling", "tool_calling"),
        ("minimal", "minimal"),
        ("full", "full"),
    ],
)
def test_each_mode_routes_to_its_builder(monkeypatch, mode, expected):
    seen = _spy_builders(monkeypatch)
    out = _call(mode)
    assert seen == [expected]
    assert out == f"graph::{expected}"


def test_hybrid_still_builds_full_and_warns(monkeypatch, caplog):
    """hybrid is a REMOVED legacy alias: it must still build the FULL graph
    (which graph builds is unchanged) but emit a one-time migration warning."""
    seen = _spy_builders(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="opsrag.server"):
        out = _call("hybrid")
    # which graph builds is UNCHANGED -- still the full graph.
    assert seen == ["full"]
    assert out == "graph::full"
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("hybrid" in m and "legacy alias" in m for m in warnings), warnings


def test_unknown_mode_raises_at_build(monkeypatch):
    """A typo'd / unknown mode must fail fast (ValueError) rather than silently
    falling through to the full graph. (Settings.agent.mode is a pydantic
    Literal, so we bypass it by setting the attr directly -- modelling a config
    built outside the validator / a future mode addition.)"""
    seen = _spy_builders(monkeypatch)
    cfg = Settings()
    # Bypass the pydantic Literal to inject an invalid mode.
    object.__setattr__(cfg.agent, "mode", "totally-bogus")
    with pytest.raises(ValueError, match="unknown agent.mode"):
        server._build_agent_graph(
            cfg, _Fake(), checkpointer=None, known_repos=[], model_router=_Fake(),
        )
    assert seen == [], "no builder should run for an unknown mode"
