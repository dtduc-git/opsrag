"""Docs-accuracy guards for the two-graph architecture + MCP count.

A prior docs pass asserted FALSE properties: it called the Neo4j knowledge
graph "zero-LLM / deterministic" (it is built by a hybrid rule + LLM
extractor by default), and claimed retrieval "never queries the graph"
absolutely (the optional light entity-graph adds 1-hop expansion). It also
left the MCP connector count diverged between README (23) and
docs/architecture.md (20).

These tests lock the corrected, verifiable facts so the regressions cannot
silently return. They read only repo files -- no secrets, no network.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


def _read(repo_root: Path, rel: str) -> str:
    path = repo_root / rel
    assert path.is_file(), f"{rel} not found at {path}"
    return path.read_text(encoding="utf-8")


def test_registry_has_27_integrations(repo_root: Path) -> None:
    """The MCP registry is the single source of truth for the connector
    count; both docs must match it. Keeps doc claims registry-accurate.
    (27 = 23 base + the 4 Billing connectors: gcp/datadog/kubecost/mongodb.)"""
    src = _read(repo_root, "opsrag/mcp/registry.py")
    # Each integration entry declares a top-level ``name="..."`` field.
    names = re.findall(r'^        name="([^"]+)"', src, flags=re.MULTILINE)
    assert len(set(names)) == 27, f"expected 27 registry integrations, got {sorted(set(names))}"


@pytest.mark.parametrize("rel", ["README.md", "docs/architecture.md"])
def test_mcp_count_matches_registry(repo_root: Path, rel: str) -> None:
    """README and docs/architecture.md must both cite 27 MCP connectors,
    never the stale 20/23."""
    text = _read(repo_root, rel)
    assert "27" in text, f"{rel} should cite the registry-accurate 27 MCP connectors"
    # No stray "20 MCP" / "(20, gated)" style stale counts.
    assert not re.search(r"\b20\s+(named\s+)?MCP", text), f"{rel} still cites a stale 20 MCP count"
    assert "(20, gated)" not in text, f"{rel} still has the stale (20, gated) box"


@pytest.mark.parametrize("rel", ["README.md", "docs/architecture.md"])
def test_neo4j_graph_not_called_zero_llm(repo_root: Path, rel: str) -> None:
    """The Neo4j knowledge graph must NOT be advertised as zero-LLM or
    deterministic: it is built by a hybrid rule + LLM extractor by default.
    The ONLY legitimate 'zero-LLM' / 'no-LLM' / 'deterministic' mentions are
    (a) the light entity-graph / RuleBasedExtractor, or (b) the explicit
    rule_based opt-out for a deterministic build."""
    text = _read(repo_root, rel)
    for m in re.finditer(r"(zero-?llm|no-?llm|deterministic)", text, flags=re.IGNORECASE):
        start = max(0, m.start() - 200)
        ctx = text[start : m.end() + 200].lower()
        legit = (
            "light entity-graph" in ctx
            or "rulebasedextractor" in ctx
            or "rule_based" in ctx
            or "rule-based" in ctx
        )
        assert legit, (
            f"{rel}: '{m.group(0)}' near offset {m.start()} attaches a "
            f"zero-LLM/deterministic claim to the wrong graph; context:\n"
            f"...{text[start : m.end() + 200]}..."
        )


@pytest.mark.parametrize("rel", ["README.md", "docs/architecture.md"])
def test_no_absolute_never_queries_graph_claim(repo_root: Path, rel: str) -> None:
    """Reject the over-broad 'never queries the graph' absolute -- the light
    entity-graph adds 1-hop expansion when enabled."""
    text = _read(repo_root, rel).lower()
    assert "never queries the graph" not in text, (
        f"{rel} keeps the absolute 'never queries the graph' claim; "
        "the light entity-graph adds 1-hop expansion when enabled"
    )


@pytest.mark.parametrize("rel", ["README.md", "docs/architecture.md"])
def test_light_entity_graph_documented(repo_root: Path, rel: str) -> None:
    """Both docs must describe the optional light entity-graph (1-hop
    expansion, off by default) so retrieval-graph behavior is accurate."""
    text = _read(repo_root, rel).lower()
    assert "light entity-graph" in text, f"{rel} omits the light entity-graph"
    assert "1-hop" in text, f"{rel} omits the 1-hop entity expansion detail"


def test_config_defaults_back_the_docs(repo_root: Path) -> None:
    """The doc claims hinge on these config defaults: entity_extraction
    defaults to the LLM-using hybrid method, and the light entity-graph is
    off by default. Lock them so the prose stays true to the code."""
    cfg = _read(repo_root, "opsrag/config.py")
    assert re.search(
        r'method:\s*Literal\[[^\]]*\]\s*=\s*"hybrid"', cfg
    ), "entity_extraction.method default is no longer 'hybrid' -- update the docs"
    # LightGraphConfig.enabled default must remain False.
    light_block = cfg[cfg.index("class LightGraphConfig") :]
    light_block = light_block[: light_block.index("class ", 5)]
    assert re.search(r"\benabled:\s*bool\s*=\s*False", light_block), (
        "LightGraphConfig.enabled default is no longer False -- update the docs"
    )


def test_cartography_gate_is_tool_binding(repo_root: Path) -> None:
    """The classifier infra_graph comments must match the ACTUAL gate:
    _cartography_enabled is 'any cartography_* MCP tool bound', not a
    'graph backend'."""
    classifier = _read(repo_root, "opsrag/agent/classifier.py")
    assert "no graph\n    # backend is bound" not in classifier
    assert "no graph backend is\n    # bound" not in classifier
    assert "cartography_* MCP\n    # tool is bound" in classifier or (
        "cartography_*\n    # MCP tool is bound" in classifier
    ), "classifier infra_graph comment should reference the cartography_* tool-binding gate"

    gate = _read(repo_root, "opsrag/agent/nodes/multi_agent.py")
    assert 'startswith("cartography_")' in gate, (
        "the actual gate (_cartography_enabled) should test cartography_* tool names"
    )
