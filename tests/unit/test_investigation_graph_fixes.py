"""Unit tests for the investigation-agent bug-cluster fixes.

Covers the verified findings remediated in the hypothesis-tree agent:
  H  -- zombie duplicate nodes (rejected dups must not linger in
        nodes_by_id / parent.children / statement_embeddings)
  G1 -- root-level hypotheses are now deduped (was sub-only)
  I2 -- dedup embeddings use a separate budget purpose and do NOT
        inflate total_tool_calls / retrieval_calls
  D1/B2 -- final chain is chosen by aggregate evidence strength
        (Citation.score + corroboration), not raw depth
  I3 -- off-topic evidence cap uses token-boundary matching and a
        narrowed generic-doc allowlist

These exercise the node closures directly with fakes -- no live LLM,
retriever, or MCP backend.
"""
from __future__ import annotations

import json

import pytest

from opsrag.agents.investigation.graph import (
    _best_validated_chain,
    _is_evidence_on_topic,
    generate_hypotheses_node,
    generate_sub_hypotheses_node,
)
from opsrag.agents.investigation.state import (
    AlertContext,
    Citation,
    HypothesisNode,
    InvestigationState,
)
from opsrag.interfaces.llm import LLMResponse


# --- fakes ----------------------------------------------------------


class _HypothesisLLM:
    """Returns a fixed hypothesis list for any llm_query_gen call."""

    def __init__(self, statements: list[str]) -> None:
        self._statements = statements
        self.calls: list[str] = []

    async def generate(self, messages, *, temperature=0.0, max_tokens=4096,
                        purpose=None, response_schema=None, **kwargs) -> LLMResponse:
        self.calls.append(purpose or "")
        payload = {
            "hypotheses": [
                {"statement": s, "rationale": "r"} for s in self._statements
            ]
        }
        return LLMResponse(
            content=json.dumps(payload),
            model="fake",
            usage={"input_tokens": 1, "output_tokens": 1},
        )


def _family_embedder():
    """Map a statement to a one-hot vector keyed by its first word, so
    statements sharing a leading token are cosine-identical (dup) and
    statements with different leading tokens are orthogonal (distinct)."""
    families = {"alpha": [1.0, 0.0, 0.0], "beta": [0.0, 1.0, 0.0],
                "gamma": [0.0, 0.0, 1.0]}

    async def _embed(stmt: str) -> list[float]:
        key = stmt.strip().split(" ", 1)[0].lower()
        return families.get(key, [0.5, 0.5, 0.5])

    return _embed


def _new_state() -> InvestigationState:
    return InvestigationState(
        alert_context=AlertContext(alert_text="api 500s", service_hint="acme-notes-api")
    )


# --- H: zombie duplicate nodes -------------------------------------


@pytest.mark.asyncio
async def test_rejected_sibling_duplicate_leaves_no_zombie() -> None:
    state = _new_state()
    parent = HypothesisNode(statement="alpha parent", depth=0, status="validated",
                            confidence=0.9)
    state.add_node(parent)
    state.current_node_id = parent.id

    # Two sub-hypotheses that embed identically -> second is a dup sibling.
    llm = _HypothesisLLM(["alpha child mechanism", "alpha child restated"])
    node_fn = generate_sub_hypotheses_node(llm, embed_query=_family_embedder())
    await node_fn(state)

    # Exactly one child accepted; the rejected dup must NOT linger.
    assert len(parent.children) == 1, "rejected dup wired into parent.children"
    accepted_id = parent.children[0]
    assert accepted_id in state.nodes_by_id
    # nodes_by_id holds only parent + the one accepted child (no zombie).
    assert len(state.nodes_by_id) == 2
    # budget node count stays consistent with the live tree.
    assert state.budget_state.total_nodes == len(state.nodes_by_id)
    # The rejected dup's embedding must be gone so it can't poison a later
    # sibling check ("duplicate-of-a-duplicate" false prune).
    assert set(state.statement_embeddings) == {accepted_id}
    # ...but the rejection is still recorded for observability.
    assert any(e.event_type == "duplicate_sibling" for e in state.agent_trace)


# --- G1: root-level dedup ------------------------------------------


@pytest.mark.asyncio
async def test_root_hypotheses_are_deduped() -> None:
    state = _new_state()
    # Two near-duplicate roots + one distinct -> only two survive.
    llm = _HypothesisLLM(["alpha root one", "alpha root two", "beta root three"])
    node_fn = generate_hypotheses_node(llm, embed_query=_family_embedder())
    await node_fn(state)

    assert len(state.root_ids) == 2, "duplicate root was not pruned"
    assert state.budget_state.total_nodes == 2
    # Both candidates were embedded for dedup...
    assert state.budget_state.embed_dedup_calls == 3
    # ...but embeds must NOT count as retrieval / tool-call budget (I2).
    assert state.budget_state.retrieval_calls == 0
    # Only the one llm_query_gen call counts toward the tool-call budget.
    assert state.budget_state.total_tool_calls == 1
    assert state.budget_state.llm_query_gen_calls == 1


# --- D1 / B2: evidence-strength chain selection ---------------------


def test_best_chain_prefers_evidence_strength_over_depth() -> None:
    state = _new_state()

    # Shallow but strongly-grounded root: 3 sources at high score.
    strong = HypothesisNode(
        statement="strong shallow", depth=0, status="validated", confidence=0.85,
        evidence=[
            Citation(source_id="s1", chunk_id="c1", score=0.9),
            Citation(source_id="s2", chunk_id="c2", score=0.9),
            Citation(source_id="s3", chunk_id="c3", score=0.9),
        ],
    )
    state.add_node(strong)

    # Deep but thinly-grounded chain: 1 source at low score per node.
    prev = None
    deep_terminal = None
    for i in range(4):
        n = HypothesisNode(
            statement=f"weak deep {i}", depth=i, status="validated", confidence=0.8,
            parent_id=(prev.id if prev else None),
            evidence=[Citation(source_id="sa", chunk_id=f"a{i}", score=0.4)],
        )
        state.add_node(n)
        prev = n
        deep_terminal = n

    chain = _best_validated_chain(state)
    # Old code returned the deepest node; the fix returns the strong one.
    assert chain[-1].id == strong.id
    assert deep_terminal is not None and chain[-1].id != deep_terminal.id


# --- I3: off-topic cap matching ------------------------------------


def _cite(source_id: str, repo: str = "") -> Citation:
    return Citation(source_id=source_id, chunk_id="c", score=0.5, repo=repo)


def test_off_topic_cap_short_hint_requires_token_boundary() -> None:
    # "api" must not match an incidental substring in "rapid-events".
    assert not _is_evidence_on_topic(_cite("samples/rapid-events/log.md"),
                                     service="api", namespace=None)
    # ...but matches a real path segment.
    assert _is_evidence_on_topic(_cite("samples/api/handler.py"),
                                 service="api", namespace=None)


def test_off_topic_cap_bare_runbook_no_longer_auto_passes() -> None:
    # A generic runbook path for an UNRELATED service should not be
    # treated as on-topic for acme-notes-api anymore.
    assert not _is_evidence_on_topic(
        _cite("samples/runbooks/other-service-failover.md"),
        service="acme-notes-api", namespace="acme",
    )
    # A specific SRE-KB marker still passes.
    assert _is_evidence_on_topic(
        _cite("confluence:sre/generic-oom-guide"),
        service="acme-notes-api", namespace="acme",
    )
