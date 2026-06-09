"""Unit tests for the P0-B (live-telemetry dispatch) + E1 (best-first)
changes to the investigation agent.

P0-B: test_hypothesis now lets the LLM pick targeted live tools
(datadog/rootly/code), dispatches them, and folds the results into the
judge's evidence with provenance tags.

E1: decide_next is best-first -- it tests EVERY pending hypothesis on the
frontier before drilling into any one branch, then expands the
strongest-evidenced validated node first.
"""
from __future__ import annotations

import json

import pytest

from opsrag.agents.investigation.graph import (
    decide_next_node,
)
# Aliased: a bare `test_hypothesis_node` import would be collected by
# pytest as a test (the `test_` prefix), not treated as the factory.
from opsrag.agents.investigation.graph import (
    test_hypothesis_node as make_test_hypothesis_node,
)
from opsrag.agents.investigation.state import (
    AlertContext,
    Citation,
    HypothesisNode,
    InvestigationState,
)
from opsrag.interfaces.llm import LLMResponse


# --- P0-B: live tool dispatch --------------------------------------


class _RoutingLLM:
    """Returns a tool plan for tool-select calls and a validated verdict
    (citing the live tool chunk) for judge calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def model_name(self) -> str:
        return "fake"

    async def generate(self, messages, *, temperature=0.0, max_tokens=4096,
                        purpose=None, response_schema=None, **kwargs) -> LLMResponse:
        self.calls.append(purpose or "")
        p = purpose or ""
        if "llm_tool_select" in p:
            content = json.dumps({
                "calls": [{
                    "tool": "datadog_search_spans",
                    "args_json": json.dumps({"service": "acme-notes-api"}),
                    "rationale": "look for error spans on the service",
                }]
            })
        elif "llm_judge" in p:
            content = json.dumps({
                "status": "validated",
                "confidence": 0.8,
                "rationale": "live spans show error status on acme-notes-api",
                "supporting_chunk_ids": ["tool:datadog_search_spans:0"],
                "refuting_chunk_ids": [],
            })
        else:
            content = "{}"
        return LLMResponse(content=content, model="fake",
                           usage={"input_tokens": 1, "output_tokens": 1})


async def _fake_retrieve(query: str, top_k: int = 6) -> list[dict]:
    return [{
        "chunk_id": "rag1",
        "source_id": "samples/runbooks/acme-notes-api.md",
        "snippet": "acme-notes-api returns 500s when the upstream DB is unhealthy.",
        "score": 0.7,
        "repo": "samples",
    }]


@pytest.mark.asyncio
async def test_test_hypothesis_dispatches_live_tools_and_tags_provenance() -> None:
    dispatched: list[tuple[str, dict]] = []

    async def _dispatch(name: str, args: dict) -> dict:
        dispatched.append((name, args))
        return {"tool": name, "error": None,
                "data": {"spans": [{"service": "acme-notes-api",
                                    "status": "error", "count": 42}]}}

    catalog = [{
        "name": "datadog_search_spans",
        "description": "Search Datadog APM spans for a service.",
        "input_schema": {"type": "object",
                         "properties": {"service": {"type": "string"}},
                         "required": ["service"]},
    }]

    state = InvestigationState(alert_context=AlertContext(
        alert_text="acme-notes-api returning 500s", service_hint="acme-notes-api"))
    node = HypothesisNode(statement="acme-notes-api 500s from upstream error", depth=0)
    state.add_node(node)
    state.current_node_id = node.id

    llm = _RoutingLLM()
    node_fn = make_test_hypothesis_node(_fake_retrieve, llm, _dispatch, catalog)
    await node_fn(state)

    # The LLM selected + we dispatched the live tool with parsed args.
    assert dispatched == [("datadog_search_spans", {"service": "acme-notes-api"})]
    assert state.budget_state.llm_tool_select_calls == 1
    assert state.budget_state.tool_dispatch_calls == 1
    # Live evidence is attached with a `tool:` provenance tag...
    tool_cites = [c for c in node.evidence if c.source_id.startswith("tool:")]
    assert tool_cites, "live tool result was not attached as evidence"
    # ...and tagged with the service so the off-topic cap treats it on-topic.
    assert tool_cites[0].repo == "acme-notes-api"
    # The hypothesis validated on the live signal (off-topic cap did NOT fire).
    assert node.status == "validated"
    assert any(e.event_type == "tool_call" for e in state.agent_trace)


@pytest.mark.asyncio
async def test_test_hypothesis_is_rag_only_when_no_dispatch() -> None:
    # Backward compatibility: no tool_dispatch -> no tool-select call.
    state = InvestigationState(alert_context=AlertContext(
        alert_text="x", service_hint="acme-notes-api"))
    node = HypothesisNode(statement="hypothesis", depth=0)
    state.add_node(node)
    state.current_node_id = node.id

    llm = _RoutingLLM()
    node_fn = make_test_hypothesis_node(_fake_retrieve, llm)  # no dispatch
    await node_fn(state)

    assert state.budget_state.tool_dispatch_calls == 0
    assert state.budget_state.llm_tool_select_calls == 0
    assert not any(c.source_id.startswith("tool:") for c in node.evidence)


# --- E1: best-first frontier ---------------------------------------


@pytest.mark.asyncio
async def test_best_first_tests_all_pending_then_expands_strongest() -> None:
    state = InvestigationState(alert_context=AlertContext(alert_text="x"))
    strong = HypothesisNode(
        statement="A strong", depth=0, status="validated", confidence=0.9,
        evidence=[Citation(source_id="s1", chunk_id="c1", score=0.9),
                  Citation(source_id="s2", chunk_id="c2", score=0.9)],
    )
    weak = HypothesisNode(
        statement="B weak", depth=0, status="validated", confidence=0.9,
        evidence=[Citation(source_id="s3", chunk_id="c3", score=0.3)],
    )
    pending = HypothesisNode(statement="C pending", depth=0)
    for n in (strong, weak, pending):
        state.add_node(n)
    # Pretend we just validated A (old code would drill into A immediately).
    state.current_node_id = strong.id

    run = decide_next_node()

    # 1. A pending hypothesis still exists -> it MUST be tested before any
    #    expansion (no early commit to the first validated branch).
    await run(state)
    assert state.current_node_id == pending.id

    # 2. Frontier exhausted -> expand the STRONGER validated root first.
    pending.status = "inconclusive"
    await run(state)
    assert state.current_node_id == strong.id
    assert strong.expanded is True

    # 3. Next frontier pick is the other validated root -> both branches
    #    get explored (best-first), neither is starved.
    await run(state)
    assert state.current_node_id == weak.id
    assert weak.expanded is True
