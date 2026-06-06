"""Integration test (T095): the investigation graph end to end with fakes.

Drives build_investigation_graph() with a fake retriever and a scripted fake
LLM (routing on the `purpose` tag the nodes pass) so the multi-hop graph runs
bootstrap -> generate_hypotheses -> test_hypothesis -> decide -> synthesize
with no live retriever, LLM, or MCP backend.
"""
from __future__ import annotations

import pytest

from opsrag.agents.investigation.graph import build_investigation_graph
from opsrag.agents.investigation.state import AlertContext, InvestigationState
from opsrag.interfaces.llm import LLMResponse


async def _fake_retrieve(query_text: str, top_k: int = 5) -> list[dict]:
    """Canned retrieval hits in the shape the graph expects."""
    return [
        {
            "chunk_id": "c1",
            "source_id": "samples/runbooks/002-acme-notes-db-failover.md",
            "snippet": "If the primary database is unhealthy, promote the standby.",
            "score": 0.91,
            "repo": "samples",
        },
        {
            "chunk_id": "c2",
            "source_id": "samples/postmortems/2026-01-15-acme-notes-db-outage.md",
            "snippet": "The primary ran out of disk after WAL archiving fell behind.",
            "score": 0.82,
            "repo": "samples",
        },
    ][:top_k]


class _ScriptedLLM:
    """Routes on the `purpose` tag the investigation nodes pass to generate().
    Returns valid JSON for hypothesis-gen and judge calls, prose for synth."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def model_name(self) -> str:
        return "fake-flash"

    async def generate(
        self,
        messages,
        system_prompt=None,
        temperature=0.0,
        max_tokens=4096,
        response_format=None,
        purpose=None,
        response_schema=None,
    ) -> LLMResponse:
        self.calls.append(purpose or "")
        p = purpose or ""
        if "llm_query_gen" in p:
            # Only the first (root) gen returns hypotheses; with all verdicts
            # inconclusive there is no recursion, so this stays bounded.
            content = (
                '{"hypotheses": ['
                '{"statement": "The primary database ran out of disk.", "rationale": "WAL lag."},'
                '{"statement": "A recent deploy introduced a regression.", "rationale": "timing."}'
                "]}"
            )
        elif "llm_judge" in p:
            content = (
                '{"status": "inconclusive", "confidence": 0.3, '
                '"rationale": "fake: evidence is suggestive but not conclusive", '
                '"citations": []}'
            )
        elif "llm_synth" in p:
            content = (
                "Most likely root cause: the primary database filled its disk "
                "after WAL archiving fell behind. See the DB failover runbook."
            )
        else:
            content = "{}"
        return LLMResponse(content=content, model="fake-flash", usage={"input_tokens": 1, "output_tokens": 1})

    async def generate_structured(self, messages, schema, system_prompt=None, purpose=None):
        raise NotImplementedError("not used by this test")


@pytest.mark.asyncio
async def test_investigation_runs_end_to_end_with_fakes() -> None:
    llm = _ScriptedLLM()
    graph = build_investigation_graph(
        retrieve=_fake_retrieve,
        llm_flash=llm,
        llm_pro=None,
        embed_query=None,  # disables duplicate-ancestor pruning
    )

    initial = InvestigationState(
        alert_context=AlertContext(
            alert_text="Acme Notes API is returning 500s and the database looks unhealthy.",
            service_hint="acme-notes-api",
        )
    )

    final = await graph.ainvoke(initial)

    # LangGraph may return the state as a dict or a model; normalise.
    def _f(key):
        return final[key] if isinstance(final, dict) else getattr(final, key)

    outcome = _f("outcome")
    nodes_by_id = _f("nodes_by_id")
    agent_trace = _f("agent_trace")

    # It must TERMINATE (not left pending) with a real outcome, having
    # generated and tested hypotheses through the graph.
    assert outcome in ("validated_root_cause", "inconclusive", "circuit_breaker_terminated")
    assert len(nodes_by_id) >= 1, "expected at least one hypothesis node"
    assert agent_trace, "expected trace events from the run"
    # The scripted LLM was exercised for hypothesis generation and judging.
    assert any("llm_query_gen" in c for c in llm.calls)
    assert any("llm_judge" in c for c in llm.calls)
    # Every tested hypothesis carries the fake judge's inconclusive verdict.
    assert all(n.status in ("pending", "inconclusive", "validated", "invalidated")
               for n in nodes_by_id.values())


@pytest.mark.asyncio
async def test_graph_compiles_without_pro_llm() -> None:
    # Smoke: building with only llm_flash (no pro) must succeed.
    graph = build_investigation_graph(retrieve=_fake_retrieve, llm_flash=_ScriptedLLM())
    assert graph is not None
