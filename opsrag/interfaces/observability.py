"""Observability provider interface."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from opsrag.interfaces.graphstore import GraphSearchResult
from opsrag.interfaces.llm import LLMResponse
from opsrag.interfaces.vectorstore import SearchResult


@runtime_checkable
class ObservabilityProvider(Protocol):
    def setup(self, project_name: str) -> None: ...
    def get_tracer(self) -> Any: ...

    async def log_retrieval(
        self,
        query: str,
        results: list[SearchResult],
        graph_results: GraphSearchResult | None,
        latency_ms: float,
        node_name: str,
    ) -> None: ...

    async def log_llm_call(
        self,
        messages: list[dict],
        response: LLMResponse,
        node_name: str,
        purpose: str,
    ) -> None: ...

    async def run_response_evals(
        self,
        query: str,
        response: str,
        context: str,
        eval_types: list[str],
    ) -> dict:
        """Run LLM-as-judge quality checks (faithfulness, relevance, etc.)."""
        ...
