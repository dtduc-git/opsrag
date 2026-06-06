"""Console-based observability -- Phase 1 stub for local dev.

Phoenix/Arize integration lands in Phase 4.
"""
from __future__ import annotations

import logging
from typing import Any

from opsrag.interfaces.graphstore import GraphSearchResult
from opsrag.interfaces.llm import LLMResponse
from opsrag.interfaces.vectorstore import SearchResult

_log = logging.getLogger("opsrag.obs")


class ConsoleObservability:
    def __init__(self, level: int = logging.INFO):
        self._project_name = "opsrag"
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )

    def setup(self, project_name: str) -> None:
        self._project_name = project_name
        _log.info("console observability active for project=%s", project_name)

    def get_tracer(self) -> Any:
        return _log

    async def log_retrieval(
        self,
        query: str,
        results: list[SearchResult],
        graph_results: GraphSearchResult | None,
        latency_ms: float,
        node_name: str,
    ) -> None:
        _log.info(
            "[%s] retrieval q=%r hits=%d graph_entities=%d latency=%.1fms",
            node_name,
            query[:80],
            len(results),
            len(graph_results.entities) if graph_results else 0,
            latency_ms,
        )

    async def log_llm_call(
        self,
        messages: list[dict],
        response: LLMResponse,
        node_name: str,
        purpose: str,
    ) -> None:
        # Token recording happens at the LLM provider layer
        # (vertex.py / anthropic.py / bedrock.py) so each call is
        # counted exactly once with the correct `purpose`. This
        # observer hook is for log streams + Phoenix spans only.
        _log.info(
            "[%s] llm purpose=%s model=%s tokens=%s latency=%.1fms",
            node_name,
            purpose,
            response.model,
            response.usage,
            response.latency_ms,
        )

    async def run_response_evals(
        self,
        query: str,
        response: str,
        context: str,
        eval_types: list[str],
    ) -> dict:
        return {kind: None for kind in eval_types}
