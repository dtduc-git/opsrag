"""Phoenix AI (Arize) observability provider.

Uses OpenInference auto-instrumentation to capture LangGraph node transitions,
LLM calls, and retrieval metrics. Sends traces via OTLP to a Phoenix collector.

Requires: arize-phoenix-otel, openinference-instrumentation-langchain
"""
from __future__ import annotations

import logging
import os
from typing import Any

from opsrag.interfaces.graphstore import GraphSearchResult
from opsrag.interfaces.llm import LLMResponse
from opsrag.interfaces.vectorstore import SearchResult

_log = logging.getLogger("opsrag.obs.phoenix")


class PhoenixObservability:
    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
    ):
        self._endpoint = endpoint
        self._api_key = api_key
        self._project_name = "opsrag"
        self._tracer: Any = None
        self._tracer_provider: Any = None

    def setup(self, project_name: str) -> None:
        self._project_name = project_name

        if self._endpoint:
            os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", self._endpoint)
        if self._api_key:
            existing = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")
            if "api_key" not in existing:
                headers = f"api_key={self._api_key}"
                if existing:
                    headers = f"{existing},{headers}"
                os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = headers

        try:
            from phoenix.otel import register

            self._tracer_provider = register(
                project_name=project_name,
                auto_instrument=True,
            )
            _log.info(
                "Phoenix observability active for project=%s endpoint=%s",
                project_name,
                self._endpoint or "default",
            )
        except ImportError:
            _log.warning(
                "phoenix.otel not installed -- falling back to no-op tracing. "
                "Install arize-phoenix-otel for full observability."
            )
        except Exception as exc:
            _log.warning("Phoenix setup failed: %s -- continuing without tracing", exc)

    def get_tracer(self) -> Any:
        if self._tracer is not None:
            return self._tracer
        try:
            from opentelemetry import trace
            self._tracer = trace.get_tracer("opsrag", "0.1.0")
        except ImportError:
            self._tracer = _log
        return self._tracer

    async def log_retrieval(
        self,
        query: str,
        results: list[SearchResult],
        graph_results: GraphSearchResult | None,
        latency_ms: float,
        node_name: str,
    ) -> None:
        tracer = self.get_tracer()
        if hasattr(tracer, "start_as_current_span"):
            with tracer.start_as_current_span(
                f"retrieval.{node_name}",
                attributes={
                    "opsrag.query": query[:200],
                    "opsrag.node": node_name,
                    "opsrag.result_count": len(results),
                    "opsrag.graph_entity_count": len(graph_results.entities) if graph_results else 0,
                    "opsrag.latency_ms": latency_ms,
                },
            ):
                pass
        else:
            _log.info(
                "[%s] retrieval q=%r hits=%d latency=%.1fms",
                node_name, query[:80], len(results), latency_ms,
            )

    async def log_llm_call(
        self,
        messages: list[dict],
        response: LLMResponse,
        node_name: str,
        purpose: str,
    ) -> None:
        # Token recording happens in the LLM provider layer; this hook is
        # for span emission + log streams only. See console.py for context.
        tracer = self.get_tracer()
        if hasattr(tracer, "start_as_current_span"):
            with tracer.start_as_current_span(
                f"llm.{node_name}.{purpose}",
                attributes={
                    "opsrag.node": node_name,
                    "opsrag.purpose": purpose,
                    "opsrag.model": response.model,
                    "opsrag.input_tokens": response.usage.get("input_tokens", 0),
                    "opsrag.output_tokens": response.usage.get("output_tokens", 0),
                    "opsrag.latency_ms": response.latency_ms,
                },
            ):
                pass
        else:
            _log.info(
                "[%s] llm purpose=%s model=%s latency=%.1fms",
                node_name, purpose, response.model, response.latency_ms,
            )

    async def run_response_evals(
        self,
        query: str,
        response: str,
        context: str,
        eval_types: list[str],
    ) -> dict:
        """Run LLM-as-judge quality checks via Phoenix.

        Full wiring is deployment-specific; this provides the hook point.
        Returns None for each type when Phoenix checking is not configured.
        """
        return {et: None for et in eval_types}
