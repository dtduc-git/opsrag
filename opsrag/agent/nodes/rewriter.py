"""Query rewriter node -- used when grading drops all retrieved chunks."""
from __future__ import annotations

from opsrag.agent.prompts import REWRITER_SYSTEM
from opsrag.interfaces.llm import LLMProvider
from opsrag.interfaces.observability import ObservabilityProvider


def rewrite_query_node(llm: LLMProvider, observability: ObservabilityProvider):
    async def _rewrite(state: dict) -> dict:
        original = state["query"]
        retries = state.get("retry_count", 0)

        try:
            response = await llm.generate(
                purpose="query-rewrite",
                messages=[
                    {
                        "role": "user",
                        "content": f"Original query: {original}\n\nRewrite it to improve retrieval.",
                    }
                ],
                system_prompt=REWRITER_SYSTEM,
                temperature=0.0,
            )
            rewritten = response.content.strip().strip('"').strip("'")
        except Exception:
            rewritten = original

        await observability.log_llm_call(
            messages=[{"role": "user", "content": original}],
            response=response if "response" in locals() else _dummy_response(),
            node_name="rewrite_query",
            purpose="query_rewrite",
        )

        return {
            "query": rewritten or original,
            "retry_count": retries + 1,
            "current_step": "rewritten",
        }

    return _rewrite


def _dummy_response():
    from opsrag.interfaces.llm import LLMResponse
    return LLMResponse(content="", model="none")
