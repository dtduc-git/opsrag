"""Query rewriter node -- used when grading drops all retrieved chunks."""
from __future__ import annotations

from opsrag.agent.anchors import extract_anchors
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

        rewritten = rewritten or original
        # Re-inject the original's exact identifiers (service slugs, filenames,
        # error strings like CrashLoopBackOff, ticket IDs) that the LLM rewrite
        # may have paraphrased away. The rewrite fires precisely when retrieval
        # already failed, so losing the literal tokens the BM25 lexical lane
        # depends on would make the retry strictly worse. Append any anchor not
        # already present so the lexical lane still matches.
        anchors = extract_anchors(original)
        low = rewritten.lower()
        missing = [a for a in anchors if a.lower() not in low]
        if missing:
            rewritten = f"{rewritten} {' '.join(missing)}"

        await observability.log_llm_call(
            messages=[{"role": "user", "content": original}],
            response=response if "response" in locals() else _dummy_response(),
            node_name="rewrite_query",
            purpose="query_rewrite",
        )

        return {
            "query": rewritten,
            # Clear the stale HyDE expansion of the ORIGINAL query. The retry
            # edge goes straight to vector_retrieve (skipping hyde_expansion), so
            # without this the dense lane would embed the old query's hypothetical
            # while BM25 uses the rewritten query -- a lane mismatch.
            "hyde_text": None,
            "retry_count": retries + 1,
            "current_step": "rewritten",
        }

    return _rewrite


def _dummy_response():
    from opsrag.interfaces.llm import LLMResponse
    return LLMResponse(content="", model="none")
