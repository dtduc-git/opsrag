"""Insufficient-information honest-answer node -- Phase 2 Step 3 (ADR-004 CRAG).

Per Yan et al. 2024 (Corrective RAG): when chunk grading + re-query exhausts
without surfacing relevant chunks, emit an honest "insufficient information"
answer rather than fabricating.

The existing OpsRAG pipeline (build_full_graph, build_hybrid_graph) already
implements the CRAG-style grade -> rewrite -> re-retrieve loop with
`max_retries`. This node fills the missing piece: when max_retries is
exhausted AND chunks are still insufficient, generate a brief honest
non-answer instead of generating with bad chunks.

The decision branch in `grade_decision` returns `"insufficient_info"` for
this state. Graph builders route that to this node, bypassing the
expensive LLM generation step and the downstream verifier (which would
have nothing to verify against).
"""
from __future__ import annotations

import logging

from opsrag.interfaces.observability import ObservabilityProvider

_log = logging.getLogger("opsrag.agent.insufficient_info")

_FALLBACK_ANSWER = (
    "I cannot find relevant information in the OpsRAG knowledge base "
    "(Confluence SRE space, Slack #devops, Rootly incidents, indexed "
    "git repos) to answer this question. The query was reformulated and "
    "re-retrieved without surfacing supporting context.\n\n"
    "Suggestions:\n"
    "- Rephrase the query with specific service names, file paths, or "
    "incident IDs you remember.\n"
    "- Check the source systems directly: the Confluence wiki, "
    "`#devops` Slack channel history, or Rootly dashboard.\n"
    "- If this is a new policy or runbook that should be in the knowledge "
    "base, please flag for indexing."
)


def insufficient_info_node(observability: ObservabilityProvider):
    """LangGraph node that emits an honest "insufficient information" answer
    when chunk grading + re-query has exhausted without finding relevant
    context. Skips both LLM generation and grounding verification (nothing
    to ground against).
    """

    async def _emit(state: dict) -> dict:
        query = state.get("query", "")
        anchors = state.get("anchors") or []
        # When the user named entities AND we routed here via the rerank
        # gate (weak retrieval despite candidates existing), produce an
        # answer specific to the missing entity instead of the generic
        # "nothing found" message.
        if anchors and state.get("merged_results"):
            entities = ", ".join(f"`{a}`" for a in anchors)
            generation = (
                f"I don't have specific information about {entities} in the "
                f"OpsRAG knowledge base. Retrieval returned related "
                f"documents but none whose source path or repository "
                f"matches the entity you named, so I can't answer "
                f"reliably from the indexed corpus.\n\n"
                f"Try:\n"
                f"- Confirm the spelling and that the entity is named "
                f"the same way in the repo / docs as in your question.\n"
                f"- If it's a GitLab repo, check the indexer source list -- "
                f"the repo may not be indexed yet.\n"
                f"- Browse the related context surfaced by retrieval "
                f"directly (Confluence / GitLab) to see if it links to "
                f"the entity you're looking for."
            )
            _log.info(
                "emitting entity-not-found insufficient_info for query=%r anchors=%s",
                query[:80], anchors,
            )
        else:
            generation = _FALLBACK_ANSWER
            _log.info("emitting insufficient_info fallback for query=%r", query[:80])
        return {
            "generation": generation,
            "generation_grounded": True,  # vacuously: no claims to fail grounding
            "current_step": "insufficient_info_emitted",
        }

    return _emit
