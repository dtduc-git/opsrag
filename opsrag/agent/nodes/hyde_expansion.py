"""HyDE (Hypothetical Document Embeddings) retrieval expansion node -- T1.5.

Before vector retrieval, ask Flash to write a short hypothetical
answer to the user's query. The retriever then embeds THAT instead
of the raw query. The embedding lands closer to actual document
text in vector space, catching phrasing mismatches like:

    user:   "how to add a secret"
    docs:   "ExternalSecret + envFrom"

The node is a no-op when:

  * ``state.query_type == 'live'`` -- live-state lookups depend on the
    user's exact wording (timestamps, IDs, current values); HyDE
    paraphrasing actively hurts.
  * The query is shorter than 4 words -- too narrow to expand
    reliably; HyDE adds noise on terse lookups.
  * Flash returns an empty / errored response -- fail open, embed the
    raw query.

Reference: Gao et al. 2022 "Precise Zero-Shot Dense Retrieval Without
Relevance Labels" (HyDE).
"""
from __future__ import annotations

import logging

from opsrag.agent.prompts import HYDE_PROMPT
from opsrag.interfaces.llm import LLMProvider
from opsrag.interfaces.observability import ObservabilityProvider

_log = logging.getLogger("opsrag.agent.hyde_expansion")

# Minimum word count for HyDE to apply. Tuned conservatively:
# < 4 words is usually a name lookup, an ID, or a one-shot config
# query where paraphrasing isn't beneficial.
_MIN_WORDS_FOR_HYDE = 4


def hyde_expansion_node(
    llm: LLMProvider,
    observability: ObservabilityProvider | None = None,
):
    """Factory: returns the hyde_expansion LangGraph node."""

    async def _expand(state: dict) -> dict:
        query = (state.get("query") or "").strip()

        # Skip on live queries -- paraphrasing erases the temporal/ID signal we
        # need for retrieval scoping. The live verdict is the CLASSIFIER's, which
        # lands in `query_category` ("live"/"mixed"); the router's `query_type`
        # has no "live" member, so the old `query_type=="live"` check was dead.
        # "mixed" (live + forensic) is treated as live, matching the classifier.
        query_category = (state.get("query_category") or "").lower()
        if query_category in ("live", "mixed"):
            _log.debug("hyde: skipped (query_category=%s)", query_category)
            return {
                "hyde_text": None,
                "current_step": "hyde_skipped_live",
            }

        # Skip on config/identifier lookups. HyDE writes a plausible-but-wrong
        # hypothetical ("be wrong on specifics", per the prompt) -- great for
        # vocabulary-gap prose queries, actively harmful for an exact lookup like
        # "what replicas value for acme-notes-be in prod values.yaml": it pulls
        # the dense lane toward INVENTED YAML keys. BM25 saves the exact match,
        # but half the fusion is degraded. Skip when the router says config_lookup
        # or the query carries exact identifiers/anchors.
        if (state.get("query_type") or "") == "config_lookup":
            _log.debug("hyde: skipped (query_type=config_lookup)")
            return {"hyde_text": None, "current_step": "hyde_skipped_config"}
        from opsrag.agent.anchors import extract_anchors
        if extract_anchors(query):
            _log.debug("hyde: skipped (query carries anchors)")
            return {"hyde_text": None, "current_step": "hyde_skipped_anchors"}

        # Skip on short queries -- too narrow to expand without drift.
        if len(query.split()) < _MIN_WORDS_FOR_HYDE:
            _log.debug("hyde: skipped (query too short: %r)", query)
            return {
                "hyde_text": None,
                "current_step": "hyde_skipped_short",
            }

        if not query:
            return {
                "hyde_text": None,
                "current_step": "hyde_skipped_empty",
            }

        try:
            response = await llm.generate(
                purpose="hyde-expansion",
                messages=[{"role": "user", "content": f"Question: {query}"}],
                system_prompt=HYDE_PROMPT,
                temperature=0.0,
                max_tokens=400,
            )
            hypothetical = (response.content or "").strip()
        except Exception as exc:
            _log.warning("hyde llm failed: %s", exc)
            hypothetical = ""

        if not hypothetical:
            # Empty response -> behave as if HyDE was off.
            return {
                "hyde_text": None,
                "current_step": "hyde_empty",
            }

        # Anchor the embedding with the original query keywords too so
        # we don't drift entirely into the hypothetical's vocabulary.
        # This is a small but consistent win in HyDE replications.
        embed_text = f"{query}\n\n{hypothetical}"

        _log.info(
            "hyde: applied (query_words=%d, hyde_chars=%d)",
            len(query.split()), len(hypothetical),
        )

        if observability is not None:
            try:
                await observability.log_llm_call(
                    messages=[{"role": "user", "content": query[:500]}],
                    response=None,
                    node_name="hyde_expansion",
                    purpose="hyde_expansion",
                )
            except Exception:
                # Observability errors must never break the graph.
                pass

        return {
            "hyde_text": embed_text,
            "current_step": "hyde_expanded",
        }

    return _expand
