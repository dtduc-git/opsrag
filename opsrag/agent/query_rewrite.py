"""Coreference / follow-up query rewriting.

When a user asks a follow-up question like "tell me more about this repo"
or "what about its config?", the bare query has no hope of matching the
right chunks via vector retrieval -- it has no entities, just pronouns.

This module rewrites such queries into a standalone form using the last
N turns from the session store. The rewritten query is what gets:

1. Embedded and looked up in the Q&A cache.
2. Fed to the agent graph (retriever, router, generator).

Heuristic gating: we only call the LLM when the query LOOKS like a
follow-up (short, has pronouns / referential phrases, lacks named
entities). Single-shot queries pass through unchanged -- rewriting them
is wasted latency and risks the LLM making the query worse.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from opsrag.agent.anchors import extract_anchors

_log = logging.getLogger("opsrag.query_rewrite")

# Pronouns and bare referential phrases that almost always need a
# previous turn's context to resolve.
_REF_PATTERNS = re.compile(
    r"\b("
    r"this|that|these|those|it|its|they|them|their"
    r"|the\s+(?:repo|repository|service|file|path|chart|config|value|env"
    r"|environment|var|variable|secret|alert|error|incident|deploy|deployment)"
    r"|(?:tell|show|give|explain)\s+me\s+more"
    r"|more\s+(?:about|on|details)"
    r"|what\s+about"
    r"|how\s+about"
    r"|and\s+(?:for|in|the)"
    r"|same\s+(?:for|in|but)"
    r"|continue|go\s+on"
    r")\b",
    re.IGNORECASE,
)

# Markers that strongly suggest a self-contained question -- skip rewrite.
# A query mentioning a service slug, an env, a file path, or a question
# phrasing with explicit nouns usually doesn't need history to resolve.
_SELF_CONTAINED_PATTERNS = re.compile(
    r"\b("
    r"all\s+(?:repos?|repositories|services)"
    r"|which\s+(?:repos?|repositories|services)"
    r")\b",
    re.IGNORECASE,
)


def looks_like_followup(query: str) -> bool:
    """Heuristic: True when query likely needs prior-turn context.

    We rewrite only when:
      - The query is short (< 12 words) AND contains a referential phrase, OR
      - The query opens with a referential phrase regardless of length.
    Plural-intent / cross-repo queries are explicitly self-contained and skipped.
    """
    q = query.strip()
    if not q:
        return False
    if _SELF_CONTAINED_PATTERNS.search(q):
        return False
    word_count = len(q.split())
    has_ref = bool(_REF_PATTERNS.search(q))
    if not has_ref:
        return False
    if word_count < 12:
        return True
    # Long query but starts with a clear follow-up cue.
    return bool(_REF_PATTERNS.match(q))


_REWRITE_SYSTEM = """You rewrite follow-up DevOps/SRE questions into standalone, fully-formed queries.

Given the recent conversation and the latest user message, output ONE
complete sentence that captures the user's intent without pronouns or
implicit context. The output MUST be a complete, grammatical question
or imperative -- never a fragment.

Rules:
1. Resolve every pronoun ("this", "it", "the service", "they") to a
   named entity from the most recent assistant turn. If the prior turn
   named multiple entities, pick the one mentioned FIRST or as the
   PRIMARY answer.
2. Preserve technical identifiers verbatim (service names, repo names,
   environment names, file paths, error codes, alert names).
3. Do NOT add new constraints, subtopics, or filters the user didn't ask
   about. Keep the user's intent intact.
4. Always produce a complete query. Never output a fragment like "Tell
   me more about" without naming the entity.

Pattern examples (placeholders <repo-a>, <svc-a>, <env-a> stand in for
the actual identifiers your deployment uses; rewrite using whatever
identifiers appeared in the prior turn):

[user] Which repo handles deployment configuration?
[assistant] Deployment config is handled primarily by `<repo-a>`
and also by `<repo-b>` for app definitions.
[user - current message] tell me more about this repo
-> Tell me more about the <repo-a> repository, including its purpose,
structure, and how it handles deployment configuration.

[user] What is the <svc-a> service?
[assistant] <svc-a> is an ingress service.
[user - current message] how do I onboard it to <env-a>?
-> How do I onboard the <svc-a> service to the <env-a> environment?

[user] Show me <svc-a>'s <env-a> config.
[assistant] <svc-a>'s <env-a> config lives in `<repo-a>/values/<svc-a>/<env-a>.yaml`.
[user - current message] and the secrets?
-> Where are the secrets defined for <svc-a> in the <env-a> environment,
and how are they wired into `<repo-a>/values/<svc-a>/<env-a>.yaml`?

Output ONLY the rewritten query text. No quotes, no preamble, no
"Rewritten:" prefix, no commentary, no trailing punctuation other than
a single question mark or period."""


def _build_rewrite_prompt(prior_messages: list[dict], current_query: str) -> str:
    """Compose the user-side prompt: last 2 turns + current query."""
    parts: list[str] = []
    # Take only the last 2 (user, assistant) pairs = up to 4 messages.
    tail = prior_messages[-4:] if len(prior_messages) > 4 else prior_messages
    for m in tail:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        # Trim long assistant turns -- only the most recent ~600 chars
        # are typically relevant for entity resolution.
        if role == "assistant" and len(content) > 600:
            content = content[:600] + "..."
        parts.append(f"[{role}]\n{content}")
    parts.append(f"[user - current message]\n{current_query.strip()}")
    parts.append("\nRewrite the current message as a standalone query.")
    return "\n\n".join(parts)


async def maybe_rewrite_query(
    *,
    query: str,
    prior_messages: list[dict] | None,
    llm: Any,
) -> str:
    """Return a rewritten standalone query if `query` looks like a follow-up
    and there's at least one prior turn to anchor on. Otherwise return the
    original query unchanged.

    Failure modes (LLM error, empty response) fall back to the original
    query -- rewriting is a recall booster, never load-bearing for correctness.
    """
    if not prior_messages:
        return query
    if not looks_like_followup(query):
        return query
    if llm is None:
        return query

    try:
        prompt = _build_rewrite_prompt(prior_messages, query)
        # gemini-2.5-flash uses 'thinking' tokens by default -- they
        # consume max_output_tokens before any visible text is emitted.
        # Setting this to 200 produced 8-token truncated outputs in
        # practice. 2000 leaves room for ~150-token thinking + the
        # short rewritten query (~50 tokens).
        resp = await llm.generate(
            purpose="query-rewrite",
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_REWRITE_SYSTEM,
            temperature=0.0,
            max_tokens=2000,
        )
        rewritten = (resp.content or "").strip()
        # Strip any surrounding quotes the model occasionally adds.
        if rewritten.startswith(("'", '"')) and rewritten.endswith(("'", '"')):
            rewritten = rewritten[1:-1].strip()
        # Strip a leading "Rewritten:" / "Standalone query:" preface.
        rewritten = re.sub(
            r"^(rewritten|standalone\s+query|query|output)\s*[:\---]\s*",
            "", rewritten, flags=re.IGNORECASE,
        ).strip()
        if not rewritten or len(rewritten) > 500:
            return query
        # Reject fragments -- the rewrite must be a complete query of at
        # least 5 words, longer than the original (it's resolving things,
        # not shortening), and ending in a sentence-ish way.
        if len(rewritten.split()) < 5:
            _log.warning("rewrite rejected (fragment): %r", rewritten)
            return query
        if len(rewritten) < len(query.strip()) * 0.8:
            _log.warning(
                "rewrite rejected (shorter than original): %r", rewritten
            )
            return query
        if rewritten.lower() == query.strip().lower():
            return query
        # Re-inject any exact identifier the LLM paraphrased away. Crucially this
        # spans the CURRENT follow-up AND the recent prior turns: the entity
        # behind "it"/"its config" usually lives in the prior turn, not the
        # follow-up ("tell me more about its config" has no anchors), and the LLM
        # may resolve "it" -> "the notes service" instead of the literal slug,
        # silently dropping the token the BM25/anchor lanes need. extract_anchors
        # only grabs identifier-shaped tokens so this stays focused; cap the
        # re-injection so a multi-entity history can't flood the query.
        # Current-follow-up anchors are always relevant. Prior-turn anchors
        # resolve "it"/"its" but risk dragging in a stale sub-topic ("and the
        # secrets?" shouldn't re-pull last turn's unrelated entities), so take
        # them only from the IMMEDIATELY prior turn and cap the total tight,
        # current-query anchors first.
        own = list(extract_anchors(query))
        prior_anchors: list[str] = []
        for m in (prior_messages[-2:] if prior_messages else []):
            prior_anchors.extend(extract_anchors(m.get("content") or ""))
        seen_a: set[str] = set()
        anchors: list[str] = []
        for a in own + prior_anchors:
            if a.lower() not in seen_a:
                seen_a.add(a.lower())
                anchors.append(a)
        low = rewritten.lower()
        missing = [a for a in anchors if a.lower() not in low][:2]
        if missing:
            rewritten = f"{rewritten} {' '.join(missing)}"
        _log.info(
            "query rewritten: %r -> %r", query.strip()[:80], rewritten[:160]
        )
        return rewritten
    except Exception as exc:
        _log.warning("query rewrite failed; using original: %s", exc)
        return query
