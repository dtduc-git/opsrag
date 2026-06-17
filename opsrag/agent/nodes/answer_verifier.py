"""Code-grounded answer verifier node (T1.2).

Runs AFTER the generator produces an answer. Asks Flash to:
  1. Extract every file path / YAML-key path / CRD name claimed in the
     answer.
  2. For each, decide whether it appears in the cited evidence chunks.
  3. Return a JSON verdict ``{"verified": [...], "unverifiable": [...]}``.

If any claim is unverifiable, we prepend a single-line hedge to the
answer (we do NOT silently strip lines -- the engineer should see what
the agent doubted). On any error (LLM failure, malformed JSON) we FAIL
CLOSED: we append a caution to the answer rather than presenting an
unverified answer as clean (we never silently pass it through).

Why Flash and not Pro: this is a constrained extract+match task, low
ambiguity, and we already pay Pro for the generator + hallucination
check. Flash keeps the per-query latency budget intact (~600ms).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from opsrag.agent.prompts import ANSWER_VERIFIER_PROMPT
from opsrag.interfaces.chunker import Chunk
from opsrag.interfaces.llm import LLMProvider
from opsrag.interfaces.observability import ObservabilityProvider
from opsrag.interfaces.vectorstore import VectorStore

_log = logging.getLogger("opsrag.agent.answer_verifier")

# Soft cap on context size sent to the verifier -- full_chunks can be
# tens of KB and we only need enough to match path/key tokens.
_MAX_EVIDENCE_CHARS = 16_000
_MAX_ANSWER_CHARS = 8_000


def _build_evidence_block(chunks: list[Chunk]) -> str:
    """Concatenate cited evidence into a single text block, capped."""
    if not chunks:
        return "(no evidence chunks)"
    parts: list[str] = []
    total = 0
    for c in chunks:
        src = getattr(c, "source_path", "") or ""
        repo = getattr(c, "repo", "") or ""
        header = f"[Source: {repo}/{src}]" if repo else f"[Source: {src}]"
        body = (getattr(c, "content", "") or "")
        snippet = f"{header}\n{body}"
        if total + len(snippet) > _MAX_EVIDENCE_CHARS:
            remaining = _MAX_EVIDENCE_CHARS - total
            if remaining > 200:
                parts.append(snippet[:remaining])
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n\n---\n\n".join(parts)


# Strip Markdown / JSON fences from the LLM verdict before json.loads.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.DOTALL)


def _parse_verdict(raw: str) -> dict[str, list[str]] | None:
    """Best-effort JSON extraction. Returns None on malformed input."""
    if not raw or not raw.strip():
        return None
    text = _FENCE_RE.sub("", raw.strip())
    # If the model surrounded JSON with prose, snip to the outermost braces.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    verified = obj.get("verified") or []
    unverifiable = obj.get("unverifiable") or []
    if not isinstance(verified, list) or not isinstance(unverifiable, list):
        return None
    # Coerce items to str so downstream formatting stays safe.
    return {
        "verified": [str(x) for x in verified if x is not None],
        "unverifiable": [str(x) for x in unverifiable if x is not None],
    }


# Fail-closed caution appended when the verifier itself could not run (LLM
# error or malformed verdict). We can't confirm the concrete claims, so we tell
# the engineer rather than silently presenting an unverified answer as clean.
_CAUTION = (
    "\n\n_Note: I could not verify the concrete file paths / keys / resource "
    "names in this answer against the corpus (verification step failed). "
    "Double-check anything load-bearing before acting on it._"
)


def _format_hedge(unverifiable: list[str]) -> str:
    """Render the user-facing hedge prefix."""
    # Cap list shown inline so the hedge stays readable.
    shown = unverifiable[:5]
    overflow = len(unverifiable) - len(shown)
    items = ", ".join(f"`{x}`" for x in shown)
    if overflow > 0:
        items += f" (+{overflow} more)"
    return (
        f"Warning: Some claims could not be verified against the corpus: "
        f"{items}. Treat with caution.\n\n"
    )


def _format_memory_evidence(mems: list | None) -> str:
    """Render recalled per-user memories as verifier evidence lines."""
    if not mems:
        return ""
    lines: list[str] = []
    for m in mems[:8]:
        text = ""
        val = getattr(m, "value", None)
        if isinstance(val, dict):
            text = val.get("memory") or val.get("text") or ""
        elif isinstance(m, dict):
            text = m.get("memory") or m.get("text") or ""
        elif isinstance(m, str):
            text = m
        text = (text or "").strip()
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines)


# Soft cap on live-tool evidence so a verbose tool dump can't crowd out the
# doc-chunk evidence block (which is already capped at _MAX_EVIDENCE_CHARS).
_MAX_TOOL_EVIDENCE_CHARS = 6_000


def _format_tool_evidence(
    audit: list | None, history: list | None
) -> str:
    """Render live MCP tool calls + their returned payloads as verifier
    evidence.

    On the multi_agent tool path the answer is grounded in what live tools
    returned (k8s pod state, prometheus alerts, a fetched Slack thread, ...),
    NOT in retrieved doc chunks -- ``final_chunks`` is often empty for a
    tool-only turn. Without surfacing the tool evidence here, the verifier
    sees "(no evidence chunks)" and spuriously flags every concrete fact the
    tools legitimately surfaced as unverifiable -> a jarring "treat with
    caution" on a correct, tool-grounded answer.

    We render BOTH:
      * the audit rows (which tools actually fired + with what args), so the
        verifier can confirm a cited tool name was really called; and
      * the ``tool_result`` payloads from ``tool_message_history`` (the text
        the tool returned), so artifact/value claims can be matched.

    Capped at ``_MAX_TOOL_EVIDENCE_CHARS`` so a noisy tool dump can't crowd
    out the doc-chunk evidence.
    """
    lines: list[str] = []

    # 1. Which tools fired (audit). Skip errored rows -- a tool that errored
    #    did not surface a fact the answer can lean on.
    called: list[str] = []
    for a in (audit or []):
        if not isinstance(a, dict):
            continue
        name = a.get("name")
        if not name or a.get("error"):
            continue
        args = a.get("args") or {}
        if args:
            called.append(f"- {name}({json.dumps(args, default=str)[:200]})")
        else:
            called.append(f"- {name}()")
    if called:
        lines.append("Tools called (succeeded):")
        lines.extend(called)

    # 2. What the tools returned (tool_result payloads from the history).
    payloads: list[str] = []
    for msg in (history or []):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "tool_result":
            continue
        name = msg.get("name") or "tool"
        resp = msg.get("response")
        if isinstance(resp, dict):
            if resp.get("error"):
                continue  # errored result is not evidence
            text = resp.get("text")
            if text is None:
                text = json.dumps(resp, default=str)
        else:
            text = str(resp) if resp is not None else ""
        text = (text or "").strip()
        if text:
            payloads.append(f"[{name}] {text}")
    if payloads:
        lines.append("\nTool results:")
        lines.append("\n\n".join(payloads))

    block = "\n".join(lines).strip()
    if not block:
        return ""
    if len(block) > _MAX_TOOL_EVIDENCE_CHARS:
        block = block[:_MAX_TOOL_EVIDENCE_CHARS]
    return block


def verify_answer_node(
    llm: LLMProvider,
    vector_store: VectorStore | None = None,
    observability: ObservabilityProvider | None = None,
):
    """Factory: returns the verify_answer LangGraph node.

    Args:
        llm: Flash-tier LLM provider.
        vector_store: reserved for future use (e.g. cross-corpus
            fallback lookup). Currently we only verify against the
            evidence the generator already saw.
        observability: optional; logs the verdict for offline eval.
    """

    async def _verify(state: dict) -> dict:
        answer = state.get("generation", "") or ""
        chunks: list[Chunk] = (
            state.get("final_chunks")
            or state.get("graded_chunks")
            or state.get("merged_results")
            or state.get("retrieved_chunks")
            or []
        )

        # Nothing to verify -- skip cleanly.
        if not answer.strip():
            return {
                "verification_result": {"skipped": True, "reason": "empty_answer"},
                "current_step": "verified",
            }

        evidence_block = _build_evidence_block(chunks)
        # Per-user memories are valid evidence too -- a recalled fact ("you own
        # the payments service") is grounded in the user's own history, not the
        # doc corpus. Without this the verifier flags legitimate memory recall
        # as "unverified" and the answer gets a jarring "treat with caution".
        mem_facts = _format_memory_evidence(state.get("user_memories"))
        if mem_facts:
            evidence_block = (
                f"{evidence_block}\n\n"
                "=== Known facts about this user (valid evidence from past "
                f"conversations) ===\n{mem_facts}"
            )
        # Live-tool evidence: on the multi_agent tool path the answer is
        # grounded in what live MCP tools returned, not in doc chunks (which
        # are often empty for a tool-only turn). Feed both the audit (which
        # tools fired) and the tool_result payloads so tool-grounded facts
        # aren't spuriously flagged unverifiable.
        tool_facts = _format_tool_evidence(
            state.get("tool_call_audit"),
            state.get("tool_message_history"),
        )
        if tool_facts:
            evidence_block = (
                f"{evidence_block}\n\n"
                "=== Live tool calls + results (valid evidence -- the answer "
                f"may legitimately cite these) ===\n{tool_facts}"
            )
        truncated_answer = answer[:_MAX_ANSWER_CHARS]

        user_msg = (
            f"Evidence chunks:\n{evidence_block}\n\n"
            f"Answer to verify:\n{truncated_answer}\n\n"
            "Return ONLY the JSON object."
        )

        verdict: dict[str, list[str]] | None = None
        try:
            response = await llm.generate(
                purpose="answer-verify",
                messages=[{"role": "user", "content": user_msg}],
                system_prompt=ANSWER_VERIFIER_PROMPT,
                temperature=0.0,
                max_tokens=1024,
            )
            verdict = _parse_verdict(response.content)
        except Exception as exc:  # network / LLM failure -> fail closed (verdict None)
            _log.warning("answer_verifier llm failed: %s", exc)
            verdict = None

        if verdict is None:
            # Fail-CLOSED: we could not verify the answer's concrete claims, so
            # append a caution rather than presenting an unverified answer as
            # clean. (Previously this passed the answer through unchanged.)
            return {
                "verification_result": {
                    "skipped": True,
                    "reason": "malformed_or_error",
                    "fail_closed": True,
                },
                "generation": answer + _CAUTION,
                "current_step": "verified",
            }

        unverifiable = verdict.get("unverifiable") or []
        verified = verdict.get("verified") or []

        result: dict[str, Any] = {
            "verification_result": {
                "verified": verified,
                "unverifiable": unverifiable,
                "skipped": False,
            },
            "current_step": "verified",
        }

        if unverifiable:
            hedged = _format_hedge(unverifiable) + answer
            result["generation"] = hedged
            _log.info(
                "answer_verifier: %d unverifiable claim(s); hedge prepended",
                len(unverifiable),
            )
        else:
            _log.debug(
                "answer_verifier: all %d claim(s) grounded", len(verified)
            )

        if observability is not None:
            try:
                await observability.log_llm_call(
                    messages=[{"role": "user", "content": user_msg[:2000]}],
                    response=None,
                    node_name="verify_answer",
                    purpose="answer_verification",
                )
            except Exception:
                # Observability errors must never break the graph.
                pass

        return result

    return _verify
