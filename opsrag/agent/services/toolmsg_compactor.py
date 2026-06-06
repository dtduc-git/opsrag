"""Tool-output summarization service.

Compresses older `tool_result` entries from `tool_message_history` when
the conversation grows past a configurable fraction of the model's
input-token budget. This lets the tool-calling loop raise its
MAX_TOOL_CALLS ceiling without blowing the context window.

Design notes
------------
- **Idempotent.** Already-summarized entries carry a `_summarized: True`
  flag; rerunning the compactor skips them. A second pass with no new
  growth makes zero LLM calls.
- **Keep-recent rule.** The N most-recent `tool_result` entries are
  preserved verbatim. The agent JUST looked at those to decide its
  next move; summarizing them breaks the chain of reasoning.
- **Identifier preservation.** The summarization prompt instructs the
  LLM that the summary will be re-read later, so it must keep verbatim:
  identifiers, error codes, file paths, counts, status strings. Verbose
  log lines and repeated structure are what get dropped.
- **Archive offload.** If an `archive_store` callable is provided, the
  full original text is sent to it (typically a Qdrant collection or
  object store) before being replaced. The replaced payload carries
  the `archive_key` so the agent can fetch the raw bytes back if a
  follow-up question demands them.
- **Token estimation.** `chars // 4` is the standard cheap heuristic
  for English; for JSON/code it's slightly conservative which is
  exactly what we want for a threshold check. We deliberately do NOT
  pull in tiktoken -- it's a heavy import for a coarse decision.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

_log = logging.getLogger("opsrag.agent.services.toolmsg_compactor")

# Tunables -- exposed as module constants so callers / tests can patch.
_CHARS_PER_TOKEN = 4
_PER_ENTRY_OVERHEAD_TOKENS = 8  # role + name + JSON wrapping cost


_SUMMARIZE_SYSTEM = """\
You are compressing a single tool-call result so it can be re-read later by an SRE agent that already saw the full output once.

The summary will REPLACE the raw output in the agent's working memory. Therefore:
- PRESERVE VERBATIM every identifier, error code, status string, file path, pod / job / pipeline name, count, timestamp, and short error message that appeared in the input. If the raw output contains "pod-abc-12345" or "exit code 137" or "/var/log/foo.log", the summary MUST contain that exact string.
- PRESERVE the structural shape: name the JSON fields / log sections that were present, so the agent knows what data existed.
- DROP verbose / repeated log lines, stack-trace boilerplate, base64 blobs, and identical-shape list items beyond the first few. Replace them with a one-line note like "(plus 47 similar lines)".
- 1 to 2 paragraphs maximum. No markdown headings. No code fences unless quoting a single short error line.
- Lead with the user-visible outcome: success / failure / item count.

Output ONLY the summary text. No preamble, no closing remarks.
"""


def _response_text(msg: dict) -> str | None:
    """Extract the raw text payload from a tool_result entry, if any.

    Returns None if the entry is not a tool_result, already summarized,
    or has no `response.text` field.
    """
    if msg.get("role") != "tool_result":
        return None
    resp = msg.get("response")
    if not isinstance(resp, dict):
        return None
    if resp.get("_summarized"):
        return None
    text = resp.get("text")
    if not isinstance(text, str):
        return None
    return text


def estimate_tokens(history: list[dict]) -> int:
    """Estimate input tokens consumed by `history`.

    For each entry we charge a small per-entry overhead (role / name /
    JSON wrapping) plus `chars // 4` for the textual payload. For
    `tool_result` entries we sum the response text (raw or summarized).
    For `user` / `assistant` / `tool_call` entries we sum the content
    or args JSON.

    Empty history -> 0.
    """
    if not history:
        return 0

    total_chars = 0
    overhead = 0
    for msg in history:
        overhead += _PER_ENTRY_OVERHEAD_TOKENS
        role = msg.get("role")
        if role == "tool_result":
            resp = msg.get("response") or {}
            if isinstance(resp, dict):
                if resp.get("_summarized"):
                    total_chars += len(resp.get("summary") or "")
                else:
                    text = resp.get("text")
                    if isinstance(text, str):
                        total_chars += len(text)
                    else:
                        # Fall back to JSON-encoded length for non-text payloads
                        # (e.g. {"error": "...", "status": 500}).
                        try:
                            total_chars += len(json.dumps(resp, default=str))
                        except Exception:
                            total_chars += len(repr(resp))
        elif role == "tool_call":
            args = msg.get("args") or {}
            try:
                total_chars += len(json.dumps(args, default=str))
            except Exception:
                total_chars += len(repr(args))
            total_chars += len(msg.get("name") or "")
        else:
            # user / assistant / anything else: prefer `content`.
            content = msg.get("content")
            if isinstance(content, str):
                total_chars += len(content)
            elif content is not None:
                try:
                    total_chars += len(json.dumps(content, default=str))
                except Exception:
                    total_chars += len(repr(content))

    return overhead + (total_chars // _CHARS_PER_TOKEN)


def _tool_result_indices(history: list[dict]) -> list[int]:
    """Return indices of tool_result entries in `history`, in order."""
    return [i for i, m in enumerate(history) if m.get("role") == "tool_result"]


def _is_already_summarized(msg: dict) -> bool:
    if msg.get("role") != "tool_result":
        return False
    resp = msg.get("response")
    return isinstance(resp, dict) and bool(resp.get("_summarized"))


async def _summarize_one(
    *,
    llm: Any,
    tool_name: str,
    raw_text: str,
) -> str:
    """Call the LLM to summarize a single tool result.

    The user message frames the raw output with the tool name so the
    model knows what kind of payload it is looking at.
    """
    user_msg = (
        f"Tool name: `{tool_name}`\n"
        f"Raw output ({len(raw_text)} chars):\n\n"
        f"{raw_text}"
    )
    resp = await llm.generate(
        messages=[{"role": "user", "content": user_msg}],
        system_prompt=_SUMMARIZE_SYSTEM,
        temperature=0.0,
        max_tokens=1024,
        purpose="toolmsg_compactor",
    )
    return (resp.content or "").strip()


async def compact_history(
    history: list[dict],
    *,
    llm: Any,
    max_input_tokens: int,
    threshold_fraction: float = 0.7,
    keep_recent_n: int = 3,
    investigation_id: str | None = None,
    archive_store: Callable[[str, str], Awaitable[None] | None] | None = None,
) -> tuple[list[dict], dict]:
    """Compress older `tool_result` entries when history grows past
    `threshold_fraction * max_input_tokens`.

    Algorithm:
        1. Estimate total tokens. If under threshold -> return unchanged.
        2. Pick eligible tool_result entries: those NOT already
           summarized AND NOT in the last `keep_recent_n` tool_results.
        3. Sort eligibles by raw `response.text` length, descending.
        4. For each, until estimated tokens drop under threshold:
             a. Summarize via `llm.generate`.
             b. If `archive_store` is set, call it with
                (archive_key, full_text). `await` if it returns a
                coroutine.
             c. Replace `response.text` with a dict containing
                `_summarized=True`, `summary`, `original_chars`,
                `archive_key`.

    Returns (new_history, stats) where stats has keys:
        - compacted: int -- number of entries summarized this call
        - tokens_before: int
        - tokens_after: int
        - summarized_entries: list[str] -- archive keys produced
    """
    threshold = int(threshold_fraction * max_input_tokens)
    tokens_before = estimate_tokens(history)

    stats: dict = {
        "compacted": 0,
        "tokens_before": tokens_before,
        "tokens_after": tokens_before,
        "summarized_entries": [],
    }

    if tokens_before < threshold:
        return history, stats

    # Work on a shallow copy of the list -- but we will deep-copy any
    # entry we mutate so the caller's history is not aliased.
    new_history: list[dict] = list(history)

    tr_indices = _tool_result_indices(new_history)
    # `kept_recent` = the last keep_recent_n tool_result indices.
    if keep_recent_n > 0:
        kept_recent = set(tr_indices[-keep_recent_n:])
    else:
        kept_recent = set()

    eligible: list[tuple[int, int]] = []  # (length, index)
    for idx in tr_indices:
        if idx in kept_recent:
            continue
        if _is_already_summarized(new_history[idx]):
            continue
        text = _response_text(new_history[idx])
        if text is None:
            continue
        eligible.append((len(text), idx))

    if not eligible:
        _log.info(
            "compact_history over threshold (%d >= %d) but no eligible entries to compact",
            tokens_before, threshold,
        )
        return new_history, stats

    # Largest first -- compressing them yields the biggest token wins.
    eligible.sort(key=lambda pair: pair[0], reverse=True)

    inv_key = investigation_id or "anon"
    running_tokens = tokens_before

    for _length, idx in eligible:
        if running_tokens < threshold:
            break

        original = new_history[idx]
        tool_name = original.get("name") or "unknown_tool"
        full_text = _response_text(original)
        if full_text is None:
            continue  # defensive; shouldn't happen given eligibility check

        try:
            summary = await _summarize_one(
                llm=llm,
                tool_name=tool_name,
                raw_text=full_text,
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "compact_history summarize failed for idx=%d tool=%s: %s",
                idx, tool_name, exc,
            )
            continue

        if not summary:
            _log.warning(
                "compact_history empty summary for idx=%d tool=%s -- keeping raw",
                idx, tool_name,
            )
            continue

        archive_key = f"{inv_key}:{tool_name}:{idx}"

        # Offload raw bytes BEFORE replacing -- if archive fails we
        # log and continue (we still have full_text locally, so the
        # user-facing answer isn't worse -- they just lose the archived
        # copy for follow-up retrieval).
        if archive_store is not None:
            try:
                result = archive_store(archive_key, full_text)
                if hasattr(result, "__await__"):
                    await result  # type: ignore[func-returns-value]
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "compact_history archive_store failed for key=%s: %s",
                    archive_key, exc,
                )

        # Deep-copy the entry we're mutating so the original `history`
        # reference is not aliased.
        replacement = dict(original)
        replacement["response"] = {
            "_summarized": True,
            "summary": summary,
            "original_chars": len(full_text),
            "archive_key": archive_key,
        }
        new_history[idx] = replacement

        stats["compacted"] += 1
        stats["summarized_entries"].append(archive_key)

        # Re-estimate so we know when to stop.
        running_tokens = estimate_tokens(new_history)
        _log.info(
            "compact_history idx=%d tool=%s original_chars=%d summary_chars=%d "
            "tokens %d->%d",
            idx, tool_name, len(full_text), len(summary),
            tokens_before, running_tokens,
        )

    stats["tokens_after"] = estimate_tokens(new_history)
    return new_history, stats


__all__ = ["compact_history", "estimate_tokens"]
