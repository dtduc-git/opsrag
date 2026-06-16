"""Phase 03 Pillar 2 -- agentic tool-calling nodes.

Three nodes orchestrate the tool-calling loop:

  - `tool_decide_node`     LLM with function-calling decides whether to
                           invoke tools (one or more), or fall through
                           to retrieval. Emits 0..N pending tool calls.
  - `tool_execute_node`    Dispatches each pending tool call to the
                           shared registry (`opsrag.mcp.GITLAB_TOOLS`),
                           records latency + error per call, appends
                           tool-result messages to the LLM history.
  - `tool_synthesize_node` After the loop ends, asks the LLM (no tools
                           this time) to write a final answer using
                           the conversation history as grounding.

Loop bound: `MAX_TOOL_CALLS = 3`. After 3 executions the loop forces
synthesis whether or not the LLM wanted to call more tools.

Audit: every tool execution writes a row to `state.tool_call_audit`
with name, args, latency_ms, and error string when applicable. This
becomes the seed for Sub-sprint 5's audit log.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from opsrag.interfaces.chunker import Chunk
from opsrag.interfaces.observability import ObservabilityProvider
from opsrag.interfaces.parser import DocType
from opsrag.mcp import ALL_MCP_TOOLS, GITLAB_TOOLS, GitLabClient, GitLabMCPError, MCPTool

_log = logging.getLogger("opsrag.agent.tool_caller")

MAX_TOOL_CALLS = 10  # bumped 3->10 2026-05-09 per user direction (testing deeper drilling)
# Per-tool result cap fed back to the LLM. Bumped from 8000 -> 32000 so
# `gitlab_get_pipeline_job` traces (200 lines x ~80 chars ~= 16KB plus
# job metadata) survive intact. Pro / Flash both handle 32KB cheaply.
_RESULT_TRUNCATE_CHARS = 32000


def _registry() -> dict[str, MCPTool]:
    """Executable tools = the SAME operator-enabled set the reasoner is offered
    (`filter_enabled(ALL_MCP_TOOLS)`), not just GitLab. Without this the
    executor rejected every non-GitLab tool the reasoner picked (code_*,
    datadog_*, rootly_*, …) as "unknown tool". `filter_enabled` reads the
    process-global active set and returns all tools when it's unset."""
    from opsrag.mcp_server.registry_loader import filter_enabled
    tools = filter_enabled(ALL_MCP_TOOLS) or list(GITLAB_TOOLS)
    return {t.name: t for t in tools}


# -- Sources-via-state (Option 2) -----------------------------------------
#
# Retrieval tools called from the agent loop (e.g., `knowledge_search`)
# return content to the LLM as a dict that the synthesizer reads to ground
# its answer. But the API response wants a `sources` array -- file paths +
# content snippets -- pulled from the agent state's `final_chunks` field.
#
# Pre-fix, the tool path set `final_chunks: []` because tool outputs were
# strings, not Chunk objects -- so confluence/runbook queries that grounded
# correctly came back with `sources: []` in the response, and goldens
# referencing those expected_sources scored SourceRecall=0.
#
# The registry below maps each retrieval-tool name to a function that
# converts its return value into `list[Chunk]`. `tool_execute_node`
# accumulates these into `state["tool_retrieved_chunks"]` (deduped by
# repo+source_path); `tool_synthesize_node` copies that list to
# `final_chunks` so the response builder picks it up.
#
# To support a new retrieval tool, add an entry to `_RETRIEVAL_EXTRACTORS`
# below -- no other changes needed.


def _extract_chunks_from_knowledge_search(result: Any) -> list[Chunk]:
    """Convert `knowledge_search`'s return dict to a list[Chunk].

    Return shape (see opsrag/mcp/knowledge.py):
      {"query": "...", "count": N,
       "results": [{"source", "repo", "score", "content", "url"?, "title"?, "priority"?}, ...]}

    Content is already capped at 1200 chars by the tool. Metadata fields
    (url/title/priority) are preserved on the Chunk's metadata dict so
    `_src_url` in api/graph.py can render clickable links.
    """
    if not isinstance(result, dict):
        return []
    hits = result.get("results") or []
    chunks: list[Chunk] = []
    for h in hits:
        if not isinstance(h, dict):
            continue
        src = h.get("source") or ""
        if not src:
            continue
        repo = h.get("repo") or ""
        # Mirror the URL key set in knowledge.py so downstream renderers
        # find page_url / permalink / web_url in the expected slots.
        meta: dict[str, Any] = {}
        if h.get("url"):
            meta["page_url"] = h["url"]
        if h.get("title"):
            meta["title"] = h["title"]
        if h.get("priority"):
            meta["priority"] = h["priority"]
        chunks.append(
            Chunk(
                id=f"mcp:knowledge_search:{repo}:{src}",
                content=h.get("content") or "",
                doc_type=DocType.GENERIC_MARKDOWN,
                source_path=src,
                repo=repo,
                metadata=meta,
                chunk_type="parent",  # arbitrary; downstream only reads source_path/content
            )
        )
    return chunks


_RETRIEVAL_EXTRACTORS: dict[str, Any] = {
    "knowledge_search": _extract_chunks_from_knowledge_search,
}


def _dedupe_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Dedupe by (repo, source_path) while preserving first-seen order.

    Multiple knowledge_search calls in the same turn can return the same
    file from different angles; the response builder also dedupes, but
    keeping the list tight here means a smaller state payload and a
    deterministic order for downstream code that iterates without dedupe.
    """
    seen: set[tuple[str, str]] = set()
    out: list[Chunk] = []
    for c in chunks:
        key = (c.repo or "", c.source_path or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _tool_specs_for_llm() -> list[dict]:
    """MCP-shape tool specs in the form `generate_with_tools` expects."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.input_schema,
        }
        for t in GITLAB_TOOLS
    ]


_SYSTEM_DECIDE = """\
You are an SRE assistant. The user has access to:
- An indexed knowledge base (runbooks, Terraform, Helm, incidents, Slack threads, Confluence) reachable via standard retrieval -- used by default.
- Live GitLab tools listed below as functions -- use these ONLY when the user asks about LIVE state that the indexed corpus cannot answer: recent pipelines, pipeline-job status / logs, recent commits, recent merge requests, deployments.

Decide:
- If the user asks about live GitLab state, call the matching function. You may call up to 3 functions across the conversation.
- Otherwise return a short text reply like "Use retrieval." and DO NOT call a function. The retrieval pipeline will handle it.

DRILLING DISCIPLINE -- for "why did X fail" / "what's wrong with X" / root-cause questions about pipelines or jobs, chain calls aggressively in one turn:
1. First call: identify the pipeline (`gitlab_get_pipeline` if you have an ID; `gitlab_list_pipelines` to find it).
2. Second call: list the failed jobs (`gitlab_list_pipeline_jobs` with `scope=failed`).
3. Third call: pull the trace tail of the FIRST failed job (`gitlab_get_pipeline_job` with `limit=200` so the synthesizer sees the actual error). This is the most valuable signal -- DO NOT skip it.
Use the full 3-call budget for failure analysis. Stopping at "two jobs failed with script_failure" without the trace is a poor answer; drill in.

When project_id is not in the user message, derive it from a GitLab URL (e.g. URL `gitlab.example.com/automation-test/unified-health-check/-/pipelines/1530036` -> project_id `automation-test/unified-health-check`, pipeline_id `1530036`). When only a service slug is named (e.g. `acme-notes-be`), default to `saas/<slug>`.

Always pass `project_id` either as a numeric id or as a URL-style path. URL paths with `/` are URL-encoded automatically by the tool.
"""


_SYSTEM_SYNTHESIZE = """\
You are an SRE assistant. You called live GitLab tools and have the results in your conversation history. Write a concise answer to the original user question using ONLY those results.

Format guidelines:
- Lead with the direct answer (status, count, timestamp, etc).
- Cite each fact inline with the tool name and the relevant ID(s) in brackets, e.g. `[gitlab_list_pipelines#1531061]`.
- If a tool errored or returned empty, say so explicitly -- do not invent results.
- Keep code/identifiers in backticks.

WHEN ANALYZING FAILURES -- you have a `gitlab_get_pipeline_job` trace tail in the history. USE IT:
- Quote the actual error line(s) from the trace, in a fenced code block.
- Identify the failure category (test failure / network / dependency / OOM / timeout / config) from the trace, not from the job's `status` field.
- If the failure is a test failure, name the failing test(s) by the trace.
- Then map back to which file or service likely caused it, if discernible from trace + commit context.
- A failure-analysis answer that does NOT quote the trace is incomplete; flag the missing trace explicitly if it wasn't fetched.
"""


def tool_decide_node(llm, observability: ObservabilityProvider):
    """Build the tool-decide node: LLM with function-calling looks at
    the user query (plus any prior tool results in the loop) and
    decides whether to invoke a tool or fall through to retrieval."""

    async def _decide(state: dict) -> dict:
        query = state.get("query") or ""
        history: list[dict] = state.get("tool_message_history") or []
        call_count = int(state.get("tool_call_count") or 0)

        # Build the LLM message history for this turn.
        if not history:
            history = [{"role": "user", "content": query}]

        # Hard loop bound -- past MAX_TOOL_CALLS we don't even ask the LLM
        # to decide; we synthesize whatever we have.
        if call_count >= MAX_TOOL_CALLS:
            _log.info(
                "tool_decide loop cap hit (%d/%d) -- forcing synthesize",
                call_count, MAX_TOOL_CALLS,
            )
            return {
                "tool_calls": [],
                "tool_message_history": history,
                "tool_path_active": True,
                "current_step": "tool_decide",
            }

        try:
            resp = await llm.generate_with_tools(
                messages=history,
                tools=_tool_specs_for_llm(),
                system_prompt=_SYSTEM_DECIDE,
                temperature=0.0,
                max_tokens=2048,
                purpose="tool_decide",
            )
        except Exception as exc:
            _log.warning("tool_decide LLM error: %s -- falling through to retrieval", exc)
            return {
                "tool_calls": [],
                "tool_path_active": False,
                "current_step": "tool_decide",
                "error": f"tool_decide_failed: {exc}",
            }

        if not resp.tool_calls:
            # LLM declined to call a function. Two cases:
            #   - call_count > 0 -> tools already ran; LLM is done -> synthesize.
            #   - call_count == 0 -> never used tools -> fall through to retrieval.
            if call_count > 0:
                _log.info(
                    "tool_decide -> synthesize (LLM done, %d call(s) executed)",
                    call_count,
                )
                return {
                    "tool_calls": [],
                    "tool_message_history": history,
                    "tool_path_active": True,
                    "current_step": "tool_decide",
                }
            _log.info("tool_decide -> retrieval (no function calls)")
            return {
                "tool_calls": [],
                "tool_path_active": False,
                "current_step": "tool_decide",
            }

        # LLM emitted function calls. Translate to plain dicts the
        # execute node can dispatch on, and persist on history so the
        # next decide turn sees them.
        pending = [{"name": tc.name, "args": tc.args} for tc in resp.tool_calls]
        _log.info(
            "tool_decide -> execute %d call(s): %s",
            len(pending), [p["name"] for p in pending],
        )
        for tc in resp.tool_calls:
            history.append({"role": "tool_call", "name": tc.name, "args": tc.args})

        return {
            "tool_calls": pending,
            "tool_message_history": history,
            "tool_path_active": True,
            "current_step": "tool_decide",
        }

    return _decide


def _safe_json(obj: Any, limit: int) -> str:
    try:
        s = json.dumps(obj, default=str)
    except Exception:
        s = repr(obj)
    if len(s) > limit:
        return s[:limit] + f"... [truncated {len(s) - limit} chars]"
    return s


def tool_execute_node(observability: ObservabilityProvider):
    """Dispatches pending tool calls. Each result is appended to
    `tool_message_history` as a `tool_result` entry so the next
    decide turn sees the full conversation."""

    async def _execute(state: dict) -> dict:
        pending: list[dict] = state.get("tool_calls") or []
        history: list[dict] = list(state.get("tool_message_history") or [])
        audit: list[dict] = list(state.get("tool_call_audit") or [])
        # Accumulate across loop iterations -- earlier iterations may have
        # already populated this field; we extend, not overwrite.
        retrieved_chunks: list[Chunk] = list(state.get("tool_retrieved_chunks") or [])
        registry = _registry()
        executed = 0
        executed_count = int(state.get("tool_call_count") or 0)

        if not pending:
            return {
                "tool_calls": [],
                "tool_results": state.get("tool_results") or [],
                "tool_message_history": history,
                "tool_call_audit": audit,
                "tool_call_count": executed_count,
                "tool_retrieved_chunks": retrieved_chunks,
                "current_step": "tool_execute",
            }

        async with GitLabClient() as client:
            for call in pending:
                if executed_count >= MAX_TOOL_CALLS:
                    _log.warning(
                        "tool_execute dropping call name=%s -- loop cap (%d) reached",
                        call.get("name"), MAX_TOOL_CALLS,
                    )
                    break
                name = call.get("name", "")
                args = call.get("args") or {}
                tool = registry.get(name)
                start = time.perf_counter()
                if tool is None:
                    err = f"unknown tool {name!r}"
                    _log.warning("tool_execute %s", err)
                    history.append({
                        "role": "tool_result", "name": name,
                        "response": {"error": err},
                    })
                    audit.append({
                        "name": name, "args": args, "latency_ms": 0.0,
                        "error": err, "ts": time.time(),
                    })
                    executed_count += 1
                    executed += 1
                    continue
                try:
                    result = await tool.call(client, args)
                    latency_ms = (time.perf_counter() - start) * 1000
                    truncated = _safe_json(result, _RESULT_TRUNCATE_CHARS)
                    # Always carry as text -- `truncated` may be a mid-string cut
                    # of a long JSON blob (e.g. trace logs >8KB), so json.loads
                    # would crash. The synthesizer reads payload as text via
                    # `_flatten_tool_history`; structured access isn't needed
                    # in this path.
                    response_payload = {"text": truncated}
                    history.append({
                        "role": "tool_result", "name": name,
                        "response": response_payload,
                    })
                    # Sources-via-state: if this is a retrieval tool, lift
                    # its result chunks into state so `tool_synthesize_node`
                    # can pass them to `final_chunks`. We parse the raw
                    # `result` dict (not the truncated JSON string) so the
                    # chunk content survives intact.
                    extractor = _RETRIEVAL_EXTRACTORS.get(name)
                    new_chunks: list[Chunk] = []
                    if extractor is not None:
                        try:
                            new_chunks = extractor(result)
                            if new_chunks:
                                retrieved_chunks.extend(new_chunks)
                                retrieved_chunks = _dedupe_chunks(retrieved_chunks)
                        except Exception as exc:
                            # Extractor bugs must not block the tool flow --
                            # log and keep going; response will fall back to
                            # the legacy empty-sources behavior for this call.
                            _log.warning(
                                "tool_execute %s chunk-extract failed: %s",
                                name, exc,
                            )
                    audit.append({
                        "name": name, "args": args,
                        "latency_ms": round(latency_ms, 1),
                        "result_chars": len(truncated),
                        "chunks_lifted": len(new_chunks),
                        "ts": time.time(),
                    })
                    _log.info(
                        "tool_execute name=%s latency=%.0fms result_chars=%d chunks_lifted=%d",
                        name, latency_ms, len(truncated), len(new_chunks),
                    )
                except GitLabMCPError as exc:
                    latency_ms = (time.perf_counter() - start) * 1000
                    err_payload = {"error": str(exc), "status": exc.status}
                    history.append({
                        "role": "tool_result", "name": name,
                        "response": err_payload,
                    })
                    audit.append({
                        "name": name, "args": args,
                        "latency_ms": round(latency_ms, 1),
                        "error": str(exc), "ts": time.time(),
                    })
                    _log.warning("tool_execute %s failed: %s", name, exc)
                except Exception as exc:
                    latency_ms = (time.perf_counter() - start) * 1000
                    history.append({
                        "role": "tool_result", "name": name,
                        "response": {"error": f"unhandled: {exc}"},
                    })
                    audit.append({
                        "name": name, "args": args,
                        "latency_ms": round(latency_ms, 1),
                        "error": f"unhandled: {exc}", "ts": time.time(),
                    })
                    _log.exception("tool_execute %s unhandled error", name)
                executed_count += 1
                executed += 1

        return {
            "tool_calls": [],  # cleared so the next loop iteration re-decides
            "tool_message_history": history,
            "tool_call_audit": audit,
            "tool_call_count": executed_count,
            "tool_retrieved_chunks": retrieved_chunks,
            "current_step": "tool_execute",
        }

    return _execute


def _flatten_tool_history(history: list[dict]) -> list[dict]:
    """Convert tool_call / tool_result messages into plain user/assistant
    turns so the synthesize call can run with `tools=[]` without
    Vertex rejecting orphan function_call parts."""
    flat: list[dict] = []
    for msg in history:
        role = msg.get("role")
        if role in ("user", "assistant"):
            flat.append(msg)
        elif role == "tool_call":
            args_str = json.dumps(msg.get("args", {}) or {}, default=str)
            flat.append({
                "role": "assistant",
                "content": f"[called tool] {msg['name']}({args_str})",
            })
        elif role == "tool_result":
            resp = msg.get("response", {}) or {}
            payload = resp.get("data") if isinstance(resp, dict) and "data" in resp else resp
            text_payload = json.dumps(payload, default=str)
            if len(text_payload) > _RESULT_TRUNCATE_CHARS:
                text_payload = text_payload[:_RESULT_TRUNCATE_CHARS] + " ...[truncated]"
            flat.append({
                "role": "user",
                "content": f"[tool_result] {msg['name']} returned: {text_payload}",
            })
    return flat


def tool_synthesize_node(llm, observability: ObservabilityProvider, model_router=None):
    """Final-answer node for the tool path. Asks the LLM (no tools
    enabled) to write a concise answer from the conversation history.

    Phase 03 Pillar 3: when `model_router` is provided, the original
    user query is classified and synthesis routes to either Flash
    (default) or Pro (multi-step / cross-source / root-cause). The
    decision is recorded on `state.model_route_decision` for audit.
    """

    async def _synthesize(state: dict) -> dict:
        history: list[dict] = state.get("tool_message_history") or []
        if not history:
            return {
                "generation": "",
                "current_step": "tool_synthesize",
                "error": "tool_synthesize called with empty history",
            }
        flattened = _flatten_tool_history(history)

        # Pick model: Pillar 3 router OR fall back to the static llm.
        chosen_llm = llm
        route_decision = None
        if model_router is not None:
            chosen_llm, route_decision = model_router.pick(state.get("query") or "")

        try:
            resp = await chosen_llm.generate(
                messages=flattened,
                system_prompt=_SYSTEM_SYNTHESIZE,
                temperature=0.0,
                max_tokens=4096,
                purpose="tool_synthesize",
            )
        except Exception as exc:
            _log.warning("tool_synthesize LLM error: %s", exc)
            return {
                "generation": "",
                "current_step": "tool_synthesize",
                "error": f"tool_synthesize_failed: {exc}",
            }

        # Build sources for the API response:
        #   - `final_chunks` = chunks retrieved by knowledge_search (and
        #     any other registered retrieval tool). This is what the
        #     response builder in api/graph.py reads into `sources`,
        #     `sources_content`, and `source_urls`.
        #   - `sources_searched` = a flat list combining concrete chunk
        #     source paths AND the tool names invoked (so audit-friendly
        #     UI surfaces still see "mcp://gitlab_list_pipelines" for
        #     live-state tool calls that don't yield Chunks).
        audit = state.get("tool_call_audit") or []
        retrieved_chunks: list[Chunk] = list(state.get("tool_retrieved_chunks") or [])
        # Stable order: retrieval-tool chunk paths first, then tool-name
        # markers for non-retrieval tools. Dedupe at the end.
        sources_searched: list[str] = []
        for c in retrieved_chunks:
            tag = f"{c.repo}/{c.source_path}" if c.repo else c.source_path
            if tag and tag not in sources_searched:
                sources_searched.append(tag)
        for entry in audit:
            tool_tag = f"mcp://{entry['name']}"
            if tool_tag not in sources_searched:
                sources_searched.append(tool_tag)
        out: dict = {
            "generation": resp.content,
            # No groundedness check runs on the tool_synthesize path, so do NOT
            # claim the answer is grounded. Leaving this False (with no
            # `grounding_checked` flag) reflects "unverified", not "failed" --
            # the QA-cache write gate keys on (grounded is False AND
            # grounding_checked is True), so an unverified tool answer is not
            # mislabelled as a grounding failure, and the prior hardcoded
            # `True` no longer lets ungrounded tool answers pass the gate.
            "generation_grounded": False,
            "final_chunks": retrieved_chunks,
            "sources_searched": sources_searched,
            "current_step": "tool_synthesize",
        }
        if route_decision is not None:
            out["model_route_decision"] = {
                "tier": route_decision.tier,
                "reason": route_decision.reason,
                "matched_patterns": route_decision.matched_patterns,
                "model": chosen_llm.model_name,
            }
            _log.info(
                "tool_synthesize tier=%s reason=%s model=%s",
                route_decision.tier, route_decision.reason, chosen_llm.model_name,
            )
        return out

    return _synthesize


def tool_decide_route(state: dict) -> str:
    """LangGraph conditional edge after `tool_decide`. Routes to
    `tool_execute` when the LLM emitted function calls (or the loop
    cap was hit and we need to synthesize), and to `vector_retrieve`
    when the LLM declined."""
    if state.get("tool_path_active"):
        # Either pending tool calls OR loop cap forces synthesize.
        return "tool_execute" if state.get("tool_calls") else "tool_synthesize"
    return "retrieval"
