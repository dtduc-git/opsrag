"""Answer generation node.

Step 1 enhancement: parent-substitution at generation time.
ParentChildChunker indexes both layers -- small children for precise
retrieval, larger parents (4x bigger) for context. The retriever
ranks against children. At generation time, we swap each retrieved
child for its parent so the LLM sees full surrounding context, not
the 256-token slice that happened to match.

Effect on token cost: ~2.5x more input tokens per query (e.g.
10 children x 256 = 2,560 -> 6 unique parents x 1024 = ~6,144).
At gemini-2.5-flash $0.30/M input that's ~$0.001 extra per query.

If the vector_store doesn't support parent lookup, or no children
have parent_chunk_id (legacy index, synthetic chunks), behavior
falls back to the original chunks transparently.
"""
from __future__ import annotations

from opsrag.agent.path_tree import (
    build_path_tree_summary_async,
    detect_target_repo,
)
from opsrag.agent.prompts import generation_system_prompt
from opsrag.interfaces.chunker import Chunk
from opsrag.interfaces.llm import LLMProvider
from opsrag.interfaces.observability import ObservabilityProvider
from opsrag.interfaces.vectorstore import VectorStore


async def _substitute_parents(
    chunks: list[Chunk], vector_store: VectorStore | None
) -> list[Chunk]:
    """Replace child chunks with their parent chunks; dedupe by parent id."""
    if not vector_store or not chunks:
        return chunks
    if not hasattr(vector_store, "get_chunks_by_chunk_ids"):
        return chunks  # Older vector store without parent lookup -> no-op.

    # Collect unique parent_chunk_id values in first-seen order.
    parent_ids: list[str] = []
    seen_parents: set[str] = set()
    for c in chunks:
        pid = getattr(c, "parent_chunk_id", None)
        if pid and getattr(c, "chunk_type", "child") == "child" and pid not in seen_parents:
            parent_ids.append(pid)
            seen_parents.add(pid)

    if not parent_ids:
        # Nothing to substitute (all parents already, or synthetic chunks).
        return chunks

    parents = await vector_store.get_chunks_by_chunk_ids(parent_ids)
    parents_by_id = {p.id: p for p in parents}

    # Build output: for each input chunk, emit its parent if it has one
    # and we found it; otherwise pass through. Dedupe -- multiple children
    # of the same parent collapse into one entry.
    out: list[Chunk] = []
    emitted: set[str] = set()
    for c in chunks:
        pid = getattr(c, "parent_chunk_id", None)
        ctype = getattr(c, "chunk_type", "child")
        if pid and ctype == "child" and pid in parents_by_id:
            parent = parents_by_id[pid]
            if parent.id not in emitted:
                out.append(parent)
                emitted.add(parent.id)
        else:
            if c.id not in emitted:
                out.append(c)
                emitted.add(c.id)
    return out


def _format_user_memories(mems: list | None) -> str:
    """Render recalled per-user memories into a compact system-prompt block.

    Accepts Memory dataclasses (``.value["memory"]``) or plain dicts/strings.
    Empty string when there's nothing to inject."""
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
    if not lines:
        return ""
    return (
        "What you remember about this user from past conversations (use only "
        "when relevant; never invent details or force it in):\n" + "\n".join(lines)
    )


def generate_node(
    llm: LLMProvider,
    observability: ObservabilityProvider,
    vector_store: VectorStore | None = None,
    answer_llm: LLMProvider | None = None,
):
    # The final answer is the user-facing voice -- use the stronger "answer"
    # model (Sonnet 4.6) when provided so replies feel like Claude; the cheap
    # nodes (route/HyDE/grade) stay on the base llm. Falls back to base llm.
    gen_llm = answer_llm or llm

    async def _generate(state: dict) -> dict:
        query = state["query"]
        chunks = (
            state.get("graded_chunks")
            or state.get("merged_results")
            or state.get("retrieved_chunks")
            or []
        )

        # Step 1: substitute children with parents for richer LLM context.
        chunks = await _substitute_parents(chunks, vector_store)

        # Stash the post-substitution chunks in state so graph.py emits them
        # as sources_content. Otherwise eval/judge sees children while LLM
        # saw parents -- mismatch tanks faithfulness scoring.
        # (Pure dict assignment; LangGraph merges this into the final state.)

        if not chunks:
            context_block = "(no context retrieved)"
        else:
            context_block = "\n\n---\n\n".join(
                f"[Source: {c.source_path}]\n{c.content}" for c in chunks
            )

        graph_ctx = ""
        gc = state.get("graph_context")
        if gc and getattr(gc, "context_text", ""):
            graph_ctx = f"\n\n=== Knowledge Graph Context ===\n{gc.context_text}"

        # Path-tree summary: when retrieved chunks share a common pivot
        # directory (e.g. `modules/`, `projects/`) across a single target
        # repo, render a 2-level tree (top-level subdir -> its subdirs) and
        # surface it alongside the raw chunks. The LLM otherwise lists 8
        # random file paths instead of enumerating the 23 module categories
        # actually present. Empty string if no useful structure is present
        # (no pivot, single-top, or too few chunks).
        anchors = state.get("anchors") or []
        target_repo = detect_target_repo(anchors, chunks)
        tree_summary = await build_path_tree_summary_async(
            chunks,
            target_repo=target_repo,
            vector_store=vector_store,
            query=query,
        )
        tree_block = ""
        if tree_summary:
            tree_block = (
                "\n\n=== Repository Structure (aggregated from retrieved sources) ===\n"
                f"{tree_summary}"
            )

        # Anchor hint: when the query named specific entities (repo slugs,
        # filenames, hyphenated module names) but NONE of the retrieved
        # sources' paths/repos contain any of those entities, the model
        # must hedge instead of pivoting to adjacent chunks (failure mode
        # 2026-05-16: query "acme-tf-state module variables" was
        # answered with PagerDuty + AlloyDB content because retrieval
        # returned them and the model had no instruction to flag the
        # mismatch).
        anchors_matched = bool(state.get("anchors_matched_in_results", True))
        anchor_hint = ""
        if anchors and not anchors_matched:
            anchor_hint = (
                "\n\nRetrieval note: the user named these specific entities -- "
                f"{', '.join(anchors)} -- but NO retrieved source's path or "
                "repository literally contains any of them. Apply the "
                "named-entity-not-in-sources rule (lead with the gap)."
            )

        # Conversation continuity: prepend the recent prior turns as real
        # messages so multi-turn references ("it", "that error", "the same
        # service") resolve and the chat feels continuous. Bounded to the last
        # few turns to cap tokens. conversation_history is a flat oldest-first
        # [{role, content}] list (see session_store.get_messages).
        history = state.get("conversation_history") or []
        history_msgs = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in history[-6:]
            if isinstance(m, dict) and m.get("content")
        ]

        messages = history_msgs + [
            {
                "role": "user",
                "content": f"Context:\n{context_block}{graph_ctx}{tree_block}{anchor_hint}\n\nQuestion: {query}",
            }
        ]

        # Personalization: inject durable per-user memories (Mem0) into the
        # system prompt so answers reflect what we know about this user.
        system_prompt = generation_system_prompt(state.get("query_type"))
        mem_block = _format_user_memories(state.get("user_memories"))
        if mem_block:
            system_prompt = f"{system_prompt}\n\n{mem_block}"

        # Regenerate loop (grounding failed -> generate again): re-running at
        # temperature 0.0 with identical context is deterministic, so it re-emits
        # the same ungrounded answer and just burns the regen budget. Warm the
        # temperature and tell the model the prior attempt failed grounding so it
        # actually changes, sticking strictly to the evidence.
        regen = int(state.get("regen_count", 0))
        gen_temp = min(0.5, 0.2 * regen)
        if regen:
            system_prompt = (
                f"{system_prompt}\n\nNOTE: a previous answer FAILED the "
                "groundedness check -- some claim was not supported by the "
                "context. Answer again using ONLY facts present in the context "
                "above; omit anything you cannot cite."
            )
        response = await gen_llm.generate(
            purpose="generation",
            messages=messages,
            system_prompt=system_prompt,
            temperature=gen_temp,
        )

        await observability.log_llm_call(
            messages=messages,
            response=response,
            node_name="generate",
            purpose="answer_generation",
        )

        return {
            "generation": response.content,
            "current_step": "generated",
            # Surface what the LLM actually saw -- graph.py prefers this over
            # graded_chunks when emitting sources_content for the API response.
            "final_chunks": chunks,
        }

    return _generate
