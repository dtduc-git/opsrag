"""Embed / cache-key invariant tests for query_with_session.

Pins the pre-graph contract (these hold regardless of any embed scheduling):
  - turn 1 (no prior): the embedded text is the ORIGINAL query (rewrite is a
    no-op) and the query is embedded once;
  - later turns: the FINAL (rewritten) query is what gets embedded AND used as
    the cache key -- so follow-up cache hits track the rewrite.

(The turn-1 speculative-embed optimization was reverted for being non-
equivalent on the transient-embed-failure path; these invariants remain the
guardrail for the serial pre-graph path.)
"""
from __future__ import annotations

import asyncio

from opsrag.agent.graph import query_with_session

# --- fakes -----------------------------------------------------------------


class _RecordingEmbedder:
    """Records every text it embeds and returns a deterministic per-text vector
    so we can assert WHICH query was embedded."""

    def __init__(self) -> None:
        self.embedded: list[str] = []

    async def embed_query(self, text):  # noqa: ANN001
        self.embedded.append(text)
        # Stable distinct vector per text (length-based is enough for identity).
        return [float(len(text)), 1.0, 0.0]


class _SessionStore:
    def __init__(self, messages):
        self._messages = messages

    async def get_messages(self, thread_id):  # noqa: ANN001
        return list(self._messages)


class _NoHitCache:
    """qa_cache that never hits -- isolates the embed/classify path."""

    async def lookup(self, *a, **k):  # noqa: ANN001, ANN002
        return None

    async def store(self, *a, **k):  # noqa: ANN001, ANN002
        return "id"


class _InvokeGraph:
    async def ainvoke(self, initial, config=None):  # noqa: ANN001
        return {
            "query": initial.get("query"),
            "generation": "an answer",
            "generation_grounded": True,
            "grounding_checked": True,
            "final_chunks": [],
            "current_step": "done",
        }


# --- L3: turn-1 parallel embed uses the original query ---------------------


def test_turn1_parallel_embed_uses_original_query():
    """No prior turns -> the embedded text is exactly the original query
    (rewrite is a no-op), and embed runs exactly once."""
    emb = _RecordingEmbedder()
    asyncio.run(
        query_with_session(
            compiled_graph=_InvokeGraph(),
            query="how do I roll back the deploy",
            user_id="u1",
            thread_id="t1",
            embedder=emb,
            qa_cache=_NoHitCache(),
            session_store=_SessionStore([]),  # empty -> turn 1
            llm=None,
        )
    )
    assert emb.embedded == ["how do I roll back the deploy"]


def test_turn1_parallel_embed_matches_serial_vector():
    """The vector produced via the turn-1 parallel path is byte-identical to
    embedding the (unchanged) query directly."""
    emb = _RecordingEmbedder()
    captured: dict = {}

    class _CaptureGraph:
        async def ainvoke(self, initial, config=None):  # noqa: ANN001
            return {
                "query": initial.get("query"),
                "generation": "a",
                "generation_grounded": True,
                "grounding_checked": True,
                "final_chunks": [],
            }

    # Spy the cache.store to capture the embedding that was actually computed.
    class _CapCache:
        async def lookup(self, *a, **k):  # noqa: ANN001, ANN002
            return None

        async def store(self, *a, embedding=None, **k):  # noqa: ANN001, ANN002
            captured["embedding"] = embedding
            return "id"

    asyncio.run(
        query_with_session(
            compiled_graph=_CaptureGraph(),
            query="what is the rollout strategy",
            user_id="u1",
            thread_id="t1",
            embedder=emb,
            qa_cache=_CapCache(),
            session_store=_SessionStore([]),
            llm=None,
        )
    )
    direct = asyncio.run(_RecordingEmbedder().embed_query("what is the rollout strategy"))
    assert captured["embedding"] == direct


def test_later_turn_embeds_rewritten_query(monkeypatch):
    """With prior turns AND a rewrite, the speculative embedding of the ORIGINAL
    query is discarded and the FINAL (rewritten) query is embedded -- so the
    cache key tracks the rewrite, not the raw follow-up."""
    emb = _RecordingEmbedder()

    async def _fake_rewrite(*, query, prior_messages, llm):  # noqa: ANN001
        return "REWRITTEN: " + query

    import opsrag.agent.query_rewrite as qr
    monkeypatch.setattr(qr, "maybe_rewrite_query", _fake_rewrite)

    captured: dict = {}

    class _CapCache:
        async def lookup(self, *a, **k):  # noqa: ANN001, ANN002
            return None

        async def store(self, *a, embedding=None, question=None, **k):  # noqa: ANN001, ANN002
            captured["embedding"] = embedding
            captured["question"] = question
            return "id"

    asyncio.run(
        query_with_session(
            compiled_graph=_InvokeGraph(),
            query="what about its config",
            user_id="u1",
            thread_id="t1",
            embedder=emb,
            qa_cache=_CapCache(),
            session_store=_SessionStore([
                {"role": "user", "content": "tell me about the payments repo"},
                {"role": "assistant", "content": "the payments repo is ..."},
            ]),
            llm=object(),  # truthy -> rewrite path runs
        )
    )
    # The FINAL embedded text is the rewritten query, and it equals the cache key.
    rewritten = "REWRITTEN: what about its config"
    direct = asyncio.run(_RecordingEmbedder().embed_query(rewritten))
    assert captured["question"] == rewritten
    assert captured["embedding"] == direct
    assert rewritten in emb.embedded
