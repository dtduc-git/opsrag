"""User-correction storage with highest-weight vector boost (T1.6).

When a user clicks the thumbs-down button and types the correct answer in
the chat UI, this module stores the corrected fact in the SAME `opsrag_v2`
Qdrant collection the regular retriever scans -- just stamped with
``priority: user-correction`` so
:func:`opsrag.vectorstores.qdrant._priority_multiplier` lifts its score
2.5x at search time (well above the 1.5x SRE-KB tier).

Design notes
------------
1.  **Shared collection on purpose.** A separate collection would skip
    the boost system entirely. We want the correction to compete in the
    normal retrieval lane -- when the same question (or a paraphrase) gets
    asked again, BM25 + dense both find it AND the boost lifts it above
    any conflicting Confluence/Slack hits.

2.  **Synthetic chunk content.** We embed
    ``"Q: <question>\\nA (user-corrected): <correct_answer>"`` so the
    chunk's vector lives near both the question semantics AND the answer
    semantics -- improves recall for paraphrased follow-ups.

3.  **Deterministic chunk id.** ``correction-<sha1(question|user_id)>`` so
    re-submitting the same correction (same user, same question) overwrites
    the prior version. Different users correcting the same question end up
    in distinct chunks -- that's intentional; a future moderation step can
    merge or dedupe.

4.  **Survives Qdrant rebuilds via audit log.** The route handler that
    calls :meth:`store_correction` ALSO writes a row to the existing
    Postgres ``opsrag_feedback`` table (``direction=2``). If Qdrant is
    nuked and re-indexed, the audit log gives operators a replay path
    (re-POST every saved correction).
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from opsrag.interfaces.embedder import EmbeddingProvider
from opsrag.vectorstores import bm25_sparse
from opsrag.vectorstores.qdrant import _chunk_point_id

_log = logging.getLogger("opsrag.correction_store")

# Reserved repo + source_path values for filtering corrections out of
# (or into) reports. The repo is set so users browsing the indexing
# dashboard can see how many corrections live in the corpus; the
# source_path is set so chat-UI source citations are still legible.
CORRECTION_REPO: str = "user-correction"
CORRECTION_SOURCE_PATH: str = "user-correction"
CORRECTION_PRIORITY: str = "user-correction"
# Use generic_markdown for doc_type because the DocType enum in
# opsrag.interfaces.parser is a closed set and the retrieval path
# `_hit_to_result` round-trips this string through `DocType(...)` --
# adding a new enum member would require a coordinated migration of the
# whole indexer. The chunk is unambiguously identified as a correction
# via `repo == "user-correction"` and `priority == "user-correction"`.
CORRECTION_DOC_TYPE: str = "generic_markdown"

# Named-vector field constants -- keep in sync with qdrant.py. We duplicate
# them rather than import the private names so the store stays decoupled
# from the QdrantVectorStore implementation details.
_DENSE = "dense"
_BM25 = "bm25"


def _deterministic_chunk_id(question: str, user_id: str) -> str:
    """Stable id keyed on (question, user_id). Re-submitting the same
    correction (e.g. user fixes a typo in their answer) overwrites the
    prior version -- no orphan duplicates."""
    key = f"{(question or '').strip().lower()}|{(user_id or '').strip().lower()}"
    # Non-security dedup id: this digest only de-duplicates re-submitted
    # corrections, it never protects secrets. usedforsecurity=False documents
    # that intent (and clears CodeQL's weak-sensitive-data-hashing finding).
    digest = hashlib.sha256(key.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    return f"correction-{digest}"


def _synthetic_chunk(question: str, correct_answer: str) -> str:
    """Compose the chunk body that gets embedded. The Q+A form gives the
    dense vector both question-anchor and answer-anchor signal -- improves
    retrieval recall for paraphrased follow-up questions."""
    return f"Q: {question.strip()}\nA (user-corrected): {correct_answer.strip()}"


class CorrectionStore:
    """User-correction storage on top of AsyncQdrantClient + embedder.

    The store doesn't manage its own Qdrant collection lifecycle -- it
    upserts into the main retrieval collection (typically ``opsrag_v2``)
    so corrections compete in the standard search lane. Collection
    creation is owned by :class:`QdrantVectorStore` which runs first
    during ingestion startup; correction upserts after that point.
    """

    def __init__(
        self,
        qdrant: AsyncQdrantClient,
        embedder: EmbeddingProvider,
        collection_name: str = "opsrag",
    ) -> None:
        self._qdrant = qdrant
        self._embedder = embedder
        self._collection = collection_name

    async def store_correction(
        self,
        *,
        question: str,
        wrong_answer: str,
        correct_answer: str,
        user_id: str = "anonymous",
        evidence_url: str | None = None,
        reviewed_by: str | None = None,
    ) -> str:
        """Inject an OPERATOR-APPROVED correction chunk into Qdrant. Returns the
        ``chunk_id`` (deterministic -- re-approval overwrites).

        Called only from the moderation approve path (see
        :mod:`opsrag.pending_corrections`); submission itself never reaches
        here. The chunk is upserted with priority ``user-correction`` so
        :func:`_priority_multiplier` lifts its dense+BM25 RRF score (1.8x) at
        search time.
        """
        if not (question and correct_answer):
            raise ValueError("question and correct_answer are required")

        chunk_id = _deterministic_chunk_id(question, user_id)
        body = _synthetic_chunk(question, correct_answer)

        # Embed via the same provider the retriever uses so cosine
        # geometry lines up -- using a different model would shift the
        # vector neighborhood and the correction wouldn't reliably
        # surface near the user's next question.
        dense_vec = await self._embedder.embed_query(body)
        sparse_vec = bm25_sparse.encode_documents([body])[0]

        created_at = time.time()
        point = qm.PointStruct(
            id=_chunk_point_id(chunk_id),
            vector={_DENSE: dense_vec, _BM25: sparse_vec},
            payload={
                "chunk_id": chunk_id,
                "content": body,
                "doc_type": CORRECTION_DOC_TYPE,
                "source_path": CORRECTION_SOURCE_PATH,
                "repo": CORRECTION_REPO,
                "parent_chunk_id": None,
                "chunk_type": "child",
                "token_count": 0,
                "metadata": {
                    "original_question": question,
                    "wrong_answer": wrong_answer or "",
                    "correct_answer": correct_answer,
                    "user_id": user_id,
                    "evidence_url": evidence_url,
                    "created_at": created_at,
                    "status": "approved",
                    "reviewed_by": reviewed_by,
                },
                # -- 2.5x boost tag -- see qdrant._priority_multiplier --
                "priority": CORRECTION_PRIORITY,
            },
        )

        await self._qdrant.upsert(
            collection_name=self._collection,
            points=[point],
            wait=True,  # ack before route returns so /query right after sees it
        )
        _log.info(
            "user-correction stored chunk_id=%s user=%s q_chars=%d a_chars=%d",
            chunk_id, user_id, len(question), len(correct_answer),
        )
        return chunk_id

    async def list_recent_corrections(self, limit: int = 50) -> list[dict[str, Any]]:
        """Scroll Qdrant for `repo=user-correction` chunks, ordered by the
        embedded ``metadata.created_at`` (Python-side sort -- Qdrant scroll
        doesn't sort by payload). Returns dicts with chunk_id + metadata,
        suitable for an SRE moderation list view.
        """
        limit = max(1, min(int(limit), 500))
        qfilter = qm.Filter(must=[
            qm.FieldCondition(key="repo", match=qm.MatchValue(value=CORRECTION_REPO)),
        ])
        # Scroll up to ~10x the requested limit so even with skew (many
        # corrections from one user) we have enough rows to sort and
        # return the freshest `limit` to the caller.
        scroll_cap = max(limit * 10, 200)
        collected: list[dict[str, Any]] = []
        cursor = None
        for _ in range(20):  # safety bound
            points, cursor = await self._qdrant.scroll(
                collection_name=self._collection,
                scroll_filter=qfilter,
                limit=500,
                offset=cursor,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                meta = payload.get("metadata") or {}
                collected.append({
                    "chunk_id": payload.get("chunk_id", str(p.id)),
                    "content": payload.get("content", ""),
                    "original_question": meta.get("original_question"),
                    "wrong_answer": meta.get("wrong_answer"),
                    "correct_answer": meta.get("correct_answer"),
                    "user_id": meta.get("user_id"),
                    "evidence_url": meta.get("evidence_url"),
                    "created_at": meta.get("created_at"),
                })
            if not cursor or len(collected) >= scroll_cap:
                break
        # Newest first -- created_at is a float epoch; None sinks to bottom.
        collected.sort(
            key=lambda r: (r.get("created_at") or 0.0),
            reverse=True,
        )
        return collected[:limit]

    async def delete_correction(self, chunk_id: str) -> bool:
        """Remove a single correction by chunk_id. Returns True if the
        delete RPC succeeded (Qdrant doesn't report whether the point
        actually existed -- a stale id returns success too)."""
        if not chunk_id:
            return False
        try:
            await self._qdrant.delete(
                collection_name=self._collection,
                points_selector=qm.PointIdsList(points=[_chunk_point_id(chunk_id)]),
                wait=True,
            )
            _log.info("user-correction deleted chunk_id=%s", chunk_id)
            return True
        except Exception as exc:
            _log.warning("user-correction delete failed chunk_id=%s: %s", chunk_id, exc)
            return False
