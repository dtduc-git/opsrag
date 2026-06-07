"""BM25 sparse vector encoder using FastEmbed.

Wraps `fastembed.SparseTextEmbedding` with the `Qdrant/bm25` model.
Provides:
  - `encode_documents(texts)` for index-time TF vectors
  - `encode_query(text)` for query-time BM25-weighted vectors

Both return Qdrant-native `SparseVector` objects (indices + values lists).
The Qdrant collection's IDF modifier handles BM25 IDF computation server-side
at query time -- we only provide TF + token IDs.

Lazy-loaded singleton: model loads on first call (~5-10 MB download cached
to `~/.cache/fastembed/`), subsequent calls reuse.

Reference: Qdrant native sparse vectors with BM25 modifier
(Qdrant 1.13+): https://qdrant.tech/articles/sparse-vectors/
"""
from __future__ import annotations

import logging
import re
import threading
from collections.abc import Iterable

from qdrant_client import models as qm

from opsrag.vectorstores.lane_weights import extract_identifiers

_log = logging.getLogger("opsrag.vectorstores.bm25_sparse")

_MODEL_NAME = "Qdrant/bm25"

# Split a compound identifier into subtokens on separators + camelCase humps.
_SUBTOKEN_SPLIT = re.compile(
    r"[_\-./:]+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
)


def _bm25_augment(text: str) -> str:
    """Append identifier SUBTOKENS so the sparse vocab can match partial symbols.

    The Qdrant/bm25 model stems English and splits on whitespace/punctuation but
    keeps `handle_webhook_callback`, `api.v1.users`, `acme-notes-be-api` as
    SINGLE tokens -- so a query for `webhook_callback` never lexically matches,
    defeating the lexical lane on a code corpus. We append the split parts
    (handle / webhook / callback) to the text fed to BM25, applied to BOTH
    documents and queries so their subtoken vocabularies line up. The original
    identifier stays in `text` (exact-symbol matches unaffected) -- this only
    ADDS recall. NB: changes the indexed sparse vectors -> needs a re-index."""
    extra: list[str] = []
    for ident in extract_identifiers(text):
        parts = [p for p in _SUBTOKEN_SPLIT.split(ident) if len(p) >= 2]
        if len(parts) > 1:
            extra.extend(parts)
    return f"{text} {' '.join(extra)}" if extra else text
_lock = threading.Lock()
_model = None
_query_model = None


def _ensure_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                # Defer import so non-sparse code paths don't pay the import cost.
                from fastembed import SparseTextEmbedding
                _log.info("loading fastembed BM25 model: %s", _MODEL_NAME)
                _model = SparseTextEmbedding(model_name=_MODEL_NAME)
                _log.info("BM25 model loaded")
    return _model


def encode_documents(texts: Iterable[str]) -> list[qm.SparseVector]:
    """Encode a batch of document texts as sparse vectors for index-time storage.

    Returns Qdrant SparseVector with int indices (token IDs from BM25 vocab)
    and float values (term frequencies, normalized per FastEmbed BM25 spec).
    Qdrant computes BM25 IDF at query time using the IDF modifier on the
    sparse vector field.
    """
    model = _ensure_model()
    out: list[qm.SparseVector] = []
    # FastEmbed's embed() yields SparseEmbedding(indices: ndarray, values: ndarray)
    for emb in model.embed([_bm25_augment(t) for t in texts]):
        out.append(qm.SparseVector(
            indices=emb.indices.tolist(),
            values=emb.values.tolist(),
        ))
    return out


def encode_query(text: str) -> qm.SparseVector:
    """Encode a single query text as a sparse vector for retrieval.

    BM25 model uses different tokenization weights at query vs document time
    (per Qdrant/bm25 model card). FastEmbed's `query_embed()` handles this.
    """
    model = _ensure_model()
    # query_embed yields a single SparseEmbedding for one query. Augment with the
    # same identifier subtokens used at index time so partial-symbol queries match.
    for emb in model.query_embed(_bm25_augment(text)):
        return qm.SparseVector(
            indices=emb.indices.tolist(),
            values=emb.values.tolist(),
        )
    # Empty query -> empty sparse vector (Qdrant tolerates this).
    return qm.SparseVector(indices=[], values=[])
