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
import threading
from collections.abc import Iterable

from qdrant_client import models as qm

_log = logging.getLogger("opsrag.vectorstores.bm25_sparse")

_MODEL_NAME = "Qdrant/bm25"
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
    for emb in model.embed(list(texts)):
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
    # query_embed yields a single SparseEmbedding for one query.
    for emb in model.query_embed(text):
        return qm.SparseVector(
            indices=emb.indices.tolist(),
            values=emb.values.tolist(),
        )
    # Empty query -> empty sparse vector (Qdrant tolerates this).
    return qm.SparseVector(indices=[], values=[])
