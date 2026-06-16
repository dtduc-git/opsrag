"""Fail-closed embedding-dimension guard (models feature, F2).

``vector_store.dimension`` is pinned to ``embedder.dimension``. A bundle /
embed-model switch (Titan 1024 <-> Vertex 768 <-> OpenAI 3072) silently
corrupts an existing Qdrant collection -- Qdrant rejects vectors of the
wrong size, and a re-create destroys the collection shared by the main
index, the QA cache, and investigations.

``assert_dimension_compatible`` reads an existing collection's dense
vector size and raises a clear error if it differs from ``expected_dim``
while ``allow_change`` is False (fail-closed). A missing collection is a
no-op (the factory will create it at the correct dimension). The check is
defensively typed so a Qdrant client-shape change never crashes startup
in a way that obscures the real cause.
"""
from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger("opsrag.vectorstore_guard")

# Named dense vector used by QdrantVectorStore (see vectorstores/qdrant.py).
_DENSE = "dense"


class DimensionMismatchError(RuntimeError):
    """Raised when an existing collection's dimension differs from the
    embedder's dimension and ``allow_dimension_change`` is False."""


def _extract_dense_size(info: Any) -> int | None:
    """Pull the dense vector size out of a Qdrant get_collection() result.

    Tolerates both single-vector (``VectorParams``) and named-vector
    (``dict[str, VectorParams]``) collection shapes, and a couple of
    historical attribute paths. Returns None if the size can't be found
    (treated as "can't verify" -> no-op, not a hard fail)."""
    try:
        params = info.config.params  # CollectionInfo.config.params
    except AttributeError:
        return None

    vectors = getattr(params, "vectors", None)
    if vectors is None:
        return None

    # Named-vector collections: dict of {name: VectorParams}.
    if isinstance(vectors, dict):
        vp = vectors.get(_DENSE)
        if vp is None and vectors:
            # Single unnamed-ish entry under some other key -- use it.
            vp = next(iter(vectors.values()))
        size = getattr(vp, "size", None)
        return int(size) if isinstance(size, int) else None

    # Single-vector collection: VectorParams directly.
    size = getattr(vectors, "size", None)
    return int(size) if isinstance(size, int) else None


async def assert_dimension_compatible(
    client: Any,
    collection: str,
    expected_dim: int,
    allow_change: bool,
) -> None:
    """Fail-closed dimension check for an existing Qdrant collection.

    - Missing collection  -> no-op (factory creates it at expected_dim).
    - Dimension matches    -> no-op.
    - Dimension differs and ``allow_change`` is False -> raise
      ``DimensionMismatchError``.
    - Dimension differs and ``allow_change`` is True  -> log a loud
      warning and continue (operator opted into a reindex).

    ``client`` is an ``AsyncQdrantClient`` (or compatible). Defensively
    typed: an unreadable collection shape is treated as "can't verify"
    and does not block startup.
    """
    # Does the collection exist? Use collection_exists when available;
    # otherwise scan get_collections(). A failure to enumerate is treated
    # as "unknown" -> no-op (don't block startup on a transient error).
    exists = await _collection_exists(client, collection)
    if exists is False:
        _log.info(
            "dimension guard: collection %r absent -- will be created at dim=%d",
            collection, expected_dim,
        )
        return
    if exists is None:
        _log.warning(
            "dimension guard: could not determine if collection %r exists; "
            "skipping check",
            collection,
        )
        return

    try:
        info = await client.get_collection(collection)
    except Exception as exc:  # noqa: BLE001 -- can't-verify, don't block
        _log.warning(
            "dimension guard: get_collection(%r) failed (%s); skipping check",
            collection, exc,
        )
        return

    actual = _extract_dense_size(info)
    if actual is None:
        _log.warning(
            "dimension guard: could not read vector size for collection %r; "
            "skipping check",
            collection,
        )
        return

    if actual == expected_dim:
        _log.info(
            "dimension guard: collection %r dim=%d matches embedder",
            collection, actual,
        )
        return

    msg = (
        f"DIMENSION_MISMATCH: collection {collection!r} has vector size "
        f"{actual} but the configured embedder produces {expected_dim}-dim "
        f"vectors. A mismatch silently corrupts retrieval (and any caches / "
        f"investigations sharing this Qdrant client). Refusing to start. "
        f"Set vector_store.allow_dimension_change=true to reindex into a "
        f"fresh collection, or revert the embed-model / cloud_provider change."
    )
    if not allow_change:
        raise DimensionMismatchError(msg)
    # allow_change=True is a TRUTHFUL opt-in to a reindex: actually DROP the
    # mismatched collection so the caller's create path rebuilds it at the
    # correct dimension. Previously this only warned and continued, leaving
    # the wrong-dim collection in place -- every subsequent upsert then failed
    # with a cryptic Qdrant size error, so "allow_dimension_change" silently
    # did nothing. Drop it loudly here. Strictly behind allow_change=True.
    _log.warning(
        "dimension guard: %s -- allow_dimension_change=true: DROPPING "
        "collection %r so it is recreated at dim=%d (operator opted into a "
        "reindex; existing vectors in this collection are discarded).",
        msg, collection, expected_dim,
    )
    try:
        await client.delete_collection(collection)
        _log.warning(
            "dimension guard: dropped collection %r (was dim=%d, expected "
            "dim=%d) -- it will be recreated empty on next ensure.",
            collection, actual, expected_dim,
        )
    except Exception as exc:  # noqa: BLE001
        # If the drop fails we must NOT silently continue -- the create path
        # would see the collection still present and skip creation, leaving
        # the mismatch in place. Surface it as the same fail-closed error.
        raise DimensionMismatchError(
            f"{msg} (allow_dimension_change=true, but DROP of collection "
            f"{collection!r} failed: {exc})"
        ) from exc


async def _collection_exists(client: Any, collection: str) -> bool | None:
    """Return True/False if existence can be determined, else None."""
    checker = getattr(client, "collection_exists", None)
    if callable(checker):
        try:
            return bool(await checker(collection))
        except Exception as exc:  # noqa: BLE001
            _log.warning("dimension guard: collection_exists failed (%s)", exc)
            return None
    # Fall back to enumerating collections.
    try:
        result = await client.get_collections()
    except Exception as exc:  # noqa: BLE001
        _log.warning("dimension guard: get_collections failed (%s)", exc)
        return None
    cols = getattr(result, "collections", None) or []
    names = {getattr(c, "name", None) for c in cols}
    return collection in names
