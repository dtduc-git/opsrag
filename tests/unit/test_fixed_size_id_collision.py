"""Fixed-size chunk id seeding -- full-content hash, no 64-char-prefix collision.

`FixedSizeChunker._make_id` used to seed the chunk-id hash with `content[:64]`,
so two distinct windows sharing a 64-char prefix (boilerplate headers, license
blocks, an edit past char 64) at the same index collided onto one id and
silently overwrote each other in Qdrant. It now hashes the FULL content,
mirroring `ParentChildChunker._make_id` exactly.

These tests pin the fix (prefix-sharing windows get DISTINCT ids) and prove
behaviour-equivalence on the unchanged dimensions: ids stay deterministic,
keep the `repo:path:idx:<16hex>` shape, and identical (content, idx) still
maps to the SAME id (same chunk == same id is intended).
"""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from opsrag.chunkers.fixed_size import FixedSizeChunker
from opsrag.chunkers.parent_child import ParentChildChunker
from opsrag.interfaces.parser import DocType, ParsedDocument
from opsrag.interfaces.scm import RepoFile


def _doc(content: str = "", doc_type: DocType = DocType.GENERIC_MARKDOWN) -> ParsedDocument:
    rf = RepoFile(
        path="a/b.py", content=content, sha="x",
        last_modified=datetime(2026, 1, 1, tzinfo=UTC),
        repo="r/p", branch="main",
    )
    return ParsedDocument(content=content, doc_type=doc_type, title="T", source=rf, sections=[])


def test_same_64char_prefix_distinct_tails_get_distinct_ids():
    """The core fix: two windows that share their first 64 chars but differ in
    the tail -- forced to the SAME idx -- must NOT collide onto one id."""
    doc = _doc()
    prefix = "x" * 64
    a = prefix + "TAIL-AAAAAAAAAAAA"
    b = prefix + "TAIL-BBBBBBBBBBBB"
    assert a[:64] == b[:64] and a != b  # fixture sanity

    id_a = FixedSizeChunker._make_id(doc, 0, a)
    id_b = FixedSizeChunker._make_id(doc, 0, b)

    assert id_a != id_b, "prefix-sharing windows at the same idx must get distinct ids"


def test_old_prefix_seed_would_have_collided():
    """Document the bug being fixed: the OLD `content[:64]` seed produced an
    IDENTICAL hash for the two prefix-sharing windows, while the new full-content
    seed does not."""
    doc = _doc()
    prefix = "x" * 64
    a = prefix + "TAIL-AAAAAAAAAAAA"
    b = prefix + "TAIL-BBBBBBBBBBBB"

    def _old_id(d, idx, content):
        h = hashlib.sha1(
            f"{d.source.repo}:{d.source.path}:{idx}:{content[:64]}".encode()
        ).hexdigest()[:16]
        return f"{d.source.repo}:{d.source.path}:{idx}:{h}"

    # Old scheme: COLLISION (the bug).
    assert _old_id(doc, 0, a) == _old_id(doc, 0, b)
    # New scheme: distinct (the fix).
    assert FixedSizeChunker._make_id(doc, 0, a) != FixedSizeChunker._make_id(doc, 0, b)


def test_id_seeds_full_content_mirroring_parent_child():
    """The new fixed-size id seed hashes the FULL content with the same
    `repo:path:<tag>:content` recipe parent_child uses, so for matching
    (repo, path, tag, content) both chunkers derive the identical 16-hex hash."""
    doc = _doc()
    content = ("y" * 200) + " unique tail"
    idx = 3

    expected_hash = hashlib.sha1(
        f"{doc.source.repo}:{doc.source.path}:{idx}:{content}".encode()
    ).hexdigest()[:16]
    expected = f"{doc.source.repo}:{doc.source.path}:{idx}:{expected_hash}"

    assert FixedSizeChunker._make_id(doc, idx, content) == expected
    # Same recipe as parent_child (tag == idx here): identical hash component.
    pc_id = ParentChildChunker._make_id(doc, idx, content)
    assert pc_id.rsplit(":", 1)[-1] == expected_hash


def test_id_is_deterministic_and_shaped():
    """Behaviour-equivalence on unchanged dimensions: same (content, idx) ->
    same id (stable / idempotent upsert), and the `repo:path:idx:<16hex>`
    shape is preserved."""
    doc = _doc()
    content = "some chunk body that is comfortably longer than sixty-four characters!!"
    assert len(content) > 64

    id1 = FixedSizeChunker._make_id(doc, 7, content)
    id2 = FixedSizeChunker._make_id(doc, 7, content)
    assert id1 == id2  # deterministic -- same chunk keeps the same id

    head, h = id1.rsplit(":", 1)
    assert head == f"{doc.source.repo}:{doc.source.path}:7"
    assert len(h) == 16 and all(c in "0123456789abcdef" for c in h)


def test_different_idx_gives_different_id():
    """idx still participates in the seed: same content at a different window
    index is a different chunk -> different id (unchanged behaviour)."""
    doc = _doc()
    content = "z" * 100
    assert FixedSizeChunker._make_id(doc, 0, content) != FixedSizeChunker._make_id(doc, 1, content)


def test_chunk_pipeline_prefix_sharing_windows_distinct_ids():
    """End-to-end through `.chunk()`: a document whose successive windows share a
    long identical prefix but diverge in the tail yields all-distinct chunk ids.

    The window stride is small so adjacent windows overlap heavily (shared
    prefix > 64 chars); only the differing tail keeps the ids apart."""
    # Long run of identical chars (shared prefix across windows) + a per-region
    # marker so window tails differ. Small chunk_size forces many windows.
    body = "".join(f"{'p' * 120}<<MARK{i:03d}>>" for i in range(40))
    chunker = FixedSizeChunker(chunk_size=40, overlap=8)
    chunks = chunker.chunk(_doc(body, DocType.GENERIC_MARKDOWN))

    assert len(chunks) > 1
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "no two windows may share an id"
