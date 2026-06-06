"""Code-aware parent-child chunking.

Source-code doc types get a larger parent budget (so a whole function/class
stays in one parent) and line-aware splitting (so overflow breaks on newlines,
never mid-identifier -- which would poison the BM25 lane). Prose paths must be
byte-identical to before, so prose chunk IDs / vectors don't churn.
"""
from __future__ import annotations

from datetime import UTC, datetime

from opsrag.chunkers.parent_child import ParentChildChunker
from opsrag.interfaces.parser import DocType, ParsedDocument
from opsrag.interfaces.scm import RepoFile


def _doc(content: str, doc_type: DocType) -> ParsedDocument:
    rf = RepoFile(
        path="a/b.py", content=content, sha="x",
        last_modified=datetime(2026, 1, 1, tzinfo=UTC),
        repo="r/p", branch="main",
    )
    # No sections -> _wrap_whole_doc path, which keeps content verbatim
    # (no heading prepend) so the size math is clean to assert on.
    return ParsedDocument(content=content, doc_type=doc_type, title="T", source=rf, sections=[])


def _parents(chunks):
    return [c for c in chunks if c.chunk_type == "parent"]


def test_prose_char_slice_unchanged():
    # Prose budget = 1024 tok * 3 chars = 3072. Hard char-slice, unchanged.
    content = "a" * 7000
    parents = _parents(ParentChildChunker().chunk(_doc(content, DocType.GENERIC_MARKDOWN)))
    assert len(parents[0].content) == 3072
    assert len(parents) == 3  # ceil(7000 / 3072)


def test_code_function_stays_whole():
    # ~4.4k chars: over the prose budget (3072) but under the code budget
    # (2048 tok * 3 = 6144). Code keeps it whole; prose would split it.
    line = "    result = compute_value_for_index(idx) + base\n"
    body = (line * 90).strip()
    assert 3072 < len(body) < 6144, f"fixture out of range: {len(body)}"

    code_parents = _parents(ParentChildChunker().chunk(_doc(body, DocType.PYTHON)))
    prose_parents = _parents(ParentChildChunker().chunk(_doc(body, DocType.GENERIC_MARKDOWN)))

    assert len(code_parents) == 1, "code: whole function should stay in one parent"
    assert len(prose_parents) >= 2, "prose: should split by the smaller budget"


def test_code_overflow_splits_on_line_boundaries():
    line = "    result = compute_value_for_index(idx) + base_offset\n"
    body = (line * 200).strip()  # ~11k chars, over the 6144 code budget
    parents = _parents(ParentChildChunker().chunk(_doc(body, DocType.PYTHON)))

    assert len(parents) >= 2
    # Lossless: parents reconstruct the (stripped) body.
    assert "".join(p.content for p in parents) == body
    # Every non-final piece ends on a newline -> no line was cut in half.
    for p in parents[:-1]:
        assert p.content.endswith("\n")
