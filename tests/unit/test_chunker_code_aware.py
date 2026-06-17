"""Code-aware parent-child chunking.

Source-code doc types get a larger parent budget (so a whole function/class
stays in one parent) and line-aware splitting (so overflow breaks on newlines,
never mid-identifier -- which would poison the BM25 lane). Char budgets are
per-content-type (config ~2.5, code ~3.5, prose ~4.0 chars/token), so a 256-tok
child is sized in tokens of THAT content type.
"""
from __future__ import annotations

from datetime import UTC, datetime

from opsrag.chunkers.parent_child import ParentChildChunker
from opsrag.interfaces.parser import DocSection, DocType, ParsedDocument
from opsrag.interfaces.scm import RepoFile


def _doc(content: str, doc_type: DocType, sections=None) -> ParsedDocument:
    rf = RepoFile(
        path="a/b.py", content=content, sha="x",
        last_modified=datetime(2026, 1, 1, tzinfo=UTC),
        repo="r/p", branch="main",
    )
    # No sections -> _wrap_whole_doc path, which keeps content verbatim
    # (no heading prepend) so the size math is clean to assert on.
    return ParsedDocument(
        content=content, doc_type=doc_type, title="T", source=rf,
        sections=sections if sections is not None else [],
    )


def _parents(chunks):
    return [c for c in chunks if c.chunk_type == "parent"]


def _children(chunks):
    return [c for c in chunks if c.chunk_type == "child"]


def test_prose_char_slice():
    # Prose budget = 1024 tok * 4.0 chars = 4096. Hard char-slice.
    content = "a" * 7000
    parents = _parents(ParentChildChunker().chunk(_doc(content, DocType.GENERIC_MARKDOWN)))
    assert len(parents[0].content) == 4096
    assert len(parents) == 2  # ceil(7000 / 4096)


def test_code_function_stays_whole():
    # ~4.4k chars: over the prose budget (1024*4.0=4096) but under the code
    # budget (2048 tok * 3.5 = 7168). Code keeps it whole; prose would split it.
    line = "    result = compute_value_for_index(idx) + base\n"
    body = (line * 90).strip()
    assert 4096 < len(body) < 7168, f"fixture out of range: {len(body)}"

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


# --- H3: overflow code children carry the enclosing symbol ------------------

def _signature_func_body(n_lines: int = 60) -> str:
    """A single Python function whose body overflows the child window so it
    splits into multiple children. The def line is the only signature."""
    line = "    total = compute_value_for_index(idx) + running_base_offset\n"
    return "def handle_inventory_webhook(idx, running_base_offset):\n" + (line * n_lines)


def test_overflow_code_child_carries_enclosing_symbol():
    # NB: re-keys code child ids -> requires a reindex (offline eval validates).
    body = _signature_func_body()
    children = _children(ParentChildChunker(child_size=64, child_overlap=8).chunk(
        _doc(body, DocType.PYTHON)
    ))
    assert len(children) >= 2, "fixture must overflow into >=2 children"
    # Child #0 starts at the signature already -> left untouched (no breadcrumb).
    assert children[0].metadata["child_index"] == 0
    assert not children[0].content.startswith("# [context]")
    assert children[0].content.lstrip().startswith("def handle_inventory_webhook")
    # EVERY overflow child (idx>=1) must contain the enclosing symbol name so
    # the BM25 lane sees it on each slice.
    for c in children[1:]:
        assert "handle_inventory_webhook" in c.content, (
            f"overflow child {c.metadata['child_index']} lost the symbol"
        )
        assert c.content.startswith("# [context]"), "breadcrumb must be prepended"


def test_overflow_code_child_uses_heading_path_breadcrumb():
    # When the parent carries a heading_path (AST/section parser symbol path),
    # the breadcrumb prefers it over the raw signature line.
    body = _signature_func_body()
    sec = DocSection(
        heading="handle_inventory_webhook",
        content=body,
        level=1,
        breadcrumb=["inventory.py", "InventoryHandler", "handle_inventory_webhook"],
    )
    children = _children(ParentChildChunker(child_size=64, child_overlap=8).chunk(
        _doc("# stub\n", DocType.PYTHON, sections=[sec])
    ))
    overflow = [c for c in children if c.metadata["child_index"] >= 1]
    assert overflow, "fixture must overflow"
    for c in overflow:
        assert "InventoryHandler > handle_inventory_webhook" in c.content


def test_prose_overflow_children_have_no_code_breadcrumb():
    # Guard: prose children are never given a code breadcrumb (byte-stable ids).
    content = ("This is a sentence about the deployment process. " * 400)
    children = _children(ParentChildChunker().chunk(_doc(content, DocType.GENERIC_MARKDOWN)))
    assert len(children) >= 2
    for c in children:
        assert not c.content.startswith("# [context]")


def test_config_overflow_children_have_no_code_breadcrumb():
    # Guard: config (line-aware but not code) gets no breadcrumb -> stable ids.
    line = "replicas: 3\nimage: registry.example.com/app:v1.2.3\n"
    content = line * 300
    children = _children(ParentChildChunker().chunk(_doc(content, DocType.KUBERNETES)))
    assert len(children) >= 2
    for c in children:
        assert not c.content.startswith("# [context]")


# --- R9: long single line must not collapse overlap / drop content ----------

def _strip_breadcrumb(content: str) -> str:
    """Drop a leading `# [context] ...` breadcrumb line (prepended to overflow
    code children, idx>=1) so coverage/overlap math sees only the real slice."""
    if content.startswith("# [context]"):
        return content.split("\n", 1)[1] if "\n" in content else ""
    return content


def _line_aware_offsets(chunker: ParentChildChunker, body, doc_type):
    """Re-run the line-aware window math (matching _split_children) to recover
    each child's (start, end) span over the parent body. Lets the test assert on
    coverage + overlap directly, independent of the breadcrumb prefix."""
    child_chars, child_overlap_chars = chunker._child_chars_for(
        _doc(body, doc_type)
    )
    spans = []
    start = 0
    last_emit_end = None
    while start < len(body):
        end = min(start + child_chars, len(body))
        if end < len(body):
            nl = body.rfind("\n", start + 1, end + 1)
            if nl > start:
                end = nl + 1
        if body[start:end].strip() and end != last_emit_end:
            spans.append((start, end))
            last_emit_end = end
        if end == len(body):
            break
        nxt = end - child_overlap_chars
        if nxt > start:
            start = nxt
        else:
            min_overlap = max(1, min(child_overlap_chars, end - start - 1))
            start = end - min_overlap
    return spans, child_chars, child_overlap_chars


def test_long_line_keeps_overlap_and_drops_no_content():
    """A code block with a line longer than child_overlap_chars still yields
    OVERLAPPING children with NO dropped content.

    R9: in the line-aware child path, when a single line is longer than
    child_overlap_chars the newline-snap pins the window end at/before where the
    overlap would begin. The old advance fell back to a fixed `step`, which
    jumped PAST that end -- silently dropping the un-emitted region AND collapsing
    the overlap to a gap exactly there, so a BM25 query straddling the boundary
    missed. The clamp must preserve a minimum overlap and never skip content.

    NB: re-keys code/config child ids -> requires a reindex (offline eval over a
    fresh index validates it).
    """
    # child_size=16, code chars/token ~3.5 -> child_chars ~56, overlap ~28. The
    # long line (a URL with VARIED chars, ~349 chars) is far longer than both the
    # window AND the overlap, forcing the snap-pinned / mid-line-cut path. Varied
    # (not a single repeated char) so each window slice is distinct.
    payload = "".join(f"k{i}=v{i}&" for i in range(40))
    long_line = f"    url = build('https://api.example.com/path?{payload}')\n"
    body = (
        "def render_settings(ctx):\n"
        + long_line
        + "    return url\n"
        "another = 1\n"
    )
    chunker = ParentChildChunker(child_size=16, child_overlap=8)
    children = _children(chunker.chunk(_doc(body, DocType.PYTHON)))
    assert len(children) >= 2, "fixture must overflow into multiple children"

    # (1) NO DROPPED CONTENT: the union of child spans must cover EVERY char of
    # the parent body (the old `step` fallback left a hole at the long-line seam).
    spans, child_chars, overlap_chars = _line_aware_offsets(
        chunker, body, DocType.PYTHON
    )
    assert overlap_chars > 0
    covered = [False] * len(body)
    for s, e in spans:
        for i in range(s, e):
            covered[i] = True
    dropped = [i for i, c in enumerate(covered) if not c]
    assert not dropped, (
        f"content dropped at offsets {dropped[:8]}...: "
        f"{''.join(body[i] for i in dropped)!r}"
    )

    # (2) OVERLAPPING CHILDREN: every consecutive pair of spans overlaps by at
    # least one char (the boundary stays searchable for BM25). The buggy code
    # produced a NEGATIVE overlap (a gap) at the short-line -> long-line seam.
    overlaps = [spans[k - 1][1] - spans[k][0] for k in range(1, len(spans))]
    assert overlaps, "fixture must produce >=2 spans"
    assert min(overlaps) >= 1, f"overlap collapsed to a gap: {overlaps}"

    # And the long line's content survives across the cut: pick a substring that
    # straddles the original short-line/long-line boundary and assert SOME child
    # contains it whole (overlap carries the seam).
    seam = "url = build"
    assert any(seam in _strip_breadcrumb(c.content) for c in children), (
        "seam content lost at the long-line boundary"
    )


def test_long_line_no_duplicate_suffix_children():
    """The clamp must not emit a burst of shrinking suffix-duplicate children.

    When the newline-snap pins the window end while the overlap-advance creeps
    `start` forward, a window whose end equals the previously-emitted child's end
    carries no new content and is skipped. Guard against a regression that would
    re-emit a burst of suffix duplicates (index bloat) at the seam.
    """
    payload = "&".join(f"field{i}=value{i}" for i in range(30))  # varied
    long_line = f"    body = encode({payload})\n"
    body = "def f():\n" + long_line + "    return body\n"
    chunker = ParentChildChunker(child_size=12, child_overlap=8)
    children = _children(chunker.chunk(_doc(body, DocType.PYTHON)))
    contents = [_strip_breadcrumb(c.content) for c in children]
    # No child may be a strict suffix of the immediately-preceding child (the
    # shrinking-suffix burst signature). Identical-content slices are impossible
    # here because the long line has varied characters.
    for a, b in zip(contents, contents[1:]):
        assert not (a.endswith(b) and a != b), (
            f"suffix-duplicate child emitted at seam: {b!r}"
        )


def test_long_line_terminates_on_no_newline_block():
    """A parent that is one enormous newline-free line must terminate and cover
    all content with overlap (no infinite loop, no dropped tail)."""
    # Varied content so windows are distinct; > child window so it overflows.
    body = "".join(f"seg{i:03d}_" for i in range(120))  # ~840 chars, no newline
    chunker = ParentChildChunker(child_size=16, child_overlap=8)
    spans, _, overlap_chars = _line_aware_offsets(chunker, body, DocType.PYTHON)
    assert len(spans) >= 2
    covered = [False] * len(body)
    for s, e in spans:
        for i in range(s, e):
            covered[i] = True
    assert all(covered), "content dropped from newline-free block"
    overlaps = [spans[k - 1][1] - spans[k][0] for k in range(1, len(spans))]
    assert min(overlaps) >= 1, f"overlap collapsed in newline-free block: {overlaps}"
