"""Parent-child chunker.

Creates two layers:
- Parent chunks: one per document section (heading + body), used as generation context
- Child chunks: smaller windows drawn from each parent, used for vector search

Searching on children gives precise hits; substituting their parent at
generation time gives the LLM enough surrounding context.
"""
from __future__ import annotations

import hashlib
import re

from opsrag.ingestion.metadata import content_hash as _content_hash
from opsrag.interfaces.chunker import Chunk
from opsrag.interfaces.parser import DocType, ParsedDocument
from opsrag.tokenization import chars_per_token_for, estimate_tokens

# A fenced-code marker: a line that STARTS (ignoring leading whitespace) with
# at least three backticks or three tildes. Matched at line boundaries so an
# inline `code` span or a mid-line ``` never registers as a fence open/close.
# Used to keep ``` blocks atomic when slicing prose (M5): a runbook command
# block must not be split as prose -- a '. ' inside a command or a '# comment'
# line inside the fence must not be read as a sentence/heading boundary.
_FENCE_RE = re.compile(r"^[ \t]*(?:`{3,}|~{3,})", re.MULTILINE)

# Sizing targets (`child_size` / `parent_max_tokens`) are expressed in tokens;
# the char budget for the slicer is `tokens * chars_per_token_for(doc.doc_type)`,
# computed PER-DOC (config ~2.5, code ~3.5, prose ~4.0) so a 256-token child is
# actually ~256 tokens of THAT content type rather than 256 tokens of "average"
# text. See opsrag.tokenization for the ratios + the re-index caveat.

# Source-code doc types. These get (a) a larger parent budget so a whole
# function/class (the AST parser emits one section per def) stays in a single
# parent instead of being hard-cut, and (b) line-aware splitting when a def
# does exceed the budget, so we break on newlines rather than mid-line /
# mid-identifier. Both matter for retrieval: a parent sliced through the middle
# of `def handle_webhook(` poisons the dense embedding, and a child sliced
# through an identifier poisons the BM25 lexical lane that the hybrid retriever
# now leans on for exact symbol matches.
_CODE_DOC_TYPES = frozenset(
    {
        DocType.PYTHON,
        DocType.JAVASCRIPT,
        DocType.TYPESCRIPT,
        DocType.GO,
        DocType.JAVA,
        DocType.SHELL,
    }
)

# Doc types that must split on LINE boundaries, never on prose sentences. This
# is CODE plus structured CONFIG (YAML/HCL/k8s/Dockerfile/alert): config has no
# sentences, so the prose snapper would cut `replicas: 3` from its key on a `. `
# inside a value and poison both the dense vector and the BM25 lexical lane --
# exactly the exact-match-sensitive content. (Separate from _CODE_DOC_TYPES,
# which ALSO grants the larger parent budget so a whole function stays intact;
# config keeps the normal budget but still wants newline-aware splitting.)
_LINE_AWARE_DOC_TYPES = _CODE_DOC_TYPES | frozenset(
    {
        DocType.KUBERNETES,
        DocType.TERRAFORM,
        DocType.HELM,
        DocType.YAML_CONFIG,
        DocType.DOCKERFILE,
        DocType.ALERT_DEFINITION,
    }
)


class ParentChildChunker:
    def __init__(
        self,
        child_size: int = 256,
        child_overlap: int = 32,
        parent_max_tokens: int = 1024,
        code_parent_max_tokens: int = 2048,
    ):
        if child_overlap >= child_size:
            raise ValueError("child_overlap must be smaller than child_size")
        self.child_size = child_size
        self.child_overlap = child_overlap
        self.parent_max_tokens = parent_max_tokens
        # Code parents get a bigger budget so most functions/classes fit whole.
        self.code_parent_max_tokens = code_parent_max_tokens

    def _child_chars_for(self, doc: ParsedDocument) -> tuple[int, int]:
        """(child_char_budget, child_overlap_chars) for this doc's content type."""
        cpt = chars_per_token_for(doc.doc_type)
        return int(self.child_size * cpt), int(self.child_overlap * cpt)

    def _parent_max_chars_for(self, doc: ParsedDocument) -> int:
        """Char budget for a parent piece -- larger for source code, scaled by
        the doc's content-type chars/token ratio."""
        cpt = chars_per_token_for(doc.doc_type)
        tokens = (
            self.code_parent_max_tokens
            if doc.doc_type in _CODE_DOC_TYPES
            else self.parent_max_tokens
        )
        return int(tokens * cpt)

    def _split_parent_text(self, text: str, max_chars: int, doc: ParsedDocument) -> list[str]:
        """Split a section body into parent-sized pieces.

        Prose uses a fixed char-slice; code AND config split on line boundaries
        so an overflow breaks between lines rather than mid-line (a YAML key:value
        pair must not be cut). ``max_chars`` is the per-doc-type budget (see
        _parent_max_chars_for), so output is NOT byte-identical to the old flat-3
        sizing -- a re-index is required for the new boundaries to take effect.
        """
        if doc.doc_type in _LINE_AWARE_DOC_TYPES:
            return self._split_by_lines(text, max_chars)
        return self._split_by_char_size(text, max_chars)

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        parents = self._build_parents(doc)
        if not parents:
            return []

        out: list[Chunk] = []
        for parent in parents:
            out.append(parent)
            out.extend(self._split_children(parent, doc))

        # Positional + dedup facets (generalized chunk metadata). Assigned
        # over the full emitted list so chunk_index/chunk_count reflect the
        # document's true ordering. content_hash enables dedup / idempotent
        # upsert. Additive only -- does not affect IDs, parent linkage, or
        # vectors.
        total = len(out)
        for i, c in enumerate(out):
            c.metadata["chunk_index"] = i
            c.metadata["chunk_count"] = total
            c.metadata.setdefault("content_hash", _content_hash(c.content))
            # Mirror the dataclass token_count into metadata so the facet is
            # visible in the payload's nested `metadata` dict alongside the
            # other positional facets (the top-level payload `token_count`
            # is unchanged).
            c.metadata.setdefault("token_count", c.token_count)
        return out

    def _build_parents(self, doc: ParsedDocument) -> list[Chunk]:
        sections = doc.sections or []
        if not sections:
            return self._wrap_whole_doc(doc)

        parents: list[Chunk] = []
        parent_max_chars = self._parent_max_chars_for(doc)
        for i, section in enumerate(sections):
            body = (f"# {section.heading}\n\n{section.content}").strip()
            if not body:
                continue
            for j, piece in enumerate(self._split_parent_text(body, parent_max_chars, doc)):
                pid = self._make_id(doc, f"parent-{i}-{j}", piece)
                parents.append(
                    Chunk(
                        id=pid,
                        content=piece,
                        doc_type=doc.doc_type,
                        source_path=doc.source.path,
                        repo=doc.source.repo,
                        metadata={
                            **doc.metadata,
                            "title": doc.title,
                            "section_heading": section.heading,
                            "section_type": section.section_type,
                            "section_level": section.level,
                            # Breadcrumb: doc title -> full heading ancestry
                            # (H1 -> H2 -> H3) when the parser supplies it, else
                            # doc title -> section heading. Children inherit it
                            # via parent.metadata; gives an H3 chunk its H1/H2
                            # scope for retrieval.
                            "heading_path": [
                                h for h in (
                                    [doc.title] + (section.breadcrumb or [section.heading])
                                ) if h
                            ],
                        },
                        parent_chunk_id=None,
                        chunk_type="parent",
                        token_count=estimate_tokens(piece, doc.doc_type),
                    )
                )
        return parents or self._wrap_whole_doc(doc)

    def _wrap_whole_doc(self, doc: ParsedDocument) -> list[Chunk]:
        text = doc.content.strip()
        if not text:
            return []
        chunks: list[Chunk] = []
        parent_max_chars = self._parent_max_chars_for(doc)
        for j, piece in enumerate(self._split_parent_text(text, parent_max_chars, doc)):
            pid = self._make_id(doc, f"parent-0-{j}", piece)
            chunks.append(
                Chunk(
                    id=pid,
                    content=piece,
                    doc_type=doc.doc_type,
                    source_path=doc.source.path,
                    repo=doc.source.repo,
                    metadata={
                        **doc.metadata,
                        "title": doc.title,
                        "heading_path": [doc.title] if doc.title else [],
                    },
                    parent_chunk_id=None,
                    chunk_type="parent",
                    token_count=estimate_tokens(piece, doc.doc_type),
                )
            )
        return chunks

    def _split_children(self, parent: Chunk, doc: ParsedDocument) -> list[Chunk]:
        text = parent.content
        out: list[Chunk] = []
        start = 0
        idx = 0
        # End offset of the last child we EMITTED. When the line-aware
        # newline-snap (below) keeps pinning the window end to the same newline
        # while the clamped overlap-advance creeps `start` forward, successive
        # windows would re-emit shrinking suffixes of the same piece. We skip a
        # window whose snapped end equals the previously-emitted end: it adds no
        # new content. Span-based (not content-based) so all-identical lines are
        # unaffected; never skips coverage (the advance still guarantees it).
        last_emit_end: int | None = None
        child_chars, child_overlap_chars = self._child_chars_for(doc)
        is_line_aware = doc.doc_type in _LINE_AWARE_DOC_TYPES
        # For CODE only: a breadcrumb of the enclosing symbol, prepended to
        # every OVERFLOW child (idx>=1) so a slice taken from the middle of a
        # function/class still carries the def line into the BM25 lexical lane.
        # Child #0 already starts at the signature, so it is left UNTOUCHED.
        # Computed once per parent; None for non-code so prose/config children
        # stay byte-identical (stable ids). See H3.
        code_header = (
            self._code_header_for(parent, text)
            if doc.doc_type in _CODE_DOC_TYPES
            else None
        )
        while start < len(text):
            end = min(start + child_chars, len(text))
            # For code/config, snap the window end back to a newline so a child
            # never cuts through an identifier (`handle_web|hook`) or a YAML
            # key:value pair -- mid-symbol slices poison the BM25 lexical lane
            # that exact-symbol queries depend on. Parents are already
            # line-aware; this brings the embedded+indexed child layer in line.
            if is_line_aware and end < len(text):
                nl = text.rfind("\n", start + 1, end + 1)
                if nl > start:
                    end = nl + 1  # keep the trailing newline with the piece
                # else: no newline inside the window -> a SINGLE line is longer
                # than child_chars (minified JS, a long YAML value, a generated
                # one-liner). We must NOT snap; the window cuts mid-line here.
                # The overlap advance below (clamped) still carries a min overlap
                # into the next window so a BM25 query straddling that cut still
                # matches -- see R9.
            elif end < len(text):
                # Prose: snap the window end back to a paragraph break, then a
                # sentence end, so a child doesn't cut mid-sentence (which
                # weakens the child embedding). Never snap a boundary INTO a
                # fenced code block (```/~~~) -- a ``` runbook command block
                # must stay atomic so a '. ' inside a command line or a '#'
                # comment isn't treated as a prose sentence/heading boundary.
                # Only snap if it doesn't shrink the window below ~60%.
                floor = start + int(child_chars * 0.6)
                fence_end = self._fence_boundary_at(text, start, end)
                if fence_end is not None:
                    # `end` landed inside a fenced block: extend to the closing
                    # fence (bounded). _fence_boundary_at returns a safe end.
                    end = fence_end
                else:
                    para = text.rfind("\n\n", floor, end)
                    if para > start:
                        end = para + 2
                    else:
                        sent = max(
                            text.rfind(". ", floor, end),
                            text.rfind("! ", floor, end),
                            text.rfind("? ", floor, end),
                            text.rfind(".\n", floor, end),
                        )
                        if sent > start:
                            end = sent + 1
            piece = text[start:end].strip()
            # Skip a window that ends exactly where the last emitted child ended:
            # it is a pure suffix produced by the clamped overlap-advance creeping
            # against a pinned newline snap, so it carries no new content. The
            # advance still runs below, so coverage is unaffected -- only the
            # redundant emit is dropped. (Never fires for prose / normal code,
            # where `end` strictly increases -> byte-identical, stable ids.)
            if piece and end != last_emit_end:
                # Prepend the enclosing-symbol breadcrumb to overflow code
                # children (idx>=1) so BM25 sees the def on every slice. Child
                # #0 already starts at the signature, so it stays untouched.
                if code_header and idx >= 1:
                    piece = f"{code_header}\n{piece}"
                cid = self._make_id(doc, f"child-{parent.id}-{idx}", piece)
                out.append(
                    Chunk(
                        id=cid,
                        content=piece,
                        doc_type=doc.doc_type,
                        source_path=doc.source.path,
                        repo=doc.source.repo,
                        metadata={**parent.metadata, "child_index": idx},
                        parent_chunk_id=parent.id,
                        chunk_type="child",
                        token_count=estimate_tokens(piece, doc.doc_type),
                    )
                )
                idx += 1
                last_emit_end = end
            if end == len(text):
                break
            # Advance with overlap measured from the (possibly snapped) end, so
            # newline-snapping never leaves a gap. For prose (no snap) `end` is
            # never pulled below `floor = start + 0.6*child_chars`, so
            # `end - child_overlap_chars > start` always holds and the next start
            # is exactly `start + (child_chars - child_overlap_chars)` -- a fixed
            # step, byte-identical to the previous behaviour -> stable chunk IDs.
            nxt = end - child_overlap_chars
            if nxt > start:
                start = nxt
            else:
                # R9: the snap pulled `end` back to (or before) where the
                # overlap would begin -- this happens in the LINE-AWARE path
                # when a single line is longer than child_overlap_chars (so the
                # newline-snap lands at/before `start + child_overlap_chars`).
                # The old fallback (`start += child_chars - child_overlap_chars`)
                # would jump PAST `end`, silently DROPPING the un-emitted region
                # after `end` and collapsing the overlap to none right where a
                # BM25 boundary-straddling query needs it.
                #
                # Instead, advance to just inside the current piece so the next
                # window (a) starts no later than `end` -> never skips content,
                # and (b) retains a minimum overlap (clamped to >=1 char, capped
                # at the configured overlap) carried back from `end`. This keeps
                # the prior line's tail / the cut long-line's head in the next
                # child so the boundary stays searchable.
                min_overlap = max(1, min(child_overlap_chars, end - start - 1))
                start = end - min_overlap
        return out

    # Compact breadcrumb (heading_path ` > ` joined) + the enclosing signature
    # line, capped so it never dwarfs the slice. Used only for overflow code
    # children -- see _split_children / H3.
    _HEADER_MAX_CHARS = 200
    # A "signature" line for the common code languages we ingest: the first
    # non-blank line that opens a symbol. Cheap heuristic (no AST) -- the
    # parent already starts at the def, so text's first matching line is the
    # enclosing symbol for the whole parent.
    _SIGNATURE_RE = re.compile(
        r"^\s*(?:export\s+|public\s+|private\s+|protected\s+|static\s+|async\s+|"
        r"final\s+|abstract\s+)*"
        r"(?:def|class|func|function|interface|type|struct|impl|trait|"
        r"fn|sub|module|package)\b.*",
        re.MULTILINE,
    )

    def _code_header_for(self, parent: Chunk, text: str) -> str | None:
        """Compact enclosing-symbol breadcrumb for overflow code children.

        The enclosing SYMBOL is the most load-bearing token for the BM25 lane,
        so prefer the first signature line in the parent body (the def/class/
        func line -- the parent starts at it). Fall back to the parent's
        heading_path (the AST/section parser's symbol path) when no signature
        line is recognizable. Returns None when neither is available so the
        caller skips prepending. Capped to _HEADER_MAX_CHARS so a long
        signature can't swamp the slice.
        """
        crumb = ""
        m = self._SIGNATURE_RE.search(text)
        if m:
            crumb = m.group(0).strip()
        path = [str(h) for h in ((parent.metadata or {}).get("heading_path") or []) if h]
        if path:
            # Drop the doc title (path[0]) so the breadcrumb is the symbol
            # ancestry, not "DocTitle > ...". A single-element path is just the
            # title -> no symbol scope (path[1:] == []), never leak the title.
            scope = path[1:]
            scope_str = " > ".join(scope)
            if not crumb:
                crumb = scope_str
            elif scope_str and scope_str not in crumb:
                crumb = f"{scope_str} > {crumb}"
        if not crumb:
            return None
        crumb = crumb.replace("\n", " ").strip()
        if len(crumb) > self._HEADER_MAX_CHARS:
            crumb = crumb[: self._HEADER_MAX_CHARS].rstrip() + "..."
        # Render as a language-neutral comment so it reads as context, not code
        # that would (e.g.) confuse a downstream syntax view. BM25 tokenizes the
        # symbol regardless of the leading `# `.
        return f"# [context] {crumb}"

    # Hard cap on how far past the target window we extend to keep a fenced
    # block atomic before giving up and splitting on newlines inside it. Keeps a
    # pathologically huge fence from producing one enormous child. (~4x the
    # default child window of 256 tok * 4 chars.)
    _FENCE_EXTEND_CAP = 4096

    @staticmethod
    def _fence_boundary_at(text: str, start: int, end: int) -> int | None:
        """If a prose child boundary at ``end`` would land INSIDE a fenced code
        block, return a safe end that keeps the fence atomic; else None.

        Scans the fence markers (```/~~~ at line start) within [start, ...].
        If ``end`` falls between an opening fence and its matching close, extend
        ``end`` to just past the close (bounded by _FENCE_EXTEND_CAP); if the
        close is beyond the cap, snap back to the LAST newline before the
        opening fence so the fence starts a fresh child instead of being cut.
        Returns None when ``end`` is not inside any fence (normal prose path).
        """
        # Find fenced spans starting at/after `start`. A fence opens with a line
        # that begins with ``` or ~~~ and closes with the next such line.
        pos = start
        n = len(text)
        while pos < end:
            fo = _FENCE_RE.search(text, pos)
            if fo is None or fo.start() >= end:
                return None  # no fence opens before `end`
            open_start = fo.start()
            # CommonMark: a fence is closed only by the SAME marker char with a
            # run >= the opening length and nothing but whitespace after -- a
            # ``` block is NOT closed by ~~~ (and vice-versa). Build a per-open
            # close matcher instead of the marker-agnostic _FENCE_RE.
            open_run = text[fo.start():fo.end()].lstrip(" \t")
            close_re = re.compile(
                rf"(?m)^[ \t]*{re.escape(open_run[0])}{{{len(open_run)},}}[ \t]*$"
            )
            # Find the matching close fence after the opening line.
            line_end = text.find("\n", fo.end())
            search_from = (line_end + 1) if line_end != -1 else n
            fc = close_re.search(text, search_from)
            if fc is None:
                # Unterminated fence: treat the rest of the parent as fenced.
                close_end = n
            else:
                ce = text.find("\n", fc.end())
                close_end = (ce + 1) if ce != -1 else n
            if end <= open_start:
                return None  # boundary is before this fence -> normal prose
            if open_start < end < close_end:
                # `end` is inside this fence. Try to extend past the close.
                if close_end - start <= ParentChildChunker._FENCE_EXTEND_CAP:
                    return close_end
                # Fence too big to keep whole within the cap: start the fence in
                # its own child by snapping back to just before the opening line.
                if open_start > start:
                    return open_start
                # Fence opens at the very start and exceeds the cap: split on a
                # newline inside it (never mid-line) rather than mid-sentence.
                nl = text.rfind("\n", start + 1, end + 1)
                return (nl + 1) if nl > start else end
            # `end` is at/after this fence's close -> keep scanning for the next.
            pos = close_end
        return None

    @staticmethod
    def _split_by_char_size(text: str, max_chars: int) -> list[str]:
        if len(text) <= max_chars:
            return [text]
        pieces: list[str] = []
        i = 0
        n = len(text)
        while i < n:
            end = min(i + max_chars, n)
            if end < n:
                # Never snap a parent boundary inside a fenced code block --
                # extend to the closing fence (bounded by _FENCE_EXTEND_CAP,
                # then fall back to a newline inside the fence). Keeps a runbook
                # ``` command block atomic at the parent layer too.
                fence_end = ParentChildChunker._fence_boundary_at(text, i, end)
                if fence_end is not None and fence_end > i:
                    end = fence_end
            pieces.append(text[i:end])
            i = end
        return pieces or [text]

    @staticmethod
    def _split_by_lines(text: str, max_chars: int) -> list[str]:
        """Line-aware split for source code.

        Greedily packs whole lines into pieces up to ``max_chars`` so breaks
        land on newlines, never mid-line. A single line longer than the budget
        (rare -- minified JS, generated code) falls back to a hard char-slice
        for that line only so the budget is still honored. Newlines between
        packed lines are preserved.
        """
        if len(text) <= max_chars:
            return [text]
        pieces: list[str] = []
        cur: list[str] = []
        cur_len = 0
        for line in text.splitlines(keepends=True):
            if len(line) > max_chars:
                # Flush what we have, then hard-slice the oversized line.
                if cur:
                    pieces.append("".join(cur))
                    cur, cur_len = [], 0
                for i in range(0, len(line), max_chars):
                    pieces.append(line[i : i + max_chars])
                continue
            if cur_len + len(line) > max_chars and cur:
                pieces.append("".join(cur))
                cur, cur_len = [], 0
            cur.append(line)
            cur_len += len(line)
        if cur:
            pieces.append("".join(cur))
        # Match _split_by_char_size's contract: pieces are used verbatim as
        # chunk content (they get .strip()'d downstream at child level, and
        # parents are stored as-is). Drop any empty trailing piece.
        return [p for p in pieces if p] or [text]

    @staticmethod
    def _make_id(doc: ParsedDocument, tag: str, content: str) -> str:
        # Hash the FULL content, not a 64-char prefix: two distinct chunks
        # that share the first 64 chars (boilerplate headers, license blocks,
        # an edit past char 64) would otherwise collide onto the same id and
        # silently overwrite each other / leave a stale vector behind.
        # NOTE: changing this scheme re-keys every chunk -> requires a reindex.
        h = hashlib.sha1(
            f"{doc.source.repo}:{doc.source.path}:{tag}:{content}".encode()
        ).hexdigest()[:16]
        return f"{doc.source.repo}:{doc.source.path}:{tag}:{h}"
