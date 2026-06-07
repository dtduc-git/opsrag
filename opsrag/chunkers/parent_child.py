"""Parent-child chunker.

Creates two layers:
- Parent chunks: one per document section (heading + body), used as generation context
- Child chunks: smaller windows drawn from each parent, used for vector search

Searching on children gives precise hits; substituting their parent at
generation time gives the LLM enough surrounding context.
"""
from __future__ import annotations

import hashlib

from opsrag.ingestion.metadata import content_hash as _content_hash
from opsrag.interfaces.chunker import Chunk
from opsrag.interfaces.parser import DocType, ParsedDocument
from opsrag.tokenization import chars_per_token_for, estimate_tokens

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
        child_chars, child_overlap_chars = self._child_chars_for(doc)
        step = child_chars - child_overlap_chars
        is_line_aware = doc.doc_type in _LINE_AWARE_DOC_TYPES
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
            elif end < len(text):
                # Prose: snap the window end back to a paragraph break, then a
                # sentence end, so a child doesn't cut mid-sentence (which
                # weakens the child embedding). Only snap if it doesn't shrink
                # the window below ~60% -- otherwise keep the char boundary.
                floor = start + int(child_chars * 0.6)
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
            if piece:
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
            if end == len(text):
                break
            # Advance with overlap measured from the (possibly snapped) end, so
            # newline-snapping never leaves a gap. For prose (no snap) this is
            # exactly `start += step` -- byte-identical output, stable chunk IDs.
            nxt = end - child_overlap_chars
            start = nxt if nxt > start else start + step
        return out

    @staticmethod
    def _split_by_char_size(text: str, max_chars: int) -> list[str]:
        if len(text) <= max_chars:
            return [text]
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

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
        h = hashlib.sha1(
            f"{doc.source.repo}:{doc.source.path}:{tag}:{content[:64]}".encode()
        ).hexdigest()[:16]
        return f"{doc.source.repo}:{doc.source.path}:{tag}:{h}"
