"""Code-structure extraction -- split source files into one DocSection per
top-level function / class / method so the enclosing-def name flows into
the chunker's `section_heading` metadata, then into contextual chunking's
prefix line.

Effect on retrieval
-------------------
Before this module, a Python chunk like:
    def _resolve_user(req):
        return db.users.get(req.headers.get("X-User"))
embedded with only path context -- "Python source in saas/acme-notes-be/...".
The function name `_resolve_user` was buried inside the chunk text, often
missed by vector retrieval on queries like "user lookup function".

After this module, the chunker emits a parent chunk per function, and
contextual chunking renders:
    [Context: Python source in saas/acme-notes-be/apps/auth/middleware.py
     -- section '_resolve_user']
    def _resolve_user(req):
        ...
The function name is now in the embedded text, near the front, where the
embedder weights it heavily.

Per-language quality
--------------------
- **Python**: stdlib `ast` -- full accuracy, handles decorators, async,
  nested classes/functions, syntax errors gracefully (returns empty
  list on parse failure -> falls back to single-section default).
- **TypeScript/JavaScript**: regex -- covers `function name`, `class Name`,
  `const name = (...) =>` arrow funcs, `name(...) { ... }` class methods.
  Misses IIFEs, object-literal methods. Best-effort, ~70-80% coverage.
- **Go**: regex -- `func Name`, `func (recv *T) Method`, `type Name struct`.
  Go syntax is regular enough that regex hits ~95% of definitions.
- **Java/Kotlin, Shell**: regex placeholder -- minimal patterns, low cost
  to add proper coverage later if retrieval data justifies it.

Module-level code (imports, top-level constants, decorators between
defs) collapses into a single `<module>` section per file. This is fine
because top-level code is usually short and queries about it ("what does
this module import?") match the path context already.
"""
from __future__ import annotations

import ast
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from opsrag.interfaces.parser import DocSection, DocType

_log = logging.getLogger("opsrag.parsers.code_structure")


@dataclass(frozen=True)
class _DefSpan:
    """Definition span in a source file.

    `name` is the surface name. For class methods, it's `ClassName.method_name`
    so the section heading carries the class context. For Go receiver-
    bound methods, it's `Receiver.Method`.

    `summary` is a one-line description scraped from the doc comment
    (Python docstring via ast / TS-JS `/** */` block / Go `//` leader).
    Empty when the def has no doc. Truncated to ~80 chars at extraction
    time so the section heading stays compact for the contextual prefix.
    """
    name: str
    kind: str           # "function" | "class" | "method" | "type"
    start_line: int     # 1-indexed, inclusive
    end_line: int       # 1-indexed, inclusive (line of last token)
    summary: str = ""   # first non-empty line of the def's doc comment


_SUMMARY_MAX_CHARS = 80


def _clean_summary(raw: str) -> str:
    """Return the first non-empty line of `raw`, stripped + truncated.

    Strips common doc-comment noise: leading `*` from JSDoc-style
    `/** * foo */`, leading `// ` for Go-style line comments.
    """
    if not raw:
        return ""
    for line in raw.splitlines():
        stripped = line.strip()
        # Drop JSDoc/JavaDoc line-leader `*` and Go line-comment `//`.
        if stripped.startswith("*"):
            stripped = stripped.lstrip("*").strip()
        if stripped.startswith("//"):
            stripped = stripped[2:].strip()
        # Skip pure-tag lines (@param/@return/...).
        if stripped.startswith("@"):
            continue
        if stripped:
            if len(stripped) > _SUMMARY_MAX_CHARS:
                stripped = stripped[: _SUMMARY_MAX_CHARS - 1].rstrip() + "..."
            return stripped
    return ""


# -- Python via stdlib ast ------------------------------------------------


def _extract_python_defs(content: str) -> list[_DefSpan]:
    """Walk the Python AST; emit one span per def, with class methods
    expanded as `ClassName.method_name`.

    Step 3a -- class methods get their own spans. The class itself still
    emits a span, but its `end_line` is trimmed to the line BEFORE the
    first method so its content is just the class header + class-level
    attributes (constants, type hints), not the method bodies.

    Step 3b -- every span carries `summary` = first line of the
    function/class/method docstring (via `ast.get_docstring`). Empty when
    no docstring. The chunker uses this in section_heading so contextual
    chunking renders `section 'Config -- Configuration for ...'` instead
    of just `section 'Config'`.

    Nested defs (function-in-function) are NOT split; they ride inside
    their parent's span and char-size split later if oversized.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        _log.debug("python AST parse failed: %s -- no structural split", exc)
        return []

    spans: list[_DefSpan] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.append(_py_span(node, name=node.name, kind="function"))
        elif isinstance(node, ast.ClassDef):
            spans.extend(_py_class_spans(node))
    return spans


def _py_span(node: ast.AST, name: str, kind: str, *, end_line: int | None = None) -> _DefSpan:
    """Build a _DefSpan from a Python AST node -- decorator-aware start,
    docstring-aware summary."""
    start = _first_decorator_line(node) or getattr(node, "lineno", 1)
    end = end_line if end_line is not None else (getattr(node, "end_lineno", None) or start)
    summary = ""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        summary = _clean_summary(ast.get_docstring(node) or "")
    return _DefSpan(name=name, kind=kind, start_line=start, end_line=end, summary=summary)


def _py_class_spans(node: ast.ClassDef) -> list[_DefSpan]:
    """Emit one span for the class header + one per method.

    The class span's end_line is trimmed to just before the first method
    so its content covers only class-level statements (header line, base
    classes, decorators, class docstring, class-level attributes).
    Methods get spans named `ClassName.method_name`.
    """
    class_start = _first_decorator_line(node) or node.lineno
    class_end = node.end_lineno or class_start
    class_summary = _clean_summary(ast.get_docstring(node) or "")

    method_spans: list[_DefSpan] = []
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method_spans.append(
                _py_span(child, name=f"{node.name}.{child.name}", kind="method")
            )

    # Trim the class span so its content excludes method bodies.
    if method_spans:
        first_method_start = min(m.start_line for m in method_spans)
        class_end = max(class_start, first_method_start - 1)

    class_span = _DefSpan(
        name=node.name,
        kind="class",
        start_line=class_start,
        end_line=class_end,
        summary=class_summary,
    )
    return [class_span, *method_spans]


def _first_decorator_line(node: ast.AST) -> int | None:
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        first = decorators[0]
        return getattr(first, "lineno", None)
    return None


# -- Regex-based extraction for non-Python languages ----------------------


# Each entry: (pattern, name_fn, kind).
# `name_fn` takes a re.Match and returns the span name. This lets Go's
# receiver-bound methods compose `Receiver.Method` from two groups while
# other languages just return `m.group(1)`.
#
# Patterns are anchored to line start (`^`) to skip nested matches inside
# strings/comments -- defensive, not bullet-proof. `re.MULTILINE` so `^`
# matches every line.
_NameFn = Callable[[re.Match], str]

_TS_PATTERNS: list[tuple[re.Pattern, _NameFn, str]] = [
    # `function name(`
    (re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*[(<]", re.M),
     lambda m: m.group(1), "function"),
    # `class Name {`
    (re.compile(r"^\s*(?:export\s+(?:default\s+)?(?:abstract\s+)?)?class\s+(\w+)\b", re.M),
     lambda m: m.group(1), "class"),
    # `const name = (...) =>` with optional return-type annotation.
    (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*[:=]\s*(?:async\s+)?(?:\([^)]*\)|\w+)(?:\s*:\s*[^={\n]+)?\s*=>", re.M),
     lambda m: m.group(1), "function"),
    # `interface Name {` / `type Name =`
    (re.compile(r"^\s*(?:export\s+)?(?:interface|type)\s+(\w+)\b", re.M),
     lambda m: m.group(1), "type"),
]

_GO_PATTERNS: list[tuple[re.Pattern, _NameFn, str]] = [
    # Receiver-bound: `func (r *Receiver) Method(` -> "Receiver.Method".
    # Step 3a: emit `Type.Method` form so contextual prefix says
    # `section 'GitLabClient.fetch_pipeline'` instead of just `section 'fetch_pipeline'`.
    (re.compile(r"^func\s+\([^)]*?\*?(\w+)\)\s+(\w+)\s*\(", re.M),
     lambda m: f"{m.group(1)}.{m.group(2)}", "method"),
    # Plain `func Name(` -- matches MAY overlap with receiver-bound at
    # the same start_line; the post-sort dedupe in _extract_regex_defs
    # handles that (first hit wins).
    (re.compile(r"^func\s+(\w+)\s*\(", re.M),
     lambda m: m.group(1), "function"),
    # `type Name struct {` or `type Name interface {`
    (re.compile(r"^type\s+(\w+)\s+(?:struct|interface)\b", re.M),
     lambda m: m.group(1), "type"),
]

_JAVA_PATTERNS: list[tuple[re.Pattern, _NameFn, str]] = [
    (re.compile(r"^\s*(?:public|private|protected|internal|open|sealed|abstract|final)?\s*class\s+(\w+)\b", re.M),
     lambda m: m.group(1), "class"),
    (re.compile(r"^\s*(?:public|private|protected|internal)?\s*(?:suspend\s+)?fun\s+(\w+)\s*\(", re.M),
     lambda m: m.group(1), "function"),
    (re.compile(r"^\s*(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?(?:[\w<>\[\],\s]+?)\s+(\w+)\s*\(", re.M),
     lambda m: m.group(1), "method"),
]

_SHELL_PATTERNS: list[tuple[re.Pattern, _NameFn, str]] = [
    (re.compile(r"^\s*(?:function\s+)?(\w+)\s*\(\s*\)\s*\{", re.M),
     lambda m: m.group(1), "function"),
]


# -- Doc-comment scraping (Step 3b) ---------------------------------------


def _doc_comment_above(lines: list[str], def_line: int, language: str) -> str:
    """Extract the doc comment IMMEDIATELY above a def line.

    Languages:
      - `block`  -> JSDoc/JavaDoc-style `/** ... */` (TS/JS/Java/Kotlin).
        Walks back from def_line until it hits the closing `*/`, then
        further back to `/**` start. Returns inner text.
      - `line`   -> Go-style `// ...` consecutive line comments. Walks back
        line-by-line while lines are `//`-prefixed.

    `def_line` is 1-indexed; lines list is 0-indexed.
    """
    if def_line <= 1:
        return ""
    idx = def_line - 2  # line immediately above the def, 0-indexed

    # Skip blank lines between def and its doc comment.
    while idx >= 0 and not lines[idx].strip():
        idx -= 1
    if idx < 0:
        return ""

    if language == "block":
        if not lines[idx].strip().endswith("*/"):
            return ""
        # Walk back collecting until we find `/**` opener.
        end_idx = idx
        while idx >= 0 and "/**" not in lines[idx]:
            idx -= 1
        if idx < 0:
            return ""
        block = "\n".join(lines[idx : end_idx + 1])
        # Strip the `/**` and `*/` markers and per-line `*` leaders.
        block = block.replace("/**", "").replace("*/", "")
        return _clean_summary(block)

    if language == "line":
        end_idx = idx
        while idx >= 0 and lines[idx].lstrip().startswith("//"):
            idx -= 1
        # Comments span [idx+1 .. end_idx].
        if idx + 1 > end_idx:
            return ""
        block = "\n".join(lines[idx + 1 : end_idx + 1])
        return _clean_summary(block)

    return ""


_DOC_COMMENT_LANGUAGES: dict[DocType, str] = {
    DocType.TYPESCRIPT: "block",
    DocType.JAVASCRIPT: "block",
    DocType.JAVA: "block",
    DocType.GO: "line",
    # Shell -- `# ...` rarely used as proper docs in this corpus, skip.
}


def _extract_regex_defs(
    content: str,
    patterns: list[tuple[re.Pattern, _NameFn, str]],
    doc_lang: str = "",
) -> list[_DefSpan]:
    """Run all language patterns; emit one span per match (deduped by
    start_line). Spans get a `summary` when `doc_lang` is set and the
    def has a recognized doc comment immediately above it.

    end_line = next-def's start - 1 (or EOF for the last). Coarse but
    robust without brace-matching.
    """
    lines = content.splitlines()
    seen_lines: set[int] = set()
    raw: list[tuple[int, str, str]] = []  # (start_line, name, kind)
    for pat, name_fn, kind in patterns:
        for m in pat.finditer(content):
            start_line = content.count("\n", 0, m.start()) + 1
            if start_line in seen_lines:
                continue  # first pattern wins per line
            seen_lines.add(start_line)
            try:
                name = name_fn(m)
            except (IndexError, AttributeError):
                continue
            if not name:
                continue
            raw.append((start_line, name, kind))

    if not raw:
        return []

    raw.sort(key=lambda t: t[0])
    total_lines = len(lines) or 1
    spans: list[_DefSpan] = []
    for i, (start, name, kind) in enumerate(raw):
        end = (raw[i + 1][0] - 1) if i + 1 < len(raw) else total_lines
        if end < start:
            end = start
        summary = _doc_comment_above(lines, start, doc_lang) if doc_lang else ""
        spans.append(_DefSpan(name=name, kind=kind,
                              start_line=start, end_line=end, summary=summary))
    return spans


# -- Public entry point ---------------------------------------------------


_REGEX_DISPATCH: dict[DocType, list[tuple[re.Pattern, _NameFn, str]]] = {
    DocType.TYPESCRIPT: _TS_PATTERNS,
    DocType.JAVASCRIPT: _TS_PATTERNS,   # JS uses same patterns; interface/type rarely match
    DocType.GO: _GO_PATTERNS,
    DocType.JAVA: _JAVA_PATTERNS,
    DocType.SHELL: _SHELL_PATTERNS,
}


# -- Step 3a for TS/JS -- per-method splitting inside class bodies ---------


# Class-body method patterns. Two shapes:
#   1. Conventional method:    `  publicMethodName(arg: T): Ret {`
#      (any leading visibility / static / async / override modifiers)
#   2. Arrow-function field:    `  signIn = async (creds): Promise<T> => {`
#      (class field assigned to an arrow function -- increasingly common
#       in React-class / NestJS-style code)
#
# Both require leading indentation (line starts with whitespace) to avoid
# matching top-level functions. Stopword list excludes JS control-flow
# keywords that share the same `name(...)` shape (`if (`, `for (`, etc.).

_TS_METHOD_PATTERN = re.compile(
    r"^\s+(?:(?:public|private|protected|static|async|override|readonly|abstract|get|set)\s+)*"
    r"(\w+)\s*[(<]",
)
_TS_ARROW_METHOD_PATTERN = re.compile(
    r"^\s+(?:(?:public|private|protected|static|readonly)\s+)*"
    r"(\w+)\s*[:=]\s*(?:async\s+)?(?:\([^)]*\)|\w+)(?:\s*:\s*[^={\n]+)?\s*=>",
)

# Names that look method-shaped but aren't -- JS keywords + a few common
# noise tokens we see inside class bodies (destructuring patterns,
# generator yields, etc.). Cheap defense against regex false positives.
_TS_METHOD_STOPWORDS: frozenset[str] = frozenset({
    "if", "for", "while", "do", "switch", "case", "return", "throw", "try",
    "catch", "finally", "else", "break", "continue", "yield", "await",
    "new", "delete", "typeof", "instanceof", "void", "function", "constructor",
})


def _extract_ts_class_methods(content: str, class_spans: list[_DefSpan]) -> list[_DefSpan]:
    """For each TS/JS class span, find indented method declarations within
    its line range. Emit `ClassName.method` spans with JSDoc summaries.

    Why this exists (Step 3a for TS): top-level pattern extraction already
    captures `class Foo {` but treats the whole class body as one section.
    For classes with 10+ methods this means all method bodies char-split
    into chunks tagged `section 'Foo'` -- undifferentiated. After this
    pass each method gets its own section, named `Foo.methodName`, with
    its own JSDoc-derived summary.

    Limitations vs. tree-sitter:
      - Doesn't validate brace nesting. A class containing an object
        literal with method-shaped keys may yield false-positive spans.
        The cost is a chunk with a slightly wrong heading -- content is
        still valid.
      - Doesn't track nested classes; methods of a class-within-a-class
        get attributed to the outer class.
    """
    if not class_spans:
        return []
    lines = content.splitlines()
    out: list[_DefSpan] = []
    for cls in class_spans:
        if cls.kind != "class":
            continue
        method_lines: list[tuple[int, str]] = []  # (1-indexed line, name)
        # Scan strictly INSIDE the class span (after the `class X {` line
        # itself; before the closing brace we approximate with end_line).
        # `range(start, end)` here is 1-indexed inclusive of the line
        # AFTER the class signature.
        for i in range(cls.start_line, min(cls.end_line, len(lines))):
            line = lines[i]
            if not (line.startswith(" ") or line.startswith("\t")):
                continue
            m = _TS_METHOD_PATTERN.match(line) or _TS_ARROW_METHOD_PATTERN.match(line)
            if not m:
                continue
            name = m.group(1)
            if name in _TS_METHOD_STOPWORDS:
                continue
            method_lines.append((i + 1, name))

        for j, (line_no, name) in enumerate(method_lines):
            # End at the line before the next method, or at the class's
            # own end_line for the final method in the class.
            next_start = method_lines[j + 1][0] if j + 1 < len(method_lines) else cls.end_line + 1
            end = max(line_no, next_start - 1)
            summary = _doc_comment_above(lines, line_no, "block")
            out.append(_DefSpan(
                name=f"{cls.name}.{name}",
                kind="method",
                start_line=line_no,
                end_line=end,
                summary=summary,
            ))
    return out


def _compose_heading(span: _DefSpan) -> str:
    """Compose section heading from span name + doc summary.

    Format: `Name -- first doc line`. Kept under ~120 chars so the
    contextual chunking prefix can include it without being awkward.
    """
    if not span.summary:
        return span.name
    head = f"{span.name} -- {span.summary}"
    # Hard cap so monster docstring first lines don't blow the heading.
    return head if len(head) <= 120 else head[:119] + "..."


def extract_def_spans(content: str, doc_type: DocType) -> list[_DefSpan]:
    """Public entry point -- return the raw `_DefSpan` list for `content`.

    Same logic as `split_by_def` but stops before converting spans into
    `DocSection`s, so callers (the code-symbol graph extractor) can use
    the line numbers and `kind` directly. The chunker keeps using
    `split_by_def` to get the section-shaped output it needs.

    Returns an empty list when:
      - the doc type has no per-language extractor
      - the file has no detectable defs (script-only / parse failure)
    """
    if not content.strip():
        return []
    if doc_type == DocType.PYTHON:
        return _extract_python_defs(content)
    patterns = _REGEX_DISPATCH.get(doc_type)
    if patterns is None:
        return []
    doc_lang = _DOC_COMMENT_LANGUAGES.get(doc_type, "")
    spans = _extract_regex_defs(content, patterns, doc_lang)
    if doc_type in (DocType.TYPESCRIPT, DocType.JAVASCRIPT):
        class_spans = [s for s in spans if s.kind == "class"]
        method_spans = _extract_ts_class_methods(content, class_spans)
        if method_spans:
            spans = spans + method_spans
    return spans


def split_by_def(content: str, doc_type: DocType) -> list[DocSection]:
    """Split a source file into DocSections, one per top-level def
    (Step 1 + 2) -- and one per class method when the language supports it
    (Step 3a: Python class methods, Go receiver-bound methods).

    Section headings include a one-line doc summary when available
    (Step 3b: Python docstrings, TS/JS/Java JSDoc blocks, Go `//`
    comments). Format: `Name -- summary`. Falls back to bare `Name`
    when no doc.

    Returns an empty list when:
      - doc_type isn't a code language we have an extractor for
      - the file has no detectable defs (script-only, module-level code)
      - parser failed (Python syntax error, etc.)
    Empty return -> caller falls back to single-section default.

    The first section is always a `<module>` capturing top-of-file
    content (imports, top-level constants, decorators) up to the first
    def -- but only when there IS content above the first def. This keeps
    short scripts as a single section instead of fragmenting into a
    pointless `<module>` + one tiny section.
    """
    if not content.strip():
        return []

    if doc_type == DocType.PYTHON:
        spans = _extract_python_defs(content)
    else:
        patterns = _REGEX_DISPATCH.get(doc_type)
        if patterns is None:
            return []
        doc_lang = _DOC_COMMENT_LANGUAGES.get(doc_type, "")
        spans = _extract_regex_defs(content, patterns, doc_lang)
        # Step 3a for TS/JS -- walk class bodies for methods, then trim
        # each class's end_line to before its first method (so the class
        # section's content is just the header, matching the Python
        # behavior in _py_class_spans).
        if doc_type in (DocType.TYPESCRIPT, DocType.JAVASCRIPT):
            class_spans = [s for s in spans if s.kind == "class"]
            method_spans = _extract_ts_class_methods(content, class_spans)
            if method_spans:
                # Build map: class_name -> earliest method start_line
                first_method: dict[str, int] = {}
                for m in method_spans:
                    cls_name = m.name.rsplit(".", 1)[0]
                    cur = first_method.get(cls_name)
                    if cur is None or m.start_line < cur:
                        first_method[cls_name] = m.start_line
                # Re-emit class spans with trimmed end_line
                trimmed: list[_DefSpan] = []
                for s in spans:
                    if s.kind == "class" and s.name in first_method:
                        new_end = max(s.start_line, first_method[s.name] - 1)
                        trimmed.append(_DefSpan(
                            name=s.name, kind=s.kind,
                            start_line=s.start_line, end_line=new_end,
                            summary=s.summary,
                        ))
                    else:
                        trimmed.append(s)
                spans = trimmed + method_spans

    if not spans:
        return []

    lines = content.splitlines()
    sections: list[DocSection] = []

    # Module-level prelude (above the first def).
    first_start = spans[0].start_line
    if first_start > 1:
        prelude_body = "\n".join(lines[: first_start - 1])
        if prelude_body.strip():
            sections.append(DocSection(
                heading="<module>",
                content=prelude_body,
                level=1,
                section_type="code_module",
            ))

    # One section per def. Sort by start_line so methods follow their
    # class header naturally.
    for span in sorted(spans, key=lambda s: (s.start_line, s.end_line)):
        body = "\n".join(lines[span.start_line - 1 : span.end_line])
        if not body.strip():
            continue
        sections.append(DocSection(
            heading=_compose_heading(span),
            content=body,
            level=1,
            section_type=f"code_{span.kind}",
        ))

    return sections
