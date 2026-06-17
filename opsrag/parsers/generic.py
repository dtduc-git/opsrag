"""Last-resort parser -- claims any text-shaped config file no other parser handles.

Catches CI YAML (GitLab/GitHub Actions/CircleCI), generic JSON/HCL config,
shell scripts, Dockerfiles, .env templates, etc. Without this, those files
are silently dropped from indexing.

Runs LAST in the parser chain. Step 2: structure-aware splitting -- YAML
files split by top-level key, JSON files split by top-level key, so each
configuration key becomes its own retrievable chunk. Falls back to single-
section for non-parseable content (Helm `.tpl` Go templates, shell scripts).
"""
from __future__ import annotations

import json
import logging

from opsrag.interfaces.parser import DocSection, DocType, ParsedDocument
from opsrag.interfaces.scm import RepoFile
from opsrag.parsers.code_structure import split_by_def

# Round-trip YAML load/dump shared with k8s.py: comments + anchors on a
# top-level key survive into the chunk text (PyYAML's safe_load/dump dropped
# them, losing ops rationale comments). Only the YAML path uses these; JSON /
# code splitting is untouched.
from opsrag.parsers.k8s import _YAML, _dump_key_slice

# Source-code DocTypes that benefit from per-function/per-class section
# splitting. Keep aligned with code_structure._REGEX_DISPATCH + the
# Python AST path. Adding a new code language? Wire it there too.
_CODE_DOC_TYPES: set[DocType] = {
    DocType.PYTHON,
    DocType.JAVASCRIPT,
    DocType.TYPESCRIPT,
    DocType.GO,
    DocType.JAVA,
    DocType.SHELL,
}

_log = logging.getLogger("opsrag.parsers.generic")
# Soft warning threshold -- sections larger than this trigger char-split
# fallback in the chunker. We keep emitting them; just log so authors notice
# YAML keys with massive values worth restructuring.
_OVERSIZE_CHARS = 8000

# Extensions we'll claim. Specific parsers (k8s, helm, terraform, markdown,
# postmortem, runbook, alert) get first dibs via parser priority order.
_GENERIC_EXTENSIONS = (
    ".yaml", ".yml",
    ".hcl",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".sh", ".bash",
    ".env",
    ".tpl",
    ".gotmpl",
    # Source code -- typical backends (Django) and frontends (Angular/
    # React/Vue). Treated as plain text by the chunker; structure-aware
    # splitting falls through to single-section since these aren't YAML/JSON.
    # Quality is good enough for first-pass retrieval; a per-language parser
    # (split by def/class) would be a future improvement.
    ".py", ".pyi",
    ".js", ".jsx", ".mjs", ".cjs",
    ".ts", ".tsx",
    ".vue",
    ".go",
    ".java", ".kt", ".kts",
    ".html", ".htm",
    ".css", ".scss", ".sass", ".less",
)

_GENERIC_BASENAMES = (
    "Dockerfile",
    "Makefile",
    ".gitlab-ci.yml",
    ".gitlab-ci.yaml",
)

_MAX_BYTES = 1_000_000  # 1 MB hard cap -- skip giant generated files
_BINARY_SAMPLE = 4096


class GenericConfigParser:
    """Fallback parser for config/script files no other parser claims."""

    def supports(self, file_path: str, content: str) -> bool:
        if not content or len(content.encode("utf-8", errors="replace")) > _MAX_BYTES:
            return False
        # Crude binary detection -- if first chunk has NUL byte it's almost
        # certainly not a config file we want to chunk as text.
        if "\x00" in content[:_BINARY_SAMPLE]:
            return False

        low = file_path.lower()
        if low.endswith(_GENERIC_EXTENSIONS):
            return True

        basename = file_path.rsplit("/", 1)[-1]
        # Exact basename match (case-sensitive for Dockerfile/Makefile).
        if basename in _GENERIC_BASENAMES:
            return True
        # Dockerfile.foo, Dockerfile_deps, etc.
        if basename.startswith("Dockerfile"):
            return True

        return False

    def detect_doc_type(self, file: RepoFile) -> DocType:
        low = file.path.lower()
        basename = file.path.rsplit("/", 1)[-1]
        if basename.startswith("Dockerfile"):
            return DocType.DOCKERFILE
        if low.endswith((".yaml", ".yml")):
            return DocType.YAML_CONFIG
        # Source-code extensions -> distinct DocType so contextual
        # chunking renders accurate labels ("Python source in ...") and
        # so future per-language enrichers can dispatch off the type.
        # Order: most-specific first (`.tsx`/`.jsx` before `.ts`/`.js`
        # would matter but `endswith` doesn't care since they don't
        # overlap as suffixes).
        if low.endswith((".py", ".pyi")):
            return DocType.PYTHON
        if low.endswith((".ts", ".tsx")):
            return DocType.TYPESCRIPT
        if low.endswith((".js", ".jsx", ".mjs", ".cjs", ".vue")):
            # Vue is HTML+JS+CSS but the JS lane is the dominant signal
            # for retrieval; bucketing under JavaScript keeps the label
            # honest without introducing a Vue-specific type that would
            # need its own per-language story.
            return DocType.JAVASCRIPT
        if low.endswith(".go"):
            return DocType.GO
        if low.endswith((".java", ".kt", ".kts")):
            return DocType.JAVA
        if low.endswith((".sh", ".bash")):
            return DocType.SHELL
        # Falls back to YAML_CONFIG for any remaining text config
        # (HCL/JSON/TOML/INI/CFG/ENV/HTML/CSS/etc.). HTML/CSS could
        # warrant their own types later if retrieval quality demands it.
        return DocType.YAML_CONFIG

    def parse(self, file: RepoFile) -> ParsedDocument:
        title = file.path.rsplit("/", 1)[-1]

        # Step 2: structure-aware splitting. Try YAML/JSON top-level-key split
        # first; fall back to a single section if file isn't parseable as
        # structured data (Helm `.tpl`, shell scripts, malformed YAML, etc.).
        sections = self._structure_split(file) or [
            DocSection(heading=title, content=file.content, level=1)
        ]

        return ParsedDocument(
            content=file.content,
            doc_type=self.detect_doc_type(file),
            title=title,
            source=file,
            metadata={
                "repo": file.repo,
                "branch": file.branch,
                "path": file.path,
                "sha": file.sha,
            },
            sections=sections,
        )

    def _structure_split(self, file: RepoFile) -> list[DocSection] | None:
        """Dispatch to a structure-aware splitter based on doc type.

        Returns None when no splitter applies -> caller wraps the whole
        file as a single DocSection.

        - YAML/JSON: split by top-level key (existing behavior).
        - Code (Python/TS/JS/Go/Java/Shell): split by top-level function
          and class definition. Each def -> one DocSection with heading
          = def name, which flows into chunker's section_heading and
          then into contextual chunking's prefix line.
        """
        doc_type = self.detect_doc_type(file)
        if doc_type in _CODE_DOC_TYPES:
            sections = split_by_def(file.content, doc_type)
            return sections or None
        low = file.path.lower()
        if low.endswith((".yaml", ".yml")):
            return self._split_yaml(file)
        if low.endswith(".json"):
            return self._split_json(file)
        return None

    def _split_yaml(self, file: RepoFile) -> list[DocSection] | None:
        try:
            docs = list(_YAML.load_all(file.content))
        except Exception:
            return None

        sections: list[DocSection] = []
        for doc_idx, doc in enumerate(docs):
            # Only mapping-shaped docs split by key. List/scalar/null -> skip.
            # ruamel returns CommentedMap (dict subclass) -- iteration/.items()
            # work unchanged.
            if not isinstance(doc, dict) or not doc:
                continue
            prefix = f"doc[{doc_idx}]." if len(docs) > 1 else ""
            for key, value in doc.items():
                heading = f"{prefix}{key}"
                # Round-trip slice: re-dump {key: value} with the key's
                # attached comment (stored on the parent map) so ops rationale
                # comments survive into the chunk text.
                body = _dump_key_slice(doc, key, value)
                if len(body) > _OVERSIZE_CHARS:
                    _log.warning(
                        "yaml key %s in %s is %d chars -- chunker will char-split",
                        heading, file.path, len(body),
                    )
                sections.append(DocSection(
                    heading=heading,
                    content=body,
                    level=1,
                    section_type="yaml_key",
                ))
        return sections or None

    def _split_json(self, file: RepoFile) -> list[DocSection] | None:
        try:
            data = json.loads(file.content)
        except Exception:
            return None
        if not isinstance(data, dict) or not data:
            return None

        sections: list[DocSection] = []
        for key, value in data.items():
            try:
                body = json.dumps({key: value}, indent=2, ensure_ascii=False)
            except Exception:
                body = f'"{key}": <unrepresentable>'
            if len(body) > _OVERSIZE_CHARS:
                _log.warning(
                    "json key %s in %s is %d chars -- chunker will char-split",
                    key, file.path, len(body),
                )
            sections.append(DocSection(
                heading=str(key),
                content=body,
                level=1,
                section_type="json_key",
            ))
        return sections or None
