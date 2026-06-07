"""Contextual chunk augmentation (Anthropic-style).

For each chunk, prepend a context string so the embedding "knows" the
doc-level scope a 256-token window otherwise lacks. Two paths:

1. **Prose docs** (LLM-generated context, ~$0.001/doc Flash):
   - RUNBOOK, POSTMORTEM, GENERIC_MARKDOWN, ARCHITECTURE, ADR

2. **Structured docs** (P1 #7 -- deterministic template, free):
   - HELM, TERRAFORM, YAML_CONFIG, KUBERNETES, DOCKERFILE, ALERT_DEFINITION
   The original review claimed these were "structurally split and have
   self-contained semantics", but that's false for the typical ops corpus:
     - `values.yaml` repeats per env (staging/preprod/prod) -- same content
       different scope; embedding alone can't tell them apart.
     - Terraform modules split across `apps.tf` / `apps_variables.tf` /
       `apps_outputs.tf` -- each fragment lacks the module name without
       the path.
   Template format: `[Context: <doc-type> in <repo>:<path> [env <env>]]`
   built deterministically from path + doc.metadata. Zero LLM cost,
   eval-stable across runs.

Cost optimization (prose path): ONE LLM call per document handles ALL
its child chunks. Full doc + numbered chunk list in; JSON array of
contexts out. For our short prose docs (~5K tokens), ~$0.001 per doc
with gemini-2.5-flash, no caching needed.

Toggle with `OPSRAG_CONTEXTUAL_CHUNKING=1` env var. On in production
(see docker-compose.yaml). When off, both paths no-op.
"""
from __future__ import annotations

import json
import logging
import os
import re

from opsrag.ingestion.metadata import content_hash as _content_hash
from opsrag.interfaces.chunker import Chunk
from opsrag.interfaces.llm import LLMProvider
from opsrag.interfaces.parser import DocType, ParsedDocument
from opsrag.tokenization import estimate_tokens

_log = logging.getLogger("opsrag.ingestion.contextual")

_PROSE_TYPES: set[DocType] = {
    DocType.RUNBOOK,
    DocType.POSTMORTEM,
    DocType.GENERIC_MARKDOWN,
    DocType.ARCHITECTURE,
    DocType.ADR,
}

# P1 #7 -- structured doc types get template-based context (no LLM call).
# `_STRUCTURED_TYPE_LABELS` is a human-readable string used in the prefix;
# kept short so it doesn't dominate the chunk's content budget.
_STRUCTURED_TYPE_LABELS: dict[DocType, str] = {
    DocType.HELM: "Helm chart values",
    DocType.TERRAFORM: "Terraform config",
    DocType.KUBERNETES: "Kubernetes manifest",
    DocType.YAML_CONFIG: "YAML config",
    DocType.DOCKERFILE: "Dockerfile",
    DocType.ALERT_DEFINITION: "Alert definition",
    # P1 #7 Step 2 (backlog): code-file contextual chunking. Path-only
    # context for now -- embeds module-path semantic signal ("apps.auth
    # .middleware") into every chunk's text so queries naming a service
    # or module surface the right file via vector retrieval, not just
    # via grep. Tree-sitter enclosing-function context is the bigger
    # win but a separate effort (~2-3 hours); the path-only baseline
    # carries ~70% of the gain because typical path segments are
    # already semantically meaningful (apps/auth/middleware.py).
    DocType.PYTHON: "Python source",
    DocType.JAVASCRIPT: "JavaScript source",
    DocType.TYPESCRIPT: "TypeScript source",
    DocType.GO: "Go source",
    DocType.JAVA: "Java/Kotlin source",
    DocType.SHELL: "Shell script",
}

# Env detection from path. Canonical names, checked in precedence order so
# `preprod` isn't mis-claimed by the `prod` rule. Token-set form (not a single
# whole-segment fullmatch) so an env spelled INSIDE a filename is caught --
# `values-prod.yaml` and `values-staging.yaml` must get DIFFERENT context
# prefixes (the headline use case), which the old `fullmatch` on `/`-segments
# silently failed. (Deployment-specific abbreviations come from
# `deployment.environments` in DeploymentContext, not hardcoded here.)
_ENV_TOKEN_MAP: list[tuple[tuple[str, ...], str]] = [
    (("staging", "stage"), "staging"),
    (("prod", "production"), "production"),
    (("dev", "develop", "development"), "dev"),
    (("qa", "test", "testing"), "qa"),
]

# Keep individual prompts under this token estimate to fit gemini-flash
# comfortably. Longer docs get split into multiple LLM calls.
_MAX_CHUNKS_PER_BATCH = 30
_MAX_DOC_CHARS = 30000  # ~7.5K tokens -- full doc included verbatim


def _extract_env_from_path(path: str) -> str:
    """Detect environment marker in a path segment.

    Returns canonical name (`production`, `staging`, `preprod`, `dev`,
    `qa`) when a segment matches; empty string when none does. Used to
    enrich structured-doc context -- same `values.yaml` in different envs
    needs different embedding context to be distinguishable.
    """
    if not path:
        return ""
    low = path.lower()
    # preprod is hyphen/underscore-spelled in the wild (`pre-prod`, `pre_prod`);
    # catch it before the generic `prod` token would mis-claim it.
    if re.search(r"pre[-_]?prod", low):
        return "preprod"
    # Tokenize on any non-alphanumeric so a filename-encoded env is caught:
    # `values-prod.yaml` -> {values, prod, yaml}. Exact-token match (not
    # substring) preserves the old false-positive guard -- `product` -> {product}
    # never matches `prod`.
    tokens = {t for t in re.split(r"[^a-z0-9]+", low) if t}
    for names, canon in _ENV_TOKEN_MAP:
        if any(n in tokens for n in names):
            return canon
    return ""


def _refresh_content_hash(chunk: Chunk) -> None:
    """Re-stamp metadata.content_hash after the chunk's content was mutated by a
    context prefix, so the dedup/idempotent-upsert hash matches the embedded
    text. No-op if the chunker never set one (e.g. fixed-size chunker)."""
    if isinstance(chunk.metadata, dict) and "content_hash" in chunk.metadata:
        chunk.metadata["content_hash"] = _content_hash(chunk.content)


def _leading_key(content: str) -> str:
    """First config key / identifier line of a chunk.

    Structured docs (values.yaml, *.tf) frequently have NO section heading, so
    without this every sibling chunk of the same file gets a byte-identical
    `[Context: ... in repo/path]` prefix -- pure boilerplate that the lexical
    lane has to wade through and that gives each chunk zero distinguishing
    signal. The leading YAML/HCL key (`replicas`, `image`, `resources`) is a
    cheap per-chunk discriminator that differs between siblings.
    """
    for line in content.splitlines():
        s = line.strip()
        if not s or s.startswith(("#", "//", "[Context:")):
            continue
        m = re.match(r"([A-Za-z_][\w.-]{1,40})\s*[:=]", s)
        if m:
            return m.group(1)
        return s[:40]
    return ""


def _build_structured_context(chunk: Chunk, doc: ParsedDocument) -> str:
    """Build a one-sentence context for a structured-doc chunk.

    Format: `<type_label> in <repo>/<path> [env <env>] [section: <section>]`

    Components, in order:
      1. Doc-type label ("Helm chart values", "Terraform config", ...)
      2. Repo + source path
      3. Env, if extractable from path
      4. Section heading, if the chunker recorded one in metadata

    The result is prepended as `[Context: ...]\\n\\n<chunk content>` --
    same wrapper format the prose path uses, so downstream code can't
    tell prose vs. structured apart by the prefix shape.
    """
    label = _STRUCTURED_TYPE_LABELS.get(doc.doc_type, "Config")
    repo = (chunk.repo or "").strip("/")
    path = (chunk.source_path or "").strip("/")
    loc = f"{repo}/{path}" if repo and path else (path or repo or "<unknown>")

    parts = [f"{label} in {loc}"]

    env = _extract_env_from_path(path)
    if env:
        parts.append(f"env {env}")

    # Section heading is set by the parent_child chunker per section.
    # For code chunks (Step 3a+3b), the heading is often
    # `Name -- first line of docstring` -- which can exceed the legacy
    # 80-char cap. Truncate instead of dropping so the embedding still
    # carries the function/class name even when the doc summary is long.
    section = ""
    if chunk.metadata:
        section = (chunk.metadata.get("section_heading") or "").strip()
    if section:
        section_display = section if len(section) <= 120 else section[:119] + "..."
        parts.append(f"section '{section_display}'")
    else:
        # No section heading (typical for YAML/HCL): use the chunk's leading
        # key so sibling chunks of the same file don't share an identical
        # prefix. Skip when it would just echo the filename.
        key = _leading_key(chunk.content)
        if key and key.lower() not in loc.lower():
            parts.append(f"key '{key}'")

    return " -- ".join(parts)

_PROMPT = """You are augmenting RAG chunks with one-sentence context that situates each
chunk inside the document. The context will be PREPENDED to the chunk
text before embedding, so phrase it to help semantic search.

Document:
<document>
{doc}
</document>

Below is a numbered list of chunks taken from this document. For each
chunk, return ONE sentence (max 25 words) describing what role it plays
in the document -- section it belongs to, what it's explaining, what
question it would answer.

Output STRICTLY as JSON with shape:
{{"contexts": ["sentence for chunk 1", "sentence for chunk 2", ...]}}

The order MUST match the chunk numbers below.

Chunks:
{chunks_block}
"""


def is_enabled() -> bool:
    return os.environ.get("OPSRAG_CONTEXTUAL_CHUNKING", "").lower() in ("1", "true", "yes")


async def augment_chunks(
    chunks: list[Chunk],
    doc: ParsedDocument,
    llm: LLMProvider,
) -> list[Chunk]:
    """Prepend per-chunk context to children of prose docs (LLM path) or
    to all chunks of structured docs (template path). Mutates `content`
    in-place on chunk objects; returns the same list.

    For prose: parents are skipped (they already have section context
    in metadata); children get an LLM-generated 1-sentence context.

    For structured: ALL chunks (parent + child) get a deterministic
    template context, since structured parents are often short (~20-line
    Chart.yaml) without section headings and benefit just as much as
    children from doc-level scope (env, repo, path).
    """
    # -- Structured-doc path: deterministic template, no LLM call -----
    if doc.doc_type in _STRUCTURED_TYPE_LABELS:
        _augment_structured(chunks, doc)
        return chunks

    # -- Prose-doc path: LLM-generated context for children -----------
    if doc.doc_type not in _PROSE_TYPES:
        return chunks
    children = [c for c in chunks if c.chunk_type == "child"]
    if not children:
        return chunks

    doc_text = (doc.content or "")[:_MAX_DOC_CHARS]
    if not doc_text.strip():
        return chunks

    # Process in batches of _MAX_CHUNKS_PER_BATCH to keep each LLM call bounded.
    batches = [
        children[i : i + _MAX_CHUNKS_PER_BATCH]
        for i in range(0, len(children), _MAX_CHUNKS_PER_BATCH)
    ]

    for batch in batches:
        contexts = await _generate_contexts(doc_text, batch, llm, doc.title)
        if not contexts:
            continue
        for child, ctx in zip(batch, contexts):
            ctx_clean = ctx.strip()
            if ctx_clean:
                child.content = f"[Context: {ctx_clean}]\n\n{child.content}"
                # Recompute: the prefix (40-80 tok) was added AFTER the chunker
                # sized this piece, so the stored token_count was stale and any
                # downstream budget/truncation logic under-counted the real
                # embedded length. Likewise refresh the dedup content_hash so it
                # describes the text that's actually embedded+stored, not the
                # pre-prefix slice.
                child.token_count = estimate_tokens(child.content, child.doc_type)
                _refresh_content_hash(child)

    return chunks


def _augment_structured(chunks: list[Chunk], doc: ParsedDocument) -> None:
    """Prepend a template context to every chunk of a structured doc.

    Idempotent within a single ingestion pass: a chunk that already
    starts with `[Context:` is skipped (in case the chunker emits the
    same chunk twice or contextual was already applied upstream).
    """
    for chunk in chunks:
        if not chunk.content:
            continue
        if chunk.content.startswith("[Context:"):
            continue
        ctx = _build_structured_context(chunk, doc)
        if ctx:
            chunk.content = f"[Context: {ctx}]\n\n{chunk.content}"
            chunk.token_count = estimate_tokens(chunk.content, chunk.doc_type)
            _refresh_content_hash(chunk)


async def _generate_contexts(
    doc_text: str, batch: list[Chunk], llm: LLMProvider, doc_title: str,
) -> list[str]:
    chunks_block = "\n\n".join(
        f"[{i + 1}] {c.content[:1200]}" for i, c in enumerate(batch)
    )
    prompt = _PROMPT.format(doc=doc_text, chunks_block=chunks_block)

    try:
        response = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048,
            purpose="contextual-chunk",
        )
    except Exception as exc:
        _log.warning("contextual generation failed for %s: %s", doc_title, exc)
        return []

    text = (response.content or "").strip()
    # Strip optional markdown fences.
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        data = json.loads(text)
    except Exception:
        _log.warning(
            "contextual generation returned non-JSON for %s; head=%s",
            doc_title, text[:200],
        )
        return []

    contexts = data.get("contexts") if isinstance(data, dict) else None
    if not isinstance(contexts, list):
        _log.warning("contextual response missing 'contexts' array for %s", doc_title)
        return []
    # Pad/truncate to match batch length defensively.
    if len(contexts) < len(batch):
        contexts = contexts + [""] * (len(batch) - len(contexts))
    return [str(c) for c in contexts[: len(batch)]]
