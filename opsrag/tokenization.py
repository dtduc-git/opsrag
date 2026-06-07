"""Token-count estimation -- single source of truth for chunker + embedder.

Loads the Gemini sentencepiece tokenizer lazily (via `vertexai.tokenization`)
on first `estimate_tokens` call, then caches it process-wide. Falls back to
a char-based estimate (`CHARS_PER_TOKEN = 3`) if the tokenizer is unavailable
(missing sentencepiece, network failure fetching the model, unsupported
model name).

Why lazy
--------
`vertexai.tokenization.get_tokenizer_for_model("gemini-1.5-flash-001")` does
a sentencepiece model deserialization that takes ~2.8s on first call. We
defer it until something actually counts tokens so unit tests, CLI tools,
and import-time scans don't pay that cost.

Why fallback
------------
- Sentencepiece is gated behind the `[vertex]` install extra. Local dev
  setups without it (or unit tests with mocks) still need a working
  `estimate_tokens()`.
- Vertex tokenizer drift: a future SDK release could rename
  `_tokenizers.get_tokenizer_for_model` (the public API is technically
  the `Tokenizer.from_pretrained` shape in newer versions). The fallback
  keeps the system running while we update.
- `text-embedding-005` is the model that *actually* sees these counts.
  Google doesn't ship a tokenizer for embedding models, so we use the
  Gemini chat tokenizer as a same-family proxy -- agrees within ~1-2%,
  but if you ever swap the embedder for one with a public tokenizer
  (BGE-large, Cohere v3 in vertex-marketplace), the fallback gives us
  a working baseline while we plumb in the new one.

The chunker's *sizing* targets (parent_max_tokens, child_size) still
multiply by `CHARS_PER_TOKEN` to produce a char budget -- sizing is the
inverse of counting, you can't ask sentencepiece "how many chars per N
tokens." That conservative bias is fine; tightening it would mean
running the tokenizer over candidate text spans during chunking, which
isn't worth the complexity.
"""
from __future__ import annotations

import logging
import os
import threading

# Char-per-token ratio for sizing targets + fallback estimator. 3 is the
# default / unknown-type / telemetry value.
CHARS_PER_TOKEN = 3

# Per-content-type chars/token for chunker SIZING (and the char-fallback
# estimator). A single flat 3 mis-sizes both ends: it OVER-fills dense config
# children (a `replicas: 3` YAML line is ~2.5 chars/token -- short keys, numbers,
# punctuation, indentation all tokenize to many short tokens, so a 768-char
# window holds ~307 tokens, 20% past the 256 target and diluting the vector) and
# UNDER-fills prose (English ~4.0 chars/token, so 768 chars is only ~192 tokens,
# 25% short -> over-fragmentation). Code sits between (~3.5: identifiers tokenize
# to fewer, longer tokens than config punctuation). Keyed by DocType.value so
# this module imports nothing from interfaces (no cycle).
#
# NB: changing these reshapes chunk char-budgets -> chunk boundaries -> chunk IDs,
# so a corpus indexed under the old flat 3 must be RE-INDEXED for the new sizing
# to take effect (existing chunks keep their old boundaries until re-ingested).
_CHARS_PER_TOKEN_BY_TYPE: dict[str, float] = {
    # config / structured (dense)
    "terraform": 2.5, "helm": 2.5, "kubernetes": 2.5, "dockerfile": 2.5,
    "alert_definition": 2.5, "yaml_config": 2.5,
    # source code
    "python": 3.5, "javascript": 3.5, "typescript": 3.5, "go": 3.5,
    "java": 3.5, "shell": 3.5,
    # prose
    "runbook": 4.0, "postmortem": 4.0, "architecture": 4.0, "adr": 4.0,
    "generic_markdown": 4.0,
}


def chars_per_token_for(doc_type) -> float:
    """Chars/token for chunker sizing, by doc type (a DocType enum or its str
    value). Falls back to the flat CHARS_PER_TOKEN for None / unknown types."""
    if doc_type is None:
        return float(CHARS_PER_TOKEN)
    key = getattr(doc_type, "value", doc_type)
    return _CHARS_PER_TOKEN_BY_TYPE.get(str(key), float(CHARS_PER_TOKEN))

_log = logging.getLogger("opsrag.tokenization")

# Which Gemini model to fetch the tokenizer for. The Gemini family shares
# a sentencepiece base; the `gemini-1.5-flash-001` tokenizer is the most
# widely-tested. Override via env to track a specific model version.
_TOKENIZER_MODEL = os.environ.get("OPSRAG_TOKENIZER_MODEL", "gemini-1.5-flash-001")

_tokenizer = None
_tokenizer_lock = threading.Lock()
_tokenizer_unavailable = False


def _get_tokenizer():
    """Return the cached tokenizer, loading it on first call.

    Returns None if loading fails -- callers fall back to the char-based
    estimator. Failures are logged once at WARNING; subsequent calls take
    the cached-failure fast path and don't re-attempt the import.
    """
    global _tokenizer, _tokenizer_unavailable
    if _tokenizer is not None:
        return _tokenizer
    if _tokenizer_unavailable:
        return None
    with _tokenizer_lock:
        if _tokenizer is not None:
            return _tokenizer
        if _tokenizer_unavailable:
            return None
        try:
            # Internal module path -- `vertexai.tokenization.get_tokenizer_for_model`
            # is re-exported here in current SDK versions. Switch to the
            # public `from vertexai.tokenization import get_tokenizer_for_model`
            # if/when Google stabilizes the namespace.
            from vertexai.tokenization._tokenizers import get_tokenizer_for_model
            _tokenizer = get_tokenizer_for_model(_TOKENIZER_MODEL)
            _log.info(
                "tokenizer loaded model=%s (using sentencepiece; fallback "
                "ratio=%d chars/token still available)",
                _TOKENIZER_MODEL, CHARS_PER_TOKEN,
            )
        except Exception as exc:
            _tokenizer_unavailable = True
            _log.warning(
                "vertexai tokenizer unavailable (%s) -- using char-based "
                "estimate CHARS_PER_TOKEN=%d for all token counts",
                exc, CHARS_PER_TOKEN,
            )
            return None
    return _tokenizer


def estimate_tokens(text: str, doc_type=None) -> int:
    """Token count for `text`.

    Uses the real Gemini sentencepiece tokenizer when available (exact, so
    `doc_type` is ignored on that path); otherwise a char-based estimate using
    the per-type ratio (`chars_per_token_for(doc_type)`, flat CHARS_PER_TOKEN
    when None). Returns 0 for empty input; always >= 1 for non-empty under the
    fallback path."""
    if not text:
        return 0
    tok = _get_tokenizer()
    if tok is not None:
        try:
            return int(tok.count_tokens(text).total_tokens)
        except Exception as exc:
            # Don't permanently disable the tokenizer on a transient
            # error -- just fall through to the char estimate for this
            # one call. Next call will retry.
            _log.warning(
                "count_tokens failed (%s) -- char-estimate fallback this call",
                exc,
            )
    return max(1, int(len(text) / chars_per_token_for(doc_type)))
