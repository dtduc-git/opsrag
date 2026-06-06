"""spaCy free-form NER for Q&A cache discriminator (sub-sprint 5.4).

Layered on top of the regex-based `_discriminator_tokens` in
`opsrag.qa_cache`. The two extractors run independently; their outputs
are unioned, and per-extractor counters let us compare/rank accuracy
over time via `/cache/summary`.

Why both
--------
- **regex** (precision-first): catches structured tokens -- years,
  semver, ticket prefixes, env names, kebab-case service IDs, GitLab
  paths. ~0% false-positive on our domain. Misses free-form
  entities.
- **spaCy** (recall-first): general-purpose NER trained on news/web.
  Catches PERSON, ORG, GPE, DATE, CARDINAL -- the cases where the user
  references a free-form named entity that the regex layer doesn't
  know about (e.g. "the Asia region", "Datadog dashboard", "Tuesday's
  outage", "ten brokers" written in words).

Cost / latency
--------------
- Cold start: load model `en_core_web_sm` once at process boot (~50MB
  RAM, ~200ms one-time). Lazy -- only paid if `OPSRAG_QA_NER_SPACY=1`.
- Per call: ~0.5-2ms for typical SRE-question-length text.
- Sync (no LLM); blocks event loop only briefly. Wrapped in
  `asyncio.to_thread` if the caller passes `await_threadpool=True`
  but default-sync is safer for single-process latency.

Graceful degrade
----------------
- Toggle off (`OPSRAG_QA_NER_SPACY=0` or default) -> no-op, returns
  empty set. The regex layer alone still catches the discriminator
  cases it covered before.
- spaCy not installed -> log once, return empty set forever.
- Model load fails -> log, return empty set.

The combination strategy in `qa_cache._discriminator_tokens` is a
simple union: cache miss when EITHER extractor's output differs. We
don't try to weight votes or use spaCy as authority -- it's an
additive recall layer.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

_log = logging.getLogger("opsrag.qa_cache.ner")

# spaCy entity labels we care about. We filter to discriminator-shaped
# entities -- labels likely to flip query meaning when the entity changes.
# Labels NOT included (deliberately):
#   - LANGUAGE, LAW, NORP -- too rare in DevOps queries
#   - FAC -- facilities; rare
#   - PRODUCT -- too noisy (matches "Kafka" everywhere it appears, but
#     since the regex layer already catches kebab-case service names,
#     PRODUCT adds little)
#   - WORK_OF_ART, EVENT -- irrelevant
_USEFUL_LABELS = frozenset({
    "PERSON",
    "ORG",
    "GPE",       # geo-political (US, EU, Asia)
    "LOC",       # location
    "DATE",      # "yesterday", "Tuesday", "Q3"
    "TIME",
    "CARDINAL",  # "three brokers", "ten pods"
    "ORDINAL",   # "first cycle", "second deployment"
    "QUANTITY",
    "PERCENT",
    "MONEY",
})


@dataclass
class NerStats:
    calls: int = 0
    tokens_extracted: int = 0
    last_load_error: str | None = None


_stats = NerStats()


def stats() -> dict:
    return {
        "calls": _stats.calls,
        "tokens_extracted": _stats.tokens_extracted,
        "last_load_error": _stats.last_load_error,
        "loaded": _nlp_singleton is not None,
        "enabled": is_enabled(),
    }


def is_enabled() -> bool:
    return os.environ.get("OPSRAG_QA_NER_SPACY", "0").lower() in ("1", "true", "yes", "on")


# Module-level singleton. The first call to `extract_entities` triggers
# the load; subsequent calls reuse. Negative-cached on import error so
# repeated calls don't keep retrying a missing model.
_nlp_singleton = None  # set on first successful load
_load_attempted = False


def _load_nlp():
    """Best-effort lazy load of the spaCy small English model. Returns
    None on failure -- caller treats that as "no NER tokens"."""
    global _nlp_singleton, _load_attempted
    if _nlp_singleton is not None:
        return _nlp_singleton
    if _load_attempted:
        return None
    _load_attempted = True
    try:
        import spacy  # type: ignore
    except ImportError as exc:
        _stats.last_load_error = f"spacy not installed: {exc}"
        _log.warning(
            "spaCy NER requested via OPSRAG_QA_NER_SPACY but spacy package is "
            "not installed -- install with `pip install -e .[ner]` and "
            "`python -m spacy download en_core_web_sm`. Falling back to "
            "regex-only discriminator.",
        )
        return None
    try:
        # Disable parser + lemmatizer -- we only need the NER pipeline.
        # Cuts model load time + per-call latency by ~3x.
        _nlp_singleton = spacy.load(
            "en_core_web_sm",
            disable=["parser", "lemmatizer", "tagger"],
        )
        _log.info(
            "spaCy NER model loaded (en_core_web_sm, pipeline=%s)",
            _nlp_singleton.pipe_names,
        )
        return _nlp_singleton
    except Exception as exc:
        _stats.last_load_error = f"spacy.load failed: {exc}"
        _log.warning(
            "spaCy en_core_web_sm load failed: %s -- install with "
            "`python -m spacy download en_core_web_sm`. Falling back to "
            "regex-only discriminator.",
            exc,
        )
        return None


def extract_entities(text: str) -> frozenset[str]:
    """Return discriminator tokens harvested by spaCy NER.

    Each token is `ner:<LABEL>:<text>`. Empty set when:
      - extractor disabled
      - spaCy not installed / model not loadable
      - input empty
      - no useful entities found

    Token format chosen so it never collides with regex-layer tokens
    (which use `year:`, `num:`, `kebab:`, etc.).
    """
    if not is_enabled() or not text:
        return frozenset()
    nlp = _load_nlp()
    if nlp is None:
        return frozenset()
    _stats.calls += 1
    try:
        # Clip very long inputs -- a single cache lookup shouldn't pay
        # multi-MB processing. SRE questions are < 500 chars typically.
        doc = nlp(text[:1000])
        tokens: set[str] = set()
        for ent in doc.ents:
            if ent.label_ not in _USEFUL_LABELS:
                continue
            # Normalize: lowercase the surface form, strip whitespace.
            surface = ent.text.strip().lower()
            if not surface or len(surface) < 2:
                continue
            tokens.add(f"ner:{ent.label_}:{surface}")
        _stats.tokens_extracted += len(tokens)
        return frozenset(tokens)
    except Exception as exc:
        _log.warning("spaCy entity extraction failed for %r: %s", text[:80], exc)
        return frozenset()


# Compare-ranking helper -- exposes the "what each extractor caught"
# breakdown so /cache/summary + audits can show overlap & disagreements
# between regex and spaCy.
@dataclass
class ExtractorBreakdown:
    regex_only: frozenset[str]
    spacy_only: frozenset[str]
    both: frozenset[str]   # tokens that BOTH layers produced (rare; their
                           # token formats differ by prefix so this is
                           # almost always empty unless a kebab token
                           # happens to coincide with an ORG entity)


def compare_breakdown(
    regex_tokens: frozenset[str],
    spacy_tokens: frozenset[str],
) -> ExtractorBreakdown:
    """Diagnostic helper -- partitions the two sets so we can rank
    extractor accuracy. The `both` bucket is mostly informational; the
    real value is `spacy_only` (catches regex misses) and `regex_only`
    (catches structured tokens spaCy doesn't tag)."""
    return ExtractorBreakdown(
        regex_only=regex_tokens - spacy_tokens,
        spacy_only=spacy_tokens - regex_tokens,
        both=regex_tokens & spacy_tokens,
    )
