"""Question normalization for the Q&A cache key.

The semantic cache keys on `(question_embedding, discriminator_tokens)`,
plus a deterministic UUID5 derived from the lower-cased question for
overwrite semantics. Polite phrasing ("please", "could you...") and
contractions ("don't" vs "do not") perturb both the embedding and the
UUID -- same intent, different cache slot. ~5-10pp hit-rate loss in
practice.

`normalize_query` strips polite prefixes/suffixes, expands common
contractions, and collapses whitespace. It does NOT touch entities
or meaningful content -- discriminator tokens stay intact downstream.

The ORIGINAL question is preserved on the cached payload (for display
and source attribution); only the cache lookup/insert key derives from
the normalized form.
"""
from __future__ import annotations

import re

# Leading politeness phrases -- anchored at the start, case-insensitive.
# Order matters: longer phrases first so "could you please" gets fully
# consumed instead of leaving "please" behind.
_LEADING_POLITE = re.compile(
    r"^\s*(?:"
    r"could you please|"
    r"can you please|"
    r"would you please|"
    r"i would like to|"
    r"i would like|"
    r"i'd like to|"
    r"i'd like|"
    r"could you|"
    r"can you|"
    r"would you|"
    r"i need to|"
    r"i want to|"
    r"i need|"
    r"i want|"
    r"help me|"
    r"please tell me|"
    r"tell me|"
    r"please"
    r")\b[\s,:]*",
    re.IGNORECASE,
)

# Trailing politeness -- anchored at the end. Strips polite tails before
# any trailing punctuation. Allow multiple (e.g. "please, thanks").
_TRAILING_POLITE = re.compile(
    r"[\s,]*(?:please|thanks(?: a lot| very much)?|thank you|thx|ty)"
    r"[\s,.!?]*$",
    re.IGNORECASE,
)

# Contractions -- expand to their full form so "don't" and "do not"
# share a cache slot. Order: apostrophe-bearing tokens first; bare
# contractions ("cant", "wont") covered too because users do type them.
# All matches are word-boundary-anchored.
_CONTRACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bwon'?t\b", re.IGNORECASE), "will not"),
    (re.compile(r"\bcan'?t\b", re.IGNORECASE), "can not"),
    (re.compile(r"\bshan'?t\b", re.IGNORECASE), "shall not"),
    (re.compile(r"\bdon'?t\b", re.IGNORECASE), "do not"),
    (re.compile(r"\bdoesn'?t\b", re.IGNORECASE), "does not"),
    (re.compile(r"\bdidn'?t\b", re.IGNORECASE), "did not"),
    (re.compile(r"\bisn'?t\b", re.IGNORECASE), "is not"),
    (re.compile(r"\baren'?t\b", re.IGNORECASE), "are not"),
    (re.compile(r"\bwasn'?t\b", re.IGNORECASE), "was not"),
    (re.compile(r"\bweren'?t\b", re.IGNORECASE), "were not"),
    (re.compile(r"\bhasn'?t\b", re.IGNORECASE), "has not"),
    (re.compile(r"\bhaven'?t\b", re.IGNORECASE), "have not"),
    (re.compile(r"\bhadn'?t\b", re.IGNORECASE), "had not"),
    (re.compile(r"\bwouldn'?t\b", re.IGNORECASE), "would not"),
    (re.compile(r"\bshouldn'?t\b", re.IGNORECASE), "should not"),
    (re.compile(r"\bcouldn'?t\b", re.IGNORECASE), "could not"),
    (re.compile(r"\bmustn'?t\b", re.IGNORECASE), "must not"),
    # Apostrophe-only contractions for "is" / "has".
    (re.compile(r"\bit's\b", re.IGNORECASE), "it is"),
    (re.compile(r"\bwhat's\b", re.IGNORECASE), "what is"),
    (re.compile(r"\bwho's\b", re.IGNORECASE), "who is"),
    (re.compile(r"\bwhere's\b", re.IGNORECASE), "where is"),
    (re.compile(r"\bwhen's\b", re.IGNORECASE), "when is"),
    (re.compile(r"\bhow's\b", re.IGNORECASE), "how is"),
    (re.compile(r"\bthat's\b", re.IGNORECASE), "that is"),
    (re.compile(r"\bthere's\b", re.IGNORECASE), "there is"),
    (re.compile(r"\bhere's\b", re.IGNORECASE), "here is"),
    (re.compile(r"\blet's\b", re.IGNORECASE), "let us"),
    # Pronoun + auxiliary contractions.
    (re.compile(r"\bi'm\b", re.IGNORECASE), "i am"),
    (re.compile(r"\bi've\b", re.IGNORECASE), "i have"),
    (re.compile(r"\bi'll\b", re.IGNORECASE), "i will"),
    (re.compile(r"\bi'd\b", re.IGNORECASE), "i would"),
    (re.compile(r"\byou're\b", re.IGNORECASE), "you are"),
    (re.compile(r"\byou've\b", re.IGNORECASE), "you have"),
    (re.compile(r"\byou'll\b", re.IGNORECASE), "you will"),
    (re.compile(r"\byou'd\b", re.IGNORECASE), "you would"),
    (re.compile(r"\bwe're\b", re.IGNORECASE), "we are"),
    (re.compile(r"\bwe've\b", re.IGNORECASE), "we have"),
    (re.compile(r"\bwe'll\b", re.IGNORECASE), "we will"),
    (re.compile(r"\bthey're\b", re.IGNORECASE), "they are"),
    (re.compile(r"\bthey've\b", re.IGNORECASE), "they have"),
    (re.compile(r"\bthey'll\b", re.IGNORECASE), "they will"),
]

_WHITESPACE = re.compile(r"\s+")


def normalize_query(text: str) -> str:
    """Normalize a user query for use as a cache key.

    Steps (in order, each operating on the running result):
      1. Lower-case
      2. Expand contractions (don't -> do not, what's -> what is, ...)
      3. Strip leading politeness ("please", "could you", "i need to", ...)
      4. Strip trailing politeness ("please", "thanks", "thank you", ...)
      5. Collapse multiple whitespace -> single space
      6. Trim

    Empty or whitespace-only input returns "".
    """
    if not text:
        return ""

    out = text.lower()

    for pattern, replacement in _CONTRACTIONS:
        out = pattern.sub(replacement, out)

    # Repeat-strip leading politeness so "please could you" collapses fully.
    prev: str | None = None
    while prev != out:
        prev = out
        out = _LEADING_POLITE.sub("", out)

    # Same for trailing -- handle "please, thanks" stacks.
    prev = None
    while prev != out:
        prev = out
        out = _TRAILING_POLITE.sub("", out)

    out = _WHITESPACE.sub(" ", out)
    return out.strip()
