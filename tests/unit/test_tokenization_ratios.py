"""Unit tests for the chunk-sizing ratio re-index safeguard (FINDING #5).

The per-doc-type chars/token ratios in opsrag.tokenization drive chunk
char-budgets -> boundaries -> IDs, so changing one silently requires a full
re-index. These tests pin the lightweight safeguard in place:

  - a `RATIOS_VERSION` stamp constant exists alongside the ratio table, and
  - `log_active_ratios` is callable and emits a WARNING naming the version
    (so an operator who edits a ratio is reminded to re-index).
"""
from __future__ import annotations

import logging

from opsrag import tokenization


def test_ratios_version_constant_exists() -> None:
    # A single greppable stamp for the ratio table -- bumped on any ratio edit.
    assert isinstance(tokenization.RATIOS_VERSION, str)
    assert tokenization.RATIOS_VERSION


def test_log_active_ratios_callable_and_warns(caplog) -> None:
    assert callable(tokenization.log_active_ratios)
    with caplog.at_level(logging.WARNING, logger="opsrag.tokenization"):
        tokenization.log_active_ratios()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "log_active_ratios must emit a WARNING-level reminder"
    msg = warnings[-1].getMessage()
    # The reminder must name the version and flag the re-index requirement.
    assert tokenization.RATIOS_VERSION in msg
    assert "re-index" in msg.lower()


def test_log_active_ratios_accepts_custom_logger() -> None:
    # The ingestion pipeline passes its own logger; the helper must honor it.
    custom = logging.getLogger("opsrag.tokenization.test-custom")
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    custom.addHandler(handler)
    custom.setLevel(logging.WARNING)
    try:
        tokenization.log_active_ratios(custom)
    finally:
        custom.removeHandler(handler)
    assert any(r.levelno == logging.WARNING for r in records)
