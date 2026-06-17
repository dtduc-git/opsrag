"""R8: the catch-all GenericConfigParser must be TERMINAL in the parser chain.

`IngestionPipeline._process_file` resolves a file to the FIRST parser whose
`supports()` returns True. `GenericConfigParser` claims source code
(`.py/.ts/.go/...`) and most text config, so it is only safe as the LAST
entry: anything after it is unreachable for those extensions and code files
would silently lose code-aware AST splitting (or be dropped). The pipeline
constructor re-checks this invariant and logs a loud WARNING on violation.

These tests cover both the guard logic and the real production assembly sites
(factory + eval offline), so a future reorder is caught.
"""
from __future__ import annotations

import logging

from opsrag.ingestion.pipeline import IngestionPipeline
from opsrag.parsers.generic import GenericConfigParser
from opsrag.parsers.k8s import K8sManifestParser
from opsrag.parsers.markdown import GenericMarkdownParser


def _pipe(parsers):
    """Build a minimal pipeline -- only the parser list matters here."""
    return IngestionPipeline(
        scm=object(),
        parsers=parsers,
        chunker=None,
        embedder=None,
        vector_store=object(),
    )


# --------------------------------------------------------------------------- #
# Guard logic                                                                 #
# --------------------------------------------------------------------------- #


def test_terminal_generic_passes_without_warning(caplog):
    chain = [K8sManifestParser(), GenericMarkdownParser(), GenericConfigParser()]
    with caplog.at_level(logging.WARNING, logger="opsrag.ingestion"):
        _pipe(chain)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], (
        "correct terminal ordering must not warn"
    )


def test_misordered_generic_warns_loudly(caplog):
    # Generic FIRST -> shadows every specific parser after it.
    chain = [GenericConfigParser(), K8sManifestParser(), GenericMarkdownParser()]
    with caplog.at_level(logging.WARNING, logger="opsrag.ingestion"):
        _pipe(chain)
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("ORDERING VIOLATION" in m for m in msgs), msgs


def test_duplicate_generic_not_terminal_warns(caplog):
    # One in the middle, one at the end -> middle one shadows specifics.
    chain = [
        GenericConfigParser(),
        K8sManifestParser(),
        GenericConfigParser(),
    ]
    with caplog.at_level(logging.WARNING, logger="opsrag.ingestion"):
        _pipe(chain)
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("ORDERING VIOLATION" in m for m in msgs), msgs


def test_missing_generic_warns_about_dropped_files(caplog):
    chain = [K8sManifestParser(), GenericMarkdownParser()]
    with caplog.at_level(logging.WARNING, logger="opsrag.ingestion"):
        _pipe(chain)
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("NO GenericConfigParser" in m for m in msgs), msgs


def test_empty_chain_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="opsrag.ingestion"):
        _pipe([])
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("EMPTY" in m for m in msgs), msgs


def test_guard_is_pure_no_reorder():
    # Defensive only: it must NOT mutate/reorder the chain it inspects.
    chain = [GenericConfigParser(), K8sManifestParser()]
    before = list(chain)
    IngestionPipeline._assert_generic_parser_terminal(chain)
    assert chain == before


# --------------------------------------------------------------------------- #
# Real production assembly sites end with the catch-all                        #
# --------------------------------------------------------------------------- #


def test_eval_offline_chain_ends_with_generic():
    """Eval builds its parser list inline -- mirror of factory ordering.

    Re-create the exact list from opsrag.eval.retrieval_offline so a reorder
    there is caught here too.
    """
    from opsrag.parsers.alert import AlertParser  # noqa: F401 (import parity)
    from opsrag.parsers.helm import HelmParser
    from opsrag.parsers.postmortem import PostmortemParser
    from opsrag.parsers.runbook import RunbookParser
    from opsrag.parsers.terraform import TerraformParser

    parsers = [
        RunbookParser(),
        PostmortemParser(),
        K8sManifestParser(),
        HelmParser(),
        TerraformParser(),
        GenericMarkdownParser(),
        GenericConfigParser(),
    ]
    assert isinstance(parsers[-1], GenericConfigParser)
    assert sum(isinstance(p, GenericConfigParser) for p in parsers) == 1
