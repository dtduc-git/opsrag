"""Contract test (T130): `audit-vendor-neutrality.sh --json` emits the
AuditReport shape from data-model.md section 5.

On a clean tree the report has empty violations and an all-zero summary; the
shape (keys + types) is asserted regardless of content.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT = REPO_ROOT / "scripts" / "audit-vendor-neutrality.sh"

CHECKS = {"proprietary_names", "non_english_text", "hardcoded_hosts"}


@pytest.fixture(scope="module")
def report() -> dict:
    # Run the (slow) full audit ONCE per module and share the parsed report
    # across the assertions, rather than once per test function.
    proc = subprocess.run(
        ["bash", str(AUDIT), "--json"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    # stdout must be a single JSON object (nothing else on stdout).
    return json.loads(proc.stdout)


def test_json_top_level_shape(report: dict) -> None:
    assert set(report) >= {"violations", "summary"}
    assert isinstance(report["violations"], list)
    assert isinstance(report["summary"], dict)


def test_summary_has_all_three_checks(report: dict) -> None:
    assert set(report["summary"]) == CHECKS
    for v in report["summary"].values():
        assert isinstance(v, int) and v >= 0


def test_clean_tree_reports_no_violations(report: dict) -> None:
    assert report["violations"] == []
    assert all(v == 0 for v in report["summary"].values())


def test_violation_records_have_expected_fields_if_any(report: dict) -> None:
    # On a clean tree this is vacuous; documents the per-violation shape.
    for v in report["violations"]:
        assert {"check", "file", "line"} <= set(v)
        assert v["check"] in CHECKS
