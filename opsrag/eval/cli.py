"""`python -m opsrag.eval` CLI.

Subcommands:
  run --tag <name>   Execute the golden set, write report to tests/eval/reports/<tag>.md
  list               Show known categories and counts.

Examples:
  python -m opsrag.eval run --tag baseline
  python -m opsrag.eval run --tag step1-parent-sub --category factual_lookup
  python -m opsrag.eval run --tag dry-run --golden _smoke
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from opsrag.eval.adapters.vertex_judge import VertexGeminiJudge
from opsrag.eval.loaders import GOLDEN_DIR, load_golden
from opsrag.eval.report import check_thresholds, write_report
from opsrag.eval.runner import run_golden

_log = logging.getLogger("opsrag.eval.cli")


def _config_summary() -> dict:
    return {
        "opsrag_url": os.environ.get("OPSRAG_URL", "http://localhost:8000"),
        "judge_model": os.environ.get("OPSRAG_JUDGE_MODEL", "gemini-2.5-pro"),
        "opsrag_llm_override": os.environ.get("OPSRAG_LLM_MODEL", "(default from config-local.yaml)"),
    }


def _assert_cache_bypassed(opsrag_url: str) -> None:
    """Fail the gate if the QA cache is live on the target server.

    A cache hit serves STORED sources, so the ranking metrics would measure the
    cache, not retrieval -- masking a regression. We can't read the server's env
    from here, so PROBE it: send a cacheable (procedural) query twice; a 2nd-call
    cache_hit means the cache is ON and the eval server is misconfigured (it must
    run with OPSRAG_DISABLE_QA_CACHE=1). Transport errors don't block the gate."""
    from opsrag.eval.runner import _query_opsrag

    probe = "what is the standard procedure to deploy a service"
    try:
        _query_opsrag(opsrag_url, probe)
        second = _query_opsrag(opsrag_url, probe)
    except Exception as exc:
        _log.warning(
            "cache-bypass probe failed (%s) -- proceeding without the assertion",
            exc,
        )
        return
    if second.get("cache_hit"):
        print(
            "EVAL GATE ABORTED: the QA cache is ENABLED on the target server "
            "(repeat probe returned cache_hit=true). Run the eval server with "
            "OPSRAG_DISABLE_QA_CACHE=1 so the ranking metrics measure retrieval, "
            "not the cache.",
            file=sys.stderr,
        )
        raise SystemExit(3)


def cmd_run(args: argparse.Namespace) -> int:
    opsrag_url = os.environ.get("OPSRAG_URL", "http://localhost:8000")
    # Before an expensive gated run, assert the server isn't serving from cache.
    if getattr(args, "gate", False):
        _assert_cache_bypassed(opsrag_url)
    judge = VertexGeminiJudge(
        model_name=os.environ.get("OPSRAG_JUDGE_MODEL", "gemini-2.5-pro"),
    )

    queries = load_golden(category=args.golden) if args.golden else load_golden(category=args.category)
    if not queries:
        scope = args.golden or args.category or "<all>"
        print(f"No golden queries loaded for scope={scope}", file=sys.stderr)
        return 2

    print(f"Running {len(queries)} queries against {opsrag_url} (tag={args.tag})", file=sys.stderr)
    results = run_golden(opsrag_url=opsrag_url, judge=judge, queries=queries)
    md_path = write_report(results, tag=args.tag, config_summary=_config_summary())
    print(f"Report -> {md_path}")

    # Regression gate (CI): fail the command if any gated metric's
    # vacuous-excluded mean drops below threshold. Off by default so local
    # exploratory runs still exit 0.
    if getattr(args, "gate", False):
        failures = check_thresholds(results)
        if failures:
            print("EVAL GATE FAILED:", file=sys.stderr)
            for f in failures:
                print(f"  - {f}", file=sys.stderr)
            return 1
        print("EVAL GATE PASSED (all gated metrics above threshold)")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    files = sorted(GOLDEN_DIR.glob("*.yaml"))
    if not files:
        print("(no golden YAML files yet)")
        return 0
    print(f"{'Category':<28}  Count")
    print("-" * 40)
    for f in files:
        cat = f.stem
        try:
            qs = load_golden(category=cat)
            print(f"{cat:<28}  {len(qs)}")
        except Exception as exc:
            print(f"{cat:<28}  ERROR: {exc}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    parser = argparse.ArgumentParser(prog="opsrag.eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="execute golden set, write report")
    p_run.add_argument("--tag", required=True, help="report tag (becomes filename stem)")
    p_run.add_argument("--category", default=None, help="restrict to single category file")
    p_run.add_argument("--golden", default=None, help="alias for --category (also loads _smoke.yaml etc)")
    p_run.add_argument(
        "--gate", action="store_true",
        help="fail (exit 1) if any gated metric's mean is below threshold (for CI)",
    )
    p_run.set_defaults(func=cmd_run)

    p_list = sub.add_parser("list", help="show available golden categories")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
