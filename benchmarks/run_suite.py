#!/usr/bin/env python3
"""CLI entry point for the ScratchV DSL benchmark suite (topic 06).

Produces JSON/Markdown reports and returns the suite exit code:

    0 = no hard failures (pass/xfail/xpass/skip)
    1 = at least one hard failure
    2 = configuration error (invalid roots, no cases, invalid options)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.dsl_suite import (  # noqa: E402
    DEFAULT_ROOTS,
    STATUS_FAIL,
    STATUS_SKIP,
    STATUS_XFAIL,
    STATUS_XPASS,
    DSLSuiteRunner,
    SuiteReport,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ScratchV DSL benchmark suite (topic 06)",
    )
    parser.add_argument(
        "--roots", action="append", default=None,
        help="Case root directory (repeatable; default: "
        + ", ".join(DEFAULT_ROOTS) + ")",
    )
    parser.add_argument("--json", default=None, help="Write JSON report")
    parser.add_argument(
        "--markdown", default=None,
        help="Write Markdown report",
    )
    parser.add_argument(
        "--backend", default="riscv", choices=("riscv", "llvm"),
        help="Compiler backend (default: riscv)",
    )
    parser.add_argument(
        "--optimize-level", default="all",
        choices=("none", "basic", "all"),
        help="Optimization level (default: all)",
    )
    parser.add_argument(
        "--reg-alloc", default="linear",
        choices=("naive", "greedy", "linear"),
        help="Register allocator (default: linear)",
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0,
        help="Per-case timeout budget in seconds for the execution oracle "
        "(default: 30); caps each case's meta timeout_s",
    )
    parser.add_argument(
        "--filter", default=None,
        help="Only run cases whose case_id contains this substring",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List discovered case ids and exit",
    )
    parser.add_argument(
        "--strict-xfail", action="store_true",
        help="Treat unexpected passes of xfail-declared cases as failures",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Only write reports; suppress the stdout summary",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots = tuple(args.roots) if args.roots else DEFAULT_ROOTS

    missing = [root for root in roots if not Path(root).is_dir()]
    if missing:
        print(
            f"error: root director{'y' if len(missing) == 1 else 'ies'} "
            f"not found: {', '.join(missing)}",
            file=sys.stderr,
        )
        return 2
    if args.timeout <= 0:
        print("error: --timeout must be positive", file=sys.stderr)
        return 2

    runner = DSLSuiteRunner(
        roots,
        backend=args.backend,
        optimize_level=args.optimize_level,
        reg_alloc=args.reg_alloc,
        timeout_s=args.timeout,
        strict_xfail=args.strict_xfail,
        verbose=False,
    )
    try:
        cases = runner.discover()
        if not cases:
            print("error: no cases discovered", file=sys.stderr)
            return 2
        if args.list:
            for case in cases:
                print(case.case_id)
            return 0
        if args.filter:
            cases = [case for case in cases if args.filter in case.case_id]
            if not cases:
                print(
                    f"error: no cases match filter {args.filter!r}",
                    file=sys.stderr,
                )
                return 2
        report = runner.run_cases(cases)
    finally:
        runner.cleanup()

    _print_summary(report, quiet=args.quiet)

    if args.json:
        report.save_json(args.json)
    if args.markdown:
        report.save_markdown(args.markdown)

    return 1 if report.fail_count > 0 else 0


def _print_summary(report: SuiteReport, *, quiet: bool) -> None:
    summary = report.summary()
    if summary["xpassed"]:
        print(
            f"warning: {summary['xpassed']} xpassed case(s) — declared "
            "xfail stages now pass; remove the obsolete declaration or run "
            "with --strict-xfail",
            file=sys.stderr,
        )
    if quiet:
        return
    print(
        "DSL benchmark suite: "
        f"{summary['passed']} passed, {summary['xfailed']} xfailed, "
        f"{summary['xpassed']} xpassed, {summary['failed']} failed, "
        f"{summary['skipped']} skipped (total {summary['total']})"
    )
    for result in report.results:
        if result.status == STATUS_FAIL:
            print(
                f"  [FAIL] {result.case_id} ({result.error_stage}): "
                f"{result.error}"
            )
        elif result.status in (STATUS_XFAIL, STATUS_XPASS, STATUS_SKIP):
            detail = result.error or ""
            suffix = f": {detail}" if detail else ""
            print(f"  [{result.status}] {result.case_id}{suffix}")


if __name__ == "__main__":
    sys.exit(main())
