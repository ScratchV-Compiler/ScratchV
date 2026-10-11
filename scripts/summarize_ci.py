#!/usr/bin/env python3
"""Summarize the unified PR run and fail closed on missing required evidence."""
from __future__ import annotations

import json
import os
import sys


REQUIRED = ("scope", "test", "benchmark", "topic06")
LABELS = {"success": "PASS", "failure": "FAIL", "cancelled": "CANCELLED", "skipped": "NOT RUN"}


def render_summary(results: object) -> tuple[str, bool]:
    valid = isinstance(results, dict)
    jobs = results if valid else {}
    scope = jobs.get("scope")
    outputs = scope.get("outputs") if isinstance(scope, dict) else None
    # A delivered/requested W4 gate must not disappear from the dependency
    # list. Only an explicit false output permits branches without W4.
    w4_flag = outputs.get("w4") if isinstance(outputs, dict) else None
    include_w4 = "w4" in jobs or w4_flag != "false"
    rows = []
    for name in (*REQUIRED, "docs", "llm", *(("w4",) if include_w4 else ())):
        job = jobs.get(name)
        outcome = job.get("result") if isinstance(job, dict) else None
        requested = True
        if name not in REQUIRED:
            flag = outputs.get(name) if isinstance(outputs, dict) else None
            requested = {"true": True, "false": False}.get(flag) if isinstance(flag, str) else None
        expected = "success" if requested else "skipped"
        passed = requested is not None and outcome == expected
        valid = valid and passed
        label = LABELS.get(outcome, "INVALID") if isinstance(outcome, str) else "INVALID"
        if outcome is None:
            label = "MISSING"
        requirement = "Required" if requested else "Not requested"
        if requested is None:
            requirement = "Invalid/missing scope output"
        rows.append(f"| {name} | {label} | {requirement}{'' if passed else ' — FAIL'} |")
    text = "\n".join([
        "## CI overview", "", f"**{'PASS' if valid else 'FAIL'}**", "",
        "| Check | Result | Scope |", "| --- | --- | --- |", *rows, "",
        "Numerical evidence and detailed reports appear in the corresponding job summaries and artifacts in this run.",
        "Full W3 IR and W4 QEMU validation are not run by ordinary PR CI; use the explicit manual validation entry. "
        "NOT RUN does not count as numerical acceptance, independent reproduction or team sign-off.", "",
    ])
    return text, valid


def main() -> int:
    try:
        results = json.loads(os.environ.get("CI_JOB_RESULTS", "{}"))
    except (ValueError, TypeError):
        results = None
    text, passed = render_summary(results)
    print(text, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
