"""Topic 9 diagnostics acceptance and same-corpus parser A/B measurements.

Run with ``python -m benchmarks.bench_dsl_diagnostics --baseline-root PATH``.
Only the standard library is imported until an isolated worker selects a checkout.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import html
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]


def measure(action: Callable[[], Any], warmup: int, groups: int, iterations: int) -> dict:
    """Time complete batches, excluding imports, file I/O and explicit GC."""
    for _ in range(warmup):
        action()
    samples = []
    for _ in range(groups):
        gc.collect()
        start = time.perf_counter()
        for _ in range(iterations):
            action()
        samples.append(time.perf_counter() - start)
    return {"samples_s": samples, "median_s": statistics.median(samples)}


def revision(root: Path) -> str | None:
    """Do not label an unpacked baseline with its enclosing repository's SHA."""
    try:
        top = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            text=True, encoding="utf-8", stderr=subprocess.DEVNULL,
        ).strip()
        if Path(top).resolve() != root.resolve():
            return None
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True, encoding="utf-8", stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_worker(args: argparse.Namespace) -> dict:
    from scratchv.frontend import dsl_extended

    parser_file = Path(dsl_extended.__file__).resolve()
    if not parser_file.is_relative_to(args.worker_root):
        raise RuntimeError(f"parser imported outside selected checkout: {parser_file}")
    paths = sorted(args.cases.glob("*.dsl"))
    if not paths:
        raise ValueError(f"no DSL cases in {args.cases}")
    sources = {path.name: path.read_text(encoding="utf-8") for path in paths}

    def parse_one(name: str, source: str) -> Any:
        try:
            # The legacy parser has no filename keyword. Use its public common API.
            return dsl_extended.ExtendedDSLParser().parse(source)
        except Exception as exc:
            raise RuntimeError(f"{name}: {exc}") from exc

    hashes = {
        name: hashlib.sha256(parse_one(name, source).dump().encode("utf-8")).hexdigest()
        for name, source in sources.items()
    }

    def batch() -> None:
        for name, source in sources.items():
            parse_one(name, source)

    return {
        **measure(batch, args.warmup, args.groups, args.iterations),
        "root": str(args.worker_root), "revision": revision(args.worker_root),
        "parser_file": str(parser_file), "parser": "ExtendedDSLParser",
        "ir_hashes": hashes,
        "case_hashes": {
            name: hashlib.sha256(source.encode("utf-8")).hexdigest()
            for name, source in sources.items()
        },
    }


def diagnostic_cases() -> list[dict]:
    """Expected locations are fixed independently of the validator output."""
    rows = [
        ("keyword_typo", "retrun x", [("E100", 1, 1)], "did you mean 'return'?"),
        ("parenthesis", "x = add(a, b", [("E101", 1, 8)], "add the missing ')'"),
        ("unknown_op", "x = ad(a, b)", [("E200", 1, 5)], "did you mean 'add'?"),
        ("arity", "x = add(a)", [("E201", 1, 9)], None),
        ("stray_closer", "endif", [("E110", 1, 1)], None),
        ("missing_closer", "if (a > b):\nx = add(a, b)", [("E111", 1, 1)], "add missing 'endif'"),
        ("duplicate_else", "if (a > b):\nelse:\nelse:\nendif", [("E112", 3, 1)], None),
        ("keyword_argument", "x = softmax(a, unknown:1)", [("E202", 1, 16)], None),
        ("numeric_argument", "x = softmax(a, axis:nope)", [("E203", 1, 21)], None),
        ("tab_crlf_unicode", "# 中文\r\n\tretrun x\r\n", [("E100", 2, 2)], "did you mean 'return'?"),
        ("three_errors", "retrun x\ny = ad(a, b)\nz = relu(a, b)\n",
         [("E100", 1, 1), ("E200", 2, 5), ("E201", 3, 10)], None),
        ("error_limit", "\n".join(f"bad statement {i}" for i in range(25)),
         [("E100", i, 1) for i in range(1, 21)], None),
    ]
    return [
        {"name": name, "source": source, "expected": expected, "hint": hint}
        for name, source, expected, hint in rows
    ]


def diagnostics_worker(args: argparse.Namespace) -> dict:
    from scratchv.frontend.dsl_extended import ExtendedDSLParser
    from scratchv.main import main as compiler_main

    results = []
    with tempfile.TemporaryDirectory(prefix="dsl-diagnostics-") as directory:
        for case in diagnostic_cases():
            source = case["source"]
            path = Path(directory) / f"{case['name']}.dsl"
            path.write_bytes(source.encode("utf-8"))

            def validate() -> Any:
                return ExtendedDSLParser().validate(source, filename=str(path))

            collector = validate()
            errors = collector.errors
            actual = [(error.error_code, error.line, error.col) for error in errors]
            rendered = collector.report()
            stdout, stderr = io.StringIO(), io.StringIO()
            output = path.with_suffix(".s")
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = compiler_main([str(path), "-o", str(output)])
            cli_text = stderr.getvalue()
            checks = {
                "codes_and_locations": actual == case["expected"],
                "source_and_filename": all(
                    error.filename == str(path)
                    and error.source_line == source.splitlines()[error.line - 1]
                    for error in errors
                ),
                "hint": case["hint"] is None or case["hint"] in rendered,
                "plain_output": "\x1b[" not in rendered + cli_text,
                "source_marker": "^" in rendered,
                "limit": collector.limit_reached == (case["name"] == "error_limit"),
                "cli_exit_code": exit_code == 1,
                "cli_diagnostics_once": cli_text.count(": error[") == len(case["expected"]),
                "cli_locations": all(
                    f"{path}:{line}:{col}: error[{code}]" in cli_text
                    for code, line, col in case["expected"]
                ),
                "no_output_artifact": not output.exists(),
            }
            results.append({
                "name": case["name"], "passed": all(checks.values()), "checks": checks,
                "source": source, "expected": case["expected"], "actual": actual,
                "actual_codes": [error.error_code for error in errors],
                "actual_count": len(errors), "limit_reached": collector.limit_reached,
                "rendered": rendered, "cli_stderr": cli_text, "cli_exit_code": exit_code,
                "validation": measure(validate, args.warmup, args.groups, args.iterations),
                "rendering": measure(collector.report, args.warmup, args.groups, args.iterations),
            })
    return {"passed": all(case["passed"] for case in results), "cases": results}


def run_worker(root: Path, mode: str, args: argparse.Namespace) -> dict:
    if not (root / "scratchv" / "frontend" / "dsl_extended.py").is_file():
        raise ValueError(f"{mode} checkout has no DSL parser: {root}")
    command = [
        sys.executable, "-I", "-X", "utf8", str(Path(__file__).resolve()),
        "--worker-root", str(root), "--worker-mode", mode,
        "--cases", str(args.cases), "--warmup", str(args.warmup),
        "--groups", str(args.groups), "--iterations", str(args.iterations),
    ]
    result = subprocess.run(
        command, cwd=root, capture_output=True, text=True, encoding="utf-8",
        timeout=args.worker_timeout,
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
    )
    if result.returncode:
        raise RuntimeError(f"{mode} worker failed at {root}:\n{result.stderr}")
    return json.loads(result.stdout)


def compare_parsing(
    baseline: dict,
    current: dict,
    target: float,
    allowed_ir_changes: set[str] | None = None,
) -> dict:
    same_cases = baseline["case_hashes"] == current["case_hashes"]
    same_ir = baseline["ir_hashes"] == current["ir_hashes"]
    changed_cases = sorted(
        name
        for name, current_hash in current["ir_hashes"].items()
        if baseline["ir_hashes"].get(name) != current_hash
    )
    allowed = allowed_ir_changes or set()
    unexpected_changes = sorted(set(changed_cases) - allowed)
    ratio = current["median_s"] / baseline["median_s"]
    return {
        "baseline": baseline, "current": current, "same_cases": same_cases,
        "ir_equal": same_ir, "ratio": ratio, "target_ratio": target,
        "target_met": ratio <= target,
        "ir_changed_cases": changed_cases,
        "allowed_ir_changes": sorted(allowed),
        "unexpected_ir_changes": unexpected_changes,
    }


def build_report(args: argparse.Namespace) -> dict:
    report: dict[str, Any] = {
        "schema_version": 1, "benchmark": "DSL diagnostics", "status": "failed",
        "environment": {
            "python": platform.python_version(), "executable": sys.executable,
            "platform": platform.platform(), "processor": platform.processor(),
            "dependencies": {name: importlib.metadata.version(name) for name in ("onnx", "numpy")},
            "worker_threads": 1,
        },
        "configuration": {
            "warmup": args.warmup, "groups": args.groups, "iterations": args.iterations,
            "performance_enforced": args.enforce_performance,
            "timing_scope": "one batch = iterations through the full corpus; parser construction + parse; excludes imports, file I/O and IR hashing",
        },
        "warnings": [],
    }
    try:
        # Record diagnostic evidence even if the legacy parser cannot parse a case.
        report["diagnostics"] = run_worker(ROOT, "diagnostics", args)
        if args.baseline_root is None:
            raise ValueError("--baseline-root is required; no baseline is fabricated")
        baseline = run_worker(args.baseline_root, "baseline", args)
        current = run_worker(ROOT, "current", args)
        parsing = compare_parsing(
            baseline,
            current,
            args.max_parse_ratio,
            set(args.allow_ir_change),
        )
        report["parsing"] = parsing
        if not parsing["same_cases"]:
            raise ValueError("baseline/current case corpus differs")
        if parsing["unexpected_ir_changes"]:
            changed = ", ".join(parsing["unexpected_ir_changes"])
            raise ValueError(f"generated IR differs for unexpected cases: {changed}")
        if not parsing["target_met"]:
            report["warnings"].append(
                f"Parsing ratio {parsing['ratio']:.3f}x exceeds target {args.max_parse_ratio:.3f}x."
            )
        if report["diagnostics"]["passed"] and (
            parsing["target_met"] or not args.enforce_performance
        ):
            report["status"] = "passed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def markdown_report(report: dict, *, include_examples: bool = True) -> str:
    lines = ["# DSL diagnostics", "", f"Functional/report status: **{report['status']}**", ""]
    if "error" in report:
        lines += ["## Error", "", "```text", report["error"], "```", ""]
    env = report["environment"]
    lines += [f"Python: {env['python']}; platform: {env['platform']}", ""]
    config = report["configuration"]
    lines += [f"Warmups: {config['warmup']}; groups: {config['groups']}; iterations/group: {config['iterations']}.",
              config["timing_scope"], ""]
    if "parsing" in report:
        parsing = report["parsing"]
        lines += ["## Correct DSL parsing", "", "| Metric | Baseline | Current |", "|---|---:|---:|"]
        for key, label in [("revision", "Revision"), ("median_s", "Batch median (s)")]:
            lines.append(f"| {label} | {parsing['baseline'][key]} | {parsing['current'][key]} |")
        lines += ["", f"Current / baseline: **{parsing['ratio']:.3f}x**; target: {parsing['target_ratio']:.3f}x; target met: **{parsing['target_met']}**.",
                  f"Same case corpus: {parsing['same_cases']}; identical IR dumps: {parsing['ir_equal']}.",
                  f"Performance target enforced: {config['performance_enforced']}.", "",
                  "Cases: " + ", ".join(parsing["current"]["case_hashes"]), ""]
        if parsing["ir_changed_cases"]:
            lines += [
                "Expected IR changes: "
                + ", ".join(parsing["ir_changed_cases"]),
                "",
            ]
    if "diagnostics" in report:
        lines += ["## Diagnostic acceptance and timings", "",
                  "Validation/rendering times are per operation (batch median divided by iterations). CLI runs are correctness checks, not timed.", "",
                  "| Case | Passed | Errors | Validate (ms) | Render (ms) |",
                  "|---|---|---:|---:|---:|"]
        for case in report["diagnostics"]["cases"]:
            scale = 1000 / config["iterations"]
            lines.append(f"| {case['name']} | {case['passed']} | {case['actual_count']} | {case['validation']['median_s'] * scale:.4f} | {case['rendering']['median_s'] * scale:.4f} |")
        for case in report["diagnostics"]["cases"]:
            if not case["passed"]:
                lines += ["", f"{case['name']} failed checks: " + ", ".join(key for key, value in case["checks"].items() if not value), ""]
        if include_examples:
            lines += ["", "## Diagnostic examples", ""]
            for case in report["diagnostics"]["cases"]:
                label = f"{case['name']}: {case['actual_count']} error(s) - view diagnostic log"
                lines += ["<details>", f"<summary>{html.escape(label)}</summary>", "",
                          "```text", case["rendered"], "```", "", "</details>", ""]
    for warning in report["warnings"]:
        lines += [f"Warning: {warning}", ""]
    return "\n".join(lines)


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, help="separate checkout of the PR base revision")
    parser.add_argument("--cases", type=Path, default=ROOT / "benchmarks" / "cases")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--groups", type=positive_int, default=10)
    parser.add_argument("--iterations", type=positive_int, default=100)
    parser.add_argument("--max-parse-ratio", type=positive_float, default=1.5)
    parser.add_argument("--enforce-performance", action="store_true")
    parser.add_argument(
        "--allow-ir-change",
        action="append",
        default=[],
        metavar="CASE",
        help="allow an expected baseline IR difference for one case filename",
    )
    parser.add_argument("--worker-timeout", type=positive_int, default=300)
    parser.add_argument("--json", action="store_true", help="emit only JSON to stdout")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--worker-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-mode", choices=("baseline", "current", "diagnostics"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    args.cases = args.cases.resolve()
    if args.baseline_root:
        args.baseline_root = args.baseline_root.resolve()
    if args.worker_root:
        args.worker_root = args.worker_root.resolve()
        sys.path.insert(0, str(args.worker_root))
        # Keep stdout a JSON channel even if imported compiler code logs there.
        with contextlib.redirect_stdout(sys.stderr):
            result = diagnostics_worker(args) if args.worker_mode == "diagnostics" else parse_worker(args)
        print(json.dumps(result, ensure_ascii=True))
        return 0
    report = build_report(args)
    json_text = json.dumps(report, indent=2, ensure_ascii=True) + "\n"
    markdown = markdown_report(report)
    html_examples = []
    for case in report.get("diagnostics", {}).get("cases", []):
        label = f"{case['name']}: {case['actual_count']} error(s) - view diagnostic log"
        html_examples.append(
            '<details><summary>' + html.escape(label) + '</summary><pre>'
            + html.escape(case["rendered"]) + '</pre></details>'
        )
    html_text = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>DSL diagnostics</title><style>'
        'body{max-width:1100px;margin:32px auto;padding:0 20px;font:15px/1.6 system-ui}'
        'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:24px}'
        'details{margin:12px 0;border:1px solid #d8dee6;border-radius:6px}'
        'summary{padding:12px;cursor:pointer}details pre{margin:0}'
        '</style><body><pre>' + html.escape(markdown_report(report, include_examples=False))
        + '</pre>' + ('<h2>Diagnostic examples</h2>' if html_examples else '')
        + ''.join(html_examples) + '</body></html>\n'
    )
    for path, content in [(args.json_output, json_text), (args.markdown, markdown), (args.html, html_text)]:
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    print(json_text if args.json else markdown, end="\n")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
