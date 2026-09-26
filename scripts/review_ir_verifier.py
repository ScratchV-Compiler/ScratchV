"""Optional, removable review tool: run tests and record their actual IR inputs.

Usage from the project root:
    .venv/bin/python -m scripts.review_ir_verifier
    .venv/bin/python -m scripts.review_ir_verifier -k 'loop or branch'
    .venv/bin/python -m scripts.review_ir_verifier --stdout

No compiler/test changes or persistent pytest hooks are required. Reports go to
ignored benchmark_reports/ by default. Exit status is pytest's test outcome,
not whether intentionally invalid IR appears in the report.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

from scratchv.analysis import IRVerifier, format_ir_error
from scratchv.analysis.ir_diagnostics import _instruction, _text, _value


DEFAULT_TESTS = [
    "tests/test_ir_verifier.py",
    "tests/test_ir_verifier_integration.py",
    "tests/test_ir_diagnostics.py",
    "tests/test_dsl_diagnostics_cli.py",
]


def value_text(value):
    return _value(value) + f" shape={value.shape!r}"


def function_text(function):
    lines = [f"function {_text(function.name)}"]
    for name in ("params", "returns", "locals"):
        values = getattr(function, name)
        lines.append(f"  {name}: " + (", ".join(value_text(v) for v in values) or "[]"))
    if not function.blocks:
        lines.append("  <no basic blocks>")
    for bi, block in enumerate(function.blocks):
        lines.append(f"  block #{bi} {_text(block.name)}:")
        for pi, instruction in enumerate(block.phi_nodes):
            lines.append(f"    phi[{pi}] {_instruction(instruction)}")
        if not block.instructions:
            lines.append("    <empty block>")
        for ii, instruction in enumerate(block.instructions):
            lines.append(f"    [{ii}] {_instruction(instruction)}")
            if instruction.dest is not None:
                dest = instruction.dest
                if dest.is_constant or dest.shape:
                    lines.append(f"        result metadata: is_constant={dest.is_constant!r} "
                                 f"const_value={dest.const_value!r} shape={dest.shape!r}")
            for oi, operand in enumerate(instruction.operands):
                if operand.shape:
                    lines.append(f"        operand[{oi}] shape={operand.shape!r}")
    return "\n".join(lines)


def program_text(program):
    lines = ["globals:"]
    lines.extend(f"  [{i}] {value_text(value)}" for i, value in enumerate(program.global_values))
    if not program.global_values:
        lines.append("  []")
    lines.extend(function_text(function) for function in program.functions)
    if not program.functions:
        lines.append("<empty Program>")
    return "\n".join(lines)


class ReviewRecorder:
    def __init__(self):
        self.tests = {}
        self.current = "<outside test>"
        self.verifying = 0

    def pytest_runtest_logstart(self, nodeid, location):
        self.current = nodeid
        self.tests.setdefault(nodeid, {"calls": [], "outcomes": [], "failures": []})

    def pytest_runtest_logreport(self, report):
        record = self.tests[report.nodeid]
        if report.when == "call" or report.failed or report.skipped:
            record["outcomes"].append(f"{report.when}: {report.outcome}")
        if report.failed:
            record["failures"].append(str(report.longrepr))

    def record(self, text):
        record = self.tests.setdefault(self.current, {"calls": [], "outcomes": [], "failures": []})
        record["calls"].append(text)

    def verify_wrapper(self, original):
        def verify(verifier, *args, **kwargs):
            before = program_text(verifier.program)
            self.verifying += 1
            try:
                issues = original(verifier, *args, **kwargs)
            except Exception as exc:
                self.record("IR input:\n" + before + f"\nEXCEPTION: {type(exc).__name__}: {exc}")
                raise
            finally:
                self.verifying -= 1
            passed = not any(issue.level.value == "error" for issue in issues)
            diagnostics = "\n\n".join(format_ir_error(issue) for issue in issues) or "<no diagnostics>"
            self.record(f"IRVerifier.verify stage={kwargs.get('stage')!r}\nIR input:\n{before}\n"
                        f"IR passed={passed}, diagnostics={len(issues)}\n{diagnostics}")
            return issues
        return verify

    def report(self, exit_code):
        lines = ["IR verifier test review", f"pytest exit_code={int(exit_code)}",
                 "Test passed means its assertions passed; IR passed=False is expected in negative cases.",
                 "All actual verification calls are listed, including repeated checks and mutated inputs.",
                 "Tests without IR calls are explicitly marked."]
        for nodeid, record in self.tests.items():
            lines += ["", "=" * 80, nodeid, "Test result: " + ", ".join(record["outcomes"])]
            if not record["calls"]:
                lines.append("<no IRVerifier.verify call in this test>")
            for index, text in enumerate(record["calls"], 1):
                lines += ["", f"--- call {index} ---", text]
            lines.extend(record["failures"])
        lines += ["", f"Total: {len(self.tests)} tests, "
                  f"{sum(len(r['calls']) for r in self.tests.values())} IR calls"]
        return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="append", help="pytest file or nodeid; repeat to select several")
    parser.add_argument("-k", default="", help="pytest expression, e.g. 'loop or branch'")
    parser.add_argument("--output", type=Path, default=Path("benchmark_reports/ir_verifier_review.txt"))
    parser.add_argument("--stdout", action="store_true", help="also print the full review to stdout")
    args = parser.parse_args()
    recorder = ReviewRecorder()
    # Patch only inside this command and restore even when pytest fails.
    with patch.object(IRVerifier, "verify", recorder.verify_wrapper(IRVerifier.verify)), \
            redirect_stdout(sys.stderr):
        result = pytest.main((args.test or DEFAULT_TESTS) + ["-q", "-k", args.k], plugins=[recorder])
    text = recorder.report(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    if args.stdout:
        print(text, end="")
    print(f"Review written to {args.output.resolve()}", file=sys.stderr)
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
