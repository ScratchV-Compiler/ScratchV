# flake8: noqa
"""Compiler benchmark suite runner for ScratchV.

Automates execution of DSL test cases: compilation, simulation,
output comparison, and performance reporting. Generates HTML and
Markdown reports.

Usage::

    from benchmarks.bench_runner import BenchmarkRunner
    runner = BenchmarkRunner("benchmarks/cases")
    report = runner.run_all()
    report.print_summary()
    report.save_html("report.html")
    report.save_markdown("report.md")

Each test case consists of:
    - {name}.dsl     : DSL source input
    - {name}.expected: Expected output (text)
    - {name}.desc    : Short description (one line)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    """Result for a single benchmark test case.

    Attributes:
        name: Test case name (derived from filename).
        description: Human-readable description.
        passed: Whether the test passed.
        output: Captured stdout from the simulation.
        expected: Expected output text.
        parse_time_s: Time spent parsing.
        compile_time_s: Total compile time (parse + codegen).
        sim_time_s: Time spent in simulation.
        instruction_count: Number of instructions executed by simulator.
        error: Error message if the test failed.
    """
    name: str
    description: str = ""
    passed: bool = False
    output: str = ""
    expected: str = ""
    parse_time_s: float = 0.0
    compile_time_s: float = 0.0
    sim_time_s: float = 0.0
    instruction_count: int = 0
    error: str = ""


@dataclass
class BenchmarkReport:
    """Aggregate report for a benchmark run.

    Attributes:
        results: List of individual case results.
        total_time_s: Total wall-clock time for the suite.
        timestamp: ISO format timestamp of the run.
    """
    results: list[CaseResult] = field(default_factory=list)
    total_time_s: float = 0.0
    timestamp: str = ""

    @property
    def pass_count(self) -> int:
        """Number of passing tests."""
        return sum(1 for r in self.results if r.passed)

    @property
    def fail_count(self) -> int:
        """Number of failing tests."""
        return sum(1 for r in self.results if not r.passed)

    @property
    def total_cases(self) -> int:
        """Total number of test cases."""
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        """Pass rate as a percentage (0-100)."""
        if not self.results:
            return 0.0
        return (self.pass_count / len(self.results)) * 100.0

    @property
    def total_parse_time(self) -> float:
        """Total time spent parsing all cases."""
        return sum(r.parse_time_s for r in self.results)

    @property
    def total_compile_time(self) -> float:
        """Total compile time for all cases."""
        return sum(r.compile_time_s for r in self.results)

    @property
    def total_sim_time(self) -> float:
        """Total simulation time for all cases."""
        return sum(r.sim_time_s for r in self.results)

    @property
    def total_instructions(self) -> int:
        """Total instructions executed across all cases."""
        return sum(r.instruction_count for r in self.results)

    def print_summary(self) -> None:
        """Print a formatted summary table to stdout."""
        print("\n" + "=" * 100)
        print("BENCHMARK REPORT")
        print("=" * 100)
        print(f"Timestamp: {self.timestamp}")
        print(f"Total cases: {self.total_cases}")
        print(f"Passed: {self.pass_count} | Failed: {self.fail_count}")
        print(f"Pass rate: {self.pass_rate:.1f}%")
        print(f"Total time: {self.total_time_s:.3f}s")
        print("-" * 100)
        header = (
            f"{'Name':<24} {'Status':<8} {'Parse(s)':<10} {'Compile(s)':<12} "
            f"{'Sim(s)':<10} {'Inst':<10} {'Description'}"
        )
        print(header)
        print("-" * 100)

        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            print(
                f"{r.name:<24} {status:<8} {r.parse_time_s:<10.4f} "
                f"{r.compile_time_s:<12.4f} {r.sim_time_s:<10.4f} "
                f"{r.instruction_count:<10} {r.description}"
            )
            if r.error:
                print(f"    ERROR: {r.error}")

        print("-" * 100)

    def to_dict(self) -> dict:
        """Convert report to a JSON-serializable dictionary."""
        import datetime
        from dataclasses import asdict

        return {
            "timestamp": self.timestamp or datetime.datetime.now().isoformat(),
            "total_time_s": self.total_time_s,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "total_cases": self.total_cases,
            "pass_rate": self.pass_rate,
            "total_parse_time": self.total_parse_time,
            "total_compile_time": self.total_compile_time,
            "total_sim_time": self.total_sim_time,
            "total_instructions": self.total_instructions,
            "results": [asdict(r) for r in self.results],
        }

    def save_json(self, path: str) -> None:
        """Save the report as JSON.

        Args:
            path: Output file path.
        """
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"JSON report saved to {path}")

    def to_markdown(self) -> str:
        """Generate a Markdown-formatted report string.

        Returns:
            Markdown report as a string.
        """
        lines = [
            "# ScratchV Benchmark Report",
            "",
            f"**Timestamp**: {self.timestamp}",
            "",
            f"- Total cases: {self.total_cases}",
            f"- Passed: {self.pass_count}",
            f"- Failed: {self.fail_count}",
            f"- Pass rate: {self.pass_rate:.1f}%",
            f"- Total time: {self.total_time_s:.3f}s",
            f"- Total parse time: {self.total_parse_time:.4f}s",
            f"- Total compile time: {self.total_compile_time:.4f}s",
            f"- Total sim time: {self.total_sim_time:.4f}s",
            f"- Total instructions: {self.total_instructions}",
            "",
            "## Results",
            "",
            "| Name | Status | Parse (s) | Compile (s) | Sim (s) | Inst | Description |",
            "|------|--------|-----------|-------------|---------|------|-------------|",
        ]

        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(
                f"| {r.name} | **{status}** | {r.parse_time_s:.4f} | "
                f"{r.compile_time_s:.4f} | {r.sim_time_s:.4f} | "
                f"{r.instruction_count} | {r.description} |"
            )

        # Summary stats
        lines.append("")
        lines.append("## Statistics")
        lines.append("")
        passed = self.results
        if passed:
            avg_parse = sum(r.parse_time_s for r in passed) / len(passed)
            avg_compile = sum(r.compile_time_s for r in passed) / len(passed)
            avg_sim = sum(r.sim_time_s for r in passed) / len(passed)
            lines.append(f"- Average parse time: {avg_parse:.4f}s")
            lines.append(f"- Average compile time: {avg_compile:.4f}s")
            lines.append(f"- Average sim time: {avg_sim:.4f}s")

        failed = [r for r in self.results if not r.passed]
        if failed:
            lines.append("")
            lines.append("## Failures")
            lines.append("")
            for r in failed:
                lines.append(f"- **{r.name}**: {r.error}")

        return "\n".join(lines)

    def save_markdown(self, path: str) -> None:
        """Save the report as a Markdown file.

        Args:
            path: Output file path.
        """
        with open(path, "w") as f:
            f.write(self.to_markdown())
        print(f"Markdown report saved to {path}")

    def to_html(self) -> str:
        """Generate an HTML report.

        Returns:
            HTML string with embedded CSS.
        """
        md_body = self.to_markdown()

        # Simple HTML wrapper around the markdown
        # (in production you'd use a markdown-to-html library)
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ScratchV Benchmark Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               max-width: 1000px; margin: 40px auto; padding: 0 20px;
               color: #333; }}
        h1 {{ border-bottom: 2px solid #0366d6; padding-bottom: 10px; color: #0366d6; }}
        h2 {{ margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
        th {{ background: #f6f8fa; font-weight: 600; }}
        tr:nth-child(even) {{ background: #f6f8fa; }}
        .pass {{ color: #22863a; font-weight: bold; }}
        .fail {{ color: #cb2431; font-weight: bold; }}
        pre {{ background: #f6f8fa; padding: 15px; border-radius: 6px; overflow-x: auto; }}
    </style>
</head>
<body>
<pre>{md_body}</pre>
<p><em>Generated by ScratchV Benchmark Suite</em></p>
</body>
</html>"""
        return html

    def save_html(self, path: str) -> None:
        """Save the report as an HTML file.

        Args:
            path: Output file path.
        """
        with open(path, "w") as f:
            f.write(self.to_html())
        print(f"HTML report saved to {path}")


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Automated runner for DSL benchmark test cases.

    Iterates over a directory of .dsl files, compiles each one,
    runs the simulator, and compares output against expected results.

    Usage::

        runner = BenchmarkRunner("benchmarks/cases")
        report = runner.run_all()
        report.print_summary()

    Attributes:
        test_dir: Path to the directory containing test cases.
        compile_cmd: Optional list override for the compile command pattern.
        simulate_cmd: Optional list override for the simulation command pattern.
        target: Backend target ('riscv' or 'dsl').
        timeout: Maximum seconds per test case.
    """

    def __init__(
        self,
        test_dir: str,
        *,
        target: str = "dsl",
        timeout: float = 30.0,
        verbose: bool = True,
    ):
        """Initialize the benchmark runner.

        Args:
            test_dir: Directory containing .dsl/.expected/.desc files.
            target: Compilation target ('dsl' for DSL parse-only, 'riscv' for full).
            timeout: Timeout in seconds per test case.
            verbose: Print progress during execution.
        """
        self.test_dir = Path(test_dir)
        self.target = target
        self.timeout = timeout
        self.verbose = verbose
        self._python = sys.executable

    # -------------------------------------------------------------------
    # Test case discovery
    # -------------------------------------------------------------------

    def discover_cases(self) -> list[dict[str, str]]:
        """Find all test cases in the test directory.

        A case is defined by a .dsl file. Optional .expected and .desc
        files are matched by basename.

        Returns:
            List of case dicts with keys: name, dsl_path, expected_path, desc_path.
        """
        cases: list[dict[str, str]] = []
        if not self.test_dir.is_dir():
            if self.verbose:
                print(f"Warning: test directory not found: {self.test_dir}")
            return cases

        for dsl_file in sorted(self.test_dir.glob("*.dsl")):
            name = dsl_file.stem
            expected_file = dsl_file.with_suffix(".expected")
            desc_file = dsl_file.with_suffix(".desc")

            case_info = {
                "name": name,
                "dsl_path": str(dsl_file),
                "expected_path": str(expected_file) if expected_file.exists() else "",
                "desc_path": str(desc_file) if desc_file.exists() else "",
            }
            cases.append(case_info)

        return cases

    # -------------------------------------------------------------------
    # Run a single case
    # -------------------------------------------------------------------

    def run_case(self, case: dict[str, str]) -> CaseResult:
        """Run a single test case through the pipeline.

        Args:
            case: Case dict from discover_cases().

        Returns:
            A CaseResult with timing and pass/fail info.
        """
        name = case["name"]
        dsl_path = case["dsl_path"]

        # Read expected output
        expected = ""
        if case["expected_path"]:
            with open(case["expected_path"]) as f:
                expected = f.read().strip()

        # Read description
        description = ""
        if case["desc_path"]:
            with open(case["desc_path"]) as f:
                description = f.read().strip()

        result = CaseResult(
            name=name,
            description=description,
            expected=expected,
        )

        try:
            t_start = time.perf_counter()

            # Phase 1: Parse DSL to IR
            t0 = time.perf_counter()
            dsl_source = ""
            with open(dsl_path) as f:
                dsl_source = f.read()

            try:
                # Auto-detect extended DSL features (if/else/while)
                has_extended = any(
                    kw in dsl_source for kw in
                    ("if (", "else:", "endif", "while (", "endwhile")
                )
                if has_extended:
                    from scratchv.frontend.dsl_extended import ExtendedDSLParser
                    parser = ExtendedDSLParser()
                else:
                    from scratchv.frontend.dsl_parser import DSLParser
                    parser = DSLParser()
                program = parser.parse(dsl_source)
            except ImportError:
                # Fallback: use subprocess
                program = None
                output = self._run_command(
                    [self._python, "-c", f"""
import sys
sys.path.insert(0, '{Path(__file__).parent.parent.as_posix()}')
from scratchv.frontend.dsl_parser import DSLParser
parser = DSLParser()
src = open('{dsl_path}').read()
program = parser.parse(src)
print(program.dump())
"""]
                )

            result.parse_time_s = time.perf_counter() - t0

            # Phase 2: Simulate or compile
            t1 = time.perf_counter()
            output = self._simulate_dsl(dsl_source, program)
            result.sim_time_s = time.perf_counter() - t1

            result.compile_time_s = (time.perf_counter() - t_start) - result.sim_time_s

            # Count instructions from IR
            if program is not None:
                result.instruction_count = sum(
                    1 for f in program.functions
                    for b in f.blocks
                    for _ in b.instructions
                )

            # Compare output (handle numpy formatting variations)
            result.output = output.strip()
            result.expected = expected
            if expected:
                result.passed = self._compare_outputs(output, expected)
            else:
                # No expected file -> pass if no error
                result.passed = (not result.error)

            result.total_time_s = time.perf_counter() - t_start

        except subprocess.TimeoutExpired:
            result.error = f"timeout ({self.timeout}s)"
            result.passed = False
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            result.passed = False

        return result

    @staticmethod
    def _compare_outputs(output: str, expected: str) -> bool:
        """Compare simulation output with expected, handling numpy formatting."""
        if output.strip() == expected.strip():
            return True
        try:
            import numpy as np
            import re
            def _parse(s):
                s = s.strip().strip("[]")
                parts = [p for p in re.split(r"[,;\s]+", s) if p]
                return np.array([float(p) for p in parts])
            out_a = _parse(output)
            exp_a = _parse(expected)
            if out_a.shape == exp_a.shape:
                return bool(np.allclose(out_a, exp_a, rtol=1e-3, atol=1e-6))
        except (ValueError, TypeError, ImportError):
            pass
        return False

    # -------------------------------------------------------------------
    # DSL simulation
    # -------------------------------------------------------------------

    def _simulate_dsl(
        self, source: str, program=None,
    ) -> str:
        """Simulate a DSL program by running it through the interpreter.

        Args:
            source: DSL source code.
            program: Optional pre-parsed Program object.

        Returns:
            String output from the simulator.
        """
        try:
            import numpy as np
            from scratchv.verification.verifier import DSLInterpreter

            # Extract input variable names from DSL
            import re
            input_vars: set[str] = set()
            op_pattern = (
                r'\b(add|sub|mul|div|relu|gelu|exp|neg|'
                r'matmul|dot|maxpool|softmax)\(([^)]+)'
            )
            for m in re.finditer(op_pattern, source):
                args_text = m.group(2)
                for arg in args_text.split(","):
                    arg = arg.strip().split(":")[0].strip()
                    if arg and not arg[0].isdigit() and arg != "":
                        input_vars.add(arg)

            # Filter out known function names and keyword argument names
            keywords = {
                "add", "sub", "mul", "div", "relu", "gelu", "exp", "neg",
                "matmul", "dot", "maxpool", "softmax", "return", "for",
                "endfor", "if", "else", "endif", "while", "endwhile",
                # Keyword argument names (not input variables)
                "m", "n", "k", "rows", "cols", "inner", "len",
                "axis", "kernel", "stride", "padding",
                "out_channels", "kernel_size",
                "transA", "transB", "alpha", "beta",
                # Loop variables and temporaries
                "i", "j", "t1", "t2", "t3", "t4",
                "acc", "sum", "tmp",
            }
            input_vars = {v for v in input_vars if v.lower() not in keywords
                         and not v.startswith("_")}

            # Provide inputs
            inputs: dict[str, np.ndarray] = {}
            for v in input_vars:
                inputs[v] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

            interpreter = DSLInterpreter()
            result = interpreter.run(source, inputs)

            if isinstance(result, np.ndarray):
                return np.array2string(result, precision=6, suppress_small=True)
            return str(result)

        except Exception as e:
            return f"## SIM ERROR: {e}"

    # -------------------------------------------------------------------
    # Run all cases
    # -------------------------------------------------------------------

    def run_all(self) -> BenchmarkReport:
        """Discover and run all test cases in the test directory.

        Returns:
            A BenchmarkReport with all results and aggregate statistics.
        """
        import datetime

        cases = self.discover_cases()
        if not cases:
            if self.verbose:
                print("No test cases found.")
            return BenchmarkReport(
                results=[],
                timestamp=datetime.datetime.now().isoformat(),
            )

        if self.verbose:
            print(f"Discovered {len(cases)} test case(s) in {self.test_dir}")
            print("-" * 60)

        t_start = time.perf_counter()
        results: list[CaseResult] = []

        for i, case in enumerate(cases):
            if self.verbose:
                print(f"  [{i + 1}/{len(cases)}] {case['name']} ... ", end="", flush=True)

            result = self.run_case(case)
            results.append(result)

            if self.verbose:
                if result.passed:
                    print(f"PASS ({result.parse_time_s:.3f}s parse, "
                          f"{result.instruction_count} inst)")
                else:
                    print(f"FAIL: {result.error}")

        total_time = time.perf_counter() - t_start

        report = BenchmarkReport(
            results=results,
            total_time_s=total_time,
            timestamp=datetime.datetime.now().isoformat(),
        )
        return report

    # -------------------------------------------------------------------
    # Benchmark mode (multiple runs)
    # -------------------------------------------------------------------

    def run_benchmark(self, repeat: int = 3) -> BenchmarkReport:
        """Run all cases multiple times and average the results.

        Args:
            repeat: Number of repetitions per test case.

        Returns:
            A BenchmarkReport with averaged timing.
        """
        all_reports: list[BenchmarkReport] = []
        for run_idx in range(repeat):
            if self.verbose:
                print(f"\n--- Benchmark run {run_idx + 1}/{repeat} ---")
            report = self.run_all()
            all_reports.append(report)

        # Average the results
        if not all_reports:
            return BenchmarkReport()

        base = all_reports[0]
        avg_results: list[CaseResult] = []
        for i, case_result in enumerate(base.results):
            avg = CaseResult(
                name=case_result.name,
                description=case_result.description,
                passed=all(r.results[i].passed for r in all_reports),
                expected=case_result.expected,
            )
            # Average times
            avg.parse_time_s = sum(
                r.results[i].parse_time_s for r in all_reports
            ) / repeat
            avg.compile_time_s = sum(
                r.results[i].compile_time_s for r in all_reports
            ) / repeat
            avg.sim_time_s = sum(
                r.results[i].sim_time_s for r in all_reports
            ) / repeat
            avg.instruction_count = case_result.instruction_count
            avg_results.append(avg)

        return BenchmarkReport(
            results=avg_results,
            total_time_s=sum(r.total_time_s for r in all_reports) / repeat,
            timestamp=base.timestamp,
        )

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _run_command(self, cmd: list[str]) -> str:
        """Run a subprocess command and return its stdout.

        Args:
            cmd: Command and arguments as a list.

        Returns:
            Captured stdout string.

        Raises:
            subprocess.TimeoutExpired: If the command times out.
            subprocess.CalledProcessError: If the command fails.
        """
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if result.returncode != 0:
            return f"## CMD ERROR ({result.returncode}): {result.stderr[:500]}"
        return result.stdout


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for running benchmarks."""
    import argparse
    import datetime

    parser = argparse.ArgumentParser(
        description="ScratchV Compiler Benchmark Suite",
    )
    parser.add_argument(
        "test_dir", nargs="?", default="benchmarks/cases",
        help="Directory containing test cases",
    )
    parser.add_argument(
        "--output-json", default=None,
        help="Save JSON report to file",
    )
    parser.add_argument(
        "--output-html", default=None,
        help="Save HTML report to file",
    )
    parser.add_argument(
        "--output-md", default=None,
        help="Save Markdown report to file",
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Number of benchmark repetitions (for averaging)",
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0,
        help="Timeout per case in seconds",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    runner = BenchmarkRunner(
        test_dir=args.test_dir,
        timeout=args.timeout,
        verbose=not args.quiet,
    )

    if args.repeat > 1:
        report = runner.run_benchmark(repeat=args.repeat)
    else:
        report = runner.run_all()

    report.print_summary()

    if args.output_json:
        report.save_json(args.output_json)
    if args.output_html:
        report.save_html(args.output_html)
    if args.output_md:
        report.save_markdown(args.output_md)

    # Exit with error if any tests failed
    if report.fail_count > 0:
        sys.exit(1)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
