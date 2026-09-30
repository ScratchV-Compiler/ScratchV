"""Correctness and full run() timing for direct Program cases.

Construction, reference evaluation and comparisons are outside timing. Each
run includes verification, plan construction, binding copies and output copy.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks.ir_interpreter_cases import (
    CASE_FACTORIES,
    TENSOR_BENCHMARK_CASES,
    compare,
    make_case,
)
from benchmarks.ir_interpreter_summary import (
    describe_case,
    describe_failure,
    render_summary,
)
from scratchv.verification.ir_interpreter import IRExecutionError, IRInterpreter


def environment():
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        np.show_config()
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "blas_config": stream.getvalue(),
        "thread_environment": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def benchmark(*, cases=None, warmup=1, repeats=3):
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    names = list(TENSOR_BENCHMARK_CASES) if cases is None else list(cases)
    if not names or any(name not in CASE_FACTORIES for name in names):
        raise ValueError("cases must be a nonempty selection of known cases")
    report = {
        "schema_version": 1,
        "environment": environment(),
        "timing_scope": "IRInterpreter.run: verification, plan, bindings, computation, return copy",
        "warmup": warmup,
        "repeats": repeats,
        "results": [],
    }
    for name in names:
        row = {"case": name, "correct": False, "status": "FAIL", "samples_s": []}
        report["results"].append(row)
        try:
            case = make_case(name)
            row.update(describe_case(case))
            row.update(
                {
                    "seed": case.seed,
                    "tensors": {
                        key: {"shape": list(x.shape), "dtype": str(x.dtype)}
                        for key, x in {**case.inputs, **case.initializers}.items()
                    },
                }
            )
            interpreter = IRInterpreter(case.program)

            def execute(interpreter=interpreter, case=case):
                return interpreter.run(
                    case.inputs, initializers=case.initializers, function_name="main"
                )

            result = execute()
            row["actual_shape"] = (
                list(result.return_value.shape)
                if result.return_value is not None
                else None
            )
            row["actual_dtype"] = (
                str(result.return_value.dtype)
                if result.return_value is not None
                else None
            )
            maximum = compare(result.return_value, case)
            for _ in range(warmup):
                compare(execute().return_value, case)
            for _ in range(repeats):
                start = time.perf_counter()
                result = execute()
                row["samples_s"].append(time.perf_counter() - start)
                maximum = max(maximum, compare(result.return_value, case))
            row.update(
                {
                    "status": "PASS",
                    "correct": True,
                    "max_abs_error": maximum,
                    "executed_steps": result.executed_steps,
                    "median_s": statistics.median(row["samples_s"]),
                    "min_s": min(row["samples_s"]),
                    "max_s": max(row["samples_s"]),
                }
            )
        except (
            IRExecutionError,
            AssertionError,
            ValueError,
            TypeError,
            OSError,
            MemoryError,
        ) as exc:
            row.update(describe_failure(exc))
    report["passed"] = all(row["correct"] for row in report["results"])
    return report


def markdown(report):
    return render_summary(report["results"], title="IR 解释器 benchmark") + (
        f"\n预热 {report['warmup']} 次，重复 {report['repeats']} 次；计时为一次完整 run，"
        "包含验证、计划、绑定、计算和返回副本。\n"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=list(CASE_FACTORIES))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("benchmark_reports/ir_interpreter.json"),
    )
    parser.add_argument(
        "--markdown", type=Path, default=Path("benchmark_reports/ir_interpreter.md")
    )
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be nonnegative and repeats positive")
    if args.json_output.resolve() == args.markdown.resolve():
        parser.error("JSON and Markdown output paths must differ")
    report = benchmark(cases=args.case, warmup=args.warmup, repeats=args.repeats)
    try:
        for path, content in (
            (
                args.json_output,
                json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
                + "\n",
            ),
            (args.markdown, markdown(report)),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"report writing failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"IR interpreter: {'PASS' if report['passed'] else 'FAIL'} ({len(report['results'])} cases)"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
