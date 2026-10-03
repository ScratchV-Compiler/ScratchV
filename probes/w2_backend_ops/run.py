"""W2 operator gate: independent formulas -> ORT -> IR -> tensor C -> RV64 QEMU.

All seven families and both optimization levels are mandatory. Missing tools,
nonfinite tensors, shape/type changes or max_abs >= 1e-5 fail the gate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from html import escape
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import onnx
import onnxruntime as ort

from probes.w2_backend_ops.cases import build_cases
from probes.w2_qwen3_small.diagnostics import tensor_diff
from scratchv.analysis.ir_verifier import verify_ir
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.pass_manager import create_optimization_pass_manager
from scratchv.runtime.riscv_tensor import build_riscv_tensor, discover_toolchain, run_riscv_tensor
from scratchv.verification.ir_interpreter import IRInterpreter

ATOL = 1e-5
LEVELS = ("none", "all")
FAMILIES = ("matmul", "elementwise", "softmax", "rmsnorm", "rope", "swiglu", "gqa")
FRONTEND_FAMILIES = ("rmsnorm", "rope", "swiglu", "gqa")


def require_valid(program):
    passed, issues = verify_ir(program)
    if not passed:
        raise ValueError("; ".join(str(issue) for issue in issues))


def reference_case(case, path):
    onnx.save(case.model, path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    expected = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"]).run(None, case.feed)[0]
    parser = ONNXParser()
    program = parser.parse(str(path))
    require_valid(program)
    return expected, parser, program


def interpret(program, bindings, feed, level):
    manager = create_optimization_pass_manager(level)
    manager.before_pass = lambda p, value: require_valid(value)
    manager.after_pass = lambda p, value: require_valid(value)
    optimized = manager.run_pipeline(copy.deepcopy(program)).data
    require_valid(optimized)
    return IRInterpreter(optimized).run(feed, initializers=bindings).return_value


def evidence():
    sources = [Path(__file__), ROOT / "probes/w2_backend_ops/cases.py",
               ROOT / "probes/w2_qwen3_small/diagnostics.py", ROOT / "scratchv/compiler.py",
               ROOT / "scratchv/frontend/onnx_parser.py", ROOT / "scratchv/ir/types.py",
               ROOT / "scratchv/ir/builder.py", ROOT / "scratchv/pass_manager.py",
               ROOT / "scratchv/pass_interface.py", ROOT / "scratchv/verification/ir_interpreter.py",
               ROOT / "scratchv/verification/ir_numpy_ops.py",
               ROOT / "scratchv/backend/tensor_c_codegen.py", ROOT / "scratchv/runtime/riscv_tensor.py",
               *sorted((ROOT / "scratchv/analysis").glob("*.py")),
               *sorted((ROOT / "scratchv/optimizer").glob("*.py"))]
    return {"environment": {"python": platform.python_version(), "executable": sys.executable,
                            "platform": platform.platform(), "numpy": np.__version__,
                            "onnx": onnx.__version__, "onnxruntime": ort.__version__},
            "source_sha256": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sources}}


def run_probe(out, report, *, cc=None, qemu=None, timeout=120):
    cases = build_cases()
    if (not cases or len({case.name for case in cases}) != len(cases)
            or set(case.family for case in cases) != set(FAMILIES)):
        raise ValueError("Cases must have unique names and cover every required operator family")
    report.update(evidence(), stage="toolchain", families=list(FAMILIES), levels=list(LEVELS),
                  planned_executions=len(cases) * len(LEVELS), cases=[])
    tools = discover_toolchain(cc=cc, qemu=qemu)
    report["toolchain"] = {"cc": list(tools.cc), "qemu": tools.qemu}
    report["stage"] = "execution"
    for case in cases:
        folder = out / case.name
        folder.mkdir()
        row = {"name": case.name, "family": case.family, "purpose": case.purpose,
               "passed": False, "stage": "reference", "executions": []}
        report["cases"].append(row)
        try:
            path = folder / "model.onnx"
            expected, parser, program = reference_case(case, path)
            np.savez(folder / "inputs.npz", **case.feed)
            np.save(folder / "numpy.npy", case.expected)
            np.save(folder / "ort.npy", expected)
            row["model_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            row["input_sha256"] = {name: hashlib.sha256(array.tobytes()).hexdigest()
                                   for name, array in case.feed.items()}
            row["ort_vs_numpy"] = tensor_diff(expected, case.expected, ATOL)
            if not row["ort_vs_numpy"]["passed"]:
                raise ValueError("ORT disagrees with independent NumPy formula")
            for level in LEVELS:
                target = folder / level
                target.mkdir()
                execution = {"optimization": level, "passed": False, "stage": "ir",
                             "qemu_process_wall_seconds": None}
                row["executions"].append(execution)
                try:
                    ir_output = interpret(program, parser.initializers, case.feed, level)
                    np.save(target / "ir.npy", ir_output)
                    execution["ir_vs_ort"] = tensor_diff(ir_output, expected, ATOL)
                    execution["ir_vs_numpy"] = tensor_diff(ir_output, case.expected, ATOL)
                    if not all(execution[key]["passed"] for key in ("ir_vs_ort", "ir_vs_numpy")):
                        raise ValueError("IR disagrees with ORT or independent NumPy formula")
                    execution["stage"] = "compile"
                    driver = CompilerDriver(CompilerConfig(backend="tensor-c", optimize_level=level, verify_ir=True))
                    result = driver.compile(str(path), str(target / "model.c"))
                    if not result.success:
                        raise RuntimeError("; ".join(result.errors))
                    executable = build_riscv_tensor(driver.tensor_artifact, target / "build", tools)
                    execution.update(elf_sha256=executable.elf_sha256, compile_command=executable.compile_command,
                                     compile_seconds=executable.compile_seconds,
                                     workspace_bytes=executable.workspace_bytes, tool_versions=executable.tool_versions,
                                     stage="qemu")
                    actual = run_riscv_tensor(executable, case.feed, target / "run", timeout=timeout)
                    execution.update(qemu_process_wall_seconds=actual.elapsed_s, command=actual.command, stage="numeric")
                    np.save(target / "qemu.npy", actual.output)
                    execution["qemu_vs_ort"] = tensor_diff(actual.output, expected, ATOL)
                    execution["qemu_vs_numpy"] = tensor_diff(actual.output, case.expected, ATOL)
                    execution["passed"] = all(execution[key]["passed"] for key in ("qemu_vs_ort", "qemu_vs_numpy"))
                    execution["stage"] = "complete"
                except Exception as exc:
                    execution["error"] = f"{type(exc).__name__}: {exc}"
                    if getattr(exc, "elapsed_s", None) is not None:
                        execution.update(qemu_process_wall_seconds=exc.elapsed_s,
                                         command=getattr(exc, "command", None),
                                         status=getattr(exc, "status", "runtime_error"))
                print(f"[{case.name}/{level}] {'PASS' if execution['passed'] else 'FAIL'} "
                      f"max_abs={execution.get('qemu_vs_ort', {}).get('max_abs')} "
                      f"stage={execution['stage']}", flush=True)
            row["passed"] = len(row["executions"]) == len(LEVELS) and all(item["passed"] for item in row["executions"])
            row["stage"] = "complete"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
    executions = [execution for case in report["cases"] for execution in case["executions"]]
    report["completed_executions"] = sum(item["stage"] == "complete" for item in executions)
    report["passed_executions"] = sum(item["passed"] for item in executions)
    measured = [item["qemu_process_wall_seconds"] for item in executions if item["qemu_process_wall_seconds"] is not None]
    report["qemu_process_wall_seconds"] = {"measured_count": len(measured),
                                           "total": sum(measured) if measured else None}
    report["passed"] = (len(executions) == report["planned_executions"]
                        and all(case["passed"] for case in report["cases"]))
    report["stage"] = "complete"


def _report_markdown(report):
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    lines = ["# W2 backend operator gate", "", f"Result: {'PASS' if report['passed'] else 'FAIL'}",
             "", "Strict max_abs < 1e-5; rtol=0; exact shape/dtype; finite values.",
             "", "| Case | Level | Result | QEMU vs ORT max_abs | QEMU process seconds |",
             "|---|---|---|---|---|"]
    for case in report.get("cases", []):
        if not case["executions"]:
            lines.append(f"| {case['name']} | not started | FAIL | - | - |")
        for item in case["executions"]:
            lines.append(f"| {case['name']} | {item['optimization']} | {'PASS' if item['passed'] else 'FAIL'} | "
                         f"{item.get('qemu_vs_ort', {}).get('max_abs')} | {item['qemu_process_wall_seconds']} |")
    lines.extend(["", "Timing includes QEMU startup, guest execution, UART transfer and exit; not guest cycles.",
                  "", "Full diagnostics, commands, environment and source hashes:", "", "```json", payload.rstrip(), "```", ""])
    return "\n".join(lines)


def write_reports(out, report):
    """Keep primary errors and publish JSON only after required evidence succeeds."""
    failures = []

    def write(name, content, *, atomic=False):
        path = out / name
        pending = out / (name + ".tmp") if atomic else path
        try:
            pending.write_text(content, encoding="utf-8")
            if atomic:
                pending.replace(path)
        except OSError as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
        finally:
            if atomic:
                try:
                    pending.unlink(missing_ok=True)
                except OSError:
                    pass

    def record_failure():
        report["report_write_errors"] = failures
        if report["passed"]:
            report.update(passed=False, stage="report-write", error="Incomplete backend gate evidence")

    def views():
        markdown = _report_markdown(report)
        write("report.md", markdown)
        write("report.html", "<!doctype html><meta charset='utf-8'><title>W2 backend operators</title>"
              "<h1>W2 backend operators</h1><pre>" + escape(markdown) + "</pre>")

    views()
    if failures:
        record_failure()
        views()
    # Write and close a sibling temporary file before publishing JSON. A late
    # write/close failure must not leave a valid PASS artifact for CI consumers.
    before_json = len(failures)
    write("report.json", json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", atomic=True)
    if len(failures) != before_json:
        record_failure()
        views()
    if failures:
        print("[unit:backend-ops] FAIL; report write errors: " + "; ".join(failures), file=sys.stderr)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", type=Path, default=ROOT / "output/w2-backend-ops")
    cli.add_argument("--cc")
    cli.add_argument("--qemu")
    cli.add_argument("--timeout", type=float, default=120)
    args = cli.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        cli.error("--timeout must be finite and positive")
    out = args.output_dir.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        cli.error("--output-dir must be empty")
    out.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "stage": "setup", "backend": "ScratchV tensor-c -> RV64GC -> QEMU",
              "atol": ATOL, "rtol": 0, "timeout_seconds": args.timeout}
    start = time.perf_counter()
    try:
        run_probe(out, report, cc=args.cc, qemu=args.qemu, timeout=args.timeout)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    report["seconds"] = time.perf_counter() - start
    write_reports(out, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
