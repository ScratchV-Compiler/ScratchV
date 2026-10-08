"""W3 combined small Attention on RV64 QEMU; strict max_abs < 1e-4."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from probes.w3_attention.cases import build_cases
from probes.w3_common import new_output_dir, source_evidence, write_reports, sha256_file, process_peak_rss_bytes
from probes.w2_backend_ops.run import reference_case, interpret
from probes.w2_qwen3_small.diagnostics import tensor_diff
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.runtime.riscv_tensor import (
    RiscVTensorExecutionError, RiscVTensorTimeoutError,
    build_riscv_tensor, discover_toolchain, run_riscv_tensor,
)

ATOL = 1e-4
LEVELS = ("none", "all")

def run_probe(out, report, *, cc=None, qemu=None, timeout=180):
    cases = build_cases()
    tools = discover_toolchain(cc=cc, qemu=qemu)
    report.update(stage="execution", planned_cases=len(cases),
                  planned_executions=len(cases)*len(LEVELS), cases=[], invariants=[],
                  toolchain={"cc":list(tools.cc), "qemu":tools.qemu})
    results = {}
    for case in cases:
        folder = out / case.name
        folder.mkdir()
        row = {"name":case.name, "valid_length":case.valid_length, "passed":False, "executions":[]}
        report["cases"].append(row)
        try:
            path = folder / "model.onnx"
            expected, parser, program = reference_case(case, path)
            row["model_sha256"] = sha256_file(path)
            np.savez(folder / "inputs.npz", **case.feed)
            row["input_sha256"] = sha256_file(folder / "inputs.npz")
            np.save(folder / "numpy.npy", case.expected)
            np.save(folder / "ort.npy", expected)
            row["ort_vs_numpy"] = tensor_diff(expected, case.expected, ATOL)
            if not row["ort_vs_numpy"]["passed"]:
                raise ValueError("ORT disagrees with independent NumPy Attention")
            results[(case.name, "ort")] = expected
            for level in LEVELS:
                target = folder / level
                target.mkdir()
                execution = {"optimization":level, "passed":False, "stage":"ir",
                             "status":"not_started", "timeout_seconds":timeout}
                row["executions"].append(execution)
                try:
                    actual_ir = interpret(program, parser.initializers, case.feed, level)
                    np.save(target / "ir.npy", actual_ir)
                    execution["ir_vs_ort"] = tensor_diff(actual_ir, expected, ATOL)
                    execution["ir_vs_numpy"] = tensor_diff(actual_ir, case.expected, ATOL)
                    if not all(execution[k]["passed"] for k in ("ir_vs_ort", "ir_vs_numpy")):
                        raise ValueError("IR Attention mismatch")
                    execution["stage"] = "compile"
                    driver = CompilerDriver(CompilerConfig(backend="tensor-c", optimize_level=level, verify_ir=True))
                    compiled = driver.compile(str(path), str(target / "model.c"))
                    if not compiled.success:
                        raise RuntimeError("; ".join(compiled.errors))
                    executable = build_riscv_tensor(driver.tensor_artifact, target / "build", tools)
                    execution.update(elf_sha256=executable.elf_sha256, compile_command=executable.compile_command,
                                     compile_seconds=executable.compile_seconds, workspace_bytes=executable.workspace_bytes,
                                     tool_versions=executable.tool_versions, stage="qemu")
                    actual = run_riscv_tensor(executable, case.feed, target / "run", timeout=timeout)
                    execution.update(qemu_process_wall_seconds=actual.elapsed_s, command=actual.command, stage="numeric")
                    np.save(target / "qemu.npy", actual.output)
                    execution["qemu_vs_ort"] = tensor_diff(actual.output, expected, ATOL)
                    execution["qemu_vs_numpy"] = tensor_diff(actual.output, case.expected, ATOL)
                    execution["passed"] = all(execution[k]["passed"] for k in ("qemu_vs_ort", "qemu_vs_numpy"))
                    execution["status"] = "success" if execution["passed"] else "numeric_failed"
                    execution["stage"] = "complete"
                    results[(case.name, "ir_"+level)] = actual_ir
                    results[(case.name, "qemu_"+level)] = actual.output
                except Exception as exc:
                    execution["passed"] = False
                    execution["status"] = getattr(exc, "status", "runtime_error")
                    execution["error"] = f"{type(exc).__name__}: {exc}"
                    if isinstance(exc, (RiscVTensorExecutionError, RiscVTensorTimeoutError)):
                        execution["qemu_process_wall_seconds"] = float(exc.elapsed_s)
                        execution["command"] = list(exc.command)
                    if isinstance(exc, RiscVTensorTimeoutError):
                        execution["timeout_seconds"] = float(exc.timeout_s)
                print(f"[{case.name}/{level}] {'PASS' if execution['passed'] else 'FAIL'}", flush=True)
            row["passed"] = all(x["passed"] for x in row["executions"]) and len(row["executions"]) == len(LEVELS)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
    for case in cases:
        if case.relation:
            base, prefix = case.relation
            for backend in ("ort", "ir_none", "ir_all", "qemu_none", "qemu_all"):
                item = {"case":case.name, "reference":base, "backend":backend, "prefix":prefix, "passed":False}
                report["invariants"].append(item)
                if (case.name,backend) in results and (base,backend) in results:
                    item["comparison"] = tensor_diff(results[(case.name,backend)][:,:,:prefix],
                                                     results[(base,backend)][:,:,:prefix], ATOL)
                    item["passed"] = item["comparison"]["passed"]
                else:
                    item["error"] = "Required execution missing"
    executions = [e for row in report["cases"] for e in row["executions"]]
    report["passed_executions"] = sum(e["passed"] for e in executions)
    durations = [e["qemu_process_wall_seconds"] for e in executions if "qemu_process_wall_seconds" in e]
    report["qemu_process_wall_seconds"] = {"total":sum(durations), "measured_count":len(durations)}
    report["passed"] = (len(executions) == report["planned_executions"] and
                        all(row["passed"] for row in report["cases"]) and
                        all(row["passed"] for row in report["invariants"]))
    report["stage"] = "complete"

def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", type=Path, required=True)
    cli.add_argument("--cc")
    cli.add_argument("--qemu")
    cli.add_argument("--timeout", type=float, default=180)
    args = cli.parse_args(argv)
    if args.timeout <= 0 or not np.isfinite(args.timeout):
        cli.error("--timeout must be finite and positive")
    out = new_output_dir(args.output_dir)
    started = time.perf_counter()
    report = {"gate":"unit:attention-backend", "scope":"W3 small combined Attention; not full Qwen3",
              "passed":False, "status":"FAIL", "stage":"source_evidence", "atol":ATOL, "rtol":0}
    try:
        report.update(source_evidence())
        run_probe(out, report, cc=args.cc, qemu=args.qemu, timeout=args.timeout)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report.update(status="PASS" if report["passed"] else "FAIL",
                  elapsed_seconds=time.perf_counter()-started, process_peak_rss_bytes=process_peak_rss_bytes())
    write_reports(out, report)
    return 0 if report["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
