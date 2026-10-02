"""Compile both real-Qwen graphs and compare actual RV64 QEMU outputs with ORT.

First export with probes/w2_qwen3_small/run.py. This gate never substitutes host
execution for QEMU, skips missing tools, or accepts logits in place of the trace.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from html import escape
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort

from probes.w2_qwen3_small.diagnostics import compare_outputs, tensor_diff, unpack_trace
from probes.w2_qwen3_small.run import ATOL, arrays_sha256, input_cases
from scratchv.analysis.ir_verifier import verify_ir
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.pass_manager import create_optimization_pass_manager
from scratchv.runtime.riscv_tensor import build_riscv_tensor, discover_toolchain, run_riscv_tensor
from scratchv.verification.ir_interpreter import IRInterpreter


def _require_valid(program, stage):
    passed, issues = verify_ir(program, stage=stage)
    if not passed:
        raise ValueError("; ".join(str(issue) for issue in issues))


def _optimize(program, level):
    manager = create_optimization_pass_manager(level)
    manager.before_pass = lambda p, data: _require_valid(data, f"before:{p.name}")
    manager.after_pass = lambda p, data: _require_valid(data, f"after:{p.name}")
    return manager.run_pipeline(copy.deepcopy(program)).data


def run_probe(model_dir, out, report, *, cc=None, qemu=None, timeout=180):
    report["stage"] = "toolchain"
    tools = discover_toolchain(cc=cc, qemu=qemu)
    report["toolchain"] = {"cc": tools.cc, "qemu": tools.qemu}
    files = {"normal": model_dir / "model.onnx", "diagnostic": model_dir / "diagnostics.onnx"}
    schema_path = model_dir / "checkpoints.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    names = [entry["name"] for entry in schema]
    if len(names) != 29 or names[-1] != "logits":
        raise ValueError("Expected the real two-layer Qwen3 schema with 29 checkpoints")
    report["checkpoints"] = schema
    report["model_sha256"] = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in files.items()}
    report["schema_sha256"] = hashlib.sha256(schema_path.read_bytes()).hexdigest()
    sources = [ROOT / "scratchv/compiler.py", Path(__file__),
               *sorted((ROOT / "scratchv/optimizer").glob("*.py")),
               ROOT / "scratchv/backend/tensor_c_codegen.py",
               ROOT / "scratchv/runtime/riscv_tensor.py"]
    report["source_sha256"] = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sources}
    report["stage"] = "compile"
    report["builds"] = []
    sessions, programs, parsers, executables = {}, {}, {}, {}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    for kind, path in files.items():
        sessions[kind] = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        parser = ONNXParser()
        original = parser.parse(str(path))
        _require_valid(original, "after-parse")
        parsers[kind] = parser
        programs[kind] = {level: _optimize(original, level) for level in ("none", "basic", "all")}
        for level in ("none", "all"):
            directory = out / "build" / f"{kind}-{level}"
            directory.mkdir(parents=True)
            driver = CompilerDriver(CompilerConfig(backend="tensor-c", optimize_level=level, verify_ir=True))
            result = driver.compile(str(path), str(directory / "model.c"))
            if not result.success:
                raise RuntimeError("; ".join(result.errors))
            artifact = driver.tensor_artifact
            executable = build_riscv_tensor(artifact, directory / "rv64", tools)
            executables[kind, level] = executable
            report["builds"].append({"graph": kind, "optimization": level,
                                     "workspace_bytes": artifact.workspace_bytes,
                                     "constant_bytes": artifact.constant_bytes,
                                     "elf_sha256": executable.elf_sha256,
                                     "compile_command": executable.compile_command,
                                     "tool_versions": executable.tool_versions,
                                     "compile_seconds": executable.compile_seconds,
                                     "optimization_stats": result.stats["optimization"]})
            print(f"[build {kind}/{level}] RV64 ELF ready; workspace={artifact.workspace_bytes}", flush=True)

    report["stage"] = "numeric"
    report["cases"] = []
    cached = {}
    for case_name, valid_length, feed in input_cases():
        report["current_case"] = case_name
        case = {"name": case_name, "valid_length": valid_length,
                "input_sha256": arrays_sha256(feed), "passed": False,
                "optimized_ir": [], "qemu": []}
        report["cases"].append(case)
        ordinary_ref = sessions["normal"].run(None, feed)[0]
        diagnostic_values = sessions["diagnostic"].run(None, feed)
        diagnostic_ref = diagnostic_values[0]
        named_ref = unpack_trace(diagnostic_ref, schema)
        # Check the supplied packed layout against independent ORT named outputs.
        ort_names = [value.name for value in sessions["diagnostic"].get_outputs()][1:]
        layout = compare_outputs(named_ref, dict(zip(ort_names, diagnostic_values[1:])), names, ATOL)
        case["ort_pack_layout"] = layout
        case["ort_normal_vs_trace"] = tensor_diff(ordinary_ref, named_ref["logits"], ATOL)
        cached[case_name] = {}
        for kind, expected in (("normal", ordinary_ref), ("diagnostic", diagnostic_ref)):
            for level in ("none", "basic", "all"):
                actual = IRInterpreter(programs[kind][level]).run(
                    feed, initializers=parsers[kind].initializers).return_value
                comparison = tensor_diff(actual, expected, ATOL)
                case["optimized_ir"].append({"graph": kind, "optimization": level, **comparison})
            for level in ("none", "all"):
                folder = out / "runs" / case_name / f"{kind}-{level}"
                execution = run_riscv_tensor(executables[kind, level], feed, folder, timeout=timeout)
                actual = execution.output
                np.save(folder / "qemu.npy", actual)
                np.save(folder / "ort.npy", expected)
                comparison = tensor_diff(actual, expected, ATOL)
                row = {"graph": kind, "optimization": level, "seconds": execution.elapsed_s,
                       "command": execution.command, **comparison}
                if kind == "diagnostic":
                    unpacked = unpack_trace(actual, schema)
                    row["checkpoints"] = compare_outputs(unpacked, named_ref, names, ATOL)
                    row["passed"] &= row["checkpoints"]["passed"]
                    logits = unpacked["logits"]
                else:
                    logits = actual
                cached[case_name][kind, level] = logits
                case["qemu"].append(row)
                print(f"[{case_name}/{kind}/{level}] {'PASS' if row['passed'] else 'FAIL'} "
                      f"max_abs={row.get('max_abs')} "
                      f"first={row.get('checkpoints', {}).get('first_divergence')}", flush=True)
        case["normal_vs_trace"] = [
            {"optimization": level, **tensor_diff(cached[case_name]["normal", level],
                                                   cached[case_name]["diagnostic", level], ATOL)}
            for level in ("none", "all")]
        case["passed"] = all(check["passed"] for check in (
            layout, case["ort_normal_vs_trace"], *case["optimized_ir"],
            *case["qemu"], *case["normal_vs_trace"]))
    report["invariants"] = []
    for name, left, right, length in (
        ("causality", "full_seed_0", "changed_future", 64),
        ("padding_isolation", "short_17", "changed_padding", 17),
    ):
        for kind in files:
            for level in ("none", "all"):
                report["invariants"].append({"name": name, "graph": kind, "optimization": level,
                    **tensor_diff(cached[left][kind, level][:, :length],
                                  cached[right][kind, level][:, :length], ATOL)})
    report["passed"] = all(case["passed"] for case in report["cases"]) and all(
        check["passed"] for check in report["invariants"])
    report["stage"] = "complete"


def write_reports(out, report):
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Two-layer Qwen3: ScratchV IR → C tensor kernels → RV64GC → QEMU", "",
             f"Result: **{'PASS' if report['passed'] else 'FAIL'}**; stage: {report['stage']}", "",
             "FP32, L=256, random weights, 29 checkpoints, strict max absolute error < 1e-5.",
             "The C cross-compiler is Zig/LLVM; this does not validate the legacy scalar assembly selector.",
             "IR optimization levels: none/basic/all. QEMU: none/all, normal and diagnostic graphs.", ""]
    if report.get("error"):
        lines += [f"Error: {report['error']}", ""]
    lines += ["| Case | Graph | IR optimization | Max abs | First divergent checkpoint | Result |",
              "|---|---|---|---:|---|---|"]
    for case in report.get("cases", []):
        for row in case.get("qemu", []):
            lines.append(f"| {case['name']} | {row['graph']} | {row['optimization']} | "
                         f"{row.get('max_abs')} | {row.get('checkpoints', {}).get('first_divergence') or 'none'} | "
                         f"{'PASS' if row['passed'] else 'FAIL'} |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    details = []
    for case in report.get("cases", []):
        details.append(f"<details><summary>{escape(case['name'])}: "
                       f"{'PASS' if case['passed'] else 'FAIL'}</summary><pre>"
                       f"{escape(json.dumps(case, indent=2))}</pre></details>")
    (out / "report.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>Qwen3 RV64 probe</title>"
        "<style>body{font:16px/1.5 system-ui;max-width:1100px;margin:32px auto}pre{white-space:pre-wrap}"
        "details{padding:12px;border:1px solid #ddd;margin:10px 0}</style>"
        f"<h1>Qwen3 RV64 QEMU: {'PASS' if report['passed'] else 'FAIL'}</h1>"
        f"<pre>{escape(chr(10).join(lines))}</pre>" + "".join(details) + "</html>", encoding="utf-8")


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model-dir", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path, default=ROOT / "output/qwen3-riscv")
    cli.add_argument("--cc")
    cli.add_argument("--qemu")
    cli.add_argument("--timeout", type=float, default=180)
    args = cli.parse_args(argv)
    out = args.output_dir.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        cli.error("--output-dir must be empty; preserve earlier evidence")
    out.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "stage": "initialization", "atol": ATOL, "rtol": 0}
    started = time.perf_counter()
    try:
        run_probe(args.model_dir.resolve(), out, report, cc=args.cc, qemu=args.qemu, timeout=args.timeout)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    report["seconds"] = time.perf_counter() - started
    write_reports(out, report)
    print(f"[RV64 gate] {'PASS' if report['passed'] else 'FAIL'}: {out / 'report.json'}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
