"""Full pinned Qwen3 IR/ORT comparison with isolated, resource-bounded workers."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from probes.w1_qwen3_export.run import verify_files
from probes.w3_common import new_output_dir, sha256_file, source_evidence, write_reports
from probes.w3_layer_diff.run import load_schema, load_arrays
from probes.w3_qwen3_full.cases import CASE_NAMES, input_cases
from probes.w3_qwen3_full.comparison import ATOL, compare_tensor, compare_positions
from probes.w3_qwen3_full.resources import spawn_owned, wait_bounded

CHECKPOINTS = ("embedding", *(f"layer_{i}.output" for i in range(28)), "final_norm")
LOGIT_SHAPE = (1, 256, 151936)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_artifact(folder, descriptor, expected_name):
    require(isinstance(descriptor, dict), f"Missing artifact {expected_name}")
    require(descriptor.get("path") == expected_name, f"Unexpected artifact path: {expected_name}")
    path = folder / expected_name
    require(path.is_file() and path.resolve().parent == folder.resolve(), f"Missing/external artifact: {expected_name}")
    require(path.stat().st_size == descriptor.get("bytes") and sha256_file(path) == descriptor.get("sha256"),
            f"Altered artifact: {expected_name}")
    return path


def load_logits(path):
    value = np.load(path, mmap_mode="r", allow_pickle=False, max_header_size=10000)
    require(value.shape == LOGIT_SHAPE and value.dtype == np.float32, "Wrong full logits shape/dtype")
    return value


def saved_arithmetic_profile(mode, recorded):
    """Validate saved arithmetic without guessing it from the auditing host CPU."""
    require(isinstance(recorded, dict), "Missing saved FP32 arithmetic profile")
    if mode == "reference":
        from scratchv.verification.fp32_reference import profile
        strategy = recorded.get("cpu_strategy")
        require(strategy in ("avx2-fma3", "avx512"), "Unsupported saved FP32 CPU profile")
        canonical = profile(cpu_strategy=strategy)
    else:
        require(mode == "native", "Unsupported saved FP32 arithmetic profile")
        canonical = {"name": "numpy-native"}
    require(recorded == canonical and all(type(recorded[key]) is type(value)
                for key, value in canonical.items()), "Noncanonical saved FP32 arithmetic profile")
    return canonical


def validate_worker(folder, backend, input_path, sources, assets, *, expected_fp32_mode=None,
                    expected_fp32_profile=None):
    require(expected_fp32_profile is None or expected_fp32_mode is not None,
            "Saved FP32 profile requires an explicit mode")
    for name in ("report.json", "report.md", "report.html"):
        require((folder/name).is_file(), f"Missing worker {name}")
    report = json.loads((folder/"report.json").read_text(encoding="utf-8"))
    require(report.get("gate") == "numeric:w3-full-worker" and report.get("schema_version") == 1,
            "Wrong worker identity")
    require(report.get("backend") == backend and report.get("optimization_level") == "none", "Wrong backend")
    require(report.get("passed") is True and report.get("status") == "PASS", "Worker not PASS")
    require(report.get("full_model_executed") is True and report.get("full_ir_executed") is (backend == "ir"),
            "Full model was not actually executed")
    require(report.get("w3_exit_accepted") is False, "A worker cannot accept the team milestone")
    require(report.get("source_sha256") == sources, "Worker production sources differ")
    require(report.get("files") == assets, "Worker assets differ from fixed model")
    require(report.get("input", {}).get("sha256") == sha256_file(input_path), "Worker input differs")
    require(type(report.get("process_peak_rss_bytes")) is int and report["process_peak_rss_bytes"] > 0,
            "Missing measured peak RSS")
    timings = report.get("stages_seconds")
    require(isinstance(timings, dict), "Missing worker execution timings")
    for mode in ("ordinary", "diagnostic"):
        name = f"{backend}_{mode}_execute"
        duration = timings.get(name)
        require(type(duration) in (int, float) and math.isfinite(duration) and duration > 0,
                f"Missing/invalid worker execution timing: {name}")
    if backend == "ort":
        runtime = report.get("ort")
        expected_runtime = {"provider": "CPUExecutionProvider", "graph_optimization": "disabled",
                            "intra_op_threads": 1, "inter_op_threads": 1, "cpu_mem_arena": False}
        require(isinstance(runtime, dict) and all(type(runtime.get(key)) is type(value)
                and runtime.get(key) == value for key, value in expected_runtime.items()),
                "Worker ORT execution profile differs")
    comparison = report.get("ordinary_diagnostic", {})
    require(comparison.get("passed") is True and comparison.get("comparison") == "exact"
            and comparison.get("max_abs") == 0.0, "Diagnostic execution changed logits")
    names = {"logits": "logits.npy", "diagnostic_logits": "diagnostic_logits.npy",
             "checkpoints": "checkpoints.npz", "checkpoint_schema": "checkpoint_schema.json"}
    artifacts = report.get("artifacts", {})
    for key, filename in names.items():
        checked_artifact(folder, artifacts.get(key), filename)
    schema = load_schema(folder/"checkpoint_schema.json")
    require(tuple(x["name"] for x in schema) == CHECKPOINTS, "Incomplete/out-of-order full checkpoints")
    require(all(x["shape"] == [1, 256, 1024] and x["dtype"] == "float32" and x.get("sequence_axis") == 1
                for x in schema), "Wrong full checkpoint contract")
    require(report.get("checkpoints") == schema, "Worker report/schema disagree")
    ordinary, diagnostic = load_logits(folder/"logits.npy"), load_logits(folder/"diagnostic_logits.npy")
    exact = compare_tensor(ordinary, diagnostic)
    require(exact["passed"] and exact["max_abs"] == 0, "Ordinary/diagnostic arrays actually differ")
    del ordinary, diagnostic
    if backend == "ir":
        ir = report.get("ir")
        require(isinstance(ir, dict), "Missing IR execution profile")
        instruction_count = ir.get("instruction_count")
        require(type(instruction_count) is int and instruction_count > 0,
                "Missing/invalid IR instruction count")
        if expected_fp32_mode is not None:
            from scratchv.verification.fp32_reference import profile
            require(report.get("fp32_mode") == expected_fp32_mode,
                    "Worker FP32 execution mode differs")
            # Live runs require the selected local strategy. Saved-array audits
            # can explicitly supply a known producer strategy from another CPU;
            # the complete canonical profile still has to match.
            expected_profile = (saved_arithmetic_profile(expected_fp32_mode, expected_fp32_profile)
                                if expected_fp32_profile is not None else
                                profile() if expected_fp32_mode == "reference" else {"name": "numpy-native"})
            require(ir.get("fp32_mode") == expected_fp32_mode
                    and ir.get("fp32_profile") == expected_profile,
                    "Worker FP32 arithmetic profile differs")
        for mode in ("ordinary", "diagnostic"):
            execution = report.get("executions", {}).get(mode, {})
            # The pinned full graph is one straight-line block, including RETURN;
            # every declared instruction must execute once in both runs.
            require(type(execution.get("executed_steps")) is int
                    and execution["executed_steps"] == instruction_count,
                    "IR execution steps do not cover the declared instruction count")
            memory = execution.get("memory_stats", {})
            require(type(memory.get("peak_numpy_storage_bytes")) is int and memory["peak_numpy_storage_bytes"] > 0,
                    "Missing IR storage measurement")
    return report


def compare_case(folder, valid_length):
    rows = []
    for filename in ("logits.npy", "diagnostic_logits.npy"):
        actual, reference = load_logits(folder/"ir"/filename), load_logits(folder/"ort"/filename)
        rows.append({"name": filename, **compare_positions(actual, reference, valid_length)})
        del actual, reference
    schema = load_schema(folder/"ir/checkpoint_schema.json")
    require(schema == load_schema(folder/"ort/checkpoint_schema.json"), "IR/ORT checkpoint schema differs")
    actual = load_arrays(folder/"ir/checkpoints.npz", schema, max_bytes=64*1024**2)
    reference = load_arrays(folder/"ort/checkpoints.npz", schema, max_bytes=64*1024**2)
    checkpoints = [{"name": entry["name"], **compare_positions(actual[entry["name"]], reference[entry["name"]], valid_length)}
                   for entry in schema]
    all_rows = checkpoints + rows
    worst = max((r for r in all_rows if r["max_abs"] is not None), key=lambda x: x["max_abs"], default=None)
    return {"passed": all(r["passed"] for r in rows), "logits": rows, "checkpoints": checkpoints,
            "max_logits_abs": max(r["max_abs"] for r in rows),
            "checkpoint_threshold_scope": "Diagnostic location only: large intermediate activations can exceed 1e-4 in one FP32 ULP; acceptance applies to complete output logits",
            "first_divergence": next((r for r in all_rows if not r["passed"]), None),
            "worst": worst, "atol": ATOL, "rtol": 0}


def invariants(out, successful):
    rows = []
    for name, reference, prefix in (("changed_future", "full_seed_0", 64), ("changed_padding", "short_17", 17)):
        if name not in successful or reference not in successful:
            continue
        for backend in ("ort", "ir"):
            a = load_logits(out/name/backend/"logits.npy")
            b = load_logits(out/reference/backend/"logits.npy")
            rows.append({"case": name, "reference": reference, "prefix": prefix, "backend": backend,
                         **compare_tensor(a[:, :prefix], b[:, :prefix])})
            del a, b
    return rows


def run_worker(args, folder, backend, input_path, row=None):
    command = [str(Path(args.python).resolve()), "-B", "-X", "utf8",
               "probes/w3_qwen3_full/worker.py", "--backend", backend,
               "--model-dir", str(args.model_dir.resolve()), "--input-file", str(input_path),
               "--output-dir", str(folder/backend), "--optimization-level", "none",
               "--fp32-mode", getattr(args, "fp32_mode", "reference")]
    row = {} if row is None else row
    row.update(backend=backend, passed=False, command=command)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    with (folder/(backend+".log")).open("w", encoding="utf-8") as stream:
        process = spawn_owned(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, **options)
        try:
            row["returncode"] = wait_bounded(process, timeout=args.worker_timeout,
                                              max_memory_bytes=int(args.max_worker_memory_gib*1024**3), row=row)
            require(row["returncode"] == 0, f"{backend} worker exited {row['returncode']}")
            row["passed"] = True
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            for filename in ("report.json", "progress.json"):
                try:
                    child = json.loads((folder/backend/filename).read_text(encoding="utf-8"))
                    row["child_stage"] = child.get("stage")
                    row["child_error"] = child.get("error")
                    break
                except (OSError, ValueError):
                    pass
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--worker-timeout", type=float, default=1800)
    parser.add_argument("--max-worker-memory-gib", type=float, default=10)
    parser.add_argument("--case", action="append", choices=CASE_NAMES)
    parser.add_argument("--fp32-mode", choices=("native", "reference"), default="reference",
                        help="Explicit IR FP32 arithmetic policy; native reproduces the original NumPy baseline")
    args = parser.parse_args(argv)
    if any(not math.isfinite(v) or v <= 0 for v in (args.worker_timeout, args.max_worker_memory_gib)):
        parser.error("Resource limits must be finite and positive")
    if args.case and len(args.case) != len(set(args.case)):
        parser.error("Duplicate case selection")
    out = new_output_dir(args.output_dir)
    selected = set(args.case or CASE_NAMES)
    start = time.perf_counter()
    report = {"gate": "numeric:ir-full-qwen3", "status": "FAIL", "passed": False,
              "optimization_level": "none", "atol": ATOL, "rtol": 0,
              "fp32_mode": args.fp32_mode,
              "full_ir_executed": False, "w3_exit_accepted": False,
              "scope": "Pinned 28-layer FP32 L256 IR vs ORT; strict 1e-4 on all ordinary/diagnostic logits including padding; 30 checkpoints localize errors",
              "cases": [], "invariants": [], "required_cases": list(CASE_NAMES),
              "selected_cases": [x for x in CASE_NAMES if x in selected], "stage": "sources"}
    try:
        report.update(source_evidence())
        report["stage"] = "assets"
        assets = verify_files(args.model_dir)
        report["files"] = assets
        successful = set()
        for name, valid, feed in input_cases():
            if name not in selected:
                continue
            report["stage"] = name
            folder = out/name
            folder.mkdir()
            input_path = folder/"inputs.npz"
            np.savez(input_path, **feed)
            row = {"name": name, "valid_length": valid, "passed": False, "workers": [],
                   "input_sha256": sha256_file(input_path)}
            report["cases"].append(row)
            for backend in ("ort", "ir"):
                worker = {"backend": backend, "passed": False}
                row["workers"].append(worker)
                run_worker(args, folder, backend, input_path, worker)
                if worker["passed"]:
                    try:
                        evidence = validate_worker(folder/backend, backend, input_path,
                                                   report["source_sha256"], assets,
                                                   expected_fp32_mode=args.fp32_mode)
                        require(evidence["input"]["valid_length"] == valid, "Worker valid length differs")
                        worker.update(report_sha256=sha256_file(folder/backend/"report.json"),
                                      process_peak_rss_bytes=evidence["process_peak_rss_bytes"],
                                      stages_seconds=evidence["stages_seconds"])
                        if backend == "ir":
                            report["full_ir_executed"] = True
                    except Exception as exc:
                        worker.update(passed=False, error=f"{type(exc).__name__}: {exc}")
                print(f"[{name}/{backend}] {'EXECUTED' if worker['passed'] else 'FAIL'}", flush=True)
            if all(w["passed"] for w in row["workers"]):
                row["comparison"] = compare_case(folder, valid)
                row["passed"] = row["comparison"]["passed"]
                successful.add(name)
                print(f"[{name}/compare] {'PASS' if row['passed'] else 'FAIL'}; logits_max_abs={row['comparison']['max_logits_abs']}; diagnostic_max_abs={row['comparison']['worst']['max_abs']}", flush=True)
            write_reports(out, report)
        report["invariants"] = invariants(out, successful)
        require(source_evidence()["source_sha256"] == report["source_sha256"], "Sources changed during full execution")
        coverage = set(selected) == set(CASE_NAMES) and len(report["cases"]) == len(CASE_NAMES) and len(report["invariants"]) == 4
        numeric = len(report["cases"]) == len(selected) and all(row["passed"] for row in report["cases"]) and all(x["passed"] for x in report["invariants"])
        report.update(coverage_complete=coverage, selected_numerical_passed=numeric,
                      passed=coverage and numeric, status="PASS" if coverage and numeric else ("PARTIAL" if numeric else "FAIL"), stage="complete")
    except KeyboardInterrupt:
        report.update(status="FAIL", passed=False, interrupted=True, error="KeyboardInterrupt: full verification cancelled")
    except Exception as exc:
        report.update(status="FAIL", passed=False, error=f"{type(exc).__name__}: {exc}")
    report["elapsed_seconds"] = time.perf_counter() - start
    write_reports(out, report)
    return 130 if report.get("interrupted") else (0 if report["passed"] else (2 if report["status"] == "PARTIAL" else 1))


if __name__ == "__main__":
    raise SystemExit(main())
