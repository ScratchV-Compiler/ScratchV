#!/usr/bin/env python3
"""Strict offline pretrained-subgraph gate; no full model is loaded or executed."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, is_dataclass
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

# Set before importing numerical runtimes. The process intentionally uses one thread.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import onnx
import onnxruntime as ort
import torch

from probes.w2_qwen3_small.diagnostics import build_diagnostic_model, unpack_trace
from probes.w3_common import new_output_dir, process_peak_rss_bytes, source_evidence, write_reports
from probes.w3_qwen3_subgraphs.assets import DIMENSIONS, Weights, sha256, verify_source
from probes.w3_qwen3_subgraphs.cases import iter_cases
from scratchv.analysis.ir_verifier import verify_ir
from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.pass_manager import create_optimization_pass_manager
from scratchv.verification.ir_interpreter import IRInterpreter

ATOL = 1e-4
LEVELS = ("none", "basic", "all")
CASE_NAMES = ("projection_q", "projection_k", "projection_v", "projection_o", "rmsnorm_hidden",
              "rmsnorm_q", "rope_q", "rmsnorm_k", "rope_k", "gqa_full", "gqa_padding",
              "gqa_changed_padding", "gqa_changed_future", "swiglu")
PINNED = {"numpy": "2.2.6", "torch": "2.7.1", "onnx": "1.18.0",
          "onnxruntime": "1.22.1", "safetensors": "0.5.3"}


def array_evidence(arrays):
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                   "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                   "finite": bool(np.isfinite(value).all()), "bytes": value.nbytes}
            for name, value in arrays.items()}


def tensor_diff(actual, expected, atol=ATOL):
    """Strict absolute gate with JSON-safe evidence for shape/nonfinite failures."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    row = {"shape": list(actual.shape), "expected_shape": list(expected.shape),
           "shape_matches": actual.shape == expected.shape, "dtype": str(actual.dtype),
           "expected_dtype": str(expected.dtype), "atol": atol, "rtol": 0,
           "finite": bool(np.isfinite(actual).all() and np.isfinite(expected).all()),
           "max_abs_error": None, "firstdiff": None, "passed": False}
    if not row["shape_matches"]:
        row["firstdiff"] = {"reason": "shape mismatch"}
        return row
    if actual.dtype != np.float32 or expected.dtype != np.float32:
        row["firstdiff"] = {"reason": "FP32 dtype mismatch"}
        return row
    with np.errstate(invalid="ignore", over="ignore"):
        error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    bad = ~np.isfinite(actual) | ~np.isfinite(expected) | (error >= atol)
    if row["finite"]:
        row["max_abs_error"] = float(error.max(initial=0))
    if bad.any():
        index = tuple(int(value) for value in np.argwhere(bad)[0])
        def safe_number(value):
            value = float(value)
            return value if np.isfinite(value) else str(value)
        row["firstdiff"] = {"index": list(index), "actual": safe_number(actual[index]),
                            "expected": safe_number(expected[index]), "abs_error": safe_number(error[index])}
    row["passed"] = bool(row["finite"] and not bad.any())
    return row


def compare(actual, expected):
    if list(actual) != list(expected):
        raise ValueError("Checkpoint names/order differ")
    rows = [{"name": name, **tensor_diff(actual[name], expected[name])} for name in expected]
    failed = next((row for row in rows if not row["passed"]), None)
    maxima = [row["max_abs_error"] for row in rows]
    return {"passed": all(row["passed"] for row in rows), "checkpoints": rows,
            "max_abs_error": None if any(value is None for value in maxima) else max(maxima, default=0),
            "first_divergence": None if failed is None else
                {"checkpoint": failed["name"], "firstdiff": failed["firstdiff"]}}


def require_valid(program):
    passed, issues = verify_ir(program)
    if not passed:
        raise ValueError("; ".join(map(str, issues)))


def execute_ir(model_path, feed, level):
    parser = ONNXParser()
    program = parser.parse(str(model_path))
    require_valid(program)
    manager = create_optimization_pass_manager(level)
    manager.before_pass = lambda unused, value: require_valid(value)
    manager.after_pass = lambda unused, value: require_valid(value)
    optimized = manager.run_pipeline(program)
    require_valid(optimized.data)
    result = IRInterpreter(optimized.data).run(feed, initializers=parser.initializers,
                                               collect_memory_stats=True)
    memory = result.memory_stats
    if is_dataclass(memory):
        memory = asdict(memory)
    return result.return_value, {"executed_steps": result.executed_steps,
        "optimization_changes": optimized.report.total_changes, "memory_stats": memory,
        "process_peak_rss_bytes": process_peak_rss_bytes()}


def attention_checks(values, feed):
    probabilities = values["probabilities"]
    blocked = feed["mask"][0, 0] < 0
    return {"blocked_probability": tensor_diff(probabilities[..., blocked],
                                                np.zeros_like(probabilities[..., blocked])),
            "probability_row_sum": tensor_diff(probabilities.sum(axis=-1),
                                                 np.ones(probabilities.shape[:-1], np.float32))}


def run_case(case, out, row):
    directory = out / case.name
    directory.mkdir()
    row.update(name=case.name, family=case.family, metadata=case.metadata,
               inputs=array_evidence(case.feed), reference=array_evidence(case.expected),
               passed=False, comparisons={}, executions={}, semantic_checks={})
    np.savez(directory / "inputs.npz", **case.feed)
    np.savez(directory / "torch.npz", **case.expected)
    model = copy.deepcopy(case.model)
    output = next(value for value in model.graph.output if value.name == "y")
    del model.graph.output[:]
    model.graph.output.append(output)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, directory / "model.onnx")
    diagnostic, schema = build_diagnostic_model(case.model, case.expected)
    onnx.save(diagnostic, directory / "diagnostics.onnx")
    (directory / "checkpoints.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    row["artifacts"] = {"directory": case.name, "model_sha256": sha256(directory / "model.onnx"),
                        "diagnostic_sha256": sha256(directory / "diagnostics.onnx")}
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(directory / "diagnostics.onnx"), options,
                                   providers=["CPUExecutionProvider"])
    values = session.run(None, case.feed)
    ort_values = {name: array for name, array in zip(case.expected, values[1:])}
    packed = unpack_trace(values[0], schema)
    row["comparisons"]["ort_pack_layout"] = compare(packed, ort_values)
    row["comparisons"]["torch_vs_ort"] = compare(ort_values, case.expected)
    np.savez(directory / "ort.npz", **ort_values)
    del session, values, packed
    session = ort.InferenceSession(str(directory / "model.onnx"), options,
                                   providers=["CPUExecutionProvider"])
    ordinary_ort = session.run(None, case.feed)[0]
    del session
    row["comparisons"]["ordinary_ort_vs_torch"] = tensor_diff(ordinary_ort, case.expected["y"])
    row["comparisons"]["ordinary_ort_vs_diagnostic"] = tensor_diff(ordinary_ort, ort_values["y"])
    np.save(directory / "ordinary_ort.npy", ordinary_ort)
    if case.family == "gqa":
        row["semantic_checks"]["torch"] = attention_checks(case.expected, case.feed)
        row["semantic_checks"]["ort"] = attention_checks(ort_values, case.feed)
    for level in LEVELS:
        try:
            packed, execution = execute_ir(directory / "diagnostics.onnx", case.feed, level)
            actual = unpack_trace(packed, schema)
            np.savez(directory / f"ir_{level}.npz", **actual)
            row["executions"][level] = {"diagnostic": execution}
            row["comparisons"][f"ir_{level}_vs_ort"] = compare(actual, ort_values)
            row["comparisons"][f"ir_{level}_vs_torch"] = compare(actual, case.expected)
            ordinary, execution = execute_ir(directory / "model.onnx", case.feed, level)
            row["executions"][level]["ordinary"] = execution
            np.save(directory / f"ordinary_ir_{level}.npy", ordinary)
            row["comparisons"][f"ordinary_ir_{level}_vs_ort"] = tensor_diff(ordinary, ordinary_ort)
            row["comparisons"][f"ordinary_ir_{level}_vs_torch"] = tensor_diff(ordinary, case.expected["y"])
            row["comparisons"][f"ordinary_ir_{level}_vs_diagnostic"] = tensor_diff(ordinary, actual["y"])
            if case.family == "gqa":
                row["semantic_checks"][f"ir_{level}"] = attention_checks(actual, case.feed)
            del packed, actual, ordinary
        except Exception as exc:
            row["executions"].setdefault(level, {})["error"] = f"{type(exc).__name__}: {exc}"
            row["comparisons"][f"ir_{level}_execution"] = {"passed": False, "error": traceback.format_exc()}
    row["passed"] = (all(value["passed"] for value in row["comparisons"].values())
                     and all(value["passed"] for checks in row["semantic_checks"].values() for value in checks.values()))
    row["first_divergence"] = next(({"comparison": name,
                                     "detail": value.get("first_divergence", value.get("firstdiff", value.get("error")))}
                                    for name, value in row["comparisons"].items() if not value["passed"]), None)


def invariants(out, length):
    rows = []
    for name, left, right, prefix in (("padding_isolation", "gqa_padding", "gqa_changed_padding", length),
                                     ("causality", "gqa_full", "gqa_changed_future", max(1, length // 4))):
        for backend in ("torch", "ort", "ir_none", "ir_basic", "ir_all"):
            try:
                with np.load(out / left / f"{backend}.npz") as a, np.load(out / right / f"{backend}.npz") as b:
                    result = compare({key: a[key][:, :, :prefix] for key in ("probabilities", "y")},
                                     {key: b[key][:, :, :prefix] for key in ("probabilities", "y")})
                rows.append({"name": name, "backend": backend, "query_prefix": prefix, **result})
            except Exception as exc:
                rows.append({"name": name, "backend": backend, "passed": False,
                             "error": f"{type(exc).__name__}: {exc}"})
    return rows


def finalize_report(report, length):
    report["coverage_complete"] = tuple(row["name"] for row in report["cases"]) == CASE_NAMES
    report["numerical_passed"] = bool(report["coverage_complete"] and all(row["passed"] for row in report["cases"])
                                      and len(report["invariants"]) == 10
                                      and all(row["passed"] for row in report["invariants"]))
    report["passed"] = report["numerical_passed"] and length == 256
    report["status"] = "PASS" if report["passed"] else ("PARTIAL" if report["numerical_passed"] else "FAIL")
    report["stage"] = "complete"
    report.pop("current_case", None)


def run_probe(source, out, report, length):
    report.update(source_evidence())
    report["stage"] = "environment"
    installed = {name: importlib.metadata.version(name).split("+")[0] for name in PINNED}
    if sys.version_info[:2] != (3, 12) or installed != PINNED:
        raise RuntimeError(f"Use pinned Python 3.12 qwen environment: {installed}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    report["environment"]["threads"] = {"torch": 1, "torch_interop": 1, "ort": 1,
        **{key: os.environ[key] for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    report["stage"] = "verify_source"
    report["source"] = verify_source(source)
    weights = Weights(source)
    report["weight_tensors"] = weights.evidence
    report["cases"] = []
    report["stage"] = "subgraphs"
    for case in iter_cases(weights, length):
        row = {"name": case.name, "passed": False}
        report["cases"].append(row)
        report["current_case"] = case.name
        started = time.perf_counter()
        try:
            run_case(case, out, row)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()
        row["seconds"] = time.perf_counter() - started
        row["process_peak_rss_bytes"] = process_peak_rss_bytes()
        print(f"[{case.name}] {'PASS' if row['passed'] else 'FAIL'} "
              f"first divergence={row.get('first_divergence', row.get('error'))}", flush=True)
        del case
        gc.collect()
    report["invariants"] = invariants(out, length)
    finalize_report(report, length)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Existing pinned HF snapshot; never downloaded")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory, refuses to overwrite")
    parser.add_argument("--seq-len", type=int, default=256, choices=range(2, 257), metavar="2..256",
                        help="Only 256 is an official gate; shorter debug runs can only be PARTIAL")
    args = parser.parse_args(argv)
    out = new_output_dir(args.output_dir)
    report = {"gate": "w3_qwen3_subgraphs", "status": "FAIL", "passed": False,
              "stage": "initializing", "scope": "official L256" if args.seq_len == 256 else "debug subset",
              "atol": ATOL, "rtol": 0, "comparison": "strict max_abs_error < 1e-4; shapes/dtypes/finite required",
              "sequence_length": args.seq_len, "dimensions": DIMENSIONS, "levels": list(LEVELS),
              "weight_origin": "Authenticated pretrained checkpoint, layer 0 only, BF16 values promoted to FP32",
              "input_origin": "NumPy PCG64 seed 20261004, synthetic N(0,1) hidden/context activations; Q/K/V cases derive from authentic projections",
              "limitations": ["Selected pretrained subgraphs, not a complete pretrained decoder-layer forward",
                              "No full-model generation or language-quality claim", "No RISC-V claim for these full dimensions",
                              "W3 1e-4 baseline does not modify W2's strict 1e-5 threshold"],
              "memory_scope": "process_peak_rss_bytes is the process lifetime peak, including Torch/ORT/IR; IR memory_stats is separately reported"}
    started = time.perf_counter()
    try:
        run_probe(args.source_dir, out, report, args.seq_len)
    except Exception as exc:
        report.update(passed=False, status="FAIL", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    report["seconds"] = time.perf_counter() - started
    report["process_peak_rss_bytes"] = process_peak_rss_bytes()
    write_reports(out, report)
    print(f"{report['status']}: {out / 'report.json'}", flush=True)
    return 0 if report["passed"] else (2 if report["status"] == "PARTIAL" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
