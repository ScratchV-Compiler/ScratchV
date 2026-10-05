"""Gate a fixed six-layer random Qwen3 through PyTorch, ORT, and host IR.

Every ordinary/diagnostic graph and none/basic/all IR path must pass strict
max_abs < 1e-5, rtol=0. No pretrained quality, full 0.6B, or QEMU claim is made.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

# Set these before NumPy/torch/ORT imports; all numerical backends use one CPU thread.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probes.w2_qwen3_small.diagnostics import (
    build_diagnostic_model, compare_outputs, tensor_diff, unpack_trace,
)
from probes.w2_qwen3_small.run import (
    ATOL, arrays_sha256, attention_checks, environment, input_cases,
    position_comparison, split_positions,
)

LEVELS = ("none", "basic", "all")
GRAPH_KINDS = ("normal", "diagnostic")
CASE_IDENTITIES = (
    ("full_seed_0", 256), ("full_seed_42", 256), ("one_token", 1),
    ("short_17", 17), ("short_255", 255), ("changed_future", 256),
    ("changed_padding", 17),
)
INVARIANT_IDENTITIES = {
    (name, backend, graph)
    for name in ("causality", "padding_isolation")
    for backend, graph in (
        ("pytorch", "diagnostic"), ("ort", "diagnostic"), ("ort", "normal"),
        *((f"ir_{level}", graph) for level in LEVELS for graph in GRAPH_KINDS),
    )
}


def trace_schema(metadata):
    """Stable named-NPZ schema, distinct from the packed ONNX return schema."""
    checkpoints = []
    for name, entry in metadata.items():
        row = {"name": name, "shape": list(entry["shape"]), "dtype": entry["dtype"],
               "sequence_axis": entry["sequence_axis"], "checkpoint": name.split(".")[-1]}
        if name.startswith("layer_"):
            row["layer"] = int(name.split(".")[0].split("_")[1])
        checkpoints.append(row)
    return {"version": 1, "checkpoints": checkpoints}


def save_trace(out, relative, values, report, *, schema, case, backend, graph, level=None):
    path = out / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **values)
    artifact = {"path": relative, "schema": schema, "case": case, "backend": backend,
                "graph": graph, "optimization": level, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "arrays_sha256": arrays_sha256(values)}
    report["trace_artifacts"].append(artifact)
    return artifact


def require_valid(program, stage):
    from scratchv.analysis.ir_verifier import verify_ir
    passed, issues = verify_ir(program, stage=stage)
    if not passed:
        raise ValueError("; ".join(str(issue) for issue in issues))


def optimized_copy(program, level):
    from scratchv.pass_manager import create_optimization_pass_manager
    manager = create_optimization_pass_manager(level)
    manager.before_pass = lambda p, data: require_valid(data, f"before:{p.name}")
    manager.after_pass = lambda p, data: require_valid(data, f"after:{p.name}")
    result = manager.run_pipeline(copy.deepcopy(program))
    require_valid(result.data, f"after-pipeline:{level}")
    return result.data


def run_ir_variant(program, initializers, feed, *, kind, level, schema, names,
                   metadata, valid_length, reference, ort_reference, out, case_name, report):
    """One independently failing path; execution failures never hide sibling paths."""
    from scratchv.verification.ir_interpreter import IRInterpreter
    row = {"passed": False, "status": "not_started", "graph": kind, "optimization": level}
    started = time.perf_counter()
    try:
        result = IRInterpreter(program).run(feed, initializers=initializers, collect_memory_stats=True)
        row["execution_seconds"] = time.perf_counter() - started
        row["executed_steps"] = result.executed_steps
        row["memory_stats"] = result.memory_stats
        if kind == "diagnostic":
            try:
                values = unpack_trace(result.return_value, schema)
            except Exception:
                # Malformed packed output cannot honestly use the named schema.
                # Preserve its original bytes for debugging before failing.
                raw = out / f"traces/{case_name}/ir_{kind}_{level}_invalid_pack.npy"
                raw.parent.mkdir(parents=True, exist_ok=True)
                np.save(raw, result.return_value, allow_pickle=False)
                row["invalid_packed_trace"] = str(raw.relative_to(out)).replace("\\", "/")
                raise
        else:
            values = {"logits": result.return_value}
        row["trace"] = save_trace(
            out, f"traces/{case_name}/ir_{kind}_{level}.npz", values, report,
            schema="trace_schema.json" if kind == "diagnostic" else "logits_schema.json",
            case=case_name, backend="ir", graph=kind, level=level)
        expected_names = names if kind == "diagnostic" else ["logits"]
        pt = {name: reference[name] for name in expected_names}
        row["pytorch_comparison"] = position_comparison(values, pt, expected_names, metadata, valid_length)
        # Missing ORT is an explicit failed requirement, even if PyTorch agrees.
        row["ort_comparison"] = (
            position_comparison(values, ort_reference, expected_names, metadata, valid_length)
            if ort_reference is not None else {"passed": False, "error": "ORT reference unavailable"})
        row["passed"] = row["pytorch_comparison"]["passed"] and row["ort_comparison"]["passed"]
        row["status"] = "success" if row["passed"] else "numeric_failed"
    except Exception as exc:
        row.update(status="runtime_error", error=f"{type(exc).__name__}: {exc}")
    row["seconds"] = time.perf_counter() - started
    return row


def case_passed(case):
    """Do not allow one successful graph/optimization to mask another failure."""
    required = [case["capture_preserves_logits"], *case["attention_checks"],
                *[case["ort"][kind] for kind in GRAPH_KINDS],
                *[case["ir"][level][kind] for level in LEVELS for kind in GRAPH_KINDS]]
    return all(item.get("passed") is True for item in required)


def finalize_numeric_report(report):
    """Require the fixed experiment, not just success among rows that survived."""
    cases = report.get("cases", [])
    invariants = report.get("invariants", [])
    case_coverage = tuple((row.get("name"), row.get("valid_length")) for row in cases) == CASE_IDENTITIES
    backend_coverage = all(
        set(row.get("ort", {})) == set(GRAPH_KINDS)
        and set(row.get("ir", {})) == set(LEVELS)
        and all(set(row["ir"][level]) == set(GRAPH_KINDS) for level in LEVELS)
        for row in cases
    )
    invariant_coverage = (
        len(invariants) == len(INVARIANT_IDENTITIES)
        and {(row.get("name"), row.get("backend"), row.get("graph")) for row in invariants}
        == INVARIANT_IDENTITIES
    )
    report["coverage"] = {"cases": case_coverage, "backends": backend_coverage,
                          "invariants": invariant_coverage,
                          "expected_cases": len(CASE_IDENTITIES), "actual_cases": len(cases),
                          "expected_invariants": len(INVARIANT_IDENTITIES), "actual_invariants": len(invariants)}
    report["coverage_complete"] = case_coverage and backend_coverage and invariant_coverage
    report["passed"] = (
        report["coverage_complete"]
        and all(case.get("passed") is True for case in cases)
        and all(row.get("passed") is True for row in invariants)
    )


def isolation_checks(out, metadata):
    """Check valid prefixes at every checkpoint for each backend and IR variant."""
    groups = [("pytorch", "reference", "diagnostic"),
              ("ort", "ort_diagnostic", "diagnostic"), ("ort", "ort_normal", "normal")]
    groups += [(f"ir_{level}", f"ir_{kind}_{level}", kind) for level in LEVELS for kind in GRAPH_KINDS]
    checks = []
    for invariant, left, right, length in (
        ("causality", "full_seed_0", "changed_future", 64),
        ("padding_isolation", "short_17", "changed_padding", 17),
    ):
        for backend, filename, kind in groups:
            row = {"name": invariant, "backend": backend, "graph": kind, "passed": False}
            try:
                with np.load(out / f"traces/{left}/{filename}.npz", allow_pickle=False) as a, np.load(
                    out / f"traces/{right}/{filename}.npz", allow_pickle=False
                ) as b:
                    names = list(metadata) if kind == "diagnostic" else ["logits"]
                    before = {name: split_positions(a[name], metadata[name]["sequence_axis"], 0, length) for name in names}
                    after = {name: split_positions(b[name], metadata[name]["sequence_axis"], 0, length) for name in names}
                    row["comparison"] = compare_outputs(after, before, names, ATOL)
                    row["passed"] = row["comparison"]["passed"]
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            checks.append(row)
    return checks


def run_probe(out, report, model_seed=0):
    report["stage"] = "environment"
    report["environment"] = environment()
    report["environment"]["threads"] = 1
    import onnx
    import onnxruntime as ort
    import torch
    from probes.w2_qwen3_small.model import export_onnx
    from probes.w3_qwen3_medium.preset import PRESET_NAME, build_medium_model
    from probes.w3_common import source_evidence
    from scratchv.frontend.onnx_parser import ONNXParser

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    report.update(stage="model", preset=PRESET_NAME, model_seed=model_seed,
                  provenance=source_evidence(), trace_artifacts=[], timings={})
    started = time.perf_counter()
    wrapper = build_medium_model(seed=model_seed)
    report["timings"]["model_build_seconds"] = time.perf_counter() - started
    names = list(wrapper.output_names)
    metadata = wrapper.checkpoint_metadata
    report["config"] = wrapper.config_metadata
    report["checkpoint_count"] = len(names)
    report["checkpoint_count_formula"] = "3 + 13 * num_hidden_layers"
    report["parameter_count"] = sum(t.numel() for t in wrapper.parameters())
    report["weights_sha256"] = arrays_sha256({
        name: tensor.detach().cpu().numpy() for name, tensor in wrapper.model.state_dict().items()})
    report["checkpoint_metadata"] = metadata
    (out / "config.json").write_text(wrapper.model.config.to_json_string(use_diff=False), encoding="utf-8")
    for filename, value in (("trace_schema.json", trace_schema(metadata)),
                            ("logits_schema.json", trace_schema({"logits": metadata["logits"]})),
                            ("checkpoint_metadata.json", metadata)):
        (out / filename).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    cases = input_cases()

    def reference(feed):
        args = [torch.from_numpy(feed[name]) for name in ("input_ids", "attention_mask")]
        with torch.inference_mode(), wrapper.capture():
            tensors = wrapper(*args)
        return {name: tensor.detach().cpu().numpy().copy() for name, tensor in zip(names, tensors)}

    started = time.perf_counter()
    first_reference = reference(cases[0][2])
    first_reference_seconds = time.perf_counter() - started
    report["stage"] = "export"
    started = time.perf_counter()
    tensors = [torch.from_numpy(cases[0][2][name]) for name in ("input_ids", "attention_mask")]
    report["export"] = export_onnx(wrapper, out / "checkpoints.onnx", *tensors)
    exported = onnx.load(out / "checkpoints.onnx")
    onnx.checker.check_model(exported, full_check=True)
    ordinary = copy.deepcopy(exported)
    logits_info = copy.deepcopy(next(value for value in ordinary.graph.output if value.name == "logits"))
    del ordinary.graph.output[:]
    ordinary.graph.output.append(logits_info)
    diagnostic, schema = build_diagnostic_model(exported, first_reference)
    report["checkpoints"] = schema
    (out / "checkpoints.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    report["onnx"] = {}
    for kind, model, filename in (("normal", ordinary, "model.onnx"),
                                  ("diagnostic", diagnostic, "diagnostics.onnx")):
        onnx.checker.check_model(model, full_check=True)
        onnx.save(model, out / filename)
        report["onnx"][kind] = {"path": filename, "nodes": len(model.graph.node),
            "operators": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
            "sha256": hashlib.sha256((out / filename).read_bytes()).hexdigest()}
    report["timings"]["export_seconds"] = time.perf_counter() - started
    report["stage"] = "prepare_backends"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    report["ort_graph_optimization"] = "ORT_DISABLE_ALL"
    sessions, parsers, programs = {}, {}, {}
    report["backends"] = {}
    for kind, filename in (("normal", "model.onnx"), ("diagnostic", "diagnostics.onnx")):
        info = report["backends"][kind] = {"ort": {"passed": False}, "ir": {}}
        started = time.perf_counter()
        try:
            sessions[kind] = ort.InferenceSession(str(out / filename), options, providers=["CPUExecutionProvider"])
            info["ort"]["passed"] = True
        except Exception as exc:
            info["ort"]["error"] = f"{type(exc).__name__}: {exc}"
        info["ort"]["prepare_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        try:
            parsers[kind] = ONNXParser()
            original = parsers[kind].parse(str(out / filename))
            require_valid(original, f"parsed:{kind}")
            info["parse_seconds"] = time.perf_counter() - started
        except Exception as exc:
            info["parse_error"] = f"{type(exc).__name__}: {exc}"
            original = None
        for level in LEVELS:
            started = time.perf_counter()
            row = info["ir"][level] = {"passed": False}
            try:
                if original is None:
                    raise RuntimeError(info["parse_error"])
                programs[kind, level] = optimized_copy(original, level)
                row["passed"] = True
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["prepare_seconds"] = time.perf_counter() - started

    report["stage"] = "numeric"
    report["cases"] = []
    for case_name, valid_length, feed in cases:
        report["current_case"] = case_name
        case_started = time.perf_counter()
        case = {"name": case_name, "valid_length": valid_length, "passed": False,
                "input_sha256": arrays_sha256(feed), "ort": {}, "ir": {}}
        report["cases"].append(case)
        np.savez(out / f"inputs_{case_name}.npz", **feed)
        started = time.perf_counter()
        pt = first_reference if case_name == cases[0][0] else reference(feed)
        case["pytorch_reference_seconds"] = (first_reference_seconds if case_name == cases[0][0]
                                              else time.perf_counter() - started)
        case["reference_trace"] = save_trace(out, f"traces/{case_name}/reference.npz", pt, report,
            schema="trace_schema.json", case=case_name, backend="pytorch", graph="diagnostic")
        with torch.inference_mode():
            plain = wrapper.model(input_ids=torch.from_numpy(feed["input_ids"]),
                attention_mask=torch.from_numpy(feed["attention_mask"]), position_ids=wrapper.positions,
                use_cache=False, return_dict=False, logits_to_keep=0)[0].numpy()
        case["capture_preserves_logits"] = tensor_diff(pt["logits"], plain, ATOL)
        case["attention_checks"] = attention_checks(pt, feed, wrapper.config_metadata)
        ort_values = {}
        for kind in GRAPH_KINDS:
            row = case["ort"][kind] = {"passed": False}
            started = time.perf_counter()
            try:
                if kind not in sessions:
                    raise RuntimeError(report["backends"][kind]["ort"]["error"])
                values = sessions[kind].run(None, feed)
                row["execution_seconds"] = time.perf_counter() - started
                if kind == "diagnostic":
                    actual_names = [output.name for output in sessions[kind].get_outputs()][1:]
                    if actual_names != names or len(values) != len(names) + 1:
                        raise ValueError("Diagnostic ONNX output order/count disagrees with schema")
                    named = dict(zip(names, values[1:]))
                    row["pack_layout"] = compare_outputs(unpack_trace(values[0], schema), named, names, ATOL)
                else:
                    named = {"logits": values[0]}
                ort_values[kind] = named
                row["trace"] = save_trace(out, f"traces/{case_name}/ort_{kind}.npz", named, report,
                    schema="trace_schema.json" if kind == "diagnostic" else "logits_schema.json",
                    case=case_name, backend="ort", graph=kind)
                ordered = names if kind == "diagnostic" else ["logits"]
                row["pytorch_comparison"] = position_comparison(named, {n: pt[n] for n in ordered}, ordered, metadata, valid_length)
                row["passed"] = row["pytorch_comparison"]["passed"] and row.get("pack_layout", {"passed": True})["passed"]
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["seconds"] = time.perf_counter() - started
        for level in LEVELS:
            case["ir"][level] = {}
            for kind in GRAPH_KINDS:
                if (kind, level) not in programs:
                    case["ir"][level][kind] = {"passed": False, "status": "preparation_failed",
                        "error": report["backends"][kind]["ir"][level]["error"]}
                    continue
                case["ir"][level][kind] = run_ir_variant(programs[kind, level], parsers[kind].initializers, feed,
                    kind=kind, level=level, schema=schema, names=names, metadata=metadata,
                    valid_length=valid_length, reference=pt, ort_reference=ort_values.get(kind),
                    out=out, case_name=case_name, report=report)
        case["passed"] = case_passed(case)
        case["seconds"] = time.perf_counter() - case_started
        print(f"[{case_name}] {'PASS' if case['passed'] else 'FAIL'} ({case['seconds']:.2f}s)", flush=True)
    report["invariants"] = isolation_checks(out, metadata)
    finalize_numeric_report(report)
    report["stage"] = "complete"
    report.pop("current_case", None)


def main(argv=None):
    # The pinned exporter prints Unicode status markers even on redirected
    # Windows GBK streams. Escaping unencodable log characters preserves the
    # actual export operation and the caller's chosen text encoding.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", type=Path, default=ROOT / "output/w3-medium")
    cli.add_argument("--model-seed", type=int, default=0)
    args = cli.parse_args(argv)
    from probes.w3_common import new_output_dir
    try:
        out = new_output_dir(args.output_dir)
    except FileExistsError:
        cli.error("--output-dir must not exist; use a new directory to preserve evidence")
    report = {"title": "Six-layer random Qwen3 medium host gate", "gate": "w3-medium-host", "passed": False,
              "stage": "initialization", "atol": ATOL, "rtol": 0,
              "acceptance": "strict max_abs < 1e-5 at all positions including padding queries",
              "scope": "official random-weight Qwen3; host PyTorch/ORT/IR only; no QEMU or pretrained quality claim"}
    started = time.perf_counter()
    try:
        run_probe(out, report, args.model_seed)
    except Exception as exc:
        report.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    report["seconds"] = time.perf_counter() - started
    report["status"] = "PASS" if report["passed"] else "FAIL"
    from probes.w3_common import process_peak_rss_bytes, write_reports
    report["process_peak_rss_bytes"] = process_peak_rss_bytes()
    report["process_peak_rss_scope"] = "whole process high-water mark, includes PyTorch exporter, ORT sessions and IR; not IR storage statistics"
    write_reports(out, report)
    print(f"[gate] {'PASS' if report['passed'] else 'FAIL'}; report: {out / 'report.json'}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
