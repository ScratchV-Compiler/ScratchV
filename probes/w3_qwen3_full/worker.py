"""Execute one complete pinned Qwen3 backend in an isolated process.

A worker PASS proves execution and ordinary/diagnostic consistency only. The
parent compares independent IR and ORT results; no worker accepts W3 itself.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import onnx

from probes.w1_qwen3_export import run as assets
from probes.w2_qwen3_parse.validation import audit_graph_structure
from probes.w3_common import (atomic_text, new_output_dir, process_peak_rss_bytes,
                             sha256_file, source_evidence, write_reports)
from probes.w3_layer_diff.run import load_arrays

GATE = "numeric:w3-full-worker"
INPUT_SCHEMA = [
    {"name": "input_ids", "shape": [1, 256], "dtype": "int64"},
    {"name": "attention_mask", "shape": [1, 1, 256, 256], "dtype": "float32"},
]
LOGITS_SHAPE = (1, 256, 151936)


def load_inputs(path):
    """Validate the pinned export's finite additive, right-padded causal mask."""
    feed = load_arrays(Path(path), INPUT_SCHEMA, max_bytes=1024 * 1024)
    ids, mask = feed["input_ids"], feed["attention_mask"]
    if np.any(ids < 0) or np.any(ids >= 151936):
        raise ValueError("input_ids must be in the fixed Qwen3 vocabulary")
    length = int(np.count_nonzero(mask[0, 0, -1] == 0))
    if not 1 <= length <= 256:
        raise ValueError("attention_mask must expose at least one valid token")
    query, key = np.arange(256)[:, None], np.arange(256)[None, :]
    wanted = np.where((key <= query) & (key < length), np.float32(0),
                      np.finfo(np.float32).min).reshape(1, 1, 256, 256)
    if not np.array_equal(mask, wanted):
        raise ValueError("attention_mask is not the pinned causal/right-padding additive mask")
    return feed, length


def checkpoint_schema(model, audit=None):
    """Locate checkpoints from audited dataflow, never layer-name substrings."""
    audit = audit_graph_structure(model) if audit is None else audit
    layers = audit.get("layers", [])
    if (audit.get("passed") is not True or audit.get("layer_count") != 28
            or [row.get("layer") for row in layers] != list(range(28))):
        raise ValueError("Checkpoint coverage requires the complete ordered 28-layer audit")
    by_name = {}
    for node in model.graph.node:
        by_name.setdefault(node.name, []).append(node)

    def output(name):
        nodes = by_name.get(name, [])
        if len(nodes) != 1 or len(nodes[0].output) != 1:
            raise ValueError(f"Checkpoint node must be unique with one output: {name}")
        return nodes[0].output[0]

    rows = [{"name": "embedding", "onnx_name": output(audit["embedding"]),
             "checkpoint": "embedding"}]
    rows += [{"name": f"layer_{row['layer']}.output", "onnx_name": row["output"],
              "layer": row["layer"], "checkpoint": "output"} for row in layers]
    rows += [{"name": "final_norm", "onnx_name": output(audit["final_norm"]),
              "checkpoint": "final_norm"}]
    definitions = {name for node in model.graph.node for name in node.output}
    names = [row["onnx_name"] for row in rows]
    if len(names) != len(set(names)) or not set(names) <= definitions:
        raise ValueError("Checkpoint tensors must be unique graph-produced values")
    for row in rows:
        row.update(shape=[1, 256, 1024], dtype="float32", sequence_axis=1)
    return rows


def checked_tensor(array, shape, name):
    if not isinstance(array, np.ndarray) or array.shape != tuple(shape) or array.dtype != np.float32:
        raise ValueError(f"{name}: expected FP32 {tuple(shape)}, got "
                         f"{getattr(array, 'shape', None)}/{getattr(array, 'dtype', None)}")
    for start in range(0, shape[1], 8):
        if not np.isfinite(array[:, start:start + 8]).all():
            raise ValueError(f"{name}: nonfinite output")
    return array


def save_array(path, value, *, archive=False):
    """Atomically persist numeric arrays; leave existing evidence untouched."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            if archive:
                np.savez(stream, **value)
            else:
                np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def evidence(path, *, array=None):
    path = Path(path)
    record = {"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
    if array is not None:
        record.update(shape=list(array.shape), dtype=str(array.dtype), all_finite=True)
    return record


def save_logits(out, name, array, report):
    checked_tensor(array, LOGITS_SHAPE, name)
    path = out / f"{name}.npy"
    save_array(path, array)
    report["artifacts"][name] = evidence(path, array=array)


def save_checkpoints(out, arrays, schema, report):
    if set(arrays) != {row["name"] for row in schema} or len(arrays) != 30:
        raise ValueError("Diagnostic output omitted/duplicated a required checkpoint")
    for row in schema:
        checked_tensor(arrays[row["name"]], row["shape"], row["name"])
    path = out / "checkpoints.npz"
    save_array(path, arrays, archive=True)
    report["artifacts"]["checkpoints"] = evidence(path)


def exact_ordinary_diagnostic(out):
    ordinary = np.load(out / "logits.npy", mmap_mode="r", allow_pickle=False)
    diagnostic = np.load(out / "diagnostic_logits.npy", mmap_mode="r", allow_pickle=False)
    checked_tensor(ordinary, LOGITS_SHAPE, "ordinary logits")
    checked_tensor(diagnostic, LOGITS_SHAPE, "diagnostic logits")
    for start in range(0, LOGITS_SHAPE[1], 8):
        if not np.array_equal(ordinary[:, start:start + 8], diagnostic[:, start:start + 8]):
            raise ValueError(f"Ordinary and diagnostic logits differ at sequence chunk {start}")
    return {"passed": True, "comparison": "exact", "max_abs": 0.0,
            "positions_compared": 256, "elements_compared": int(np.prod(LOGITS_SHAPE))}


def diagnostic_graph(model, schema, model_dir, out):
    """Add graph outputs without re-exporting or copying the external weights."""
    model = onnx.ModelProto.FromString(model.SerializeToString())
    for row in schema:
        model.graph.output.append(onnx.helper.make_tensor_value_info(
            row["onnx_name"], onnx.TensorProto.FLOAT, row["shape"]))
    # External references are now relative to the derived metadata's directory.
    # Source manifest verification already bounded and validated these files.
    for tensor in assets.tensors(model):
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            for entry in tensor.external_data:
                if entry.key == "location":
                    original = (model_dir / entry.value).resolve()
                    if not original.is_relative_to(model_dir.resolve()) or not original.is_file():
                        raise ValueError("Invalid external tensor location in diagnostic model")
                    entry.value = os.path.relpath(original, out).replace("\\", "/")
    target = out / "diagnostic.onnx"
    target.write_bytes(model.SerializeToString())
    return target


def execute_ort(model_dir, out, feed, schema, model, report, stage):
    import onnxruntime as ort

    def session(path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.enable_cpu_mem_arena = False
        return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    runtime = stage("ort_ordinary_session", lambda: session(model_dir / "model.onnx"))
    logits = stage("ort_ordinary_execute", lambda: runtime.run(["logits"], feed)[0])
    stage("ordinary_save", lambda: save_logits(out, "logits", logits, report))
    del logits, runtime
    gc.collect()
    diagnostic = stage("diagnostic_graph", lambda: diagnostic_graph(model, schema, model_dir, out))
    report["artifacts"]["diagnostic_model"] = evidence(diagnostic)
    runtime = stage("ort_diagnostic_session", lambda: session(diagnostic))
    names = ["logits", *[row["onnx_name"] for row in schema]]
    results = stage("ort_diagnostic_execute", lambda: runtime.run(names, feed))
    if len(results) != 31:
        raise ValueError("ORT diagnostic did not return all 31 outputs")
    stage("diagnostic_save", lambda: save_logits(out, "diagnostic_logits", results[0], report))
    arrays = {row["name"]: value for row, value in zip(schema, results[1:])}
    stage("checkpoints_save", lambda: save_checkpoints(out, arrays, schema, report))
    report["ort"] = {"provider": "CPUExecutionProvider", "graph_optimization": "disabled",
                     "intra_op_threads": 1, "inter_op_threads": 1, "cpu_mem_arena": False}


def execute_ir(model_dir, out, feed, schema, model, report, stage):
    from scratchv.frontend.onnx_parser import ONNXParser
    from scratchv.verification.ir_interpreter import IRInterpreter
    from scratchv.verification.fp32_reference import profile

    fp32_mode = report.get("fp32_mode", "reference")

    parser = ONNXParser()
    program = stage("scratchv_parse", lambda: parser.parse(str(model_dir / "model.onnx"),
                                                          mmap_external_data=True))
    for value in parser.initializers.values():
        value.setflags(write=False)
    mapping = {}
    for row in schema:
        value = parser._value_map.get(row["onnx_name"])
        if value is None or value.name in mapping:
            raise ValueError(f"Missing/non-unique IR checkpoint mapping: {row['name']}")
        mapping[value.name] = row
    report["ir_checkpoint_bindings"] = {name: row["name"] for name, row in mapping.items()}
    report["executions"] = {}
    for kind in ("ordinary", "diagnostic"):
        arrays = {}

        def observe(name, value):
            if name in mapping:
                row = mapping[name]
                if row["name"] in arrays:
                    raise ValueError(f"Checkpoint executed multiple times: {row['name']}")
                arrays[row["name"]] = checked_tensor(value, row["shape"], row["name"]).copy()

        result = stage(f"ir_{kind}_execute", lambda: IRInterpreter(program).run(
            feed, initializers=parser.initializers, collect_memory_stats=True,
            memory_mode="last_use", copy_initializers=False,
            fp32_mode=fp32_mode,
            observer=observe if kind == "diagnostic" else None))
        report["executions"][kind] = {"executed_steps": result.executed_steps,
                                      "memory_stats": result.memory_stats}
        label = "logits" if kind == "ordinary" else "diagnostic_logits"
        stage(f"{kind}_save", lambda: save_logits(out, label, result.return_value, report))
        if kind == "diagnostic":
            stage("checkpoints_save", lambda: save_checkpoints(out, arrays, schema, report))
        del result, arrays
        gc.collect()
    report["ir"] = {"optimization_level": "none", "memory_mode": "last_use",
                    "fp32_mode": fp32_mode,
                    "fp32_profile": profile() if fp32_mode == "reference" else {"name": "numpy-native"},
                    "copy_initializers": False, "mmap_external_data": True,
                    "instruction_count": sum(len(b.instructions) for f in program.functions for b in f.blocks)}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=("ort", "ir"), required=True)
    cli.add_argument("--model-dir", type=Path, required=True)
    cli.add_argument("--input-file", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path, required=True)
    cli.add_argument("--optimization-level", choices=("none",), default="none")
    cli.add_argument("--fp32-mode", choices=("native", "reference"), default="reference")
    args = cli.parse_args(argv)
    out = new_output_dir(args.output_dir)
    started = time.perf_counter()
    report = {"schema_version": 1, "gate": GATE, "backend": args.backend,
              "optimization_level": args.optimization_level, "status": "FAIL", "passed": False,
              "fp32_mode": args.fp32_mode if args.backend == "ir" else None,
              "stage": "initialization", "stages_seconds": {}, "artifacts": {},
              "full_ir_executed": False, "full_model_executed": False, "w3_exit_accepted": False,
              "scope": "Complete single-backend execution and exact ordinary/diagnostic consistency. "
                       "Independent IR/ORT comparison belongs to the parent gate; not W3 team acceptance.",
              "model_id": assets.MANIFEST["model_id"], "revision": assets.MANIFEST["revision"]}

    def stage(name, operation):
        report["stage"] = name
        atomic_text(out / "progress.json", json.dumps({"stage": name,
                    "elapsed_seconds": time.perf_counter() - started,
                    "process_peak_rss_bytes": process_peak_rss_bytes()}) + "\n")
        print(f"[{args.backend}] {name}", flush=True)
        begin = time.perf_counter()
        try:
            return operation()
        finally:
            report["stages_seconds"][name] = time.perf_counter() - begin

    try:
        report.update(stage("source_evidence", source_evidence))
        model_dir = args.model_dir.resolve()
        report["model_dir"] = str(model_dir)
        report["files"] = stage("asset_hashes", lambda: assets.verify_files(model_dir))
        report["onnx_contract"] = stage("onnx_contract", lambda: assets.inspect_model(model_dir, report["files"]))
        feed, length = stage("input_validation", lambda: load_inputs(args.input_file))
        report["input"] = evidence(args.input_file)
        report["input"].update(path=str(args.input_file.resolve()), valid_length=length)
        model = stage("metadata_load", lambda: onnx.load(str(model_dir / "model.onnx"), load_external_data=False))
        report["graph_audit"] = stage("graph_audit", lambda: audit_graph_structure(model))
        schema = stage("checkpoint_selection", lambda: checkpoint_schema(model, report["graph_audit"]))
        report["checkpoints"] = schema
        atomic_text(out / "checkpoint_schema.json", json.dumps({"version": 1, "checkpoints": schema}, indent=2) + "\n")
        report["artifacts"]["checkpoint_schema"] = evidence(out / "checkpoint_schema.json")
        executor = execute_ort if args.backend == "ort" else execute_ir
        executor(model_dir, out, feed, schema, model, report, stage)
        report["full_model_executed"] = True
        report["full_ir_executed"] = args.backend == "ir"
        report["ordinary_diagnostic"] = stage("ordinary_diagnostic_consistency", lambda: exact_ordinary_diagnostic(out))
        final_sources = stage("source_recheck", source_evidence)["source_sha256"]
        if final_sources != report["source_sha256"]:
            raise ValueError("Production sources changed during numerical execution")
        report.update(passed=True, status="PASS", stage="complete")
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        if isinstance(exc, KeyboardInterrupt):
            report["interrupted"] = True
    report.update(elapsed_seconds=time.perf_counter() - started,
                  process_peak_rss_bytes=process_peak_rss_bytes(),
                  peak_scope="Lifetime resident high-water mark of this single backend worker, "
                             "including both ordinary and diagnostic runs; not isolated kernel memory")
    write_reports(out, report)
    print(f"[{GATE}] {report['status']}; {out / 'report.json'}", flush=True)
    return 130 if report.get("interrupted") else 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
