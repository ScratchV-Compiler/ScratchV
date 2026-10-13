"""Localize full-Qwen3 numerical differences using identical layer inputs.

A completed diagnosis is not a numerical gate: large intermediate activations
may differ by more than 1e-4 in a few FP32 ULPs. The full-model runner remains
the authority for full logits. This tool never changes execution precision.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import onnx

from probes.w1_qwen3_export.run import verify_files
from probes.w3_common import (new_output_dir, process_peak_rss_bytes, sha256_file,
                             source_evidence, write_reports)
from probes.w3_layer_diff.run import load_arrays, load_schema
from probes.w3_qwen3_full.comparison import compare_tensor
from probes.w3_qwen3_full.run import validate_worker
from probes.w3_qwen3_full.worker import checkpoint_schema, load_inputs


def require(value, message):
    if not value:
        raise ValueError(message)


def selected_layers(values):
    values = [0] if values is None else list(values)
    require(bool(values) and all(type(v) is int and 0 <= v < 28 for v in values),
            "Layers must be integer indices in [0, 27]")
    require(len(values) == len(set(values)), "Duplicate layer selection")
    return sorted(values)


def compare_queries(actual, reference, valid, axis):
    require(actual.ndim > axis and actual.shape[axis] == 256 and 1 <= valid <= 256,
            "Expected complete L256 query axis and valid length")
    row = compare_tensor(actual, reference)
    prefix = [slice(None)] * actual.ndim
    suffix = prefix.copy()
    prefix[axis], suffix[axis] = slice(0, valid), slice(valid, None)
    row["valid_queries"] = compare_tensor(actual[tuple(prefix)], reference[tuple(prefix)])
    row["padding_queries"] = (compare_tensor(actual[tuple(suffix)], reference[tuple(suffix)])
                              if valid < 256 else None)
    row["query_axis"] = axis
    return row


def probability_diagnostics(probabilities, mask):
    require(probabilities.dtype == np.float32 and probabilities.ndim == 4
            and probabilities.shape[0] == 1 and probabilities.shape[2:] == (256, 256)
            and np.isfinite(probabilities).all(), "Invalid attention probabilities")
    require(mask.shape == (1, 1, 256, 256) and mask.dtype == np.float32,
            "Invalid attention mask")
    invisible = np.broadcast_to(mask < 0, probabilities.shape)
    return {"masked_probability_max_abs": float(np.abs(probabilities[invisible]).max()),
            "row_sum_max_abs_error": float(np.abs(probabilities.sum(-1, dtype=np.float64) - 1).max()),
            "minimum_probability": float(probabilities.min())}


def fp64_error_oracle(op, feed, ir_value, ort_value):
    """Double precision is only an independent local error estimate."""
    require(ir_value.dtype == ort_value.dtype == np.float32,
            "Execution outputs must remain FP32")
    if op == "MatMul":
        truth = np.matmul(feed["a"].astype(np.float64), feed["b"].astype(np.float64))
    elif op == "ReduceMean":
        truth = feed["x"].astype(np.float64).mean(-1, keepdims=True)
    elif op == "Softmax":
        x = feed["x"].astype(np.float64)
        exp = np.exp(x - x.max(-1, keepdims=True))
        truth = exp / exp.sum(-1, keepdims=True)
    else:
        raise ValueError("Unsupported local diagnostic operation")
    return {"scope": "diagnostic only; neither execution result nor acceptance output",
            "ir_max_abs": float(np.abs(ir_value - truth).max()),
            "ort_max_abs": float(np.abs(ort_value - truth).max())}


class LayerSlice:
    """Extract exact pinned-graph ancestors; never reconstruct model arithmetic."""

    def __init__(self, model, model_dir):
        self.model, self.model_dir = model, Path(model_dir)
        self.nodes = list(model.graph.node)
        self.producers = {name: node for node in self.nodes for name in node.output}
        self.info = {v.name: v for v in [*model.graph.value_info, *model.graph.input, *model.graph.output]}
        self.initializers = {v.name: v for v in model.graph.initializer}
        require(len(self.producers) == sum(len(n.output) for n in self.nodes),
                "Duplicate graph values")

    @staticmethod
    def one(nodes, description):
        rows = list(nodes)
        require(len(rows) == 1, f"Expected unique {description}")
        return rows[0]

    def norm(self, weight):
        node = self.one((n for n in self.nodes if n.op_type == "Mul" and weight in n.input), weight)
        value = next(x for x in node.input if x != weight)
        cast = self.producers[value]
        require(cast.op_type == "Cast", "Unexpected normalization cast")
        mul = self.producers[cast.input[0]]
        require(mul.op_type == "Mul", "Unexpected normalization product")
        cast = self.producers[mul.input[0]]
        require(cast.op_type == "Cast", "Unexpected normalization input")
        return cast.input[0], node.output[0]

    def projection(self, weight):
        tr = self.one((n for n in self.nodes if n.op_type == "Transpose" and weight in n.input), weight)
        mat = self.one((n for n in self.nodes if n.op_type == "MatMul" and tr.output[0] in n.input), weight + " MatMul")
        return mat.input[0], mat.output[0]

    def describe(self, index):
        p = f"model.model.layers.{index}."
        hidden, input_norm = self.norm(p + "input_layernorm.weight")
        residual, postnorm = self.norm(p + "post_attention_layernorm.weight")
        _, qnorm = self.norm(p + "self_attn.q_norm.weight")
        _, knorm = self.norm(p + "self_attn.k_norm.weight")
        _, vproj = self.projection(p + "self_attn.v_proj.weight")
        context, attn_output = self.projection(p + "self_attn.o_proj.weight")
        _, mlp = self.projection(p + "mlp.down_proj.weight")
        final = self.one((n for n in self.nodes if n.op_type == "Add" and mlp in n.input), "MLP residual").output[0]
        begin, end = self.nodes.index(self.producers[input_norm]), self.nodes.index(self.producers[final])
        block = self.nodes[begin:end + 1]
        softmax = self.one((n for n in block if n.op_type == "Softmax"), "layer Softmax").output[0]
        probs = self.one((n for n in block if n.op_type == "Cast" and softmax in n.input), "Softmax cast").output[0]

        def rope(norm):
            tr = self.one((n for n in block if n.op_type == "Transpose" and norm in n.input), "norm transpose")
            mul = self.one((n for n in block if n.op_type == "Mul" and tr.output[0] in n.input), "RoPE cosine product")
            return self.one((n for n in block if n.op_type == "Add" and mul.output[0] in n.input), "RoPE sum").output[0]

        return hidden, {"input_norm": input_norm, "q_norm": qnorm, "k_norm": knorm, "v_proj": vproj,
                        "rope_q": rope(qnorm), "rope_k": rope(knorm), "attn_probs": probs,
                        "attn_context": context, "attn_output": attn_output, "attn_residual": residual,
                        "post_attention_norm": postnorm, "mlp": mlp, "residual": final}

    def build(self, hidden, outputs):
        stop, needed, queue, chosen = {hidden, "attention_mask"}, set(outputs), list(outputs), set()
        while queue:
            value = queue.pop()
            if value in stop or value not in self.producers:
                continue
            node = self.producers[value]
            if node.name in chosen:
                continue
            chosen.add(node.name)
            for name in node.input:
                if name and name not in needed:
                    needed.add(name)
                    queue.append(name)
        nodes = [copy.deepcopy(n) for n in self.nodes if n.name in chosen]
        weights = [onnx.numpy_helper.from_array(onnx.numpy_helper.to_array(t, base_dir=str(self.model_dir)), name)
                   for name, t in self.initializers.items() if name in needed and name not in stop]
        for node in nodes:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR and attr.t.external_data:
                    attr.t.CopyFrom(onnx.numpy_helper.from_array(
                        onnx.numpy_helper.to_array(attr.t, base_dir=str(self.model_dir)), attr.t.name))
        graph = onnx.helper.make_graph(nodes, "pinned_layer_slice",
                 [copy.deepcopy(self.info[x]) for x in sorted(stop) if x in needed],
                 [copy.deepcopy(self.info[x]) for x in outputs], initializer=weights)
        result = onnx.helper.make_model(graph, opset_imports=self.model.opset_import)
        result.ir_version = self.model.ir_version
        onnx.checker.check_model(result)
        return result


def run_pair(model, feed, outputs, path, *, fp32_mode="native"):
    """Actual ORT and production Parser/IR execution, with FP32 feeds unchanged."""
    import onnxruntime as ort
    from scratchv.frontend.onnx_parser import ONNXParser
    from scratchv.verification.ir_interpreter import IRInterpreter

    require(all(v.dtype == np.float32 and np.isfinite(v).all() for v in feed.values()),
            "Local execution feeds must be finite FP32")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    diagnostic_path = path.with_name(path.stem + "-diagnostic.onnx")
    onnx.save_model(model, diagnostic_path)
    session = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    ort_values = dict(zip(outputs, session.run(list(outputs.values()), feed)))
    ordinary = copy.deepcopy(model)
    wanted = outputs[next(reversed(outputs))]
    last = next(v for v in ordinary.graph.output if v.name == wanted)
    last = copy.deepcopy(last)
    del ordinary.graph.output[:]
    ordinary.graph.output.append(last)
    onnx.save_model(ordinary, path)
    parser = ONNXParser()
    program = parser.parse(str(path))
    for value in parser.initializers.values():
        value.setflags(write=False)
    bindings = {parser._value_map[value].name: name for name, value in outputs.items()}
    require(len(bindings) == len(outputs), "Aliased/ambiguous diagnostic checkpoints")
    ir_values = {}

    def observe(name, value):
        if name in bindings:
            require(bindings[name] not in ir_values, "Checkpoint executed twice")
            ir_values[bindings[name]] = value.copy()

    result = IRInterpreter(program).run(feed, initializers=parser.initializers,
              memory_mode="last_use", copy_initializers=False, observer=observe,
              fp32_mode=fp32_mode)
    require(set(ir_values) == set(outputs), "Missing executed checkpoint")
    require(all(v.dtype == np.float32 and np.isfinite(v).all() for v in [*ir_values.values(), *ort_values.values()]),
            "Nonfinite/non-FP32 execution output")
    return ir_values, ort_values, {"executed_steps": result.executed_steps,
                                 "fp32_mode": fp32_mode,
                                 "model_sha256": sha256_file(path), "model_file": path.name,
                                 "ort_model_file": diagnostic_path.name,
                                 "ort_model_sha256": sha256_file(diagnostic_path),
                                 "checkpoint_onnx_names": dict(outputs)}


def primitive(op, feed, shape):
    inputs, initializers, attrs = list(feed), [], {}
    if op == "ReduceMean":
        inputs += ["axes"]
        initializers = [onnx.numpy_helper.from_array(np.array([-1], dtype=np.int64), "axes")]
        attrs["keepdims"] = 1
    elif op == "Softmax":
        attrs["axis"] = -1
    graph = onnx.helper.make_graph([onnx.helper.make_node(op, inputs, ["y"], **attrs)], op,
           [onnx.helper.make_tensor_value_info(k, 1, v.shape) for k, v in feed.items()],
           [onnx.helper.make_tensor_value_info("y", 1, shape)], initializer=initializers)
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
    model.ir_version = 10
    return model


def local_primitives(extract, ort_values, mask, valid, out, *, fp32_mode="native"):
    rows = []

    def run(name, op, feed, shape, axis=1):
        ir, ort_values, evidence = run_pair(primitive(op, feed, shape), feed, {"result": "y"},
                                           out / f"op-{name}.onnx", fp32_mode=fp32_mode)
        a, b = ir["result"], ort_values["result"]
        row = {"name": name, "op": op, "comparison": compare_queries(a, b, valid, axis),
               "fp64_error_oracle": fp64_error_oracle(op, feed, a, b), **evidence}
        if op == "Softmax":
            row["probabilities"] = {"ir": probability_diagnostics(a, mask), "ort": probability_diagnostics(b, mask)}
        rows.append(row)
        values_path = out / f"op-{name}-values.npz"
        np.savez(values_path, ir=a, ort=b, **feed)
        row["values"] = {"file": values_path.name, "bytes": values_path.stat().st_size,
                         "sha256": sha256_file(values_path)}
        return b

    run("input_norm_reduce_mean", "ReduceMean", {"x": ort_values["square"]}, [1, 256, 1])
    for key, width in (("q", 2048), ("k", 1024)):
        name = f"model.model.layers.0.self_attn.{key}_proj.weight"
        weight = onnx.numpy_helper.to_array(extract.initializers[name], base_dir=str(extract.model_dir)).T.copy()
        run(f"projection_{key}", "MatMul", {"a": ort_values["input_norm"], "b": weight}, [1, 256, width])
    q, k = ort_values["rope_q"], np.repeat(ort_values["rope_k"], 2, axis=1)
    scores = run("attention_scores", "MatMul", {"a": q, "b": k.transpose(0, 1, 3, 2).copy()}, [1, 16, 256, 256], 2)
    scores = scores * np.float32(128 ** -0.5) + mask
    run("attention_softmax", "Softmax", {"x": scores}, [1, 16, 256, 256], 2)
    return rows


def validate_reference_metadata(record, schema, expected_schema, files, input_hash):
    require(record.get("gate") == "numeric:w3-full-worker" and record.get("backend") == "ort"
            and record.get("passed") is True and record.get("status") == "PASS", "Expected executed ORT worker")
    require(record.get("files") == files, "Reference model assets differ")
    require(record.get("input", {}).get("sha256") == input_hash, "Reference inputs differ")
    require(schema == expected_schema and record.get("checkpoints") == schema,
            "Reference schema differs from fixed graph dataflow")


def validate_producer_pair(ort_record, ir_record):
    require(bool(ort_record.get("source_sha256"))
            and ort_record["source_sha256"] == ir_record.get("source_sha256"),
            "Historical IR/ORT producer source fingerprints differ")


def producer_fp32_mode(ir_record):
    details = ir_record.get("ir", {})
    mode = details.get("fp32_mode", "native")
    require(mode in ("native", "reference"), "Unknown producer FP32 mode")
    require(ir_record.get("fp32_mode", mode) == mode, "Producer FP32 modes disagree")
    if mode == "reference":
        from scratchv.verification.fp32_reference import profile
        require(details.get("fp32_profile") == profile(),
                "Unsupported historical FP32 reference profile")
    else:
        require("fp32_profile" not in details or details["fp32_profile"] == {"name": "numpy-native"},
                "Unsupported historical native FP32 profile")
    return mode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, action="append", help="Repeat for selected indices 0..27; default 0")
    args = parser.parse_args(argv)
    try:
        layers = selected_layers(args.layer)
    except ValueError as exc:
        parser.error(str(exc))
    out = new_output_dir(args.output_dir)
    report = {"gate": "diagnostic:w3-full-localize", "status": "FAIL", "passed": False,
              "diagnostic_complete": False, "numeric_gate_passed": False, "w3_exit_accepted": False,
              "scope": "Selected exact layer slices and equal-input FP32 primitives; cannot accept the full-model gate",
              "selected_layers": layers, "layers": [], "primitives": [],
              "execution_dtype": "float32", "fp64_scope": "local error oracle only"}
    start = time.perf_counter()
    try:
        report.update(source_evidence())
        files = verify_files(args.model_dir)
        feed, valid = load_inputs(args.case_dir / "inputs.npz")
        model = onnx.load(args.model_dir / "model.onnx", load_external_data=False)
        expected = checkpoint_schema(model)
        folder = args.case_dir / "ort"
        schema = load_schema(folder / "checkpoint_schema.json")
        record = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        ir_folder = args.case_dir / "ir"
        ir_record = json.loads((ir_folder / "report.json").read_text(encoding="utf-8"))
        validate_producer_pair(record, ir_record)
        input_hash = sha256_file(args.case_dir / "inputs.npz")
        validate_reference_metadata(record, schema, expected, files, input_hash)
        # Historical source identity is retained rather than falsely claiming a
        # new localizer file existed when that complete reference was generated.
        validate_worker(folder, "ort", args.case_dir / "inputs.npz", record["source_sha256"], files)
        validate_worker(ir_folder, "ir", args.case_dir / "inputs.npz", record["source_sha256"], files)
        fp32_mode = producer_fp32_mode(ir_record)
        report["fp32_mode"] = fp32_mode
        states = load_arrays(folder / "checkpoints.npz", schema, max_bytes=64 * 1024**2)
        report.update(files=files, valid_length=valid, input_sha256=input_hash,
                      reference_report_sha256=sha256_file(folder / "report.json"),
                      paired_ir_report_sha256=sha256_file(ir_folder / "report.json"),
                      reference_source_sha256=record["source_sha256"])
        extract = LayerSlice(model, args.model_dir)
        for index in layers:
            hidden, names = extract.describe(index)
            require(hidden == expected[index]["onnx_name"] and names["residual"] == expected[index + 1]["onnx_name"],
                    "Extracted layer boundary differs from audited graph")
            if index == 0:
                names = {"square": "pow_1", "mean": "mean", "epsilon_add": "add", "rsqrt": "rsqrt", **names}
            state = states["embedding" if index == 0 else f"layer_{index - 1}.output"]
            pair_feed = {hidden: state, "attention_mask": feed["attention_mask"]}
            graph = extract.build(hidden, list(names.values()))
            ir, ort_values, evidence = run_pair(graph, pair_feed, names, out / f"layer-{index:02d}.onnx",
                                               fp32_mode=fp32_mode)
            comparisons = {name: compare_queries(ir[name], ort_values[name], valid,
                            2 if name in ("rope_q", "rope_k", "attn_probs") else 1) for name in names}
            report["layers"].append({"layer": index, "same_input": comparisons, **evidence,
                "probabilities": {"ir": probability_diagnostics(ir["attn_probs"], feed["attention_mask"]),
                                  "ort": probability_diagnostics(ort_values["attn_probs"], feed["attention_mask"])}})
            values_path = out / f"layer-{index:02d}-values.npz"
            np.savez(values_path, **{"ir_" + k: v for k, v in ir.items()},
                     **{"ort_" + k: v for k, v in ort_values.items()})
            report["layers"][-1]["values"] = {"file": values_path.name, "bytes": values_path.stat().st_size,
                                             "sha256": sha256_file(values_path)}
            if index == 0:
                report["primitives"] = local_primitives(extract, ort_values, feed["attention_mask"], valid,
                                                        out, fp32_mode=fp32_mode)
            del graph, ir, ort_values
            gc.collect()
        require(source_evidence()["source_sha256"] == report["source_sha256"], "Sources changed during diagnosis")
        report.update(status="COMPLETE", passed=True, diagnostic_complete=True)
    except KeyboardInterrupt:
        report.update(interrupted=True, error="KeyboardInterrupt: local diagnosis cancelled")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report.update(elapsed_seconds=time.perf_counter() - start, process_peak_rss_bytes=process_peak_rss_bytes())
    write_reports(out, report)
    return 130 if report.get("interrupted") else (0 if report["diagnostic_complete"] else 1)


if __name__ == "__main__":
    raise SystemExit(main())
