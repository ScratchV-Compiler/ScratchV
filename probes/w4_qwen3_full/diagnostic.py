"""Locate FP32 drift at one audited Qwen3 checkpoint using actual RV64 QEMU.

This executes a dependency slice, never the complete W4 numeric gate. ORT and
QEMU consume the same fixed input; original external ONNX weights are retained.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from probes.w1_qwen3_export import run as assets
from probes.w2_qwen3_parse.validation import audit_graph_structure
from probes.w3_common import atomic_text, new_output_dir, sha256_file, source_evidence, write_reports
from probes.w3_qwen3_full.cases import CASE_NAMES, input_cases
from probes.w3_qwen3_full.comparison import compare_positions
from probes.w3_qwen3_full.worker import checkpoint_schema, checked_tensor, save_array
from probes.w4_qwen3_full.run import ATOL, save_executable, spec_json
from scratchv.backend.tensor_c_codegen import TensorCCodegen
from scratchv.ir.types import BasicBlock, DataType, Function, Instruction, OpCode, Program
from scratchv.runtime.riscv_external import build_external, run_external
from scratchv.runtime.riscv_tensor import discover_toolchain
from scratchv.runtime.weight_bundle import write_weight_bundle

CHECKPOINT_NAMES = ("embedding", *(f"layer_{i}.output" for i in range(28)), "final_norm")


def select_checkpoint(model, name, audit=None):
    rows = checkpoint_schema(model, audit)
    matches = [row for row in rows if row["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Unknown/nonunique audited checkpoint: {name}")
    return matches[0]


def slice_ir(program, initializers, target_name):
    """Copy the pure straight-line reverse dependency slice, retaining all inputs.

    Tensor arrays stay borrowed: no model-sized copies or source IR mutations.
    Literal operands remain in instructions, while unused global bindings vanish.
    """
    if len(program.functions) != 1:
        raise ValueError("Checkpoint slicing requires one function")
    function = program.functions[0]
    if len(function.blocks) != 1 or function.blocks[0].phi_nodes:
        raise ValueError("Checkpoint slicing requires one straight-line block")
    instructions = function.blocks[0].instructions
    if (not instructions or instructions[-1].opcode != OpCode.RETURN
            or any(instruction.opcode == OpCode.RETURN for instruction in instructions[:-1])):
        raise ValueError("Checkpoint slicing requires exactly one final RETURN")
    definitions = {value.name: value for value in [*program.global_values, *function.params]}
    if len(definitions) != len(program.global_values) + len(function.params):
        raise ValueError("Duplicate input/global definition")
    for instruction in instructions:
        if ((instruction.opcode.is_control_flow() and instruction.opcode != OpCode.RETURN)
                or instruction.target or instruction.opcode in (OpCode.LOAD, OpCode.STORE, OpCode.ALLOCA)):
            raise ValueError("Checkpoint slicing requires pure tensor dataflow")
        if instruction.dest is not None:
            if instruction.dest.name in definitions:
                raise ValueError(f"Duplicate SSA definition: {instruction.dest.name}")
            definitions[instruction.dest.name] = instruction.dest
    target = definitions.get(target_name)
    if target is None or target.dtype != DataType.FLOAT32:
        raise ValueError("Checkpoint must name a defined FP32 tensor")
    needed, selected = {target_name}, []
    for instruction in reversed(instructions):
        if instruction.opcode == OpCode.RETURN:
            continue
        if instruction.dest is not None and instruction.dest.name in needed:
            selected.append(instruction)
            needed.update(value.name for value in instruction.operands)
    selected.reverse()
    globals_ = [value for value in program.global_values if value.name in needed]
    available = {value.name for value in [*globals_, *function.params]}
    for instruction in selected:
        if any(value.name not in available and not value.is_constant for value in instruction.operands):
            raise ValueError("Checkpoint dependency has no preceding definition")
        available.add(instruction.dest.name)
    if target_name not in available:
        raise ValueError("Checkpoint dependency is unresolved")
    result = Program()
    result.global_values = globals_
    entry = BasicBlock(function.blocks[0].name)
    entry.instructions = [*selected, Instruction(OpCode.RETURN, operands=[target])]
    result.functions = [Function(function.name, params=list(function.params), blocks=[entry])]
    result = copy.deepcopy(result)
    bindings = {value.name: initializers[value.name] for value in globals_ if value.name in initializers}
    return result, bindings


def slice_onnx(model, row, model_dir, output_path):
    """Save a pruned metadata graph referencing the original external files.

    The caller verifies the original pinned files before invoking this helper.
    No external tensor is materialized in protobuf, copied or re-exported.
    """
    import onnx

    model_dir, output_path = Path(model_dir).resolve(), Path(output_path).resolve()
    name = row["onnx_name"]
    producers = {}
    for node in model.graph.node:
        for output in node.output:
            if output and output in producers:
                raise ValueError(f"Duplicate ONNX producer: {output}")
            if output:
                producers[output] = node
    if name not in producers:
        raise ValueError("Checkpoint must be an ONNX graph-produced tensor")
    needed, selected = {name}, []
    for node in reversed(model.graph.node):
        if any(output in needed for output in node.output if output):
            if any(attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                   for attribute in node.attribute):
                raise ValueError("Checkpoint slice cannot contain implicit subgraph captures")
            selected.append(node)
            needed.update(value for value in node.input if value)
    selected.reverse()
    initializers = [value for value in model.graph.initializer if value.name in needed]
    available = {value.name for value in [*model.graph.input, *initializers]}
    for node in selected:
        if not {value for value in node.input if value} <= available:
            raise ValueError("ONNX slice has an unresolved/non-topological dependency")
        available.update(value for value in node.output if value)
    result = onnx.ModelProto()
    result.CopyFrom(model)
    del result.graph.node[:]
    result.graph.node.extend(selected)
    del result.graph.initializer[:]
    result.graph.initializer.extend(initializers)
    del result.graph.output[:]
    result.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, row["shape"]))
    retained_info = [value for value in result.graph.value_info if value.name in needed]
    del result.graph.value_info[:]
    result.graph.value_info.extend(retained_info)
    # Sparse weights and graph-attribute tensor references are unsupported by
    # the fixed dense model, and must not silently escape the dependency walk.
    if result.graph.sparse_initializer:
        raise ValueError("Sparse initializers are outside the fixed checkpoint contract")
    for tensor in assets.tensors(result):
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            locations = [entry for entry in tensor.external_data if entry.key == "location"]
            if len(locations) != 1:
                raise ValueError("External tensor requires exactly one location")
            original = (model_dir / locations[0].value.replace("\\", "/")).resolve()
            if not original.is_relative_to(model_dir) or not original.is_file():
                raise ValueError("External checkpoint weight escapes the trusted model directory")
            locations[0].value = os.path.relpath(original, output_path.parent).replace("\\", "/")
    with output_path.open("xb") as stream:
        stream.write(result.SerializeToString())
    return {"path": output_path.name, "sha256": sha256_file(output_path),
            "nodes": len(selected), "initializers": len(initializers),
            "inputs": [value.name for value in result.graph.input], "output": name}


def execute_ort(metadata, row, feed, output_path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_cpu_mem_arena = False
    session = ort.InferenceSession(str(metadata), options, providers=["CPUExecutionProvider"])
    value = checked_tensor(session.run([row["onnx_name"]], feed)[0], row["shape"], row["name"])
    save_array(output_path, value)
    del value, session
    gc.collect()
    return {"provider": "CPUExecutionProvider", "graph_optimization": "disabled",
            "intra_op_threads": 1, "inter_op_threads": 1, "cpu_mem_arena": False,
            "output_sha256": sha256_file(output_path)}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model-dir", required=True, type=Path)
    cli.add_argument("--output-dir", required=True, type=Path)
    cli.add_argument("--checkpoint", choices=CHECKPOINT_NAMES, default="layer_0.output")
    cli.add_argument("--case", choices=CASE_NAMES, default="full_seed_0")
    cli.add_argument("--cc")
    cli.add_argument("--qemu")
    cli.add_argument("--matmul-policy", choices=("sequential", "blocked_fma"), default="sequential")
    cli.add_argument("--kernel-calls", action=argparse.BooleanOptionalAction, default=True,
                     help="Emit independent tensor kernel functions (default: enabled)")
    cli.add_argument("--timeout", type=float, default=3600)
    cli.add_argument("--build-timeout", type=float, default=900)
    args = cli.parse_args(argv)
    if any(not math.isfinite(limit) or limit <= 0 for limit in (args.timeout, args.build_timeout)):
        cli.error("Timeouts must be finite and positive")
    out = new_output_dir(args.output_dir)
    started = time.perf_counter()
    report = {"schema_version": 1, "gate": "diagnostic:qemu-qwen3-checkpoint",
              "passed": False, "status": "FAIL", "stage": "initialization",
              "stages_seconds": {}, "checkpoint": args.checkpoint, "case": args.case,
              "matmul_policy": args.matmul_policy, "kernel_calls": args.kernel_calls,
              "scope": "One reverse dependency checkpoint slice; not the complete W4 numerical gate",
              "full_qemu_forward_executed": False, "numeric_gate_passed": False,
              "w4_team_accepted": False, "qemu_checkpoint_executed": False}

    def stage(name, operation):
        report["stage"] = name
        atomic_text(out / "progress.json", json.dumps({"stage": name,
                    "elapsed_s": time.perf_counter() - started}) + "\n")
        print(f"[W4 diagnostic] {name}", flush=True)
        begin = time.perf_counter()
        try:
            return operation()
        finally:
            report["stages_seconds"][name] = time.perf_counter() - begin
            write_reports(out, report)

    try:
        report.update(stage("source_evidence", source_evidence))
        model_dir = args.model_dir.resolve()
        report["model_files"] = stage("asset_hashes", lambda: assets.verify_files(model_dir))
        import onnx
        model = stage("metadata_load", lambda: onnx.load(str(model_dir / "model.onnx"), load_external_data=False))
        audit = stage("28_layer_audit", lambda: audit_graph_structure(model))
        row = stage("select_checkpoint", lambda: select_checkpoint(model, args.checkpoint, audit))
        report["checkpoint_schema"] = row
        from scratchv.frontend.onnx_parser import ONNXParser
        parser = ONNXParser()
        program = stage("parse", lambda: parser.parse(str(model_dir / "model.onnx"), mmap_external_data=True))
        target = parser._value_map.get(row["onnx_name"])
        if target is None:
            raise ValueError("Audited ONNX checkpoint has no IR value binding")
        sliced, bindings = stage("slice_ir", lambda: slice_ir(program, parser.initializers, target.name))
        report["ir_slice"] = {"checkpoint_value": target.name, "initializers": len(bindings),
                              "instructions": len(sliced.functions[0].blocks[0].instructions),
                              "original_instructions": len(program.functions[0].blocks[0].instructions)}
        atomic_text(out / "slice.ir.txt", sliced.dump())
        artifact = stage("external_codegen", lambda: TensorCCodegen(sliced, bindings,
            constant_storage="external", max_constant_bytes=3 * 1024**3,
            max_workspace_bytes=1024**3, kernel_calls=args.kernel_calls,
            matmul_policy=args.matmul_policy).generate())
        if list(artifact.output.shape) != row["shape"] or artifact.output.dtype != DataType.FLOAT32:
            raise ValueError("Generated checkpoint output disagrees with audited schema")
        report["codegen"] = {"source_bytes": len(artifact.source.encode()),
                              "workspace_bytes": artifact.workspace_bytes,
                              "constant_bytes": artifact.constant_bytes,
                              "kernel_calls": artifact.kernel_calls,
                              "matmul_policy": artifact.matmul_policy,
                              "compile_flags": list(artifact.compile_flags),
                              "output": spec_json(artifact.output)}
        stage("write_weight_bundle", lambda: write_weight_bundle(artifact.external_weights,
            artifact.external_initializers, out / "weights"))
        tools = discover_toolchain(args.cc, args.qemu)
        exe = stage("build_riscv_checkpoint", lambda: build_external(artifact, out / "weights",
            out / "build", tools, timeout=args.build_timeout))
        save_executable(exe, out)
        report.update(build=exe.evidence, memory_layout=exe.layout)
        report["derived_onnx"] = stage("slice_onnx", lambda: slice_onnx(model, row, model_dir, out / "checkpoint.onnx"))
        del artifact, bindings, sliced, program, parser, target, model
        gc.collect()
        _, valid_length, feed = next(case for case in input_cases() if case[0] == args.case)
        save_array(out / "inputs.npz", feed, archive=True)
        report["input_sha256"] = sha256_file(out / "inputs.npz")
        report["ort"] = stage("ort_checkpoint", lambda: execute_ort(out / "checkpoint.onnx", row, feed, out / "ort.npy"))
        actual, run = stage("qemu_checkpoint", lambda: run_external(exe, feed, out / "qemu", timeout=args.timeout))
        report.update(qemu=run, qemu_checkpoint_executed=True)
        expected = np.load(out / "ort.npy", mmap_mode="r", allow_pickle=False)
        checked_tensor(actual, row["shape"], "RV64 checkpoint")
        report["comparison"] = stage("compare_checkpoint", lambda: compare_positions(actual, expected, valid_length, atol=ATOL))
        report["diagnostic_threshold_passed"] = report["comparison"]["passed"]
        report.update(passed=True, status="COMPLETE", stage="complete")
        report["completion_meaning"] = "Diagnostic execution and comparison complete; checkpoint error does not accept or reject the full W4 logits gate"
        del actual, expected
    except BaseException as exc:
        report.update(passed=False, status="FAIL", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    report["elapsed_seconds"] = time.perf_counter() - started
    write_reports(out, report)
    print(f"[W4 diagnostic] {report['status']}: {out / 'report.json'}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
