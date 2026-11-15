"""Compare a materialized and a fused Transpose → MatMul → Softmax graph on four QEMU cores."""
import argparse
import json
import os
from pathlib import Path
from statistics import median
import subprocess

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper

from probes.w6_performance import run
from probes.w6_performance.attention_chain_kernel import attention_chain_artifact
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import Value
from scratchv.runtime.riscv_external import build_external
from scratchv.runtime.riscv_tensor import discover_toolchain
from scratchv.runtime.weight_bundle import write_weight_bundle


CASE = "transpose-matmul-softmax"
INPUT_SHAPE = (16, 256, 128)
OUTPUT_SHAPE = (16, 256, 256)
PERMUTATION = (0, 2, 1)
SEED = 4108
VARIANTS = ("separate", "fused")
ROUNDS = 8


def chain_program():
    query = Value("q", shape=INPUT_SHAPE)
    key = Value("k", shape=INPUT_SHAPE)
    builder = IRBuilder()
    builder.new_function("main", [query, key])
    builder.new_block("entry")
    scores = builder.matmul(query, builder.transpose(key, PERMUTATION))
    builder.ret(builder.softmax(scores, axis=-1))
    return builder.program


def ort_reference(inputs):
    nodes = [
        helper.make_node("Transpose", ["k"], ["kt"], perm=PERMUTATION),
        helper.make_node("MatMul", ["q", "kt"], ["scores"]),
        helper.make_node("Softmax", ["scores"], ["probabilities"], axis=-1),
    ]
    graph = helper.make_graph(
        nodes, CASE, [helper.make_tensor_value_info(name, TensorProto.FLOAT, INPUT_SHAPE)
                      for name in ("q", "k")],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, OUTPUT_SHAPE)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    return session.run(None, inputs)[0]


def prepare(directory, tools):
    rng = np.random.default_rng(SEED)
    inputs = {name: rng.standard_normal(INPUT_SHAPE).astype(np.float32) for name in ("q", "k")}
    reference = ort_reference(inputs)
    np.save(directory / "ort.npy", reference, allow_pickle=False)
    program = chain_program()
    bundle = directory / "weights"
    builds = {}
    for variant in VARIANTS:
        artifact = attention_chain_artifact(
            program, fused=variant == "fused", nonlinear_policy="fp32-reference")
        if artifact.external_weights or artifact.external_initializers:
            raise ValueError("The Attention chain expects two runtime inputs and no weights")
        if not bundle.exists():
            write_weight_bundle(artifact.external_weights, artifact.external_initializers, bundle)
        exe = build_external(artifact, bundle, directory / variant / "build", tools, cores=4)
        if (exe.cores, exe.evidence["qemu_workspace_layout"], exe.evidence["matmul_tile"],
                exe.evidence["matmul_rhs_layout"], exe.evidence["nonlinear_policy"]) != (
                4, "none", "2x4", "original", "fp32-reference"):
            raise ValueError(f"{variant}: unexpected guest build configuration")
        if ([spec.name for spec in exe.inputs] != ["q", "k"] or
                any(spec.shape != INPUT_SHAPE for spec in exe.inputs) or
                exe.output.shape != OUTPUT_SHAPE):
            raise ValueError(f"{variant}: unexpected input/output ABI")
        if exe.layout["regions"]["workspace"]["nbytes"] != artifact.workspace_bytes:
            raise ValueError(f"{variant}: guest memory layout disagrees with generated workspace")
        builds[variant] = (exe, artifact.workspace_bytes)
    return inputs, reference, builds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 4:
        parser.error("Four available logical CPUs are required")
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    tools = discover_toolchain()
    prepared = prepare(directory, tools)
    reference = prepared[1]
    expected_max = np.argmax(reference, axis=-1)
    rows, baseline = [], {}
    with (directory / "runs.jsonl").open("x", encoding="utf-8") as log:
        for rep in range(ROUNDS + 1):
            phase = "warmup" if rep == 0 else "measure"
            order = VARIANTS if rep == 0 or rep % 2 else VARIANTS[::-1]
            for variant in order:
                folder = directory / variant / f"run-{rep}"
                row = run.measure(CASE, variant, "2x4", "original", phase, rep, prepared, folder, baseline)
                # measure() checked the whole output against ORT and compared both guests bit-for-bit.
                output = np.memmap(folder / "output.bin", dtype="<f4", mode="r", shape=OUTPUT_SHAPE)
                row.update(rhs_source="runtime", argmax_rows_matched=int(np.sum(
                    np.argmax(output, axis=-1) == expected_max)), row_count=int(expected_max.size),
                    max_row_sum_error=float(np.max(np.abs(np.sum(output, axis=-1, dtype=np.float64) - 1))))
                rows.append(row)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                print(variant, phase, rep, f"{row['forward_s'] * 1000:.2f} ms", flush=True)
    summary = {}
    for variant in VARIANTS:
        trials = [r for r in rows if r["variant"] == variant and r["phase"] == "measure"]
        forward = [r["forward_s"] for r in trials]
        summary[variant] = dict(
            forward_s=median(forward), min_forward_s=min(forward), max_forward_s=max(forward),
            qemu_s=median(r["qemu_s"] for r in trials), workspace_bytes=trials[0]["workspace_bytes"],
            stack_reserved_bytes=prepared[2][variant][0].layout["regions"]["stack"]["nbytes"],
            qemu_peak_bytes=median(r["qemu_peak_bytes"] for r in trials),
            combined_peak_bytes=median(r["combined_peak_bytes"] for r in trials),
            max_abs_vs_ort=max(r["max_abs_vs_ort"] for r in trials),
            max_ort_bit_mismatches=max(r["ort_bit_mismatches"] for r in trials),
            min_argmax_rows_matched=min(r["argmax_rows_matched"] for r in trials),
            max_row_sum_error=max(r["max_row_sum_error"] for r in trials),
        )
    separate, fused = (summary[v] for v in VARIANTS)
    result = dict(
        case=CASE, ops=["Transpose(k,[0,2,1])", "MatMul(q,kt)", "Softmax(axis=-1)"],
        seed=SEED, input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, outputs_per_run=int(reference.size),
        rows_per_run=int(expected_max.size), cores=4, affinity=affinity, dtype="FP32",
        qemu_cpu="rv64", workspace_layout="none", matmul_policy="blocked_fma", matmul_tile="2x4",
        nonlinear_policy="fp32-reference", ort=ort.__version__, ort_graph_optimizations=False,
        qemu=subprocess.check_output([tools.qemu, "--version"], text=True).splitlines()[0],
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=run.ROOT, text=True).strip(),
        warmups_per_variant=1, measured_per_variant=ROUNDS,
        metric="median wall seconds from QEMU launch through UART completion frame",
        forward_speedup=separate["forward_s"] / fused["forward_s"],
        workspace_reduction_bytes=separate["workspace_bytes"] - fused["workspace_bytes"],
        workspace_scope="generator ABI workspace, excluding inputs, caller output and reserved core stacks",
        stack_scope="total reserved guest stack for all cores; actual stack use was not sampled",
        max_abs_gate=1e-3, all_variants_bitwise_equal=all(r["bitwise_equal"] for r in rows),
        summary=summary, runs=rows,
    )
    (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                           encoding="utf-8")
    print(f"Combined graph speedup: {result['forward_speedup']:.3f}x", flush=True)


if __name__ == "__main__":
    main()
