"""Measure two consecutive one-query Attention steps at 255 and 256 valid positions."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import re
from statistics import median
import struct
import subprocess
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper

from probes.w6_performance.attention_decode_sequence import sequence_artifact
from probes.w6_performance.attention_decode_sequence_full import full_sequence_artifact
from probes.w6_performance.attention_head_groups import head_groups_artifact
from probes.w6_performance.lm_head_timing import read_timebase
from probes.w6_performance.run import measure, ort_matmul
from scratchv.runtime import riscv_external
from scratchv.runtime.riscv_tensor import discover_toolchain
from scratchv.runtime.weight_bundle import write_weight_bundle


BASELINE = "native-transpose-2x4"
DIRECT = "native-paired-dot"
HEAD_GROUPS = "head-groups-2x4"
LENGTHS = (255, 256)


def ort_context(query, keys_t, values):
    """Return every output component for 16 query heads, using ordinary ONNX operators."""
    inputs = {"q": query, "kt": keys_t, "v": values}
    scale = helper.make_tensor("scale", TensorProto.FLOAT, [], [float(1 / np.sqrt(128))])
    graph = helper.make_graph([
        helper.make_node("MatMul", ["q", "kt"], ["scores"]),
        helper.make_node("Mul", ["scores", "scale"], ["scaled"]),
        helper.make_node("Softmax", ["scaled"], ["probabilities"], axis=-1),
        helper.make_node("MatMul", ["probabilities", "v"], ["context"]),
    ], "single_query_attention", [
        helper.make_tensor_value_info(k, TensorProto.FLOAT, v.shape) for k, v in inputs.items()
    ], [helper.make_tensor_value_info("context", TensorProto.FLOAT, query.shape)], [scale])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    return session.run(None, inputs)[0]


def make_case(seed, full_attention=False):
    rng = np.random.default_rng(seed)
    history = rng.standard_normal((8, 254, 128)).astype(np.float32)
    queries = rng.standard_normal((2, 8, 2, 128)).astype(np.float32)
    new_keys = rng.standard_normal((2, 8, 128)).astype(np.float32)
    inputs = dict(history=history, queries=queries, new_keys=new_keys)
    if full_attention:
        inputs["value_history"] = rng.standard_normal(history.shape).astype(np.float32)
        inputs["new_values"] = rng.standard_normal(new_keys.shape).astype(np.float32)
    expected = []
    for step, length in enumerate(LENGTHS):
        keys = np.concatenate([history, new_keys[:step + 1].transpose(1, 0, 2)], axis=1)
        # Compare against the independent 16-head formulation of grouped-query attention.
        all_heads = np.ascontiguousarray(np.repeat(keys, 2, axis=0).transpose(0, 2, 1))
        q = queries[step].reshape(16, 1, 128)
        if full_attention:
            values = np.concatenate([inputs["value_history"], inputs["new_values"][:step + 1].transpose(1, 0, 2)], axis=1)
            context = ort_context(q, all_heads, np.repeat(values, 2, axis=0))
            expected.append(context.reshape(8, 2, 128))
        else:
            expected.append(ort_matmul(f"decode_{length}", q, all_heads).ravel())
    return inputs, np.stack(expected) if full_attention else np.concatenate(expected)


def build_variants(out, tools, instrumented, full_attention, names):
    builds, addresses = {}, {}
    artifact_builder = full_sequence_artifact if full_attention else sequence_artifact
    expected_shape = (2, 8, 2, 128) if full_attention else (16 * sum(LENGTHS),)
    counter_bytes = 96 if full_attention else 64
    symbol = "w6_attention_sequence_ticks" if full_attention else "w6_decode_sequence_ticks"
    out.mkdir(parents=True)
    bundle = out / "weights"
    for name in names:
        if name == HEAD_GROUPS:
            artifact = head_groups_artifact(instrumented=instrumented)
        else:
            artifact = artifact_builder(name, instrumented=instrumented)
        if not bundle.exists():
            write_weight_bundle(artifact.external_weights, artifact.external_initializers, bundle)
        exe = riscv_external.build_external(artifact, bundle, out / name, tools, cores=4)
        if (exe.cores != 4 or exe.weights or exe.output.shape != expected_shape
                or exe.evidence["qemu_workspace_layout"] != "none" or artifact.matmul_rhs_panels):
            raise ValueError("Unexpected shape, weights or platform configuration for the two-step sequence")
        builds[name] = exe, artifact.workspace_bytes
        if instrumented:
            symbols = subprocess.check_output(["nm", "-S", exe.elf_path], text=True)
            matches = re.findall(rf"^(\w+)\s+(\w+)\s+[bBdD]\s+{symbol}$", symbols, re.M)
            if len(matches) != 1 or int(matches[0][1], 16) != counter_bytes:
                raise ValueError(f"Could not locate {counter_bytes} bytes of timing counters in the measured ELF")
            addresses[name] = int(matches[0][0], 16)
            if addresses[name] % 8:
                raise ValueError("Invalid counter alignment")
    return builds, addresses


def capture_step_ticks(address, nbytes=64):
    class TimingQMP(riscv_external.QMP):
        def command(self, name, arguments=None):
            response = super().command(name, arguments)
            if name == "stop":
                reply = super().command("human-monitor-command", {
                    "command-line": f"pmemsave 0x{address:x} {nbytes} sequence-ticks.bin",
                })
                if reply.strip():
                    raise RuntimeError(f"QMP could not read sequence counters: {reply}")
            return response
    return patch.object(riscv_external, "QMP", TimingQMP)


def decode_ticks(folder, hz, full_attention=False, *, separate_stages=True):
    count = 6 if full_attention else 4
    ticks = struct.unpack(f"<{2 * count}Q", (folder / "sequence-ticks.bin").read_bytes())
    valid_ticks = ticks if separate_stages else tuple(ticks[i] for i in (0, 1, 4, 5, 6, 7, 10, 11))
    if valid_ticks[0] == 0 or any(a > b for a, b in zip(valid_ticks, valid_ticks[1:])):
        raise ValueError(f"Invalid sequence clock: {ticks}")
    result = {"guest_ticks": ticks}
    for step, length in enumerate(LENGTHS):
        segment = ticks[step * count:(step + 1) * count]
        op_start, op_end = segment[1], segment[-2]
        if op_start >= op_end:
            raise ValueError(f"Invalid step {length} timing: {ticks}")
        result[f"operator_{length}_ms"] = (op_end - op_start) * 1000 / hz
        result[f"step_{length}_ms"] = (segment[-1] - segment[0]) * 1000 / hz
        if full_attention and separate_stages:
            for stage, begin, end in zip(("qk", "scale_softmax", "pv"), segment[1:4], segment[2:5]):
                if begin >= end:
                    raise ValueError(f"Invalid {stage} timing at {folder}")
                result[f"{stage}_{length}_ms"] = (end - begin) * 1000 / hz
    return result


def summarize(rows, fields, names=(BASELINE, DIRECT)):
    pairs = {}
    for row in rows:
        if row["phase"] == "measure":
            pairs.setdefault((row["round"], row["seed"]), {})[row["variant"]] = row
    baseline, candidate = names
    label = "direct" if candidate == DIRECT else "candidate"
    groups = {}
    for field in fields:
        originals = [p[baseline][field] for p in pairs.values()]
        other = [p[candidate][field] for p in pairs.values()]
        groups[field] = {
            "baseline_median": median(originals), f"{label}_median": median(other),
            "ratio_of_medians": median(originals) / median(other),
            "median_paired_speedup": median(a / b for a, b in zip(originals, other)),
            f"{label}_faster_pairs": sum(b < a for a, b in zip(originals, other)),
            "pairs": len(pairs), "baseline_range": [min(originals), max(originals)],
            f"{label}_range": [min(other), max(other)],
        }
    return groups


def run_batch(out, index, args, variants, hz, log):
    result, cases, canonical = {}, {}, {}
    # Each seed sees both orders within this batch. Reverse the initial order in batch B.
    seeds = [args.seed + index * (args.pairs // 2 + 1) + i for i in range(args.pairs // 2 + 1)]
    for seed in seeds:
        inputs, reference = make_case(seed, args.full_attention)
        cases[seed] = inputs, reference
        np.save(out / f"ort-{seed}.npy", reference, allow_pickle=False)
    for mode in (("plain", "timed") if index == 0 else ("timed", "plain")):
        builds, addresses = variants[mode]
        rows = []
        names = tuple(builds)
        for rep in range(args.pairs + 1):
            offset = max(0, rep - 1)
            seed = seeds[1 + offset % (args.pairs // 2)] if rep else seeds[0]
            order = list(names)
            if rep and (offset % (args.pairs // 2) + offset // (args.pairs // 2) + index) % 2:
                order.reverse()
            inputs, reference = cases[seed]
            for position, name in enumerate(order):
                folder = out / mode / f"round-{rep:02d}" / name
                nbytes = 96 if args.full_attention else 64
                timer = capture_step_ticks(addresses[name], nbytes) if addresses else nullcontext()
                with timer:
                    row = measure(str(seed), name, "2x4", "native", "measure" if rep else "warmup",
                                  rep, (inputs, reference, builds), folder, canonical, require_rss=False)
                observed = np.fromfile(folder / "output.bin", dtype="<f4")
                split = 16 * 128 if args.full_attention else 16 * LENGTHS[0]
                pieces = np.split(np.abs(observed.astype("f8") - reference.ravel().astype("f8")), [split])
                row.update(batch=index + 1, mode=mode, seed=seed, order=position,
                           rhs_source="runtime", errors_by_step=[float(p.max()) for p in pieces],
                           sequence_uart_ms=row["forward_s"] * 1000)
                if addresses:
                    row.update(decode_ticks(folder, hz, args.full_attention, separate_stages=name != HEAD_GROUPS))
                    if row["step_255_ms"] + row["step_256_ms"] > row["sequence_uart_ms"] + 1:
                        raise ValueError(f"Guest step intervals exceeded wall time: {folder}")
                rows.append(row)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
        regions = ("operator", "step")
        if args.full_attention and HEAD_GROUPS not in names:
            regions = ("qk", "scale_softmax", "pv", *regions)
        fields = (["sequence_uart_ms"] if mode == "plain" else
                  [f"{part}_{length}_ms" for length in LENGTHS for part in regions])
        result[mode] = {
            "max_abs_vs_ort_by_step": [max(r["errors_by_step"][i] for r in rows) for i in range(2)],
            "measured_pairs": args.pairs, "warmup_pairs": 1, "summary": summarize(rows, fields, names),
            "workspace_bytes": {name: n for name, (_, n) in builds.items()},
        }
        print(f"Batch {index + 1} {mode}: {json.dumps(result[mode]['summary'])}", flush=True)
    return {"warmup_seed": seeds[0], "measured_seeds": seeds[1:], **result}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", required=True, type=Path)
    cli.add_argument("--seed", type=int, default=7100)
    cli.add_argument("--full-attention", action="store_true",
                     help="Append K and V, scale QK scores, run Softmax and return P @ V at each step")
    comparison = cli.add_mutually_exclusive_group()
    comparison.add_argument("--compare-transpose-to-head-groups", action="store_true",
                            help="Compare the original K transpose to direct K with one dispatch per step by K/V head")
    comparison.add_argument("--compare-head-groups", action="store_true",
                            help="Compare separate operators to one parallel dispatch by KV head, keeping the same 2x4 P @ V")
    cli.add_argument("--pairs", type=int, default=12, help="Even number of measured pairs in each batch and mode")
    args = cli.parse_args()
    if args.compare_transpose_to_head_groups:
        names = (BASELINE, HEAD_GROUPS)
    elif args.compare_head_groups:
        names = (DIRECT, HEAD_GROUPS)
    else:
        names = (BASELINE, DIRECT)
    if HEAD_GROUPS in names:
        args.full_attention = True
    if args.pairs < 2 or args.pairs % 2 or args.seed < 0:
        cli.error("Use a nonnegative seed and a positive even pair count")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 4:
        cli.error("Four available logical CPUs are required")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    tools = discover_toolchain()
    hz = read_timebase(tools.qemu, out / "clock")
    variants = {mode: build_variants(out / "elf" / mode, tools, mode == "timed", args.full_attention, names)
                for mode in ("plain", "timed")}
    result = {
        "baseline": names[0], "candidate": names[1],
        "cores": 4, "affinity": affinity, "dtype": "float32", "workspace_layout": "none",
        "qemu": subprocess.check_output([tools.qemu, "--version"], text=True).splitlines()[0],
        "ort": ort.__version__, "timebase_hz": hz, "positions": LENGTHS,
        "reference": "ORT CPU MatMul, duplicate each of eight K heads for the 16 query heads",
        "timing": "Host sequence_uart_ms spans QEMU launch through the two-step completion UART frame. "
                  "Guest operator times score calculation and finite checks. Guest step includes append, input checks and score copy. "
                  "Absolute clocks and timing modes are separate; each unit is milliseconds.",
        "batches": {},
    }
    if args.full_attention:
        result.update(
            full_attention=True,
            reference="ORT CPU MatMul(Q, K^T) -> Mul(1/sqrt(128)) -> Softmax(axis=-1) -> MatMul(P, V), "
                      "explicitly repeating K and V for each pair of query heads; all 4096 output components.",
            timing="Plain ELF: host QEMU launch through the final UART frame for both steps including cache "
                   "initialization. Instrumented ELF: operator = QK through P @ V; scale_softmax includes both "
                   "scaling and Softmax; step adds Q checks, appending and checking K/V, and saving the context. "
                   "Includes only valid positions; all timings are elapsed milliseconds.",
        )
    if HEAD_GROUPS in names:
        result.update(value_tile="2x4",
                      timing="Plain ELF: host QEMU launch through UART completion of both steps. "
                             "Timed ELF: each operator spans QK, scaling, Softmax and P @ V, including finite checks. "
                             "Each step also checks Q, appends K/V and saves the context. "
                             "Groups progress independently, so only the combined operator time is comparable.")
    with (out / "runs.jsonl").open("x", encoding="utf-8") as log:
        for index in range(2):
            directory = out / f"batch-{index + 1}"
            directory.mkdir()
            result["batches"][str(index + 1)] = run_batch(directory, index, args, variants, hz, log)
            (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
