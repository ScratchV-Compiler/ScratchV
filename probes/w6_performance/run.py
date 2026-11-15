"""Measure two fixed FP32 MatMuls with the existing four-core RV64 runner (Linux)."""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import threading
import time

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

from probes.w3_qwen3_full.resources import process_memory
from scratchv.backend.tensor_c_codegen import TensorCCodegen
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import Value
from scratchv.runtime.riscv_external import FRAME, MAGIC, build_external, run_external
from scratchv.runtime.riscv_tensor import discover_toolchain
from scratchv.runtime.weight_bundle import write_weight_bundle


CASES = {
    "dense": ((256, 1024), (1024, 1024)),
    "attention_scores": ((16, 256, 128), (16, 128, 256)),
}
TILES = ("1x4", "2x4")
# These are chosen before timing; additional runs are diagnostics, not an automatic winner.
REQUIRED_VARIANTS = {"dense": "2x4-column-panel", "attention_scores": "4x4-checked-parallel-copy"}


def configurations(case, compare_candidates):
    if not compare_candidates:
        return [(tile, tile, "original") for tile in TILES]
    return [
        ("1x4-original", "1x4", "original"),
        ("2x4-original", "2x4", "original"),
        ("1x8-original", "1x8", "original"),
        ((REQUIRED_VARIANTS[case], "2x4", "column-panel") if case == "dense"
         else (REQUIRED_VARIANTS[case], "4x4", "runtime-panel")),
    ]

# Compare both inner loops against the same two dynamic inputs.
ATTENTION_LOOPS = [
    ("1x4-original", "1x4", "original"),
    ("4x4-runtime-panel", "4x4", "runtime-panel"),
    ("4x4-k8", "4x4", "runtime-panel"),
    ("4x4-head-panels", "4x4", "runtime-panel"),
]
CLAIM_VARIANTS = (
    ("4x4-checked-claim4", "4x4", "runtime-panel"),
    ("4x4-checked-parallel-copy", "4x4", "runtime-panel"),
    ("4x4-checked-claim16", "4x4", "runtime-panel"),
)
CLAIM_BLOCKS = {variant[0]: blocks for variant, blocks in zip(CLAIM_VARIANTS, (4, 8, 16))}
# Each entry specifies the row tile and whether the destination rows are cached.
WIDE_SCALAR_ROWS = {"3x8": 3, "2x8": 2}
Q_PAIR_VARIANTS = {"5x4-q-unroll2": False, "5x4-q-pair": True}
SCALAR_CANDIDATES = {"5x4": (5, False), "4x4-cached": (4, True), "5x4-cached": (5, True),
                     "4x4-claim-panel": (4, False), "5x4-cached-claim-panel": (5, True),
                     "6x4-cached": (6, True), "5x4-bitwise": (5, True),
                     "5x4-first-pack": (5, True), "5x4-preboxed-k": (5, True), "5x4-paired-k": (5, True),
                     "5x4-paired-bitwise": (5, True), "5x4-static": (5, True),
                     "5x4-bounded-output": (5, True), "5x4-normal-output": (5, True),
                     **{name: (4, False) for name in WIDE_SCALAR_ROWS},
                     **{name: (5, True) for name in Q_PAIR_VARIANTS}}
ROOT = Path(__file__).resolve().parents[2]


def ort_matmul(case, x, weight):
    left, right = x.shape, weight.shape
    shape = (*left[:-1], right[-1])
    def tensor(name, dims):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)

    model = helper.make_model(helper.make_graph(
        [helper.make_node("MatMul", ["x", "weight"], ["y"])], case,
        [tensor("x", left), tensor("weight", right)], [tensor("y", shape)]),
        opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    return session.run(None, {"x": x, "weight": weight})[0]


def prepare_case(case, directory, tools, *, compare_candidates=False, variants=None):
    left, right = CASES[case]
    rng = np.random.default_rng(4108)
    x = rng.standard_normal(left).astype(np.float32)
    weight = rng.standard_normal(right).astype(np.float32)
    reference = ort_matmul(case, x, weight)

    ir = IRBuilder()
    lhs, rhs = Value("x", shape=left), Value("weight", shape=right)
    dynamic_rhs = compare_candidates and case == "attention_scores"
    inputs = {"x": x, "weight": weight} if dynamic_rhs else {"x": x}
    bindings = {} if dynamic_rhs else {"weight": weight}
    ir.new_function("main", [lhs, rhs] if dynamic_rhs else [lhs])
    ir.new_block("entry")
    if not dynamic_rhs:
        ir.program.global_values.append(rhs)
    ir.ret(ir.matmul(lhs, rhs))
    directory.mkdir()
    np.save(directory / "ort.npy", reference, allow_pickle=False)
    build = {}
    for variant, tile, layout in (variants if variants is not None else configurations(case, compare_candidates)):
        kernel_options = {}
        if layout in ("runtime-panel", "rvv-direct"):
            from probes.w6_performance.attention_panel import RuntimeScorePanel
            # The underlying graph ABI remains original. This probe replaces only its MatMul kernel.
            generator, codegen_tile, storage_layout = RuntimeScorePanel, "2x4", "original"
            if variant in SCALAR_CANDIDATES:
                rows, cached = SCALAR_CANDIDATES[variant]
                kernel_options.update(row_tile=rows, cache_output=cached, claim_blocks=8,
                                      loop_variant="shared-claim-panels" if variant.endswith("-claim-panel") else "shared-checked")
            elif variant in ("4x4-k8", "4x4-head-panels", "4x4-shared-blocks"):
                kernel_options["loop_variant"] = variant.removeprefix("4x4-")
            elif variant in ("4x4-checked-publish64", *CLAIM_BLOCKS):
                kernel_options["loop_variant"] = "shared-checked"
                if variant in CLAIM_BLOCKS:
                    kernel_options["claim_blocks"] = CLAIM_BLOCKS[variant]
            elif variant == "rvv-publish64":
                kernel_options["loop_variant"] = "rvv"
            elif variant == "4x4-shared-publish64":
                kernel_options["loop_variant"] = "shared-blocks"
        else:
            generator, codegen_tile, storage_layout = TensorCCodegen, tile, layout
        artifact = generator(ir.program, bindings, constant_storage="external", kernel_calls=True,
                             matmul_policy="blocked_fma", matmul_tile=codegen_tile,
                             matmul_rhs_layout=storage_layout, qemu_workspace_layout="none", **kernel_options).generate()
        parallel_copy = variant in CLAIM_BLOCKS or variant in SCALAR_CANDIDATES
        if parallel_copy or variant in ("1x4-publish64", "4x4-shared-publish64", "rvv-publish64", "4x4-checked-publish64"):
            from probes.w6_performance.output_copy import with_block_output
            artifact = with_block_output(artifact, parallel=parallel_copy)
        if variant == "5x4-first-pack":
            from probes.w6_performance.attention_first_pack import with_first_pass_packing
            artifact = with_first_pass_packing(artifact)
        elif variant == "5x4-preboxed-k":
            from probes.w6_performance.attention_preboxed_k import with_preboxed_k
            artifact = with_preboxed_k(artifact)
        if variant in ("5x4-bitwise", "5x4-paired-bitwise", "5x4-static", "5x4-bounded-output",
                       "5x4-normal-output", *Q_PAIR_VARIANTS):
            from probes.w6_performance.attention_scalar_epilogue import with_bitwise_writeback
            artifact = with_bitwise_writeback(artifact)
        if variant == "5x4-static":
            from probes.w6_performance.attention_static_schedule import with_static_schedule
            artifact = with_static_schedule(artifact)
        if variant in ("5x4-bounded-output", *Q_PAIR_VARIANTS):
            from probes.w6_performance.attention_bounded_output import with_bounded_output
            artifact = with_bounded_output(artifact)
        if variant in Q_PAIR_VARIANTS:
            from probes.w6_performance.attention_paired_q import with_paired_q
            artifact = with_paired_q(artifact, packed_load=Q_PAIR_VARIANTS[variant])
        if variant == "5x4-normal-output":
            from probes.w6_performance.attention_lattice_output import with_normal_output
            artifact = with_normal_output(artifact)
        if variant in WIDE_SCALAR_ROWS:
            from probes.w6_performance.attention_wide_scalar import with_wide_scalar
            artifact = with_wide_scalar(artifact, rows=WIDE_SCALAR_ROWS[variant])
        if variant in ("5x4-paired-k", "5x4-paired-bitwise"):
            from probes.w6_performance.attention_preboxed_k import with_paired_k
            artifact = with_paired_k(artifact)
        bundle = directory / ("weights-column-panel" if storage_layout == "column-panel" else "weights")
        if not bundle.exists():
            write_weight_bundle(artifact.external_weights, artifact.external_initializers, bundle)
        exe = build_external(artifact, bundle, directory / variant / "build", tools, cores=4)
        if (exe.cores, exe.evidence["matmul_tile"], exe.evidence["qemu_workspace_layout"],
                exe.evidence["matmul_rhs_layout"]) != (4, codegen_tile, "none", storage_layout):
            raise ValueError("Unexpected build configuration")
        if bool(artifact.matmul_rhs_panels) != (layout == "column-panel"):
            raise ValueError("RHS panel selection did not match this experiment")
        if dynamic_rhs and (exe.weights or {i.name for i in exe.inputs} != set(inputs)):
            raise ValueError("Attention requires two runtime inputs and no weight bundle tensors")
        build[variant] = (exe, artifact.workspace_bytes)
    return inputs, reference, build


def collect_pair(stop, observed, *, uart_path=None):
    """Sample the controller and its QEMU child in the same /proc scan."""
    parent = os.getpid()
    children = Path(f"/proc/{parent}/task/{parent}/children")
    first_qemu_time = None
    while not stop.is_set():
        try:
            for pid in children.read_text().split():
                if not (Path("/proc") / pid / "comm").read_text().startswith("qemu-system"):
                    continue
                if uart_path is not None and (Path("/proc") / pid / "cwd").readlink() != uart_path.parent:
                    continue  # Exclude a paused QEMU used to inspect the boot layout.
                guest = process_memory(int(pid))["rss_bytes"]
                controller = process_memory(parent)["rss_bytes"]
                observed["samples"] += 1
                sampled_at = time.perf_counter()
                if first_qemu_time is None:
                    first_qemu_time = sampled_at
                combined = controller + guest
                if combined > observed["combined_peak_bytes"]:
                    observed.update(combined_peak_bytes=combined, controller_bytes_at_peak=controller,
                                    qemu_bytes_at_peak=guest,
                                    peak_seconds_since_first_qemu_sample=sampled_at - first_qemu_time)
                    if uart_path is not None:
                        try:
                            observed["uart_bytes_at_peak"] = uart_path.stat().st_size
                        except FileNotFoundError:
                            observed["uart_bytes_at_peak"] = 0
        except (FileNotFoundError, ProcessLookupError):
            observed["exit_races"] += 1
        except (OSError, ValueError):
            observed["read_errors"] += 1
        stop.wait(0.05)


def measure(case, variant, tile, layout, phase, rep, prepared, folder, baseline, *, require_rss=True):
    inputs, reference, builds = prepared
    exe, workspace = builds[variant]
    observed = dict(combined_peak_bytes=0, samples=0, read_errors=0, exit_races=0)
    stop = threading.Event()
    sampler = threading.Thread(target=collect_pair, args=(stop, observed), daemon=True)
    sampler.start()
    before = time.perf_counter(), time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    try:
        output, report = run_external(exe, inputs, folder, timeout=60)
    finally:
        after = time.perf_counter(), time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
        stop.set()
        sampler.join()
    clock_diff = (after[0] - before[0]) - (after[1] - before[1])
    if not report["passed"] or report["timing_mode"] != "event" or abs(clock_diff) > 0.02:
        raise RuntimeError(f"Invalid event timing or guest status at {folder}")
    if output.shape != reference.shape or output.dtype != np.float32 or not np.isfinite(output).all():
        raise ValueError("Invalid result shape, type or values")
    error = float(np.max(np.abs(output.astype(np.float64) - reference.astype(np.float64))))
    if not np.isfinite(error) or error >= 1e-3:
        raise ValueError(f"{folder}: max_abs_vs_ort={error}")
    if case not in baseline:
        baseline[case] = output.copy()
    if not np.array_equal(output.view(np.uint32), baseline[case].view(np.uint32)):
        raise ValueError(f"{folder}: FP32 bits differ between runs")
    qemu_peak = report["sampled_peak_qemu_rss_bytes"] or None
    if require_rss and (not observed["samples"] or not qemu_peak):
        raise RuntimeError(f"No RSS samples at {folder}")
    if not observed["samples"]:
        for key in ("combined_peak_bytes", "controller_bytes_at_peak", "guest_bytes_at_peak"):
            observed[key] = None
    return dict(case=case, tile=tile, variant=variant, rhs_layout=layout, phase=phase, round=rep,
                rhs_source="runtime" if "weight" in inputs else "constant", forward_s=report["forward_wall_s"],
                qemu_s=report["qemu_wall_s"], qemu_peak_bytes=qemu_peak,
                qemu_rss_samples=report["qemu_rss_observed_samples"], qemu_rss_errors=report["qemu_rss_sample_errors"],
                workspace_bytes=workspace, max_abs_vs_ort=error, bitwise_equal=True,
                ort_bit_mismatches=int(np.count_nonzero(output.view(np.uint32) != reference.view(np.uint32))),
                clock_delta_s=clock_diff, **observed)


def check_tail_overflow(prepared, candidates, directory):
    inputs, _, builds = prepared
    query, key = inputs["x"].copy(), inputs["weight"].copy()
    query[-1, -1] = np.finfo(np.float32).max
    key[-1, :, -1] = 2
    errors = {}
    for name in candidates:
        exe, _ = builds[name]
        folder = directory / name / "tail-overflow"
        try:
            run_external(exe, {"x": query, "weight": key}, folder, timeout=60)
        except RuntimeError as error:
            if str(error) != "RV64 guest failed with status 0x00000002":
                raise
        else:
            raise ValueError(f"{name}: expected numerical failure on row 255")
        report = json.loads((folder / "run.json").read_text())
        frame = FRAME.unpack((folder / "uart.bin").read_bytes())
        if (frame != (MAGIC, 2, exe.output.nbytes) or report["passed"] or not report["qemu_reaped"]
                or report.get("cleanup_errors") or (folder / "output.bin").exists()):
            raise ValueError(f"{name}: unexpected error frame, output or QEMU cleanup")
        errors[name] = dict(guest_status=2, row=255, qemu_reaped=True, output_dumped=False)
    return errors


def check_fp32_values(prepared, candidates, directory):
    inputs, _, builds = prepared
    query, key = np.zeros_like(inputs['x']), np.zeros_like(inputs['weight'])
    # One final product on every row. Halving the least subnormal exercises
    # RNE zero canonicalization; multiplying the largest finite value by 1 is exact.
    query[:, :, -1] = np.resize(np.array([1, 0.5, -1, -0.5], dtype=np.float32), query.shape[1])
    patterns = np.array([0, 0x80000000, 1, 0x80000001, 0x7f7fffff, 0xff7fffff, 0x3f800000,
                         0xbf800000], dtype=np.uint32).view(np.float32)
    key[:, -1, :] = np.resize(patterns, key.shape[-1])
    reference = ort_matmul('attention_scores', query, key)
    exact = query[:, :, -1, None] * key[:, None, -1, :]
    exact[exact == 0] = np.float32(0)
    if not np.array_equal(reference, exact):
        raise ValueError('ORT single-product reference differs numerically on FP32 edge cases')
    # ORT MatMul can return -0 for an underflowing final FMA. The score kernel's
    # final addition of +0 maps that result to +0 under RNE. Test both contracts.
    signed_zero_differences = int(np.count_nonzero(reference.view(np.uint32) != exact.view(np.uint32)))
    np.save(directory / 'fp32-values-ort.npy', reference, allow_pickle=False)
    control = '4x4-checked-parallel-copy'
    verified = [control, *candidates]
    for name in verified:
        exe, _ = builds[name]
        output, report = run_external(exe, {'x': query, 'weight': key}, directory / name / 'fp32-values', timeout=60)
        if (not report['passed'] or not np.array_equal(output, reference)
                or not np.array_equal(output.view(np.uint32), exact.view(np.uint32))):
            raise ValueError(f'{name}: FP32 edge cases differ from ORT numerically or the expected score bits')
    return dict(variants=verified, elements=reference.size, max_abs_vs_ort=0.0,
                sign_only_zero_differences_vs_ort=signed_zero_differences, bit_mismatches_vs_expected=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--attention-inner-loop", nargs="?", const="tiles", choices=("tiles", "work-sharing", "output-copy", "rvv", "data-movement", "claim-granularity"),
                       help="Compare dynamic Attention kernels or block scheduling without instrumentation")
    modes.add_argument("--compare-candidates", action="store_true",
                        help="Measure three production tiles and the preselected Dense/Attention alternatives")
    modes.add_argument("--attention-scalar", nargs="+", choices=tuple(SCALAR_CANDIDATES),
                       help="Compare selected scalar Attention candidates against the original and current kernels")
    args = parser.parse_args()
    if args.attention_scalar and len(set(args.attention_scalar)) != len(args.attention_scalar):
        parser.error("Specify each scalar candidate once")
    attention_only = bool(args.attention_inner_loop or args.attention_scalar)
    dest = args.output_dir.resolve()
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 4:
        parser.error("Four available logical CPUs are required")
    dest.mkdir(parents=True, exist_ok=False)
    tools = discover_toolchain()
    if args.attention_scalar:
        group = [ATTENTION_LOOPS[0], CLAIM_VARIANTS[1]]
        for name in args.attention_scalar:
            tile = name if name in WIDE_SCALAR_ROWS else f"{SCALAR_CANDIDATES[name][0]}x4"
            group.append((name, tile, "runtime-panel"))
        variants = {"attention_scores": group}
    elif args.attention_inner_loop:
        if args.attention_inner_loop == "rvv":
            from probes.w6_performance.rvv_platform import RVV_CPU, rvv_toolchain
            tools = rvv_toolchain(tools, dest)
        shared = ("4x4-shared-blocks", "4x4", "runtime-panel")
        groups = {
            "tiles": ATTENTION_LOOPS,
            "claim-granularity": CLAIM_VARIANTS,
            "data-movement": [
                ATTENTION_LOOPS[0], ("4x4-shared-publish64", "4x4", "runtime-panel"),
                ("4x4-checked-publish64", "4x4", "runtime-panel"),
                ("4x4-checked-parallel-copy", "4x4", "runtime-panel"),
            ],
            "rvv": [
                ATTENTION_LOOPS[0], ("4x4-shared-publish64", "4x4", "runtime-panel"),
                ("rvv-publish64", "RVV", "rvv-direct"),
            ],
            "work-sharing": [*ATTENTION_LOOPS[:2], shared],
            "output-copy": [
                ATTENTION_LOOPS[0], ("1x4-publish64", "1x4", "original"),
                shared, ("4x4-shared-publish64", "4x4", "runtime-panel"),
            ],
        }
        variants = {"attention_scores": groups[args.attention_inner_loop]}
    else:
        variants = {case: configurations(case, args.compare_candidates) for case in CASES}
    dynamic = args.compare_candidates or attention_only
    prepared = {case: prepare_case(case, dest / case, tools, compare_candidates=dynamic, variants=group)
                for case, group in variants.items()}
    rounds = 2 * len(variants["attention_scores"]) if attention_only else 3
    baseline, rows = {}, []
    with (dest / "runs.jsonl").open("x", encoding="utf-8") as log:
        for rep in range(rounds + 1):
            phase = "warmup" if rep == 0 else "measure"
            for case, group in variants.items():
                order = group
                if attention_only and rep:
                    # Put each implementation in every position twice.
                    start = (rep - 1) % len(group)
                    order = group[start:] + group[:start]
                    if rep > len(group):
                        order = order[::-1]
                elif rep and rep % 2 == 0:
                    order = group[::-1]
                for variant, tile, layout in order:
                    row = measure(case, variant, tile, layout, phase, rep, prepared[case],
                                  dest / case / variant / f"run-{rep}", baseline)
                    if (args.attention_scalar or args.attention_inner_loop in ("rvv", "data-movement", "claim-granularity")) and row["ort_bit_mismatches"]:
                        raise ValueError(f"{variant}: output differs from ORT at the bit level")
                    if args.attention_inner_loop == "claim-granularity":
                        row["claim_blocks"] = CLAIM_BLOCKS[variant]
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    rows.append(row)
                    print(case, variant, phase, rep, round(row["forward_s"], 4), flush=True)
    summary = []
    for case in variants:
        for variant, tile, layout in variants[case]:
            group = [r for r in rows if r["case"] == case and r["variant"] == variant and r["phase"] == "measure"]
            summary.append(dict(case=case, tile=tile, variant=variant, rhs_layout=layout, rhs_source=group[0]["rhs_source"],
                **{key: statistics.median(r[key] for r in group)
                   for key in ("forward_s", "qemu_s", "qemu_peak_bytes", "combined_peak_bytes")},
                max_abs_vs_ort=max(r["max_abs_vs_ort"] for r in group),
                workspace_bytes=group[0]["workspace_bytes"]))
    result = dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,text=True).strip(),
                  numpy=np.__version__, ort=ort.__version__, onnx=onnx.__version__, python=platform.python_version(),
                  qemu=subprocess.check_output([tools.qemu,"--version"],text=True).splitlines()[0],
                  affinity=affinity, seed=4108, cores=4, dtype="float32", matmul="blocked_fma",
                  rhs_layout="per-variant" if dynamic else "original", workspace_layout="none",
                  shapes={case: CASES[case] for case in variants}, summary=summary, runs=rows)
    if attention_only:
        result["metric"] = "median forward_s: QEMU start to UART; no guest instrumentation"
        if args.attention_inner_loop == "claim-granularity":
            result["baseline_variant"] = "4x4-checked-parallel-copy"
            result["claim_blocks"] = CLAIM_BLOCKS
        else:
            result["baseline_variant"] = "1x4-original"
            result["target_speedup"] = 1.5
    if args.attention_scalar:
        result["current_variant"] = "4x4-checked-parallel-copy"
        result["tail_overflow"] = check_tail_overflow(prepared["attention_scores"], args.attention_scalar,
                                                     dest / "attention_scores")
        result["fp32_values_verified"] = check_fp32_values(prepared["attention_scores"], args.attention_scalar,
                                                         dest / "attention_scores")
    if args.attention_inner_loop == "rvv":
        result["qemu_cpu"] = RVV_CPU
    if args.compare_candidates:
        result["required_variants"] = REQUIRED_VARIANTS
    (dest / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
