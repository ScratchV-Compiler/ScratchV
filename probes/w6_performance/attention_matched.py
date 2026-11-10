"""Compare scalar and RVV 4×4 or 4×8 QK tiles with the same inputs and four-core schedule."""
import argparse
import json
import os
from pathlib import Path
from statistics import median
import subprocess

import numpy as np

from probes.w6_performance.attention_matched_kernel import matched_kernel_source
from probes.w6_performance.attention_profile import ARRAY, TIMER, capture_counters
from probes.w6_performance.lm_head_timing import read_timebase
from probes.w6_performance.run import ROOT, ort_matmul
from probes.w6_performance.rvv_platform import RVV_CPU, rvv_toolchain
from scratchv.backend.tensor_c_codegen import TensorCArtifact, TensorSpec
from scratchv.ir.types import DataType
from scratchv.runtime.riscv_external import build_external, run_external
from scratchv.runtime.riscv_tensor import discover_toolchain
from scratchv.runtime.weight_bundle import write_weight_bundle


TILES = 256
REPEATS = 16
PAIRS = 8
MODES = ("scalar", "rvv")


GUEST = r"""
typedef void (*w6_tile_fn)(const float *, const float *, float *);
extern void sv_parallel_for(size_t, void (*)(void *, size_t, size_t), void *);
struct w6_tiles {
    const float *q, *k;
    float *out;
    w6_tile_fn fn;
    size_t repeats;
};

static void w6_run_tiles(void *opaque, size_t begin, size_t end) {
    struct w6_tiles *ctx = opaque;
    size_t core;
    __asm__ volatile("csrr %0, mhartid" : "=r"(core));
    if (ctx->repeats == 1) {
        size_t vl;
        w6_matched_enable();
        __asm__ volatile(".option push\n.option arch,+v\n"
                         "vsetivli %0,%1,e32,m1,ta,ma\n.option pop"
                         : "=r"(vl) : "i"(W6_COLUMNS) : "memory");
        w6_ticks[core][5] = vl;
    }
    w6_ticks[core][0] = w6_now();
    for (size_t rep = 0; rep < ctx->repeats; ++rep) {
        for (size_t tile = begin; tile < end; ++tile) {
            ctx->fn(ctx->q + tile * 4 * 128, ctx->k + tile * 128 * W6_COLUMNS,
                    ctx->out + (rep * W6_TILES + tile) * 4 * W6_COLUMNS);
        }
    }
    w6_ticks[core][1] = w6_now();
    w6_ticks[core][2] = begin;
    w6_ticks[core][3] = end;
    w6_ticks[core][4] = ctx->repeats;
    size_t vl;
    __asm__ volatile(".option push\n.option arch,+v\ncsrr %0,vl\n.option pop"
                     : "=r"(vl) :: "memory");
    w6_ticks[core][6] = vl;
}

int scratchv_run_external(const void *const *in, const void *const *weights,
                          void *workspace, size_t workspace_size, void *out) {
    const int32_t mode = *(const int32_t *)in[2];
    if (mode != 0 && mode != 1) return 1;
    struct w6_tiles ctx = {in[0], in[1], out, mode ? w6_matched_rvv : w6_matched_scalar, 1};
    sv_parallel_for(W6_TILES, w6_run_tiles, &ctx);
    ctx.repeats = W6_REPEATS;
    w6_ticks[4][0] = w6_now();
    sv_parallel_for(W6_TILES, w6_run_tiles, &ctx);
    w6_ticks[4][1] = w6_now();
    w6_ticks[4][2] = mode;
    return 0;
}
"""


def prepare_inputs(columns=4):
    rng = np.random.default_rng(4108)
    query = rng.standard_normal((16, 256, 128)).astype(np.float32)
    key = rng.standard_normal((16, 128, 256)).astype(np.float32)
    # Keep whole depth-128 dot products. Each head contributes 16 disjoint
    # row blocks and the same packed arrays go to both implementations.
    q = np.empty((TILES, 4, 128), dtype=np.float32)
    k = np.empty((TILES, 128, columns), dtype=np.float32)
    for tile in range(TILES):
        head, group = tile % 16, tile // 16
        row = 4 * ((4 * group + head) % 64)
        column = columns * ((group + 3 * head) % (256 // columns))
        q[tile] = query[head, row:row + 4]
        k[tile] = key[head, :, column:column + columns]
    reference = ort_matmul("matched_attention_tiles", q, k)
    return q, k, reference


def build(directory, tools, columns=4):
    source = ("#include <stddef.h>\n#include <stdint.h>\n" + ARRAY + TIMER
              + matched_kernel_source(columns) + "\n"
              + f"#define W6_TILES {TILES}\n#define W6_REPEATS {REPEATS}\n#define W6_COLUMNS {columns}\n"
              + GUEST)
    artifact = TensorCArtifact(source=source, inputs=(
        TensorSpec("q", DataType.FLOAT32, (TILES, 4, 128)),
        TensorSpec("k", DataType.FLOAT32, (TILES, 128, columns)),
        TensorSpec("mode", DataType.INT32, (1,))),
        output=TensorSpec("scores", DataType.FLOAT32, (REPEATS, TILES, 4, columns)),
        workspace_bytes=0, constant_bytes=0, function_name="scratchv_run_external",
        constant_storage="external", matmul_policy="blocked_fma", qemu_workspace_layout="none")
    bundle = directory / "weights"
    write_weight_bundle((), (), bundle)
    exe = build_external(artifact, bundle, directory / "build", tools, cores=4)
    if exe.cores != 4 or exe.weights or exe.evidence["qemu_workspace_layout"] != "none":
        raise ValueError("Unexpected ELF configuration for the four-core short loop")
    return exe


def actual_configuration(folder, report, exe):
    actual = (folder / "qemu-argv.txt").read_text().splitlines()
    expected = list(report["command"])
    expected[0] = actual[0]
    expected[expected.index("-cpu") + 1] = RVV_CPU
    if (actual != expected or actual[actual.index("-smp") + 1] != "4" or "-icount" in actual
            or actual[actual.index("-kernel") + 1] != str(exe.elf_path)):
        raise ValueError(f"{folder}: unexpected effective QEMU command")
    # run_external allocates a new localhost port for QMP in every process.
    actual[actual.index("-qmp") + 1] = "<QMP endpoint>"
    return actual


def measure(exe, folder, mode, q, k, reference, hz):
    mode_index = MODES.index(mode)
    inputs = {"q": q, "k": k, "mode": np.array([mode_index], dtype=np.int32)}
    with capture_counters(exe):
        output, report = run_external(exe, inputs, folder, timeout=60)
    if (not report["passed"] or not report["qemu_reaped"] or report.get("cleanup_errors")
            or report["timing_mode"] != "event" or report["qemu_workspace_layout"] != "none"):
        raise ValueError(f"{folder}: incomplete UART run or cleanup")
    expected_shape = (REPEATS, *reference.shape)
    if output.shape != expected_shape or output.dtype != np.float32 or not np.isfinite(output).all():
        raise ValueError(f"{folder}: invalid output array")
    mismatches = int(np.count_nonzero(output.view(np.uint32) != reference.view(np.uint32)))
    error = float(np.max(np.abs(output.astype(np.float64) - reference.astype(np.float64))))
    if mismatches or error != 0.0:
        raise ValueError(f"{folder}: {mismatches} differing FP32 scores, max_abs_vs_ort={error}")
    ticks = np.fromfile(folder / "per-core.bin", dtype="<u8").reshape(5, 8).tolist()
    start, stop, observed_mode = ticks[4][:3]
    if (observed_mode != mode_index or not 0 < start < stop
            or (stop - start) / hz > report["forward_wall_s"]):
        raise ValueError(f"{folder}: invalid guest-clock interval or runtime mode")
    columns = reference.shape[-1]
    for core, (begin, end, first, limit, repetitions, warmup_vl, final_vl, _) in enumerate(ticks[:4]):
        if (not start <= begin < end <= stop or first != core * TILES // 4
                or limit != (core + 1) * TILES // 4 or repetitions != REPEATS):
            raise ValueError(f"{folder}: invalid interval or work assignment for core {core}")
        if (warmup_vl, final_vl) != (columns, columns):
            raise ValueError(f"{folder}: core {core} requested {columns} FP32 lanes, "
                             f"observed {warmup_vl} at warmup and {final_vl} after timed calls")
    seconds = (stop - start) / hz
    row = dict(mode=mode, ticks=ticks, dispatch_s=seconds, ns_per_score=seconds * 1e9 / output.size,
               uart_wall_s=report["forward_wall_s"], qemu_wall_s=report["qemu_wall_s"],
               max_abs_vs_ort=error, ort_bit_mismatches=mismatches, verified_scores=int(output.size))
    return row, actual_configuration(folder, report, exe)


def summarize(rows):
    stats = {}
    for mode in MODES:
        trials = [r for r in rows if r["mode"] == mode and r["round"] > 0]
        costs = [r["ns_per_score"] for r in trials]
        stats[mode] = dict(ns_per_score=median(costs), range_ns=[min(costs), max(costs)],
                           median_dispatch_ms=median(r["dispatch_s"] * 1000 for r in trials),
                           median_uart_ms=median(r["uart_wall_s"] * 1000 for r in trials))
    deltas = []
    for rep in range(1, PAIRS + 1):
        pair = {r["mode"]: r["ns_per_score"] for r in rows if r["round"] == rep}
        deltas.append(pair["scalar"] - pair["rvv"])
    ratio = stats["scalar"]["ns_per_score"] / stats["rvv"]["ns_per_score"]
    wins = sum(x > 0 for x in deltas)
    return dict(modes=stats, scalar_over_rvv=ratio, paired_scalar_minus_rvv_ns=deltas,
                median_paired_gain_ns=median(deltas), rvv_faster_pairs=wins,
                rvv_advantage=bool(ratio > 1 and wins > PAIRS // 2 and median(deltas) > 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--columns", choices=(4, 8), default=4, type=int, help="FP32 output columns per tile")
    args = parser.parse_args()
    if len(os.sched_getaffinity(0)) < 4:
        parser.error("Four available CPUs are required")
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    tools = rvv_toolchain(discover_toolchain(), directory)
    hz = read_timebase(tools.qemu, directory / "timer")
    columns = args.columns
    exe = build(directory, tools, columns)
    q, k, ref = prepare_inputs(columns)
    np.save(directory / "ort.npy", ref, allow_pickle=False)
    batches, configuration = [], None
    with (directory / "runs.jsonl").open("x", encoding="utf-8") as journal:
        for batch in (1, 2):
            rows = []
            for rep in range(PAIRS + 1):
                order = MODES if (rep + batch) % 2 == 0 else MODES[::-1]
                for mode in order:
                    folder = directory / f"batch-{batch}" / mode / f"run-{rep}"
                    row, actual = measure(exe, folder, mode, q, k, ref, hz)
                    if configuration is not None and actual != configuration:
                        raise ValueError(f"{folder}: effective QEMU configuration changed")
                    configuration = actual
                    row.update(batch=batch, round=rep, order=list(order), folder=str(folder.relative_to(directory)))
                    rows.append(row)
                    journal.write(json.dumps(row, allow_nan=False) + "\n")
                    journal.flush()
                    print(batch, rep, mode, f"{row['ns_per_score']:.2f} ns/score", flush=True)
            batches.append(dict(batch=batch, summary=summarize(rows), runs=rows))
    result = dict(cores=4, dtype="FP32", workspace_layout="none", qemu_cpu=RVV_CPU, seed=4108,
                  qemu=subprocess.check_output([tools.qemu, "--version"], text=True).splitlines()[0],
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  includes_local_changes=True, common_elf=str(exe.elf_path.relative_to(directory)),
                  affinity=sorted(os.sched_getaffinity(0)), tile=[4, columns], depth=128, tiles=TILES, repeats=REPEATS,
                  vector_length_check="Per core, ticks[5] is the e32,m1 VL from warmup, ticks[6] is the VL CSR "
                                      "after timed tile calls; both must match the tile width.",
                  work_assignment="Each core owns a fixed contiguous quarter of the tiles on every pass.",
                  source_coordinates="For tile t, h=t%16 and g=t//16. Use head h, rows starting at "
                                     f"4*((4*g+h)%64), columns starting at {columns}*((g+3*h)%{256 // columns}).",
                  distinct_scores=ref.size, measured_scores_per_run=REPEATS * ref.size, timebase_hz=hz,
                  metric="Wall interval on core 0 across dispatch and join, divided by all timed scores on four "
                         "cores. Each process warms the same tile kernel on all four cores before measurement. "
                         "UART wall includes startup, input loading and the untimed warm pass. Timing is throughput.",
                  warmup_pairs=1, measured_pairs=PAIRS, batches=batches,
                  stable_rvv_advantage=all(b["summary"]["rvv_advantage"] for b in batches))
    (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
