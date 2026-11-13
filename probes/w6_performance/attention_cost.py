"""Measure the marginal cost of extra FP32 loads or fused multiply-adds in a four-core score MatMul."""
import argparse
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import re
from statistics import median
import subprocess
from unittest.mock import patch

import numpy as np

from probes.w6_performance import attention_panel, run
from probes.w6_performance.attention_profile import CASE, replace_once
from probes.w6_performance.lm_head_timing import _SYMBOL, _TIMER, capture_ticks, read_timebase, read_timing, timing_address
from scratchv.runtime import riscv_external


MODES = ("stock", "matched", "extra-flw", "extra-fma")
K_ITERATIONS = 16 * (256 // 4) * (256 // 4) * 128


def diagnostic_tile(mode):
    # Keep accumulator, operand regs, loop shape and work scheduling identical.
    ops = ["fmv.w.x f25,zero"] + [f"fmv.w.x f{i},zero" for i in range(16)]
    loads = [(f"f{16+r}", f"{r*512}(%[x])") for r in range(4)] + [
        (f"f{20+c}", f"{c*4}(%[b])") for c in range(4)]
    ops += [".global w6_hot_start", "w6_hot_start:"] + [f"flw {reg},{mem}" for reg, mem in loads]
    if mode == "extra-flw":
        ops += [f"flw f24,{mem}" for _, mem in loads]
    for r in range(4):
        for c in range(4):
            acc = 4*r+c
            operands = f"f{16+r},f{20+c},f{acc}"
            if mode == "extra-fma":
                ops.append(f"fmadd.s f24,{operands}")
            ops.append(f"fmadd.s f{acc},{operands}")
    ops += ["addi %[x],%[x],4", "addi %[b],%[b],16", "bne %[b],%[end],w6_hot_start",
            ".global w6_hot_end", "w6_hot_end:"]
    for r in range(4):
        for c in range(4):
            acc = 4*r+c
            ops += [f"fadd.s f{acc},f25,f{acc}", f"fsw f{acc},{4*c}(%[o])"]
        if r != 3:
            ops.append("addi %[o],%[o],1024")
    strings = "\n".join(json.dumps(op + "\n\t") for op in ops)
    clobbers = ",".join(f'"f{i}"' for i in range(26))
    return f"""
static __attribute__((noinline)) void w6_diagnostic_tile(const float *x, const float *b, float *out) {{
    const float *end = b + 512;
    __asm__ volatile(
{strings}
        : [x] "+&r"(x), [b] "+&r"(b), [o] "+&r"(out)
        : [end] "r"(end)
        : "memory", {clobbers});
}}
"""


def kernel(stock, mode):
    if mode == "stock":
        return stock
    start = stock.index("            float acc[4][4]")
    # Only replace the row tile. Statements after the enclosing task loop are independent.
    end = stock.index("\n        }\n", start)
    tile = stock[start:end]
    if "sv_finite(" in tile or "return " in tile:
        raise ValueError("The marginal-cost diagnostic requires a scalar tile with external output checks")
    invocation = ("            w6_diagnostic_tile(ctx->x + (head * 256 + row) * 128, &panel[0][0],\n"
                  "                               ctx->output + (head * 256 + row) * 256 + col);")
    return diagnostic_tile(mode) + replace_once(stock, tile, invocation)


def traced_build(artifact, *args, **kwargs):
    source = replace_once(artifact.source, "#include <stdint.h>", "#include <stdint.h>\n" + _TIMER)
    source, matches = re.subn(
        r"^sv_parallel_for\([^;\n]+\);",
        lambda m: f"{_SYMBOL}[0]=sv_w6_read_time();\n{m[0]}\n{_SYMBOL}[1]=sv_w6_read_time();",
        source, flags=re.MULTILINE)
    if matches != 1:
        raise ValueError("Expected exactly one score dispatch")
    return riscv_external.build_external(replace(artifact, source=source), *args, **kwargs)


def hot_instructions(elf):
    dump = subprocess.check_output(["llvm-objdump", "-d", "--no-show-raw-insn", str(elf)], text=True)
    labels = re.findall(r"(?ms)^[0-9a-f]+ <w6_hot_start>:\n(.*?)^[0-9a-f]+ <w6_hot_end>:", dump)
    if len(labels) != 1:
        raise ValueError(f"Expected exactly one labelled inner loop in {elf}")
    return Counter(re.findall(r"(?m)^\s*[0-9a-f]+:\s+([a-z][.\w]*)", labels[0]))


def build_cases(directory, tools):
    stock = attention_panel._score_kernel()
    prepared, addresses, counts = {}, {}, {"stock": None}
    for mode in MODES:
        variants = [(mode, "4x4", "runtime-panel")]
        with (patch.object(attention_panel, "_score_kernel", return_value=kernel(stock, mode)),
              patch.object(run, "build_external", traced_build)):
            prepared[mode] = run.prepare_case(CASE, directory / mode, tools, compare_candidates=True, variants=variants)
        executable = prepared[mode][2][mode][0]
        addresses[mode] = timing_address(executable)
        if mode != "stock":
            counts[mode] = hot_instructions(executable.elf_path)
    base = counts["matched"]
    if (base["flw"] != 8 or base["fmadd.s"] != 16 or counts["extra-flw"] != base + Counter(flw=8)
            or counts["extra-fma"] != base + Counter({"fmadd.s": 16})):
        raise ValueError(f"Unexpected changes to the measured instruction mix: {counts}")
    # The shared small-graph builder uses fixed inputs and an independent ORT reference.
    if any(not np.array_equal(item[1], prepared["stock"][1]) for item in prepared.values()):
        raise ValueError("Different ORT reference scores across builds")
    return prepared, addresses, counts


def summarize(rows):
    measurements = {m: [row for row in rows if row["phase"] == "measure" and row["variant"] == m]
                    for m in MODES}
    summary = {m: {k: median(row[k] for row in trials) for k in ("dispatch_s", "forward_s", "qemu_s")}
               for m, trials in measurements.items()}
    matched = {row["round"]: row["dispatch_s"] for row in measurements["matched"]}
    for mode, added in (("extra-flw", 8), ("extra-fma", 16)):
        deltas = [r["dispatch_s"] - matched[r["round"]] for r in measurements[mode]]
        summary[mode].update(paired_dispatch_delta_s=deltas, median_paired_delta_s=median(deltas),
                             extra_instructions=K_ITERATIONS * added)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if len(os.sched_getaffinity(0)) < 4:
        parser.error("Four available logical CPUs are required")
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    tools = run.discover_toolchain()
    prepared, addresses, counts = build_cases(directory, tools)
    hz = read_timebase(tools.qemu, directory / "dtb")
    rows, baseline = [], {}
    # Every implementation appears twice in every position across eight timed rounds.
    for rep in range(1 + 2 * len(MODES)):
        start = max(rep - 1, 0) % len(MODES)
        order = MODES[start:] + MODES[:start]
        if rep > len(MODES):
            order = order[::-1]
        for mode in order:
            folder = directory / mode / f"run-{rep}"
            with capture_ticks(addresses[mode]):
                row = run.measure(CASE, mode, "4x4", "runtime-panel", "measure" if rep else "warmup",
                                  rep, prepared[mode], folder, baseline)
            timing = read_timing(folder, hz)
            if row["max_abs_vs_ort"] != 0 or row["ort_bit_mismatches"] or not row["bitwise_equal"]:
                raise ValueError(f"{mode}: complete output differs from ORT or another run")
            row.update(dispatch_s=timing["guest_lm_head_s"], dispatch_ticks=timing["guest_elapsed_ticks"])
            rows.append(row)
            print(rep, mode, round(row["dispatch_s"], 4), round(row["forward_s"], 4), flush=True)
    result = dict(case=CASE, cores=4, dtype="float32", workspace_padding="none", seed=4108, kiters=K_ITERATIONS,
                  matched_elf_loop_counts=counts, output_scores=1048576, timebase_hz=hz,
                  method="One warmup and eight rotated/reversed trials on the same inputs. Guest counters bracket "
                         "the dispatch outside the K loop. Extra hot loads or duplicate FMAs discard their results "
                         "in a scratch register. Paired differences measure marginal cost with this access pattern.",
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=run.ROOT, text=True).strip(),
                  summary=summarize(rows), runs=rows)
    (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
