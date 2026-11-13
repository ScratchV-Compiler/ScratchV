"""Measure packing, multiply/writeback and four-core synchronization for the W6 score MatMul."""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import re
from statistics import median
import subprocess
from unittest.mock import patch

import numpy as np

from probes.w6_performance import run
from probes.w6_performance.lm_head_timing import read_timebase
from scratchv.runtime import riscv_external as runtime


CASE = "attention_scores"
VARIANTS = (("1x4-original", "1x4", "original"), ("4x4-runtime-panel", "4x4", "runtime-panel"))
PARTS = ("critical_pack_ms", "critical_mul_store_ms", "critical_bookkeeping_ms", "critical_schedule_ms")
# One cache line per callback and one for the dispatcher. Blank slots are padding.
ARRAY = "volatile unsigned long long w6_ticks[5][8] __attribute__((aligned(64)));"
CLAIM_ARRAY = "volatile unsigned long long w6_ticks[10][8] __attribute__((aligned(64)));"
TIMER = """
static inline unsigned long long w6_now(void) {
    unsigned long long v;
    __asm__ volatile("rdtime %0" : "=r"(v) :: "memory");
    return v;
}
"""


def replace_once(source, marker, new):
    if source.count(marker) != 1:
        raise ValueError(f"Expected one timing marker for {marker!r}")
    return source.replace(marker, new)


def model_source(source):
    source = replace_once(source, "#include <stdint.h>", "#include <stdint.h>\n" + ARRAY + TIMER)
    callbacks = list(re.finditer(r"static void (\w+_range)\(void \*opaque, size_t begin, size_t end\) {", source))
    if len(callbacks) != 1 or callbacks[0][1] not in ("sv_matmul_sv_v2_range", "w6_score_range"):
        raise ValueError("Expected the original score MatMul or 4×4 panel callback")
    entry = callbacks[0]
    end = source.index("\n}", entry.end())
    body = source[entry.end():end]
    if entry[1] == "w6_score_range":
        body = replace_once(body, "float panel[128][4];", "float panel[128][4];\n    uint64_t packed=0, calc=0;")
        start_packing = "        for (size_t k = 0; k < 128; ++k)\n            for (size_t lane"
        body = replace_once(body, start_packing, "        uint64_t p0=w6_now();\n" + start_packing)
        rows = "        for (size_t row = 0;"
        body = replace_once(body, rows, "        uint64_t p1=w6_now();\n" + rows)
        # Read once per four-column panel; never insert a timer in the FMA loop.
        body = replace_once(body, "\n        }\n    }", """
        }
        uint64_t p2=w6_now(); packed+=p1-p0; calc+=p2-p1;
    }""")
        body += "\n    w6_ticks[core][2]=packed; w6_ticks[core][3]=calc;"
    begin = """
    size_t core;
    __asm__ volatile("csrr %0, mhartid" : "=r"(core));
    w6_ticks[core][0] = w6_now();"""
    source = source[:entry.end()] + begin + body + "\n    w6_ticks[core][1] = w6_now();" + source[end:]
    source, count = re.subn(r"^sv_parallel_for\([^;\n]+\);",
                            lambda m: "w6_ticks[4][0]=w6_now();\n" + m[0] + "\nw6_ticks[4][1]=w6_now();",
                            source, flags=re.MULTILINE)
    if count != 1:
        raise ValueError("Expected exactly one score MatMul dispatch")
    return source


def worker_source(source):
    start = source.index("void sv_parallel_for(usize")
    stop = source.index("void sv_trap_report(", start)
    wfi = '__asm__ volatile("wfi" ::: "memory");'
    sections = source[start:stop].split(wfi)
    if len(sections) != 3:
        raise ValueError("Expected one wait for readiness and one wait for completion")
    dispatch = sections[0]
    for slot, section in zip((2, 3), sections[1:]):
        dispatch += f"{{ unsigned long long a=w6_now(); {wfi} w6_ticks[4][{slot}]+=w6_now()-a; }}" + section
    return "extern " + ARRAY + TIMER + source[:start] + dispatch + source[stop:]


def trace_build(artifact, *args, **kwargs):
    with patch.object(runtime, "WORKER_SOURCE", worker_source(runtime.WORKER_SOURCE)):
        return runtime.build_external(replace(artifact, source=model_source(artifact.source)), *args, **kwargs)


def _rewrite_callback(source, name, transform):
    signature = f"static void {name}(void *opaque, size_t begin, size_t end) {{"
    if source.count(signature) != 1:
        raise ValueError(f"Expected one callback {name}")
    start = source.index(signature) + len(signature)
    end = source.index("\n}", start)
    return source[:start] + transform(source[start:end]) + source[end:]


def claim_model_source(source):
    """Accumulate repeated score batches; time the score and copy dispatches separately."""
    source = replace_once(source, "#include <stdint.h>", "#include <stdint.h>\n" + CLAIM_ARRAY + TIMER)
    core_id = '\n    size_t core;\n    __asm__ volatile("csrr %0, mhartid" : "=r"(core));'

    def score(body):
        body = replace_once(body, "float panel[128][4];",
                            "float panel[128][4];\n    uint64_t packed=0, calc=0;")
        packing = "        for (size_t k = 0; k < 128; ++k)\n            for (size_t lane"
        body = replace_once(body, packing, "        uint64_t p0=w6_now();\n" + packing)
        rows = "        for (size_t row = 0;"
        body = replace_once(body, rows, "        uint64_t p1=w6_now();\n" + rows)
        body = replace_once(body, "\n        }\n    }", """
        }
        uint64_t p2=w6_now(); packed+=p1-p0; calc+=p2-p1;
    }""")
        return core_id + body + """
    w6_ticks[core][2]+=packed;
    w6_ticks[core][3]+=calc;
    w6_ticks[core][4]+=end-begin;
    w6_ticks[core][5]+=1;"""

    def claim(body):
        body = replace_once(body, "if (first >= 1024) return;",
                            "if (first >= 1024) { w6_ticks[core][1]=w6_now(); return; }")
        return core_id + "\n    w6_ticks[core][0]=w6_now();" + body

    def copy(body):
        return core_id + "\n    w6_ticks[5+core][0]=w6_now();" + body + """
    w6_ticks[5+core][2]=end-begin;
    w6_ticks[5+core][1]=w6_now();"""

    source = _rewrite_callback(source, "w6_score_range", score)
    source = _rewrite_callback(source, "w6_claim_scores", claim)
    source = _rewrite_callback(source, "w6_publish_range", copy)
    for row, call in ((4, "sv_parallel_for(4ULL, w6_claim_scores, &ctx);"),
                      (9, "sv_parallel_for(blocks, w6_publish_range, &ctx);")):
        source = replace_once(source, call, f"w6_ticks[{row}][0]=w6_now();\n    " + call
                              + f"\n    w6_ticks[{row}][1]=w6_now();")
    return source


def claim_worker_source(source):
    """The first dispatch is MatMul, the second is its final output copy."""
    start = source.index("void sv_parallel_for(usize")
    stop = source.index("void sv_trap_report(", start)
    dispatch = source[start:stop]
    signature = "void sv_parallel_for(usize count, sv_task_fn fn, void *context) {"
    dispatch = replace_once(dispatch, signature, signature + """
    static unsigned w6_dispatch_index;
    unsigned w6_counter_row = w6_dispatch_index++ == 0 ? 4 : 9;""")
    wfi = '__asm__ volatile("wfi" ::: "memory");'
    sections = dispatch.split(wfi)
    if len(sections) != 3:
        raise ValueError("Expected readiness and completion waits in the dispatcher")
    dispatch = sections[0]
    for slot, section in zip((2, 3), sections[1:]):
        dispatch += (f"{{ unsigned long long a=w6_now(); {wfi} "
                     f"w6_ticks[w6_counter_row][{slot}]+=w6_now()-a; }}" + section)
    return "extern " + CLAIM_ARRAY + TIMER + source[:start] + dispatch + source[stop:]


def claim_trace_build(artifact, *args, **kwargs):
    with patch.object(runtime, "WORKER_SOURCE", claim_worker_source(runtime.WORKER_SOURCE)):
        return runtime.build_external(replace(artifact, source=claim_model_source(artifact.source)),
                                      *args, **kwargs)


def capture_counters(executable, *, rows=5):
    size = rows * 8 * 8
    listing = subprocess.check_output(["nm", "-S", executable.elf_path], text=True)
    matches = re.findall(r"^([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+[bBdD]\s+w6_ticks$", listing, re.MULTILINE)
    if len(matches) != 1 or int(matches[0][1], 16) != size:
        raise ValueError(f"Expected one {size}-byte timer array in the measured ELF")
    address = int(matches[0][0], 16)

    class TimingQMP(runtime.QMP):
        def command(self, name, arguments=None):
            response = super().command(name, arguments)
            if name == "stop":
                message = super().command("human-monitor-command", {
                    "command-line": f"pmemsave 0x{address:x} {size} per-core.bin",
                })
                if message.strip():
                    raise RuntimeError(f"Could not read score MatMul counters: {message}")
            return response

    return patch.object(runtime, "QMP", TimingQMP)


def decode(ticks, hz, is_panel):
    cores, main = ticks[:4], ticks[4]
    start, stop, ready_wfi, join_wfi = main[:4]
    if not 0 < start < stop or hz <= 0:
        raise ValueError("Invalid dispatch interval or QEMU timer frequency")
    if ready_wfi > cores[0][0] - start or join_wfi > stop - cores[0][1]:
        raise ValueError("Main core WFI duration exceeds its dispatch/wait interval")

    def ms(delta):
        return delta * 1000 / hz

    per_core = []
    for core, (a, b, pack, calc, *_) in enumerate(cores):
        if not start <= a < b <= stop or pack + calc > b - a or (is_panel and not (pack and calc)):
            raise ValueError(f"Invalid phase counters for core {core}")
        loop = calc if is_panel else b - a
        per_core.append(dict(core=core, entry_ms=ms(a-start), exit_ms=ms(b-start), packed_ms=ms(pack),
                             mul_store_ms=ms(loop), callback_other_ms=ms(b-a-pack-loop),
                             before_callback_ms=ms(a-start), after_callback_ms=ms(stop-b)))
    last = max(per_core, key=lambda record: record["exit_ms"])
    return dict(dispatch_ms=ms(stop-start), critical_core=last["core"], critical_pack_ms=last["packed_ms"],
                critical_mul_store_ms=last["mul_store_ms"], critical_bookkeeping_ms=last["callback_other_ms"],
                critical_schedule_ms=last["before_callback_ms"] + last["after_callback_ms"],
                main_wait_after_work_ms=per_core[0]["after_callback_ms"],
                main_ready_wfi_ms=ms(ready_wfi), main_join_wfi_ms=ms(join_wfi), cores=per_core)


def decode_claims(ticks, hz, claim_blocks):
    """Decode elapsed intervals; callback tail intervals are not worker WFI measurements."""
    if (type(claim_blocks) is not int or claim_blocks not in (4, 8, 16) or type(hz) not in (int, float)
            or not math.isfinite(hz) or hz <= 0
            or len(ticks) != 10 or any(len(row) != 8 for row in ticks)
            or any(type(value) is not int or value < 0 for row in ticks for value in row)):
        raise ValueError("Invalid claim-profile counters or configuration")
    start, stop, ready_wfi, join_wfi = ticks[4][:4]
    copy_start, copy_stop, copy_ready_wfi, copy_join_wfi = ticks[9][:4]
    if not 0 < start < stop <= copy_start < copy_stop:
        raise ValueError("Invalid or overlapping MatMul/copy dispatch intervals")

    def ms(delta):
        return delta * 1000 / hz

    cores, copy_cores = [], []
    for core in range(4):
        a, b, pack, calc, tasks, claims = ticks[core][:6]
        if (not start <= a < b <= stop or pack + calc > b-a
                or tasks != claims * claim_blocks or tasks > 1024
                or (tasks > 0 and not (pack and calc))
                or (tasks == 0 and (pack or calc))):
            raise ValueError(f"Invalid MatMul counters or claimed tasks for core {core}")
        cores.append(dict(core=core, entry_ms=ms(a-start), exit_ms=ms(b-start),
                          packed_ms=ms(pack), mul_store_ms=ms(calc),
                          callback_ms=ms(b-a), callback_other_ms=ms(b-a-pack-calc),
                          before_callback_ms=ms(a-start), after_callback_ms=ms(stop-b),
                          claimed_blocks=tasks, successful_claims=claims))
        ca, cb, blocks = ticks[5+core][:3]
        if not copy_start <= ca < cb <= copy_stop or blocks != 16384:
            raise ValueError(f"Invalid copy counters or block coverage for core {core}")
        copy_cores.append(dict(core=core, entry_ms=ms(ca-copy_start), exit_ms=ms(cb-copy_start),
                               callback_ms=ms(cb-ca), before_callback_ms=ms(ca-copy_start),
                               after_callback_ms=ms(copy_stop-cb), copied_64byte_blocks=blocks))
    if sum(core["claimed_blocks"] for core in cores) != 1024:
        raise ValueError("Claimed tasks do not cover all 1024 score blocks")
    for row, group, a, b in ((ticks[4], ticks[:4], start, stop),
                             (ticks[9], ticks[5:9], copy_start, copy_stop)):
        if row[2] > group[0][0]-a or row[3] > b-group[0][1]:
            raise ValueError("Main-core WFI exceeds its dispatch/wait interval")
    last = max(cores, key=lambda core: core["exit_ms"])
    return dict(dispatch_ms=ms(stop-start), critical_core=last["core"],
                critical_pack_ms=last["packed_ms"], critical_mul_store_ms=last["mul_store_ms"],
                critical_bookkeeping_ms=last["callback_other_ms"],
                critical_schedule_ms=last["before_callback_ms"]+last["after_callback_ms"],
                main_wait_after_work_ms=cores[0]["after_callback_ms"],
                main_ready_wfi_ms=ms(ready_wfi), main_join_wfi_ms=ms(join_wfi),
                finish_spread_ms=max(c["exit_ms"] for c in cores)-min(c["exit_ms"] for c in cores),
                worker_tail_mean_ms=sum(c["after_callback_ms"] for c in cores[1:])/3,
                worker_tail_max_ms=max(c["after_callback_ms"] for c in cores[1:]),
                cores=cores, copy_dispatch_ms=ms(copy_stop-copy_start),
                copy_main_wait_after_work_ms=copy_cores[0]["after_callback_ms"],
                copy_main_ready_wfi_ms=ms(copy_ready_wfi), copy_main_join_wfi_ms=ms(copy_join_wfi),
                copy_cores=copy_cores)


def summarize(rows, variant):
    selected = [r for r in rows if r["phase"] == "measure" and r["variant"] == variant]
    plain = median(r["forward_s"] for r in selected if r["mode"] == "plain")
    diagnostic = median(r["forward_s"] for r in selected if r["mode"] == "diagnostic")
    phases = [r["phase_times"] for r in selected if r["mode"] == "diagnostic"]
    fields = {field: median(r[field] for r in phases) for field in phases[0] if field.endswith("_ms")}
    for key in PARTS:
        fields[key[:-3] + "_median_pct"] = 100 * median(r[key] / r["dispatch_ms"] for r in phases)
    # These four pooled values sum to 100%. Component-wise medians need not do so.
    duration = sum(r["dispatch_ms"] for r in phases)
    fields.update(critical_cores=[r["critical_core"] for r in phases], plain_median_forward_s=plain,
                  diagnostic_median_forward_s=diagnostic, instrumented_minus_plain_pct=100*(diagnostic/plain-1),
                  pooled_dispatch_pct={key: 100 * sum(r[key] for r in phases) / duration for key in PARTS})
    return fields


def _claim_comparison(dest, tools):
    variants = run.CLAIM_VARIANTS
    chunks = dict(zip((name for name, _, _ in variants), (4, 8, 16)))
    hz = read_timebase(tools.qemu, dest / "timer")
    plain = run.prepare_case(CASE, dest / "plain", tools, compare_candidates=True, variants=variants)
    with patch.object(run, "build_external", claim_trace_build):
        diagnostic = run.prepare_case(CASE, dest / "diagnostic", tools,
                                      compare_candidates=True, variants=variants)
    if (set(plain[0]) != set(diagnostic[0])
            or any(not np.array_equal(plain[0][k].view(np.uint32), diagnostic[0][k].view(np.uint32))
                   for k in plain[0])
            or not np.array_equal(plain[1].view(np.uint32), diagnostic[1].view(np.uint32))):
        raise ValueError("Plain and diagnostic inputs or ORT references differ")

    rows, baseline = [], {}
    # Timing and instrumentation are separate batches. The first six plain rounds
    # put each candidate in each position twice, including both directions.
    with (dest / "runs.jsonl").open("x", encoding="utf-8") as log:
        for mode, prepared, rounds in (("plain", plain, 6), ("diagnostic", diagnostic, 3)):
            traced = mode == "diagnostic"
            for round_id in range(rounds+1):
                offset = (round_id-1) % 3 if round_id else 0
                order = variants[offset:] + variants[:offset]
                if round_id > 3:
                    order = tuple(reversed(order))
                for variant, tile, layout in order:
                    folder = dest / mode / variant / f"run-{round_id}"
                    with capture_counters(prepared[2][variant][0], rows=10) if traced else nullcontext():
                        row = run.measure(CASE, variant, tile, layout, "measure" if round_id else "warmup",
                                          round_id, prepared, folder, baseline)
                    if row["ort_bit_mismatches"]:
                        raise ValueError(f"{folder}: output is not bitwise equal to ORT")
                    row.update(mode=mode, claim_blocks=chunks[variant])
                    if traced:
                        ticks = np.fromfile(folder / "per-core.bin", dtype="<u8").reshape(10, 8).tolist()
                        phases = decode_claims(ticks, hz, chunks[variant])
                        span = (ticks[9][1]-ticks[4][0]) / hz
                        if span > row["forward_s"] + .002:
                            raise ValueError(f"{folder}: guest diagnostic span exceeds the UART forward interval")
                        row.update(ticks=ticks, phase_times=phases, diagnostic_span_s=span)
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    log.flush()
                    rows.append(row)
                    print(variant, mode, round_id, round(row["forward_s"], 5), flush=True)

    stats = {name: summarize(rows, name) for name, _, _ in variants}
    reference = stats["4x4-checked-parallel-copy"]["plain_median_forward_s"]
    for name, summary in stats.items():
        summary["claim_blocks"] = chunks[name]
        summary["speedup_vs_claim8"] = reference / summary["plain_median_forward_s"]
        selected = [r for r in rows if r["variant"] == name and r["phase"] == "measure"]
        summary["plain_forward_range_s"] = [min(r["forward_s"] for r in selected if r["mode"] == "plain"),
                                             max(r["forward_s"] for r in selected if r["mode"] == "plain")]
        phases = [r["phase_times"] for r in selected if r["mode"] == "diagnostic"]
        summary["per_core_medians"] = [
            {"core": core, **{key: median(p["cores"][core][key] for p in phases)
                              for key in phases[0]["cores"][core] if key != "core"}}
            for core in range(4)]
    data = dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=run.ROOT, text=True).strip(),
                includes_local_changes=True, cores=4, dtype="float32", padding="none", seed=4108,
                elements=1048576, shapes=run.CASES[CASE], timebase_hz=hz, baseline_variant="4x4-checked-parallel-copy",
                statistic="Each variant: plain 1 warmup + 6 measured rounds; then diagnostic 1 warmup + 3 measured rounds. "
                          "Cyclic candidate order; plain final three rounds reverse the cyclic order.",
                scope="Elapsed callback/dispatch time includes vCPU descheduling. Worker tail intervals run from callback "
                      "completion to dispatch return; they are not measured worker WFI. Only main-core WFI is timed. "
                      "MatMul and output-copy dispatches are separate; waits overlap other cores' work.",
                statistics=stats, runs=rows)
    (dest / "result.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", required=True, type=Path)
    cli.add_argument("--claim-granularity", action="store_true",
                     help="Compare the checked scalar kernel claiming 4, 8 or 16 column blocks")
    args = cli.parse_args()
    if len(os.sched_getaffinity(0)) < 4:
        cli.error("Four available logical CPUs are required")
    dest = args.output_dir.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    tools = run.discover_toolchain()
    if args.claim_granularity:
        _claim_comparison(dest, tools)
        return
    hz = read_timebase(tools.qemu, dest / "timer")
    plain = run.prepare_case(CASE, dest / "plain", tools, compare_candidates=True, variants=VARIANTS)
    with patch.object(run, "build_external", trace_build):
        diag = run.prepare_case(CASE, dest / "diagnostic", tools, compare_candidates=True, variants=VARIANTS)

    rows, baseline = [], {}
    for round_id in range(4):
        for variant, tile, layout in (VARIANTS if round_id % 2 else reversed(VARIANTS)):
            for mode in (("plain", "diagnostic") if round_id % 2 else ("diagnostic", "plain")):
                traced = mode == "diagnostic"
                prepared = diag if traced else plain
                folder = dest / mode / variant / f"run-{round_id}"
                with capture_counters(prepared[2][variant][0]) if traced else nullcontext():
                    row = run.measure(CASE, variant, tile, layout, "measure" if round_id else "warmup",
                                      round_id, prepared, folder, baseline)
                row["mode"] = mode
                if traced:
                    ticks = np.fromfile(folder / "per-core.bin", dtype="<u8").reshape(5, 8).tolist()
                    row.update(ticks=ticks, phase_times=decode(ticks, hz, layout == "runtime-panel"))
                rows.append(row)
                print(variant, mode, round_id, round(row["forward_s"], 5), flush=True)
    stats = {name: summarize(rows, name) for name, _, _ in VARIANTS}
    data = dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=run.ROOT, text=True).strip(),
                cores=4, dtype="float32", padding="none", seed=4108, elements=1048576, timebase_hz=hz,
                statistic="1 warmup and 3 measured runs per variant and mode, with alternating order",
                scope="Elapsed guest time on the last-finishing core, including memory access and vCPU descheduling. "
                      "Main-core wait/WFI overlaps work on other cores.",
                plain_1x4_speedup=stats[VARIANTS[0][0]]["plain_median_forward_s"]/stats[VARIANTS[1][0]]["plain_median_forward_s"],
                statistics=stats, runs=rows)
    (dest / "result.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
