"""Break down the 5×4 score kernel on the last-finishing core of each actual run."""
import argparse
from dataclasses import replace
from itertools import permutations
import json
import os
from pathlib import Path
from statistics import median
import subprocess
from unittest.mock import patch

import numpy as np

from probes.w6_performance import run
from probes.w6_performance.attention_profile import (
    CASE, CLAIM_ARRAY, _rewrite_callback, capture_counters, claim_worker_source, replace_once,
)
from probes.w6_performance.lm_head_timing import read_timebase
from scratchv.runtime import riscv_external as runtime


MODES = ("outer", "panel", "tile")
NAMES = ("5x4-bitwise", "5x4-bounded-output", "5x4-normal-output")
VARIANTS = [(name, "5x4", "runtime-panel") for name in NAMES]
CORE_ID = '\n    size_t core; __asm__ volatile("csrr %0, mhartid" : "=r"(core));'
CLOCK_ASM = r'fence iorw,iorw\n\trdtime %0\n\tfence iorw,iorw'
CLOCK = f'static inline unsigned long long w6_now(void) {{ unsigned long long t; __asm__ volatile("{CLOCK_ASM}" : "=r"(t) :: "memory"); return t; }}'


def split_tile(body, rows):
    """Keep the FMA-to-writeback boundary ordered even for register-only operations."""
    start = body.index(f"            float acc[{rows}][4]")
    mid = body.index(f"            for (size_t r = 0; r < {rows}; ++r) {{\n                float *const out_row", start)
    end = body.index("\n            }", mid) + len("\n            }")
    inputs = ", ".join(f'"f"(acc[{r}][{c}])' for r in range(rows) for c in range(4))
    stamp = (f'            uint64_t write_begin;\n'
             f'            __asm__ volatile("{CLOCK_ASM}" : "=r"(write_begin) : {inputs} : "memory");\n')
    # Each accumulator is an asm input above, so its FMA must have completed.
    # The tied no-op below makes its consumers depend on the timestamp. Four
    # outputs per asm keep within GCC's operand limit even for twenty sums.
    for r in range(rows):
        outputs = ", ".join(f'"+f"(acc[{r}][{c}])' for c in range(4))
        stamp += f'            __asm__ volatile("" : {outputs} : "r"(write_begin) : "memory");\n'
    return (body[:start] + "            uint64_t fma_begin=w6_now();\n" + body[start:mid] + stamp +
            body[mid:end] + "\n            uint64_t write_end=w6_now();\n"
            "            fma_ticks += write_begin-fma_begin; write_ticks += write_end-write_begin;" + body[end:])


def instrument(source, mode):
    if mode not in MODES:
        raise ValueError("Unexpected score profiling mode")
    source = replace_once(source, "#include <stdint.h>", "#include <stdint.h>\n" + CLAIM_ARRAY + CLOCK)
    for variant in ("score", "normal"):
        score = f"w6_{variant}_range"
        claim = "w6_claim_scores" if variant == "score" else "w6_normal_claim_scores"
        if f"static void {score}(" not in source:
            continue

        def claim_body(body):
            body = replace_once(body, "if (first >= 1024) return;",
                                "if (first >= 1024) { w6_ticks[core][1]=w6_now(); return; }")
            return CORE_ID + "\n    w6_ticks[core][0]=w6_now();" + body

        def score_body(body):
            counters = "w6_ticks[core][6] += end-begin; ++w6_ticks[core][7];"
            if mode == "outer":
                return CORE_ID + body + "\n    " + counters
            body = replace_once(body, "float panel[128][4];", "float panel[128][4];\n"
                                "    uint64_t pack_ticks=0, tile_ticks=0, fma_ticks=0, write_ticks=0;")
            pack = "        for (size_t k = 0; k < 128; ++k)\n            for (size_t lane"
            body = replace_once(body, pack, "        uint64_t pack_begin=w6_now();\n" + pack)
            rows = "        for (size_t row = 0; row + 5 <= 256;"
            body = replace_once(body, rows, "        uint64_t tiles_begin=w6_now();\n" + rows)
            end_task = body.rindex("\n    }")
            body = (body[:end_task] + "\n        uint64_t tiles_end=w6_now();\n"
                    "        pack_ticks+=tiles_begin-pack_begin; tile_ticks+=tiles_end-tiles_begin;"
                    + body[end_task:])
            if mode == "tile":
                for height in (5, 1):
                    body = split_tile(body, height)
            return (CORE_ID + body + "\n    " + counters +
                    "\n    w6_ticks[core][2]+=pack_ticks; w6_ticks[core][3]+=tile_ticks;"
                    "\n    w6_ticks[core][4]+=fma_ticks; w6_ticks[core][5]+=write_ticks;")

        source = _rewrite_callback(source, score, score_body)
        source = _rewrite_callback(source, claim, claim_body)
        dispatch = f"sv_parallel_for(4ULL, {claim}, &ctx);"
        source = replace_once(source, dispatch, "w6_ticks[4][0]=w6_now();\n" + dispatch +
                              "\nw6_ticks[4][1]=w6_now();")
    for slot, marker in ((0, "const float *sv_v0 ="), (1, "int sv_status_0 ="),
                         (2, "if (sv_status_0) return sv_status_0;")):
        source = replace_once(source, marker, f"w6_ticks[8][{slot}]=w6_now();\n  " + marker)
    publish = "w6_publish_parallel(output, sv_v2, 4194304ULL);"
    bounded = "bounded" if "int bounded = 1;" in source else "0"
    return replace_once(source, publish, publish + f"\n  w6_ticks[8][3]=w6_now(); w6_ticks[8][4]={bounded};")


def decode(ticks, hz, mode, name, forward_s):
    scan, begin, end, publish_end, direct = ticks[8][:5]
    dispatch, join, ready_wfi, done_wfi = ticks[4][:4]
    if not (0 < scan < begin <= dispatch < join <= end <= publish_end and
            (publish_end-scan)/hz <= forward_s and direct == int(name != "5x4-bitwise")):
        raise ValueError("Invalid outer kernel timing or output branch")

    def ms(delta):
        return 1000*delta/hz

    per_core = []
    for core, (start, finish, pack, tiles, fma, write, tasks, claims) in enumerate(ticks[:4]):
        if not (dispatch <= start < finish <= join and tasks == 8*claims and
                fma + write <= tiles and pack + tiles <= finish-start):
            raise ValueError(f"Invalid time or claimed work on core {core}")
        if mode == "outer" and (pack or tiles) or mode != "tile" and (fma or write):
            raise ValueError("Counters inconsistent with the instrumentation level")
        if tasks and ((mode != "outer" and not (pack and tiles)) or
                      (mode == "tile" and not (fma and write))):
            raise ValueError("Missing timing for nonempty work")
        per_core.append(dict(core=core, entry_ms=ms(start-begin), exit_ms=ms(finish-begin),
                             callback_ms=ms(finish-start), pack_ms=ms(pack), tiles_ms=ms(tiles),
                             fma_ms=ms(fma), write_ms=ms(write), claimed_blocks=tasks,
                             tile_other_ms=ms(tiles-fma-write), callback_other_ms=ms(finish-start-pack-tiles),
                             tail_to_join_ms=ms(join-finish)))
    if sum(r["claimed_blocks"] for r in per_core) != 1024:
        raise ValueError("The score callbacks did not cover every four-column block")
    if ready_wfi > ticks[0][0]-dispatch or done_wfi > join-ticks[0][1]:
        raise ValueError("Main-core WFI outside the corresponding pre/post-callback interval")
    last_core = max(range(4), key=lambda i: ticks[i][1])
    last = per_core[last_core]
    after = ms(end-ticks[last_core][1])
    parts = dict(critical_entry_ms=last["entry_ms"], critical_pack_ms=last["pack_ms"],
                 critical_fma_ms=last["fma_ms"], critical_write_ms=last["write_ms"],
                 critical_tile_other_ms=last["tile_other_ms"], critical_callback_other_ms=last["callback_other_ms"],
                 critical_join_return_ms=after)
    if abs(sum(parts.values())-ms(end-begin)) > 1e-6:
        raise ValueError("Critical path does not partition the kernel's elapsed time")
    return dict(kernel_ms=ms(end-begin), dispatch_ms=ms(join-dispatch), input_ms=ms(begin-scan),
                publish_ms=ms(publish_end-end), direct_output=bool(direct), cores=per_core,
                critical_core=last_core, critical_mul_and_write_ms=last["tiles_ms"], **parts,
                finish_spread_ms=ms(max(r[1] for r in ticks[:4])-min(r[1] for r in ticks[:4])),
                main_ready_wfi_ms=ms(ready_wfi), main_join_wfi_ms=ms(done_wfi))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if len(os.sched_getaffinity(0)) < 4:
        parser.error("Four available CPUs are required")
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    tools = run.discover_toolchain()
    hz = read_timebase(tools.qemu, directory / "dtb")
    prepared_by_mode = {}
    for mode in MODES:
        root = directory / mode / CASE
        root.parent.mkdir()

        def build(artifact, *args, **kwargs):
            source = instrument(artifact.source, mode)
            with patch.object(runtime, "WORKER_SOURCE", claim_worker_source(runtime.WORKER_SOURCE)):
                return runtime.build_external(replace(artifact, source=source), *args, **kwargs)

        with patch.object(run, "build_external", build):
            prepared_by_mode[mode] = run.prepare_case(CASE, root, tools, compare_candidates=True, variants=VARIANTS)
    for prepared in prepared_by_mode.values():
        control = prepared_by_mode["outer"]
        if any(not np.array_equal(v.view(np.uint32), control[0][k].view(np.uint32)) for k,v in prepared[0].items()) or not np.array_equal(prepared[1].view(np.uint32), control[1].view(np.uint32)):
            raise ValueError("Instrumentation builds must use identical inputs and ORT reference bits")
    rows, baseline = [], {}
    candidate_orders = list(permutations(NAMES))
    with (directory / "runs.jsonl").open("x", encoding="utf-8") as log:
        for rep, mode_order in enumerate((MODES, *permutations(MODES))):
            for mode in mode_order:
                prepared = prepared_by_mode[mode]
                root = directory / mode / CASE
                order = candidate_orders[(rep-1+2*MODES.index(mode)) % len(candidate_orders)] if rep else NAMES
                for name in order:
                    exe = prepared[2][name][0]
                    folder = root / name / f"run-{rep}"
                    with capture_counters(exe, rows=10):
                        row = run.measure(CASE, name, "5x4", "runtime-panel", "measure" if rep else "warmup",
                                          rep, prepared, folder, baseline)
                    ticks = np.fromfile(folder / "per-core.bin", dtype="<u8").reshape(10, 8).tolist()
                    report = json.loads((folder / "run.json").read_text())
                    cmd = report["command"]
                    if (row["max_abs_vs_ort"] or row["ort_bit_mismatches"] or not report["qemu_reaped"]
                            or report.get("cleanup_errors") or report["timing_mode"] != "event"
                            or report["qemu_workspace_layout"] != "none" or cmd[cmd.index("-smp")+1] != "4"
                            or cmd[cmd.index("-cpu")+1] != "rv64" or "-icount" in cmd):
                        raise ValueError(f"Invalid four-core run or output at {folder}")
                    phases = decode(ticks, hz, mode, name, row["forward_s"])
                    row.update(mode=mode, order=list(order), mode_order=list(mode_order), ticks=ticks, phases=phases)
                    rows.append(row)
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    log.flush()
                    print(mode, name, rep, round(phases["kernel_ms"], 2), flush=True)
    summary = {}
    for name in NAMES:
        summary[name] = {}
        for mode in MODES:
            formal = [r for r in rows if r["mode"] == mode and r["variant"] == name and r["phase"] == "measure"]
            summary[name][mode] = {k: median(r["phases"][k] for r in formal)
                                   for k in formal[0]["phases"] if k.endswith("_ms")}
            summary[name][mode]["uart_ms"] = 1000*median(r["forward_s"] for r in formal)
    result = dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=run.ROOT, text=True).strip(),
                  cores=4, dtype="float32", padding="none", seed=4108, timebase_hz=hz,
                  method="outer: entry/kernel/callback and WFI. panel: add two adjacent panel intervals. "
                         "tile: additionally time all 51 five-row tiles and the last one-row tile per panel. "
                         "Critical fields select the callback finishing last on each run; their raw ticks "
                         "partition the kernel interval. Per-core tails and main WFI overlap this time. "
                         "Tile arithmetic includes accumulator initialization, Q/K loads and loop overhead. "
                         "These are elapsed guest timestamps, not pure FMA cycles or host CPU time.",
                  summary=summary, runs=rows)
    (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
