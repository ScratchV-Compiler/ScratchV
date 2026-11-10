"""Record WFI intervals on the real worker/dispatcher paths, including a worker still asleep at QMP stop."""
import argparse
from itertools import permutations
import json
import os
from pathlib import Path
from statistics import median
import struct
import subprocess
from unittest.mock import patch

from probes.w6_performance import run
from probes.w6_performance.attention_profile import capture_counters, replace_once
from probes.w6_performance.lm_head_timing import read_timebase
from scratchv.runtime import riscv_external as runtime


MODES = ("5x4-bitwise", "5x4-bounded-output", "5x4-normal-output")
CASE = "attention_scores"
EVENTS = 128
# 4 event heads, 2 dispatch rows, 8 callback rows, 4 banks of WFI events.
ROWS = 14 + 4 * EVENTS
WFI = '__asm__ volatile("wfi" ::: "memory");'
KINDS = {1: "worker", 2: "ready", 3: "join"}
TRACE = r"""
volatile unsigned long long w6_ticks[W6_ROWS][8] __attribute__((aligned(64)));
static inline unsigned long long w6_now(void) {
    unsigned long long v;
    __asm__ volatile("fence iorw,iorw\n\trdtime %0\n\tfence iorw,iorw" : "=r"(v) :: "memory");
    return v;
}
static inline void w6_wait(unsigned core, unsigned kind, unsigned target) {
    // Only this core writes its event and head rows. Never overwrite an old interval.
    unsigned long long event = w6_ticks[core][0];
    if (event >= W6_EVENTS) {
        w6_ticks[core][1] = 1;
        __asm__ volatile("wfi" ::: "memory");
        return;
    }
    volatile unsigned long long *row = w6_ticks[14 + core * W6_EVENTS + event];
    row[2] = kind;
    row[3] = target;
    row[0] = w6_now();
    w6_ticks[core][0] = event + 1;
    __asm__ volatile("fence rw,rw\n\twfi" ::: "memory");
    row[1] = w6_now();
}
"""


def instrument(worker_source):
    src = worker_source
    src = replace_once(src, "if (begin != end) fn(context, begin, end);",
                       """usize index = __atomic_load_n(&sv_generation, __ATOMIC_RELAXED) - 1;
    if (index >= 2) { w6_ticks[core][1] = 1; return; }
    volatile unsigned long long *cb = w6_ticks[6 + index * SCRATCHV_CORES + core];
    cb[0] = w6_now();
    if (begin != end) fn(context, begin, end);
    cb[1] = w6_now();""")
    worker = "if (generation == seen) {\n            " + WFI
    src = replace_once(src, worker, 'if (generation == seen) {\n            w6_wait(core, 1, core);')
    start = src.index("void sv_parallel_for(usize")
    stop = src.index("void sv_trap_report(", start)
    dispatch = src[start:stop]
    signature = "void sv_parallel_for(usize count, sv_task_fn fn, void *context) {"
    dispatch = replace_once(dispatch, signature, signature + """
    usize index = __atomic_load_n(&sv_generation, __ATOMIC_RELAXED);
    if (index >= 2) { w6_ticks[0][1] = 1; return; }
    volatile unsigned long long *slice = w6_ticks[4 + index];
    slice[0] = w6_now();""")
    pieces = dispatch.split(WFI)
    if len(pieces) != 3:
        raise ValueError("Expected ready and join WFI in the real dispatcher")
    dispatch = pieces[0] + "w6_wait(0, 2, 0);" + pieces[1] + "w6_wait(0, 3, core);" + pieces[2]
    dispatch = replace_once(dispatch, "    usize generation=", "    slice[2] = w6_now();\n    usize generation=")
    dispatch = replace_once(dispatch, "    sv_check_workers();\n}", "    sv_check_workers();\n    slice[1] = w6_now();\n}")
    return (f"#define W6_ROWS {ROWS}\n#define W6_EVENTS {EVENTS}\n" + TRACE + src[:start] + dispatch + src[stop:])


def traced_build(artifact, *args, **kwargs):
    with patch.object(runtime, "WORKER_SOURCE", instrument(runtime.WORKER_SOURCE)):
        return runtime.build_external(artifact, *args, **kwargs)


def overlap(event, start, stop):
    # An open event has been entered but no exit was published by QMP stop.
    a, b, *_ = event
    return max(0, min(b or stop, stop) - max(a, start))


def decode(path, hz, mode, forward_s):
    records = list(struct.iter_unpack("<8Q", path.read_bytes()))
    if len(records) != ROWS:
        raise ValueError("Unexpected trace size")
    history = {}
    for core in range(4):
        count, overflow = records[core][:2]
        if count > EVENTS or overflow:
            raise ValueError(f"Trace overflow for core {core}")
        events = [r[:4] for r in records[14 + core*EVENTS:14 + core*EVENTS + count]]
        previous_end = 0
        for i, (start, end, phase, target) in enumerate(events):
            if not start or start < previous_end or (end and end < start):
                raise ValueError("Nonmonotone WFI records")
            if not end and (core == 0 or i + 1 != count):
                raise ValueError("Only the final idle worker event can remain open")
            if core and (phase != 1 or target != core):
                raise ValueError("Invalid worker WFI phase")
            if not core and not ((phase == 2 and target == 0) or (phase == 3 and 1 <= target < 4)):
                raise ValueError("Invalid dispatcher WFI phase")
            previous_end = end
        history[core] = events

    ms = 1000 / hz
    slices = {}
    previous_end = 0
    for index, label in enumerate(("score", "publish")):
        start, end, ready_end = records[4 + index][:3]
        if not start and index == 1 and mode != MODES[0]:
            if any(records[4+index]) or any(any(r) for r in records[6+4*index:10+4*index]):
                raise ValueError("Unused dispatch has recorded events")
            continue
        if not (previous_end < start <= ready_end < end and (end-start)/hz <= forward_s):
            raise ValueError(f"Invalid {label} dispatch")
        if index == 1 and mode != MODES[0]:
            raise ValueError("Unexpected publish dispatch on the bounded-input score graph")
        previous_end = end
        cores = {}
        for core in range(4):
            a, b = records[6 + 4*index + core][:2]
            if not (start <= a < b <= end):
                raise ValueError(f"Invalid {label} callback, core {core}")
            events = history[core]
            before = sum(overlap(e, start, a) for e in events)
            after = sum(overlap(e, b, end) for e in events)
            inside = sum(overlap(e, a, b) for e in events)
            if inside or before > a-start or after > end-b:
                raise ValueError("WFI recorded inside an executing callback or outside its wait window")
            if core == 0:
                ready = sum(overlap(e, start, end) for e in events if e[2] == 2)
                join = sum(overlap(e, start, end) for e in events if e[2] == 3)
                if ready != before or join != after or ready > ready_end-start:
                    raise ValueError("Main WFI phases do not partition pre/post callback waits")
            cores[core] = dict(callback_start_tick=a, callback_end_tick=b, callback_ms=(b-a)*ms,
                               pre_callback_ms=(a-start)*ms, post_callback_ms=(end-b)*ms,
                               pre_wfi_ms=before*ms, post_wfi_ms=after*ms,
                               pre_other_ms=(a-start-before)*ms, post_other_ms=(end-b-after)*ms,
                               overlapping_wfi_count=sum(overlap(e, start, end) > 0 for e in events),
                               open_wfi_intersects=any(not e[1] and overlap(e, start, end) > 0 for e in events))
        last = max(cores, key=lambda c: cores[c]["callback_end_tick"])
        slices[label] = dict(start_tick=start, end_tick=end, dispatch_ms=(end-start)*ms,
                             ready_check_end_tick=ready_end, main_ready_wfi_ms=cores[0]["pre_wfi_ms"],
                             main_join_wfi_ms=cores[0]["post_wfi_ms"], last_callback_core=last, cores=cores)
    if len(slices) != (2 if mode == MODES[0] else 1):
        raise ValueError("Wrong dispatch count for fixed input")
    event_rows = {core: [dict(enter=a, exit=b or None, kind=KINDS[k], target=t) for a, b, k, t in rows]
                  for core, rows in history.items()}
    return dict(dispatches=slices, events=event_rows)


def summarize(rows):
    summary = {}
    fields = ("callback_ms", "pre_callback_ms", "post_callback_ms", "pre_wfi_ms", "post_wfi_ms",
              "pre_other_ms", "post_other_ms")
    for mode in MODES:
        formal = [r for r in rows if r["variant"] == mode and r["phase"] == "measure"]
        summary[mode] = {}
        for label in formal[0]["dispatches"]:
            data = [r["dispatches"][label] for r in formal]
            result = {k: median(s[k] for s in data)
                      for k in ("dispatch_ms", "main_ready_wfi_ms", "main_join_wfi_ms")}
            result["cores"] = {core: {k: median(s["cores"][core][k] for s in data) for k in fields}
                               for core in range(4)}
            summary[mode][label] = result
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(os.sched_getaffinity(0)) < 4:
        parser.error("Expected four available CPUs")
    dest = args.output_dir.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    tools = run.discover_toolchain()
    hz = read_timebase(tools.qemu, dest / "timer")
    with patch.object(run, "build_external", traced_build):
        prepared = run.prepare_case(CASE, dest / CASE, tools, compare_candidates=True,
                                    variants=[(name, "5x4", "runtime-panel") for name in MODES])
    rows, baseline = [], {}
    with (dest / "runs.jsonl").open("x", encoding="utf-8") as log:
        for rep, order in enumerate((MODES, *permutations(MODES))):
            for mode in order:
                exe = prepared[2][mode][0]
                folder = dest / CASE / mode / f"run-{rep}"
                with capture_counters(exe, rows=ROWS):
                    row = run.measure(CASE, mode, "5x4", "runtime-panel", "measure" if rep else "warmup",
                                      rep, prepared, folder, baseline)
                report = json.loads((folder / "run.json").read_text())
                cmd = report["command"]
                if (row["ort_bit_mismatches"] or row["max_abs_vs_ort"] or not row["bitwise_equal"]
                        or not report["qemu_reaped"] or report.get("cleanup_errors")
                        or report["timing_mode"] != "event" or report["qemu_workspace_layout"] != "none"
                        or cmd[cmd.index("-smp") + 1] != "4" or cmd[cmd.index("-cpu") + 1] != "rv64"
                        or "-icount" in cmd):
                    raise ValueError("Unexpected output, timing or QEMU configuration")
                row.update(decode(folder / "per-core.bin", hz, mode, row["forward_s"]), order=list(order))
                rows.append(row)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                score = row["dispatches"]["score"]
                print(rep, mode, round(score["dispatch_ms"], 2), round(score["main_join_wfi_ms"], 2), flush=True)
    result = dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=run.ROOT, text=True).strip(),
                  cores=4, dtype="FP32", workspace_layout="none", timebase_hz=hz, guest_event_capacity=EVENTS,
                  method="Fenced guest timestamps bracket the worker, ready and join WFI instructions. "
                         "Clip each event to the dispatch's pre/post callback windows; an open event "
                         "continues to the end of the window. These elapsed intervals include logging "
                         "and host scheduling and may overlap work on other cores.",
                  summary=summarize(rows), runs=rows)
    (dest / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
