"""Measure input finite checks on the fixed four-core Attention score MatMul."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
from statistics import median
import subprocess
from unittest.mock import patch

import numpy as np

from probes.w6_performance import run
from probes.w6_performance.attention_input_checks import with_input_checks
from probes.w6_performance.attention_profile import capture_counters
from probes.w6_performance.lm_head_timing import read_timebase
from scratchv.runtime import riscv_external as runtime
from scratchv.runtime.riscv_tensor import input_layout

CASE = "attention_scores"
KERNEL = "4x4-checked-parallel-copy"
POLICIES = ("serial", "parallel")
ROUNDS = {"diagnostic": 6, "plain": 12}
INPUT_ELEMENTS = 524288


def prepare(directory, tools):
    """Build all arms from the same checked 8-block artifact and ORT reference."""
    builds = {}

    def build_variants(artifact, bundle, build_dir, toolchain, **options):
        for mode in ROUNDS:
            for policy in POLICIES:
                modified = with_input_checks(artifact, parallel=policy == "parallel",
                                             timed=mode == "diagnostic")
                executable = runtime.build_external(modified, bundle, directory / mode / policy / "build",
                                                    toolchain, **options)
                if (executable.cores != 4 or executable.evidence["qemu_workspace_layout"] != "none"
                        or executable.weights or modified.workspace_bytes != 4 * 1024**2):
                    raise ValueError("Unexpected configuration for the Attention input check")
                builds[mode, policy] = {KERNEL: (executable, modified.workspace_bytes)}
        return builds["plain", "serial"][KERNEL][0]

    with patch.object(run, "build_external", build_variants):
        inputs, reference, _ = run.prepare_case(CASE, directory / "reference", tools,
                                                 compare_candidates=True,
                                                 variants=[(KERNEL, "4x4", "runtime-panel")])
    return {key: (inputs, reference, value) for key, value in builds.items()}


def scan_times(ticks, hz, policy, forward_s):
    start, stop, serial_q, serial_k = ticks[4][:4]
    if not 0 < start < stop or (stop - start) / hz > forward_s + .002:
        raise ValueError("Input check interval outside the UART forward time")
    duration = (stop - start) * 1000 / hz
    record = {"combined_ms": duration, "percent_of_forward": 100 * duration / (forward_s * 1000)}
    if policy == "serial":
        if min(serial_q, serial_k) <= 0 or serial_q + serial_k > stop - start:
            raise ValueError("Invalid serial Q/K timing")
        return {**record, "q_ms": serial_q * 1000 / hz, "k_ms": serial_k * 1000 / hz}

    workers = []
    for core, row in enumerate(ticks[:4]):
        begin, q_end, k_end, end, elements = row[:5]
        if not start <= begin < q_end < k_end <= end <= stop or elements != INPUT_ELEMENTS // 4:
            raise ValueError(f"Invalid input check interval or coverage on core {core}")
        workers.append({"core": core, "elements_per_input": elements,
                        "q_ms": (q_end - begin) * 1000 / hz,
                        "k_ms": (k_end - q_end) * 1000 / hz,
                        "exit_ms": (end - start) * 1000 / hz})
    last = max(workers, key=lambda core: core["exit_ms"])
    return {**record, "q_ms": last["q_ms"], "k_ms": last["k_ms"],
            "last_core": last["core"], "per_core": workers}


def summarize(rows):
    stats = {}
    for policy in POLICIES:
        selected = [row for row in rows if row["phase"] == "measure" and row["input_checks"] == policy]
        diagnostic = [row["scan_times"] for row in selected if row["mode"] == "diagnostic"]
        times = [row["forward_s"] * 1000 for row in selected if row["mode"] == "plain"]
        stats[policy] = {
            "scan_medians": {key: median(row[key] for row in diagnostic)
                             for key in ("q_ms", "k_ms", "combined_ms", "percent_of_forward")},
            "forward_median_ms": median(times), "forward_range_ms": [min(times), max(times)],
            "first_half_median_ms": median(times[:len(times)//2]),
            "second_half_median_ms": median(times[len(times)//2:]),
        }
    paired_ms = []
    for rep in range(1, ROUNDS["plain"] + 1):
        pair = {row["input_checks"]: row["forward_s"] for row in rows
                if row["mode"] == "plain" and row["round"] == rep}
        paired_ms.append(1000 * (pair["serial"] - pair["parallel"]))
    return {"policies": stats, "paired_serial_minus_parallel_ms": paired_ms,
            "parallel_wins": sum(diff > 0 for diff in paired_ms),
            "forward_speedup": stats["serial"]["forward_median_ms"] / stats["parallel"]["forward_median_ms"]}


def check_errors(prepared_by_variant, dest):
    """Use the same plain ELF files as the successful serial/parallel measurements."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    results = []
    for variant in ("serial", "parallel"):
        inputs, _, builds = prepared_by_variant[variant]
        exe, _ = builds[KERNEL]
        offsets, _ = input_layout(exe.inputs)
        offset_by_name = {spec.name: offset for spec, offset in zip(exe.inputs, offsets)}
        # Choose the first Q element of slice 2 and the last K element of slice 3.
        cases = (("q_nan", "x", inputs["x"].size // 2, np.nan),
                 ("k_inf", "weight", inputs["weight"].size - 1, np.inf))
        for name, tensor, index, value in cases:
            changed = inputs[tensor].copy()
            changed.flat[index] = value
            directory = dest / variant / name
            try:
                runtime.run_external(exe, {**inputs, tensor: changed}, directory, timeout=60)
            except RuntimeError as error:
                if str(error) != "RV64 guest failed with status 0x00000002":
                    raise
            else:
                raise ValueError(f"{directory}: expected a guest numerical error")

            with (directory / "inputs.bin").open("rb") as stream:
                stream.seek(offset_by_name[tensor] + 4 * index)
                if stream.read(4) != np.asarray(value, dtype="<f4").tobytes():
                    raise ValueError(f"{directory}: unexpected packed input at the injected index")
            report = json.loads((directory / "run.json").read_text())
            frame = runtime.FRAME.unpack((directory / "uart.bin").read_bytes())
            if frame != (runtime.MAGIC, 2, exe.output.nbytes):
                raise ValueError(f"{directory}: unexpected completion frame {frame}")
            dumped = (directory / "output.bin").exists()
            if (report["stage"] != "qemu" or report["status"] != "FAIL" or report["passed"]
                    or report["guest_completed"] or not report["qemu_reaped"] or dumped
                    or report.get("cleanup_errors") or report["cores"] != 4
                    or report["qemu_workspace_layout"] != "none"):
                raise ValueError(f"{directory}: unexpected error propagation or cleanup")
            results.append(dict(variant=variant, case=name, input_name=tensor, flat_index=index,
                                guest_status=frame[1], qemu_reaped=report["qemu_reaped"],
                                output_dumped=dumped, report=str(directory.relative_to(dest) / "run.json")))
    (dest / "result.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def measure(directory):
    if len(os.sched_getaffinity(0)) < 4:
        raise ValueError("Four available CPUs are required")
    directory.mkdir(parents=True, exist_ok=False)
    tools = run.discover_toolchain()
    hz = read_timebase(tools.qemu, directory / "timer")
    built = prepare(directory, tools)
    rows, baseline = [], {}
    with (directory / "runs.jsonl").open("x", encoding="utf-8") as log:
        for mode, rounds in ROUNDS.items():
            for rep in range(rounds + 1):
                for policy in POLICIES if rep % 2 else reversed(POLICIES):
                    prepared = built[mode, policy]
                    folder = directory / mode / policy / f"run-{rep}"
                    executable = prepared[2][KERNEL][0]
                    with capture_counters(executable) if mode == "diagnostic" else nullcontext():
                        row = run.measure(CASE, KERNEL, "4x4", "runtime-panel",
                                          "measure" if rep else "warmup", rep, prepared, folder, baseline)
                    if row["ort_bit_mismatches"]:
                        raise ValueError(f"{folder}: output differs from ORT")
                    row.update(mode=mode, input_checks=policy)
                    if mode == "diagnostic":
                        ticks = np.fromfile(folder / "per-core.bin", dtype="<u8").reshape(5, 8).tolist()
                        row.update(ticks=ticks, scan_times=scan_times(ticks, hz, policy, row["forward_s"]))
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    log.flush()
                    rows.append(row)
                    print(mode, policy, rep, f"{row['forward_s']*1000:.2f} ms", flush=True)
    result = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=run.ROOT, text=True).strip(),
              "includes_local_changes": True, "case": CASE, "shapes": run.CASES[CASE], "seed": 4108,
              "claim_blocks": 8, "cores": 4, "dtype": "float32", "padding": "none",
              "elements_per_input": INPUT_ELEMENTS, "outputs": 1048576, "timebase_hz": hz,
              "warmups_per_arm": 1, "formal_rounds_per_arm": ROUNDS,
              "order": "Serial first on odd rounds; parallel first on even rounds.",
              "statistics": summarize(rows), "runs": rows}
    result["invalid_input_cases"] = check_errors({policy: built["plain", policy] for policy in POLICIES},
                                               directory / "errors")
    (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", type=Path, required=True)
    args = cli.parse_args()
    measure(args.output_dir.resolve())


if __name__ == "__main__":
    main()
