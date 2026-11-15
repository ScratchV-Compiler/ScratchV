"""Isolated QEMU LM Head A/B. Run from the repository root on Linux."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time

MODULE = "probes.w6_performance.lm_head"
VARIANTS = ("baseline", "vocab_blocked")
VOCABS = {"small": 257, "real": 151936}
ROOT = Path(__file__).resolve().parents[2]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(assets, directory, size):
    from probes.w4_qwen3_full.run import save_executable
    from probes.w6_performance.lm_head_case import load_case
    from probes.w6_performance.lm_head_kernel import generate_vocab_blocked
    from probes.w6_performance.lm_head_timing import instrument, read_timebase, timing_address
    from scratchv.backend.tensor_c_codegen import TensorCCodegen
    from scratchv.runtime.riscv_external import build_external
    from scratchv.runtime.riscv_tensor import discover_toolchain
    from scratchv.runtime.weight_bundle import write_weight_bundle
    import numpy as np
    import onnxruntime as ort

    case = load_case(assets, VOCABS[size])
    tools = discover_toolchain()
    directory.mkdir()
    prepared = {"vocab": case.vocab, "cores": 4, "padding": "none", "dtype": "float32",
                "python": platform.python_version(), "numpy": np.__version__, "ort_package": ort.__version__,
                "active_cpus": sorted(os.sched_getaffinity(0)), "variants": {},
                "timebase_hz": read_timebase(tools.qemu, directory)}
    for variant in VARIANTS:
        if variant == "baseline":
            artifact = TensorCCodegen(case.program, case.bindings, constant_storage="external",
                kernel_calls=True, matmul_policy="blocked_fma", matmul_tile="2x4", matmul_rhs_layout="original",
                qemu_workspace_layout="none", max_constant_bytes=case.weight.nbytes,
                max_workspace_bytes=case.weight.nbytes + 2 * case.reference.nbytes + 8 * 1024**2).generate()
        else:
            artifact = generate_vocab_blocked(case.program, case.bindings)
        if (len(artifact.external_weights) != 1 or artifact.external_weights[0].shape != case.weight.shape
                or len(artifact.external_initializers) != 1 or not np.shares_memory(artifact.external_initializers[0], case.weight)
                or artifact.matmul_rhs_panels or artifact.matmul_rhs_layout != "original"
                or artifact.qemu_workspace_layout != "none" or tuple(artifact.output.shape) != case.reference.shape):
            raise ValueError("The shared, original-layout weight and complete logits are required")
        if variant == "baseline":
            write_weight_bundle(artifact.external_weights, artifact.external_initializers, directory / "weights")
        config = directory / variant
        config.mkdir()
        exe = build_external(instrument(artifact), directory / "weights", config / "build", tools, cores=4)
        if exe.cores != 4 or exe.output.nbytes != case.reference.nbytes:
            raise ValueError("Unexpected ELF ABI")
        save_executable(exe, config)
        prepared["variants"][variant] = dict(timing_address=timing_address(exe), workspace_bytes=artifact.workspace_bytes,
                weight_bytes=case.weight.nbytes, output_bytes=exe.output.nbytes, guest_ram_mib=exe.layout["memory_mib"],
                physical_weight_tensors=len(exe.weights), compile_s=exe.evidence["compile_seconds"], toolchain=exe.evidence["versions"])
    write_json(directory / "setup.json", prepared)


def validate(actual, reference, base_path):
    import numpy as np
    if actual.shape != reference.shape or actual.dtype != np.float32:
        raise ValueError("Incomplete output or unexpected output type")
    previous = (np.memmap(base_path, dtype="<f4", mode="r", shape=actual.shape) if base_path else None)
    maximum, mismatch_rows, different_bits = 0.0, [], 0
    selected = []
    for row in range(actual.shape[1]):
        out, ref = actual[0, row], reference[0, row]
        if not np.isfinite(out).all() or not np.isfinite(ref).all():
            raise ValueError(f"Nonfinite row {row}")
        maximum = max(maximum, float(np.abs(out.astype(np.float64) - ref).max()))
        token = int(np.argmax(out))
        selected.append(token)
        if token != int(np.argmax(ref)):
            mismatch_rows.append(row)
        if previous is not None:
            different_bits += int(np.count_nonzero(out.view(np.uint32) != previous[0, row].view(np.uint32)))
    return dict(elements=actual.size, max_abs_vs_ort=maximum, greedy_mismatch_rows=mismatch_rows,
                greedy_matches=len(selected)-len(mismatch_rows), token_ids=selected, finite=True,
                bit_mismatches_vs_baseline_warmup=different_bits if previous is not None else None)


def run_trial(assets, directory, size, variant, round_id):
    from probes.w4_qwen3_full.run import load_executable
    from probes.w6_performance.lm_head_case import load_case
    from probes.w6_performance.lm_head_timing import capture_ticks, read_timing
    from probes.w6_performance.run import collect_pair
    from scratchv.runtime.riscv_external import run_external

    case = load_case(assets, VOCABS[size])
    setup = json.loads((directory / "setup.json").read_text())
    exe = load_executable(directory / variant / "executable.json")
    folder = directory / variant / f"run-{round_id}"
    observed = dict(combined_peak_bytes=0, samples=0, read_errors=0, exit_races=0)
    stop = threading.Event()
    thread = threading.Thread(target=collect_pair, args=(stop, observed), daemon=True)
    thread.start()
    wall0, raw0 = time.perf_counter(), time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    try:
        with capture_ticks(setup["variants"][variant]["timing_address"], short_qemu_highwater=size == "small") as highwater:
            actual, run = run_external(exe, case.inputs, folder, timeout=1200)
    finally:
        raw = time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - raw0
        wall = time.perf_counter() - wall0
        stop.set()
        thread.join()
    measured = read_timing(folder, setup["timebase_hz"])
    if not run["passed"] or run["exit_code"] != 0 or run["timing_mode"] != "event" or run["cores"] != 4:
        raise ValueError("Incomplete RV64 execution")
    baseline = directory / "baseline/run-0/output.bin"
    numeric = validate(actual, case.reference, None if round_id == 0 and variant == "baseline" else baseline)
    write_json(folder / "numeric.json", numeric)
    row = dict(round=round_id, phase="measure" if round_id else "warmup", variant=variant, case=size,
               guest_s=measured["guest_lm_head_s"], guest_timing=measured,
               forward_s=run["forward_wall_s"], qemu_s=run["qemu_wall_s"],
               qemu_peak_bytes=run["sampled_peak_qemu_rss_bytes"], qemu_samples=run["qemu_rss_observed_samples"],
               qemu_read_errors=run["qemu_rss_sample_errors"], **observed, **highwater, clock_delta_s=wall-raw,
               **{k:v for k,v in numeric.items() if k != "token_ids"})
    write_json(folder / "measurement.json", row)
    if (numeric["max_abs_vs_ort"] >= 1e-3 or numeric["greedy_mismatch_rows"] or
            numeric["bit_mismatches_vs_baseline_warmup"] or not observed["samples"]
            or not row["qemu_peak_bytes"] or not 0 < row["guest_s"] <= row["forward_s"] + 0.02
            or abs(wall-raw) > max(0.02,wall*0.0006)):
        raise ValueError(f"Measurement or output checks failed. See {folder / 'measurement.json'}")
    print(f"{size}/{variant}/{round_id}: guest {row['guest_s']:.3f}s, UART {row['forward_s']:.3f}s, "
          f"max_abs {numeric['max_abs_vs_ort']:.3g}, tokens {numeric['greedy_matches']}/256", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--small-only", action="store_true")
    parser.add_argument("--worker", choices=("prepare","trial"), help=argparse.SUPPRESS)
    parser.add_argument("--size", choices=VOCABS, help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=VARIANTS, help=argparse.SUPPRESS)
    parser.add_argument("--round-id", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    assets, dest = args.assets.resolve(), args.output_dir.resolve()
    if args.worker == "prepare":
        return prepare(assets, dest / args.size, args.size)
    if args.worker == "trial":
        return run_trial(assets, dest / args.size,args.size,args.variant,args.round_id)
    if len(os.sched_getaffinity(0)) < 4:
        parser.error("Four available logical CPUs are required")
    dest.mkdir(parents=True,exist_ok=False)
    cmd = [sys.executable,"-B","-u","-m",MODULE,"--assets",str(assets),"--output-dir",str(dest)]
    result = dict(commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                  started_utc=datetime.now(timezone.utc).isoformat(), cases={})
    for size in (("small",) if args.small_only else ("small","real")):
        subprocess.run(cmd+["--worker","prepare","--size",size],check=True,cwd=ROOT)
        rows=[]
        for rep in range(4):
            for variant in (VARIANTS[::-1] if rep==2 else VARIANTS):
                print(f"Start {size}/{variant}/{rep} at {datetime.now(timezone.utc).isoformat()}",flush=True)
                subprocess.run(cmd+["--worker","trial","--size",size,"--variant",variant,"--round-id",str(rep)],check=True,cwd=ROOT)
                rows.append(json.loads((dest/size/variant/f"run-{rep}/measurement.json").read_text()))
        summary={}
        for variant in VARIANTS:
            measured = [r for r in rows if r["variant"]==variant and r["phase"]=="measure"]
            summary[variant]={k:statistics.median(r[k] for r in measured) for k in
                              ("guest_s","forward_s","qemu_s","qemu_peak_bytes","combined_peak_bytes")}
            if size == "small":
                summary[variant]["qemu_highwater_bytes"] = statistics.median(r["qemu_highwater_bytes"] for r in measured)
        result["cases"][size] = dict(setup=json.loads((dest/size/"setup.json").read_text()),runs=rows,summary=summary)
        write_json(dest/"result.json",result)
        print(f"{size}: completed all numeric checks",flush=True)


if __name__=="__main__":
    main()
