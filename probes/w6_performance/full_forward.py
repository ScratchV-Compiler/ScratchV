"""Compare full_seed_0 forwards with optional LM Head and QEMU weight loading changes."""
import argparse
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import numpy as np

from probes.w4_qwen3_full import run as full
from probes.w6_performance.lm_head import validate
from probes.w6_performance.run import collect_pair


ROOT = Path(__file__).resolve().parents[2]
POLICIES = ("original", "vocab-blocked")
SHAPE = (1, 256, 151936)
CASE = "full_seed_0"
FLAGS = ["--case", CASE, "--cores", "4", "--matmul-policy", "blocked_fma",
         "--matmul-tile", "2x4", "--matmul-rhs-layout", "column-panel",
         "--mean-policy", "two-four-lane-last-axis", "--nonlinear-policy", "fp32-reference",
         "--qemu-workspace-layout", "none", "--timeout", "1800"]
MODULE = "probes.w6_performance.full_forward"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def policy_for(arm):
    return "original" if arm == "original" else "vocab-blocked"


def measured_run(model_dir, directory, arm, transport):
    policy = policy_for(arm)
    run_dir = directory / arm / "forward"
    built = directory / policy / "build" / "executable.json"
    real_run = full.run_external
    resources = {}

    def observed(executable, inputs, run_path, **kwargs):
        if resources or Path(run_path) != run_dir / CASE / "qemu":
            raise RuntimeError("Expected exactly one complete full_seed_0 guest per worker")
        pair = dict(combined_peak_bytes=0, samples=0, read_errors=0, exit_races=0)
        stop = threading.Event()
        watch = threading.Thread(target=collect_pair, args=(stop, pair), kwargs={"uart_path": Path(run_path) / "uart.bin"}, daemon=True)
        # The file-backed arm may also spawn a paused QEMU to inspect its boot layout.
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        wall0, raw0 = time.perf_counter(), time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
        watch.start()
        try:
            output, record = real_run(executable, inputs, run_path, **kwargs)
        finally:
            stop.set()
            watch.join()
        wall = time.perf_counter()-wall0
        raw = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)-raw0
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        prior_peak = max(before.ru_maxrss * 1024, record["weight_transport"].get("children_maxrss_before_model_bytes", 0))
        highwater = after.ru_maxrss * 1024 if after.ru_maxrss * 1024 > prior_peak else None
        resources.update(**pair, qemu_weight_transport=record["qemu_weight_transport"], forward_s=record["forward_wall_s"], qemu_s=record["qemu_wall_s"],
                         qemu_sampled_peak_bytes=record["sampled_peak_qemu_rss_bytes"],
                         qemu_samples=record["qemu_rss_observed_samples"], qemu_read_errors=record["qemu_rss_sample_errors"],
                         qemu_exit_peak_bytes=highwater, child_cpu_s=(after.ru_utime-before.ru_utime)+(after.ru_stime-before.ru_stime),
                         timing_mode=record["timing_mode"], clock_delta_s=wall-raw, runner_wall_s=wall)
        if record["qemu_weight_transport"] != transport:
            raise RuntimeError("Measured transport does not match the requested configuration")
        staged = record["weight_transport"]
        resources["staging"] = {key: staged[key] for key in (
            "total_bytes", "copied_bytes", "prepare_elapsed_s", "post_validation_elapsed_s",
            "ram_bytes", "allocated_disk_bytes", "children_maxrss_before_model_bytes",
            "boot_probe", "retained", "retained_bytes") if key in staged}
        save(run_dir / "full-resources.json", resources)
        if (not pair["samples"] or record["timing_mode"] != "event" or
                abs(wall-raw) > max(0.02, wall*0.0006)):
            raise RuntimeError("Incomplete or inconsistent full forward observation")
        return output, record

    with patch.object(full, "run_external", observed):
        rc = full.main(["--model-dir", str(model_dir), "--output-dir", str(run_dir),
                        "--reuse-build", str(built), "--lm-head-policy", policy, "--qemu-weight-transport", transport, *FLAGS])
    if rc or not resources:
        raise RuntimeError(f"{policy} forward failed; inspect {run_dir}/report.json")


def compare(directory, candidate_transport="loader"):
    numeric, arms = {}, {}
    original = directory / POLICIES[0] / "forward"
    first = read(original / "report.json")
    canonical = np.load(original / CASE / "ort.npy", mmap_mode="r", allow_pickle=False)
    assert canonical.shape == SHAPE and canonical.dtype == np.float32
    measured_arms = ("original", "lm-head-only", "vocab-blocked") if candidate_transport != "loader" else POLICIES
    for arm in measured_arms:
        policy = policy_for(arm)
        path = directory / arm / "forward"
        report, stats = read(path / "report.json"), read(path / "full-resources.json")
        if not report["passed"] or [r["name"] for r in report["cases"]] != [CASE]:
            raise RuntimeError("Missing full_seed_0 result")
        for key in ("source_sha256", "weight_sha256", "model_files", "matmul_policy", "mean_policy",
                    "nonlinear_policy", "cores", "matmul_tile", "matmul_rhs_layout", "qemu_workspace_layout_requested"):
            if report[key] != first[key]:
                raise RuntimeError(f"Mismatched A/B {key}")
        if report["build"]["matmul_rhs_panels"] != first["build"]["matmul_rhs_panels"]:
            raise RuntimeError("Other RHS panel choices differ")
        if report["lm_head_policy"] != policy or report["build"].get("lm_head_policy","original") != policy:
            raise RuntimeError(f"Build/report policy mismatch for {policy}")
        with np.load(original / CASE / "inputs.npz") as a, np.load(path / CASE / "inputs.npz") as b:
            if a.files != b.files or any(not np.array_equal(a[k], b[k]) for k in a.files):
                raise RuntimeError("A/B model inputs differ")
        reference = np.load(path / CASE / "ort.npy", mmap_mode="r", allow_pickle=False)
        if (reference.shape != SHAPE or reference.dtype != np.float32 or
                not np.array_equal(canonical.view(np.uint32),reference.view(np.uint32))):
            raise RuntimeError("A/B ORT reference arrays differ")
        raw = path / CASE / "qemu/output.bin"
        if raw.stat().st_size != canonical.nbytes:
            raise RuntimeError("Incomplete logits dump")
        output = np.memmap(raw, dtype="<f4", mode="r", shape=SHAPE)
        base_output = original / CASE / "qemu/output.bin" if policy != "original" else None
        result = validate(output, canonical, base_output)
        result["bit_mismatches_vs_original"] = result.pop("bit_mismatches_vs_baseline_warmup")
        numeric[arm] = result
        build = read(directory / policy / "build/report.json")
        arms[arm] = dict(**stats, lm_head_policy=policy, workspace_bytes=build["codegen"]["workspace_bytes"],
                constant_bytes=build["codegen"]["constant_bytes"], weight_tensors=build["codegen"]["weights"],
                rhs_panels=len(report["build"]["matmul_rhs_panels"]), guest_ram_mib=report["memory_layout"]["memory_mib"],
                fma_passed=report["fma_conformance"]["passed"], tools=report["execution_tools"],
                build_seconds=report["build"]["compile_seconds"])
    a,b = (arms[k] for k in POLICIES)
    passed = all(v["finite"] and v["max_abs_vs_ort"] < 1e-3 and not v["greedy_mismatch_rows"]
                 for v in numeric.values())
    summary = dict(status="PASS" if passed else "MISMATCH", case=CASE, layers=28, shape=SHAPE,
                   git_head=first["git"]["head"], worktree_dirty=first["git"]["dirty"], flags=FLAGS,
                   python=platform.python_version(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                   trials_per_arm=1, arms=arms, numeric=numeric, ort_bitwise_equal=True, inputs_equal=True,
                   same_weight_manifest=True, same_other_rhs_panels=True,
                   ort_greedy_ids=canonical[0].argmax(axis=-1).tolist(),
                   speedup={k:a[k]/b[k] for k in ("forward_s","qemu_s")},
                   rss_reduction=1-b["combined_peak_bytes"]/a["combined_peak_bytes"])
    summary["candidate_weight_transport"] = candidate_transport
    summary["controls"] = ["lm-head-only"] if candidate_transport != "loader" else []
    summary["measurement_order"] = list(measured_arms)
    if summary["controls"]:
        control, candidate = (read(directory / arm / "forward" / CASE / "qemu" / "run.json")
                              for arm in ("lm-head-only", "vocab-blocked"))
        for field, key in (("same_candidate_elf", "elf_sha256"),
                           ("same_candidate_bundle", "weight_sha256"),
                           ("same_candidate_output", "output_sha256")):
            # These values are already recorded by the W4 runner.
            summary[field] = bool(control[key]) and control[key] == candidate[key]
            passed = passed and summary[field]
        summary["status"] = "PASS" if passed else "MISMATCH"
    save(directory / "comparison.json", summary)
    if not passed:
        raise RuntimeError("Full logits or greedy row validation failed; see comparison.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-weight-transport", choices=("loader", "file-backed-ram"), default="loader",
                        help="file-backed-ram also measures the same candidate ELF with loader as a control")
    parser.add_argument("--run-built", choices=(*POLICIES, "lm-head-only"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    dest, model = args.output_dir.resolve(), args.model_dir.resolve()
    if args.run_built:
        transport = args.candidate_weight_transport if args.run_built == "vocab-blocked" else "loader"
        return measured_run(model, dest, args.run_built, transport)
    if len(os.sched_getaffinity(0)) < 4:
        parser.error("Four logical CPUs are required")
    dest.mkdir(parents=True, exist_ok=False)
    for policy in POLICIES:
        build = dest / policy / "build"
        build.parent.mkdir()
        subprocess.run([sys.executable, "-B", "-u", "-m", "probes.w4_qwen3_full.run",
                        "--model-dir", str(model), "--output-dir", str(build), *FLAGS,
                        "--lm-head-policy", policy, "--build-only"], cwd=ROOT, check=True)
    order = ("original", "lm-head-only", "vocab-blocked") if args.candidate_weight_transport != "loader" else POLICIES
    for arm in order:
        print("Measuring", arm, flush=True)
        subprocess.run([sys.executable,"-B","-u","-m",MODULE,"--model-dir",str(model),
                        "--output-dir",str(dest),"--run-built",arm,
                        "--candidate-weight-transport",args.candidate_weight_transport],cwd=ROOT,check=True)
    compare(dest, args.candidate_weight_transport)


if __name__ == "__main__":
    main()
