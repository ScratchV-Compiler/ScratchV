"""Recompute W4 numeric evidence from retained raw arrays, without QEMU or ORT.

This proves consistency of supplied evidence only. It does not authenticate a
self-authored report or replace independent compiler/runtime reproduction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probes.w3_common import sha256_file
from probes.w3_qwen3_full.cases import CASE_NAMES, input_cases
from probes.w3_qwen3_full.worker import load_inputs
from probes.w4_qwen3_full.run import compare_logits, write_reports
from scratchv.backend.tensor_c_codegen import TensorSpec
from scratchv.ir.types import DataType
from scratchv.runtime.riscv_external import decode_completion
from scratchv.runtime.riscv_tensor import pack_inputs


SHAPE = (1, 256, 151936)
NBYTES = int(np.prod(SHAPE)) * 4
SPECS = (TensorSpec("input_ids", DataType.INT64, (1, 256)),
         TensorSpec("attention_mask", DataType.FLOAT32, (1, 1, 256, 256)))


def audit(directory):
    directory = Path(directory).resolve()
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    required = {"build:riscv-full", "smoke:qemu-full", "numeric:qemu-full-qwen3"}
    if (report.get("passed") is not True or report.get("status") != "PASS"
            or report.get("full_qemu_forward_executed") is not True
            or any(report.get("gates", {}).get(name) is not True for name in required)):
        raise ValueError("Source report does not claim all full-model gates passed")
    coverage = report.get("coverage")
    wanted = list(CASE_NAMES) if coverage == "all" else [coverage]
    if not wanted or any(name not in CASE_NAMES for name in wanted):
        raise ValueError("Invalid source coverage")
    if [case.get("name") for case in report.get("cases", [])] != wanted:
        raise ValueError("Case records do not match declared coverage")
    fixed_inputs = {name: (length, feed) for name, length, feed in input_cases()}
    results = []
    for case in report["cases"]:
        out = directory / case["name"]
        for filename, expected in (("inputs.npz", case["input_sha256"]),
                                    ("ort.npy", case["ort_sha256"]),
                                    ("qemu/output.bin", case["qemu"]["output_sha256"]),
                                    ("qemu/inputs.bin", case["qemu"]["input_sha256"])):
            if sha256_file(out / filename) != expected:
                raise ValueError(f"Retained artifact hash mismatch: {case['name']}/{filename}")
        feed, length = load_inputs(out / "inputs.npz")
        fixed_length, fixed_feed = fixed_inputs[case["name"]]
        if (type(case["valid_length"]) is not int or length != fixed_length
                or case["valid_length"] != fixed_length
                or set(feed) != set(fixed_feed)
                or any(not np.array_equal(feed[name], value) for name, value in fixed_feed.items())):
            raise ValueError("Retained input does not match the fixed input case")
        if pack_inputs(SPECS, feed) != (out / "qemu/inputs.bin").read_bytes():
            raise ValueError("Retained raw loader input differs from reference feed")
        qemu = json.loads((out / "qemu/run.json").read_text(encoding="utf-8"))
        if (json.dumps(qemu, sort_keys=True) != json.dumps(case["qemu"], sort_keys=True)
                or qemu.get("guest_completed") is not True or type(qemu.get("exit_code")) is not int
                or qemu["exit_code"] != 0 or qemu.get("passed") is not True
                or qemu.get("status") != "PASS" or qemu.get("cleanup_errors") or qemu.get("error")):
            raise ValueError("Retained run record differs or has no guest completion")
        decode_completion((out / "qemu/uart.bin").read_bytes(), NBYTES)
        if (out / "qemu/output.bin").stat().st_size != NBYTES:
            raise ValueError("QEMU output length differs from full logits contract")
        actual = np.memmap(out / "qemu/output.bin", dtype="<f4", mode="r", shape=SHAPE)
        expected = np.load(out / "ort.npy", allow_pickle=False, mmap_mode="r")
        numeric = compare_logits(actual, expected)
        if (json.dumps(numeric, sort_keys=True) != json.dumps(case["numeric"], sort_keys=True)
                or not numeric["passed"] or case.get("passed") is not True):
            raise ValueError("Recomputed numerical comparison differs or does not pass")
        results.append({"name": case["name"], "numeric": numeric})
    return {"passed": True, "status": "PASS", "stage": "complete",
            "gate": "audit:w4-retained-numeric", "gates": {"retained_numeric_consistency": True},
            "source_report_sha256": sha256_file(directory / "report.json"), "cases": results,
            "scope": "Retained input/output and numeric consistency only; not independent execution, "
                     "source authenticity, build validation, team acceptance or nightly"}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--report-dir", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path, required=True)
    args = cli.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        report = audit(args.report_dir)
    except Exception as exc:
        report = {"passed": False, "status": "FAIL", "stage": "audit", "gates": {},
                  "error": f"{type(exc).__name__}: {exc}"}
    write_reports(args.output_dir, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
