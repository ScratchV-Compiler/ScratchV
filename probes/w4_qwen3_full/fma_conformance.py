"""Strict RV64 FMADD.S preflight: rounding must not depend on sticky NX.

This diagnostic deliberately writes FCSR to exercise both software/hardware
emulator paths. Production kernels must not clear flags as a workaround.
No tolerance, host BLAS, ORT result or approximate fmaf is used as its oracle.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import struct
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probes.w3_common import atomic_text, new_output_dir, sha256_file
from scratchv.backend.tensor_c_codegen import TensorCArtifact, TensorSpec
from scratchv.ir.types import DataType
from scratchv.runtime.riscv_external import build_external, run_external
from scratchv.runtime.riscv_tensor import discover_toolchain
from scratchv.runtime.weight_bundle import write_weight_bundle

CASES = (
    {"name": "rounding_sensitive", "input_bits": (0xBE58B5EB, 0x3ED3B8B4, 0xBF01BEA9),
     "expected_bits": 0xBF1825F2, "inexact": 1},
    {"name": "fused_cancellation", "input_bits": (0x3F800001, 0x3F7FFFFE, 0xBF800000),
     "expected_bits": 0xA8800000, "inexact": 0},
)
# Repeat the clean state after NX to exclude an accidental one-off/order effect.
MODES = ((0, 0), (0, 1), (0, 0), (7, 0), (7, 1), (7, 0))

SOURCE = r'''#include <stddef.h>
#include <stdint.h>
int scratchv_run_external(const void *const inputs[], const void *const weights[],
    void *workspace, size_t workspace_bytes, void *output) {
    const int64_t *bits = (const int64_t *)inputs[0];
    int64_t *rows = (int64_t *)output;
    (void)weights; (void)workspace; (void)workspace_bytes;
    for (size_t t = 0; t < 2; ++t) {
        union { uint32_t bits; float f; } a, b, c, result;
        a.bits = (uint32_t)bits[t*3]; b.bits = (uint32_t)bits[t*3+1];
        c.bits = (uint32_t)bits[t*3+2];
        for (size_t i = 0; i < 6; ++i) {
            unsigned long requested = (i == 1 || i == 4) ? 1 : 0;
            unsigned long before, after;
            if (i < 3) {
                __asm__ volatile("csrw fcsr, %3\n\tcsrr %0, fcsr\n\t"
                    "fmadd.s %1, %4, %5, %6, rne\n\tcsrr %2, fcsr"
                    : "=&r"(before), "=&f"(result.f), "=&r"(after)
                    : "r"(requested), "f"(a.f), "f"(b.f), "f"(c.f) : "memory");
            } else {
                __asm__ volatile("csrw fcsr, %3\n\tcsrr %0, fcsr\n\t"
                    "fmadd.s %1, %4, %5, %6, dyn\n\tcsrr %2, fcsr"
                    : "=&r"(before), "=&f"(result.f), "=&r"(after)
                    : "r"(requested), "f"(a.f), "f"(b.f), "f"(c.f) : "memory");
            }
            size_t offset = (t*6+i)*5;
            rows[offset] = i < 3 ? 0 : 7;
            rows[offset+1] = requested; rows[offset+2] = before;
            rows[offset+3] = result.bits; rows[offset+4] = after;
        }
    }
    return 0;
}
'''


def exact_value(bits):
    """Decode a finite binary32 bit pattern to an exact rational, no FP math."""
    sign, exponent, significand = bits >> 31, (bits >> 23) & 255, bits & 0x7FFFFF
    if exponent == 255:
        raise ValueError("Finite binary32 required")
    if exponent:
        significand |= 1 << 23
    shift = exponent - 150 if exponent else -149
    return (-1 if sign else 1) * Fraction(significand) * Fraction(2) ** shift


def verify_oracle():
    """Prove each recorded result is the unique nearest representable value."""
    for case in CASES:
        a, b, c = map(exact_value, case["input_bits"])
        exact = a*b+c
        bits = case["expected_bits"]
        difference = abs(exact - exact_value(bits))
        if not all(difference < abs(exact - exact_value(neighbor)) for neighbor in (bits-1, bits+1)):
            raise AssertionError("The fixed FMA oracle is not uniquely nearest")
        if int(difference != 0) != case["inexact"]:
            raise AssertionError("Incorrect oracle inexact flag")


def artifact():
    return TensorCArtifact(SOURCE, (TensorSpec("bits", DataType.INT64, (2, 3)),),
        TensorSpec("observations", DataType.INT64, (2, 6, 5)), 0, 0,
        function_name="scratchv_run_external", constant_storage="external")


def compare_observations(observations):
    verify_oracle()
    observations = np.asarray(observations)
    if observations.shape != (2, 6, 5) or observations.dtype != np.dtype("int64"):
        raise ValueError("FMA observations require int64 [2,6,5]")
    rows = []
    for t, case in enumerate(CASES):
        for i, (rm, initial_flags) in enumerate(MODES):
            actual_rm, requested, before, bits, after = map(int, observations[t, i])
            expected_after = initial_flags | case["inexact"]
            bits_match = bits == case["expected_bits"]
            csr_match = (actual_rm == rm and requested == initial_flags
                         and before == initial_flags and after == expected_after)
            rows.append({"case": case["name"], "repeat": i, "instruction_rm": rm,
                "requested_fflags": initial_flags, "fcsr_before": before, "fcsr_after": after,
                "expected_fcsr_after": expected_after, "actual_bits": f"0x{bits:08x}",
                "expected_bits": f"0x{case['expected_bits']:08x}",
                "bits_match": bits_match, "csr_match": csr_match,
                "passed": bits_match and csr_match})
    return {"passed": all(row["passed"] for row in rows), "criterion": "exact binary32 bits and FCSR",
            "rows": rows, "failed_rows": sum(not row["passed"] for row in rows)}


def fmadd_encodings(elf_path):
    """Record actual single-precision FMADD encodings in executable ELF sections."""
    data = Path(elf_path).read_bytes()
    if len(data) < 64 or data[:7] != b"\x7fELF\x02\x01\x01":
        raise ValueError("Expected little-endian ELF64")
    section_offset = struct.unpack_from("<Q", data, 40)[0]
    entry_size, count = struct.unpack_from("<HH", data, 58)
    if entry_size != 64 or section_offset + count * entry_size > len(data):
        raise ValueError("Invalid ELF section headers")
    result = []
    for i in range(count):
        header = section_offset + i*entry_size
        flags, address, offset, size = struct.unpack_from("<QQQQ", data, header+8)
        if not flags & 4:
            continue
        if offset+size > len(data):
            raise ValueError("Truncated executable ELF section")
        position = 0
        while position+2 <= size:
            half = struct.unpack_from("<H", data, offset+position)[0]
            length = 2 if half & 3 != 3 else 4
            if length == 4:
                if half & 31 == 31 or position+4 > size:
                    raise ValueError("Unsupported/truncated RV64 instruction")
                word = struct.unpack_from("<I", data, offset+position)[0]
                if word & 127 == 0x43 and (word >> 25) & 3 == 0:
                    result.append({"address": hex(address+position), "word": f"0x{word:08x}",
                                   "rm": (word >> 12) & 7})
            position += length
    if not {0, 7} <= {row["rm"] for row in result}:
        raise ValueError("ELF must contain both static RNE and dynamic FMADD.S")
    return result


def run_probe(out, *, cc=None, qemu=None, timeout=30.0):
    out = new_output_dir(out)
    started = time.perf_counter()
    report = {"schema_version": 1, "passed": False, "status": "FAIL", "stage": "oracle",
        "scope": "RV64 FMADD.S conformance, not model accuracy or performance",
        "cases": [{**case, "input_bits": [f"0x{x:08x}" for x in case["input_bits"]],
                   "expected_bits": f"0x{case['expected_bits']:08x}"} for case in CASES]}
    try:
        verify_oracle()
        report["stage"] = "toolchain"
        toolchain = discover_toolchain(cc=cc, qemu=qemu)
        report["tool_binary_sha256"] = {"cc": sha256_file(toolchain.cc[0]),
                                         "qemu": sha256_file(toolchain.qemu)}
        report["probe_sha256"] = sha256_file(__file__)
        model = artifact()
        write_weight_bundle(model.external_weights, model.external_initializers, out / "weights")
        report["stage"] = "build"
        executable = build_external(model, out / "weights", out / "build", toolchain, timeout=timeout)
        report["build"] = executable.evidence
        report["fmadd_encodings"] = fmadd_encodings(executable.elf_path)
        inputs = np.array([case["input_bits"] for case in CASES], dtype=np.int64)
        np.save(out / "input.npy", inputs, allow_pickle=False)
        report["stage"] = "qemu"
        observations, run_report = run_external(executable, {"bits": inputs}, out / "run", timeout=timeout)
        np.save(out / "observations.npy", observations, allow_pickle=False)
        report["run"] = run_report
        report["stage"] = "conformance"
        report["comparison"] = compare_observations(observations)
        report["passed"] = bool(report["comparison"]["passed"] and run_report["passed"])
        report["status"] = "PASS" if report["passed"] else "FAIL"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        atomic_text(out / "traceback.txt", traceback.format_exc())
    report["elapsed_s"] = time.perf_counter() - started
    atomic_text(out / "report.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cc")
    parser.add_argument("--qemu")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    report = run_probe(args.out, cc=args.cc, qemu=args.qemu, timeout=args.timeout)
    print(json.dumps({"status": report["status"], "stage": report["stage"],
                      "report": str(Path(args.out).resolve() / "report.json")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
