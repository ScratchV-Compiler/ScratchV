#!/usr/bin/env python3
"""Recheck saved W3 full-model evidence without executing either model backend."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from probes.w1_qwen3_export.run import MANIFEST, verify_files
from probes.w3_common import new_output_dir, sha256_file, source_evidence, write_reports
from probes.w3_layer_diff.run import load_arrays
from probes.w3_qwen3_full import run as full
from probes.w3_qwen3_full.cases import CASE_NAMES, input_cases

MAX_JSON_BYTES = 8 * 1024**2
HASH = re.compile(r"[0-9a-f]{64}\Z")
SCOPE = ("Offline consistency and numerical recheck of saved seven-case arrays. "
         "No model execution, independent-person reproduction, current-source execution, "
         "Linux execution, Nightly run, or team acceptance is established. "
         "Hashes establish consistency, not producer authenticity; retain a trusted report hash.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    require(path.is_file() and path.stat().st_size <= MAX_JSON_BYTES,
            f"Missing or oversized JSON: {path.name}")
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    def finite_float(value):
        number = float(value)
        require(math.isfinite(number), f"Nonfinite JSON number: {value}")
        return number
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique,
                       parse_constant=reject, parse_float=finite_float)
    require(isinstance(value, dict), f"JSON object required: {path.name}")
    return value


def safe_file(root, relative):
    path = root / relative
    require(path.is_file() and path.resolve().is_relative_to(root.resolve()),
            f"Missing/external evidence file: {relative}")
    return path


def source_difference(recorded, current):
    require(isinstance(recorded, dict) and bool(recorded), "Missing producer source hashes")
    for name, digest in recorded.items():
        require(isinstance(name, str) and name and "\\" not in name and ":" not in name
                and not name.startswith("/") and all(p not in ("", ".", "..") for p in name.split("/"))
                and isinstance(digest, str) and HASH.fullmatch(digest), "Invalid producer source hash")
    return {"matches_current": recorded == current,
            "changed": [{"path": name, "recorded_sha256": recorded[name], "current_sha256": current[name]}
                        for name in sorted(recorded.keys() & current.keys()) if recorded[name] != current[name]],
            "added_since_recording": sorted(current.keys() - recorded.keys()),
            "missing_from_current": sorted(recorded.keys() - current.keys()),
            "scope": "Source mismatch is reported, not silently treated as current-source execution. "
                     "This audit recomputes stored-array comparisons only."}


def verify(directory, *, model_dir=None, expected_report_sha256=None, progress=None):
    """Reuse the runner's numerical validators; never call its worker launcher."""
    directory = Path(directory).resolve()
    top_path = safe_file(directory, "report.json")
    digest = sha256_file(top_path)
    if expected_report_sha256 is not None:
        require(isinstance(expected_report_sha256, str) and HASH.fullmatch(expected_report_sha256),
                "Expected report SHA-256 must be 64 lowercase hexadecimal characters")
        require(digest == expected_report_sha256, "Trusted report SHA-256 differs")
    top = read_json(top_path)
    require(top.get("gate") == "numeric:ir-full-qwen3" and top.get("stage") == "complete",
            "Wrong/incomplete full-model report")
    require(top.get("passed") is True and top.get("status") == "PASS"
            and top.get("coverage_complete") is True and top.get("selected_numerical_passed") is True
            and top.get("full_ir_executed") is True and top.get("w3_exit_accepted") is False,
            "Producer report is not a complete passing numerical run")
    require(top.get("optimization_level") == "none" and top.get("atol") == full.ATOL
            and top.get("rtol") == 0 and top.get("fp32_mode") in ("native", "reference"),
            "Wrong numerical contract")
    require(top.get("required_cases") == list(CASE_NAMES) and top.get("selected_cases") == list(CASE_NAMES),
            "Seven-case coverage required")
    rows = top.get("cases")
    require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows)
            and [row.get("name") for row in rows] == list(CASE_NAMES), "Missing/duplicate/reordered cases")
    require(top.get("files") == MANIFEST["files"], "Producer assets differ from pinned manifest")
    current = source_evidence()
    differences = source_difference(top.get("source_sha256"), current["source_sha256"])
    if model_dir is not None:
        require(verify_files(model_dir) == top["files"], "Local model assets differ")

    verified, profiles = [], []
    for row, (name, valid, wanted) in zip(rows, input_cases()):
        require(row.get("name") == name and type(row.get("valid_length")) is int
                and row["valid_length"] == valid and row.get("passed") is True, "Wrong case contract")
        folder = directory / name
        input_path = safe_file(directory, f"{name}/inputs.npz")
        require(row.get("input_sha256") == sha256_file(input_path), f"Altered input: {name}")
        schema = [{"name": key, "shape": list(value.shape), "dtype": str(value.dtype)}
                  for key, value in wanted.items()]
        feed = load_arrays(input_path, schema, max_bytes=1024**2)
        require(all(np.array_equal(feed[key], value) for key, value in wanted.items()),
                f"Saved inputs differ from deterministic case: {name}")
        workers = row.get("workers")
        require(isinstance(workers, list) and all(isinstance(w, dict) for w in workers)
                and [w.get("backend") for w in workers] == ["ort", "ir"], "Missing/duplicate case workers")
        for saved in workers:
            backend = saved["backend"]
            report_path = safe_file(directory, f"{name}/{backend}/report.json")
            require(saved.get("passed") is True and type(saved.get("returncode")) is int
                    and saved["returncode"] == 0 and saved.get("report_sha256") == sha256_file(report_path),
                    f"Invalid/altered worker report: {name}/{backend}")
            report = read_json(report_path)
            # Preflight strict JSON and containment before invoking the shared reader.
            for filename in ("report.md", "report.html", "logits.npy", "diagnostic_logits.npy",
                             "checkpoints.npz", "checkpoint_schema.json"):
                safe_file(directory, f"{name}/{backend}/{filename}")
            read_json(folder / backend / "checkpoint_schema.json")
            saved_profile = (full.saved_arithmetic_profile(top["fp32_mode"],
                                report.get("ir", {}).get("fp32_profile")) if backend == "ir" else None)
            evidence = full.validate_worker(folder / backend, backend, input_path,
                                            top["source_sha256"], top["files"],
                                            expected_fp32_mode=top["fp32_mode"],
                                            expected_fp32_profile=saved_profile)
            require(evidence == report, "Worker report changed during audit")
            require(evidence["input"].get("valid_length") == valid, "Wrong worker valid length")
            if backend == "ir":
                profiles.append(evidence["ir"]["fp32_profile"])
        comparison = full.compare_case(folder, valid)
        require(comparison["passed"], f"Recomputed logits fail threshold: {name}")
        require(comparison == row.get("comparison"), f"Stored comparison differs from arrays: {name}")
        verified.append({"name": name, "passed": True, "valid_length": valid,
                         "input_sha256": row["input_sha256"], "comparison": comparison})
        if progress:
            progress(name)
    require(len(profiles) == len(CASE_NAMES) and all(p == profiles[0] for p in profiles),
            "Inconsistent numerical profiles")
    invariants = full.invariants(directory, set(CASE_NAMES))
    require(len(invariants) == 4 and all(row["passed"] for row in invariants), "Recomputed invariants fail")
    require(invariants == top.get("invariants"), "Stored invariants differ from arrays")
    require(sha256_file(top_path) == digest, "Producer report changed during audit")
    return {"gate": "audit:w3-full-saved-evidence", "schema_version": 1, "status": "PASS", "passed": True,
            "scope": SCOPE, "model_executed": False, "independent_reproduction": False,
            "w3_exit_accepted": False, "producer_report_sha256": digest,
            "trusted_report_hash_checked": expected_report_sha256 is not None,
            "pinned_model_files_checked": model_dir is not None, "producer_environment": top.get("environment"),
            "producer_git": top.get("git"), "auditor": current, "source_comparison": differences,
            "fp32_mode": top["fp32_mode"], "fp32_profile": profiles[0], "cases": verified,
            "invariants": invariants, "coverage_complete": True,
            "checkpoint_threshold_scope": "Checkpoints are finite FP32 diagnostics; the strict numerical gate "
                                          "applies to every ordinary/diagnostic output logit, including padding."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, help="Optionally verify local pinned model file hashes; never execute it")
    parser.add_argument("--expected-report-sha256", help="Optional independently retained producer report digest")
    args = parser.parse_args(argv)
    out = new_output_dir(args.output_dir)
    start = time.perf_counter()
    result = {"gate": "audit:w3-full-saved-evidence", "schema_version": 1, "passed": False,
              "status": "FAIL", "scope": SCOPE, "model_executed": False,
              "independent_reproduction": False, "w3_exit_accepted": False}
    try:
        result = verify(args.evidence_dir, model_dir=args.model_dir,
                        expected_report_sha256=args.expected_report_sha256,
                        progress=lambda name: print(f"[{name}] saved arrays verified", flush=True))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_seconds"] = time.perf_counter() - start
    write_reports(out, result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
