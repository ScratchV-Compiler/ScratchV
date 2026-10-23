#!/usr/bin/env python3
"""Check the existing W5 CLI report; never execute or rehash model artifacts."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
POLICIES = {"harts": 4, "matmul_policy": "blocked_fma", "matmul_tile": "2x4",
            "matmul_rhs_layout": "column-panel", "mean_policy": "two-four-lane-last-axis",
            "nonlinear_policy": "fp32-reference", "qemu_workspace_layout": "none"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_report(report, reference):
    """Require all three frozen eight-token cases and their actual QEMU evidence."""
    require(isinstance(report, dict) and isinstance(reference, dict), "Expected report objects")
    require(reference.get("status") == "frozen", "Reference is not frozen")
    cases = reference["cases"]
    require([case["name"] for case in cases] == ["france", "germany", "japan"]
            and all(case["max_new_tokens"] == len(case["generated_token_ids"]) == 8
                    for case in cases), "Expected the frozen three-by-eight contract")
    require(report.get("passed") is True and report.get("status") == "PASS"
            and report.get("stage") == "complete" and report.get("full_model_gate") == "PASS"
            and report.get("mode") == "raw" and not report.get("error"),
            "Generation did not complete the full reference gate")
    build = report.get("build", {})
    for key, value in POLICIES.items():
        require(type(report.get(key)) is type(value) and report[key] == value
                and type(build.get(key)) is type(value) and build[key] == value,
                f"Unexpected {key} in report/build")
    audit = build.get("w5_graph_audit", {})
    require(audit.get("passed") is True and type(audit.get("layer_count")) is int
            and audit["layer_count"] == 28, "Missing complete 28-layer graph audit")
    require(all(type(report.get(key)) is int and report[key] == 24
                for key in ("forward_attempts", "qemu_forwards")), "Expected 24 executed forwards")
    for key, size in (("cases", 3), ("numeric_steps", 24), ("runs", 24)):
        require(isinstance(report.get(key), list) and len(report[key]) == size,
                f"Incomplete or extra {key}")
    fma = report.get("fma_conformance", {})
    comparison = fma.get("comparison", {})
    rows = comparison.get("rows", [])
    require(fma.get("passed") is True and fma.get("status") == "PASS"
            and comparison.get("passed") is True
            and comparison.get("criterion") == "exact binary32 bits and FCSR"
            and type(comparison.get("failed_rows")) is int and comparison["failed_rows"] == 0
            and isinstance(rows, list) and len(rows) == 12
            and all(all(row.get(key) is True for key in ("passed", "bits_match", "csr_match"))
                    for row in rows), "FMA exact bit/FCSR conformance did not pass")
    # Compare existing identities only: no new hashes or extra artifact reads.
    elf, weights = build.get("sha256"), report["runs"][0].get("weight_sha256")
    require(all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                for value in (elf, weights)), "Missing existing ELF/bundle identity")
    index = 0
    for actual, frozen in zip(report["cases"], cases):
        require(actual.get("reference_passed") is True
                and all(actual.get(key) == frozen[key] for key in (
                    "name", "prompt", "max_new_tokens", "prompt_token_ids",
                    "generated_token_ids", "generated_text", "stop_reason"))
                and frozen["required_substring"] in actual["generated_text"]
                and all(type(actual.get(key)) is int and actual[key] == 8
                        for key in ("numeric_step_count", "forward_count")),
                f"Frozen output mismatch: {frozen['name']}")
        for local_step, token in enumerate(frozen["generated_token_ids"]):
            numeric, run = report["numeric_steps"][index], report["runs"][index]
            valid_length = len(frozen["prompt_token_ids"]) + local_step
            require(numeric.get("case") == frozen["name"]
                    and type(numeric.get("step")) is int and numeric["step"] == local_step
                    and type(numeric.get("valid_length")) is int
                    and numeric["valid_length"] == valid_length
                    and all(numeric.get(key) is True for key in (
                        "passed", "selected_row_passed", "reference_token_passed"))
                    and all(type(numeric.get(key)) is int and numeric[key] == token
                            for key in ("qemu_token_id", "ort_token_id", "reference_token_id")),
                    f"Wrong or missing token evidence at {frozen['name']}:{local_step}")
            error = numeric.get("max_abs")
            require(type(error) in (int, float) and math.isfinite(error) and 0 <= error < 1e-3
                    and type(numeric.get("atol")) in (int, float) and numeric["atol"] == 1e-3
                    and type(numeric.get("rtol")) in (int, float) and numeric["rtol"] == 0
                    and type(numeric.get("elements_compared")) is int
                    and numeric["elements_compared"] == 256 * 151936,
                    f"Full-logits strict tolerance failed at {frozen['name']}:{local_step}")
            require(run.get("passed") is True and run.get("status") == "PASS"
                    and run.get("guest_completed") is True and run.get("qemu_reaped") is True
                    and type(run.get("exit_code")) is int and run["exit_code"] == 0
                    and not run.get("cleanup_errors") and not run.get("error")
                    and type(run.get("step")) is int and run["step"] == index
                    and type(run.get("valid_length")) is int and run["valid_length"] == valid_length
                    and run.get("elf_sha256") == elf and run.get("weight_sha256") == weights
                    and all(type(run.get(key)) is type(POLICIES[key]) and run[key] == POLICIES[key]
                            for key in ("harts", "mean_policy", "nonlinear_policy", "qemu_workspace_layout")),
                    f"QEMU completion, cleanup or reuse failed at forward {index}")
            index += 1


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--report", type=Path, required=True)
    cli.add_argument("--reference", type=Path, default=ROOT / "probes/w5_generation/references.json")
    cli.add_argument("--exit-code-file", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args(argv)
    result = {"status": "FAIL", "passed": False, "scope": "Three frozen cases, 24 full QEMU forwards",
              "commit": os.environ.get("GITHUB_SHA"), "report": str(args.report),
              "defaults_changed": False, "raw_logits_uploaded": False}
    try:
        require(args.exit_code_file.read_text().strip() == "0", "Generation process did not exit successfully")
        report = json.loads(args.report.read_text(encoding="utf-8"))
        validate_report(report, json.loads(args.reference.read_text(encoding="utf-8")))
        result.update(status="PASS", passed=True, numeric_steps=24, cases=3,
                      max_abs=max(row["max_abs"] for row in report["numeric_steps"]),
                      elapsed_seconds=report.get("elapsed_seconds"))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    text = (f"## W5 generation {result['status']}\n\n{result['scope']}. "
            "4 hart; explicit Mean/nonlinear reference policies; padding off. "
            "This is not acceptance of the unchanged default arithmetic configuration.\n\n"
            + (f"24/24 numeric steps, 3/3 frozen outputs; max_abs={result['max_abs']:.9g}; "
               f"elapsed={result['elapsed_seconds']} s.\n" if result["passed"]
               else f"Not accepted: {result['error']}\n"))
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
