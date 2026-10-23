"""Model-free W5 CI evidence checks using the frozen generation contract."""

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
POLICIES = {
    "harts": 4,
    "qemu_workspace_layout": "none",
    "matmul_policy": "blocked_fma",
    "mean_policy": "two-four-lane-last-axis",
    "nonlinear_policy": "fp32-reference",
    "matmul_tile": "2x4",
    "matmul_rhs_layout": "column-panel",
}


@pytest.fixture
def gate():
    spec = spec_from_file_location("w5_ci_gate_tests", ROOT / "scripts/check_w5_ci.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def reference():
    return json.loads((ROOT / "probes/w5_generation/references.json").read_text())


@pytest.fixture
def report(reference):
    cases, numeric_steps, runs = [], [], []
    for expected in reference["cases"]:
        cases.append({
            key: deepcopy(expected[key])
            for key in ("name", "prompt", "max_new_tokens", "prompt_token_ids",
                        "generated_token_ids", "generated_text", "stop_reason")
        })
        cases[-1].update(numeric_step_count=8, forward_count=8, reference_passed=True)
        for step in expected["steps"]:
            numeric_steps.append({
                "passed": True, "selected_row_passed": True, "max_abs": 0.0002,
                "atol": 0.001, "rtol": 0.0, "elements_compared": 38895616,
                "case": expected["name"], "step": step["step"],
                "valid_length": step["valid_length"],
                "qemu_token_id": step["token_id"], "ort_token_id": step["token_id"],
                "reference_token_id": step["token_id"], "reference_token_passed": True,
            })
            runs.append({
                **POLICIES, "step": len(runs), "valid_length": step["valid_length"],
                "passed": True, "status": "PASS", "guest_completed": True,
                "qemu_reaped": True, "exit_code": 0,
                "elf_sha256": "a" * 64, "weight_sha256": "b" * 64,
            })
    rows = []
    for case, bits, inexact in (
        ("rounding_sensitive", "0xbf1825f2", 1),
        ("fused_cancellation", "0xa8800000", 0),
    ):
        for repeat in range(6):
            requested = int(repeat in (1, 4))
            rows.append({
                "case": case, "repeat": repeat, "instruction_rm": 0 if repeat < 3 else 7,
                "requested_fflags": requested, "fcsr_before": requested,
                "fcsr_after": requested | inexact, "expected_fcsr_after": requested | inexact,
                "actual_bits": bits, "expected_bits": bits,
                "bits_match": True, "csr_match": True, "passed": True,
            })
    return {
        **POLICIES, "schema_version": 1, "status": "PASS", "passed": True,
        "stage": "complete", "mode": "raw", "harts_requested": 4,
        "cases": cases, "numeric_steps": numeric_steps, "runs": runs,
        "qemu_forwards": 24, "forward_attempts": 24, "full_model_gate": "PASS",
        "build": {
            **POLICIES, "sha256": "a" * 64,
            "w5_graph_audit": {"passed": True, "layer_count": 28, "tied_embeddings": True},
        },
        "fma_conformance": {
            "passed": True, "status": "PASS",
            "comparison": {"passed": True, "criterion": "exact binary32 bits and FCSR",
                           "rows": rows, "failed_rows": 0},
        },
    }


def test_complete_three_by_eight_evidence_passes_without_diagnostic_extras(gate, report, reference):
    assert len(report["cases"]) == 3
    assert len(report["runs"]) == len(report["numeric_steps"]) == 24
    # Production reports need no optional frozen_* diagnostics or output arrays.
    assert not any(key.startswith("frozen_") for key in report["numeric_steps"][0])
    assert gate.validate_report(report, reference) is None


@pytest.mark.parametrize("collection", ["cases", "numeric_steps", "runs"])
@pytest.mark.parametrize("damage", ["missing", "duplicate"])
def test_incomplete_or_duplicate_evidence_cannot_be_hidden_by_pass(gate, report, reference, collection, damage):
    if damage == "missing":
        report[collection].pop()
    else:
        report[collection][-1] = deepcopy(report[collection][0])
    with pytest.raises(ValueError):
        gate.validate_report(report, reference)


@pytest.mark.parametrize("field,value", [
    ("generated_token_ids", [12095] * 8),
    ("generated_text", "Paris"),
    ("stop_reason", "eos"),
    ("prompt_token_ids", [1, 2, 3, 4, 5]),
    ("numeric_step_count", 7),
])
def test_frozen_output_and_history_must_match(gate, report, reference, field, value):
    report["cases"][0][field] = value
    with pytest.raises(ValueError):
        gate.validate_report(report, reference)


@pytest.mark.parametrize("field,value", [
    ("max_abs", 0.001),
    ("max_abs", float("nan")),
    ("max_abs", float("inf")),
    ("atol", 0.01),
    ("rtol", 0.001),
    ("elements_compared", 151936),
    ("passed", False),
    ("qemu_token_id", 0),
    ("ort_token_id", 0),
    ("valid_length", 6),
])
def test_each_step_needs_strict_full_logits_and_frozen_token_evidence(gate, report, reference, field, value):
    report["numeric_steps"][0][field] = value
    with pytest.raises(ValueError):
        gate.validate_report(report, reference)


@pytest.mark.parametrize("field,value", [
    ("qemu_reaped", False),
    ("guest_completed", False),
    ("exit_code", 130),
    ("status", "CANCELLED"),
    ("error", "deadline expired"),
    ("cleanup_errors", ["unable to reap QEMU"]),
    ("elf_sha256", "c" * 64),
    ("weight_sha256", "c" * 64),
])
def test_cancel_failure_or_different_executable_cannot_pass(gate, report, reference, field, value):
    report["runs"][-1][field] = value
    with pytest.raises(ValueError):
        gate.validate_report(report, reference)


@pytest.mark.parametrize("location,field,value", [
    ("top", "harts", 1),
    ("top", "qemu_workspace_layout", "two-buffer-pad-4k"),
    ("top", "matmul_tile", "1x4"),
    ("build", "mean_policy", "sequential"),
    ("build", "nonlinear_policy", "libm"),
    ("run", "mean_policy", "sequential"),
    ("run", "nonlinear_policy", "libm"),
])
def test_claimed_strategy_must_match_build_and_every_run(gate, report, reference, location, field, value):
    target = report if location == "top" else report["build"] if location == "build" else report["runs"][7]
    target[field] = value
    with pytest.raises(ValueError):
        gate.validate_report(report, reference)


def test_small_graph_cannot_claim_full_model_gate(gate, report, reference):
    report["build"]["w5_graph_audit"]["layer_count"] = 2
    with pytest.raises(ValueError):
        gate.validate_report(report, reference)


@pytest.mark.parametrize("damage", ["missing", "failed"])
def test_fma_summary_cannot_mask_incomplete_or_failing_rows(gate, report, reference, damage):
    rows = report["fma_conformance"]["comparison"]["rows"]
    if damage == "missing":
        rows.pop()
    else:
        rows[7]["csr_match"] = False
    with pytest.raises(ValueError):
        gate.validate_report(report, reference)


@pytest.mark.parametrize("exit_text,expected", [("0\n", 0), ("130\n", 1), ("", 1)])
def test_cli_checks_real_exit_code_and_writes_status(gate, report, tmp_path, monkeypatch, exit_text, expected):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    exit_path = tmp_path / "exit-code.txt"
    exit_path.write_text(exit_text)
    output = tmp_path / "gate.json"
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    code = gate.main(["--report", str(report_path), "--exit-code-file", str(exit_path),
                      "--output", str(output)])
    assert code == expected
    result = json.loads(output.read_text())
    assert result["status"] == ("PASS" if expected == 0 else "FAIL")
    assert result["passed"] is (expected == 0)
    assert result["status"] in summary.read_text()


def test_missing_run_report_produces_failure_artifact(gate, tmp_path):
    exit_path = tmp_path / "exit-code.txt"
    exit_path.write_text("0\n")
    output = tmp_path / "gate.json"
    assert gate.main(["--report", str(tmp_path / "missing.json"),
                      "--exit-code-file", str(exit_path), "--output", str(output)]) == 1
    result = json.loads(output.read_text())
    assert result["status"] == "FAIL" and result["passed"] is False
