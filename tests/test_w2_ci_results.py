"""A skipped heavy frontend job must never yield green W2 acceptance."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/check_w2_ci_results.py"
SPEC = importlib.util.spec_from_file_location("w2_ci_results", SCRIPT)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def successful_jobs():
    return {name: {"result": "success"} for name in gate.REQUIRED_JOBS}


@pytest.mark.parametrize("name", gate.REQUIRED_JOBS)
@pytest.mark.parametrize("outcome", ["skipped", "failure", "cancelled", "pending", None, True])
def test_required_job_must_actually_succeed(name, outcome):
    results = successful_jobs()
    results[name]["result"] = outcome
    with pytest.raises(ValueError, match=name):
        gate.validate_results(results)


@pytest.mark.parametrize("payload", [{}, [], None, {"llm-deploy": {"result": "success"}},
                                     {"llm-deploy": "success", "full-qwen3-frontend": {"result": "success"}}])
def test_absent_or_malformed_job_results_fail(payload):
    with pytest.raises(ValueError):
        gate.validate_results(payload)


@pytest.mark.parametrize("payload,expected", [("invalid-json", 1), ("{}", 1),
                                            (json.dumps(successful_jobs()), 0)])
def test_cli_exit_matches_gate_status(monkeypatch, payload, expected):
    monkeypatch.setenv("W2_JOB_RESULTS", payload)
    result = subprocess.run([sys.executable, "-B", str(SCRIPT)], capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == expected
    assert ("PASS:" if expected == 0 else "FAIL:") in result.stdout


def test_optional_full_ort_skip_does_not_replace_required_frontend():
    results = successful_jobs()
    results["full-qwen3-onnx"] = {"result": "skipped"}
    gate.validate_results(results)
