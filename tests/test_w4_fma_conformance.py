"""Exact FMA preflight: a one-ULP error is a failure, never an xfail."""
import json
import os

import numpy as np
import pytest

from probes.w4_qwen3_full import fma_conformance as probe
from scratchv.verification.fp32_fma import fma


def correct_observations():
    return np.array([[[rm, flags, flags, case["expected_bits"], flags | case["inexact"]]
                      for rm, flags in probe.MODES] for case in probe.CASES], dtype=np.int64)


def test_recorded_oracle_has_independent_exact_rational_proof():
    probe.verify_oracle()
    assert probe.exact_value(0xA8800000) == -probe.Fraction(1, 2**46)
    for case in probe.CASES:
        values = np.array(case["input_bits"], dtype=np.uint32).view(np.float32)
        result = fma(*values)
        assert int(result.view(np.uint32)) == case["expected_bits"]


def test_exact_results_and_sticky_flags_pass():
    report = probe.compare_observations(correct_observations())
    assert report["passed"] and report["failed_rows"] == 0
    assert len(report["rows"]) == 12


@pytest.mark.parametrize("row,column,value", [(1, 3, 0xBF1825F3), (4, 3, 0xBF1825F3),
    (0, 2, 1), (1, 4, 0), (0, 0, 7), (1, 1, 0), (3, 2, 32)])
def test_one_ulp_or_wrong_csr_is_a_failure(row, column, value):
    observations = correct_observations()
    observations[0, row, column] = value
    report = probe.compare_observations(observations)
    assert not report["passed"] and report["failed_rows"] == 1


@pytest.mark.parametrize("data", [np.zeros((2, 6, 5), np.float32), np.zeros((6, 5), np.int64)])
def test_malformed_observations_rejected(data):
    with pytest.raises(ValueError, match="int64"):
        probe.compare_observations(data)


def test_failure_report_survives_missing_toolchain(tmp_path, monkeypatch):
    def missing(**kwargs):
        raise FileNotFoundError("missing explicit QEMU")
    monkeypatch.setattr(probe, "discover_toolchain", missing)
    out = tmp_path / "evidence"
    report = probe.run_probe(out)
    assert not report["passed"] and report["status"] == "FAIL"
    assert report["stage"] == "toolchain" and "missing explicit QEMU" in report["error"]
    assert json.loads((out / "report.json").read_text())["passed"] is False
    with pytest.raises(FileExistsError):
        probe.run_probe(out)


def test_cli_propagates_conformance_failure(monkeypatch, capsys):
    monkeypatch.setattr(probe, "run_probe", lambda *args, **kwargs:
        {"passed": False, "status": "FAIL", "stage": "conformance"})
    assert probe.main(["--out", "unused-output"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "FAIL"


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="requires explicit SCRATCHV_QEMU/CC")
def test_real_rv64_fma_exact_conformance(tmp_path):
    report = probe.run_probe(tmp_path / "fma")
    assert report["passed"], json.dumps(report.get("comparison", report), indent=2)
    assert {0, 7} <= {row["rm"] for row in report["fmadd_encodings"]}
