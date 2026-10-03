"""Explicit artifact selection must fail closed before numeric integration."""
import json

import numpy as np
import pytest

from probes.w2_qwen3_small.run import input_cases, provenance
from tests.qwen_artifacts import load_qwen_artifact_inputs
from tests.test_optimizer_numeric_semantics import real_qwen_artifacts
from tests.test_qwen3_small_riscv_gate import model_evidence  # noqa: F401


@pytest.fixture
def integration_artifacts(model_evidence):
    directory, report = model_evidence
    report["provenance"] = provenance()
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    for name, _, feed in input_cases():
        np.savez(directory / f"inputs_{name}.npz", **feed)
    return directory, report


def test_unconfigured_artifact_integration_is_explicitly_optional(monkeypatch):
    monkeypatch.delenv("SCRATCHV_QWEN_ARTIFACT_DIR", raising=False)
    with pytest.raises(pytest.skip.Exception, match="SCRATCHV_QWEN_ARTIFACT_DIR"):
        real_qwen_artifacts.__wrapped__()


@pytest.mark.parametrize("value", ["", " "])
def test_explicit_empty_artifact_setting_fails(monkeypatch, value):
    monkeypatch.setenv("SCRATCHV_QWEN_ARTIFACT_DIR", value)
    with pytest.raises(pytest.fail.Exception, match="must not be empty"):
        real_qwen_artifacts.__wrapped__()


def test_explicit_missing_directory_fails_instead_of_skipping(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRATCHV_QWEN_ARTIFACT_DIR", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        real_qwen_artifacts.__wrapped__()


def test_valid_external_directory_binds_actual_inputs(integration_artifacts):
    directory, report = integration_artifacts
    model, feeds, evidence = load_qwen_artifact_inputs(directory)
    assert model == directory / "model.onnx"
    assert list(feeds) == [name for name, _, _ in input_cases()]
    assert evidence["provenance"] == report["provenance"]
    for name, _, expected in input_cases():
        for key in expected:
            np.testing.assert_array_equal(feeds[name][key], expected[key])


@pytest.mark.parametrize("fault", ["missing_report", "missing_model", "failed_report", "changed_model",
                                   "missing_sources", "changed_sources", "missing_input", "extra_input",
                                   "renamed_input", "changed_input", "changed_dtype"])
def test_bad_explicit_artifacts_cannot_pass(integration_artifacts, fault):
    directory, report = integration_artifacts
    if fault == "failed_report":
        report["passed"] = False
    elif fault == "missing_sources":
        report["provenance"].pop("source_sha256")
    elif fault == "changed_sources":
        key = next(iter(report["provenance"]["source_sha256"]))
        report["provenance"]["source_sha256"][key] = "0" * 64
    elif fault == "changed_model":
        (directory / "model.onnx").write_bytes(b"different model")
    elif fault in {"changed_input", "changed_dtype", "extra_input"}:
        name, _, feed = input_cases()[0]
        if fault == "changed_input":
            feed["input_ids"][0, 0] += 1
        elif fault == "changed_dtype":
            feed["input_ids"] = feed["input_ids"].astype(np.int32)
        else:
            name = "unexpected_case"
        np.savez(directory / f"inputs_{name}.npz", **feed)
    elif fault == "renamed_input":
        (directory / "inputs_full_seed_0.npz").rename(directory / "inputs_wrong_name.npz")
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    for key, filename in (("missing_report", "report.json"), ("missing_model", "model.onnx"),
                          ("missing_input", "inputs_full_seed_0.npz")):
        if fault == key:
            (directory / filename).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        load_qwen_artifact_inputs(directory)
