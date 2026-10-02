"""Exercise the real export and prove corrupt IR cannot produce a green gate.

The LLM workflow runs this file in the pinned CPU environment. The generic
compiler job has no torch/transformers dependency and skips this module.
"""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from probes.w2_qwen3_small import run as probe
from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.verification.ir_interpreter import IRInterpreter


def read_report(path):
    return json.loads((path / "report.json").read_text(encoding="utf-8"))


def test_official_two_layer_export_passes_all_checkpoints_and_invariants(tmp_path):
    assert probe.main(["--output-dir", str(tmp_path)]) == 0
    report = read_report(tmp_path)
    assert report["passed"] and report["stage"] == "complete"
    assert len(report["checkpoints"]) == 29
    assert len(report["cases"]) == 7
    assert len(report["invariants"]) == 6
    for case in report["cases"]:
        assert case["passed"]
        assert all(result["passed"] for result in case["ordinary_logits"].values())
        assert all(result["passed"] for result in case["attention_checks"])
    assert all(check["passed"] for check in report["invariants"])
    assert (tmp_path / "model.onnx").exists()
    assert (tmp_path / "diagnostics.onnx").exists()
    from probes.w2_qwen3_small.riscv import validated_model_artifacts

    _, schema, evidence = validated_model_artifacts(tmp_path)
    assert schema == report["checkpoints"]
    assert evidence["model_sha256"]["normal"] == report["onnx"]["model_sha256"]


@pytest.mark.parametrize("target", ["ordinary", "diagnostic"])
def test_corrupted_ir_fails_even_when_other_execution_path_passes(tmp_path, monkeypatch, target):
    original = IRInterpreter.run

    def corrupted(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        ordinary = result.return_value.ndim == 3
        if ordinary == (target == "ordinary"):
            result.return_value.flat[0] += 0.25
        return result

    monkeypatch.setattr(IRInterpreter, "run", corrupted)
    assert probe.main(["--output-dir", str(tmp_path)]) == 1
    report = read_report(tmp_path)
    assert not report["passed"]
    assert report["stage"] == "complete" and "error" not in report
    first = report["cases"][0]
    assert first["pytorch_vs_ort"]["passed"]
    assert first["ort_pack_layout"]["passed"]
    if target == "diagnostic":
        assert first["ir_vs_ort"]["first_divergence"] == "token_embedding"
        divergence = first["ir_vs_ort"]["checkpoints"][0]
        assert divergence["worst_index"] == [0, 0, 0]
        assert divergence["max_abs"] > 0.2
        assert first["ordinary_logits"]["ir_vs_ort"]["passed"]
    else:
        assert first["ir_vs_ort"]["passed"]
        assert not first["ordinary_logits"]["ir_vs_ort"]["passed"]


@pytest.mark.parametrize("stage", ["parse", "execute"])
def test_parser_and_interpreter_exceptions_fail_and_preserve_evidence(tmp_path, monkeypatch, stage):
    def broken(*args, **kwargs):
        raise RuntimeError(f"injected {stage} failure")

    monkeypatch.setattr(ONNXParser if stage == "parse" else IRInterpreter,
                        "parse" if stage == "parse" else "run", broken)
    assert probe.main(["--output-dir", str(tmp_path)]) == 1
    report = read_report(tmp_path)
    assert not report["passed"]
    assert report["error"] == f"RuntimeError: injected {stage} failure"
    assert report["stage"] == ("parse" if stage == "parse" else "numeric")
    if stage == "execute":
        assert report["current_case"] == "full_seed_0"
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "model.onnx").exists()
