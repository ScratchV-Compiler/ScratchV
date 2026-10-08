"""Semantic and evidence failure regressions for W3 preparation."""
import json
import numpy as np
import pytest
from probes import w3_common
from probes.w3_attention.cases import build_cases
from onnx import numpy_helper
from probes.w2_backend_ops.run import reference_case, interpret
from probes.w2_qwen3_small.diagnostics import tensor_diff

@pytest.mark.parametrize("case", build_cases(), ids=lambda c:c.name)
def test_combined_attention_ir(case, tmp_path):
    expected, parser, program = reference_case(case, tmp_path / "attention.onnx")
    assert tensor_diff(expected, case.expected, 1e-4)["passed"]
    for level in ("none","all"):
        actual = interpret(program, parser.initializers, case.feed, level)
        assert tensor_diff(actual, expected, 1e-4)["passed"]

def test_attention_independent_mask_relations():
    cases = {c.name:c for c in build_cases()}
    for case in cases.values():
        if case.relation:
            base, prefix = case.relation
            assert tensor_diff(case.expected[:,:,:prefix], cases[base].expected[:,:,:prefix], 1e-4)["passed"]
    assert not np.allclose(cases["future17"].expected[:,:,5:], cases["full17"].expected[:,:,5:])

def test_mask_oracle_rejects_dropping_a_legal_key(tmp_path):
    case = build_cases()[0]
    tensor = next(t for t in case.model.graph.initializer if t.name == "mask")
    wrong = numpy_helper.to_array(tensor).copy()
    wrong[0,0,7,3] = np.finfo(np.float32).min
    tensor.CopyFrom(numpy_helper.from_array(wrong, "mask"))
    actual, _, _ = reference_case(case, tmp_path / "bad-mask.onnx")
    assert not tensor_diff(actual, case.expected, 1e-4)["passed"]

@pytest.mark.parametrize("failed_name", ["report.md","report.html","report.json"])
def test_report_failure_cannot_leave_success(tmp_path, monkeypatch, failed_name):
    real = w3_common.atomic_text
    def broken(path, text):
        if path.name == failed_name:
            # A write can fail after content was already placed in the target.
            path.write_text(text, encoding="utf-8")
            raise OSError("injected publish failure")
        real(path,text)
    monkeypatch.setattr(w3_common,"atomic_text",broken)
    report = {"passed":True,"status":"PASS"}
    with pytest.raises(OSError):
        w3_common.write_reports(tmp_path,report)
    assert report["passed"] is False
    assert not (tmp_path/"report.md").exists()
    assert not (tmp_path/"report.html").exists()
    if (tmp_path/"report.json").exists():
        assert json.loads((tmp_path/"report.json").read_text(encoding="utf-8"))["passed"] is False

def test_new_output_cannot_reuse_empty_directory(tmp_path):
    with pytest.raises(FileExistsError):
        w3_common.new_output_dir(tmp_path)

def test_report_normal_and_nonfinite(tmp_path):
    w3_common.write_reports(tmp_path, {"passed":True,"status":"PASS","gate":"test"})
    assert json.loads((tmp_path/"report.json").read_text(encoding="utf-8"))["passed"]
    with pytest.raises(ValueError):
        w3_common.write_reports(tmp_path, {"passed":True,"status":"PASS","value":float("nan")})
    assert not (tmp_path/"report.json").exists()

def test_peak_memory_is_explicit_quantity():
    result = w3_common.process_peak_rss_bytes()
    assert result is None or (isinstance(result,int) and result>0)
