"""The diagnosis must preserve evidence boundaries and FP32 execution."""
import copy

import numpy as np
import pytest

from probes.w3_qwen3_full import localize as mod


def test_localizer_preserves_historical_arithmetic_mode():
    from scratchv.verification.fp32_reference import profile
    assert mod.producer_fp32_mode({}) == "native"
    record = {"fp32_mode": "reference", "ir": {"fp32_mode": "reference", "fp32_profile": profile()}}
    assert mod.producer_fp32_mode(record) == "reference"
    record["ir"]["fp32_profile"]["matmul_k_block"] = 64
    with pytest.raises(ValueError, match="profile"):
        mod.producer_fp32_mode(record)


def test_localizer_accepts_explicit_native_profile():
    assert mod.producer_fp32_mode({"fp32_mode": "native", "ir": {
        "fp32_mode": "native", "fp32_profile": {"name": "numpy-native"}}}) == "native"


@pytest.mark.parametrize("profile", [None, {}, {"name": "unknown"},
                                     {"name": "numpy-fp32-reference-v1"}])
def test_localizer_rejects_inconsistent_native_profile(profile):
    with pytest.raises(ValueError, match="profile"):
        mod.producer_fp32_mode({"ir": {"fp32_mode": "native", "fp32_profile": profile}})


@pytest.mark.parametrize("record", [
    {"ir": {"fp32_mode": "other"}},
    {"fp32_mode": "reference", "ir": {"fp32_mode": "native"}},
])
def test_localizer_rejects_unknown_or_inconsistent_arithmetic_modes(record):
    with pytest.raises(ValueError, match="FP32"):
        mod.producer_fp32_mode(record)


@pytest.mark.parametrize("values", [[], [0, 0], [-1], [28], [True], [0.5]])
def test_layer_selection_rejects_missing_duplicate_or_invalid(values):
    with pytest.raises(ValueError):
        mod.selected_layers(values)


def test_layer_selection_default_and_order():
    assert mod.selected_layers(None) == [0]
    assert mod.selected_layers([27, 0, 2]) == [0, 2, 27]


def reference_metadata():
    schema = [{"name": "embedding", "shape": [1, 256, 1024], "dtype": "float32"}]
    files = [{"name": "model.onnx", "sha256": "fixed"}]
    record = {"gate": "numeric:w3-full-worker", "backend": "ort", "passed": True,
              "status": "PASS", "files": files, "input": {"sha256": "input"}, "checkpoints": schema}
    return record, schema, files


@pytest.mark.parametrize("mismatch", ["assets", "inputs", "schema", "backend", "failed_worker"])
def test_reference_metadata_mismatch_is_rejected(mismatch):
    record, schema, files = reference_metadata()
    record = copy.deepcopy(record)
    if mismatch == "assets":
        record["files"][0]["sha256"] = "other"
    elif mismatch == "inputs":
        record["input"]["sha256"] = "other"
    elif mismatch == "schema":
        record["checkpoints"][0]["shape"] = [1, 1, 1024]
    elif mismatch == "backend":
        record["backend"] = "ir"
    else:
        record["passed"] = False
    with pytest.raises(ValueError):
        mod.validate_reference_metadata(record, schema, schema, files, "input")


def test_reference_metadata_requires_fixed_graph_schema():
    record, schema, files = reference_metadata()
    with pytest.raises(ValueError, match="schema"):
        mod.validate_reference_metadata(record, schema, [{"name": "forged"}], files, "input")


@pytest.mark.parametrize("other", [None, {}, {"file.py": "different"}])
def test_historical_producer_fingerprints_must_match(other):
    with pytest.raises(ValueError, match="fingerprints"):
        mod.validate_producer_pair({"source_sha256": {"file.py": "original"}}, {"source_sha256": other})


def test_historical_producer_identity_need_not_equal_new_tool_sources():
    old = {"source_sha256": {"old_worker.py": "original"}}
    mod.validate_producer_pair(old, copy.deepcopy(old))


def test_query_comparison_keeps_padding_failure():
    ref = np.zeros((1, 2, 256, 3), dtype=np.float32)
    actual = ref.copy()
    actual[0, 1, 200, 2] = np.float32(0.01)
    result = mod.compare_queries(actual, ref, 17, 2)
    assert result["valid_queries"]["passed"]
    assert not result["passed"] and not result["padding_queries"]["passed"]
    assert result["worst_index"] == [0, 1, 200, 2]


def test_diagnostic_oracle_does_not_mutate_or_promote_execution():
    a = np.arange(12, dtype=np.float32).reshape(3, 4) / 7
    b = np.arange(20, dtype=np.float32).reshape(4, 5) / 11
    result = a @ b
    before = result.copy()
    report = mod.fp64_error_oracle("MatMul", {"a": a, "b": b}, result, result)
    assert "diagnostic only" in report["scope"]
    assert a.dtype == b.dtype == result.dtype == np.float32
    np.testing.assert_array_equal(result, before)
    with pytest.raises(ValueError, match="FP32"):
        mod.fp64_error_oracle("MatMul", {"a": a, "b": b}, result.astype(np.float64), result)


def test_same_input_actual_softmax_has_no_mask_leak(tmp_path):
    query, key = np.arange(256)[:, None], np.arange(256)[None, :]
    mask = np.where((key <= query) & (key < 17), np.float32(0),
                    np.finfo(np.float32).min).reshape(1, 1, 256, 256)
    scores = np.sin(np.arange(256 * 256, dtype=np.float32)).reshape(1, 1, 256, 256) + mask
    feed = {"x": scores}
    model = mod.primitive("Softmax", feed, scores.shape)
    ir, ort, evidence = mod.run_pair(model, feed, {"result": "y"}, tmp_path / "softmax.onnx")
    assert evidence["executed_steps"] > 0
    for output in (ir["result"], ort["result"]):
        assert output.dtype == np.float32
        report = mod.probability_diagnostics(output, mask)
        assert report["masked_probability_max_abs"] == 0
        assert report["row_sum_max_abs_error"] < 1e-6
    assert mod.compare_queries(ir["result"], ort["result"], 17, 2)["passed"]


def test_constant_value_ints_is_readonly_before_borrowing(tmp_path):
    import onnx
    x = np.arange(256, dtype=np.float32).reshape(1, 256, 1)
    graph = onnx.helper.make_graph([
        onnx.helper.make_node("Constant", [], ["raw"], value_ints=[-2]),
        onnx.helper.make_node("Cast", ["raw"], ["float"], to=onnx.TensorProto.FLOAT),
        onnx.helper.make_node("Abs", ["float"], ["positive"]),
        onnx.helper.make_node("Add", ["x", "positive"], ["y"]),
    ], "control_constant", [onnx.helper.make_tensor_value_info("x", 1, x.shape)],
       [onnx.helper.make_tensor_value_info("y", 1, x.shape)])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
    model.ir_version = 10
    ir, ort, _ = mod.run_pair(model, {"x": x}, {"result": "y"}, tmp_path / "constant.onnx")
    np.testing.assert_array_equal(ir["result"], x + 2)
    np.testing.assert_array_equal(ort["result"], x + 2)


@pytest.mark.parametrize("kind", ["double", "nan"])
def test_execution_rejects_promoted_or_nonfinite_input(tmp_path, kind):
    feed = {"x": np.ones((1, 256, 4), dtype=np.float32)}
    model = mod.primitive("ReduceMean", feed, [1, 256, 1])
    if kind == "double":
        feed["x"] = feed["x"].astype(np.float64)
    else:
        feed["x"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite FP32"):
        mod.run_pair(model, feed, {"result": "y"}, tmp_path / "invalid.onnx")


def test_execution_failure_reports_failure_not_diagnostic_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "source_evidence", lambda: {"source_sha256": {}})
    def fail(_):
        raise ValueError("corrupt asset")
    monkeypatch.setattr(mod, "verify_files", fail)
    out = tmp_path / "diagnostic"
    assert mod.main(["--model-dir", str(tmp_path), "--case-dir", str(tmp_path),
                     "--output-dir", str(out)]) == 1
    import json
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "FAIL"
    assert not report["diagnostic_complete"] and not report["numeric_gate_passed"]
    assert not report["w3_exit_accepted"]


def test_interrupt_keeps_failure_report(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "source_evidence", lambda: {"source_sha256": {}})
    def interrupt(_):
        raise KeyboardInterrupt
    monkeypatch.setattr(mod, "verify_files", interrupt)
    out = tmp_path / "cancelled"
    assert mod.main(["--model-dir", str(tmp_path), "--case-dir", str(tmp_path),
                     "--output-dir", str(out)]) == 130
    import json
    report = json.loads((out / "report.json").read_text())
    assert report["interrupted"] and report["status"] == "FAIL"
    assert not report["diagnostic_complete"]
