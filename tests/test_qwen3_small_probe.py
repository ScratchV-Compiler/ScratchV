"""Checkpoint diagnostics without Torch, downloads or pretrained model weights."""

import copy
import json

import numpy as np
import onnx
import pytest
from onnx import TensorProto as T, helper, numpy_helper

from probes.w2_qwen3_small.diagnostics import (
    build_diagnostic_model, compare_outputs, tensor_diff, unpack_trace,
)
from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.verification.ir_interpreter import IRInterpreter


def small_model():
    """Include scalar/tensor checkpoints and intentional diagnostic-name collisions."""
    x = np.array([[-3, 2, -1], [4, -5, 6]], dtype=np.float32)
    nodes = [helper.make_node("Abs", ["__trace_flat_0"], ["hidden"], name="__trace_reshape_0"),
             helper.make_node("Mul", ["hidden", "__trace_flat_shape"], ["trace_pack"]),
             helper.make_node("ReduceMean", ["trace_pack"], ["mean"],
                              axes=[0, 1], keepdims=0, name="__trace_concat")]
    graph = helper.make_graph(nodes, "diagnostic_test",
                              [helper.make_tensor_value_info("__trace_flat_0", T.FLOAT, [2, 3])],
                              [helper.make_tensor_value_info("hidden", T.FLOAT, [2, 3]),
                               helper.make_tensor_value_info("trace_pack", T.FLOAT, [2, 3]),
                               helper.make_tensor_value_info("mean", T.FLOAT, [])],
                              [numpy_helper.from_array(np.array(2, np.float32), "__trace_flat_shape")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    references = {"trace_pack": np.abs(x) * np.float32(2), "hidden": np.abs(x),
                  "mean": np.array(7, dtype=np.float32)}
    return model, {"__trace_flat_0": x}, references


def test_packing_preserves_original_model_and_avoids_all_name_collisions():
    model, _, references = small_model()
    before = model.SerializeToString()
    diagnostic, schema = build_diagnostic_model(model, references)
    assert model.SerializeToString() == before
    assert diagnostic is not model
    assert [entry["name"] for entry in schema] == list(references)
    assert [entry["offset"] for entry in schema] == [0, 6, 12]
    assert [entry["size"] for entry in schema] == [6, 6, 1]
    assert [output.name for output in diagnostic.graph.output[1:]] == ["hidden", "trace_pack", "mean"]
    assert diagnostic.graph.output[0].name == "trace_pack_1"
    assert tuple(dim.dim_value for dim in diagnostic.graph.output[0].type.tensor_type.shape.dim) == (13,)
    defined = [value.name for value in diagnostic.graph.input] + [value.name for value in diagnostic.graph.initializer]
    defined += [name for node in diagnostic.graph.node for name in node.output]
    assert len(defined) == len(set(defined))
    onnx.checker.check_model(diagnostic, full_check=True)


def test_all_named_checkpoints_roundtrip_with_one_ir_run_and_independent_ort(tmp_path, monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    model, feed, references = small_model()
    diagnostic, schema = build_diagnostic_model(model, references)
    path = tmp_path / "diagnostic.onnx"
    onnx.save(diagnostic, path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    outputs = dict(zip((output.name for output in session.get_outputs()), session.run(None, feed)))
    packed_name = diagnostic.graph.output[0].name
    ort_checkpoints = unpack_trace(outputs[packed_name], schema)
    for name in references:
        np.testing.assert_array_equal(ort_checkpoints[name], outputs[name])
        np.testing.assert_array_equal(outputs[name], references[name])
    parser = ONNXParser()
    program = parser.parse(str(path))
    original = IRInterpreter.run
    calls = []

    def count(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(IRInterpreter, "run", count)
    result = IRInterpreter(program).run(feed, initializers=parser.initializers)
    checkpoints = unpack_trace(result.return_value, schema)
    assert calls == [1]
    assert result.executed_steps > len(model.graph.node)
    assert compare_outputs(checkpoints, references, list(references))["passed"]
    for name in references:
        np.testing.assert_array_equal(checkpoints[name], outputs[name])


@pytest.mark.parametrize("mutation,match", [
    ("missing", "exactly match"), ("extra", "exactly match"),
    ("reference_dtype", "FP32"), ("reference_shape", "shape disagrees"),
    ("output_dtype", "FP32"), ("dynamic", "static shape"),
    ("missing_shape", "static shape"), ("duplicate", "duplicate"),
])
def test_packing_rejects_invalid_output_contract(mutation, match):
    model, _, references = small_model()
    if mutation == "missing":
        references.pop("mean")
    elif mutation == "extra":
        references["extra"] = np.array(0, dtype=np.float32)
    elif mutation == "reference_dtype":
        references["mean"] = references["mean"].astype(np.float64)
    elif mutation == "reference_shape":
        references["mean"] = references["mean"].reshape(1)
    elif mutation == "output_dtype":
        model.graph.output[0].type.tensor_type.elem_type = T.DOUBLE
    elif mutation == "dynamic":
        model.graph.output[0].type.tensor_type.shape.dim[1].dim_param = "sequence"
    elif mutation == "missing_shape":
        model.graph.output[0].type.tensor_type.ClearField("shape")
    elif mutation == "duplicate":
        model.graph.output.append(copy.deepcopy(model.graph.output[0]))
    with pytest.raises(ValueError, match=match):
        build_diagnostic_model(model, references)


def test_unpack_preserves_scalar_and_empty_shapes_and_copies_data():
    schema = [dict(name="empty", shape=[2, 0, 3], offset=0, size=0),
              dict(name="scalar", shape=[], offset=0, size=1)]
    packed = np.array([7], dtype=np.float32)
    outputs = unpack_trace(packed, schema)
    assert outputs["empty"].shape == (2, 0, 3)
    assert outputs["scalar"].shape == ()
    outputs["scalar"][...] = 9
    assert packed[0] == 7
    empty = unpack_trace(np.empty(0, dtype=np.float32), schema[:1])
    assert empty["empty"].shape == (2, 0, 3)


@pytest.mark.parametrize("mutation", ["dtype", "rank", "length", "duplicate", "offset",
                                     "gap", "size", "negative", "boolean", "missing", "extra"])
def test_unpack_rejects_malformed_pack_or_schema(mutation):
    packed = np.arange(4, dtype=np.float32)
    schema = [dict(name="a", shape=[2], offset=0, size=2),
              dict(name="b", shape=[2], offset=2, size=2)]
    if mutation == "dtype":
        packed = packed.astype(np.float64)
    elif mutation == "rank":
        packed = packed.reshape(2, 2)
    elif mutation == "length":
        packed = packed[:3]
    elif mutation == "duplicate":
        schema[1]["name"] = "a"
    elif mutation == "offset":
        schema[1]["offset"] = 0
    elif mutation == "gap":
        schema[1]["offset"] = 3
    elif mutation == "size":
        schema[0]["size"] = 1
    elif mutation == "negative":
        schema[0]["shape"] = [-2]
    elif mutation == "boolean":
        schema[0]["offset"] = False
    elif mutation == "missing":
        schema[0].pop("size")
    elif mutation == "extra":
        schema[0]["unexpected"] = 1
    with pytest.raises(ValueError):
        unpack_trace(packed, schema)


def test_comparison_finds_first_failed_checkpoint_and_worst_element():
    expected = {name: np.zeros((2, 3), dtype=np.float32) for name in ("logits", "layer0", "layer1")}
    actual = {name: value.copy() for name, value in expected.items()}
    actual["layer0"][1, 2] = np.float32(0.25)
    actual["logits"][0, 0] = np.float32(0.5)
    report = compare_outputs(actual, expected, ["layer0", "layer1", "logits"])
    assert not report["passed"]
    assert report["first_divergence"] == "layer0"
    assert len(report["checkpoints"]) == 3
    first = report["checkpoints"][0]
    assert first["worst_index"] == [1, 2]
    assert first["max_abs"] == first["actual_value"] == 0.25
    assert first["expected_value"] == 0
    assert report["checkpoints"][1]["passed"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("change", ["missing_actual", "extra_actual", "missing_expected", "duplicate_order", "string_order"])
def test_comparison_never_silently_drops_checkpoints(change):
    actual = {"a": np.zeros(1, dtype=np.float32), "b": np.zeros(1, dtype=np.float32)}
    expected = dict(actual)
    order = ["a", "b"]
    if change == "missing_actual":
        actual.pop("b")
    elif change == "extra_actual":
        actual["c"] = actual["a"]
    elif change == "missing_expected":
        expected.pop("b")
    elif change == "duplicate_order":
        order = ["a", "a", "b"]
    elif change == "string_order":
        order = "ab"
    with pytest.raises(ValueError):
        compare_outputs(actual, expected, order)


@pytest.mark.parametrize("value,label", [(np.nan, "NaN"), (np.inf, "+Inf"), (-np.inf, "-Inf")])
@pytest.mark.parametrize("side", ["actual", "expected"])
def test_nonfinite_results_fail_with_json_safe_diagnostics(value, label, side):
    actual, expected = np.zeros((2,), np.float32), np.zeros((2,), np.float32)
    (actual if side == "actual" else expected)[1] = value
    report = tensor_diff(actual, expected)
    assert not report["passed"] and not report["finite"]
    assert report["max_abs"] is None and report["worst_index"] == [1]
    assert report[side + "_value"] == label
    json.dumps(report, allow_nan=False)


def test_shape_dtype_and_tolerance_boundaries_do_not_pass():
    assert not tensor_diff(np.array(1, np.float32), np.array([1], np.float32))["passed"]
    assert not tensor_diff(np.ones(1, np.float64), np.ones(1, np.float32))["passed"]
    assert not tensor_diff(None, np.ones(1, np.float32))["passed"]
    report = tensor_diff(np.array(1e-5, np.float64), np.array(0, np.float64), atol=1e-5)
    assert not report["passed"] and report["max_abs"] == 1e-5
    assert report["worst_index"] == []
    assert tensor_diff(np.array(0.5e-5), np.array(0.0), atol=1e-5)["passed"]


@pytest.mark.parametrize("atol", [0, -1, np.nan, np.inf, True, "1e-5", np.complex64(1e-5 + 1j)])
def test_invalid_tolerance_cannot_disable_numeric_gate(atol):
    with pytest.raises(ValueError, match="finite and positive"):
        tensor_diff(np.zeros(1), np.zeros(1), atol)


def test_empty_and_large_integer_outputs_have_explicit_correct_semantics():
    report = tensor_diff(np.empty((2, 0), np.float32), np.empty((2, 0), np.float32))
    assert report["passed"] and report["empty"]
    assert report["max_abs"] == 0 and report["worst_index"] is None
    report = tensor_diff(np.array(2**60 + 1, np.int64), np.array(2**60, np.int64))
    assert not report["passed"] and report["max_abs"] == 1
    report = tensor_diff(np.array(np.finfo(np.float64).max), np.array(-np.finfo(np.float64).max))
    assert not report["passed"] and report["max_abs"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("failure", ["numeric", "nonfinite", "exception"])
def test_cli_failures_return_nonzero_and_keep_json_evidence(tmp_path, monkeypatch, failure):
    from probes.w2_qwen3_small import run as probe

    def injected_probe(out, report, model_seed):
        report["stage"] = "numeric"
        if failure == "exception":
            raise RuntimeError("injected execution failure")
        actual = {"layer0": np.array([np.nan if failure == "nonfinite" else 0.25], np.float32)}
        expected = {"layer0": np.zeros(1, np.float32)}
        comparison = compare_outputs(actual, expected, ["layer0"])
        report["cases"] = [{"name": "injected", "valid_length": 1,
                            "ir_vs_ort": comparison, "passed": comparison["passed"]}]
        report["passed"] = comparison["passed"]

    monkeypatch.setattr(probe, "run_probe", injected_probe)
    output = tmp_path / "failed_probe"
    assert probe.main(["--output-dir", str(output)]) == 1
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert not report["passed"]
    assert report["stage"] == "numeric"
    if failure == "exception":
        assert report["error"] == "RuntimeError: injected execution failure"
    else:
        comparison = report["cases"][0]["ir_vs_ort"]
        assert comparison["first_divergence"] == "layer0"
        assert not comparison["checkpoints"][0]["passed"]
    json.dumps(report, allow_nan=False)
    assert "Result: **FAIL**" in (output / "report.md").read_text(encoding="utf-8")


def test_cli_success_returns_zero_and_refuses_to_overwrite_evidence(tmp_path, monkeypatch):
    from probes.w2_qwen3_small import run as probe

    def injected_probe(out, report, model_seed):
        report.update(passed=True, stage="complete")

    monkeypatch.setattr(probe, "run_probe", injected_probe)
    output = tmp_path / "passed_probe"
    assert probe.main(["--output-dir", str(output)]) == 0
    report_path = output / "report.json"
    before = report_path.read_bytes()
    assert json.loads(before)["passed"]
    with pytest.raises(SystemExit) as exc:
        probe.main(["--output-dir", str(output)])
    assert exc.value.code == 2
    assert report_path.read_bytes() == before
