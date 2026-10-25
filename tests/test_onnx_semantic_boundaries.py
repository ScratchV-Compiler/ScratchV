"""Operator versions and optional attributes must not silently change ONNX."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from scratchv.frontend.onnx_parser import ONNXParseError, ONNXParser
from scratchv.verification.ir_interpreter import IRInterpreter

ort = pytest.importorskip("onnxruntime")


def _model(tmp_path, node, data, output_shape, *, arrays=None, opset=18):
    graph = helper.make_graph(
        [node], "semantic_boundary",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, data.shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        [numpy_helper.from_array(value, name) for name, value in (arrays or {}).items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=10)
    onnx.checker.check_model(model)
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    expected = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"x": data})[0]
    return path, expected


def _compare(path, data, expected):
    parser = ONNXParser()
    program = parser.parse(str(path))
    actual = IRInterpreter(program).run({"x": data}, initializers=parser.initializers).return_value
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)


@pytest.mark.parametrize("alpha,beta", [(2.0, 3.0), (-0.5, 0.25), (0.0, 2.0), (1.0, 1.0)])
@pytest.mark.parametrize("trans_a,trans_b", [(0, 0), (1, 1)])
@pytest.mark.parametrize("bias", [True, False])
def test_gemm_preserves_coefficients_transpose_and_optional_bias(tmp_path, alpha, beta, trans_a, trans_b, bias):
    data = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
    weights = np.array([[2, -1], [3, 2], [-2, 4]], np.float32)
    data = data.T.copy() if trans_a else data
    weights = weights.T.copy() if trans_b else weights
    arrays = {"w": weights}
    names = ["x", "w"]
    if bias:
        arrays["b"] = np.array([5, -3], np.float32)
        names.append("b")
    node = helper.make_node("Gemm", names, ["y"], alpha=alpha, beta=beta, transA=trans_a, transB=trans_b)
    path, expected = _model(tmp_path, node, data, [2, 2], arrays=arrays)
    _compare(path, data, expected)


@pytest.mark.parametrize("opset", [11, 12])
def test_legacy_softmax_fails_explicitly_instead_of_normalizing_only_last_axis(tmp_path, opset):
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    path, expected = _model(tmp_path, helper.make_node("Softmax", ["x"], ["y"]), data, data.shape, opset=opset)
    # Legacy default axis=1 flattens dimensions 1..rank-1 into each row.
    np.testing.assert_allclose(expected.reshape(2, -1).sum(axis=1), np.ones(2), atol=1e-6)
    with pytest.raises(ONNXParseError, match="Softmax.*opset.*13"):
        ONNXParser().parse(str(path))


@pytest.mark.parametrize("opset", [13, 18])
def test_modern_softmax_default_last_axis_matches_ort(tmp_path, opset):
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    path, expected = _model(tmp_path, helper.make_node("Softmax", ["x"], ["y"]), data, data.shape, opset=opset)
    _compare(path, data, expected)


@pytest.mark.parametrize("approximate", [None, "none"])
def test_exact_gelu_fails_explicitly_instead_of_silently_using_tanh(tmp_path, approximate):
    data = np.array([-3, -2, -1, 0, 1, 2, 3], np.float32)
    attrs = {} if approximate is None else {"approximate": approximate}
    path, _ = _model(tmp_path, helper.make_node("Gelu", ["x"], ["y"], **attrs), data, data.shape, opset=20)
    with pytest.raises(ONNXParseError, match="Gelu.*tanh"):
        ONNXParser().parse(str(path))


def test_gelu_explicit_tanh_matches_ort(tmp_path):
    data = np.array([-3, -2, -1, 0, 1, 2, 3], np.float32)
    path, expected = _model(tmp_path, helper.make_node("Gelu", ["x"], ["y"], approximate="tanh"), data, data.shape, opset=20)
    _compare(path, data, expected)
