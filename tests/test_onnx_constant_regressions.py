"""Constant arithmetic preserves tensor semantics and original names stay unique."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto as T, helper, numpy_helper

from scratchv.analysis.ir_verifier import verify_ir
from scratchv.frontend.onnx_parser import ONNXParser, ONNXParseError
from scratchv.verification.ir_interpreter import IRInterpreter

ort = pytest.importorskip("onnxruntime")


def compare(tmp_path, nodes, feed, output_type, output_shape, initializers=()):
    inputs = [helper.make_tensor_value_info(
        name, helper.np_dtype_to_tensor_dtype(data.dtype), data.shape)
        for name, data in feed.items()]
    model = helper.make_model(helper.make_graph(
        nodes, "constant_regression", inputs,
        [helper.make_tensor_value_info("y", output_type, output_shape)], initializers),
        opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    parser = ONNXParser()
    program = parser.parse(str(path))
    assert verify_ir(program) == (True, [])
    actual = IRInterpreter(program).run(feed, initializers=parser.initializers).return_value
    expected = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, feed)[0]
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)
    return actual


@pytest.mark.parametrize("op,values", [
    ("Add", [16777216.0, 1.0, 1.0]),
    ("Mul", [1.5892002582550049, 1.3721024990081787, 2.7498621940612793]),
])
def test_scalar_float32_arithmetic_rounds_after_each_operation(tmp_path, op, values):
    nodes = [helper.make_node("Constant", [], [name], value_float=value)
             for name, value in zip(("a", "b", "c"), values)]
    nodes += [helper.make_node(op, ["a", "b"], ["ab"]),
              helper.make_node(op, ["ab", "c"], ["y"])]
    compare(tmp_path, nodes, {}, T.FLOAT, [])


@pytest.mark.parametrize("dtype", ["int32", "int64"])
@pytest.mark.parametrize("op,right", [("Add", 1), ("Mul", 2)])
def test_scalar_integer_arithmetic_wraps_at_declared_width(tmp_path, dtype, op, right):
    nodes = [helper.make_node("Constant", [], [name], value=numpy_helper.from_array(data))
             for name, data in (("a", np.array(np.iinfo(dtype).max, dtype)),
                                ("b", np.array(right, dtype)))]
    nodes.append(helper.make_node(op, ["a", "b"], ["y"]))
    compare(tmp_path, nodes, {}, T.INT32 if dtype == "int32" else T.INT64, [])


def test_constant_name_cannot_shadow_computed_static_shape(tmp_path):
    nodes = [helper.make_node("Constant", [], ["v_1"], value_ints=[-2, 3]),
             helper.make_node("Abs", ["v_1"], ["shape"]),
             helper.make_node("Expand", ["x", "shape"], ["y"])]
    compare(tmp_path, nodes, {"x": np.array([[1, 2, 3]], dtype="float32")}, T.FLOAT, [2, 3])


def test_scalar_constant_name_cannot_shadow_generated_load(tmp_path):
    nodes = [helper.make_node("Constant", [], ["v_1"], value_int=7),
             helper.make_node("Cast", ["v_1"], ["y"], to=T.FLOAT)]
    compare(tmp_path, nodes, {}, T.FLOAT, [])


def test_input_name_cannot_shadow_generated_result(tmp_path):
    nodes = [helper.make_node("Abs", ["v_1"], ["y"])]
    compare(tmp_path, nodes, {"v_1": np.array([-3, 1], dtype="int64")}, T.INT64, [2])


def test_initializer_name_cannot_shadow_generated_result(tmp_path):
    nodes = [helper.make_node("Abs", ["v_1"], ["y"])]
    compare(tmp_path, nodes, {}, T.INT64, [2],
            [numpy_helper.from_array(np.array([-3, 1], dtype="int64"), "v_1")])


def test_later_constant_names_are_reserved_before_translation(tmp_path):
    nodes = [helper.make_node("Abs", ["x"], ["positive"]),
             helper.make_node("Constant", [], ["v_2"], value_float=3.0),
             helper.make_node("Add", ["positive", "v_2"], ["y"])]
    compare(tmp_path, nodes, {"x": np.array([-1, 2], dtype="float32")}, T.FLOAT, [2])


def test_scalar_arithmetic_remains_usable_in_static_shape_chains(tmp_path):
    nodes = [helper.make_node("Constant", [], ["two"], value_int=2),
             helper.make_node("Constant", [], ["one"], value_int=1),
             helper.make_node("Add", ["two", "one"], ["three"]),
             helper.make_node("Mul", ["three", "two"], ["six"]),
             helper.make_node("Constant", [], ["axis"], value_ints=[0]),
             helper.make_node("Unsqueeze", ["six", "axis"], ["shape"]),
             helper.make_node("Reshape", ["x", "shape"], ["y"])]
    compare(tmp_path, nodes, {"x": np.arange(6, dtype="float32").reshape(2, 3)}, T.FLOAT, [6])


@pytest.mark.parametrize("input_shape,requested_shape", [
    ((), (2**32, 2**32)),
    ((4096, 1), (1, 4096)),
])
def test_static_expand_size_guard_precedes_evaluation(monkeypatch, input_shape, requested_shape):
    parser = ONNXParser()
    parser.builder.new_function("guard")
    parser.builder.new_block()
    source = parser._bind_constant("source", np.ones(input_shape, dtype="int64"))
    result = parser.builder.expand(source, requested_shape)
    parser._producers[result.name] = parser.builder.current_block.instructions[-1]

    def reject_evaluation(*args):
        pytest.fail("oversized constant expression reached NumPy evaluation")

    monkeypatch.setattr("scratchv.verification.ir_numpy_ops.compute", reject_evaluation)
    with pytest.raises(ONNXParseError, match="exceeds 4096 elements"):
        parser._constant_array(result)
