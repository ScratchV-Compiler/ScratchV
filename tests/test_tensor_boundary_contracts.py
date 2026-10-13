"""Static tensor boundaries shared by the ONNX frontend, IR and C ABI."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from scratchv.backend.tensor_c_codegen import TensorCCodegen
from scratchv.frontend.onnx_parser import ONNXParseError, ONNXParser
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, Value
from scratchv.runtime.riscv_tensor import pack_inputs
from scratchv.verification.ir_interpreter import IRExecutionError, IRInterpreter


def _builder(*parameters):
    builder = IRBuilder()
    builder.new_function("main", list(parameters))
    builder.new_block("entry")
    return builder


def _neg_model(tmp_path, shape):
    graph = helper.make_graph(
        [helper.make_node("Neg", ["x"], ["y"])], "neg",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    path = tmp_path / "neg.onnx"
    onnx.save(model, path)
    return path


@pytest.mark.parametrize("shape", [["N"], [None], [1, "sequence"], None],
                         ids=["symbolic", "unknown-dimension", "mixed", "unknown-rank"])
def test_parser_rejects_unresolved_input_shape_before_static_codegen(tmp_path, shape):
    # An absent shape field means unknown rank, not a rank-zero scalar.
    with pytest.raises(ONNXParseError, match="static|dynamic|unknown"):
        ONNXParser().parse(str(_neg_model(tmp_path, shape)))


@pytest.mark.parametrize("weight_shape", [["N"], [None]])
def test_initializer_backed_graph_input_uses_fixed_initializer_shape(tmp_path, weight_shape):
    # Older exporters also list initializer weights in graph.input. ONNX
    # inference does not refine that input's symbolic/unknown dimension, but
    # this frontend binds it as a fixed initializer rather than a parameter.
    weights = np.array([1.0, 2.0], dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "weights"], ["y"])], "legacy-weights",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2]),
         helper.make_tensor_value_info("weights", TensorProto.FLOAT, weight_shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
        [numpy_helper.from_array(weights, name="weights")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.checker.check_model(model, full_check=True)
    path = tmp_path / "legacy-weights.onnx"
    onnx.save(model, path)
    parser = ONNXParser()
    program = parser.parse(str(path))
    assert [value.name for value in program.functions[0].params] == ["x"]
    assert program.global_values[0].shape == (2,)
    data = np.array([3.0, 4.0], dtype=np.float32)
    returned = IRInterpreter(program).run({"x": data}, initializers=parser.initializers).return_value
    np.testing.assert_array_equal(returned, [4.0, 6.0])
    artifact = TensorCCodegen(program, initializers=parser.initializers,
                             constant_storage="external").generate()
    assert artifact.output.shape == (2,)
    assert [spec.name for spec in artifact.inputs] == ["x"]
    assert artifact.external_weights[0].shape == (2,)


@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (1, 256)])
def test_explicit_static_scalar_zero_and_l256_shapes_survive_all_boundaries(tmp_path, shape):
    parser = ONNXParser()
    program = parser.parse(str(_neg_model(tmp_path, shape)))
    assert program.functions[0].params[0].shape == shape
    values = np.arange(np.prod(shape, dtype=int), dtype=np.float32).reshape(shape)
    result = IRInterpreter(program).run({"x": values}, initializers=parser.initializers).return_value
    assert result.shape == shape
    np.testing.assert_array_equal(result, -values)
    artifact = TensorCCodegen(program, initializers=parser.initializers,
                             constant_storage="external").generate()
    assert artifact.inputs[0].shape == artifact.output.shape == shape
    assert pack_inputs(artifact.inputs, {"x": values}) == values.tobytes()


@pytest.mark.parametrize("dtype,numpy_dtype", [(DataType.FLOAT32, np.float32),
                                              (DataType.INT64, np.int64)])
@pytest.mark.parametrize("binding", ["input", "initializer"])
@pytest.mark.parametrize("array_shape", [(1,), (2,), (1, 1)])
def test_scalar_bindings_reject_non_scalar_arrays(dtype, numpy_dtype, binding, array_shape):
    value = Value("x", dtype=dtype, shape=())
    builder = _builder(value) if binding == "input" else _builder()
    if binding == "initializer":
        builder.program.global_values.append(value)
    builder.ret(value)
    data = np.ones(array_shape, dtype=numpy_dtype)
    with pytest.raises(IRExecutionError, match="ShapeError.*expected \\(\\)"):
        IRInterpreter(builder.program).run(
            {"x": data} if binding == "input" else {},
            initializers={"x": data} if binding == "initializer" else {},
        )


@pytest.mark.parametrize("binding", ["input", "initializer"])
def test_scalar_binding_and_inferred_tensor_intermediate_remain_supported(binding):
    scalar, tensor = Value("scale", shape=()), Value("x", shape=(2, 3))
    builder = _builder(scalar, tensor) if binding == "input" else _builder(tensor)
    if binding == "initializer":
        builder.program.global_values.append(scalar)
    product = builder.mul(tensor, scalar)
    assert product.shape == ()  # Builder has not inferred this intermediate yet.
    builder.ret(product)
    value = np.array(2.0, dtype=np.float32)
    value.setflags(write=False)
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    feed = {"x": data}
    initializers = {}
    if binding == "input":
        feed["scale"] = value
    else:
        initializers["scale"] = value
    result = IRInterpreter(builder.program).run(feed, initializers=initializers,
                                               copy_initializers=False,
                                               memory_mode="last_use").return_value
    np.testing.assert_array_equal(result, data * value)
    assert result.shape == (2, 3)
    artifact = TensorCCodegen(builder.program, initializers=initializers,
                             constant_storage="external").generate()
    assert artifact.output.shape == (2, 3)
    assert product.shape == ()  # Neither consumer mutates the shared IR shape.


@pytest.mark.parametrize("shape", [(-1,), (True,), (1.5,), ("N",), None])
@pytest.mark.parametrize("binding", ["input", "initializer"])
def test_static_bindings_reject_invalid_shape_metadata(shape, binding):
    value = Value("x", shape=shape)
    builder = _builder(value) if binding == "input" else _builder()
    if binding == "initializer":
        builder.program.global_values.append(value)
    builder.ret(value)
    data = np.ones(1, dtype=np.float32)
    with pytest.raises(IRExecutionError, match="ShapeError"):
        IRInterpreter(builder.program).run(
            {"x": data} if binding == "input" else {},
            initializers={"x": data} if binding == "initializer" else {},
        )


@pytest.mark.parametrize("shape", [(), (3,), (-1,), (True,), None])
def test_explicit_function_return_shape_is_a_static_contract(shape):
    x = Value("x", shape=(2,))
    builder = _builder(x)
    builder.current_func.returns = [Value("result", shape=shape)]
    builder.ret(builder.neg(x))
    with pytest.raises(IRExecutionError, match="ShapeError"):
        IRInterpreter(builder.program).run({"x": np.ones(2, dtype=np.float32)})


@pytest.mark.parametrize("shape", [(), (2, 3)])
@pytest.mark.parametrize("declared", [False, True])
def test_matching_or_omitted_return_signature_preserves_inferred_intermediates(shape, declared):
    x = Value("x", shape=shape)
    builder = _builder(x)
    result = builder.neg(x)
    assert result.shape == ()
    builder.ret(result)
    if declared:
        builder.current_func.returns = [Value("result", shape=shape)]
    data = np.ones(shape, dtype=np.float32)
    returned = IRInterpreter(builder.program).run({"x": data}).return_value
    np.testing.assert_array_equal(returned, -data)
    assert returned.shape == shape
    assert result.shape == ()


@pytest.mark.parametrize("mode", ["wrong-dtype", "multiple-returns", "missing-value"])
def test_explicit_return_dtype_and_count_are_rejected(mode):
    x = Value("x", shape=(2,))
    builder = _builder(x)
    dtype = DataType.INT64 if mode == "wrong-dtype" else DataType.FLOAT32
    builder.current_func.returns = [Value("declared", dtype=dtype, shape=(2,))]
    if mode == "multiple-returns":
        builder.current_func.returns.append(Value("second", shape=(2,)))
    builder.ret(None if mode == "missing-value" else x)
    with pytest.raises(IRExecutionError, match="InvalidProgram"):
        IRInterpreter(builder.program).run({"x": np.ones(2, dtype=np.float32)})


@pytest.mark.parametrize("reference_shape", [(), (3,)])
def test_return_reference_shape_cannot_override_parameter_shape(reference_shape):
    x = Value("x", shape=(2,))
    builder = _builder(x)
    builder.ret(Value("x", shape=reference_shape))
    with pytest.raises(IRExecutionError, match="ShapeError"):
        IRInterpreter(builder.program).run({"x": np.ones(2, dtype=np.float32)})
