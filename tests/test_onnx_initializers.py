"""ONNX initializers must survive parsing as definitions in the shared IR."""

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from scratchv.analysis.ir_verifier import verify_ir
from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.ir.types import DataType, OpCode


def save_model(tmp_path, nodes, inputs, outputs, initializers):
    graph = helper.make_graph(nodes, "initializers", inputs, outputs, initializers)
    model = helper.make_model(graph)
    onnx.checker.check_model(model)
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    return path


@pytest.mark.parametrize("initializer_is_input", [False, True])
def test_gemm_weights_and_bias_are_global_definitions(tmp_path, initializer_is_input):
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])]
    if initializer_is_input:
        inputs.append(helper.make_tensor_value_info("weight", TensorProto.FLOAT, [2, 3]))
    path = save_model(
        tmp_path,
        [helper.make_node("Gemm", ["x", "weight", "bias"], ["y"], transB=1)],
        inputs,
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
        [numpy_helper.from_array(np.ones((2, 3), dtype=np.float32), "weight"),
         numpy_helper.from_array(np.zeros(2, dtype=np.float32), "bias")],
    )
    program = ONNXParser().parse(str(path))
    function = program.functions[0]
    globals_ = {value.name: value for value in program.global_values}
    assert list(globals_) == ["weight", "bias"]
    assert len(program.global_values) == 2
    assert [value.name for value in function.params] == ["x"]
    assert globals_["weight"].shape == (2, 3)
    assert globals_["bias"].shape == (2,)
    assert all(value.dtype is DataType.FLOAT32 for value in globals_.values())
    gemm = function.blocks[0].instructions[0]
    assert gemm.operands[1] is globals_["weight"]
    assert gemm.operands[2] is globals_["bias"]
    assert verify_ir(program) == (True, [])


@pytest.mark.parametrize("dtype,onnx_type,ir_type", [
    (np.float32, TensorProto.FLOAT, DataType.FLOAT32),
    (np.float64, TensorProto.DOUBLE, DataType.FLOAT64),
    (np.int32, TensorProto.INT32, DataType.INT32),
    (np.int64, TensorProto.INT64, DataType.INT64),
])
def test_scalar_initializer_keeps_its_numeric_type(tmp_path, dtype, onnx_type, ir_type):
    path = save_model(
        tmp_path, [], [],
        [helper.make_tensor_value_info("scalar", onnx_type, [1])],
        [numpy_helper.from_array(np.array([7], dtype=dtype), "scalar")],
    )
    program = ONNXParser().parse(str(path))
    scalar, = program.global_values
    assert scalar.name == "scalar" and scalar.dtype is ir_type
    assert scalar.is_constant and scalar.const_value == 7
    expected_type = int if dtype in (np.int32, np.int64) else float
    assert type(scalar.const_value) is expected_type
    instructions = program.functions[0].blocks[0].instructions
    assert type(instructions[0].attrs["value"]) is expected_type
    assert instructions[-1].operands[0] is scalar
    assert verify_ir(program) == (True, [])


def test_cnn_initializers_pass_existing_ir_verifier():
    path = Path(__file__).resolve().parents[1] / "models" / "graph" / "cnn.onnx"
    if not path.exists():
        pytest.skip("cnn.onnx model not found")
    model = onnx.load(path)
    program = ONNXParser().parse(str(path))
    globals_ = {value.name: value for value in program.global_values}
    assert set(globals_) == {value.name for value in model.graph.initializer}
    assert len(globals_) == len(program.global_values)
    assert verify_ir(program) == (True, [])
    for block in program.functions[0].blocks:
        for instruction in block.instructions:
            if instruction.opcode in (OpCode.CONV, OpCode.GEMM):
                for operand in instruction.operands[1:]:
                    assert operand is globals_[operand.name]
