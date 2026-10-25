"""Encoded constants must win over Python float values end to end.

Covers the RV32 overflow policy of extended integer folding, the separation
between tensor and scalar initializer modes in a reused parser, and exact
bit-pattern fidelity including negative zero.
"""

import numpy as np
import pytest

from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType
from scratchv.optimizer.constant_folding import ConstantFolder
from scratchv.verification.ir_interpreter import IRInterpreter
from tests.topic28_runtime import execute_i32


class TestEncodedConstants:
    @pytest.mark.parametrize("operation", ["div(a,b)", "abs(a)"])
    @pytest.mark.parametrize("passes", [None, ("constant-folding",)])
    def test_integer_overflow_wraps_and_folds_once(self, tmp_path, operation, passes):
        source = (
            "a = const(-2147483648,dtype:i32)\n"
            "b = const(-1,dtype:i32)\n"
            f"result = {operation}\nreturn result"
        )
        compiled = CompilerDriver(CompilerConfig(
            use_extended_isel=True, verify_ir=True,
            optimize_level="all", passes=passes,
        )).compile("", dsl_source=source)
        assert compiled.success, compiled.errors
        folding = next(row for row in compiled.stats["optimization"]["passes"]
                       if row["name"] == "constant-folding")
        assert folding["changes"] == 1
        assert execute_i32(compiled.output_text, tmp_path) == -2147483648

    def test_parser_reuse_keeps_tensor_and_scalar_initializer_modes_separate(self, tmp_path):
        import onnx

        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("Add", ["x", "c"], ["y"])], "main",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
            [onnx.numpy_helper.from_array(np.array([3.0], np.float32), "c")],
        )
        path = tmp_path / "singleton.onnx"
        onnx.save(onnx.helper.make_model(graph), path)
        parser = ONNXParser()
        scalar_program = parser.parse(str(path), scalar_initializers=True)
        constant, = scalar_program.global_values
        assert constant.is_constant and constant.bit_pattern == 0x40400000
        assert constant.shape == (1,)
        tensor_program = parser.parse(str(path))
        tensor, = tensor_program.global_values
        assert not tensor.is_constant and tensor.bit_pattern is None
        actual = IRInterpreter(tensor_program).run(
            {"x": np.array([4.0], np.float32)}, initializers=parser.initializers,
        ).return_value
        np.testing.assert_array_equal(actual, np.array([7.0], np.float32))

    @pytest.mark.parametrize("definition", ["instruction", "global", "inline"])
    def test_checked_folding_and_execution_honor_encoded_constant(self, definition):
        builder = IRBuilder()
        builder.new_function("main")
        builder.new_block("entry")
        if definition == "instruction":
            encoded = builder.load_const(99.0, DataType.FLOAT32, bit_pattern=0x40000000)
        else:
            encoded = builder.make_const(99.0, DataType.FLOAT32)
            encoded.bit_pattern = 0x40000000
            if definition == "global":
                builder.program.global_values.append(encoded)
        builder.ret(builder.mul(encoded, builder.make_const(3.0)))
        before = IRInterpreter(builder.program).run({}).return_value
        assert ConstantFolder().optimize(builder.program) == 1
        after = IRInterpreter(builder.program).run({}).return_value
        assert before == after == np.float32(6.0)

    def test_checked_execution_keeps_encoded_negative_zero(self):
        builder = IRBuilder()
        builder.new_function("main")
        builder.new_block("entry")
        builder.ret(builder.load_const(0.0, DataType.FLOAT32, bit_pattern=0x80000000))
        result = IRInterpreter(builder.program).run({}).return_value
        assert result.dtype == np.float32 and result == 0 and np.signbit(result)
