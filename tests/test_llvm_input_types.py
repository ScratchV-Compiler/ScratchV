"""LLVM input rejection and verified output regressions for Topic 28 R1."""

import numpy as np
import onnx
import pytest

from scratchv.backend.llvm_codegen import LLVMCodegen
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.frontend.dsl_parser import DSLParser
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType


def compile_add(tmp_path, frontend, dtype, level):
    """Compile the same scalar constant Add through either shared frontend."""
    output = tmp_path / "out.ll"
    driver = CompilerDriver(CompilerConfig(backend="llvm", optimize_level=level))
    if frontend == "dsl":
        source = (
            f"a = const(7,dtype:{dtype.value})\n"
            f"b = const(3,dtype:{dtype.value})\n"
            "r = add(a,b)\nreturn r"
        )
        result = driver.compile("", str(output), dsl_source=source)
    else:
        numpy_type, onnx_type = {
            DataType.INT32: (np.int32, onnx.TensorProto.INT32),
            DataType.FLOAT32: (np.float32, onnx.TensorProto.FLOAT),
            DataType.FLOAT64: (np.float64, onnx.TensorProto.DOUBLE),
        }[dtype]
        initializers = [
            onnx.numpy_helper.from_array(np.asarray(value, dtype=numpy_type), name)
            for name, value in (("a", 7), ("b", 3))
        ]
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("Add", ["a", "b"], ["r"])],
            "main",
            [],
            [onnx.helper.make_tensor_value_info("r", onnx_type, [])],
            initializers,
        )
        model = onnx.helper.make_model(
            graph,
            opset_imports=[onnx.helper.make_opsetid("", 13)],
        )
        onnx.checker.check_model(model)
        model_path = tmp_path / "add.onnx"
        onnx.save(model, model_path)
        result = driver.compile(str(model_path), str(output))
    return result, output


class TestLLVMInputTypes:
    @pytest.mark.parametrize("frontend", ["dsl", "onnx"])
    @pytest.mark.parametrize("level", ["none", "basic", "all"])
    def test_llvm_rejects_i32_add_without_output(self, tmp_path, frontend, level):
        result, output = compile_add(tmp_path, frontend, DataType.INT32, level)

        assert not result.success
        assert "LLVM backend does not support load_const for i32" in result.errors[0]
        assert result.output_text == ""
        assert not output.exists()

    @pytest.mark.parametrize("operation", ["Add", "Sub", "Mul", "Div"])
    @pytest.mark.parametrize("level", ["none", "basic", "all"])
    def test_llvm_rejects_nonconstant_onnx_i32_arithmetic(
        self, tmp_path, operation, level
    ):
        # Runtime parameters prevent constant folding from hiding arithmetic lowering.
        inputs = [
            onnx.helper.make_tensor_value_info(name, onnx.TensorProto.INT32, [])
            for name in ("a", "b")
        ]
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node(operation, ["a", "b"], ["r"])],
            "main",
            inputs,
            [onnx.helper.make_tensor_value_info("r", onnx.TensorProto.INT32, [])],
        )
        model = onnx.helper.make_model(
            graph,
            opset_imports=[onnx.helper.make_opsetid("", 13)],
        )
        onnx.checker.check_model(model)
        source = tmp_path / "arithmetic.onnx"
        onnx.save(model, source)
        output = tmp_path / "out.ll"
        driver = CompilerDriver(
            CompilerConfig(
                backend="llvm",
                optimize_level=level,
                verify_ir=True,
            )
        )

        result = driver.compile(str(source), str(output))

        assert not result.success
        assert (
            f"LLVM backend does not support {operation.lower()} for i32"
            in result.errors[0]
        )
        assert result.output_text == ""
        assert not output.exists()

    def test_llvm_rejects_unused_i32_before_optimization_and_preserves_output(
        self,
        tmp_path,
        monkeypatch,
    ):
        driver = CompilerDriver(
            CompilerConfig(
                backend="llvm",
                optimize_level="all",
                verify_ir=True,
            )
        )

        def unexpected_optimization(program):
            pytest.fail("unsupported LLVM input must be rejected before optimization")

        monkeypatch.setattr(driver, "_run_optimizations", unexpected_optimization)
        output = tmp_path / "existing.ll"
        output.write_bytes(b"existing output\x00")
        result = driver.compile(
            "",
            str(output),
            dsl_source="unused = const(7,dtype:i32)\nreturn 1.0",
        )

        assert not result.success
        assert "load_const for i32" in result.errors[0]
        assert output.read_bytes() == b"existing output\x00"

    @pytest.mark.parametrize("operation", ["add", "sub", "mul", "div", "neg"])
    def test_direct_llvm_codegen_rejects_integer_arithmetic_without_constants(
        self, operation
    ):
        builder = IRBuilder()
        lhs = builder.make_value("lhs", DataType.INT32)
        rhs = builder.make_value("rhs", DataType.INT32)
        builder.new_function("main", [lhs, rhs])
        builder.new_block("entry")
        operands = [lhs] if operation == "neg" else [lhs, rhs]
        builder.ret(getattr(builder, operation)(*operands))

        with pytest.raises(ValueError, match=f"{operation} for i32"):
            LLVMCodegen(builder.program).emit()

    def test_direct_llvm_save_preserves_file_on_unsupported_constant(self, tmp_path):
        builder = IRBuilder()
        builder.new_function("main")
        builder.new_block("entry")
        builder.ret(builder.load_const(7, DataType.INT64))
        output = tmp_path / "existing.ll"
        output.write_text("existing output")

        with pytest.raises(ValueError, match="load_const for i64"):
            LLVMCodegen(builder.program).save(str(output))

        assert output.read_text() == "existing output"

    @pytest.mark.parametrize("frontend", ["dsl", "onnx"])
    @pytest.mark.parametrize("level", ["none", "basic", "all"])
    @pytest.mark.parametrize("dtype", [DataType.FLOAT32, DataType.FLOAT64])
    def test_llvm_float_add_output_parses_and_verifies(
        self, tmp_path, frontend, level, dtype
    ):
        llvm = pytest.importorskip("llvmlite.binding")
        result, output = compile_add(tmp_path, frontend, dtype, level)

        assert result.success, result.errors
        module = llvm.parse_assembly(result.output_text)
        module.verify()
        expected_type = "float" if dtype is DataType.FLOAT32 else "double"
        assert f"define {expected_type} @main()" in str(module)
        assert output.read_text() == result.output_text

    def test_llvm_integer_loop_control_still_parses_and_verifies(self):
        llvm = pytest.importorskip("llvmlite.binding")
        program = DSLParser().parse("for i = 0, 3\nendfor\nreturn 0")

        llvm.parse_assembly(LLVMCodegen(program).emit()).verify()
