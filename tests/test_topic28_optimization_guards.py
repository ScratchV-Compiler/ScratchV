"""Optimization must preserve actual hard-float values and fflags.

Inputs and expected bit patterns are independent of the optimizer and are
checked by Clang/LLD-linked executables running under QEMU, not opcode shape.
"""

from __future__ import annotations

import struct

import pytest

from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, OpCode
from scratchv.optimizer.constant_folding import ConstantFolder
from tests.topic28_runtime import execute_fp


LEVELS = ("none", "basic", "all")


def fp_bits(value, width):
    return int.from_bytes(struct.pack("<f" if width == 32 else "<d", value), "little")


def make_parameter_model(tmp_path, dtype, case):
    """A real function input prevents constant folding masking R4 defects."""
    onnx = pytest.importorskip("onnx")
    tensor_type = onnx.TensorProto.FLOAT if dtype == "f32" else onnx.TensorProto.DOUBLE
    if case.startswith("mul-one"):
        nodes, value = [onnx.helper.make_node("Mul", ["x", "c"], ["y"])], 1.0
    elif case.startswith("add-zero"):
        nodes, value = [onnx.helper.make_node("Add", ["x", "c"], ["y"])], 0.0
    elif case == "mul-zero-infinity":
        nodes, value = [onnx.helper.make_node("Mul", ["x", "c"], ["y"])], 0.0
    else:
        nodes, value = [
            onnx.helper.make_node("Mul", ["x", "c"], ["p"]),
            onnx.helper.make_node("Add", ["p", "x"], ["y"]),
        ], 3.0
    graph = onnx.helper.make_graph(
        nodes,
        "main",
        [onnx.helper.make_tensor_value_info("x", tensor_type, [1])],
        [onnx.helper.make_tensor_value_info("y", tensor_type, [1])],
        [onnx.helper.make_tensor("c", tensor_type, [1], [value])],
    )
    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)
    path = tmp_path / "parameter.onnx"
    onnx.save(model, path)
    return path


class TestFloatingPointOptimizationGuards:
    @pytest.mark.parametrize("level", LEVELS)
    @pytest.mark.parametrize(
        "case,dtype,source,expected,flags",
        [
            (
                "f32-input-quantization",
                "f32",
                "a = const(16777217,dtype:f32)\nb = const(-16777216,dtype:f32)\n"
                "r = add(a,b)\nreturn r",
                0x00000000,
                0,
            ),
            (
                "f32-quantization-chain",
                "f32",
                "a = const(16777217,dtype:f32)\nb = const(-16777216,dtype:f32)\n"
                "one = const(1.0,dtype:f32)\np = add(a,b)\nr = add(p,one)\nreturn r",
                0x3F800000,
                0,
            ),
            (
                "f64-rne-midpoint",
                "f64",
                "a = const(1.0,dtype:f64)\nb = const(1.1102230246251565e-16,dtype:f64)\n"
                "r = add(a,b)\nreturn r",
                0x3FF0000000000000,
                1,
            ),
            (
                "f64-rne-midpoint-chain",
                "f64",
                "a = const(1.0,dtype:f64)\nb = const(1.1102230246251565e-16,dtype:f64)\n"
                "p = add(a,b)\nr = sub(p,a)\nreturn r",
                0x0000000000000000,
                1,
            ),
        ],
    )
    def test_constants_preserve_target_precision_rounding_and_flags(
        self,
        level,
        case,
        dtype,
        source,
        expected,
        flags,
        tmp_path,
    ):
        # Value/fflags semantics belong to the optimizer, not to register
        # allocation; allocator coverage lives in the backend suites.
        result = CompilerDriver(
            CompilerConfig(
                use_extended_isel=True,
                target_isa="rv32imfd",
                target_abi="ilp32d",
                reg_alloc="greedy",
                optimize_level=level,
                verify_ir=True,
            )
        ).compile("", str(tmp_path / "compiled.s"), dsl_source=source)
        assert result.success, (case, result.errors)
        actual = execute_fp(result.output_text, tmp_path, 32 if dtype == "f32" else 64)
        assert actual == (expected, flags), (case, level, actual)

    @pytest.mark.parametrize("level", LEVELS)
    @pytest.mark.parametrize("dtype", ("f32", "f64"))
    @pytest.mark.parametrize(
        "case,argument,expected_value,flags",
        [
            ("mul-one", 2.0, 2.0, 0),
            ("add-zero", 2.0, 2.0, 0),
            ("muladd", 2.0, 8.0, 0),
            ("mul-one-negative-zero", -0.0, -0.0, 0),
            ("add-zero-negative-zero", -0.0, 0.0, 0),
            ("mul-zero-infinity", float("inf"), float("nan"), 16),
        ],
    )
    def test_parameter_optimization_preserves_bits_and_flags(
        self,
        level,
        dtype,
        case,
        argument,
        expected_value,
        flags,
        tmp_path,
    ):
        path = make_parameter_model(tmp_path, dtype, case)
        result = CompilerDriver(
            CompilerConfig(
                use_extended_isel=True,
                target_isa="rv32imfd",
                target_abi="ilp32d",
                reg_alloc="greedy",
                optimize_level=level,
                verify_ir=True,
            )
        ).compile(str(path), str(tmp_path / "compiled.s"))
        assert result.success, (case, result.errors)
        width = 32 if dtype == "f32" else 64
        expected_bits = (
            (0x7FC00000 if width == 32 else 0x7FF8000000000000)
            if case == "mul-zero-infinity"
            else fp_bits(expected_value, width)
        )
        actual = execute_fp(result.output_text, tmp_path, width, argument=argument)
        assert actual == (expected_bits, flags), (case, level, actual)

    @pytest.mark.parametrize(
        "passes",
        [
            ("ir-peephole",),
            ("muladd-fusion",),
            ("constant-folding", "ir-peephole", "muladd-fusion"),
        ],
    )
    def test_explicit_pass_selection_cannot_bypass_hard_float_guards(
        self, passes, tmp_path
    ):
        path = make_parameter_model(tmp_path, "f64", "muladd")
        result = CompilerDriver(
            CompilerConfig(
                use_extended_isel=True,
                target_isa="rv32imfd",
                target_abi="ilp32d",
                optimize_level="none",
                passes=passes,
                verify_ir=True,
            )
        ).compile(str(path), str(tmp_path / "compiled.s"))
        assert result.success, result.errors
        assert execute_fp(result.output_text, tmp_path, 64, argument=2.0) == (
            fp_bits(8.0, 64),
            0,
        )

    @pytest.mark.parametrize(
        "dtype,source,expected,flags",
        [
            (
                "f32",
                "a = const(16777217,dtype:f32)\nb = const(-16777216,dtype:f32)\n"
                "r = add(a,b)\nreturn r",
                0x00000000,
                0,
            ),
            (
                "f64",
                "a = const(1.0,dtype:f64)\nb = const(1.1102230246251565e-16,dtype:f64)\n"
                "r = add(a,b)\nreturn r",
                0x3FF0000000000000,
                1,
            ),
        ],
    )
    def test_explicit_folding_pass_uses_the_same_precision_guard(
        self,
        dtype,
        source,
        expected,
        flags,
        tmp_path,
    ):
        result = CompilerDriver(
            CompilerConfig(
                use_extended_isel=True,
                target_isa="rv32imfd",
                target_abi="ilp32d",
                optimize_level="none",
                passes=("constant-folding", "constant-folding"),
                verify_ir=True,
            )
        ).compile("", str(tmp_path / "compiled.s"), dsl_source=source)
        assert result.success, result.errors
        assert execute_fp(
            result.output_text, tmp_path, 32 if dtype == "f32" else 64
        ) == (expected, flags)
        assert [item["changes"] for item in result.stats["optimization"]["passes"]] == [
            0,
            0,
        ]

    @pytest.mark.parametrize("dtype", (DataType.FLOAT32, DataType.FLOAT64))
    @pytest.mark.parametrize(
        "opcode",
        (
            OpCode.ADD,
            OpCode.SUB,
            OpCode.MUL,
            OpCode.DIV,
            OpCode.REM,
            OpCode.MIN,
            OpCode.MAX,
            OpCode.ABS,
        ),
    )
    def test_safe_folder_does_not_mutate_floating_instruction_or_value(
        self, dtype, opcode
    ):
        builder = IRBuilder()
        builder.new_function("main")
        block = builder.new_block("entry")
        a = builder.make_const(3.0, dtype)
        b = builder.make_const(2.0, dtype)
        result = builder.make_value(dtype=dtype)
        instruction = builder._emit(
            opcode, result, [a] if opcode is OpCode.ABS else [a, b]
        )
        builder.ret(result)
        assert (
            ConstantFolder(preserve_float_semantics=True).optimize(builder.program) == 0
        )
        assert block.instructions[0] is instruction
        assert not result.is_constant and result.const_value is None

    def test_safe_folder_retains_integer_wraparound_folding(self):
        builder = IRBuilder()
        builder.new_function("main")
        block = builder.new_block("entry")
        result = builder.add(
            builder.make_const(0x7FFFFFFF, DataType.INT32),
            builder.make_const(1, DataType.INT32),
        )
        builder.ret(result)
        assert (
            ConstantFolder(preserve_float_semantics=True).optimize(builder.program) == 1
        )
        assert block.instructions[0].opcode is OpCode.LOAD_CONST
        assert result.is_constant and result.const_value == -0x80000000
