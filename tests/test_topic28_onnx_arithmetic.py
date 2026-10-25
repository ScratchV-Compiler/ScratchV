"""ONNX constant arithmetic and variadic scalar Min/Max.

Every generated model is checked by the installed ONNX checker. Execution
tests compare binary results and fflags, not merely compilation or asm text.
"""

from __future__ import annotations

import struct

import pytest

from scratchv.analysis.ir_verifier import verify_ir
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.frontend.onnx_parser import ONNXParseError, ONNXParser
from scratchv.ir.types import DataType, OpCode
from scratchv.optimizer.constant_folding import ConstantFolder


onnx = pytest.importorskip("onnx")
TYPES = (("FLOAT", DataType.FLOAT32, 32), ("DOUBLE", DataType.FLOAT64, 64))


def _model(tmp_path, nodes, constants, tensor_type, *, parameter=False):
    graph = onnx.helper.make_graph(
        nodes,
        "main",
        [onnx.helper.make_tensor_value_info("x", tensor_type, [])] if parameter else [],
        [onnx.helper.make_tensor_value_info("y", tensor_type, [])],
        [
            onnx.helper.make_tensor(name, tensor_type, [], [value])
            for name, value in constants.items()
        ],
    )
    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)
    path = tmp_path / "review.onnx"
    onnx.save(model, path)
    return path


def _instructions(program):
    return [
        instruction
        for function in program.functions
        for block in function.blocks
        for instruction in block.instructions
    ]


def _bits(value, width):
    return int.from_bytes(struct.pack("<f" if width == 32 else "<d", value), "little")


def _compile(path, tmp_path, mode, optimization="none"):
    result = CompilerDriver(
        CompilerConfig(
            use_extended_isel=True,
            reg_alloc=mode,
            verify_ir=True,
            target_isa="rv32imfd",
            target_abi="ilp32d",
            optimize_level=optimization,
        )
    ).compile(str(path), str(tmp_path / "review.s"))
    assert result.success, result.errors
    return result.output_text


def _execute(assembly, tmp_path, width, expected, flags=0, argument=None):
    # Imported lazily so parser-only regressions remain runnable without tools.
    from tests.topic28_runtime import execute_fp

    actual = execute_fp(assembly, tmp_path, width, argument=argument)
    assert actual == (
        _bits(expected, width),
        flags,
    ), f"actual bits/fflags={actual}, expected={(_bits(expected, width), flags)}"


class TestONNXArithmeticSemantics:
    @pytest.mark.parametrize("type_name,ir_type,width", TYPES)
    @pytest.mark.parametrize(
        "operator,opcode", (("Add", OpCode.ADD), ("Mul", OpCode.MUL))
    )
    def test_float_constants_keep_defined_target_arithmetic(
        self,
        tmp_path,
        type_name,
        ir_type,
        width,
        operator,
        opcode,
    ):
        path = _model(
            tmp_path,
            [onnx.helper.make_node(operator, ["a", "b"], ["y"])],
            {"a": 2.0, "b": 3.0},
            getattr(onnx.TensorProto, type_name),
        )
        program = ONNXParser().parse(str(path))
        assert verify_ir(program) == (True, [])
        (operation,) = [
            instruction
            for instruction in _instructions(program)
            if instruction.opcode is opcode
        ]
        returned = _instructions(program)[-1].operands[0]
        assert returned is operation.dest
        assert returned.dtype is ir_type and not returned.is_constant

    @pytest.mark.parametrize("type_name,ir_type,width", TYPES)
    @pytest.mark.parametrize("operator,expected", (("Add", 5.0), ("Mul", 6.0)))
    @pytest.mark.parametrize("continued", (False, True))
    def test_constant_arithmetic_returns_and_remains_usable(
        self,
        tmp_path,
        type_name,
        ir_type,
        width,
        operator,
        expected,
        continued,
    ):
        nodes = [
            onnx.helper.make_node(operator, ["a", "b"], ["first" if continued else "y"])
        ]
        if continued:
            nodes.append(onnx.helper.make_node("Sub", ["first", "b"], ["y"]))
            expected -= 3.0
        path = _model(
            tmp_path, nodes, {"a": 2.0, "b": 3.0}, getattr(onnx.TensorProto, type_name)
        )
        _execute(_compile(path, tmp_path, "greedy"), tmp_path, width, expected)

    @pytest.mark.parametrize("type_name,ir_type,width", TYPES)
    @pytest.mark.parametrize("operator", ("Add", "Mul"))
    @pytest.mark.parametrize("optimization", ("none", "basic", "all"))
    def test_constant_arithmetic_keeps_hardware_rounding_and_flags(
        self,
        tmp_path,
        type_name,
        ir_type,
        width,
        operator,
        optimization,
    ):
        precision = 24 if width == 32 else 53
        if operator == "Add":
            constants = {"a": 1.0, "b": 2.0**-precision}
            expected = 1.0  # exactly halfway, RNE keeps the even mantissa
        else:
            constants = {
                "a": 1.0 + 2.0 ** -(precision - 1),
                "b": 1.0 + 2.0 ** -(precision - 1),
            }
            expected = 1.0 + 2.0 ** -(precision - 2)
        path = _model(
            tmp_path,
            [onnx.helper.make_node(operator, ["a", "b"], ["y"])],
            constants,
            getattr(onnx.TensorProto, type_name),
        )
        _execute(
            _compile(path, tmp_path, "greedy", optimization),
            tmp_path,
            width,
            expected,
            flags=1,
        )

    @pytest.mark.parametrize("operator,expected", (("Add", 5), ("Mul", 6)))
    def test_integer_constant_fold_is_deferred_to_typed_folder(
        self, tmp_path, operator, expected
    ):
        path = _model(
            tmp_path,
            [onnx.helper.make_node(operator, ["a", "b"], ["y"])],
            {"a": 2, "b": 3},
            onnx.TensorProto.INT32,
        )
        program = ONNXParser().parse(str(path))
        returned = _instructions(program)[-1].operands[0]
        assert not returned.is_constant
        assert verify_ir(program) == (True, [])
        assert ConstantFolder().optimize(program) == 1
        assert verify_ir(program) == (True, [])
        assert returned.dtype is DataType.INT32
        assert returned.is_constant and type(returned.const_value) is int
        assert returned.const_value == expected

    @pytest.mark.parametrize("type_name,ir_type,width", TYPES)
    @pytest.mark.parametrize("operator", ("Min", "Max"))
    @pytest.mark.parametrize("count", (1, 2, 3, 5))
    @pytest.mark.parametrize("parameter", (False, True))
    def test_all_variadic_inputs_and_single_input_are_preserved(
        self,
        tmp_path,
        type_name,
        ir_type,
        width,
        operator,
        count,
        parameter,
    ):
        schema = onnx.defs.get_schema(operator)
        assert schema.min_input == 1
        assert (
            schema.inputs[0].option == onnx.defs.OpSchema.FormalParameterOption.Variadic
        )
        values = (
            [5.0, 3.0, 1.0, -2.0, -4.0]
            if operator == "Min"
            else [1.0, 3.0, 5.0, 7.0, 9.0]
        )[:count]
        names = [
            "x" if parameter and index == 0 else f"c{index}" for index in range(count)
        ]
        constants = {name: value for name, value in zip(names, values) if name != "x"}
        path = _model(
            tmp_path,
            [onnx.helper.make_node(operator, names, ["y"])],
            constants,
            getattr(onnx.TensorProto, type_name),
            parameter=parameter,
        )
        program = ONNXParser().parse(str(path))
        assert verify_ir(program) == (True, [])
        expected = min(values) if operator == "Min" else max(values)
        _execute(
            _compile(path, tmp_path, "greedy"),
            tmp_path,
            width,
            expected,
            argument=values[0] if parameter else None,
        )
        opcode = OpCode.MIN if operator == "Min" else OpCode.MAX
        reductions = [
            instruction
            for instruction in _instructions(program)
            if instruction.opcode is opcode
        ]
        assert len(reductions) == count - 1
        assert all(
            len(instruction.operands) == 2
            and instruction.dest.dtype is ir_type
            and all(value.dtype is ir_type for value in instruction.operands)
            for instruction in reductions
        )
        assert _instructions(program)[-1].operands[0].dtype is ir_type

    @pytest.mark.parametrize("operator", ("min", "max"))
    def test_empty_variadic_input_has_a_clear_diagnostic(self, operator):
        parser = ONNXParser()
        with pytest.raises(ONNXParseError, match="requires at least one input"):
            getattr(parser, f"_handle_{operator}")(None, [], ["y"])
