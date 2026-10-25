"""Focused tests for extended integer instruction selection."""

from __future__ import annotations

import pytest

from scratchv.backend.inst_select_ext import ExtendedInstructionSelector
from scratchv.backend.instruction_select import InstructionSelector
from scratchv.backend.machine_types import MachineInstr, MachineOp, MachineOperand
from scratchv.ir.types import (
    BasicBlock,
    DataType,
    Function,
    Instruction,
    OpCode,
    Program,
    Value,
)


I32_MIN = -(1 << 31)
I32_MAX = (1 << 31) - 1
MASK32 = (1 << 32) - 1


def _program_with(instruction: Instruction, *locals_: Value) -> Program:
    block = BasicBlock("entry")
    block.add(instruction)
    function = Function("test", blocks=[block], locals=list(locals_))
    program = Program()
    program.add_function(function)
    return program


def _constant(name: str, value: int) -> Value:
    return Value(
        name=name,
        dtype=DataType.INT32,
        is_constant=True,
        const_value=value,
    )


def _signed32(value: int) -> int:
    value &= MASK32
    return value - (1 << 32) if value & (1 << 31) else value


def _execute_integer_sequence(
    instructions: list[MachineInstr], initial: dict[str, int]
) -> dict[str, int]:
    """Execute the small RV32I subset used by the selector sequences."""
    registers = {name: value & MASK32 for name, value in initial.items()}

    def read(operand: MachineOperand | None) -> int:
        assert operand is not None
        if operand.kind == "imm":
            return int(operand.value) & MASK32
        if operand.kind == "reg" and operand.value in {"x0", "zero"}:
            return 0
        return registers[str(operand.value)]

    for instruction in instructions:
        if instruction.op in {MachineOp.LABEL, MachineOp.JALR}:
            continue
        if instruction.dst is None or instruction.dst.kind == "reg":
            continue
        dst = str(instruction.dst.value)
        if instruction.op == MachineOp.LI:
            result = read(instruction.src1)
        elif instruction.op == MachineOp.SRAI:
            result = _signed32(read(instruction.src1)) >> read(instruction.src2)
        elif instruction.op == MachineOp.SLT:
            result = int(
                _signed32(read(instruction.src1)) < _signed32(read(instruction.src2))
            )
        elif instruction.op == MachineOp.SUB:
            result = read(instruction.src1) - read(instruction.src2)
        elif instruction.op == MachineOp.XOR:
            result = read(instruction.src1) ^ read(instruction.src2)
        elif instruction.op == MachineOp.AND:
            result = read(instruction.src1) & read(instruction.src2)
        else:
            continue
        registers[dst] = result & MASK32
    return registers


class TestIntegerInstructionSelection:
    @pytest.mark.parametrize(
        "selector_type", [InstructionSelector, ExtendedInstructionSelector]
    )
    def test_div_materializes_both_constant_sources(self, selector_type: type) -> None:
        lhs = _constant("lhs", -17)
        rhs = _constant("rhs", 5)
        dst = Value("quotient", DataType.INT32)
        instruction = Instruction(OpCode.DIV, dst, [lhs, rhs])

        instructions = selector_type(_program_with(instruction)).run()
        div = next(item for item in instructions if item.op == MachineOp.DIV)

        assert div.src1 is not None and div.src1.kind == "vreg"
        assert div.src2 is not None and div.src2.kind == "vreg"
        assert [item.op for item in instructions].count(MachineOp.LI) == 2

    def test_rem_materializes_constant_sources_before_register_only_rem(self) -> None:
        lhs = _constant("lhs", -17)
        rhs = _constant("rhs", 5)
        dst = Value("remainder", DataType.INT32)
        instruction = Instruction(OpCode.REM, dst, [lhs, rhs])
        selector = ExtendedInstructionSelector(_program_with(instruction))

        instructions = selector.run()
        rem = next(item for item in instructions if item.op == MachineOp.REM)

        assert rem.op == MachineOp.REM
        assert rem.src1 is not None and rem.src1.kind == "vreg"
        assert rem.src2 is not None and rem.src2.kind == "vreg"
        rem_index = instructions.index(rem)
        assert [item.op for item in instructions[rem_index - 2 : rem_index]] == [
            MachineOp.LI,
            MachineOp.LI,
        ]

    @pytest.mark.parametrize(
        ("opcode", "lhs", "rhs", "expected"),
        [
            (OpCode.MIN, I32_MIN, 0, I32_MIN),
            (OpCode.MIN, 7, -9, -9),
            (OpCode.MIN, I32_MAX, I32_MAX, I32_MAX),
            (OpCode.MAX, I32_MAX, 0, I32_MAX),
            (OpCode.MAX, -7, 9, 9),
            (OpCode.MAX, I32_MIN, I32_MIN, I32_MIN),
        ],
    )
    def test_min_max_full_width_mask_sequences(
        self, opcode: OpCode, lhs: int, rhs: int, expected: int
    ) -> None:
        left = Value("left", DataType.INT32)
        right = Value("right", DataType.INT32)
        result = Value("result", DataType.INT32)
        instruction = Instruction(opcode, result, [left, right])
        selector = ExtendedInstructionSelector(
            _program_with(instruction, left, right, result)
        )

        selector.run()
        registers = _execute_integer_sequence(
            selector._instructions, {"left": lhs, "right": rhs}
        )

        assert _signed32(registers["result"]) == expected
        mask_builders = [
            item
            for item in selector._instructions
            if item.op == MachineOp.SUB and item.src1 == MachineOperand.reg("x0")
        ]
        assert len(mask_builders) == 1

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0, 0), (19, 19), (-19, 19), (I32_MIN, I32_MIN)],
    )
    def test_abs_sequence_preserves_rv32_twos_complement_semantics(
        self, value: int, expected: int
    ) -> None:
        source = Value("source", DataType.INT32)
        result = Value("result", DataType.INT32)
        instruction = Instruction(OpCode.ABS, result, [source])
        selector = ExtendedInstructionSelector(
            _program_with(instruction, source, result)
        )

        selector.run()
        registers = _execute_integer_sequence(selector._instructions, {"source": value})

        assert _signed32(registers["result"]) == expected

    def test_selector_temporaries_do_not_collide_with_ir_names(self) -> None:
        adversarial_name = "__scratchv_abs_sign_1"
        source = Value(adversarial_name, DataType.INT32)
        result = Value("result", DataType.INT32)
        instruction = Instruction(OpCode.ABS, result, [source])
        selector = ExtendedInstructionSelector(
            _program_with(instruction, source, result)
        )

        selector.run()
        generated_defs = {
            str(item.dst.value)
            for item in selector._instructions
            if item.dst is not None and item.dst.kind == "vreg"
        }

        assert adversarial_name not in generated_defs

    def test_extended_ir_surface_does_not_advertise_idiv_or_mod(self) -> None:
        selector = ExtendedInstructionSelector(Program())

        assert "rem" in selector.supported_ops
        assert "idiv" not in selector.supported_ops
        assert "mod" not in selector.supported_ops
