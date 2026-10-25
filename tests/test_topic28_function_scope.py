"""R2: independent functions may reuse every IR value name."""

import struct

import pytest

from scratchv.analysis.ir_verifier import verify_ir
from scratchv.backend.allocation import collect_register_types
from scratchv.backend.inst_select_ext import ExtendedInstructionSelector
from scratchv.backend.instruction_select import InstructionSelector
from scratchv.backend.machine_types import MachineOp, RegisterClass
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType as T, Program
from tests.topic28_runtime import observe
from tests.topic28_support import MODES, allocate, compile_ir, configuration


TYPE_PAIRS = ((T.FLOAT64, T.FLOAT64), (T.FLOAT64, T.FLOAT32))


def independent_functions(types, *, pressure=False, reserved_names=False):
    program = Program()
    for index, dtype in enumerate(types):
        builder = IRBuilder()
        builder.new_function(f"f{index}")
        builder.new_block("entry")
        pointer = builder.alloca(1, dtype)
        builder.store(pointer, builder.load_const(2 + 3 * index, dtype))
        values = (
            [builder.load_const(value, dtype) for value in range(1, 17)]
            if pressure
            else []
        )
        if pressure:
            # More simultaneously live FPR values than the allocator's pool.
            builder.br("use")
            builder.new_block("use")
        result = builder.load(pointer, dtype)
        for value in values:
            result = builder.add(result, value)
        builder.ret(result)
        if reserved_names:
            instructions = builder.program.functions[0].blocks[0].instructions
            for number, instruction in enumerate(instructions, 2):
                if instruction.dest is not None:
                    instruction.dest.name = f"__scratchv_f0_{number}"
        program.functions.extend(builder.program.functions)
    assert verify_ir(program) == (True, [])
    return program


def registers_by_function(instructions):
    result = {}
    current = None
    for instruction in instructions:
        if instruction.op is MachineOp.LABEL and instruction.label_role == "function":
            current = instruction.target
            result[current] = set()
        for operand in (instruction.dst, instruction.src1, instruction.src2):
            if operand is not None and operand.kind == "vreg":
                result[current].add(str(operand.value))
    return result


class TestFunctionValueScope:
    @pytest.mark.parametrize("types", TYPE_PAIRS)
    @pytest.mark.parametrize("reserved_names", [False, True])
    def test_selection_preserves_local_slots_and_function_value_identity(
        self, types, reserved_names
    ):
        program = independent_functions(types, reserved_names=reserved_names)
        before = program.dump()
        source_names = {
            instruction.dest.name
            for function in program.functions
            for block in function.blocks
            for instruction in block.instructions
            if instruction.dest is not None
        }
        selector = ExtendedInstructionSelector(program)
        instructions = selector.run()
        registers = registers_by_function(instructions)

        assert registers["f0"].isdisjoint(registers["f1"])
        assert len(selector.local_slots) == 2
        assert len({slot.offset for slot in selector.local_slots.values()}) == 2
        value_types = collect_register_types(instructions)
        for index, dtype in enumerate(types):
            names = registers[f"f{index}"]
            assert names.isdisjoint(source_names)
            slots = [
                slot for name, slot in selector.local_slots.items() if name in names
            ]
            assert len(slots) == 1
            width = 8 if dtype is T.FLOAT64 else 4
            assert (slots[0].size, slots[0].alignment) == (width, width)
            assert {
                value_types[name].bit_width
                for name in names
                if value_types[name].reg_class is RegisterClass.FPR
            } == {width * 8}
        assert program.dump() == before
        assert verify_ir(program) == (True, [])

    @pytest.mark.parametrize("types", TYPE_PAIRS)
    def test_naive_spill_storage_keeps_same_named_function_values_separate(self, types):
        instructions = ExtendedInstructionSelector(independent_functions(types)).run()
        registers = registers_by_function(instructions)
        allocation = allocate(instructions, "naive")
        spills = set(allocation.spill_slots)

        assert spills & registers["f0"]
        assert spills & registers["f1"]
        assert (spills & registers["f0"]).isdisjoint(spills & registers["f1"])

    def test_multifunction_selection_can_be_repeated_without_changing_identity(self):
        selector = ExtendedInstructionSelector(independent_functions(TYPE_PAIRS[0]))
        first = selector.run()
        slots = dict(selector.local_slots)

        assert selector.run() == first
        assert selector.local_slots == slots

    def test_base_selector_also_distinguishes_same_named_integer_values(self):
        instructions = InstructionSelector(
            independent_functions((T.INT32, T.INT32))
        ).run()
        registers = registers_by_function(instructions)

        assert registers["f0"].isdisjoint(registers["f1"])

    @pytest.mark.parametrize("types", TYPE_PAIRS)
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("pressure", [False, True])
    def test_same_named_functions_execute_independently(
        self, tmp_path, types, mode, pressure
    ):
        assembly = compile_ir(
            independent_functions(types, pressure=pressure),
            tmp_path,
            configuration(mode, "none"),
        )
        if pressure:
            assert "[regalloc:spill]" in assembly
        for index, dtype in enumerate(types):
            path = tmp_path / f"f{index}"
            path.mkdir()
            value = 2 + 3 * index + (sum(range(1, 17)) if pressure else 0)
            encoding = "<d" if dtype is T.FLOAT64 else "<f"
            expected = int.from_bytes(struct.pack(encoding, value), "little")
            assert observe(
                assembly, path, dtype, function=f"f{index}", separate=True
            ) == (expected, 0)
