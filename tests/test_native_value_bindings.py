"""Native scalar code must resolve named IR bindings before literal hints."""

import numpy as np
import pytest

from scratchv.backend.asm_emit import AsmEmitter
from scratchv.backend.instruction_select import InstructionSelector
from scratchv.backend.register_alloc import RegisterAllocator
from scratchv.backend.riscv_encoder import RISCVAEncoder
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, Value
from scratchv.simulator.rv32_emulator import REG_ID, RV32Emulator
from scratchv.verification.ir_interpreter import IRInterpreter


def _execute_native(program, inputs=None):
    """Encode actual RV32 instructions and supply scalar entry bindings."""
    allocator = RegisterAllocator(InstructionSelector(program).run(), mode="greedy")
    assembly = AsmEmitter(allocator.run()).emit()
    machine = RV32Emulator(mem_size=4 * 1024 * 1024)
    machine.load_code(RISCVAEncoder().assemble(assembly))
    for name, value in (inputs or {}).items():
        # The scalar selector exposes virtual inputs; bind their allocated
        # entry registers as the caller would, independently of literal hints.
        if name in allocator.register_map:
            machine.regs[REG_ID[allocator.register_map[name]]] = int(value)
    assert machine.run(max_instr=128) < 128
    return machine.regs[REG_ID["a0"]]


@pytest.mark.parametrize("binding", ["parameter", "hinted_parameter", "load", "load_const", "arithmetic"])
def test_named_binding_wins_over_conflicting_literal_hint(binding):
    builder = IRBuilder()
    inputs = {}
    parameter = Value("x", DataType.INT32)
    if binding == "hinted_parameter":
        parameter.is_constant = True
        parameter.const_value = 99
    builder.new_function("main", [parameter] if "parameter" in binding else [])
    builder.new_block("entry")
    if "parameter" in binding:
        result = parameter
        inputs["x"] = np.array(7, dtype=np.int32)
    elif binding == "load":
        pointer = builder.alloca(4, DataType.INT32)
        builder.store(pointer, builder.load_const(7, DataType.INT32))
        result = builder.load(pointer)
    elif binding == "load_const":
        result = builder.load_const(7, DataType.INT32)
    else:
        result = builder.add(builder.load_const(3, DataType.INT32),
                             builder.load_const(4, DataType.INT32))
    # A reference can be a separate Value object; the name determines its
    # binding. Existing IR contracts permit this stale constant hint.
    builder.ret(Value(result.name, DataType.INT32, is_constant=True, const_value=99))
    reference = IRInterpreter(builder.program).run(inputs).return_value
    assert reference.item() == 7
    assert _execute_native(builder.program, inputs) == reference.item()


def test_unbound_literal_still_executes_as_an_immediate():
    builder = IRBuilder()
    builder.new_function("main")
    builder.new_block("entry")
    builder.ret(Value("literal", DataType.INT32, is_constant=True, const_value=23))
    assert _execute_native(builder.program) == 23


@pytest.mark.parametrize("hint", [None, 99])
def test_scalar_global_definition_wins_over_reference_metadata(hint):
    builder = IRBuilder()
    builder.program.global_values.append(Value(
        "weight", DataType.INT32, is_constant=True, const_value=7,
    ))
    builder.new_function("main")
    builder.new_block("entry")
    builder.ret(Value("weight", DataType.INT32, is_constant=hint is not None, const_value=hint))
    reference = IRInterpreter(builder.program).run({}).return_value
    assert reference.item() == 7
    assert _execute_native(builder.program) == reference.item()


def test_binding_names_are_scoped_to_the_current_function():
    builder = IRBuilder()
    parameter = Value("shared", DataType.INT32)
    builder.new_function("first", [parameter])
    builder.new_block("entry")
    builder.ret(parameter)
    builder.new_function("second")
    builder.new_block("entry")
    builder.ret(Value("shared", DataType.INT32, is_constant=True, const_value=23))
    selected = InstructionSelector(builder.program).run()
    returns = [instruction for instruction in selected if instruction.comment == "return value"]
    assert [(instruction.src1.kind, instruction.src1.value) for instruction in returns] == [
        ("vreg", "shared"), ("imm", 23),
    ]
