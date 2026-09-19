"""Semantic regression tests for Topic 01 DSL control flow."""

from pathlib import Path

import pytest
from llvmlite import binding as llvm

from scratchv.backend.instruction_select import InstructionSelector
from scratchv.backend.asm_emit import AsmEmitter
from scratchv.backend.register_alloc import RegisterAllocator
from scratchv.backend.llvm_codegen import LLVMCodegen
from scratchv.backend.machine_types import MachineOp
from scratchv.frontend.dsl_extended import ExtendedDSLParser
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, OpCode


def _execute_llvm_scalar(source: str) -> float:
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    program = ExtendedDSLParser().parse(source)
    llvm_ir = LLVMCodegen(program).emit()
    module = llvm.parse_assembly(llvm_ir)
    module.verify()
    target = llvm.Target.from_default_triple()
    engine = llvm.create_mcjit_compiler(module, target.create_target_machine())
    engine.finalize_object()
    import ctypes

    function = ctypes.CFUNCTYPE(ctypes.c_float)(
        engine.get_function_address("main")
    )
    return function()


@pytest.mark.parametrize(
    ("operator", "llvm_predicate", "machine_op"),
    [
        ("==", "icmp eq", MachineOp.BEQ),
        ("!=", "icmp ne", MachineOp.BNE),
        ("<", "icmp slt", MachineOp.BLT),
        (">", "icmp sgt", MachineOp.BLT),
        ("<=", "icmp sle", MachineOp.BGE),
        (">=", "icmp sge", MachineOp.BGE),
    ],
)
def test_compare_branch_has_explicit_backend_lowering(
    operator, llvm_predicate, machine_op,
):
    builder = IRBuilder()
    builder.new_function("main")
    builder.new_block("entry")
    lhs = builder.make_value("lhs", dtype=DataType.INT32)
    rhs = builder.make_value("rhs", dtype=DataType.INT32)
    builder.br_compare(lhs, operator, rhs, "yes", "no")
    builder.new_block("yes")
    builder.ret(lhs)
    builder.new_block("no")
    builder.ret(rhs)

    llvm_ir = LLVMCodegen(builder.program).emit()
    assert llvm_predicate in llvm_ir
    assert "br i1" in llvm_ir
    llvm.parse_assembly(llvm_ir).verify()

    machine = InstructionSelector(builder.program).run()
    assert any(instr.op == machine_op for instr in machine)
    allocated = RegisterAllocator(machine, mode="greedy").run()
    assembly = AsmEmitter(allocated).emit()
    assert ".yes:" in assembly
    assert ".no:" in assembly
    assert ".yes" in next(
        line for line in assembly.splitlines()
        if line.strip().startswith(machine_op.value)
    )


def test_while_condition_reads_value_updated_by_loop_body():
    program = ExtendedDSLParser().parse(
        """
        i = add(0, 0)
        while (i < 3):
            i = add(i, 1)
        endwhile
        return i
        """
    )
    function = program.functions[0]
    header = next(block for block in function.blocks if block.name.startswith("while_hdr"))
    branch = header.instructions[-1]
    assert branch.opcode == OpCode.BR_IF
    header_load = next(
        instr for instr in header.instructions if instr.opcode == OpCode.LOAD
    )

    body = next(block for block in function.blocks if block.name.startswith("while_body"))
    body_store = next(
        instr for instr in body.instructions
        if instr.opcode == OpCode.STORE
    )
    assert branch.operands[0] == header_load.dest
    assert body_store.operands[0] == header_load.operands[0]


def test_if_else_result_is_loaded_after_branch_merge():
    program = ExtendedDSLParser().parse(
        """
        a = add(3, 0)
        b = add(4, 0)
        if (a < b):
            result = add(a, b)
        else:
            result = sub(a, b)
        endif
        return result
        """
    )
    function = program.functions[0]
    merge = next(block for block in function.blocks if block.name.startswith("if_end"))
    assert [instr.opcode for instr in merge.instructions] == [
        OpCode.LOAD,
        OpCode.RETURN,
    ]
    result_slot = merge.instructions[0].operands[0]
    branch_stores = [
        instr
        for block in function.blocks
        for instr in block.instructions
        if instr.opcode == OpCode.STORE and instr.operands[0] == result_slot
    ]
    assert len(branch_stores) == 2


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            """
            a = add(3, 0)
            b = add(4, 0)
            if (a < b):
                result = add(a, b)
            else:
                result = sub(a, b)
            endif
            return result
            """,
            7.0,
        ),
        (
            """
            i = add(1, 0)
            total = add(0, 0)
            while (i <= 5):
                total = add(total, i)
                i = add(i, 1)
            endwhile
            return total
            """,
            15.0,
        ),
        (
            """
            i = add(0, 0)
            total = add(0, 0)
            while (i < 3):
                j = add(0, 0)
                while (j < 2):
                    if (i >= 0):
                        total = add(total, 1)
                    else:
                        total = sub(total, 1)
                    endif
                    j = add(j, 1)
                endwhile
                i = add(i, 1)
            endwhile
            return total
            """,
            6.0,
        ),
    ],
)
def test_control_flow_executes_with_llvm(source, expected):
    assert _execute_llvm_scalar(source) == pytest.approx(expected)


def test_return_is_never_followed_by_another_terminator():
    program = ExtendedDSLParser().parse(
        """
        if (a > 0):
            return a
        else:
            return 0
        endif
        """
    )
    for block in program.functions[0].blocks:
        returns = [
            index for index, instr in enumerate(block.instructions)
            if instr.opcode == OpCode.RETURN
        ]
        if returns:
            assert returns[-1] == len(block.instructions) - 1

    llvm.parse_assembly(LLVMCodegen(program).emit()).verify()


@pytest.mark.parametrize(
    "condition",
    [
        "if (a + b > c):",
        "if (a < b < c):",
        "while (call(a) != 0):",
    ],
)
def test_conditions_reject_non_atomic_operands(condition):
    parser = ExtendedDSLParser()
    assert parser._parse_condition(condition) is None
    collector = parser.validate(f"{condition}\nendif")
    assert collector.has_errors


def test_riscv_mutable_slots_use_distinct_stack_addresses():
    program = ExtendedDSLParser().parse(
        """
        i = add(0, 0)
        total = add(0, 0)
        while (i < 1):
            total = add(total, 1)
            i = add(i, 1)
        endwhile
        return total
        """
    )
    machine = InstructionSelector(program).run()
    alloca_offsets = [
        instr.src2.value
        for instr in machine
        if instr.op == MachineOp.ADDI
        and instr.src1 is not None
        and instr.src1.value == "sp"
        and instr.src2 is not None
    ]
    assert alloca_offsets == [-4, -8]

    allocated = RegisterAllocator(machine, mode="greedy").run()
    assembly = AsmEmitter(allocated).emit()
    stores = [line.strip() for line in assembly.splitlines() if line.strip().startswith("sw ")]
    loads = [line.strip() for line in assembly.splitlines() if line.strip().startswith("lw ")]
    assert any(", 0(" in line for line in stores)
    assert any(", 0(" in line for line in loads)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("if_else.dsl", 7.0),
        ("while_sum.dsl", 15.0),
        ("nested_loop.dsl", 6.0),
    ],
)
def test_topic01_examples_execute_with_llvm(filename, expected):
    source = (
        Path(__file__).parents[1] / "examples" / "topic01" / filename
    ).read_text(encoding="utf-8")
    assert _execute_llvm_scalar(source) == pytest.approx(expected)
