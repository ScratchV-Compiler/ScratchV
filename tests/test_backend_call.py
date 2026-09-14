"""Tests for minimal CALL lowering in the RISC-V backend (Topic 15)."""

import pytest

from scratchv.backend.asm_emit import AsmEmitter
from scratchv.backend.instruction_select import (
    InstructionSelector, UnsupportedCallError,
)
from scratchv.backend.machine_types import MachineOp, MachineOperand
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import OpCode


def _program_with_call(n_args: int = 2, is_tail: bool = False,
                       has_ret: bool = True):
    b = IRBuilder()
    params = [b.make_value(name=f"p{i}") for i in range(n_args)]
    b.new_function("inc", params=params)
    b.new_block("entry")
    if params:
        total = params[0]
        for p in params[1:]:
            total = b.add(total, p)
        b.ret(total)
    else:
        b.ret()

    b.new_function("main")
    b.new_block("entry")
    args = [b.make_value(name=f"x{i}") for i in range(n_args)]
    r = b.call("inc", args, has_ret=has_ret, is_tail=is_tail)
    b.ret(r)
    return b.program


def _call_instr(program):
    return next(
        ins
        for func in program.functions
        for block in func.blocks
        for ins in block.instructions
        if ins.opcode is OpCode.CALL
    )


def test_default_raises_unsupported_call_error():
    program = _program_with_call()
    sel = InstructionSelector(program)
    with pytest.raises(UnsupportedCallError) as excinfo:
        sel.run()
    msg = str(excinfo.value)
    assert msg.startswith("CALL ")
    assert "ABI" in msg
    assert "inc" in msg
    assert "'main'" in msg


def test_minimal_call_asm():
    program = _program_with_call()
    call = _call_instr(program)

    sel = InstructionSelector(program, allow_uninlined_calls=True)
    instrs = sel.run()
    ops = [i.op.value for i in instrs]
    assert "jal" in ops
    assert "mv" in ops

    jal_index = next(
        idx for idx, i in enumerate(instrs) if i.op is MachineOp.JAL)
    jal = instrs[jal_index]
    assert jal.dst == MachineOperand.reg("ra")
    assert jal.comment == "inc"

    assert instrs[jal_index - 2].op is MachineOp.MV
    assert instrs[jal_index - 2].dst == MachineOperand.reg("a0")
    assert instrs[jal_index - 2].src1 == MachineOperand.vreg("x0")
    assert instrs[jal_index - 1].op is MachineOp.MV
    assert instrs[jal_index - 1].dst == MachineOperand.reg("a1")
    assert instrs[jal_index - 1].src1 == MachineOperand.vreg("x1")

    ret_move = instrs[jal_index + 1]
    assert ret_move.op is MachineOp.MV
    assert ret_move.dst == MachineOperand.vreg(call.dest.name)
    assert ret_move.src1 == MachineOperand.reg("a0")

    asm = AsmEmitter(instrs).emit()
    assert "jal ra, inc" in asm


def test_minimal_void_call_has_no_return_move():
    program = _program_with_call(n_args=1, has_ret=False)
    sel = InstructionSelector(program, allow_uninlined_calls=True)
    instrs = sel.run()
    jal_index = next(
        idx for idx, i in enumerate(instrs) if i.op is MachineOp.JAL)
    assert not any(
        i.op is MachineOp.MV for i in instrs[jal_index + 1:])


def test_more_than_8_args_always_raises():
    program = _program_with_call(n_args=9)
    sel = InstructionSelector(program, allow_uninlined_calls=True)
    with pytest.raises(UnsupportedCallError) as excinfo:
        sel.run()
    assert "args > 8" in str(excinfo.value)


def test_tail_call_always_raises():
    program = _program_with_call(is_tail=True)
    sel = InstructionSelector(program, allow_uninlined_calls=True)
    with pytest.raises(UnsupportedCallError) as excinfo:
        sel.run()
    assert "tail" in str(excinfo.value)


def test_driver_default_rejects_residual_call():
    from scratchv.compiler import CompilerConfig, CompilerDriver

    program = _program_with_call()
    driver = CompilerDriver(CompilerConfig())
    with pytest.raises(UnsupportedCallError):
        driver._generate_riscv_linear(program)


def test_driver_minimal_call_codegen_emits_jal():
    from scratchv.compiler import CompilerConfig, CompilerDriver

    program = _program_with_call()
    driver = CompilerDriver(CompilerConfig(
        minimal_call_codegen=True, reg_alloc="greedy"))
    asm = driver._generate_riscv_linear(program)
    assert "jal ra, inc" in asm


def test_compile_reports_codegen_error_and_writes_nothing(tmp_path):
    from scratchv.compiler import CompilerConfig, CompilerDriver

    program = _program_with_call()
    output = tmp_path / "out.s"
    driver = CompilerDriver(CompilerConfig())
    driver._parse = lambda *args, **kwargs: program

    result = driver.compile("dummy.onnx", str(output))

    assert result.success is False
    assert len(result.errors) == 1
    assert result.errors[0].startswith("Codegen error:")
    assert "ABI" in result.errors[0]
    assert "inc" in result.errors[0]
    assert not output.exists()
