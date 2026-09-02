"""Topic17 P1: CFG liveness, executable spills, calls, and greedy reloads."""

from __future__ import annotations

import re
import random

import pytest

from scratchv.backend import regalloc_linear, regalloc_linear_v1_5
from scratchv.backend.asm_emit import AsmEmitter
from scratchv.backend.machine_types import MachineInstr, MachineOp, MachineOperand
from scratchv.backend.register_alloc import RegisterAllocator
from scratchv.backend.riscv_encoder import RISCVAEncoder


ALLOCATOR_MODULES = (regalloc_linear, regalloc_linear_v1_5)


def _pressure_machine() -> list[MachineInstr]:
    v = MachineOperand.vreg
    imm = MachineOperand.immediate
    return [
        MachineInstr(MachineOp.LI, v("v0"), imm(1)),
        MachineInstr(MachineOp.LI, v("v1"), imm(2)),
        MachineInstr(MachineOp.LI, v("v2"), imm(3)),
        MachineInstr(MachineOp.ADD, v("v3"), v("v0"), v("v1")),
        MachineInstr(MachineOp.ADD, v("v4"), v("v2"), v("v3")),
        MachineInstr(MachineOp.ADD, v("v5"), v("v4"), v("v0")),
        MachineInstr(MachineOp.MV, MachineOperand.reg("a0"), v("v5")),
    ]


@pytest.mark.parametrize("allocator_module", ALLOCATOR_MODULES)
def test_two_register_spill_rewrite_executes_with_distinct_sources(allocator_module):
    pytest.importorskip("tinyfive")
    from scratchv.simulator.tinyfive import ProfiledMachine

    allocator = allocator_module.LinearScanAllocator(["t0", "t1"])
    assembly = allocator.emit(
        allocator_module.block_from_machine_instrs(_pressure_machine())
    )
    program = "li sp, 2048\n" + assembly + "\n.done:\nj .done"
    binary = RISCVAEncoder().assemble(program)
    words = [
        int.from_bytes(binary[offset:offset + 4], "little")
        for offset in range(0, len(binary), 4)
    ]
    machine = ProfiledMachine(mem_size=4096)
    machine.load_binary(words, origin=0)
    machine.run(instructions=len(words) + 2, start=0, strict=True)

    assert machine.get_reg(10) == 7  # a0
    add_lines = [line for line in assembly.splitlines() if line.strip().startswith("add ")]
    for line in add_lines:
        operands = line.split("#", 1)[0].replace(",", " ").split()[1:]
        assert operands[1] != operands[2]


@pytest.mark.parametrize("allocator_module", ALLOCATOR_MODULES)
def test_impossible_source_pressure_fails_instead_of_clobbering(allocator_module):
    block = allocator_module.block_from_machine_instrs(_pressure_machine()[:4])
    allocator = allocator_module.LinearScanAllocator(["t0"])

    with pytest.raises(RuntimeError, match="distinct register uses"):
        allocator.emit(block)


@pytest.mark.parametrize("allocator_module", ALLOCATOR_MODULES)
def test_destination_can_reuse_a_still_live_source_after_saving_it(allocator_module):
    pytest.importorskip("tinyfive")
    from scratchv.simulator.tinyfive import ProfiledMachine

    v = MachineOperand.vreg
    imm = MachineOperand.immediate
    machine_ir = [
        MachineInstr(MachineOp.LI, v("left"), imm(3)),
        MachineInstr(MachineOp.LI, v("right"), imm(4)),
        # Both sources remain live after this definition.
        MachineInstr(MachineOp.ADD, v("sum"), v("left"), v("right")),
        MachineInstr(MachineOp.ADD, v("left_again"), v("left"), v("sum")),
        MachineInstr(MachineOp.ADD, v("answer"), v("left_again"), v("right")),
        MachineInstr(MachineOp.MV, MachineOperand.reg("a0"), v("answer")),
    ]
    allocator = allocator_module.LinearScanAllocator(["t0", "t1"])
    asm = allocator.emit(allocator_module.block_from_machine_instrs(machine_ir))
    binary = RISCVAEncoder().assemble(
        "li sp, 2048\n" + asm + "\n.done:\nj .done"
    )
    words = [
        int.from_bytes(binary[offset:offset + 4], "little")
        for offset in range(0, len(binary), 4)
    ]
    machine = ProfiledMachine(mem_size=4096)
    machine.load_binary(words, origin=0)
    machine.run(instructions=len(words) + 2, start=0, strict=True)

    assert machine.get_reg(10) == 14


@pytest.mark.parametrize("allocator_module", ALLOCATOR_MODULES)
def test_randomized_straight_line_programs_match_reference(allocator_module):
    """Execution-level differential check for allocation and vreg leakage."""
    pytest.importorskip("tinyfive")
    from scratchv.simulator.tinyfive import ProfiledMachine

    opcodes = [MachineOp.ADD, MachineOp.SUB, MachineOp.XOR, MachineOp.AND]
    evaluators = {
        MachineOp.ADD: lambda a, b: a + b,
        MachineOp.SUB: lambda a, b: a - b,
        MachineOp.XOR: lambda a, b: a ^ b,
        MachineOp.AND: lambda a, b: a & b,
    }
    v = MachineOperand.vreg
    imm = MachineOperand.immediate

    for seed in range(12):
        rng = random.Random(seed)
        machine_ir: list[MachineInstr] = []
        values: dict[str, int] = {}
        names: list[str] = []
        for index in range(6):
            name = f"v{index}"
            value = rng.randrange(0, 256)
            machine_ir.append(MachineInstr(MachineOp.LI, v(name), imm(value)))
            values[name] = value
            names.append(name)

        for index in range(24):
            left, right = rng.sample(names, 2)
            opcode = rng.choice(opcodes)
            name = f"tmp{index}"
            result = evaluators[opcode](values[left], values[right]) & 0xFFFFFFFF
            machine_ir.append(MachineInstr(opcode, v(name), v(left), v(right)))
            values[name] = result
            names.append(name)

        answer = names[-1]
        machine_ir.append(
            MachineInstr(MachineOp.MV, MachineOperand.reg("a0"), v(answer))
        )
        allocator = allocator_module.LinearScanAllocator(["t0", "t1", "t2"])
        asm = allocator.emit(allocator_module.block_from_machine_instrs(machine_ir))
        binary = RISCVAEncoder().assemble(
            "li sp, 4096\n" + asm + "\n.done:\nj .done"
        )
        words = [
            int.from_bytes(binary[offset:offset + 4], "little")
            for offset in range(0, len(binary), 4)
        ]
        profile = ProfiledMachine(mem_size=8192)
        profile.load_binary(words, origin=0)
        profile.run(instructions=len(words) + 2, start=0, strict=True)

        assert profile.get_reg(10) & 0xFFFFFFFF == values[answer], seed


@pytest.mark.parametrize("allocator_module", ALLOCATOR_MODULES)
def test_cfg_splits_targets_and_carries_live_value(allocator_module):
    v = MachineOperand.vreg
    imm = MachineOperand.immediate
    machine = [
        MachineInstr(MachineOp.LABEL, comment="main"),
        MachineInstr(MachineOp.LI, v("condition"), imm(1)),
        MachineInstr(MachineOp.LI, v("carried"), imm(7)),
        MachineInstr(MachineOp.BNEZ, v("condition"), comment=".then"),
        MachineInstr(MachineOp.ADDI, v("else_value"), v("carried"), imm(1)),
        MachineInstr(MachineOp.J, comment=".join"),
        MachineInstr(MachineOp.LABEL, comment=".then"),
        MachineInstr(MachineOp.ADDI, v("then_value"), v("carried"), imm(2)),
        MachineInstr(MachineOp.LABEL, comment=".join"),
        MachineInstr(MachineOp.MV, v("result"), v("carried")),
    ]
    allocator = allocator_module.LinearScanAllocator(["t0", "t1", "s0"])
    block = allocator_module.block_from_machine_instrs(machine)
    allocator.compute_live_intervals(block)
    cfg = allocator.cfg

    assert ".then" in cfg.by_name
    assert ".join" in cfg.by_name
    assert ".then" in cfg.blocks[0].successors
    assert "carried" in cfg.by_name[".then"].live_in
    assert "carried" in cfg.by_name[".join"].live_in


@pytest.mark.parametrize("allocator_module", ALLOCATOR_MODULES)
@pytest.mark.parametrize("condition, expected", [(0, 23), (1, 9)])
def test_cfg_spill_handoff_executes_on_both_branch_paths(
    allocator_module, condition, expected
):
    pytest.importorskip("tinyfive")
    from scratchv.simulator.tinyfive import ProfiledMachine

    v = MachineOperand.vreg
    imm = MachineOperand.immediate
    machine_ir = [
        MachineInstr(MachineOp.LABEL, comment="main"),
        MachineInstr(MachineOp.LI, v("carried_left"), imm(7)),
        MachineInstr(MachineOp.LI, v("carried_right"), imm(9)),
        MachineInstr(MachineOp.LI, v("condition"), imm(condition)),
        MachineInstr(MachineOp.BNEZ, v("condition"), comment=".then"),
        MachineInstr(
            MachineOp.ADD,
            v("branch_result"),
            v("carried_left"),
            v("carried_right"),
        ),
        MachineInstr(MachineOp.J, comment=".join"),
        MachineInstr(MachineOp.LABEL, comment=".then"),
        MachineInstr(
            MachineOp.SUB,
            v("branch_result"),
            v("carried_right"),
            v("carried_left"),
        ),
        MachineInstr(MachineOp.LABEL, comment=".join"),
        MachineInstr(
            MachineOp.ADD,
            v("answer"),
            v("branch_result"),
            v("carried_left"),
        ),
        MachineInstr(
            MachineOp.MV, MachineOperand.reg("a0"), v("answer")
        ),
    ]
    allocator = allocator_module.LinearScanAllocator(["t0", "t1"])
    assembly = allocator.emit(
        allocator_module.block_from_machine_instrs(machine_ir)
    )
    binary = RISCVAEncoder().assemble(
        "li sp, 2048\n" + assembly + "\n.done:\nj .done"
    )
    words = [
        int.from_bytes(binary[offset:offset + 4], "little")
        for offset in range(0, len(binary), 4)
    ]
    profile = ProfiledMachine(mem_size=4096)
    profile.load_binary(words, origin=0)
    profile.run(instructions=len(words) + 4, start=0, strict=True)

    assert profile.get_reg(10) == expected


@pytest.mark.parametrize("allocator_module", ALLOCATOR_MODULES)
def test_call_spills_caller_saved_value_and_leaves_callee_saved_value(allocator_module):
    v = MachineOperand.vreg
    imm = MachineOperand.immediate
    machine = [
        MachineInstr(MachineOp.LI, v("value"), imm(7)),
        MachineInstr(MachineOp.CALL, comment="helper"),
        MachineInstr(MachineOp.MV, v("result"), v("value")),
    ]

    caller = allocator_module.LinearScanAllocator(["t0", "s0"])
    caller_asm = caller.emit(allocator_module.block_from_machine_instrs(machine))
    assert re.search(r"sw t0, .*# spill value", caller_asm)
    assert re.search(r"lw t0, .*# reload value", caller_asm)

    callee = allocator_module.LinearScanAllocator(["s0", "t0"])
    callee_asm = callee.emit(allocator_module.block_from_machine_instrs(machine))
    assert "# spill value" not in callee_asm
    assert "# reload value" not in callee_asm


def test_greedy_spill_is_reloaded_before_later_use():
    v = MachineOperand.vreg
    imm = MachineOperand.immediate
    instructions = [
        MachineInstr(MachineOp.LI, v(f"v{i}"), imm(i)) for i in range(20)
    ]
    instructions.extend(
        MachineInstr(MachineOp.ADD, v(f"sum{i}"), v(f"v{i}"), v(f"v{(i + 1) % 20}"))
        for i in range(20)
    )

    allocated = RegisterAllocator(instructions, mode="greedy").run()
    asm = AsmEmitter(allocated).emit()

    assert "# spill v" in asm
    assert "# reload v" in asm
    assert not any(
        operand.kind == "vreg"
        for instr in allocated
        for operand in (instr.dst, instr.src1, instr.src2)
        if operand is not None
    )
    RISCVAEncoder().assemble(asm)


def test_greedy_spill_reload_executes_correctly():
    pytest.importorskip("tinyfive")
    from scratchv.simulator.tinyfive import ProfiledMachine

    v = MachineOperand.vreg
    imm = MachineOperand.immediate
    instructions = [
        MachineInstr(MachineOp.LI, v(f"v{i}"), imm(i)) for i in range(20)
    ]
    instructions.append(MachineInstr(MachineOp.MV, v("acc"), v("v0")))
    instructions.extend(
        MachineInstr(MachineOp.ADD, v("acc"), v("acc"), v(f"v{i}"))
        for i in range(1, 20)
    )
    instructions.append(
        MachineInstr(MachineOp.MV, MachineOperand.reg("a0"), v("acc"))
    )

    asm = AsmEmitter(RegisterAllocator(instructions, mode="greedy").run()).emit()
    binary = RISCVAEncoder().assemble(
        "li sp, 2048\n" + asm + "\n.done:\nj .done"
    )
    words = [
        int.from_bytes(binary[offset:offset + 4], "little")
        for offset in range(0, len(binary), 4)
    ]
    machine = ProfiledMachine(mem_size=4096)
    machine.load_binary(words, origin=0)
    machine.run(instructions=len(words) + 2, start=0, strict=True)

    assert machine.get_reg(10) == sum(range(20))


def test_call_encodes_as_local_jal_and_rejects_missing_target():
    call = "call .helper\naddi a0, x0, 1\n.helper:\njalr x0, ra, 0"
    direct = "jal ra, .helper\naddi a0, x0, 1\n.helper:\njalr x0, ra, 0"

    assert RISCVAEncoder().assemble(call) == RISCVAEncoder().assemble(direct)
    with pytest.raises(ValueError, match="undefined branch target"):
        RISCVAEncoder().assemble("call .missing")
