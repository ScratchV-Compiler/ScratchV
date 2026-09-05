"""Register-allocation tests for machine pseudo-instructions."""

import pytest

from scratchv.backend import regalloc_linear, regalloc_linear_v1_5
from scratchv.backend.instruction_select import InstructionSelector
from scratchv.backend.machine_semantics import OP_SEM
from scratchv.backend.machine_types import (
    ALL_REGS,
    MachineInstr,
    MachineOp,
    MachineOperand,
)
from scratchv.backend.riscv_encoder import RISCVAEncoder
from scratchv.ir.types import Program


ALLOCATOR_MODULES = (regalloc_linear, regalloc_linear_v1_5)


def _run_rv32(assembly: str, instruction_limit: int = 16):
    pytest.importorskip("tinyfive")
    from scratchv.simulator.tinyfive import ProfiledMachine

    binary = RISCVAEncoder().assemble(assembly)
    words = [
        int.from_bytes(binary[offset:offset + 4], "little")
        for offset in range(0, len(binary), 4)
    ]
    machine = ProfiledMachine(mem_size=4096)
    machine.load_binary(words, origin=0)
    machine.run(instructions=instruction_limit, start=0, strict=True)
    return machine


@pytest.mark.parametrize("allocator_module", ALLOCATOR_MODULES)
def test_mv_has_explicit_def_use_semantics(allocator_module):
    move = MachineInstr(
        MachineOp.MV,
        MachineOperand.vreg("copy"),
        MachineOperand.vreg("source"),
    )

    converted = allocator_module.block_from_machine_instrs([move])[0]

    assert OP_SEM[MachineOp.MV].is_pseudo
    assert converted.defines == {"copy"}
    assert converted.uses == {"source"}


@pytest.mark.parametrize("allocator_module", ALLOCATOR_MODULES)
def test_mv_allocates_like_addi(allocator_module):
    source = MachineOperand.vreg("source")
    copy = MachineOperand.vreg("copy")
    pseudo = [
        MachineInstr(MachineOp.LI, source, MachineOperand.immediate(7)),
        MachineInstr(MachineOp.MV, copy, source),
    ]
    expanded = [
        MachineInstr(MachineOp.LI, source, MachineOperand.immediate(7)),
        MachineInstr(
            MachineOp.ADDI,
            copy,
            source,
            MachineOperand.immediate(0),
        ),
    ]

    pseudo_alloc = allocator_module.LinearScanAllocator(["t0", "t1"])
    pseudo_asm = pseudo_alloc.emit(
        allocator_module.block_from_machine_instrs(pseudo)
    )
    expanded_alloc = allocator_module.LinearScanAllocator(["t0", "t1"])
    expanded_asm = expanded_alloc.emit(
        allocator_module.block_from_machine_instrs(expanded)
    )

    assert "%" not in pseudo_asm
    assert "source" not in pseudo_asm
    assert "copy" not in pseudo_asm
    assert len(pseudo_alloc._spill_slots) == len(expanded_alloc._spill_slots)
    assert RISCVAEncoder().assemble(pseudo_asm) == RISCVAEncoder().assemble(
        expanded_asm
    )


def test_move_helper_uses_li_for_an_immediate_source():
    selector = InstructionSelector(Program())

    selector._emit_move(
        MachineOperand.vreg("copy"),
        MachineOperand.immediate(42),
        comment="constant copy",
    )

    assert selector._instructions == [
        MachineInstr(
            MachineOp.LI,
            MachineOperand.vreg("copy"),
            MachineOperand.immediate(42),
            comment="constant copy",
        )
    ]


def test_mv_executes_as_a_register_copy():
    machine = _run_rv32("li t0, 42\nmv t1, t0\n.done:\nj .done")

    assert machine.get_reg(6) == 42  # t1 / x6


@pytest.mark.parametrize("allocator_module", ALLOCATOR_MODULES)
def test_li_defines_only_its_destination(allocator_module):
    load_immediate = MachineInstr(
        MachineOp.LI,
        MachineOperand.vreg("constant"),
        MachineOperand.immediate(0x12345),
    )

    converted = allocator_module.block_from_machine_instrs([load_immediate])[0]

    assert OP_SEM[MachineOp.LI].is_pseudo
    assert OP_SEM[MachineOp.LI].immediate_positions == (1,)
    assert converted.defines == {"constant"}
    assert converted.uses == set()


@pytest.mark.parametrize("allocator_module", ALLOCATOR_MODULES)
def test_li_small_immediate_matches_addi_pressure_and_encoding(allocator_module):
    destination = MachineOperand.vreg("constant")
    pseudo = [
        MachineInstr(
            MachineOp.LI,
            destination,
            MachineOperand.immediate(7),
        )
    ]
    expanded = [
        MachineInstr(
            MachineOp.ADDI,
            destination,
            MachineOperand.reg("x0"),
            MachineOperand.immediate(7),
        )
    ]

    pseudo_alloc = allocator_module.LinearScanAllocator(["t0"])
    pseudo_asm = pseudo_alloc.emit(
        allocator_module.block_from_machine_instrs(pseudo)
    )
    expanded_alloc = allocator_module.LinearScanAllocator(["t0"])
    expanded_asm = expanded_alloc.emit(
        allocator_module.block_from_machine_instrs(expanded)
    )

    assert len(pseudo_alloc._spill_slots) == len(expanded_alloc._spill_slots)
    assert RISCVAEncoder().assemble(pseudo_asm) == RISCVAEncoder().assemble(
        expanded_asm
    )


def test_li_large_immediate_expands_to_two_real_instructions():
    pseudo = "li t0, 0x12345"
    expanded = "lui t0, 18\naddi t0, t0, 837"

    encoded = RISCVAEncoder().assemble(pseudo)

    assert len(encoded) == 8
    assert encoded == RISCVAEncoder().assemble(expanded)


def test_li_large_immediate_executes_with_exact_value():
    machine = _run_rv32("li t0, 0x12345\n.done:\nj .done")

    assert machine.get_reg(5) == 0x12345  # t0 / x5


@pytest.mark.parametrize("allocator_module", ALLOCATOR_MODULES)
def test_max_tracks_register_and_immediate_operands(allocator_module):
    register_max = MachineInstr(
        MachineOp.MAX,
        MachineOperand.vreg("result"),
        MachineOperand.vreg("left"),
        MachineOperand.vreg("right"),
    )
    immediate_max = MachineInstr(
        MachineOp.MAX,
        MachineOperand.vreg("left"),
        MachineOperand.vreg("left"),
        MachineOperand.immediate(0),
    )

    register_inst, immediate_inst = allocator_module.block_from_machine_instrs(
        [register_max, immediate_max]
    )

    assert OP_SEM[MachineOp.MAX].is_pseudo
    assert register_inst.defines == {"result"}
    assert register_inst.uses == {"left", "right"}
    assert immediate_inst.defines == {"left"}
    assert immediate_inst.uses == {"left"}


def test_max_helper_keeps_supported_zero_immediate():
    selector = InstructionSelector(Program())

    selector._emit_max(
        MachineOperand.vreg("result"),
        MachineOperand.vreg("left"),
        MachineOperand.immediate(0),
        comment="relu",
    )

    assert selector._instructions == [
        MachineInstr(
            MachineOp.MAX,
            MachineOperand.vreg("result"),
            MachineOperand.vreg("left"),
            MachineOperand.immediate(0),
            comment="relu",
        )
    ]


def test_max_helper_materializes_nonzero_immediate():
    selector = InstructionSelector(Program())

    selector._emit_max(
        MachineOperand.vreg("result"),
        MachineOperand.vreg("left"),
        MachineOperand.immediate(7),
        comment="max",
    )

    temp = MachineOperand.vreg("__scratchv_max_rhs_1")
    assert selector._instructions == [
        MachineInstr(
            MachineOp.LI,
            temp,
            MachineOperand.immediate(7),
            comment="materialize max rhs",
        ),
        MachineInstr(
            MachineOp.MAX,
            MachineOperand.vreg("result"),
            MachineOperand.vreg("left"),
            temp,
            comment="max",
        ),
    ]


@pytest.mark.parametrize("allocator_module", ALLOCATOR_MODULES)
def test_max_pseudo_and_expansion_have_equal_spill_pressure(allocator_module):
    pseudo = [
        MachineInstr(
            MachineOp.LI,
            MachineOperand.vreg("left"),
            MachineOperand.immediate(3),
        ),
        MachineInstr(
            MachineOp.LI,
            MachineOperand.vreg("right"),
            MachineOperand.immediate(7),
        ),
        MachineInstr(
            MachineOp.MAX,
            MachineOperand.vreg("result"),
            MachineOperand.vreg("left"),
            MachineOperand.vreg("right"),
        ),
    ]
    expanded = [
        allocator_module.LsInstruction(
            0, "li", ["left", "3"], defines={"left"}
        ),
        allocator_module.LsInstruction(
            1, "li", ["right", "7"], defines={"right"}
        ),
        allocator_module.LsInstruction(
            2,
            "bge",
            ["left", "right", ".max_then"],
            uses={"left", "right"},
        ),
        allocator_module.LsInstruction(
            3,
            "addi",
            ["result", "right", "0"],
            defines={"result"},
            uses={"right"},
        ),
        allocator_module.LsInstruction(4, "j", [".max_end"]),
        allocator_module.LsInstruction(5, ".label", [".max_then"]),
        allocator_module.LsInstruction(
            6,
            "addi",
            ["result", "left", "0"],
            defines={"result"},
            uses={"left"},
        ),
        allocator_module.LsInstruction(7, ".label", [".max_end"]),
    ]

    pseudo_alloc = allocator_module.LinearScanAllocator(["t0", "t1"])
    pseudo_alloc.allocate(
        pseudo_alloc.compute_live_intervals(
            allocator_module.block_from_machine_instrs(pseudo)
        )
    )
    expanded_alloc = allocator_module.LinearScanAllocator(["t0", "t1"])
    expanded_alloc.allocate(expanded_alloc.compute_live_intervals(expanded))

    assert len(pseudo_alloc._spill_slots) == len(expanded_alloc._spill_slots)


def test_max_register_rhs_expands_to_copy_the_rhs_not_zero():
    pseudo = "max t2, t0, t1"
    expanded = """\
bge t0, t1, .__max_then_0
addi t2, t1, 0
j .__max_end_0
.__max_then_0:
addi t2, t0, 0
.__max_end_0:
"""

    assert RISCVAEncoder().assemble(pseudo) == RISCVAEncoder().assemble(expanded)


def test_max_rejects_nonzero_immediate_rhs_without_hidden_vreg_semantics():
    with pytest.raises(ValueError, match="supports only zero"):
        RISCVAEncoder().assemble("max t2, t0, 7")


@pytest.mark.parametrize(
    "left, right, expected",
    [(2, 3, 3), (3, 2, 3), (-4, -2, -2), (7, 7, 7)],
)
def test_max_executes_for_both_control_flow_paths(left, right, expected):
    machine = _run_rv32(
        f"li t0, {left}\nli t1, {right}\nmax t2, t0, t1\n"
        ".done:\nj .done"
    )

    assert machine.get_reg(7) == expected  # t2 / x7


@pytest.mark.parametrize(
    "assembly, result_reg, expected",
    [
        ("li t0, 2\nli t1, 3\nmax t0, t0, t1", 5, 3),
        ("li t0, 3\nli t1, 2\nmax t1, t0, t1", 6, 3),
    ],
)
def test_max_is_correct_when_destination_aliases_a_source(
    assembly, result_reg, expected
):
    machine = _run_rv32(assembly + "\n.done:\nj .done")

    assert machine.get_reg(result_reg) == expected


def test_max_internal_labels_do_not_collide_with_user_labels():
    machine = _run_rv32(
        ".__max_then_0:\n"
        "li t0, 2\n"
        "li t1, 3\n"
        "max t2, t0, t1\n"
        ".done:\n"
        "j .done"
    )

    assert machine.get_reg(7) == 3


def test_branch_immediate_fails_instead_of_clobbering_a_busy_temp():
    all_temps_are_live_in_text = "\n".join(
        [f"add t{i}, t{i}, t{i}" for i in range(7)]
        + ["beq s0, 5, .done", ".done:", "j .done"]
    )

    with pytest.raises(ValueError, match="needs a free temporary register"):
        RISCVAEncoder().assemble(all_temps_are_live_in_text)


@pytest.mark.parametrize("allocator_module", ALLOCATOR_MODULES)
def test_label_emits_gas_syntax_and_has_no_register_semantics(allocator_module):
    label = MachineInstr(MachineOp.LABEL, comment=".target")
    load = MachineInstr(
        MachineOp.LI,
        MachineOperand.vreg("value"),
        MachineOperand.immediate(1),
    )

    block = allocator_module.block_from_machine_instrs([label, load])
    assembly = allocator_module.LinearScanAllocator(["t0"]).emit(block)

    assert OP_SEM[MachineOp.LABEL].is_label
    assert block[0].defines == set()
    assert block[0].uses == set()
    assert assembly.splitlines()[0] == ".target:"
    assert ".label" not in assembly
    assert len(RISCVAEncoder().assemble(assembly)) == 4


@pytest.mark.parametrize("allocator_module", ALLOCATOR_MODULES)
def test_bnez_uses_condition_and_emits_target_operand(allocator_module):
    machine = [
        MachineInstr(
            MachineOp.LI,
            MachineOperand.vreg("condition"),
            MachineOperand.immediate(1),
        ),
        MachineInstr(
            MachineOp.BNEZ,
            MachineOperand.vreg("condition"),
            comment=".taken",
        ),
        MachineInstr(MachineOp.LABEL, comment=".taken"),
    ]

    block = allocator_module.block_from_machine_instrs(machine)
    branch = block[1]
    allocator = allocator_module.LinearScanAllocator(["t0"])
    intervals = allocator.compute_live_intervals(block)
    assembly = allocator.emit(block)
    condition = next(iv for iv in intervals if iv.vreg == "condition")

    assert OP_SEM[MachineOp.BNEZ].is_terminator
    assert branch.defines == set()
    assert branch.uses == {"condition"}
    assert branch.operands == ["condition", ".taken"]
    assert branch.comment == ""
    assert condition.uses == {1}
    assert condition.end == 2
    assert "bnez t0, .taken" in assembly
    RISCVAEncoder().assemble(assembly)


def test_bnez_encoding_matches_bne_against_zero():
    pseudo = "bnez t0, .taken\naddi t1, x0, 0\n.taken:\naddi t1, x0, 1"
    expanded = "bne t0, x0, .taken\naddi t1, x0, 0\n.taken:\naddi t1, x0, 1"

    assert RISCVAEncoder().assemble(pseudo) == RISCVAEncoder().assemble(expanded)


@pytest.mark.parametrize("condition, expected", [(0, 1), (5, 2)])
def test_bnez_executes_taken_and_not_taken_paths(condition, expected):
    machine = _run_rv32(
        f"li t0, {condition}\n"
        "li t1, 0\n"
        "bnez t0, .taken\n"
        "li t1, 1\n"
        "j .done\n"
        ".taken:\n"
        "li t1, 2\n"
        ".done:\n"
        "j .done"
    )

    assert machine.get_reg(6) == expected  # t1 / x6


@pytest.mark.parametrize(
    "assembly, message",
    [
        ("bnez t0", "expects exactly 2 operands"),
        ("bnez t0, .missing", "undefined branch target"),
    ],
)
def test_bnez_rejects_missing_target_information(assembly, message):
    with pytest.raises(ValueError, match=message):
        RISCVAEncoder().assemble(assembly)


@pytest.mark.parametrize("allocator_module", ALLOCATOR_MODULES)
def test_j_emits_target_operand_without_register_pressure(allocator_module):
    machine = [
        MachineInstr(MachineOp.J, comment=".target"),
        MachineInstr(
            MachineOp.LI,
            MachineOperand.vreg("skipped"),
            MachineOperand.immediate(0),
        ),
        MachineInstr(MachineOp.LABEL, comment=".target"),
    ]

    block = allocator_module.block_from_machine_instrs(machine)
    jump = block[0]
    assembly = allocator_module.LinearScanAllocator(["t0"]).emit(block)

    assert OP_SEM[MachineOp.J].is_terminator
    assert jump.defines == set()
    assert jump.uses == set()
    assert jump.operands == [".target"]
    assert jump.comment == ""
    assert assembly.splitlines()[0] == "  j .target"
    RISCVAEncoder().assemble(assembly)


def test_j_encoding_matches_jal_with_zero_destination():
    pseudo = "j .target\naddi t0, x0, 0\n.target:\naddi t0, x0, 1"
    expanded = "jal x0, .target\naddi t0, x0, 0\n.target:\naddi t0, x0, 1"

    assert RISCVAEncoder().assemble(pseudo) == RISCVAEncoder().assemble(expanded)


def test_j_executes_without_falling_through():
    machine = _run_rv32(
        "li t0, 0\n"
        "j .target\n"
        "li t0, 1\n"
        ".target:\n"
        "li t0, 2\n"
        ".done:\n"
        "j .done"
    )

    assert machine.get_reg(5) == 2  # t0 / x5


@pytest.mark.parametrize(
    "assembly, message",
    [
        ("j", "expects exactly 1 operand"),
        ("j .missing", "undefined branch target"),
    ],
)
def test_j_rejects_missing_target_information(assembly, message):
    with pytest.raises(ValueError, match=message):
        RISCVAEncoder().assemble(assembly)


@pytest.mark.parametrize("allocator_module", ALLOCATOR_MODULES)
def test_jalr_tracks_link_definition_and_base_use(allocator_module):
    jump = MachineInstr(
        MachineOp.JALR,
        MachineOperand.vreg("link"),
        MachineOperand.vreg("base"),
        MachineOperand.immediate(0),
    )
    ret = MachineInstr(
        MachineOp.JALR,
        MachineOperand.reg("zero"),
        MachineOperand.reg("ra"),
        comment="ret",
    )

    jump_inst, ret_inst = allocator_module.block_from_machine_instrs(
        [jump, ret]
    )

    assert jump_inst.defines == {"link"}
    assert jump_inst.uses == {"base"}
    assert ret_inst.defines == set()
    assert ret_inst.uses == set()
    assert RISCVAEncoder().assemble(ret_inst.to_asm())


@pytest.mark.parametrize("allocator_module", ALLOCATOR_MODULES)
@pytest.mark.parametrize(
    "opcode", [MachineOp.BEQ, MachineOp.BNE, MachineOp.BLT, MachineOp.BGE]
)
def test_true_branch_operands_are_uses_and_target_is_emitted(
    allocator_module, opcode
):
    branch = MachineInstr(
        opcode,
        MachineOperand.vreg("left"),
        MachineOperand.vreg("right"),
        comment=".target",
    )

    converted = allocator_module.block_from_machine_instrs([branch])[0]

    assert converted.defines == set()
    assert converted.uses == {"left", "right"}
    assert converted.operands == ["left", "right", ".target"]


def test_call_metadata_records_abi_clobbers_without_calling_it_a_terminator():
    semantics = OP_SEM[MachineOp.CALL]

    assert semantics.is_call
    assert not semantics.is_terminator
    assert semantics.implicit_defs == {"ra"}
    assert {"ra", "a0", "a7", "t0", "t6"} <= semantics.clobbers


@pytest.mark.parametrize("allocator_module", ALLOCATOR_MODULES)
def test_default_register_bank_matches_canonical_19_register_bank(
    allocator_module,
):
    allocator = allocator_module.LinearScanAllocator()

    assert allocator.phys_regs == ALL_REGS
    assert len(allocator.phys_regs) == 19


def test_every_machine_opcode_has_explicit_semantics():
    assert set(OP_SEM) == set(MachineOp)


@pytest.mark.parametrize("allocator_module", ALLOCATOR_MODULES)
def test_pseudo_pipeline_leaks_no_arbitrary_virtual_register_names(
    allocator_module,
):
    v = MachineOperand.vreg
    imm = MachineOperand.immediate
    names = {
        "input_tensor",
        "weight_tensor",
        "maximum_value",
        "copied_value",
    }
    machine = [
        MachineInstr(MachineOp.LI, v("input_tensor"), imm(3)),
        MachineInstr(MachineOp.LI, v("weight_tensor"), imm(7)),
        MachineInstr(
            MachineOp.MAX,
            v("maximum_value"),
            v("input_tensor"),
            v("weight_tensor"),
        ),
        MachineInstr(
            MachineOp.MV, v("copied_value"), v("maximum_value")
        ),
        MachineInstr(
            MachineOp.BNEZ, v("copied_value"), comment=".taken"
        ),
        MachineInstr(MachineOp.J, comment=".done"),
        MachineInstr(MachineOp.LABEL, comment=".taken"),
        MachineInstr(
            MachineOp.MV, MachineOperand.reg("a0"), v("copied_value")
        ),
        MachineInstr(MachineOp.LABEL, comment=".done"),
    ]
    allocator = allocator_module.LinearScanAllocator(["t0", "t1", "t2"])
    assembly = allocator.emit(
        allocator_module.block_from_machine_instrs(machine)
    )

    assert not names & set(assembly.replace(",", " ").split())
    assert "%" not in assembly
    RISCVAEncoder().assemble(assembly)


@pytest.mark.parametrize("allocator_module", ALLOCATOR_MODULES)
def test_store_operands_are_uses_not_definitions(allocator_module):
    store = MachineInstr(
        MachineOp.SW,
        MachineOperand.vreg("value"),
        MachineOperand.vreg("address"),
    )

    converted = allocator_module.block_from_machine_instrs([store])[0]

    assert converted.defines == set()
    assert converted.uses == {"value", "address"}


@pytest.mark.parametrize("allocator_module", ALLOCATOR_MODULES)
@pytest.mark.parametrize("opcode", [MachineOp.BNEZ, MachineOp.J])
def test_control_pseudo_round_trip_preserves_target_comment(
    allocator_module, opcode
):
    condition = (
        MachineOperand.vreg("condition") if opcode is MachineOp.BNEZ else None
    )
    original = MachineInstr(opcode, condition, comment=".target")

    block = allocator_module.block_from_machine_instrs([original])
    converted = allocator_module.machine_instrs_from_block(block)[0]

    assert converted == original


@pytest.mark.parametrize("allocator_module", ALLOCATOR_MODULES)
def test_round_trip_does_not_misclassify_vreg_name_prefix(allocator_module):
    block = [
        allocator_module.LsInstruction(
            0,
            "mv",
            ["a_temporary", "source"],
            defines={"a_temporary"},
            uses={"source"},
        )
    ]

    converted = allocator_module.machine_instrs_from_block(block)[0]

    assert converted.dst == MachineOperand.vreg("a_temporary")
    assert converted.src1 == MachineOperand.vreg("source")


@pytest.mark.parametrize("allocator_module", ALLOCATOR_MODULES)
def test_round_trip_rejects_unknown_opcode_instead_of_falling_back_to_mv(
    allocator_module,
):
    block = [allocator_module.LsInstruction(0, "not-an-op")]

    with pytest.raises(ValueError, match="not-an-op"):
        allocator_module.machine_instrs_from_block(block)
