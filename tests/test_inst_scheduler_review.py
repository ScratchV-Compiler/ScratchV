"""Regressions for the findings in docs/topic-18/topic18-review.md."""

from pathlib import Path

import pytest

from benchmarks.run_inst_scheduler_case import _instruction_count
from scratchv.backend.inst_scheduler import (
    InstructionScheduler, ScheduleConfig, parse_instructions, schedule_assembly,
)
from scratchv.backend.machine_types import MachineInstr, MachineOp, MachineOperand
from scratchv.backend.regalloc_linear import (
    LsInstruction, block_from_machine_instrs, machine_instrs_from_block,
)
from scratchv.backend.riscv_encoder import assemble_to_binary
from scratchv.backend.schedule_model import ScheduleModel, estimate_order
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.main import main

CASE = "lw t0, 0(a0)\nadd t1, t0, t2\naddi t3, t3, 1\nsw t1, 4(a0)\n"


@pytest.mark.parametrize("literal", ['"a;b"', r'"a\\b"', '"."', '"a#;b"', r'"a\";b"'])
def test_quoted_data_does_not_disable_text_scheduling(literal):
    source = f".data\nmessage: .asciz {literal}\n.text\n" + CASE
    result = schedule_assembly(source, ScheduleConfig(strict=True))
    assert result.changed
    assert result.asm_text.startswith(source[:source.index(".text")])
    assert result.stats["saved_cycles"] == 1


@pytest.mark.parametrize("transition", [
    '.pushsection .rodata\n.word 1\n.popsection\n',
    '.pushsection .rodata\n.pushsection .bss\n.zero 4\n.popsection\n.popsection\n',
    '.section .rodata\n.word 1\n.previous\n',
    '.section .rodata\n.word 1\n.previous\n.previous\n.previous\n',
])
def test_section_state_restores_executable_code(transition):
    source = ".text\n" + transition + CASE
    result = schedule_assembly(source, ScheduleConfig(strict=True))
    assert result.changed
    assert result.stats["input_instructions"] == 4
    assert result.stats["coverage_ratio"] == 1.0


@pytest.mark.parametrize("section", ['.section .rodata,"ax"', '.section .text,"a"'])
def test_data_or_nonexecutable_flags_keep_text_pinned(section):
    source = section + "\n" + CASE
    assert schedule_assembly(source).asm_text == source
    assert not parse_instructions(source)


def test_custom_executable_section_flags_survive_reselection():
    source = '.section .custom,"ax"\n.data\n.section .custom\n' + CASE
    assert schedule_assembly(source).changed


@pytest.mark.parametrize("directive", [".popsection", ".previous"])
def test_invalid_section_restore_is_visible_and_recovers(directive):
    source = directive + "\n" + CASE + ".text\n" + CASE
    result = schedule_assembly(source)
    assert result.asm_text.startswith(directive + "\n" + CASE)
    assert result.changed
    assert result.diagnostics[0]["line"] == 1
    assert result.diagnostics[0]["severity"] == "warning"


def test_unsafe_layout_reports_actual_line_and_no_coverage():
    result = schedule_assembly(CASE + "j 8\n")
    assert not result.changed
    assert result.diagnostics[0]["line"] == 5
    assert result.stats["coverage_ratio"] == 0.0
    assert result.stats["unmodeled_instructions"] == 5


def test_display_labels_and_noncode_have_one_counting_scope():
    source = ".text\n_op_/layer1/Conv:\n" + CASE + ".data\nadd t0, t1, t2\n"
    result = schedule_assembly(source)
    assert result.stats["input_instructions"] == _instruction_count(source) == 4
    assert result.stats["modeled_instructions"] == 4
    assert "_op_/layer1/Conv:\n" in result.asm_text


def test_partial_coverage_identifies_unsupported_opcodes():
    source = CASE + "max t0, t1, t2\nmv t0, 1\n"
    result = schedule_assembly(source)
    assert result.stats["unmodeled_by_opcode"] == {"max": 1, "mv": 1}
    assert result.stats["coverage_ratio"] == 4 / 6
    assert result.stats["mean_region_size"] == 4
    assert any(d["severity"] == "warning" for d in result.diagnostics)
    assert "unmodeled: 2" in result.report()


def test_waw_completion_is_part_of_the_cost_model():
    instructions = parse_instructions("mul t0, t1, t2\nli t0, 1\n")
    assert estimate_order(instructions).issue_cycles == (0, 3)
    graph = InstructionScheduler().build_dag(instructions)
    assert graph[1].predecessors == [(graph[0], 3)]


def test_latency_sensitive_gain_is_rejected():
    source = "sw t0, 8(a0)\nmul t1, t2, t3\nlw t4, 0(a0)\n"
    result = schedule_assembly(source, ScheduleConfig(strict=True))
    assert result.asm_text == source
    assert result.stats["sensitivity_rejected_regions"] == 1
    row = result.stats["regions"][0]
    assert (row["original_cycles"], row["candidate_cycles"], row["final_cycles"]) == (5, 4, 5)
    assert row["sensitivity_original_cycles"] == row["sensitivity_candidate_cycles"] == 4
    assert row["status"] == "model_sensitive"


@pytest.mark.parametrize("op", ["div", "divu", "rem", "remu"])
def test_no_instruction_can_cross_a_division(op):
    source = f"li t4, 1\n{op} t0, t1, t2\nli t3, 2\n"
    scheduler = InstructionScheduler()
    instructions = parse_instructions(source)
    assert [i.id for i in scheduler.schedule(scheduler.build_dag(instructions))] == [0, 1, 2]
    assert schedule_assembly(source).asm_text == source


def test_empty_legacy_latency_overrides_accept_an_explicit_model():
    model = ScheduleModel(overrides={"mul": 7})
    assert InstructionScheduler(latency_model={}, model=model).model is model


def test_report_has_its_own_cli_channel(tmp_path, capsys):
    source = tmp_path / "case.dsl"
    source.write_text("x = add(a, b)\nreturn x\n")
    status = main([str(source), "-o", str(tmp_path / "out.s"), "--schedule", "--schedule-report"])
    output = capsys.readouterr().err
    assert status == 0
    assert "\nInstruction Scheduling Report" in output
    assert "note: Instruction Scheduling Report" not in output


@pytest.mark.parametrize("filename", ["014_for_dot.dsl", "017_while_sum.dsl", "019_nested_loop.dsl"])
@pytest.mark.parametrize("enabled", [False, True])
def test_linear_control_flow_is_encoded_and_has_symbolic_targets(tmp_path, filename, enabled):
    source = Path(__file__).parents[1] / "benchmarks" / "cases" / filename
    result = CompilerDriver(CompilerConfig(reg_alloc="linear", schedule=enabled)).compile(
        str(source), str(tmp_path / "out.s"),
    )
    assert result.success, result.errors
    binary = assemble_to_binary(result.output_text)
    assert len(binary) > 0 and len(binary) % 4 == 0
    assert ".label" not in result.output_text
    targets = [i.effects.target for i in parse_instructions(result.output_text) if i.effects.target]
    assert targets
    assert all(f"{target}:" in result.output_text for target in targets)
    if enabled:
        assert not any("numeric control-flow target" in d["reason"]
                       for d in result.stats["schedule"]["diagnostics"])


def test_linear_target_is_not_renamed_and_first_branch_operand_is_a_use():
    v = MachineOperand.vreg
    machine = MachineInstr(MachineOp.BNE, v("condition"), v("other"), comment="condition")
    block = block_from_machine_instrs([machine])
    assert block[0].uses == {"condition", "other"}
    assert not block[0].defines
    assert block[0].to_asm({"condition": "t0", "other": "t1"}).strip() == "bne t0, t1, condition"
    restored = machine_instrs_from_block(block)[0]
    assert restored.comment == "condition"
    explicit = LsInstruction(0, "j", ["target"], comment="an ordinary comment")
    assert explicit.to_asm().strip() == "j target  # an ordinary comment"


def test_zero_sources_are_canonical_but_immediates_keep_their_meaning():
    r, n = MachineOperand.reg, MachineOperand.immediate
    assert MachineInstr(MachineOp.SLT, r("t0"), n(0), r("t1")).src1 == r("zero")
    assert MachineInstr(MachineOp.MV, r("t0"), n(0)).src1 == r("zero")
    immediate = MachineInstr(MachineOp.ADDI, r("t0"), n(0), n(0))
    assert immediate.src1 == r("zero") and immediate.src2 == n(0)
    assert MachineInstr(MachineOp.LI, r("t0"), n(0)).src1 == n(0)
