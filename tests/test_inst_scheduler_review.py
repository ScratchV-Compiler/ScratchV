"""Scheduler safety regressions; current behavior is documented in docs/topic-18/."""

from pathlib import Path

import pytest

from benchmarks.run_inst_scheduler_case import _instruction_count
from scratchv.backend.inst_scheduler import (
    InstructionScheduler, ScheduleConfig, parse_instructions, schedule_assembly,
)
from scratchv.backend.llvm_mca import metrics as llvm_metrics
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.main import main

CASE = "lw t0, 0(a0)\nadd t1, t0, t2\nsw t1, 4(a0)\naddi t3, t3, 1\n"


@pytest.mark.parametrize("literal", ['"a;b"', r'"a\\b"', '"."', '"a#;b"', r'"a\";b"'])
def test_quoted_data_does_not_disable_text_scheduling(literal):
    source = f".data\nmessage: .asciz {literal}\n.text\n" + CASE
    result = schedule_assembly(source, ScheduleConfig(strict=True))
    assert result.changed
    assert result.asm_text.startswith(source[:source.index(".text")])
    assert result.stats["saved_cycles"] == 2


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
    instructions = parse_instructions("div t0, t1, t2\nli t0, 1\n")
    assert llvm_metrics(instructions)["issue_cycles"] == [0, 63]
    graph = InstructionScheduler().build_dag(instructions)
    assert graph[1].predecessors == [(graph[0], 0)]


def test_llvm_regression_is_rejected(monkeypatch):
    # Moving the increment after the store is legal but increases LLVM cycles.
    source = "lw t0,0(a0)\naddi t3,t3,1\nadd t1,t0,t2\nsw t1,4(a0)\n"
    monkeypatch.setattr(InstructionScheduler, "schedule", lambda self, dag: [dag[i].inst for i in (0, 2, 3, 1)])
    result = schedule_assembly(source, ScheduleConfig(strict=True))
    assert result.asm_text == source
    row = result.stats["regions"][0]
    assert row["candidate_cycles"] > row["original_cycles"] == row["final_cycles"]
    assert row["status"] == "no_improvement"


@pytest.mark.parametrize("op", ["div", "divu", "rem", "remu"])
def test_no_instruction_can_cross_a_division(op):
    source = f"li t4, 1\n{op} t0, t1, t2\nli t3, 2\n"
    scheduler = InstructionScheduler()
    instructions = parse_instructions(source)
    assert [i.id for i in scheduler.schedule(scheduler.build_dag(instructions))] == [0, 1, 2]
    assert schedule_assembly(source).asm_text == source


def test_report_has_its_own_cli_channel(tmp_path, capsys):
    source = tmp_path / "case.dsl"
    source.write_text("x = add(a, b)\nreturn x\n")
    status = main([str(source), "-o", str(tmp_path / "out.s"), "--schedule", "--schedule-report"])
    output = capsys.readouterr().err
    assert status == 0
    assert "\nInstruction Scheduling Report" in output
    assert "note: Instruction Scheduling Report" not in output
