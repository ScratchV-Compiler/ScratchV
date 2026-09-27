"""Exact dependencies, scheduling times, source preservation and failure paths."""

import random
from dataclasses import FrozenInstanceError

import pytest

from scratchv.backend.inst_scheduler import (
    InstructionScheduler,
    SchedInst,
    ScheduleConfig,
    ScheduleError,
    machine_instrs_from_scheduled,
    parse_instructions,
    schedule_assembly,
)
from scratchv.backend.llvm_mca import metrics as llvm_metrics
from scratchv.backend.schedule_verify import verify_schedule

EXAMPLE = "lw t0, 0(a0)\nadd t1, t0, t2\nmul t3, t3, t4\nmul t5, t3, t6\nret\n"
EXPECTED = "lw t0, 0(a0)\nmul t3, t3, t4\nadd t1, t0, t2\nmul t5, t3, t6\nret\n"


@pytest.mark.parametrize(
    "source,reads,writes",
    [
        ("lw t0, -4(sp)", {"x2"}, {"x5"}),
        ("lw x0, 0x10(fp)", {"x8"}, set()),
        ("sw t0, 0(a0)", {"x5", "x10"}, set()),
        ("sw sp(-4), t0", {"x2", "x5"}, set()),
        ("lw t0, sp(-4)", {"x2"}, {"x5"}),
        ("add t0, x5, zero", {"x5"}, {"x5"}),
        ("ret", {"x1"}, set()),
        ("beq t0, t1, target", {"x5", "x6"}, set()),
        ("jalr zero, 0(ra)", {"x1"}, set()),
        ("jalr zero, ra", {"x1"}, set()),
        ("fsw fa0, 4(a0)", {"f10", "x10"}, set()),
        ("fld fs0, (sp)", {"x2"}, {"f8"}),
        ("fadd.s fa0, ft0, fs0, rtz", {"f0", "f8"}, {"f10"}),
        ("fcvt.w.s a0, fa0, rtz", {"f10"}, {"x10"}),
        ("fmv.s.x ft0, t0", {"x5"}, {"f0"}),
        ("fmadd.d fa0, fa1, fa2, fa3", {"f11", "f12", "f13"}, {"f10"}),
    ],
)
def test_exact_instruction_objects(source, reads, writes):
    inst = parse_instructions(source)[0]
    assert not inst.effects.barrier_reason
    assert inst.uses == reads
    assert inst.defines == writes
    assert inst.raw_line == source
    with pytest.raises(FrozenInstanceError):
        inst.opcode = "nop"


@pytest.mark.parametrize(
    "source",
    [
        "custom t0, t1",
        "call f",
        "jal f",
        "jal ra, f",
        "jalr ra, t0, 0",
        "ecall",
        "fence",
        "csrr t0, fflags",
        "amoadd.w t0, t1, (a0)",
        "li t0, 4098",
        "li.d f0, 1.0",
        "la t0, label",
        "auipc t0, 1",
        "addi t0, t1, %lo(symbol)",
        "lw t0, broken",
        "add %v0, t0, t1",
        "add t0, t1",
        "slli t0, t1, 32",
        "fld f32, 0(a0)",
        "fmin.s f0, f1, f2, rtz",
        "feq.s t0, f1, f2, rtz",
    ],
)
def test_unknown_forms_are_explicit_boundaries(source):
    inst = parse_instructions(source)[0]
    assert inst.effects.barrier_reason
    result = schedule_assembly(EXAMPLE + source + "\n" + EXAMPLE)
    assert result.asm_text == EXPECTED + source + "\n" + EXPECTED
    assert result.diagnostics


def graph(source):
    return InstructionScheduler().build_dag(parse_instructions(source))


def test_three_register_dependencies_and_deduplication():
    nodes = graph("add t1, t0, t2\nmul t0, t3, t4\nadd t0, t0, t1")
    assert nodes[1].predecessors == [(nodes[0], 0)]
    assert nodes[1].edge_kinds[0] == {"WAR"}
    assert nodes[2].predecessors == [(nodes[0], 0), (nodes[1], 0)]
    assert nodes[2].edge_kinds[1] == {"RAW", "WAW"}
    assert nodes[2].edge_kinds[0] == {"RAW"}


def test_all_readers_must_precede_overwrite():
    nodes = graph("add t1, t0, t2\nsub t3, t0, t4\nli x5, 9")
    assert nodes[2].predecessors == [(nodes[0], 0), (nodes[1], 0)]


def test_zero_has_no_register_dependencies_but_load_still_has_memory_effect():
    nodes = graph("lw zero, 0(a0)\nadd t0, x0, t1\nsw t2, 4(a0)")
    assert nodes[1].predecessors == []
    assert nodes[2].predecessors == [(nodes[0], 0)]
    assert nodes[2].edge_kinds[0] == {"memory"}


@pytest.mark.parametrize(
    "first,second",
    [
        ("lw t0, 0(a0)", "lw t1, 4(a0)"),
        ("lw t0, 0(a0)", "sw t1, 4(a0)"),
        ("sw t0, 0(a0)", "lw t1, 4(a0)"),
        ("sw t0, 0(a0)", "sw t1, 4(a0)"),
    ],
)
def test_memory_order_even_with_different_offsets(first, second):
    nodes = graph(first + "\n" + second)
    assert nodes[1].predecessors == [(nodes[0], 0)]
    assert nodes[1].edge_kinds[0] == {"memory"}


def test_fp_status_effects_are_ordered():
    nodes = graph("fadd.s f0, f1, f2\nfdiv.s f3, f4, f5")
    assert nodes[1].edge_kinds[0] == {"fp-effects"}


def test_exact_heights_and_load_use_wait():
    nodes = graph("lw t0, 0(a0)\nmul t1, t0, t2\nadd t3, t1, t4")
    assert [n.priority for n in nodes] == [3, 2, 1]
    result = schedule_assembly(EXAMPLE, ScheduleConfig(strict=True))
    assert result.asm_text == EXPECTED
    assert result.stats["original_cycles"] == 9
    assert result.stats["final_cycles"] == 7
    assert result.stats["original_stalls"] == 2
    assert result.stats["final_stalls"] == 1
    row = result.stats["regions"][0]
    assert row["original_issue_cycles"] == [0, 1, 2, 5, 6]
    assert row["candidate_issue_cycles"] == [0, 0, 1, 3, 4]


def test_division_preserves_order_and_llvm_models_occupancy():
    scheduler = InstructionScheduler()
    original = parse_instructions("div t0, t1, t2\nadd t3, t4, t5\nrem t6, t1, t2")
    result = scheduler.schedule(scheduler.build_dag(original))
    assert [i.id for i in result] == [0, 1, 2]
    measured = llvm_metrics(result)
    assert measured["issue_cycles"] == [0, 63, 65]
    assert measured["cycles"] == 131


@pytest.mark.parametrize(
    "newline,last_newline", [("\n", True), ("\n", False), ("\r\n", True)]
)
def test_labels_directives_comments_and_newlines_survive(newline, last_newline):
    original = (
        ".text\n.globl f\nf:\nlw t0, 0(a0) # load\n"
        "add t1, t0, t2 # consume\nmul t3, t3, t4 # independent\nmul t5, t3, t6\nret\n"
        ".size f, .-f\n\n# notes\n.data\nvalue: .word 3\n"
    )
    expected = original.replace(
        "lw t0, 0(a0) # load\nadd t1, t0, t2 # consume\nmul t3, t3, t4 # independent",
        "lw t0, 0(a0) # load\nmul t3, t3, t4 # independent\nadd t1, t0, t2 # consume",
    )
    if not last_newline:
        original, expected = original.rstrip("\n"), expected.rstrip("\n")
    assert schedule_assembly(
        original.replace("\n", newline)
    ).asm_text == expected.replace("\n", newline)


@pytest.mark.parametrize(
    "separator", ["\n", "# comment\n", ".align 2\n", "label:\n", "1:\n"]
)
def test_no_crossing_non_instruction_lines(separator):
    original = "lw t0, 0(a0)\nadd t1, t0, t2\n" + separator + "addi t3, t3, 1\n"
    assert schedule_assembly(original).asm_text == original


def test_inline_label_and_terminator_bundle_stay_fixed():
    source = "entry: lw t0, 0(a0)\n" + EXAMPLE.replace(
        "ret\n", "beq t1, t2, target\nj other\n"
    )
    result = schedule_assembly(source, ScheduleConfig(strict=True))
    assert result.asm_text.startswith("entry: lw t0, 0(a0)\n")
    assert result.asm_text.endswith("beq t1, t2, target\nj other\n")
    assert result.stats["modeled_instructions"] == 6


@pytest.mark.parametrize(
    "source", ["", "# notes", "ret\n", "1:\n2:\n", ".data\n" + EXAMPLE]
)
def test_empty_short_and_non_code_inputs(source):
    assert schedule_assembly(source, ScheduleConfig(strict=True)).asm_text == source


@pytest.mark.parametrize(
    "unsafe",
    [
        ".macro f\n.endm",
        '.include "extra.s"',
        "j 8",
        "beq t0, t1, 8",
        "add t0, t1, t2; ret",
        "lw t0, .-label(a0)",
    ],
)
def test_unsafe_assembly_layout_preserves_entire_input(unsafe):
    source = EXAMPLE + unsafe + "\n"
    result = schedule_assembly(source, ScheduleConfig(strict=True))
    assert result.asm_text == source
    assert "entire input preserved" in result.diagnostics[0]["reason"]


def test_low_level_api_cannot_silently_drop_boundaries():
    with pytest.raises(ScheduleError, match="Multiple regions"):
        graph("lw t0, 0(a0)\nlabel:\nadd t1, t0, t2")
    with pytest.raises(ValueError):
        machine_instrs_from_scheduled([SchedInst(0, "made_up")])


def test_control_target_metadata_cannot_override_assembly():
    with pytest.raises(ValueError, match="Conflicting instruction targets"):
        SchedInst(0, "beq", ["t0", "t1", ".left"], target=".right")


@pytest.mark.parametrize("source", ["call helper", "jal ra, helper", "jal zero, helper"])
def test_machine_conversion_preserves_direct_calls_and_jumps(source):
    from scratchv.backend.asm_emit import AsmEmitter

    original = parse_instructions(source)
    restored = parse_instructions(AsmEmitter(machine_instrs_from_scheduled(original)).emit())
    assert [(i.opcode, i.operands, i.target) for i in restored] == [
        (i.opcode, i.operands, i.target) for i in original
    ]


def test_verifier_checks_reads_without_relying_on_graph():
    original = parse_instructions("add t1, t0, t2\nmul t0, t3, t4")
    with pytest.raises(ScheduleError, match="Register"):
        verify_schedule(original, list(reversed(original)))
    with pytest.raises(ScheduleError, match="identities"):
        verify_schedule(original, [original[0], original[0]])


def test_failure_restores_whole_region_and_strict_mode_raises(monkeypatch):
    def broken(self, dag):
        return [n.inst for n in dag[:-1]]

    monkeypatch.setattr(InstructionScheduler, "schedule", broken)
    result = schedule_assembly(EXAMPLE)
    assert result.asm_text == EXAMPLE
    assert result.stats["regions"][0]["status"] == "restored"
    assert result.diagnostics[0]["severity"] == "warning"
    with pytest.raises(ScheduleError, match="Line 1"):
        schedule_assembly(EXAMPLE, ScheduleConfig(strict=True))


def test_cycles_in_graph_never_return_partial_output():
    scheduler = InstructionScheduler()
    nodes = graph("li t0, 1\nli t1, 2")
    nodes[0].predecessors.append((nodes[1], 0))
    nodes[1].successors.append((nodes[0], 0))
    nodes[1].predecessors.append((nodes[0], 0))
    nodes[0].successors.append((nodes[1], 0))
    with pytest.raises(ScheduleError, match="cycle"):
        scheduler.schedule(nodes)


def test_equal_or_worse_candidates_are_not_applied(monkeypatch):
    def reverse(self, dag):
        return [n.inst for n in reversed(dag)]

    monkeypatch.setattr(InstructionScheduler, "schedule", reverse)
    for source in ("li t0, 1\nli t1, 2\n", "mul t0, t1, t2\nli t3, 3\n"):
        result = schedule_assembly(source, ScheduleConfig(strict=True))
        assert result.asm_text == source
        assert result.stats["regions"][0]["status"] == "no_improvement"
        assert result.stats["saved_cycles"] == 0


def test_large_region_skip_and_iterative_dependency_chain():
    source = "addi t0, t0, 1\n" * 1100
    result = schedule_assembly(source)
    assert result.asm_text == source
    assert result.stats["skipped_regions"] == 1
    result = schedule_assembly(
        source, ScheduleConfig(strict=True, max_region_size=1200)
    )
    assert result.stats["modeled_instructions"] == 1100


def test_determinism_and_idempotence_on_legal_physical_registers():
    rng = random.Random(18)
    for _ in range(50):
        source = EXAMPLE.replace("ret\n", "")
        for _ in range(30):
            dst, a, b = (rng.choice(["t0", "t1", "t2", "t3", "t4"]) for _ in range(3))
            source += f"{rng.choice(['add', 'sub', 'mul'])} {dst}, {a}, {b}\n"
        first = schedule_assembly(source, ScheduleConfig(strict=True))
        second = schedule_assembly(source, ScheduleConfig(strict=True))
        assert first.asm_text == second.asm_text
        assert first.stats["regions"] == second.stats["regions"]
        assert schedule_assembly(first.asm_text).asm_text == first.asm_text
