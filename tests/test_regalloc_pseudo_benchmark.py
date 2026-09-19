"""Coverage and execution tests for the per-pseudo benchmark."""

import json

import pytest

from benchmarks.test_regalloc import bench_cnn, bench_pseudo, bench_regalloc_linear
from benchmarks.test_regalloc.semantics_compare import (
    compare_semantics,
    legacy_positional_block,
)
from scratchv.backend.machine_types import ALL_REGS, MachineInstr, MachineOperand
from scratchv.backend.machine_types import MachineOp
from scripts import bench_single_ops


EXPECTED_MACHINE_PSEUDOS = {
    MachineOp.MV,
    MachineOp.LI,
    MachineOp.MAX,
    MachineOp.BNEZ,
    MachineOp.J,
    MachineOp.CALL,
    MachineOp.LABEL,
}


def test_benchmark_matrix_covers_every_supported_machine_pseudo() -> None:
    cases = bench_pseudo.machine_pseudo_cases()

    assert bench_pseudo.BENCHMARKED_MACHINE_PSEUDOS == EXPECTED_MACHINE_PSEUDOS
    assert {case.opcode for case in cases} == EXPECTED_MACHINE_PSEUDOS
    assert len(cases) == len(EXPECTED_MACHINE_PSEUDOS)
    for case in cases:
        assert any(instr.op is case.opcode for instr in case.instructions)


def test_benchmark_matrix_covers_encoder_only_pseudos() -> None:
    cases = bench_pseudo.assembler_pseudo_cases()

    assert bench_pseudo.BENCHMARKED_ASSEMBLER_PSEUDOS == {"nop", "ret"}
    assert {case.name for case in cases} == {"nop", "ret"}


def test_every_pseudo_allocates_encodes_and_executes() -> None:
    stats = bench_pseudo.run_bench(repeats=1)

    assert stats["case_count"] == 9
    assert stats["valid"]
    assert stats["spill_slots"] == 0
    assert stats["spill_stores"] == 0
    assert stats["reloads"] == 0
    assert all(case["valid"] for case in stats["cases"])
    assert all(case["actual_a0"] == case["expected_a0"] for case in stats["cases"])


def test_pseudo_benchmark_rejects_non_positive_repeats() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        bench_pseudo.run_bench(repeats=0)


def test_pseudo_metrics_are_report_serializable() -> None:
    stats = bench_pseudo.run_bench(repeats=1)
    public_stats = {
        key: value for key, value in stats.items() if not key.startswith("_")
    }
    results = {"4. Pseudo Instructions": public_stats}

    json.dumps(results)
    html = bench_regalloc_linear._make_html(results, 0.0)
    markdown = bench_regalloc_linear._make_markdown(results)
    assert "4. Pseudo Instructions" in html
    assert "4. Pseudo Instructions" in markdown
    assert "Pseudo-instruction detail" in markdown
    assert "legacy positional" in markdown
    assert "Metric definitions" in markdown
    assert "`Spills`/`Reloads`" in markdown
    assert "Physical-register banks used" in markdown


def test_summary_uses_total_vregs_and_true_pressure_peak() -> None:
    results = {
        "2. Dense Computation": {
            "mean_s": 0.001,
            "stdev_s": 0.0,
            "vreg_count": 30,
            "phys_reg_count": 5,
            "reg_spill_count": 60,
            "peak_active": 5,
            "pressure_peak": 30,
            "reloads": 74,
            "asm_lines": 214,
            "valid": True,
        }
    }

    markdown = bench_regalloc_linear._make_markdown(results)
    html = bench_regalloc_linear._make_html(results, 0.0)

    assert "| 2. Dense Computation | 1.000 | 0.000 | 30 | 60 | 30 | 74 |" in markdown
    assert "Dense Computation=5" in markdown
    assert "<td>30</td><td>60</td><td>30</td><td>74</td>" in html
    assert "This is not the number of spilled virtual registers" in html


def test_bnez_detail_exposes_the_fixed_operand_semantics() -> None:
    case = next(case for case in bench_pseudo.machine_pseudo_cases()
                if case.opcode is MachineOp.BNEZ)
    target = next(instruction for instruction in case.instructions
                  if instruction.op is MachineOp.BNEZ)
    legacy = legacy_positional_block([target])[0]

    assert legacy.defines == {"condition"}
    assert legacy.uses == set()

    result = next(
        result for result in bench_pseudo.run_bench(repeats=1)["cases"]
        if result["name"] == "bnez"
    )
    assert result["legacy_defs"] == ("condition",)
    assert result["semantic_defs"] == ()
    assert result["semantic_uses"] == ("condition",)
    assert result["expanded_rv32_instructions"] == 1
    assert [point["physical_register_count"]
            for point in result["pressure_sweep"]] == [2, 3, 5, 8, 12, 19]
    tight = result["pressure_sweep"][0]
    assert tight["before"]["pressure_peak"] == 2
    assert tight["after"]["pressure_peak"] == 3
    assert not tight["before"]["valid"]
    assert tight["after"]["valid"]
    assert tight["after"]["spill_stores"] > 0


def test_semantics_comparison_uses_one_allocator_and_input() -> None:
    vreg = MachineOperand.vreg
    instructions = [
        MachineInstr(MachineOp.LI, vreg("condition"), MachineOperand.immediate(1)),
        MachineInstr(MachineOp.BNEZ, vreg("condition"), comment=".done"),
        MachineInstr(MachineOp.LABEL, comment=".done"),
    ]

    comparison = compare_semantics(instructions, list(ALL_REGS[:2]))

    assert comparison["baseline"] == "legacy positional dst/src inference"
    assert comparison["current"] == "machine_semantics.py"
    assert comparison["before"]["intervals"]["condition"]["uses"] == []
    assert comparison["after"]["intervals"]["condition"]["uses"] == [1]


def test_single_operator_report_exposes_pressure_sweep() -> None:
    counts = bench_single_ops._parse_pressure_regs("2,3,5,19")
    aggregates = {
        "relu": {
            "model_count": 1,
            "machine_instructions": 4,
            "semantics_differences": 1,
            "pseudo_counts": {"max": 1},
            "pressure_sweep": {
                str(count): {
                    "before_valid": True,
                    "after_valid": True,
                    "before": {
                        "pressure_peak": 2,
                        "spill_slots": 1,
                        "spill_stores": 1,
                        "reloads": 1,
                        "static_instructions": 6,
                        "encoded_instructions": 7,
                    },
                    "after": {
                        "pressure_peak": 3,
                        "spill_slots": 2,
                        "spill_stores": 2,
                        "reloads": 2,
                        "static_instructions": 8,
                        "encoded_instructions": 9,
                    },
                }
                for count in counts
            },
        }
    }

    markdown = bench_single_ops._regalloc_markdown(aggregates, counts)

    assert counts == (2, 3, 5, 19)
    assert "Changed semantics" in markdown
    assert "Pressure before/after" in markdown
    assert "| relu | 2 | 1 | 4 | 1 | max:1 | 2 / 3 |" in markdown


def test_reports_show_comparable_before_after_optimization() -> None:
    stats = {
        "greedy_time_s": 0.002,
        "mean_s": 0.001,
        "greedy_static_instrs": 100,
        "sv_static_instrs": 75,
        "greedy_spill_slots": 20,
        "spill_slots": 0,
        "greedy_spill_stores": 40,
        "spill_stores": 10,
        "greedy_reloads": 50,
        "reloads": 25,
        "greedy_asm_valid": True,
        "greedy_emu_passed": True,
        "asm_valid": True,
        "emu_passed": True,
    }
    stats["optimization_comparison"] = (
        bench_cnn._build_optimization_comparison(stats)
    )
    results = {"3. CNN Integration": stats}

    markdown = bench_regalloc_linear._make_markdown(results)
    html = bench_regalloc_linear._make_html(results, 0.0)

    assert "CNN allocator comparison: Greedy vs LinearScan" in markdown
    assert "Before (Greedy allocator)" in markdown
    assert "After (Topic17 LinearScan)" in markdown
    assert "| Allocation mean | 2.000 ms | 1.000 ms | 50.00% better |" in markdown
    assert (
        "| Static instructions | 100 instructions | 75 instructions | "
        "25.00% better |"
    ) in markdown
    assert "Correctness (assembly + emulator): **PASS -> PASS**" in markdown
    assert "CNN allocator comparison: Greedy vs LinearScan" in html
    assert "25.00% better" in html


def test_optimization_percentage_handles_zero_baseline() -> None:
    assert bench_cnn._improvement_pct(0, 0) == 0
    assert bench_cnn._improvement_pct(0, 1) is None
