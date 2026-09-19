"""Coverage and execution tests for the per-pseudo benchmark."""

import json

import pytest

from benchmarks.test_regalloc import bench_cnn, bench_pseudo, bench_regalloc_linear
from scratchv.backend.machine_types import MachineOp


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

    assert "CNN Register Allocation: Before vs After" in markdown
    assert "Before (Greedy allocator)" in markdown
    assert "After (Topic17 LinearScan)" in markdown
    assert "| Allocation mean | 2.000 ms | 1.000 ms | 50.00% better |" in markdown
    assert (
        "| Static instructions | 100 instructions | 75 instructions | "
        "25.00% better |"
    ) in markdown
    assert "Correctness (assembly + emulator): **PASS -> PASS**" in markdown
    assert "CNN Register Allocation: Before vs After" in html
    assert "25.00% better" in html


def test_optimization_percentage_handles_zero_baseline() -> None:
    assert bench_cnn._improvement_pct(0, 0) == 0
    assert bench_cnn._improvement_pct(0, 1) is None
