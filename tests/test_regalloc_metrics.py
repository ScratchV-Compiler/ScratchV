"""Topic17 acceptance tests for pressure and spill metric alignment."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.test_regalloc import bench_cnn, bench_dense
from scratchv.backend.machine_types import ALL_REGS
from scratchv.backend.regalloc_metrics import (
    count_spill_reload_sites,
    peak_live_intervals,
)


def test_peak_pressure_counts_overlap_instead_of_cumulative_spills():
    intervals = [
        SimpleNamespace(start=0, end=2),
        SimpleNamespace(start=1, end=2),
        SimpleNamespace(start=3, end=5),
        SimpleNamespace(start=4, end=5),
    ]

    assert peak_live_intervals(intervals) == 2


def test_spill_metrics_ignore_ordinary_memory_operations():
    assembly = """\
sw t0, 0(sp)  # model store
lw t1, 0(sp)  # model load
sw t4, 4(sp)  # spill user-comment-only
lw t4, 4(sp)  # reload user-comment-only
sw t2, -4(sp)  # spill value [regalloc:spill]
lw t2, -4(sp)  # reload value [regalloc:reload]
sw t3, -8(sp)  # evict other [regalloc:spill]
"""

    assert count_spill_reload_sites(assembly) == (2, 1)


def test_dense_benchmark_separates_sites_slots_reloads_and_pressure():
    block = bench_dense._gen_block(num_insts=80, num_vregs=30)
    stats = bench_dense.bench_allocate(block, list(ALL_REGS[:5]), 1)

    assert stats["reg_spill_count"] == stats["spill_stores"]
    assert stats["vreg_count"] == 30
    assert stats["phys_reg_count"] == 5
    assert stats["spill_stores"] > stats["spill_slots"] > 0
    assert stats["reloads"] > 0
    assert stats["pressure_peak"] > 5
    assert stats["pressure_excess_peak"] == stats["pressure_peak"] - 5
    assert stats["execution_valid"]
    assert stats["actual_a0"] == stats["expected_a0"]


def test_topic17_cnn_uses_19_regs_and_passes_real_assembly_validation():
    model = Path(__file__).parents[1] / "models" / "graph" / "cnn.onnx"

    stats = bench_cnn.bench_allocate(str(model), list(ALL_REGS), repeats=1)

    assert stats["asm_valid"], stats["asm_errors"]
    assert len(stats["_alloc"].phys_regs) == 19
    assert stats["phys_reg_count"] == 19
    assert stats["pressure_peak"] == 11
    assert stats["pressure_excess_peak"] == 0
    assert stats["spill_slots"] == 0
    assert stats["spill_stores"] == 0
    assert stats["reloads"] == 0
    assert stats["reg_spill_count"] == stats["spill_stores"]


def test_topic17_cnn_executes_the_allocated_assembly_it_measures():
    model = Path(__file__).parents[1] / "models" / "graph" / "cnn.onnx"

    stats = bench_cnn.run_bench(str(model), list(ALL_REGS), repeats=1)

    assert stats["emu_passed"], stats["emu_error"]
    assert stats["actual_a0"] == stats["expected_a0"]
    assert stats["greedy_asm_valid"], stats["greedy_asm_errors"]
    assert stats["greedy_emu_passed"], stats["greedy_emu_error"]
    comparison = stats["optimization_comparison"]
    assert comparison["same_machine_ir"]
    assert comparison["correctness"] == {"before": True, "after": True}
    assert (
        comparison["metrics"]["static_instructions"]["before"]
        == stats["greedy_static_instrs"]
    )
    assert (
        comparison["metrics"]["static_instructions"]["after"]
        == stats["sv_static_instrs"]
    )
    assert comparison["metrics"]["spill_slots"]["improvement_pct"] == 100


def test_real_assembly_validation_rejects_unresolved_named_vreg():
    errors = bench_cnn._validate_asm("add layer1.bias, t0, t1")

    assert errors
    assert "unknown register" in errors[0]


@pytest.mark.parametrize(
    ("asm_valid", "emu_passed", "expected"),
    [(True, True, True), (False, True, False), (True, False, False)],
)
def test_cnn_benchmark_requires_assembly_and_emulator_validity(
    monkeypatch: pytest.MonkeyPatch,
    asm_valid: bool,
    emu_passed: bool,
    expected: bool,
) -> None:
    monkeypatch.setattr(
        bench_cnn,
        "bench_allocate",
        lambda *_args, **_kwargs: {
            "asm_valid": asm_valid,
            "sv_static_instrs": 1,
        },
    )
    monkeypatch.setattr(
        bench_cnn,
        "_run_emulator",
        lambda *_args, **_kwargs: {"passed": emu_passed},
    )
    monkeypatch.setattr(
        bench_cnn,
        "_llvm_compare",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    stats = bench_cnn.run_bench("unused.onnx", repeats=1)

    assert stats["valid"] is expected
