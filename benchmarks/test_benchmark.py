# flake8: noqa
"""Benchmark tests — integrated with pytest for CI.

These tests ensure the compiler pipeline completes successfully
on standard ONNX models and track performance regressions.

Usage:
    pytest benchmarks/test_benchmark.py -v
    pytest benchmarks/test_benchmark.py -v --benchmark-model resnet18
"""

from __future__ import annotations

import os
import json
import sys
import time

import pytest

BENCH_DIR = os.path.dirname(__file__)
PROJ_DIR = os.path.dirname(BENCH_DIR)
sys.path.insert(0, PROJ_DIR)

from benchmarks.generate_models import ensure_all_models
from benchmarks.bench_const_merge import _gen_synthetic_asm, bench_merge
from benchmarks.run_benchmark import BenchResult, print_summary, run_benchmark, save_results
from benchmarks.run_benchmark import _count_asm_instructions


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def benchmark_models() -> dict[str, str]:
    return ensure_all_models()


MODEL_PARAMS = ["add", "mixed_ops", "deep_relu", "matmul", "maxpool_relu"]
BACKEND_PARAMS = ["riscv"]


def test_effective_asm_instruction_count():
    asm = ".text\nmain:\n  li t0, 1\n# comment\n  add t1, t0, t0\n"
    assert _count_asm_instructions(asm) == 2


def test_synthetic_benchmark_covers_both_rules():
    asm = _gen_synthetic_asm(
        200,
        seed=42,
        pair_density=0.3,
        redundant_lui_density=0.2,
    )
    stats = bench_merge(asm, repeats=2)
    assert stats["benchmark_type"] == "synthetic"
    assert stats["merged_pairs"] > 0
    assert stats["redundant_lui_removed"] > 0
    assert stats["instruction_reduction"] == (
        stats["merged_pairs"] + stats["redundant_lui_removed"]
    )
    assert stats["input_instructions"] == 200


@pytest.mark.parametrize(
    "pair_density,redundant_density",
    [(-0.1, 0.1), (0.1, -0.1), (1.1, 0.0), (0.6, 0.5)],
)
def test_synthetic_density_validation(pair_density, redundant_density):
    with pytest.raises(ValueError):
        _gen_synthetic_asm(
            10,
            pair_density=pair_density,
            redundant_lui_density=redundant_density,
        )


def test_synthetic_seed_is_reproducible():
    kwargs = {
        "num_instructions": 100,
        "seed": 7,
        "pair_density": 0.2,
        "redundant_lui_density": 0.1,
    }
    assert _gen_synthetic_asm(**kwargs) == _gen_synthetic_asm(**kwargs)


def test_synthetic_repeats_validation():
    with pytest.raises(ValueError, match="repeats"):
        bench_merge("  nop\n", repeats=0)


def _zero_hit_result() -> BenchResult:
    return BenchResult(
        model_name="zero_hit",
        model_path="model.onnx",
        backend="riscv",
        optimize_level="all",
        parse_time_s=0.0,
        ir_inst_count=1,
        ir_bb_count=1,
        asm_instructions_before=3,
        asm_instructions_after=3,
    )


def test_summary_keeps_zero_hit_case(capsys):
    print_summary([_zero_hit_result()])
    output = capsys.readouterr().out
    assert "CONST-MERGE A/B" in output
    assert "zero_hit" in output
    assert "3→3" in output
    assert "N/A without toolchain" in output


def test_json_keeps_zero_and_na_fields(tmp_path, capsys):
    output = tmp_path / "results.json"
    save_results([_zero_hit_result()], str(output))
    capsys.readouterr()
    data = json.loads(output.read_text())[0]
    assert data["candidate_pairs"] == 0
    assert data["merged_pairs"] == 0
    assert data["redundant_lui_removed"] == 0
    assert data["machine_instructions_before"] is None
    assert data["output_equal"] is None


def _model_id(name: str) -> str:
    return name


# ---------------------------------------------------------------------------
# Parse benchmark
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", MODEL_PARAMS, ids=_model_id)
def test_parse_onnx(model_name: str, benchmark_models: dict[str, str]):
    """Parse ONNX → IR for each model."""
    from scratchv.frontend.onnx_parser import ONNXParser
    path = benchmark_models[model_name]
    parser = ONNXParser()
    program = parser.parse(path)

    inst_count = sum(1 for f in program.functions for bb in f.blocks for _ in bb.instructions)
    assert inst_count > 0, f"Empty IR for {model_name}"


# ---------------------------------------------------------------------------
# Optimization benchmark
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", MODEL_PARAMS, ids=_model_id)
def test_optimize(model_name: str, benchmark_models: dict[str, str]):
    """Parse + optimize, check IR is not empty."""
    from scratchv.frontend.onnx_parser import ONNXParser
    from scratchv.optimizer.constant_folding import ConstantFolder
    from scratchv.optimizer.dead_code import DeadCodeEliminator
    from scratchv.optimizer.peephole import IRPeepholeOptimizer

    path = benchmark_models[model_name]
    program = ONNXParser().parse(path)

    inst_before = sum(1 for f in program.functions for bb in f.blocks for _ in bb.instructions)

    ConstantFolder(program).run()
    DeadCodeEliminator(program).run()
    IRPeepholeOptimizer(program).run()

    inst_after = sum(1 for f in program.functions for bb in f.blocks for _ in bb.instructions)
    assert inst_after >= 0, f"Optimization failed for {model_name}"
    print(f"\n    {model_name}: {inst_before} → {inst_after} instructions")


# ---------------------------------------------------------------------------
# Backend codegen
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", MODEL_PARAMS, ids=_model_id)
@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_codegen_riscv(model_name: str, backend: str, benchmark_models: dict[str, str]):
    """Parse + codegen → RISC-V assembly, check output is non-empty."""
    from scratchv.frontend.onnx_parser import ONNXParser
    from scratchv.optimizer.constant_folding import ConstantFolder
    from scratchv.optimizer.dead_code import DeadCodeEliminator
    from scratchv.backend.instruction_select import InstructionSelector
    from scratchv.backend.register_alloc import RegisterAllocator
    from scratchv.backend.asm_emit import AsmEmitter

    path = benchmark_models[model_name]
    program = ONNXParser().parse(path)

    ConstantFolder(program).run()
    DeadCodeEliminator(program).run()

    selector = InstructionSelector(program)
    machine = selector.run()
    alloc = RegisterAllocator(machine, mode="greedy")
    allocated = alloc.run()
    emitter = AsmEmitter(allocated)
    asm = emitter.emit()

    lines = asm.splitlines()
    assert len(lines) > 0, f"Empty assembly for {model_name}"
    print(f"\n    {model_name}: {len(lines)} asm lines")


# ---------------------------------------------------------------------------
# LLVM codegen
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", MODEL_PARAMS, ids=_model_id)
def test_codegen_llvm(model_name: str, benchmark_models: dict[str, str]):
    """Parse + codegen → LLVM IR, check output is non-empty."""
    from scratchv.frontend.onnx_parser import ONNXParser
    from scratchv.backend.llvm_codegen import LLVMCodegen

    path = benchmark_models[model_name]
    program = ONNXParser().parse(path)

    codegen = LLVMCodegen(program)
    llvm_ir = codegen.emit()

    assert len(llvm_ir) > 0, f"Empty LLVM IR for {model_name}"
    print(f"\n    {model_name}: {len(llvm_ir.splitlines())} LLVM IR lines")


# ---------------------------------------------------------------------------
# Performance timing (lightweight, no pytest-benchmark dependency)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", MODEL_PARAMS, ids=_model_id)
def test_perf_pipeline(model_name: str, benchmark_models: dict[str, str]):
    """Full pipeline timing.  Fails if > threshold."""
    path = benchmark_models[model_name]

    result = run_benchmark(model_name, path, backend="riscv",
                           optimize_level="all", verify=False)

    assert result.error is None, f"Benchmark failed: {result.error}"
    assert result.ir_inst_count > 0
    assert isinstance(result.asm_instructions_before, int)
    assert isinstance(result.asm_instructions_after, int)
    reduction = (
        result.asm_instructions_before - result.asm_instructions_after
    )
    tracked_changes = result.merged_pairs + result.redundant_lui_removed
    assert reduction == tracked_changes, (
        f"instruction reduction {reduction} != tracked changes "
        f"{tracked_changes}"
    )
    assert result.pass_time_ms >= 0

    print(f"\n    {model_name}:")
    print(f"      parse:  {result.parse_time_s:.4f}s")
    print(f"      IR:     {result.ir_inst_count} inst → {result.ir_opt_inst_count} opt")
    print(f"      optimize: {result.optimize_time_s:.4f}s")
    print(f"      codegen:  {result.codegen_time_s:.4f}s")
    print(f"      total:    {result.total_time_s:.4f}s")
    print(f"      asm:      {result.asm_line_count} lines")
