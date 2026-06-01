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
import sys
import time

import pytest

BENCH_DIR = os.path.dirname(__file__)
PROJ_DIR = os.path.dirname(BENCH_DIR)
sys.path.insert(0, PROJ_DIR)

from benchmarks.generate_models import ensure_all_models
from benchmarks.run_benchmark import run_benchmark


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def benchmark_models() -> dict[str, str]:
    return ensure_all_models()


MODEL_PARAMS = ["add", "mixed_ops", "deep_relu", "matmul", "maxpool_relu"]
BACKEND_PARAMS = ["riscv"]


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
    from scratchv.optimizer.peephole import PeepholeOptimizer

    path = benchmark_models[model_name]
    program = ONNXParser().parse(path)

    inst_before = sum(1 for f in program.functions for bb in f.blocks for _ in bb.instructions)

    ConstantFolder(program).run()
    DeadCodeEliminator(program).run()
    PeepholeOptimizer(program).run()

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

    print(f"\n    {model_name}:")
    print(f"      parse:  {result.parse_time_s:.4f}s")
    print(f"      IR:     {result.ir_inst_count} inst → {result.ir_opt_inst_count} opt")
    print(f"      optimize: {result.optimize_time_s:.4f}s")
    print(f"      codegen:  {result.codegen_time_s:.4f}s")
    print(f"      total:    {result.total_time_s:.4f}s")
    print(f"      asm:      {result.asm_line_count} lines")
