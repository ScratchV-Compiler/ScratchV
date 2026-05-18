# flake8: noqa
"""ScratchV compiler benchmark suite.

Measures compilation pipeline performance across ONNX models:
- Parse time (ONNX → IR)
- IR size (instruction count)
- Optimization time & effectiveness
- Codegen time (RISC-V / LLVM)
- Verification correctness

Usage:
    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --model resnet18
    python benchmarks/run_benchmark.py --backend llvm
    python benchmarks/run_benchmark.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

BENCH_DIR = os.path.dirname(__file__)
PROJ_DIR = os.path.dirname(BENCH_DIR)
sys.path.insert(0, PROJ_DIR)

from benchmarks.generate_models import ensure_all_models
from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.frontend.dsl_parser import DSLParser
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import Program


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    model_name: str
    model_path: str
    backend: str
    optimize_level: str
    parse_time_s: float
    ir_inst_count: int
    ir_bb_count: int
    optimize_time_s: float = 0.0
    ir_opt_inst_count: int = 0
    codegen_time_s: float = 0.0
    asm_line_count: int = 0
    total_time_s: float = 0.0
    verified: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Compiler pipeline
# ---------------------------------------------------------------------------

def _count_ir(program: Program) -> tuple[int, int]:
    inst = sum(1 for f in program.functions for bb in f.blocks for _ in bb.instructions)
    bb = sum(len(f.blocks) for f in program.functions)
    return inst, bb


def _parse_onnx(path: str) -> Program:
    parser = ONNXParser()
    return parser.parse(path)


def _optimize(program: Program, level: str) -> float:
    """Run optimizations. Returns elapsed time in seconds."""
    if level == "none":
        return 0.0

    t0 = time.perf_counter()

    from scratchv.optimizer.constant_folding import ConstantFolder
    from scratchv.optimizer.dead_code import DeadCodeEliminator

    ConstantFolder(program).run()
    DeadCodeEliminator(program).run()

    if level == "all":
        from scratchv.optimizer.peephole import IRPeepholeOptimizer
        from scratchv.optimizer.muladd_fusion import MulAddFusion
        from scratchv.optimizer.licm import LICM
        IRPeepholeOptimizer(program).run()
        MulAddFusion(program).run()
        LICM(program).run()

    return time.perf_counter() - t0


def _codegen_riscv(program: Program) -> tuple[str, float]:
    t0 = time.perf_counter()
    from scratchv.backend.instruction_select import InstructionSelector
    from scratchv.backend.register_alloc import RegisterAllocator
    from scratchv.backend.asm_emit import AsmEmitter
    selector = InstructionSelector(program)
    machine = selector.run()
    alloc = RegisterAllocator(machine, mode="greedy")
    allocated = alloc.run()
    emitter = AsmEmitter(allocated)
    asm = emitter.emit()
    elapsed = time.perf_counter() - t0
    return asm, elapsed


def _codegen_llvm(program: Program) -> tuple[str, float]:
    t0 = time.perf_counter()
    from scratchv.backend.llvm_codegen import LLVMCodegen
    codegen = LLVMCodegen(program)
    ir_str = codegen.emit()
    elapsed = time.perf_counter() - t0
    return ir_str, elapsed


def _verify_onnx(model_path: str, program: Program, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Verify compiled result against ONNX Runtime reference."""
    try:
        from scratchv.verification.verifier import ONNXVerifier
        verifier = ONNXVerifier(rtol=rtol, atol=atol)
        verifier.verify(model_path, program)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_benchmark(model_name: str, model_path: str, *,
                  backend: str = "riscv",
                  optimize_level: str = "all",
                  verify: bool = False) -> BenchResult:
    """Run compilation benchmark on a single model."""
    result = BenchResult(
        model_name=model_name,
        model_path=model_path,
        backend=backend,
        optimize_level=optimize_level,
        parse_time_s=0.0,
        ir_inst_count=0,
        ir_bb_count=0,
    )
    t_start = time.perf_counter()

    try:
        # 1. Parse
        t0 = time.perf_counter()
        program = _parse_onnx(model_path)
        result.parse_time_s = time.perf_counter() - t0
        result.ir_inst_count, result.ir_bb_count = _count_ir(program)

        # 2. Optimize
        if optimize_level != "none":
            result.optimize_time_s = _optimize(program, optimize_level)
            result.ir_opt_inst_count, _ = _count_ir(program)
        else:
            result.ir_opt_inst_count = result.ir_inst_count

        # 3. Codegen
        if backend == "llvm":
            asm_str, result.codegen_time_s = _codegen_llvm(program)
        else:
            asm_str, result.codegen_time_s = _codegen_riscv(program)
        result.asm_line_count = len(asm_str.splitlines())

        # 4. Verify
        if verify:
            result.verified = _verify_onnx(model_path, program)

    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"

    result.total_time_s = time.perf_counter() - t_start
    return result


def run_all_benchmarks(models: dict[str, str], backend: str = "riscv",
                       optimize_level: str = "all",
                       verify: bool = False) -> list[BenchResult]:
    """Run benchmarks on all models."""
    results = []
    for name, path in models.items():
        print(f"  Benchmarking {name} ({path}) ...", end=" ", flush=True)
        r = run_benchmark(name, path, backend=backend,
                          optimize_level=optimize_level, verify=verify)
        if r.error:
            print(f"ERROR: {r.error}")
        else:
            print(f"done ({r.total_time_s:.3f}s, {r.ir_inst_count} IR inst, "
                  f"{r.asm_line_count} asm lines)")
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_summary(results: list[BenchResult]):
    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY")
    print("=" * 90)
    header = f"{'Model':<16} {'Backend':<8} {'Parse(s)':<10} {'IR inst':<8} {'Opt(s)':<10} {'CG(s)':<10} {'Total(s)':<10} {'Asm':<8} {'OK':<5}"
    print(header)
    print("-" * 90)
    for r in results:
        opt_str = f"{r.ir_inst_count}→{r.ir_opt_inst_count}" if r.optimize_level != "none" else str(r.ir_inst_count)
        verified = "✓" if r.verified else ("✗" if r.error else "-")
        print(f"{r.model_name:<16} {r.backend:<8} {r.parse_time_s:<10.4f} {opt_str:<8} "
              f"{r.optimize_time_s:<10.4f} {r.codegen_time_s:<10.4f} {r.total_time_s:<10.4f} "
              f"{r.asm_line_count:<8} {verified:<5}")
        if r.error:
            print(f"  ERROR: {r.error}")
    print("-" * 90)

    # Totals
    total_parse = sum(r.parse_time_s for r in results)
    total_opt = sum(r.optimize_time_s for r in results)
    total_cg = sum(r.codegen_time_s for r in results)
    total_all = sum(r.total_time_s for r in results)
    print(f"{'TOTAL':<16} {'':<8} {total_parse:<10.4f} {'':<8} {total_opt:<10.4f} {total_cg:<10.4f} {total_all:<10.4f}")
    print(f"\nModels benchmarked: {len(results)}")
    errors = [r for r in results if r.error]
    if errors:
        print(f"Errors: {len(errors)}")
        for r in errors:
            print(f"  - {r.model_name}: {r.error}")


def save_results(results: list[BenchResult], output_path: str):
    data = [asdict(r) for r in results]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ScratchV compiler benchmark")
    parser.add_argument("--model", type=str, default=None,
                        help="Run only a specific model (simple_cnn, add_128, resnet18)")
    parser.add_argument("--backend", choices=["riscv", "llvm"], default="riscv")
    parser.add_argument("--optimize", choices=["none", "basic", "all"], default="all")
    parser.add_argument("--verify", action="store_true", help="Verify against ONNX Runtime")
    parser.add_argument("--output", type=str, default=None, help="Save JSON results")
    parser.add_argument("--list", action="store_true", help="List available models")
    args = parser.parse_args()

    models = ensure_all_models()

    if args.list:
        for name, path in models.items():
            size_kb = os.path.getsize(path) / 1024
            print(f"  {name}: {path} ({size_kb:.1f} KB)")
        return

    if args.model:
        if args.model not in models:
            print(f"Unknown model: {args.model}. Available: {list(models.keys())}")
            sys.exit(1)
        models = {args.model: models[args.model]}

    print(f"ScratchV Benchmark Suite")
    print(f"  Backend: {args.backend}, Optimize: {args.optimize}, Verify: {args.verify}")
    print(f"  Models: {len(models)}")

    results = run_all_benchmarks(models, backend=args.backend,
                                 optimize_level=args.optimize, verify=args.verify)
    print_summary(results)

    if args.output:
        save_results(results, args.output)

    # Exit with error if any benchmark failed
    errors = [r for r in results if r.error]
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
