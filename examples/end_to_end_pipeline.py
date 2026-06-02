#!/usr/bin/env python3
"""End-to-end pipeline: ONNX model → LLVM IR → verify against ONNX Runtime.

Usage:
    python examples/end_to_end_pipeline.py
    python examples/end_to_end_pipeline.py --backend riscv

This demonstrates the complete ScratchV flow:
  1. Generate an ONNX model OR use DSL
  2. Parse → IR → Optimize → Codegen (RISC-V or LLVM)
  3. Verify against numpy/ONNX Runtime reference
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np


def demo_add(backend: str):
    """Simple A + B with both backends."""
    print("\n" + "=" * 60)
    print("DEMO 1: Element-wise Add")
    print("=" * 60)

    dsl_source = "y = add(a, b)\nreturn y"

    # Compile
    from scratchv.frontend.dsl_parser import DSLParser
    parser = DSLParser()
    program = parser.parse(dsl_source)

    if backend == "llvm":
        from scratchv.backend.llvm_codegen import LLVMCodegen
        codegen = LLVMCodegen(program)
        output = codegen.emit()
        print(f"\nLLVM IR output:\n{output[:500]}...\n")
    else:
        from scratchv.backend.instruction_select import InstructionSelector
        from scratchv.backend.register_alloc import RegisterAllocator
        from scratchv.backend.asm_emit import AsmEmitter
        selector = InstructionSelector(program)
        machine = selector.run()
        alloc = RegisterAllocator(machine, mode="greedy")
        allocated = alloc.run()
        emitter = AsmEmitter(allocated)
        output = emitter.emit()
        print(f"\nRISC-V Assembly output:\n{output[:600]}...\n")

    # Verify
    from scratchv.verification.verifier import DSLInterpreter
    interpreter = DSLInterpreter()
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([5.0, 6.0, 7.0, 8.0])
    result = interpreter.run(dsl_source, {"a": a, "b": b})
    expected = a + b
    print(f"  Input a: {a}")
    print(f"  Input b: {b}")
    print(f"  Expected (a+b): {expected}")
    print(f"  Reference result: {result}")
    assert np.allclose(result, expected), "Reference mismatch!"
    print("  ✓ Reference verification passed")


def demo_relu(backend: str):
    """ReLU activation."""
    print("\n" + "=" * 60)
    print("DEMO 2: ReLU Activation")
    print("=" * 60)

    dsl_source = "y = relu(x)\nreturn y"

    from scratchv.frontend.dsl_parser import DSLParser
    parser = DSLParser()
    program = parser.parse(dsl_source)

    # Show LLVM IR for ReLU
    from scratchv.backend.llvm_codegen import LLVMCodegen
    codegen = LLVMCodegen(program)
    llvm_ir = codegen.emit()
    print(f"\nLLVM IR for ReLU:\n{llvm_ir}\n")

    # Verify
    from scratchv.verification.verifier import DSLInterpreter
    interpreter = DSLInterpreter()
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = interpreter.run(dsl_source, {"x": x})
    expected = np.maximum(x, 0.0)
    print(f"  Input: {x}")
    print(f"  ReLU output: {result}")
    assert np.allclose(result, expected), "Reference mismatch!"
    print("  ✓ Reference verification passed")


def demo_matmul(backend: str):
    """Matrix multiplication with optimizations."""
    print("\n" + "=" * 60)
    print("DEMO 3: Matrix Multiplication (with optimizations)")
    print("=" * 60)

    dsl_source = "c = matmul(A, B, m:2, n:2, k:2)\nreturn c"

    from scratchv.frontend.dsl_parser import DSLParser
    parser = DSLParser()
    program = parser.parse(dsl_source)

    # Optimize
    from scratchv.optimizer.constant_folding import ConstantFolder
    from scratchv.optimizer.dead_code import DeadCodeEliminator
    folder = ConstantFolder(program)
    folded = folder.run()
    elim = DeadCodeEliminator(program)
    eliminated = elim.run()
    print(f"  Optimizer: {folded} folded, {eliminated} eliminated")

    from scratchv.backend.llvm_codegen import LLVMCodegen
    codegen = LLVMCodegen(program)
    llvm_ir = codegen.emit()
    print(f"\nLLVM IR (MatMul):\n{llvm_ir}\n")

    # Verify
    from scratchv.verification.verifier import DSLInterpreter
    interpreter = DSLInterpreter()
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    result = interpreter.run(dsl_source, {"A": A, "B": B})
    expected = A @ B
    print(f"  A:\n{A}")
    print(f"  B:\n{B}")
    print(f"  Expected (A@B):\n{expected}")
    print(f"  Reference result:\n{result}")
    assert np.allclose(result, expected), "Reference mismatch!"
    print("  ✓ Reference verification passed")


def demo_optimized_pipeline():
    """Show how the optimizer improves LLVM IR."""
    print("\n" + "=" * 60)
    print("DEMO 4: Optimizer Impact on LLVM IR")
    print("=" * 60)

    dsl_source = """
x = add(a, b)
y = mul(x, 1.0)
z = add(y, 0.0)
return z
"""

    from scratchv.frontend.dsl_parser import DSLParser

    # Without optimization
    parser1 = DSLParser()
    program1 = parser1.parse(dsl_source)
    from scratchv.backend.llvm_codegen import LLVMCodegen
    codegen1 = LLVMCodegen(program1)
    print("Before optimization:")
    print(codegen1.emit()[:400])
    print("...")

    # With optimization
    parser2 = DSLParser()
    program2 = parser2.parse(dsl_source)
    from scratchv.optimizer.constant_folding import ConstantFolder
    from scratchv.optimizer.dead_code import DeadCodeEliminator
    from scratchv.optimizer.peephole import IRPeepholeOptimizer
    folder = ConstantFolder(program2)
    folder.run()
    elim = DeadCodeEliminator(program2)
    elim.run()
    peep = IRPeepholeOptimizer(program2)
    peep.run()
    codegen2 = LLVMCodegen(program2)
    print("After optimization (fold + dce + peephole):")
    print(codegen2.emit()[:400])
    print("...")


def demo_llvm_to_file():
    """Save LLVM IR to .ll file for use with llc/opt."""
    print("\n" + "=" * 60)
    print("DEMO 5: Save LLVM IR to File (for llc/opt)")
    print("=" * 60)

    dsl_source = "y = relu(x)\nreturn y"

    from scratchv.frontend.dsl_parser import DSLParser
    from scratchv.backend.llvm_codegen import LLVMCodegen
    parser = DSLParser()
    program = parser.parse(dsl_source)
    codegen = LLVMCodegen(program)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".ll", mode="w", delete=False) as f:
        f.write(codegen.emit())
        path = f.name

    print(f"  LLVM IR saved to: {path}")
    print(f"  To compile: llc {path} -o {path.replace('.ll', '.s')}")
    print(f"  To optimize: opt -O2 {path} -o {path.replace('.ll', '.opt.bc')}")
    print(f"  To run JIT: lli {path}")

    # Cleanup
    os.unlink(path)


def main():
    parser = argparse.ArgumentParser(description="ScratchV end-to-end pipeline demo")
    parser.add_argument("--backend", choices=["riscv", "llvm"], default="llvm",
                        help="Target backend")
    parser.add_argument("--demo", type=int, choices=[1, 2, 3, 4, 5], default=None,
                        help="Run specific demo only")
    args = parser.parse_args()

    print(f"ScratchV End-to-End Pipeline (backend: {args.backend})")
    print(f"{'=' * 60}")

    demos = {
        1: lambda: demo_add(args.backend),
        2: lambda: demo_relu(args.backend),
        3: lambda: demo_matmul(args.backend),
        4: demo_optimized_pipeline,
        5: demo_llvm_to_file,
    }

    if args.demo:
        demos[args.demo]()
    else:
        for i in range(1, 6):
            demos[i]()

    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
