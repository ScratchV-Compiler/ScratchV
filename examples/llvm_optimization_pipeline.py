#!/usr/bin/env python3
"""Demonstrate LLVM optimization pipeline through opt-level analysis.

Shows how the ScratchV optimizer + LLVM backend work together.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    print("ScratchV LLVM Optimization Pipeline Demo")
    print("=" * 60)

    # A DSL program with optimization opportunities
    dsl_source = """
x = add(input, bias)
y = mul(x, 1.0)        # peephole: redundant mul by 1
z = add(y, 0.0)        # peephole: redundant add by 0
t = mul(z, scale)      # this one stays
w = add(t, offset)
result = relu(w)
return result
"""

    from scratchv.frontend.dsl_parser import DSLParser
    from scratchv.backend.llvm_codegen import LLVMCodegen
    from scratchv.ir.printer import IRPrinter

    # --- Without optimization ---
    print("\n[Without optimization]")
    parser = DSLParser()
    program = parser.parse(dsl_source)

    codegen = LLVMCodegen(program)
    unopt_ir = codegen.emit()
    line_count_unopt = len(unopt_ir.strip().split("\n"))
    print(f"  LLVM IR lines: {line_count_unopt}")
    print(f"  Contains 'fmul': {'fmul' in unopt_ir}")
    print(f"  Redundant ops preserved (x*1, y+0)")

    # --- With basic optimization ---
    print("\n[With basic optimization: fold + dce]")
    parser2 = DSLParser()
    program2 = parser2.parse(dsl_source)

    from scratchv.optimizer.constant_folding import ConstantFolder
    from scratchv.optimizer.dead_code import DeadCodeEliminator
    folder = ConstantFolder(program2)
    folded = folder.run()
    elim = DeadCodeEliminator(program2)
    eliminated = elim.run()
    print(f"  Folded: {folded}, Eliminated: {eliminated}")

    codegen2 = LLVMCodegen(program2)
    basic_ir = codegen2.emit()
    line_count_basic = len(basic_ir.strip().split("\n"))
    print(f"  LLVM IR lines: {line_count_basic}")

    # --- With full optimization ---
    print("\n[With full optimization: fold + dce + peephole]")
    parser3 = DSLParser()
    program3 = parser3.parse(dsl_source)

    folder3 = ConstantFolder(program3)
    folder3.run()
    elim3 = DeadCodeEliminator(program3)
    elim3.run()
    from scratchv.optimizer.peephole import PeepholeOptimizer
    peep = PeepholeOptimizer(program3)
    peeped = peep.run()
    print(f"  Folded+DCE+Peephole: {peeped} optimizations")

    codegen3 = LLVMCodegen(program3)
    opt_ir = codegen3.emit()
    line_count_opt = len(opt_ir.strip().split("\n"))
    print(f"  LLVM IR lines: {line_count_opt}")

    # Summary
    print("\n" + "=" * 60)
    print("Optimization Summary:")
    print(f"  Unoptimized:   {line_count_unopt} lines")
    print(f"  Basic opt:     {line_count_basic} lines")
    print(f"  Full opt:      {line_count_opt} lines")
    reduction = ((line_count_unopt - line_count_opt) / line_count_unopt) * 100
    print(f"  Reduction:     {reduction:.1f}%")

    # Show the optimized LLVM IR
    print("\nOptimized LLVM IR:")
    print("-" * 40)
    print(opt_ir)

    # Check for key patterns
    has_fmul = "fmul" in opt_ir
    has_fadd = "fadd" in opt_ir
    has_select = "select" in opt_ir  # ReLU pattern
    print(f"\n  Has fmul (real mul): {has_fmul}")
    print(f"  Has fadd (real add): {has_fadd}")
    print(f"  Has select (ReLU):   {has_select}")


if __name__ == "__main__":
    main()
