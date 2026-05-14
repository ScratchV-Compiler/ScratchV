#!/usr/bin/env python3
"""Example: verify generated assembly with TinyFive and count instructions.

Usage:
    python examples/verify_with_tinyfive.py examples/simple_add.dsl
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scratchv.frontend.dsl_parser import DSLParser
from scratchv.backend.instruction_select import InstructionSelector
from scratchv.backend.register_alloc import RegisterAllocator
from scratchv.backend.asm_emit import AsmEmitter
from scratchv.optimizer.constant_folding import ConstantFolder
from scratchv.optimizer.dead_code import DeadCodeEliminator


def compile_and_count(path: str, optimize: bool = False) -> tuple[str, int]:
    """Compile a DSL file and count instructions."""
    with open(path) as f:
        source = f.read()

    parser = DSLParser()
    program = parser.parse(source)

    if optimize:
        folder = ConstantFolder(program)
        folded = folder.run()
        elim = DeadCodeEliminator(program)
        eliminated = elim.run()
        print(f"  Optimization: {folded} folded, {eliminated} eliminated")

    selector = InstructionSelector(program)
    instrs = selector.run()
    alloc = RegisterAllocator(instrs, mode="greedy")
    allocated = alloc.run()
    emitter = AsmEmitter(allocated)
    asm = emitter.emit()

    # Try to verify with TinyFive
    from scratchv.simulator.tinyfive import verify_assembly
    result = verify_assembly(asm, verbose=True)

    return asm, result.get("instr_count", 0)


def main():
    if len(sys.argv) < 2:
        print("Usage: python examples/verify_with_tinyfive.py <file.dsl>")
        sys.exit(1)

    path = sys.argv[1]

    print(f"Compiling: {path}")
    print("=" * 40)

    # Without optimization
    print("\nWithout optimization:")
    asm_before, count_before = compile_and_count(path, optimize=False)

    # With optimization
    print("\nWith optimization:")
    asm_after, count_after = compile_and_count(path, optimize=True)

    print("\n" + "=" * 40)
    print(f"Instructions before: {count_before}")
    print(f"Instructions after:  {count_after}")
    if count_before > 0 and count_after > 0:
        reduction = ((count_before - count_after) / count_before) * 100
        print(f"Reduction: {reduction:.1f}%")
    elif count_before > 0:
        print("Note: TinyFive not available — instruction counts are 0")
        print("Install with: pip install tinyfive numpy")


if __name__ == "__main__":
    main()
