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
from scratchv.compiler import create_optimization_pass_manager


def compile_and_count(path: str, optimize: bool = False) -> tuple[str, int]:
    """Compile a DSL file and count instructions."""
    with open(path) as f:
        source = f.read()

    parser = DSLParser()
    program = parser.parse(source)

    if optimize:
        # "basic" is the stable constant-folding -> dead-code-elim pipeline.
        # run() optimizes program in place and returns an immutable report.
        report = create_optimization_pass_manager("basic").run(program)
        changes_by_name = {
            execution.name: execution.changes
            for execution in report.executions
        }
        folded_changes = changes_by_name["constant-folding"]
        eliminated_changes = changes_by_name["dead-code-elim"]
        print(
            "  Optimization: "
            f"{folded_changes} constant fold(s), "
            f"{eliminated_changes} dead-code elimination(s)"
        )

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
