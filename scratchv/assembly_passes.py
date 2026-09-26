"""PassManager adapters for the existing post-codegen implementations."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from scratchv.pass_interface import CompilerPass, PassResult
from scratchv.pass_manager import PassRegistry


class AssemblyPass(CompilerPass):
    """Adapt a text transformation or read-only analysis to CompilerPass."""

    def __init__(self, name: str, action: Callable[[str], PassResult]) -> None:
        self._name = name
        self._action = action

    @property
    def name(self) -> str:
        return self._name

    def run(self, input_data: Any) -> PassResult:
        if not isinstance(input_data, str):
            raise TypeError(f"{self.name} requires assembly text")
        return self._action(input_data)


def _peephole(text: str) -> PassResult:
    from scratchv.backend.asm_peephole import AsmPeepholeOptimizer

    optimizer = AsmPeepholeOptimizer()
    output, changes = optimizer.optimize(text)
    warnings = []
    if changes:
        warnings.append(
            f"Asm peephole: {changes} changes, "
            f"{optimizer.instructions_saved} instr saved "
            f"({optimizer.instructions_before}->{optimizer.instructions_after})"
        )
    return PassResult(output, changes, warnings=warnings)


def _const_merge(text: str) -> PassResult:
    from scratchv.backend.const_merge import merge_constants_detailed

    output, stats = merge_constants_detailed(text)
    warnings = []
    if stats.total_changes:
        warnings.append(
            f"Const merge: {stats.total_changes} changes "
            f"({stats.merged_pairs} pairs, "
            f"{stats.redundant_lui_removed} redundant lui)"
        )
    return PassResult(output, stats.total_changes, warnings=warnings)


def _schedule(text: str) -> PassResult:
    from scratchv.backend.inst_scheduler import InstructionScheduler, parse_instructions

    scheduler = InstructionScheduler()
    instructions = parse_instructions(text)
    scheduled = scheduler.schedule(scheduler.build_dag(instructions))
    output = "\n".join(
        f"  {instruction.opcode} " + ", ".join(instruction.operands)
        for instruction in scheduled
    )
    # Existing scheduler has no edit count: report whether the text changed.
    return PassResult(output, int(output != text))


def _beautify(text: str) -> PassResult:
    from scratchv.backend.asm_beautifier import beautify_asm

    output = beautify_asm(text)
    return PassResult(output, int(output != text))


def _count(text: str) -> PassResult:
    from scratchv.backend.inst_counter import count_instructions

    counts = count_instructions(text)
    total = sum(
        value
        for key, value in counts.items()
        if not key.startswith("_") and isinstance(value, int)
    )
    return PassResult(text, warnings=[f"Instruction count: {total}"])


def create_assembly_registry() -> PassRegistry:
    """Keep assembly names separate from the IR registry to prevent misrouting."""
    registry = PassRegistry()
    for name, action in (
        ("asm-peephole", _peephole),
        ("const-merge", _const_merge),
        ("schedule", _schedule),
        ("beautify", _beautify),
        ("count-instr", _count),
    ):
        registry.register(name, partial(AssemblyPass, name, action))
    return registry
