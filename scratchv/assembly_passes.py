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


def _schedule(
    text: str,
    *,
    strict: bool = False,
    report: bool = False,
    llvm_mca: str | None = None,
    stats: dict | None = None,
) -> PassResult:
    from scratchv.backend.inst_scheduler import ScheduleConfig, schedule_assembly

    result = schedule_assembly(text, ScheduleConfig(strict=strict, llvm_mca=llvm_mca))
    if stats is not None:
        stats["schedule"] = {**result.stats, "diagnostics": result.diagnostics}
        if report:
            stats["schedule"]["report"] = result.report()
    warnings = [
        f"Schedule line {item['line']}: {item['reason']}"
        for item in result.diagnostics
        if item["severity"] == "warning"
    ]
    # Report text changes separately from the scheduler's performance metrics.
    return PassResult(result.asm_text, int(result.asm_text != text), warnings=warnings)


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


def create_assembly_registry(
    *,
    schedule_strict: bool = False,
    schedule_report: bool = False,
    llvm_mca: str | None = None,
    stats: dict | None = None,
) -> PassRegistry:
    """Keep assembly names separate from the IR registry to prevent misrouting."""
    registry = PassRegistry()
    for name, action in (
        ("asm-peephole", _peephole),
        ("const-merge", _const_merge),
        (
            "schedule",
            partial(
                _schedule,
                strict=schedule_strict,
                report=schedule_report,
                llvm_mca=llvm_mca,
                stats=stats,
            ),
        ),
        ("beautify", _beautify),
        ("count-instr", _count),
    ):
        registry.register(name, partial(AssemblyPass, name, action))
    return registry
