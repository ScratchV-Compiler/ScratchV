"""Ordered pass execution and LLVM-inspired named pipeline construction."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from scratchv.ir.types import Program
from scratchv.pass_interface import (
    CompilerPass,
    OptimizationPass,
    OptimizationPassError,
    OptimizationReport,
    PassExecutionStats,
    PassResult,
)


@dataclass(frozen=True)
class PipelineResult:
    """Final data, ordered statistics and diagnostics of a generic pipeline."""

    data: Any
    report: OptimizationReport
    warnings: tuple[str, ...] = ()


class PassManager:
    """Register and run optimization, transformation and analysis passes in order.

    Usage::

        pm = PassManager()
        pm.register(ConstantFolder())
        pm.register(DeadCodeEliminator())
        report = pm.run(program)
    """

    def __init__(self, name: str = "pipeline"):
        if not isinstance(name, str):
            raise TypeError("pipeline name must be a string")
        if not name.strip():
            raise ValueError("pipeline name must not be empty")
        self._name = name
        self._passes: list[OptimizationPass | CompilerPass] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def passes(self) -> list[OptimizationPass | CompilerPass]:
        return list(self._passes)

    def register(
        self, pass_: OptimizationPass | CompilerPass, *, enabled: bool = True
    ) -> PassManager:
        """Append an enabled pass; equivalent to guarding registration with if."""
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        if not isinstance(pass_, (OptimizationPass, CompilerPass)):
            raise TypeError(
                "registered pass must implement OptimizationPass or CompilerPass"
            )
        if not isinstance(pass_.name, str):
            raise TypeError("optimization pass name must be a string")
        if not pass_.name.strip():
            raise ValueError("optimization pass name must not be empty")
        if enabled:
            self._passes.append(pass_)
        return self

    def add(self, pass_: OptimizationPass | CompilerPass) -> PassManager:
        """Compatibility alias for :meth:`register`."""
        return self.register(pass_)

    def run(self, program: Program) -> OptimizationReport:
        """Optimize ``program`` in place and return ordered execution stats."""
        if not isinstance(program, Program):
            raise TypeError("PassManager.run() requires a Program")

        return self._execute(program, in_place=True).report

    def run_pipeline(self, input_data: Any) -> PipelineResult:
        """Chain functional/analysis/transform passes and return the final data."""
        return self._execute(input_data, in_place=False)

    def _execute(self, input_data: Any, *, in_place: bool) -> PipelineResult:
        data = input_data
        warnings: list[str] = []
        executions: list[PassExecutionStats] = []
        for index, pass_ in enumerate(self._passes):
            t0 = time.perf_counter()
            try:
                if isinstance(pass_, OptimizationPass):
                    if not isinstance(data, Program):
                        raise TypeError("OptimizationPass requires a Program")
                    result = PassResult(data, changes=pass_.optimize(data))
                else:
                    result = pass_.run(data)
                    if not isinstance(result, PassResult):
                        raise TypeError("CompilerPass must return PassResult")
                    if not result.success:
                        raise ValueError(result.message or "pass produced no data")
                if in_place and result.data is not input_data:
                    raise ValueError(
                        "run() requires in-place IR passes; use run_pipeline() for replacement data"
                    )
                changes = result.changes
                self._validate_change_count(changes)
                data = result.data
                warnings.extend(result.warnings)
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                completed_report = self._make_report(executions)
                raise OptimizationPassError(
                    pass_index=index,
                    pass_name=pass_.name,
                    elapsed_seconds=elapsed,
                    completed_report=completed_report,
                    cause=exc,
                ) from exc
            elapsed = time.perf_counter() - t0
            executions.append(
                PassExecutionStats(
                    index=index,
                    name=pass_.name,
                    changes=changes,
                    elapsed_seconds=elapsed,
                )
            )

        return PipelineResult(data, self._make_report(executions), tuple(warnings))

    @staticmethod
    def _validate_change_count(changes: int) -> None:
        if isinstance(changes, bool) or not isinstance(changes, int):
            raise TypeError("optimization pass must return an integer change count")
        if changes < 0:
            raise ValueError("optimization pass change count must be non-negative")

    def _make_report(self, executions: list[PassExecutionStats]) -> OptimizationReport:
        records = tuple(executions)
        return OptimizationReport(
            pipeline_name=self._name,
            executions=records,
            total_changes=sum(item.changes for item in records),
            elapsed_seconds=sum((item.elapsed_seconds for item in records), 0.0),
        )

    def report(self) -> str:
        """Return a summary of all registered passes."""
        lines = [f"PassManager '{self._name}' ({len(self._passes)} passes):"]
        for p in self._passes:
            lines.append(f"  {p.name}")
        return "\n".join(lines)


class PassRegistry:
    """Map stable names to factories without scheduling or constructing passes.

    Registration makes a pass available. ``build`` selects an ordered pipeline,
    analogous to LLVM PassBuilder's separation of registration and scheduling.
    Each selected occurrence gets a fresh instance; duplicates are intentional.
    """

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], OptimizationPass | CompilerPass]] = {}

    @property
    def names(self) -> tuple[str, ...]:
        """Return available names in registration order."""
        return tuple(self._factories)

    def register(
        self, name: str, factory: Callable[[], OptimizationPass | CompilerPass]
    ) -> None:
        """Register a factory once; duplicate names are errors, not overrides."""
        if not isinstance(name, str) or not name or name.strip() != name or "," in name:
            raise ValueError(
                "pass registry name must be non-empty, trimmed and comma-free"
            )
        if name in self._factories:
            raise ValueError(f"pass already registered: {name}")
        if not callable(factory):
            raise TypeError("pass factory must be callable")
        self._factories[name] = factory

    def build(
        self,
        names: Iterable[str],
        *,
        disabled: Iterable[str] = (),
        pipeline_name: str = "pipeline",
    ) -> PassManager:
        """Validate all names, then construct only selected, enabled passes."""
        if isinstance(names, str) or isinstance(disabled, str):
            raise TypeError("pass names must be an iterable of names, not a string")
        selected, excluded = tuple(names), tuple(disabled)
        for name in (*selected, *excluded):
            if not isinstance(name, str) or name not in self._factories:
                raise ValueError(
                    f"unknown pass: {name!r}; available: {', '.join(self.names)}"
                )
        manager = PassManager(pipeline_name)
        for name in selected:
            if name in excluded:
                continue
            pass_ = self._factories[name]()
            if not isinstance(pass_, (OptimizationPass, CompilerPass)):
                raise TypeError(f"factory for {name!r} did not return a compiler pass")
            if pass_.name != name:
                raise ValueError(
                    f"factory for {name!r} returned pass named {pass_.name!r}"
                )
            manager.register(pass_)
        return manager


def create_optimization_registry() -> PassRegistry:
    """Register existing IR algorithms without changing their semantics."""
    from scratchv.optimizer.constant_folding import ConstantFolder
    from scratchv.optimizer.dead_code import DeadCodeEliminator
    from scratchv.optimizer.licm import LICM
    from scratchv.optimizer.muladd_fusion import MulAddFusion
    from scratchv.optimizer.peephole import IRPeepholeOptimizer

    registry = PassRegistry()
    for pass_type in (
        ConstantFolder,
        DeadCodeEliminator,
        IRPeepholeOptimizer,
        MulAddFusion,
        LICM,
    ):
        registry.register(pass_type.name, pass_type)
    return registry


def create_optimization_pass_manager(
    level: str,
    *,
    passes: Iterable[str] | None = None,
    disabled_passes: Iterable[str] = (),
) -> PassManager:
    """Build a preset or explicit IR pipeline, then apply named exclusions.

    ``passes=None`` selects the level preset; an empty iterable explicitly
    selects no passes. Exclusions remove every occurrence of a name.
    """
    if not isinstance(level, str):
        raise TypeError("optimization level must be a string")
    presets = {
        "none": (),
        "basic": ("constant-folding", "dead-code-elim"),
        "all": (
            "constant-folding",
            "dead-code-elim",
            "ir-peephole",
            "muladd-fusion",
            "licm",
        ),
    }
    if level not in presets:
        raise ValueError("optimization level must be one of: none, basic, all")
    return create_optimization_registry().build(
        presets[level] if passes is None else passes,
        disabled=disabled_passes,
        pipeline_name="optimizer",
    )
