"""Unified compiler pass interfaces for ScratchV.

IR optimization passes use the strongly typed ``OptimizationPass`` contract.
The older generic ``CompilerPass`` and ``PassResult`` types remain available
for compatibility with non-optimization callers.

Usage::

    from scratchv.pass_interface import CompilerPass, PassResult

    class MyPass(CompilerPass):
        @property
        def name(self) -> str:
            return "my-pass"

        def run(self, input_data):
            # ... transform ...
            return PassResult(
                data=output, changes=42,
                message="42 patterns folded",
            )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from scratchv.ir.types import Program


class OptimizationPass(ABC):
    """Abstract interface implemented by every IR optimization pass."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a stable, non-empty name for reports and diagnostics."""
        ...

    @abstractmethod
    def optimize(self, program: Program) -> int:
        """Optimize ``program`` in place and return this run's change count."""
        ...


@dataclass(frozen=True)
class PassExecutionStats:
    """Statistics for one execution in an optimization pipeline."""

    index: int
    name: str
    changes: int
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise TypeError("pass execution index must be an integer")
        if self.index < 0:
            raise ValueError("pass execution index must be non-negative")
        if not isinstance(self.name, str):
            raise TypeError("pass execution name must be a string")
        if not self.name.strip():
            raise ValueError("pass execution name must not be empty")
        if isinstance(self.changes, bool) or not isinstance(self.changes, int):
            raise TypeError("pass changes must be an integer")
        if self.changes < 0:
            raise ValueError("pass changes must be non-negative")
        if isinstance(self.elapsed_seconds, bool) or not isinstance(
            self.elapsed_seconds, (int, float)
        ):
            raise TypeError("pass elapsed time must be numeric")
        if self.elapsed_seconds < 0:
            raise ValueError("pass elapsed time must be non-negative")


@dataclass(frozen=True)
class OptimizationReport:
    """Ordered statistics produced by one optimization pipeline run."""

    pipeline_name: str
    executions: tuple[PassExecutionStats, ...]
    total_changes: int
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline_name, str):
            raise TypeError("pipeline name must be a string")
        if not self.pipeline_name.strip():
            raise ValueError("pipeline name must not be empty")
        if not isinstance(self.executions, tuple):
            raise TypeError("pipeline executions must be a tuple")
        if [item.index for item in self.executions] != list(
            range(len(self.executions))
        ):
            raise ValueError("pass execution indexes must be consecutive")
        if isinstance(self.total_changes, bool) or not isinstance(
            self.total_changes, int
        ):
            raise TypeError("total changes must be an integer")
        if self.total_changes < 0:
            raise ValueError("total changes must be non-negative")
        if self.total_changes != sum(item.changes for item in self.executions):
            raise ValueError("total changes must equal the execution sum")
        if isinstance(self.elapsed_seconds, bool) or not isinstance(
            self.elapsed_seconds, (int, float)
        ):
            raise TypeError("pipeline elapsed time must be numeric")
        if self.elapsed_seconds < 0:
            raise ValueError("pipeline elapsed time must be non-negative")


class OptimizationPassError(RuntimeError):
    """A pass failed or returned a value that violates its contract."""

    def __init__(
        self,
        pass_index: int,
        pass_name: str,
        elapsed_seconds: float,
        completed_report: OptimizationReport,
        cause: Exception,
    ) -> None:
        self.pass_index = pass_index
        self.pass_name = pass_name
        self.elapsed_seconds = elapsed_seconds
        self.completed_report = completed_report
        self.cause = cause

        super().__init__(
            f"optimization pass #{pass_index} '{pass_name}' failed: {cause}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PassResult — uniform return value for all passes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PassResult:
    """Result of running a compiler pass.

    Attributes:
        data:     The transformed data (IR Program, machine instr list,
                  assembly text, etc.).
        changes:  Number of transformations / fixes applied.
        message:  Human-readable summary (e.g. "3 constants folded").
        warnings: Non-fatal issues discovered during the pass.
    """

    data: Any
    changes: int = 0
    message: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """A pass succeeds if it produced data (warnings are non-fatal)."""
        return self.data is not None


# ═══════════════════════════════════════════════════════════════════════════════
# CompilerPass — abstract base
# ═══════════════════════════════════════════════════════════════════════════════

class CompilerPass(ABC):
    """Abstract base for all compiler passes.

    Subclasses must implement ``name`` and ``run``.  ``stats`` is optional
    and defaults to an empty dict.

    The legacy ``run`` method accepts and returns arbitrary data. IR
    optimization pipelines instead use the typed ``OptimizationPass``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique, human-readable pass name (e.g. 'constant-folding')."""
        ...

    @abstractmethod
    def run(self, input_data: Any) -> PassResult:
        """Execute the pass on *input_data* and return a ``PassResult``."""
        ...

    @property
    def stats(self) -> dict[str, Any]:
        """Optional per-pass statistics (e.g. timing, counts)."""
        return {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"
