"""Unified compiler pass interface for ScratchV.

All optimisation, analysis, and code-generation passes implement this
protocol so they can be composed by ``PassManager`` and ``CompilerDriver``.

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

    The ``run`` method accepts and returns arbitrary data — passes are
    composed by ``PassManager`` which chains their inputs and outputs.
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
