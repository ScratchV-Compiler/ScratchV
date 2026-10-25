"""Registered compatibility pass; fusion requires a valid IR/lowering first."""

from __future__ import annotations

from scratchv.ir.types import Program
from scratchv.pass_interface import OptimizationPass


class MulAddFusion(OptimizationPass):
    """Preserve MUL/ADD until a fused opcode has a signature and lowering.

    RV32IM has no integer multiply-add instruction. A three-operand ADD is
    invalid IR, and both selectors consume only two sources. This pass stays
    registered so presets/explicit pipelines remain compatible, but performs
    no transformations. FP fusion would additionally require an explicit
    contraction/rounding policy. The old constructor switch is retained for
    callers, not as permission to emit the invalid legacy pseudo.
    """

    name = "muladd-fusion"

    def __init__(self, *, preserve_float_semantics: bool = False):
        self.preserve_float_semantics = preserve_float_semantics

    def optimize(self, program: Program) -> int:
        """Return zero changes without inventing an unsupported instruction."""
        return 0
