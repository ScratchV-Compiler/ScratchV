"""Retain mul/add until shared IR defines a supported fusion contract."""

from __future__ import annotations

from scratchv.ir.types import Program
from scratchv.pass_interface import OptimizationPass


class MulAddFusion(OptimizationPass):
    """Compatibility pass; separate multiply/add preserve checked semantics.

    Shared IR ADD is binary and has no fused_mul_add attribute. A real fusion
    needs an explicit opcode with interpreter/backend support, defined rounding
    and intermediate-error behavior, and proof that the multiply has no other
    users. Until those exist, the registered pass conservatively retains both
    instructions, including shared multiply results.
    """

    name = "muladd-fusion"

    def optimize(self, program: Program) -> int:
        return 0
