"""Extended Instruction Selector for RISC-V Backend.

Extends the base instruction selector with additional operations:
- sqrt (via library call if no hardware support)
- min/max (using RISC-V branchless or branch sequences)
- abs (absolute value)
- float64 (double-precision floating point) support
- div, rem, mod operations

Usage::

    from scratchv.backend.inst_select_ext import ExtendedInstructionSelector
    selector = ExtendedInstructionSelector(program)
    machine_instrs = selector.run()
"""

from __future__ import annotations

from typing import Optional

# moved import above
from scratchv.backend.instruction_select import InstructionSelector
from scratchv.backend.machine_types import (
    MachineOp, MachineOperand,
)
from scratchv.ir.types import DataType, Instruction, Program


class ExtendedInstructionSelector(InstructionSelector):
    """Extended instruction selector with additional RISC-V op support.

    Extends the base ``InstructionSelector`` to add:
    - sqrt (via software library call or hardware F extension)
    - min/max (branchless sequences)
    - abs (absolute value)
    - float64 (double-precision) operations
    - div, rem, mod (integer)

    Parameters
    ----------
    program:
        The ScratchV IR Program to select instructions for.
    enable_fp64:
        If True, enable float64 (D extension) support.
    use_hardware_sqrt:
        If True, use ``fsqrt.s``/``fsqrt.d`` (requires F/D extension).
        If False, emit a library call to ``sqrtf``/``sqrt``.
    """

    def __init__(self, program: Program, *,
                 enable_fp64: bool = True,
                 use_hardware_sqrt: bool = False):
        super().__init__(program)
        self.enable_fp64 = enable_fp64
        self.use_hardware_sqrt = use_hardware_sqrt
        self._current_dtype: Optional[DataType] = None

    # ------------------------------------------------------------------
    # Base overrides
    # ------------------------------------------------------------------

    def _select_instruction(self, instr: Instruction) -> None:
        """Override to add dtype tracking."""
        if instr.dest is not None:
            self._current_dtype = instr.dest.dtype
        super()._select_instruction(instr)

    # ------------------------------------------------------------------
    # sqrt
    # ------------------------------------------------------------------

    def _select_sqrt(self, instr: Instruction) -> None:
        """Select instruction for sqrt.

        If hardware F extension is available, use ``fsqrt.s``.
        Otherwise emit a library call to ``sqrtf`` (float) or ``sqrt``
        (double).
        """
        src = self._op(instr, 0)
        dst = self._dst(instr)
        dtype = instr.dest.dtype if instr.dest else DataType.FLOAT32

        if self.use_hardware_sqrt:
            if dtype == DataType.FLOAT64:
                self._emit(MachineOp.SQRT_D, dst, src,
                           comment="fsqrt.d (hardware)")
            else:
                self._emit(MachineOp.SQRT_S, dst, src,
                           comment="fsqrt.s (hardware)")
        else:
            # Library call: argument in a0, result in a0
            if src and src.kind != "imm":
                self._emit(MachineOp.MV, MachineOperand.reg("a0"), src,
                           comment="sqrt arg -> a0")
            func = "sqrt" if dtype == DataType.FLOAT64 else "sqrtf"
            self._emit(MachineOp.CALL, comment=func)
            if dst:
                self._emit(MachineOp.MV, dst, MachineOperand.reg("a0"),
                           comment="sqrt result")

    # ------------------------------------------------------------------
    # min / max
    # ------------------------------------------------------------------

    def _select_min(self, instr: Instruction) -> None:
        """Select instruction for min(a, b).

        Integer min (branchless):
            slt tmp, a, b       # tmp = (a < b)
            sub dst, b, a       # diff = b - a
            and tmp, tmp, dst   # mask = tmp & diff
            add dst, a, tmp     # dst = a + mask
        """
        a = self._op(instr, 0)
        b = self._op(instr, 1)
        dst = self._dst(instr)

        if (self.enable_fp64 and instr.dest
                and instr.dest.dtype == DataType.FLOAT64):
            # Use FMIN.D pseudo (expands to branchless sequence)
            self._emit(MachineOp.FMIN_D, dst, a, b, comment="fmin.d")
        else:
            tmp = MachineOperand.vreg("_min_tmp1")
            self._emit(MachineOp.SLT, tmp, a, b, comment="min: slt")
            diff = MachineOperand.vreg("_min_tmp2")
            self._emit(MachineOp.SUB, diff, b, a, comment="min: sub")
            and_tmp = MachineOperand.vreg("_min_tmp3")
            self._emit(
                MachineOp.AND, and_tmp, tmp, diff, comment="min: and"
            )
            if dst:
                self._emit(
                    MachineOp.ADD, dst, a, and_tmp, comment="min result"
                )

    def _select_max(self, instr: Instruction) -> None:
        """Select instruction for max(a, b).

        Uses the existing `max` pseudo-instruction from the base selector,
        or a branchless sequence if not available.
        """
        a = self._op(instr, 0)
        b = self._op(instr, 1)
        dst = self._dst(instr)

        if (self.enable_fp64 and instr.dest
                and instr.dest.dtype == DataType.FLOAT64):
            self._emit(MachineOp.FMAX_D, dst, a, b, comment="fmax.d")
        else:
            # Use existing MAX pseudo (base selector has this)
            self._emit(MachineOp.MAX, dst, a, b, comment="max")

    # ------------------------------------------------------------------
    # abs
    # ------------------------------------------------------------------

    def _select_abs(self, instr: Instruction) -> None:
        """Select instruction for abs(x).

        Integer abs (branchless):
            srai tmp, x, 31     # sign bit broadcast
            xor dst, x, tmp     # invert bits if negative
            sub dst, dst, tmp   # add 1 if negative
        """
        src = self._op(instr, 0)
        dst = self._dst(instr)

        if (self.enable_fp64 and instr.dest
                and instr.dest.dtype == DataType.FLOAT64):
            # fabs.d: clear the sign bit
            self._emit(MachineOp.FABS_D, dst, src, comment="fabs.d")
        else:
            tmp1 = MachineOperand.vreg("_abs_tmp1")
            imm31 = MachineOperand.immediate(31)
            self._emit(
                MachineOp.SRAI, tmp1, src, imm31, comment="abs: srai 31"
            )
            tmp2 = MachineOperand.vreg("_abs_tmp2")
            self._emit(
                MachineOp.XOR, tmp2, src, tmp1, comment="abs: xor"
            )
            if dst:
                self._emit(
                    MachineOp.SUB, dst, tmp2, tmp1, comment="abs: sub"
                )

    # ------------------------------------------------------------------
    # div / rem / mod (integer)
    # ------------------------------------------------------------------

    def _select_idiv(self, instr: Instruction) -> None:
        """Select instruction for integer division."""
        a = self._op(instr, 0)
        b = self._op(instr, 1)
        dst = self._dst(instr)
        self._emit(MachineOp.DIV, dst, a, b, comment="div")

    def _select_rem(self, instr: Instruction) -> None:
        """Select instruction for integer remainder."""
        a = self._op(instr, 0)
        b = self._op(instr, 1)
        dst = self._dst(instr)
        self._emit(MachineOp.REM, dst, a, b, comment="rem")

    def _select_mod(self, instr: Instruction) -> None:
        """Select instruction for modulo (synonym of rem for non-negative)."""
        self._select_rem(instr)

    # ------------------------------------------------------------------
    # float64 (D extension)
    # ------------------------------------------------------------------

    def _select_load_f64(self, instr: Instruction) -> None:
        """Load a 64-bit float from memory."""
        src = self._op(instr, 0)
        dst = self._dst(instr)
        self._emit(MachineOp.FLD, dst, src, comment="fld (load f64)")

    def _select_store_f64(self, instr: Instruction) -> None:
        """Store a 64-bit float to memory."""
        val = self._op(instr, 0)
        addr = self._op(instr, 1)
        self._emit(MachineOp.FSD, val, addr, comment="fsd (store f64)")

    def _select_fadd_d(self, instr: Instruction) -> None:
        """Add two float64 values."""
        a = self._op(instr, 0)
        b = self._op(instr, 1)
        dst = self._dst(instr)
        self._emit(MachineOp.FADD_D, dst, a, b, comment="fadd.d")

    def _select_fsub_d(self, instr: Instruction) -> None:
        """Subtract two float64 values."""
        a = self._op(instr, 0)
        b = self._op(instr, 1)
        dst = self._dst(instr)
        self._emit(MachineOp.FSUB_D, dst, a, b, comment="fsub.d")

    def _select_fmul_d(self, instr: Instruction) -> None:
        """Multiply two float64 values."""
        a = self._op(instr, 0)
        b = self._op(instr, 1)
        dst = self._dst(instr)
        self._emit(MachineOp.FMUL_D, dst, a, b, comment="fmul.d")

    def _select_fdiv_d(self, instr: Instruction) -> None:
        """Divide two float64 values."""
        a = self._op(instr, 0)
        b = self._op(instr, 1)
        dst = self._dst(instr)
        self._emit(MachineOp.FDIV_D, dst, a, b, comment="fdiv.d")

    def _select_fcmp_l_d(self, instr: Instruction) -> None:
        """Float64 less-than comparison."""
        a = self._op(instr, 0)
        b = self._op(instr, 1)
        dst = self._dst(instr)
        self._emit(MachineOp.FLT_D, dst, a, b, comment="flt.d")

    def _select_fcmp_eq_d(self, instr: Instruction) -> None:
        """Float64 equality comparison."""
        a = self._op(instr, 0)
        b = self._op(instr, 1)
        dst = self._dst(instr)
        self._emit(MachineOp.FEQ_D, dst, a, b, comment="feq.d")

    def _select_fcvt_s_d(self, instr: Instruction) -> None:
        """Convert float64 to float32."""
        src = self._op(instr, 0)
        dst = self._dst(instr)
        self._emit(MachineOp.FCVT_S_D, dst, src, comment="fcvt.s.d")

    def _select_fcvt_d_s(self, instr: Instruction) -> None:
        """Convert float32 to float64."""
        src = self._op(instr, 0)
        dst = self._dst(instr)
        self._emit(MachineOp.FCVT_D_S, dst, src, comment="fcvt.d.s")

    def _select_load_const_f64(self, instr: Instruction) -> None:
        """Load a float64 constant."""
        raw_val = instr.attrs.get("value", 0.0)
        assert isinstance(raw_val, (int, float))
        dst = self._dst(instr)
        # Emit as a load from a constant pool (simplified: use li + fcvt)
        # For now, mark as a float64 constant load
        self._emit(MachineOp.LI_D, dst,
                   MachineOperand.immediate(int(raw_val)),
                   comment=f"const f64 {raw_val}")

    # ------------------------------------------------------------------
    # Type-aware overrides for existing ops
    # ------------------------------------------------------------------

    def _select_add(self, instr: Instruction) -> None:
        if self._is_fp64(instr):
            self._select_fadd_d(instr)
        else:
            super()._select_add(instr)

    def _select_sub(self, instr: Instruction) -> None:
        if self._is_fp64(instr):
            self._select_fsub_d(instr)
        else:
            super()._select_sub(instr)

    def _select_mul(self, instr: Instruction) -> None:
        if self._is_fp64(instr):
            self._select_fmul_d(instr)
        else:
            super()._select_mul(instr)

    def _select_div(self, instr: Instruction) -> None:
        if self._is_fp64(instr):
            self._select_fdiv_d(instr)
        elif instr.dest and instr.dest.dtype == DataType.INT32:
            self._select_idiv(instr)
        else:
            super()._select_div(instr)

    def _select_load(self, instr: Instruction) -> None:
        if self._is_fp64(instr):
            self._select_load_f64(instr)
        else:
            super()._select_load(instr)

    def _select_store(self, instr: Instruction) -> None:
        if self._is_fp64(instr):
            self._select_store_f64(instr)
        else:
            super()._select_store(instr)

    def _select_load_const(self, instr: Instruction) -> None:
        if self._is_fp64(instr):
            self._select_load_const_f64(instr)
        else:
            super()._select_load_const(instr)

    def _select_neg(self, instr: Instruction) -> None:
        """Negate: for float64 use fneg.d, for int use sub x0 - x."""
        if self._is_fp64(instr):
            src = self._op(instr, 0)
            dst = self._dst(instr)
            self._emit(MachineOp.FNEG_D, dst, src, comment="fneg.d")
        else:
            super()._select_neg(instr)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_fp64(self, instr: Instruction) -> bool:
        """Check if an instruction operates on float64 data."""
        if not self.enable_fp64:
            return False
        if instr.dest is not None and instr.dest.dtype == DataType.FLOAT64:
            return True
        for op in instr.operands:
            if op.dtype == DataType.FLOAT64:
                return True
        return False

    @property
    def supported_ops(self) -> list[str]:
        """Return list of all supported opcodes in this selector."""
        base_ops = [
            "add", "sub", "mul", "div", "neg", "load_const",
            "load", "store", "alloca", "relu", "gelu", "softmax",
            "maxpool", "for", "endfor", "br", "br_if", "return",
            "label", "matmul", "dot", "conv", "gemm", "sigmoid",
            "reshape", "exp",
        ]
        extended_ops = [
            "sqrt", "min", "max", "abs",
            "idiv", "rem", "mod",
        ]
        fp64_ops = [
            "load_f64", "store_f64", "fadd_d", "fsub_d", "fmul_d",
            "fdiv_d", "fcmp_l_d", "fcmp_eq_d", "fcvt_s_d", "fcvt_d_s",
            "load_const_f64",
        ]
        return (
            base_ops + extended_ops
            + (fp64_ops if self.enable_fp64 else [])
        )


# New MachineOp entries for the extended selector are now defined directly
# in the MachineOp enum in scratchv.backend.register_alloc.
