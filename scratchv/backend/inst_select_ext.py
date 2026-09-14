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

import struct
from typing import Optional

# moved import above
from scratchv.backend.instruction_select import InstructionSelector
from scratchv.backend.machine_types import (
    MachineOp, MachineOperand,
)
from scratchv.ir.types import DataType, Instruction, Program, Value


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

    # fp64-specific opcodes gated by ``enable_fp64``.
    _FP64_OPCODES: frozenset[str] = frozenset({
        "load_f64", "store_f64", "load_const_f64",
        "fadd_d", "fsub_d", "fmul_d", "fdiv_d",
        "fcmp_l_d", "fcmp_eq_d", "fcvt_s_d", "fcvt_d_s",
    })

    def __init__(self, program: Program, *,
                 enable_fp64: bool = True,
                 use_hardware_sqrt: bool = False):
        super().__init__(program)
        self.enable_fp64 = enable_fp64
        self.use_hardware_sqrt = use_hardware_sqrt
        self._current_dtype: Optional[DataType] = None
        self._temp_counter = 0
        # Names of values defined by an emitted instruction (e.g. the dest
        # of ``load_const_f64``).  Such "constants" are addressable vregs
        # and must not be re-materialized as literals.
        self._defined_names: set[str] = set()

    def run(self) -> list:
        """Select instructions, resetting the temp counter for determinism."""
        self._temp_counter = 0
        self._defined_names = {
            instr.dest.name
            for func in self.program.functions
            for block in func.blocks
            for instr in block.instructions
            if instr.dest is not None
        }
        return super().run()

    # ------------------------------------------------------------------
    # Base overrides
    # ------------------------------------------------------------------

    def _select_instruction(self, instr: Instruction) -> None:
        """Override to add fp64 gating and dtype tracking."""
        if not self.enable_fp64 and instr.opcode.value in self._FP64_OPCODES:
            raise ValueError(
                f"opcode '{instr.opcode.value}' requires enable_fp64=True "
                f"(ExtendedInstructionSelector)")
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

        Float32 literal arguments are materialized by their exact IEEE-754
        bit pattern (never ``int()``-truncated); float64 literals raise
        ``ValueError`` (no 64-bit materialization in RV32IM).
        """
        if self._involves_fp64(instr):
            self._require_fp64(instr)
        self._check_dtype(
            instr, (DataType.FLOAT32, DataType.FLOAT64), "sqrt")
        dst = self._dst(instr)
        dtype = instr.dest.dtype

        if self.use_hardware_sqrt:
            src = self._materialized_op(instr, 0, prefix="sqrt_imm")
            if dtype == DataType.FLOAT64:
                self._emit(MachineOp.SQRT_D, dst, src,
                           comment="fsqrt.d (hardware)")
            else:
                self._emit(MachineOp.SQRT_S, dst, src,
                           comment="fsqrt.s (hardware)")
            return

        # Library call: argument in a0, result in a0
        if self._literal_operand(instr.operands[0]):
            bits = self._constant_bits(instr, instr.operands[0])
            self._emit(MachineOp.LI, MachineOperand.reg("a0"),
                       MachineOperand.immediate(bits),
                       comment="sqrt arg -> a0")
        else:
            src = self._op(instr, 0)
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

        Integer min (branchless, 0/-1 mask):
            slt tmp, a, b        # tmp = (a < b)
            sub mask, x0, tmp    # mask = 0 or -1
            sub diff, a, b       # diff = a - b
            and diff, diff, mask # diff = (a < b) ? a - b : 0
            add dst, b, diff     # dst = (a < b) ? a : b

        Literal operands are materialized first: ``sub``/``and``/``add``
        have no immediate form, and f64 literals have no materialization
        at all (fail-loud).
        """
        if self._involves_fp64(instr):
            self._require_fp64(instr)
        self._check_dtype(
            instr,
            (DataType.INT32, DataType.INT64, DataType.FLOAT64),
            "min")
        a = self._materialized_op(instr, 0, prefix="min_const")
        b = self._materialized_op(instr, 1, prefix="min_const")
        dst = self._dst(instr)

        if instr.dest.dtype == DataType.FLOAT64:
            # Use FMIN.D pseudo (expands to branchless sequence)
            self._emit(MachineOp.FMIN_D, dst, a, b, comment="fmin.d")
        else:
            tmp = self._fresh_temp("min_slt")
            self._emit(MachineOp.SLT, tmp, a, b, comment="min: slt")
            mask = self._fresh_temp("min_mask")
            self._emit(MachineOp.SUB, mask, MachineOperand.reg("x0"), tmp,
                       comment="min: mask")
            diff = self._fresh_temp("min_sub")
            self._emit(MachineOp.SUB, diff, a, b, comment="min: sub")
            self._emit(MachineOp.AND, diff, diff, mask, comment="min: and")
            self._emit(
                MachineOp.ADD, dst, b, diff, comment="min result"
            )

    def _select_max(self, instr: Instruction) -> None:
        """Select instruction for max(a, b).

        Integer max (branchless, 0/-1 mask, symmetric to min):
            slt tmp, a, b        # tmp = (a < b)
            sub mask, x0, tmp    # mask = 0 or -1
            sub diff, b, a       # diff = b - a
            and diff, diff, mask # diff = (a < b) ? b - a : 0
            add dst, a, diff     # dst = (a < b) ? b : a

        The machine-level ``max`` pseudo is not used: its encoder fallback
        branch is broken (falls back to x0 instead of rs2), so the pseudo
        computes max incorrectly for a < b.
        """
        if self._involves_fp64(instr):
            self._require_fp64(instr)
        self._check_dtype(
            instr,
            (DataType.INT32, DataType.INT64, DataType.FLOAT64),
            "max")
        a = self._materialized_op(instr, 0, prefix="max_const")
        b = self._materialized_op(instr, 1, prefix="max_const")
        dst = self._dst(instr)

        if instr.dest.dtype == DataType.FLOAT64:
            self._emit(MachineOp.FMAX_D, dst, a, b, comment="fmax.d")
        else:
            tmp = self._fresh_temp("max_slt")
            self._emit(MachineOp.SLT, tmp, a, b, comment="max: slt")
            mask = self._fresh_temp("max_mask")
            self._emit(MachineOp.SUB, mask, MachineOperand.reg("x0"), tmp,
                       comment="max: mask")
            diff = self._fresh_temp("max_sub")
            self._emit(MachineOp.SUB, diff, b, a, comment="max: sub")
            self._emit(MachineOp.AND, diff, diff, mask, comment="max: and")
            self._emit(
                MachineOp.ADD, dst, a, diff, comment="max result"
            )

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
        if self._involves_fp64(instr):
            self._require_fp64(instr)
        self._check_dtype(
            instr,
            (DataType.INT32, DataType.INT64, DataType.FLOAT64),
            "abs")
        src = self._materialized_op(instr, 0, prefix="abs_const")
        dst = self._dst(instr)

        if instr.dest.dtype == DataType.FLOAT64:
            # fabs.d: clear the sign bit
            self._emit(MachineOp.FABS_D, dst, src, comment="fabs.d")
        else:
            tmp1 = self._fresh_temp("abs_srai")
            imm31 = MachineOperand.immediate(31)
            self._emit(
                MachineOp.SRAI, tmp1, src, imm31, comment="abs: srai 31"
            )
            tmp2 = self._fresh_temp("abs_xor")
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
        self._check_dtype(instr, (DataType.INT32,), "idiv")
        a = self._materialized_op(instr, 0, prefix="idiv_const")
        b = self._materialized_op(instr, 1, prefix="idiv_const")
        dst = self._dst(instr)
        self._emit(MachineOp.DIV, dst, a, b, comment="div")

    def _select_rem(self, instr: Instruction) -> None:
        """Select instruction for integer remainder."""
        self._check_dtype(instr, (DataType.INT32,), "rem")
        a = self._materialized_op(instr, 0, prefix="rem_const")
        b = self._materialized_op(instr, 1, prefix="rem_const")
        dst = self._dst(instr)
        self._emit(MachineOp.REM, dst, a, b, comment="rem")

    def _select_mod(self, instr: Instruction) -> None:
        """Select instruction for modulo (synonym of rem for non-negative)."""
        self._check_dtype(instr, (DataType.INT32,), "mod")
        self._select_rem(instr)

    # ------------------------------------------------------------------
    # float64 (D extension)
    # ------------------------------------------------------------------

    def _select_load_f64(self, instr: Instruction) -> None:
        """Load a 64-bit float from memory."""
        self._check_signature(
            instr, (DataType.FLOAT64,),
            (DataType.INT32, DataType.INT64), "load_f64")
        src = self._op(instr, 0)
        dst = self._dst(instr)
        self._emit(MachineOp.FLD, dst, src, comment="fld (load f64)")

    def _select_store_f64(self, instr: Instruction) -> None:
        """Store a 64-bit float to memory (operands: [addr, value])."""
        if len(instr.operands) < 2:
            raise ValueError(
                "store_f64 requires operands [addr, value], got "
                f"{len(instr.operands)} operand(s)")
        addr = instr.operands[0]
        if addr.dtype not in (DataType.INT32, DataType.INT64):
            raise ValueError(
                "store_f64 requires an INT32/INT64 address operand, got "
                f"{addr.dtype.value}")
        val = instr.operands[1]
        if val.dtype != DataType.FLOAT64:
            raise ValueError(
                "store_f64 requires a FLOAT64 value operand, got "
                f"{val.dtype.value}")
        addr_op = self._op(instr, 0)
        val_op = self._materialized_op(instr, 1, prefix="store_f64_val")
        self._emit(MachineOp.FSD, val_op, addr_op, comment="fsd (store f64)")

    def _select_fadd_d(self, instr: Instruction) -> None:
        """Add two float64 values."""
        self._check_dtype(instr, (DataType.FLOAT64,), "fadd_d")
        a = self._materialized_op(instr, 0, prefix="fadd_d_const")
        b = self._materialized_op(instr, 1, prefix="fadd_d_const")
        dst = self._dst(instr)
        self._emit(MachineOp.FADD_D, dst, a, b, comment="fadd.d")

    def _select_fsub_d(self, instr: Instruction) -> None:
        """Subtract two float64 values."""
        self._check_dtype(instr, (DataType.FLOAT64,), "fsub_d")
        a = self._materialized_op(instr, 0, prefix="fsub_d_const")
        b = self._materialized_op(instr, 1, prefix="fsub_d_const")
        dst = self._dst(instr)
        self._emit(MachineOp.FSUB_D, dst, a, b, comment="fsub.d")

    def _select_fmul_d(self, instr: Instruction) -> None:
        """Multiply two float64 values."""
        self._check_dtype(instr, (DataType.FLOAT64,), "fmul_d")
        a = self._materialized_op(instr, 0, prefix="fmul_d_const")
        b = self._materialized_op(instr, 1, prefix="fmul_d_const")
        dst = self._dst(instr)
        self._emit(MachineOp.FMUL_D, dst, a, b, comment="fmul.d")

    def _select_fdiv_d(self, instr: Instruction) -> None:
        """Divide two float64 values."""
        self._check_dtype(instr, (DataType.FLOAT64,), "fdiv_d")
        a = self._materialized_op(instr, 0, prefix="fdiv_d_const")
        b = self._materialized_op(instr, 1, prefix="fdiv_d_const")
        dst = self._dst(instr)
        self._emit(MachineOp.FDIV_D, dst, a, b, comment="fdiv.d")

    def _select_fcmp_l_d(self, instr: Instruction) -> None:
        """Float64 less-than comparison."""
        self._check_signature(
            instr, (DataType.INT32,), (DataType.FLOAT64,), "fcmp_l_d")
        a = self._materialized_op(instr, 0, prefix="fcmp_l_d_const")
        b = self._materialized_op(instr, 1, prefix="fcmp_l_d_const")
        dst = self._dst(instr)
        self._emit(MachineOp.FLT_D, dst, a, b, comment="flt.d")

    def _select_fcmp_eq_d(self, instr: Instruction) -> None:
        """Float64 equality comparison."""
        self._check_signature(
            instr, (DataType.INT32,), (DataType.FLOAT64,), "fcmp_eq_d")
        a = self._materialized_op(instr, 0, prefix="fcmp_eq_d_const")
        b = self._materialized_op(instr, 1, prefix="fcmp_eq_d_const")
        dst = self._dst(instr)
        self._emit(MachineOp.FEQ_D, dst, a, b, comment="feq.d")

    def _select_fcvt_s_d(self, instr: Instruction) -> None:
        """Convert float64 to float32."""
        self._check_signature(
            instr, (DataType.FLOAT32,), (DataType.FLOAT64,), "fcvt_s_d")
        src = self._materialized_op(instr, 0, prefix="fcvt_s_d_const")
        dst = self._dst(instr)
        self._emit(MachineOp.FCVT_S_D, dst, src, comment="fcvt.s.d")

    def _select_fcvt_d_s(self, instr: Instruction) -> None:
        """Convert float32 to float64."""
        self._check_signature(
            instr, (DataType.FLOAT64,), (DataType.FLOAT32,), "fcvt_d_s")
        src = self._materialized_op(instr, 0, prefix="fcvt_d_s_const")
        dst = self._dst(instr)
        self._emit(MachineOp.FCVT_D_S, dst, src, comment="fcvt.d.s")

    def _select_load_const_f64(self, instr: Instruction) -> None:
        """Load a float64 constant (exact IEEE-754 bit pattern)."""
        self._check_signature(
            instr, (DataType.FLOAT64,), (), "load_const_f64")
        raw_val = instr.attrs.get("value")
        if not isinstance(raw_val, (int, float)):
            raise ValueError(
                "load_const_f64 requires numeric attrs['value'], got "
                f"{raw_val!r}")
        bits = struct.unpack("<Q", struct.pack("<d", float(raw_val)))[0]
        dst = self._dst(instr)
        self._emit(MachineOp.LI_D, dst,
                   MachineOperand.immediate(bits),
                   comment=f"f64 {raw_val!r} bits=0x{bits:016x}")

    # ------------------------------------------------------------------
    # Type-aware overrides for existing ops
    # ------------------------------------------------------------------

    def _select_add(self, instr: Instruction) -> None:
        if self._involves_fp64(instr):
            self._require_fp64(instr)
            self._select_fadd_d(instr)
        else:
            super()._select_add(instr)

    def _select_sub(self, instr: Instruction) -> None:
        if self._involves_fp64(instr):
            self._require_fp64(instr)
            self._select_fsub_d(instr)
        else:
            super()._select_sub(instr)

    def _select_mul(self, instr: Instruction) -> None:
        if self._involves_fp64(instr):
            self._require_fp64(instr)
            self._select_fmul_d(instr)
        else:
            super()._select_mul(instr)

    def _select_div(self, instr: Instruction) -> None:
        if self._involves_fp64(instr):
            self._require_fp64(instr)
            self._select_fdiv_d(instr)
        elif instr.dest is not None and instr.dest.dtype == DataType.INT32:
            self._select_idiv(instr)
        else:
            super()._select_div(instr)

    def _select_load(self, instr: Instruction) -> None:
        if self._involves_fp64(instr):
            self._require_fp64(instr)
            self._select_load_f64(instr)
        else:
            super()._select_load(instr)

    def _select_store(self, instr: Instruction) -> None:
        if self._involves_fp64(instr):
            self._require_fp64(instr)
            self._select_store_f64(instr)
        else:
            super()._select_store(instr)

    def _select_load_const(self, instr: Instruction) -> None:
        if self._involves_fp64(instr):
            self._require_fp64(instr)
            self._select_load_const_f64(instr)
        else:
            super()._select_load_const(instr)

    def _select_neg(self, instr: Instruction) -> None:
        """Negate: for float64 use fneg.d, for int use sub x0 - x."""
        if self._involves_fp64(instr):
            self._require_fp64(instr)
            self._check_dtype(instr, (DataType.FLOAT64,), "neg")
            src = self._materialized_op(instr, 0, prefix="neg_const")
            dst = self._dst(instr)
            self._emit(MachineOp.FNEG_D, dst, src, comment="fneg.d")
        else:
            super()._select_neg(instr)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fresh_temp(self, prefix: str) -> MachineOperand:
        """Return a fresh deterministic virtual register temp."""
        self._temp_counter += 1
        return MachineOperand.vreg(f"__{prefix}_{self._temp_counter}")

    def _involves_fp64(self, instr: Instruction) -> bool:
        """Detect float64 involvement (pure check, ignores enable_fp64)."""
        if instr.dest is not None and instr.dest.dtype == DataType.FLOAT64:
            return True
        for op in instr.operands:
            if op.dtype == DataType.FLOAT64:
                return True
        return False

    def _require_fp64(self, instr: Instruction) -> None:
        """Raise if float64 support is disabled."""
        if not self.enable_fp64:
            raise ValueError(
                f"instruction '{instr.opcode.value}' involves FLOAT64 but "
                f"enable_fp64=False (ExtendedInstructionSelector)")

    def _materialized_op(self, instr: Instruction, idx: int,
                         prefix: str = "const") -> MachineOperand:
        """Return operand *idx* as a register, materializing literals.

        The base ``_op`` routes every constant through ``int()``; for float
        literals that silently changes the value.  Literal operands are
        instead materialized with an exact ``LI`` (f32 bit pattern) into a
        fresh temp; f64 literals fail loud (RV32IM has no 64-bit constant
        materialization).
        """
        op = instr.operands[idx]
        if not self._literal_operand(op):
            return MachineOperand.vreg(op.name)
        bits = self._constant_bits(instr, op)
        tmp = self._fresh_temp(prefix)
        self._emit(MachineOp.LI, tmp, MachineOperand.immediate(bits),
                   comment=f"const {op.const_value!r}")
        return tmp

    def _literal_operand(self, op: Value) -> bool:
        """True if *op* is a literal constant (no defining instruction)."""
        if not (op.is_constant and op.const_value is not None):
            return False
        return op.name not in self._defined_names

    def _constant_bits(self, instr: Instruction, op: Value) -> int:
        """Exact 32-bit materialization of a literal constant operand."""
        if op.dtype == DataType.FLOAT64:
            raise ValueError(
                f"{instr.opcode.value} cannot materialize a FLOAT64 literal "
                f"({op.const_value!r}): RV32IM has no 64-bit constant "
                f"materialization; use load_const_f64")
        if op.dtype == DataType.FLOAT32:
            return self._fp32_bits(op.const_value)
        return int(op.const_value)

    @staticmethod
    def _fp32_bits(value) -> int:
        """IEEE-754 f32 bit pattern as a signed RV32 immediate."""
        return struct.unpack("<i", struct.pack("<f", float(value)))[0]

    def _check_signature(self, instr: Instruction,
                         dest_allowed: tuple[DataType, ...],
                         operand_allowed: tuple[DataType, ...],
                         opname: str) -> None:
        """Validate dest existence and exact dtype sets for an opcode."""
        if instr.dest is None:
            raise ValueError(f"{opname} requires a destination value")
        if instr.dest.dtype not in dest_allowed:
            allowed_vals = ", ".join(d.value for d in dest_allowed)
            raise ValueError(
                f"{opname} requires destination dtype in "
                f"({allowed_vals}), got {instr.dest.dtype.value}")
        for op in instr.operands:
            if op.dtype not in operand_allowed:
                allowed_vals = ", ".join(d.value for d in operand_allowed)
                raise ValueError(
                    f"{opname} requires operand dtype in "
                    f"({allowed_vals}), got {op.dtype.value}")

    def _check_dtype(self, instr: Instruction,
                     allowed: tuple[DataType, ...],
                     opname: str) -> None:
        """Guard dest/operand dtypes; all operands must match the dest type."""
        self._check_signature(instr, allowed, allowed, opname)
        for op in instr.operands:
            if op.dtype != instr.dest.dtype:
                raise ValueError(
                    f"{opname} requires operands to match destination dtype "
                    f"{instr.dest.dtype.value}, got {op.dtype.value}")

    def _is_fp64(self, instr: Instruction) -> bool:
        """Legacy compatibility: float64 involvement with fp64 enabled."""
        return self.enable_fp64 and self._involves_fp64(instr)

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
