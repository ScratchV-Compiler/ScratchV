"""Extended Instruction Selector for RISC-V Backend.

Extends the base instruction selector with additional operations:
- sqrt (hardware F/D instructions)
- min/max (using RISC-V branchless or branch sequences)
- abs (absolute value)
- float64 (double-precision floating point) support
- typed div and rem operations

Usage::

    from scratchv.backend.inst_select_ext import ExtendedInstructionSelector
    selector = ExtendedInstructionSelector(program)
    machine_instrs = selector.run()
"""

from __future__ import annotations

import struct

from scratchv.backend.instruction_select import InstructionSelector
from scratchv.backend.machine_types import (
    FloatConstant,
    MachineOp,
    MachineOperand,
    RegisterClass,
)
from scratchv.backend.target_features import TargetFeatures
from scratchv.backend.machine_semantics import validate_register_classes
from scratchv.ir.types import DataType, Instruction, OpCode, Program, valid_float_bit_pattern


def validate_extended_types(program: Program) -> None:
    """Reject unsupported element types/shapes before optimization or lowering.

    Shared IR can express i64 for other backends. The extended RV32 path
    supports only i32/f32/f64, including unused values that DCE might hide.
    """
    values = list(program.global_values)
    parameter_ids = set()
    for func in program.functions:
        parameter_ids.update(id(value) for value in func.params)
        values.extend([*func.params, *func.returns, *func.locals])
        for block in func.blocks:
            for instr in (*block.phi_nodes, *block.instructions):
                if instr.opcode is OpCode.LOAD_CONST:
                    bits = instr.attrs.get("bit_pattern")
                    if bits is not None:
                        if instr.dest is None or not valid_float_bit_pattern(instr.dest.dtype, bits):
                            raise ValueError("invalid LOAD_CONST bit_pattern for result dtype")
                        if instr.dest.bit_pattern is not None and instr.dest.bit_pattern != bits:
                            raise ValueError("constant bit_pattern differs from result metadata")
                values.extend(instr.operands)
                if instr.dest is not None:
                    values.append(instr.dest)
    for value in values:
        if value.dtype not in (DataType.INT32, DataType.FLOAT32, DataType.FLOAT64):
            dtype = getattr(value.dtype, "value", repr(value.dtype))
            raise ValueError(
                f"extended RISC-V backend does not support {dtype} value {value.name!r}; "
                "supported element types are i32, f32, f64"
            )
    for value in values:
        if value.bit_pattern is not None and (
                not value.is_constant or not valid_float_bit_pattern(value.dtype, value.bit_pattern)):
            raise ValueError(f"invalid constant bit_pattern for {value.dtype.value} value {value.name!r}")
        if any(dimension != 1 for dimension in value.shape):
            boundary = ("extended RISC-V ABI supports one scalar parameter only"
                        if id(value) in parameter_ids
                        else "extended RISC-V supports scalar shapes only")
            raise ValueError(f"{boundary}: {value.name!r} has shape {value.shape}")


class ExtendedInstructionSelector(InstructionSelector):
    """Extended instruction selector with additional RISC-V op support.

    Extends the base ``InstructionSelector`` to add:
    - sqrt (hardware F/D instructions)
    - min/max (branchless sequences)
    - abs (absolute value)
    - float64 (double-precision) operations
    - typed div and rem (integer)

    Parameters
    ----------
    program:
        The ScratchV IR Program to select instructions for.
    features:
        ISA/ABI requirements. Direct use defaults to RV32IMFD/ILP32D;
        CompilerDriver always passes its explicit target.
    """

    _TYPED_FP_OPS = frozenset({
        OpCode.ADD,
        OpCode.SUB,
        OpCode.MUL,
        OpCode.DIV,
        OpCode.REM,
        OpCode.NEG,
        OpCode.MIN,
        OpCode.MAX,
        OpCode.ABS,
        OpCode.SQRT,
        OpCode.LOAD,
        OpCode.STORE,
        OpCode.LOAD_CONST,
        OpCode.BR_IF,
        OpCode.RETURN,
    })

    def __init__(
        self,
        program: Program,
        *,
        features: TargetFeatures | None = None,
    ):
        super().__init__(program)
        if features is None:
            features = TargetFeatures("rv32imfd", "ilp32d")
        self.features = features
        self._constant_pool: list[FloatConstant] = []
        self._constant_labels: dict[tuple[DataType, int], str] = {}
        self._fp_definitions: set[str] = set()

    @property
    def constant_pool(self) -> tuple[FloatConstant, ...]:
        return tuple(self._constant_pool)

    def run(self):
        validate_extended_types(self.program)
        self._validate_fp_definition_order()
        self._constant_pool = []
        self._constant_labels = {}
        self.local_slots = {}
        self.local_bytes = 0
        instructions = super().run()
        validate_register_classes(instructions)
        return instructions

    def _validate_fp_definition_order(self) -> None:
        """Keep CFG dominance checks when accepting definitions later in layout.

        Reuse the IR verifier's CFG normalization. This boundary checks only
        FP data definitions; it does not require legacy GPR live-ins or memory
        addresses to acquire new frontend declarations.
        """
        from scratchv.analysis.ir_verifier import ErrorLevel, verify_ir

        data_names = set()
        for func in self.program.functions:
            for value in func.params:
                if value.dtype in (DataType.FLOAT32, DataType.FLOAT64):
                    data_names.add(value.name)
            for block in func.blocks:
                for instr in block.instructions:
                    if instr.dest is not None and instr.opcode is not OpCode.ALLOCA:
                        if instr.dest.dtype in (DataType.FLOAT32, DataType.FLOAT64):
                            data_names.add(instr.dest.name)
                    operands = (instr.operands[1:] if instr.opcode in (OpCode.LOAD, OpCode.STORE)
                                else instr.operands)
                    data_names.update(value.name for value in operands
                                      if value.dtype in (DataType.FLOAT32, DataType.FLOAT64))
        if data_names:
            _, issues = verify_ir(self.program)
            for issue in issues:
                if (issue.level is ErrorLevel.ERROR and issue.value_name in data_names
                        and issue.rule in {"def-before-use", "ssa-validity"}
                        and issue.message != "value has no definition"):
                    raise ValueError(f"invalid floating machine definition: {issue}")

    # ------------------------------------------------------------------
    # Base overrides
    # ------------------------------------------------------------------

    def _select_instruction(self, instr: Instruction) -> None:
        """Require typed rules and track floating machine definitions."""
        if (
            instr.opcode is not OpCode.ALLOCA
            and self._is_float(instr)
            and instr.opcode not in self._TYPED_FP_OPS
        ):
            raise ValueError(
                f"{instr.opcode.value} has no typed floating-point "
                "selection rule"
            )
        super()._select_instruction(instr)
        if (instr.dest is not None and instr.opcode is not OpCode.ALLOCA
                and instr.dest.dtype in (DataType.FLOAT32, DataType.FLOAT64)):
            self._fp_definitions.add(instr.dest.name)

    def _select_function(self, func) -> None:
        # A definition can dominate a use while appearing later in block layout.
        # Include only sources with actual lowering rules, never ALLOCA/phi.
        self._fp_definitions = {
            value.name for value in [*func.params, *self.program.global_values]
            if value.dtype in (DataType.FLOAT32, DataType.FLOAT64)
            and (value in func.params or value.is_constant)
        }
        self._fp_definitions.update(
            instr.dest.name for block in func.blocks for instr in block.instructions
            if instr.dest is not None and instr.opcode in self._TYPED_FP_OPS
            and instr.dest.dtype in (DataType.FLOAT32, DataType.FLOAT64)
        )
        float_params = [value for value in func.params if value.dtype in (DataType.FLOAT32, DataType.FLOAT64)]
        if float_params and (len(func.params) != 1 or any(dimension != 1 for dimension in func.params[0].shape)):
            raise ValueError("extended hard-float ABI currently supports one scalar parameter only")
        if func.params and not float_params:
            if (len(func.params) != 1
                    or any(dimension != 1 for dimension in func.params[0].shape)):
                raise ValueError("extended integer ABI currently supports one scalar parameter only")
        self._emit_label(func.name, is_function=True)
        self._stack_offset = 0
        if float_params:
            value = float_params[0]
            width = 32 if value.dtype is DataType.FLOAT32 else 64
            self._require_fp_extension(OpCode.RETURN, value.dtype)
            self._emit(
                MachineOp.FMV_S if width == 32 else MachineOp.FMV_D,
                MachineOperand.vreg(value.name, RegisterClass.FPR, width),
                MachineOperand.reg("fa0", RegisterClass.FPR, width),
                comment="scalar hard-float ABI parameter",
            )
            self._fp_definitions.add(value.name)
        elif func.params:
            self._emit(
                MachineOp.MV, MachineOperand.vreg(func.params[0].name),
                MachineOperand.reg("a0"), comment="scalar integer ABI parameter",
            )
        # ONNX scalar initializers are IR global definitions. Materialize their
        # actual names before uses; the parser's separate LOAD_CONST temp does
        # not define these globals in the machine register stream.
        for value in self.program.global_values:
            if value.is_constant and value.dtype in (DataType.FLOAT32, DataType.FLOAT64):
                attrs = {"value": value.const_value}
                if value.bit_pattern is not None:
                    attrs["bit_pattern"] = value.bit_pattern
                self._select_load_const(Instruction(
                    OpCode.LOAD_CONST, value, attrs=attrs,
                ))
        for block in func.blocks:
            self._emit_label(self._block_label(block.name))
            for instr in block.instructions:
                self._select_instruction(instr)

    def _namespace_function_registers(self, start: int, function_index: int) -> dict[str, str]:
        names = super()._namespace_function_registers(start, function_index)
        # Rename this function's objects before the next function reuses IR names.
        self.local_slots = {
            names.get(name, name): slot for name, slot in self.local_slots.items()
        }
        return names

    def _select_alloca(self, instr: Instruction) -> None:
        from scratchv.backend.allocation import SpillSlot

        width = 8 if instr.dest.dtype is DataType.FLOAT64 else 4
        size = int(instr.attrs.get("size", 1)) * width
        if size <= 0:
            raise ValueError("local stack allocation must be positive")
        self.local_bytes = (self.local_bytes + size + width - 1) // width * width
        offset = -self.local_bytes
        self.local_slots[instr.dest.name] = SpillSlot(offset, size, width, RegisterClass.GPR)
        self._emit(
            MachineOp.ADDI, MachineOperand.vreg(instr.dest.name),
            MachineOperand.reg("sp"), MachineOperand.immediate(offset),
            comment="local stack address [regalloc:local]",
        )

    # ------------------------------------------------------------------
    # sqrt
    # ------------------------------------------------------------------

    def _select_sqrt(self, instr: Instruction) -> None:
        """Select the hardware F/D square-root path.

        Targets without F/D are rejected; libm fallback is not delivered.
        """
        dtype = self._float_dtype(instr)
        self._require_fp_extension(OpCode.SQRT, dtype)
        op = MachineOp.SQRT_D if dtype is DataType.FLOAT64 else MachineOp.SQRT_S
        self._emit(
            op,
            self._fp_dst(instr),
            self._fp_op(instr, 0),
            comment=op.value,
            rounding_mode="rne",
        )

    # ------------------------------------------------------------------
    # min / max
    # ------------------------------------------------------------------

    def _select_fp_min_max(self, instr: Instruction) -> None:
        """Use minimumNumber/maximumNumber, optionally propagating ONNX NaNs.

        Quiet self-comparisons identify NaNs without setting NV for qNaN.
        Only the NaN path adds the operands to produce a canonical quiet NaN;
        adding finite values or opposite infinities would change fflags.
        """
        dtype = self._float_dtype(instr)
        self._require_fp_extension(instr.opcode, dtype)
        double = dtype is DataType.FLOAT64
        if instr.opcode is OpCode.MIN:
            op = MachineOp.FMIN_D if double else MachineOp.FMIN_S
        else:
            op = MachineOp.FMAX_D if double else MachineOp.FMAX_S
        dst, lhs, rhs = self._fp_dst(instr), self._fp_op(instr, 0), self._fp_op(instr, 1)
        self._emit(op, dst, lhs, rhs, comment=op.value)
        if not instr.attrs.get("propagate_nan", False):
            return
        compare = MachineOp.FEQ_D if double else MachineOp.FEQ_S
        lhs_number = self._fresh_vreg("minmax_lhs_number")
        rhs_number = self._fresh_vreg("minmax_rhs_number")
        both_numbers = self._fresh_vreg("minmax_both_numbers")
        done = self._fresh_label("minmax_done")
        self._emit(compare, lhs_number, lhs, lhs, comment="ONNX: lhs is not NaN")
        self._emit(compare, rhs_number, rhs, rhs, comment="ONNX: rhs is not NaN")
        self._emit(MachineOp.AND, both_numbers, lhs_number, rhs_number)
        self._emit(MachineOp.BNEZ, both_numbers, target=done)
        self._emit(
            MachineOp.FADD_D if double else MachineOp.FADD_S,
            dst, lhs, rhs, rounding_mode="rne", comment="ONNX: propagate NaN",
        )
        self._emit_label(done)

    def _select_min(self, instr: Instruction) -> None:
        """Select instruction for min(a, b).

        Integer min (branchless):
            slt cond, a, b       # cond = (a < b), either 0 or 1
            sub mask, x0, cond   # mask = 0 or 0xffffffff
            xor diff, a, b
            and selected, diff, mask
            xor dst, b, selected # a if a < b, otherwise b
        """
        dtype = instr.dest.dtype if instr.dest is not None else None
        if dtype in (DataType.FLOAT32, DataType.FLOAT64):
            self._select_fp_min_max(instr)
        elif dtype == DataType.INT32:
            a = self._op(instr, 0)
            b = self._op(instr, 1)
            dst = self._dst(instr)
            a = self._materialize_int_operand(a, prefix="min_lhs")
            b = self._materialize_int_operand(b, prefix="min_rhs")
            cond = self._fresh_vreg("min_cond")
            mask = self._fresh_vreg("min_mask")
            diff = self._fresh_vreg("min_diff")
            selected = self._fresh_vreg("min_selected")
            self._emit(MachineOp.SLT, cond, a, b, comment="min: a < b")
            self._emit(
                MachineOp.SUB,
                mask,
                MachineOperand.reg("x0"),
                cond,
                comment="min: full-width mask",
            )
            self._emit(MachineOp.XOR, diff, a, b, comment="min: differing bits")
            self._emit(
                MachineOp.AND,
                selected,
                diff,
                mask,
                comment="min: select differing bits",
            )
            if dst:
                self._emit(
                    MachineOp.XOR, dst, b, selected, comment="min result"
                )
        else:
            raise ValueError(f"min does not support {dtype.value if dtype else 'missing result'}")

    def _select_max(self, instr: Instruction) -> None:
        """Select instruction for max(a, b).

        Integer max uses the same full-width-mask selection scheme as min.
        """
        dtype = instr.dest.dtype if instr.dest is not None else None
        if dtype in (DataType.FLOAT32, DataType.FLOAT64):
            self._select_fp_min_max(instr)
        elif dtype == DataType.INT32:
            a = self._op(instr, 0)
            b = self._op(instr, 1)
            dst = self._dst(instr)
            a = self._materialize_int_operand(a, prefix="max_lhs")
            b = self._materialize_int_operand(b, prefix="max_rhs")
            cond = self._fresh_vreg("max_cond")
            mask = self._fresh_vreg("max_mask")
            diff = self._fresh_vreg("max_diff")
            selected = self._fresh_vreg("max_selected")
            self._emit(MachineOp.SLT, cond, a, b, comment="max: a < b")
            self._emit(
                MachineOp.SUB,
                mask,
                MachineOperand.reg("x0"),
                cond,
                comment="max: full-width mask",
            )
            self._emit(MachineOp.XOR, diff, a, b, comment="max: differing bits")
            self._emit(
                MachineOp.AND,
                selected,
                diff,
                mask,
                comment="max: select differing bits",
            )
            if dst:
                self._emit(
                    MachineOp.XOR, dst, a, selected, comment="max result"
                )
        else:
            raise ValueError(f"max does not support {dtype.value if dtype else 'missing result'}")

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
        dtype = instr.dest.dtype if instr.dest is not None else None
        if dtype == DataType.FLOAT64:
            self._require_fp_extension(OpCode.ABS, dtype)
            self._emit(
                MachineOp.FABS_D, self._fp_dst(instr),
                self._fp_op(instr, 0), comment="fabs.d",
            )
        elif dtype == DataType.FLOAT32:
            self._require_fp_extension(OpCode.ABS, dtype)
            self._emit(
                MachineOp.FABS_S, self._fp_dst(instr),
                self._fp_op(instr, 0), comment="fabs.s",
            )
        elif dtype == DataType.INT32:
            src = self._op(instr, 0)
            dst = self._dst(instr)
            src = self._materialize_int_operand(src, prefix="abs_src")
            tmp1 = self._fresh_vreg("abs_sign")
            imm31 = MachineOperand.immediate(31)
            self._emit(
                MachineOp.SRAI, tmp1, src, imm31, comment="abs: srai 31"
            )
            tmp2 = self._fresh_vreg("abs_xor")
            self._emit(
                MachineOp.XOR, tmp2, src, tmp1, comment="abs: xor"
            )
            if dst:
                self._emit(
                    MachineOp.SUB, dst, tmp2, tmp1, comment="abs: sub"
                )
        else:
            raise ValueError(f"abs does not support {dtype.value if dtype else 'missing result'}")

    # ------------------------------------------------------------------
    # div / rem (integer)
    # ------------------------------------------------------------------

    def _select_rem(self, instr: Instruction) -> None:
        """Select instruction for integer remainder."""
        dtype = instr.dest.dtype if instr.dest is not None else None
        if dtype != DataType.INT32:
            raise ValueError(
                f"rem does not support {dtype.value if dtype else 'missing result'}"
            )
        a = self._materialize_int_operand(
            self._op(instr, 0), prefix="rem_lhs"
        )
        b = self._materialize_int_operand(
            self._op(instr, 1), prefix="rem_rhs"
        )
        dst = self._dst(instr)
        self._emit(MachineOp.REM, dst, a, b, comment="rem")

    # ------------------------------------------------------------------
    # Type-aware floating-point selection
    # ------------------------------------------------------------------

    _FP_BINARY_OPS = {
        OpCode.ADD: (MachineOp.FADD_S, MachineOp.FADD_D),
        OpCode.SUB: (MachineOp.FSUB_S, MachineOp.FSUB_D),
        OpCode.MUL: (MachineOp.FMUL_S, MachineOp.FMUL_D),
        OpCode.DIV: (MachineOp.FDIV_S, MachineOp.FDIV_D),
    }

    def _select_fp_binary(self, instr: Instruction) -> None:
        dtype = self._float_dtype(instr)
        self._require_fp_extension(instr.opcode, dtype)
        single, double = self._FP_BINARY_OPS[instr.opcode]
        op = double if dtype is DataType.FLOAT64 else single
        self._emit(
            op,
            self._fp_dst(instr),
            self._fp_op(instr, 0),
            self._fp_op(instr, 1),
            comment=op.value,
            rounding_mode="rne",
        )

    def _select_add(self, instr: Instruction) -> None:
        if self._is_float(instr):
            self._select_fp_binary(instr)
        else:
            super()._select_add(instr)

    def _select_sub(self, instr: Instruction) -> None:
        if self._is_float(instr):
            self._select_fp_binary(instr)
        else:
            super()._select_sub(instr)

    def _select_mul(self, instr: Instruction) -> None:
        if self._is_float(instr):
            self._select_fp_binary(instr)
        else:
            super()._select_mul(instr)

    def _select_div(self, instr: Instruction) -> None:
        if self._is_float(instr):
            self._select_fp_binary(instr)
        else:
            super()._select_div(instr)

    def _select_load(self, instr: Instruction) -> None:
        if (
            instr.dest is None
            or instr.dest.dtype not in (DataType.FLOAT32, DataType.FLOAT64)
        ):
            super()._select_load(instr)
            return
        dtype = instr.dest.dtype
        self._require_fp_extension(OpCode.LOAD, dtype)
        op = MachineOp.FLD if dtype is DataType.FLOAT64 else MachineOp.FLW
        self._emit(
            op,
            self._fp_dst(instr),
            self._materialize_int_operand(self._op(instr, 0), prefix="fp_load_addr"),
            comment=f"{op.value} (load {dtype.value})",
        )

    def _select_store(self, instr: Instruction) -> None:
        value = instr.operands[1]
        if value.dtype not in (DataType.FLOAT32, DataType.FLOAT64):
            super()._select_store(instr)
            return
        dtype = value.dtype
        self._require_fp_extension(OpCode.STORE, dtype)
        op = MachineOp.FSD if dtype is DataType.FLOAT64 else MachineOp.FSW
        # IR order is [address, value]; machine order is [value, address].
        self._emit(
            op,
            self._fp_op(instr, 1),
            self._materialize_int_operand(self._op(instr, 0), prefix="fp_store_addr"),
            comment=f"{op.value} (store {dtype.value})",
        )

    def _select_load_const(self, instr: Instruction) -> None:
        if (
            instr.dest is None
            or instr.dest.dtype not in (DataType.FLOAT32, DataType.FLOAT64)
        ):
            super()._select_load_const(instr)
            return
        dtype = instr.dest.dtype
        self._require_fp_extension(OpCode.LOAD_CONST, dtype)
        label = self._intern_float_constant(instr, dtype)
        address = self._fresh_vreg("fp_const_addr")
        self._emit(
            MachineOp.LA,
            address,
            MachineOperand.symbol(label),
            comment=f"address of {label}",
        )
        load_op = MachineOp.FLD if dtype is DataType.FLOAT64 else MachineOp.FLW
        self._emit(
            load_op,
            self._fp_dst(instr),
            address,
            comment=f"bit-exact {dtype.value} constant",
        )
        self._fp_definitions.add(instr.dest.name)

    def _select_neg(self, instr: Instruction) -> None:
        if not self._is_float(instr):
            super()._select_neg(instr)
            return
        dtype = self._float_dtype(instr)
        self._require_fp_extension(OpCode.NEG, dtype)
        op = MachineOp.FNEG_D if dtype is DataType.FLOAT64 else MachineOp.FNEG_S
        self._emit(
            op, self._fp_dst(instr), self._fp_op(instr, 0),
            comment=op.value,
        )

    def _select_br_if(self, instr: Instruction) -> None:
        if not (
            len(instr.operands) == 2
            and instr.operands[0].dtype in (DataType.FLOAT32, DataType.FLOAT64)
        ):
            super()._select_br_if(instr)
            return
        dtype = instr.operands[0].dtype
        self._require_fp_extension(OpCode.BR_IF, dtype)
        targets = (instr.target or ",").split(",")
        true_target = self._block_label(targets[0].strip())
        false_target = self._block_label(targets[1].strip())
        lhs = self._fp_op(instr, 0)
        rhs = self._fp_op(instr, 1)
        predicate = instr.attrs.get("cmp_op")
        suffix_ops = (
            (MachineOp.FEQ_S, MachineOp.FLT_S, MachineOp.FLE_S)
            if dtype is DataType.FLOAT32
            else (MachineOp.FEQ_D, MachineOp.FLT_D, MachineOp.FLE_D)
        )
        feq, flt, fle = suffix_ops
        comparisons = {
            "==": (feq, lhs, rhs, False),
            "!=": (feq, lhs, rhs, True),
            "<": (flt, lhs, rhs, False),
            ">": (flt, rhs, lhs, False),
            "<=": (fle, lhs, rhs, False),
            ">=": (fle, rhs, lhs, False),
        }
        if predicate not in comparisons:
            raise ValueError(f"unsupported comparison operator: {predicate}")
        compare_op, first, second, invert = comparisons[predicate]
        condition = self._fresh_vreg("fp_cmp")
        self._emit(compare_op, condition, first, second, comment=str(predicate))
        if invert:
            self._emit(
                MachineOp.BEQ,
                condition,
                MachineOperand.reg("x0"),
                target=true_target,
            )
        else:
            self._emit(MachineOp.BNEZ, condition, target=true_target)
        self._emit(MachineOp.J, target=false_target)

    def _select_return(self, instr: Instruction) -> None:
        if instr.operands and instr.operands[0].dtype in (
            DataType.FLOAT32, DataType.FLOAT64,
        ):
            self._require_fp_extension(
                OpCode.RETURN, instr.operands[0].dtype,
            )
            width = 32 if instr.operands[0].dtype is DataType.FLOAT32 else 64
            self._emit(
                MachineOp.FMV_S if width == 32 else MachineOp.FMV_D,
                MachineOperand.reg("fa0", RegisterClass.FPR, width),
                self._fp_op(instr, 0),
                comment="floating ABI return",
            )
            self._emit(
                MachineOp.JALR,
                MachineOperand.reg("zero"),
                MachineOperand.reg("ra"),
                comment="return",
            )
            return
        super()._select_return(instr)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_float(instr: Instruction) -> bool:
        values = instr.operands + (
            [instr.dest] if instr.dest is not None else []
        )
        return any(
            value.dtype in (DataType.FLOAT32, DataType.FLOAT64)
            for value in values
        )

    @staticmethod
    def _float_dtype(instr: Instruction) -> DataType:
        values = (
            ([instr.dest] if instr.dest is not None else [])
            + instr.operands
        )
        dtypes = {value.dtype for value in values}
        if (
            len(dtypes) != 1
            or next(iter(dtypes), None)
            not in (DataType.FLOAT32, DataType.FLOAT64)
        ):
            rendered = ", ".join(sorted(dtype.value for dtype in dtypes))
            raise ValueError(
                "floating instruction requires one consistent dtype"
                + (f": {rendered}" if rendered else "")
            )
        return next(iter(dtypes))

    def _require_fp_extension(
        self, opcode: OpCode, dtype: DataType,
    ) -> None:
        self.features.require_float(opcode, dtype)

    def _fp_op(self, instr: Instruction, index: int) -> MachineOperand:
        value = instr.operands[index]
        if value.name not in self._fp_definitions:
            raise ValueError(
                f"{value.dtype.value} value {value.name!r} has no machine definition; "
                "use LOAD_CONST, a scalar constant global, an ABI parameter "
                "or a previously defined floating-point result"
            )
        return MachineOperand.vreg(
            value.name, RegisterClass.FPR,
            32 if value.dtype is DataType.FLOAT32 else 64,
        )

    @staticmethod
    def _fp_dst(instr: Instruction) -> MachineOperand | None:
        if instr.dest is None:
            return None
        return MachineOperand.vreg(
            instr.dest.name, RegisterClass.FPR,
            32 if instr.dest.dtype is DataType.FLOAT32 else 64,
        )

    def _intern_float_constant(
        self, instr: Instruction, dtype: DataType,
    ) -> str:
        raw_bits = instr.attrs.get("bit_pattern")
        value_bits = instr.dest.bit_pattern if instr.dest is not None else None
        if raw_bits is not None and value_bits is not None and raw_bits != value_bits:
            raise ValueError("constant bit_pattern differs from result metadata")
        if raw_bits is None:
            raw_bits = value_bits
        width = 32 if dtype is DataType.FLOAT32 else 64
        if raw_bits is not None:
            if (isinstance(raw_bits, bool) or not isinstance(raw_bits, int)
                    or not 0 <= raw_bits < (1 << width)):
                raise ValueError(
                    f"invalid {dtype.value} constant bit_pattern"
                )
            bits = raw_bits
        else:
            raw_value = instr.attrs.get("value")
            if not isinstance(raw_value, (int, float)):
                raise ValueError(f"missing {dtype.value} constant value")
            if dtype is DataType.FLOAT32:
                bits = struct.unpack(">I", struct.pack(">f", float(raw_value)))[0]
            else:
                bits = struct.unpack(">Q", struct.pack(">d", float(raw_value)))[0]
        key = (dtype, bits)
        label = self._constant_labels.get(key)
        if label is not None:
            return label
        index = sum(
            entry.bit_width == width for entry in self._constant_pool
        )
        label = self._claim_label(f".LC_{dtype.value}_{index}")
        self._constant_labels[key] = label
        self._constant_pool.append(FloatConstant(label, width, bits))
        return label

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
            "rem",
        ]
        return base_ops + extended_ops


# New MachineOp entries for the extended selector are now defined directly
# in the MachineOp enum in scratchv.backend.register_alloc.
