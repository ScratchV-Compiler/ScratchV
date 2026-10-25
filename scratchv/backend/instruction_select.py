"""Instruction selection: IR instructions to RISC-V pseudo-instructions.

This phase lowers each IR instruction to a sequence of RISC-V machine
instructions, producing a flat list of MachineInstrs that still use
virtual registers.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from scratchv.ir.types import Instruction, Function, Program
from scratchv.backend.symbols import validate_function_symbol
from scratchv.backend.machine_types import (
    MachineInstr,
    MachineOp,
    MachineOperand,
    ZERO_REG,
)


class InstructionSelector:
    """Select RISC-V instructions for each IR instruction."""

    def __init__(self, program: Program):
        self.program = program
        self._instructions: list[MachineInstr] = []
        self._label_counter = 0
        self._stack_offset = 0
        self._temp_counter = 0
        self._loop_stack: list[dict] = []
        self._reserved_vreg_names = self._collect_ir_value_names()
        self._block_labels: dict[str, str] = {}
        self._reserved_labels: set[str] = set()
        self._bound_value_names: set[str] = set()
        self._globals = {value.name: value for value in program.global_values}

    def _collect_ir_value_names(self) -> set[str]:
        """Reserve every user-visible IR name before creating temporaries."""
        names = {value.name for value in self.program.global_values}
        for func in self.program.functions:
            names.update(value.name for value in func.params)
            names.update(value.name for value in func.returns)
            names.update(value.name for value in func.locals)
            for block in func.blocks:
                for instr in (*block.phi_nodes, *block.instructions):
                    if instr.dest is not None:
                        names.add(instr.dest.name)
                    names.update(value.name for value in instr.operands)
        return names

    def run(self) -> list[MachineInstr]:
        """Select instructions for all functions.

        Returns flat list of MachineInstrs.
        """
        self._instructions = []
        self._label_counter = 0
        self._temp_counter = 0
        self._loop_stack = []
        self._reserved_vreg_names = self._collect_ir_value_names()
        for func in self.program.functions:
            validate_function_symbol(func.name)
        self._reserved_labels = {func.name for func in self.program.functions}
        self._reserved_labels.update(
            instr.target for func in self.program.functions
            for block in func.blocks for instr in block.instructions
            if instr.opcode.value == "label" and instr.target
        )
        multiple_functions = len(self.program.functions) > 1
        block_labels = []
        for function_index, func in enumerate(self.program.functions):
            # Numeric namespaces avoid concatenation collisions between user
            # function/block names. Reserve all blocks before any pool or
            # internal labels are generated; retain nonconflicting legacy names.
            block_labels.append({
                block.name: self._claim_label(
                    f".Lsv_f{function_index}_b{block_index}"
                    if multiple_functions else f".{block.name}"
                )
                for block_index, block in enumerate(func.blocks)
            })
        for function_index, (func, labels) in enumerate(zip(self.program.functions, block_labels)):
            self._block_labels = labels
            # Resolve named definitions before literal hints for every selector,
            # including subclasses with their own function/ABI lowering.
            self._bound_value_names = {value.name for value in func.params}
            self._bound_value_names.update(
                instr.dest.name
                for block in func.blocks
                for instr in (*block.phi_nodes, *block.instructions)
                if instr.dest is not None
            )
            start = len(self._instructions)
            self._select_function(func)
            if multiple_functions:
                self._namespace_function_registers(start, function_index)
        return self._instructions

    def _namespace_function_registers(self, start: int, function_index: int) -> dict[str, str]:
        """Give function-local vregs unique identities in the flat machine stream.

        Rewrite only machine operands, preserving the caller's IR names and
        operand class/width. Fresh names avoid every source and temporary name.
        Subclasses use the returned mapping to rename associated local objects.
        """
        names: dict[str, str] = {}

        def scoped(operand: MachineOperand | None) -> MachineOperand | None:
            if operand is None or operand.kind != "vreg":
                return operand
            name = str(operand.value)
            if name not in names:
                names[name] = str(self._fresh_vreg(f"f{function_index}").value)
            return replace(operand, value=names[name])

        for index in range(start, len(self._instructions)):
            instruction = self._instructions[index]
            self._instructions[index] = replace(
                instruction, dst=scoped(instruction.dst),
                src1=scoped(instruction.src1), src2=scoped(instruction.src2),
            )
        return names

    def _claim_label(self, preferred: str) -> str:
        """Allocate one symbol in the shared block/pool/internal namespace."""
        label = preferred
        suffix = 0
        while label in self._reserved_labels:
            suffix += 1
            label = f"{preferred}_{suffix}"
        self._reserved_labels.add(label)
        return label

    def _fresh_label(self, prefix: str = "L") -> str:
        self._label_counter += 1
        return self._claim_label(f".L{prefix}_{self._label_counter}")

    def _fresh_vreg(self, prefix: str = "tmp") -> MachineOperand:
        """Return a compiler-owned virtual register with a unique name.

        IR names are collected up front, so even deliberately adversarial
        source names cannot alias a selector temporary.  Generated names are
        reserved as well, making the helper safe for every selector method.
        """
        while True:
            self._temp_counter += 1
            name = f"__scratchv_{prefix}_{self._temp_counter}"
            if name not in self._reserved_vreg_names:
                self._reserved_vreg_names.add(name)
                return MachineOperand.vreg(name)

    def _materialize_int_operand(
        self,
        operand: MachineOperand,
        *,
        prefix: str,
    ) -> MachineOperand:
        """Materialize an integer immediate for a register-only instruction."""
        if operand.kind != "imm":
            return operand
        temp = self._fresh_vreg(prefix)
        self._emit(MachineOp.LI, temp, operand, comment="materialize integer operand")
        return temp

    def _select_function(self, func: Function) -> None:
        # Function prologue label
        self._emit_label(func.name, is_function=True)
        self._stack_offset = 0

        for block in func.blocks:
            self._emit_label(self._block_label(block.name))
            for instr in block.instructions:
                self._select_instruction(instr)

    def _select_instruction(self, instr: Instruction) -> None:
        handler = getattr(self, f"_select_{instr.opcode.value}", None)
        if handler is None:
            raise ValueError(f"No instruction selection for opcode: {instr.opcode}")
        handler(instr)

    def _emit(self, op: MachineOp, dst=None, src1=None, src2=None,
              comment: str = "", target: Optional[str] = None,
              rounding_mode: Optional[str] = None) -> None:
        self._instructions.append(
            MachineInstr(
                op, dst, src1, src2, comment, target, rounding_mode,
            )
        )

    def _emit_move(self, dst: MachineOperand, src: MachineOperand,
                   comment: str = "") -> None:
        """Emit a legal copy pseudo for either a register or an immediate."""
        if src.kind == "imm":
            self._emit(MachineOp.LI, dst, src, comment=comment)
        else:
            self._emit(MachineOp.MV, dst, src, comment=comment)

    def _emit_max(self, dst: MachineOperand | None,
                  lhs: MachineOperand, rhs: MachineOperand,
                  comment: str = "") -> None:
        """Emit a MAX pseudo in the canonical form accepted by the encoder.

        ``MAX`` is commutative, so an immediate left operand is first moved
        to the right.  The encoder accepts an immediate right operand only
        when it is zero; other immediates are materialized in a fresh virtual
        register before emitting the pseudo.
        """
        if dst is None:
            return

        if lhs.kind == "imm" and rhs.kind == "imm":
            value = max(int(lhs.value), int(rhs.value))
            self._emit(
                MachineOp.LI,
                dst,
                MachineOperand.immediate(value),
                comment=comment,
            )
            return

        if lhs.kind == "imm":
            lhs, rhs = rhs, lhs

        if rhs.kind == "imm" and int(rhs.value) != 0:
            rhs_reg = self._fresh_vreg("max_rhs")
            self._emit(
                MachineOp.LI,
                rhs_reg,
                rhs,
                comment="materialize max rhs",
            )
            rhs = rhs_reg

        self._emit(MachineOp.MAX, dst, lhs, rhs, comment=comment)

    def _emit_label(self, name: str, *, is_function: bool = False) -> None:
        self._instructions.append(MachineInstr(
            MachineOp.LABEL, target=name,
            label_role="function" if is_function else "block",
        ))

    def _op(self, instr: Instruction, idx: int):
        """Get an operand from an IR instruction as a machine operand."""
        op = instr.operands[idx]
        if op.name in self._bound_value_names:
            return MachineOperand.vreg(op.name)
        # Global scalar literals are defined by Program.global_values, not by
        # potentially incomplete or stale metadata on a separate reference.
        op = self._globals.get(op.name, op)
        if op.is_constant and op.const_value is not None:
            return MachineOperand.immediate(int(op.const_value))
        return MachineOperand.vreg(op.name)

    @staticmethod
    def _reg_op(instr: Instruction, idx: int) -> MachineOperand:
        """Return an operand through its materialized virtual register."""
        return MachineOperand.vreg(instr.operands[idx].name)

    def _dst(self, instr: Instruction):
        if instr.dest is None:
            return None
        return MachineOperand.vreg(instr.dest.name)

    # --- Per-opcode selectors ---

    def _select_load_const(self, instr: Instruction) -> None:
        raw_val = instr.attrs.get("value", 0)
        assert isinstance(raw_val, (int, float))
        val = int(raw_val)
        dst = self._dst(instr)
        # LI pseudo-instruction (expands to addi x0, imm or lui+addi)
        self._emit(
            MachineOp.LI,
            dst,
            MachineOperand.immediate(int(val)),
            comment=f"const {val}",
        )

    def _select_add(self, instr: Instruction) -> None:
        self._select_int_binary(instr, MachineOp.ADD)

    def _select_sub(self, instr: Instruction) -> None:
        self._select_int_binary(instr, MachineOp.SUB)

    def _select_mul(self, instr: Instruction) -> None:
        self._select_int_binary(instr, MachineOp.MUL)

    def _select_int_binary(self, instr: Instruction, op: MachineOp) -> None:
        """Register-only integer operations cannot consume inline constants."""
        lhs = self._materialize_int_operand(
            self._op(instr, 0), prefix=f"{op.value}_lhs"
        )
        rhs = self._materialize_int_operand(
            self._op(instr, 1), prefix=f"{op.value}_rhs"
        )
        self._emit(
            op, self._dst(instr), lhs, rhs
        )

    def _select_div(self, instr: Instruction) -> None:
        lhs = self._materialize_int_operand(
            self._op(instr, 0), prefix="div_lhs"
        )
        rhs = self._materialize_int_operand(
            self._op(instr, 1), prefix="div_rhs"
        )
        self._emit(
            MachineOp.DIV, self._dst(instr), lhs, rhs
        )

    def _select_neg(self, instr: Instruction) -> None:
        # RISC-V: sub rd, x0, rs
        src = self._materialize_int_operand(self._op(instr, 0), prefix="neg_src")
        self._emit(
            MachineOp.SUB,
            self._dst(instr),
            MachineOperand.reg("x0"),
            src,
        )

    def _select_exp(self, instr: Instruction) -> None:
        # exp(x) approximated as max(0, 1+x) for simplicity (pure RV32I)
        src = self._op(instr, 0)
        dst = self._dst(instr)
        if dst is None:
            return
        self._emit(
            MachineOp.ADDI,
            dst,
            src,
            MachineOperand.immediate(1),
            comment="exp approx: 1+x",
        )
        self._emit_max(
            dst, dst, MachineOperand.immediate(0), comment="relu clamp"
        )

    def _select_relu(self, instr: Instruction) -> None:
        """ReLU(x) = max(x, 0).  Use:  max rd, rs, x0"""
        src = self._op(instr, 0)
        dst = self._dst(instr)
        self._emit_max(dst, src, MachineOperand.immediate(0))

    def _select_gelu(self, instr: Instruction) -> None:
        # GELU approx: x * relu(x) / 2 (simplified, pure RV32IM)
        src = self._op(instr, 0)
        dst = self._dst(instr)
        if dst is None:
            return
        tmp = self._fresh_vreg("gelu_relu")
        self._emit_max(
            tmp, src, MachineOperand.immediate(0), comment="relu(x)"
        )
        self._emit(MachineOp.MUL, dst, src, tmp, comment="x * relu(x)")
        divisor = self._materialize_int_operand(
            MachineOperand.immediate(2), prefix="gelu_divisor"
        )
        self._emit(MachineOp.DIV, dst, dst, divisor, comment="/ 2")

    def _select_softmax(self, instr: Instruction) -> None:
        # softmax ≈ identity (pure RV32I passthrough)
        src = self._op(instr, 0)
        dst = self._dst(instr)
        if dst and src:
            self._emit_move(dst, src, comment="softmax passthrough")

    def _select_reshape(self, instr: Instruction) -> None:
        # Reshape is a no-op: just copy the value
        src = self._op(instr, 0)
        dst = self._dst(instr)
        if dst and src:
            self._emit_move(dst, src, comment="reshape")

    def _select_load(self, instr: Instruction) -> None:
        address = self._materialize_int_operand(self._op(instr, 0), prefix="load_addr")
        self._emit(MachineOp.LW, self._dst(instr), address)

    def _select_store(self, instr: Instruction) -> None:
        # Machine SW follows the standard order: value, address.
        value = self._materialize_int_operand(self._op(instr, 1), prefix="store_value")
        address = self._materialize_int_operand(self._op(instr, 0), prefix="store_addr")
        self._emit(MachineOp.SW, value, address)

    def _select_alloca(self, instr: Instruction) -> None:
        raw_size = instr.attrs.get("size", 4)
        assert isinstance(raw_size, int)
        size = max(raw_size, 1)
        size = (size + 3) // 4 * 4
        self._stack_offset += size
        dst = self._dst(instr)
        self._emit(
            MachineOp.ADDI,
            dst,
            MachineOperand.reg("sp"),
            MachineOperand.immediate(-self._stack_offset),
            comment=f"alloca {size}",
        )

    def _select_for(self, instr: Instruction) -> None:
        """Begin a for loop: set up loop variable and branch to loop header."""
        iv = self._dst(instr)
        raw_start = instr.attrs.get("start", 0)
        assert isinstance(raw_start, int)
        start = raw_start
        raw_end = instr.attrs.get("end", 0)
        assert isinstance(raw_end, int)
        end = raw_end
        step = instr.attrs.get("step", 1)
        if (isinstance(step, bool) or not isinstance(step, int)
                or not 0 < step <= 0x7fff_ffff):
            raise ValueError("FOR step must be a positive i32 integer")

        # Emit loop header label (will be patched)
        header_label = self._fresh_label("loop_header")
        body_label = self._fresh_label("loop_body")
        exit_label = self._fresh_label("loop_exit")

        # Initialize loop variable
        self._emit(
            MachineOp.LI, iv, MachineOperand.immediate(start), comment="loop init"
        )

        # Branch to loop body
        # Store loop context for endfor to use.  Nested loops must keep a
        # stack so the inner ENDFOR cannot clobber the outer loop's labels.
        self._loop_stack.append({
            "iv": iv,
            "end": end,
            "step": step,
            "header": header_label,
            "body": body_label,
            "exit": exit_label,
        })

        # Recreate the constant in the header, just before its only use. A
        # definition outside the header stays live across the backedge and
        # forces local allocators to spill/reload an otherwise cheap literal.
        self._emit_label(header_label)
        if end != 0 and -2048 <= end <= 2047:
            # SLTI consumes the bound directly. Avoid executing a separate
            # constant-loading ADDI on every trip through the header.
            condition = self._fresh_vreg("loop_condition")
            self._emit(MachineOp.SLTI, condition, iv, MachineOperand.immediate(end))
            self._emit(MachineOp.BEQ, condition, MachineOperand.reg(ZERO_REG), target=exit_label)
        else:
            end_val = self._materialize_int_operand(
                MachineOperand.immediate(int(end)), prefix="loop_end"
            )
            self._emit(MachineOp.BGE, iv, end_val, target=exit_label)
        self._emit_label(body_label)

    def _select_endfor(self, instr: Instruction) -> None:
        """End a for loop: increment and branch back."""
        if not self._loop_stack:
            raise ValueError("endfor without matching for")
        ctx = self._loop_stack.pop()

        iv = ctx["iv"]
        step = ctx["step"]
        # With a positive step greater than one the last increment can overflow
        # i32 before reaching the header. Such an overshoot ends the finite loop.
        previous = None
        if step > 1:
            previous = self._fresh_vreg("loop_previous")
            self._emit_move(previous, iv)
        if step <= 2047:
            self._emit(MachineOp.ADDI, iv, iv, MachineOperand.immediate(step), comment="loop inc")
        else:
            increment = self._materialize_int_operand(
                MachineOperand.immediate(step), prefix="loop_step"
            )
            self._emit(MachineOp.ADD, iv, iv, increment, comment="loop inc")
        if previous is not None:
            self._emit(MachineOp.BLT, iv, previous, target=ctx["exit"])
        # Jump back to header
        self._emit(MachineOp.J, target=ctx["header"])
        # Exit label
        self._emit_label(ctx["exit"])

    def _select_br(self, instr: Instruction) -> None:
        self._emit(
            MachineOp.J,
            target=self._block_label(instr.target or ""),
        )

    def _select_br_if(self, instr: Instruction) -> None:
        targets = (instr.target or ",").split(",")
        true_target = self._block_label(
            targets[0].strip() if len(targets) > 0 else ""
        )
        false_target = self._block_label(
            targets[1].strip() if len(targets) > 1 else ""
        )

        if len(instr.operands) == 2 and "cmp_op" in instr.attrs:
            # Inline constants are valid IR operands too; do not assume every
            # literal was accompanied by a LOAD_CONST definition.
            lhs = self._materialize_int_operand(self._op(instr, 0), prefix="branch_lhs")
            rhs = self._materialize_int_operand(self._op(instr, 1), prefix="branch_rhs")
            operator = instr.attrs["cmp_op"]
            branches = {
                "==": (MachineOp.BEQ, lhs, rhs),
                "!=": (MachineOp.BNE, lhs, rhs),
                "<": (MachineOp.BLT, lhs, rhs),
                ">": (MachineOp.BLT, rhs, lhs),
                "<=": (MachineOp.BGE, rhs, lhs),
                ">=": (MachineOp.BGE, lhs, rhs),
            }
            if operator not in branches:
                raise ValueError(
                    f"unsupported comparison operator: {operator}"
                )
            branch_op, first, second = branches[operator]
            self._emit(
                branch_op, first, second, target=true_target,
            )
        elif len(instr.operands) == 1:
            cond = self._materialize_int_operand(self._op(instr, 0), prefix="branch_cond")
            self._emit(MachineOp.BNEZ, cond, target=true_target)
        else:
            raise ValueError(
                "br_if expects a boolean or comparison operands"
            )
        self._emit(MachineOp.J, target=false_target)

    def _block_label(self, name: str) -> str:
        if name in self._block_labels:
            return self._block_labels[name]
        if not name or name.startswith("."):
            return name
        return f".{name}"

    def _select_return(self, instr: Instruction) -> None:
        if instr.operands:
            self._emit_move(
                MachineOperand.reg("a0"),
                self._op(instr, 0),
                comment="return value",
            )
        self._emit(
            MachineOp.JALR,
            MachineOperand.reg("zero"),
            MachineOperand.reg("ra"),
            comment="ret",
        )

    def _select_matmul(self, instr: Instruction) -> None:
        a_reg = self._op(instr, 0)
        b_reg = self._op(instr, 1)
        dst = self._dst(instr)
        if dst:
            self._emit(MachineOp.MUL, dst, a_reg, b_reg, comment="matmul: a * b")

    def _select_dot(self, instr: Instruction) -> None:
        a_reg = self._op(instr, 0)
        b_reg = self._op(instr, 1)
        dst = self._dst(instr)
        if dst:
            self._emit(MachineOp.MUL, dst, a_reg, b_reg, comment="dot: a * b")

    def _select_label(self, instr: Instruction) -> None:
        self._emit_label(instr.target or "")

    def _select_sigmoid(self, instr: Instruction) -> None:
        """Sigmoid inline: x<0 → 0, x>1 → 1, else x. Pure integer RV32I."""
        src = self._op(instr, 0)
        dst = self._dst(instr)
        if dst is None or src is None:
            return
        # sigmoid approximation using integer ops only:
        # if x > 0: result = min(x, 1) else result = 0
        # slti t0, src, 1  → t0 = (src < 1) ? 1 : 0
        # bnez t0, keep    → if < 1, keep value
        # li dst, 1        → else clamp to 1
        # keep: mv dst, src
        keep_label = self._fresh_label("sig_keep")
        below_one = self._fresh_vreg("sig_below_one")
        self._emit(
            MachineOp.SLT,
            below_one,
            src,
            MachineOperand.immediate(1),
            comment="src < 1 ?",
        )
        self._emit(MachineOp.BNEZ, below_one, target=keep_label)
        self._emit(MachineOp.LI, dst, MachineOperand.immediate(1), comment="clamp to 1")
        # Branch over the mv
        done_label = self._fresh_label("sig_done")
        self._emit(MachineOp.J, target=done_label)
        self._emit_label(keep_label)
        self._emit_move(dst, src, comment="keep src")
        self._emit_label(done_label)
        # Now dst = min(src, 1). If src < 0, result = 0
        above_zero = self._fresh_vreg("sig_above_zero")
        self._emit(
            MachineOp.SLT,
            above_zero,
            MachineOperand.immediate(0),
            src,
            comment="0 < src ?",
        )
        zero_label = self._fresh_label("sig_zero")
        self._emit(MachineOp.BNEZ, above_zero, target=zero_label)
        self._emit(MachineOp.LI, dst, MachineOperand.immediate(0), comment="clamp to 0")
        self._emit_label(zero_label)

    def _select_conv(self, instr: Instruction) -> None:
        """Conv2D: real RISC-V MAC inline (simplified single-MAC)."""
        dst = self._dst(instr)
        x_reg = self._op(instr, 0)
        w_reg = self._op(instr, 1)
        b_reg = self._op(instr, 2)
        if dst:
            # acc = bias (mv bias to dest)
            self._emit_move(dst, b_reg, comment="acc = bias")
            # tmp = x * w (MUL for MAC)
            tmp_vreg = self._fresh_vreg("conv_mac")
            self._emit(MachineOp.MUL, tmp_vreg, x_reg, w_reg, comment="tmp = x * w")
            # dst = dst + tmp (acc += x*w)
            self._emit(MachineOp.ADD, dst, dst, tmp_vreg, comment="acc += x*w")

    def _select_gemm(self, instr: Instruction) -> None:
        """GEMM inline: real RISC-V MUL+ADD MAC."""
        if any(instr.attrs.get(name, 1) != 1 for name in ("alpha", "beta")):
            raise ValueError("Legacy RISC-V GEMM does not support nondefault alpha/beta scaling")
        dst = self._dst(instr)
        a_reg = self._op(instr, 0)
        w_reg = self._op(instr, 1)
        b_reg = self._op(instr, 2)
        if dst:
            self._emit_move(dst, b_reg, comment="acc = bias")
            tmp_vreg = self._fresh_vreg("gemm_mac")
            self._emit(MachineOp.MUL, tmp_vreg, a_reg, w_reg, comment="tmp = a * w")
            self._emit(MachineOp.ADD, dst, dst, tmp_vreg, comment="acc += a*w")

    def _select_maxpool(self, instr: Instruction) -> None:
        """MaxPool inline: RISC-V SLT + branch → max."""
        src = self._op(instr, 0)
        dst = self._dst(instr)
        if dst is None or src is None:
            return
        # max(x, 0) using SLT + branch
        gt_label = self._fresh_label("mp_gt")
        above_zero = self._fresh_vreg("maxpool_above_zero")
        self._emit(
            MachineOp.SLT,
            above_zero,
            MachineOperand.immediate(0),
            src,
            comment="0 < x ?",
        )
        self._emit(MachineOp.BNEZ, above_zero, target=gt_label)
        self._emit(MachineOp.LI, dst, MachineOperand.immediate(0), comment="result = 0")
        done_label = self._fresh_label("mp_done")
        self._emit(MachineOp.J, target=done_label)
        self._emit_label(gt_label)
        self._emit_move(dst, src, comment="result = x")
        self._emit_label(done_label)
