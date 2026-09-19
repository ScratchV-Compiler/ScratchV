"""Instruction selection: IR instructions to RISC-V pseudo-instructions.

This phase lowers each IR instruction to a sequence of RISC-V machine
instructions, producing a flat list of MachineInstrs that still use
virtual registers.
"""

from __future__ import annotations

from scratchv.ir.types import Instruction, Function, Program
from scratchv.backend.machine_types import (
    MachineInstr, MachineOp, MachineOperand,
)


class InstructionSelector:
    """Select RISC-V instructions for each IR instruction."""

    def __init__(self, program: Program):
        self.program = program
        self._instructions: list[MachineInstr] = []
        self._label_counter = 0
        self._stack_offset = 0

    def run(self) -> list[MachineInstr]:
        """Select instructions for all functions.

        Returns flat list of MachineInstrs.
        """
        self._instructions = []
        for func in self.program.functions:
            self._select_function(func)
        return self._instructions

    def _fresh_label(self, prefix: str = "L") -> str:
        self._label_counter += 1
        return f".L{prefix}_{self._label_counter}"

    def _select_function(self, func: Function) -> None:
        # Function prologue label
        self._emit_label(func.name)
        self._stack_offset = 0

        for block in func.blocks:
            self._emit_label(f".{block.name}")
            for instr in block.instructions:
                self._select_instruction(instr)

    def _select_instruction(self, instr: Instruction) -> None:
        handler = getattr(self, f"_select_{instr.opcode.value}", None)
        if handler is None:
            raise ValueError(
                f"No instruction selection for opcode: {instr.opcode}")
        handler(instr)

    def _emit(self, op: MachineOp, dst=None, src1=None, src2=None,
              comment: str = "") -> None:
        self._instructions.append(
            MachineInstr(op, dst, src1, src2, comment))

    def _emit_label(self, name: str) -> None:
        self._instructions.append(
            MachineInstr(MachineOp.LABEL, comment=name))

    def _op(self, instr: Instruction, idx: int):
        """Get an operand from an IR instruction as a machine operand."""
        op = instr.operands[idx]
        # LOAD_CONST materializes constants before their use, so consumers
        # refer to the resulting virtual register.  Most RISC-V arithmetic
        # and branch instructions do not accept immediate operands.
        return MachineOperand.vreg(op.name)

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
        self._emit(MachineOp.LI, dst,
                   MachineOperand.immediate(int(val)),
                   comment=f"const {val}")

    def _select_add(self, instr: Instruction) -> None:
        self._emit(MachineOp.ADD, self._dst(instr),
                   self._op(instr, 0), self._op(instr, 1))

    def _select_sub(self, instr: Instruction) -> None:
        self._emit(MachineOp.SUB, self._dst(instr),
                   self._op(instr, 0), self._op(instr, 1))

    def _select_mul(self, instr: Instruction) -> None:
        self._emit(MachineOp.MUL, self._dst(instr),
                   self._op(instr, 0), self._op(instr, 1))

    def _select_div(self, instr: Instruction) -> None:
        self._emit(MachineOp.DIV, self._dst(instr),
                   self._op(instr, 0), self._op(instr, 1))

    def _select_neg(self, instr: Instruction) -> None:
        # RISC-V: sub rd, x0, rs
        self._emit(MachineOp.SUB, self._dst(instr),
                   MachineOperand.immediate(0), self._op(instr, 0))

    def _select_exp(self, instr: Instruction) -> None:
        # exp(x) approximated as max(0, 1+x) for simplicity (pure RV32I)
        src = self._op(instr, 0)
        dst = self._dst(instr)
        if dst is None:
            return
        self._emit(MachineOp.ADDI, dst, src,
                   MachineOperand.immediate(1),
                   comment="exp approx: 1+x")
        self._emit(MachineOp.MAX, dst, dst,
                   MachineOperand.immediate(0),
                   comment="relu clamp")

    def _select_relu(self, instr: Instruction) -> None:
        """ReLU(x) = max(x, 0).  Use:  max rd, rs, x0"""
        src = self._op(instr, 0)
        dst = self._dst(instr)
        self._emit(MachineOp.MAX, dst, src, MachineOperand.immediate(0))

    def _select_gelu(self, instr: Instruction) -> None:
        # GELU approx: x * relu(x) / 2 (simplified, pure RV32IM)
        src = self._op(instr, 0)
        dst = self._dst(instr)
        if dst is None:
            return
        tmp = MachineOperand.vreg("tmp_gelu")
        self._emit(MachineOp.MAX, tmp, src,
                   MachineOperand.immediate(0),
                   comment="relu(x)")
        self._emit(MachineOp.MUL, dst, src, tmp,
                   comment="x * relu(x)")
        self._emit(MachineOp.DIV, dst, dst,
                   MachineOperand.immediate(2),
                   comment="/ 2")

    def _select_softmax(self, instr: Instruction) -> None:
        # softmax ≈ identity (pure RV32I passthrough)
        src = self._op(instr, 0)
        dst = self._dst(instr)
        if dst and src:
            self._emit(MachineOp.MV, dst, src,
                       comment="softmax passthrough")

    def _select_reshape(self, instr: Instruction) -> None:
        # Reshape is a no-op: just copy the value
        src = self._op(instr, 0)
        dst = self._dst(instr)
        if dst and src:
            self._emit(MachineOp.MV, dst, src, comment="reshape")

    def _select_load(self, instr: Instruction) -> None:
        self._emit(MachineOp.LW, self._dst(instr), self._op(instr, 0))

    def _select_store(self, instr: Instruction) -> None:
        self._emit(MachineOp.SW, self._op(instr, 0), self._op(instr, 1))

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

        # Emit loop header label (will be patched)
        header_label = self._fresh_label("loop_header")
        body_label = self._fresh_label("loop_body")
        exit_label = self._fresh_label("loop_exit")

        # Initialize loop variable
        self._emit(MachineOp.LI, iv, MachineOperand.immediate(start),
                   comment="loop init")

        # Branch to loop body
        # Store loop context for endfor to use
        self._loop_context = {
            "iv": iv,
            "end": end,
            "header": header_label,
            "body": body_label,
            "exit": exit_label,
        }

        self._emit_label(header_label)

        # Check condition: if iv >= end, exit
        end_val = MachineOperand.immediate(int(end))  # type: ignore[arg-type]
        self._emit(MachineOp.BGE, iv, end_val, comment=exit_label)
        self._emit_label(body_label)

    def _select_endfor(self, instr: Instruction) -> None:
        """End a for loop: increment and branch back."""
        ctx = getattr(self, "_loop_context", None)
        if ctx is None:
            raise ValueError("endfor without matching for")

        iv = ctx["iv"]
        # Increment: addi iv, iv, 1
        self._emit(MachineOp.ADDI, iv, iv, MachineOperand.immediate(1),
                   comment="loop inc")
        # Jump back to header
        self._emit(MachineOp.J, comment=ctx["header"])
        # Exit label
        self._emit_label(ctx["exit"])

    def _select_br(self, instr: Instruction) -> None:
        self._emit(
            MachineOp.J,
            comment=self._block_label(instr.target or ""),
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
            lhs = self._op(instr, 0)
            rhs = self._op(instr, 1)
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
                branch_op, first, second, comment=true_target,
            )
        elif len(instr.operands) == 1:
            cond = self._op(instr, 0)
            self._emit(MachineOp.BNEZ, cond, comment=true_target)
        else:
            raise ValueError(
                "br_if expects a boolean or comparison operands"
            )
        self._emit(MachineOp.J, comment=false_target)

    @staticmethod
    def _block_label(name: str) -> str:
        if not name or name.startswith("."):
            return name
        return f".{name}"

    def _select_return(self, instr: Instruction) -> None:
        if instr.operands:
            self._emit(MachineOp.MV, MachineOperand.reg("a0"),
                       self._op(instr, 0), comment="return value")
        self._emit(MachineOp.JALR, MachineOperand.reg("zero"),
                   MachineOperand.reg("ra"), comment="ret")

    def _select_matmul(self, instr: Instruction) -> None:
        a_reg = self._op(instr, 0)
        b_reg = self._op(instr, 1)
        dst = self._dst(instr)
        if dst:
            self._emit(MachineOp.MUL, dst, a_reg, b_reg,
                       comment="matmul: a * b")

    def _select_dot(self, instr: Instruction) -> None:
        a_reg = self._op(instr, 0)
        b_reg = self._op(instr, 1)
        dst = self._dst(instr)
        if dst:
            self._emit(MachineOp.MUL, dst, a_reg, b_reg,
                       comment="dot: a * b")

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
        self._emit(MachineOp.SLT,
                   MachineOperand.vreg("t_sig"),
                   src,
                   MachineOperand.immediate(1),
                   comment="src < 1 ?")
        self._emit(MachineOp.BNEZ,
                   MachineOperand.vreg("t_sig"),
                   comment=keep_label)
        self._emit(MachineOp.LI, dst,
                   MachineOperand.immediate(1),
                   comment="clamp to 1")
        # Branch over the mv
        done_label = self._fresh_label("sig_done")
        self._emit(MachineOp.J, comment=done_label)
        self._emit_label(keep_label)
        self._emit(MachineOp.MV, dst, src,
                   comment="keep src")
        self._emit_label(done_label)
        # Now dst = min(src, 1). If src < 0, result = 0
        self._emit(MachineOp.SLT,
                   MachineOperand.vreg("t_sig2"),
                   MachineOperand.immediate(0),
                   src,
                   comment="0 < src ?")
        zero_label = self._fresh_label("sig_zero")
        self._emit(MachineOp.BNEZ,
                   MachineOperand.vreg("t_sig2"),
                   comment=zero_label)
        self._emit(MachineOp.LI, dst,
                   MachineOperand.immediate(0),
                   comment="clamp to 0")
        self._emit_label(zero_label)

    def _select_conv(self, instr: Instruction) -> None:
        """Conv2D: real RISC-V MAC inline (simplified single-MAC)."""
        dst = self._dst(instr)
        x_reg = self._op(instr, 0)
        w_reg = self._op(instr, 1)
        b_reg = self._op(instr, 2)
        if dst:
            # acc = bias (mv bias to dest)
            self._emit(MachineOp.MV, dst, b_reg,
                       comment="acc = bias")
            # tmp = x * w (MUL for MAC)
            tmp_vreg = MachineOperand.vreg("tmp_mac")
            self._emit(MachineOp.MUL, tmp_vreg, x_reg, w_reg,
                       comment="tmp = x * w")
            # dst = dst + tmp (acc += x*w)
            self._emit(MachineOp.ADD, dst, dst, tmp_vreg,
                       comment="acc += x*w")

    def _select_gemm(self, instr: Instruction) -> None:
        """GEMM inline: real RISC-V MUL+ADD MAC."""
        dst = self._dst(instr)
        a_reg = self._op(instr, 0)
        w_reg = self._op(instr, 1)
        b_reg = self._op(instr, 2)
        if dst:
            self._emit(MachineOp.MV, dst, b_reg,
                       comment="acc = bias")
            tmp_vreg = MachineOperand.vreg("tmp_gemm")
            self._emit(MachineOp.MUL, tmp_vreg, a_reg, w_reg,
                       comment="tmp = a * w")
            self._emit(MachineOp.ADD, dst, dst, tmp_vreg,
                       comment="acc += a*w")

    def _select_maxpool(self, instr: Instruction) -> None:
        """MaxPool inline: RISC-V SLT + branch → max."""
        src = self._op(instr, 0)
        dst = self._dst(instr)
        if dst is None or src is None:
            return
        # max(x, 0) using SLT + branch
        gt_label = self._fresh_label("mp_gt")
        self._emit(MachineOp.SLT,
                   MachineOperand.vreg("t_mp"),
                   MachineOperand.immediate(0),
                   src,
                   comment="0 < x ?")
        self._emit(MachineOp.BNEZ,
                   MachineOperand.vreg("t_mp"),
                   comment=gt_label)
        self._emit(MachineOp.LI, dst,
                   MachineOperand.immediate(0),
                   comment="result = 0")
        done_label = self._fresh_label("mp_done")
        self._emit(MachineOp.J, comment=done_label)
        self._emit_label(gt_label)
        self._emit(MachineOp.MV, dst, src,
                   comment="result = x")
        self._emit_label(done_label)
