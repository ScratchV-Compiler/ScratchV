"""Instruction selection: maps IR instructions to RISC-V-like pseudo-instructions.

This phase lowers each IR instruction to a sequence of RISC-V machine instructions,
producing a flat list of MachineInstrs that still use virtual registers.
"""

from __future__ import annotations

from scratchv.ir.types import OpCode, Instruction, BasicBlock, Function, Program
from scratchv.backend.register_alloc import MachineInstr, MachineOp, MachineOperand


class InstructionSelector:
    """Select RISC-V instructions for each IR instruction."""

    def __init__(self, program: Program):
        self.program = program
        self._instructions: list[MachineInstr] = []
        self._label_counter = 0

    def run(self) -> list[MachineInstr]:
        """Select instructions for all functions. Returns flat list of MachineInstrs."""
        self._instructions = []
        for func in self.program.functions:
            self._select_function(func)
        return self._instructions

    def _fresh_label(self, prefix: str = "L") -> str:
        self._label_counter += 1
        return f"{prefix}_{self._label_counter}"

    def _select_function(self, func: Function) -> None:
        # Function prologue label
        self._emit_label(func.name)

        for block in func.blocks:
            self._emit_label(f".{block.name}")
            for instr in block.instructions:
                self._select_instruction(instr)

    def _select_instruction(self, instr: Instruction) -> None:
        handler = getattr(self, f"_select_{instr.opcode.value}", None)
        if handler is None:
            raise ValueError(f"No instruction selection for opcode: {instr.opcode}")
        handler(instr)

    def _emit(self, op: MachineOp, dst=None, src1=None, src2=None, comment: str = "") -> None:
        self._instructions.append(MachineInstr(op, dst, src1, src2, comment))

    def _emit_label(self, name: str) -> None:
        self._instructions.append(MachineInstr(MachineOp.LABEL, comment=name))

    def _op(self, instr: Instruction, idx: int):
        """Get an operand from an IR instruction, converting constants inline."""
        op = instr.operands[idx]
        # Small integers can be encoded as immediate operands
        if op.is_constant and op.const_value is not None:
            return MachineOperand.immediate(int(op.const_value))
        return MachineOperand.vreg(op.name)

    def _dst(self, instr: Instruction):
        if instr.dest is None:
            return None
        return MachineOperand.vreg(instr.dest.name)

    # --- Per-opcode selectors ---

    def _select_load_const(self, instr: Instruction) -> None:
        val = instr.attrs.get("value", 0)
        dst = self._dst(instr)
        # Use LI pseudo-instruction (expanded to addi x0, imm or lui+addi)
        self._emit(MachineOp.LI, dst, MachineOperand.immediate(int(val)), comment=f"const {val}")

    def _select_add(self, instr: Instruction) -> None:
        self._emit(MachineOp.ADD, self._dst(instr), self._op(instr, 0), self._op(instr, 1))

    def _select_sub(self, instr: Instruction) -> None:
        self._emit(MachineOp.SUB, self._dst(instr), self._op(instr, 0), self._op(instr, 1))

    def _select_mul(self, instr: Instruction) -> None:
        self._emit(MachineOp.MUL, self._dst(instr), self._op(instr, 0), self._op(instr, 1))

    def _select_div(self, instr: Instruction) -> None:
        self._emit(MachineOp.DIV, self._dst(instr), self._op(instr, 0), self._op(instr, 1))

    def _select_neg(self, instr: Instruction) -> None:
        # RISC-V: sub rd, x0, rs
        self._emit(MachineOp.SUB, self._dst(instr),
                   MachineOperand.immediate(0), self._op(instr, 0))

    def _select_exp(self, instr: Instruction) -> None:
        # Placeholder: exp is not a native RISC-V instruction.
        # For now, emit a call to an external exp helper.
        dst = self._dst(instr)
        self._emit(MachineOp.CALL, comment="exp")
        if dst:
            self._emit(MachineOp.MV, dst, MachineOperand.vreg("a0"))

    def _select_relu(self, instr: Instruction) -> None:
        """ReLU(x) = max(x, 0).  Use:  max rd, rs, x0"""
        src = self._op(instr, 0)
        dst = self._dst(instr)
        self._emit(MachineOp.MAX, dst, src, MachineOperand.immediate(0))

    def _select_gelu(self, instr: Instruction) -> None:
        # GELU(x) ≈ x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # Simplified: call external gelu helper
        dst = self._dst(instr)
        self._emit(MachineOp.CALL, comment="gelu")
        if dst:
            self._emit(MachineOp.MV, dst, MachineOperand.vreg("a0"))

    def _select_softmax(self, instr: Instruction) -> None:
        dst = self._dst(instr)
        self._emit(MachineOp.CALL, comment="softmax")
        if dst:
            self._emit(MachineOp.MV, dst, MachineOperand.vreg("a0"))

    def _select_maxpool(self, instr: Instruction) -> None:
        self._emit(MachineOp.CALL, comment="maxpool")

    def _select_load(self, instr: Instruction) -> None:
        self._emit(MachineOp.LW, self._dst(instr), self._op(instr, 0))

    def _select_store(self, instr: Instruction) -> None:
        self._emit(MachineOp.SW, self._op(instr, 0), self._op(instr, 1))

    def _select_alloca(self, instr: Instruction) -> None:
        size = instr.attrs.get("size", 4)
        dst = self._dst(instr)
        # Subtract from sp to allocate
        self._emit(MachineOp.ADDI, dst, MachineOperand.vreg("sp"),
                   MachineOperand.immediate(-size), comment=f"alloca {size}")

    def _select_for(self, instr: Instruction) -> None:
        """Begin a for loop: set up loop variable and branch to loop header."""
        iv = self._dst(instr)
        start = instr.attrs.get("start", 0)
        end = instr.attrs.get("end", 0)

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
        end_val = MachineOperand.immediate(end)
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
        self._emit(MachineOp.J, comment=instr.target or "")

    def _select_br_if(self, instr: Instruction) -> None:
        cond = self._op(instr, 0)
        targets = (instr.target or ",").split(",")
        true_target = targets[0] if len(targets) > 0 else ""
        false_target = targets[1] if len(targets) > 1 else ""

        # bnez cond, true_label; j false_label
        self._emit(MachineOp.BNEZ, cond, comment=true_target)
        self._emit(MachineOp.J, comment=false_target)

    def _select_return(self, instr: Instruction) -> None:
        if instr.operands:
            self._emit(MachineOp.MV, MachineOperand.vreg("a0"),
                       self._op(instr, 0), comment="return value")
        self._emit(MachineOp.JALR, MachineOperand.vreg("zero"),
                   MachineOperand.vreg("ra"), comment="ret")

    def _select_matmul(self, instr: Instruction) -> None:
        a = self._op(instr, 0)
        b = self._op(instr, 1)
        m = instr.attrs.get("m", 1)
        n = instr.attrs.get("n", 1)
        k = instr.attrs.get("k", 1)
        dst = self._dst(instr)

        # Generate nested loops: for i in range(m): for j in range(n): sum += a[i,k] * b[k,j]
        # Allocate temp for sum
        sum_reg = MachineOperand.vreg("matmul_sum")
        self._emit(MachineOp.LI, sum_reg, MachineOperand.immediate(0))

        # We emit a call to a runtime matmul helper for now
        self._emit(MachineOp.CALL, comment=f"matmul m={m} n={n} k={k}")
        if dst:
            self._emit(MachineOp.MV, dst, MachineOperand.vreg("a0"))

    def _select_dot(self, instr: Instruction) -> None:
        a = self._op(instr, 0)
        b = self._op(instr, 1)
        length = instr.attrs.get("length", 1)
        dst = self._dst(instr)

        self._emit(MachineOp.CALL, comment=f"dot len={length}")
        if dst:
            self._emit(MachineOp.MV, dst, MachineOperand.vreg("a0"))

    def _select_label(self, instr: Instruction) -> None:
        self._emit_label(instr.target or "")
