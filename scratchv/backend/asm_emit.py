"""Assembly emitter: converts MachineInstr list to RISC-V assembly text.

Produces GNU Assembler (GAS) syntax suitable for ``riscv64-unknown-elf-gcc``
or ``riscv64-linux-gnu-gcc``.
"""

from __future__ import annotations

from scratchv.backend.register_alloc import (
    MachineInstr, MachineOp, MachineOperand,
)


# RW───RV32IM pseudo-instruction expansion ────────────────────────────────────

_OP_NAMES = {
    MachineOp.ADD: "add",
    MachineOp.ADDI: "addi",
    MachineOp.SUB: "sub",
    MachineOp.MUL: "mul",
    MachineOp.DIV: "div",
    MachineOp.MAX: "max",
    MachineOp.LW: "lw",
    MachineOp.SW: "sw",
    MachineOp.J: "j",
    MachineOp.JAL: "jal",
    MachineOp.JALR: "jalr",
    MachineOp.BEQ: "beq",
    MachineOp.BNE: "bne",
    MachineOp.BLT: "blt",
    MachineOp.BGE: "bge",
    MachineOp.BNEZ: "bnez",
    MachineOp.LI: "li",
    MachineOp.MV: "mv",
    MachineOp.CALL: "call",
}


def _fmt_op(op: MachineOperand | None) -> str:
    if op is None:
        return ""
    # strip % from vreg names since we've resolved them
    return str(op).lstrip("%")


class AsmEmitter:
    """Emit RISC-V assembly text from machine instructions."""

    def __init__(self, instructions: list[MachineInstr]):
        self.instructions = instructions

    def emit(self) -> str:
        """Produce a complete assembly source string."""
        lines = [
            ".text",
            ".align 2",
        ]

        in_function = False
        for instr in self.instructions:
            if instr.op == MachineOp.LABEL:
                label = instr.comment
                if not label.startswith("."):  # function label
                    if in_function:
                        lines.append(f"  .size {label}, .-{label}")
                    lines.extend([
                        f"  .globl {label}",
                        f"  .type {label}, @function",
                        f"{label}:",
                    ])
                    in_function = True
                else:
                    lines.append(f"{label}:")
            elif instr.op == MachineOp.SECTION:
                lines.append(f"  .section .{instr.comment}")
            else:
                lines.append(f"  {self._format_instr(instr)}")

        if in_function and self.instructions:
            last_label = None
            for instr in reversed(self.instructions):
                if instr.op == MachineOp.LABEL:
                    last_label = instr.comment
                    break
            if last_label and not last_label.startswith("."):
                lines.append(f"  .size {last_label}, .-{last_label}")

        lines.append("")
        return "\n".join(lines)

    def _format_instr(self, instr: MachineInstr) -> str:
        op_name = _OP_NAMES.get(instr.op)
        if op_name is None:
            return f"  # {instr.op.value} {instr.comment}".strip()

        parts = [f"  {op_name}"]

        # Format operands
        operands = []
        for op in (instr.dst, instr.src1, instr.src2):
            if op is not None:
                operands.append(_fmt_op(op))

        if operands:
            parts.append(" " + ", ".join(operands))

        if instr.comment:
            parts.append(f"  # {instr.comment}")

        return "".join(parts)
