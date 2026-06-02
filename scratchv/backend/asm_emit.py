"""Assembly emitter: converts MachineInstr list to RISC-V assembly text.

Produces GNU Assembler (GAS) syntax suitable for ``riscv64-unknown-elf-gcc``
or ``riscv64-linux-gnu-gcc``.
"""

from __future__ import annotations

from scratchv.backend.machine_types import (
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
    MachineOp.SRAI: "srai",
    MachineOp.XOR: "xor",
    MachineOp.AND: "and",
    MachineOp.SLT: "slt",
    MachineOp.REM: "rem",
    MachineOp.LW: "lw",
    MachineOp.SW: "sw",
    MachineOp.FLD: "fld",
    MachineOp.FSD: "fsd",
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
    MachineOp.SQRT_S: "fsqrt.s",
    MachineOp.SQRT_D: "fsqrt.d",
    MachineOp.FMIN_D: "fmin.d",
    MachineOp.FMAX_D: "fmax.d",
    MachineOp.FABS_D: "fabs.d",
    MachineOp.FNEG_D: "fneg.d",
    MachineOp.FADD_D: "fadd.d",
    MachineOp.FSUB_D: "fsub.d",
    MachineOp.FMUL_D: "fmul.d",
    MachineOp.FDIV_D: "fdiv.d",
    MachineOp.FLT_D: "flt.d",
    MachineOp.FEQ_D: "feq.d",
    MachineOp.FCVT_S_D: "fcvt.s.d",
    MachineOp.FCVT_D_S: "fcvt.d.s",
    MachineOp.LI_D: "li.d",
    MachineOp.FADD_S: "fadd.s",
    MachineOp.FSUB_S: "fsub.s",
    MachineOp.FMUL_S: "fmul.s",
    MachineOp.FDIV_S: "fdiv.s",
    MachineOp.FMAX_S: "fmax.s",
    MachineOp.FMIN_S: "fmin.s",
    MachineOp.FLE_S: "fle.s",
    MachineOp.FLT_S: "flt.s",
    MachineOp.FEQ_S: "feq.s",
    MachineOp.FSQRT_S: "fsqrt.s",
    MachineOp.FLW: "flw",
    MachineOp.FSW: "fsw",
    MachineOp.FMV_S: "fmv.s",
    MachineOp.FMV_S_X: "fmv.s.x",
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

        # Branch/jump/call use comment as target label
        if instr.op in (MachineOp.CALL, MachineOp.J, MachineOp.JAL,
                        MachineOp.BNEZ, MachineOp.BEQ, MachineOp.BNE,
                        MachineOp.BLT, MachineOp.BGE) and instr.comment:
            operands = []
            for op in (instr.dst, instr.src1, instr.src2):
                if op is not None:
                    operands.append(_fmt_op(op))
            if operands:
                return f"  {op_name} {', '.join(operands)}, {instr.comment}"
            else:
                return f"  {op_name} {instr.comment}"

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
