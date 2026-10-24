"""Assembly emitter: converts MachineInstr list to RISC-V assembly text.

Produces GNU Assembler (GAS) syntax suitable for ``riscv64-unknown-elf-gcc``
or ``riscv64-linux-gnu-gcc``.
"""

from __future__ import annotations

import re

from scratchv.backend.machine_types import (
    FloatConstant, MachineInstr, MachineOp, MachineOperand,
)
from scratchv.backend.symbols import (
    label_reference, validate_call_symbol, validate_function_symbol, validate_label_symbol,
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
    MachineOp.SLTI: "slti",
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
    MachineOp.LA: "la",
    MachineOp.MV: "mv",
    MachineOp.CALL: "call",
    MachineOp.SQRT_S: "fsqrt.s",
    MachineOp.SQRT_D: "fsqrt.d",
    MachineOp.FMIN_D: "fmin.d",
    MachineOp.FMAX_D: "fmax.d",
    MachineOp.FABS_D: "fabs.d",
    MachineOp.FNEG_D: "fneg.d",
    MachineOp.FNEG_S: "fneg.s",
    MachineOp.FADD_D: "fadd.d",
    MachineOp.FSUB_D: "fsub.d",
    MachineOp.FMUL_D: "fmul.d",
    MachineOp.FDIV_D: "fdiv.d",
    MachineOp.FLT_D: "flt.d",
    MachineOp.FLE_D: "fle.d",
    MachineOp.FEQ_D: "feq.d",
    MachineOp.FCVT_S_D: "fcvt.s.d",
    MachineOp.FCVT_D_S: "fcvt.d.s",
    MachineOp.FADD_S: "fadd.s",
    MachineOp.FSUB_S: "fsub.s",
    MachineOp.FMUL_S: "fmul.s",
    MachineOp.FDIV_S: "fdiv.s",
    MachineOp.FMAX_S: "fmax.s",
    MachineOp.FMIN_S: "fmin.s",
    MachineOp.FABS_S: "fabs.s",
    MachineOp.FLE_S: "fle.s",
    MachineOp.FLT_S: "flt.s",
    MachineOp.FEQ_S: "feq.s",
    MachineOp.FSQRT_S: "fsqrt.s",
    MachineOp.FLW: "flw",
    MachineOp.FSW: "fsw",
    MachineOp.FMV_S: "fmv.s",
    MachineOp.FMV_D: "fmv.d",
    MachineOp.FMV_S_X: "fmv.s.x",
}


def _fmt_op(op: MachineOperand | None) -> str:
    if op is None:
        return ""
    if op.kind == "symbol":
        return label_reference(str(op.value))
    # strip % from vreg names since we've resolved them
    return str(op).lstrip("%")


def _memory_address(op: MachineOperand) -> str:
    address = _fmt_op(op)
    legacy = re.fullmatch(r"([A-Za-z][A-Za-z0-9]*)\((-?\d+)\)", address)
    if legacy:
        return f"{legacy.group(2)}({legacy.group(1)})"
    if "(" not in address:
        return f"0({address})"
    return address


def _comment_suffix(instr: MachineInstr) -> str:
    return f"  # {instr.comment}" if instr.comment else ""


class AsmEmitter:
    """Emit RISC-V assembly text from machine instructions."""

    def __init__(
        self,
        instructions: list[MachineInstr],
        constant_pool: tuple[FloatConstant, ...] | list[FloatConstant] = (),
    ):
        self.instructions = instructions
        self.constant_pool = tuple(constant_pool)

    def emit(self) -> str:
        """Produce a complete assembly source string."""
        lines: list[str] = []
        section_symbols = {".text", *(
            f".{instr.comment}" for instr in self.instructions
            if instr.op is MachineOp.SECTION
        )}
        if self.constant_pool:
            section_symbols.add(".rodata")
        for label in [*(entry.label for entry in self.constant_pool), *(
            instr.target if instr.target is not None else instr.comment
            for instr in self.instructions if instr.op is MachineOp.LABEL
        )]:
            validate_label_symbol(label, section_symbols)
        if self.constant_pool:
            lines.append(".section .rodata")
            for entry in self.constant_pool:
                alignment = 2 if entry.bit_width == 32 else 3
                directive = ".word" if entry.bit_width == 32 else ".quad"
                digits = entry.bit_width // 4
                lines.extend([
                    f".p2align {alignment}",
                    f"{entry.label}:",
                    f"  {directive} 0x{entry.bit_pattern:0{digits}x}",
                ])
        lines.extend([".text", ".align 2"])

        function_name = None
        current_section = ".text"
        closed_function = None
        for instr in self.instructions:
            if instr.op == MachineOp.LABEL:
                label = instr.target if instr.target is not None else instr.comment
                is_function = (instr.label_role == "function" if instr.label_role is not None
                               else not label.startswith("."))
                if is_function:
                    validate_function_symbol(label)
                    if function_name:
                        lines.append(f"  .size {function_name}, .-{function_name}")
                    lines.extend([
                        f"  .globl {label}",
                        f"  .type {label}, @function",
                        f"{label}:",
                    ])
                    function_name = label
                    closed_function = None
                else:
                    if closed_function:
                        raise ValueError(f"function {closed_function!r} cannot continue across output sections")
                    lines.append(f"{label}:")
            elif instr.op == MachineOp.SECTION:
                section = f".{instr.comment}"
                if section != current_section and function_name:
                    # Compute size before switching away from the function's
                    # section. ABI prologue/epilogue insertion stays inside it.
                    lines.append(f"  .size {function_name}, .-{function_name}")
                    closed_function, function_name = function_name, None
                lines.append(f"  .section {section}")
                current_section = section
            else:
                if closed_function:
                    raise ValueError(f"function {closed_function!r} cannot continue across output sections")
                lines.append(f"  {self._format_instr(instr)}")

        if function_name:
            lines.append(f"  .size {function_name}, .-{function_name}")

        lines.append("")
        return "\n".join(lines)

    def _format_instr(self, instr: MachineInstr) -> str:
        if instr.op is MachineOp.LI_D:
            raise ValueError(
                "li.d is not a standard RISC-V instruction; use the "
                "bit-exact floating constant pool"
            )
        op_name = _OP_NAMES.get(instr.op)
        if op_name is None:
            raise ValueError(f"unsupported assembly opcode: {instr.op.value}")

        # Branch/jump/call target: structured target, fallback to comment
        target = instr.target if instr.target is not None else instr.comment
        if instr.op is MachineOp.CALL:
            validate_call_symbol(target)
        elif instr.op in (MachineOp.J, MachineOp.JAL, MachineOp.BNEZ,
                          MachineOp.BEQ, MachineOp.BNE, MachineOp.BLT, MachineOp.BGE) and target:
            target = label_reference(target)

        if instr.op in (MachineOp.LW, MachineOp.FLW, MachineOp.FLD) and instr.dst and instr.src1:
            address = _memory_address(instr.src1)
            return (
                f"  {op_name} {_fmt_op(instr.dst)}, {address}"
                f"{_comment_suffix(instr)}"
            )

        if instr.op in (MachineOp.SW, MachineOp.FSW, MachineOp.FSD) and instr.dst and instr.src1:
            address = _memory_address(instr.src1)
            return (
                f"  {op_name} {_fmt_op(instr.dst)}, {address}"
                f"{_comment_suffix(instr)}"
            )

        # Branch/jump/call use the structured target resolved above.
        if instr.op in (MachineOp.CALL, MachineOp.J, MachineOp.JAL,
                        MachineOp.BNEZ, MachineOp.BEQ, MachineOp.BNE,
                        MachineOp.BLT, MachineOp.BGE) and target:
            operands = []
            for op in (instr.dst, instr.src1, instr.src2):
                if op is not None:
                    operands.append(_fmt_op(op))
            if operands:
                return f"  {op_name} {', '.join(operands)}, {target}"
            else:
                return f"  {op_name} {target}"

        parts = [f"  {op_name}"]

        # Format operands
        operands = []
        for op in (instr.dst, instr.src1, instr.src2):
            if op is not None:
                operands.append(_fmt_op(op))
        if instr.rounding_mode:
            operands.append(instr.rounding_mode)

        if operands:
            parts.append(" " + ", ".join(operands))

        if instr.comment:
            parts.append(f"  # {instr.comment}")

        return "".join(parts)
