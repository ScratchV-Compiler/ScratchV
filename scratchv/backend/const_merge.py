"""Constant Load Merge Optimizer for RISC-V.

Detects and merges lui+addi instruction pairs into single li
pseudo-instructions, and eliminates redundant lui instructions
across basic blocks.

Usage::

    from scratchv.backend.const_merge import merge_constants
    optimized_asm, changes = merge_constants(asm_text)
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from scratchv.backend._asm_parser import (
    ParsedAsmLine,
    canonical_reg,
    lines_to_asm,
    parse_asm,
    parse_line,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class AsmInst(ParsedAsmLine):
    """Backward-compatible parsed instruction using the shared parser."""

    def __init__(self, raw: str, lineno: int = 0):
        parsed = parse_line(raw, lineno=lineno)
        super().__init__(
            raw=parsed.raw,
            label=parsed.label,
            opcode=parsed.opcode,
            operands=parsed.operands,
            comment=parsed.comment,
            lineno=parsed.lineno,
            is_directive=parsed.is_directive,
        )

    def __repr__(self) -> str:
        return f"AsmInst({self.opcode}, {self.operands})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_asm(asm_text: str) -> list[ParsedAsmLine]:
    """Compatibility wrapper around the shared assembly parser."""
    return parse_asm(asm_text.strip())


def _insts_to_asm(insts: list[ParsedAsmLine]) -> str:
    """Compatibility wrapper around the shared assembly serializer."""
    return lines_to_asm(insts)


def _parse_imm(s: str) -> Optional[int]:
    """Parse a decimal or prefixed numeric immediate."""
    try:
        text = s.strip()
        try:
            return int(text, 0)
        except ValueError:
            # Python rejects decimal strings such as ``010`` with base 0.
            return int(text, 10)
    except ValueError:
        return None


def _sign_extend_12(val: int) -> int:
    """Sign-extend a 12-bit value."""
    val = val & 0xFFF
    if val & 0x800:
        val -= 0x1000
    return val


def _u20(val: int) -> int:
    """Extract upper 20 bits for LUI, accounting for sign extension of addi."""
    return (val + 0x800) >> 12


def _l12(val: int) -> int:
    """Extract lower 12 bits (sign-extended) for ADDI."""
    return _sign_extend_12(val & 0xFFF)


def _parse_lui_imm(text: str) -> Optional[int]:
    """Parse a representable 20-bit LUI immediate.

    Both signed 20-bit spelling and the unsigned encoded field are accepted.
    Values outside that range are rejected instead of silently truncated.
    """
    value = _parse_imm(text)
    if value is None or not -(1 << 19) <= value <= 0xFFFFF:
        return None
    return value & 0xFFFFF


def _parse_addi_imm(text: str) -> Optional[int]:
    """Parse a 12-bit ADDI immediate in signed or encoded-field spelling."""
    value = _parse_imm(text)
    if value is None or not -(1 << 11) <= value <= 0xFFF:
        return None
    return value


def _signed_rv32(value: int) -> int:
    """Normalize an integer to its signed RV32 representation."""
    value &= 0xFFFFFFFF
    return value if value < 0x80000000 else value - 0x100000000


def _is_separator(line: ParsedAsmLine) -> bool:
    """Return whether a line is only whitespace or a comment."""
    return line.opcode is None and line.label is None


def _merge_lui_addi_once(
    insts: list[ParsedAsmLine],
) -> tuple[list[ParsedAsmLine], int]:
    """Safely merge one scan's eligible LUI+ADDI sequences."""
    result: list[ParsedAsmLine] = []
    changes = 0
    i = 0

    while i < len(insts):
        lui = insts[i]
        if lui.opcode != "lui" or len(lui.operands) != 2:
            result.append(lui)
            i += 1
            continue

        j = i + 1
        while j < len(insts) and _is_separator(insts[j]):
            j += 1
        if j >= len(insts):
            result.append(lui)
            i += 1
            continue

        addi = insts[j]
        if (
            addi.opcode != "addi"
            or addi.label is not None
            or len(addi.operands) != 3
        ):
            result.append(lui)
            i += 1
            continue

        rd = canonical_reg(lui.operands[0])
        if (
            rd != canonical_reg(addi.operands[0])
            or rd != canonical_reg(addi.operands[1])
        ):
            result.append(lui)
            i += 1
            continue

        imm_hi = _parse_lui_imm(lui.operands[1])
        imm_lo = _parse_addi_imm(addi.operands[2])
        if imm_hi is None or imm_lo is None:
            result.append(lui)
            i += 1
            continue

        final_value = _signed_rv32(
            (imm_hi << 12) + _sign_extend_12(imm_lo),
        )
        comments = [comment for comment in (lui.comment, addi.comment) if comment]
        comments.append(f"merged lui+addi -> {final_value}")
        result.append(ParsedAsmLine(
            raw="",
            label=lui.label,
            opcode="li",
            operands=[lui.operands[0], str(final_value)],
            comment="; ".join(comments),
            lineno=lui.lineno,
        ))
        # Comments and blank lines are not instructions, so retain them.
        result.extend(insts[i + 1:j])
        changes += 1
        i = j + 1

    return result, changes


# ---------------------------------------------------------------------------
# Constant merge optimization
# ---------------------------------------------------------------------------

# Standard register names
_STANDARD_REGS = {
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
    "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15",
    "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23",
    "x24", "x25", "x26", "x27", "x28", "x29", "x30", "x31",
    "zero", "ra", "sp", "gp", "tp",
    "t0", "t1", "t2", "t3", "t4", "t5", "t6",
    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8",
    "s9", "s10", "s11",
    "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
    "fp",
}


def _is_reg(s: str) -> bool:
    """Check if a string is a known register name."""
    return s.strip() in _STANDARD_REGS


def _is_clobbered(inst: ParsedAsmLine, reg: str) -> bool:
    """Check if an instruction writes to the given register."""
    if inst.opcode is None:
        return False
    if not inst.operands:
        return False
    # For most instructions, the first operand is the destination
    dst_clobbers = {
        "add", "addi", "sub", "mul", "div", "rem", "sll", "srl", "sra",
        "xor", "or", "and", "slt", "sltu",
        "lui", "li", "mv", "lw", "lh", "lb", "lbu", "lhu",
        "auipc", "jal", "jalr",
        "xori", "ori", "andi", "slli", "srli", "srai",
        "slti", "sltiu",
    }
    if inst.opcode in dst_clobbers:
        return canonical_reg(inst.operands[0]) == canonical_reg(reg)
    # For stores, the first operand is the value (doesn't clobber dest reg)
    # For branches, no destination
    return False


def merge_constants(asm_text: str) -> tuple[str, int]:
    """Merge lui+addi pairs and eliminate redundant lui instructions.

    Parameters
    ----------
    asm_text:
        Input RISC-V assembly text.

    Returns
    -------
    Tuple of (optimized_assembly_string, number_of_changes_made).
    """
    insts = _parse_asm(asm_text)
    total_changes = 0

    # --- Pass 1: Merge safe lui+addi pairs into li ---
    insts, merged = _merge_lui_addi_once(insts)
    total_changes += merged

    # --- Pass 2: Eliminate redundant lui ---
    # Track the last upper-immediate value loaded into each register
    # If a new lui loads the same value into the same register (and the
    # register hasn't been clobbered in between), the second lui is redundant.
    new_insts = []
    lui_state: dict[str, Optional[int]] = {}  # reg -> upper imm value

    for inst in insts:
        if inst.opcode == "lui" and inst.operands:
            rd = canonical_reg(inst.operands[0])
            imm = (
                _parse_imm(inst.operands[1])
                if len(inst.operands) > 1 else None
            )
            if rd in lui_state and lui_state[rd] == imm:
                # Redundant: skip it, add a comment to the next instruction
                total_changes += 1
                # Replace with a comment
                comment_inst = AsmInst("")
                comment_inst.comment = (
                    f"peephole: removed redundant lui {rd}, {imm}"
                )
                new_insts.append(comment_inst)
                continue
            else:
                lui_state[rd] = imm
        else:
            # If this instruction writes to a tracked register, clear tracking
            for reg in list(lui_state.keys()):
                if _is_clobbered(inst, reg):
                    lui_state[reg] = None

        new_insts.append(inst)

    return _insts_to_asm(new_insts), total_changes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RISC-V Constant Load Merge Optimizer",
    )
    parser.add_argument(
        "input", type=str,
        help="Input assembly file (.s)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file (default: stdout)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print optimization statistics to stderr",
    )

    args = parser.parse_args()

    with open(args.input, "r") as f:
        asm_text = f.read()

    result, changes = merge_constants(asm_text)

    if args.verbose:
        print(
            f"Constant merge: {changes} change(s) applied",
            file=sys.stderr,
        )

    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
    else:
        print(result)


if __name__ == "__main__":
    main()
