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
import re
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class AsmInst:
    """Represents one parsed assembly instruction."""

    def __init__(self, raw: str, lineno: int = 0):
        self.raw = raw
        self.lineno = lineno
        self.label: Optional[str] = None
        self.opcode: Optional[str] = None
        self.operands: list[str] = []
        self.comment: Optional[str] = None
        self._parse()

    def _parse(self) -> None:
        """Parse the raw line into components."""
        stripped = self.raw.strip()

        # Empty line or pure comment
        if not stripped or stripped.startswith("#"):
            self.comment = stripped.lstrip("#").strip()
            return

        # Separate code from comment
        code = stripped
        if "#" in stripped:
            idx = stripped.find("#")
            code = stripped[:idx].strip()
            self.comment = stripped[idx + 1:].strip()

        # Check for label
        label_match = re.match(r'^([A-Za-z_.][A-Za-z0-9_.]*):\s*(.*)', code)
        if label_match:
            self.label = label_match.group(1)
            code = label_match.group(2).strip()

        if not code:
            return

        # Extract opcode and operands
        tokens = code.replace(",", " ").split()
        if not tokens:
            return

        self.opcode = tokens[0].lower().lstrip(".")
        self.operands = tokens[1:] if len(tokens) > 1 else []

    def to_asm(self) -> str:
        """Reconstruct the assembly line."""
        parts = []
        if self.label:
            parts.append(f"{self.label}:")
        if self.opcode:
            parts.append(f"  {self.opcode}")
            if self.operands:
                parts.append(" " + ", ".join(self.operands))
        if self.comment:
            parts.append(f"  # {self.comment}")
        result = "".join(parts)
        if not result.strip() and self.raw.strip() == "":
            return ""
        return result

    def __repr__(self) -> str:
        return f"AsmInst({self.opcode}, {self.operands})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_asm(asm_text: str) -> list[AsmInst]:
    """Parse assembly text into AsmInst objects."""
    lines = asm_text.strip().split("\n")
    return [AsmInst(line, lineno=i) for i, line in enumerate(lines)]


def _insts_to_asm(insts: list[AsmInst]) -> str:
    """Convert AsmInst list back to assembly string."""
    return "\n".join(inst.to_asm() for inst in insts)


def _parse_imm(s: str) -> Optional[int]:
    """Parse an immediate value string to int."""
    try:
        s = s.strip()
        if s.startswith("0x") or s.startswith("0X"):
            return int(s, 16)
        return int(s)
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


def _is_clobbered(inst: AsmInst, reg: str) -> bool:
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
        return inst.operands[0] == reg
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

    # --- Pass 1: Merge adjacent lui+addi pairs into li ---
    new_insts: list[AsmInst] = []
    i = 0
    while i < len(insts):
        inst = insts[i]

        # Check for lui followed by addi
        if inst.opcode == "lui" and i + 1 < len(insts):
            next_inst = insts[i + 1]
            if (next_inst.opcode == "addi"
                    and inst.operands and next_inst.operands):
                # Check: rd of lui == rd of addi, and rd == rs1 of addi
                lui_rd = inst.operands[0]
                if (len(next_inst.operands) >= 3
                        and next_inst.operands[0] == lui_rd
                        and next_inst.operands[1] == lui_rd):
                    # Merge
                    imm_hi = (
                        _parse_imm(inst.operands[1])
                        if len(inst.operands) > 1 else None
                    )
                    imm_lo = (
                        _parse_imm(next_inst.operands[2])
                        if len(next_inst.operands) > 2 else None
                    )

                    if imm_hi is not None and imm_lo is not None:
                        # Compute final constant
                        final_val = (imm_hi << 12) + _sign_extend_12(imm_lo)
                        # Replace with li
                        new_inst = AsmInst("")
                        new_inst.opcode = "li"
                        new_inst.operands = [lui_rd, str(final_val)]
                        new_inst.comment = (
                            f"merged lui+addi -> {final_val}"
                        )
                        new_insts.append(new_inst)
                        total_changes += 1
                        i += 2
                        continue

        new_insts.append(inst)
        i += 1

    insts = new_insts

    # --- Pass 2: Eliminate redundant lui ---
    # Track the last upper-immediate value loaded into each register
    # If a new lui loads the same value into the same register (and the
    # register hasn't been clobbered in between), the second lui is redundant.
    new_insts = []
    lui_state: dict[str, Optional[int]] = {}  # reg -> upper imm value

    for inst in insts:
        if inst.opcode == "lui" and inst.operands:
            rd = inst.operands[0]
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
