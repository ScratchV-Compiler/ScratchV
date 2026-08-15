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
from dataclasses import dataclass
from typing import Optional

from scratchv.backend._asm_parser import (
    ParsedAsmLine,
    canonical_reg,
    classify_def_use,
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

@dataclass
class ConstantMergeStats:
    """Categorized results from one constant-merge optimization run."""

    candidate_pairs: int = 0
    merged_pairs: int = 0
    redundant_lui_removed: int = 0
    iterations: int = 0

    @property
    def total_changes(self) -> int:
        """Return all transformations while preserving the legacy count."""
        return self.merged_pairs + self.redundant_lui_removed

_CONTROL_FLOW_OPCODES = {
    "beq", "bne", "blt", "bge", "bltu", "bgeu",
    "beqz", "bnez", "blez", "bgtz", "bltz", "bgez",
    "j", "jr", "jal", "jalr", "call", "tail", "ret",
}

_KNOWN_OPCODES = {
    "add", "addi", "sub", "mul", "div", "divu", "rem", "remu",
    "sll", "slli", "srl", "srli", "sra", "srai",
    "xor", "xori", "or", "ori", "and", "andi",
    "slt", "slti", "sltu", "sltiu",
    "lui", "auipc", "li", "mv", "neg", "not", "seqz", "snez",
    "lw", "lh", "lb", "lbu", "lhu", "sw", "sh", "sb", "nop",
    "max", "min", "maxu", "minu",
} | _CONTROL_FLOW_OPCODES


def _count_merge_candidates(insts: list[ParsedAsmLine]) -> int:
    """Count initially mergeable LUI+ADDI sequences."""
    candidates = 0
    for i, lui in enumerate(insts):
        if lui.opcode != "lui" or len(lui.operands) != 2:
            continue
        j = i + 1
        while j < len(insts) and _is_separator(insts[j]):
            j += 1
        if j >= len(insts):
            continue
        addi = insts[j]
        if (
            addi.opcode != "addi"
            or addi.label is not None
            or len(addi.operands) != 3
        ):
            continue
        rd = canonical_reg(lui.operands[0])
        if (
            rd == canonical_reg(addi.operands[0])
            and rd == canonical_reg(addi.operands[1])
            and _parse_lui_imm(lui.operands[1]) is not None
            and _parse_addi_imm(addi.operands[2]) is not None
        ):
            candidates += 1
    return candidates


def _remove_redundant_lui_once(
    insts: list[ParsedAsmLine],
) -> tuple[list[ParsedAsmLine], int]:
    """Delete provably redundant LUI instructions within basic blocks."""
    result: list[ParsedAsmLine] = []
    lui_state: dict[str, int] = {}
    changes = 0

    for inst in insts:
        # A label starts a new basic block, including ``label: instruction``.
        if inst.label is not None:
            lui_state.clear()

        if inst.opcode is None:
            result.append(inst)
            continue

        # Directives can change sections or assembler state.  Do not carry
        # register facts through a directive whose semantics are not modeled.
        if inst.is_directive:
            lui_state.clear()
            result.append(inst)
            continue

        opcode = inst.opcode
        if opcode == "lui" and len(inst.operands) == 2:
            rd = canonical_reg(inst.operands[0])
            imm = _parse_lui_imm(inst.operands[1])
            if imm is not None:
                if lui_state.get(rd) == imm:
                    comment = (
                        f"peephole: removed redundant lui "
                        f"{inst.operands[0]}, {inst.operands[1]}"
                    )
                    if inst.comment:
                        comment += f"; {inst.comment}"
                    result.append(ParsedAsmLine(
                        raw=f"  # {comment}",
                        comment=comment,
                        lineno=inst.lineno,
                    ))
                    changes += 1
                    # The earlier LUI still defines the same value, so state
                    # deliberately remains unchanged after deleting this one.
                    continue
                lui_state[rd] = imm
                result.append(inst)
                continue

        if opcode in _CONTROL_FLOW_OPCODES:
            lui_state.clear()
        elif opcode not in _KNOWN_OPCODES:
            # An unknown instruction may write any register.  Clearing all
            # facts loses an optimization opportunity but preserves safety.
            lui_state.clear()
        else:
            defines, _ = classify_def_use(inst)
            for reg in defines:
                lui_state.pop(canonical_reg(reg), None)

        result.append(inst)

    return result, changes


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
    optimized, stats = merge_constants_detailed(asm_text)
    return optimized, stats.total_changes


def merge_constants_detailed(
    asm_text: str,
    *,
    max_iterations: Optional[int] = None,
) -> tuple[str, ConstantMergeStats]:
    """Optimize assembly to a fixed point and return detailed statistics.

    Redundant LUI removal runs before pair merging so deleting a duplicate can
    expose the earlier LUI to a following ADDI in the same iteration.
    """
    insts = _parse_asm(asm_text)
    stats = ConstantMergeStats(
        candidate_pairs=_count_merge_candidates(insts),
    )
    limit = max_iterations if max_iterations is not None else max(1, len(insts))

    for _ in range(max(0, limit)):
        insts, removed = _remove_redundant_lui_once(insts)
        insts, merged = _merge_lui_addi_once(insts)
        stats.iterations += 1
        stats.redundant_lui_removed += removed
        stats.merged_pairs += merged
        if removed == 0 and merged == 0:
            break

    return _insts_to_asm(insts), stats


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
