"""Shared assembly parsing utilities for RISC-V backend tools.

Provides a unified ``ParsedAsmLine`` dataclass and parsing functions
used by all assembly-level passes (beautifier, peephole, const-merge,
inst-counter, inst-scheduler).  Eliminates the duplicated regex-based
line parsers that previously existed in each tool.

Usage::

    from scratchv.backend._asm_parser import ParsedAsmLine, parse_asm, lines_to_asm

    parsed = parse_asm(asm_text)
    for line in parsed:
        if line.opcode == "add":
            ...
    output = lines_to_asm(parsed)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# ParsedAsmLine — unified assembly line representation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParsedAsmLine:
    """Represents one parsed line of RISC-V assembly.

    Attributes:
        raw:        Original line text (preserved verbatim).
        label:      Label name without colon, or None.
        opcode:     Instruction mnemonic (lowercased, without leading dot),
                    or None for labels / empty lines.
        operands:   List of operand strings (split by comma, parens respected).
        comment:    Trailing comment text (without leading ``#``), or None.
        lineno:     0-based line index in the source.
        is_directive: True if opcode started with ``.`` (e.g. ``.text``).
    """

    raw: str
    label: Optional[str] = None
    opcode: Optional[str] = None
    operands: list[str] = field(default_factory=list)
    comment: Optional[str] = None
    lineno: int = 0
    is_directive: bool = False

    def __str__(self) -> str:
        return self.to_asm()

    def to_asm(self) -> str:
        """Reconstruct the assembly line."""
        # Empty lines
        if self.opcode is None and self.label is None and not self.raw.strip():
            return ""
        # Comment-only lines
        if self.opcode is None and self.label is None and self.comment:
            return self.raw
        # Label-only lines
        if self.label is not None and self.opcode is None and not self.operands:
            return f"{self.label}:"
        # Instruction lines (possibly with label)
        parts: list[str] = []
        if self.label:
            parts.append(f"{self.label}:")
        if self.opcode:
            op_str = f".{self.opcode}" if self.is_directive else self.opcode
            parts.append(f"  {op_str}")
            if self.operands:
                parts.append(" " + ", ".join(self.operands))
        if self.comment:
            parts.append(f"  # {self.comment}")
        return "".join(parts)

    @property
    def is_empty(self) -> bool:
        """True if this line is empty or whitespace-only."""
        return (
            self.opcode is None
            and self.label is None
            and not self.raw.strip()
        )

    @property
    def is_label_only(self) -> bool:
        """True if this line is a bare label (no instruction)."""
        return (
            self.label is not None
            and self.opcode is None
            and not self.operands
        )

    @property
    def is_comment_only(self) -> bool:
        """True if this line is a comment-only line."""
        stripped = self.raw.strip()
        return bool(stripped.startswith("#"))


# ═══════════════════════════════════════════════════════════════════════════════
# Line-parsing regex
# ═══════════════════════════════════════════════════════════════════════════════

_LINE_RE = re.compile(
    r'^\s*'
    r'(?P<label>[A-Za-z_.][A-Za-z0-9_.]*:)?\s*'
    r'(?P<opcode>\.?\w[\w.]*)?\s*'
    r'(?P<operands>[^#]*?)'
    r'(?:\s*#\s*(?P<comment>.*))?'
    r'$'
)

# Assembler directives (their raw form includes the leading dot)
_DIRECTIVES: set[str] = {
    ".text", ".data", ".bss", ".rodata", ".section",
    ".globl", ".global", ".type", ".size", ".align",
    ".file", ".loc", ".cfi_startproc", ".cfi_endproc",
    ".cfi_def_cfa", ".cfi_offset", ".cfi_restore",
    ".byte", ".word", ".dword", ".half", ".quad",
    ".string", ".asciz", ".ascii", ".zero", ".skip",
    ".balign", ".p2align", ".option", ".set",
}

# Opcodes that do NOT write to their first operand (stores, branches, jumps)
_NON_DEF_OPCODES: set[str] = {
    "sw", "sh", "sb", "fsd", "fsw",
    "beq", "bne", "blt", "bge", "bltu", "bgeu",
    "beqz", "bnez", "blez", "bgtz", "bltz", "bgez",
    "j", "jal", "jalr", "ret", "jr",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Parsing
# ═══════════════════════════════════════════════════════════════════════════════

def _split_operands(s: str) -> list[str]:
    """Split operand string by comma, respecting parenthesised groups.

    Example: ``"x1, 0(x2)"`` → ``["x1", "0(x2)"]``
    """
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in s:
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
    if current or not parts:
        parts.append("".join(current).strip())
    return [p for p in parts if p]


def parse_line(line: str, lineno: int = 0) -> ParsedAsmLine:
    """Parse a single assembly line into a ``ParsedAsmLine`` object.

    Args:
        line: Raw assembly text line.
        lineno: 0-based line index (for error reporting).

    Returns:
        A ParsedAsmLine with all fields populated.
    """
    m = _LINE_RE.match(line)
    if m is None:
        return ParsedAsmLine(raw=line, lineno=lineno)

    # Label
    label = m.group("label")
    if label is not None:
        label = label.rstrip(":")

    # Opcode
    opcode_raw = m.group("opcode")
    opcode: Optional[str] = None
    is_directive = False
    if opcode_raw is not None:
        opcode_raw = opcode_raw.strip()
        if opcode_raw.startswith("."):
            is_directive = True
            opcode = opcode_raw.lstrip(".").lower()
        else:
            opcode = opcode_raw.lower()

    # Operands
    operands_raw = (m.group("operands") or "").strip()
    operands = _split_operands(operands_raw)

    # Comment
    comment = m.group("comment")
    if comment is not None:
        comment = comment.strip()

    return ParsedAsmLine(
        raw=line,
        label=label,
        opcode=opcode,
        operands=operands,
        comment=comment,
        lineno=lineno,
        is_directive=is_directive,
    )


def parse_asm(asm_text: str) -> list[ParsedAsmLine]:
    """Parse full assembly text into a list of ``ParsedAsmLine`` objects.

    Args:
        asm_text: Raw RISC-V assembly source.

    Returns:
        List of ParsedAsmLine, one per source line (including blanks).
    """
    lines = asm_text.split("\n")
    return [parse_line(line, lineno=i) for i, line in enumerate(lines)]


def lines_to_asm(lines: list[ParsedAsmLine]) -> str:
    """Convert a list of ``ParsedAsmLine`` back to assembly text.

    Args:
        lines: Parsed assembly lines.

    Returns:
        Reconstructed assembly source string.
    """
    return "\n".join(line.to_asm() for line in lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Operand classification helpers
# ═══════════════════════════════════════════════════════════════════════════════

def classify_def_use(line: ParsedAsmLine) -> tuple[set[str], set[str]]:
    """Classify operands of an instruction into defines and uses.

    For most RISC-V instructions the first operand is the destination
    (def), and remaining operands are sources (uses).  Stores, branches,
    and jumps are exceptions.

    Args:
        line: A parsed assembly line with an opcode.

    Returns:
        Tuple of ``(defines, uses)`` sets of register names.
    """
    defines: set[str] = set()
    uses: set[str] = set()

    if line.opcode is None or line.is_directive:
        return defines, uses

    for i, op in enumerate(line.operands):
        # Extract base register from memory operands like ``16(sp)``
        m = re.match(r'-?\d+\((\w+)\)', op)
        if m:
            uses.add(m.group(1))
            continue

        # Register-like operands
        if _looks_like_reg(op):
            if i == 0 and line.opcode not in _NON_DEF_OPCODES:
                defines.add(op)
            else:
                uses.add(op)

    return defines, uses


def _looks_like_reg(s: str) -> bool:
    """Heuristic: does the string look like a RISC-V register name?"""
    if not s:
        return False
    # ABI names
    if s in ("zero", "ra", "sp", "gp", "tp", "fp"):
        return True
    # x0–x31
    if re.match(r'^x([0-9]|[12][0-9]|3[01])$', s):
        return True
    # a0–a7, t0–t6, s0–s11
    if re.match(r'^[ats]([0-9]|1[01])$', s):
        return True
    # f0–f31
    if re.match(r'^f([0-9]|[12][0-9]|3[01])$', s):
        return True
    return False


def extract_offset(op: str) -> str:
    """Extract offset from a memory operand like ``'16(sp)'``."""
    m = re.match(r'^(-?\d+)\(', op)
    if m:
        return m.group(1)
    return op.lstrip("(").rstrip(")")


def extract_base_reg(op: str) -> str:
    """Extract base register from a memory operand like ``'16(sp)'``."""
    m = re.search(r'\((\w+)\)', op)
    if m:
        return m.group(1)
    return ""
