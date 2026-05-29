"""RISC-V Assembly Beautifier.

Parses RISC-V assembly text and outputs a formatted, aligned version with
semantic comments and section headers for improved readability.

Usage as module::

    from scratchv.backend.asm_beautifier import beautify_asm
    pretty = beautify_asm(raw_asm)

Usage as CLI::

    python -m scratchv.backend.asm_beautifier input.s -o output.s
"""

from __future__ import annotations

import argparse
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Instruction comment templates
# ---------------------------------------------------------------------------

_RV_REG_NAMES: dict[str, str] = {
    "x0": "zero", "x1": "ra", "x2": "sp", "x3": "gp",
    "x4": "tp", "x5": "t0", "x6": "t1", "x7": "t2",
    "x8": "s0/fp", "x9": "s1", "x10": "a0", "x11": "a1",
    "x12": "a2", "x13": "a3", "x14": "a4", "x15": "a5",
    "x16": "a6", "x17": "a7", "x18": "s2", "x19": "s3",
    "x20": "s4", "x21": "s5", "x22": "s6", "x23": "s7",
    "x24": "s8", "x25": "s9", "x26": "s10", "x27": "s11",
    "x28": "t3", "x29": "t4", "x30": "t5", "x31": "t6",
}


def _anon_reg(r: str) -> str:
    """Return ABI name for a register string like 'x5' or 't0'."""
    r = r.strip().lstrip("%")
    return _RV_REG_NAMES.get(r, r)


# Mapping: instruction mnemonic -> comment template
# {rd}, {rs1}, {rs2}, {imm} are replaced at format time.
_INST_COMMENTS: dict[str, str] = {
    # Integer arithmetic
    "add":   "{rd} = {rs1} + {rs2}",
    "sub":   "{rd} = {rs1} - {rs2}",
    "addi":  "{rd} = {rs1} + {imm}",
    "slli":  "{rd} = {rs1} << {imm}",
    "srli":  "{rd} = {rs1} >> {imm} (logical)",
    "srai":  "{rd} = {rs1} >> {imm} (arithmetic)",
    "sll":   "{rd} = {rs1} << {rs2}",
    "srl":   "{rd} = {rs1} >> {rs2} (logical)",
    "sra":   "{rd} = {rs1} >> {rs2} (arithmetic)",
    "mul":   "{rd} = {rs1} * {rs2}",
    "div":   "{rd} = {rs1} / {rs2}",
    "rem":   "{rd} = {rs1} % {rs2}",
    "xor":   "{rd} = {rs1} XOR {rs2}",
    "or":    "{rd} = {rs1} | {rs2}",
    "and":   "{rd} = {rs1} & {rs2}",
    "xori":  "{rd} = {rs1} XOR {imm}",
    "ori":   "{rd} = {rs1} | {imm}",
    "andi":  "{rd} = {rs1} & {imm}",
    "slt":   "{rd} = ({rs1} < {rs2}) ? 1 : 0",
    "sltu":  "{rd} = ({rs1} < {rs2}) ? 1 : 0 (unsigned)",
    "slti":  "{rd} = ({rs1} < {imm}) ? 1 : 0",
    "sltiu": "{rd} = ({rs1} < {imm}) ? 1 : 0 (unsigned)",
    # Memory
    "lw":    "{rd} = MEM[{rs1} + {imm}]",
    "lh":    "{rd} = MEM16[{rs1} + {imm}]",
    "lb":    "{rd} = MEM8[{rs1} + {imm}]",
    "lbu":   "{rd} = MEM8[{rs1} + {imm}] (unsigned)",
    "lhu":   "{rd} = MEM16[{rs1} + {imm}] (unsigned)",
    "sw":    "MEM[{rs1} + {imm}] = {rd}",
    "sh":    "MEM16[{rs1} + {imm}] = {rd} (low 16b)",
    "sb":    "MEM8[{rs1} + {imm}] = {rd} (low 8b)",
    # Upper immediate
    "lui":   "{rd} = {imm} << 12",
    "auipc": "{rd} = PC + ({imm} << 12)",
    # Branches
    "beq":   "if {rs1} == {rs2} goto {rd}",
    "bne":   "if {rs1} != {rs2} goto {rd}",
    "blt":   "if {rs1} < {rs2} goto {rd}",
    "bge":   "if {rs1} >= {rs2} goto {rd}",
    "bltu":  "if {rs1} < {rs2} goto {rd} (unsigned)",
    "bgeu":  "if {rs1} >= {rs2} goto {rd} (unsigned)",
    # Jumps
    "j":     "goto {rd}",
    "jal":   "{rd} = PC+4; goto {imm}",
    "jalr":  "{rd} = PC+4; goto {rs1}+{imm}",
    # Pseudo-instructions
    "li":    "{rd} = {imm}",
    "mv":    "{rd} = {rs1}",
    "not":   "{rd} = ~{rs1}",
    "neg":   "{rd} = -{rs1}",
    "seqz":  "{rd} = ({rs1} == 0) ? 1 : 0",
    "snez":  "{rd} = ({rs1} != 0) ? 1 : 0",
    "bnez":  "if {rs1} != 0 goto {rd}",
    "beqz":  "if {rs1} == 0 goto {rd}",
    "call":  "call {rd}",
    "ret":   "return",
    "nop":   "no operation",
    # M-extension
    "mulh":  "{rd} = ({rs1} * {rs2})[63:32]",
    "divu":  "{rd} = {rs1} / {rs2} (unsigned)",
    "remu":  "{rd} = {rs1} % {rs2} (unsigned)",
    # F/D extensions
    "fadd.s": "{rd} = {rs1} + {rs2} (f32)",
    "fsub.s": "{rd} = {rs1} - {rs2} (f32)",
    "fmul.s": "{rd} = {rs1} * {rs2} (f32)",
    "fdiv.s": "{rd} = {rs1} / {rs2} (f32)",
    "flw":    "{rd} = MEM[{rs1} + {imm}] (f32)",
    "fsw":    "MEM[{rs1} + {imm}] = {rd} (f32)",
    # Custom pseudo (ScratchV)
    "max":   "{rd} = max({rs1}, {rs2})",
}


# ---------------------------------------------------------------------------
# Line parsing
# ---------------------------------------------------------------------------

# Regex: optional label at start, then instruction, operands, comment
_LINE_RE = re.compile(
    r'^\s*'
    r'(?P<label>[A-Za-z_.][A-Za-z0-9_.]*:)?\s*'
    r'(?P<opcode>\.?\w[\w.]*)?\s*'
    r'(?P<operands>[^#]*?)'
    r'(?:\s*#\s*(?P<comment>.*))?'
    r'$'
)


def _parse_line(line: str) -> dict:
    """Parse one assembly line into a dictionary of components.

    Returns dict with keys: raw, label, opcode, operands, comment.
    If the line is empty or comment-only, opcode will be None.
    """
    m = _LINE_RE.match(line)
    if m is None:
        return {"raw": line, "label": None, "opcode": None,
                "operands": "", "comment": None}

    label = m.group("label")
    if label is not None:
        label = label.rstrip(":")

    opcode = m.group("opcode")
    if opcode is not None:
        opcode = opcode.strip()

    operands_raw = (m.group("operands") or "").strip()
    comment = m.group("comment")
    if comment is not None:
        comment = comment.strip()

    # Split operands by comma, respecting parenthesised groups
    operands_list = [o.strip() for o in _split_operands(operands_raw)]

    return {
        "raw": line,
        "label": label,
        "opcode": opcode,
        "operands_str": operands_raw,
        "operands": operands_list,
        "comment": comment,
    }


def _split_operands(s: str) -> list[str]:
    """Split operand string by comma, respecting parenthesised groups.

    Example: "x1, 0(x2)" -> ["x1", "0(x2)"]
    """
    parts = []
    depth = 0
    current = []
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


# ---------------------------------------------------------------------------
# Comment generation
# ---------------------------------------------------------------------------

def _gen_comment(opcode: str, operands: list[str]) -> str:
    """Generate a human-readable semantic comment for an instruction."""
    if opcode is None or not opcode:
        return ""

    # Strip directive prefix (.) for matching
    base = opcode.lstrip(".")
    template = _INST_COMMENTS.get(base)
    if template is None:
        # For unknown instructions, still try to describe registers
        parts = [opcode]
        reg_parts = ", ".join(operands)
        if reg_parts:
            parts.append(reg_parts)
        return "; ".join(parts)

    # Build substitution context
    ctx: dict[str, str] = {}
    padded = list(operands) + [""] * 5  # pad so we don't index out of range

    # Map operand positions to template roles
    # Convention: first operand is rd, second is rs1, third is rs2
    ctx["rd"] = padded[0] if len(padded) > 0 else ""
    ctx["rs1"] = padded[1] if len(padded) > 1 else ""
    ctx["rs2"] = padded[2] if len(padded) > 2 else ""
    ctx["imm"] = padded[1] if len(padded) > 1 else (
        ""  # immediate often is operand 1
    )

    # For branch/jump, the label is often last operand
    if base in ("beq", "bne", "blt", "bge", "bltu", "bgeu", "bnez", "beqz",
                "j", "jal", "call"):
        ctx["rd"] = padded[-1] if len(operands) > 0 else ""

    # For lui/li, imm is operand 1
    if base in ("li", "lui"):
        ctx["imm"] = padded[1] if len(padded) > 1 else ""

    # For load/store memory operands
    if base in ("lw", "sw", "lh", "sh", "lb", "sb", "lbu", "lhu"):
        ctx["imm"] = (
            _extract_offset(padded[-1]) if len(operands) > 0 else ""
        )
        # rs1 is the base register inside parentheses
        base_reg = _extract_base_reg(padded[-1]) if len(operands) > 0 else ""
        ctx["rs1"] = base_reg or ctx["rs1"]
        ctx["rd"] = padded[0] if len(padded) > 0 else ""

    try:
        return template.format(**ctx)
    except KeyError:
        return ""


def _extract_offset(op: str) -> str:
    """Extract offset from memory operand like '16(sp)'."""
    m = re.match(r'^(-?\d+)\(', op)
    if m:
        return m.group(1)
    return op.lstrip("(").rstrip(")")


def _extract_base_reg(op: str) -> str:
    """Extract base register from memory operand like '16(sp)'."""
    m = re.search(r'\((\w+)\)', op)
    if m:
        return m.group(1)
    return ""


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

def _detect_section(opcode: str) -> Optional[str]:
    """Return section name if opcode indicates a section change."""
    if opcode is None:
        return None
    if opcode == ".text":
        return "CODE"
    if opcode == ".data":
        return "DATA"
    if opcode == ".bss":
        return "BSS"
    if opcode == ".rodata":
        return "READ-ONLY DATA"
    return None


def _is_function_label(label: str) -> bool:
    """Detect if a label looks like a function entry point."""
    if not label:
        return False
    return (not label.startswith(".") and not label.startswith("L")
            and not label.startswith("loop")
            and not label.startswith("_L"))


# ---------------------------------------------------------------------------
# Main beautifier
# ---------------------------------------------------------------------------

def beautify_asm(asm_text: str, align: bool = True,
                 add_comments: bool = True) -> str:
    """Beautify RISC-V assembly text.

    Parameters
    ----------
    asm_text:
        Raw RISC-V assembly source text.
    align:
        If True, align labels, opcodes, and operands into fixed-width columns.
    add_comments:
        If True, add semantic comments to each instruction line.

    Returns
    -------
    Formatted assembly string.
    """
    lines = asm_text.strip().split("\n")
    parsed_lines = [_parse_line(ln) for ln in lines]

    # Compute column widths (first pass)
    max_label = 0
    max_opcode = 0
    max_operands = 0
    for p in parsed_lines:
        label_len = len(p["label"]) if p["label"] else 0
        opcode_len = len(p["opcode"]) if p["opcode"] else 0
        ops_len = len(p["operands_str"]) if p["operands_str"] else 0
        if label_len > max_label:
            max_label = label_len
        if opcode_len > max_opcode:
            max_opcode = opcode_len
        if ops_len > max_operands:
            max_operands = ops_len

    # Clamp widths
    max_label = min(max(max_label, 0), 30)
    max_opcode = min(max(max_opcode, 8), 12)
    max_operands = min(max(max_operands, 0), 40)

    output_lines: list[str] = []
    current_section: Optional[str] = None
    prev_was_empty = False

    for p in parsed_lines:
        # Detect section changes and insert headers
        section = _detect_section(p.get("opcode", ""))
        if section and section != current_section:
            current_section = section
            if output_lines:
                output_lines.append("")
            output_lines.append(f"# {'=' * 60}")
            output_lines.append(f"#  {section} SECTION")
            output_lines.append(f"# {'=' * 60}")

        # Function entry labels
        raw_label = p.get("label")
        if raw_label and _is_function_label(raw_label):
            if output_lines and not prev_was_empty:
                output_lines.append("")
            output_lines.append(f"# --- Function: {raw_label} ---")

        # Format the line
        formatted = _format_line(
            p, align, max_label, max_opcode,
            max_operands, add_comments,
        )
        output_lines.append(formatted)

        prev_was_empty = (
            p["opcode"] is None and p["label"] is None
            and p["raw"].strip() == ""
        )

    return "\n".join(output_lines) + "\n"


def _format_line(parsed: dict, align: bool, max_label: int,
                 max_opcode: int, max_operands: int,
                 add_comments: bool) -> str:
    """Format a single parsed assembly line."""
    raw = parsed["raw"]

    # Empty lines: preserve
    if not raw.strip():
        return ""

    # Comment-only lines: preserve as-is
    stripped = raw.strip()
    if stripped.startswith("#"):
        return raw

    label = parsed.get("label")
    opcode = parsed.get("opcode")
    operands_str = parsed.get("operands_str", "")
    user_comment = parsed.get("comment")

    # Build the line
    parts: list[str] = []

    if label is not None:
        if align:
            parts.append(f"{label + ':':<{max_label + 1}}")
        else:
            parts.append(f"{label}:")

    if opcode is not None:
        if align:
            # Align after label
            if label is not None:
                parts.append(f"{opcode:<{max_opcode}}")
            else:
                # Indent opcode when no label
                parts.append(f"{opcode:<{max_opcode + 2}}")
        else:
            parts.append(opcode)
    else:
        # Directive or label-only line
        if label is not None:
            if align:
                return f"{label + ':':<{max_label + 1}}"
            else:
                return f"{label}:"
        return raw

    # Operands
    if operands_str:
        if align:
            parts.append(f"{operands_str:<{max_operands}}")
        else:
            parts.append(operands_str)

    # Join with spaces
    result = " ".join(parts).rstrip()

    # Add comments
    comment_parts = []
    if user_comment:
        comment_parts.append(user_comment)

    if add_comments and opcode:
        operands_list = parsed.get("operands", [])
        semantic = _gen_comment(opcode, operands_list)
        if semantic:
            # Only add if different from user comment
            if not user_comment or semantic not in user_comment:
                comment_parts.append(semantic)

    if comment_parts:
        result += "  # " + "  |  ".join(comment_parts)

    return result


# ---------------------------------------------------------------------------
# File-level helpers
# ---------------------------------------------------------------------------

def beautify_file(input_path: str, output_path: Optional[str] = None,
                  align: bool = True, add_comments: bool = True) -> str:
    """Read an assembly file, beautify it, optionally write to output.

    Parameters
    ----------
    input_path:
        Path to the input .s file.
    output_path:
        Path to write output. If None, return the string only.
    align:
        Align columns.
    add_comments:
        Add semantic comments.

    Returns
    -------
    The beautified assembly string.
    """
    with open(input_path, "r") as f:
        asm_text = f.read()

    result = beautify_asm(asm_text, align=align, add_comments=add_comments)

    if output_path:
        with open(output_path, "w") as f:
            f.write(result)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point for the assembly beautifier."""
    parser = argparse.ArgumentParser(
        description=(
            "RISC-V Assembly Beautifier - format and annotate .s files"
        ),
    )
    parser.add_argument(
        "input", type=str,
        help="Input assembly file (.s)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file (default: print to stdout)",
    )
    parser.add_argument(
        "--no-align", action="store_true",
        help="Disable column alignment",
    )
    parser.add_argument(
        "--no-comments", action="store_true",
        help="Disable automatic semantic comments",
    )

    args = parser.parse_args(argv)

    result = beautify_file(
        args.input,
        output_path=args.output,
        align=not args.no_align,
        add_comments=not args.no_comments,
    )

    if args.output is None:
        print(result)


if __name__ == "__main__":
    main()
