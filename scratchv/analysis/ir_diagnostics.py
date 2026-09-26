"""Context snapshots and human-readable rendering for IR diagnostics.

Snapshots are captured during verification, before later passes can mutate IR.
Positions are original IR indices, never fabricated source line numbers.
"""

from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass
from typing import Optional, TextIO


@dataclass(frozen=True)
class IRContextLine:
    label: str
    text: str
    marked: bool = False
    column: int = 0
    length: int = 1


def _text(value) -> str:
    """Keep identifiers and attributes on one terminal-safe display line."""
    return "".join(char if char.isprintable() else repr(char)[1:-1]
                   for char in str(value))


def _display_width(text):
    return sum(0 if unicodedata.combining(char) else
               2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
               for char in text)


def _value(value) -> str:
    dtype = getattr(value.dtype, "value", value.dtype)
    text = f"${_text(value.name)}: {_text(dtype)}"
    if value.is_constant:
        text += f" ({_text(repr(value.const_value))})"
    return text


def _instruction(instruction) -> str:
    opcode = getattr(instruction.opcode, "value", instruction.opcode)
    result = _text(opcode)
    if instruction.dest is not None:
        # The constant payload belongs to the literal/attrs, not the LHS.
        dest = instruction.dest
        dtype = getattr(dest.dtype, "value", dest.dtype)
        result = f"${_text(dest.name)}: {_text(dtype)} = " + result
    if instruction.operands:
        result += " " + ", ".join(_value(v) for v in instruction.operands)
    if instruction.target is not None:
        result += " -> " + _text(instruction.target)
    for name, value in instruction.attrs.items():
        display = _text(repr(value))
        if len(display) > 120:
            display = display[:117] + "..."
        result += f" [{_text(name)}={display}]"
    return result


def capture_ir_context(program, location, value_name=None, *, radius=2):
    """Return bounded, immutable context using ordinal (not name) lookup."""
    fi, bi, ii, kind, item = location
    lines = []

    def add(label, text, marked=False):
        token = f"${_text(value_name)}:" if value_name is not None else ""
        column = text.rfind(token) if token else -1
        length = len(token) - 1 if column >= 0 else len(text)
        lines.append(IRContextLine(str(label), text, marked, max(column, 0),
                                   max(length, 1)))

    if fi < 0:
        add(f"global #{item}", _value(program.global_values[item]), True)
        return tuple(lines)
    function = program.functions[fi]
    add("", f"function {_text(function.name)} (#{fi})")
    if bi < 0:
        collections = {"param": function.params, "return": function.returns,
                       "local": function.locals}
        if kind in collections:
            add(f"{kind} #{item}", _value(collections[kind][item]), True)
        else:
            add("", "<no basic blocks>" if not function.blocks else
                f"return signature: {', '.join(_value(v) for v in function.returns)}", True)
        return tuple(lines)
    block = function.blocks[bi]
    add("", f"block {_text(block.name)} (#{bi})", ii < 0)
    if not block.instructions:
        add("", "<empty block>")
        return tuple(lines)
    start = max(0, ii - radius) if ii >= 0 else 0
    stop = min(len(block.instructions), ii + radius + 1 if ii >= 0 else radius + 1)
    if start:
        add("...", "")
    for index in range(start, stop):
        add(index, _instruction(block.instructions[index]), index == ii)
    if stop < len(block.instructions):
        add("...", "")
    return tuple(lines)


_HINTS = {
    "undefined-value": "Register external values as parameters/globals, or define the value before using it.",
    "use-before-definition": "Move the definition before this use; an instruction cannot use its own result.",
    "not-dominating": "Ensure the value is defined on every path to this use. Loop bodies may execute zero times.",
    "duplicate-definition": "Give each parameter/global/result a unique name; locals declarations are not definitions.",
    "invalid-dtype": "Use a DataType enum member (FLOAT32, FLOAT64, INT32 or INT64), not a string.",
    "invalid-constant": "Match the literal to its dtype; integer constants must be non-bool integers within the signed range.",
    "constant-disagreement": "Make LOAD_CONST attrs['value'] agree exactly with the result's constant metadata.",
    "operand-count": "Check the opcode signature and supply the required number of operands.",
    "destination": "Check whether this opcode requires or forbids a result value.",
    "unreachable": "Check incoming branches and loop exits; this block cannot be reached from the entry.",
    "FOR without ENDFOR": "Close this FOR with a matching ENDFOR in the same function.",
    "ENDFOR without FOR": "Check loop nesting and add the matching FOR before this ENDFOR.",
    "unsupported representation: LABEL": "Use BasicBlock names and explicit branches; LABEL is not supported by this verifier.",
    "unsupported representation: phi_nodes": "This verifier does not support phi_nodes; use a supported representation before verification.",
    "instruction after explicit terminator": "Keep BR, BR_IF or RETURN last in its block; place subsequent code in a properly connected block.",
    "explicit branch to function entry": "Use a separate loop header instead of branching back to the function entry.",
}
_RULE_HINTS = {
    "def-before-use": "Check the definition and all control-flow paths reaching this use.",
    "label-existence": "Use unique, nonempty block names and branch to a block in the current function.",
    "block-termination": "End the block with RETURN, BR or BR_IF, or a valid structured-loop continuation.",
    "type-consistency": "Check operand/result types, literal metadata and opcode attributes; implicit casts are not performed.",
    "control-flow-integrity": "Check branch targets, terminators and FOR/ENDFOR nesting.",
    "ssa-validity": "Use a unique name for each static definition.",
    "entry-existence": "Add an entry basic block with a valid terminator to this function.",
}


def ir_fix_hint(rule, reason):
    return _HINTS.get(reason, _RULE_HINTS.get(rule))


def format_ir_error(issue, *, use_color=False) -> str:
    """Render one diagnostic. ``str(issue)`` remains the compact API format."""
    severity = issue.level.value
    color = "\033[31m" if severity == "error" else "\033[33m"

    def highlight(text):
        return color + text + "\033[0m" if use_color else text

    parts = [highlight(_text(str(issue)))]
    if issue.context:
        width = max(len(line.label) for line in issue.context)
        for line in issue.context:
            arrow = "-->" if line.marked else "   "
            prefix = f"{arrow} {line.label:>{width}} | "
            parts.append(prefix + line.text)
            if line.marked:
                padding = _display_width(line.text[:line.column])
                length = _display_width(line.text[line.column:line.column + line.length])
                marker = " " * padding + "^" + "~" * max(length - 1, 0)
                parts.append(" " * (width + 4) + " | " + highlight(marker))
    if issue.fix_hint:
        parts.append("note: " + issue.fix_hint)
    return "\n".join(parts)


def render_ir_error(issue, *, stream: TextIO, use_color: Optional[bool] = None) -> str:
    """Like the DSL renderer, honor the output stream and NO_COLOR."""
    if use_color is None:
        use_color = bool(getattr(stream, "isatty", lambda: False)()) and "NO_COLOR" not in os.environ
    return format_ir_error(issue, use_color=use_color)
