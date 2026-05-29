"""Extended DSL parser with if/else and while control-flow support.

Extends the existing DSLParser to parse conditionals (if/else) and
while loops, generating proper IR with labels and conditional branches.

New syntax supported:
    if (a > b):
        ...
    else:
        ...
    endif

    while (i < 10):
        ...
    endwhile

The extended parser follows the same patterns as the base DSLParser:
recursive-descent parsing, variable-to-Value tracking, and IR generation
via IRBuilder.
"""

from __future__ import annotations

import re
from typing import Optional

# moved import above
from scratchv.frontend.dsl_parser import DSLParser, DSLParseError
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import OpCode, Program, Value


# ---------------------------------------------------------------------------
# Conditional expression node
# ---------------------------------------------------------------------------

class CondExpr:
    """Represents a parsed conditional expression for if/while guards.

    Supports comparison operators: ==, !=, <, >, <=, >=.
    Each operand can be a variable name or numeric literal.
    """

    def __init__(self, lhs: str, op: str, rhs: str):
        self.lhs = lhs.strip()
        self.op = op.strip()
        self.rhs = rhs.strip()

    def resolve(self, parser: ExtendedDSLParser) -> tuple[Value, str]:
        """Resolve operands and return (lhs_val, operator, rhs_val)."""
        lhs_val = parser._resolve(self.lhs)
        rhs_val = parser._resolve(self.rhs)
        return lhs_val, self.op, rhs_val

    def __repr__(self) -> str:
        return f"CondExpr({self.lhs} {self.op} {self.rhs})"


# ---------------------------------------------------------------------------
# Extended DSL Parser
# ---------------------------------------------------------------------------

class ExtendedDSLParser(DSLParser):
    """DSL parser with extended control-flow constructs (if/else, while).

    Inherits all arithmetic and NN-operation parsing from DSLParser and adds
    support for conditional branching and while loops.

    Usage::

        parser = ExtendedDSLParser()
        program = parser.parse(dsl_source_text)

    The parser generates IR with:
    - Labels for branch targets and loop headers
    - Conditional branch instructions (cmp + br_if)
    - Proper control-flow structure for nested constructs
    """

    def __init__(self):
        super().__init__()
        # Label counters for generating unique block names
        self._label_counter: int = 0
        # Stack for tracking nested while-loop labels
        self._while_stack: list[dict[str, str]] = []

    # -----------------------------------------------------------------------
    # Label generation
    # -----------------------------------------------------------------------

    def _fresh_label(self, prefix: str = "L") -> str:
        """Generate a unique label name."""
        self._label_counter += 1
        return f"{prefix}{self._label_counter}"

    # -----------------------------------------------------------------------
    # Core parse method (overrides base)
    # -----------------------------------------------------------------------

    def parse(self, text: str) -> Program:
        """Parse DSL text into IR Program, supporting if/else and while.

        Args:
            text: The DSL source code as a string.

        Returns:
            A Program object containing the generated IR.
        """
        # Strip comments before splitting to handle block-level constructs
        lines_raw = text.split("\n")
        lines: list[str] = []
        for line in lines_raw:
            line = line.strip()
            if not line or line.startswith("#"):
                lines.append("")  # keep blank for indexing
            else:
                # Inline comment removal
                comment_idx = line.find(" #")
                if comment_idx >= 0:
                    line = line[:comment_idx].strip()
                    if not line:
                        lines.append("")
                    else:
                        lines.append(line)
                else:
                    lines.append(line)

        self.builder = IRBuilder()
        self._vars: dict[str] = {}
        self._loop_stack: list[str] = []
        self._label_counter = 0
        self._while_stack = []

        self.builder.new_function("main")
        self.builder.new_block("entry")

        idx = 0
        while idx < len(lines):
            line = lines[idx]
            if not line:
                idx += 1
                continue

            if line.startswith("if "):
                idx = self._parse_if_block(lines, idx)
            elif line.startswith("while "):
                idx = self._parse_while_block(lines, idx)
            else:
                self._parse_line(line)
                idx += 1

        # Ensure function ends with a return
        if not self._loop_stack and not self._while_stack:
            block = self.builder.current_block
            if block and block.instructions:
                has_ret = (
                    block.instructions[-1].opcode.name == "RETURN"
                )
            else:
                has_ret = False
            if not has_ret:
                self.builder.ret()

        return self.builder.program

    # -----------------------------------------------------------------------
    # if / else / endif parsing
    # -----------------------------------------------------------------------

    def _parse_if_block(self, lines: list[str], start_idx: int) -> int:
        """Parse an if/else/endif block starting at start_idx.

        Returns the index of the next line after 'endif'.
        """
        line = lines[start_idx]
        cond = self._parse_condition(line)
        if cond is None:
            raise DSLParseError(f"Invalid if condition: {line}")

        then_label = self._fresh_label("if_then")
        else_label = self._fresh_label("if_else")
        endif_label = self._fresh_label("if_end")

        # Resolve condition and emit conditional branch
        lhs_val, op_str, rhs_val = cond.resolve(self)
        self.builder._emit(
            OpCode.BR_IF,
            operands=[lhs_val, rhs_val],
            target=f"{then_label},{else_label}",
            cmp_op=op_str,
        )

        # Parse then branch
        self.builder.new_block(then_label)
        idx = start_idx + 1
        while idx < len(lines):
            inner_line = lines[idx]
            if not inner_line:
                idx += 1
                continue
            if inner_line == "else:" or inner_line == "else":
                break
            if inner_line == "endif":
                break
            if inner_line.startswith("if "):
                idx = self._parse_if_block(lines, idx)
            elif inner_line.startswith("while "):
                idx = self._parse_while_block(lines, idx)
            else:
                self._parse_line(inner_line)
                idx += 1

        # Terminate then branch with jump to endif
        self.builder.br(endif_label)

        # Check for else branch
        has_else = False
        if idx < len(lines) and lines[idx] in ("else:", "else"):
            has_else = True
            idx += 1
            self.builder.new_block(else_label)
            while idx < len(lines):
                inner_line = lines[idx]
                if not inner_line:
                    idx += 1
                    continue
                if inner_line == "endif":
                    break
                if inner_line.startswith("if "):
                    idx = self._parse_if_block(lines, idx)
                elif inner_line.startswith("while "):
                    idx = self._parse_while_block(lines, idx)
                else:
                    self._parse_line(inner_line)
                    idx += 1
            self.builder.br(endif_label)

        if not has_else:
            # Else block exists but is empty - just jumps to endif
            self.builder.new_block(else_label)
            self.builder.br(endif_label)

        if idx < len(lines) and lines[idx] == "endif":
            idx += 1

        self.builder.new_block(endif_label)
        return idx

    # -----------------------------------------------------------------------
    # while / endwhile parsing
    # -----------------------------------------------------------------------

    def _parse_while_block(self, lines: list[str], start_idx: int) -> int:
        """Parse a while/endwhile block starting at start_idx.

        Returns the index of the next line after 'endwhile'.
        """
        line = lines[start_idx]
        cond = self._parse_condition(line)
        if cond is None:
            raise DSLParseError(f"Invalid while condition: {line}")

        header_label = self._fresh_label("while_hdr")
        body_label = self._fresh_label("while_body")
        exit_label = self._fresh_label("while_exit")

        # Push while context for nested loop support
        self._while_stack.append({
            "header": header_label,
            "body": body_label,
            "exit": exit_label,
        })

        # Header: evaluate condition, branch to body or exit
        self.builder.br(header_label)
        self.builder.new_block(header_label)
        lhs_val, op_str, rhs_val = cond.resolve(self)
        self.builder._emit(
            OpCode.BR_IF,
            operands=[lhs_val, rhs_val],
            target=f"{body_label},{exit_label}",
            cmp_op=op_str,
        )

        # Body
        self.builder.new_block(body_label)
        idx = start_idx + 1
        while idx < len(lines):
            inner_line = lines[idx]
            if not inner_line:
                idx += 1
                continue
            if inner_line == "endwhile":
                break
            if inner_line.startswith("if "):
                idx = self._parse_if_block(lines, idx)
            elif inner_line.startswith("while "):
                idx = self._parse_while_block(lines, idx)
            else:
                self._parse_line(inner_line)
                idx += 1

        # Jump back to header
        self.builder.br(header_label)

        if idx < len(lines) and lines[idx] == "endwhile":
            idx += 1

        self.builder.new_block(exit_label)
        self._while_stack.pop()
        return idx

    # -----------------------------------------------------------------------
    # Condition parsing
    # -----------------------------------------------------------------------

    _COND_PATTERN = re.compile(
        r'^(?:if|while)\s*\(\s*(.+?)\s*'
        r'(==|!=|<=|>=|<|>)\s*(.+?)\s*\)\s*:?\s*$'
    )

    def _parse_condition(self, line: str) -> Optional[CondExpr]:
        """Attempt to parse a condition from if/while line.

        Returns a CondExpr or None if the line doesn't match.
        """
        m = self._COND_PATTERN.match(line)
        if not m:
            return None
        return CondExpr(lhs=m.group(1), op=m.group(2), rhs=m.group(3))

    # -----------------------------------------------------------------------
    # Comparison IR generation
    # -----------------------------------------------------------------------

    def _emit_cmp(self, lhs: Value, op_str: str, rhs: Value) -> Value:
        """Create a comparison value and return it.

        The comparison is materialized by the subsequent br_if instruction,
        which takes this value as its condition operand and stores the
        comparison details in its attrs.

        Args:
            lhs: Left operand Value.
            op_str: Comparison operator string (==, !=, <, >, <=, >=).
            rhs: Right operand Value.

        Returns:
            A Value representing the comparison result.
        """
        # Create a condition value to use as br_if's condition operand
        dest = self.builder.make_value(name=self._fresh_label("cmp"))
        # The actual comparison operands and operator are passed to br_if
        return dest

    # -----------------------------------------------------------------------
    # Override _parse_line to handle extended keywords
    # -----------------------------------------------------------------------

    def _parse_line(self, line: str) -> None:
        """Parse a single DSL line, delegating to base for standard ops."""
        # Keywords we handle at the block level
        if line in ("endif", "endwhile", "else", "else:"):
            return
        if line.startswith("if ") or line.startswith("while "):
            return
        super()._parse_line(line)

    # -----------------------------------------------------------------------
    # Convenience: create a stand-alone label block
    # -----------------------------------------------------------------------

    def _label(self, name: str) -> None:
        """Create a new block with the given label name.

        Args:
            name: The label/block name.
        """
        self.builder.new_block(name)
