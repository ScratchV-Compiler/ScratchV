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

Diagnostics: strict mode (default) fails fast with the first structured
error from the shared pre-validation pass, then raises ``DSLSyntaxError``
on any remaining rich error; passing an ``ErrorCollector`` records errors
and recovers/skips bad lines or blocks, returning a partial Program.
"""

from __future__ import annotations

import re
from typing import Optional

from scratchv.frontend.dsl_errors import (
    DSLParseError,
    DSLSyntaxError,
    ErrorCollector,
    ErrorCode,
)
from scratchv.frontend.dsl_parser import DSLParser
from scratchv.frontend.dsl_validator import DSLValidator
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import OpCode, Program, Value

__all__ = [
    "ExtendedDSLParser",
    "CondExpr",
    "DSLParseError",
    "DSLSyntaxError",
]


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
        self._while_stack: list[dict[str, str | int]] = []

    # -----------------------------------------------------------------------
    # Label generation
    # -----------------------------------------------------------------------

    def _fresh_label(self, prefix: str = "L") -> str:
        """Generate a unique label name."""
        self._label_counter += 1
        return f"{prefix}{self._label_counter}"

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def validate(
        self,
        text: str,
        *,
        filename: Optional[str] = None,
        max_errors: int = 20,
    ) -> ErrorCollector:
        """Validate base and extended DSL syntax without creating IR."""
        return DSLValidator(extended=True).validate(
            text, filename=filename, max_errors=max_errors,
        )

    # -----------------------------------------------------------------------
    # Core parse method (overrides base)
    # -----------------------------------------------------------------------

    def parse(
        self,
        text: str,
        filename: Optional[str] = None,
        collector: Optional[ErrorCollector] = None,
    ) -> Program:
        """Parse DSL text into IR Program, supporting if/else and while.

        Args:
            text: The DSL source code as a string.
            filename: Optional source filename for diagnostics.
            collector: Optional ErrorCollector; when provided, errors are
                collected and parsing recovers instead of raising.

        Returns:
            A Program object containing the generated IR (partial when the
            collector recorded errors).
        """
        if collector is None:
            preflight = self.validate(text, filename=filename)
            if preflight.has_errors:
                raise preflight.errors[0]

        raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        lines: list[str] = []
        for raw in raw_lines:
            line = raw.strip()
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
        self._vars = {}
        self._loop_stack = []
        self._for_positions = []
        self._raw_lines = raw_lines
        self._filename = filename
        self._collector = collector
        self._line_no = 0
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

            if re.match(r"^if\b", line):
                idx = self._parse_if_block(lines, idx)
            elif re.match(r"^while\b", line):
                idx = self._parse_while_block(lines, idx)
            else:
                self._parse_line(line, idx + 1)
                idx += 1

        unclosed = bool(self._loop_stack or self._while_stack)
        self._report_unclosed_for()
        self._report_unclosed_while()

        # Ensure function ends with a return
        if (
            not self._loop_stack
            and not self._while_stack
            and not unclosed
        ):
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
    # Block parsing helpers
    # -----------------------------------------------------------------------

    def _report_unclosed_while(self) -> None:
        """Report every unclosed ``while`` at EOF (LIFO), clearing the stack."""
        while self._while_stack:
            ctx = self._while_stack.pop()
            line_no = int(ctx.get("line", self._line_no))
            col = int(ctx.get("col", 1))
            self._report_error(
                line_no, col,
                "missing 'endwhile' for 'while' opened here",
                ErrorCode.SYN_MISSING_TERMINATOR,
                fix_hint="add 'endwhile' to close this block",
            )

    @staticmethod
    def _condition_hint(line: str) -> str:
        """Build a fix hint for E202 based on paren balance."""
        if line.count("(") > line.count(")"):
            return "missing closing ')'"
        if line.count(")") > line.count("("):
            return "missing opening '('"
        return (
            "expected one of ==, !=, <, >, <=, >= "
            "and parentheses around each operand"
        )

    def _parse_block(
        self,
        lines: list[str],
        start_idx: int,
        terminators: tuple[str, ...],
        opener_kind: str,
        opener_line: int,
        opener_col: int,
    ) -> tuple[int, Optional[str]]:
        """Parse block body until a terminator.

        Returns ``(next_index, terminator)``. Terminators ``endif`` and
        ``endwhile`` are never consumed here: they are returned to the caller
        so an unmatched one can be handed to the enclosing block (or reported
        as stray at the top level). ``endfor`` is delegated to the base
        statement parser.

        Args:
            lines: Stripped/comment-free source lines (blank = skipped).
            start_idx: First index of the body.
            terminators: Tokens that end this block (e.g. ``else``, ``endif``).
            opener_kind: ``"if"`` or ``"while"`` (diagnostics only).
            opener_line: 1-based line of the opening keyword.
            opener_col: 1-based column of the opening keyword.
        """
        idx = start_idx
        while idx < len(lines):
            line = lines[idx]
            if not line:
                idx += 1
                continue
            if line in ("endif", "endwhile"):
                return idx, line  # never consume; caller decides
            if line in ("else", "else:"):
                if line in terminators:
                    return idx, line
                self._report_error(
                    idx + 1,
                    self._col_of(
                        idx + 1, "else", self._line_indent(idx + 1) + 1,
                    ),
                    "'else' without matching 'if'",
                    ErrorCode.SYN_STRAY_TERMINATOR,
                    fix_hint="remove this line or add a matching 'if'",
                )
                idx += 1
                continue
            if line == "endfor":
                self._parse_line(line, idx + 1)
                idx += 1
                continue
            if re.match(r"^if\b", line):
                idx = self._parse_if_block(lines, idx)
            elif re.match(r"^while\b", line):
                idx = self._parse_while_block(lines, idx)
            else:
                self._parse_line(line, idx + 1)
                idx += 1
        return idx, None

    def _recover_after_bad_header(
        self, lines: list[str], start_idx: int, opener_kind: str,
    ) -> int:
        """Skip a block whose header failed to parse (E202 already reported).

        Scans forward respecting nested openers until the matching terminator
        (consumed) or a foreign terminator at depth 0 (not consumed) or EOF.
        Never reports errors itself, to avoid cascading diagnostics.
        """
        end_tok = "endif" if opener_kind == "if" else "endwhile"
        idx = start_idx + 1
        depth = 0
        while idx < len(lines):
            line = lines[idx]
            if not line:
                idx += 1
                continue
            if re.match(r"^if\b", line) or re.match(r"^while\b", line):
                depth += 1
            elif line in ("endif", "endwhile"):
                if depth > 0:
                    depth -= 1
                elif line == end_tok:
                    return idx + 1
                else:
                    return idx
            idx += 1
        return len(lines)

    # -----------------------------------------------------------------------
    # if / else / endif parsing
    # -----------------------------------------------------------------------

    def _parse_if_block(self, lines: list[str], start_idx: int) -> int:
        """Parse an if/else/endif block starting at start_idx.

        Returns the index of the next line after 'endif' (or after the
        last consumed line on error paths).
        """
        line = lines[start_idx]
        opener_line = start_idx + 1
        opener_col = self._col_of(
            opener_line, "if", self._line_indent(opener_line) + 1,
        )
        cond = self._parse_condition(line)
        if cond is None:
            self._report_error(
                opener_line, opener_col,
                "invalid condition in 'if'; "
                "expected 'if (<expr>) <op> (<expr>):'",
                ErrorCode.SYN_INVALID_CONDITION,
                fix_hint=self._condition_hint(line),
            )
            return self._recover_after_bad_header(lines, start_idx, "if")

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
        idx, term = self._parse_block(
            lines, start_idx + 1, ("else", "else:", "endif"),
            "if", opener_line, opener_col,
        )

        # Terminate then branch with jump to endif
        self.builder.br(endif_label)

        if term in ("else", "else:"):
            idx += 1
            self.builder.new_block(else_label)
            idx, term2 = self._parse_block(
                lines, idx, ("endif",), "if", opener_line, opener_col,
            )
            self.builder.br(endif_label)
            if term2 != "endif":
                self._report_error(
                    opener_line, opener_col,
                    "missing 'endif' for 'if' opened here",
                    ErrorCode.SYN_MISSING_TERMINATOR,
                    fix_hint="add 'endif' to close this block",
                )
            else:
                idx += 1
        else:
            # No else branch: keep the (empty) else block for IR shape
            self.builder.new_block(else_label)
            self.builder.br(endif_label)
            if term == "endif":
                idx += 1
            else:
                # EOF or foreign terminator (e.g. 'endwhile'): not consumed
                self._report_error(
                    opener_line, opener_col,
                    "missing 'endif' for 'if' opened here",
                    ErrorCode.SYN_MISSING_TERMINATOR,
                    fix_hint="add 'endif' to close this block",
                )

        self.builder.new_block(endif_label)
        return idx

    # -----------------------------------------------------------------------
    # while / endwhile parsing
    # -----------------------------------------------------------------------

    def _parse_while_block(self, lines: list[str], start_idx: int) -> int:
        """Parse a while/endwhile block starting at start_idx.

        Returns the index of the next line after 'endwhile' (or after the
        last consumed line on error paths).
        """
        line = lines[start_idx]
        opener_line = start_idx + 1
        opener_col = self._col_of(
            opener_line, "while", self._line_indent(opener_line) + 1,
        )
        cond = self._parse_condition(line)
        if cond is None:
            self._report_error(
                opener_line, opener_col,
                "invalid condition in 'while'; "
                "expected 'while (<expr>) <op> (<expr>):'",
                ErrorCode.SYN_INVALID_CONDITION,
                fix_hint=self._condition_hint(line),
            )
            return self._recover_after_bad_header(lines, start_idx, "while")

        header_label = self._fresh_label("while_hdr")
        body_label = self._fresh_label("while_body")
        exit_label = self._fresh_label("while_exit")

        # Push while context for nested loop support
        self._while_stack.append({
            "header": header_label,
            "body": body_label,
            "exit": exit_label,
            "line": opener_line,
            "col": opener_col,
        })

        try:
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
            idx, term = self._parse_block(
                lines, start_idx + 1, ("endwhile",),
                "while", opener_line, opener_col,
            )

            # Jump back to header
            self.builder.br(header_label)

            if term == "endwhile":
                idx += 1
            else:
                # EOF or foreign terminator (e.g. 'endif'): not consumed
                self._report_error(
                    opener_line, opener_col,
                    "missing 'endwhile' for 'while' opened here",
                    ErrorCode.SYN_MISSING_TERMINATOR,
                    fix_hint="add 'endwhile' to close this block",
                )

            self.builder.new_block(exit_label)
            return idx
        finally:
            self._while_stack.pop()

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

    def _parse_line(self, line: str, line_no: int = 0) -> None:
        """Parse a single DSL line, delegating to base for standard ops."""
        # Stray terminators reaching statement level are errors
        if line in ("endif", "endwhile"):
            opener = "if" if line == "endif" else "while"
            self._report_error(
                line_no,
                self._col_of(
                    line_no, line, self._line_indent(line_no) + 1,
                ),
                f"'{line}' without matching '{opener}'",
                ErrorCode.SYN_STRAY_TERMINATOR,
                fix_hint=f"remove this line or add a matching '{opener}'",
            )
            return
        if line in ("else", "else:"):
            self._report_error(
                line_no,
                self._col_of(
                    line_no, "else", self._line_indent(line_no) + 1,
                ),
                "'else' without matching 'if'",
                ErrorCode.SYN_STRAY_TERMINATOR,
                fix_hint="remove this line or add a matching 'if'",
            )
            return
        if re.match(r"^if\b", line) or re.match(r"^while\b", line):
            return
        super()._parse_line(line, line_no)

    # -----------------------------------------------------------------------
    # Convenience: create a stand-alone label block
    # -----------------------------------------------------------------------

    def _label(self, name: str) -> None:
        """Create a new block with the given label name.

        Args:
            name: The label/block name.
        """
        self.builder.new_block(name)
