"""Simple DSL parser for testing IR and backend without ONNX dependency.

DSL syntax (one operation per line):
  # comment
  name = add(a, b)
  name = mul(a, b)
  name = relu(x)
  name = matmul(a, b, rows:m, cols:n, inner:k)
  name = dot(a, b, len:N)
  name = gelu(x)
  name = softmax(x, axis:-1)
  name = maxpool(x, kernel:2, stride:2)
  name = exp(x)
  name = neg(x)
  return name
  for i = 0, N    # start a loop
  endfor          # end a loop
"""

from __future__ import annotations

import re
from typing import Optional

from scratchv.frontend.dsl_errors import (
    DSLParseError,
    DSLSyntaxError,
    ErrorCollector,
    ErrorCode,
    suggest_op,
    suggest_spelling,
    _ARITY_HINTS,
)
from scratchv.frontend.dsl_validator import DSLValidator, OP_SIGNATURES
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import Value, Program

# Re-exported for backward compatibility:
#   from scratchv.frontend.dsl_parser import DSLParseError
__all__ = [
    "DSLParser",
    "DSLParseError",
    "DSLSyntaxError",
    "ErrorCode",
    "ErrorCollector",
]

# Expected count of plain (non-kwarg) arguments per operator (E302).
_ARITY: dict[str, int] = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "neg": 1,
    "exp": 1,
    "relu": 1,
    "gelu": 1,
    "dot": 2,
    "matmul": 2,
    "softmax": 1,
    "maxpool": 1,
}

# Numeric literal accepted for numeric kwargs (mirrors validator `_NUMBER`).
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")


class DSLParser:
    """Parses a simple DSL text into an IR Program.

    ``parse(text)`` runs in strict mode: a structural pre-validation pass
    (``validate``) fails fast with the first structured error, then the
    rich parser raises ``DSLSyntaxError`` on the first remaining error.
    Passing a non-None ``collector`` switches to collecting mode: errors
    are recorded with full recovery/suggestions, returning a *partial*
    Program that must not be used for code generation.
    """

    def __init__(self):
        self.builder = IRBuilder()
        self._vars: dict[str, Value] = {}
        self._loop_stack: list[str] = []
        # Diagnostic context (see dev contract 0.7)
        self._raw_lines: list[str] = []
        self._filename: Optional[str] = None
        self._collector: Optional[ErrorCollector] = None
        self._for_positions: list[tuple[int, int]] = []
        self._line_no: int = 0

    # -- validation ---------------------------------------------------------

    def validate(
        self,
        text: str,
        *,
        filename: Optional[str] = None,
        max_errors: int = 20,
    ) -> ErrorCollector:
        """Validate DSL syntax without constructing IR."""
        return DSLValidator().validate(
            text, filename=filename, max_errors=max_errors,
        )

    @staticmethod
    def supported_operations() -> set[str]:
        """Return the operator names this parser can lower."""
        return {
            "add", "sub", "mul", "div", "neg", "exp", "relu", "gelu",
            "dot", "matmul", "softmax", "maxpool",
        }

    # -- diagnostics --------------------------------------------------------

    def _line_indent(self, line_no: int) -> int:
        """Return the leading-whitespace width of a physical line."""
        if 0 < line_no <= len(self._raw_lines):
            raw = self._raw_lines[line_no - 1]
            return len(raw) - len(raw.lstrip())
        return 0

    def _col_of(self, line_no: int, needle: str, fallback: int = 1) -> int:
        """Return the 1-based column of ``needle`` in a physical line."""
        if 0 < line_no <= len(self._raw_lines):
            idx = self._raw_lines[line_no - 1].find(needle)
            if idx >= 0:
                return idx + 1
        return max(fallback, 1)

    def _report_error(
        self,
        line_no: int,
        col: int,
        message: str,
        error_code: str,
        fix_hint: Optional[str] = None,
    ) -> None:
        """Build/report a DSLSyntaxError (raise in strict, record otherwise)."""
        raw = (
            self._raw_lines[line_no - 1]
            if 0 < line_no <= len(self._raw_lines)
            else ""
        )
        err = DSLSyntaxError(
            line=line_no,
            col=max(col, 1),
            message=message,
            source_line=raw,
            filename=self._filename,
            fix_hint=fix_hint,
            error_code=error_code,
        )
        if self._collector is not None:
            self._collector.add(err)
            return
        raise err

    def _report_unclosed_for(self) -> None:
        """Report every unclosed ``for`` at EOF (LIFO), then clear the stack."""
        while self._loop_stack:
            self._loop_stack.pop()
            if self._for_positions:
                line_no, col = self._for_positions.pop()
            else:
                line_no, col = self._line_no, 1
            self._report_error(
                line_no, col,
                "missing 'endfor' for 'for' opened here",
                ErrorCode.SYN_MISSING_TERMINATOR,
                fix_hint="add 'endfor' to close this block",
            )

    # -- parsing ------------------------------------------------------------

    def parse(
        self,
        text: str,
        filename: Optional[str] = None,
        collector: Optional[ErrorCollector] = None,
    ) -> Program:
        if collector is None:
            preflight = self.validate(text, filename=filename)
            if preflight.has_errors:
                raise preflight.errors[0]

        raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

        self.builder = IRBuilder()
        self._vars = {}
        self._loop_stack = []
        self._for_positions = []
        self._raw_lines = raw_lines
        self._filename = filename
        self._collector = collector
        self._line_no = 0

        self.builder.new_function("main")
        self.builder.new_block("entry")

        for i, raw in enumerate(raw_lines):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            self._line_no = i + 1
            self._parse_line(line, i + 1)

        unclosed_loop = bool(self._loop_stack)
        self._report_unclosed_for()

        if not self._loop_stack and not unclosed_loop:
            block = self.builder.current_block
            if block and block.instructions:
                has_ret = block.instructions[-1].opcode.name == "RETURN"
            else:
                has_ret = False
            if not has_ret:
                self.builder.ret()
        return self.builder.program

    def _parse_line(self, line: str, line_no: int = 0) -> None:
        # Strip trailing inline comment (same convention as extended parser)
        line = line.split(" #", 1)[0].strip()
        if not line:
            return

        indent = self._line_indent(line_no)

        # for i = start, end
        m = re.fullmatch(r"for\s+(\w+)\s*=\s*(\d+)\s*,\s*(\d+)", line)
        if m:
            iv = self.builder.for_loop(int(m.group(2)), int(m.group(3)))
            self._vars[m.group(1)] = iv
            self._loop_stack.append(m.group(1))
            self._for_positions.append((line_no, indent + 1))
            return

        if line == "endfor":
            if not self._loop_stack:
                self._report_error(
                    line_no, self._col_of(line_no, "endfor", indent + 1),
                    "'endfor' without matching 'for'",
                    ErrorCode.SYN_STRAY_TERMINATOR,
                    fix_hint="remove this line or add a matching 'for'",
                )
                return
            self._loop_stack.pop()
            if self._for_positions:
                self._for_positions.pop()
            self.builder.endfor()
            return

        # return [var]
        m = re.fullmatch(r"return\s+(\S+)", line)
        if m:
            val = self._resolve(m.group(1))
            self.builder.ret(val)
            return

        # name = op(args)
        m = re.match(r"^(\w+)\s*=\s*(\w+)\s*\((.*)\)\s*$", line)
        if m:
            dest_name = m.group(1)
            op_name = m.group(2)
            args_text = m.group(3)

            # Nested function call: inner '(' inside the argument list
            if "(" in args_text:
                col = indent + m.start(3) + args_text.find("(") + 1
                self._report_error(
                    line_no, col,
                    "nested function call is not supported",
                    ErrorCode.SYN_NESTED_CALL,
                    fix_hint="assign the inner call to a temporary variable first",
                )
                return

            # Stray closing paren in the argument list
            if ")" in args_text:
                col = indent + m.start(3) + args_text.find(")") + 1
                self._report_error(
                    line_no, col,
                    "cannot parse statement; expected 'name = op(args)'",
                    ErrorCode.SYN_INVALID_STATEMENT,
                    fix_hint="missing opening '('",
                )
                return

            args = [a.strip() for a in args_text.split(",") if a.strip()]
            op_col = indent + m.start(2) + 1
            result = self._dispatch_op(op_name, args, line_no, op_col)
            if result is None:
                # Error already reported; do not pollute _vars.
                return
            self._vars[dest_name] = result
            return

        # Fallback: illegal character scan, then generic statement error
        illegal = re.search(r"[^A-Za-z0-9_(),:=.+\-*/#%\s]", line)
        if illegal:
            ch = illegal.group(0)
            col = indent + illegal.start() + 1
            self._report_error(
                line_no, col,
                f"unexpected character '{ch}'",
                ErrorCode.LEX_ILLEGAL_CHAR,
                fix_hint=f"remove or replace '{ch}'",
            )
            return

        hint: Optional[str] = None
        if line.count("(") > line.count(")"):
            hint = "missing closing ')'"
        elif line.count(")") > line.count("("):
            hint = "missing opening '('"
        self._report_error(
            line_no, indent + 1,
            "cannot parse statement; expected 'name = op(args)'",
            ErrorCode.SYN_INVALID_STATEMENT,
            fix_hint=hint,
        )

    def _resolve(self, name: str) -> Value:
        """Resolve a variable name or literal to a Value.

        Note: undefined variables are created on first access (E303 is a
        reserved error code and intentionally not raised).
        """
        if name in self._vars:
            return self._vars[name]
        try:
            val = float(name)
            return self.builder.load_const(val)
        except ValueError:
            pass
        # Create a variable on first access
        v = self.builder.make_value(name=name)
        self._vars[name] = v
        return v

    def _parse_kwargs(
            self, args: list[str], op: str,
            line_no: int = 0, col: int = 1,
    ) -> Optional[tuple[list[str], dict[str, int | float | str]]]:
        """Split ``args`` into plain and keyword arguments.

        Keyword arguments are validated against the operator signature
        (``OP_SIGNATURES``): unknown keys and non-numeric values for numeric
        kwargs are reported as E304. Returns ``None`` when an error was
        reported (strict mode raises before returning).
        """
        signature = OP_SIGNATURES.get(op)
        allowed = (
            signature.optional_kwargs | signature.required_kwargs
            if signature is not None
            else frozenset()
        )
        numeric = (
            signature.numeric_kwargs if signature is not None else frozenset()
        )
        kwargs: dict[str, int | float | str] = {}
        plain: list[str] = []
        for a in args:
            if ":" not in a:
                plain.append(a)
                continue
            k, v = a.split(":", 1)
            k = k.strip()
            v = v.strip()
            err_col = self._col_of(line_no, k, col)
            if k not in allowed:
                self._report_error(
                    line_no, err_col,
                    f"invalid keyword argument '{k}'",
                    ErrorCode.SEM_UNKNOWN_KWARG,
                    fix_hint=f"'{k}' is not accepted by {op}()",
                )
                return None
            if k in numeric and _NUMBER.fullmatch(v) is None:
                self._report_error(
                    line_no, err_col,
                    f"'{k}' requires a numeric value",
                    ErrorCode.SEM_UNKNOWN_KWARG,
                    fix_hint=f"pass a number for '{k}', got '{v}'",
                )
                return None
            kwargs[k] = self._parse_value(v)
        return plain, kwargs

    def _parse_value(self, s: str) -> int | float | str:
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            return s

    def _dispatch_op(
        self, op: str, args: list[str],
        line_no: int = 0, col: int = 1,
    ) -> Optional[Value]:
        resolved: list[Value] = []
        handlers = {
            "add": lambda: self.builder.add(resolved[0], resolved[1]),
            "sub": lambda: self.builder.sub(resolved[0], resolved[1]),
            "mul": lambda: self.builder.mul(resolved[0], resolved[1]),
            "div": lambda: self.builder.div(resolved[0], resolved[1]),
            "neg": lambda: self.builder.neg(resolved[0]),
            "exp": lambda: self.builder.exp(resolved[0]),
            "relu": lambda: self.builder.relu(resolved[0]),
            "gelu": lambda: self.builder.gelu(resolved[0]),
            "dot": lambda: self.builder.dot(
                resolved[0], resolved[1],
                kwargs.get("len", kwargs.get("length", 1)),
            ),
            "matmul": lambda: self.builder.matmul(
                resolved[0], resolved[1],
                kwargs.get("rows", kwargs.get("m", 1)),
                kwargs.get("cols", kwargs.get("n", 1)),
                kwargs.get("inner", kwargs.get("k", 1)),
            ),
            "softmax": lambda: self.builder.softmax(
                resolved[0], kwargs.get("axis", -1),
            ),
            "maxpool": lambda: self.builder.maxpool(
                resolved[0],
                kwargs.get("kernel", 2),
                kwargs.get("stride", 2),
            ),
        }
        assert set(handlers) == self.supported_operations()

        # Unknown operator: report before resolving args (no side effects)
        if op not in handlers:
            raw = (
                self._raw_lines[line_no - 1]
                if 0 < line_no <= len(self._raw_lines)
                else ""
            )
            hint = suggest_spelling(raw, col) or suggest_op(op, handlers)
            self._report_error(
                line_no, col,
                f"unknown operation '{op}'",
                ErrorCode.SEM_UNKNOWN_OP,
                fix_hint=hint,
            )
            return None

        parsed = self._parse_kwargs(args, op, line_no, col)
        if parsed is None:
            return None
        plain, kwargs = parsed
        expected = _ARITY.get(op)
        if expected is not None and len(plain) != expected:
            hint = _ARITY_HINTS.get(op)
            if hint is None:
                hint = f"{op}() requires exactly {expected} arguments"
            arg_word = "argument" if expected == 1 else "arguments"
            self._report_error(
                line_no, col,
                f"{op}() expects {expected} {arg_word}, got {len(plain)}",
                ErrorCode.SEM_ARITY,
                fix_hint=hint,
            )
            return None

        resolved = [self._resolve(a) for a in plain]
        handler = handlers.get(op)
        if handler is None:
            return None
        return handler()
