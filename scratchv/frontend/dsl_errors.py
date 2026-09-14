"""DSL error beautifier with gcc/clang-style error messages.

Provides:
- DSLParseError: backward-compatible base class for all DSL parse errors
- DSLSyntaxError: enriched exception with line, column, message, source_line
- ErrorCode: stable error-code constants (E1xx lexical / E2xx syntax / E3xx semantic)
- format_error(): produces gcc/clang-style formatted error output
- render_error(): renders an error with color chosen from the target stream
- ErrorCollector: collects multiple errors before reporting
- ANSI color support for enhanced readability

Example output::

    test.dsl:5:12: error[E301]: unexpected token 'retrun'
      5 | result = retrun(x)
        |          ^~~~~~
    note: did you mean 'return'?
"""

from __future__ import annotations

import difflib
import enum
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, Optional, TextIO


# ---------------------------------------------------------------------------
# ANSI color codes
# ---------------------------------------------------------------------------

class Color(enum.Enum):
    """ANSI terminal color codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"


def _color(text: str, color: Color) -> str:
    """Wrap text with ANSI color codes."""
    return f"{color.value}{text}{Color.RESET.value}"


# ---------------------------------------------------------------------------
# Error-code constants (stable; only append, never reuse)
# ---------------------------------------------------------------------------

class ErrorCode:
    """Stable DSL diagnostic error codes.

    Encoding: ``E1xx`` lexical, ``E2xx`` syntax, ``E3xx`` semantic.
    """

    # Lexical
    LEX_ILLEGAL_CHAR = "E101"
    # Syntax
    SYN_INVALID_STATEMENT = "E201"
    SYN_INVALID_CONDITION = "E202"
    SYN_MISSING_TERMINATOR = "E203"
    SYN_STRAY_TERMINATOR = "E204"
    SYN_NESTED_CALL = "E205"
    # Semantic
    SEM_UNKNOWN_OP = "E301"
    SEM_ARITY = "E302"
    SEM_UNKNOWN_KWARG = "E304"


# ---------------------------------------------------------------------------
# Fix suggestion database
# ---------------------------------------------------------------------------

_SUGGESTIONS: dict[str, str] = {
    "retrun": "did you mean 'return'?",
    "endiff": "did you mean 'endif'?",
    "endwhie": "did you mean 'endwhile'?",
    "matmal": "did you mean 'matmul'?",
    "reul": "did you mean 'relu'?",
    "geul": "did you mean 'gelu'?",
    "sofmax": "did you mean 'softmax'?",
    "maxpol": "did you mean 'maxpool'?",
    "enfor": "did you mean 'endfor'?",
    "ednfor": "did you mean 'endfor'?",
}

# Arity hints keyed by bare operator name (served by E302).
_ARITY_HINTS: dict[str, str] = {
    "add": "add() requires exactly 2 arguments",
    "sub": "sub() requires exactly 2 arguments",
    "mul": "mul() requires exactly 2 arguments",
    "div": "div() requires exactly 2 arguments",
    "neg": "neg() requires exactly 1 argument",
    "exp": "exp() requires exactly 1 argument",
    "relu": "relu() requires exactly 1 argument",
    "gelu": "gelu() requires exactly 1 argument",
    "dot": "dot() requires exactly 2 arguments",
    "matmul": "matmul() requires exactly 2 arguments",
    "softmax": "softmax() requires exactly 1 argument",
    "maxpool": "maxpool() requires exactly 1 argument",
}

_COMMON_FIXES: dict[str, str] = {
    "missing_right_paren": "missing closing ')'",
    "missing_left_paren": "missing opening '('",
    "missing_colon": "if/while statement requires ':' after condition",
    "unexpected_keyword": "unexpected keyword -- check spelling",
    "unterminated_block": (
        "missing 'endif', 'endwhile', or 'endfor'"
    ),
    "nested_block_error": "nested block not properly closed",
    "undefined_variable": "variable used before assignment",
    "invalid_operator": (
        "unsupported comparison operator -- use ==, !=, <, >, <=, >="
    ),
}


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class DSLParseError(Exception):
    """Base class for DSL parse errors (backward-compatible catch-all)."""


@dataclass(init=False)
class DSLSyntaxError(DSLParseError):
    """Enriched syntax error with precise location information.

    Attributes:
        line: 1-based line number of the error.
        col: 1-based column number of the error.
        message: Human-readable error description.
        source_line: The content of the line containing the error.
        filename: Optional source filename for display.
        fix_hint: Optional suggestion for fixing the error.
        error_code: Optional error code string for categorization.
        end_col: Optional 1-based column just past the erroneous span.
        suggestion: Read/write alias of ``fix_hint``.
    """

    line: int
    col: int
    message: str
    source_line: str = ""
    filename: Optional[str] = None
    fix_hint: Optional[str] = None
    error_code: Optional[str] = None
    end_col: Optional[int] = None

    def __init__(
        self,
        line: int,
        col: int,
        message: str,
        source_line: str = "",
        filename: Optional[str] = None,
        fix_hint: Optional[str] = None,
        error_code: Optional[str] = None,
        *,
        suggestion: Optional[str] = None,
        end_col: Optional[int] = None,
    ) -> None:
        if fix_hint is None:
            fix_hint = suggestion
        self.line = line
        self.col = col
        self.message = message
        self.source_line = source_line
        self.filename = filename
        self.fix_hint = fix_hint
        self.error_code = error_code
        self.end_col = end_col
        Exception.__init__(self, message)

    @property
    def suggestion(self) -> Optional[str]:
        """Read/write alias for ``fix_hint``."""
        return self.fix_hint

    @suggestion.setter
    def suggestion(self, value: Optional[str]) -> None:
        self.fix_hint = value

    def __str__(self) -> str:
        return format_error(self, use_color=False)


# ---------------------------------------------------------------------------
# Suggestion helpers
# ---------------------------------------------------------------------------

_IDENT_RE = re.compile(r"[A-Za-z_]\w*")


def _identifier_at(source_line: str, col: int) -> Optional[str]:
    """Return the identifier containing the 1-based column ``col``."""
    if not source_line or col <= 0:
        return None
    idx = col - 1
    for m in _IDENT_RE.finditer(source_line):
        if m.start() <= idx < m.end():
            return m.group(0)
    return None


def suggest_spelling(
    source_line: str,
    col: int = 0,
) -> Optional[str]:
    """Suggest a spelling fix from ``_SUGGESTIONS``.

    The identifier covering ``col`` (1-based) is checked first, then the
    whole line is scanned. Matching is case-insensitive and exact.

    Returns:
        A suggestion such as ``"did you mean 'return'?"`` or ``None``.
    """
    if not source_line:
        return None
    if col > 0:
        token = _identifier_at(source_line, col)
        if token is not None:
            hint = _SUGGESTIONS.get(token.lower())
            if hint:
                return hint
    for token in _IDENT_RE.findall(source_line):
        hint = _SUGGESTIONS.get(token.lower())
        if hint:
            return hint
    return None


def suggest_op(
    op: str,
    candidates: Iterable[str],
    cutoff: float = 0.6,
) -> Optional[str]:
    """Suggest the closest known operator via ``difflib``.

    Returns:
        A suggestion such as ``"did you mean 'mul'?"`` or ``None``.
    """
    matches = difflib.get_close_matches(op, list(candidates), n=1, cutoff=cutoff)
    if matches:
        return f"did you mean '{matches[0]}'?"
    return None


def _compute_suggestion(
    message: str,
    source_line: str,
    error_code: Optional[str] = None,
) -> Optional[str]:
    """Heuristically compute a fix suggestion.

    Priority: error-code specific hints, spelling library, then keyword-based
    common fixes. Returns ``None`` when nothing applies.
    """
    if error_code == ErrorCode.SEM_ARITY:
        m = re.match(r"(\w+)\(\)\s+expects\s+", message)
        if m:
            op = m.group(1)
            hint = _ARITY_HINTS.get(op)
            if hint:
                return hint
            n = re.search(r"expects\s+(\d+)", message)
            if n:
                return f"{op}() requires exactly {n.group(1)} arguments"

    hint = suggest_spelling(source_line)
    if hint:
        return hint

    msg_lower = message.lower()
    if "unterminated" in msg_lower or "missing end" in msg_lower:
        return _COMMON_FIXES["unterminated_block"]
    if "unexpected" in msg_lower:
        return _COMMON_FIXES["unexpected_keyword"]
    if "undefined" in msg_lower or "not defined" in msg_lower:
        return _COMMON_FIXES["undefined_variable"]
    if "paren" in msg_lower or "(" in msg_lower:
        if "missing" in msg_lower:
            return _COMMON_FIXES["missing_right_paren"]
    if "operator" in msg_lower:
        return _COMMON_FIXES["invalid_operator"]

    return None


# ---------------------------------------------------------------------------
# Error formatting functions
# ---------------------------------------------------------------------------

def format_error(
    err: DSLSyntaxError,
    use_color: bool = True,
    context_lines: int = 0,
    show_column_marker: bool = True,
    source: Optional[str] = None,
) -> str:
    """Format a DSLSyntaxError as a gcc/clang-style error message.

    Output format::

        filename:line:col: error[code]: message
          line | source_line
               |  ^ marker
        note: fix suggestion

    Args:
        err: The DSLSyntaxError to format.
        use_color: Whether to use ANSI color codes.
        context_lines: Number of context lines shown before the error line
            (only rendered when ``source`` is provided).
        show_column_marker: Whether to show the caret marker.
        source: Full source text used to render real context lines.

    Returns:
        A formatted error string (no trailing newline).
    """
    parts: list[str] = []

    # Header: location + error label + message
    location = f"{err.filename or '<dsl>'}:{err.line}:{err.col}: "
    error_label = f"error[{err.error_code}]" if err.error_code else "error"
    if use_color:
        parts.append(
            f"{_color(location, Color.BOLD)}"
            f"{_color(error_label, Color.RED)}: {err.message}"
        )
    else:
        parts.append(f"{location}{error_label}: {err.message}")

    line_str = str(err.line)
    gutter_src = f"  {line_str} | "
    gutter_mark = " " * (3 + len(line_str)) + "| "
    assert len(gutter_src) == len(gutter_mark)

    # Optional context lines (only with real source text)
    if context_lines > 0 and source:
        src_lines = source.split("\n")
        if 0 < err.line <= len(src_lines):
            start = max(1, err.line - context_lines)
            for n in range(start, err.line):
                ctx_text = src_lines[n - 1].expandtabs(4)
                if use_color:
                    parts.append(
                        f"{_color(f'  {n} |', Color.GRAY)} {ctx_text}"
                    )
                else:
                    parts.append(f"  {n} | {ctx_text}")

    # Source line display
    if err.source_line:
        display_source = err.source_line.expandtabs(4)
        if use_color:
            parts.append(
                f"{_color(f'  {line_str} |', Color.GRAY)} {display_source}"
            )
        else:
            parts.append(f"{gutter_src}{display_source}")

        # Column marker, aligned with the expanded source display
        if show_column_marker:
            raw_start = max(err.col - 1, 0)
            display_start = len(err.source_line[:raw_start].expandtabs(4))
            if err.end_col is not None:
                raw_end = max(err.end_col - 1, raw_start + 1)
                token_len = max(
                    len(err.source_line[:raw_end].expandtabs(4))
                    - display_start,
                    1,
                )
            else:
                token_len = _estimate_token_length(err.source_line, raw_start)
            caret = _color("^", Color.GREEN) if use_color else "^"
            marker = (
                gutter_mark
                + " " * display_start
                + caret
                + "~" * max(token_len - 1, 1)
            )
            parts.append(marker)

    # Fix suggestion (explicit hint wins over heuristic)
    hint = err.fix_hint or _compute_suggestion(
        err.message, err.source_line, err.error_code,
    )
    if hint:
        if use_color:
            parts.append(f"{_color('note', Color.CYAN)}: {hint}")
        else:
            parts.append(f"note: {hint}")

    return "\n".join(parts)


def render_error(
    err: DSLSyntaxError,
    *,
    stream: TextIO,
    use_color: Optional[bool] = None,
) -> str:
    """Render an error, selecting color from the destination stream."""
    if use_color is None:
        is_tty = bool(getattr(stream, "isatty", lambda: False)())
        use_color = is_tty and "NO_COLOR" not in os.environ
    return format_error(err, use_color=use_color)


def _estimate_token_length(source_line: str, col_start: int) -> int:
    """Estimate the length of the token at the given column position.

    Args:
        source_line: The source line.
        col_start: 0-based column index of the start of the token.

    Returns:
        Estimated length of the token in characters (at least 1).
    """
    if col_start < 0 or col_start >= len(source_line):
        return 1
    token_end = col_start
    while token_end < len(source_line):
        ch = source_line[token_end]
        if ch.isalnum() or ch == "_":
            token_end += 1
        else:
            break
    return max(token_end - col_start, 1)


# ---------------------------------------------------------------------------
# ErrorCollector
# ---------------------------------------------------------------------------

class ErrorCollector:
    """Collects multiple DSLSyntaxErrors before reporting them all at once.

    This allows the parser to continue after the first error to find more
    errors, providing a better developer experience.

    Usage::

        collector = ErrorCollector(filename="test.dsl")
        program = parser.parse(source, filename="test.dsl", collector=collector)
        if collector.has_errors:
            print(collector.report())
    """

    def __init__(
        self,
        filename: Optional[str] = None,
        use_color: bool = True,
        max_errors: int = 20,
        source: Optional[str] = None,
        context_lines: int = 0,
    ):
        """Initialize the error collector.

        Args:
            filename: Source filename for display.
            use_color: Whether to use ANSI colors in output.
            max_errors: Maximum number of stored errors (must be >= 1).
            source: Full source text used for context rendering.
            context_lines: Context lines shown before each error line.

        Raises:
            ValueError: If ``max_errors`` is less than 1 (a zero limit would
                silently report "no errors" while suppressing everything).
        """
        if max_errors < 1:
            raise ValueError(
                f"max_errors must be >= 1, got {max_errors}"
            )
        self.filename = filename
        self.use_color = use_color
        self.max_errors = max_errors
        self.source = source
        self.context_lines = context_lines
        self._errors: list[DSLSyntaxError] = []
        self._keys: set[tuple[object, ...]] = set()
        self.limit_reached = False
        self._suppressed: int = 0

    @property
    def errors(self) -> list[DSLSyntaxError]:
        """Return the collected errors, de-duplicated and position-sorted."""
        return sorted(
            self._errors,
            key=lambda err: (err.line, err.col, err.error_code or ""),
        )

    @property
    def has_errors(self) -> bool:
        """Check if any errors have been collected."""
        return len(self._errors) > 0

    @property
    def error_count(self) -> int:
        """Return the number of collected (non-suppressed) errors."""
        return len(self._errors)

    @property
    def suppressed_count(self) -> int:
        """Return the number of errors dropped after ``max_errors``."""
        return self._suppressed

    def add(self, err: DSLSyntaxError) -> None:
        """Add an error to the collector.

        Duplicates by ``(filename, line, col, error_code, message)`` are
        ignored. Once ``max_errors`` is reached, further unique errors are
        only counted in ``suppressed_count`` and flip ``limit_reached``.
        """
        if err.filename is None and self.filename is not None:
            err.filename = self.filename
        key = (
            err.filename, err.line, err.col, err.error_code, err.message,
        )
        if key in self._keys:
            return
        self._keys.add(key)
        if len(self._errors) >= self.max_errors:
            self.limit_reached = True
            self._suppressed += 1
            return
        self._errors.append(err)

    def add_error(
        self,
        line: int,
        col: int,
        message: str,
        source_line: str = "",
        fix_hint: Optional[str] = None,
        error_code: Optional[str] = None,
        end_col: Optional[int] = None,
    ) -> None:
        """Convenience method to add an error by components."""
        self.add(DSLSyntaxError(
            line=line,
            col=col,
            message=message,
            source_line=source_line,
            filename=self.filename,
            fix_hint=fix_hint,
            error_code=error_code,
            end_col=end_col,
        ))

    def report(self) -> str:
        """Format all collected errors and return as a string.

        Returns:
            Formatted error report (``""`` when there are no errors).
        """
        if not self._errors:
            return ""

        header = f"--- {len(self._errors)} error(s) found ---"
        if self.use_color:
            header = _color(header, Color.BOLD)
        parts: list[str] = [header]

        for err in self.errors:
            parts.append(format_error(
                err,
                use_color=self.use_color,
                context_lines=self.context_lines,
                source=self.source,
            ))

        if self.limit_reached:
            note = (
                f"note: error limit ({self.max_errors}) reached; "
                f"{self._suppressed} further errors suppressed"
            )
            if self.use_color:
                note = _color(note, Color.CYAN)
            parts.append(note)

        return "\n".join(parts)

    def report_and_exit(self, exit_code: int = 1) -> None:
        """Print errors and exit if any errors were collected."""
        if self._errors:
            print(self.report(), file=sys.stderr)
            sys.exit(exit_code)

    def clear(self) -> None:
        """Clear all collected errors and reset limit/dedup state."""
        self._errors.clear()
        self._keys.clear()
        self.limit_reached = False
        self._suppressed = 0


# ---------------------------------------------------------------------------
# Helper: quickly create an error from a parse context
# ---------------------------------------------------------------------------

def make_error(
    line: int,
    col: int,
    message: str,
    source_line: str = "",
    filename: Optional[str] = None,
    fix_hint: Optional[str] = None,
    error_code: Optional[str] = None,
    end_col: Optional[int] = None,
) -> DSLSyntaxError:
    """Factory function to create a DSLSyntaxError."""
    return DSLSyntaxError(
        line=line,
        col=col,
        message=message,
        source_line=source_line,
        filename=filename,
        fix_hint=fix_hint,
        error_code=error_code,
        end_col=end_col,
    )
