"""DSL error beautifier with gcc/clang-style error messages.

Provides:
- DSLSyntaxError: enriched exception with line, column, message, source_line
- format_error(): produces gcc/clang-style formatted error output
- ErrorCollector: collects multiple errors before reporting
- ANSI color support for enhanced readability

Example output::

    test.dsl:5:12: error: unexpected token 'retrun'
      5 | result = retrun(x)
        |          ^~~~~~~
    note: did you mean 'return'?
"""

from __future__ import annotations

import enum
import sys
from dataclasses import dataclass
from typing import Optional


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
    "add(": "add() requires exactly 2 arguments",
    "mul(": "mul() requires exactly 2 arguments",
    "sub(": "sub() requires exactly 2 arguments",
    "div(": "div() requires exactly 2 arguments",
    "matmul(": (
        "matmul() requires rows:, cols:, inner: kwargs "
        "(e.g., m:2, n:2, k:2)"
    ),
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
# DSLSyntaxError
# ---------------------------------------------------------------------------

@dataclass
class DSLSyntaxError(Exception):
    """Enriched syntax error with precise location information.

    Attributes:
        line: 1-based line number of the error.
        col: 1-based column number of the error.
        message: Human-readable error description.
        source_line: The content of the line containing the error.
        filename: Optional source filename for display.
        fix_hint: Optional suggestion for fixing the error.
        error_code: Optional error code string for categorization.
    """

    line: int
    col: int
    message: str
    source_line: str = ""
    filename: Optional[str] = None
    fix_hint: Optional[str] = None
    error_code: Optional[str] = None

    def __str__(self) -> str:
        return format_error(self, use_color=False)


# ---------------------------------------------------------------------------
# Error formatting functions
# ---------------------------------------------------------------------------

def _compute_suggestion(message: str, source_line: str) -> Optional[str]:
    """Heuristically compute a fix suggestion based on the error message
    and source line content.

    Args:
        message: The error message text.
        source_line: The full source line content.

    Returns:
        A human-readable suggestion string, or None.
    """
    # Check for known misspellings in source line
    words = source_line.strip().split()
    for word in words:
        clean = word.strip("(){},:=* ")
        if clean.lower() in _SUGGESTIONS:
            return _SUGGESTIONS[clean.lower()]

    # Check common patterns in message
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


def format_error(
    err: DSLSyntaxError,
    use_color: bool = True,
    context_lines: int = 0,
    show_column_marker: bool = True,
) -> str:
    """Format a DSLSyntaxError as a gcc/clang-style error message.

    Output format::

        filename:line:col: error: message
          line | source_line
                |  ^ marker
        note: fix suggestion

    Args:
        err: The DSLSyntaxError to format.
        use_color: Whether to use ANSI color codes.
        context_lines: Number of context lines to show before the error line.
        show_column_marker: Whether to show the caret/carrot marker.

    Returns:
        A formatted error string.
    """
    parts: list[str] = []

    # Build location prefix
    location = ""
    if err.filename:
        location += err.filename
    location += f":{err.line}:{err.col}: "

    # Error header
    error_tag = "error"
    if use_color:
        location = _color(location, Color.BOLD)
        error_tag = _color("error", Color.RED)
        parts.append(f"{location}{error_tag}: {err.message}")
    else:
        parts.append(f"{location}error: {err.message}")

    # Error code
    if err.error_code:
        parts[-1] += f" [{err.error_code}]"

    # Source line display
    if err.source_line:
        # Optionally show context lines before
        if context_lines > 0:
            for ctx_off in range(-context_lines, 0):
                ctx_line_num = err.line + ctx_off
                if ctx_line_num > 0:
                    ctx_indicator = (
                        " |" if context_lines > 1 else " "
                    )
                    parts.append(f"  {ctx_line_num}{ctx_indicator}")

        # Error line
        if use_color:
            line_prefix = _color(f"  {err.line} |", Color.GRAY)
            parts.append(f"{line_prefix} {err.source_line}")
        else:
            parts.append(f"  {err.line} | {err.source_line}")

        # Column marker
        if show_column_marker:
            token_len = _estimate_token_length(
                err.source_line, err.col - 1,
            )
            marker = " " * (err.col + 3) + "^"
            if use_color:
                marker = (
                    " " * (err.col + 3)
                    + _color("^", Color.GREEN)
                )
            # Add tildes to indicate token length
            marker += "~" * (max(token_len - 1, 1))
            parts.append(marker)

    # Fix suggestion
    hint = err.fix_hint or _compute_suggestion(err.message, err.source_line)
    if hint:
        if use_color:
            note_tag = _color("note", Color.CYAN)
            parts.append(f"{note_tag}: {hint}")
        else:
            parts.append(f"note: {hint}")

    return "\n".join(parts)


def _estimate_token_length(source_line: str, col_start: int) -> int:
    """Estimate the length of the token at the given column position.

    Args:
        source_line: The source line.
        col_start: 0-based column index of the start of the token.

    Returns:
        Estimated length of the token in characters.
    """
    if col_start >= len(source_line):
        return 1
    token_end = col_start
    while token_end < len(source_line) and source_line[token_end].isalnum():
        token_end += 1
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
        try:
            parser.parse(source)
        except DSLSyntaxError as e:
            collector.add(e)
        collector.report()
    """

    def __init__(
        self,
        filename: Optional[str] = None,
        use_color: bool = True,
        max_errors: int = 20,
    ):
        """Initialize the error collector.

        Args:
            filename: Source filename for display.
            use_color: Whether to use ANSI colors in output.
            max_errors: Maximum number of errors to collect before giving up.
        """
        self.filename = filename
        self.use_color = use_color
        self.max_errors = max_errors
        self._errors: list[DSLSyntaxError] = []

    @property
    def errors(self) -> list[DSLSyntaxError]:
        """Return the collected errors."""
        return list(self._errors)

    @property
    def has_errors(self) -> bool:
        """Check if any errors have been collected."""
        return len(self._errors) > 0

    @property
    def error_count(self) -> int:
        """Return the number of collected errors."""
        return len(self._errors)

    def add(self, err: DSLSyntaxError) -> None:
        """Add an error to the collector.

        Args:
            err: A DSLSyntaxError instance.
        """
        if len(self._errors) >= self.max_errors:
            if not getattr(self, "_max_error_warned", False):
                self._max_error_warned = True
                msg = (
                    f"error limit ({self.max_errors}) reached; "
                    f"further errors suppressed"
                )
                self._errors.append(DSLSyntaxError(
                    line=0, col=0, message=msg,
                    filename=self.filename,
                ))
            return
        if err.filename is None and self.filename is not None:
            err.filename = self.filename
        self._errors.append(err)

    def add_error(
        self,
        line: int,
        col: int,
        message: str,
        source_line: str = "",
        fix_hint: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> None:
        """Convenience method to add an error by components.

        Args:
            line: 1-based line number.
            col: 1-based column number.
            message: Error message.
            source_line: Source line content.
            fix_hint: Optional fix suggestion.
            error_code: Optional error code.
        """
        self.add(DSLSyntaxError(
            line=line,
            col=col,
            message=message,
            source_line=source_line,
            filename=self.filename,
            fix_hint=fix_hint,
            error_code=error_code,
        ))

    def report(self) -> str:
        """Format all collected errors and return as a string.

        Returns:
            Formatted error report string.
        """
        if not self._errors:
            return ""

        parts: list[str] = []
        if self.use_color:
            parts.append(_color(
                f"--- {len(self._errors)} error(s) found ---",
                Color.BOLD,
            ))
        else:
            parts.append(f"--- {len(self._errors)} error(s) found ---")

        for err in self._errors:
            parts.append(format_error(err, use_color=self.use_color))

        return "\n".join(parts)

    def report_and_exit(self, exit_code: int = 1) -> None:
        """Print errors and exit if any errors were collected.

        Args:
            exit_code: Process exit code to use.
        """
        if self._errors:
            print(self.report(), file=sys.stderr)
            sys.exit(exit_code)

    def clear(self) -> None:
        """Clear all collected errors."""
        self._errors.clear()


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
) -> DSLSyntaxError:
    """Factory function to create a DSLSyntaxError.

    Args:
        line: 1-based line number.
        col: 1-based column number.
        message: Error description.
        source_line: Content of the erroneous line.
        filename: Source filename.
        fix_hint: Fix suggestion.
        error_code: Error code.

    Returns:
        A DSLSyntaxError instance.
    """
    return DSLSyntaxError(
        line=line,
        col=col,
        message=message,
        source_line=source_line,
        filename=filename,
        fix_hint=fix_hint,
        error_code=error_code,
    )
