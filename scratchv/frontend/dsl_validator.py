"""Source-preserving validation shared by ScratchV DSL parsers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

from scratchv.frontend.dsl_errors import ErrorCollector


@dataclass(frozen=True)
class SourceBuffer:
    """Immutable DSL source with one-based physical-line access."""

    text: str
    filename: str = "<dsl>"

    @property
    def lines(self) -> tuple[str, ...]:
        return tuple(self.text.replace("\r\n", "\n").replace("\r", "\n").split("\n"))

    @property
    def line_count(self) -> int:
        return len(self.lines)

    def line_text(self, line: int) -> str:
        if line < 1 or line > self.line_count:
            return ""
        return self.lines[line - 1]


@dataclass(frozen=True)
class OpSignature:
    positional: int
    optional_kwargs: frozenset[str] = frozenset()
    required_kwargs: frozenset[str] = frozenset()
    numeric_kwargs: frozenset[str] = frozenset()


OP_SIGNATURES: dict[str, OpSignature] = {
    "add": OpSignature(2),
    "sub": OpSignature(2),
    "mul": OpSignature(2),
    "div": OpSignature(2),
    "neg": OpSignature(1),
    "exp": OpSignature(1),
    "relu": OpSignature(1),
    "gelu": OpSignature(1),
    "dot": OpSignature(2, frozenset({"len", "length"}), numeric_kwargs=frozenset({"len", "length"})),
    "matmul": OpSignature(2, frozenset({"rows", "cols", "inner", "m", "n", "k"}), numeric_kwargs=frozenset({"rows", "cols", "inner", "m", "n", "k"})),
    "softmax": OpSignature(1, frozenset({"axis"}), numeric_kwargs=frozenset({"axis"})),
    "maxpool": OpSignature(1, frozenset({"kernel", "stride"}), numeric_kwargs=frozenset({"kernel", "stride"})),
}


_IDENTIFIER = re.compile(r"^(?!\d)\w+$", re.UNICODE)
_ASSIGNMENT = re.compile(
    r"^(?P<dest>[^\s=]+)\s*=\s*(?P<op>[A-Za-z_]\w*)\s*"
    r"\((?P<args>.*)\)\s*$",
    re.UNICODE,
)
_FOR = re.compile(
    r"^for\s+(?P<var>[^\s=]+)\s*=\s*(?P<start>\d+)\s*,\s*"
    r"(?P<end>\d+)\s*$",
)
_CONDITION = re.compile(
    r"^(?P<kind>if|while)\s*\(\s*.+?\s*"
    r"(?:==|!=|<=|>=|<|>)\s*.+?\s*\)\s*:?\s*$"
)
_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


@dataclass
class BlockFrame:
    kind: Literal["if", "while", "for"]
    line: int
    col: int
    source_line: str
    saw_else: bool = False


class DSLValidator:
    """Validate DSL syntax without constructing IR."""

    def __init__(self, *, extended: bool = False):
        self.extended = extended

    def validate(
        self,
        text: str,
        *,
        filename: Optional[str] = None,
        max_errors: int = 20,
    ) -> ErrorCollector:
        display_name = filename or "<dsl>"
        source = SourceBuffer(text, display_name)
        collector = ErrorCollector(
            filename=display_name, use_color=False, max_errors=max_errors,
        )
        stack: list[BlockFrame] = []

        for line_no, raw_line in enumerate(source.lines, start=1):
            if collector.limit_reached:
                break
            statement = self._statement(raw_line)
            if not statement:
                continue
            col = len(raw_line) - len(raw_line.lstrip()) + 1

            if statement.startswith("for "):
                match = _FOR.fullmatch(statement)
                if match is None:
                    self._add(collector, line_no, col, raw_line, "E100", "cannot parse for statement")
                elif not _IDENTIFIER.fullmatch(match.group("var")):
                    self._add(collector, line_no, col + 4, raw_line, "E103", "invalid loop variable")
                else:
                    stack.append(BlockFrame("for", line_no, col, raw_line))
                continue

            if statement in {"endfor", "endif", "endwhile"}:
                self._close_block(statement, line_no, col, raw_line, stack, collector)
                continue

            if statement in {"else", "else:"}:
                self._else(line_no, col, raw_line, stack, collector)
                continue

            if statement.startswith(("if ", "while ")):
                if not self.extended:
                    self._add(collector, line_no, col, raw_line, "E100", "cannot parse statement")
                    continue
                kind = "if" if statement.startswith("if ") else "while"
                if _CONDITION.fullmatch(statement) is None:
                    code = "E101" if "(" not in statement or ")" not in statement else "E100"
                    hint = "add matching parentheses around the condition" if code == "E101" else None
                    self._add(collector, line_no, col, raw_line, code, f"invalid {kind} condition", fix_hint=hint)
                else:
                    stack.append(BlockFrame(kind, line_no, col, raw_line))
                continue

            if statement.startswith("return"):
                if re.fullmatch(r"return\s+\S+", statement) is None:
                    self._add(collector, line_no, col, raw_line, "E100", "return requires a value")
                continue

            self._validate_assignment(statement, line_no, col, raw_line, collector)

        for frame in stack:
            if collector.limit_reached:
                break
            expected = {"if": "endif", "while": "endwhile", "for": "endfor"}[frame.kind]
            self._add(
                collector, frame.line, frame.col, frame.source_line, "E111",
                f"unterminated {frame.kind} block",
                fix_hint=f"add missing '{expected}'",
            )
        return collector

    @staticmethod
    def _statement(raw_line: str) -> str:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            return ""
        comment = stripped.find(" #")
        if comment >= 0:
            stripped = stripped[:comment].rstrip()
        return stripped

    def _validate_assignment(
        self, statement: str, line: int, col: int, raw: str,
        collector: ErrorCollector,
    ) -> None:
        left_count = statement.count("(")
        right_count = statement.count(")")
        if left_count != right_count:
            if left_count < right_count:
                depth = 0
                unmatched_right = 0
                for index, char in enumerate(raw):
                    if char == "(":
                        depth += 1
                    elif char == ")":
                        if depth == 0:
                            unmatched_right = index
                            break
                        depth -= 1
                error_col = unmatched_right + 1
                message = "missing opening '('"
                hint = "add the missing '('"
            else:
                openings: list[int] = []
                for index, char in enumerate(raw):
                    if char == "(":
                        openings.append(index)
                    elif char == ")" and openings:
                        openings.pop()
                error_col = openings[0] + 1
                message = "missing closing ')'"
                hint = "add the missing ')'"
            self._add(
                collector, line, error_col, raw, "E101", message,
                fix_hint=hint,
            )
            return
        match = _ASSIGNMENT.fullmatch(statement)
        if match is None:
            if "=" in statement and "(" in statement and ")" not in statement:
                paren_col = raw.find("(") + 1
                self._add(collector, line, paren_col, raw, "E101", "missing closing ')'", fix_hint="add the missing ')'")
            else:
                hint = "did you mean 'return'?" if statement.split(maxsplit=1)[0].lower() == "retrun" else None
                self._add(collector, line, col, raw, "E100", "cannot parse statement", fix_hint=hint)
            return

        dest = match.group("dest")
        if _IDENTIFIER.fullmatch(dest) is None:
            self._add(collector, line, raw.find(dest) + 1, raw, "E103", f"invalid identifier '{dest}'", end_col=raw.find(dest) + len(dest) + 1)
            return

        op = match.group("op")
        op_col = raw.find(op, raw.find("=")) + 1
        signature = OP_SIGNATURES.get(op)
        if signature is None:
            hint = "did you mean 'add'?" if op == "ad" else None
            self._add(collector, line, op_col, raw, "E200", f"unsupported operation '{op}'", fix_hint=hint, end_col=op_col + len(op))
            return

        args_text = match.group("args")
        args = [item.strip() for item in args_text.split(",") if item.strip()]
        positional: list[str] = []
        kwargs: dict[str, str] = {}
        args_col = raw.find("(", op_col - 1) + 2
        for arg in args:
            if ":" not in arg:
                positional.append(arg)
                continue
            key, value = (part.strip() for part in arg.split(":", 1))
            if key in kwargs or key not in signature.optional_kwargs | signature.required_kwargs:
                self._add(collector, line, raw.find(key, args_col - 1) + 1, raw, "E202", f"invalid keyword argument '{key}'")
                return
            if not value:
                self._add(collector, line, raw.find(key, args_col - 1) + len(key) + 2, raw, "E203", f"missing value for '{key}'")
                return
            if key in signature.numeric_kwargs and _NUMBER.fullmatch(value) is None:
                self._add(collector, line, raw.find(value, args_col - 1) + 1, raw, "E203", f"'{key}' requires a numeric value")
                return
            kwargs[key] = value

        if len(positional) != signature.positional:
            self._add(
                collector, line, args_col, raw, "E201",
                f"operation '{op}' expects {signature.positional} positional argument(s), got {len(positional)}",
            )
            return
        missing = signature.required_kwargs - kwargs.keys()
        if missing:
            name = sorted(missing)[0]
            self._add(collector, line, args_col, raw, "E202", f"missing keyword argument '{name}'")

    def _close_block(
        self, closer: str, line: int, col: int, raw: str,
        stack: list[BlockFrame], collector: ErrorCollector,
    ) -> None:
        expected_kind = {"endif": "if", "endwhile": "while", "endfor": "for"}[closer]
        if not stack:
            self._add(collector, line, col, raw, "E110", f"'{closer}' without matching {expected_kind}")
            return
        if stack[-1].kind == expected_kind:
            stack.pop()
            return
        expected_closer = {"if": "endif", "while": "endwhile", "for": "endfor"}[stack[-1].kind]
        self._add(collector, line, col, raw, "E110", f"found '{closer}', expected '{expected_closer}'")
        matching = next((i for i in range(len(stack) - 1, -1, -1) if stack[i].kind == expected_kind), None)
        if matching is None:
            stack.pop()
        else:
            del stack[matching:]

    def _else(
        self, line: int, col: int, raw: str,
        stack: list[BlockFrame], collector: ErrorCollector,
    ) -> None:
        if not stack or stack[-1].kind != "if":
            self._add(collector, line, col, raw, "E112", "else without matching if")
        elif stack[-1].saw_else:
            self._add(collector, line, col, raw, "E112", "duplicate else in if block")
        else:
            stack[-1].saw_else = True

    @staticmethod
    def _add(
        collector: ErrorCollector, line: int, col: int, source_line: str,
        code: str, message: str, *, fix_hint: Optional[str] = None,
        end_col: Optional[int] = None,
    ) -> None:
        collector.add_error(
            line, max(col, 1), message, source_line=source_line,
            fix_hint=fix_hint, error_code=code, end_col=end_col,
        )
