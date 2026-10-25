"""One lexical contract for DSL validation and condition parsing."""

from __future__ import annotations

import re


DECIMAL = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
NUMBER = re.compile(rf"^[+-]?(?:{DECIMAL}|inf|nan)$", re.IGNORECASE)
INTEGER = re.compile(r"^[+-]?\d+$")
CONDITION_ATOM = rf"(?:[^\W\d]\w*|[+-]?{DECIMAL})"
CONDITION = re.compile(
    rf"^(?P<kind>if|while)\s*\(\s*(?P<lhs>{CONDITION_ATOM})\s*"
    rf"(?P<op>==|!=|<=|>=|<|>)\s*(?P<rhs>{CONDITION_ATOM})\s*\)\s*:?\s*$",
    re.UNICODE,
)


def parse_i32_constant(text: str) -> int:
    """Parse the typed const contract, without truncation or silent wrapping."""
    if INTEGER.fullmatch(text) is None:
        raise ValueError("i32 constant requires an integer value")
    # Remove leading zeros before int(): even an extremely long source
    # spelling should produce our range diagnostic, not Python's digit limit.
    digits = text.lstrip("+-").lstrip("0") or "0"
    if len(digits) > 10:
        raise ValueError("i32 constant outside signed 32-bit range")
    value = int(digits) * (-1 if text.startswith("-") else 1)
    if not -(1 << 31) <= value <= (1 << 31) - 1:
        raise ValueError("i32 constant outside signed 32-bit range")
    return value
