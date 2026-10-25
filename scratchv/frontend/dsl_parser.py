"""Simple DSL parser for testing IR and backend without ONNX dependency.

DSL syntax (one operation per line):
  # comment
  name = add(a, b)
  name = mul(a, b)
  name = const(1, dtype:i32)
  name = rem(a, b)  # mod(a, b) is an alias
  name = min(a, b)
  name = max(a, b)
  name = abs(a)
  name = sqrt(a)
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
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, Value, Program
from scratchv.frontend.dsl_errors import DSLParseError, ErrorCollector
from scratchv.frontend.dsl_validator import DSLValidator
from scratchv.frontend.dsl_lexical import parse_i32_constant


class DSLParser:
    """Parses a simple DSL text into an IR Program."""

    def __init__(self):
        self.builder = IRBuilder()
        self._vars: dict[str, Value] = {}
        self._loop_stack: list[str] = []

    def validate(
        self, text: str, *, filename: str | None = None,
        max_errors: int = 20,
    ) -> ErrorCollector:
        return DSLValidator().validate(
            text, filename=filename, max_errors=max_errors,
        )

    @staticmethod
    def supported_operations() -> set[str]:
        return {
            "const", "add", "sub", "mul", "div", "rem", "mod", "min",
            "max", "abs", "neg", "exp", "relu", "gelu", "dot", "matmul",
            "sqrt", "softmax", "maxpool",
        }

    def parse(self, text: str, *, filename: str | None = None) -> Program:
        collector = self.validate(text, filename=filename)
        if collector.has_errors:
            raise collector.errors[0]

        self.builder = IRBuilder()
        self._vars = {}
        self._loop_stack = []
        lines = text.strip().split("\n")
        self.builder.new_function("main")
        self.builder.new_block("entry")

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            self._parse_line(line)

        if not self._loop_stack:
            block = self.builder.current_block
            if block and block.instructions:
                has_ret = block.instructions[-1].opcode.name == "RETURN"
            else:
                has_ret = False
            if not has_ret:
                self.builder.ret()
        return self.builder.program

    def _parse_line(self, line: str) -> None:
        # for i = start, end
        m = re.match(r"for\s+(\w+)\s*=\s*(\d+)\s*,\s*(\d+)", line)
        if m:
            iv = self.builder.for_loop(int(m.group(2)), int(m.group(3)))
            self._vars[m.group(1)] = iv
            self._loop_stack.append(m.group(1))
            return

        if line == "endfor":
            if not self._loop_stack:
                raise DSLParseError("endfor without matching for")
            self._loop_stack.pop()
            self.builder.endfor()
            return

        # return [var]
        m = re.match(r"return\s*(\S+)", line)
        if m:
            val = self._resolve(m.group(1))
            self.builder.ret(val)
            return

        # name = op(args)
        m = re.match(r"(\w+)\s*=\s*(\w+)\((.+)\)", line)
        if not m:
            raise DSLParseError(f"Cannot parse line: {line}")

        dest_name = m.group(1)
        op_name = m.group(2)
        args_text = m.group(3)
        args = [a.strip() for a in args_text.split(",") if a.strip()]

        try:
            result = self._dispatch_op(op_name, args)
        except DSLParseError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise DSLParseError(str(exc)) from exc
        self._vars[dest_name] = result

    def _resolve(self, name: str) -> Value:
        """Resolve a variable name or literal to a Value."""
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
            self, args: list[str],
    ) -> tuple[list[str], dict[str, int | float | str]]:
        kwargs: dict[str, int | float | str] = {}
        plain: list[str] = []
        for a in args:
            if ":" in a:
                k, v = a.split(":", 1)
                kwargs[k.strip()] = self._parse_value(v.strip())
            else:
                plain.append(a)
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

    def _dispatch_op(self, op: str, args: list[str]) -> Value:
        plain, kwargs = self._parse_kwargs(args)

        if op == "const":
            dtype_name = kwargs.get("dtype", "f32")
            dtype = {
                "i32": DataType.INT32,
                "f32": DataType.FLOAT32,
                "f64": DataType.FLOAT64,
            }[dtype_name]
            try:
                value = (
                    parse_i32_constant(plain[0])
                    if dtype is DataType.INT32
                    else float(plain[0])
                )
            except ValueError as exc:
                raise DSLParseError(
                    f"Invalid {dtype.value} constant: {exc} ({plain[0]})"
                ) from exc
            return self.builder.load_const(value, dtype=dtype)

        resolved = [self._resolve(a) for a in plain]

        handlers = {
            "const": lambda: self.builder.load_const(0.0),  # handled above
            "add": lambda: self.builder.add(resolved[0], resolved[1]),
            "sub": lambda: self.builder.sub(resolved[0], resolved[1]),
            "mul": lambda: self.builder.mul(resolved[0], resolved[1]),
            "div": lambda: self.builder.div(resolved[0], resolved[1]),
            "rem": lambda: self.builder.rem(resolved[0], resolved[1]),
            "mod": lambda: self.builder.rem(resolved[0], resolved[1]),
            "min": lambda: self.builder.min(resolved[0], resolved[1]),
            "max": lambda: self.builder.max(resolved[0], resolved[1]),
            "abs": lambda: self.builder.abs(resolved[0]),
            "sqrt": lambda: self.builder.sqrt(resolved[0]),
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
        handler = handlers.get(op)
        if handler is None:
            raise DSLParseError(f"Unsupported op: {op}")
        return handler()
