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
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import Value, DataType, Program


class DSLParseError(Exception):
    pass


class DSLParser:
    """Parses a simple DSL text into an IR Program."""

    def __init__(self):
        self.builder = IRBuilder()
        self._vars: dict[str, Value] = {}
        self._loop_stack: list[str] = []

    def parse(self, text: str) -> Program:
        lines = text.strip().split("\n")
        func = self.builder.new_function("main")
        self.builder.new_block("entry")

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            self._parse_line(line)

        if not self._loop_stack:
            block = self.builder.current_block
            has_ret = block and block.instructions and block.instructions[-1].opcode.name == "RETURN"
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

        result = self._dispatch_op(op_name, args)
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

    def _parse_kwargs(self, args: list[str]) -> dict:
        kwargs = {}
        plain = []
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
        resolved = [self._resolve(a) for a in plain]

        handlers = {
            "add": lambda: self.builder.add(resolved[0], resolved[1]),
            "sub": lambda: self.builder.sub(resolved[0], resolved[1]),
            "mul": lambda: self.builder.mul(resolved[0], resolved[1]),
            "div": lambda: self.builder.div(resolved[0], resolved[1]),
            "neg": lambda: self.builder.neg(resolved[0]),
            "exp": lambda: self.builder.exp(resolved[0]),
            "relu": lambda: self.builder.relu(resolved[0]),
            "gelu": lambda: self.builder.gelu(resolved[0]),
            "dot": lambda: self.builder.dot(resolved[0], resolved[1], kwargs.get("len", kwargs.get("length", 1))),
            "matmul": lambda: self.builder.matmul(
                resolved[0], resolved[1],
                kwargs.get("rows", kwargs.get("m", 1)),
                kwargs.get("cols", kwargs.get("n", 1)),
                kwargs.get("inner", kwargs.get("k", 1)),
            ),
            "softmax": lambda: self.builder.softmax(resolved[0], kwargs.get("axis", -1)),
            "maxpool": lambda: self.builder.maxpool(resolved[0], kwargs.get("kernel", 2), kwargs.get("stride", 2)),
        }
        handler = handlers.get(op)
        if handler is None:
            raise DSLParseError(f"Unsupported op: {op}")
        return handler()
