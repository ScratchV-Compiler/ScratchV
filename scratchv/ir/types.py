"""Intermediate Representation (IR) type definitions.

Uses a three-address code style representation with explicit basic blocks,
suitable for direct translation to RISC-V assembly.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional, Union


class OpCode(enum.Enum):
    """All supported IR operation codes."""

    # Arithmetic (binary)
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    # Arithmetic (unary)
    NEG = "neg"
    EXP = "exp"
    # Memory / data
    LOAD = "load"
    STORE = "store"
    LOAD_CONST = "load_const"
    ALLOCA = "alloca"
    # Control flow
    FOR = "for"
    ENDFOR = "endfor"
    BR = "br"
    BR_IF = "br_if"
    LABEL = "label"
    RETURN = "return"
    # Neural-network ops
    MATMUL = "matmul"
    RELU = "relu"
    MAXPOOL = "maxpool"
    SOFTMAX = "softmax"
    GELU = "gelu"
    DOT = "dot"
    CONV = "conv"
    GEMM = "gemm"
    SIGMOID = "sigmoid"
    # Shape / data movement
    TRANSPOSE = "transpose"
    RESHAPE = "reshape"
    CONCAT = "concat"

    def is_arith(self) -> bool:
        return self in (OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV)

    def is_nn(self) -> bool:
        return self in (
            OpCode.MATMUL,
            OpCode.RELU,
            OpCode.MAXPOOL,
            OpCode.SOFTMAX,
            OpCode.GELU,
            OpCode.DOT,
            OpCode.EXP,
            OpCode.CONV,
            OpCode.GEMM,
            OpCode.SIGMOID,
        )

    def is_control_flow(self) -> bool:
        return self in (
            OpCode.FOR, OpCode.ENDFOR, OpCode.BR,
            OpCode.BR_IF, OpCode.RETURN)


class DataType(enum.Enum):
    """Element data types."""
    FLOAT32 = "f32"
    INT32 = "i32"
    FLOAT64 = "f64"
    INT64 = "i64"

    @staticmethod
    def from_onnx(elem_type: int) -> DataType:
        mapping = {
            1: DataType.FLOAT32, 6: DataType.INT32,
            7: DataType.INT64, 11: DataType.FLOAT64,
        }
        return mapping.get(elem_type, DataType.FLOAT32)


@dataclass
class Value:
    """An SSA-like typed value (instruction result or function arg)."""
    name: str
    dtype: DataType = DataType.FLOAT32
    is_constant: bool = False
    const_value: Optional[Union[float, int]] = None
    shape: tuple[int, ...] = ()


@dataclass
class Instruction:
    """A single three-address-code instruction."""
    opcode: OpCode
    dest: Optional[Value] = None
    operands: list[Value] = field(default_factory=list)
    attrs: dict[str, object] = field(default_factory=dict)
    # For control flow: target block name or loop bounds
    target: Optional[str] = None

    def __repr__(self) -> str:
        parts = [self.opcode.value]
        if self.dest:
            parts.append(f"${self.dest.name}")
        for v in self.operands:
            parts.append(f"${v.name}")
        if self.target:
            parts.append(f"-> {self.target}")
        if self.attrs:
            for attr_k, attr_v in self.attrs.items():
                parts.append(f"[{attr_k}={attr_v}]")
        return " ".join(parts)


class BasicBlock:
    """A basic block: a straight-line sequence of instructions with a label."""

    def __init__(self, name: str):
        self.name = name
        self.instructions: list[Instruction] = []
        self.phi_nodes: list[Instruction] = []

    def add(self, instr: Instruction) -> None:
        self.instructions.append(instr)

    def __repr__(self) -> str:
        lines = [f".{self.name}:"]
        for inst in self.instructions:
            lines.append(f"  {inst}")
        return "\n".join(lines)


@dataclass
class Function:
    """An IR function: a collection of basic blocks forming a CFG."""
    name: str
    params: list[Value] = field(default_factory=list)
    returns: list[Value] = field(default_factory=list)
    blocks: list[BasicBlock] = field(default_factory=list)
    # Local variables declared in this function
    locals: list[Value] = field(default_factory=list)

    def add_block(self, block: BasicBlock) -> BasicBlock:
        self.blocks.append(block)
        return block

    def new_block(self, name: str) -> BasicBlock:
        existing = {b.name for b in self.blocks}
        candidate = name
        i = 0
        while candidate in existing:
            candidate = f"{name}_{i}"
            i += 1
        block = BasicBlock(candidate)
        self.blocks.append(block)
        return block


class Program:
    """The top-level IR container: a list of functions."""

    def __init__(self):
        self.functions: list[Function] = []
        self.global_values: list[Value] = []

    def add_function(self, func: Function) -> None:
        self.functions.append(func)

    def dump(self) -> str:
        lines = []
        for func in self.functions:
            lines.append(f"fun ${func.name}(")
            if func.params:
                params_str = ", ".join(
                    f"${{p.name}}: {p.dtype.value}"
                    for p in func.params)
                lines.append("  params: " + params_str)
            for block in func.blocks:
                lines.append(f"  .{block.name}:")
                for inst in block.instructions:
                    rhs = f"{inst.opcode.value}"
                    if inst.operands:
                        ops_str = " ".join(
                            f"${v.name}" for v in inst.operands)
                        rhs += " " + ops_str
                    if inst.target:
                        rhs += f" -> {inst.target}"
                    if inst.attrs:
                        for k, v in inst.attrs.items():
                            rhs += f" [{k}={v}]"
                    if inst.dest:
                        lines.append(f"    ${inst.dest.name} = {rhs}")
                    else:
                        lines.append(f"    {rhs}")
            lines.append("")
        return "\n".join(lines)
