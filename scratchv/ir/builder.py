"""IR builder: a helper to construct IR instructions conveniently."""

from __future__ import annotations

from scratchv.ir.types import (
    OpCode,
    DataType,
    Value,
    Instruction,
    BasicBlock,
    Function,
    Program,
)


class IRBuilder:
    """Tracks current function, block, and unique name counter."""

    def __init__(self):
        self.program = Program()
        self.current_func: Function | None = None
        self.current_block: BasicBlock | None = None
        self._name_counter = 0

    def _fresh(self, prefix: str = "v") -> str:
        self._name_counter += 1
        return f"{prefix}_{self._name_counter}"

    def _emit(self, opcode: OpCode, dest: Value | None = None,
              operands: list[Value] | None = None,
              **attrs) -> Instruction:
        instr = Instruction(
            opcode=opcode, dest=dest,
            operands=operands or [], attrs=attrs)
        if self.current_block is not None:
            self.current_block.add(instr)
        return instr

    # --- Function ---

    def new_function(
            self, name: str,
            params: list[Value] | None = None,
    ) -> Function:
        func = Function(name=name, params=params or [])
        self.program.add_function(func)
        self.current_func = func
        return func

    def new_block(self, name: str = "entry") -> BasicBlock:
        assert self.current_func is not None
        block = self.current_func.new_block(name)
        self.current_block = block
        return block

    # --- Values ---

    def make_value(self, name: str | None = None,
                   dtype: DataType = DataType.FLOAT32,
                   is_constant: bool = False,
                   const_value: float | int | None = None) -> Value:
        return Value(name=name or self._fresh(), dtype=dtype,
                     is_constant=is_constant, const_value=const_value)

    def make_const(
            self, value: float | int,
            dtype: DataType = DataType.FLOAT32,
    ) -> Value:
        return self.make_value(
            dtype=dtype, is_constant=True, const_value=value,
        )

    # --- Instructions ---

    def add(self, lhs: Value, rhs: Value) -> Value:
        dest = self.make_value()
        self._emit(OpCode.ADD, dest, [lhs, rhs])
        return dest

    def sub(self, lhs: Value, rhs: Value) -> Value:
        dest = self.make_value()
        self._emit(OpCode.SUB, dest, [lhs, rhs])
        return dest

    def mul(self, lhs: Value, rhs: Value) -> Value:
        dest = self.make_value()
        self._emit(OpCode.MUL, dest, [lhs, rhs])
        return dest

    def div(self, lhs: Value, rhs: Value) -> Value:
        dest = self.make_value()
        self._emit(OpCode.DIV, dest, [lhs, rhs])
        return dest

    def neg(self, val: Value) -> Value:
        dest = self.make_value()
        self._emit(OpCode.NEG, dest, [val])
        return dest

    def exp(self, val: Value) -> Value:
        dest = self.make_value()
        self._emit(OpCode.EXP, dest, [val])
        return dest

    def load_const(
            self, val: float | int,
            dtype: DataType = DataType.FLOAT32,
    ) -> Value:
        dest = self.make_value(dtype=dtype, is_constant=True, const_value=val)
        self._emit(OpCode.LOAD_CONST, dest, value=val)
        return dest

    def load(self, ptr: Value) -> Value:
        dest = self.make_value()
        self._emit(OpCode.LOAD, dest, [ptr])
        return dest

    def store(self, ptr: Value, val: Value) -> Instruction:
        return self._emit(OpCode.STORE, operands=[ptr, val])

    def alloca(self, size: int, dtype: DataType = DataType.FLOAT32) -> Value:
        dest = self.make_value(dtype=dtype)
        self._emit(OpCode.ALLOCA, dest, size=size)
        return dest

    def for_loop(self, start: int, end: int, step: int = 1) -> Value:
        """Start a for loop. Returns the loop variable."""
        iv = self.make_value(dtype=DataType.INT32)
        self._emit(OpCode.FOR, iv, start=start, end=end, step=step)
        return iv

    def endfor(self) -> Instruction:
        return self._emit(OpCode.ENDFOR)

    def br(self, target_block: str) -> Instruction:
        return self._emit(OpCode.BR, target=target_block)

    def br_if(self, cond: Value, true_block: str,
              false_block: str) -> Instruction:
        return self._emit(
            OpCode.BR_IF, operands=[cond],
            target=f"{true_block},{false_block}")

    def ret(self, val: Value | None = None) -> Instruction:
        operands = [val] if val else []
        return self._emit(OpCode.RETURN, operands=operands)

    def relu(self, val: Value) -> Value:
        dest = self.make_value()
        self._emit(OpCode.RELU, dest, [val])
        return dest

    def matmul(self, a: Value, b: Value, m: int, n: int, k: int) -> Value:
        dest = self.make_value()
        self._emit(OpCode.MATMUL, dest, [a, b], m=m, n=n, k=k)
        return dest

    def dot(self, a: Value, b: Value, length: int) -> Value:
        dest = self.make_value()
        self._emit(OpCode.DOT, dest, [a, b], length=length)
        return dest

    def maxpool(self, val: Value, kernel: int, stride: int) -> Value:
        dest = self.make_value()
        self._emit(OpCode.MAXPOOL, dest, [val], kernel=kernel, stride=stride)
        return dest

    def gelu(self, val: Value) -> Value:
        dest = self.make_value()
        self._emit(OpCode.GELU, dest, [val])
        return dest

    def softmax(self, val: Value, axis: int = -1) -> Value:
        dest = self.make_value()
        self._emit(OpCode.SOFTMAX, dest, [val], axis=axis)
        return dest
