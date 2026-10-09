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
        # Extract target from attrs to set it as a proper field
        target = attrs.pop("target", None)
        instr = Instruction(
            opcode=opcode, dest=dest,
            operands=operands or [], attrs=attrs,
            target=target,
        )
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
        dest = self.make_value(dtype=lhs.dtype)
        self._emit(OpCode.ADD, dest, [lhs, rhs])
        return dest

    def sub(self, lhs: Value, rhs: Value) -> Value:
        dest = self.make_value(dtype=lhs.dtype)
        self._emit(OpCode.SUB, dest, [lhs, rhs])
        return dest

    def mul(self, lhs: Value, rhs: Value) -> Value:
        dest = self.make_value(dtype=lhs.dtype)
        self._emit(OpCode.MUL, dest, [lhs, rhs])
        return dest

    def div(self, lhs: Value, rhs: Value) -> Value:
        dest = self.make_value(dtype=lhs.dtype)
        self._emit(OpCode.DIV, dest, [lhs, rhs])
        return dest

    def neg(self, val: Value) -> Value:
        dest = self.make_value(dtype=val.dtype)
        self._emit(OpCode.NEG, dest, [val])
        return dest

    def exp(self, val: Value) -> Value:
        dest = self.make_value(dtype=val.dtype)
        self._emit(OpCode.EXP, dest, [val])
        return dest

    def abs(self, val: Value) -> Value:
        dest = self.make_value(dtype=val.dtype)
        dest.shape = val.shape
        self._emit(OpCode.ABS, dest, [val])
        return dest

    def cos(self, val: Value) -> Value:
        dest = self.make_value(dtype=val.dtype)
        dest.shape = val.shape
        self._emit(OpCode.COS, dest, [val])
        return dest

    def sin(self, val: Value) -> Value:
        dest = self.make_value(dtype=val.dtype)
        dest.shape = val.shape
        self._emit(OpCode.SIN, dest, [val])
        return dest

    def reciprocal(self, val: Value) -> Value:
        dest = self.make_value(dtype=val.dtype)
        dest.shape = val.shape
        self._emit(OpCode.RECIPROCAL, dest, [val])
        return dest

    def cast(self, val: Value, dtype: DataType) -> Value:
        """Convert elements; the result type is the conversion target."""
        if not isinstance(dtype, DataType):
            raise ValueError("CAST target must be a supported DataType")
        dest = self.make_value(dtype=dtype)
        dest.shape = val.shape
        self._emit(OpCode.CAST, dest, [val])
        return dest

    def pow(self, base: Value, exponent: Value) -> Value:
        """Broadcast power; exponent type may differ from the base/result type."""
        dest = self.make_value(dtype=base.dtype)
        self._emit(OpCode.POW, dest, [base, exponent])
        return dest

    def load_const(
            self, val: float | int,
            dtype: DataType = DataType.FLOAT32,
    ) -> Value:
        dest = self.make_value(dtype=dtype, is_constant=True, const_value=val)
        self._emit(OpCode.LOAD_CONST, dest, value=val)
        return dest

    def load(self, ptr: Value) -> Value:
        dest = self.make_value(dtype=ptr.dtype)
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

    def br_compare(
        self,
        lhs: Value,
        operator: str,
        rhs: Value,
        true_block: str,
        false_block: str,
    ) -> Instruction:
        """Branch by comparing two values with a supported DSL operator."""
        if operator not in {"==", "!=", "<", ">", "<=", ">="}:
            raise ValueError(f"unsupported comparison operator: {operator}")
        if lhs.dtype != rhs.dtype:
            raise ValueError("comparison operands must have the same type")
        return self._emit(
            OpCode.BR_IF,
            operands=[lhs, rhs],
            target=f"{true_block},{false_block}",
            cmp_op=operator,
        )

    def ret(self, val: Value | None = None) -> Instruction:
        operands = [val] if val else []
        return self._emit(OpCode.RETURN, operands=operands)

    def relu(self, val: Value) -> Value:
        dest = self.make_value(dtype=val.dtype)
        self._emit(OpCode.RELU, dest, [val])
        return dest

    def matmul(self, a: Value, b: Value, m: int | None = None,
               n: int | None = None, k: int | None = None) -> Value:
        dest = self.make_value(dtype=a.dtype)
        dimensions = (m, n, k)
        if any(v is not None for v in dimensions) and not all(v is not None for v in dimensions):
            raise ValueError("MATMUL m/n/k must be supplied together")
        attrs = {} if m is None else {"m": m, "n": n, "k": k}
        self._emit(OpCode.MATMUL, dest, [a, b], **attrs)
        return dest

    def dot(self, a: Value, b: Value, length: int) -> Value:
        dest = self.make_value(dtype=a.dtype)
        self._emit(OpCode.DOT, dest, [a, b], length=length)
        return dest

    def maxpool(self, val: Value, kernel: int, stride: int) -> Value:
        dest = self.make_value(dtype=val.dtype)
        self._emit(OpCode.MAXPOOL, dest, [val], kernel=kernel, stride=stride)
        return dest

    def gelu(self, val: Value) -> Value:
        dest = self.make_value(dtype=val.dtype)
        self._emit(OpCode.GELU, dest, [val])
        return dest

    def softmax(self, val: Value, axis: int = -1) -> Value:
        dest = self.make_value(dtype=val.dtype)
        self._emit(OpCode.SOFTMAX, dest, [val], axis=axis)
        return dest

    def conv(self, x: Value, w: Value, b: Value,
             out_channels: int,
             kernel_size: int = 3,
             stride: int = 1,
             padding: int = 1) -> Value:
        dest = self.make_value(dtype=x.dtype)
        self._emit(OpCode.CONV, dest, [x, w, b],
                   out_channels=out_channels,
                   kernel_size=kernel_size,
                   stride=stride, padding=padding)
        return dest

    def gemm(self, a: Value, w: Value, b: Value,
             trans_a: bool = False, trans_b: bool = False, *,
             alpha: float = 1.0, beta: float = 1.0) -> Value:
        dest = self.make_value(dtype=a.dtype)
        scales = {}
        if alpha != 1:
            scales["alpha"] = alpha
        if beta != 1:
            scales["beta"] = beta
        self._emit(OpCode.GEMM, dest, [a, w, b],
                   trans_a=trans_a, trans_b=trans_b, **scales)
        return dest

    def sigmoid(self, val: Value) -> Value:
        dest = self.make_value(dtype=val.dtype)
        self._emit(OpCode.SIGMOID, dest, [val])
        return dest

    def reshape(self, val: Value, shape: tuple) -> Value:
        dest = self.make_value(dtype=val.dtype)
        self._emit(OpCode.RESHAPE, dest, [val], shape=shape)
        return dest

    def sqrt(self, val: Value) -> Value:
        dest = self.make_value(dtype=val.dtype)
        self._emit(OpCode.SQRT, dest, [val])
        return dest

    def reduce_mean(self, val: Value, axes: tuple[int, ...] | None = None,
                    keepdims: bool = True) -> Value:
        dest = self.make_value(dtype=val.dtype)
        attrs = {"keepdims": keepdims}
        if axes is not None:
            attrs["axes"] = tuple(axes)
        self._emit(OpCode.REDUCE_MEAN, dest, [val], **attrs)
        return dest

    def transpose(self, val: Value, perm: tuple[int, ...] | None = None) -> Value:
        dest = self.make_value(dtype=val.dtype)
        self._emit(OpCode.TRANSPOSE, dest, [val], **({} if perm is None else {"perm": tuple(perm)}))
        return dest

    def concat(self, values: list[Value], axis: int) -> Value:
        if not values:
            raise ValueError("CONCAT requires at least one value")
        dest = self.make_value(dtype=values[0].dtype)
        self._emit(OpCode.CONCAT, dest, values, axis=axis)
        return dest

    def gather(self, data: Value, indices: Value, axis: int = 0) -> Value:
        dest = self.make_value(dtype=data.dtype)
        self._emit(OpCode.GATHER, dest, [data, indices], axis=axis)
        return dest

    def slice(self, val: Value, starts: tuple[int, ...], ends: tuple[int, ...],
              axes: tuple[int, ...] | None = None,
              steps: tuple[int, ...] | None = None) -> Value:
        dest = self.make_value(dtype=val.dtype)
        attrs = {"starts": tuple(starts), "ends": tuple(ends)}
        if axes is not None:
            attrs["axes"] = tuple(axes)
        if steps is not None:
            attrs["steps"] = tuple(steps)
        self._emit(OpCode.SLICE, dest, [val], **attrs)
        return dest

    def unsqueeze(self, val: Value, axes: tuple[int, ...]) -> Value:
        dest = self.make_value(dtype=val.dtype)
        self._emit(OpCode.UNSQUEEZE, dest, [val], axes=tuple(axes))
        return dest

    def expand(self, val: Value, shape: tuple[int, ...]) -> Value:
        dest = self.make_value(dtype=val.dtype)
        self._emit(OpCode.EXPAND, dest, [val], shape=tuple(shape))
        return dest
