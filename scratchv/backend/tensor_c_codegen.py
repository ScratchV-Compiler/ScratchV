"""Static tensor IR -> portable C, independent of the legacy scalar selector.

The C ABI is ``int scratchv_run(const void *const inputs[], void *output)``.
Buffers are contiguous, native-endian and suitably aligned; output must not
overlap an input buffer. The caller owns
input/output storage; intermediates use a bounded static arena with SSA-lifetime
reuse. Consequently generated functions are not reentrant. Compile as C11 with
``-fno-fast-math -ffp-contract=off -fno-strict-aliasing`` and link a conforming
libm plus memcpy/memset. No tensor storage is placed on the stack.

Supported element types are FP32, INT32 and INT64. Control flow, dynamic shapes,
empty reductions and nonfinite arithmetic are rejected explicitly. This backend
does not silently lower tensor operations to scalar integer instructions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from scratchv.ir.types import DataType, OpCode, Program, Value
from scratchv.verification.ir_numpy_ops import OpError, check_instruction


class TensorCCodegenError(ValueError):
    """A graph cannot be represented by the static C tensor contract."""


_TYPES = {DataType.FLOAT32: ("float", np.dtype("float32")),
          DataType.INT32: ("int32_t", np.dtype("int32")),
          DataType.INT64: ("int64_t", np.dtype("int64"))}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: DataType
    shape: tuple[int, ...]

    @property
    def numpy_dtype(self):
        return _TYPES[self.dtype][1]

    @property
    def size(self):
        return math.prod(self.shape)

    @property
    def nbytes(self):
        return self.size * self.numpy_dtype.itemsize


@dataclass(frozen=True)
class TensorCArtifact:
    source: str
    inputs: tuple[TensorSpec, ...]
    output: TensorSpec
    workspace_bytes: int
    constant_bytes: int
    function_name: str = "scratchv_run"
    compile_flags: tuple[str, ...] = ("-fno-fast-math", "-ffp-contract=off", "-fno-strict-aliasing")


def _shape(shape):
    if not isinstance(shape, (tuple, list)) or any(
        isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer)) or d < 0
        for d in shape
    ):
        raise TensorCCodegenError(f"Expected static nonnegative dimensions, got {shape!r}")
    result = tuple(int(d) for d in shape)
    if len(result) > 16 or math.prod(result) > 2**31 - 1:
        raise TensorCCodegenError("Tensor rank/element count exceeds static backend limit")
    return result


def _strides(shape):
    return tuple(math.prod(shape[i + 1:]) for i in range(len(shape)))


def _axis(axis, rank):
    if not -rank <= axis < rank:
        raise TensorCCodegenError(f"axis {axis} outside rank {rank}")
    return axis % rank


def _axes(axes, rank):
    normalized = tuple(_axis(a, rank) for a in axes)
    if len(set(normalized)) != len(normalized):
        raise TensorCCodegenError("Duplicate axes")
    return normalized


def _broadcast(*shapes):
    try:
        return _shape(np.broadcast_shapes(*shapes))
    except ValueError as exc:
        raise TensorCCodegenError(f"Invalid broadcast: {shapes}") from exc


def _offset(shape, coefficients, index="i", base=0):
    """Flat-index coordinates dotted with signed input strides."""
    terms = [str(base)] if base else []
    for dim, stride, coefficient in zip(shape, _strides(shape), coefficients):
        if dim > 1 and coefficient:
            coordinate = f"(({index} / {stride}ULL) % {dim}ULL)"
            terms.append(f"((int64_t){coordinate} * ({coefficient}LL))")
    return " + ".join(terms) or "0"


def _broadcast_offset(output, source, index="i"):
    if source == output:
        return index
    padded = (1,) * (len(output) - len(source)) + source
    strides = (0,) * (len(output) - len(source)) + _strides(source)
    return _offset(output, [s if d != 1 else 0 for d, s in zip(padded, strides)], index)


def _literal(value, dtype):
    if dtype == DataType.FLOAT32:
        value = np.float32(value)
        if not np.isfinite(value):
            raise TensorCCodegenError("Nonfinite initializer/constant")
        return float(value).hex() + "f"
    integer = int(value)
    bounds = np.iinfo(_TYPES[dtype][1])
    if integer != value or not bounds.min <= integer <= bounds.max:
        raise TensorCCodegenError("Integer constant outside declared dtype")
    if integer == -(1 << 63):
        return "(-9223372036854775807LL - 1LL)"
    return f"{integer}LL"


_PRELUDE = r'''/* Generated static FP32 tensor graph. Inputs/output must not alias workspace. */
#include <stdint.h>
#include <stddef.h>
extern float expf(float);
extern float sinf(float);
extern float cosf(float);
extern float sqrtf(float);
extern float powf(float, float);
static int sv_finite(float x) {
    union { float f; uint32_t u; } v; v.f=x;
    return (v.u & 0x7f800000U) != 0x7f800000U;
}
/* status: 1 invalid pointer, 2 invalid numeric/domain, 3 gather bounds. */
static float sv_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    float z=expf(x); return z / (1.0f + z);
}
'''


class TensorCCodegen:
    """Generate one straight-line function with strict tensor/dtype validation."""

    def __init__(self, program: Program, initializers: Mapping[str, np.ndarray] | None = None,
                 *, max_workspace_bytes=256 * 1024 * 1024,
                 max_constant_bytes=256 * 1024 * 1024):
        self.program = program
        self.initializers = {} if initializers is None else dict(initializers)
        self.max_workspace_bytes = max_workspace_bytes
        self.max_constant_bytes = max_constant_bytes

    def generate(self) -> TensorCArtifact:
        if len(self.program.functions) != 1:
            raise TensorCCodegenError("Exactly one function is required")
        func = self.program.functions[0]
        if len(func.blocks) != 1 or func.blocks[0].phi_nodes:
            raise TensorCCodegenError("Only one straight-line block without phi is supported")
        instructions = func.blocks[0].instructions
        if (not instructions or instructions[-1].opcode != OpCode.RETURN
                or len(instructions[-1].operands) != 1
                or any(i.opcode == OpCode.RETURN for i in instructions[:-1])):
            raise TensorCCodegenError("Exactly one final tensor RETURN is required")
        if instructions[-1].attrs or instructions[-1].dest is not None or instructions[-1].target:
            raise TensorCCodegenError("RETURN must not have attributes, destination or target")
        if any(value.is_constant for value in func.params):
            raise TensorCCodegenError("Function parameters cannot be constants")
        for bound in (self.max_workspace_bytes, self.max_constant_bytes):
            if isinstance(bound, bool) or not isinstance(bound, int) or bound < 0:
                raise TensorCCodegenError("Memory limits must be nonnegative integers")
        self.specs, self.names, self.constants, self.body = {}, {}, [], []
        self.constant_bytes = 0
        self._counter = 0
        defined = set()
        for value in [*self.program.global_values, *func.params]:
            if value.name in defined:
                raise TensorCCodegenError(f"Duplicate definition {value.name}")
            defined.add(value.name)
            self._register(value)
        expected_initializers = {v.name for v in self.program.global_values}
        if set(self.initializers) - expected_initializers:
            raise TensorCCodegenError("Initializer bindings not declared in Program")
        for value in self.program.global_values:
            data = self.initializers.get(value.name)
            if data is None and value.is_constant:
                data = np.asarray(value.const_value, dtype=self.specs[value.name].numpy_dtype)
            if data is None:
                raise TensorCCodegenError(f"Missing initializer tensor data: {value.name}")
            self._constant(value, data)
        inputs = tuple(self.specs[p.name] for p in func.params)
        for index, spec in enumerate(inputs):
            ctype = _TYPES[spec.dtype][0]
            self.body.append(f"if (!inputs || !inputs[{index}]) return 1;")
            self.body.append(f"const {ctype} *{self.names[spec.name]} = (const {ctype} *)inputs[{index}];")
            self._finite_check(spec)

        # Last-use allocation keeps tensors live until their final consumer has
        # finished. The destination is allocated before operands are released.
        last = {}
        for index, instruction in enumerate(instructions):
            for operand in instruction.operands:
                last[operand.name] = index
        free, allocations, high = [], {}, 0
        for index, instruction in enumerate(instructions[:-1]):
            if instruction.dest is None or instruction.dest.name in defined:
                raise TensorCCodegenError(f"Instruction {index}: missing or duplicate SSA destination")
            if instruction.dest.dtype not in _TYPES:
                raise TensorCCodegenError(f"Unsupported dtype {instruction.dest.dtype}")
            if instruction.target:
                raise TensorCCodegenError("Tensor instructions cannot have control-flow targets")
            for operand in instruction.operands:
                if operand.name not in self.specs:
                    if not operand.is_constant:
                        raise TensorCCodegenError(f"Undefined operand {operand.name}")
                    self._register(operand)
                    self._constant(operand, np.asarray(operand.const_value,
                                   dtype=self.specs[operand.name].numpy_dtype))
                    defined.add(operand.name)
                existing = self.specs[operand.name]
                if operand.dtype != existing.dtype:
                    raise TensorCCodegenError(f"Inconsistent operand dtype: {operand.name}")
            try:
                check_instruction(instruction)
                shape, details = self._infer(instruction)
            except (OpError, ValueError, TypeError, IndexError) as exc:
                raise TensorCCodegenError(f"{index}:{instruction.opcode.value}: {exc}") from exc
            value = instruction.dest
            if value.shape and _shape(value.shape) != shape:
                raise TensorCCodegenError(f"{value.name}: declared shape {value.shape} != inferred {shape}")
            spec = self._register(value, shape)
            defined.add(value.name)
            needed = (max(spec.nbytes, 1) + 7) // 8 * 8
            choice = next((j for j, (_, size) in enumerate(free) if size >= needed), None)
            if choice is None:
                offset = high
                high += needed
            else:
                offset, block_size = free.pop(choice)
                if block_size > needed:
                    free.append((offset + needed, block_size - needed))
            if high > self.max_workspace_bytes:
                raise TensorCCodegenError(f"Static workspace exceeds {self.max_workspace_bytes} bytes")
            allocations[value.name] = (offset, needed)
            self.body.append(f"/* {index}: {instruction.opcode.value} */")
            self.body.append(f"{_TYPES[spec.dtype][0]} *{self.names[value.name]} = "
                             f"({_TYPES[spec.dtype][0]} *)(sv_arena.bytes + {offset}ULL);")
            self._emit(instruction, spec, details)
            self._finite_check(spec)
            for name in list(allocations):
                if last.get(name, index) <= index:
                    free.append(allocations.pop(name))
            free.sort()
            merged = []
            for offset, size in free:
                if merged and merged[-1][0] + merged[-1][1] == offset:
                    merged[-1] = (merged[-1][0], merged[-1][1] + size)
                else:
                    merged.append((offset, size))
            free = merged
        returned = instructions[-1].operands[0]
        if returned.name not in self.specs:
            if not returned.is_constant:
                raise TensorCCodegenError("Undefined return value")
            self._register(returned)
            self._constant(returned, np.asarray(returned.const_value,
                           dtype=self.specs[returned.name].numpy_dtype))
        output = self.specs[returned.name]
        ctype = _TYPES[output.dtype][0]
        self.body += ["if (!output) return 1;",
                      f"for (size_t i=0; i<{output.size}ULL; ++i) (({ctype} *)output)[i] = {self.names[output.name]}[i];",
                      "return 0;"]
        source = (_PRELUDE + "\n" + "\n".join(self.constants)
                  + f"\nstatic union {{ uint64_t align; unsigned char bytes[{max(high, 8)}]; }} sv_arena;\n"
                  + "#if defined(_WIN32)\n__declspec(dllexport)\n#endif\n"
                  + "int scratchv_run(const void *const inputs[], void *output) {\n  "
                  + "\n  ".join(self.body) + "\n}\n")
        return TensorCArtifact(source, inputs, output, high, self.constant_bytes)

    def _register(self, value, shape=None):
        if value.dtype not in _TYPES:
            raise TensorCCodegenError(f"Unsupported dtype {value.dtype}; use FP32/INT32/INT64")
        shape = _shape(value.shape if shape is None else shape)
        spec = TensorSpec(value.name, value.dtype, shape)
        self.specs[value.name] = spec
        self.names[value.name] = f"sv_v{self._counter}"
        self._counter += 1
        return spec

    def _constant(self, value, data):
        spec = self.specs[value.name]
        if not isinstance(data, np.ndarray) or data.shape != spec.shape or data.dtype != spec.numpy_dtype:
            raise TensorCCodegenError(f"Initializer {value.name} dtype/shape disagrees with IR")
        self.constant_bytes += spec.nbytes
        if self.constant_bytes > self.max_constant_bytes:
            raise TensorCCodegenError("Constant tensor storage exceeds configured limit")
        values = [_literal(v.item(), spec.dtype) for v in data.ravel()]
        lines = [", ".join(values[i:i + 8]) for i in range(0, len(values), 8)]
        self.constants.append(f"static const {_TYPES[spec.dtype][0]} {self.names[value.name]}"
                              f"[{max(spec.size, 1)}] = {{\n" + ",\n".join(lines or ["0"]) + "\n};")

    def _finite_check(self, spec):
        if spec.dtype == DataType.FLOAT32:
            self.body.append(f"for (size_t i=0; i<{spec.size}ULL; ++i) "
                             f"if (!sv_finite({self.names[spec.name]}[i])) return 2;")

    def _infer(self, instruction):
        op, attrs = instruction.opcode, instruction.attrs
        xs = [self.specs[x.name] for x in instruction.operands]
        dtype = instruction.dest.dtype
        unary = {OpCode.NEG, OpCode.ABS, OpCode.COS, OpCode.SIN, OpCode.RECIPROCAL,
                 OpCode.EXP, OpCode.SQRT, OpCode.SIGMOID, OpCode.RELU, OpCode.CAST}
        binary = {OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.POW}
        if op == OpCode.LOAD_CONST:
            if xs:
                raise TensorCCodegenError("LOAD_CONST cannot have operands")
            _literal(attrs["value"], dtype)
            return (), {}
        expected_count = 1 if op in unary or op in {
            OpCode.REDUCE_MEAN, OpCode.SOFTMAX, OpCode.RESHAPE, OpCode.TRANSPOSE,
            OpCode.SLICE, OpCode.UNSQUEEZE, OpCode.EXPAND} else 2
        if op == OpCode.CONCAT:
            if not xs:
                raise TensorCCodegenError("Concat needs at least one input")
        elif len(xs) != expected_count:
            raise TensorCCodegenError(f"Expected {expected_count} operands")
        if op != OpCode.CAST and dtype != xs[0].dtype:
            raise TensorCCodegenError("Output dtype must match first operand")
        if op in unary:
            return xs[0].shape, {}
        if op in binary:
            if op != OpCode.POW and xs[0].dtype != xs[1].dtype:
                raise TensorCCodegenError("Arithmetic operands must have equal dtype")
            if op == OpCode.POW and dtype != DataType.FLOAT32 and xs[1].dtype == DataType.FLOAT32:
                raise TensorCCodegenError("Integer Pow requires integer exponent")
            return _broadcast(*(x.shape for x in xs)), {}
        if op == OpCode.MATMUL:
            if any(x.dtype != DataType.FLOAT32 or not x.shape for x in xs):
                raise TensorCCodegenError("MatMul requires nonscalar FP32 tensors")
            a, b = xs[0].shape, xs[1].shape
            if "m" in attrs:
                m, n, k = (attrs[key] for key in ("m", "n", "k"))
                if len(a) == len(b) == 1 and math.prod(a) == m*k and math.prod(b) == k*n:
                    a, b = (m, k), (k, n)
                elif a != (m, k) or b != (k, n):
                    raise TensorCCodegenError("MatMul m/n/k disagree with tensor dimensions")
            av, bv = len(a) == 1, len(b) == 1
            a = (1, *a) if av else a
            b = (*b, 1) if bv else b
            if a[-1] != b[-2]:
                raise TensorCCodegenError("MatMul inner dimensions disagree")
            batch = _broadcast(a[:-2], b[:-2])
            result = batch + (() if av else (a[-2],)) + (() if bv else (b[-1],))
            return result, dict(a=a, b=b, batch=batch)
        x = xs[0]
        if op == OpCode.SOFTMAX:
            axis = _axis(attrs.get("axis", -1), len(x.shape))
            if not x.shape[axis]:
                raise TensorCCodegenError("Empty Softmax axis")
            return x.shape, dict(axis=axis)
        if op == OpCode.REDUCE_MEAN:
            axes = _axes(attrs.get("axes") or tuple(range(len(x.shape))), len(x.shape))
            if any(not x.shape[a] for a in axes):
                raise TensorCCodegenError("Cannot ReduceMean an empty axis")
            keep = attrs.get("keepdims", True)
            result = tuple(1 if a in axes else d for a, d in enumerate(x.shape)) if keep else tuple(
                d for a, d in enumerate(x.shape) if a not in axes)
            return result, dict(axes=axes)
        if op == OpCode.RESHAPE:
            target = list(attrs["shape"])
            for axis, dim in enumerate(target):
                if dim == 0:
                    if axis >= len(x.shape):
                        raise TensorCCodegenError("Reshape zero cannot copy absent input axis")
                    target[axis] = x.shape[axis]
            if -1 in target:
                denominator = math.prod(d for d in target if d != -1)
                if denominator == 0 or x.size % denominator:
                    raise TensorCCodegenError("Reshape inference is ambiguous or nonintegral")
                target[target.index(-1)] = x.size // denominator
            if math.prod(target) != x.size:
                raise TensorCCodegenError("Reshape element counts differ")
            return _shape(target), {}
        if op == OpCode.TRANSPOSE:
            perm = tuple(attrs.get("perm", tuple(reversed(range(len(x.shape))))))
            if sorted(perm) != list(range(len(x.shape))):
                raise TensorCCodegenError("Transpose requires a complete permutation")
            return tuple(x.shape[a] for a in perm), dict(perm=perm)
        if op == OpCode.CONCAT:
            axis = _axis(attrs["axis"], len(x.shape))
            for operand in xs:
                if (operand.dtype != dtype or len(operand.shape) != len(x.shape) or any(
                    a != axis and d != x.shape[a] for a, d in enumerate(operand.shape))):
                    raise TensorCCodegenError("Concat input dimensions/dtypes disagree")
            result = list(x.shape)
            result[axis] = sum(v.shape[axis] for v in xs)
            return tuple(result), dict(axis=axis)
        if op == OpCode.GATHER:
            if xs[1].dtype not in (DataType.INT32, DataType.INT64):
                raise TensorCCodegenError("Gather indices must be integer tensors")
            axis = _axis(attrs.get("axis", 0), len(x.shape))
            return x.shape[:axis] + xs[1].shape + x.shape[axis+1:], dict(axis=axis)
        if op == OpCode.SLICE:
            axes = _axes(attrs.get("axes", tuple(range(len(attrs["starts"])))), len(x.shape))
            steps = attrs.get("steps", (1,) * len(axes))
            starts, factors, result = [0] * len(x.shape), [1] * len(x.shape), list(x.shape)
            for axis, start, end, step in zip(axes, attrs["starts"], attrs["ends"], steps):
                dimension = x.shape[axis]
                start = start + dimension if start < 0 else start
                end = end + dimension if end < 0 else end
                if step > 0:
                    start, end = max(0, min(dimension, start)), max(0, min(dimension, end))
                else:
                    start, end = max(0, min(dimension-1, start)), max(-1, min(dimension-1, end))
                starts[axis], factors[axis] = start, step
                result[axis] = len(range(start, end, step)) if dimension else 0
            return tuple(result), dict(starts=starts, steps=factors)
        if op == OpCode.UNSQUEEZE:
            rank = len(x.shape) + len(attrs["axes"])
            axes = _axes(attrs["axes"], rank)
            dims = iter(x.shape)
            return tuple(1 if a in axes else next(dims) for a in range(rank)), {}
        if op == OpCode.EXPAND:
            return _broadcast(x.shape, tuple(attrs["shape"])), {}
        raise TensorCCodegenError(f"Unsupported tensor opcode: {op.value}")

    def _emit(self, instruction, output, details):
        op, attrs = instruction.opcode, instruction.attrs
        xs = [self.specs[x.name] for x in instruction.operands]
        dst = self.names[output.name]
        dtype, size = output.dtype, output.size
        names = [self.names[x.name] for x in xs]
        if not size:
            return
        if op == OpCode.LOAD_CONST:
            self.body.append(f"{dst}[0] = {_literal(attrs['value'], dtype)};")
            return
        refs = [f"{name}[{_broadcast_offset(output.shape, x.shape)}]"
                for name, x in zip(names, xs)] if op in {
                    OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.POW, OpCode.EXPAND} else []
        if op in {OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.POW}:
            a, b = refs
            checks = ""
            symbol = {OpCode.ADD: "+", OpCode.SUB: "-", OpCode.MUL: "*", OpCode.DIV: "/"}.get(op)
            if op == OpCode.POW:
                if dtype != DataType.FLOAT32:
                    ctype = _TYPES[dtype][0]
                    expression = f"({ctype})acc"
                    checks = (f"if ({b}<0) return 2; {ctype} acc=1, factor={a}; uint64_t exponent=(uint64_t){b}; "
                              "while (exponent) { if ((exponent & 1ULL) && __builtin_mul_overflow(acc,factor,&acc)) return 2; "
                              "exponent >>= 1; if (exponent && __builtin_mul_overflow(factor,factor,&factor)) return 2; } ")
                else:
                    # powf is exact about integer parity only while an exponent
                    # fits FP32. Reject larger runtime integer exponents instead
                    # of silently rounding them (Qwen uses INT64 scalar 2).
                    if xs[1].dtype != DataType.FLOAT32:
                        checks += f"if ({b}>16777216LL || {b}<(-16777216LL)) return 2; "
                    checks += f"if ({a}==0.0f && {b}<0) return 2; "
                    expression = f"powf({a}, (float){b})"
            elif dtype == DataType.FLOAT32:
                if op == OpCode.DIV:
                    checks = f"if ({b}==0.0f) return 2; "
                expression = f"{a} {symbol} {b}"
            else:
                bits = output.numpy_dtype.itemsize * 8
                ctype = _TYPES[dtype][0]
                if op == OpCode.DIV:
                    minimum = _literal(np.iinfo(output.numpy_dtype).min, dtype)
                    checks = f"if ({b}==0 || ({a}=={minimum} && {b}==-1)) return 2; "
                    expression = f"{a} / {b}"
                else:
                    expression = f"({ctype})((uint{bits}_t){a} {symbol} (uint{bits}_t){b})"
            self.body.append(f"for (size_t i=0; i<{size}ULL; ++i) {{ {checks}{dst}[i]={expression}; }}")
            return
        if op in {OpCode.NEG, OpCode.ABS, OpCode.COS, OpCode.SIN, OpCode.RECIPROCAL,
                  OpCode.EXP, OpCode.SQRT, OpCode.SIGMOID, OpCode.RELU, OpCode.CAST}:
            a, checks = f"{names[0]}[i]", ""
            if op == OpCode.CAST:
                if dtype != DataType.FLOAT32 and xs[0].dtype == DataType.FLOAT32:
                    bound = float(1 << (output.numpy_dtype.itemsize * 8 - 1)).hex() + "f"
                    checks = f"if ({a} < -{bound} || {a} >= {bound}) return 2; "
                expression = f"({_TYPES[dtype][0]}){a}"
            elif op == OpCode.NEG:
                expression = f"-{a}" if dtype == DataType.FLOAT32 else (
                    f"({_TYPES[dtype][0]})(0ULL-(uint{output.numpy_dtype.itemsize*8}_t){a})")
            elif op == OpCode.ABS:
                if dtype != DataType.FLOAT32:
                    checks = f"if ({a}=={_literal(np.iinfo(output.numpy_dtype).min,dtype)}) return 2; "
                expression = (f"__builtin_fabsf({a})" if dtype == DataType.FLOAT32
                              else f"({a}<0 ? -{a} : {a})")
            elif op == OpCode.RECIPROCAL:
                checks, expression = f"if ({a}==0.0f) return 2; ", f"1.0f/{a}"
            elif op == OpCode.RELU:
                expression = f"({a}>0 ? {a} : 0)"
            else:
                function = {OpCode.COS:"cosf", OpCode.SIN:"sinf", OpCode.EXP:"expf",
                            OpCode.SQRT:"sqrtf", OpCode.SIGMOID:"sv_sigmoid"}[op]
                if op == OpCode.SQRT:
                    checks = f"if ({a}<0.0f) return 2; "
                expression = f"{function}({a})"
            self.body.append(f"for (size_t i=0; i<{size}ULL; ++i) {{ {checks}{dst}[i]={expression}; }}")
            return
        x, src = xs[0], names[0]
        if op in {OpCode.RESHAPE, OpCode.UNSQUEEZE}:
            expression = f"{src}[i]"
        elif op == OpCode.EXPAND:
            expression = refs[0]
        elif op == OpCode.TRANSPOSE:
            offset = _offset(output.shape, [_strides(x.shape)[a] for a in details["perm"]])
            expression = f"{src}[{offset}]"
        elif op == OpCode.SLICE:
            strides = _strides(x.shape)
            offset = _offset(output.shape, [a*b for a,b in zip(strides,details["steps"])],
                             base=sum(a*b for a,b in zip(strides,details["starts"])))
            expression = f"{src}[{offset}]"
        else:
            expression = None
        if expression is not None:
            self.body.append(f"for (size_t i=0; i<{size}ULL; ++i) {dst}[i]={expression};")
            return
        if op == OpCode.MATMUL:
            a, b, batch = details["a"], details["b"], details["batch"]
            m, k, n = a[-2], a[-1], b[-1]
            left = _broadcast_offset(batch, a[:-2], "batch")
            right = _broadcast_offset(batch, b[:-2], "batch")
            self.body.append(f"for (size_t batch=0; batch<{math.prod(batch)}ULL; ++batch) "
                f"for (size_t m=0; m<{m}ULL; ++m) for (size_t n=0; n<{n}ULL; ++n) {{ "
                "float acc=0.0f; "
                f"for (size_t k=0; k<{k}ULL; ++k) acc += {names[0]}[({left})*{m*k}ULL+m*{k}ULL+k] "
                f"* {names[1]}[({right})*{k*n}ULL+k*{n}ULL+n]; "
                f"{dst}[batch*{m*n}ULL+m*{n}ULL+n]=acc; }}")
        elif op == OpCode.REDUCE_MEAN:
            axes = details["axes"]
            remaining = tuple(a for a in range(len(x.shape)) if a not in axes)
            outer_shape = tuple(x.shape[a] for a in remaining)
            reduced_shape = tuple(x.shape[a] for a in axes)
            outer_offset = _offset(outer_shape, [_strides(x.shape)[a] for a in remaining], "i")
            red_offset = _offset(reduced_shape, [_strides(x.shape)[a] for a in axes], "r")
            count = math.prod(reduced_shape)
            self.body.append(f"for (size_t i=0; i<{size}ULL; ++i) {{ float acc=0.0f; "
                f"for (size_t r=0; r<{count}ULL; ++r) acc += {src}[({outer_offset})+({red_offset})]; "
                f"{dst}[i]=acc/{float(count).hex()}f; }}")
        elif op == OpCode.SOFTMAX:
            axis = details["axis"]
            outer, dimension, inner = math.prod(x.shape[:axis]), x.shape[axis], math.prod(x.shape[axis+1:])
            off = f"o*{dimension*inner}ULL+j*{inner}ULL+t"
            self.body.append(f"for (size_t o=0; o<{outer}ULL; ++o) for (size_t t=0; t<{inner}ULL; ++t) {{ "
                "float maximum=-0x1.fffffep+127f, sum=0.0f; "
                f"for (size_t j=0; j<{dimension}ULL; ++j) if ({src}[{off}]>maximum) maximum={src}[{off}]; "
                f"for (size_t j=0; j<{dimension}ULL; ++j) {{ float shifted={src}[{off}]-maximum; "
                "if (!sv_finite(shifted)) return 2; float v=expf(shifted); "
                f"{dst}[{off}]=v; sum+=v; }} "
                f"for (size_t j=0; j<{dimension}ULL; ++j) {dst}[{off}]/=sum; }}")
        elif op == OpCode.GATHER:
            axis, count = details["axis"], xs[1].size
            dimension, inner = x.shape[axis], math.prod(x.shape[axis+1:])
            self.body.append(f"for (size_t j=0; j<{count}ULL; ++j) if ({names[1]}[j]<-{dimension}LL "
                             f"|| {names[1]}[j]>={dimension}LL) return 3;")
            self.body.append(f"for (size_t i=0; i<{size}ULL; ++i) {{ int64_t selected="
                f"{names[1]}[(i/{inner}ULL)%{count}ULL]; if (selected<0) selected+={dimension}LL; "
                f"{dst}[i]={src}[(i/{inner*count}ULL)*{dimension*inner}ULL+selected*{inner}LL+i%{inner}ULL]; }}")
        elif op == OpCode.CONCAT:
            axis = details["axis"]
            inner, total, start = math.prod(output.shape[axis+1:]), output.shape[axis], 0
            for operand, name in zip(xs, names):
                chunk = operand.shape[axis] * inner
                if operand.size:
                    self.body.append(f"for (size_t i=0; i<{operand.size}ULL; ++i) "
                        f"{dst}[(i/{chunk}ULL)*{total*inner}ULL+{start*inner}ULL+i%{chunk}ULL]={name}[i];")
                start += operand.shape[axis]
        else:
            raise TensorCCodegenError(f"No C lowering for {op.value}")
