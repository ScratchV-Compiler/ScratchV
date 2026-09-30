"""Conservative proofs for speculative execution under checked IR semantics.

Unknown shapes/ranges are not proofs. Attribute validation and scalar constant
evaluation reuse the interpreter's stateless kernels, so numeric error rules do
not diverge. No input tensor or whole program is executed by this analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from scratchv.ir.types import DataType, Instruction, OpCode, Value
from scratchv.verification.ir_numpy_ops import (
    DTYPES,
    KERNELS,
    OpError,
    axes,
    axis,
    check_instruction,
    compute,
)

_INTEGERS = (DataType.INT32, DataType.INT64)
_SCALAR_OPS = {
    OpCode.LOAD_CONST,
    OpCode.ADD,
    OpCode.SUB,
    OpCode.MUL,
    OpCode.DIV,
    OpCode.NEG,
    OpCode.EXP,
    OpCode.SQRT,
    OpCode.REDUCE_MEAN,
    OpCode.RELU,
    OpCode.SIGMOID,
    OpCode.GELU,
}


@dataclass(frozen=True)
class Facts:
    shape: tuple[int, ...] | None = None
    finite: bool = False
    # Parameters may contain -inf masks, but binding rejects NaN and +inf.
    finite_or_neginf: bool = False
    lower: int | float | None = None
    upper: int | float | None = None
    constant: np.ndarray | None = None
    tensor: bool = True


class HoistSafety:
    def __init__(self, program, function):
        self.parameters = {v.name for v in function.params}
        self.definitions = {
            v.name: v for v in [*program.global_values, *function.params]
        }
        self.definitions.update(
            (i.dest.name, i)
            for b in function.blocks
            for i in b.instructions
            if i.dest is not None
        )
        self.cache = {}
        self.visiting = set()

    def facts(self, value: Value) -> Facts:
        name = value.name
        if name in self.cache:
            return self.cache[name]
        if name in self.visiting:
            return Facts(tensor=False)
        self.visiting.add(name)
        definition = self.definitions.get(name, value)
        if isinstance(definition, Value):
            if definition.is_constant and name not in self.parameters:
                result = self._constant(definition.const_value, definition.dtype)
            else:
                result = Facts(
                    shape=definition.shape or None,
                    finite=definition.dtype in _INTEGERS,
                    finite_or_neginf=True,
                )
        elif definition.opcode == OpCode.ALLOCA:
            result = Facts(tensor=False)
        elif definition.opcode == OpCode.FOR:
            result = Facts(shape=(), finite=True, finite_or_neginf=True)
        elif definition.opcode == OpCode.LOAD:
            # STORE accepts scalar masks/literals without a finite-data check.
            # A prior LOAD is available, but its float range remains unknown.
            integral = definition.dest.dtype in _INTEGERS
            result = Facts(shape=(), finite=integral, finite_or_neginf=integral)
        else:
            result = self._prove(definition)
            if result is None:
                # An available, successfully executed instruction produced a
                # checked finite tensor. This is not permission to hoist it.
                result = Facts(
                    shape=definition.dest.shape or None,
                    finite=True,
                    finite_or_neginf=True,
                )
        self.visiting.remove(name)
        self.cache[name] = result
        return result

    @staticmethod
    def _constant(data, dtype):
        try:
            with np.errstate(over="raise", invalid="raise"):
                array = np.asarray(data, dtype=DTYPES[dtype])
            if array.shape != () or not np.isfinite(array).all():
                return Facts(tensor=False)
            value = array.item()
            return Facts((), True, True, value, value, array)
        except (TypeError, ValueError, OverflowError, FloatingPointError):
            return Facts(tensor=False)

    def is_safe(self, instr: Instruction) -> bool:
        return self._prove(instr) is not None

    def can_hoist_when_guaranteed(self, instr: Instruction) -> bool:
        """Allow pure runtime computations only with a must-execute proof.

        This does not prove that the operation succeeds. LICM must separately
        preserve execution and ordering before using this result. Memory and
        control-flow instructions are excluded from the stateless kernels.
        """
        if instr.dest is None or instr.opcode not in KERNELS:
            return False
        try:
            check_instruction(instr)
            operands = [self.facts(v) for v in instr.operands]
            if any(not facts.tensor for facts in operands):
                return False
            # Safe scalar constants already pass is_safe(). Keep known failing
            # constant expressions at their source location for diagnostics.
            return not (
                instr.opcode in _SCALAR_OPS
                and all(facts.constant is not None for facts in operands)
            )
        except (OpError, TypeError, ValueError, IndexError, OverflowError):
            return False

    def _prove(self, instr):
        if instr.dest is None or instr.opcode not in KERNELS:
            # LOAD/STORE/ALLOCA and control flow have no stateless kernel.
            return None
        try:
            check_instruction(instr)
            operands = [self.facts(v) for v in instr.operands]
            if any(not f.tensor for f in operands):
                return None
            result = self._prove_kernel(instr, operands)
            if result is not None and (
                not result.tensor
                or (instr.dest.shape and result.shape != instr.dest.shape)
            ):
                return None
            return result
        except (
            OpError,
            TypeError,
            ValueError,
            IndexError,
            OverflowError,
            FloatingPointError,
        ):
            return None

    def _prove_kernel(self, instr, fs):
        op, attrs, dtype = instr.opcode, instr.attrs, instr.dest.dtype
        integral = dtype in _INTEGERS
        # Only scalar operations are evaluated, with scalar constant operands.
        # Tensor shape attributes cannot trigger large compile-time allocations.
        if op in _SCALAR_OPS and all(f.constant is not None for f in fs):
            result = compute(instr, [f.constant for f in fs])
            return self._constant(result, dtype)

        if op in (OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV):
            if not integral or any(f.shape is None for f in fs):
                return None  # Floating arithmetic can overflow; shapes can fail.
            shape = np.broadcast_shapes(*(f.shape for f in fs))
            if op == OpCode.DIV:
                divisor = fs[1].constant
                if divisor is None or divisor.item() == 0:
                    return None
                if divisor.item() == -1 and (
                    fs[0].lower is None or fs[0].lower <= np.iinfo(DTYPES[dtype]).min
                ):
                    return None
            return Facts(shape, True, True)
        if op == OpCode.NEG and (integral or fs[0].finite):
            return Facts(fs[0].shape, True, True)
        if op == OpCode.RELU and fs[0].finite_or_neginf:
            return Facts(fs[0].shape, True, True, 0, fs[0].upper)
        if op == OpCode.SIGMOID and fs[0].finite_or_neginf:
            return Facts(fs[0].shape, True, True, 0, 1)
        if op == OpCode.SQRT:
            if not fs[0].finite or fs[0].lower is None or fs[0].lower < 0:
                return None
            return Facts(fs[0].shape, True, True, 0)

        # Remaining rules require proven concrete shapes. Copying floating
        # masks through a shape operation can itself fail the finite-result rule.
        if any(f.shape is None or not f.finite for f in fs):
            return None
        shapes = [f.shape for f in fs]
        shape = shapes[0] if shapes else None
        if op == OpCode.TRANSPOSE:
            perm = attrs.get("perm", tuple(reversed(range(len(shape)))))
            if len(perm) != len(shape) or set(perm) != set(range(len(shape))):
                return None
            shape = tuple(shape[a] for a in perm)
        elif op == OpCode.RESHAPE:
            target = tuple(
                shape[i] if d == 0 else d for i, d in enumerate(attrs["shape"])
            )
            size = math.prod(shape)
            if -1 in target:
                known = math.prod(d for d in target if d != -1)
                if known == 0 or size % known:
                    return None
                target = tuple(size // known if d == -1 else d for d in target)
            if math.prod(target) != size:
                return None
            shape = target
        elif op == OpCode.UNSQUEEZE:
            selected = axes(attrs["axes"], len(shape) + len(attrs["axes"]))
            source = iter(shape)
            shape = tuple(
                1 if a in selected else next(source)
                for a in range(len(shape) + len(selected))
            )
        elif op == OpCode.EXPAND:
            shape = np.broadcast_shapes(shape, tuple(attrs["shape"]))
        elif op == OpCode.SLICE:
            selected = axes(
                attrs.get("axes", tuple(range(len(attrs["starts"])))), len(shape)
            )
            output = list(shape)
            steps = attrs.get("steps", (1,) * len(selected))
            for a, start, end, step in zip(
                selected, attrs["starts"], attrs["ends"], steps
            ):
                output[a] = len(range(*slice(start, end, step).indices(shape[a])))
            shape = tuple(output)
        elif op == OpCode.GATHER:
            a = axis(attrs.get("axis", 0), len(shape))
            indices = fs[1]
            if (
                indices.constant is None
                or not -shape[a] <= indices.constant.item() < shape[a]
            ):
                return None
            shape = shape[:a] + indices.shape + shape[a + 1 :]
        elif op == OpCode.CONCAT:
            a = axis(attrs["axis"], len(shape))
            if any(
                len(s) != len(shape)
                or any(s[i] != shape[i] for i in range(len(shape)) if i != a)
                for s in shapes
            ):
                return None
            shape = tuple(
                sum(s[i] for s in shapes) if i == a else shape[i]
                for i in range(len(shape))
            )
        elif op == OpCode.REDUCE_MEAN:
            selected = attrs.get("axes")
            selected = (
                axes(selected, len(shape)) if selected else tuple(range(len(shape)))
            )
            count = math.prod(shape[a] for a in selected)
            if count == 0 or fs[0].lower is None or fs[0].upper is None:
                return None
            # A margin accounts for rounding during summation. Finite inputs
            # alone do not prove that NumPy's intermediate sum cannot overflow.
            if max(abs(fs[0].lower), abs(fs[0].upper)) > np.finfo(DTYPES[dtype]).max / (
                2 * count
            ):
                return None
            shape = (
                tuple(1 if a in selected else d for a, d in enumerate(shape))
                if attrs.get("keepdims", True)
                else tuple(d for a, d in enumerate(shape) if a not in selected)
            )
        else:
            return None  # New/unsupported proof rules default to retaining code.
        return Facts(shape, True, True, fs[0].lower, fs[0].upper)
