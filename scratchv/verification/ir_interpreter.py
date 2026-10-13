"""Execute a shared Program with strict bindings and real control flow.

Integer ADD/SUB/MUL/NEG wrap at the declared width; DIV truncates toward zero
and rejects zero and min/-1. ALLOCA sizes are bytes; LOAD/STORE access the
first scalar slot only and reject reads before STORE. No host pointers exist.
Every run owns fresh state and, by default, copies external arrays and results.
Explicit straight-line execution options can borrow immutable weights and
release values after their final use. Returned results are always copied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from scratchv.analysis.adapters import IRCFGAdapter, IRSourcePosition
from scratchv.analysis.cfg import build_cfg
from scratchv.analysis.ir_verifier import VerificationError, verify_ir
from scratchv.ir.types import OpCode, Program
from scratchv.verification.ir_numpy_ops import (
    DTYPES,
    OpError,
    check_instruction,
    compute,
    integer,
)


class IRExecutionError(RuntimeError):
    """Execution error with an original IR location (zero-based index)."""

    def __init__(
        self,
        code,
        message,
        *,
        function_name=None,
        position=None,
        opcode=None,
        value_name=None,
    ):
        self.code = code
        self.function_name = function_name
        self.block_name = position.block_name if position else None
        self.instruction_index = position.instruction_index if position else None
        self.stage = position.stage if position else None
        self.opcode = opcode
        self.value_name = value_name
        context = " ".join(
            f"{key}={value}"
            for key, value in (
                ("function", self.function_name),
                ("block", self.block_name),
                ("instruction", self.instruction_index),
                ("stage", self.stage),
                ("opcode", opcode.value if opcode else None),
                ("value", value_name),
            )
            if value is not None
        )
        super().__init__(f"{code} {context}: {message}")


@dataclass(frozen=True)
class ExecutionResult:
    return_value: np.ndarray | None
    executed_steps: int
    diagnostics: tuple[VerificationError, ...] = ()
    memory_stats: dict | None = None


@dataclass
class _MemorySlot:
    storage: np.ndarray
    initialized: bool = False


def _backing_storage(array):
    """Identify the owner of a view without keeping new array references alive."""
    owner = array
    seen = set()
    while id(owner) not in seen:
        seen.add(id(owner))
        base = getattr(owner, "base", None)
        if base is None and isinstance(owner, memoryview):
            base = owner.obj
        if base is None:
            break
        owner = base
    try:
        size = memoryview(owner).nbytes
    except TypeError:
        # Normal NumPy views (including broadcast/strided views) terminate in
        # a buffer owner. Fall back to the root array for unusual owners.
        size = getattr(owner, "nbytes", array.nbytes)
    return id(owner), int(size)


class _MemoryObservation:
    """Optional instruction-boundary accounting; never releases/copies values."""

    def __init__(self, values, params, globals_):
        self.input_bytes = sum(values[name].nbytes for name in params)
        self.initializer_bytes = sum(
            values[name].nbytes for name in globals_ if name in values
        )
        self.peak_logical = 0
        self.peak_storage = 0
        self.observe(values)

    @staticmethod
    def snapshot(values, returned=None):
        arrays = [value.storage if isinstance(value, _MemorySlot) else value
                  for value in values.values()]
        if returned is not None:
            arrays.append(returned)
        storage = {}
        for array in arrays:
            key, size = _backing_storage(array)
            storage[key] = max(size, storage.get(key, 0))
        return sum(int(array.nbytes) for array in arrays), sum(storage.values())

    def observe(self, values, returned=None):
        logical, storage = self.snapshot(values, returned)
        self.peak_logical = max(self.peak_logical, logical)
        self.peak_storage = max(self.peak_storage, storage)

    def finish(self, values, returned):
        self.observe(values, returned)
        logical, storage = self.snapshot(values)
        return {
            "input_logical_bytes": int(self.input_bytes),
            "initializer_logical_bytes": int(self.initializer_bytes),
            "peak_live_logical_bytes": self.peak_logical,
            "peak_numpy_storage_bytes": self.peak_storage,
            "retained_values": len(values),
            "retained_logical_bytes": logical,
            "retained_numpy_storage_bytes": storage,
            "return_logical_bytes": 0 if returned is None else int(returned.nbytes),
            "scope": "interpreter retained values at instruction boundaries; final return copy included in peaks",
            "notes": [
                "Logical bytes count each retained binding, including views and ALLOCA storage.",
                "Storage bytes deduplicate shared backing buffer owners, including NumPy views.",
                "Initializer bytes count bound referenced globals, including scalar global constants.",
                "Retained values exclude the independent return copy; peaks include it.",
                "Counts currently bound buffers, including borrowed initializers; excludes other caller-held arrays, temporary kernel buffers, Python objects and allocator overhead; not process RSS.",
                "Observation does not change existing copying, value retention, control flow or numeric semantics.",
            ],
        }


_CONTROL_ATTRS = {
    OpCode.FOR: {"start", "end", "step"},
    OpCode.ENDFOR: set(),
    OpCode.BR: set(),
    OpCode.BR_IF: {"cmp_op"},
    OpCode.RETURN: set(),
    OpCode.ALLOCA: {"size"},
    OpCode.LOAD: set(),
    OpCode.STORE: set(),
}

# An explicit allow-list makes future effectful instructions fail closed in
# the opt-in borrowing/liveness modes. Every listed kernel treats inputs as
# immutable and returns an array or a view of an input.
_PURE_OPS = {
    OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.POW,
    OpCode.NEG, OpCode.ABS, OpCode.COS, OpCode.SIN, OpCode.RECIPROCAL,
    OpCode.EXP, OpCode.SQRT, OpCode.REDUCE_MEAN, OpCode.LOAD_CONST,
    OpCode.CAST, OpCode.MATMUL, OpCode.RELU, OpCode.MAXPOOL, OpCode.SOFTMAX,
    OpCode.GELU, OpCode.DOT, OpCode.CONV, OpCode.GEMM, OpCode.SIGMOID,
    OpCode.TRANSPOSE, OpCode.RESHAPE, OpCode.CONCAT, OpCode.GATHER,
    OpCode.SLICE, OpCode.UNSQUEEZE, OpCode.EXPAND,
}


class IRInterpreter:
    def __init__(self, program: Program):
        self.program = program

    def run(
        self,
        inputs: Mapping[str, np.ndarray],
        *,
        initializers: Mapping[str, np.ndarray] | None = None,
        function_name: str | None = None,
        max_steps: int = 1_000_000,
        collect_memory_stats: bool = False,
        copy_initializers: bool = True,
        memory_mode: str = "retain_all",
        observer: Callable[[str, np.ndarray], None] | None = None,
        fp32_mode: str = "native",
    ) -> ExecutionResult:
        """Run verified IR with strict bindings.

        Parameters and bound initializers require their exact declared static
        shape, including ``()`` for scalars. Builder intermediate values may
        retain the default ``()`` until the operation infers its result shape.
        An optional ``Function.returns`` signature is also exact; omitting it
        preserves the actual returned tensor's inferred shape and dtype.

        ``copy_initializers=False`` borrows only read-only NumPy arrays. The
        caller must not mutate their backing storage for the duration of this
        synchronous call. ``memory_mode='last_use'`` releases bindings after
        their final instruction use; it does not skip dead instructions or
        their errors. Both options require a pure, single-block SSA function
        ending in one RETURN; unsupported control flow fails explicitly.

        ``observer`` receives each numeric destination and a temporary
        read-only view after validation, before reclamation. It must not mutate
        backing storage or retain arrays if bounded memory is desired. Save
        selected checkpoints synchronously; observer allocations are excluded
        from interpreter accounting. Inputs and results are always copied.

        ``fp32_mode='reference'`` selects a versioned NumPy FP32 evaluation
        order and polynomial profile for numerical comparisons. It does not
        call ORT, change graph optimization, or change non-FP32 kernels. The
        default ``native`` retains the original NumPy evaluation policy.
        """
        if not isinstance(inputs, Mapping) or (
            initializers is not None and not isinstance(initializers, Mapping)
        ):
            raise IRExecutionError(
                "BindingError", "inputs and initializers must be mappings"
            )
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or max_steps < 1
        ):
            raise IRExecutionError(
                "InvalidOptions", "max_steps must be positive integer"
            )
        if not isinstance(collect_memory_stats, bool):
            raise IRExecutionError("InvalidOptions", "collect_memory_stats must be boolean")
        if not isinstance(copy_initializers, bool):
            raise IRExecutionError("InvalidOptions", "copy_initializers must be boolean")
        if not isinstance(memory_mode, str) or memory_mode not in ("retain_all", "last_use"):
            raise IRExecutionError("InvalidOptions", "memory_mode must be retain_all or last_use")
        if observer is not None and not callable(observer):
            raise IRExecutionError("InvalidOptions", "observer must be callable or None")
        if not isinstance(fp32_mode, str) or fp32_mode not in ("native", "reference"):
            raise IRExecutionError("InvalidOptions", "fp32_mode must be native or reference")
        passed, issues = verify_ir(self.program, stage="before-execution")
        if not passed:
            issue = next(i for i in issues if i.level.value == "error")
            position = (
                IRSourcePosition(issue.block_name, issue.instruction_index)
                if issue.block_name is not None and issue.instruction_index is not None
                else None
            )
            raise IRExecutionError(
                "InvalidProgram",
                str(issue),
                function_name=issue.function_name,
                position=position,
                value_name=issue.value_name,
            )
        functions = self.program.functions
        if function_name is None:
            if len(functions) != 1:
                raise IRExecutionError(
                    "EntryError",
                    "specify function_name when Program does not have exactly one function",
                )
            function = functions[0]
        else:
            matches = [f for f in functions if f.name == function_name]
            if len(matches) != 1:
                raise IRExecutionError(
                    "EntryError", f"entry must uniquely exist: {function_name}"
                )
            function = matches[0]

        def error(code, message, position=None, instr=None, value_name=None):
            if value_name is None and instr is not None:
                value_name = (
                    instr.dest.name
                    if instr.dest is not None
                    else (instr.operands[0].name if instr.operands else None)
                )
            return IRExecutionError(
                code,
                message,
                function_name=function.name,
                position=position,
                opcode=instr.opcode if instr else None,
                value_name=value_name,
            )

        last_uses = None
        if memory_mode == "last_use" or not copy_initializers:
            if len(function.blocks) != 1 or function.blocks[0].phi_nodes:
                raise error("InvalidOptions", "memory optimization requires a pure single-block function")
            instructions = function.blocks[0].instructions
            if (not instructions or instructions[-1].opcode != OpCode.RETURN
                    or any(inst.opcode not in _PURE_OPS for inst in instructions[:-1])):
                raise error("InvalidOptions", "memory optimization requires pure instructions and one final RETURN")
            if memory_mode == "last_use":
                last_uses = {}
                for index, inst in enumerate(instructions):
                    for operand in inst.operands:
                        last_uses[operand.name] = index

        for block in function.blocks:
            for index, instr in enumerate(block.instructions):
                position = IRSourcePosition(block.name, index)
                try:
                    if instr.opcode in _CONTROL_ATTRS:
                        unknown = set(instr.attrs) - _CONTROL_ATTRS[instr.opcode]
                        if unknown:
                            raise OpError(
                                "UnsupportedAttribute",
                                f"unsupported attributes: {sorted(unknown)}",
                            )
                        if instr.opcode == OpCode.ALLOCA:
                            size = integer(instr.attrs.get("size", 4), "size")
                            itemsize = DTYPES[instr.dest.dtype].itemsize
                            if size < itemsize or size % itemsize:
                                raise OpError(
                                    "MemoryError",
                                    "ALLOCA byte size must be positive and dtype-aligned",
                                )
                    else:
                        check_instruction(instr)
                except OpError as exc:
                    raise error(exc.code, str(exc), position, instr) from exc

        values = {}
        globals_ = {v.name: v for v in self.program.global_values}
        params = {v.name: v for v in function.params}
        if set(inputs) != set(params):
            raise error(
                "BindingError",
                f"input keys must match parameters: expected {sorted(params)}, got {sorted(inputs)}",
            )
        initializers = {} if initializers is None else initializers
        extra = set(initializers) - set(globals_)
        if extra or set(initializers) & set(params):
            raise error(
                "BindingError",
                f"invalid initializer bindings: {sorted(extra | (set(initializers) & set(params)))}",
            )

        def checked_shape(value, actual_shape, position=None, instr=None):
            # Boundary declarations describe actual tensors: () is a scalar,
            # not an unspecified shape. Builder intermediates are inferred
            # separately and do not enter this check unless explicitly shaped.
            if not isinstance(value.shape, (tuple, list)) or any(
                isinstance(d, (bool, np.bool_))
                or not isinstance(d, (int, np.integer)) or d < 0
                for d in value.shape
            ):
                raise error(
                    "ShapeError", "tensor boundary requires a static nonnegative shape",
                    position, instr,
                    value_name=value.name,
                )
            shape = tuple(int(d) for d in value.shape)
            if actual_shape != shape:
                raise error(
                    "ShapeError",
                    f"expected {shape}, got {actual_shape}",
                    position, instr,
                    value_name=value.name,
                )

        def checked_array(value, data, *, copy=False):
            if not isinstance(data, np.ndarray):
                raise error(
                    "DTypeError", "binding must be a NumPy array", value_name=value.name
                )
            if data.dtype != DTYPES[value.dtype]:
                raise error(
                    "DTypeError",
                    f"expected {DTYPES[value.dtype]}, got {data.dtype}",
                    value_name=value.name,
                )
            checked_shape(value, data.shape)
            if np.issubdtype(data.dtype, np.floating) and (
                np.isnan(data).any() or np.isposinf(data).any()
            ):
                raise error(
                    "NumericError",
                    "NaN and positive infinity are not supported",
                    value_name=value.name,
                )
            return data.copy() if copy else data

        for name, value in params.items():
            values[name] = checked_array(value, inputs[name], copy=True)
        referenced = {
            v.name
            for b in function.blocks
            for inst in b.instructions
            for v in inst.operands
        }
        for name in referenced & set(globals_):
            value = globals_[name]
            if name in initializers:
                data = checked_array(value, initializers[name], copy=copy_initializers)
                if not copy_initializers:
                    if data.flags.writeable:
                        raise error("BindingError", "borrowed initializer must be read-only", value_name=name)
                    data = data.view()
                    data.setflags(write=False)
                if value.is_constant:
                    literal = np.asarray(value.const_value, dtype=DTYPES[value.dtype])
                    if data.shape != () or not np.array_equal(data, literal):
                        raise error(
                            "BindingError",
                            "initializer disagrees with scalar constant",
                            value_name=name,
                        )
                values[name] = data
            elif value.is_constant:
                values[name] = np.asarray(value.const_value, dtype=DTYPES[value.dtype])
            else:
                raise error(
                    "BindingError", "global tensor data missing", value_name=name
                )

        memory = _MemoryObservation(values, params, globals_) if collect_memory_stats else None
        reclaimed_bindings = 0

        def reclaim(names):
            nonlocal reclaimed_bindings
            for name in names:
                if name in values:
                    del values[name]
                    reclaimed_bindings += 1

        if last_uses is not None:
            reclaim([name for name in values if name not in last_uses])

        def memory_result(returned):
            if memory is None:
                return None
            stats = memory.finish(values, returned)
            if memory_mode != "retain_all" or not copy_initializers or observer is not None:
                stats.update(memory_mode=memory_mode, copy_initializers=copy_initializers,
                             reclaimed_bindings=reclaimed_bindings, observer_enabled=observer is not None)
                stats["notes"].append(
                    "Opt-in policy applies: borrowed buffers remain resident if retained by the caller; "
                    "reclaimed bindings and observer allocations are not process RSS reductions.")
            return stats

        adapter = IRCFGAdapter(function)
        cfg = build_cfg(adapter)
        plan = adapter.execution_plan
        current, offset, steps = cfg.entry, 0, 0

        def resolve(value):
            if value.name in values:
                return values[value.name]
            if value.is_constant:
                return np.asarray(value.const_value, dtype=DTYPES[value.dtype])
            raise OpError(
                "UndefinedValue", f"value not defined on this path: {value.name}"
            )

        while True:
            block = cfg.nodes[current]
            if offset >= len(block.instructions):
                targets = cfg.successors(current)
                if len(targets) != 1:
                    raise error(
                        "InvalidProgram", "block has no unique legal continuation"
                    )
                current, offset = targets[0], 0
                continue
            instr = block.instructions[offset]
            position = plan.origins[id(instr)]
            if steps >= max_steps:
                raise error(
                    "StepLimitExceeded",
                    f"exceeded {max_steps} executed instructions",
                    position,
                    instr,
                )
            steps += 1
            offset += 1
            try:
                op = instr.opcode
                if op == OpCode.BR:
                    current, offset = instr.target, 0
                    continue
                if op == OpCode.BR_IF:
                    operands = [resolve(v) for v in instr.operands]
                    if any(
                        not isinstance(x, np.ndarray) or x.shape != () for x in operands
                    ):
                        raise OpError(
                            "ShapeError", "branch operands must be scalar arrays"
                        )
                    if len(operands) == 1:
                        condition = bool(operands[0] != 0)
                    else:
                        a, b = operands
                        condition = bool(
                            {
                                "==": np.equal,
                                "!=": np.not_equal,
                                "<": np.less,
                                "<=": np.less_equal,
                                ">": np.greater,
                                ">=": np.greater_equal,
                            }[instr.attrs["cmp_op"]](a, b)
                        )
                    current = instr.target.split(",")[0 if condition else 1].strip()
                    offset = 0
                    continue
                operands = [resolve(v) for v in instr.operands]
                if op == OpCode.RETURN:
                    if not operands:
                        return ExecutionResult(None, steps, tuple(issues),
                                               memory_result(None))
                    value = operands[0]
                    if not isinstance(value, np.ndarray):
                        raise OpError("MemoryError", "cannot return memory reference")
                    reference = instr.operands[0]
                    inferred_return = isinstance(reference.shape, (tuple, list)) and len(reference.shape) == 0
                    if not inferred_return or reference.name in params or reference.name in globals_:
                        checked_shape(reference, value.shape, position, instr)
                    if function.returns:
                        # verify_ir already checks declaration count and dtype.
                        checked_shape(function.returns[0], value.shape, position, instr)
                    if (
                        np.issubdtype(value.dtype, np.floating)
                        and not np.isfinite(value).all()
                    ):
                        raise OpError("NumericError", "return value must be finite")
                    returned = value.copy()
                    return ExecutionResult(returned, steps, tuple(issues),
                                           memory_result(returned))
                if op == OpCode.ALLOCA:
                    size = instr.attrs.get("size", 4)
                    dtype = DTYPES[instr.dest.dtype]
                    values[instr.dest.name] = _MemorySlot(
                        np.empty(size // dtype.itemsize, dtype=dtype)
                    )
                    if memory:
                        memory.observe(values)
                    continue
                if op in (OpCode.LOAD, OpCode.STORE):
                    slot = operands[0]
                    if not isinstance(slot, _MemorySlot):
                        raise OpError(
                            "MemoryError", "LOAD/STORE requires local ALLOCA reference"
                        )
                    if op == OpCode.STORE:
                        data = operands[1]
                        if (
                            not isinstance(data, np.ndarray)
                            or data.shape != ()
                            or data.dtype != slot.storage.dtype
                        ):
                            raise OpError(
                                "MemoryError",
                                "STORE requires scalar with matching element dtype",
                            )
                        slot.storage[0] = data
                        slot.initialized = True
                        continue
                    if not slot.initialized:
                        raise OpError("MemoryError", "LOAD before STORE")
                    result = np.asarray(slot.storage[0])
                else:
                    if any(not isinstance(x, np.ndarray) for x in operands):
                        raise OpError(
                            "MemoryError",
                            "numeric operation cannot use memory reference",
                        )
                    if position.stage == "for-step" and op == OpCode.ADD:
                        total = int(operands[0]) + int(operands[1])
                        bounds = np.iinfo(np.int32)
                        if total < bounds.min or total > bounds.max:
                            raise OpError("NumericError", "loop counter overflows i32")
                    result = (compute(instr, operands) if fp32_mode == "native"
                              else compute(instr, operands, fp32_mode=fp32_mode))
                if result.dtype != DTYPES[instr.dest.dtype]:
                    raise OpError(
                        "DTypeError",
                        f"kernel returned {result.dtype}, expected {DTYPES[instr.dest.dtype]}",
                    )
                if (
                    instr.dest.shape
                    and all(isinstance(d, int) and d >= 0 for d in instr.dest.shape)
                    and result.shape != instr.dest.shape
                ):
                    raise OpError(
                        "ShapeError",
                        f"result expected {instr.dest.shape}, got {result.shape}",
                    )
                values[instr.dest.name] = result
                if observer is not None:
                    observed = result.view()
                    observed.setflags(write=False)
                    try:
                        observer(instr.dest.name, observed)
                    except Exception as exc:
                        raise error("ObserverError", str(exc), position, instr) from exc
                    finally:
                        del observed
                if memory:
                    memory.observe(values)
                if last_uses is not None:
                    index = position.instruction_index
                    reclaim({value.name for value in instr.operands
                             if last_uses.get(value.name) == index}
                            | ({instr.dest.name} if instr.dest.name not in last_uses else set()))
                    # Do not accidentally pin reclaimed arrays in loop locals.
                    operands = []
                    result = None
            except OpError as exc:
                raise error(exc.code, str(exc), position, instr) from exc
            except FloatingPointError as exc:
                raise error("NumericError", str(exc), position, instr) from exc
            except (
                ValueError,
                TypeError,
                IndexError,
                OverflowError,
                MemoryError,
            ) as exc:
                raise error(
                    (
                        "ShapeError"
                        if isinstance(exc, (ValueError, IndexError))
                        else "ExecutionError"
                    ),
                    str(exc),
                    position,
                    instr,
                ) from exc
