"""Execute a shared Program with strict bindings and real control flow.

Integer ADD/SUB/MUL/NEG wrap at the declared width; DIV truncates toward zero
and rejects zero and min/-1. ALLOCA sizes are bytes; LOAD/STORE access the
first scalar slot only and reject reads before STORE. No host pointers exist.
Every run owns fresh state and copies external arrays and returned results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

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


@dataclass
class _MemorySlot:
    storage: np.ndarray
    initialized: bool = False


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
    ) -> ExecutionResult:
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
            if (
                value.shape
                and all(isinstance(d, int) and d >= 0 for d in value.shape)
                and data.shape != value.shape
            ):
                raise error(
                    "ShapeError",
                    f"expected {value.shape}, got {data.shape}",
                    value_name=value.name,
                )
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
                data = checked_array(value, initializers[name], copy=True)
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
                        return ExecutionResult(None, steps, tuple(issues))
                    value = operands[0]
                    if not isinstance(value, np.ndarray):
                        raise OpError("MemoryError", "cannot return memory reference")
                    if (
                        np.issubdtype(value.dtype, np.floating)
                        and not np.isfinite(value).all()
                    ):
                        raise OpError("NumericError", "return value must be finite")
                    return ExecutionResult(value.copy(), steps, tuple(issues))
                if op == OpCode.ALLOCA:
                    size = instr.attrs.get("size", 4)
                    dtype = DTYPES[instr.dest.dtype]
                    values[instr.dest.name] = _MemorySlot(
                        np.empty(size // dtype.itemsize, dtype=dtype)
                    )
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
                    result = compute(instr, operands)
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
