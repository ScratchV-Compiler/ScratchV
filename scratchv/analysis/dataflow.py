"""Reusable data-flow framework for ScratchV CFG analyses.

Liveness is a *backward* analysis, while constant propagation is a *forward*
analysis.  They use different transfer functions and lattices, but the same
worklist engine.  This module provides that engine plus a small constant
propagation analysis that consumes the unified CFG.
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from scratchv.analysis.cfg import (
    BlockId,
    ControlFlowGraph,
    ValueId,
)


class Direction(enum.Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


class DataflowAnalysis(Protocol):
    """Contract for an analysis solved by :func:`run_dataflow`."""

    direction: Direction

    def initial(self) -> Any:
        """Return the initial lattice value for a block."""

    def boundary(self, block: BlockId) -> Any:
        """Return the boundary value for the analysis direction."""

    def meet(self, values: Sequence[Any]) -> Any:
        """Combine incoming values (must be commutative/associative)."""

    def transfer(self, block: BlockId, value: Any) -> Any:
        """Apply a block's instructions to an incoming value."""


@dataclass(frozen=True)
class DataflowResult:
    """Results of a fixed-point data-flow solve."""

    in_values: Mapping[BlockId, Any]
    out_values: Mapping[BlockId, Any]


def run_dataflow(
    cfg: ControlFlowGraph,
    analysis: DataflowAnalysis,
) -> DataflowResult:
    """Solve a monotone data-flow problem over ``cfg``.

    * Forward analyses propagate from ``cfg.entry`` toward successors.
    * Backward analyses propagate from blocks with no successors toward
      predecessors.

    The solver is deterministic: neighbour values are passed to ``meet`` in
    sorted block-name order.
    """

    if analysis.direction is Direction.FORWARD:
        return _run_forward(cfg, analysis)
    if analysis.direction is Direction.BACKWARD:
        return _run_backward(cfg, analysis)
    raise ValueError(f"unsupported dataflow direction: {analysis.direction}")


def _run_forward(
    cfg: ControlFlowGraph,
    analysis: DataflowAnalysis,
) -> DataflowResult:
    in_values = {block: analysis.initial() for block in cfg.nodes}
    out_values = {block: analysis.initial() for block in cfg.nodes}

    in_values[cfg.entry] = analysis.boundary(cfg.entry)
    out_values[cfg.entry] = analysis.transfer(
        cfg.entry, in_values[cfg.entry]
    )

    worklist = deque(cfg.successors(cfg.entry))
    in_worklist = set(worklist)

    while worklist:
        block = worklist.popleft()
        in_worklist.discard(block)

        preds = sorted(cfg.predecessors(block))
        if preds:
            new_in = analysis.meet(
                [out_values[pred] for pred in preds]
            )
        else:
            new_in = (
                analysis.boundary(block)
                if block == cfg.entry
                else analysis.initial()
            )

        if new_in != in_values[block]:
            in_values[block] = new_in
            new_out = analysis.transfer(block, new_in)
            if new_out != out_values[block]:
                out_values[block] = new_out
                for succ in sorted(cfg.successors(block)):
                    if succ not in in_worklist:
                        worklist.append(succ)
                        in_worklist.add(succ)

    return DataflowResult(in_values=in_values, out_values=out_values)


def _run_backward(
    cfg: ControlFlowGraph,
    analysis: DataflowAnalysis,
) -> DataflowResult:
    in_values = {block: analysis.initial() for block in cfg.nodes}
    out_values = {block: analysis.initial() for block in cfg.nodes}

    worklist = deque()
    in_worklist = set()

    for block in cfg.nodes:
        if not cfg.successors(block):
            out_values[block] = analysis.boundary(block)
            in_values[block] = analysis.transfer(block, out_values[block])
            for pred in cfg.predecessors(block):
                if pred not in in_worklist:
                    worklist.append(pred)
                    in_worklist.add(pred)

    while worklist:
        block = worklist.popleft()
        in_worklist.discard(block)

        succs = sorted(cfg.successors(block))
        if succs:
            new_out = analysis.meet(
                [in_values[succ] for succ in succs]
            )
        else:
            new_out = analysis.boundary(block)

        if new_out != out_values[block]:
            out_values[block] = new_out
            new_in = analysis.transfer(block, new_out)
            if new_in != in_values[block]:
                in_values[block] = new_in
                for pred in sorted(cfg.predecessors(block)):
                    if pred not in in_worklist:
                        worklist.append(pred)
                        in_worklist.add(pred)

    return DataflowResult(in_values=in_values, out_values=out_values)


# ---------------------------------------------------------------------------
# Constant propagation
# ---------------------------------------------------------------------------

class _Constant:
    __slots__ = ("value",)

    def __init__(self, value: int | float):
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Constant) and self.value == other.value

    def __hash__(self) -> int:
        return hash(("const", self.value))

    def __repr__(self) -> str:
        return f"const({self.value!r})"


class _Overdefined:
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Overdefined)

    def __hash__(self) -> int:
        return hash("overdefined")

    def __repr__(self) -> str:
        return "overdefined"


OVERDEFINED = _Overdefined()


def _meet_envs(
    values: Sequence[Mapping[ValueId, Any]],
) -> dict[ValueId, Any]:
    """Meet a set of constant environments.

    Missing keys mean "undefined", the top of the constant lattice.  Meeting
    the same constant with itself keeps the constant; different constants, or
    any explicit ``OVERDEFINED``, degrade to ``OVERDEFINED``.
    """

    if not values:
        return {}

    result: dict[ValueId, Any] = {}
    keys: set[ValueId] = set()
    for env in values:
        keys.update(env.keys())

    for key in keys:
        combined: Any = None
        first = True
        for env in values:
            value = env.get(key)
            if value is None:
                continue
            if first:
                combined = value
                first = False
            elif combined is OVERDEFINED:
                continue
            elif value is OVERDEFINED or combined != value:
                combined = OVERDEFINED
        if combined is not None:
            result[key] = combined

    return result


def _constant_value(instr: Any) -> int | float | None:
    opcode = getattr(instr, "opcode", None)
    if opcode is None:
        return None
    op_name = getattr(opcode, "name", str(opcode))
    if op_name != "LOAD_CONST":
        return None
    raw = getattr(instr, "attrs", {}).get("value")
    if isinstance(raw, (int, float)):
        return raw
    return None


def _value_name(value: Any) -> ValueId | None:
    name = getattr(value, "name", None)
    return name if isinstance(name, str) else None


class ConstantPropagation:
    """Forward constant propagation over an IR CFG.

    The lattice is per-value: missing means undefined (top), ``_Constant``
    means known, and ``OVERDEFINED`` means join produced conflicting values.
    This analysis intentionally stops at foldability/join facts and does not
    rewrite instructions.
    """

    direction = Direction.FORWARD

    def __init__(self, cfg: ControlFlowGraph):
        self.cfg = cfg

    def initial(self) -> dict[ValueId, Any]:
        return {}

    def boundary(self, block: BlockId) -> dict[ValueId, Any]:
        return {}

    def meet(
        self,
        values: Sequence[Mapping[ValueId, Any]],
    ) -> dict[ValueId, Any]:
        return _meet_envs(values)

    def transfer(
        self,
        block: BlockId,
        value: Mapping[ValueId, Any],
    ) -> dict[ValueId, Any]:
        env = dict(value)
        node = self.cfg.nodes.get(block)
        if node is None:
            return env

        instructions = (
            [] if isinstance(node.instructions, int)
            else list(node.instructions)
        )

        for instr in instructions:
            opcode = getattr(instr, "opcode", None)
            if opcode is None:
                continue
            op_name = getattr(opcode, "name", str(opcode))

            dest = getattr(instr, "dest", None)
            dest_id = _value_name(dest) if dest is not None else None

            if op_name == "LOAD_CONST":
                const = _constant_value(instr)
                if dest_id is not None and const is not None:
                    env[dest_id] = _Constant(const)
                elif dest_id is not None:
                    env[dest_id] = OVERDEFINED
                continue

            if op_name in ("ADD", "SUB", "MUL", "DIV"):
                operands = getattr(instr, "operands", []) or []
                if len(operands) >= 2:
                    left = self._resolve_operand(operands[0], env)
                    right = self._resolve_operand(operands[1], env)
                    if dest_id is not None:
                        env[dest_id] = self._eval_binary(
                            op_name, left, right
                        )
                elif dest_id is not None:
                    env[dest_id] = OVERDEFINED
                continue

            # Any other definition conservatively becomes overdefined.
            if dest_id is not None:
                env[dest_id] = OVERDEFINED

        return env

    @staticmethod
    def _resolve_operand(operand: Any, env: Mapping[ValueId, Any]) -> Any:
        if getattr(operand, "is_constant", False):
            value = getattr(operand, "const_value", None)
            if isinstance(value, (int, float)):
                return _Constant(value)
        return env.get(_value_name(operand))

    @staticmethod
    def _eval_binary(
        op_name: str,
        left: Any,
        right: Any,
    ) -> Any:
        if not isinstance(left, _Constant) or not isinstance(right, _Constant):
            return OVERDEFINED

        lhs = left.value
        rhs = right.value
        try:
            if op_name == "ADD":
                return _Constant(lhs + rhs)
            if op_name == "SUB":
                return _Constant(lhs - rhs)
            if op_name == "MUL":
                return _Constant(lhs * rhs)
            if op_name == "DIV":
                if rhs == 0:
                    return OVERDEFINED
                return _Constant(lhs / rhs)
        except (ArithmeticError, TypeError):
            return OVERDEFINED
        return OVERDEFINED

    def run(self) -> DataflowResult:
        return run_dataflow(self.cfg, self)


__all__ = [
    "Direction",
    "DataflowAnalysis",
    "DataflowResult",
    "run_dataflow",
    "ConstantPropagation",
    "OVERDEFINED",
]