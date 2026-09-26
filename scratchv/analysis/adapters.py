"""CFG adapters for IR and machine instruction streams.

The unified CFG builder only understands labels, terminators, targets, and
fallthrough.  These adapters translate ScratchV's two concrete instruction
representations into that vocabulary.

IR adapter
----------
* Normalises ``FOR``/``ENDFOR`` into ``LOAD_CONST`` + ``BR`` + ``BR_IF`` +
  labels before the CFG is built.
* Preserves explicit IR block names by emitting a ``LABEL`` at each block
  boundary.

Machine adapter
---------------
* Works on the flat ``list[MachineInstr]`` produced by instruction selection.
* Partitions the stream at ``LABEL`` instructions and real terminators.
* ``CALL`` is *not* a terminator and therefore keeps fallthrough.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from scratchv.ir.types import (
    BasicBlock,
    DataType,
    Function,
    Instruction,
    OpCode,
    Value,
)
from scratchv.backend.machine_types import MachineInstr, MachineOp
from scratchv.backend.machine_semantics import get_machine_semantics

from scratchv.analysis.cfg import CFGAdapter
from scratchv.analysis.usedef import IRUseDefProvider, MachineUseDefProvider


@dataclass
class _NormalizedIRBlock:
    name: str
    instructions: list[Instruction]


# ---------------------------------------------------------------------------
# IR helper constructors
# ---------------------------------------------------------------------------

def _label(name: str) -> Instruction:
    return Instruction(opcode=OpCode.LABEL, target=name)


def _br(target: str) -> Instruction:
    return Instruction(opcode=OpCode.BR, target=target)


def _br_if(
    lhs: Value,
    rhs: Value,
    true_target: str,
    false_target: str,
    cmp_op: str,
) -> Instruction:
    return Instruction(
        opcode=OpCode.BR_IF,
        operands=[lhs, rhs],
        target=f"{true_target},{false_target}",
        attrs={"cmp_op": cmp_op},
    )


def _load_const(dest: Value, value: int) -> Instruction:
    return Instruction(
        opcode=OpCode.LOAD_CONST,
        dest=dest,
        attrs={"value": value},
    )


def _add(dest: Value, lhs: Value, rhs: Value) -> Instruction:
    return Instruction(opcode=OpCode.ADD, dest=dest, operands=[lhs, rhs])


def _const_int(name: str, value: int) -> Value:
    return Value(
        name=name,
        dtype=DataType.INT32,
        is_constant=True,
        const_value=value,
    )


def _normalize_for_endfor(
    function: Function,
) -> list[Instruction]:
    """Normalise FOR/ENDFOR before partitioning.

    The resulting stream only uses labels and the normal control-flow opcodes
    ``BR``/``BR_IF``/``RETURN``.  Loop variables are still initialised and
    incremented, which keeps later use/def analysis faithful.
    """

    flat: list[Instruction] = []
    for block in function.blocks:
        flat.append(_label(block.name))
        flat.extend(block.instructions)

    normalized: list[Instruction] = []
    loop_stack: list[dict[str, Any]] = []
    synthetic = 0

    for instr in flat:
        if instr.opcode is OpCode.FOR:
            synthetic += 1
            iv = instr.dest
            if iv is None:
                raise ValueError("FOR instruction is missing its loop variable")

            start = int(instr.attrs.get("start", 0))
            end = int(instr.attrs.get("end", 0))
            step = int(instr.attrs.get("step", 1))

            header = f"for_hdr{synthetic}"
            body = f"for_body{synthetic}"
            exit_label = f"for_exit{synthetic}"

            end_value = _const_int(f"for_end_{synthetic}", end)
            step_value = _const_int(f"for_step_{synthetic}", step)

            normalized.append(_load_const(iv, start))
            normalized.append(_br(header))
            normalized.append(_label(header))
            normalized.append(
                _br_if(iv, end_value, exit_label, body, cmp_op=">=")
            )
            normalized.append(_label(body))

            loop_stack.append(
                {
                    "iv": iv,
                    "step": step_value,
                    "header": header,
                    "exit": exit_label,
                }
            )
            continue

        if instr.opcode is OpCode.ENDFOR:
            if not loop_stack:
                raise ValueError("ENDFOR without matching FOR")
            ctx = loop_stack.pop()
            normalized.append(_add(ctx["iv"], ctx["iv"], ctx["step"]))
            normalized.append(_br(ctx["header"]))
            normalized.append(_label(ctx["exit"]))
            continue

        normalized.append(instr)

    if loop_stack:
        raise ValueError("unterminated FOR loop")

    return normalized


def _partition_ir_stream(
    instructions: list[Instruction],
) -> list[_NormalizedIRBlock]:
    """Partition a normalised IR stream at labels and terminators.

    Empty basic blocks are preserved so analyses and validation can report
    them consistently.
    """

    blocks: list[_NormalizedIRBlock] = []
    current: list[Instruction] = []
    current_name: Optional[str] = None
    auto_id = 0

    for instr in instructions:
        if instr.opcode is OpCode.LABEL:
            if current_name is not None:
                blocks.append(
                    _NormalizedIRBlock(name=current_name, instructions=current)
                )
                auto_id += 1
            current_name = instr.target or f"b{auto_id}"
            current = []
            continue

        if current_name is None:
            current_name = "entry" if not blocks else f"b{auto_id}"
            if blocks:
                auto_id += 1

        current.append(instr)

        if instr.opcode in (OpCode.BR, OpCode.BR_IF, OpCode.RETURN):
            blocks.append(
                _NormalizedIRBlock(name=current_name, instructions=current)
            )
            current_name = None
            current = []

    if current_name is not None or current:
        blocks.append(
            _NormalizedIRBlock(name=current_name or f"b{auto_id}",
                               instructions=current)
        )

    return blocks


class IRCFGAdapter:
    """Adapter for an IR ``Function``."""

    def __init__(self, function: Function):
        self._function = function
        normalized = _normalize_for_endfor(function)
        self._blocks = _partition_ir_stream(normalized)
        self.use_def_provider = IRUseDefProvider()

    @property
    def function_name(self) -> str:
        return self._function.name

    def blocks(self) -> Sequence[Any]:
        return self._blocks

    def block_name(self, block: Any) -> str:
        return block.name

    def instructions(self, block: Any) -> Sequence[Instruction]:
        return block.instructions

    def is_label(self, instr: Instruction) -> bool:
        return instr.opcode is OpCode.LABEL

    def is_terminator(self, instr: Instruction) -> bool:
        return instr.opcode in (OpCode.BR, OpCode.BR_IF, OpCode.RETURN)

    def branch_targets(self, instr: Instruction) -> Sequence[str]:
        if instr.opcode is OpCode.BR:
            return [instr.target] if instr.target else []
        if instr.opcode is OpCode.BR_IF:
            return [
                item.strip()
                for item in (instr.target or "").split(",")
                if item.strip()
            ]
        return []

    def has_fallthrough(self, instr: Instruction) -> bool:
        if instr.opcode is not OpCode.BR_IF:
            return False
        return len(self.branch_targets(instr)) == 1

    def opcode_name(self, instr: Instruction) -> str:
        return instr.opcode.name

    def uses(self, instr: Instruction):
        return self.use_def_provider.uses(instr)

    def defs(self, instr: Instruction):
        return self.use_def_provider.defs(instr)

    def edge_uses(self, pred, succ):
        return self.use_def_provider.edge_uses(pred, succ)

    def phi_defs(self, block):
        return self.use_def_provider.phi_defs(block)


# ---------------------------------------------------------------------------
# Machine adapter
# ---------------------------------------------------------------------------

@dataclass
class _MachineBlock:
    name: str
    instructions: list[MachineInstr]


_MACHINE_TERMINATORS = {
    MachineOp.J,
    MachineOp.JAL,
    MachineOp.JALR,
    MachineOp.BEQ,
    MachineOp.BNE,
    MachineOp.BLT,
    MachineOp.BGE,
    MachineOp.BNEZ,
}

_MACHINE_CONDITIONAL_BRANCHES = {
    MachineOp.BEQ,
    MachineOp.BNE,
    MachineOp.BLT,
    MachineOp.BGE,
    MachineOp.BNEZ,
}


def _strip_function_label(
    instructions: Sequence[MachineInstr],
) -> list[MachineInstr]:
    """Drop the leading function-name label emitted by instruction selection.

    ``InstructionSelector`` emits ``LABEL main`` followed by ``LABEL .entry``.
    The first label is not a basic-block boundary; the first block label is
    the second label.  We use the convention that function labels do not begin
    with ``"."``.
    """

    instrs = list(instructions)
    if not instrs or instrs[0].op is not MachineOp.LABEL:
        return instrs

    first_target = instrs[0].target or instrs[0].comment or ""
    if first_target.startswith("."):
        return instrs
    return instrs[1:]


def _partition_machine_stream(
    instructions: Sequence[MachineInstr],
) -> list[_MachineBlock]:
    """Partition a flat machine stream into named basic blocks.

    Empty blocks are preserved so block boundaries remain observable.
    """

    blocks: list[_MachineBlock] = []
    current: list[MachineInstr] = []
    current_name: Optional[str] = None
    auto_id = 0

    for instr in instructions:
        if instr.op is MachineOp.LABEL:
            if current_name is not None:
                blocks.append(
                    _MachineBlock(name=current_name, instructions=current)
                )
                auto_id += 1
            current_name = instr.target or instr.comment or f"b{auto_id}"
            current = []
            continue

        if current_name is None:
            current_name = "entry" if not blocks else f"b{auto_id}"
            if blocks:
                auto_id += 1

        current.append(instr)

        if instr.op in _MACHINE_TERMINATORS:
            blocks.append(
                _MachineBlock(name=current_name, instructions=current)
            )
            current_name = None
            current = []

    if current_name is not None or current:
        blocks.append(
            _MachineBlock(name=current_name or f"b{auto_id}",
                          instructions=current)
        )

    return blocks


class MachineCFGAdapter:
    """Adapter for one function's flat machine instruction list."""

    def __init__(self, function_name: str, instructions: Sequence[MachineInstr]):
        self._function_name = function_name
        self._blocks = _partition_machine_stream(
            _strip_function_label(instructions)
        )
        self.use_def_provider = MachineUseDefProvider()

    @property
    def function_name(self) -> str:
        return self._function_name

    def blocks(self) -> Sequence[Any]:
        return self._blocks

    def block_name(self, block: Any) -> str:
        return block.name

    def instructions(self, block: Any) -> Sequence[MachineInstr]:
        return block.instructions

    def is_label(self, instr: MachineInstr) -> bool:
        return instr.op is MachineOp.LABEL

    def is_terminator(self, instr: MachineInstr) -> bool:
        return instr.op in _MACHINE_TERMINATORS

    def branch_targets(self, instr: MachineInstr) -> Sequence[str]:
        target = instr.target
        if not target and get_machine_semantics(instr.op).target_from_comment:
            target = instr.comment
        return [target] if target else []

    def has_fallthrough(self, instr: MachineInstr) -> bool:
        return instr.op in _MACHINE_CONDITIONAL_BRANCHES

    def opcode_name(self, instr: MachineInstr) -> str:
        return instr.op.name

    def uses(self, instr: MachineInstr):
        return self.use_def_provider.uses(instr)

    def defs(self, instr: MachineInstr):
        return self.use_def_provider.defs(instr)

    def edge_uses(self, pred, succ):
        return self.use_def_provider.edge_uses(pred, succ)

    def phi_defs(self, block):
        return self.use_def_provider.phi_defs(block)

    def clobbers(self, instr: MachineInstr):
        return self.use_def_provider.clobbers(instr)


__all__ = [
    "IRCFGAdapter",
    "MachineCFGAdapter",
]
