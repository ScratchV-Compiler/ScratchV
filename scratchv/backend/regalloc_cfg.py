"""Control-flow and liveness analysis for machine-level register allocation.

The linear allocators consume a flat ``LsInstruction`` stream.  This module
recovers basic blocks from labels and terminators, builds successor edges, and
computes the conventional backward ``live_in``/``live_out`` data-flow sets.
It deliberately has no dependency on either allocator implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from scratchv.backend.machine_semantics import get_machine_semantics
from scratchv.backend.machine_types import MachineOp


_CONDITIONAL_BRANCHES = {"beq", "bne", "blt", "bge", "bnez"}
_DIRECT_JUMPS = {"j", "jal"}


@dataclass
class MachineBasicBlock:
    """One recovered machine basic block and its liveness facts."""

    name: str
    instructions: list[Any]
    start: int
    end: int
    successors: set[str] = field(default_factory=set)
    predecessors: set[str] = field(default_factory=set)
    uses: set[str] = field(default_factory=set)
    defines: set[str] = field(default_factory=set)
    live_in: set[str] = field(default_factory=set)
    live_out: set[str] = field(default_factory=set)


@dataclass
class MachineCFG:
    """Recovered control-flow graph for a flat machine instruction stream."""

    blocks: list[MachineBasicBlock]
    by_name: dict[str, MachineBasicBlock]
    instruction_to_block: dict[int, str]


def _semantics(opcode: str):
    try:
        return get_machine_semantics(MachineOp(opcode))
    except ValueError:
        return None


def _is_terminator(inst: Any) -> bool:
    semantics = _semantics(inst.opcode)
    return bool(semantics and semantics.is_terminator)


def _target(inst: Any) -> str | None:
    semantics = _semantics(inst.opcode)
    if not semantics or not semantics.target_from_comment:
        return None
    return inst.operands[-1] if inst.operands else None


def analyze_control_flow(instructions: list[Any]) -> MachineCFG:
    """Split *instructions* into blocks and compute live-in/live-out sets.

    ``call`` is intentionally not a terminator: it has a fallthrough edge and
    its ABI clobbers are handled by allocation/code generation, not the CFG.
    Direct targets outside this stream (for example an external ``jal``) do
    not create an internal successor edge.
    """

    if not instructions:
        return MachineCFG([], {}, {})

    leaders = {0}
    for index, inst in enumerate(instructions):
        if inst.opcode == ".label":
            leaders.add(index)
        if _is_terminator(inst) and index + 1 < len(instructions):
            leaders.add(index + 1)

    starts = sorted(leaders)
    blocks: list[MachineBasicBlock] = []
    instruction_to_block: dict[int, str] = {}
    for ordinal, start_index in enumerate(starts):
        stop_index = starts[ordinal + 1] if ordinal + 1 < len(starts) else len(instructions)
        body = instructions[start_index:stop_index]
        first = body[0]
        if first.opcode == ".label":
            name = first.operands[0] if first.operands else first.comment
        else:
            name = f".__ls_block_{ordinal}"
        if not name:
            name = f".__ls_block_{ordinal}"
        block = MachineBasicBlock(
            name=name,
            instructions=body,
            start=body[0].id,
            end=body[-1].id + 1,
        )
        for inst in body:
            instruction_to_block[inst.id] = name
            block.uses |= inst.uses - block.defines
            block.defines |= inst.defines
        blocks.append(block)

    by_name = {block.name: block for block in blocks}
    if len(by_name) != len(blocks):
        raise ValueError("duplicate machine basic-block label")

    for index, block in enumerate(blocks):
        last = block.instructions[-1]
        target = _target(last)
        fallthrough = blocks[index + 1].name if index + 1 < len(blocks) else None

        if last.opcode in _CONDITIONAL_BRANCHES:
            if target in by_name:
                block.successors.add(target)
            if fallthrough is not None:
                block.successors.add(fallthrough)
        elif last.opcode in _DIRECT_JUMPS:
            if target in by_name:
                block.successors.add(target)
        elif last.opcode == "jalr":
            pass  # Indirect target / return: no statically known successor.
        elif fallthrough is not None:
            block.successors.add(fallthrough)

    for block in blocks:
        for successor in block.successors:
            by_name[successor].predecessors.add(block.name)

    changed = True
    while changed:
        changed = False
        for block in reversed(blocks):
            live_out = set().union(
                *(by_name[name].live_in for name in block.successors)
            ) if block.successors else set()
            live_in = block.uses | (live_out - block.defines)
            if live_in != block.live_in or live_out != block.live_out:
                block.live_in = live_in
                block.live_out = live_out
                changed = True

    return MachineCFG(blocks, by_name, instruction_to_block)


def apply_cfg_liveness(intervals: list[Any], cfg: MachineCFG) -> list[Any]:
    """Extend intervals to cover the block boundaries required by the CFG.

    The allocator still uses conservative single ranges, but those ranges now
    include values carried through blocks even when a block contains no local
    use.  This is the safe first step before lifetime-hole/split-interval work.
    """

    by_vreg = {interval.vreg: interval for interval in intervals}
    for block in cfg.blocks:
        for vreg in block.live_in:
            interval = by_vreg.get(vreg)
            if interval is not None:
                interval.start = min(interval.start, block.start)
        for vreg in block.live_out:
            interval = by_vreg.get(vreg)
            if interval is not None:
                interval.end = max(interval.end, block.end)
    return sorted(intervals, key=lambda iv: (iv.start, iv.end, iv.vreg))
