"""Compatibility layer for linear-scan allocator CFG/liveness.

The linear-scan allocators historically recovered blocks and liveness from a
flat ``LsInstruction`` stream inside this module.  The implementation below is
now a thin adapter over the unified CFG and liveness APIs in
``scratchv.analysis``:

* ``build_cfg`` partitions the stream and builds successor/predecessor edges.
* ``analyze_liveness`` computes block live-in/live-out sets with the generic
  backward worklist solver.

The old ``MachineCFG``/``MachineBasicBlock`` API is retained only so existing
allocator call sites remain source-compatible while they consume the unified
analysis results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from scratchv.analysis.cfg import CFGAdapter, build_cfg
from scratchv.analysis.liveness import analyze_liveness
from scratchv.backend.machine_semantics import get_machine_semantics
from scratchv.backend.machine_types import MachineOp


_CONDITIONAL_BRANCHES = {"beq", "bne", "blt", "bge", "bnez"}


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


@dataclass
class _LsBlock:
    name: str
    instructions: list[Any]


def _semantics(opcode: str):
    if opcode == ".label":
        return None
    try:
        return get_machine_semantics(MachineOp(opcode))
    except ValueError:
        return None


def _is_terminator(inst: Any) -> bool:
    semantics = _semantics(inst.opcode)
    return bool(semantics and semantics.is_terminator)


def _label_name(inst: Any) -> str:
    if inst.operands:
        return str(inst.operands[0])
    return str(inst.comment or "")


def _branch_target(inst: Any) -> Optional[str]:
    semantics = _semantics(inst.opcode)
    if not semantics or not semantics.target_from_comment:
        return None
    target = getattr(inst, "target", None)
    if target:
        return str(target)
    if inst.operands:
        return str(inst.operands[-1])
    return str(inst.comment or "") or None


class _LsCFGAdapter:
    """CFG adapter for the linear-scan ``LsInstruction`` representation."""

    def __init__(self, instructions: Sequence[Any]):
        self._blocks = _partition_ls_stream(instructions)

    @property
    def function_name(self) -> str:
        return "__linear_scan"

    def blocks(self) -> Sequence[_LsBlock]:
        return self._blocks

    def block_name(self, block: _LsBlock) -> str:
        return block.name

    def instructions(self, block: _LsBlock) -> Sequence[Any]:
        return block.instructions

    def is_label(self, instr: Any) -> bool:
        return instr.opcode == ".label"

    def is_terminator(self, instr: Any) -> bool:
        return _is_terminator(instr)

    def branch_targets(self, instr: Any) -> Sequence[str]:
        target = _branch_target(instr)
        return [target] if target else []

    def has_fallthrough(self, instr: Any) -> bool:
        return instr.opcode in _CONDITIONAL_BRANCHES

    def opcode_name(self, instr: Any) -> str:
        return instr.opcode


class _LsUseDefProvider:
    """Use/def provider for ``LsInstruction`` values.

    ``LsInstruction`` already stores the register-allocation use/def sets,
    which are themselves derived from ``machine_semantics.py`` during
    ``block_from_machine_instrs``.
    """

    def uses(self, instr: Any):
        return frozenset(instr.uses)

    def defs(self, instr: Any):
        return frozenset(instr.defines)

    def edge_uses(self, pred, succ):
        return frozenset()

    def phi_defs(self, block):
        return frozenset()


def _partition_ls_stream(instructions: Sequence[Any]) -> list[_LsBlock]:
    """Split a flat LsInstruction stream at labels and terminators."""

    if not instructions:
        return []

    leaders = {0}
    for index, inst in enumerate(instructions):
        if inst.opcode == ".label":
            leaders.add(index)
        if _is_terminator(inst) and index + 1 < len(instructions):
            leaders.add(index + 1)

    starts = sorted(leaders)
    blocks: list[_LsBlock] = []
    for ordinal, start_index in enumerate(starts):
        stop_index = (
            starts[ordinal + 1]
            if ordinal + 1 < len(starts)
            else len(instructions)
        )
        body = list(instructions[start_index:stop_index])
        first = body[0]
        if first.opcode == ".label":
            name = _label_name(first)
        else:
            name = f".__ls_block_{ordinal}"
        if not name:
            name = f".__ls_block_{ordinal}"
        blocks.append(_LsBlock(name=name, instructions=body))

    return blocks


def analyze_control_flow(instructions: list[Any]) -> MachineCFG:
    """Split *instructions* into blocks and compute liveness.

    This is the compatibility entry point used by both linear-scan allocators.
    It delegates to :func:`scratchv.analysis.cfg.build_cfg` and
    :func:`scratchv.analysis.liveness.analyze_liveness` rather than
    reimplementing graph construction or data-flow iteration locally.
    """

    if not instructions:
        return MachineCFG([], {}, {})

    cfg = build_cfg(_LsCFGAdapter(instructions))
    liveness = analyze_liveness(cfg, _LsUseDefProvider())

    blocks: list[MachineBasicBlock] = []
    by_name: dict[str, MachineBasicBlock] = {}
    instruction_to_block: dict[int, str] = {}

    for block_id, node in cfg.nodes.items():
        block_instructions = (
            list(node.instructions)
            if not isinstance(node.instructions, int)
            else []
        )
        body = block_instructions
        if not body:
            start = 0
            end = 0
        else:
            start = min(inst.id for inst in body)
            end = max(inst.id for inst in body) + 1

        facts = liveness.blocks.get(block_id)
        block = MachineBasicBlock(
            name=block_id,
            instructions=body,
            start=start,
            end=end,
            successors=set(cfg.successors(block_id)),
            predecessors=set(cfg.predecessors(block_id)),
            uses=set(facts.uses) if facts else set(),
            defines=set(facts.defs) if facts else set(),
            live_in=set(facts.live_in) if facts else set(),
            live_out=set(facts.live_out) if facts else set(),
        )
        blocks.append(block)
        by_name[block_id] = block
        for inst in body:
            instruction_to_block[inst.id] = block_id

    # Preserve deterministic list order even though the underlying CFG uses a
    # dict: most callers use ``by_name``, but benchmarks sometimes inspect
    # ``blocks`` directly.
    blocks.sort(key=lambda block: block.start)

    return MachineCFG(blocks, by_name, instruction_to_block)


def apply_cfg_liveness(intervals: list[Any], cfg: MachineCFG) -> list[Any]:
    """Extend intervals to cover the block boundaries required by the CFG.

    The allocator still uses conservative single ranges, but those ranges now
    include values carried through blocks even when a block contains no local
    use.  ``cfg`` is the result of :func:`analyze_control_flow`, whose
    ``live_in``/``live_out`` sets come from the unified liveness analysis.
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


__all__ = [
    "MachineBasicBlock",
    "MachineCFG",
    "analyze_control_flow",
    "apply_cfg_liveness",
]
