"""Move invariant instructions proven safe to speculate or certain to execute.

Only structured loops contained in one IR block are transformed. Safety follows
the checked IR semantics: purity alone does not permit speculative execution.
"""

from __future__ import annotations

from dataclasses import replace

from scratchv.analysis.adapters import IRCFGAdapter
from scratchv.analysis.cfg import build_cfg, compute_dominators
from scratchv.analysis.ir_verifier import verify_ir
from scratchv.ir.types import BasicBlock, Function, Instruction, OpCode, Program
from scratchv.optimizer.hoist_safety import HoistSafety
from scratchv.pass_interface import OptimizationPass


class LICM(OptimizationPass):
    """Hoist only operations proved safe on every input allowed by the IR."""

    name = "licm"

    def optimize(self, program: Program) -> int:
        # Undefined values and broken CFGs cannot establish availability proofs.
        if not verify_ir(program)[0]:
            return 0
        changes = 0
        for func in program.functions:
            for block in func.blocks:
                changes += self._process_region(
                    program, func, block, 0, len(block.instructions)
                )
        return changes

    def _process_region(self, program, func, block, start, stop):
        changes = 0
        instrs = block.instructions
        i = start
        while i < stop:
            if instrs[i].opcode != OpCode.FOR:
                i += 1
                continue
            end = self._find_matching_endfor(instrs, i)
            if end is None or end >= stop:
                i += 1
                continue
            # Inner-loop hoists stay inside their enclosing loop until that
            # loop independently proves them safe and invariant.
            changes += self._process_region(program, func, block, i + 1, end)
            safety = HoistSafety(program, func)
            available = self._available_before(program, func, block, i)
            hoisted, kept = [], []
            depth = 0
            # Verified FOR bounds are constant i32 values with positive step.
            # The straight-line prefix executes on the first iteration when
            # start < end. Do not carry this proof past a retained instruction
            # that may fail, affect memory, or change control flow.
            guaranteed_prefix = instrs[i].attrs["start"] < instrs[i].attrs["end"]
            for instr in instrs[i + 1 : end]:
                if instr.opcode == OpCode.FOR:
                    depth += 1
                safe_to_speculate = depth == 0 and safety.is_safe(instr)
                operands_available = all(
                    v.name in available
                    or (v.is_constant and v.name not in safety.definitions)
                    for v in instr.operands
                )
                can_move = depth == 0 and operands_available and (
                    safe_to_speculate
                    or (guaranteed_prefix and safety.can_hoist_when_guaranteed(instr))
                )
                if can_move:
                    hoisted.append(instr)
                    available.add(instr.dest.name)
                else:
                    kept.append(instr)
                # Hoisted computations retain their relative order, including
                # possible failures. A safe retained computation cannot prevent
                # reaching the next instruction; all other retained operations
                # stop the must-execute proof.
                guaranteed_prefix = guaranteed_prefix and (
                    can_move or safe_to_speculate
                )
                if instr.opcode == OpCode.ENDFOR:
                    depth -= 1
            # Rebuild once, preserving both dependency order and instruction
            # identity. Removing/inserting by stale indices can move the FOR.
            instrs[i : end + 1] = hoisted + [instrs[i]] + kept + [instrs[end]]
            changes += len(hoisted)
            i = end + 1  # Region length never changes.
        return changes

    @staticmethod
    def _available_before(
        program: Program, func: Function, block: BasicBlock, index: int
    ):
        adapter = IRCFGAdapter(func)
        cfg = build_cfg(adapter)
        reachable = cfg.reachable_nodes
        live = replace(
            cfg,
            nodes={n: cfg.nodes[n] for n in reachable},
            edges=[
                e for e in cfg.edges if e.source in reachable and e.target in reachable
            ],
        )
        dom = compute_dominators(live)
        locations = {}
        insertion = None
        for name, node in cfg.nodes.items():
            for offset, instr in enumerate(node.instructions):
                position = adapter.execution_plan.origins.get(id(instr))
                if position is None:
                    continue
                if (
                    position.block_name == block.name
                    and position.instruction_index == index
                    and position.stage == "for-init"
                    and instr.opcode == OpCode.LOAD_CONST
                ):
                    insertion = (name, offset)
                if instr.dest and position.stage != "for-step":
                    locations[instr.dest.name] = (name, offset)
        if insertion is None or insertion[0] not in reachable:
            return set()
        available = {v.name for v in [*program.global_values, *func.params]}
        node, offset = insertion
        for value, (def_node, def_offset) in locations.items():
            if (def_node == node and def_offset < offset) or (
                def_node != node and def_node in dom[node]
            ):
                available.add(value)
        return available

    @staticmethod
    def _find_matching_endfor(instrs: list[Instruction], start: int):
        depth = 0
        for i in range(start, len(instrs)):
            if instrs[i].opcode == OpCode.FOR:
                depth += 1
            elif instrs[i].opcode == OpCode.ENDFOR:
                depth -= 1
                if depth == 0:
                    return i
        return None
