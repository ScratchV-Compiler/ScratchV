"""Function inlining pass (Topic 15).

Replaces eligible CALL sites with a renamed clone of the callee body.
Conservative v1: recursive callees, loops, oversized bodies and arity
mismatches are rejected and reported in :attr:`Inliner.warnings`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from scratchv.ir.types import (
    BasicBlock, Function, Instruction, OpCode, Program, Value,
)


@dataclass
class InlinerConfig:
    """Tunables for the inliner (all decisions are deterministic)."""

    max_instrs: int = 32
    single_site_only: bool = False
    growth_budget: int = 256
    reject_loops: bool = True
    allow_ret_drop: bool = False
    max_rounds: int = 4


def _body_size(func: Function) -> int:
    return sum(len(b.instructions) for b in func.blocks)


def _has_loop(func: Function) -> bool:
    return any(
        ins.opcode in (OpCode.FOR, OpCode.ENDFOR)
        for b in func.blocks for ins in b.instructions
    )


def _build_call_graph(
        program: Program,
        callee_index: dict[str, Function]) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {f.name: set() for f in program.functions}
    for func in program.functions:
        for block in func.blocks:
            for ins in block.instructions:
                if ins.opcode is OpCode.CALL and ins.target in callee_index:
                    graph[func.name].add(ins.target)
    return graph


def _recursive_functions(graph: dict[str, set[str]]) -> set[str]:
    recursive: set[str] = set()
    for start in graph:
        seen: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            for nxt in graph.get(node, ()):
                if nxt == start:
                    recursive.add(start)
                    stack.clear()
                    break
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return recursive


def _collect_call_sites(
        func: Function) -> list[tuple[BasicBlock, int, Instruction]]:
    sites: list[tuple[BasicBlock, int, Instruction]] = []
    for block in func.blocks:
        for idx, ins in enumerate(block.instructions):
            if ins.opcode is OpCode.CALL:
                sites.append((block, idx, ins))
    return sites


def _unique_block_name(func: Function, base: str) -> str:
    existing = {b.name for b in func.blocks}
    candidate = base
    i = 0
    while candidate in existing:
        candidate = f"{base}_{i}"
        i += 1
    return candidate


def _caller_names(func: Function) -> set[str]:
    names: set[str] = set()
    for param in func.params:
        names.add(param.name)
    for local in func.locals:
        names.add(local.name)
    for block in func.blocks:
        for ins in block.instructions:
            if ins.dest is not None:
                names.add(ins.dest.name)
            for v in ins.operands:
                names.add(v.name)
    return names


def _unique(used: set[str], base: str) -> str:
    candidate = base
    i = 0
    while candidate in used:
        candidate = f"{base}_{i}"
        i += 1
    used.add(candidate)
    return candidate


def _rewrite_target(target: Optional[str],
                    block_map: dict[str, BasicBlock]) -> Optional[str]:
    if target is None:
        return None
    if "," in target:
        parts = [p.strip() for p in target.split(",")]
        return ",".join(
            block_map[p].name if p in block_map else p for p in parts)
    return block_map[target].name if target in block_map else target


class Inliner:
    """Inline eligible CALL sites inside ``program`` (in place)."""

    def __init__(self, program: Program,
                 config: Optional[InlinerConfig] = None):
        self.program = program
        self.config = config or InlinerConfig()
        self._stats: dict[str, int] = {
            "inlined": 0, "rejected": 0, "rounds": 0}
        self.warnings: list[str] = []

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def run(self) -> int:
        """Run the inliner; returns the number of inlined call sites."""
        self._stats = {"inlined": 0, "rejected": 0, "rounds": 0}
        self.warnings = []

        callee_index = {f.name: f for f in self.program.functions}
        graph = _build_call_graph(self.program, callee_index)
        recursive = _recursive_functions(graph)
        cloned: dict[str, int] = defaultdict(int)
        processed: set[int] = set()

        for round_no in range(1, self.config.max_rounds + 1):
            changed = False
            for caller in list(self.program.functions):
                while True:
                    site = None
                    for candidate in _collect_call_sites(caller):
                        if id(candidate[2]) not in processed:
                            site = candidate
                            break
                    if site is None:
                        break

                    block, idx, call = site
                    callee = callee_index.get(call.target)
                    ok, reason = self._check_eligible(
                        site, callee_index, recursive, cloned)
                    if not ok:
                        self.warnings.append(
                            f"inliner: skip {call.target} at "
                            f"{caller.name}.{block.name}[{idx}]: {reason}")
                        self._stats["rejected"] += 1
                        processed.add(id(call))
                        continue

                    k = self._stats["inlined"]
                    self._inline_site(caller, block, idx, call, callee, k)
                    cloned[callee.name] += _body_size(callee)
                    self._stats["inlined"] += 1
                    changed = True

            self._stats["rounds"] = round_no
            if not changed:
                break

        return self._stats["inlined"]

    def _count_sites(self, callee: Function) -> int:
        return sum(
            1
            for func in self.program.functions
            for block in func.blocks
            for ins in block.instructions
            if ins.opcode is OpCode.CALL and ins.target == callee.name
        )

    def _check_eligible(
            self,
            site: tuple[BasicBlock, int, Instruction],
            callee_index: dict[str, Function],
            recursive: set[str],
            cloned: dict[str, int],
    ) -> tuple[bool, str]:
        _block, _idx, call = site
        callee = callee_index.get(call.target)
        if callee is None:
            return False, "callee_not_found"
        if not callee.blocks:
            return False, "empty_callee"

        argc = call.attrs.get("argc")
        if not isinstance(argc, int) or argc != len(call.operands):
            return False, "malformed_call"
        if call.attrs.get("is_tail", False):
            return False, "tail_unsupported"
        if len(call.operands) != len(callee.params):
            return False, "argc_mismatch"
        if callee.name in recursive:
            return False, "recursive_callee"
        if self.config.reject_loops and _has_loop(callee):
            return False, "loop_body_unsupported"

        valued = [
            ins
            for b in callee.blocks
            for ins in b.instructions
            if ins.opcode is OpCode.RETURN and ins.operands
        ]
        if len(valued) > 1:
            return False, "multiple_valued_returns"
        if valued and call.dest is None and not self.config.allow_ret_drop:
            return False, "ret_arity_mismatch"
        if not valued and call.dest is not None:
            return False, "ret_arity_mismatch"

        size = _body_size(callee)
        if size > self.config.max_instrs:
            return False, (
                f"body_too_large ({size} > {self.config.max_instrs})")
        if cloned[callee.name] + size > self.config.growth_budget:
            return False, "growth_budget_exceeded"
        if self.config.single_site_only and self._count_sites(callee) > 1:
            return False, "multiple_call_sites"
        return True, ""

    def _inline_site(self, caller: Function, block: BasicBlock, idx: int,
                     call: Instruction, callee: Function, k: int) -> None:
        cont_name = _unique_block_name(caller, f"{caller.name}_inl{k}_cont")

        block_map: dict[str, BasicBlock] = {}
        for callee_block in callee.blocks:
            block_map[callee_block.name] = caller.new_block(
                f"{callee.name}_{callee_block.name}_inl{k}")

        used = _caller_names(caller)
        value_map: dict[int, Value] = {}
        callee_dests = {
            id(ins.dest)
            for b in callee.blocks
            for ins in b.instructions
            if ins.dest is not None
        }

        def map_value(v: Value) -> Value:
            if v.is_constant and id(v) not in callee_dests:
                return v
            if id(v) not in value_map:
                value_map[id(v)] = Value(
                    name=_unique(used, f"{v.name}_inl{k}"), dtype=v.dtype)
            return value_map[id(v)]

        for param, arg in zip(callee.params, call.operands):
            value_map[id(param)] = arg
            for callee_block in callee.blocks:
                for ins in callee_block.instructions:
                    candidates = ([ins.dest] if ins.dest is not None else [])
                    candidates.extend(ins.operands)
                    for v in candidates:
                        if v.name == param.name and id(v) not in value_map:
                            value_map[id(v)] = arg

        ret_value: Optional[Value] = None
        for callee_block in callee.blocks:
            new_block = block_map[callee_block.name]
            for ins in callee_block.instructions:
                if ins.opcode is OpCode.RETURN:
                    if ins.operands:
                        ret_value = map_value(ins.operands[0])
                    new_block.add(Instruction(OpCode.BR, target=cont_name))
                    continue
                new_block.add(Instruction(
                    opcode=ins.opcode,
                    dest=map_value(ins.dest) if ins.dest is not None else None,
                    operands=[map_value(v) for v in ins.operands],
                    attrs=dict(ins.attrs),
                    target=_rewrite_target(ins.target, block_map),
                ))

        tail = block.instructions[idx + 1:]
        block.instructions = block.instructions[:idx] + [
            Instruction(
                OpCode.BR,
                target=block_map[callee.blocks[0].name].name,
            )
        ]
        cont = caller.new_block(cont_name)
        if tail:
            cont.instructions.extend(tail)
        else:
            self.warnings.append(
                f"inliner: call at end of block {block.name} (invalid IR)")

        if call.dest is not None and ret_value is not None:
            for caller_block in caller.blocks:
                for ins in caller_block.instructions:
                    ins.operands = [
                        ret_value if op is call.dest else op
                        for op in ins.operands
                    ]
