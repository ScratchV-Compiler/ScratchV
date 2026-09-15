"""Safe local RISC-V scheduling over immutable instruction objects.

LLVM references: ScheduleDAGInstrs::addPhysRegDeps, SUnit::ComputeHeight,
MachineScheduler::getSchedRegions, PostRASchedulerList::ListScheduleTopDown.
"""

from __future__ import annotations

import argparse
import heapq
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ._asm_parser import ParsedAsmLine, parse_line
from .schedule_model import ScheduleModel, estimate_order
from .schedule_semantics import DIVIDE, SchedInst, ScheduleError, integer, register_name
from .schedule_verify import verify_schedule


@dataclass(eq=False)
class DAGNode:
    inst: SchedInst
    predecessors: list[tuple[DAGNode, int]] = field(default_factory=list)
    successors: list[tuple[DAGNode, int]] = field(default_factory=list)
    scheduled: bool = False
    ready_time: int = 0
    priority: int = 0
    edge_kinds: dict[int, frozenset[str]] = field(default_factory=dict)


class InstructionScheduler:
    """Deterministic single-region scheduling using physical register effects."""

    def __init__(
        self,
        latency_model: dict[str, int] | None = None,
        model: ScheduleModel | None = None,
    ):
        if model is not None and latency_model:
            raise ValueError("Choose a model or latency overrides, not both")
        self.model = model or ScheduleModel(overrides=latency_model or {})
        self.latency_model = dict(self.model.overrides)
        self._nodes: list[DAGNode] = []

    def build_dag(self, instructions: Sequence[SchedInst]) -> list[DAGNode]:
        """Record register, memory, FP effect and fixed-terminator order."""
        if len({i.id for i in instructions}) != len(instructions):
            raise ScheduleError("Duplicate instruction IDs")
        if len({i.region for i in instructions}) > 1:
            raise ScheduleError("Multiple regions: use schedule_assembly()")
        self._nodes = nodes = [DAGNode(i) for i in instructions]
        edges: dict[tuple[int, int], tuple[int, set[str]]] = {}

        def edge(a: int, b: int, distance: int, kind: str) -> None:
            if a == b:
                return
            if a > b:
                raise ScheduleError("Backward dependency in original order")
            old_distance, kinds = edges.get((a, b), (0, set()))
            kinds.add(kind)
            edges[a, b] = max(distance, old_distance), kinds

        writes: dict[str, int] = {}
        reads: dict[str, set[int]] = {}
        memory = fp = terminal = None
        for index, inst in enumerate(instructions):
            self.model.timing(inst)
            if terminal is not None and not inst.terminator:
                raise ScheduleError("Body instruction after a terminator")
            for reg in inst.uses:
                if reg in writes:
                    producer = writes[reg]
                    edge(
                        producer,
                        index,
                        self.model.timing(instructions[producer]).latency,
                        "RAW",
                    )
                reads.setdefault(reg, set()).add(index)
            for reg in inst.defines:
                if reg in writes:
                    producer = writes[reg]
                    distance = max(
                        1, self.model.timing(instructions[producer]).latency
                        - self.model.timing(inst).latency,
                    )
                    edge(producer, index, distance, "WAW")
                for reader in reads.get(reg, ()):
                    edge(reader, index, 1, "WAR")
                reads[reg] = set()
                writes[reg] = index
            if inst.effects.memory != "none":
                if memory is not None:
                    edge(memory, index, 1, "memory")
                memory = index
            if inst.effects.fp_flags:
                if fp is not None:
                    edge(fp, index, 1, "fp-effects")
                fp = index
            if inst.terminator:
                if terminal is None:
                    for body in range(index):
                        edge(body, index, 1, "control")
                else:
                    edge(terminal, index, 1, "control")
                terminal = index

        # Division timing/dispatch behaviour differs substantially by target.
        # Keep every div/rem on the same side of every other instruction until
        # a target-specific policy is validated. Between anchors, normal list
        # scheduling still applies. Only link each intervening interval once.
        anchor = None
        for index, inst in enumerate(instructions):
            if anchor is not None:
                edge(anchor, index, 1, "division-order")
            if inst.opcode in DIVIDE:
                for previous in range(0 if anchor is None else anchor + 1, index):
                    edge(previous, index, 1, "division-order")
                anchor = index
        for (a, b), (distance, kinds) in sorted(edges.items()):
            nodes[a].successors.append((nodes[b], distance))
            nodes[b].predecessors.append((nodes[a], distance))
            nodes[b].edge_kinds[nodes[a].inst.id] = frozenset(kinds)
        self._compute_priorities()
        return nodes

    def _compute_priorities(self) -> None:
        # Edges point forward in original order; visit successors first.
        for node in reversed(self._nodes):
            node.priority = max(
                self.model.timing(node.inst).latency,
                max((d + succ.priority for succ, d in node.successors), default=0),
            )

    def schedule(self, dag: list[DAGNode]) -> list[SchedInst]:
        """Use a time-ordered pending queue and priority queues per resource."""
        if len({n.inst.id for n in dag}) != len(dag):
            raise ScheduleError("Duplicate graph nodes")
        members = set(dag)
        if any(p not in members for n in dag for p, _ in n.predecessors):
            raise ScheduleError("Dependency outside scheduling region")
        remaining = {n: len(n.predecessors) for n in dag}
        order = {n: index for index, n in enumerate(dag)}
        timings = {n: self.model.timing(n.inst) for n in dag}
        pending = []
        available: dict[str, list] = {}
        resources: dict[str, int] = {}
        for node in dag:
            node.scheduled = False
            node.ready_time = 0
            if not remaining[node]:
                heapq.heappush(pending, (0, order[node], node))
        clock = 0
        result = []
        while len(result) < len(dag):
            while pending and pending[0][0] <= clock:
                _, index, node = heapq.heappop(pending)
                queue = available.setdefault(timings[node].resource, [])
                heapq.heappush(queue, (-node.priority, node.ready_time, index, node))
            choices = [
                queue[0]
                for resource, queue in available.items()
                if queue and resources.get(resource, 0) <= clock
            ]
            if not choices:
                future = [pending[0][0]] if pending else []
                future.extend(
                    resources[resource]
                    for resource, queue in available.items()
                    if queue and resources.get(resource, 0) > clock
                )
                if not future:
                    raise ScheduleError("Dependency cycle or inconsistent graph")
                clock = min(future)
                continue
            _, _, _, node = min(choices)
            timing = timings[node]
            heapq.heappop(available[timing.resource])
            node.scheduled = True
            result.append(node.inst)
            resources[timing.resource] = clock + timing.occupancy
            for succ, distance in node.successors:
                if succ not in remaining:
                    raise ScheduleError("Successor outside scheduling region")
                succ.ready_time = max(succ.ready_time, clock + distance)
                remaining[succ] -= 1
                if remaining[succ] == 0:
                    heapq.heappush(pending, (succ.ready_time, order[succ], succ))
            clock += timings[node].issue_occupancy
        return result

    def estimate_cycles(self, instructions: Sequence[SchedInst]) -> int:
        return estimate_order(instructions, self.model).cycles

    def report(
        self, original: Sequence[SchedInst], scheduled: Sequence[SchedInst]
    ) -> str:
        before = self.estimate_cycles(original)
        after = self.estimate_cycles(scheduled)
        return (
            f"Instruction Scheduling Report ({self.model.name}; local model estimate)\n"
            f"  Estimated cycles (original): {before}\n"
            f"  Estimated cycles (scheduled): {after}\n"
            f"  Improvement: {before - after} cycles"
        )


@dataclass(frozen=True)
class AssemblyLine:
    parsed: ParsedAsmLine
    ending: str
    instruction: SchedInst | None
    executable: bool
    diagnostic: str = ""


def _code_outside_strings(raw: str) -> str:
    """Mask quoted strings and remove comments before checking layout syntax."""
    code = []
    quoted = escaped = False
    for char in raw:
        if quoted:
            code.append(" ")
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == "#":
            break
        elif char == '"':
            quoted = True
            code.append(" ")
        else:
            code.append(char)
    return "".join(code)


def _read_lines(asm_text: str) -> list[AssemblyLine]:
    """Keep source layout separately from executable, typed instructions."""
    result = []
    region = 0
    section = (".text", True)
    previous_section = None
    section_stack = []
    section_flags = {".text": True}
    previous_terminal = False
    for index, raw in enumerate(asm_text.splitlines(keepends=True)):
        text = raw.rstrip("\r\n")
        ending = raw[len(text) :]
        parsed = parse_line(text, index)
        # Standalone compiler listings also have display labels containing '/'.
        # Preserve these as boundaries rather than counting them as opcodes.
        display = _code_outside_strings(text).strip()
        if parsed.label is None and re.fullmatch(r"[^\s:]+:", display):
            parsed = ParsedAsmLine(raw=text, label=display[:-1], lineno=index)
        message = ""
        if parsed.is_directive:
            op = parsed.opcode
            if op in {"text", "data", "bss", "rodata", "section", "pushsection"}:
                if op == "pushsection":
                    section_stack.append(section)
                name = (
                    "." + op if op in {"text", "data", "bss", "rodata"}
                    else parsed.operands[0].strip('"') if parsed.operands else ""
                )
                executable = section_flags.get(
                    name, name == ".text" or name.startswith(".text.")
                )
                if op in {"section", "pushsection"} and len(parsed.operands) > 1:
                    executable = "x" in parsed.operands[1].strip('"')
                # Explicitly named data stays pinned even with unusual flags.
                if any(name == prefix or name.startswith(prefix + ".")
                       for prefix in (".data", ".bss", ".rodata", ".sdata", ".sbss")):
                    executable = False
                section_flags[name] = executable
                previous_section, section = section, (name, executable)
            elif op == "popsection":
                if section_stack:
                    previous_section, section = section, section_stack.pop()
                else:
                    section = ("", False)
                    message = "unmatched .popsection; awaiting an explicit section"
            elif op == "previous":
                if previous_section is not None:
                    section, previous_section = previous_section, section
                else:
                    section = ("", False)
                    message = ".previous has no prior section; awaiting an explicit section"
        executable = section[1]
        inst = None
        if parsed.label or parsed.is_directive or not parsed.opcode:
            region += 1
        if parsed.opcode and not parsed.is_directive and executable:
            if previous_terminal:
                probe = SchedInst(index, parsed.opcode, parsed.operands)
                if not probe.terminator:
                    region += 1
            inst = SchedInst(
                index, parsed.opcode, parsed.operands, raw_line=text, region=region
            )
            previous_terminal = inst.terminator
            if parsed.label or inst.effects.barrier_reason:
                region += 1
        else:
            previous_terminal = False
        result.append(AssemblyLine(parsed, ending, inst, executable, message))
    return result


def parse_instructions(asm_text: str) -> list[SchedInst]:
    """Read instructions with region IDs. Use schedule_assembly to rewrite files.

    Non-instruction lines are excluded here, but region IDs preserve boundaries:
    build_dag rejects instructions from multiple regions.
    """
    return [
        line.instruction
        for line in _read_lines(asm_text)
        if line.instruction is not None
    ]


@dataclass(frozen=True)
class ScheduleConfig:
    strict: bool = False
    max_region_size: int = 1024
    model: ScheduleModel = field(default_factory=ScheduleModel)

    def __post_init__(self) -> None:
        if self.max_region_size < 1:
            raise ValueError("max_region_size must be positive")


@dataclass
class ScheduleResult:
    asm_text: str
    stats: dict
    diagnostics: list[dict]
    changed: bool

    def report(self) -> str:
        s = self.stats
        return (
            f"Instruction Scheduling Report ({s['model']}; local static model estimate)\n"
            f"  Estimated cycles: {s['original_cycles']} -> {s['final_cycles']}\n"
            f"  Saved cycles: {s['saved_cycles']}; moved instructions: {s['moved_instructions']}\n"
            f"  Modeled instructions: {s['modeled_instructions']}/{s['input_instructions']}\n"
            f"  Coverage: {s['coverage_ratio']:.1%}; unmodeled: {s['unmodeled_instructions']}\n"
            f"  Region size: mean {s['mean_region_size']:.2f}, max {s['max_region_instructions']}\n"
            f"  Applied regions: {s['applied_regions']}; skipped: {s['skipped_regions']}\n"
            "  Regions assume ready inputs; this is not whole-program runtime."
        )


def _unsafe_layout(lines: Sequence[AssemblyLine]) -> tuple[int, str] | None:
    # These constructs can change the interpretation of later instructions.
    unsafe = {
        "macro",
        "endm",
        "rept",
        "endr",
        "irp",
        "irpc",
        "include",
        "incbin",
        "org",
        "set",
        "equ",
        "equiv",
        "if",
        "ifdef",
        "ifndef",
        "else",
        "endif",
    }
    for line in lines:
        p = line.parsed
        code = _code_outside_strings(p.raw)
        if (
            ";" in code
            or "\\" in code
            or (p.opcode != "size" and re.search(r"(?<![\w.])\.(?![\w.])", code))
        ):
            return p.lineno, "compound statement, continuation or current-address expression"
        if p.is_directive and (p.opcode in unsafe or (p.opcode or "").startswith("if")):
            return p.lineno, "assembler macro, conditional or layout directive"
        if (
            line.instruction
            and line.instruction.terminator
            and p.operands
            and (
                line.instruction.effects.barrier_reason
                and integer(p.operands[-1]) is not None
            )
        ):
            return p.lineno, "numeric control-flow target"
        if (
            p.opcode in {"j", "jal"}
            and p.operands
            and integer(p.operands[-1]) is not None
        ):
            return p.lineno, "numeric control-flow target"
    return None


def schedule_assembly(
    asm_text: str, config: ScheduleConfig | None = None
) -> ScheduleResult:
    """Verify each candidate before applying it; errors restore the whole region."""
    config = config or ScheduleConfig()
    sensitivity_model = config.model.sensitivity_model()
    started = time.perf_counter()
    lines = _read_lines(asm_text)
    output = [line.parsed.raw + line.ending for line in lines]
    diagnostics: list[dict] = []
    stats = {
        "model": config.model.name,
        "model_overrides": dict(config.model.overrides),
        "division_policy": "preserve-relative-order; conservative blocking issue",
        "sensitivity_model": sensitivity_model.name,
        "sensitivity_overrides": dict(sensitivity_model.overrides),
        "sensitivity_rejected_regions": 0,
        "estimate_scope": "sum of local static regions; ready inputs; no cache/branch prediction",
        "input_instructions": sum(line.instruction is not None for line in lines),
        "modeled_instructions": 0,
        "moved_instructions": 0,
        "applied_regions": 0,
        "skipped_regions": 0,
        "original_cycles": 0,
        "candidate_cycles": 0,
        "final_cycles": 0,
        "original_stalls": 0,
        "final_stalls": 0,
        "regions": [],
    }

    def diagnostic(line: int, reason: str, severity: str = "info") -> None:
        diagnostics.append({"line": line + 1, "reason": reason, "severity": severity})

    for line in lines:
        if line.diagnostic:
            diagnostic(line.parsed.lineno, line.diagnostic, "warning")
    modeled_ids: set[int] = set()
    unsafe = _unsafe_layout(lines)
    if unsafe:
        line, reason = unsafe
        diagnostic(line, reason + "; entire input preserved", "warning")
        stats["skipped_regions"] = 1
    else:
        regions: list[list[SchedInst]] = []
        current: list[SchedInst] = []
        for line in lines:
            inst = line.instruction
            fixed = (
                inst is None
                or line.parsed.label is not None
                or bool(inst.effects.barrier_reason)
            )
            if current and (fixed or inst.region != current[-1].region):
                regions.append(current)
                current = []
            if not fixed:
                current.append(inst)
            elif inst is not None:
                diagnostic(
                    inst.id,
                    inst.effects.barrier_reason or "instruction shares a label line",
                )
        if current:
            regions.append(current)
        for original in regions:
            row = {
                "start_line": original[0].id + 1,
                "end_line": original[-1].id + 1,
                "instructions": len(original),
                "status": "unchanged",
                "moved": 0,
                "original_cycles": None,
                "candidate_cycles": None,
                "final_cycles": None,
                "dependencies": {},
            }
            stats["regions"].append(row)
            if len(original) > config.max_region_size:
                row["status"] = "skipped"
                stats["skipped_regions"] += 1
                diagnostic(original[0].id, "region size exceeds configured limit")
                continue
            try:
                scheduler = InstructionScheduler(model=config.model)
                dag = scheduler.build_dag(original)
                candidate = scheduler.schedule(dag)
                verify_schedule(original, candidate, dag)
                before = estimate_order(original, config.model)
                after = estimate_order(candidate, config.model)
                sensitivity_before = estimate_order(original, sensitivity_model)
                sensitivity_after = estimate_order(candidate, sensitivity_model)
                sensitive = (after.cycles < before.cycles
                             and sensitivity_after.cycles >= sensitivity_before.cycles)
                applied = after.cycles < before.cycles and not sensitive
                stats["sensitivity_rejected_regions"] += int(sensitive)
                final = candidate if applied else original
                verify_schedule(original, final, dag)
                final_estimate = after if applied else before
                row.update(
                    status="applied" if applied else "model_sensitive" if sensitive else "no_improvement",
                    original_cycles=before.cycles,
                    candidate_cycles=after.cycles,
                    final_cycles=final_estimate.cycles,
                    sensitivity_original_cycles=sensitivity_before.cycles,
                    sensitivity_candidate_cycles=sensitivity_after.cycles,
                    original_issue_cycles=list(before.issue_cycles),
                    candidate_issue_cycles=list(after.issue_cycles),
                )
                for node in dag:
                    for kinds in node.edge_kinds.values():
                        for kind in sorted(kinds):
                            row["dependencies"][kind] = (
                                row["dependencies"].get(kind, 0) + 1
                            )
                row["moved"] = sum(a is not b for a, b in zip(original, final))
                # Keep newline characters at the destination slots (including EOF).
                for slot, inst in zip(original, final):
                    output[slot.id] = inst.raw_line + lines[slot.id].ending
                stats["modeled_instructions"] += len(original)
                modeled_ids.update(inst.id for inst in original)
                stats["moved_instructions"] += row["moved"]
                stats["applied_regions"] += int(applied)
                stats["original_cycles"] += before.cycles
                stats["candidate_cycles"] += after.cycles
                stats["final_cycles"] += final_estimate.cycles
                stats["original_stalls"] += before.stalls
                stats["final_stalls"] += final_estimate.stalls
            except ScheduleError as exc:
                if config.strict:
                    raise ScheduleError(f"Line {original[0].id + 1}: {exc}") from exc
                row["status"] = "restored"
                row["reason"] = str(exc)
                stats["skipped_regions"] += 1
                diagnostic(original[0].id, f"original order restored: {exc}", "warning")
    result = "".join(output)
    stats["saved_cycles"] = stats["original_cycles"] - stats["final_cycles"]
    stats["output_instructions"] = stats["input_instructions"]
    stats["unmodeled_instructions"] = stats["input_instructions"] - len(modeled_ids)
    stats["coverage_ratio"] = (
        len(modeled_ids) / stats["input_instructions"] if stats["input_instructions"] else 0.0
    )
    counts: dict[str, int] = {}
    for line in lines:
        if line.instruction is not None and line.instruction.id not in modeled_ids:
            op = line.instruction.opcode
            counts[op] = counts.get(op, 0) + 1
    stats["unmodeled_by_opcode"] = dict(sorted(counts.items()))
    sizes = [row["instructions"] for row in stats["regions"]]
    stats["region_count"] = len(sizes)
    stats["mean_region_size"] = sum(sizes) / len(sizes) if sizes else 0.0
    stats["max_region_instructions"] = max(sizes, default=0)
    if stats["unmodeled_instructions"]:
        first = next(line for line in lines
                     if line.instruction is not None and line.instruction.id not in modeled_ids)
        diagnostic(first.parsed.lineno,
                   f"scheduling coverage {stats['coverage_ratio']:.1%}; "
                   f"{stats['unmodeled_instructions']}/{stats['input_instructions']} "
                   "instructions unmodeled; see unmodeled_by_opcode and diagnostics",
                   "warning")
    stats["elapsed_seconds"] = time.perf_counter() - started
    return ScheduleResult(result, stats, diagnostics, result != asm_text)


def machine_instrs_from_scheduled(scheduled: Sequence[SchedInst]) -> list:
    """Legacy conversion for simple operands; reject lossy/unknown conversions."""
    from .machine_types import MachineInstr, MachineOp, MachineOperand

    result = []
    for inst in scheduled:
        if (
            inst.effects.barrier_reason
            or inst.effects.memory != "none"
            or inst.terminator
        ):
            raise ValueError(
                "This instruction cannot be losslessly converted to MachineInstr"
            )
        try:
            opcode = MachineOp(inst.opcode)
        except ValueError as exc:
            raise ValueError(f"Unsupported MachineOp: {inst.opcode}") from exc
        operands = []
        for value in inst.operands:
            reg, imm = register_name(value), integer(value)
            if reg is not None:
                operands.append(MachineOperand.reg(reg))
            elif imm is not None:
                operands.append(MachineOperand.immediate(imm))
            else:
                raise ValueError(f"Unsupported machine operand: {value}")
        if len(operands) > 3:
            raise ValueError("Too many operands for MachineInstr")
        operands.extend([None] * (3 - len(operands)))
        result.append(MachineInstr(opcode, *operands))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safe local RISC-V instruction scheduling"
    )
    parser.add_argument("input")
    parser.add_argument("-o", "--output")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--report-json")
    args = parser.parse_args()
    with open(args.input, newline="") as source:
        asm_text = source.read()
    try:
        result = schedule_assembly(asm_text, ScheduleConfig(strict=args.strict))
    except ScheduleError as exc:
        parser.exit(1, f"Scheduling failed: {exc}\n")
    if args.report:
        print(result.report(), file=sys.stderr)
    for diagnostic in result.diagnostics:
        print(
            f"Schedule line {diagnostic['line']}: {diagnostic['reason']}",
            file=sys.stderr,
        )
    if args.report_json:
        Path(args.report_json).write_text(
            json.dumps(
                {"stats": result.stats, "diagnostics": result.diagnostics}, indent=2
            )
            + "\n"
        )
    if args.output:
        with open(args.output, "w", newline="") as destination:
            destination.write(result.asm_text)
    else:
        sys.stdout.write(result.asm_text)


if __name__ == "__main__":
    main()
