"""Safe local RISC-V scheduling over immutable instruction objects.

LLVM references: ScheduleDAGInstrs::addPhysRegDeps, SUnit::ComputeHeight,
MachineScheduler::getSchedRegions, PostRASchedulerList::ListScheduleTopDown.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ._asm_parser import ParsedAsmLine, parse_line
from .llvm_mca import LLVMError, metrics as llvm_metrics, model_metadata
from .schedule_semantics import DIVIDE, SchedInst, ScheduleError, integer, register_name
from .schedule_verify import verify_schedule


@dataclass(eq=False)
class DAGNode:
    inst: SchedInst
    predecessors: list[tuple[DAGNode, int]] = field(default_factory=list)
    successors: list[tuple[DAGNode, int]] = field(default_factory=list)
    scheduled: bool = False
    priority: int = 0
    edge_kinds: dict[int, frozenset[str]] = field(default_factory=dict)


class InstructionScheduler:
    """Deterministic single-region scheduling using physical register effects."""

    def __init__(self, llvm_mca: str | None = None):
        self.llvm_mca = llvm_mca
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
            if terminal is not None and not inst.terminator:
                raise ScheduleError("Body instruction after a terminator")
            for reg in inst.uses:
                if reg in writes:
                    producer = writes[reg]
                    edge(producer, index, 0, "RAW")
                reads.setdefault(reg, set()).add(index)
            for reg in inst.defines:
                if reg in writes:
                    producer = writes[reg]
                    edge(producer, index, 0, "WAW")
                for reader in reads.get(reg, ()):
                    edge(reader, index, 0, "WAR")
                reads[reg] = set()
                writes[reg] = index
            if inst.effects.memory != "none":
                if memory is not None:
                    edge(memory, index, 0, "memory")
                memory = index
            if inst.effects.fp_flags:
                if fp is not None:
                    edge(fp, index, 0, "fp-effects")
                fp = index
            if inst.terminator:
                if terminal is None:
                    for body in range(index):
                        edge(body, index, 0, "control")
                else:
                    edge(terminal, index, 0, "control")
                terminal = index

        # Division timing/dispatch behaviour differs substantially by target.
        # Keep every div/rem on the same side of every other instruction until
        # a target-specific policy is validated. Between anchors, normal list
        # scheduling still applies. Only link each intervening interval once.
        anchor = None
        for index, inst in enumerate(instructions):
            if anchor is not None:
                edge(anchor, index, 0, "division-order")
            if inst.opcode in DIVIDE:
                for previous in range(0 if anchor is None else anchor + 1, index):
                    edge(previous, index, 0, "division-order")
                anchor = index
        for (a, b), (distance, kinds) in sorted(edges.items()):
            nodes[a].successors.append((nodes[b], distance))
            nodes[b].predecessors.append((nodes[a], distance))
            nodes[b].edge_kinds[nodes[a].inst.id] = frozenset(kinds)
        self._compute_priorities()
        return nodes

    def _compute_priorities(self) -> None:
        # Dependency depth is an ordering heuristic, not a CPU cycle estimate.
        for node in reversed(self._nodes):
            node.priority = 1 + max((succ.priority for succ, _ in node.successors), default=0)

    def schedule(self, dag: list[DAGNode]) -> list[SchedInst]:
        """Propose a legal order, separating producers from consumers where possible.

        Ready-list scores use graph depth and instruction positions, never
        simulated cycles. LLVM MCA alone decides whether to accept the order.
        """
        if len({n.inst.id for n in dag}) != len(dag):
            raise ScheduleError("Duplicate graph nodes")
        members = set(dag)
        if any(p not in members for n in dag for p, _ in n.predecessors):
            raise ScheduleError("Dependency outside scheduling region")
        remaining = {n: len(n.predecessors) for n in dag}
        order = {n: index for index, n in enumerate(dag)}
        ready = []
        for node in dag:
            node.scheduled = False
            if not remaining[node]:
                ready.append(node)
        positions = {}
        result = []
        while ready:
            def rank(node):
                latest_producer = max(
                    (positions[p] for p, _ in node.predecessors
                     if "RAW" in node.edge_kinds.get(p.inst.id, ())), default=-1)
                return latest_producer, -node.priority, order[node]

            node = min(ready, key=rank)
            ready.remove(node)
            positions[node] = len(result)
            node.scheduled = True
            result.append(node.inst)
            for succ, distance in node.successors:
                if succ not in remaining:
                    raise ScheduleError("Successor outside scheduling region")
                remaining[succ] -= 1
                if remaining[succ] == 0:
                    ready.append(succ)
        if len(result) != len(dag):
            raise ScheduleError("Dependency cycle or inconsistent graph")
        return result

    def estimate_cycles(self, instructions: Sequence[SchedInst]) -> int:
        return llvm_metrics(instructions, self.llvm_mca)["cycles"] if instructions else 0

    def report(
        self, original: Sequence[SchedInst], scheduled: Sequence[SchedInst]
    ) -> str:
        before = self.estimate_cycles(original)
        after = self.estimate_cycles(scheduled)
        return (
            f"Instruction Scheduling Report (llvm-mca/sifive-e76; local static estimate)\n"
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
    llvm_mca: str | None = None

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
        a, b = s["metrics_before"], s["metrics_after"]
        return (
            f"Instruction Scheduling Report ({s['model']}; LLVM MCA static estimate)\n"
            f"  Estimated cycles: {s['original_cycles']} -> {s['final_cycles']}\n"
            f"  Saved cycles: {s['saved_cycles']}; moved instructions: {s['moved_instructions']}\n"
            f"  Modeled instructions: {s['modeled_instructions']}/{s['input_instructions']}\n"
            f"  Coverage: {s['coverage_ratio']:.1%}; unmodeled: {s['unmodeled_instructions']}\n"
            f"  Region size: mean {s['mean_region_size']:.2f}, max {s['max_region_instructions']}\n"
            f"  Applied regions: {s['applied_regions']}; skipped: {s['skipped_regions']}\n"
            f"  Peak issue parallelism: {a['peak_parallelism']} -> {b['peak_parallelism']}\n"
            "  Critical path: N/A (not collected for this in-order target)\n"
            f"  Issue bubbles (cycles, excluding drain): {a['bubbles']} -> {b['bubbles']}\n"
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
    started = time.perf_counter()
    lines = _read_lines(asm_text)
    output = [line.parsed.raw + line.ending for line in lines]
    diagnostics: list[dict] = []
    stats = {
        "model": "llvm-mca/sifive-e76",
        "model_metadata": None,
        "division_policy": "preserve-relative-order",
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
                scheduler = InstructionScheduler(llvm_mca=config.llvm_mca)
                dag = scheduler.build_dag(original)
                candidate = scheduler.schedule(dag)
                verify_schedule(original, candidate, dag)
                if stats["model_metadata"] is None:
                    stats["model_metadata"] = model_metadata(config.llvm_mca)
                before = llvm_metrics(original, config.llvm_mca)
                after = llvm_metrics(candidate, config.llvm_mca)
                applied = after["cycles"] < before["cycles"]
                final = candidate if applied else original
                verify_schedule(original, final, dag)
                final_estimate = after if applied else before
                row.update(
                    metrics_before=before, metrics_after=final_estimate,
                    final_issue_cycles=final_estimate["issue_cycles"],
                    status="applied" if applied else "no_improvement",
                    original_cycles=before["cycles"],
                    candidate_cycles=after["cycles"],
                    candidate_source=after["source"],
                    final_cycles=final_estimate["cycles"],
                    original_issue_cycles=before["issue_cycles"],
                    candidate_issue_cycles=after["issue_cycles"],
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
                stats["original_cycles"] += before["cycles"]
                stats["candidate_cycles"] += after["cycles"]
                stats["final_cycles"] += final_estimate["cycles"]
                stats["original_stalls"] += before["bubbles"]
                stats["final_stalls"] += final_estimate["bubbles"]
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
    for phase in ("before", "after"):
        rows = [r[f"metrics_{phase}"] for r in stats["regions"] if f"metrics_{phase}" in r]
        cycles = sum(r["cycles"] for r in rows)
        bubbles = sum(r["bubbles"] for r in rows)
        span = sum(r["issue_span"] for r in rows)
        stats[f"metrics_{phase}"] = {
            "cycles": cycles,
            "peak_parallelism": max((r["peak_parallelism"] for r in rows), default=0),
            "critical_path_max": None,
            "bubbles": bubbles,
            "bubble_ratio": bubbles / span if span else 0.0,
            "issue_span": span,
            "drain_cycles": cycles - span,
            "ipc": stats["modeled_instructions"] / cycles if cycles else 0.0,
        }
    stats["elapsed_seconds"] = time.perf_counter() - started
    return ScheduleResult(result, stats, diagnostics, result != asm_text)


def machine_instrs_from_scheduled(scheduled: Sequence[SchedInst]) -> list:
    """Convert supported operands and preserve explicit CFG control targets."""
    from .machine_types import MachineInstr, MachineOp, MachineOperand

    result = []
    for inst in scheduled:
        if inst.opcode == ".label" and inst.target and not inst.operands:
            result.append(MachineInstr(MachineOp.LABEL, target=inst.target))
            continue
        direct_control = inst.opcode in {"beq", "bne", "blt", "bge", "bnez", "j", "jal", "call"}
        direct_control = direct_control and inst.effects.target is not None
        if (
            (inst.effects.barrier_reason and not direct_control)
            or inst.effects.memory != "none"
            or (inst.terminator and not direct_control)
        ):
            raise ValueError(
                "This instruction cannot be losslessly converted to MachineInstr"
            )
        try:
            opcode = MachineOp(inst.opcode)
        except ValueError as exc:
            raise ValueError(f"Unsupported MachineOp: {inst.opcode}") from exc
        operands = []
        source_operands = inst.operands[:-1] if direct_control else inst.operands
        for value in source_operands:
            reg, imm = register_name(value), integer(value)
            if reg is not None:
                operands.append(MachineOperand.reg(value))
            elif imm is not None:
                operands.append(MachineOperand.immediate(imm))
            else:
                raise ValueError(f"Unsupported machine operand: {value}")
        if len(operands) > 3:
            raise ValueError("Too many operands for MachineInstr")
        if direct_control and opcode != MachineOp.JAL:
            operands.extend([None] * (2 - len(operands)))
            result.append(MachineInstr(opcode, None, *operands, target=inst.target))
        else:
            operands.extend([None] * (3 - len(operands)))
            result.append(MachineInstr(opcode, *operands, target=inst.target))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safe local RISC-V instruction scheduling"
    )
    parser.add_argument("input")
    parser.add_argument("-o", "--output")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--llvm-mca", help="LLVM MCA executable; default: LLVM_MCA or llvm-mca")
    parser.add_argument("--report-json")
    args = parser.parse_args()
    with open(args.input, newline="") as source:
        asm_text = source.read()
    try:
        result = schedule_assembly(asm_text, ScheduleConfig(strict=args.strict, llvm_mca=args.llvm_mca))
    except (ScheduleError, LLVMError) as exc:
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
