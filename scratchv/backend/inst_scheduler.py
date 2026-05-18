"""Instruction Scheduler for RISC-V (List Scheduling).

Reorders instructions within a basic block to reduce pipeline stalls
caused by data dependencies, using list scheduling with critical-path
priority.

Usage::

    from scratchv.backend.inst_scheduler import InstructionScheduler
    sched = InstructionScheduler()
    dag = sched.build_dag(instructions)
    scheduled = sched.schedule(dag)
"""

from __future__ import annotations

import argparse
import re as _re
import sys
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Latency model
# ---------------------------------------------------------------------------

# Default RISC-V latency model (cycles before result is available)
_DEFAULT_LATENCY: dict[str, int] = {
    # Integer ALU
    "add": 1, "addi": 1, "sub": 1,
    "sll": 1, "srl": 1, "sra": 1,
    "xor": 1, "or": 1, "and": 1,
    "xori": 1, "ori": 1, "andi": 1,
    "slli": 1, "srli": 1, "srai": 1,
    "slt": 1, "sltu": 1, "slti": 1, "sltiu": 1,
    # M extension
    "mul": 3,
    "mulh": 3, "mulhsu": 3, "mulhu": 3,
    "div": 16, "divu": 16,
    "rem": 16, "remu": 16,
    # Memory loads (cache hit assumed)
    "lw": 2, "lh": 2, "lb": 2, "lbu": 2, "lhu": 2,
    # Memory stores (non-blocking for subsequent loads)
    "sw": 0, "sh": 0, "sb": 0,
    # Branches (resolved in decode in simple cores, 1 cycle otherwise)
    "beq": 1, "bne": 1, "blt": 1, "bge": 1,
    "bltu": 1, "bgeu": 1,
    # Jumps
    "j": 0, "jal": 0, "jalr": 0, "ret": 0,
    # Pseudo
    "li": 1, "mv": 1, "nop": 0, "call": 3,
    "lui": 1, "auipc": 1,
    "max": 1,  # ScratchV pseudo
}


# ---------------------------------------------------------------------------
# Instruction representation
# ---------------------------------------------------------------------------

@dataclass
class SchedInst:
    """An instruction node for the scheduler.

    Attributes
    ----------
    id:
        Unique index in the input list.
    opcode:
        Instruction mnemonic.
    operands:
        List of operand strings.
    defines:
        Set of register names that this instruction writes.
    uses:
        Set of register names that this instruction reads.
    raw_line:
        Original assembly text line.
    """
    id: int
    opcode: str
    operands: list[str] = field(default_factory=list)
    defines: set[str] = field(default_factory=set)
    uses: set[str] = field(default_factory=set)
    raw_line: str = ""

    def __repr__(self) -> str:
        return (f"SchedInst({self.id}, {self.opcode}, "
                f"def={self.defines}, use={self.uses})")


# ---------------------------------------------------------------------------
# DAG node
# ---------------------------------------------------------------------------

@dataclass
class DAGNode:
    """A node in the instruction dependency DAG.

    Attributes
    ----------
    inst:
        The instruction this node represents.
    predecessors:
        List of edges (pred_node, latency) that must complete before this node.
    successors:
        List of edges (succ_node, latency) that depend on this node.
    scheduled:
        Whether this node has been scheduled.
    ready_time:
        Earliest cycle this node can be issued.
    priority:
        Critical path length (used for list scheduling priority).
    """
    inst: SchedInst
    predecessors: list[tuple["DAGNode", int]] = field(default_factory=list)
    successors: list[tuple["DAGNode", int]] = field(default_factory=list)
    scheduled: bool = False
    ready_time: int = 0
    priority: int = 0

    def __repr__(self) -> str:
        return (f"DAGNode(id={self.inst.id}, {self.inst.opcode}, "
                f"prio={self.priority}, pred={len(self.predecessors)})")


# ---------------------------------------------------------------------------
# Instruction scheduler
# ---------------------------------------------------------------------------

class InstructionScheduler:
    """List scheduler for RISC-V basic blocks.

    Parameters
    ----------
    latency_model:
        Dict mapping opcode strings to execution latency (cycles).
        If None, uses the default RISC-V latency model.

    Usage::

        sched = InstructionScheduler()
        dag = sched.build_dag(instructions)
        scheduled_insts = sched.schedule(dag)
        # scheduled_insts is a list of SchedInst in scheduled order
    """

    def __init__(self, latency_model: Optional[dict[str, int]] = None):
        self.latency_model: dict[str, int] = (
            latency_model if latency_model is not None
            else dict(_DEFAULT_LATENCY)
        )
        self._original_order: list[SchedInst] = []
        self._nodes: list[DAGNode] = []
        self._schedule_time: int = 0

    # ------------------------------------------------------------------
    # DAG construction
    # ------------------------------------------------------------------

    def build_dag(self, instructions: list[SchedInst]) -> list[DAGNode]:
        """Build a dependency DAG from a list of instructions.

        Parameters
        ----------
        instructions:
            List of SchedInst objects.

        Returns
        -------
        List of DAGNode objects representing the dependency DAG.
        """
        self._original_order = list(instructions)
        self._nodes = []

        # Create nodes
        id_to_node: dict[int, DAGNode] = {}
        for inst in instructions:
            node = DAGNode(inst=inst)
            self._nodes.append(node)
            id_to_node[inst.id] = node

        # Build dependency edges (RAW hazards)
        # For each register, track the last instruction that defined it
        last_def: dict[str, DAGNode] = {}

        for inst in instructions:
            node = id_to_node[inst.id]

            # RAW: each use depends on the last definition
            for use_reg in inst.uses:
                if use_reg in last_def:
                    pred = last_def[use_reg]
                    latency = self._get_latency(pred.inst.opcode)
                    node.predecessors.append((pred, latency))
                    pred.successors.append((node, latency))

            # WAW: later definitions of same register depend on earlier
            for def_reg in inst.defines:
                if def_reg in last_def and last_def[def_reg] is not node:
                    pred = last_def[def_reg]
                    latency = self._get_latency(pred.inst.opcode)
                    node.predecessors.append((pred, latency))
                    pred.successors.append((node, latency))

                last_def[def_reg] = node

        # Compute priorities (critical path length from each node)
        self._compute_priorities()

        return self._nodes

    def _get_latency(self, opcode: str) -> int:
        """Get the latency for an opcode."""
        return self.latency_model.get(opcode, 1)

    def _compute_priorities(self) -> None:
        """Compute the critical path length (priority) for each node.

        Priority = longest path from this node to a leaf (no successors),
        where edge weights are latencies.
        """
        # Topological sort for reverse traversal
        visited: set[int] = set()
        order: list[DAGNode] = []

        def _dfs(n: DAGNode) -> None:
            if n.inst.id in visited:
                return
            visited.add(n.inst.id)
            for succ, _ in n.successors:
                _dfs(succ)
            order.append(n)

        for node in self._nodes:
            _dfs(node)

        # Compute priorities in reverse topological order
        for node in reversed(order):
            max_succ_prio = 0
            for succ, lat in node.successors:
                max_succ_prio = max(max_succ_prio, succ.priority + lat)
            node.priority = (
                max_succ_prio + self._get_latency(node.inst.opcode)
            )

    # ------------------------------------------------------------------
    # List scheduling
    # ------------------------------------------------------------------

    def schedule(self, dag: list[DAGNode]) -> list[SchedInst]:
        """Perform list scheduling on the DAG.

        Parameters
        ----------
        dag:
            List of DAGNode objects (output of ``build_dag``).

        Returns
        -------
        List of SchedInst in scheduled order.
        """
        self._nodes = dag
        self._schedule_time = 0

        # Reset scheduling state
        for node in self._nodes:
            node.scheduled = False
            node.ready_time = 0

        # Ready queue: nodes with no unscheduled predecessors
        # Priority queue ordered by: higher priority first, then original order
        ready: list[DAGNode] = []
        result: list[SchedInst] = []

        # Find initial ready nodes
        for node in self._nodes:
            if not node.predecessors or all(
                not p.scheduled for p, _ in node.predecessors
            ):
                pass  # we'll process below

        # Main scheduling loop
        remaining = {id(node): node for node in self._nodes}
        clock = 0

        while remaining:
            # Find nodes whose predecessors are all scheduled
            ready = []
            for node in remaining.values():
                if all(p.scheduled for p, _ in node.predecessors):
                    ready.append(node)

            if not ready:
                # Deadlock: should not happen for acyclic DAG
                break

            # Sort ready nodes by priority (descending), then original id
            ready.sort(key=lambda n: (-n.priority, n.inst.id))

            # Pick the best node
            node = ready[0]
            node.scheduled = True
            node.ready_time = clock
            result.append(node.inst)
            del remaining[id(node)]

            # Advance clock by the instruction's latency
            clock += self._get_latency(node.inst.opcode)

        self._schedule_time = clock
        return result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def estimate_cycles(self, instructions: list[SchedInst]) -> int:
        """Estimate total execution cycles for a sequence (naive in-order)."""
        total = 0
        for inst in instructions:
            total += self._get_latency(inst.opcode)
        return total

    def report(self, original: list[SchedInst],
               scheduled: list[SchedInst]) -> str:
        """Return a comparison report between original and scheduled order."""
        orig_cycles = self.estimate_cycles(original)
        sched_cycles = self.estimate_cycles(scheduled)
        improvement = orig_cycles - sched_cycles
        pct = (improvement / orig_cycles * 100) if orig_cycles > 0 else 0.0

        lines = []
        lines.append("Instruction Scheduling Report")
        lines.append(f"  Original instructions: {len(original)}")
        lines.append(f"  Estimated cycles (original): {orig_cycles}")
        lines.append(f"  Estimated cycles (scheduled): {sched_cycles}")
        lines.append(
            f"  Improvement: {improvement} cycles ({pct:.1f}%)"
        )
        lines.append("  Scheduled order:")
        for i, inst in enumerate(scheduled):
            ops = ", ".join(inst.operands) if inst.operands else ""
            lines.append(f"    {i}: {inst.opcode} {ops}".rstrip())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Assembly parsing helpers
# ---------------------------------------------------------------------------

_LINE_RE = _re.compile(
    r'^\s*'
    r'(?:[A-Za-z_.][A-Za-z0-9_.]*:\s*)?'
    r'\.?(?P<opcode>[a-zA-Z][a-zA-Z0-9.]*)?\s*'
    r'(?P<operands>[^#]*)'
)


def parse_instructions(asm_text: str) -> list[SchedInst]:
    """Parse RISC-V assembly text into a list of SchedInst for scheduling.

    Parameters
    ----------
    asm_text:
        Raw RISC-V assembly text.

    Returns
    -------
    List of SchedInst objects.
    """
    result: list[SchedInst] = []
    lines = asm_text.strip().split("\n")
    idx = 0
    for line in lines:
        stripped = line.strip()
        code = stripped.split("#")[0].strip()
        if not code:
            continue

        m = _LINE_RE.match(stripped)
        if m is None:
            continue

        opcode = m.group("opcode")
        if opcode is None:
            continue
        opcode = opcode.lower().lstrip(".")

        # Skip labels
        if opcode.endswith(":"):
            continue
        # Skip assembler directives
        if opcode.startswith("."):
            continue

        operands_str = (m.group("operands") or "").strip()
        operands = [
            o.strip() for o in operands_str.split(",") if o.strip()
        ]

        # Classify operands as defines/uses
        defines: set[str] = set()
        uses: set[str] = set()
        pure_ops: list[str] = []

        for i, op in enumerate(operands):
            pure_ops.append(op)
            # First operand is usually the destination for ALU-type
            if i == 0 and opcode not in (
                "sw", "sh", "sb", "beq", "bne",
                "blt", "bge", "bltu", "bgeu",
                "j", "jal", "ret",
            ):
                defines.add(op)
            else:
                # Extract register from memory operands like "16(sp)"
                m2 = _re.match(r'\d+\((\w+)\)', op)
                if m2:
                    uses.add(m2.group(1))
                elif (op.startswith("x") or op.startswith("a") or
                      op.startswith("t") or op.startswith("s") or
                      op.startswith("f")):
                    uses.add(op)

        # For stores, first operand is a use (value to store)
        if opcode in ("sw", "sh", "sb"):
            if operands and operands[0] in defines:
                defines.remove(operands[0])
                uses.add(operands[0])

        # Handle labels and import from existing backend
        if opcode.startswith("."):
            defines = set()
            uses = set()

        result.append(SchedInst(
            id=idx,
            opcode=opcode,
            operands=pure_ops,
            defines=defines,
            uses=uses,
            raw_line=line,
        ))
        idx += 1

    return result


def machine_instrs_from_scheduled(
        scheduled: list[SchedInst],
) -> list:  # list of MachineInstr
    """Convert scheduled SchedInst list back to MachineInstr list.

    This enables the instruction scheduler's output to be consumed by
    ``RegisterAllocator`` and ``AsmEmitter``.

    Parameters
    ----------
    scheduled:
        List of SchedInst objects in scheduled order.

    Returns
    -------
    List of MachineInstr objects.
    """
    from scratchv.backend.machine_types import MachineInstr, MachineOp, MachineOperand

    result = []
    for inst in scheduled:
        if inst.opcode == ".label":
            result.append(MachineInstr(
                MachineOp.LABEL, comment="",
            ))
            continue

        try:
            mop = MachineOp(inst.opcode)
        except ValueError:
            mop = MachineOp.MV

        def _to_mop(s: str) -> MachineOperand:
            if s.startswith("x") or s.startswith("a") or s.startswith("t") or \
               s.startswith("s") or s.startswith("f") or \
               s in ("zero", "ra", "sp", "gp", "tp", "fp"):
                return MachineOperand.reg(s)
            try:
                return MachineOperand.immediate(int(s))
            except ValueError:
                return MachineOperand.vreg(s)

        ops = [_to_mop(o) for o in inst.operands]
        dst = ops[0] if len(ops) >= 1 else None
        src1 = ops[1] if len(ops) >= 2 else None
        src2 = ops[2] if len(ops) >= 3 else None

        result.append(MachineInstr(mop, dst, src1, src2, inst.raw_line))

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RISC-V Instruction Scheduler (List Scheduling)",
    )
    parser.add_argument(
        "input", type=str,
        help="Input assembly file (.s)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file (default: stdout)",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Print scheduling report to stderr",
    )

    args = parser.parse_args()

    with open(args.input, "r") as f:
        asm_text = f.read()

    instructions = parse_instructions(asm_text)
    sched = InstructionScheduler()
    dag = sched.build_dag(instructions)
    scheduled = sched.schedule(dag)

    output_lines = [
        f"  {inst.opcode} " + ", ".join(inst.operands)
        for inst in scheduled
    ]
    result = "\n".join(output_lines)

    if args.report:
        print(sched.report(instructions, scheduled), file=sys.stderr)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
    else:
        print(result)


if __name__ == "__main__":
    main()
