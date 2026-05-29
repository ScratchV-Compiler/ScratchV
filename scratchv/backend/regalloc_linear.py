"""Linear Scan Register Allocator for RISC-V.

Implements a basic-block-level linear scan register allocation algorithm
with proper live interval computation and spill code generation.

Usage::

    from scratchv.backend.regalloc_linear import LinearScanAllocator
    allocator = LinearScanAllocator()
    intervals = allocator.compute_live_intervals(block_instructions)
    allocator.allocate(intervals)
    result = allocator.get_allocated_code()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# RISC-V register definitions
# ---------------------------------------------------------------------------

# Allocatable integer registers (excludes x0/zero, sp, gp, tp, ra)
_INT_REGS = [
    # Argument/temp registers (caller-saved)
    "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",   # x10-x17
    "t0", "t1", "t2", "t3", "t4", "t5", "t6",          # x5-x7, x28-x31
    # Saved registers (callee-saved)
    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",    # x8-x9, x18-x23
    "s8", "s9", "s10", "s11",                           # x24-x27
]

_FP_REGS = [
    "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7",
    "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15",
    "f16", "f17", "f18", "f19",
    "f20", "f21", "f22", "f23",
    "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31",
]

_DEFAULT_PHYS_REGS = _INT_REGS

# Standard register map
_REG_NUMS: dict[str, int] = {
    "x0": 0, "zero": 0,
    "ra": 1, "x1": 1,
    "sp": 2, "x2": 2,
    "gp": 3, "x3": 3,
    "tp": 4, "x4": 4,
    "t0": 5, "x5": 5,
    "t1": 6, "x6": 6,
    "t2": 7, "x7": 7,
    "s0": 8, "fp": 8, "x8": 8,
    "s1": 9, "x9": 9,
    "a0": 10, "x10": 10,
    "a1": 11, "x11": 11,
    "a2": 12, "x12": 12,
    "a3": 13, "x13": 13,
    "a4": 14, "x14": 14,
    "a5": 15, "x15": 15,
    "a6": 16, "x16": 16,
    "a7": 17, "x17": 17,
    "s2": 18, "x18": 18,
    "s3": 19, "x19": 19,
    "s4": 20, "x20": 20,
    "s5": 21, "x21": 21,
    "s6": 22, "x22": 22,
    "s7": 23, "x23": 23,
    "s8": 24, "x24": 24,
    "s9": 25, "x25": 25,
    "s10": 26, "x26": 26,
    "s11": 27, "x27": 27,
    "t3": 28, "x28": 28,
    "t4": 29, "x29": 29,
    "t5": 30, "x30": 30,
    "t6": 31, "x31": 31,
}


# ---------------------------------------------------------------------------
# Instruction representation
# ---------------------------------------------------------------------------

@dataclass
class LsInstruction:
    """An instruction for the linear scan allocator.

    Attributes
    ----------
    id:
        Unique index within the basic block.
    opcode:
        Instruction mnemonic (e.g. "add", "lw", "sw").
    operands:
        List of operand strings (register names, immediates).
    defines:
        Set of virtual register names written by this instruction.
    uses:
        Set of virtual register names read by this instruction.
    comment:
        Optional comment string.
    """
    id: int
    opcode: str
    operands: list[str] = field(default_factory=list)
    defines: set[str] = field(default_factory=set)
    uses: set[str] = field(default_factory=set)
    comment: str = ""

    def __repr__(self) -> str:
        return (f"LsInstruction({self.id}, {self.opcode}, "
                f"def={self.defines}, use={self.uses})")

    def to_asm(self, rename: Optional[dict[str, str]] = None) -> str:
        """Emit this instruction as assembly after register renaming."""
        ops = self.operands[:]
        if rename:
            ops = [rename.get(o, o) for o in ops]
        parts = [f"  {self.opcode}"]
        if ops:
            parts.append(" " + ", ".join(ops))
        if self.comment:
            parts.append(f"  # {self.comment}")
        return "".join(parts)


# ---------------------------------------------------------------------------
# Live interval
# ---------------------------------------------------------------------------

@dataclass
class LiveInterval:
    """Live interval for a single virtual register in a basic block.

    Attributes
    ----------
    vreg:
        Virtual register name.
    start:
        Instruction index of the first definition.
    end:
        Instruction index of the last use (exclusive bound).
    uses:
        Set of instruction indices where this vreg is used.
    """
    vreg: str
    start: int
    end: int
    uses: set[int] = field(default_factory=set)

    def overlaps(self, other: "LiveInterval") -> bool:
        """Check if two intervals overlap."""
        return self.start < other.end and other.start < self.end

    def contains(self, pos: int) -> bool:
        """Check if a position is within this interval."""
        return self.start <= pos < self.end

    def __repr__(self) -> str:
        return f"LiveInterval({self.vreg}, [{self.start}, {self.end}))"


# ---------------------------------------------------------------------------
# Linear scan allocator
# ---------------------------------------------------------------------------

class LinearScanAllocator:
    """Linear scan register allocator for RISC-V.

    Parameters
    ----------
    phys_regs:
        List of physical register names available for allocation.
        Defaults to all integer registers (excluding special-purpose regs).

    Attributes
    ----------
    stack_slot:
        Current stack slot offset (negative, grows downward).
    alloc_map:
        Mapping from virtual register to assigned physical register.
    spill_code:
        List of spill load/store instructions inserted during allocation.
    """

    def __init__(self, phys_regs: Optional[list[str]] = None):
        self.phys_regs: list[str] = (
            phys_regs if phys_regs is not None
            else list(_DEFAULT_PHYS_REGS)
        )
        self.stack_slot: int = 0
        self.alloc_map: dict[str, str] = {}
        self.spill_code: list[tuple[int, str, str]] = (
            []  # (position, op, operand)
        )
        self._spill_slots: dict[str, int] = {}  # vreg -> slot offset

    # ------------------------------------------------------------------
    # Live interval computation
    # ------------------------------------------------------------------

    def compute_live_intervals(
            self, block: list[LsInstruction],
    ) -> list[LiveInterval]:
        """Compute live intervals for all virtual registers in a basic block.

        Parameters
        ----------
        block:
            List of LsInstruction objects in instruction order.

        Returns
        -------
        List of LiveInterval objects sorted by start position.
        """
        # Collect all virtual register names
        vregs: set[str] = set()
        for inst in block:
            vregs |= inst.defines
            vregs |= inst.uses

        intervals: list[LiveInterval] = []

        for vreg in vregs:
            start = -1
            end = -1
            uses = set()

            for inst in block:
                if vreg in inst.defines:
                    if start == -1:
                        start = inst.id
                if vreg in inst.uses:
                    uses.add(inst.id)
                    end = max(end, inst.id + 1)
                if vreg in inst.defines and vreg in inst.uses:
                    # define and use in same instruction
                    uses.add(inst.id)
                    if start == -1:
                        start = inst.id
                    end = max(end, inst.id + 1)

            if start == -1:
                start = 0  # live-in parameter

            if end == -1:
                end = start + 1

            intervals.append(LiveInterval(
                vreg=vreg, start=start, end=end, uses=uses,
            ))

        return sorted(intervals, key=lambda iv: iv.start)

    # ------------------------------------------------------------------
    # Linear scan allocation
    # ------------------------------------------------------------------

    def allocate(self, intervals: list[LiveInterval]) -> dict[str, str]:
        """Perform linear scan register allocation.

        Parameters
        ----------
        intervals:
            Sorted list of live intervals (by start position).

        Returns
        -------
        Mapping from virtual register name to physical register name.
        """
        self.alloc_map.clear()
        self.spill_code.clear()
        self._spill_slots.clear()

        # Active list: (interval, phys_reg) sorted by increasing end
        active: list[tuple[LiveInterval, str]] = []
        free_regs: list[str] = list(self.phys_regs)

        for interval in intervals:
            # Expire old intervals
            self._expire_old_intervals(active, interval.start, free_regs)

            if free_regs:
                # Assign a free register
                reg = free_regs.pop(0)
                self.alloc_map[interval.vreg] = reg
                active.append((interval, reg))
            else:
                # Need to spill
                spill = self.spill(interval, active, free_regs)
                if spill is not None:
                    # Spill freed a register
                    reg = (
                        free_regs.pop(0) if free_regs
                        else self.phys_regs[0]
                    )
                    self.alloc_map[interval.vreg] = reg
                    active.append((interval, reg))

        return dict(self.alloc_map)

    def _expire_old_intervals(self, active: list[tuple[LiveInterval, str]],
                              current_pos: int,
                              free_regs: list[str]) -> None:
        """Remove intervals from active list that have ended."""
        i = 0
        while i < len(active):
            interval, reg = active[i]
            if interval.end <= current_pos:
                free_regs.append(reg)
                active.pop(i)
            else:
                i += 1

    def spill(self, current: LiveInterval,
              active: list[tuple[LiveInterval, str]],
              free_regs: list[str]) -> Optional[str]:
        """Select a register to spill and emit spill code.

        Chooses the active interval with the farthest end position to spill.

        Parameters
        ----------
        current:
            The live interval that needs a register.
        active:
            Currently active intervals.
        free_regs:
            List of free registers (will be appended to if a spill succeeds).

        Returns
        -------
        The physical register freed by spilling, or None if no spill possible.
        """
        if not active:
            return None

        # Find the active interval with the farthest end
        spill_idx = 0
        farthest_end = active[0][0].end

        for i, (interval, _) in enumerate(active):
            if interval.end > farthest_end:
                farthest_end = interval.end
                spill_idx = i

        spill_interval, spill_reg = active[spill_idx]

        # Only spill if the current interval ends earlier
        if current.end <= spill_interval.end:
            # Spill the farthest interval
            slot = self._get_spill_slot(spill_interval.vreg)
            active.pop(spill_idx)
            # Emit store after the definition point
            self.spill_code.append(
                (spill_interval.start, "sw",
                 f"{spill_reg}, {slot}(sp)  # spill {spill_interval.vreg}")
            )
            free_regs.append(spill_reg)
            return spill_reg

        # Otherwise, spill the current interval
        slot = self._get_spill_slot(current.vreg)
        self.spill_code.append(
            (current.start, "sw",
             f"{self.phys_regs[0]}, {slot}(sp)  # spill {current.vreg}")
        )
        return None

    def _get_spill_slot(self, vreg: str) -> int:
        """Get or allocate a stack slot for a virtual register."""
        if vreg not in self._spill_slots:
            self.stack_slot -= 4
            self._spill_slots[vreg] = self.stack_slot
        return self._spill_slots[vreg]

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    def get_allocated_code(self, block: list[LsInstruction]) -> str:
        """Generate allocated assembly code for the block.

        Parameters
        ----------
        block:
            The original block of LsInstruction objects.

        Returns
        -------
        RISC-V assembly text with physical registers and spill code.
        """
        lines: list[str] = []
        rename = self.alloc_map

        # Build a position -> spill load map
        spill_loads: dict[int, list[str]] = {}
        for pos, op, operand in self.spill_code:
            if "sw" in op:
                lines.append(f"  {op} {operand}")
            else:
                if pos not in spill_loads:
                    spill_loads[pos] = []
                spill_loads[pos].append(f"  {op} {operand}")

        for inst in block:
            # Insert spill loads before instruction
            if inst.id in spill_loads:
                for load_line in spill_loads[inst.id]:
                    lines.append(load_line)

            lines.append(inst.to_asm(rename))

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def report(self) -> str:
        """Return a string summary of the allocation result."""
        total = len(self.alloc_map)
        spilled = len(self._spill_slots)
        parts = []
        parts.append("Linear Scan Register Allocation Report")
        parts.append(f"  Virtual registers allocated: {total}")
        parts.append(f"  Stack spill slots used: {spilled}")
        parts.append(
            f"  Physical registers available: {len(self.phys_regs)}"
        )
        if self._spill_slots:
            parts.append("  Spill details:")
            for vreg, slot in self._spill_slots.items():
                parts.append(f"    {vreg}: sp+{slot}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helper: convert MachineInstr list to LsInstruction list
# ---------------------------------------------------------------------------

def block_from_machine_instrs(
        instrs: list,  # list of MachineInstr
) -> list[LsInstruction]:
    """Convert MachineInstr list to LsInstruction list.

    Parameters
    ----------
    instrs:
        List of MachineInstr objects from register_alloc module.

    Returns
    -------
    List of LsInstruction objects ready for linear scan allocator.
    """
    result = []
    for i, mi in enumerate(instrs):
        defines: set[str] = set()
        uses: set[str] = set()
        operands: list[str] = []

        for op in (mi.dst, mi.src1, mi.src2):
            if op is None:
                continue
            op_str = str(op).lstrip("%")
            if op.kind == "vreg":
                # For the destination operand position
                if op is mi.dst:
                    defines.add(op_str)
                    operands.append(op_str)
                else:
                    uses.add(op_str)
                    operands.append(op_str)
            else:
                operands.append(op_str)

        if mi.op.value == ".label":
            result.append(LsInstruction(
                id=i, opcode=".label", operands=[mi.comment],
                comment=mi.comment,
            ))
        else:
            result.append(LsInstruction(
                id=i,
                opcode=mi.op.value,
                operands=operands,
                defines=defines,
                uses=uses,
                comment=mi.comment,
            ))

    return result
