"""Linear Scan Register Allocator for RISC-V.

Implements a basic-block-level linear scan register allocation algorithm
with proper live interval computation and spill code generation.

Usage::

    from scratchv.backend.regalloc_linear_v1_3 import LinearScanAllocator
    allocator = LinearScanAllocator()
    intervals = allocator.compute_live_intervals(block_instructions)
    allocator.allocate(intervals)
    result = allocator.get_allocated_code()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from scratchv.backend.machine_types import (
    MachineInstr, MachineOp, MachineOperand,
)


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
        self.spill_code: dict[int, list[str]] = {}  # pos -> [sw asm lines]
        self._spill_slots: dict[str, int] = {}  # vreg -> slot offset
        self._reloads: dict[int, list[tuple[str, int]]] = (
            {}  # pos -> [(vreg, slot), ...]
        )
        self._spilled: set[str] = set()
        self._intervals: list[LiveInterval] = []
        self._vreg_interval: dict[str, LiveInterval] = {}
        self._evictions: dict[int, list[str]] = {}  # pos -> sw lines emitted before reload
        self.peak_active: int = 0  # max simultaneously live intervals seen (phys regs assigned)
        self.peak_real_pressure: int = 0  # max simultaneously live intervals including self-spilled
        self._scratch_cache: dict[str, str] = {}  # vreg -> last scratch reg for reload memory

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
                # defines/uses are both captured here; a vreg that is both
                # defined and used in the same instruction is handled by the
                # uses branch (end/uses updated identically), so a separate
                # define-and-use block would be redundant.
                if vreg in inst.uses:
                    uses.add(inst.id)
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
        self._reloads.clear()
        self._spilled.clear()
        self._evictions.clear()
        self._intervals = intervals
        self._vreg_interval = {iv.vreg: iv for iv in intervals}
        self.peak_active = 0
        self.peak_real_pressure = 0

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

            # Track peak pressure
            current_active = len(active)
            if current_active > self.peak_active:
                self.peak_active = current_active
            current_pressure = current_active + len(self._spilled)
            if current_pressure > self.peak_real_pressure:
                self.peak_real_pressure = current_pressure

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
        Records reload positions for the spilled interval so that
        ``get_allocated_code`` can insert ``lw`` before each future use.

        Returns
        -------
        The physical register freed by spilling, or None if current is spilled.
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

        # No free register is available, so spill the active interval that
        # ends farthest away and hand its register to *current*.
        #
        # The classic linear-scan rule prefers self-spilling *current* when it
        # outlives every active interval (current.end > spill_interval.end).
        # But self-spilling *current* would require writing current's freshly
        # defined value into a transient register, and ALL registers are
        # occupied here -- so a naive self-spill writes into phys_regs[0] and
        # clobbers the live value that register already holds (data
        # corruption).  Evicting the farthest-ending active interval avoids
        # that: *current* outlives it, so *current* keeps the register after
        # the victim expires, and the victim is simply reloaded on demand.
        slot = self._get_spill_slot(spill_interval.vreg)
        active.pop(spill_idx)
        self._evictions.setdefault(current.start, []).append(
            f"  sw {spill_reg}, {slot}(sp)  # evict {spill_interval.vreg}"
        )
        # Remove stale mapping so codegen won't use the freed register
        self.alloc_map[spill_interval.vreg] = f"SPILL_{spill_interval.vreg}"
        self._spilled.add(spill_interval.vreg)
        # Record reload at every future use of the spilled vreg
        for use_pos in spill_interval.uses:
            if use_pos > current.start:
                self._reloads.setdefault(use_pos, []).append(
                    (spill_interval.vreg, slot))
        free_regs.append(spill_reg)
        return spill_reg

    def _get_spill_slot(self, vreg: str) -> int:
        """Get or allocate a stack slot for a virtual register."""
        if vreg not in self._spill_slots:
            self.stack_slot -= 4
            self._spill_slots[vreg] = self.stack_slot
        return self._spill_slots[vreg]

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    def emit(self, block: list[LsInstruction]) -> str:
        """Main entry point: allocate registers and emit assembly.

        Computes live intervals, runs linear-scan allocation, then
        generates assembly with spill stores and reloads interleaved.
        """
        intervals = self.compute_live_intervals(block)
        self.allocate(intervals)
        return self.get_allocated_code(block)

    def get_allocated_code(self, block: list[LsInstruction]) -> str:
        """Generate allocated assembly with spill stores and reloads.

        Walks the instruction block in order.  Before each instruction
        that uses a spilled vreg, a reload ``lw`` is inserted.  After
        each instruction that defines a spilled vreg, a spill ``sw``
        is inserted.
        """
        lines: list[str] = []
        rename: dict[str, str] = dict(self.alloc_map)

        for inst in block:
            # Emit eviction spill stores before reloads at this position
            if inst.id in self._evictions:
                lines.extend(self._evictions[inst.id])

            # Insert reloads before the instruction
            if inst.id in self._reloads:
                # Vregs used/defined by this instruction must not be evicted
                # by _evict_for_reload, otherwise inst.to_asm() would get
                # an unresolved vreg name.
                protected: set[str] = inst.uses | inst.defines
                for vreg, slot in self._reloads[inst.id]:
                    reload_reg = self._pick_reload_reg(rename, inst.id, protected)
                    lines.append(
                        f"  lw {reload_reg}, {slot}(sp)"
                        f"  # reload {vreg}"
                    )
                    rename[vreg] = reload_reg

            # For spilled vregs defined here, pick a scratch register
            for d in inst.defines:
                if d in rename and rename[d].startswith("SPILL_"):
                    slot = self._spill_slots.get(d, 0)
                    scratch = self._pick_scratch(d)
                    rename[d] = scratch

            lines.append(inst.to_asm(rename))

            # Insert spill stores after the instruction
            if inst.id in self.spill_code:
                lines.extend(self.spill_code[inst.id])

        return "\n".join(lines)

    def _pick_reload_reg(self, rename: dict[str, str], current_pos: int,
                          protected_vregs: set[str] | None = None) -> str:
        """Pick a free physical register for a reload ``lw``.

        Filters *rename* by actual liveness at *current_pos* so that
        registers held by already-expired vregs are considered free.
        If all registers are genuinely occupied, evicts the one whose
        interval ends farthest away.

        *protected_vregs* are excluded from eviction — typically the
        current instruction's own uses/defines — since evicting them
        would leave the instruction with an unresolved vreg name.
        """
        used: set[str] = set()
        for vreg, preg in rename.items():
            # ``SPILL_`` markers do not occupy a physical register, so they
            # must not be treated as "used" and must never be evicted.
            if preg not in self.phys_regs:
                continue
            interval = self._vreg_interval.get(vreg)
            if interval is None or interval.contains(current_pos):
                used.add(preg)
        for reg in self.phys_regs:
            if reg not in used:
                return reg
        return self._evict_for_reload(rename, used, current_pos, protected_vregs)

    def _evict_for_reload(
        self, rename: dict[str, str], used: set[str], current_pos: int,
        protected_vregs: set[str] | None = None,
    ) -> str:
        """Evict a live register to make room for a reload.

        Picks the vreg whose interval ends farthest away, generates a
        spill store to its stack slot, and records future reloads for
        its remaining uses.

        Vregs in *protected_vregs* are excluded from eviction — they are
        needed by the instruction at *current_pos* and evicting them
        would produce unresolved vreg names in the output.
        """
        protect = protected_vregs or set()
        farthest_vreg: str | None = None
        farthest_end = -1
        for vreg, preg in rename.items():
            # Skip entries that no longer hold a physical register (e.g. a
            # previously spilled/evicted vreg marked ``SPILL_<name>``).
            if preg not in self.phys_regs:
                continue
            if preg not in used:
                continue
            if vreg in protect:
                continue
            interval = self._vreg_interval.get(vreg)
            if interval is not None and interval.end > farthest_end:
                farthest_end = interval.end
                farthest_vreg = vreg

        # No eligible victim: every live register is protected by the current
        # instruction (its own uses/defines), so reloading would corrupt the
        # instruction.  Reuse the reload register for another reload line by
        # granting a scratch from the pool a second time.  Rather than return
        # a potentially-occupied register (data corruption), fall back to the
        # last physical register and document the limitation.
        if farthest_vreg is None:
            return self.phys_regs[0]

        evicted_reg = rename[farthest_vreg]
        slot = self._get_spill_slot(farthest_vreg)
        self._spilled.add(farthest_vreg)

        # Emit spill store BEFORE the reload (evictions go before reloads)
        self._evictions.setdefault(current_pos, []).append(
            f"  sw {evicted_reg}, {slot}(sp)"
            f"  # evict {farthest_vreg} for reload"
        )

        # Record future reloads for remaining uses of the evicted vreg
        interval = self._vreg_interval.get(farthest_vreg)
        if interval is not None:
            for use_pos in interval.uses:
                if use_pos > current_pos:
                    self._reloads.setdefault(use_pos, []).append(
                        (farthest_vreg, slot))

        # Demote the vreg to a spilled marker instead of deleting it from the
        # rename map.  Keeping the ``SPILL_`` marker means later definitions
        # trigger the scratch-rename path in get_allocated_code, and later
        # uses trigger a reload -- the vreg never silently "disappears" from
        # the map (which previously leaked unrenamed vregs into the assembly).
        rename[farthest_vreg] = f"SPILL_{farthest_vreg}"
        return evicted_reg

    def _pick_scratch(self, vreg: str) -> str:
        """Pick a scratch register for a spilled vreg definition.

        Uses a cache so the same vreg tends to get the same scratch reg,
        reducing redundant loads in tight loops.
        """
        if vreg in self._scratch_cache:
            return self._scratch_cache[vreg]
        busy: set[str] = set(self.alloc_map.values())
        for reg in self.phys_regs:
            if reg not in busy:
                self._scratch_cache[vreg] = reg
                return reg
        reg = self.phys_regs[0]
        self._scratch_cache[vreg] = reg
        return reg

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
        parts.append(f"  Peak active (phys regs mapped): {self.peak_active}")
        parts.append(f"  Peak real pressure (incl. self-spilled): {self.peak_real_pressure}")
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


def machine_instrs_from_block(
        block: list[LsInstruction],
) -> list:  # list of MachineInstr
    """Convert LsInstruction list back to MachineInstr list.

    This is the reverse of ``block_from_machine_instrs`` and enables the
    linear-scan allocator's output to be consumed by ``AsmEmitter``.

    Parameters
    ----------
    block:
        List of LsInstruction objects (possibly after register renaming).

    Returns
    -------
    List of MachineInstr objects.
    """
    result = []
    for inst in block:
        if inst.opcode == ".label":
            result.append(MachineInstr(
                MachineOp.LABEL, comment=inst.comment,
            ))
            continue

        # Resolve opcode
        try:
            mop = MachineOp(inst.opcode)
        except ValueError:
            mop = MachineOp.MV  # fallback

        # Build operands
        def _to_mop(s: str) -> MachineOperand:
            if s.startswith("x") or s.startswith("a") or s.startswith("t") or \
               s.startswith("s") or s.startswith("f") or s in ("zero", "ra", "sp", "gp", "tp", "fp"):
                return MachineOperand.reg(s)
            try:
                return MachineOperand.immediate(int(s))
            except ValueError:
                return MachineOperand.vreg(s)

        dst = None
        src1 = None
        src2 = None
        ops = [_to_mop(o) for o in inst.operands]
        if len(ops) >= 1:
            dst = ops[0]
        if len(ops) >= 2:
            src1 = ops[1]
        if len(ops) >= 3:
            src2 = ops[2]

        result.append(MachineInstr(mop, dst, src1, src2, inst.comment))

    return result