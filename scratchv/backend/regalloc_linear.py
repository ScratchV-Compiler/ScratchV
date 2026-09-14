"""Linear Scan Register Allocator for RISC-V.

Implements a basic-block-level linear scan register allocation algorithm
with proper live interval computation, spill/reload code generation, and a
strict no-alias invariant at every program point.

This module is the topic-17 converged principal source (the former
``regalloc_linear_v1_5.py`` implementation, with the reload alias P0 fixed).
It exposes three error classes for fail-loudly behaviour:

* ``RegAllocError``        -- base class for impossible allocation states.
* ``RegisterAliasError``   -- two live vregs claim the same physical register.
* ``SpillFallbackError``   -- strict mode cannot find a reload/scratch register.

Usage::

    from scratchv.backend.regalloc_linear import LinearScanAllocator
    allocator = LinearScanAllocator()
    allocated = allocator.allocate_block(block_instructions)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Collection, Optional

from scratchv.backend.machine_types import (
    ALL_REGS,
    CALLEE_SAVED,
    REG_NUMS,
    MachineInstr,
    MachineOp,
    MachineOperand,
)


# ---------------------------------------------------------------------------
# RISC-V register definitions
# ---------------------------------------------------------------------------

# Compatibility alias: the allocatable integer register pool is the single
# source of truth in ``machine_types.ALL_REGS`` (27 = a0-a7 + t0-t6 + s0-s11).
_INT_REGS: list[str] = list(ALL_REGS)

_DEFAULT_PHYS_REGS: list[str] = ALL_REGS

# Compatibility alias for the canonical register-number table.
_REG_NUMS: dict[str, int] = REG_NUMS

_FP_REGS = [
    "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7",
    "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15",
    "f16", "f17", "f18", "f19",
    "f20", "f21", "f22", "f23",
    "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31",
]

# Jump/branch mnemonics whose label target is carried in ``comment`` for the
# machine path and re-materialised as the final operand by ``to_asm``.
_BRANCH_TARGET_OPS = frozenset({
    "j", "jal", "call", "beq", "bne", "blt", "bge", "bnez",
})

_MEM_OPERAND_RE = re.compile(r"^(-?\d+)\((\w+)\)$")

# Machine ops whose first operand slot is NOT a destination: branches and
# jumps carry a condition/target there, stores carry the value, and calls
# define nothing.  Getting this wrong makes the allocator treat a live-in
# branch condition as a fresh definition (stale write-back + missing reload).
_NO_DST_OPS = frozenset({
    MachineOp.BEQ, MachineOp.BNE, MachineOp.BLT, MachineOp.BGE,
    MachineOp.BNEZ, MachineOp.J, MachineOp.JAL, MachineOp.JALR,
    MachineOp.SW, MachineOp.FSW, MachineOp.FSD,
    MachineOp.CALL, MachineOp.LABEL,
})


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class RegAllocError(RuntimeError):
    """Register allocation cannot produce a correct result (fail loudly)."""


class RegisterAliasError(RegAllocError):
    """Invariant I1/I2 violated: two live vregs claim one physical register."""


class SpillFallbackError(RegAllocError):
    """Strict mode found no scratch/reload register for a spilled vreg."""


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
        Instruction mnemonic (e.g. "add", "lw", "sw"); labels use ".label".
    operands:
        List of operand strings (register names, immediates, ``off(sp)``).
    defines:
        Set of virtual register names written by this instruction.
    uses:
        Set of virtual register names read by this instruction.
    comment:
        Optional comment string.  For branches/jumps it carries the label
        target, and for labels it carries the label name.
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
        """Emit this instruction as assembly after register renaming.

        * ``.label`` instructions render as ``name:``.
        * Jump/branch instructions re-materialise their label target from
          ``comment`` as the final operand (``j .Lend`` / ``bnez a0, .Lend``).
        """
        if self.opcode == ".label":
            name = self.operands[0] if self.operands else self.comment
            return f"{name}:"

        ops = self.operands[:]
        if rename:
            ops = [rename.get(o, o) for o in ops]

        target: Optional[str] = None
        if self.opcode in _BRANCH_TARGET_OPS and self.comment:
            target = self.comment

        parts = [f"  {self.opcode}"]
        rendered = ops + ([target] if target is not None else [])
        if rendered:
            parts.append(" " + ", ".join(rendered))
        if self.comment and target is None:
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
        Instruction index of the first definition (0 for live-in values).
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
        """Check if a position is within this interval (half-open)."""
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
        Physical register names available for allocation.  Defaults to
        ``machine_types.ALL_REGS`` (27 integer registers).
    stack_base:
        Byte offset of this block's spill area relative to the function
        frame base.  Slots are assigned as ``stack_base + 4 * index``.
    strict:
        ``True`` (default): degenerate paths raise ``RegAllocError``.
        ``False``: pressure-measurement mode, counts fallbacks instead.
    pre_spilled:
        Vregs forced to memory (cross-block values).  Each use gets a
        reload and each redefinition is written back to the slot.

    Attributes
    ----------
    alloc_map:
        vreg -> physical register, or ``"SPILL_<vreg>"`` marker.
    spill_code:
        position -> spill write-back store lines.
    """

    def __init__(
        self,
        phys_regs: Optional[list[str]] = None,
        *,
        stack_base: int = 0,
        strict: bool = True,
        pre_spilled: Collection[str] = (),
        slot_hints: Optional[dict[str, int]] = None,
    ):
        self.phys_regs: list[str] = (
            list(phys_regs) if phys_regs is not None
            else list(_DEFAULT_PHYS_REGS)
        )
        self.stack_base: int = int(stack_base)
        self.strict: bool = bool(strict)
        self.pre_spilled: tuple[str, ...] = tuple(pre_spilled)
        # Function-wide slot assignments (cross-block forced-spill values)
        # must be identical in every block, so callers pin them here.
        self._slot_hints: dict[str, int] = dict(slot_hints or {})

        self.stack_slot: int = self.stack_base
        self.alloc_map: dict[str, str] = {}
        self.spill_code: dict[int, list[str]] = {}  # pos -> [sw asm lines]
        self._spill_slots: dict[str, int] = {}  # vreg -> frame offset
        self._reloads: dict[int, list[tuple[str, int]]] = (
            {}  # pos -> [(vreg, slot), ...]
        )
        self._spilled: set[str] = set()
        self._intervals: list[LiveInterval] = []
        self._vreg_interval: dict[str, LiveInterval] = {}
        self._evictions: dict[int, list[str]] = {}  # pos -> sw lines
        self._eviction_events: dict[int, list[tuple[str, int, str]]] = (
            {}  # pos -> [(victim, slot, preg)] allocation-time evictions
        )
        self.peak_active: int = 0
        self.peak_real_pressure: int = 0
        self._scratch_cache: dict[str, str] = {}
        self.fallback_count: int = 0

    # ------------------------------------------------------------------
    # Read-only state (frame allocator / tests)
    # ------------------------------------------------------------------

    @property
    def spill_slots(self) -> dict[str, int]:
        """vreg -> non-negative frame offset."""
        return dict(self._spill_slots)

    @property
    def spill_bytes(self) -> int:
        """Bytes of spill area consumed by this block."""
        return 4 * len(self._spill_slots)

    @property
    def used_callee_saved(self) -> set[str]:
        """Callee-saved registers actually assigned by this allocator."""
        return {r for r in self.alloc_map.values() if r in CALLEE_SAVED}

    # ------------------------------------------------------------------
    # Live interval computation
    # ------------------------------------------------------------------

    def compute_live_intervals(
            self, block: list[LsInstruction],
    ) -> list[LiveInterval]:
        """Compute live intervals for all virtual registers in a block.

        Returns intervals sorted by ``(start, end, vreg)`` so allocation is
        deterministic regardless of set iteration order.
        """
        vregs: set[str] = set()
        for inst in block:
            vregs |= inst.defines
            vregs |= inst.uses

        intervals: list[LiveInterval] = []

        for vreg in vregs:
            start = -1
            end = -1
            uses: set[int] = set()

            for inst in block:
                if vreg in inst.defines and start == -1:
                    start = inst.id
                if vreg in inst.uses:
                    uses.add(inst.id)
                    end = max(end, inst.id + 1)

            if start == -1:
                start = 0  # live-in parameter
            if end == -1:
                end = start + 1  # pure definition

            intervals.append(LiveInterval(
                vreg=vreg, start=start, end=end, uses=uses,
            ))

        return sorted(intervals, key=lambda iv: (iv.start, iv.end, iv.vreg))

    # ------------------------------------------------------------------
    # Linear scan allocation
    # ------------------------------------------------------------------

    def allocate(self, intervals: list[LiveInterval]) -> dict[str, str]:
        """Perform linear scan register allocation.

        Returns a mapping from virtual register name to physical register
        name (or ``"SPILL_<vreg>"`` for spilled vregs).
        """
        self.alloc_map.clear()
        self.spill_code.clear()
        self._spill_slots.clear()
        self._reloads.clear()
        self._spilled.clear()
        self._evictions.clear()
        self._eviction_events.clear()
        self._scratch_cache.clear()
        self.fallback_count = 0
        self.stack_slot = self.stack_base
        self.peak_active = 0
        self.peak_real_pressure = 0

        intervals = sorted(
            intervals, key=lambda iv: (iv.start, iv.end, iv.vreg))
        self._intervals = intervals
        self._vreg_interval = {iv.vreg: iv for iv in intervals}

        # Seed forced-spill values (cross-block vregs): every use reloads,
        # every redefinition is written back by the codegen path.
        for v in sorted(self.pre_spilled):
            self._spilled.add(v)
            self.alloc_map[v] = f"SPILL_{v}"
            slot = self._get_spill_slot(v)
            iv = self._vreg_interval.get(v)
            if iv is not None:
                for use_pos in sorted(iv.uses):
                    self._reloads.setdefault(use_pos, []).append((v, slot))

        active: list[tuple[LiveInterval, str]] = []
        free_regs: list[str] = list(self.phys_regs)

        for interval in intervals:
            if interval.vreg in self._spilled:
                continue  # pre-spilled (forced-spill) values stay in memory
            self._expire_old_intervals(active, interval.start, free_regs)

            if free_regs:
                reg = free_regs.pop(0)
            else:
                self._spill_for_allocation(interval, active, free_regs)
                reg = free_regs.pop(0)
            self.alloc_map[interval.vreg] = reg
            active.append((interval, reg))

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

    def _spill_for_allocation(
        self,
        current: LiveInterval,
        active: list[tuple[LiveInterval, str]],
        free_regs: list[str],
    ) -> None:
        """Evict the farthest-ending active interval to free its register.

        Reloads are registered for every use of the victim.  Because codegen
        replays the block linearly from the final spill state, uses that
        precede the eviction point also need a reload; the slot is kept
        current by the definition write-back (defined victims) or by the
        entry store emitted for live-in victims.  The freed register is
        appended to *free_regs*.
        """
        if not active:
            raise RegAllocError(
                "register allocation: no physical register available for "
                f"{current.vreg} at position {current.start} "
                f"(pool size {len(self.phys_regs)})"
            )

        spill_idx = 0
        farthest_end = active[0][0].end
        for i, (interval, _) in enumerate(active):
            if interval.end > farthest_end:
                farthest_end = interval.end
                spill_idx = i

        victim, victim_reg = active.pop(spill_idx)
        slot = self._get_spill_slot(victim.vreg)
        self._eviction_events.setdefault(current.start, []).append(
            (victim.vreg, slot, victim_reg)
        )
        self.alloc_map[victim.vreg] = f"SPILL_{victim.vreg}"
        self._spilled.add(victim.vreg)
        # Register reloads for every use: codegen replays the block linearly
        # from the final (post-allocation) spill state, so uses that precede
        # the eviction point also need a reload.  The slot is kept current by
        # the definition write-back (defined victims) or by the entry store
        # emitted for live-in victims.
        for use_pos in sorted(victim.uses):
            self._reloads.setdefault(use_pos, []).append(
                (victim.vreg, slot))
        free_regs.append(victim_reg)
        return None

    def _get_spill_slot(self, vreg: str) -> int:
        """Get or allocate a non-negative frame offset for a vreg."""
        hint = self._slot_hints.get(vreg)
        if hint is not None:
            self._spill_slots[vreg] = hint
            return hint
        if vreg not in self._spill_slots:
            self._spill_slots[vreg] = self.stack_slot
            self.stack_slot += 4
        return self._spill_slots[vreg]

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    def emit(self, block: list[LsInstruction]) -> str:
        """Main entry point: allocate registers and emit assembly text."""
        return self.get_allocated_code(block)

    def get_allocated_code(self, block: list[LsInstruction]) -> str:
        """Emit allocated assembly text with spill/reload instructions."""
        allocated = self.allocate_block(block)
        return "\n".join(inst.to_asm() for inst in allocated)

    def allocate_block(
        self, block: list[LsInstruction],
    ) -> list[LsInstruction]:
        """Allocate + rename a block, returning a new instruction sequence.

        The returned sequence contains the original instructions with
        physical register operands plus inserted ``lw``/``sw`` spill and
        reload instructions.  No operand retains a vreg or ``SPILL_``
        prefix.
        """
        intervals = self.compute_live_intervals(block)
        self.allocate(intervals)
        return self._build_allocated_block(block)

    def emit_machine_instrs(
        self, instrs: list[MachineInstr],
    ) -> list[MachineInstr]:
        """MachineInstr-level entry point for a single basic block."""
        return machine_instrs_from_block(
            self.allocate_block(block_from_machine_instrs(instrs))
        )

    def _build_allocated_block(
        self, block: list[LsInstruction],
    ) -> list[LsInstruction]:
        out: list[LsInstruction] = []
        rename: dict[str, str] = dict(self.alloc_map)

        first_def: dict[str, int] = {}
        for inst in block:
            for d in inst.defines:
                if d not in first_def:
                    first_def[d] = inst.id

        # Allocation-time evictions.  For a victim defined in this block the
        # definition write-back keeps its slot current, so no store is needed
        # (and its allocation-time register may be stale).  A live-in victim
        # has no write-back, so its incoming value is captured at block entry
        # before any scratch/reload can reuse the register.
        entry_id = block[0].id if block else 0
        for evict_pos in sorted(self._eviction_events):
            for victim, slot, victim_reg in self._eviction_events[evict_pos]:
                if victim in first_def:
                    continue
                line = f"  sw {victim_reg}, {slot}(sp)  # evict {victim}"
                out.append(_parse_line(line, entry_id))
                self._evictions.setdefault(entry_id, []).append(line)
                rename[victim] = f"SPILL_{victim}"

        for inst in block:
            owners = self._occupied_at(inst.id, inst)
            loaded: dict[str, str] = {}

            # Reloads: dedup per vreg within one instruction slot, never
            # share a reload register across different vregs.
            for vreg, slot in self._reloads.get(inst.id, []):
                reg, stores = self._pick_reload_reg(
                    inst, vreg, slot, rename, owners, loaded)
                for line in stores:
                    out.append(_parse_line(line, inst.id))
                if vreg not in loaded:
                    out.append(LsInstruction(
                        inst.id, "lw", [reg, f"{slot}(sp)"],
                        comment=f"reload {vreg}",
                    ))
                    loaded[vreg] = reg
                    rename[vreg] = reg

            busy = set(owners) | set(loaded.values())

            # Write back redefinitions of spilled vregs.
            for d in sorted(inst.defines):
                if d not in self._spilled:
                    continue
                slot = self._spill_slots.get(d, self.stack_base)
                cur = rename.get(d)
                if cur is None or str(cur).startswith("SPILL_"):
                    cur = self._pick_scratch(d, busy=busy)
                    rename[d] = cur
                    busy.add(cur)
                self.spill_code.setdefault(inst.id, []).append(
                    f"  sw {cur}, {slot}(sp)  # store redefined {d}"
                )

            renamed_ops = [rename.get(o, o) for o in inst.operands]
            out.append(LsInstruction(
                inst.id, inst.opcode, renamed_ops,
                defines=set(inst.defines), uses=set(inst.uses),
                comment=inst.comment,
            ))

            for line in self.spill_code.get(inst.id, []):
                out.append(_parse_line(line, inst.id))

        return out

    def _occupied_at(self, pos: int, inst: LsInstruction) -> dict[str, str]:
        """Return the exclusive ``preg -> vreg`` ownership map at *pos*.

        A vreg occupies its register when its interval strictly spans the
        position, or when it is read-modify-written by the instruction at
        *pos*.  A pure definition at ``start == pos`` does not occupy the
        register (invariant I3), so reloads may target it.

        Raises ``RegisterAliasError`` if two live vregs claim one register
        (invariant I1).
        """
        owners: dict[str, str] = {}
        for v, r in self.alloc_map.items():
            if v in self._spilled:
                continue
            if r not in self.phys_regs:
                continue
            iv = self._vreg_interval.get(v)
            if iv is None:
                continue
            if iv.start == pos:
                # At the interval start a pure definition does not occupy
                # its register yet (invariant I3); a live-in value (synthetic
                # start 0 without a definition here) or a read-modify-write
                # does.
                occupies = not (v in inst.defines and v not in inst.uses)
            else:
                occupies = iv.start < pos < iv.end
            if not occupies:
                continue
            prev = owners.get(r)
            if prev is not None and prev != v:
                raise RegisterAliasError(
                    f"position {pos}: {prev} and {v} both claim {r}")
            owners[r] = v
        return owners

    def _select_victim(
        self, inst: LsInstruction, owners: dict[str, str], used: set[str],
    ) -> Optional[tuple[str, str]]:
        """Select an evictable (vreg, preg); return None if none qualifies.

        Constraints: the register must actually be in use, the vreg must
        not be an operand of the current instruction, the vreg must have a
        future use (otherwise eviction buys nothing), and the register
        must be exclusively owned.  Ties break by vreg name for determinism.
        """
        protected = inst.uses | inst.defines
        best: Optional[str] = None
        best_end = -1
        for r, v in owners.items():
            if r not in used:
                continue
            if v in protected:
                continue
            iv = self._vreg_interval.get(v)
            if iv is None:
                continue
            if not any(u > inst.id for u in iv.uses):
                continue
            if iv.end > best_end or (iv.end == best_end
                                     and best is not None and v < best):
                best, best_end = v, iv.end
        if best is None:
            return None
        return best, self.alloc_map[best]

    def _pick_reload_reg(
        self,
        inst: LsInstruction,
        vreg: str,
        slot: int,
        rename: dict[str, str],
        owners: dict[str, str],
        loaded: dict[str, str],
    ) -> tuple[str, list[str]]:
        """Pick a reload target register.

        Returns ``(register, store_lines_to_emit_before_the_lw)``.  A vreg
        already reloaded in this instruction reuses its binding (no second
        ``lw``); different vregs never share a reload register (invariant
        I2/I4).
        """
        if vreg in loaded:
            return loaded[vreg], []

        used = set(owners) | set(loaded.values())

        for r in self.phys_regs:
            if r not in used:
                return r, []

        picked = self._select_victim(inst, owners, used)
        if picked is None:
            if self.strict:
                raise SpillFallbackError(
                    "regalloc: cannot find a reload register for "
                    f"{vreg} at position {inst.id}: all registers are held "
                    f"by active values or by the instruction operands "
                    f"{sorted(inst.uses | inst.defines)}"
                )
            self.fallback_count += 1
            if loaded:
                return next(iter(loaded.values())), []
            for r in self.phys_regs:
                if owners.get(r) not in (inst.uses | inst.defines):
                    return r, []
            return self.phys_regs[0], []

        victim, reg = picked
        slot_v = self._get_spill_slot(victim)
        stores = [
            f"  sw {reg}, {slot_v}(sp)  # evict {victim} for reload"
        ]
        self._spilled.add(victim)
        rename[victim] = f"SPILL_{victim}"
        iv = self._vreg_interval.get(victim)
        if iv is not None:
            for u in sorted(iv.uses):
                if u > inst.id:
                    self._reloads.setdefault(u, []).append((victim, slot_v))
        return reg, stores

    def _pick_scratch(self, vreg: str, busy: set[str]) -> str:
        """Pick a scratch register for a spilled vreg definition.

        *busy* holds every register that must not be clobbered at this
        point (live owners, reload targets, previously chosen scratches).
        """
        candidate = self._scratch_cache.get(vreg)
        if candidate is not None and candidate not in busy:
            return candidate
        for reg in self.phys_regs:
            if reg not in busy:
                self._scratch_cache[vreg] = reg
                return reg
        if self.strict:
            raise SpillFallbackError(
                "regalloc: no scratch register available for redefined "
                f"spilled vreg {vreg}: busy={sorted(busy)}"
            )
        self.fallback_count += 1
        if candidate is not None:
            return candidate
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
        parts.append(
            f"  Peak real pressure (incl. self-spilled): "
            f"{self.peak_real_pressure}")
        parts.append(
            f"  Physical registers available: {len(self.phys_regs)}"
        )
        if self.fallback_count:
            parts.append(f"  Non-strict fallbacks: {self.fallback_count}")
        if self._spill_slots:
            parts.append("  Spill details (frame-relative, non-negative):")
            for vreg, slot in self._spill_slots.items():
                parts.append(f"    {vreg}: sp+{slot}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers: spill line <-> LsInstruction, MachineInstr conversion
# ---------------------------------------------------------------------------

def _parse_line(line: str, inst_id: int) -> LsInstruction:
    """Convert an internally generated spill/reload line to LsInstruction."""
    if "#" in line:
        body, comment = line.split("#", 1)
        comment = comment.strip()
    else:
        body, comment = line, ""
    tokens = body.strip().split(None, 1)
    opcode = tokens[0]
    operands = (
        [o.strip() for o in tokens[1].split(",") if o.strip()]
        if len(tokens) > 1 else []
    )
    return LsInstruction(inst_id, opcode, operands, comment=comment)


def _to_mop(s: str) -> MachineOperand:
    """Convert an operand string to a MachineOperand (exact-match table)."""
    if s in REG_NUMS:
        return MachineOperand.reg(s)
    mem = _MEM_OPERAND_RE.match(s)
    if mem is not None:
        return MachineOperand.mem(int(mem.group(1)), mem.group(2))
    try:
        return MachineOperand.immediate(int(s))
    except ValueError:
        return MachineOperand.vreg(s)


def block_from_machine_instrs(
        instrs: list,  # list of MachineInstr
) -> list[LsInstruction]:
    """Convert MachineInstr list to LsInstruction list.

    Physical-register operands (including operands mis-typed as ``vreg``
    whose names are real registers, e.g. ``sp``) are not treated as
    allocatable virtual registers.
    """
    result = []
    for i, mi in enumerate(instrs):
        defines: set[str] = set()
        uses: set[str] = set()
        operands: list[str] = []
        has_dst = mi.op not in _NO_DST_OPS

        for idx, op in enumerate((mi.dst, mi.src1, mi.src2)):
            if op is None:
                continue
            op_str = str(op).lstrip("%")
            if op.kind == "vreg" and op_str not in REG_NUMS:
                if has_dst and idx == 0:
                    defines.add(op_str)
                else:
                    uses.add(op_str)
            operands.append(op_str)

        if mi.op.value == ".label":
            name = mi.comment or (operands[0] if operands else "")
            result.append(LsInstruction(
                id=i, opcode=".label", operands=[name], comment=name,
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

    Enables the linear-scan allocator's output to be consumed by
    ``AsmEmitter``.  Branch targets stay in ``comment``; memory operands
    (``off(sp)``) become ``MachineOperand.mem``.
    """
    result = []
    for inst in block:
        if inst.opcode == ".label":
            name = inst.comment or (inst.operands[0] if inst.operands else "")
            result.append(MachineInstr(MachineOp.LABEL, comment=name))
            continue

        try:
            mop = MachineOp(inst.opcode)
        except ValueError as exc:
            raise RegAllocError(
                f"unsupported opcode {inst.opcode!r} in linear-scan "
                "output"
            ) from exc

        ops = [_to_mop(o) for o in inst.operands]
        dst = ops[0] if len(ops) >= 1 else None
        src1 = ops[1] if len(ops) >= 2 else None
        src2 = ops[2] if len(ops) >= 3 else None

        result.append(MachineInstr(mop, dst, src1, src2, inst.comment))

    return result
