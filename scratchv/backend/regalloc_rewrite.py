"""Correct spill rewriting shared by the linear-scan allocator variants."""

from __future__ import annotations

from typing import Any

from scratchv.backend.machine_semantics import get_machine_semantics
from scratchv.backend.machine_types import MachineOp


def rewrite_with_spills(allocator: Any, instructions: list[Any]) -> str:
    """Rewrite virtual operands, inserting executable spill/reload code.

    The allocation pass supplies preferred registers and pressure/victim
    decisions.  This final rewrite owns the actual register contents.  In
    particular it guarantees that distinct source vregs of one instruction
    occupy distinct registers, canonicalizes live values to stack slots at
    CFG boundaries, and invalidates ABI-clobbered registers across calls.
    """

    if not instructions:
        return ""
    if not allocator.phys_regs:
        raise RuntimeError("regalloc: physical register pool is empty")

    # Allocation-time event lists are planning artifacts.  Rebuilding the
    # actual transfers here avoids stale same-position eviction/reload state.
    allocator.spill_code.clear()
    allocator._reloads.clear()
    allocator._evictions.clear()

    cfg = allocator.cfg
    block_by_instruction = {
        inst.id: block for block in cfg.blocks for inst in block.instructions
    }
    block_first = {block.start: block for block in cfg.blocks}
    block_last = {block.end - 1: block for block in cfg.blocks}

    all_defines = set().union(*(inst.defines for inst in instructions))
    # Once allocation contains a split/spilled interval, reloads can evict a
    # nominally register-resident value on only one predecessor.  Canonicalize
    # every edge-live value through its stack slot in that case so all incoming
    # paths agree at joins.  The set is fixed before emission; deriving it from
    # mutable rewrite state would make correctness depend on textual block
    # order.
    canonical_vregs: set[str] = set()
    if allocator._spilled:
        canonical_vregs = set().union(
            *(block.live_out for block in cfg.blocks)
        )
    locations: dict[str, str] = {}
    reg_owner: dict[str, str] = {}
    stack_current: set[str] = set()
    rename: dict[str, str] = {}
    lines: list[str] = []

    def has_later_use(vreg: str, position: int, block: Any) -> bool:
        interval = allocator._vreg_interval.get(vreg)
        return bool(
            (interval and any(use > position for use in interval.uses))
            or vreg in block.live_out
        )

    def forget_register(reg: str) -> None:
        owner = reg_owner.pop(reg, None)
        if owner is not None and locations.get(owner) == reg:
            locations.pop(owner, None)

    def store_owner(reg: str, position: int, block: Any, reason: str) -> None:
        owner = reg_owner.get(reg)
        if owner is None or owner in stack_current:
            return
        if not has_later_use(owner, position, block):
            return
        slot = allocator._get_spill_slot(owner)
        allocator._spilled.add(owner)
        lines.append(f"  sw {reg}, {slot}(sp)  # {reason} {owner}")
        stack_current.add(owner)

    def claim_register(vreg: str, reg: str) -> None:
        old = reg_owner.get(reg)
        if old is not None and old != vreg:
            locations.pop(old, None)
        previous = locations.get(vreg)
        if previous is not None and previous != reg:
            reg_owner.pop(previous, None)
        reg_owner[reg] = vreg
        locations[vreg] = reg
        rename[vreg] = reg

    def choose_register(
        vreg: str,
        position: int,
        block: Any,
        protected: set[str],
        preferred: str | None = None,
    ) -> str:
        candidates = []
        if preferred in allocator.phys_regs:
            candidates.append(preferred)
        candidates.extend(reg for reg in allocator.phys_regs if reg not in candidates)

        # Keep the allocator's global assignment stable across CFG edges.
        # Even if another register is currently dead, silently choosing it
        # would make successor blocks read the value from the wrong place.
        if preferred in candidates and preferred not in protected:
            owner = reg_owner.get(preferred)
            if owner is not None and has_later_use(owner, position, block):
                store_owner(preferred, position, block, "evict")
            forget_register(preferred)
            return preferred

        for reg in candidates:
            if reg in protected:
                continue
            owner = reg_owner.get(reg)
            if owner is None or not has_later_use(owner, position, block):
                forget_register(reg)
                return reg

        victims = [reg for reg in candidates if reg not in protected]
        if not victims:
            raise RuntimeError(
                "regalloc: instruction at position "
                f"{position} needs more distinct source registers than the "
                f"{len(allocator.phys_regs)}-register pool provides"
            )

        def next_use(reg: str) -> int:
            owner = reg_owner.get(reg)
            interval = allocator._vreg_interval.get(owner) if owner else None
            future = [use for use in interval.uses if use > position] if interval else []
            return min(future) if future else 1 << 30

        victim = max(victims, key=next_use)
        store_owner(victim, position, block, "evict")
        forget_register(victim)
        return victim

    def store_live_out(block: Any, position: int) -> None:
        for vreg in sorted(block.live_out):
            # Values with a stable global physical assignment already have
            # the same location on every edge.  Only split/spilled values need
            # the canonical stack hand-off between basic blocks.
            if vreg not in canonical_vregs \
                    and allocator.alloc_map.get(vreg) in allocator.phys_regs \
                    and vreg not in allocator._spilled:
                continue
            reg = locations.get(vreg)
            if reg is None or vreg in stack_current:
                continue
            slot = allocator._get_spill_slot(vreg)
            allocator._spilled.add(vreg)
            lines.append(
                f"  sw {reg}, {slot}(sp)  # spill {vreg} at block boundary"
            )
            stack_current.add(vreg)

    for inst in instructions:
        block = block_by_instruction[inst.id]
        if inst.id in block_first:
            locations.clear()
            reg_owner.clear()
            rename.clear()
            stack_current.clear()
            stack_current.update(canonical_vregs & block.live_in)
            # Non-spilled intervals keep one global physical assignment, so
            # every predecessor agrees on their location.  Spilled intervals
            # cross an edge through their canonical stack slot instead.
            for vreg in sorted(block.live_in):
                preferred = allocator.alloc_map.get(vreg)
                if preferred in allocator.phys_regs \
                        and vreg not in canonical_vregs \
                        and vreg not in allocator._spilled \
                        and preferred not in reg_owner:
                    claim_register(vreg, preferred)

        semantics = None
        try:
            semantics = get_machine_semantics(MachineOp(inst.opcode))
        except ValueError:
            pass

        ordered_uses: list[str] = []
        for operand in inst.operands:
            if operand in inst.uses and operand not in ordered_uses:
                ordered_uses.append(operand)
        # Keep malformed/custom LsInstruction tests deterministic too.
        ordered_uses.extend(sorted(inst.uses - set(ordered_uses)))

        if len(ordered_uses) > len(allocator.phys_regs):
            raise RuntimeError(
                "regalloc: instruction at position "
                f"{inst.id} has {len(ordered_uses)} distinct register uses, "
                f"but only {len(allocator.phys_regs)} physical registers"
            )

        # Protect resident values for *all* sources before emitting any load.
        # Otherwise loading the first spilled source could evict a second
        # source whose value has not yet been consumed by the instruction.
        protected: set[str] = {
            locations[vreg]
            for vreg in ordered_uses
            if vreg in locations and reg_owner.get(locations[vreg]) == vreg
        }
        for vreg in ordered_uses:
            resident = locations.get(vreg)
            if resident is not None and reg_owner.get(resident) == vreg:
                reg = resident
            elif vreg not in stack_current and vreg not in all_defines:
                preferred = allocator.alloc_map.get(vreg)
                reg = choose_register(vreg, inst.id, block, protected, preferred)
            else:
                if vreg not in stack_current:
                    raise RuntimeError(
                        f"regalloc: value {vreg!r} has no resident register "
                        f"or initialized spill slot at position {inst.id}"
                    )
                preferred = allocator.alloc_map.get(vreg)
                reg = choose_register(vreg, inst.id, block, protected, preferred)
                slot = allocator._get_spill_slot(vreg)
                lines.append(f"  lw {reg}, {slot}(sp)  # reload {vreg}")
            claim_register(vreg, reg)
            protected.add(reg)

        ordered_defines: list[str] = []
        for operand in inst.operands:
            if operand in inst.defines and operand not in ordered_defines:
                ordered_defines.append(operand)
        ordered_defines.extend(sorted(inst.defines - set(ordered_defines)))

        # RISC-V reads sources before writing rd, so a destination may reuse
        # any source register.  If that source remains live, choose_register
        # first writes its old value to the canonical spill slot; the already
        # materialized source operand still names the same register for this
        # instruction, and later uses reload the saved value.
        definition_protected: set[str] = set()
        for vreg in ordered_defines:
            resident = locations.get(vreg)
            if resident is not None and reg_owner.get(resident) == vreg:
                # A pure redefinition may overwrite its own old value.  Do
                # not classify that overwrite as an eviction merely because
                # the interval also contains uses of the newly defined value.
                reg = resident
            else:
                preferred = allocator.alloc_map.get(vreg)
                reg = choose_register(
                    vreg, inst.id, block, definition_protected, preferred
                )
            claim_register(vreg, reg)
            definition_protected.add(reg)

        is_last = inst.id in block_last
        if is_last and semantics and semantics.is_terminator:
            store_live_out(block, inst.id)

        # Calls fall through but clobber caller-saved registers.  Save only
        # values actually live afterwards and reload them lazily on demand.
        if semantics and semantics.is_call:
            for reg in list(reg_owner):
                if reg in semantics.clobbers:
                    store_owner(reg, inst.id, block, "spill")

        lines.append(inst.to_asm(rename))

        for vreg in ordered_defines:
            stack_current.discard(vreg)

        # Once allocation has split/spilled a vreg, its stack slot is the
        # canonical value between reloads.  Every later definition must update
        # that slot before another instruction can evict the transient result.
        for vreg in ordered_defines:
            if vreg not in allocator._spilled or not has_later_use(vreg, inst.id, block):
                continue
            reg = locations[vreg]
            slot = allocator._get_spill_slot(vreg)
            lines.append(
                f"  sw {reg}, {slot}(sp)  # store redefined {vreg}"
            )
            stack_current.add(vreg)
            forget_register(reg)

        if semantics and semantics.is_call:
            for reg in list(reg_owner):
                if reg in semantics.clobbers:
                    forget_register(reg)

        if is_last and not (semantics and semantics.is_terminator):
            store_live_out(block, inst.id)

    return "\n".join(lines)
