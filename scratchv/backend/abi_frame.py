"""RISC-V ABI stack-frame finalization for allocated assembly."""

from __future__ import annotations

import re
from dataclasses import replace

from scratchv.backend.machine_types import CALLEE_SAVED
from scratchv.backend.allocation import AllocationResult


def _function_slots(body, spill_slots, local_slots):
    """Compact only this function's referenced slots, retaining source offsets.

    Allocation statistics and temporary offsets cover the whole program.
    Their offsets remain lookup keys; each function gets its own runtime frame.
    """
    spill_offsets = set()
    local_offsets = set()
    for line in body:
        if "[regalloc:local]" in line:
            match = re.search(r"addi\s+\w+\s*,\s*sp\s*,\s*(-\d+)", line)
            if match:
                local_offsets.add(int(match.group(1)))
        elif "[regalloc:" in line:
            spill_offsets.update(int(offset) for offset in re.findall(r"(-\d+)\(sp\)", line))

    def compact(offsets, slots, kind):
        result = {}
        size = 0
        for offset in sorted(offsets, reverse=True):
            if offset not in slots:
                raise ValueError(f"{kind} access has no layout entry: {offset}")
            slot = slots[offset]
            size = (size + slot.size + slot.alignment - 1) // slot.alignment * slot.alignment
            result[offset] = replace(slot, offset=-size)
        return result, size

    spills, spill_bytes = compact(spill_offsets, spill_slots, "spill")
    locals_, local_bytes = compact(local_offsets, local_slots, "local")
    return spills, spill_bytes, locals_, local_bytes


def apply_abi_frames(assembly: str, spill_slot_count: int | AllocationResult) -> str:
    """Reserve spill storage and preserve used callee-saved registers.

    Register allocators use compact negative offsets while rewriting.  This
    final production-code step gives each emitted function a real, aligned
    frame and rebases tagged allocator spill accesses into that frame.
    """
    layout = spill_slot_count if isinstance(spill_slot_count, AllocationResult) else None
    spill_bytes = layout.spill_bytes if layout is not None else spill_slot_count * 4
    if spill_bytes < 0:
        raise ValueError("spill bytes must be nonnegative")
    slots_by_offset = {}
    local_slots = layout.local_slots if layout is not None else {}
    local_bytes = layout.local_bytes if layout is not None else 0
    if local_bytes < 0:
        raise ValueError("local bytes must be nonnegative")
    if layout is not None:
        occupied = set()
        for slot in layout.spill_slots.values():
            if (slot.size not in {4, 8} or slot.alignment != slot.size
                    or slot.offset >= 0 or slot.offset % slot.alignment
                    or -slot.offset > spill_bytes):
                raise ValueError("invalid spill slot size, alignment or offset")
            addresses = set(range(slot.offset, slot.offset + slot.size))
            if occupied & addresses:
                raise ValueError("overlapping spill slots")
            occupied.update(addresses)
            slots_by_offset[slot.offset] = slot
        occupied.clear()
        for slot in local_slots.values():
            if (slot.size <= 0 or slot.alignment not in {4, 8}
                    or slot.offset >= 0 or slot.offset % slot.alignment
                    or -slot.offset > local_bytes):
                raise ValueError("invalid local stack layout")
            addresses = set(range(slot.offset, slot.offset + slot.size))
            if occupied & addresses:
                raise ValueError("overlapping local stack slots")
            occupied.update(addresses)
    lines = assembly.splitlines()
    # Emitted function metadata, rather than a user's label prefix, determines
    # the role of dot-prefixed symbols. Keep the fallback for legacy raw input.
    declared_functions = {
        match.group(1)
        for line in lines
        if (match := re.fullmatch(
            r"\.type\s+([^\s,]+)\s*,\s*@function", line.partition("#")[0].strip()
        ))
    }
    function_starts = [
        index
        for index, line in enumerate(lines)
        if re.fullmatch(r"[A-Za-z_.$][\w.$]*:", line.strip())
        and (line.strip()[:-1] in declared_functions if declared_functions
             else not line.strip().startswith("."))
    ]
    for position in reversed(range(len(function_starts))):
        start = function_starts[position]
        end = (
            function_starts[position + 1]
            if position + 1 < len(function_starts)
            else len(lines)
        )
        body = lines[start + 1:end]
        body_text = "\n".join(body)
        saved = [
            reg
            for reg in CALLEE_SAVED
            if re.search(rf"(?<![\w.]){re.escape(reg)}(?![\w.])", body_text)
        ]
        if re.search(r"^\s*(?:call\s|jal\s+ra\s*,)", body_text, re.MULTILINE):
            saved.insert(0, "ra")
        function_spills = slots_by_offset
        local_by_offset = {slot.offset: slot for slot in local_slots.values()}
        function_spill_bytes, function_local_bytes = spill_bytes, local_bytes
        if layout is not None and len(function_starts) > 1:
            (function_spills, function_spill_bytes, local_by_offset,
             function_local_bytes) = _function_slots(body, slots_by_offset, local_by_offset)
        alignment = max((slot.alignment for slot in [*function_spills.values(), *local_by_offset.values()]), default=4)
        spill_end = (function_spill_bytes + alignment - 1) // alignment * alignment
        local_end = (function_local_bytes + alignment - 1) // alignment * alignment
        spill_base = (4 * len(saved) + alignment - 1) // alignment * alignment
        payload = spill_base + local_end + spill_end
        if payload == 0:
            continue
        frame_size = (payload + 15) // 16 * 16
        if frame_size > 2032:
            raise ValueError(
                "RISC-V stack frame exceeds the encodable 2032-byte limit: "
                f"{frame_size} bytes"
            )

        rewritten: list[str] = []
        for line in body:
            if "[regalloc:" in line:
                if "[regalloc:local]" in line:
                    def rebase_local(match):
                        offset = int(match.group(2))
                        if offset not in local_by_offset:
                            raise ValueError(f"local access has no layout entry: {offset}")
                        return f"{match.group(1)}{spill_base + local_end + local_by_offset[offset].offset}"
                    line = re.sub(r"(addi\s+\w+\s*,\s*sp\s*,\s*)(-\d+)", rebase_local, line)
                def rebase(match):
                    offset = int(match.group(1))
                    if layout is not None and offset not in function_spills:
                        raise ValueError(f"spill access has no layout entry: {offset}")
                    # Preserve old integer assembly exactly. Mixed frames place
                    # the aligned spill region after saves, then tail padding.
                    base = (spill_base + local_end + spill_end if function_local_bytes or alignment == 8
                            or any(slot.reg_class.value == "fpr" for slot in function_spills.values())
                            else frame_size)
                    return f"{base + (function_spills[offset].offset if layout is not None else offset)}(sp)"
                line = re.sub(
                    r"(-\d+)\(sp\)",
                    rebase,
                    line,
                )
            stripped = line.partition("#")[0].strip()
            if stripped == "ret" or re.match(
                r"jalr\s+(?:zero|x0)\s*,\s*(?:ra|x1)(?:\s*,\s*0)?$",
                stripped,
            ):
                for slot, reg in reversed(list(enumerate(saved))):
                    rewritten.append(f"  lw {reg}, {slot * 4}(sp)  # ABI restore")
                rewritten.append(f"  addi sp, sp, {frame_size}  # destroy stack frame")
            rewritten.append(line)

        prologue = [f"  addi sp, sp, -{frame_size}  # create stack frame"]
        prologue.extend(
            f"  sw {reg}, {slot * 4}(sp)  # ABI save"
            for slot, reg in enumerate(saved)
        )
        lines[start + 1:end] = prologue + rewritten

    suffix = "\n" if assembly.endswith("\n") else ""
    return "\n".join(lines) + suffix
