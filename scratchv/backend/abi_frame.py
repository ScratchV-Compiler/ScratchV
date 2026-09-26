"""RISC-V ABI stack-frame finalization for allocated assembly."""

from __future__ import annotations

import re

from scratchv.backend.machine_types import CALLEE_SAVED


def apply_abi_frames(assembly: str, spill_slot_count: int) -> str:
    """Reserve spill storage and preserve used callee-saved registers.

    Register allocators use compact negative offsets while rewriting.  This
    final production-code step gives each emitted function a real, aligned
    frame and rebases tagged allocator spill accesses into that frame.
    """
    lines = assembly.splitlines()
    function_starts = [
        index
        for index, line in enumerate(lines)
        if re.fullmatch(r"[A-Za-z_$][\w.$]*:", line.strip())
        and not line.strip().startswith(".")
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
        payload = 4 * (spill_slot_count + len(saved))
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
                line = re.sub(
                    r"(-\d+)\(sp\)",
                    lambda match: f"{frame_size + int(match.group(1))}(sp)",
                    line,
                )
            stripped = line.strip()
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
