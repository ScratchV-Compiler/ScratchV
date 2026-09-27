"""Checks independent of graph construction and scheduling decisions."""

from __future__ import annotations

from typing import Sequence

from .schedule_semantics import SchedInst, ScheduleError


def _reads_and_final_writes(instructions: Sequence[SchedInst]) -> tuple[dict, dict]:
    writes: dict[str, int] = {}
    reads = {}
    for inst in instructions:
        for reg in inst.uses:
            reads[inst.id, reg] = writes.get(reg)  # None means the incoming value.
        for reg in inst.defines:
            writes[reg] = inst.id
    return reads, writes


def verify_schedule(
    original: Sequence[SchedInst], candidate: Sequence[SchedInst], dag: Sequence = ()
) -> None:
    """Reject changes to identity, boundaries, register values or effects."""
    expected = {inst.id: inst for inst in original}
    actual = {inst.id: inst for inst in candidate}
    if (
        len(expected) != len(original)
        or len(actual) != len(candidate)
        or expected.keys() != actual.keys()
    ):
        raise ScheduleError("Instruction identities lost or duplicated")
    if any(actual[key] is not inst for key, inst in expected.items()):
        raise ScheduleError("Instruction object or original text was replaced")
    for before, after in zip(original, candidate):
        if before.region != after.region:
            raise ScheduleError("Instruction crossed a region boundary")
        if (before.terminator or before.effects.barrier_reason) and before is not after:
            raise ScheduleError("Pinned instruction moved")
    if _reads_and_final_writes(original) != _reads_and_final_writes(candidate):
        raise ScheduleError("Register read source or final write changed")
    for effect in (lambda i: i.effects.memory != "none", lambda i: i.effects.fp_flags):
        if [i.id for i in original if effect(i)] != [
            i.id for i in candidate if effect(i)
        ]:
            raise ScheduleError("Memory or floating-point effects reordered")
    positions = {inst.id: index for index, inst in enumerate(candidate)}
    for node in dag:
        for pred, _ in node.predecessors:
            if positions[pred.inst.id] >= positions[node.inst.id]:
                raise ScheduleError("Dependency order violated")
