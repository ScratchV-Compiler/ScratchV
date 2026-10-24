"""Allocation outputs must describe only the current call on a reused allocator."""

from copy import deepcopy

import pytest

from scratchv.backend.allocation import AllocationResult
from scratchv.backend.machine_types import (
    FP_TEMP_REGS,
    MachineInstr as MI,
    MachineOp as O,
    MachineOperand as P,
    RegisterClass as RC,
)
from scratchv.backend.regalloc_linear import (
    LinearScanAllocator,
    block_from_machine_instrs,
)


def _stream(kind, seed=7, count=1):
    is_float = kind != "i32"
    width = 64 if kind == "f64" else 32
    reg_class = RC.FPR if is_float else RC.GPR

    def value(index):
        return P.vreg(f"v{index}", reg_class, width)

    load = O.FLD if width == 64 else O.FLW
    add = (O.FADD_D if width == 64 else O.FADD_S) if is_float else O.ADD
    move = (O.FMV_D if width == 64 else O.FMV_S) if is_float else O.MV
    instructions = [
        (
            MI(load, value(index), P.reg(f"{8 * (seed + index)}(sp)"))
            if is_float
            else MI(O.LI, value(index), P.immediate(seed + index))
        )
        for index in range(count)
    ]
    instructions.extend(
        MI(add, value(0), value(0), value(index)) for index in range(1, count)
    )
    instructions.append(
        MI(
            move,
            P.reg("fa0" if is_float else "a0", reg_class, width),
            value(0),
        )
    )
    return instructions


def _run(allocator, entry, instructions):
    if entry == "allocate_result":
        return allocator.allocate_result(instructions)
    text = allocator.emit(block_from_machine_instrs(instructions))
    assert isinstance(text, str)
    if instructions:
        assert text
    else:
        assert text == ""
    return allocator.allocation_result


def _assert_empty(allocator, result):
    assert result == AllocationResult([], {}, {}, 0)
    assert allocator.spill_slot_count == 0
    assert allocator.stack_slot == 0
    assert allocator.spill_store_count == 0
    assert allocator.reload_load_count == 0


class TestRegisterAllocationResultReuse:
    @pytest.mark.parametrize("entry", ["allocate_result", "emit"])
    @pytest.mark.parametrize("kind", ["i32", "f32", "f64"])
    def test_nonempty_empty_nonempty_results_belong_to_each_call(self, entry, kind):
        allocator = LinearScanAllocator()
        first = _run(allocator, entry, _stream(kind))
        assert first.instructions
        assert set(first.register_map) == {"v0"}
        snapshot = deepcopy(first)

        empty = _run(allocator, entry, [])
        _assert_empty(allocator, empty)
        # Clearing the allocator must not clear instructions held by its caller.
        assert first == snapshot

        current_stream = _stream(kind, seed=11)
        current = _run(allocator, entry, current_stream)
        assert current == LinearScanAllocator().allocate_result(current_stream)
        assert current != snapshot
        assert first == snapshot
        assert empty == AllocationResult([], {}, {}, 0)

    @pytest.mark.parametrize("entry", ["allocate_result", "emit"])
    def test_fresh_and_repeated_empty_calls_return_empty_results(self, entry):
        allocator = LinearScanAllocator()
        first = _run(allocator, entry, [])
        _assert_empty(allocator, first)
        _assert_empty(allocator, _run(allocator, entry, []))

        _run(allocator, entry, _stream("i32"))
        assert first == AllocationResult([], {}, {}, 0)

    @pytest.mark.parametrize("entry", ["allocate_result", "emit"])
    @pytest.mark.parametrize("kind", ["i32", "f32", "f64"])
    def test_empty_call_discards_spill_instructions_and_layout(self, entry, kind):
        allocator = LinearScanAllocator(["t0", "t1"])
        count = 4 if kind == "i32" else len(FP_TEMP_REGS) + 1
        first = _run(allocator, entry, _stream(kind, count=count))
        assert first.spill_slots
        assert first.spill_bytes > 0
        assert allocator.spill_store_count > 0
        assert allocator.reload_load_count > 0
        store = {"i32": O.SW, "f32": O.FSW, "f64": O.FSD}[kind]
        assert any(instr.op is store for instr in first.instructions)
        snapshot = deepcopy(first)

        _assert_empty(allocator, _run(allocator, entry, []))
        assert first == snapshot

        current_stream = _stream(kind, seed=11)
        current = _run(allocator, entry, current_stream)
        assert current == LinearScanAllocator(["t0", "t1"]).allocate_result(
            current_stream
        )
        assert current.spill_slots == {}
        assert current.spill_bytes == 0
        assert first == snapshot

    @pytest.mark.parametrize("first_entry", ["allocate_result", "emit"])
    def test_switching_interfaces_and_widths_does_not_reuse_old_output(
        self, first_entry
    ):
        allocator = LinearScanAllocator()
        first = _run(allocator, first_entry, _stream("f64"))
        snapshot = deepcopy(first)
        other_entry = "emit" if first_entry == "allocate_result" else "allocate_result"

        empty = _run(allocator, other_entry, [])
        _assert_empty(allocator, empty)
        # Reuse the same virtual name with a different width in the next call.
        current_stream = _stream("f32", seed=11)
        current = _run(allocator, first_entry, current_stream)
        assert current == LinearScanAllocator().allocate_result(current_stream)
        assert first == snapshot
        assert empty == AllocationResult([], {}, {}, 0)
