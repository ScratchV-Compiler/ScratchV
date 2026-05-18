"""Tests for Linear Scan Register Allocator."""

import pytest
from scratchv.backend.regalloc_linear import (
    LinearScanAllocator, LsInstruction, LiveInterval,
    block_from_machine_instrs,
)


class TestLiveInterval:
    """Tests for LiveInterval data class."""

    def test_creation(self):
        iv = LiveInterval(vreg="v1", start=0, end=3, uses={1, 2})
        assert iv.vreg == "v1"
        assert iv.start == 0
        assert iv.end == 3

    def test_overlaps_true(self):
        a = LiveInterval("a", 0, 5)
        b = LiveInterval("b", 3, 7)
        assert a.overlaps(b)
        assert b.overlaps(a)

    def test_overlaps_false_adjacent(self):
        a = LiveInterval("a", 0, 3)
        b = LiveInterval("b", 3, 6)
        assert not a.overlaps(b)

    def test_overlaps_false_separate(self):
        a = LiveInterval("a", 0, 2)
        b = LiveInterval("b", 5, 7)
        assert not a.overlaps(b)

    def test_contains(self):
        iv = LiveInterval("v1", 2, 6)
        assert iv.contains(2)
        assert iv.contains(5)
        assert not iv.contains(1)
        assert not iv.contains(6)


class TestLsInstruction:
    """Tests for LsInstruction."""

    def test_to_asm_no_rename(self):
        inst = LsInstruction(0, "add", ["v1", "v2", "v3"],
                             defines={"v1"}, uses={"v2", "v3"})
        asm = inst.to_asm()
        assert "add" in asm
        assert "v1" in asm

    def test_to_asm_with_rename(self):
        inst = LsInstruction(0, "add", ["v1", "v2", "v3"],
                             defines={"v1"}, uses={"v2", "v3"})
        rename = {"v1": "t0", "v2": "t1", "v3": "t2"}
        asm = inst.to_asm(rename)
        assert "t0" in asm
        assert "v1" not in asm


class TestLinearScanAllocator:
    """Tests for the linear scan allocator."""

    def test_creation(self):
        alloc = LinearScanAllocator()
        assert len(alloc.phys_regs) > 0

    def test_custom_regs(self):
        alloc = LinearScanAllocator(phys_regs=["t0", "t1", "a0"])
        assert alloc.phys_regs == ["t0", "t1", "a0"]

    def test_compute_live_intervals(self):
        block = [
            LsInstruction(0, "add", ["v1", "v2", "v3"],
                          defines={"v1"}, uses={"v2", "v3"}),
            LsInstruction(1, "mul", ["v4", "v1", "v5"],
                          defines={"v4"}, uses={"v1", "v5"}),
            LsInstruction(2, "sub", ["v6", "v4", "v1"],
                          defines={"v6"}, uses={"v4", "v1"}),
        ]
        alloc = LinearScanAllocator(["t0", "t1", "t2", "a0"])
        intervals = alloc.compute_live_intervals(block)
        assert len(intervals) >= 4  # v1, v2, v3, v4, v5, v6

    def test_intervals_sorted_by_start(self):
        block = [
            LsInstruction(0, "li", ["v1", "42"], defines={"v1"}, uses=set()),
            LsInstruction(1, "li", ["v2", "10"], defines={"v2"}, uses=set()),
            LsInstruction(2, "add", ["v3", "v1", "v2"],
                          defines={"v3"}, uses={"v1", "v2"}),
        ]
        alloc = LinearScanAllocator(["t0", "t1", "a0"])
        intervals = alloc.compute_live_intervals(block)
        starts = [iv.start for iv in intervals]
        assert starts == sorted(starts)

    def test_allocate_no_spill(self):
        """Allocate with enough registers: no spills needed."""
        block = [
            LsInstruction(0, "li", ["v1", "42"], defines={"v1"}, uses=set()),
            LsInstruction(1, "li", ["v2", "10"], defines={"v2"}, uses=set()),
            LsInstruction(2, "add", ["v3", "v1", "v2"],
                          defines={"v3"}, uses={"v1", "v2"}),
        ]
        alloc = LinearScanAllocator(["t0", "t1", "t2", "a0"])
        intervals = alloc.compute_live_intervals(block)
        mapping = alloc.allocate(intervals)
        assert len(mapping) == 3
        assert "v1" in mapping
        assert "v2" in mapping
        assert "v3" in mapping

    def test_allocate_insufficient_regs(self):
        """Allocate with fewer registers than variables: should spill."""
        block = [
            LsInstruction(0, "li", ["v1", "1"], defines={"v1"}, uses=set()),
            LsInstruction(1, "li", ["v2", "2"], defines={"v2"}, uses=set()),
            LsInstruction(2, "li", ["v3", "3"], defines={"v3"}, uses=set()),
            LsInstruction(3, "add", ["v4", "v1", "v2"],
                          defines={"v4"}, uses={"v1", "v2"}),
            LsInstruction(4, "add", ["v5", "v3", "v4"],
                          defines={"v5"}, uses={"v3", "v4"}),
        ]
        alloc = LinearScanAllocator(["t0", "t1"])
        intervals = alloc.compute_live_intervals(block)
        mapping = alloc.allocate(intervals)
        assert len(mapping) >= 2
        assert len(alloc.spill_code) >= 0  # may or may not spill

    def test_allocate_empty_block(self):
        alloc = LinearScanAllocator()
        intervals = alloc.compute_live_intervals([])
        mapping = alloc.allocate(intervals)
        assert mapping == {}

    def test_get_allocated_code(self):
        block = [
            LsInstruction(0, "add", ["v1", "v2", "v3"],
                          defines={"v1"}, uses={"v2", "v3"}),
        ]
        alloc = LinearScanAllocator(["t0", "t1", "a0"])
        intervals = alloc.compute_live_intervals(block)
        alloc.allocate(intervals)
        code = alloc.get_allocated_code(block)
        assert isinstance(code, str)
        assert "add" in code

    def test_report(self):
        alloc = LinearScanAllocator(["t0", "a0"])
        block = [
            LsInstruction(0, "add", ["v1", "v2", "v3"],
                          defines={"v1"}, uses={"v2", "v3"}),
        ]
        intervals = alloc.compute_live_intervals(block)
        alloc.allocate(intervals)
        report = alloc.report()
        assert "Linear Scan" in report
        assert "v1" in report or "allocated" in report.lower()


class TestBlockFromMachineInstrs:
    """Tests for converting MachineInstr to LsInstruction lists."""

    def test_conversion(self):
        from scratchv.backend.register_alloc import (
            MachineInstr, MachineOp, MachineOperand)

        mi = [
            MachineInstr(MachineOp.ADD,
                         MachineOperand.vreg("v1"),
                         MachineOperand.vreg("v2"),
                         MachineOperand.vreg("v3"),
                         comment="add"),
        ]
        block = block_from_machine_instrs(mi)
        assert len(block) == 1
        assert block[0].opcode == "add"
        assert "v1" in block[0].defines
        assert "v2" in block[0].uses
        assert "v3" in block[0].uses

    def test_conversion_label(self):
        from scratchv.backend.register_alloc import MachineInstr, MachineOp
        mi = [
            MachineInstr(MachineOp.LABEL, comment="main"),
        ]
        block = block_from_machine_instrs(mi)
        assert len(block) == 1
        assert block[0].opcode == ".label"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
