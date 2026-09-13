"""Regression tests for the register-allocation changes in PR #37.

The implementation under test is intentionally kept in the PR-specific
module.  The module is not present on the pre-PR main branch, so this file is
skipped there and becomes active as soon as the PR is checked out by CI.
"""

from __future__ import annotations

import re

import pytest


regalloc = pytest.importorskip(
    "scratchv.backend.regalloc_linear_v1_5",
    reason="PR #37 register allocator is not present on this branch",
)


def _pressure_block():
    """Build a legal two-register block that requires spill/reload."""
    instruction = regalloc.LsInstruction
    return [
        instruction(0, "li", ["v0", "1"], defines={"v0"}),
        instruction(1, "li", ["v1", "2"], defines={"v1"}),
        instruction(2, "li", ["v2", "3"], defines={"v2"}),
        instruction(3, "add", ["v3", "v0", "v1"],
                    defines={"v3"}, uses={"v0", "v1"}),
        instruction(4, "add", ["v4", "v2", "v3"],
                    defines={"v4"}, uses={"v2", "v3"}),
        instruction(5, "add", ["v5", "v4", "v0"],
                    defines={"v5"}, uses={"v4", "v0"}),
    ]


def test_pressure_spill_reload_does_not_leak_virtual_registers():
    """Eviction/reload must leave only physical registers in assembly."""
    allocator = regalloc.LinearScanAllocator(phys_regs=["t0", "t1"])
    assembly = allocator.emit(_pressure_block())

    assert "  sw " in assembly
    assert "  lw " in assembly
    assert "SPILL_" not in assembly

    # Ignore comments, which mention the original vreg for diagnostics.
    operands = "\n".join(
        line.split("#", 1)[0] for line in assembly.splitlines()
    )
    assert not re.search(r"(?<![A-Za-z0-9_])v[0-9]+(?![A-Za-z0-9_])", operands)


def test_spilled_vreg_redefinition_is_written_back_before_reload():
    """A pure redefinition must update its spill slot for later uses."""
    instruction = regalloc.LsInstruction
    block = [
        instruction(0, "li", ["v0", "1"], defines={"v0"}),
        instruction(1, "li", ["v1", "2"], defines={"v1"}),
        instruction(2, "li", ["v2", "3"], defines={"v2"}),
        instruction(3, "add", ["v3", "v0", "v1"],
                    defines={"v3"}, uses={"v0", "v1"}),
        # v0 is redefined without reading its old value.
        instruction(4, "li", ["v0", "9"], defines={"v0"}),
        instruction(5, "add", ["v4", "v0", "v2"],
                    defines={"v4"}, uses={"v0", "v2"}),
    ]
    allocator = regalloc.LinearScanAllocator(phys_regs=["t0", "t1"])
    assembly = allocator.emit(block)
    lines = assembly.splitlines()

    redefine = next(i for i, line in enumerate(lines) if ", 9" in line)
    writeback = next(
        i for i, line in enumerate(lines[redefine + 1:], redefine + 1)
        if "sw " in line and "store redefined v0" in line
    )
    assert writeback > redefine
    assert any("reload v0" in line for line in lines[writeback + 1:])


def test_machine_operand_round_trip_uses_exact_register_names():
    """Names such as ``a_temp`` remain vregs while ``a0`` is a register."""
    instruction = regalloc.LsInstruction
    block = [instruction(
        0,
        "add",
        ["a_temp", "a0", "42"],
        defines={"a_temp"},
        uses={"a0"},
    )]

    converted = regalloc.machine_instrs_from_block(block)[0]

    assert converted.dst.kind == "vreg"
    assert converted.dst.value == "a_temp"
    assert converted.src1.kind == "reg"
    assert converted.src1.value == "a0"
    assert converted.src2.kind == "imm"
    assert converted.src2.value == 42
