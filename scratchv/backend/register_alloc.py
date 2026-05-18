"""Register allocation for RISC-V.

Implements two strategies:
1. Naive: map every virtual register to a stack slot (load/store).
2. Greedy: simple local greedy allocator using callee-saved regs first.

Machine instruction types (MachineOp, MachineOperand, MachineInstr) are
defined in ``scratchv.backend.machine_types`` and re-exported here for
backward compatibility.
"""

from __future__ import annotations

from typing import Optional

from scratchv.backend.machine_types import (  # noqa: F401 — re-export
    ALL_REGS,
    ARG_REGS,
    CALLEE_SAVED,
    MachineInstr,
    MachineOp,
    MachineOperand,
    STACK_BASE,
    TEMP_REGS,
    ZERO_REG,
)

# Re-export for backward compatibility — new code should import directly
# from scratchv.backend.machine_types.
__all__ = [
    "MachineOp",
    "MachineOperand",
    "MachineInstr",
    "CALLEE_SAVED",
    "TEMP_REGS",
    "ARG_REGS",
    "ALL_REGS",
    "STACK_BASE",
    "ZERO_REG",
    "RegisterAllocator",
]


class RegisterAllocator:
    """Register allocator that maps vregs to physical RISC-V registers.

    Mode 'naive': spill everything to stack, for maximum correctness.
    Mode 'greedy': simple local allocator using temp registers first.
    """

    def __init__(self, instructions: list[MachineInstr], mode: str = "greedy"):
        self.instructions = instructions
        self.mode = mode
        self._vreg_map: dict[str, str] = {}  # vreg_name -> phys_reg
        self._spill_slots: dict[str, int] = {}  # vreg_name -> stack offset
        self._next_spill = 0
        # Track which physical registers are currently allocated
        self._reg_pool: dict[str, Optional[str]] = {r: None for r in ALL_REGS}
        self._output: list[MachineInstr] = []

    def run(self) -> list[MachineInstr]:
        if self.mode == "naive":
            return self._allocate_naive()
        else:
            return self._allocate_greedy()

    def _allocate_naive(self) -> list[MachineInstr]:
        """Spill every virtual register to the stack."""
        self._output = []
        for instr in self.instructions:
            if instr.op == MachineOp.LABEL:
                self._emit(instr)
                continue

            # Before: spill src operands that are vregs
            src1 = self._resolve_src(instr.src1)
            src2 = self._resolve_src(instr.src2)
            dst = self._resolve_dst(instr.dst)

            if instr.dst and instr.dst.kind == "vreg":
                dst = self._spill_operand(instr.dst)

            self._emit(MachineInstr(instr.op, dst, src1, src2, instr.comment))

            # After: store dst back to stack if it's a vreg
            if instr.dst and instr.dst.kind == "vreg":
                v = instr.dst.value
                assert isinstance(v, str)
                slot = self._get_spill_slot(v)
                mem = f"{STACK_BASE}({-slot})" if slot > 0 else "0(sp)"
                self._emit(MachineInstr(
                    MachineOp.SW,
                    MachineOperand.reg(mem),
                    dst if dst else MachineOperand.reg("zero"),
                    comment=f"spill {instr.dst.value}",
                ))

        return self._output

    def _allocate_greedy(self) -> list[MachineInstr]:
        """Simple greedy allocator: assign physical registers to vregs."""
        self._output = []
        self._vreg_map.clear()
        self._reg_pool = {r: None for r in ALL_REGS}

        for instr in self.instructions:
            if instr.op == MachineOp.LABEL:
                self._flush_regs()
                self._emit(instr)
                continue

            src1 = self._resolve_src(instr.src1)
            src2 = self._resolve_src(instr.src2)
            dst = self._resolve_dst(instr.dst)

            # Allocate destination register
            if instr.dst and instr.dst.kind == "vreg" \
                    and instr.dst.value not in self._vreg_map:
                v2 = instr.dst.value
                assert isinstance(v2, str)
                reg_name = self._assign_reg(v2)
                dst = MachineOperand.reg(reg_name)
            elif instr.dst and instr.dst.kind == "vreg":
                v3 = instr.dst.value
                assert isinstance(v3, str)
                dst = MachineOperand.reg(self._vreg_map[v3])

            self._emit(MachineInstr(instr.op, dst, src1, src2, instr.comment))

        return self._output

    def _resolve_src(self, op: MachineOperand | None) -> MachineOperand | None:
        if op is None:
            return None
        if op.kind == "imm":
            return op
        if op.kind == "reg":
            return op
        if op.kind == "vreg":
            if op.value in self._vreg_map:
                r = self._vreg_map[op.value]  # type: ignore[index]
                return MachineOperand.reg(r)
            # Assign a register
            v = op.value
            assert isinstance(v, str)
            reg = self._assign_reg(v)
            return MachineOperand.reg(reg)
        return op

    def _resolve_dst(self, op: MachineOperand | None) -> MachineOperand | None:
        if op is None:
            return None
        if op.kind == "reg":
            return op
        if op.kind == "vreg":
            if op.value in self._vreg_map:
                r2 = self._vreg_map[op.value]  # type: ignore[index]
                return MachineOperand.reg(r2)
            v = op.value
            assert isinstance(v, str)
            reg = self._assign_reg(v)
            return MachineOperand.reg(reg)
        return op

    def _assign_reg(self, vreg_name: str) -> str:
        """Assign a physical register to a virtual register."""
        if vreg_name in self._vreg_map:
            return self._vreg_map[vreg_name]

        # Find a free register
        for phys_reg, occupant in self._reg_pool.items():
            if occupant is None:
                self._reg_pool[phys_reg] = vreg_name
                self._vreg_map[vreg_name] = phys_reg
                return phys_reg

        # No free register: spill the one used longest ago (simple LRU)
        lru_reg = TEMP_REGS[0]
        lru_vreg = self._reg_pool[lru_reg]
        if lru_vreg:
            # Spill: store to stack
            slot = self._get_spill_slot(lru_vreg)
            mem = f"{STACK_BASE}({-slot})"
            self._emit(MachineInstr(
                MachineOp.SW, MachineOperand.reg(mem),
                MachineOperand.reg(lru_reg),
                comment=f"spill {lru_vreg}",
            ))
        self._reg_pool[lru_reg] = vreg_name
        self._vreg_map[vreg_name] = lru_reg
        return lru_reg

    def _flush_regs(self) -> None:
        """Spill all registers at basic block boundaries."""
        for phys_reg, vreg_name in list(self._reg_pool.items()):
            if vreg_name is not None:
                slot = self._get_spill_slot(vreg_name)  # type: ignore[arg-type]
                mem = f"{STACK_BASE}({-slot})"
                self._emit(MachineInstr(
                    MachineOp.SW, MachineOperand.reg(mem),
                    MachineOperand.reg(phys_reg),
                    comment=f"spill {vreg_name}",
                ))
                self._reg_pool[phys_reg] = None
        self._vreg_map.clear()

    def _get_spill_slot(self, vreg_name: str) -> int:
        if vreg_name not in self._spill_slots:
            self._next_spill -= 4
            self._spill_slots[vreg_name] = self._next_spill
        return self._spill_slots[vreg_name]

    def _spill_operand(self, op: MachineOperand) -> MachineOperand:
        """Return a temp register holding the spilled value."""
        v = op.value
        assert isinstance(v, str)
        slot = self._get_spill_slot(v)
        temp = MachineOperand.reg("t0")
        mem = f"{STACK_BASE}({-slot})" if slot != 0 else "0(sp)"
        self._emit(MachineInstr(MachineOp.LW, temp,
                                MachineOperand.reg(mem),
                                comment=f"load {op.value}"))
        return temp

    def _emit(self, instr: MachineInstr) -> None:
        self._output.append(instr)
