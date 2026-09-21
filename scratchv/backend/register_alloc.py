"""Register allocation for RISC-V.

Implements two strategies:
1. Naive: map every virtual register to a stack slot (load/store).
2. Greedy: simple local greedy allocator using callee-saved regs first.

Machine instruction types (MachineOp, MachineOperand, MachineInstr) are
defined in ``scratchv.backend.machine_types`` and re-exported here for
backward compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from scratchv.backend.machine_semantics import (
    get_machine_semantics,
    virtual_register_defs_uses,
)

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

if TYPE_CHECKING:
    from scratchv.analysis.cfg import ControlFlowGraph
    from scratchv.analysis.liveness import LivenessResult

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
        self._remaining_uses: dict[str, int] = {}
        self.cfg: Optional["ControlFlowGraph"] = None
        self.liveness: Optional["LivenessResult"] = None
        self._current_live_after: frozenset[str] = frozenset()

    @property
    def spill_slot_count(self) -> int:
        """Return the number of unique stack slots reserved for spills."""

        return len(self._spill_slots)

    def run(self) -> list[MachineInstr]:
        if self.mode == "naive":
            return self._allocate_naive()
        else:
            return self._allocate_greedy()

    def _allocate_naive(self) -> list[MachineInstr]:
        """Spill every virtual register to the stack."""
        self._output = []
        self._spill_slots.clear()
        self._next_spill = 0
        for instr in self.instructions:
            if instr.op == MachineOp.LABEL:
                self._emit(instr)
                continue

            semantics = get_machine_semantics(instr.op)
            operands = [instr.dst, instr.src1, instr.src2]
            resolved = list(operands)
            explicit_regs = {
                str(operand.value)
                for operand in operands
                if operand is not None
                and operand.kind == "reg"
                and operand.value in ALL_REGS
            }
            scratch_regs = [
                reg for reg in ALL_REGS if reg not in explicit_regs
            ]
            scratch_by_vreg: dict[str, str] = {}

            def scratch_for(vreg: str) -> str:
                if vreg in scratch_by_vreg:
                    return scratch_by_vreg[vreg]
                used = set(scratch_by_vreg.values())
                try:
                    reg = next(reg for reg in scratch_regs if reg not in used)
                except StopIteration as exc:
                    raise RuntimeError(
                        "naive regalloc: instruction needs more distinct "
                        "scratch registers than are available"
                    ) from exc
                scratch_by_vreg[vreg] = reg
                return reg

            # Materialize every distinct virtual source into its own scratch
            # register.  Reusing one fixed temporary silently overwrites the
            # first source of binary instructions.
            for position in semantics.uses:
                operand = operands[position]
                if operand is None or operand.kind != "vreg":
                    continue
                vreg = str(operand.value)
                reg = scratch_for(vreg)
                slot = self._get_spill_slot(vreg)
                self._emit(MachineInstr(
                    MachineOp.LW,
                    MachineOperand.reg(reg),
                    MachineOperand.reg(f"{slot}({STACK_BASE})"),
                    comment=f"reload {vreg} [regalloc:reload]",
                ))
                resolved[position] = MachineOperand.reg(reg)

            # A destination never needs its old stack value unless the opcode
            # also marks that same virtual register as a use.
            for position in semantics.defs:
                operand = operands[position]
                if operand is None or operand.kind != "vreg":
                    continue
                vreg = str(operand.value)
                resolved[position] = MachineOperand.reg(scratch_for(vreg))

            self._emit(MachineInstr(
                instr.op,
                resolved[0],
                resolved[1],
                resolved[2],
                instr.comment,
                instr.target,
            ))

            stored: set[str] = set()
            for position in semantics.defs:
                operand = operands[position]
                if operand is None or operand.kind != "vreg":
                    continue
                vreg = str(operand.value)
                if vreg in stored:
                    continue
                stored.add(vreg)
                reg = scratch_by_vreg[vreg]
                slot = self._get_spill_slot(vreg)
                self._emit(MachineInstr(
                    MachineOp.SW,
                    MachineOperand.reg(reg),
                    MachineOperand.reg(f"{slot}({STACK_BASE})"),
                    comment=f"spill {vreg} [regalloc:spill]",
                ))

        return self._output

    def _allocate_greedy(self) -> list[MachineInstr]:
        """Allocate locally using unified CFG liveness at block barriers."""
        # Keep analysis imports local: analysis adapters depend on backend
        # machine types, while backend.__init__ re-exports this allocator.
        from scratchv.analysis.adapters import MachineCFGAdapter
        from scratchv.analysis.cfg import build_cfg
        from scratchv.analysis.liveness import analyze_liveness
        from scratchv.analysis.usedef import MachineUseDefProvider

        self._output = []
        self._vreg_map.clear()
        self._spill_slots.clear()
        self._next_spill = 0
        self._reg_pool = {r: None for r in ALL_REGS}
        self._remaining_uses = {}
        for instr in self.instructions:
            _, uses = virtual_register_defs_uses(instr)
            for vreg in uses:
                self._remaining_uses[vreg] = self._remaining_uses.get(vreg, 0) + 1

        self.cfg = build_cfg(MachineCFGAdapter("greedy", self.instructions))
        self.liveness = analyze_liveness(self.cfg, MachineUseDefProvider())
        live_after_by_instr: dict[int, frozenset[str]] = {}
        for node in self.cfg.nodes.values():
            if isinstance(node.instructions, int):
                continue
            for instr_id, block_instr in zip(
                node.instruction_ids, node.instructions
            ):
                live_after_by_instr[id(block_instr)] = self.liveness.live_after[
                    instr_id
                ]

        fallthrough_boundaries = {
            index - 1
            for index, instr in enumerate(self.instructions)
            if index > 0 and instr.op == MachineOp.LABEL
        }

        for index, instr in enumerate(self.instructions):
            if instr.op == MachineOp.LABEL:
                self._vreg_map.clear()
                self._reg_pool = {r: None for r in ALL_REGS}
                self._emit(instr)
                continue

            semantics = get_machine_semantics(instr.op)
            live_after = live_after_by_instr.get(id(instr), frozenset())
            self._current_live_after = live_after
            operands = [instr.dst, instr.src1, instr.src2]
            resolved = list(operands)
            explicit_uses = {
                str(operands[position].value)
                for position in semantics.uses
                if operands[position] is not None
                and operands[position].kind == "reg"
                and operands[position].value in ALL_REGS
            }
            explicit_defs = {
                str(operands[position].value)
                for position in semantics.defs
                if operands[position] is not None
                and operands[position].kind == "reg"
                and operands[position].value in ALL_REGS
            }
            for phys_reg in explicit_defs:
                owner = self._reg_pool[phys_reg]
                if owner is not None and owner in live_after:
                    self._emit_spill(owner, phys_reg)
            reserved: set[str] = set(explicit_uses)

            # Resolve every use first so a destination can safely alias a
            # source whose last use is this instruction.
            for position in semantics.uses:
                operand = operands[position]
                resolved[position] = self._resolve_src(operand, reserved)
                resolved_operand = resolved[position]
                if resolved_operand is not None and resolved_operand.kind == "reg" \
                        and resolved_operand.value in ALL_REGS:
                    reserved.add(str(resolved_operand.value))

            reusable = {
                self._vreg_map[vreg]
                for vreg in (
                    str(operands[position].value)
                    for position in semantics.uses
                    if operands[position] is not None
                    and operands[position].kind == "vreg"
                )
                if vreg not in live_after
                and vreg in self._vreg_map
            }

            for position in semantics.defs:
                operand = operands[position]
                if position in semantics.uses:
                    continue
                resolved[position] = self._resolve_dst(
                    operand, reserved - reusable
                )

            allocated = MachineInstr(
                instr.op,
                resolved[0],
                resolved[1],
                resolved[2],
                instr.comment,
                instr.target,
            )

            # A call is not a CFG terminator, but it invalidates caller-saved
            # mappings.  At calls and CFG boundaries, only values reported
            # live by the unified analysis need a canonical stack copy.
            if semantics.is_call:
                self._flush_clobbered(semantics.clobbers, live_after)
            elif semantics.is_terminator:
                defines, _ = virtual_register_defs_uses(instr)
                if defines:
                    raise RuntimeError(
                        "greedy regalloc does not support a control-flow "
                        "terminator defining a virtual register"
                    )
                self._flush_regs(live_after)
            self._emit(allocated)

            # A fixed physical destination overwrites any virtual value that
            # happened to occupy that register.
            for phys_reg in explicit_defs:
                owner = self._reg_pool[phys_reg]
                if owner is not None:
                    self._vreg_map.pop(owner, None)
                self._reg_pool[phys_reg] = None

            _, uses = virtual_register_defs_uses(instr)
            for vreg in uses:
                self._remaining_uses[vreg] -= 1
            for vreg in list(self._vreg_map):
                if vreg not in live_after:
                    self._release_vreg(vreg)

            # A label may be reached either by fallthrough or by another CFG
            # edge.  Canonicalize the fallthrough predecessor before the label
            # so every incoming edge observes initialized spill slots.
            if index in fallthrough_boundaries and not semantics.is_terminator:
                self._flush_regs(live_after)

        self._current_live_after = frozenset()
        return self._output

    def _resolve_src(
        self, op: MachineOperand | None, avoid_regs: set[str] | None = None,
    ) -> MachineOperand | None:
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
            v = op.value
            assert isinstance(v, str)
            reg = self._assign_reg(
                v, reload=v in self._spill_slots, avoid_regs=avoid_regs
            )
            return MachineOperand.reg(reg)
        return op

    def _resolve_dst(
        self, op: MachineOperand | None, avoid_regs: set[str] | None = None,
    ) -> MachineOperand | None:
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
            reg = self._assign_reg(v, reload=False, avoid_regs=avoid_regs)
            return MachineOperand.reg(reg)
        return op

    def _assign_reg(
        self,
        vreg_name: str,
        *,
        reload: bool = False,
        avoid_regs: set[str] | None = None,
    ) -> str:
        """Assign a physical register to a virtual register."""
        if vreg_name in self._vreg_map:
            return self._vreg_map[vreg_name]

        avoid = avoid_regs or set()
        for phys_reg, occupant in self._reg_pool.items():
            if occupant is None and phys_reg not in avoid:
                self._reg_pool[phys_reg] = vreg_name
                self._vreg_map[vreg_name] = phys_reg
                if reload:
                    self._emit_reload(vreg_name, phys_reg)
                return phys_reg

        # Pick an unprotected victim whose next use is farthest away.
        candidates = [reg for reg in ALL_REGS if reg not in avoid]
        if not candidates:
            raise RuntimeError(
                "greedy regalloc: instruction needs more simultaneous "
                f"register operands than the {len(ALL_REGS)}-register bank"
            )

        def remaining(reg: str) -> int:
            owner = self._reg_pool[reg]
            return self._remaining_uses.get(owner or "", 0)

        lru_reg = min(candidates, key=remaining)
        lru_vreg = self._reg_pool[lru_reg]
        if lru_vreg:
            if lru_vreg in self._current_live_after:
                self._emit_spill(lru_vreg, lru_reg)
            self._vreg_map.pop(lru_vreg, None)
        self._reg_pool[lru_reg] = vreg_name
        self._vreg_map[vreg_name] = lru_reg
        if reload:
            self._emit_reload(vreg_name, lru_reg)
        return lru_reg

    def _flush_regs(self, live_values: frozenset[str]) -> None:
        """Spill all registers at basic block boundaries."""
        for phys_reg, vreg_name in list(self._reg_pool.items()):
            if vreg_name is not None:
                if vreg_name in live_values:
                    self._emit_spill(vreg_name, phys_reg)
                self._reg_pool[phys_reg] = None
        self._vreg_map.clear()

    def _flush_clobbered(
        self,
        clobbers: frozenset[str],
        live_values: frozenset[str],
    ) -> None:
        """Canonicalize values held in ABI-clobbered registers before call."""
        for phys_reg in clobbers:
            if phys_reg not in self._reg_pool:
                continue
            vreg_name = self._reg_pool[phys_reg]
            if vreg_name is None:
                continue
            if vreg_name in live_values:
                self._emit_spill(vreg_name, phys_reg)
            self._vreg_map.pop(vreg_name, None)
            self._reg_pool[phys_reg] = None

    def _release_vreg(self, vreg_name: str) -> None:
        reg = self._vreg_map.pop(vreg_name, None)
        if reg is not None and self._reg_pool.get(reg) == vreg_name:
            self._reg_pool[reg] = None

    def _emit_spill(self, vreg_name: str, phys_reg: str) -> None:
        slot = self._get_spill_slot(vreg_name)
        mem = f"{slot}({STACK_BASE})"
        self._emit(MachineInstr(
            MachineOp.SW,
            MachineOperand.reg(phys_reg),
            MachineOperand.reg(mem),
            comment=f"spill {vreg_name} [regalloc:spill]",
        ))

    def _emit_reload(self, vreg_name: str, phys_reg: str) -> None:
        slot = self._get_spill_slot(vreg_name)
        mem = f"{slot}({STACK_BASE})"
        self._emit(MachineInstr(
            MachineOp.LW,
            MachineOperand.reg(phys_reg),
            MachineOperand.reg(mem),
            comment=f"reload {vreg_name} [regalloc:reload]",
        ))

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
        mem = f"{slot}({STACK_BASE})"
        self._emit(MachineInstr(MachineOp.LW, temp,
                                MachineOperand.reg(mem),
                                comment=f"load {op.value}"))
        return temp

    def _emit(self, instr: MachineInstr) -> None:
        self._output.append(instr)
