"""Scalar lowering of phase-1 vector ops (Topic 29).

``VectorScalarExpander`` converts each vector IR instruction into a
sequence of plain RV32IM machine instructions, one per lane, so that the
existing register allocator, assembly emitter, encoder and RV32IM
emulator can all execute vectorized programs without modification.

Two encoder-facing constraints shape the emitted code:

* Scratch lane addresses are materialized in a fixed physical register
  (``a1``) and referenced with an explicit ``(0)`` offset, because the
  RV32IM encoder only accepts ``lw rd, rs1(offset)`` /
  ``sw rs1(offset), rs2`` memory forms; a bare ``lw rd, rs1`` operand is
  silently encoded as ``rs1 = x0``.  ``a1`` is outside the allocatable
  register sets, so no allocated vreg can collide with it.
* ``VRELU`` uses the ``zero`` register (not the immediate ``0``) as the
  ``max`` pseudo-instruction's second operand, so the encoder's expansion
  does not need to borrow a temporary general-purpose register.
"""

from __future__ import annotations

from scratchv.backend.machine_types import (
    MachineInstr,
    MachineOp,
    MachineOperand,
)
from scratchv.ir.types import Instruction, OpCode, Value


class VectorLoweringError(ValueError):
    """Raised when a vector op cannot be lowered to scalar machine code."""


_BINARY_OPS: dict[OpCode, MachineOp] = {
    OpCode.VADD: MachineOp.ADD,
    OpCode.VSUB: MachineOp.SUB,
    OpCode.VMUL: MachineOp.MUL,
    OpCode.VDIV: MachineOp.DIV,
}

# Fixed physical scratch register for lane addresses.  Must not be a
# member of ``machine_types.ALL_REGS`` (t0-t6, s0-s11).
_ADDR_SCRATCH = "a1"


class VectorScalarExpander:
    """Expand vector IR instructions into per-lane scalar machine code."""

    def __init__(self, *, default_width: int = 4) -> None:
        self._default_width = default_width
        self._lanes: dict[str, list[MachineOperand]] = {}
        self._counter = 0

    # ── Lifecycle ───────────────────────────────────────────────────────

    def begin_function(self, func_name: str) -> None:
        """Reset per-function lowering state."""
        self._lanes.clear()
        self._counter = 0

    # ── Public API ──────────────────────────────────────────────────────

    def expand(self, instr: Instruction) -> list[MachineInstr]:
        """Lower one vector instruction to a list of machine instructions."""
        op = instr.opcode
        if op is OpCode.VLOAD:
            return self._expand_load(instr)
        if op is OpCode.VSTORE:
            return self._expand_store(instr)
        if op is OpCode.VBCAST:
            return self._expand_bcast(instr)
        if op in _BINARY_OPS:
            return self._expand_binary(instr)
        if op is OpCode.VRELU:
            return self._expand_unary(instr)
        raise VectorLoweringError(
            f"unsupported vector op: '{op.value}'")

    # ── Shape helpers ───────────────────────────────────────────────────

    def _width(self, instr: Instruction) -> int:
        raw = instr.attrs.get("width", self._default_width)
        try:
            return int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return self._default_width

    def _operand_reg(self, value: Value,
                     out: list[MachineInstr]) -> MachineOperand:
        """A register operand for a scalar value; constants become LI."""
        if value.is_constant and value.const_value is not None:
            return self._materialize_const(int(value.const_value), out)
        return MachineOperand.vreg(value.name)

    def _materialize_const(self, raw: int,
                           out: list[MachineInstr]) -> MachineOperand:
        self._counter += 1
        tmp = MachineOperand.vreg(f"vcst_{self._counter}")
        out.append(MachineInstr(
            MachineOp.LI, tmp, MachineOperand.immediate(raw),
            comment=f"vector const {raw}"))
        return tmp

    def _lanes_of(self, value: Value, width: int,
                  out: list[MachineInstr]) -> list[MachineOperand]:
        """Return the lane operands bound to *value*."""
        bound = self._lanes.get(value.name)
        if bound is not None:
            if len(bound) != width:
                raise VectorLoweringError(
                    f"vector value '{value.name}' has {len(bound)} lane(s), "
                    f"expected {width}")
            return bound
        if value.is_constant and value.const_value is not None:
            tmp = self._materialize_const(int(value.const_value), out)
            return [tmp] * width
        if value.shape:
            raise VectorLoweringError(
                f"vector value '{value.name}' has no lane binding")
        return [MachineOperand.vreg(value.name)] * width

    # ── Per-op expansion ────────────────────────────────────────────────

    def _lane_addr(self, base: MachineOperand, k: int, out: list[MachineInstr],
                   name_hint: str) -> MachineOperand:
        """Materialize lane *k*'s address (``base + 4k``) in a scratch reg."""
        out.append(MachineInstr(
            MachineOp.ADDI,
            MachineOperand.reg(_ADDR_SCRATCH),
            base,
            MachineOperand.immediate(4 * k),
            comment=f"{name_hint} lane {k} addr"))
        return MachineOperand.reg(f"{_ADDR_SCRATCH}(0)")

    def _expand_load(self, instr: Instruction) -> list[MachineInstr]:
        out: list[MachineInstr] = []
        width = self._width(instr)
        dest = instr.dest
        if dest is None:
            raise VectorLoweringError("vload without destination")
        addr = self._operand_reg(instr.operands[0], out)
        lanes: list[MachineOperand] = []
        for k in range(width):
            lane = MachineOperand.vreg(f"{dest.name}__lane{k}")
            a = self._lane_addr(addr, k, out, dest.name)
            out.append(MachineInstr(MachineOp.LW, lane, a))
            lanes.append(lane)
        self._lanes[dest.name] = lanes
        return out

    def _expand_store(self, instr: Instruction) -> list[MachineInstr]:
        out: list[MachineInstr] = []
        width = self._width(instr)
        addr = self._operand_reg(instr.operands[0], out)
        vec = instr.operands[1]
        lanes = self._lanes_of(vec, width, out)
        for k in range(width):
            a = self._lane_addr(addr, k, out, vec.name)
            out.append(MachineInstr(MachineOp.SW, a, lanes[k]))
        return out

    def _expand_bcast(self, instr: Instruction) -> list[MachineInstr]:
        out: list[MachineInstr] = []
        width = self._width(instr)
        dest = instr.dest
        if dest is None:
            raise VectorLoweringError("vbcast without destination")
        src = self._operand_reg(instr.operands[0], out)
        self._lanes[dest.name] = [src] * width
        return out

    def _expand_binary(self, instr: Instruction) -> list[MachineInstr]:
        out: list[MachineInstr] = []
        width = self._width(instr)
        dest = instr.dest
        if dest is None:
            raise VectorLoweringError(
                f"{instr.opcode.value} without destination")
        va = self._lanes_of(instr.operands[0], width, out)
        vb = self._lanes_of(instr.operands[1], width, out)
        mop = _BINARY_OPS[instr.opcode]
        lanes: list[MachineOperand] = []
        for k in range(width):
            lane = MachineOperand.vreg(f"{dest.name}__lane{k}")
            out.append(MachineInstr(mop, lane, va[k], vb[k]))
            lanes.append(lane)
        self._lanes[dest.name] = lanes
        return out

    def _expand_unary(self, instr: Instruction) -> list[MachineInstr]:
        out: list[MachineInstr] = []
        width = self._width(instr)
        dest = instr.dest
        if dest is None:
            raise VectorLoweringError("vrelu without destination")
        va = self._lanes_of(instr.operands[0], width, out)
        zero = MachineOperand.reg("zero")
        lanes: list[MachineOperand] = []
        for k in range(width):
            lane = MachineOperand.vreg(f"{dest.name}__lane{k}")
            out.append(MachineInstr(MachineOp.MAX, lane, va[k], zero))
            lanes.append(lane)
        self._lanes[dest.name] = lanes
        return out
