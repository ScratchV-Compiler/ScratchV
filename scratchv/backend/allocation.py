"""Typed register allocation results and byte-addressed spill storage."""

from __future__ import annotations

from dataclasses import dataclass, field

from scratchv.backend.machine_semantics import validate_register_classes
from scratchv.backend.machine_types import MachineInstr, MachineOp, RegisterClass


@dataclass(frozen=True)
class RegisterType:
    reg_class: RegisterClass = RegisterClass.GPR
    bit_width: int = 32

    def __post_init__(self) -> None:
        if self.bit_width not in {32, 64} or (self.reg_class is RegisterClass.GPR and self.bit_width != 32):
            raise ValueError("unsupported register class/width combination")

    @property
    def load_op(self) -> MachineOp:
        if self.reg_class is RegisterClass.GPR:
            return MachineOp.LW
        return MachineOp.FLW if self.bit_width == 32 else MachineOp.FLD

    @property
    def store_op(self) -> MachineOp:
        if self.reg_class is RegisterClass.GPR:
            return MachineOp.SW
        return MachineOp.FSW if self.bit_width == 32 else MachineOp.FSD


@dataclass(frozen=True)
class SpillSlot:
    # Negative temporary offset, rebased by the ABI finalizer.
    offset: int
    size: int
    alignment: int
    reg_class: RegisterClass


@dataclass
class AllocationResult:
    instructions: list[MachineInstr]
    register_map: dict[str, str]
    spill_slots: dict[str, SpillSlot]
    spill_bytes: int
    local_slots: dict[str, SpillSlot] = field(default_factory=dict)
    local_bytes: int = 0


class SpillLayout:
    def __init__(self) -> None:
        self.slots: dict[str, SpillSlot] = {}
        self.size = 0

    def reserve(self, name: str, value_type: RegisterType) -> SpillSlot:
        existing = self.slots.get(name)
        if existing is not None and (existing.size != value_type.bit_width // 8
                                     or existing.reg_class is not value_type.reg_class):
            raise ValueError(f"conflicting spill slot metadata for {name}")
        if name not in self.slots:
            size = value_type.bit_width // 8
            self.size = (self.size + size + size - 1) // size * size
            self.slots[name] = SpillSlot(
                -self.size, size, size, value_type.reg_class,
            )
        return self.slots[name]


def collect_register_types(instructions: list[MachineInstr]) -> dict[str, RegisterType]:
    validate_register_classes(instructions)
    result: dict[str, RegisterType] = {}
    for instr in instructions:
        widths = {"fcvt.s.d": (32, 64, None), "fcvt.d.s": (64, 32, None)}.get(instr.op.value)
        if widths is None:
            width = (64 if instr.op.value.endswith(".d") or instr.op in {MachineOp.FLD, MachineOp.FSD}
                     else 32)
            widths = (width, width, width)
        for position, operand in enumerate((instr.dst, instr.src1, instr.src2)):
            if (operand is not None and operand.reg_class is RegisterClass.FPR
                    and operand.bit_width is not None and operand.bit_width != widths[position]):
                raise ValueError(f"{instr.op.value} operand {position} has incompatible bit width")
        for operand in (instr.dst, instr.src1, instr.src2):
            if operand is None or operand.kind != "vreg":
                continue
            if operand.reg_class is RegisterClass.FPR and operand.bit_width is None:
                raise ValueError(f"FPR virtual register {operand.value} is missing bit width")
            value_type = RegisterType(operand.reg_class, operand.bit_width or 32)
            previous = result.setdefault(str(operand.value), value_type)
            if previous != value_type:
                raise ValueError(f"virtual register {operand.value} has conflicting width or class")
    return result


def validate_allocated(instructions: list[MachineInstr]) -> None:
    validate_register_classes(instructions)
    for instr in instructions:
        for operand in (instr.dst, instr.src1, instr.src2):
            if operand is not None and operand.kind == "vreg":
                raise ValueError(f"unallocated virtual register: {operand.value}")
