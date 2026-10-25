"""Machine instruction types for the RISC-V backend.

This module defines the core types shared by all backend passes:
machine opcodes, operands, and instructions.  Extracted from
``register_alloc.py`` so that passes can import these types without
pulling in register-allocation logic.

Usage::

    from scratchv.backend.machine_types import (
        MachineOp, MachineOperand, MachineInstr,
        CALLEE_SAVED, TEMP_REGS, ARG_REGS, ALL_REGS, STACK_BASE, ZERO_REG,
    )
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# MachineOp — RISC-V machine instruction opcodes
# ═══════════════════════════════════════════════════════════════════════════════

class MachineOp(enum.Enum):
    """RISC-V machine instruction opcodes used by the compiler."""
    # ALU
    ADD = "add"
    ADDI = "addi"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    MAX = "max"     # pseudo: max rd, rs1, rs2
    SRAI = "srai"
    XOR = "xor"
    AND = "and"
    SLT = "slt"
    SLTI = "slti"
    REM = "rem"
    # Memory
    LW = "lw"
    SW = "sw"
    FLD = "fld"
    FSD = "fsd"
    # Control
    J = "j"
    JAL = "jal"
    JALR = "jalr"
    BEQ = "beq"
    BNE = "bne"
    BLT = "blt"
    BGE = "bge"
    BNEZ = "bnez"   # pseudo
    # Pseudo
    LI = "li"
    LA = "la"
    MV = "mv"
    CALL = "call"
    LABEL = ".label"
    # Directive
    SECTION = ".section"
    GLOBL = ".globl"
    SIZE = ".size"
    TYPE = ".type"
    # Float (F/D extension)
    SQRT_S = "fsqrt.s"
    SQRT_D = "fsqrt.d"
    FMIN_D = "fmin.d"
    FMAX_D = "fmax.d"
    FABS_D = "fabs.d"
    FNEG_D = "fneg.d"
    FNEG_S = "fneg.s"
    FADD_D = "fadd.d"
    FSUB_D = "fsub.d"
    FMUL_D = "fmul.d"
    FDIV_D = "fdiv.d"
    FLT_D = "flt.d"
    FLE_D = "fle.d"
    FEQ_D = "feq.d"
    FCVT_S_D = "fcvt.s.d"
    FCVT_D_S = "fcvt.d.s"
    LI_D = "li.d"
    # Float single-precision
    FADD_S = "fadd.s"
    FSUB_S = "fsub.s"
    FMUL_S = "fmul.s"
    FDIV_S = "fdiv.s"
    FMAX_S = "fmax.s"
    FMIN_S = "fmin.s"
    FABS_S = "fabs.s"
    FLE_S = "fle.s"
    FLT_S = "flt.s"
    FEQ_S = "feq.s"
    FSQRT_S = "fsqrt.s"
    FLW = "flw"
    FSW = "fsw"
    FMV_S = "fmv.s"
    FMV_D = "fmv.d"
    FMV_S_X = "fmv.s.x"


class RegisterClass(enum.Enum):
    """Architectural register file required by an operand."""

    GPR = "gpr"
    FPR = "fpr"


_ROUNDING_MODE_OPS = frozenset({
    MachineOp.FADD_S, MachineOp.FADD_D,
    MachineOp.FSUB_S, MachineOp.FSUB_D,
    MachineOp.FMUL_S, MachineOp.FMUL_D,
    MachineOp.FDIV_S, MachineOp.FDIV_D,
    MachineOp.SQRT_S, MachineOp.SQRT_D,
    MachineOp.FCVT_S_D, MachineOp.FCVT_D_S,
})


# ═══════════════════════════════════════════════════════════════════════════════
# MachineOperand — register or immediate operand
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MachineOperand:
    """A register or immediate operand."""

    kind: str  # "reg", "imm", "vreg"
    value: str | int
    reg_class: RegisterClass | None = None
    bit_width: int | None = None

    def __post_init__(self) -> None:
        if self.kind in {"reg", "vreg"} and self.reg_class is None:
            self.reg_class = RegisterClass.GPR
        if self.kind not in {"reg", "vreg"} and self.reg_class is not None:
            raise ValueError(f"{self.kind} operand cannot have a register class")
        if self.bit_width is not None and (
            self.kind not in {"reg", "vreg"} or self.bit_width not in {32, 64}
        ):
            raise ValueError("register width must be 32 or 64 bits")
        if self.reg_class is RegisterClass.GPR and self.bit_width == 64:
            raise ValueError("RV32 GPR operands must be 32 bits")

    @staticmethod
    def vreg(
        name: str,
        reg_class: RegisterClass = RegisterClass.GPR,
        bit_width: int | None = None,
    ) -> "MachineOperand":
        """Create a virtual register operand."""
        return MachineOperand("vreg", name, reg_class, bit_width)

    @staticmethod
    def immediate(val: int) -> "MachineOperand":
        """Create an immediate (constant) operand."""
        return MachineOperand("imm", val)

    @staticmethod
    def symbol(name: str) -> "MachineOperand":
        """Create a non-register symbolic assembly operand."""
        return MachineOperand("symbol", name)

    @staticmethod
    def reg(
        name: str,
        reg_class: RegisterClass = RegisterClass.GPR,
        bit_width: int | None = None,
    ) -> "MachineOperand":
        """Create a physical register operand."""
        return MachineOperand("reg", name, reg_class, bit_width)

    def __repr__(self) -> str:
        if self.kind == "imm":
            return str(self.value)
        if self.kind == "symbol":
            return str(self.value)
        return f"%{self.value}"


# ═══════════════════════════════════════════════════════════════════════════════
# MachineInstr — a single machine-level instruction
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MachineInstr:
    """A machine-level instruction using virtual or physical registers."""

    op: MachineOp
    dst: Optional[MachineOperand] = None
    src1: Optional[MachineOperand] = None
    src2: Optional[MachineOperand] = None
    comment: str = ""
    target: Optional[str] = None
    rounding_mode: Optional[str] = None
    label_role: Optional[str] = None

    def __post_init__(self) -> None:
        if self.label_role is not None:
            if self.op is not MachineOp.LABEL or self.label_role not in {"function", "block"}:
                raise ValueError("label_role requires a function or block LABEL")
        if self.rounding_mode is not None:
            self.rounding_mode = self.rounding_mode.lower()
            if self.rounding_mode not in {
                "rne", "rtz", "rdn", "rup", "rmm", "dyn",
            }:
                raise ValueError(
                    f"unsupported RISC-V rounding mode: {self.rounding_mode}"
                )
            if self.op not in _ROUNDING_MODE_OPS:
                raise ValueError(
                    f"{self.op.value} does not take a rounding mode"
                )

    def __repr__(self) -> str:
        parts = [self.op.value]
        for op in (self.dst, self.src1, self.src2):
            if op is not None:
                parts.append(str(op))
        if self.target:
            parts.append(self.target)
        if self.rounding_mode:
            parts.append(self.rounding_mode)
        s = " ".join(parts)
        if self.comment:
            s += f"  # {self.comment}"
        return s


@dataclass(frozen=True)
class FloatConstant:
    """One bit-exact entry in a compilation unit's floating constant pool."""

    label: str
    bit_width: int
    bit_pattern: int

    def __post_init__(self) -> None:
        if self.bit_width not in {32, 64}:
            raise ValueError("floating constant width must be 32 or 64 bits")
        if not 0 <= self.bit_pattern < (1 << self.bit_width):
            raise ValueError("floating constant bit pattern is out of range")


# ═══════════════════════════════════════════════════════════════════════════════
# RISC-V register sets
# ═══════════════════════════════════════════════════════════════════════════════

# Callee-saved registers (preserved across function calls)
CALLEE_SAVED: list[str] = [
    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
    "s8", "s9", "s10", "s11",
]

# Caller-saved temporary registers
TEMP_REGS: list[str] = ["t0", "t1", "t2", "t3", "t4", "t5", "t6"]

# Argument / return-value registers
ARG_REGS: list[str] = ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"]

# All allocatable integer registers (19 total)
ALL_REGS: list[str] = TEMP_REGS + CALLEE_SAVED

# Only caller-saved FP temporaries are allocated; fa*/fs* are ABI-owned.
FP_TEMP_REGS: list[str] = [f"ft{i}" for i in range(12)]
FP_ARG_REGS: list[str] = [f"fa{i}" for i in range(8)]


def canonical_register_name(name: str) -> str:
    """Normalize architectural aliases when tracking physical ownership."""
    integer_names = ["zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
                     "s0", "s1", *ARG_REGS, *CALLEE_SAVED[2:], *TEMP_REGS[3:]]
    float_names = [*FP_TEMP_REGS[:8], "fs0", "fs1", *FP_ARG_REGS,
                   *(f"fs{i}" for i in range(2, 12)), *FP_TEMP_REGS[8:]]
    aliases = {f"x{i}": alias for i, alias in enumerate(integer_names)}
    aliases.update({f"f{i}": alias for i, alias in enumerate(float_names)})
    aliases["fp"] = "s0"
    return aliases.get(name, name)


def physical_register_class(name: str) -> RegisterClass:
    """Classify architectural names, including ABI aliases and memory bases."""
    import re

    memory = re.fullmatch(r"-?\d+\((\w+)\)", name)
    if memory:
        if physical_register_class(memory.group(1)) is not RegisterClass.GPR:
            raise ValueError("memory address base must be a GPR")
        return RegisterClass.GPR
    if name in {
        *ALL_REGS, *ARG_REGS, "zero", "ra", "sp", "gp", "tp", "fp",
        *(f"x{i}" for i in range(32)),
    }:
        return RegisterClass.GPR
    if name in {
        *FP_TEMP_REGS, *FP_ARG_REGS, *(f"fs{i}" for i in range(12)),
        *(f"f{i}" for i in range(32)),
    }:
        return RegisterClass.FPR
    raise ValueError(f"unknown physical register: {name}")

# Special-purpose registers
STACK_BASE: str = "sp"
ZERO_REG: str = "x0"
