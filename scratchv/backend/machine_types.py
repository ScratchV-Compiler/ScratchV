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
    FADD_D = "fadd.d"
    FSUB_D = "fsub.d"
    FMUL_D = "fmul.d"
    FDIV_D = "fdiv.d"
    FLT_D = "flt.d"
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
    FLE_S = "fle.s"
    FLT_S = "flt.s"
    FEQ_S = "feq.s"
    FSQRT_S = "fsqrt.s"
    FLW = "flw"
    FSW = "fsw"
    FMV_S = "fmv.s"
    FMV_S_X = "fmv.s.x"


# ═══════════════════════════════════════════════════════════════════════════════
# MachineOperand — register or immediate operand
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MachineOperand:
    """A register or immediate operand."""

    kind: str  # "reg", "imm", "vreg"
    value: str | int

    @staticmethod
    def vreg(name: str) -> "MachineOperand":
        """Create a virtual register operand."""
        return MachineOperand("vreg", name)

    @staticmethod
    def immediate(val: int) -> "MachineOperand":
        """Create an immediate (constant) operand."""
        return MachineOperand("imm", val)

    @staticmethod
    def reg(name: str) -> "MachineOperand":
        """Create a physical register operand."""
        return MachineOperand("reg", name)

    def __repr__(self) -> str:
        if self.kind == "imm":
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

    def __repr__(self) -> str:
        parts = [self.op.value]
        for op in (self.dst, self.src1, self.src2):
            if op is not None:
                parts.append(str(op))
        s = " ".join(parts)
        if self.comment:
            s += f"  # {self.comment}"
        return s


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

# Special-purpose registers
STACK_BASE: str = "sp"
ZERO_REG: str = "x0"
