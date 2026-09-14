"""Machine instruction types for the RISC-V backend.

This module defines the core types shared by all backend passes:
machine opcodes, operands, and instructions.  Extracted from
``register_alloc.py`` so that passes can import these types without
pulling in register-allocation logic.

Usage::

    from scratchv.backend.machine_types import (
        MachineOp, MachineOperand, MachineInstr,
        CALLEE_SAVED, TEMP_REGS, ARG_REGS, ALL_REGS, GREEDY_REGS,
        REG_NUMS, STACK_BASE, ZERO_REG,
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
    """A register, immediate, or memory operand."""

    kind: str  # "reg", "imm", "vreg", "mem"
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

    @staticmethod
    def mem(offset: int, base: str = "sp") -> "MachineOperand":
        """Create a memory operand, formatted as ``offset(base)``."""
        return MachineOperand("mem", f"{offset}({base})")

    def __repr__(self) -> str:
        if self.kind in ("imm", "mem"):
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

# Caller-saved registers: arguments + temporaries
CALLER_SAVED: list[str] = ARG_REGS + TEMP_REGS

# All allocatable integer registers (27 total):
# a0-a7 + t0-t6 + s0-s11, order shared with the linear-scan allocator pool.
ALL_REGS: list[str] = CALLER_SAVED + CALLEE_SAVED

# Legacy greedy allocator pool (19): temporaries + callee-saved.  Frozen so
# that the greedy allocation order is unchanged by the ALL_REGS correction.
GREEDY_REGS: list[str] = TEMP_REGS + CALLEE_SAVED

# Canonical RISC-V register-number table (includes x-aliases and fp).
REG_NUMS: dict[str, int] = {
    "x0": 0, "zero": 0,
    "ra": 1, "x1": 1,
    "sp": 2, "x2": 2,
    "gp": 3, "x3": 3,
    "tp": 4, "x4": 4,
    "t0": 5, "x5": 5,
    "t1": 6, "x6": 6,
    "t2": 7, "x7": 7,
    "s0": 8, "fp": 8, "x8": 8,
    "s1": 9, "x9": 9,
    "a0": 10, "x10": 10,
    "a1": 11, "x11": 11,
    "a2": 12, "x12": 12,
    "a3": 13, "x13": 13,
    "a4": 14, "x14": 14,
    "a5": 15, "x15": 15,
    "a6": 16, "x16": 16,
    "a7": 17, "x17": 17,
    "s2": 18, "x18": 18,
    "s3": 19, "x19": 19,
    "s4": 20, "x20": 20,
    "s5": 21, "x21": 21,
    "s6": 22, "x22": 22,
    "s7": 23, "x23": 23,
    "s8": 24, "x24": 24,
    "s9": 25, "x25": 25,
    "s10": 26, "x26": 26,
    "s11": 27, "x27": 27,
    "t3": 28, "x28": 28,
    "t4": 29, "x29": 29,
    "t5": 30, "x30": 30,
    "t6": 31, "x31": 31,
}

# Special-purpose registers
STACK_BASE: str = "sp"
ZERO_REG: str = "x0"
