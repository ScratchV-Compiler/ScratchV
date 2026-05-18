"""RISC-V RV32IM instruction encoder.

Converts assembly text to 32-bit machine code words. Supports the
subset of instructions emitted by the ScratchV compiler backend.
"""

from __future__ import annotations

import struct
from enum import IntEnum


# ── RISC-V opcodes ────────────────────────────────────────────────────

class RVOpcode(IntEnum):
    """RISC-V opcode map."""
    LOAD = 0b0000011
    STORE = 0b0100011
    BRANCH = 0b1100011
    JALR = 0b1100111
    JAL = 0b1101111
    OP_IMM = 0b0010011
    OP = 0b0110011
    LUI = 0b0110111
    AUIPC = 0b0010111


# ── funct3 ────────────────────────────────────────────────────────────

F3_ADD_SUB = 0b000
F3_SLL = 0b001
F3_SLT = 0b010
F3_SLTU = 0b011
F3_XOR = 0b100
F3_SRL_SRA = 0b101
F3_OR = 0b110
F3_AND = 0b111

F3_BEQ = 0b000
F3_BNE = 0b001
F3_BLT = 0b100
F3_BGE = 0b101
F3_BLTU = 0b110
F3_BGEU = 0b111

F3_LB = 0b000
F3_LH = 0b001
F3_LW = 0b010
F3_LBU = 0b100
F3_LHU = 0b101

F3_SB = 0b000
F3_SH = 0b001
F3_SW = 0b010


# ── funct7 ────────────────────────────────────────────────────────────

F7_ADD = 0b0000000
F7_SUB = 0b0100000
F7_MUL = 0b0000001
F7_MULDIV = 0b0000001  # M extension base funct7

# ── Register map ──────────────────────────────────────────────────────

REG_MAP: dict[str, int] = {
    "zero": 0, "x0": 0,
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


def _reg_num(name: str) -> int:
    name = name.strip().lstrip("%")
    if name in REG_MAP:
        return REG_MAP[name]
    # Handle stack-pointer offset syntax: "16(sp)", "-4(sp)"
    if "(" in name and ")" in name:
        base = name[name.index("(") + 1:name.index(")")]
        return REG_MAP.get(base, 0)
    return 0


def _sext(val: int, bits: int) -> int:
    """Sign-extend val to bits width."""
    mask = (1 << bits) - 1
    val = val & mask
    if val >> (bits - 1):
        val -= (1 << bits)
    return val


# ── Instruction encoders ──────────────────────────────────────────────

def _r_type(rd: int, rs1: int, rs2: int,
            funct3: int, funct7: int) -> int:
    return ((funct7 << 25) | (rs2 << 20) | (rs1 << 15)
            | (funct3 << 12) | (rd << 7) | RVOpcode.OP)


def _i_type(rd: int, rs1: int, imm: int, funct3: int,
            opcode: RVOpcode = RVOpcode.OP_IMM) -> int:
    return ((_sext(imm, 12) << 20) | (rs1 << 15)
            | (funct3 << 12) | (rd << 7) | opcode)


def _s_type(rs1: int, rs2: int, imm: int,
            funct3: int) -> int:
    imm = _sext(imm, 12)
    return ((imm >> 5) << 25) | (rs2 << 20) | (rs1 << 15) \
        | (funct3 << 12) | ((imm & 0x1F) << 7) | RVOpcode.STORE


def _b_type(rs1: int, rs2: int, imm: int,
            funct3: int) -> int:
    imm = _sext(imm, 13)
    b12 = (imm >> 12) & 1
    b10_5 = (imm >> 5) & 0x3F
    b4_1 = (imm >> 1) & 0xF
    b11 = (imm >> 11) & 1
    return ((b12 << 31) | (b10_5 << 25) | (rs2 << 20)
            | (rs1 << 15) | (funct3 << 12) | (b4_1 << 8)
            | (b11 << 7) | RVOpcode.BRANCH)


def _u_type(rd: int, imm: int) -> int:
    return ((_sext(imm, 20) << 12) | (rd << 7)
            | RVOpcode.LUI)


def _j_type(rd: int, imm: int) -> int:
    imm = _sext(imm, 21)
    b20 = (imm >> 20) & 1
    b10_1 = (imm >> 1) & 0x3FF
    b11 = (imm >> 11) & 1
    b19_12 = (imm >> 12) & 0xFF
    return ((b20 << 31) | (b19_12 << 12) | (b11 << 20)
            | (b10_1 << 21) | (rd << 7) | RVOpcode.JAL)


# ── High-level assembler ──────────────────────────────────────────────

class RISCVAEncoder:
    """Encode RISC-V assembly text to binary."""

    def __init__(self):
        self.labels: dict[str, int] = {}  # label -> instruction index
        self.pending_fixups: list[tuple[int, str, str]] = []

    def assemble(self, asm_text: str) -> bytearray:
        """Assemble RISC-V assembly text to flat binary."""
        lines = asm_text.strip().split("\n")
        instructions: list[tuple] = []  # (encoded_word, comment)

        # Pass 1: collect labels and encode
        for line in lines:
            line = line.split("#")[0].strip()
            if not line:
                continue

            # Skip directives (but not local labels like .Lxxx)
            if line.startswith(".") and not line.startswith(".L"):
                continue

            # Label detection (including .L local labels)
            if line.endswith(":"):
                name = line[:-1].strip()
                self.labels[name] = len(instructions)
                continue

            # Parse instruction
            encoded = self._encode_line(line, len(instructions))
            if encoded is not None:
                instructions.append(encoded)

        # Pass 2: apply label fixups
        result = bytearray()
        for idx, (word, fixup) in enumerate(instructions):
            if fixup is not None:
                word = self._apply_fixup(word, fixup, idx)
            result.extend(struct.pack("<I", word))

        return result

    def _encode_line(
            self, line: str, idx: int,
    ) -> tuple[int, tuple[str, str] | None] | None:
        """Encode a single assembly line."""
        # Tokenize
        tokens = line.replace(",", " ").split()
        if not tokens:
            return None

        op = tokens[0].lower()
        operands = tokens[1:]

        fixup = None

        if op == "add":
            rd = _reg_num(operands[0])
            rs1 = _reg_num(operands[1])
            rs2 = _reg_num(operands[2])
            word = _r_type(rd, rs1, rs2, F3_ADD_SUB, F7_ADD)
        elif op == "sub":
            rd = _reg_num(operands[0])
            rs1 = _reg_num(operands[1])
            rs2 = _reg_num(operands[2])
            word = _r_type(rd, rs1, rs2, F3_ADD_SUB, F7_SUB)
        elif op == "mul":
            rd = _reg_num(operands[0])
            rs1 = _reg_num(operands[1])
            rs2 = _reg_num(operands[2])
            word = _r_type(rd, rs1, rs2, F3_ADD_SUB, F7_MUL)
        elif op == "div":
            rd = _reg_num(operands[0])
            rs1 = _reg_num(operands[1])
            rs2 = _reg_num(operands[2])
            word = _r_type(rd, rs1, rs2, 0b100, F7_MULDIV)
        elif op == "addi":
            rd = _reg_num(operands[0])
            rs1 = _reg_num(operands[1])
            imm = self._parse_imm(operands[2])
            word = _i_type(rd, rs1, imm, F3_ADD_SUB)
        elif op == "lw":
            rd = _reg_num(operands[0])
            offset, rs1 = self._parse_mem(operands[1])
            word = _i_type(rd, rs1, offset, F3_LW, RVOpcode.LOAD)
        elif op == "sw":
            rs2 = _reg_num(operands[0])
            offset, rs1 = self._parse_mem(operands[1])
            word = _s_type(rs1, rs2, offset, F3_SW)
        elif op == "beq":
            rs1 = _reg_num(operands[0])
            rs2 = _reg_num(operands[1])
            label = operands[2]
            fixup = ("b", label)
            word = _b_type(rs1, rs2, 0, F3_BEQ)
        elif op == "bne":
            rs1 = _reg_num(operands[0])
            rs2 = _reg_num(operands[1])
            label = operands[2]
            fixup = ("b", label)
            word = _b_type(rs1, rs2, 0, F3_BNE)
        elif op == "blt":
            rs1 = _reg_num(operands[0])
            rs2 = _reg_num(operands[1])
            label = operands[2]
            fixup = ("b", label)
            word = _b_type(rs1, rs2, 0, F3_BLT)
        elif op == "bge":
            rs1 = _reg_num(operands[0])
            rs2 = _reg_num(operands[1])
            label = operands[2]
            fixup = ("b", label)
            word = _b_type(rs1, rs2, 0, F3_BGE)
        elif op == "bnez":
            rs1 = _reg_num(operands[0])
            label = operands[1]
            fixup = ("b", label)
            word = _b_type(rs1, 0, 0, F3_BNE)
        elif op == "j" or op == "jal":
            label = operands[0]
            fixup = ("j", label)
            word = _j_type(0, 0)
        elif op == "jalr":
            rd = _reg_num(operands[0])
            rs1 = _reg_num(operands[1])
            offset = self._parse_imm(operands[2]) if len(operands) > 2 else 0
            word = _i_type(rd, rs1, offset, 0, RVOpcode.JALR)
        elif op == "li":
            rd = _reg_num(operands[0])
            imm = self._parse_imm(operands[1])
            if -2048 <= imm <= 2047:
                word = _i_type(rd, 0, imm, F3_ADD_SUB)
            else:
                # lui + addi sequence — will be handled later
                upper = (imm + 0x800) >> 12
                word = _u_type(rd, upper)
                # Store second instruction
                self._pending_li = (rd, imm & 0xFFF)
        elif op == "mv":
            rd = _reg_num(operands[0])
            rs = _reg_num(operands[1])
            word = _i_type(rd, rs, 0, F3_ADD_SUB)
        elif op == "call":
            if operands:
                label = operands[0]
                fixup = ("call", label)
                word = _u_type(1, 0)
            else:
                # call without label (runtime call, target in comment)
                # Encode as auipc ra, 0 + jalr (nop-like, handled by emulator)
                word = _i_type(1, 1, 0, 0, RVOpcode.JALR)
                # Store runtime call info for later fixup
                fixup = ("runtime_call", "")
        elif op == "ret":
            word = _i_type(0, 1, 0, 0, RVOpcode.JALR)
        elif op == "lui":
            rd = _reg_num(operands[0])
            imm = self._parse_imm(operands[1])
            word = _u_type(rd, imm)
        elif op == "slt":
            rd = _reg_num(operands[0])
            rs1 = _reg_num(operands[1])
            if operands[2].startswith("%") or operands[2] in REG_MAP:
                rs2 = _reg_num(operands[2])
                word = _r_type(rd, rs1, rs2, F3_SLT, 0b0000000)
            else:
                imm = self._parse_imm(operands[2])
                word = _i_type(rd, rs1, imm, F3_SLT)
        elif op == "max":
            # Expand pseudo: blt rs1, rs2, +8; mv rd, rs2; j +8; mv rd, rs1
            # For single instruction encoding, emit as add (simplified)
            rd = _reg_num(operands[0]) if len(operands) > 0 else 0
            rs1 = _reg_num(operands[1]) if len(operands) > 1 else 0
            if len(operands) > 2 and (operands[2].startswith("%") or operands[2] in REG_MAP):
                rs2 = _reg_num(operands[2])
            else:
                rs2 = 0
            word = _r_type(rd, rs1, rs2, F3_ADD_SUB, 0b0000000)
            # Store as multi-instruction expansion
            self._pending_max = (rd, rs1, rs2)
            fixup = ("max_expand", "")
        elif op == "nop":
            word = _i_type(0, 0, 0, F3_ADD_SUB)
        else:
            raise ValueError(f"Unknown instruction: {op}")

        return (word, fixup)

    def _apply_fixup(self, word: int, fixup: tuple, current_idx: int) -> int:
        """Apply a label fixup to an already-encoded instruction."""
        kind, label = fixup
        if kind == "runtime_call":
            return word
        if kind == "max_expand":
            return word  # already encoded

        target_idx = self.labels.get(label, current_idx)
        offset = target_idx - current_idx

        if kind == "b":
            byte_offset = offset * 4
            rs1 = (word >> 15) & 0x1F
            rs2 = (word >> 20) & 0x1F
            funct3 = (word >> 12) & 0x7
            return _b_type(rs1, rs2, byte_offset, funct3)
        elif kind == "j":
            byte_offset = offset * 4
            return _j_type(0, byte_offset)
        elif kind == "call":
            byte_offset = offset * 4
            return _u_type(1, byte_offset >> 12)
        return word

    def _parse_imm(self, s: str) -> int:
        s = s.strip()
        if s.startswith("0x"):
            return int(s, 16)
        if s.startswith("-"):
            return int(s)
        return int(s)

    def _parse_mem(self, s: str) -> tuple[int, int]:
        """Parse memory operand like '16(sp)' -> (offset, rs1)."""
        s = s.strip()
        if "(" in s and ")" in s:
            offset_str = s[:s.index("(")]
            base = s[s.index("(") + 1:s.index(")")]
            offset = self._parse_imm(offset_str) if offset_str else 0
            return offset, _reg_num(base)
        return 0, 0


def assemble_to_binary(asm_text: str) -> bytearray:
    """Convenience function: assemble RISC-V text to binary."""
    encoder = RISCVAEncoder()
    return encoder.assemble(asm_text)
