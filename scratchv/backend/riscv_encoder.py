"""RISC-V RV32IM instruction encoder.

Converts assembly text to 32-bit machine code words. Supports the
subset of instructions emitted by the ScratchV compiler backend.
"""

from __future__ import annotations

import re
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
        self._max_counter = 0
        self._temp_reg = 0

    # ── Pseudo-instruction expansion ──────────────────────────────────

    def _find_free_temp(self, asm_text: str) -> int:
        """Scan assembly text for used registers; return first free temp.

        Preference order: t6, t5, t4, t3, t2, t1, t0 (x31 down to x5).
        """
        used = set()
        for match in re.finditer(
            r'\b(zero|ra|sp|gp|tp|t[0-6]|s\d+|a\d+|fp|x\d+)\b',
            asm_text,
        ):
            name = match.group(1)
            if name in REG_MAP:
                used.add(REG_MAP[name])
        for r in [31, 30, 29, 28, 7, 6, 5]:
            if r not in used:
                return r
        return 31  # fallback

    def _expand_pseudo(self, line: str) -> list[str]:
        """Expand one possibly-pseudo line into standard RISC-V lines.

        Returns a list of lines (may be empty for skipped directives).
        """
        if not line or line.startswith(".") or line.endswith(":"):
            return [line]

        tokens = line.replace(",", " ").split()
        if not tokens:
            return [line]

        op = tokens[0].lower()

        # li rd, imm -> addi rd, x0, imm (small values), otherwise the
        # canonical LUI/ADDI pair.  A single RISC-V instruction cannot encode
        # an arbitrary 32-bit immediate; keeping a large ``li`` as one encoded
        # word silently drops its low 12 bits.
        if op == "li" and len(tokens) >= 3:
            rd = tokens[1]
            imm = self._parse_imm(tokens[2])
            return self._expand_li(rd, imm)

        # max rd, rs1, rs2  →  4-instruction sequence
        if op == "max":
            rd = tokens[1] if len(tokens) > 1 else "x0"
            rs1 = tokens[2] if len(tokens) > 2 else "x0"
            rs2 = tokens[3] if len(tokens) > 3 else "x0"
            n = self._max_counter
            self._max_counter += 1
            return [
                f"bge {rs1}, {rs2}, .__max_then_{n}",
                f"addi {rd}, x0, 0",
                f"j .__max_end_{n}",
                f".__max_then_{n}:",
                f"addi {rd}, {rs1}, 0",
                f".__max_end_{n}:",
            ]

        # Branch-with-immediate:  beq/bne/blt/bge rs1, imm, label
        #   → li xTEMP, imm; beq/bne/blt/bge rs1, xTEMP, label
        if op in ("beq", "bne", "blt", "bge"):
            if len(tokens) >= 4:
                op2 = tokens[2].rstrip(",")
                if op2 not in REG_MAP and not op2.startswith("x") and not op2.startswith("%"):
                    try:
                        imm = int(op2)
                    except ValueError:
                        pass
                    else:
                        temp = f"x{self._temp_reg}"
                        label = tokens[3]
                        return self._expand_li(temp, imm) + [
                            f"{op} {tokens[1]}, {temp}, {label}",
                        ]

        return [line]

    @staticmethod
    def _expand_li(rd: str, imm: int) -> list[str]:
        """Expand ``li`` into semantically equivalent RV32 instructions."""
        if not -(1 << 31) <= imm <= (1 << 31) - 1:
            raise ValueError(f"li immediate out of RV32 range: {imm}")
        if -2048 <= imm <= 2047:
            return [f"addi {rd}, x0, {imm}"]

        upper = (imm + 0x800) >> 12
        lower = imm - (upper << 12)
        result = [f"lui {rd}, {upper}"]
        if lower:
            result.append(f"addi {rd}, {rd}, {lower}")
        return result

    # ── Assembly pass ─────────────────────────────────────────────────

    def assemble(self, asm_text: str) -> bytearray:
        """Assemble RISC-V assembly text to flat binary."""
        # Pre-scan: find a free temp register for pseudo expansion
        clean_text = "\n".join(
            line.split("#")[0] for line in asm_text.split("\n")
        )
        self._temp_reg = self._find_free_temp(clean_text)

        lines = asm_text.strip().split("\n")
        instructions: list[tuple] = []  # (encoded_word, fixup_or_None)

        # Pass 1: expand pseudos, collect labels, encode
        for line in lines:
            line = line.split("#")[0].strip()
            if not line:
                continue

            # Expand pseudo-instructions (one line → possibly many)
            expanded = self._expand_pseudo(line)

            for exp_line in expanded:
                exp_line = exp_line.split("#")[0].strip()
                if not exp_line:
                    continue

                # Skip directives, but keep all labels (including .L and .__)
                if exp_line.startswith(".") and not exp_line.endswith(":"):
                    continue

                if exp_line.endswith(":"):
                    name = exp_line[:-1].strip()
                    self.labels[name] = len(instructions)
                    continue

                encoded = self._encode_line(exp_line, len(instructions))
                if encoded is not None:
                    instructions.append(encoded)

        # Pass 2: apply label fixups
        result = bytearray()
        for idx, (word, fixup) in enumerate(instructions):
            if fixup is not None:
                word = self._apply_fixup(word, fixup, idx)
            # Some encoders intentionally build signed Python integers when
            # bit 31 is set.  Machine words are always the low 32 bits.
            result.extend(struct.pack("<I", word & 0xFFFFFFFF))

        return result

    def _encode_line(
            self, line: str, idx: int,
    ) -> tuple[int, tuple[str, str] | None] | None:
        """Encode a single assembly line (standard RISC-V only — pseudos
        should already be expanded by ``_expand_pseudo``)."""
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
        elif op == "rem":
            rd = _reg_num(operands[0])
            rs1 = _reg_num(operands[1])
            rs2 = _reg_num(operands[2])
            word = _r_type(rd, rs1, rs2, 0b110, F7_MULDIV)
        elif op == "addi":
            rd = _reg_num(operands[0])
            rs1 = _reg_num(operands[1])
            imm = self._parse_imm(operands[2])
            word = _i_type(rd, rs1, imm, F3_ADD_SUB)
        elif op == "srai":
            rd = _reg_num(operands[0])
            rs1 = _reg_num(operands[1])
            shamt = self._parse_imm(operands[2]) & 0x1F
            word = _i_type(rd, rs1, shamt | (0b0100000 << 5), F3_SRL_SRA)
            # Shamt is encoded in lower 5 bits of the 12-bit immediate;
            # the upper 7 bits are 0100000 for SRAI.
            imm12 = shamt | (0b0100000 << 5)
            word = _i_type(rd, rs1, imm12, F3_SRL_SRA)
        elif op == "xor":
            rd = _reg_num(operands[0])
            rs1 = _reg_num(operands[1])
            rs2 = _reg_num(operands[2])
            word = _r_type(rd, rs1, rs2, F3_XOR, 0b0000000)
        elif op == "and":
            rd = _reg_num(operands[0])
            rs1 = _reg_num(operands[1])
            rs2 = _reg_num(operands[2])
            word = _r_type(rd, rs1, rs2, F3_AND, 0b0000000)
        elif op == "lw":
            rd = _reg_num(operands[0])
            mem_op = operands[1]
            if "(" in mem_op and ")" in mem_op:
                before = mem_op[:mem_op.index("(")]
                after = mem_op[mem_op.index("(") + 1:mem_op.index(")")]
                if before in REG_MAP or before.startswith("x"):
                    # Compiler syntax: lw rd, rs1(offset)
                    rs1 = _reg_num(before)
                    offset = self._parse_imm(after) if after else 0
                else:
                    # Standard syntax: lw rd, offset(rs1)
                    offset = self._parse_imm(before) if before else 0
                    rs1 = _reg_num(after)
            else:
                offset, rs1 = 0, 0
            word = _i_type(rd, rs1, offset, F3_LW, RVOpcode.LOAD)
        elif op == "sw":
            # Handle both standard (sw rs2, offset(rs1)) and compiler
            # (sw rs1(offset), rs2) syntax.
            if "(" in operands[0]:
                # Compiler syntax:  sw rs1(offset), rs2
                mem = operands[0].strip()
                base = mem[:mem.index("(")]
                off_str = mem[mem.index("(") + 1:mem.index(")")]
                offset = self._parse_imm(off_str) if off_str else 0
                rs1 = _reg_num(base)
                rs2 = _reg_num(operands[1])
            else:
                # Standard syntax:  sw rs2, offset(rs1)
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
            raise ValueError("li must be expanded before encoding")
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
                word = _i_type(1, 1, 0, 0, RVOpcode.JALR)
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
