"""Symbolic standalone finalization for safe assembly peephole optimization."""

from __future__ import annotations

import re
from typing import Any

from scratchv.backend.asm_peephole import AsmPeepholeOptimizer, _parse_asm

from .onnx_to_riscv_standalone import (
    _ABI_NAMES,
    _F3_AND,
    _F3_BEQ,
    _F3_BGE,
    _F3_BLT,
    _F3_BLTU,
    _F3_BNE,
    _F3_OR,
    _F3_XOR,
    _R_ZERO,
    _disasm_one,
    rv_add,
    rv_addi,
    rv_auipc,
    rv_btype,
    rv_itype,
    rv_j,
    rv_jal,
    rv_jalr,
    rv_li,
    rv_lui,
    rv_lw,
    rv_mul,
    rv_mulh,
    rv_nop,
    rv_or,
    rv_ret,
    rv_slli,
    rv_slt,
    rv_slti,
    rv_srai,
    rv_srli,
    rv_sub,
    rv_sw,
    rv_xor,
)

_REGISTERS = {name: number for number, name in _ABI_NAMES.items()}
_REGISTERS.update({f"x{number}": number for number in range(32)})
_MEMORY_OPERAND = re.compile(r"^([^()]+)\(([^()]+)\)$")
_BRANCH_FUNCT3 = {
    "beq": _F3_BEQ,
    "bne": _F3_BNE,
    "blt": _F3_BLT,
    "bge": _F3_BGE,
    "bltu": _F3_BLTU,
}


class PeepholeFinalizer:
    """Finalize an emitter through symbolic relocation and optional peephole."""

    def __init__(
        self,
        *,
        enabled: bool,
        optimizer: AsmPeepholeOptimizer | None = None,
    ):
        self.enabled = enabled
        self.optimizer = optimizer or AsmPeepholeOptimizer()
        self.input_assembly = ""
        self.output_assembly = ""
        self.relocation_validation = False

    def __call__(self, emitter: Any) -> None:
        self.input_assembly = symbolic_assembly(emitter)
        if not self.enabled:
            emitter.resolve_fixups()
            self.output_assembly = emitter.disassemble()
            self.relocation_validation = True
            return
        self.output_assembly, _ = self.optimizer.optimize(self.input_assembly)
        _reassemble(emitter, self.output_assembly)
        self.relocation_validation = True


def _label_definitions(emitter: Any) -> tuple[list[str], dict[str, str]]:
    names = list(getattr(emitter, "label_history", []))
    if not names:
        names = list(emitter.labels)
    if len(names) != len(set(names)):
        raise ValueError("duplicate label definition")
    if set(names) != set(emitter.labels):
        raise ValueError("label history does not match emitter labels")
    mapping = {name: f"L{index:04d}" for index, name in enumerate(names)}
    return names, mapping


def symbolic_assembly(emitter: Any) -> str:
    """Render unresolved emitter words with safe symbolic branch targets."""

    names, label_map = _label_definitions(emitter)
    labels_at: dict[int, list[str]] = {}
    for name in names:
        labels_at.setdefault(emitter.labels[name], []).append(label_map[name])

    fixups: dict[int, tuple[str, str]] = {}
    for index, kind, label in emitter.pending_fixups:
        if index in fixups:
            raise ValueError(f"duplicate relocation at instruction {index}")
        if label not in emitter.labels:
            raise ValueError(f"undefined label '{label}'")
        if index < 0 or index >= len(emitter.code):
            raise ValueError(f"relocation index out of range: {index}")
        fixups[index] = (kind, label_map[label])

    lines: list[str] = []
    for index, word in enumerate(emitter.code):
        lines.extend(f"{label}:" for label in labels_at.get(index, []))
        if index in emitter.protected_indices:
            instruction = f".word 0x{word:08x}"
        else:
            instruction = _disasm_one(word)
            fixup = fixups.get(index)
            if fixup is not None:
                kind, target = fixup
                if kind not in {"b", "j"}:
                    raise ValueError(f"unsupported relocation kind '{kind}'")
                if "," in instruction:
                    instruction = instruction.rsplit(",", 1)[0] + f", {target}"
                else:
                    instruction = f"{instruction.split()[0]} {target}"
        comment = emitter.comments.get(index, "")
        if comment and not instruction.startswith(".word"):
            instruction += f"  # {comment}"
        lines.append(instruction)
    for label in labels_at.get(len(emitter.code), []):
        lines.append(f"{label}:")
    return "\n".join(lines)


def _register(text: str) -> int:
    try:
        return _REGISTERS[text.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown register '{text}'") from exc


def _immediate(text: str) -> int:
    try:
        return int(text, 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid immediate '{text}'") from exc


def _target(text: str, labels: dict[str, int], index: int) -> int:
    try:
        return _immediate(text)
    except ValueError:
        if text not in labels:
            raise ValueError(f"undefined label '{text}'")
        return (labels[text] - index) * 4


def _memory_operand(text: str) -> tuple[int, int]:
    match = _MEMORY_OPERAND.match(text)
    if match is None:
        raise ValueError(f"invalid memory operand '{text}'")
    return _immediate(match.group(1)), _register(match.group(2))


def _validate_signed(value: int, low: int, high: int, kind: str) -> None:
    if not low <= value <= high:
        raise ValueError(f"{kind} offset {value} out of range [{low}, {high}]")
    if value % 2:
        raise ValueError(f"{kind} offset {value} is not 2-byte aligned")


def _encode_instruction(
    opcode: str,
    operands: list[str],
    index: int,
    labels: dict[str, int],
) -> list[int]:
    if opcode == ".word":
        if len(operands) != 1:
            raise ValueError(".word requires one operand")
        return [_immediate(operands[0]) & 0xFFFFFFFF]
    if opcode == "nop":
        return [rv_nop()]
    if opcode == "li":
        if len(operands) != 2:
            raise ValueError("li requires rd and immediate")
        return list(rv_li(_register(operands[0]), _immediate(operands[1])))
    if opcode == "mv":
        if len(operands) != 2:
            raise ValueError("mv requires rd and rs")
        return [rv_addi(_register(operands[0]), _register(operands[1]), 0)]
    if opcode in {"addi", "slti", "xori", "ori", "andi"}:
        if len(operands) != 3:
            raise ValueError(f"{opcode} requires rd, rs1 and immediate")
        rd, rs1, imm = (
            _register(operands[0]),
            _register(operands[1]),
            _immediate(operands[2]),
        )
        if not -2048 <= imm <= 2047:
            raise ValueError(f"{opcode} immediate {imm} out of range")
        builders = {
            "addi": rv_addi,
            "slti": rv_slti,
            "xori": lambda a, b, c: rv_itype(a, b, c, _F3_XOR),
            "ori": lambda a, b, c: rv_itype(a, b, c, _F3_OR),
            "andi": lambda a, b, c: rv_itype(a, b, c, _F3_AND),
        }
        return [builders[opcode](rd, rs1, imm)]
    if opcode in {"slli", "srli", "srai"}:
        if len(operands) != 3:
            raise ValueError(f"{opcode} requires rd, rs1 and shift")
        rd, rs1, shift = (
            _register(operands[0]),
            _register(operands[1]),
            _immediate(operands[2]),
        )
        if not 0 <= shift <= 31:
            raise ValueError(f"shift {shift} out of range")
        return [
            {"slli": rv_slli, "srli": rv_srli, "srai": rv_srai}[opcode](rd, rs1, shift)
        ]
    if opcode in {"add", "sub", "mul", "mulh", "div", "slt", "xor", "or", "and"}:
        if len(operands) != 3:
            raise ValueError(f"{opcode} requires rd, rs1 and rs2")
        rd, rs1, rs2 = map(_register, operands)
        builders = {
            "add": rv_add,
            "sub": rv_sub,
            "mul": rv_mul,
            "mulh": rv_mulh,
            "div": lambda a, b, c: rv_mul(a, b, c) | (0b100 << 12),
            "slt": rv_slt,
            "xor": rv_xor,
            "or": rv_or,
            "and": lambda a, b, c: rv_add(a, b, c) | (0b111 << 12),
        }
        return [builders[opcode](rd, rs1, rs2)]
    if opcode in {"lw", "sw"}:
        if len(operands) != 2:
            raise ValueError(f"{opcode} requires register and memory operand")
        register = _register(operands[0])
        offset, base = _memory_operand(operands[1])
        if not -2048 <= offset <= 2047:
            raise ValueError(f"{opcode} offset {offset} out of range")
        return [
            (
                rv_lw(register, base, offset)
                if opcode == "lw"
                else rv_sw(base, register, offset)
            )
        ]
    if opcode in _BRANCH_FUNCT3:
        if len(operands) != 3:
            raise ValueError(f"{opcode} requires rs1, rs2 and target")
        offset = _target(operands[2], labels, index)
        _validate_signed(offset, -4096, 4094, "branch")
        return [
            rv_btype(
                _register(operands[0]),
                _register(operands[1]),
                offset,
                _BRANCH_FUNCT3[opcode],
            )
        ]
    if opcode == "j":
        if len(operands) != 1:
            raise ValueError("j requires a target")
        offset = _target(operands[0], labels, index)
        _validate_signed(offset, -(1 << 20), (1 << 20) - 2, "jump")
        return [rv_j(offset)]
    if opcode == "jal":
        if len(operands) != 2:
            raise ValueError("jal requires rd and target")
        offset = _target(operands[1], labels, index)
        _validate_signed(offset, -(1 << 20), (1 << 20) - 2, "jump")
        return [rv_jal(_register(operands[0]), offset)]
    if opcode == "jalr":
        if len(operands) != 3:
            raise ValueError("jalr requires rd, rs1 and immediate")
        return [
            rv_jalr(
                _register(operands[0]), _register(operands[1]), _immediate(operands[2])
            )
        ]
    if opcode == "ret":
        return [rv_ret()]
    if opcode == "jr":
        if len(operands) != 1:
            raise ValueError("jr requires rs1")
        return [rv_jalr(_R_ZERO, _register(operands[0]), 0)]
    if opcode in {"lui", "auipc"}:
        if len(operands) != 2:
            raise ValueError(f"{opcode} requires rd and immediate")
        builder = rv_lui if opcode == "lui" else rv_auipc
        return [builder(_register(operands[0]), _immediate(operands[1]))]
    raise ValueError(f"cannot encode optimized instruction '{opcode}'")


def _line_width(line: Any) -> int:
    if line.opcode is None:
        return 0
    if line.opcode == "li":
        if len(line.operands) != 2:
            raise ValueError("li requires rd and immediate")
        return len(rv_li(_register(line.operands[0]), _immediate(line.operands[1])))
    return 1


def _reassemble(emitter: Any, assembly: str) -> None:
    lines = _parse_asm(assembly)
    labels: dict[str, int] = {}
    widths: list[int] = []
    index = 0
    for line in lines:
        if line.label is not None:
            if line.label in labels:
                raise ValueError(f"duplicate label definition '{line.label}'")
            labels[line.label] = index
        width = _line_width(line)
        widths.append(width)
        index += width

    code: list[int] = []
    comments: dict[int, str] = {}
    index = 0
    for line, width in zip(lines, widths):
        if line.opcode is None:
            if line.label is not None:
                continue
            if line.raw.strip() and not line.raw.strip().startswith("#"):
                raise ValueError(f"cannot parse assembly line: {line.raw!r}")
            continue
        words = _encode_instruction(line.opcode, line.operands, index, labels)
        if len(words) != width:
            raise ValueError(f"instruction width changed for '{line.raw}'")
        if line.comment:
            comments[index] = line.comment
        code.extend(words)
        index += len(words)

    inverse_labels = {
        safe: original for original, safe in _label_definitions(emitter)[1].items()
    }
    emitter.code = code
    emitter.labels = {
        inverse_labels.get(name, name): position for name, position in labels.items()
    }
    emitter.label_history = list(emitter.labels)
    emitter.pending_fixups.clear()
    emitter.comments = comments
