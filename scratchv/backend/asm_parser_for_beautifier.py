"""
Usage:

    from scratchv.backend._asm_parser_for_beautifier import ParsedAsmLine, parse_asm, parse_asm_line

    line = parse_asm_line("add a0, a1, a2")
    print(line.opcode)
    print(line.operands)

    parsed = parse_asm(asm_text)
    for item in parsed:
        if item.opcode == "add":
            print(item.operands)
"""

import re

from dataclasses import dataclass, field
from typing import Literal, Optional

ParseStatus = Literal[
    "valid",
    "incomplete_operands",
    "unknown_opcode",
    "metadata_label",
    "malformed",
]

@dataclass(frozen=True)
class InstructionSpec:
    """一类指令允许的操作数数量及其语义角色。"""

    min_operands: int
    max_operands: int
    operand_roles: tuple[str, ...] = ()

@dataclass
class ParsedAsmLine:
    """一行RISC-V汇编代码的解析结果。"""
    raw: str
    label: Optional[str] = None
    opcode: Optional[str] = None
    operands_str: str = ""
    operands: list[str] = field(default_factory=list)
    comment: Optional[str] = None
    lineno: int = 0
    parse_status: ParseStatus = "valid"

    @property
    def field_lengths(self) -> dict[str, int]:
        """返回用于列宽统计的字段长度。"""

        return {
            "label": len(self.label or ""),
            "opcode": len(self.opcode or ""),
            "operands_str": len(self.operands_str),
            "comment": len(self.comment or ""),
        }

def _spec_group(
    opcodes: str,
    *roles: str,
) -> dict[str, InstructionSpec]:
    """批量创建具有相同操作数角色的指令规格。"""

    spec = InstructionSpec(len(roles), len(roles), roles)
    return {opcode: spec for opcode in opcodes.split()}


INSTRUCTION_SPECS: dict[str, InstructionSpec] = {
    # 三操作数的寄存器运算：rd, rs1, rs2。
    **_spec_group(
        "add sub sll slt sltu xor srl sra or and "
        "mul mulh div divu rem remu max",
        "rd", "rs1", "rs2",
    ),
    # 三操作数的立即数运算：rd, rs1, imm。
    **_spec_group(
        "addi slti sltiu xori ori andi slli srli srai",
        "rd", "rs1", "imm",
    ),
    # 常见三操作数单精度浮点运算：rd, rs1, rs2。
    **_spec_group(
        "fadd.s fsub.s fmul.s fdiv.s",
        "rd", "rs1", "rs2",
    ),
    # load 和 store 都有两个操作数，但第一个操作数角色不同。
    **_spec_group(
        "lb lh lw ld lbu lhu flw fld",
        "rd", "memory",
    ),
    **_spec_group("sb sh sw sd fsw fsd", "rs2", "memory"),
    # 条件分支使用两个源寄存器和一个跳转目标。
    **_spec_group(
        "beq bne blt bge bltu bgeu",
        "rs1", "rs2", "target",
    ),
    # 常见 U 型指令和汇编器伪指令。
    **_spec_group("lui auipc li", "rd", "imm"),
    **_spec_group("mv not neg seqz snez", "rd", "rs"),
    **_spec_group(
        "beqz bnez blez bgez bltz bgtz",
        "rs", "target",
    ),
    **_spec_group("j call tail", "target"),
    **_spec_group("ret nop ecall ebreak"),
    "la": InstructionSpec(2, 2, ("rd", "symbol")),
    "jr": InstructionSpec(1, 1, ("rs",)),
    "jal": InstructionSpec(1, 2, ("rd", "target")),
    "jalr": InstructionSpec(1, 3, ("rd", "rs1", "imm")),
    "fence": InstructionSpec(
        0, 2, ("predecessor", "successor")
    ),
}

# 汇编器指示不属于 CPU 指令，不进行操作数数量校验。
DIRECTIVES = frozenset(
    """
    .text .data .bss .rodata .section .globl .global .type .size
    .align .file .loc .cfi_startproc .cfi_endproc .cfi_def_cfa
    .cfi_offset .cfi_restore .byte .half .word .dword .quad
    .2byte .4byte .8byte .float .double
    .string .asciz .ascii .zero .space .skip
    .balign .p2align .option .set
    """.split()
)

# 正则只负责识别字段边界；括号、字符串和逗号由扫描函数处理。
_NORMAL_LABEL_RE = re.compile(r"^[A-Za-z_.$][A-Za-z0-9_.$]*$")
_METADATA_LABEL_RES = (
    re.compile(r"^_op_/[^:\s]+$"),
    re.compile(r"^_op_PPQ_Operation_\d+_\d+$"),
)
_LABEL_PREFIX_RE = re.compile(
    r"^(?P<label>[^\s:]+):(?P<body>.*)$"
)
_OPCODE_RE = re.compile(
    r"^(?P<opcode>"
    r"(?:\.[A-Za-z0-9][A-Za-z0-9_.]*"
    r"|[A-Za-z][A-Za-z0-9_.]*)"
    r")"
    r"(?:\s+(?P<operands>.*?))?$"
)


def _is_escaped(text: str, index: int) -> bool:
    """Return whether the character at ``index`` is backslash-escaped.

    An odd-length run of consecutive backslashes escapes the character;
    an even-length run represents complete backslash pairs and does not.
    """

    backslash_count = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslash_count += 1
        cursor -= 1
    return backslash_count % 2 == 1


def _is_metadata_label(label: Optional[str]) -> bool:
    """判断标签是否是metadata标签。"""
    if not label:
        return False

    for pattern in _METADATA_LABEL_RES:
        if pattern.fullmatch(label):
            return True

    return False

def _split_comment(
    line: str,
) -> tuple[str, Optional[str], bool]:
    """在字符串和括号之外查找行尾注释。"""

    in_string = False
    depth = 0

    for index, char in enumerate(line):
        # 引号内的 # 和括号都作为普通字符处理。
        if char == '"' and not _is_escaped(line, index):
            in_string = not in_string
        elif not in_string:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    return line.rstrip(), None, False
            elif char == "#" and depth == 0:
                return (
                    line[:index].rstrip(),
                    line[index + 1 :].strip(),
                    True,
                )

    return line.rstrip(), None, not in_string and depth == 0

def _has_unsafe_space(operand: str) -> bool:
    """检查操作数内部是否存输入错误，如add a0 a1 a2。"""

    in_string = False
    depth = 0

    for index, char in enumerate(operand):
        if char == '"' and not _is_escaped(operand, index):
            in_string = not in_string
        elif not in_string:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char.isspace() and depth == 0:
                return True

    return False

def _split_operands(
    text: str,
    *,
    allow_top_level_spaces: bool = False,
) -> Optional[list[str]]:
    """按顶层逗号拆分；格式异常时返回 None。

    普通指令的单个操作数不允许包含顶层空格，以便识别缺失逗号的
    ``add a0 a1 a2``。汇编器指示具有独立语法，可通过参数允许空格。
    """

    if not text:
        return []

    operands: list[str] = []
    current: list[str] = []

    in_string = False
    depth = 0

    for index, char in enumerate(text):
        if char == '"' and not _is_escaped(text, index):
            in_string = not in_string
        elif not in_string:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    return None
            elif char == "," and depth == 0:
                operand = "".join(current).strip()
                if not operand:
                    return None
                operands.append(operand)
                current = []
                continue

        current.append(char)

    last_operand = "".join(current).strip()
    if in_string or depth != 0 or not last_operand:
        return None

    operands.append(last_operand)
    if (
        not allow_top_level_spaces
        and any(_has_unsafe_space(item) for item in operands)
    ):
        return None

    return operands

def _split_label(
    code: str,
) -> tuple[Optional[str], str, bool]:
    """提取可选标签并验证标签格式。"""

    stripped = code.strip()
    match = _LABEL_PREFIX_RE.fullmatch(stripped)
    if match is None:
        return None, stripped, True

    label = match.group("label")
    body = match.group("body").strip()
    label_is_valid = bool(
        _NORMAL_LABEL_RE.fullmatch(label)
        or _is_metadata_label(label)
    )

    return label, body, label_is_valid and not body.startswith(":")

def _status_for(
    label: Optional[str],
    opcode: Optional[str],
    operands: list[str],
) -> ParseStatus:
    """根据标签、opcode 和操作数数量确定解析状态。"""

    if _is_metadata_label(label):
        return "metadata_label"
    # 空行、普通标签和已登记汇编器指示均可安全处理。
    if opcode is None or opcode in DIRECTIVES:
        return "valid"

    spec = INSTRUCTION_SPECS.get(opcode)
    if spec is None:
        return "unknown_opcode"
    if len(operands) < spec.min_operands:
        return "incomplete_operands"
    if len(operands) > spec.max_operands:
        return "malformed"
    return "valid"

def parse_asm_line(
    line: str,
    lineno: int = 0,
) -> ParsedAsmLine:
    """将一行汇编解析为 ``ParsedAsmLine``。"""

    code, comment, structure_is_valid = _split_comment(line)
    if not structure_is_valid:
        return ParsedAsmLine(
            raw=line,
            comment=comment,
            lineno=lineno,
            parse_status="malformed",
        )

    if not code.strip():
        return ParsedAsmLine(
            raw=line,
            comment=comment,
            lineno=lineno,
        )

    label, body, label_is_valid = _split_label(code)
    if not label_is_valid:
        return ParsedAsmLine(
            raw=line,
            comment=comment,
            lineno=lineno,
            parse_status="malformed",
        )

    if not body:
        status: ParseStatus = (
            "metadata_label"
            if _is_metadata_label(label)
            else "valid"
        )
        return ParsedAsmLine(
            raw=line,
            label=label,
            comment=comment,
            lineno=lineno,
            parse_status=status,
        )

    match = _OPCODE_RE.fullmatch(body)
    if match is None:
        return ParsedAsmLine(
            raw=line,
            label=label,
            comment=comment,
            lineno=lineno,
            parse_status="malformed",
        )

    opcode = match.group("opcode").lower()
    operands_str = (match.group("operands") or "").strip()
    operands = _split_operands(
        operands_str,
        allow_top_level_spaces=opcode in DIRECTIVES,
    )

    if operands is None:
        return ParsedAsmLine(
            raw=line,
            label=label,
            opcode=opcode,
            operands_str=operands_str,
            comment=comment,
            lineno=lineno,
            parse_status="malformed",
        )

    return ParsedAsmLine(
        raw=line,
        label=label,
        opcode=opcode,
        operands_str=operands_str,
        operands=operands,
        comment=comment,
        lineno=lineno,
        parse_status=_status_for(label, opcode, operands),
    )

def parse_asm(asm_text: str) -> list[ParsedAsmLine]:
    """按原顺序解析完整汇编文本并记录行号。"""

    # 统一三种换行符后再逐行解析，raw 不包含换行符本身。
    normalized = asm_text.replace("\r\n", "\n").replace("\r", "\n")
    return [
        parse_asm_line(line, lineno=index)
        for index, line in enumerate(normalized.split("\n"))
    ]
