"""基于美化器专用解析器实现 RISC-V 汇编字段对齐。

"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

if __package__:
    from .asm_parser_for_beautifier import (
        DIRECTIVES,
        INSTRUCTION_SPECS,
        ParsedAsmLine,
        _is_metadata_label,
        parse_asm,
    )
else:
    from asm_parser_for_beautifier import (
        DIRECTIVES,
        INSTRUCTION_SPECS,
        ParsedAsmLine,
        _is_metadata_label,
        parse_asm,
    )


LABEL_WIDTH_LIMIT = 30
OPCODE_WIDTH_MIN = 8
OPCODE_WIDTH_LIMIT = 12
OPERANDS_WIDTH_MIN = 15
OPERANDS_WIDTH_LIMIT = 40
FIELD_SEPARATOR = "  "
ALIGNABLE_STATUSES = frozenset({"valid"})

_OPERAND_MISSING_WARNING = "[warning: operand missing]"
_UNKNOWN_OPCODE_WARNING = "[warning: unknown opcode]"
_MALFORMED_WARNING = "[warning: malformed instruction]"
_PARSE_STATUS_WARNINGS: Mapping[str, str] = MappingProxyType(
    {
        "incomplete_operands": _OPERAND_MISSING_WARNING,
        "unknown_opcode": _UNKNOWN_OPCODE_WARNING,
        "malformed": _MALFORMED_WARNING,
    }
)

SECTION_BAR = "# " + "=" * 60
SECTION_MARKERS: Mapping[str, tuple[str, str, str]] = MappingProxyType(
    {
        ".text": (SECTION_BAR, "#  CODE SECTION", SECTION_BAR),
        ".data": (SECTION_BAR, "#  DATA SECTION", SECTION_BAR),
        ".bss": (SECTION_BAR, "#  BSS SECTION", SECTION_BAR),
        ".rodata": (
            SECTION_BAR,
            "#  READ-ONLY DATA SECTION",
            SECTION_BAR,
        ),
    }
)
_SECTION_MARKER_LINES = frozenset(
    marker_line
    for marker in SECTION_MARKERS.values()
    for marker_line in marker
)
_FUNCTION_MARKER_PREFIX = "# --- Function: "
_FUNCTION_MARKER_SUFFIX = " ---"
_CONTROL_FLOW_LABEL_RE = re.compile(
    r"^(?:\.L.*|_?L\d+|loop\d+)$",
    re.IGNORECASE,
)


INSTRUCTION_COMMENT_TEMPLATES: Mapping[str, Mapping[str, str]] = (
    MappingProxyType(
        {
            "integer_arithmetic": MappingProxyType(
                {
                    "add": "{rd} = {rs1} + {rs2}",
                    "sub": "{rd} = {rs1} - {rs2}",
                    "sll": "{rd} = {rs1} << {rs2}",
                    "slt": "{rd} = ({rs1} < {rs2}) ? 1 : 0",
                    "sltu": "{rd} = ({rs1} < {rs2}) ? 1 : 0 (unsigned)",
                    "xor": "{rd} = {rs1} XOR {rs2}",
                    "srl": "{rd} = {rs1} >> {rs2} (logical)",
                    "sra": "{rd} = {rs1} >> {rs2} (arithmetic)",
                    "or": "{rd} = {rs1} | {rs2}",
                    "and": "{rd} = {rs1} & {rs2}",
                }
            ),
            "immediate_arithmetic": MappingProxyType(
                {
                    "addi": "{rd} = {rs1} + {imm}",
                    "slti": "{rd} = ({rs1} < {imm}) ? 1 : 0",
                    "sltiu": "{rd} = ({rs1} < {imm}) ? 1 : 0 (unsigned)",
                    "xori": "{rd} = {rs1} XOR {imm}",
                    "ori": "{rd} = {rs1} | {imm}",
                    "andi": "{rd} = {rs1} & {imm}",
                    "slli": "{rd} = {rs1} << {imm}",
                    "srli": "{rd} = {rs1} >> {imm} (logical)",
                    "srai": "{rd} = {rs1} >> {imm} (arithmetic)",
                }
            ),
            "multiply_divide": MappingProxyType(
                {
                    "mul": "{rd} = {rs1} * {rs2}",
                    "mulh": "{rd} = ({rs1} * {rs2})[63:32]",
                    "div": "{rd} = {rs1} / {rs2}",
                    "divu": "{rd} = {rs1} / {rs2} (unsigned)",
                    "rem": "{rd} = {rs1} % {rs2}",
                    "remu": "{rd} = {rs1} % {rs2} (unsigned)",
                }
            ),
            "floating_point": MappingProxyType(
                {
                    "fadd.s": "{rd} = {rs1} + {rs2} (f32)",
                    "fsub.s": "{rd} = {rs1} - {rs2} (f32)",
                    "fmul.s": "{rd} = {rs1} * {rs2} (f32)",
                    "fdiv.s": "{rd} = {rs1} / {rs2} (f32)",
                }
            ),
            "memory": MappingProxyType(
                {
                    "lb": "{rd} = MEM8[{rs1} + {imm}]",
                    "lh": "{rd} = MEM16[{rs1} + {imm}]",
                    "lw": "{rd} = MEM[{rs1} + {imm}]",
                    "ld": "{rd} = MEM64[{rs1} + {imm}]",
                    "lbu": "{rd} = MEM8[{rs1} + {imm}] (unsigned)",
                    "lhu": "{rd} = MEM16[{rs1} + {imm}] (unsigned)",
                    "flw": "{rd} = MEM[{rs1} + {imm}] (f32)",
                    "fld": "{rd} = MEM[{rs1} + {imm}] (f64)",
                    "sb": "MEM8[{rs1} + {imm}] = {rs2} (low 8b)",
                    "sh": "MEM16[{rs1} + {imm}] = {rs2} (low 16b)",
                    "sw": "MEM[{rs1} + {imm}] = {rs2}",
                    "sd": "MEM64[{rs1} + {imm}] = {rs2}",
                    "fsw": "MEM[{rs1} + {imm}] = {rs2} (f32)",
                    "fsd": "MEM[{rs1} + {imm}] = {rs2} (f64)",
                }
            ),
            "control_flow": MappingProxyType(
                {
                    "beq": "if {rs1} == {rs2} goto {target}",
                    "bne": "if {rs1} != {rs2} goto {target}",
                    "blt": "if {rs1} < {rs2} goto {target}",
                    "bge": "if {rs1} >= {rs2} goto {target}",
                    "bltu": "if {rs1} < {rs2} goto {target} (unsigned)",
                    "bgeu": "if {rs1} >= {rs2} goto {target} (unsigned)",
                    "jal": "{rd} = PC+4; goto {target}",
                    "jalr": "{rd} = PC+4; goto {rs1}+{imm}",
                }
            ),
            "upper_immediate": MappingProxyType(
                {
                    "lui": "{rd} = {imm} << 12",
                    "auipc": "{rd} = PC + ({imm} << 12)",
                }
            ),
            "system": MappingProxyType(
                {
                    "fence": "memory barrier ({predecessor} -> {successor})",
                    "ecall": "environment call",
                    "ebreak": "debugger breakpoint",
                }
            ),
            "pseudo_instruction": MappingProxyType(
                {
                    "li": "{rd} = {imm}",
                    "mv": "{rd} = {rs}",
                    "not": "{rd} = ~{rs}",
                    "neg": "{rd} = -{rs}",
                    "seqz": "{rd} = ({rs} == 0) ? 1 : 0",
                    "snez": "{rd} = ({rs} != 0) ? 1 : 0",
                    "beqz": "if {rs} == 0 goto {target}",
                    "bnez": "if {rs} != 0 goto {target}",
                    "blez": "if {rs} <= 0 goto {target}",
                    "bgez": "if {rs} >= 0 goto {target}",
                    "bltz": "if {rs} < 0 goto {target}",
                    "bgtz": "if {rs} > 0 goto {target}",
                    "j": "goto {target}",
                    "call": "call {target}",
                    "tail": "tail call {target}",
                    "ret": "return",
                    "nop": "no operation",
                    "la": "{rd} = address({symbol})",
                    "jr": "goto {rs}",
                }
            ),
            "scratchv_custom": MappingProxyType(
                {
                    "max": "{rd} = max({rs1}, {rs2}) (ScratchV custom)",
                }
            ),
        }
    )
)

_INST_COMMENTS: Mapping[str, str] = MappingProxyType(
    {
        opcode: template
        for group in INSTRUCTION_COMMENT_TEMPLATES.values()
        for opcode, template in group.items()
    }
)

_ABI_REGISTER_NAMES: Mapping[str, str] = MappingProxyType(
    {
        "x0": "zero", "x1": "ra", "x2": "sp", "x3": "gp",
        "x4": "tp", "x5": "t0", "x6": "t1", "x7": "t2",
        "x8": "s0/fp", "x9": "s1", "x10": "a0", "x11": "a1",
        "x12": "a2", "x13": "a3", "x14": "a4", "x15": "a5",
        "x16": "a6", "x17": "a7", "x18": "s2", "x19": "s3",
        "x20": "s4", "x21": "s5", "x22": "s6", "x23": "s7",
        "x24": "s8", "x25": "s9", "x26": "s10", "x27": "s11",
        "x28": "t3", "x29": "t4", "x30": "t5", "x31": "t6",
    }
)

_MEMORY_OPERAND_RE = re.compile(
    r"^(?P<imm>[^()]*)\((?P<rs1>[^()]+)\)$"
)

_REGISTER_ROLES = frozenset({"rd", "rs", "rs1", "rs2"})
_LOAD_OPCODES = frozenset({"lb", "lh", "lw", "ld", "lbu", "lhu", "flw", "fld"})
_STORE_OPCODES = frozenset({"sb", "sh", "sw", "sd", "fsw", "fsd"})


@dataclass(frozen=True)
class ColumnWidths:
    """三个对齐字段实际使用的填充宽度。"""

    label: int = 0
    opcode: int = 0
    operands: int = 0


def _register_name(register: str, abi_register_names: bool) -> str:
    """按开关决定是否在自动注释中使用 ABI 寄存器别名。"""

    if not abi_register_names:
        return register
    return _ABI_REGISTER_NAMES.get(register, register)


def _memory_operand_context(
    operand: str,
    abi_register_names: bool,
) -> dict[str, str] | None:
    """从 ``imm(rs1)`` 内存操作数中提取偏移和基址寄存器。"""

    match = _MEMORY_OPERAND_RE.fullmatch(operand.strip())
    if match is None:
        return None

    return {
        "imm": match.group("imm").strip() or "0",
        "rs1": _register_name(
            match.group("rs1").strip(),
            abi_register_names,
        ),
    }


def _gen_comment(
    opcode: str | None,
    operands: Sequence[str],
    abi_register_names: bool = False,
) -> str:
    """为已登记且操作数完整的指令生成可读语义注释。"""

    if not opcode:
        return ""

    base_opcode = opcode.lstrip(".").lower()
    template = _INST_COMMENTS.get(base_opcode)
    if template is None:
        return ""

    spec = INSTRUCTION_SPECS.get(base_opcode)
    if spec is None or not (
        spec.min_operands <= len(operands) <= spec.max_operands
    ):
        return ""

    context: dict[str, str] = {}
    if base_opcode in _LOAD_OPCODES | _STORE_OPCODES:
        if len(operands) != 2:
            return ""
        memory_context = _memory_operand_context(
            operands[1],
            abi_register_names,
        )
        if memory_context is None:
            return ""
        context = memory_context
        role = "rd" if base_opcode in _LOAD_OPCODES else "rs2"
        context[role] = _register_name(
            operands[0],
            abi_register_names,
        )
    elif base_opcode == "jal":
        if len(operands) == 1:
            rd, target = "ra", operands[0]
        elif len(operands) == 2:
            rd, target = operands
        else:
            return ""
        context = {
            "rd": _register_name(rd, abi_register_names),
            "target": target,
        }
    elif base_opcode == "jalr":
        if len(operands) == 1:
            rd, rs1, imm = "ra", operands[0], "0"
        elif len(operands) == 2:
            rd, rs1 = operands
            imm = "0"
        else:
            rd, rs1, imm = operands
        context = {
            "rd": _register_name(rd, abi_register_names),
            "rs1": _register_name(rs1, abi_register_names),
            "imm": imm,
        }
    elif base_opcode == "fence":
        predecessor = operands[0] if operands else "iorw"
        successor = operands[1] if len(operands) == 2 else "iorw"
        context = {
            "predecessor": predecessor,
            "successor": successor,
        }
    else:
        for role, operand in zip(spec.operand_roles, operands):
            if role in _REGISTER_ROLES:
                context[role] = _register_name(
                    operand,
                    abi_register_names,
                )
            else:
                context[role] = operand

    return template.format_map(context)


def _is_alignable(line: ParsedAsmLine) -> bool:
    """判断解析行是否可以安全参与列对齐。"""

    if line.parse_status not in ALIGNABLE_STATUSES:
        return False
    if line.opcode in DIRECTIVES:
        return False

    # 空行和纯注释行没有会影响列宽的字段。
    return line.label is not None or line.opcode is not None


def _format_operands(line: ParsedAsmLine) -> str:
    """用逗号加一个空格连接顶层操作数。"""

    if not line.operands:
        return line.operands_str
    return ", ".join(line.operands)


def scan_column_widths(
    lines: Sequence[ParsedAsmLine],
) -> ColumnWidths:
    """扫描可对齐行并返回受上限约束的填充宽度。

    操作数宽度按照逗号空格规范化后的文本计算。操作码和操作数使用固定
    最小宽度；宽度上限只限制填充，不会截断超长字段。
    """

    max_label = 0
    max_opcode = 0
    max_operands = 0

    for line in lines:
        if not _is_alignable(line):
            continue

        lengths = line.field_lengths
        max_label = max(max_label, lengths["label"])
        max_opcode = max(max_opcode, lengths["opcode"])
        max_operands = max(max_operands, len(_format_operands(line)))

    return ColumnWidths(
        label=min(max_label, LABEL_WIDTH_LIMIT),
        opcode=min(
            max(max_opcode, OPCODE_WIDTH_MIN),
            OPCODE_WIDTH_LIMIT,
        ),
        operands=min(
            max(max_operands, OPERANDS_WIDTH_MIN),
            OPERANDS_WIDTH_LIMIT,
        ),
    )


def _merge_comments(
    original: str | None,
    automatic: str | None = None,
) -> str | None:
    """合并后续阶段生成的自动注释，同时保留原始注释。"""

    if original and automatic:
        return f"{original} | {automatic}"
    return original or automatic


def _automatic_comment(
    line: ParsedAsmLine,
    add_comments: bool,
    abi_register_names: bool,
) -> str | None:
    """按照解析状态和配置决定是否生成指令语义注释。"""

    warning = _PARSE_STATUS_WARNINGS.get(line.parse_status)
    if warning is not None:
        if line.comment == warning or (
            line.comment is not None
            and line.comment.endswith(f" | {warning}")
        ):
            return None
        return warning

    if (
        not add_comments
        or line.parse_status != "valid"
        or line.opcode in DIRECTIVES
        or (line.opcode == "nop" and line.comment is not None)
    ):
        return None

    return _gen_comment(
        line.opcode,
        line.operands,
        abi_register_names,
    ) or None


def _format_unaligned_line(
    line: ParsedAsmLine,
    automatic_comment: str | None,
    comment_column: int | None = None,
) -> str:
    """保留原行布局，并在需要时追加自动注释或对齐后的警告。"""

    if automatic_comment is None:
        return line.raw
    if line.comment is not None:
        return f"{line.raw} | {automatic_comment}"
    padding_width = len(FIELD_SEPARATOR)
    if comment_column is not None:
        padding_width = max(padding_width, comment_column - len(line.raw))
    return f"{line.raw}{' ' * padding_width}# {automatic_comment}"


def _comment_column(widths: ColumnWidths) -> int:
    """返回无标签指令统一使用的注释起始列。"""

    return (
        widths.opcode
        + len(FIELD_SEPARATOR)
        + widths.operands
        + len(FIELD_SEPARATOR)
    )


def _format_comment_only(
    comment: str,
    widths: ColumnWidths,
) -> str:
    """将纯注释放到与指令行注释相同的列。"""

    return " " * _comment_column(widths) + f"# {comment}"


def _format_parsed_line(
    line: ParsedAsmLine,
    widths: ColumnWidths,
    *,
    automatic_comment: str | None = None,
) -> str:
    """格式化一条安全解析行，并规范化操作数分隔空格。"""

    if (
        line.raw.strip() in _SECTION_MARKER_LINES
        or _is_function_marker_line(line.raw)
    ):
        return line.raw.strip()

    if (
        line.parse_status == "valid"
        and line.label is None
        and line.opcode is None
        and line.comment is not None
    ):
        return _format_comment_only(line.comment, widths)

    if not _is_alignable(line):
        return _format_unaligned_line(
            line,
            automatic_comment,
            comment_column=_comment_column(widths),
        )

    label_text = f"{line.label}:" if line.label is not None else ""
    comment = _merge_comments(line.comment, automatic_comment)

    if line.opcode is None:
        if comment:
            return f"{label_text}{FIELD_SEPARATOR}# {comment}"
        return label_text

    prefix = ""
    if label_text:
        # 扫描得到的标签长度不包含冒号。
        prefix = label_text.ljust(widths.label + 1) + FIELD_SEPARATOR

    opcode_field = line.opcode.ljust(widths.opcode)
    code = prefix + opcode_field

    operands_text = _format_operands(line)
    if operands_text:
        code += FIELD_SEPARATOR + operands_text.ljust(widths.operands)
    elif comment:
        # 为无操作数指令保留操作数列，使其注释与其他指令的注释对齐。
        code += FIELD_SEPARATOR + "".ljust(widths.operands)

    if comment:
        return f"{code}{FIELD_SEPARATOR}# {comment}"
    return code.rstrip()


def _append_section_marker(output: list[str], opcode: str | None) -> None:
    """在分段指示前插入尚未存在的三行段标记。"""

    marker = SECTION_MARKERS.get(opcode or "")
    if marker is None:
        return
    if tuple(item.strip() for item in output[-len(marker):]) == marker:
        return
    output.extend(marker)


def _function_marker(function_name: str) -> str:
    """返回设计文档规定的函数标题。"""

    return (
        f"{_FUNCTION_MARKER_PREFIX}{function_name}"
        f"{_FUNCTION_MARKER_SUFFIX}"
    )


def _is_function_marker_line(raw: str) -> bool:
    """判断一行是否是已生成的函数标题。"""

    stripped = raw.strip()
    return (
        stripped.startswith(_FUNCTION_MARKER_PREFIX)
        and stripped.endswith(_FUNCTION_MARKER_SUFFIX)
        and len(stripped) > (
            len(_FUNCTION_MARKER_PREFIX) + len(_FUNCTION_MARKER_SUFFIX)
        )
    )


def _ensure_blank_before_function_marker(output: list[str]) -> None:
    """除输出首行外，保证函数标题前至少存在一个空行。"""

    if output and output[-1].strip():
        output.append("")


def _has_unspaced_function_marker(
    lines: Sequence[ParsedAsmLine],
) -> bool:
    """判断原输入是否有缺少前置空行的已有函数标题。"""

    return any(
        index > 0
        and bool(lines[index - 1].raw.strip())
        and _is_function_marker_line(line.raw)
        for index, line in enumerate(lines)
    )


def _collect_function_labels(
    lines: Sequence[ParsedAsmLine],
) -> frozenset[str]:
    """按段、声明和调用关系收集本文件中的函数入口标签。"""

    current_section: str | None = None
    text_labels: set[str] = set()
    explicit_functions: set[str] = set()
    global_symbols: set[str] = set()
    call_targets: set[str] = set()

    for line in lines:
        if line.opcode in SECTION_MARKERS:
            current_section = line.opcode

        if (
            current_section == ".text"
            and line.label is not None
            and line.parse_status == "valid"
            and not _is_metadata_label(line.label)
        ):
            text_labels.add(line.label)

        if line.opcode == ".type" and len(line.operands) >= 2:
            if line.operands[1].strip().lower() == "@function":
                explicit_functions.add(line.operands[0].strip())
        elif line.opcode in {".globl", ".global"}:
            global_symbols.update(
                operand.strip() for operand in line.operands if operand.strip()
            )
        elif line.opcode == "call" and len(line.operands) == 1:
            call_targets.add(line.operands[0].strip())

    declared_functions = text_labels & (
        explicit_functions | global_symbols
    )
    called_functions = {
        target
        for target in call_targets & text_labels
        if _CONTROL_FLOW_LABEL_RE.fullmatch(target) is None
    }
    main_function = {"main"} if "main" in text_labels else set()
    return frozenset(
        declared_functions | called_functions | main_function
    )


def _append_function_marker(
    output: list[str],
    line: ParsedAsmLine,
    function_labels: frozenset[str],
) -> None:
    """在已识别的函数标签前插入尚未存在的函数标题。"""

    if line.label not in function_labels:
        return
    marker = _function_marker(line.label)
    if output and output[-1].strip() == marker:
        return
    _ensure_blank_before_function_marker(output)
    output.append(marker)


def _section_needs_trailing_blank(
    lines: Sequence[ParsedAsmLine],
    index: int,
) -> bool:
    """分段指示后仅在原输入没有空行时补充一个空行。"""

    return (
        lines[index].opcode in SECTION_MARKERS
        and (
            index + 1 == len(lines)
            or bool(lines[index + 1].raw.strip())
        )
    )


def beautify_asm(
    asm_text: str,
    align: bool = True,
    add_comments: bool = True,
    abi_register_names: bool = False,
) -> str:
    """返回完成标签、操作码和操作数字段对齐的汇编文本。

    四类分段指示和已识别函数前插入对应标题。关闭对齐时保留原始字段布局；
    对齐时普通标签始终独占一行，其他汇编器指示、未知行、异常行、元数据标签
    和空行按 ``raw`` 原样输出。可安全格式化的运行时指令会规范化顶层操作数
    分隔空格。
    """

    parsed_lines = parse_asm(asm_text)
    function_labels = _collect_function_labels(parsed_lines)

    if (
        not align
        and not add_comments
        and not function_labels
        and not any(line.opcode in SECTION_MARKERS for line in parsed_lines)
        and not _has_unspaced_function_marker(parsed_lines)
        and all(
            line.parse_status not in _PARSE_STATUS_WARNINGS
            for line in parsed_lines
        )
    ):
        return asm_text

    if not align:
        output: list[str] = []
        for index, line in enumerate(parsed_lines):
            _append_section_marker(output, line.opcode)
            if _is_function_marker_line(line.raw):
                _ensure_blank_before_function_marker(output)
            _append_function_marker(output, line, function_labels)
            output.append(
                _format_unaligned_line(
                    line,
                    _automatic_comment(
                        line,
                        add_comments,
                        abi_register_names,
                    ),
                )
            )
            if _section_needs_trailing_blank(parsed_lines, index):
                output.append("")
        return "\n".join(output)

    widths = scan_column_widths(parsed_lines)
    output = []

    for index, line in enumerate(parsed_lines):
        _append_section_marker(output, line.opcode)
        if _is_function_marker_line(line.raw):
            _ensure_blank_before_function_marker(output)
        _append_function_marker(output, line, function_labels)

        automatic_comment = _automatic_comment(
            line,
            add_comments,
            abi_register_names,
        )
        if (
            line.label is not None
            and _is_alignable(line)
        ):
            output.append(f"{line.label}:")
            if line.opcode is not None:
                output.append(
                    _format_parsed_line(
                        replace(line, label=None),
                        widths,
                        automatic_comment=automatic_comment,
                    )
                )
            elif line.comment is not None:
                output.append(_format_comment_only(line.comment, widths))
            if _section_needs_trailing_blank(parsed_lines, index):
                output.append("")
            continue

        output.append(
            _format_parsed_line(
                line,
                widths,
                automatic_comment=automatic_comment,
            )
        )
        if _section_needs_trailing_blank(parsed_lines, index):
            output.append("")

    return "\n".join(output)


def _write_text_atomic(
    output_path: Path,
    text: str,
    encoding: str,
) -> None:
    """通过同目录临时文件原子替换输出文件。"""

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            newline="",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(text)
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def beautify_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    align: bool = True,
    add_comments: bool = True,
    abi_register_names: bool = False,
    encoding: str = "utf-8",
) -> str:
    """读取并美化汇编文件，可选地将结果原子写入目标文件。"""

    source_path = Path(input_path)
    result = beautify_asm(
        source_path.read_text(encoding=encoding),
        align=align,
        add_comments=add_comments,
        abi_register_names=abi_register_names,
    )

    if output_path is not None:
        _write_text_atomic(Path(output_path), result, encoding)
    return result


def _build_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="Beautify a RISC-V assembly file.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="input assembly file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="write the beautified assembly to this file",
    )
    parser.add_argument(
        "--no-comments",
        action="store_true",
        help="disable generated semantic comments",
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="disable opcode, operand, and comment alignment",
    )
    parser.add_argument(
        "--abi-register-names",
        action="store_true",
        help="use ABI register names in generated comments",
    )
    return parser


def _print_file_error(path: Path, error: BaseException) -> None:
    """按单行格式向标准错误报告文件处理失败。"""

    print(f"error: {path}: {error}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    """运行独立命令行工具并返回进程退出码。"""

    args = _build_argument_parser().parse_args(argv)
    try:
        result = beautify_file(
            args.input,
            align=not args.no_align,
            add_comments=not args.no_comments,
            abi_register_names=args.abi_register_names,
        )
    except (OSError, UnicodeError) as error:
        _print_file_error(args.input, error)
        return 1

    if args.output is None:
        sys.stdout.write(result)
        return 0

    try:
        _write_text_atomic(args.output, result, "utf-8")
    except (OSError, UnicodeError) as error:
        _print_file_error(args.output, error)
        return 1
    return 0


__all__ = [
    "ColumnWidths",
    "LABEL_WIDTH_LIMIT",
    "OPCODE_WIDTH_MIN",
    "OPCODE_WIDTH_LIMIT",
    "OPERANDS_WIDTH_MIN",
    "OPERANDS_WIDTH_LIMIT",
    "SECTION_BAR",
    "SECTION_MARKERS",
    "INSTRUCTION_COMMENT_TEMPLATES",
    "beautify_asm",
    "beautify_file",
    "main",
    "scan_column_widths",
    "_gen_comment",
]


if __name__ == "__main__":
    raise SystemExit(main())
