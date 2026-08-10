"""Functional integration tests for the RISC-V assembly beautifier."""

from pathlib import Path

import pytest

import scratchv.backend.asm_beautifier as asm_beautifier

from scratchv.backend.asm_beautifier import (
    ColumnWidths,
    FIELD_SEPARATOR,
    OPCODE_WIDTH_MIN,
    OPERANDS_WIDTH_MIN,
    beautify_asm,
    beautify_file,
    scan_column_widths,
)
from scratchv.backend.asm_parser_for_beautifier import parse_asm


SECTION_BAR = "# " + "=" * 60


def _standard_comment_column() -> int:

    return (
        OPCODE_WIDTH_MIN
        + len(FIELD_SEPARATOR)
        + OPERANDS_WIDTH_MIN
        + len(FIELD_SEPARATOR)
    )


def test_incomplete_operands_keep_raw_code_and_do_not_split_label() -> None:
    source_lines = [
        "  addi   a0 # user note",
        ".Lbad: beq a0,a1",
    ]

    result = beautify_asm("\n".join(source_lines), add_comments=False)

    first_line, second_line = result.splitlines()
    assert first_line == (
        "  addi   a0 # user note | [warning: operand missing]"
    )
    assert second_line.startswith(".Lbad: beq a0,a1")
    assert second_line.index("#") == _standard_comment_column()
    assert second_line.endswith("# [warning: operand missing]")


def test_incomplete_warning_is_not_disabled_with_alignment_and_comments() -> None:
    result = beautify_asm(
        "  addi   a0",
        align=False,
        add_comments=False,
    )

    assert result == "  addi   a0  # [warning: operand missing]"


@pytest.mark.parametrize(
    ("source", "warning"),
    [
        ("custom_op x1,x2,x3", "[warning: unknown opcode]"),
        ("li,,a0,,3", "[warning: malformed instruction]"),
        ("main::", "[warning: malformed instruction]"),
        ("this is not asm", "[warning: malformed instruction]"),
        ('.asciz "unterminated', "[warning: malformed instruction]"),
    ],
)
@pytest.mark.parametrize("align", [True, False])
def test_unknown_and_malformed_lines_keep_raw_code_and_add_warning(
    source: str,
    warning: str,
    align: bool,
) -> None:
    result = beautify_asm(
        source,
        align=align,
        add_comments=False,
    )

    expected_comment_column = (
        max(_standard_comment_column(), len(source) + 2)
        if align
        else len(source) + 2
    )
    assert result.startswith(source)
    assert result.index("#") == expected_comment_column
    assert result.endswith(f"# {warning}")


@pytest.mark.parametrize(
    ("source", "warning"),
    [
        (
            "custom_op x1,x2 # user note",
            "[warning: unknown opcode]",
        ),
        (
            "li,,a0,,3 # user note",
            "[warning: malformed instruction]",
        ),
    ],
)
def test_unsafe_line_keeps_user_comment_before_warning(
    source: str,
    warning: str,
) -> None:
    result = beautify_asm(source, add_comments=False)

    assert result == f"{source} | {warning}"


def test_unknown_opcode_takes_priority_over_operand_count() -> None:
    result = beautify_asm("custom_add a0", add_comments=False)

    assert result.endswith("# [warning: unknown opcode]")
    assert "operand missing" not in result


@pytest.mark.parametrize(
    "source",
    [
        "add a0",
        "custom_op x1,x2",
        "li,,a0,,3",
    ],
)
def test_generated_warning_is_not_duplicated(source: str) -> None:
    first_result = beautify_asm(source, add_comments=False)
    second_result = beautify_asm(first_result, add_comments=False)

    assert second_result == first_result
    assert second_result.count("[warning:") == 1


def test_all_unsafe_statuses_are_excluded_from_scanned_widths() -> None:
    source = (
        "add a0,a1,a2\n"
        "very_long_unknown_opcode x1,x2,x3\n"
        "li,,a0,,3"
    )

    assert scan_column_widths(parse_asm(source)) == ColumnWidths(
        label=0,
        opcode=8,
        operands=15,
    )


def test_short_unsafe_warning_aligns_with_normal_comment() -> None:
    source = "add a0,a1,a2\ncustom_op x1,x2"

    normal, unsafe = beautify_asm(source).splitlines()
    
    assert normal.index("#") == unsafe.index("#")


def test_default_options_merge_comments_in_structured_program() -> None:
    source = (
        ".text\n"
        "main:\n"
        "add x1,x2,x3 # user note\n"
        "custom_add a0 # unknown note"
    )

    result = beautify_asm(source)
    lines = result.splitlines()

    assert lines[:7] == [
        SECTION_BAR,
        "#  CODE SECTION",
        SECTION_BAR,
        ".text",
        "",
        "# --- Function: main ---",
        "main:",
    ]
    assert lines[7].endswith("# user note | x1 = x2 + x3")
    assert lines[7].index("#") == _standard_comment_column()
    assert lines[8] == (
        "custom_add a0 # unknown note | [warning: unknown opcode]"
    )


def test_long_unsafe_line_keeps_two_spaces_before_warning() -> None:
    source = "very_long_custom_opcode_name x1,x2"

    result = beautify_asm(source, add_comments=False)

    assert result == f"{source}  # [warning: unknown opcode]"


def test_malformed_line_does_not_stop_following_valid_line() -> None:
    source = "main::\nadd a0,a1,a2"

    malformed, valid = beautify_asm(source, align=False).splitlines()

    assert malformed == "main::  # [warning: malformed instruction]"
    assert valid == "add a0,a1,a2  # a0 = a1 + a2"


@pytest.mark.parametrize(
    ("directive", "section_name"),
    [
        (".text", "CODE"),
        (".data", "DATA"),
        (".bss", "BSS"),
        (".rodata", "READ-ONLY DATA"),
    ],
)
def test_section_marker_is_inserted_before_directive(
    directive: str,
    section_name: str,
) -> None:
    result = beautify_asm(directive, add_comments=False)

    assert result.splitlines() == [
        SECTION_BAR,
        f"#  {section_name} SECTION",
        SECTION_BAR,
        directive,
    ]


def test_section_markers_are_inserted_without_alignment() -> None:
    source = ".text\nadd x1,x2,x3"

    result = beautify_asm(source, align=False, add_comments=False)

    assert result.splitlines() == [
        SECTION_BAR,
        "#  CODE SECTION",
        SECTION_BAR,
        ".text",
        "",
        "add x1,x2,x3",
    ]


def test_existing_section_marker_is_not_duplicated() -> None:
    source = "\n".join(
        [
            SECTION_BAR,
            "#  DATA SECTION",
            SECTION_BAR,
            ".data",
            "",
        ]
    )

    result = beautify_asm(source, add_comments=False)

    assert result == source
    assert beautify_asm(result, add_comments=False) == result


def test_standalone_comment_aligns_while_unsafe_and_blank_lines_are_preserved() -> None:
    source_lines = [
        "add a0, a1, a2 # inline",
        "# standalone",
        "   ",
        "_op_/layer1.0/Conv_5:",
        "custom_op x1,x2 # keep",
        "main::",
    ]

    result = beautify_asm(
        "\n".join(source_lines),
        add_comments=False,
    )
    result_lines = result.split("\n")

    assert result_lines[0].index("#") == result_lines[1].index("#")
    assert result_lines[1].lstrip() == "# standalone"
    assert result_lines[2:] == [
        source_lines[2],
        source_lines[3],
        "custom_op x1,x2 # keep | [warning: unknown opcode]",
        "main::                     # [warning: malformed instruction]",
    ]


def test_declared_inline_function_label_is_written_separately() -> None:
    source = (
        ".text\n"
        ".globl main\n"
        "main: add a0, a1, a2 # first instruction"
    )

    result = beautify_asm(source, add_comments=False)
    lines = result.splitlines()

    assert lines[-3] == "# --- Function: main ---"
    assert lines[-2] == "main:"
    assert lines[-1].startswith("add")
    assert not lines[-1].startswith(" ")
    assert "# first instruction" in lines[-1]


def test_ordinary_inline_label_is_also_written_separately() -> None:
    source = ".text\n.L1: addi a0, a0, 1"

    result = beautify_asm(source, add_comments=False)
    lines = result.splitlines()

    assert lines[-2] == ".L1:"
    assert lines[-1].startswith("addi")
    assert not lines[-1].startswith(" ")


@pytest.mark.parametrize(
    "declaration",
    [
        ".type worker, @function",
        ".globl worker",
        ".global worker",
    ],
)
def test_explicit_local_function_gets_function_marker(
    declaration: str,
) -> None:
    source = f".text\n{declaration}\nworker:\nret"

    result = beautify_asm(source, add_comments=False)

    assert result.splitlines()[-3:] == [
        "# --- Function: worker ---",
        "worker:",
        "ret",
    ]
    assert result.index(declaration) < result.index(
        "# --- Function: worker ---"
    )
    marker_index = result.splitlines().index("# --- Function: worker ---")
    assert result.splitlines()[marker_index - 1] == ""


def test_text_main_is_a_function_without_declaration() -> None:
    result = beautify_asm(".text\nmain:\nret", add_comments=False)

    assert result.splitlines()[-3:] == [
        "# --- Function: main ---",
        "main:",
        "ret",
    ]


def test_direct_call_target_defined_in_text_is_a_function() -> None:
    source = ".text\ncall worker\nworker:\nret"

    result = beautify_asm(source, add_comments=False)

    assert "# --- Function: worker ---\nworker:" in result


@pytest.mark.parametrize(
    "label",
    [".L1", "L1", "_L1", "loop1", "ordinary_label"],
)
def test_control_and_unconfirmed_labels_are_not_functions(label: str) -> None:
    source = f".text\n{label}:\naddi a0,a0,1"

    result = beautify_asm(source, add_comments=False)

    assert f"# --- Function: {label} ---" not in result
    assert f"{label}:" in result


def test_data_label_is_not_a_function_even_when_global() -> None:
    source = ".globl buffer\n.data\nbuffer:\n.word 1"

    result = beautify_asm(source, add_comments=False)

    assert "# --- Function: buffer ---" not in result
    assert "buffer:" in result


@pytest.mark.parametrize(
    "metadata_label",
    [
        "_op_/layer1.0/Conv_5",
        "_op_PPQ_Operation_6_29",
    ],
)
def test_metadata_label_is_never_a_function(
    metadata_label: str,
) -> None:
    source = (
        f".text\n.globl {metadata_label}\n"
        f"{metadata_label}:"
    )

    result = beautify_asm(source, add_comments=False)

    assert f"# --- Function: {metadata_label} ---" not in result
    assert result.splitlines()[-1] == f"{metadata_label}:"


def test_no_align_adds_marker_but_preserves_inline_function_line() -> None:
    source = ".text\n.globl worker\nworker:  add a0,a1,a2"

    result = beautify_asm(source, align=False, add_comments=False)

    assert result.splitlines()[-3:] == [
        "",
        "# --- Function: worker ---",
        "worker:  add a0,a1,a2",
    ]


def test_existing_function_marker_is_not_duplicated() -> None:
    source = (
        ".text\n.globl worker\n"
        "# --- Function: worker ---\nworker:\nret"
    )

    result = beautify_asm(source, add_comments=False)

    assert result.count("# --- Function: worker ---") == 1
    assert ".globl worker\n\n# --- Function: worker ---" in result
    assert beautify_asm(result, add_comments=False) == result


def test_function_marker_on_first_line_has_no_leading_blank() -> None:
    source = "# --- Function: worker ---\nworker:"

    result = beautify_asm(source, align=False, add_comments=False)

    assert result == source
    assert not result.startswith("\n")


def test_unspaced_existing_marker_gets_one_blank_without_alignment() -> None:
    source = (
        "addi a0,a0,1\n"
        "# --- Function: worker ---\n"
        "worker:"
    )

    result = beautify_asm(source, align=False, add_comments=False)

    assert result == (
        "addi a0,a0,1\n\n"
        "# --- Function: worker ---\n"
        "worker:"
    )
    assert beautify_asm(
        result,
        align=False,
        add_comments=False,
    ) == result


def test_consecutive_functions_have_one_blank_before_each_marker() -> None:
    source = (
        ".text\n"
        ".type first, @function\n"
        ".type second, @function\n"
        "first:\nret\n"
        "second:\nret"
    )

    result = beautify_asm(source, add_comments=False)

    assert ".type second, @function\n\n# --- Function: first ---" in result
    assert "first:\nret\n\n# --- Function: second ---" in result
    assert beautify_asm(result, add_comments=False) == result


def test_complete_program_receives_structure_comments_and_alignment() -> None:
    source = "\n".join(
        [
            ".text",
            "main:",
            "addi sp, sp, -32",
            "sw ra, 28(sp)",
            "li a5, 3",
            "li a4, 5",
            "add a5, a5, a4",
            "ret",
        ]
    )

    result = beautify_asm(source)
    lines = result.splitlines()

    assert lines[:7] == [
        SECTION_BAR,
        "#  CODE SECTION",
        SECTION_BAR,
        ".text",
        "",
        "# --- Function: main ---",
        "main:",
    ]

    expected_instructions = [
        ("addi", "sp, sp, -32", "sp = sp + -32"),
        ("sw", "ra, 28(sp)", "MEM[sp + 28] = ra"),
        ("li", "a5, 3", "a5 = 3"),
        ("li", "a4, 5", "a4 = 5"),
        ("add", "a5, a5, a4", "a5 = a5 + a4"),
        ("ret", None, "return"),
    ]
    comment_columns: list[int] = []

    for line, (opcode, operands, comment) in zip(
        lines[7:],
        expected_instructions,
    ):
        code, actual_comment = line.split("#", maxsplit=1)
        assert code.strip().split(maxsplit=1)[0] == opcode
        assert actual_comment.strip() == comment
        if operands is not None:
            assert operands in code
        comment_columns.append(line.index("#"))

    assert len(comment_columns) == len(expected_instructions)
    assert len(set(comment_columns)) == 1


def test_beautify_file_returns_and_writes_result(tmp_path: Path) -> None:
    input_path = tmp_path / "input.s"
    output_path = tmp_path / "output.s"
    input_path.write_text("add x1,x2,x3", encoding="utf-8")

    result = beautify_file(
        input_path,
        output_path,
        abi_register_names=True,
    )

    assert output_path.read_text(encoding="utf-8") == result
    assert "# ra = sp + gp" in result


def test_beautify_file_does_not_replace_output_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.s"
    output_path = tmp_path / "output.s"
    input_path.write_text("ret", encoding="utf-8")
    output_path.write_text("original", encoding="utf-8")

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(asm_beautifier.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        beautify_file(input_path, output_path)

    assert output_path.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.glob(".output.s.*.tmp")) == []
