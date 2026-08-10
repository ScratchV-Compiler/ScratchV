"""Functional unit tests for assembly formatting behavior."""

from scratchv.backend.asm_beautifier import (
    ColumnWidths,
    beautify_asm,
    scan_column_widths,
)
from scratchv.backend.asm_parser_for_beautifier import parse_asm


def test_two_pass_alignment_normalizes_operands_and_preserves_comment() -> None:
    source = (
        "add a0,a1,  a2 # sum\n"
        "sw ra,28(sp)\n"
        "addi a0"
    )

    result = beautify_asm(source, add_comments=False)
    add_line, sw_line, incomplete_line = result.splitlines()

    assert add_line == "add       a0, a1, a2       # sum"
    assert sw_line == "sw        ra, 28(sp)"
    assert incomplete_line == (
        "addi a0                    # [warning: operand missing]"
    )
    assert add_line.index("a0, a1, a2") == sw_line.index("ra, 28(sp)")


def test_incomplete_operands_do_not_affect_scanned_widths() -> None:
    parsed = parse_asm("add a0,a1,a2\naddi a0")

    assert scan_column_widths(parsed) == ColumnWidths(
        label=0,
        opcode=8,
        operands=15,
    )


def test_directives_are_preserved_and_do_not_affect_column_widths() -> None:
    directives = (
        '.loc 1 2 0\n'
        '.word 1,2,  3\n'
        '.asciz "a,b"\n'
        '.Ldata: .word 4,5'
    )
    source = directives + "\nadd a0,a1,a2"

    result = beautify_asm(source, add_comments=False)
    widths = scan_column_widths(parse_asm(source))

    assert result.splitlines()[:4] == directives.splitlines()
    assert result.splitlines()[4] == "add       a0, a1, a2"
    assert widths == ColumnWidths(label=0, opcode=8, operands=15)


def test_short_fields_use_minimum_widths() -> None:
    parsed = parse_asm("ret\nadd a0,a1,a2")

    assert scan_column_widths(parsed) == ColumnWidths(
        label=0,
        opcode=8,
        operands=15,
    )

    ret_line, add_line = beautify_asm("ret\nadd a0,a1,a2").splitlines()
    assert add_line.index("a0, a1, a2") == 10
    assert ret_line.index("#") == add_line.index("#") == 27


def test_padding_caps_do_not_truncate_long_fields() -> None:
    long_label = "label_" + "x" * 40
    long_operand = "1" * 50
    source = f"{long_label}: li a0,{long_operand}"
    parsed = parse_asm(source)

    widths = scan_column_widths(parsed)
    result = beautify_asm(source, add_comments=False)

    assert widths == ColumnWidths(label=30, opcode=8, operands=40)
    assert long_label in result
    assert long_operand in result


def test_label_comment_is_moved_to_a_separate_aligned_line() -> None:
    source = "add a0, a1, a2 # inline\n.Ldone: # target"

    result = beautify_asm(source, add_comments=False)
    inline, label, standalone_comment = result.splitlines()

    assert label == ".Ldone:"
    assert inline.index("#") == standalone_comment.index("#")
    assert standalone_comment.lstrip() == "# target"


def test_no_align_returns_original_newlines_and_whitespace() -> None:
    source = "  add a0,a1,a2\r\n\t# note\r\n"

    assert beautify_asm(
        source,
        align=False,
        add_comments=False,
    ) == source


def test_trailing_newline_is_preserved_when_aligning() -> None:
    result = beautify_asm(
        "add a0, a1, a2\n",
        add_comments=False,
    )

    assert result.endswith("\n")
