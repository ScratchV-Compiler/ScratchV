"""Tests for extracting fields from one RISC-V assembly line.

``parse_asm_line`` returns a ``ParsedAsmLine`` dataclass, so tests use
attribute access.
"""

import pytest

try:
    from scratchv.backend.asm_parser_for_beautifier import parse_asm_line
except ModuleNotFoundError as error:
    if error.name != "scratchv":
        raise
    from asm_parser_for_beautifier import parse_asm_line


class TestValidAsmLine:
    """Test valid instructions, labels, directives, comments, and operands."""

    def test_simple_instruction(self):
        source = "add x1, x2, x3"
        result = parse_asm_line(source)

        assert result.raw == source
        assert result.label is None
        assert result.opcode == "add"
        assert result.operands_str == "x1, x2, x3"
        assert result.operands == ["x1", "x2", "x3"]
        assert result.comment is None
        assert result.parse_status == "valid"

    def test_label_instruction_operands_and_comment(self):
        source = "main: addi sp, sp, -16 # create stack frame"
        result = parse_asm_line(source)

        assert result.raw == source
        assert result.label == "main"
        assert result.opcode == "addi"
        assert result.operands_str == "sp, sp, -16"
        assert result.operands == ["sp", "sp", "-16"]
        assert result.comment == "create stack frame"
        assert result.parse_status == "valid"

    def test_label_only(self):
        result = parse_asm_line("function1:")

        assert result.label == "function1"
        assert result.opcode is None
        assert result.operands_str == ""
        assert result.operands == []
        assert result.comment is None
        assert result.parse_status == "valid"

    def test_directive_without_operands(self):
        result = parse_asm_line(".text")

        assert result.label is None
        assert result.opcode == ".text"
        assert result.operands_str == ""
        assert result.operands == []
        assert result.comment is None
        assert result.parse_status == "valid"

    @pytest.mark.parametrize(
        ("source", "expected_operands_str", "expected_operands"),
        [
            (
                '.file 1 "main.c"',
                '1 "main.c"',
                ['1 "main.c"'],
            ),
            (
                ".loc 1 20 0",
                "1 20 0",
                ["1 20 0"],
            ),
            (
                ".set buffer_size, 4 + 8",
                "buffer_size, 4 + 8",
                ["buffer_size", "4 + 8"],
            ),
        ],
    )
    def test_directive_allows_top_level_spaces(
        self,
        source,
        expected_operands_str,
        expected_operands,
    ):
        result = parse_asm_line(source)

        assert result.operands_str == expected_operands_str
        assert result.operands == expected_operands
        assert result.parse_status == "valid"

    def test_pseudo_instruction(self):
        result = parse_asm_line("li a0, 42")

        assert result.label is None
        assert result.opcode == "li"
        assert result.operands_str == "a0, 42"
        assert result.operands == ["a0", "42"]
        assert result.comment is None
        assert result.parse_status == "valid"

    @pytest.mark.parametrize(
        "source",
        [
            "mulh a0, a1, a2",
            "fadd.s ft0, ft1, ft2",
        ],
    )
    def test_representative_extended_instruction_is_valid(self, source):
        result = parse_asm_line(source)

        assert result.parse_status == "valid"

    @pytest.mark.parametrize(
        ("source", "expected_opcode"),
        [
            (".float 1.0", ".float"),
            (".space 16", ".space"),
            (".2byte 1", ".2byte"),
        ],
    )
    def test_representative_extended_data_directive_is_valid(
        self,
        source,
        expected_opcode,
    ):
        result = parse_asm_line(source)

        assert result.opcode == expected_opcode
        assert result.parse_status == "valid"

    def test_memory_operand_is_not_split_at_parentheses(self):
        result = parse_asm_line("lw a0, 8(sp)")

        assert result.label is None
        assert result.opcode == "lw"
        assert result.operands_str == "a0, 8(sp)"
        assert result.operands == ["a0", "8(sp)"]
        assert result.comment is None
        assert result.parse_status == "valid"

    def test_comment_only(self):
        result = parse_asm_line("# This is a comment")

        assert result.label is None
        assert result.opcode is None
        assert result.operands_str == ""
        assert result.operands == []
        assert result.comment == "This is a comment"
        assert result.parse_status == "valid"

    def test_empty_line(self):
        result = parse_asm_line("")

        assert result.raw == ""
        assert result.label is None
        assert result.opcode is None
        assert result.operands_str == ""
        assert result.operands == []
        assert result.comment is None
        assert result.parse_status == "valid"


class TestEscapedQuotes:
    """Test that only unescaped quotes change string-scanning state."""

    def test_escaped_quote_keeps_hash_inside_string(self):
        source = r'.asciz "a\"#b" # trailing comment'
        result = parse_asm_line(source)

        assert result.opcode == ".asciz"
        assert result.operands_str == r'"a\"#b"'
        assert result.operands == [r'"a\"#b"']
        assert result.comment == "trailing comment"
        assert result.parse_status == "valid"

    def test_escaped_quote_keeps_comma_and_space_inside_string(self):
        source = r'.asciz "a\", b c"'
        result = parse_asm_line(source)

        assert result.operands_str == r'"a\", b c"'
        assert result.operands == [r'"a\", b c"']
        assert result.comment is None
        assert result.parse_status == "valid"

    @pytest.mark.parametrize(
        ("source", "expected_operands", "expected_comment"),
        [
            (
                r'.asciz "a\\\"#b"',
                [r'"a\\\"#b"'],
                None,
            ),
            (
                r'.asciz "a\\" # trailing comment',
                [r'"a\\"'],
                "trailing comment",
            ),
        ],
    )
    def test_backslash_parity_controls_quote_escaping(
        self,
        source,
        expected_operands,
        expected_comment,
    ):
        result = parse_asm_line(source)

        assert result.operands == expected_operands
        assert result.comment == expected_comment
        assert result.parse_status == "valid"


class TestMetadataLabel:
    """Test ScratchV operator metadata labels separately from function labels."""

    @pytest.mark.parametrize(
        ("source", "expected_label"),
        [
            (
                "_op_/layer1.0/Conv_5:",
                "_op_/layer1.0/Conv_5",
            ),
            (
                "_op_PPQ_Operation_6_29:",
                "_op_PPQ_Operation_6_29",
            ),
        ],
    )
    def test_operator_metadata_label(self, source, expected_label):
        result = parse_asm_line(source)

        assert result.raw == source
        assert result.label == expected_label
        assert result.opcode is None
        assert result.operands_str == ""
        assert result.operands == []
        assert result.comment is None
        assert result.parse_status == "metadata_label"
        assert result.field_lengths == {
            "label": len(expected_label),
            "opcode": 0,
            "operands_str": 0,
            "comment": 0,
        }

    def test_internal_done_label_is_not_metadata(self):
        result = parse_asm_line("_input_copy_done:")

        assert result.label == "_input_copy_done"
        assert result.parse_status == "valid"

    def test_nop_with_operator_comment_preserves_original_comment(self):
        source = (
            "nop # --- Conv: input1, layer1.0.weight"
            " -> /layer1.0/Conv_output_0"
        )
        result = parse_asm_line(source)

        assert result.raw == source
        assert result.label is None
        assert result.opcode == "nop"
        assert result.operands_str == ""
        assert result.operands == []
        assert result.comment == (
            "--- Conv: input1, layer1.0.weight"
            " -> /layer1.0/Conv_output_0"
        )
        assert result.parse_status == "valid"


class TestValidFieldLengths:
    """Test field lengths used by the later column-alignment pass.

    Lengths use the parsed field values:
    - ``label`` excludes the trailing colon;
    - ``comment`` excludes ``#`` and surrounding separator spaces;
    - ``operands_str`` preserves its internal spaces;
    - a missing field has length 0.
    """

    @pytest.mark.parametrize(
        ("source", "expected_lengths"),
        [
            (
                "main: addi sp, sp, -16 # create stack frame",
                {
                    "label": 4,
                    "opcode": 4,
                    "operands_str": 11,
                    "comment": 18,
                },
            ),
            (
                "add x1, x2, x3",
                {
                    "label": 0,
                    "opcode": 3,
                    "operands_str": 10,
                    "comment": 0,
                },
            ),
            (
                "function1:",
                {
                    "label": 9,
                    "opcode": 0,
                    "operands_str": 0,
                    "comment": 0,
                },
            ),
            (
                ".text",
                {
                    "label": 0,
                    "opcode": 5,
                    "operands_str": 0,
                    "comment": 0,
                },
            ),
            (
                "# This is a comment",
                {
                    "label": 0,
                    "opcode": 0,
                    "operands_str": 0,
                    "comment": 17,
                },
            ),
        ],
    )
    def test_valid_line_field_lengths(self, source, expected_lengths):
        result = parse_asm_line(source)

        assert result.parse_status == "valid"
        assert result.field_lengths == expected_lengths


class TestParseStatus:
    """Test the parse states implemented in the current stage."""

    @pytest.mark.parametrize(
        ("source", "opcode", "operands_str", "operands"),
        [
            ("add a0", "add", "a0", ["a0"]),
            ("lw t0", "lw", "t0", ["t0"]),
            ("beq a0,a1", "beq", "a0,a1", ["a0", "a1"]),
        ],
    )
    def test_incomplete_operands(
        self,
        source,
        opcode,
        operands_str,
        operands,
    ):
        result = parse_asm_line(source)

        assert result.raw == source
        assert result.opcode == opcode
        assert result.operands_str == operands_str
        assert result.operands == operands
        assert result.parse_status == "incomplete_operands"

    def test_unknown_opcode(self):
        source = "custom_op x1,x2,x3 # keep this comment"
        result = parse_asm_line(source)

        assert result.raw == source
        assert result.label is None
        assert result.opcode == "custom_op"
        assert result.operands_str == "x1,x2,x3"
        assert result.operands == ["x1", "x2", "x3"]
        assert result.comment == "keep this comment"
        assert result.parse_status == "unknown_opcode"

    @pytest.mark.parametrize(
        "source",
        [
            "li,,a0,,3",
            "sw ra,,28(sp)",
            "main::",
            "this is not asm",
            "add a0 a1 a2",
            "2byte 1",
            '.asciz "unterminated',
            "lw a0, 8(sp",
        ],
    )
    def test_malformed_line_is_preserved(self, source):
        result = parse_asm_line(source)

        assert result.raw == source
        assert result.parse_status == "malformed"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
