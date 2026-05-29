"""Tests for RISC-V Assembly Beautifier."""

import pytest
from scratchv.backend.asm_beautifier import (
    beautify_asm, _parse_line, _gen_comment,
)


class TestParseLine:
    """Tests for line parsing."""

    def test_parse_simple_instruction(self):
        result = _parse_line("  add x1, x2, x3  # test")
        assert result["opcode"] == "add"
        assert result["operands"] == ["x1", "x2", "x3"]
        assert result["comment"] == "test"

    def test_parse_labeled_instruction(self):
        result = _parse_line("main:  addi sp, sp, -16")
        assert result["label"] == "main"
        assert result["opcode"] == "addi"
        assert result["operands"] == ["sp", "sp", "-16"]

    def test_parse_label_only(self):
        result = _parse_line("loop_start:")
        assert result["label"] == "loop_start"
        assert result["opcode"] is None

    def test_parse_comment_only(self):
        result = _parse_line("# this is a comment")
        assert result["opcode"] is None
        assert result["comment"] == "this is a comment"

    def test_parse_directive(self):
        result = _parse_line(".text")
        assert result["opcode"] == ".text"

    def test_parse_empty_line(self):
        result = _parse_line("")
        assert result["opcode"] is None
        assert result["label"] is None

    def test_parse_memory_operand(self):
        result = _parse_line("  lw x1, 8(sp)")
        assert result["operands"] == ["x1", "8(sp)"]


class TestGenComment:
    """Tests for semantic comment generation."""

    def test_add_comment(self):
        comment = _gen_comment("add", ["x1", "x2", "x3"])
        assert "x1" in comment
        assert "x2" in comment
        assert "x3" in comment

    def test_li_comment(self):
        comment = _gen_comment("li", ["t0", "42"])
        assert "42" in comment

    def test_lw_comment(self):
        comment = _gen_comment("lw", ["a0", "0(sp)"])
        assert "MEM" in comment
        assert "sp" in comment

    def test_mv_comment(self):
        comment = _gen_comment("mv", ["t1", "t0"])
        assert "t1" in comment
        assert "t0" in comment

    def test_j_comment(self):
        comment = _gen_comment("j", ["loop_start"])
        assert "goto" in comment

    def test_beq_comment(self):
        comment = _gen_comment("beq", ["t0", "t1", "label"])
        assert "==" in comment
        assert "label" in comment

    def test_unknown_opcode(self):
        comment = _gen_comment("custom_inst", ["x1", "x2"])
        # Should still produce something (opcode name)
        assert "custom_inst" in comment

    def test_nop_comment(self):
        comment = _gen_comment("nop", [])
        assert "no operation" in comment.lower()


class TestBeautifyAsm:
    """Integration tests for the full beautifier."""

    def test_basic_formatting(self):
        asm = ".text\nmain:\n  add x1, x2, x3\n"
        result = beautify_asm(asm, align=True, add_comments=True)
        assert ".text" in result
        assert "main:" in result
        assert "add" in result

    def test_no_comments_mode(self):
        asm = ".text\nmain:\n  add x1, x2, x3\n"
        result = beautify_asm(asm, align=True, add_comments=False)
        assert "=" in result.lower() or "add" in result.lower()

    def test_empty_input(self):
        result = beautify_asm("")
        assert result == "\n"

    def test_section_header_inserted(self):
        asm = ".text\n  add x1, x2, x3\n.data\n  .word 42\n"
        result = beautify_asm(asm, align=True, add_comments=True)
        assert "CODE SECTION" in result or "code" in result.lower()
        assert "DATA SECTION" in result or "data" in result.lower()

    def test_function_header_inserted(self):
        asm = "foo:\n  li a0, 1\n  ret\n"
        result = beautify_asm(asm, align=True, add_comments=True)
        assert "foo" in result
        assert "Function" in result or "---" in result

    def test_preserves_original_comment(self):
        asm = "# original comment\n  add x1, x2, x3\n"
        result = beautify_asm(asm, align=True, add_comments=True)
        assert "original comment" in result

    def test_pseudo_instruction_li(self):
        asm = "  li t0, 100\n"
        result = beautify_asm(asm, align=True, add_comments=True)
        assert "t0" in result
        assert "100" in result

    def test_complex_sequence(self):
        asm = (
            ".text\n"
            "main:\n"
            "  addi sp, sp, -16\n"
            "  sw ra, 12(sp)\n"
            "  li a0, 5\n"
            "  li a1, 3\n"
            "  jal multiply\n"
            "  lw ra, 12(sp)\n"
            "  addi sp, sp, 16\n"
            "  ret\n"
        )
        result = beautify_asm(asm, align=True, add_comments=True)
        assert "sp" in result
        assert "ret" in result
        assert "li" in result
        assert "jal" in result


class TestCli:
    """Test CLI behavior (import only, don't invoke argparse)."""

    def test_main_importable(self):
        from scratchv.backend.asm_beautifier import main
        assert callable(main)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
