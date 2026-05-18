"""Tests for RISC-V Instruction Counter."""

import os
import tempfile

import pytest
from scratchv.backend.inst_counter import (
    count_instructions, format_table, compare_files, ComparisonResult,
    _extract_opcode, _classify_opcode,
)


class TestExtractOpcode:
    """Tests for opcode extraction from assembly lines."""

    def test_extract_add(self):
        assert _extract_opcode("  add x1, x2, x3") == "add"

    def test_extract_with_comment(self):
        assert _extract_opcode("  lw a0, 0(sp)  # load arg") == "lw"

    def test_extract_label_line(self):
        assert _extract_opcode("main:") is None

    def test_extract_pure_comment(self):
        assert _extract_opcode("# this is a comment") is None

    def test_extract_empty(self):
        assert _extract_opcode("") is None

    def test_extract_directive(self):
        assert _extract_opcode(".text") is None
        assert _extract_opcode(".globl main") is None

    def test_extract_pseudo(self):
        assert _extract_opcode("  li t0, 42") == "li"
        assert _extract_opcode("  mv t1, t0") == "mv"


class TestClassifyOpcode:
    """Tests for opcode classification."""

    def test_alu(self):
        assert _classify_opcode("add") == "ALU"
        assert _classify_opcode("sub") == "ALU"
        assert _classify_opcode("mul") == "ALU"
        assert _classify_opcode("lui") == "ALU"

    def test_mem(self):
        assert _classify_opcode("lw") == "MEM"
        assert _classify_opcode("sw") == "MEM"

    def test_branch(self):
        assert _classify_opcode("beq") == "BRANCH"
        assert _classify_opcode("bne") == "BRANCH"

    def test_jump(self):
        assert _classify_opcode("j") == "JUMP"
        assert _classify_opcode("jal") == "JUMP"
        assert _classify_opcode("ret") == "JUMP"

    def test_pseudo(self):
        assert _classify_opcode("li") == "PSEUDO"
        assert _classify_opcode("mv") == "PSEUDO"
        assert _classify_opcode("call") == "PSEUDO"

    def test_misc(self):
        assert _classify_opcode("unknown_op") == "MISC"


class TestCountInstructions:
    """Tests for the count_instructions function."""

    @property
    def sample_asm(self):
        return (
            ".text\n"
            "main:\n"
            "  add x1, x2, x3\n"
            "  addi x1, x1, 1\n"
            "  sub x2, x1, x3\n"
            "  lw x4, 0(sp)\n"
            "  sw x4, 4(sp)\n"
            "  beq x1, x2, exit\n"
            "  j main\n"
            "exit:\n"
            "  ret\n"
            "  li t0, 42\n"
            "  mv t1, t0\n"
        )

    def test_count_categories_present(self):
        counts = count_instructions(self.sample_asm)
        for cat in ["ALU", "MEM", "BRANCH", "JUMP", "PSEUDO", "MISC"]:
            assert cat in counts

    def test_count_alu(self):
        counts = count_instructions(self.sample_asm)
        # add, addi, sub = 3 ALU
        assert counts["ALU"] == 3

    def test_count_mem(self):
        counts = count_instructions(self.sample_asm)
        # lw, sw = 2 MEM
        assert counts["MEM"] == 2

    def test_count_branch(self):
        counts = count_instructions(self.sample_asm)
        # beq = 1 BRANCH
        assert counts["BRANCH"] == 1

    def test_count_jump(self):
        counts = count_instructions(self.sample_asm)
        # j, ret = 2 JUMP
        assert counts["JUMP"] == 2

    def test_count_pseudo(self):
        counts = count_instructions(self.sample_asm)
        # li, mv = 2 PSEUDO
        assert counts["PSEUDO"] == 2

    def test_detailed_counts(self):
        counts = count_instructions(self.sample_asm)
        detailed = counts.get("_detailed")
        assert detailed is not None
        assert len(detailed) > 0

    def test_total_matches(self):
        counts = count_instructions(self.sample_asm)
        total = sum(v for k, v in counts.items()
                    if not k.startswith("_") and isinstance(v, int))
        # 3 ALU + 2 MEM + 1 BRANCH + 2 JUMP + 2 PSEUDO = 10
        # (.text directive is excluded)
        assert total >= 10

    def test_empty_asm(self):
        counts = count_instructions("")
        for cat in ["ALU", "MEM", "BRANCH", "JUMP", "PSEUDO", "MISC"]:
            assert counts[cat] == 0

    def test_only_comments(self):
        counts = count_instructions("# comment\n# another comment\n")
        total = sum(v for k, v in counts.items()
                    if not k.startswith("_") and isinstance(v, int))
        assert total == 0


class TestFormatTable:
    """Tests for table formatting."""

    def test_format_produces_string(self):
        counts = count_instructions("  add x1, x2, x3\n")
        table = format_table(counts)
        assert isinstance(table, str)
        assert "TOTAL" in table
        assert "ALU" in table

    def test_format_with_detailed(self):
        counts = count_instructions("  add x1, x2, x3\n  lw x4, 0(sp)\n")
        table = format_table(counts)
        assert "add" in table or "Per-instruction" in table


class TestCompareFiles:
    """Tests for file comparison."""

    def test_compare_two_files(self):
        f1 = tempfile.NamedTemporaryFile(
            mode="w", suffix=".s", delete=False)
        f1.write(".text\nmain:\n  add x1, x2, x3\n  lw x4, 0(sp)\n  ret\n")
        f1_path = f1.name
        f2 = tempfile.NamedTemporaryFile(
            mode="w", suffix=".s", delete=False)
        f2.write(".text\nmain:\n  add x1, x2, x3\n  div x5, x6, x7\n  ret\n")
        f2_path = f2.name

        try:
            result = compare_files([f1_path, f2_path])
            assert isinstance(result, ComparisonResult)
            assert len(result.files) == 2
            assert len(result.counts) == 2
        finally:
            os.unlink(f1_path)
            os.unlink(f2_path)


class TestHtmlReport:
    """Tests for HTML report generation."""

    def test_generate_html_file(self):
        from scratchv.backend.inst_counter import generate_html_report

        counts = count_instructions("  add x1, x2, x3\n  lw x4, 0(sp)\n")
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            generate_html_report(counts, output_path)
            with open(output_path, "r") as f:
                content = f.read()
            assert "<html" in content or "<table" in content
            assert "ALU" in content
            assert "MEM" in content
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
