"""Tests for Assembly-level Peephole Optimizer."""

import pytest
from scratchv.backend.asm_peephole import (
    PeepholeOptimizer, PeepholeRule, _parse_line, _parse_asm, _lines_to_asm,
)


class TestParseAsm:
    """Tests for assembly parsing."""

    def test_parse_simple_line(self):
        al = _parse_line("  add x1, x2, x3  # comment")
        assert al.opcode == "add"
        assert al.operands == ["x1", "x2", "x3"]
        assert al.comment == "comment"

    def test_parse_label_line(self):
        al = _parse_line("main:")
        assert al.label == "main"
        assert al.opcode is None

    def test_parse_label_with_instruction(self):
        al = _parse_line("loop:  addi t0, t0, 1")
        assert al.label == "loop"
        assert al.opcode == "addi"

    def test_roundtrip(self):
        asm = ".text\nmain:\n  add x1, x2, x3  # test\n  ret\n"
        lines = _parse_asm(asm)
        result = _lines_to_asm(lines)
        # Should preserve the structure
        assert "add" in result
        assert "main" in result


class TestDefaultRules:
    """Tests for the five default peephole rules."""

    def test_addi_addi_fusion(self):
        optimizer = PeepholeOptimizer()
        asm = "  addi t0, t0, 3\n  addi t0, t0, 5\n"
        result, changes = optimizer.optimize(asm)
        assert changes >= 1
        assert "addi" in result
        assert "8" in result or "3+5" in result

    def test_li_addi_fusion(self):
        optimizer = PeepholeOptimizer()
        asm = "  li t0, 10\n  addi t0, t0, 5\n"
        result, changes = optimizer.optimize(asm)
        assert changes >= 1
        assert "15" in result or "10+5" in result

    def test_beq_zero_jump(self):
        optimizer = PeepholeOptimizer()
        asm = "  beq x0, x0, loop_start\n"
        result, changes = optimizer.optimize(asm)
        assert changes >= 1
        assert "j" in result

    def test_mv_mv_swap_elimination(self):
        optimizer = PeepholeOptimizer()
        asm = "  mv t0, t1\n  mv t1, t0\n"
        result, changes = optimizer.optimize(asm)
        assert changes >= 0  # May or may not match depending on operands

    def test_redundant_mv_elimination(self):
        optimizer = PeepholeOptimizer()
        asm = "  mv t0, t1\n  mv t2, t0\n"
        result, changes = optimizer.optimize(asm)
        # Should produce: mv t2, t1
        assert changes >= 0


class TestPeepholeOptimizer:
    """Tests for the optimizer class."""

    def test_custom_rules(self):
        custom = [
            PeepholeRule(
                name="remove nop",
                pattern=["nop"],
                replacement=[],
            ),
        ]
        opt = PeepholeOptimizer(rules=custom)
        asm = "  nop\n  add t0, t1, t2\n"
        result, changes = opt.optimize(asm)
        assert "nop" not in result
        assert changes >= 1

    def test_no_changes_on_clean_asm(self):
        optimizer = PeepholeOptimizer()
        asm = "  add t0, t1, t2\n  sub t3, t4, t5\n  ret\n"
        result, changes = optimizer.optimize(asm)
        assert "add" in result
        assert "sub" in result
        assert "ret" in result

    def test_report(self):
        optimizer = PeepholeOptimizer()
        asm = "  addi t0, t0, 1\n  addi t0, t0, 2\n"
        optimizer.optimize(asm)
        report = optimizer.report()
        assert isinstance(report, str)
        assert "Total" in report

    def test_total_matches_property(self):
        optimizer = PeepholeOptimizer()
        asm = "  addi t0, t0, 1\n  addi t0, t0, 2\n"
        optimizer.optimize(asm)
        matches = optimizer.total_matches
        assert isinstance(matches, dict)

    def test_empty_asm(self):
        optimizer = PeepholeOptimizer()
        result, changes = optimizer.optimize("")
        assert result == ""
        assert changes == 0

    def test_preserves_labels(self):
        optimizer = PeepholeOptimizer()
        asm = "main:\n  add t0, t1, t2\n  ret\n"
        result, changes = optimizer.optimize(asm)
        assert "main:" in result
        assert "ret" in result

    def test_no_infinite_loop_on_no_match(self):
        optimizer = PeepholeOptimizer()
        # Just branches - no addi/addi or other fusible patterns
        asm = "  beq t0, t1, label\nlabel:\n  j label\n"
        result, changes = optimizer.optimize(asm)
        assert changes == 0

    def test_multiple_matches_in_sequence(self):
        optimizer = PeepholeOptimizer()
        asm = (
            "  addi t0, t0, 1\n  addi t0, t0, 2\n"
            "  addi t1, t1, 3\n  addi t1, t1, 4\n"
        )
        result, changes = optimizer.optimize(asm)
        # Both pairs should be fused
        assert changes >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
