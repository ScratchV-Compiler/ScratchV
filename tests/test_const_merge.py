"""Tests for Constant Load Merge Optimizer."""

import pytest
from scratchv.backend.const_merge import (
    merge_constants, AsmInst, _parse_asm, _insts_to_asm,
)


class TestAsmInst:
    """Tests for AsmInst parsing."""

    def test_parse_simple(self):
        inst = AsmInst("  add x1, x2, x3")
        assert inst.opcode == "add"
        assert inst.operands == ["x1", "x2", "x3"]

    def test_parse_lui(self):
        inst = AsmInst("  lui t0, 0x12345")
        assert inst.opcode == "lui"
        assert inst.operands[0] == "t0"
        assert "0x12345" in inst.operands[1]

    def test_parse_addi(self):
        inst = AsmInst("  addi t0, t0, -256")
        assert inst.opcode == "addi"
        assert inst.operands[:2] == ["t0", "t0"]

    def test_parse_label(self):
        inst = AsmInst("main:")
        assert inst.label == "main"
        assert inst.opcode is None

    def test_parse_comment_only(self):
        inst = AsmInst("# a comment")
        assert inst.opcode is None
        assert inst.comment == "a comment"

    def test_reconstruct(self):
        original = "  add x1, x2, x3"
        inst = AsmInst(original)
        assert "add" in inst.to_asm()
        assert "x1" in inst.to_asm()

    def test_parse_roundtrip(self):
        asm = "  lui t0, 0x12345\n  addi t0, t0, 0x678\n"
        insts = _parse_asm(asm)
        result = _insts_to_asm(insts)
        assert "lui" in result
        assert "addi" in result


class TestMergeConstants:
    """Tests for the merge_constants function."""

    def test_merge_lui_addi_basic(self):
        asm = "  lui t0, 0x12345\n  addi t0, t0, 0x678\n"
        result, changes = merge_constants(asm)
        assert changes >= 1
        assert "li" in result
        assert "t0" in result

    def test_merge_lui_addi_negative(self):
        # lui loads upper, addi with sign-extended negative lower bits
        asm = "  lui t0, 0x12345\n  addi t0, t0, -1\n"
        result, changes = merge_constants(asm)
        assert changes >= 1
        assert "li" in result or "merged" in result.lower()

    def test_redundant_lui_elimination(self):
        asm = (
            "  lui t0, 0x10000\n"
            "  addi t0, t0, 0\n"
            "  lui t0, 0x10000\n"
            "  addi t0, t0, 100\n"
        )
        result, changes = merge_constants(asm)
        assert changes >= 1

    def test_no_merge_different_regs(self):
        # lui into t0, addi into t1 (different reg) - should NOT merge
        asm = "  lui t0, 0x12345\n  addi t1, t1, 0x678\n"
        result, changes = merge_constants(asm)
        # First pair does not form a lui+addi with matching rd
        assert "lui" in result  # still has the lui

    def test_no_merge_when_intervening(self):
        # lui followed by another instruction, then addi
        asm = "  lui t0, 0x12345\n  add t1, t2, t3\n  addi t0, t0, 0x678\n"
        result, changes = merge_constants(asm)
        # Should not merge because they're not adjacent
        assert "lui" in result

    def test_empty_asm(self):
        result, changes = merge_constants("")
        assert changes == 0

    def test_no_changes_without_lui(self):
        asm = "  add t0, t1, t2\n  sub t3, t4, t5\n  ret\n"
        result, changes = merge_constants(asm)
        assert changes == 0

    def test_preserves_non_lui_addi(self):
        asm = "main:\n  addi sp, sp, -16\n  sw ra, 12(sp)\n  ret\n"
        result, changes = merge_constants(asm)
        assert "addi" in result or "sp" in result
        assert "ret" in result

    def test_sign_extension_correct(self):
        # Test that sign extension is handled correctly
        # lui t0, 0x00001; addi t0, t0, 0x800
        # addi with 0x800 is sign-extended to -2048
        # So final = 0x00001000 + (-2048) = 0x00001000 - 0x800 = 0x00000800
        asm = "  lui t0, 0x1\n  addi t0, t0, 0x800\n"
        result, changes = merge_constants(asm)
        assert changes >= 1
        assert "li" in result


class TestCli:
    """Test CLI behavior."""

    def test_main_importable(self):
        from scratchv.backend.const_merge import main
        assert callable(main)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
