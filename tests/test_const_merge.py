"""Tests for Constant Load Merge Optimizer."""

import pytest
from scratchv.backend._asm_parser import (
    ParsedAsmLine, canonical_reg, is_integer_reg,
)
from scratchv.backend.const_merge import (
    AsmInst, ConstantMergeStats, _insts_to_asm, _parse_asm,
    merge_constants, merge_constants_detailed,
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

    def test_uses_shared_parser_representation(self):
        insts = _parse_asm(".text\n  lw t0, 16(sp) # load")
        assert all(isinstance(inst, ParsedAsmLine) for inst in insts)
        assert insts[0].is_directive
        assert insts[1].operands == ["t0", "16(sp)"]
        assert insts[1].comment == "load"


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
        assert result == asm

    @pytest.mark.parametrize(
        "asm",
        [
            "\n\n  add t0, t1, t2\n\n",
            "   \n\t\n",
            "   # indented comment\n",
        ],
    )
    def test_no_change_preserves_whitespace_exactly(self, asm):
        result, changes = merge_constants(asm)
        assert changes == 0
        assert result == asm

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
        assert "2048" in result

    def test_rv32_result_is_normalized_to_signed_value(self):
        asm = "  lui t0, 0x80000\n  addi t0, t0, 0\n"
        result, changes = merge_constants(asm)
        assert changes == 1
        assert "li t0, -2147483648" in result

    def test_negative_hex_immediates(self):
        asm = "  lui t0, -0x1\n  addi t0, t0, -0x1\n"
        result, changes = merge_constants(asm)
        assert changes == 1
        assert "li t0, -4097" in result

    @pytest.mark.parametrize("imm", ["0x100000", "-0x80001"])
    def test_out_of_range_lui_is_not_truncated(self, imm):
        asm = f"  lui t0, {imm}\n  addi t0, t0, 1\n"
        result, changes = merge_constants(asm)
        assert changes == 0
        assert imm in result

    @pytest.mark.parametrize("imm", ["0x1000", "-2049"])
    def test_out_of_range_addi_is_not_truncated(self, imm):
        asm = f"  lui t0, 1\n  addi t0, t0, {imm}\n"
        result, changes = merge_constants(asm)
        assert changes == 0
        assert imm in result

    def test_relocation_expression_is_not_merged(self):
        asm = "  lui t0, %hi(symbol)\n  addi t0, t0, %lo(symbol)\n"
        result, changes = merge_constants(asm)
        assert changes == 0
        assert "%hi(symbol)" in result
        assert "%lo(symbol)" in result

    def test_different_addi_source_is_not_merged(self):
        asm = "  lui t0, 1\n  addi t0, t1, 2\n"
        result, changes = merge_constants(asm)
        assert changes == 0
        assert "lui" in result and "addi" in result

    def test_comment_and_blank_between_pair_are_preserved(self):
        asm = "  lui t0, 1 # upper\n# keep me\n\n  addi t0, t0, 2 # lower\n"
        result, changes = merge_constants(asm)
        assert changes == 1
        assert "li t0, 4098" in result
        assert "# keep me" in result
        assert "upper" in result and "lower" in result

    def test_intervening_label_prevents_merge(self):
        asm = "  lui t0, 1\nL1:\n  addi t0, t0, 2\n"
        result, changes = merge_constants(asm)
        assert changes == 0
        assert "lui" in result and "addi" in result

    def test_label_on_lui_is_preserved(self):
        asm = "L0: lui t0, 1\n  addi t0, t0, 2\n"
        result, changes = merge_constants(asm)
        assert changes == 1
        assert "L0:  li t0, 4098" in result

    def test_merge_with_abi_and_x_register_aliases(self):
        asm = "  lui t0, 1\n  addi x5, t0, 2\n"
        result, changes = merge_constants(asm)
        assert changes == 1
        assert "li t0, 4098" in result

    @pytest.mark.parametrize("register", ["foo", "x32", "v0", "f0"])
    def test_invalid_or_non_integer_register_is_not_merged(self, register):
        asm = f"  lui {register}, 1\n  addi {register}, {register}, 2\n"
        result, stats = merge_constants_detailed(asm)
        assert result == asm
        assert stats.candidate_pairs == 1
        assert stats.merged_pairs == 0
        assert stats.total_changes == 0

    def test_alias_clobber_prevents_redundant_lui_removal(self):
        asm = "  lui t0, 1\n  add x5, a0, a1\n  lui t0, 1\n"
        result, changes = merge_constants(asm)
        assert changes == 0
        assert result.count("lui") == 2

    def test_different_lui_value_is_not_redundant(self):
        asm = "  lui t0, 1\n  add a0, a1, a2\n  lui t0, 2\n"
        result, changes = merge_constants(asm)
        assert changes == 0
        assert sum(inst.opcode == "lui" for inst in _parse_asm(result)) == 2

    def test_redundant_lui_does_not_cross_label(self):
        asm = "  lui t0, 1\nL1:\n  lui t0, 1\n"
        result, changes = merge_constants(asm)
        assert changes == 0
        assert sum(inst.opcode == "lui" for inst in _parse_asm(result)) == 2

    @pytest.mark.parametrize(
        "boundary",
        [
            "  beq a0, zero, L1",
            "  j L1",
            "  call helper",
            "  jal ra, helper",
            "  jalr ra, 0(t1)",
            "  ret",
            "  jr ra",
        ],
    )
    def test_redundant_lui_does_not_cross_control_flow(self, boundary):
        asm = f"  lui t0, 1\n{boundary}\n  lui t0, 1\n"
        result, changes = merge_constants(asm)
        assert changes == 0
        assert sum(inst.opcode == "lui" for inst in _parse_asm(result)) == 2

    def test_unknown_instruction_clears_lui_state(self):
        asm = "  lui t0, 1\n  custom.op a0, a1\n  lui t0, 1\n"
        result, changes = merge_constants(asm)
        assert changes == 0
        assert sum(inst.opcode == "lui" for inst in _parse_asm(result)) == 2

    def test_directive_clears_lui_state(self):
        asm = "  lui t0, 1\n.section .text\n  lui t0, 1\n"
        result, changes = merge_constants(asm)
        assert changes == 0
        assert sum(inst.opcode == "lui" for inst in _parse_asm(result)) == 2

    def test_removed_redundant_lui_preserves_comment(self):
        asm = (
            "  lui t0, 1\n"
            "  add a0, a1, a2\n"
            "  lui t0, 1 # duplicate high bits\n"
        )
        result, changes = merge_constants(asm)
        assert changes == 1
        assert "duplicate high bits" in result
        assert sum(inst.opcode == "lui" for inst in _parse_asm(result)) == 1

    def test_redundant_lui_recognizes_aliases(self):
        asm = "  lui t0, 1\n  add a0, a1, a2\n  lui x5, 1\n"
        result, changes = merge_constants(asm)
        assert changes == 1
        assert sum(inst.opcode == "lui" for inst in _parse_asm(result)) == 1

    def test_register_canonicalization(self):
        assert canonical_reg("t0") == "x5"
        assert canonical_reg("fp") == "x8"
        assert canonical_reg("s0") == "x8"
        assert canonical_reg("a0") == "x10"
        assert canonical_reg("X31") == "x31"
        assert is_integer_reg("t0")
        assert is_integer_reg("X31")
        assert not is_integer_reg("x32")
        assert not is_integer_reg("foo")

    def test_invalid_register_lui_is_not_tracked_as_redundant(self):
        asm = "  lui foo, 1\n  lui foo, 1\n"
        result, changes = merge_constants(asm)
        assert result == asm
        assert changes == 0

    def test_fixed_point_exposes_pair_after_redundant_lui(self):
        asm = "  lui t0, 1\n  lui x5, 1\n  addi t0, x5, 2\n"
        result, stats = merge_constants_detailed(asm)
        assert isinstance(stats, ConstantMergeStats)
        assert stats.candidate_pairs == 1
        assert stats.redundant_lui_removed == 1
        assert stats.merged_pairs == 1
        assert stats.total_changes == 2
        assert stats.iterations == 2
        assert "li t0, 4098" in result
        assert not any(inst.opcode == "lui" for inst in _parse_asm(result))

    def test_detailed_statistics_distinguish_rule_types(self):
        asm = (
            "  lui t0, 1\n"
            "  addi t0, t0, 2\n"
            "  lui t1, 3\n"
            "  add a0, a1, a2\n"
            "  lui x6, 3\n"
        )
        _, stats = merge_constants_detailed(asm)
        assert stats.candidate_pairs == 1
        assert stats.merged_pairs == 1
        assert stats.redundant_lui_removed == 1
        assert stats.total_changes == 2

    def test_legacy_api_returns_detailed_total(self):
        asm = "  lui t0, 1\n  lui x5, 1\n  addi t0, x5, 2\n"
        detailed_result, stats = merge_constants_detailed(asm)
        legacy_result, changes = merge_constants(asm)
        assert legacy_result == detailed_result
        assert changes == stats.total_changes == 2

    def test_optimization_is_idempotent(self):
        asm = "  lui t0, 1\n  lui x5, 1\n  addi t0, x5, 2\n"
        once, first_stats = merge_constants_detailed(asm)
        twice, second_stats = merge_constants_detailed(once)
        assert first_stats.total_changes == 2
        assert twice == once
        assert second_stats.total_changes == 0

    def test_optimization_is_textually_idempotent_with_trailing_separator(self):
        asm = "  lui t0, 1\n# between\n\n  addi t0, t0, 2\n"
        once, first_stats = merge_constants_detailed(asm)
        twice, second_stats = merge_constants_detailed(once)
        assert first_stats.total_changes == 1
        assert twice == once
        assert second_stats.total_changes == 0

    @pytest.mark.parametrize(
        "asm",
        [
            "  lui t0, 1\n  addi t1, t0, 2\n",
            "  lui t0, %hi(symbol)\n  addi t0, t0, %lo(symbol)\n",
            "  lui t0, 0x100000\n  addi t0, t0, 1\n",
        ],
    )
    def test_candidates_include_structural_pairs_rejected_for_safety(self, asm):
        result, stats = merge_constants_detailed(asm)
        assert result == asm
        assert stats.candidate_pairs == 1
        assert stats.merged_pairs == 0

    def test_max_iterations_zero_disables_transformations(self):
        asm = "  lui t0, 1\n  addi t0, t0, 2\n"
        result, stats = merge_constants_detailed(asm, max_iterations=0)
        assert stats.iterations == 0
        assert stats.total_changes == 0
        assert "lui" in result and "addi" in result


class TestCli:
    """Test CLI behavior."""

    def test_main_importable(self):
        from scratchv.backend.const_merge import main
        assert callable(main)

    def test_cli_writes_output_and_verbose_stats(
        self, tmp_path, capsys, monkeypatch,
    ):
        from scratchv.backend.const_merge import main

        source = tmp_path / "input.s"
        output = tmp_path / "output.s"
        source.write_text("  lui t0, 1\n  addi t0, t0, 2\n")
        monkeypatch.setattr(
            "sys.argv",
            ["const_merge", str(source), "-o", str(output), "-v"],
        )

        main()

        captured = capsys.readouterr()
        assert "candidate pairs: 1" in captured.err
        assert "merged lui+addi pairs: 1" in captured.err
        assert "redundant lui removed: 0" in captured.err
        assert "total transformations: 1" in captured.err
        assert "li t0, 4098" in output.read_text()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
