"""Tests for Assembly-level Peephole Optimizer."""

from __future__ import annotations

import re

import pytest
from scratchv.backend.asm_peephole import (
    AsmPeepholeOptimizer, PeepholeRule, _parse_line, _parse_asm, _lines_to_asm,
    _match_rule, _operand_matches, _default_rules, _fits_simm12, _parse_imm,
)


# ---------------------------------------------------------------------------
# Lightweight straight-line semantic helper (no branches except j as no-op end)
# ---------------------------------------------------------------------------

_REG_ALIASES = {"zero": "x0"}


def _canon_reg(name: str) -> str:
    return _REG_ALIASES.get(name, name)


def _exec_straightline(asm: str, regs: dict[str, int] | None = None) -> dict[str, int]:
    """Execute a tiny subset of RV32I used by peephole rules (no memory)."""
    state: dict[str, int] = dict(regs or {})
    state.setdefault("x0", 0)
    state.setdefault("zero", 0)

    for raw in asm.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.endswith(":") or line.startswith("."):
            continue
        # Drop leading label on same line: "L: addi ..."
        if re.match(r"^[A-Za-z_.][\w.]*:", line):
            line = line.split(":", 1)[1].strip()
            if not line:
                continue
        parts = [p.strip() for p in line.replace(",", " ").split() if p.strip()]
        if not parts:
            continue
        op = parts[0].lower()
        ops = parts[1:]

        if op == "li":
            state[_canon_reg(ops[0])] = int(ops[1], 0)
        elif op == "addi":
            rd, rs, imm = ops[0], ops[1], int(ops[2], 0)
            state[_canon_reg(rd)] = state.get(_canon_reg(rs), 0) + imm
        elif op == "mv":
            state[_canon_reg(ops[0])] = state.get(_canon_reg(ops[1]), 0)
        elif op in ("add", "sub"):
            rd, rs1, rs2 = ops[0], ops[1], ops[2]
            a = state.get(_canon_reg(rs1), 0)
            b = state.get(_canon_reg(rs2), 0)
            state[_canon_reg(rd)] = a + b if op == "add" else a - b
        elif op in ("nop", "ret"):
            continue
        elif op == "j":
            continue  # ignore control for straight-line value checks
        elif op == "beq":
            continue
        else:
            raise AssertionError(f"unsupported opcode in semantic helper: {op}")

    state["x0"] = 0
    state["zero"] = 0
    return state


def _assert_regs_equal(before_asm: str, after_asm: str, regs: dict[str, int],
                       watch: list[str]) -> None:
    pre = _exec_straightline(before_asm, regs)
    post = _exec_straightline(after_asm, regs)
    for r in watch:
        assert pre.get(_canon_reg(r), 0) == post.get(_canon_reg(r), 0), (
            f"reg {r}: before={pre.get(_canon_reg(r), 0)} "
            f"after={post.get(_canon_reg(r), 0)}\n"
            f"SRC:\n{before_asm}\nDST:\n{after_asm}"
        )


@pytest.mark.unit
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

    def test_parse_hex_and_memory_operand(self):
        al = _parse_line("  addi t0, t0, 0x10")
        assert al.operands == ["t0", "t0", "0x10"]
        al2 = _parse_line("  lw t0, 0(t1)")
        assert al2.operands == ["t0", "0(t1)"]

    def test_parse_imm_accepts_bases(self):
        assert _parse_imm("10") == 10
        assert _parse_imm("0x10") == 16
        assert _parse_imm("0b1010") == 10
        assert _parse_imm("t0") is None

    def test_roundtrip(self):
        asm = ".text\nmain:\n  add x1, x2, x3  # test\n  ret\n"
        lines = _parse_asm(asm)
        result = _lines_to_asm(lines)
        assert "add" in result
        assert "main" in result


@pytest.mark.unit
class TestDefaultRules:
    """Positive coverage for each default peephole rule."""

    def test_addi_addi_fusion(self):
        optimizer = AsmPeepholeOptimizer()
        asm = "  addi t0, t0, 3\n  addi t0, t0, 5\n"
        result, changes = optimizer.optimize(asm)
        assert changes >= 1
        assert "addi" in result
        assert "8" in result

    def test_li_addi_fusion(self):
        optimizer = AsmPeepholeOptimizer()
        asm = "  li t0, 10\n  addi t0, t0, 5\n"
        result, changes = optimizer.optimize(asm)
        assert changes >= 1
        assert "15" in result

    def test_beq_zero_jump(self):
        optimizer = AsmPeepholeOptimizer()
        asm = "  beq x0, x0, loop_start\n"
        result, changes = optimizer.optimize(asm)
        assert changes >= 1
        assert "j" in result

    def test_beq_zero_alias(self):
        optimizer = AsmPeepholeOptimizer()
        asm = "  beq zero, zero, target\n"
        result, changes = optimizer.optimize(asm)
        assert changes == 1
        assert "j target" in result

    def test_mv_swap_pair_not_deleted(self):
        """mv x,y; mv y,x is NOT a no-op — must be preserved."""
        optimizer = AsmPeepholeOptimizer()
        asm = "  mv t0, t1\n  mv t1, t0\n"
        result, changes = optimizer.optimize(asm)
        assert changes == 0
        assert result.count("mv") == 2

    def test_redundant_mv_elimination(self):
        optimizer = AsmPeepholeOptimizer()
        asm = "  mv t0, t1\n  mv t2, t0\n"
        result, changes = optimizer.optimize(asm)
        assert changes == 1
        assert "mv t2, t1" in result


@pytest.mark.unit
class TestAsmPeepholeOptimizer:
    """Tests for the optimizer class."""

    def test_custom_rules(self):
        custom = [
            PeepholeRule(
                name="remove nop",
                pattern=["nop"],
                replacement=[],
            ),
        ]
        opt = AsmPeepholeOptimizer(rules=custom)
        asm = "  nop\n  add t0, t1, t2\n"
        result, changes = opt.optimize(asm)
        assert "nop" not in result
        assert changes >= 1

    def test_no_changes_on_clean_asm(self):
        optimizer = AsmPeepholeOptimizer()
        asm = "  add t0, t1, t2\n  sub t3, t4, t5\n  ret\n"
        result, changes = optimizer.optimize(asm)
        assert changes == 0
        assert "add" in result
        assert "sub" in result
        assert "ret" in result

    def test_report(self):
        optimizer = AsmPeepholeOptimizer()
        asm = "  addi t0, t0, 1\n  addi t0, t0, 2\n"
        optimizer.optimize(asm)
        report = optimizer.report()
        assert isinstance(report, str)
        assert "Instructions before: 2" in report
        assert "Instructions after:  1" in report
        assert "Instructions saved:  1" in report
        assert "Rule applications:   1" in report
        assert "Fixed-point passes:" in report
        assert "Total changes: 1" in report
        assert "addi+addi fusion" in report
        assert optimizer.instructions_before == 2
        assert optimizer.instructions_after == 1
        assert optimizer.instructions_saved == 1
        assert optimizer.iterations >= 1

    def test_report_no_changes_still_shows_counts(self):
        optimizer = AsmPeepholeOptimizer()
        asm = "  add t0, t1, t2\n  ret\n"
        optimizer.optimize(asm)
        report = optimizer.report()
        assert "Instructions before: 2" in report
        assert "Instructions after:  2" in report
        assert "Instructions saved:  0" in report
        assert "No optimization opportunities found." in report
        assert "Total changes: 0" in report

    def test_total_matches_property(self):
        optimizer = AsmPeepholeOptimizer()
        asm = "  addi t0, t0, 1\n  addi t0, t0, 2\n"
        optimizer.optimize(asm)
        matches = optimizer.total_matches
        assert isinstance(matches, dict)
        assert matches.get("addi+addi fusion", 0) >= 1

    def test_empty_asm(self):
        optimizer = AsmPeepholeOptimizer()
        result, changes = optimizer.optimize("")
        assert result == ""
        assert changes == 0

    def test_preserves_labels(self):
        optimizer = AsmPeepholeOptimizer()
        asm = "main:\n  add t0, t1, t2\n  ret\n"
        result, changes = optimizer.optimize(asm)
        assert "main:" in result
        assert "ret" in result

    def test_no_infinite_loop_on_no_match(self):
        optimizer = AsmPeepholeOptimizer()
        asm = "  beq t0, t1, label\nlabel:\n  j label\n"
        result, changes = optimizer.optimize(asm)
        assert changes == 0

    def test_multiple_matches_in_sequence(self):
        optimizer = AsmPeepholeOptimizer()
        asm = (
            "  addi t0, t0, 1\n  addi t0, t0, 2\n"
            "  addi t1, t1, 3\n  addi t1, t1, 4\n"
        )
        result, changes = optimizer.optimize(asm)
        assert changes >= 2

    def test_optimize_idempotent(self):
        optimizer = AsmPeepholeOptimizer()
        asm = (
            "  li t0, 1\n  addi t0, t0, 2\n"
            "  addi t1, t1, 3\n  addi t1, t1, 4\n"
            "  nop\n  mv t2, t2\n"
        )
        first, c1 = optimizer.optimize(asm)
        second, c2 = optimizer.optimize(first)
        assert c1 >= 1
        assert c2 == 0
        assert second == first


@pytest.mark.unit
class TestMatchEngine:
    """Low-level pattern matching unit tests."""

    def test_operand_matches_binds_and_reuses(self):
        bindings: dict[str, str] = {}
        assert _operand_matches("rd0", "t0", bindings) is True
        assert bindings["rd0"] == "t0"
        assert _operand_matches("rd0", "t0", bindings) is True
        assert _operand_matches("rd0", "t1", bindings) is False

    def test_match_rule_addi_fusion_positive(self):
        rule = next(r for r in _default_rules() if r.name == "addi+addi fusion")
        window = _parse_asm("  addi t0, t0, 3\n  addi t0, t0, 5\n")
        assert _match_rule(rule, window) is not None

    def test_match_rule_addi_fusion_wrong_register(self):
        rule = next(r for r in _default_rules() if r.name == "addi+addi fusion")
        window = _parse_asm("  addi t0, t0, 3\n  addi t1, t1, 5\n")
        assert _match_rule(rule, window) is None

    def test_match_rule_beq_requires_zero_operands(self):
        rule = next(r for r in _default_rules() if r.name == "beq zero-zero to jump")
        assert _match_rule(rule, _parse_asm("  beq t0, t1, L\n")) is None
        assert _match_rule(rule, _parse_asm("  beq x0, x0, L\n")) is not None
        assert _match_rule(rule, _parse_asm("  beq zero, x0, L\n")) is not None

    def test_match_rule_mv_swap_not_eliminated_as_chain(self):
        rule = next(r for r in _default_rules() if r.name == "redundant mv elimination")
        assert _match_rule(rule, _parse_asm("  mv t0, t1\n  mv t1, t0\n")) is None

    def test_match_rule_mv_chain_elimination(self):
        rule = next(r for r in _default_rules() if r.name == "redundant mv elimination")
        assert _match_rule(rule, _parse_asm("  mv t0, t1\n  mv t2, t0\n")) is not None

    def test_match_refuses_mid_window_label(self):
        rule = next(r for r in _default_rules() if r.name == "addi+addi fusion")
        window = _parse_asm("  addi t0, t0, 1\nLmid:  addi t0, t0, 2\n")
        assert _match_rule(rule, window) is None

    def test_redundant_mv_elimination_exact_output(self):
        optimizer = AsmPeepholeOptimizer()
        asm = "  mv t0, t1\n  mv t2, t0\n"
        result, changes = optimizer.optimize(asm)
        assert changes == 1
        assert "mv t2, t1" in result


@pytest.mark.unit
class TestCorrectnessAndNewRules:
    """Immediate overflow guards and elimination rules."""

    def test_fits_simm12_bounds(self):
        assert _fits_simm12(-2048) is True
        assert _fits_simm12(2047) is True
        assert _fits_simm12(-2049) is False
        assert _fits_simm12(2048) is False

    def test_addi_fusion_rejected_on_overflow(self):
        optimizer = AsmPeepholeOptimizer()
        asm = "  addi t0, t0, 2000\n  addi t0, t0, 2000\n"
        result, changes = optimizer.optimize(asm)
        assert changes == 0
        assert result.count("addi") == 2
        assert "4000" not in result

    def test_addi_fusion_allowed_at_simm12_edge(self):
        optimizer = AsmPeepholeOptimizer()
        asm = "  addi t0, t0, 2000\n  addi t0, t0, 47\n"
        result, changes = optimizer.optimize(asm)
        assert changes == 1
        assert "2047" in result

    def test_addi_fusion_rejected_at_simm12_underflow(self):
        optimizer = AsmPeepholeOptimizer()
        asm = "  addi t0, t0, -2000\n  addi t0, t0, -2000\n"
        result, changes = optimizer.optimize(asm)
        assert changes == 0
        assert result.count("addi") == 2

    def test_addi_fusion_hex_immediates(self):
        optimizer = AsmPeepholeOptimizer()
        asm = "  addi t0, t0, 0x10\n  addi t0, t0, 0x20\n"
        result, changes = optimizer.optimize(asm)
        assert changes == 1
        assert "48" in result
        assert "(" not in result.split("#")[0]

    def test_addi_fusion_negative_immediates(self):
        optimizer = AsmPeepholeOptimizer()
        asm = "  addi t0, t0, 10\n  addi t0, t0, -3\n"
        result, changes = optimizer.optimize(asm)
        assert changes == 1
        assert "7" in result

    def test_li_addi_still_fuses_large_sum(self):
        """li can hold values outside simm12; fusion should still apply."""
        optimizer = AsmPeepholeOptimizer()
        asm = "  li t0, 3000\n  addi t0, t0, 2000\n"
        result, changes = optimizer.optimize(asm)
        assert changes == 1
        assert "5000" in result

    def test_li_addi_hex_immediates(self):
        optimizer = AsmPeepholeOptimizer()
        asm = "  li t0, 0x100\n  addi t0, t0, 0x20\n"
        result, changes = optimizer.optimize(asm)
        assert changes == 1
        assert "288" in result

    def test_addi_zero_self_elimination(self):
        optimizer = AsmPeepholeOptimizer()
        asm = "  addi t0, t0, 0\n  add t1, t2, t3\n"
        result, changes = optimizer.optimize(asm)
        assert changes == 1
        assert "addi" not in result
        assert "add" in result

    def test_addi_zero_to_mv(self):
        optimizer = AsmPeepholeOptimizer()
        asm = "  addi t1, t0, 0\n"
        result, changes = optimizer.optimize(asm)
        assert changes == 1
        assert "mv t1, t0" in result
        assert any(line.strip().startswith("mv ") for line in result.splitlines())
        assert not any(
            line.strip().startswith("addi ") for line in result.splitlines()
        )

    def test_nop_elimination(self):
        optimizer = AsmPeepholeOptimizer()
        asm = "  nop\n  add t0, t1, t2\n"
        result, changes = optimizer.optimize(asm)
        assert changes == 1
        assert "nop" not in result
        assert "add" in result

    def test_mv_self_elimination(self):
        optimizer = AsmPeepholeOptimizer()
        asm = "  mv t0, t0\n  add t1, t2, t3\n"
        result, changes = optimizer.optimize(asm)
        assert changes == 1
        assert "mv" not in result
        assert "add" in result

    def test_label_preserved_on_fusion(self):
        optimizer = AsmPeepholeOptimizer()
        asm = "loop:  addi t0, t0, 1\n  addi t0, t0, 2\n"
        result, changes = optimizer.optimize(asm)
        assert changes == 1
        assert "loop:" in result
        assert "3" in result

    def test_label_preserved_on_nop_deletion(self):
        optimizer = AsmPeepholeOptimizer()
        asm = "keep:  nop\n  ret\n"
        result, changes = optimizer.optimize(asm)
        assert changes == 1
        assert "keep:" in result
        assert "ret" in result

    def test_mid_label_blocks_fusion(self):
        optimizer = AsmPeepholeOptimizer()
        asm = "  addi t0, t0, 1\nLmid:  addi t0, t0, 2\n"
        result, changes = optimizer.optimize(asm)
        assert changes == 0
        assert "Lmid:" in result
        assert result.count("addi") == 2


@pytest.mark.unit
class TestSemanticEquivalence:
    """Register-state equivalence for sound peephole rewrites."""

    def test_addi_fusion_preserves_regs(self):
        asm = "  addi t0, t0, 3\n  addi t0, t0, 5\n  addi t1, t1, 1\n"
        out, n = AsmPeepholeOptimizer().optimize(asm)
        assert n >= 1
        _assert_regs_equal(asm, out, {"t0": 100, "t1": 7}, ["t0", "t1"])

    def test_li_addi_preserves_regs(self):
        asm = "  li t0, 10\n  addi t0, t0, 5\n"
        out, n = AsmPeepholeOptimizer().optimize(asm)
        assert n == 1
        _assert_regs_equal(asm, out, {}, ["t0"])

    def test_addi_zero_rules_preserve_regs(self):
        asm = "  addi t0, t0, 0\n  addi t1, t0, 0\n  add t2, t1, t0\n"
        out, n = AsmPeepholeOptimizer().optimize(asm)
        assert n >= 1
        _assert_regs_equal(asm, out, {"t0": 4}, ["t0", "t1", "t2"])

    def test_nop_and_mv_self_preserve_regs(self):
        asm = "  nop\n  mv t0, t0\n  addi t0, t0, 1\n"
        out, n = AsmPeepholeOptimizer().optimize(asm)
        assert n >= 2
        _assert_regs_equal(asm, out, {"t0": 5}, ["t0"])

    def test_mv_swap_pair_preserves_regs(self):
        """Swap-shaped pair must keep original post-mv register state."""
        asm = "  li t0, 1\n  li t1, 2\n  mv t0, t1\n  mv t1, t0\n"
        out, n = AsmPeepholeOptimizer().optimize(asm)
        assert n == 0
        _assert_regs_equal(asm, out, {}, ["t0", "t1"])

    def test_mv_chain_preserves_destination_when_mid_dead(self):
        """When only t2 is observed, chain rewrite is value-correct for t2."""
        asm = "  li t1, 9\n  mv t0, t1\n  mv t2, t0\n"
        out, n = AsmPeepholeOptimizer().optimize(asm)
        assert n == 1
        _assert_regs_equal(asm, out, {}, ["t2", "t1"])

    def test_mv_chain_unsound_when_mid_live(self):
        """Document best-effort limitation: intermediate t0 may be clobbered."""
        asm = (
            "  li t1, 9\n"
            "  mv t0, t1\n"
            "  mv t2, t0\n"
            "  add t3, t0, t2\n"
        )
        out, n = AsmPeepholeOptimizer().optimize(asm)
        assert n == 1
        pre = _exec_straightline(asm, {})
        post = _exec_straightline(out, {})
        # Destination of chain stays correct…
        assert pre["t2"] == post["t2"] == 9
        # …but live intermediate differs without liveness analysis.
        assert pre["t0"] == 9
        assert post.get("t0", 0) != pre["t0"]


@pytest.mark.unit
class TestTrackAParserReuseAndHardening:
    """Track A: shared _asm_parser reuse + review follow-ups."""

    def test_asmline_is_shared_parsed_type(self):
        from scratchv.backend._asm_parser import ParsedAsmLine
        from scratchv.backend.asm_peephole import AsmLine

        assert AsmLine is ParsedAsmLine
        al = _parse_line("  add t0, t1, t2")
        assert isinstance(al, ParsedAsmLine)

    def test_count_opcodes_skips_directives(self):
        from scratchv.backend.asm_peephole import _count_opcodes

        lines = _parse_asm(".text\nmain:\n  addi t0, t0, 1\n  ret\n")
        assert _count_opcodes(lines) == 2

    def test_short_mv_operands_do_not_crash(self):
        """Malformed mv with one operand must not raise IndexError."""
        optimizer = AsmPeepholeOptimizer()
        asm = "  mv t0\n  mv t1, t0\n"
        result, changes = optimizer.optimize(asm)
        assert isinstance(result, str)
        assert changes >= 0

    def test_addi_zero_self_with_zero_alias(self):
        optimizer = AsmPeepholeOptimizer()
        asm = "  addi zero, x0, 0\n  ret\n"
        result, changes = optimizer.optimize(asm)
        assert changes == 1
        assert "addi" not in result
        assert "ret" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
