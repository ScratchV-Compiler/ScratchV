"""Comprehensive tests for RISC-V Instruction Counter (Topic 12)."""

import json
import os
import tempfile
from pathlib import Path

import pytest
from scratchv.backend.inst_counter import (
    count_instructions,
    count_instructions_file,
    format_table,
    compare_files,
    compare_stats,
    analyze_instructions,
    analyze_file,
    render_text,
    render_json,
    render_markdown,
    render_html,
    render_chart,
    classify_opcode,
    InstructionStats,
    ComparisonResult,
    CategoryDelta,
    from_emulator_counts,
    _extract_opcode,
    _classify_opcode,
    _CATEGORY_ORDER,
    main,
)
from scratchv.compiler import CompilerConfig, CompilerDriver

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "inst_counter"


class TestExtractOpcode:
    """Tests for opcode extraction and parsing edge cases."""

    def test_extract_basic(self):
        assert _extract_opcode("  add x1, x2, x3") == "add"
        assert _extract_opcode("  lw a0, 0(sp)  # load arg") == "lw"

    def test_extract_labels(self):
        assert _extract_opcode("main:") is None
        assert _extract_opcode(".L1:") is None
        assert _extract_opcode("1:") is None
        assert _extract_opcode("name$part:") is None

    def test_extract_inline_labels(self):
        assert _extract_opcode("loop: addi t0, t0, -1") == "addi"
        assert _extract_opcode(".L1: bnez t0, .L1") == "bnez"
        assert _extract_opcode("10: nop") == "nop"
        assert _extract_opcode("name$part: add a0, a1, a2") == "add"

    def test_extract_pure_comment_and_empty(self):
        assert _extract_opcode("# this is a comment") is None
        assert _extract_opcode("") is None
        assert _extract_opcode("   \t  \r\n") is None

    def test_extract_directive(self):
        assert _extract_opcode(".text") is None
        assert _extract_opcode(".globl main") is None
        assert _extract_opcode(".word 0x13") is None
        assert _extract_opcode(".cfi_startproc") is None
        assert _extract_opcode(".option arch, +c") is None

    def test_extract_dotted_and_compressed(self):
        assert _extract_opcode("  fadd.s fa0, fa1, fa2") == "fadd.s"
        assert _extract_opcode("  c.addi a0, 1") == "c.addi"
        assert _extract_opcode("  amoadd.w.aq a0, a1, (a2)") == "amoadd.w.aq"


class TestClassifyOpcode:
    """Tests for instruction classification across all categories."""

    def test_alu(self):
        assert _classify_opcode("add") == "ALU"
        assert _classify_opcode("sub") == "ALU"
        assert _classify_opcode("mul") == "ALU"
        assert _classify_opcode("div") == "ALU"
        assert _classify_opcode("lui") == "ALU"
        assert _classify_opcode("c.addi") == "ALU"
        assert _classify_opcode("addw") == "ALU"

    def test_fp(self):
        assert _classify_opcode("fadd.s") == "FP"
        assert _classify_opcode("fmadd.d") == "FP"
        assert _classify_opcode("fsqrt.s") == "FP"
        assert _classify_opcode("fcvt.w.s") == "FP"
        assert _classify_opcode("fmv.x.w") == "FP"
        assert _classify_opcode("fneg.s") == "FP"

    def test_mem(self):
        assert _classify_opcode("lw") == "MEM"
        assert _classify_opcode("sw") == "MEM"
        assert _classify_opcode("flw") == "MEM"
        assert _classify_opcode("fsd") == "MEM"
        assert _classify_opcode("c.lw") == "MEM"

    def test_branch(self):
        assert _classify_opcode("beq") == "BRANCH"
        assert _classify_opcode("bne") == "BRANCH"
        assert _classify_opcode("bltu") == "BRANCH"
        assert _classify_opcode("bnez") == "BRANCH"
        assert _classify_opcode("c.beqz") == "BRANCH"

    def test_jump(self):
        assert _classify_opcode("j") == "JUMP"
        assert _classify_opcode("jal") == "JUMP"
        assert _classify_opcode("ret") == "JUMP"
        assert _classify_opcode("call") == "JUMP"
        assert _classify_opcode("c.j") == "JUMP"

    def test_atomic(self):
        assert _classify_opcode("lr.w") == "ATOMIC"
        assert _classify_opcode("sc.w") == "ATOMIC"
        assert _classify_opcode("amoadd.w") == "ATOMIC"
        # Atomic ordering suffixes
        assert _classify_opcode("amoadd.w.aq") == "ATOMIC"
        assert _classify_opcode("amoswap.d.aqrl") == "ATOMIC"
        assert _classify_opcode("lr.w.rl") == "ATOMIC"

    def test_system(self):
        assert _classify_opcode("fence") == "SYSTEM"
        assert _classify_opcode("fence.i") == "SYSTEM"
        assert _classify_opcode("csrrw") == "SYSTEM"
        assert _classify_opcode("ecall") == "SYSTEM"
        assert _classify_opcode("sfence.vma") == "SYSTEM"

    def test_pseudo(self):
        assert _classify_opcode("li") == "PSEUDO"
        assert _classify_opcode("mv") == "PSEUDO"
        assert _classify_opcode("nop") == "PSEUDO"
        assert _classify_opcode("neg") == "PSEUDO"
        assert _classify_opcode("c.nop") == "PSEUDO"

    def test_misc(self):
        assert _classify_opcode("unknown_custom_op") == "MISC"
        info = classify_opcode("unknown_custom_op")
        assert info.category == "MISC"
        assert info.extension == "unknown"


class TestInstructionStats:
    """Tests for InstructionStats data model invariants and serialization."""

    def test_invariants_and_immutability(self):
        cat_counts = {c: 0 for c in _CATEGORY_ORDER}
        cat_counts["ALU"] = 2
        cat_counts["MEM"] = 1
        stats = InstructionStats(
            category_counts=cat_counts,
            opcode_counts={"add": 2, "lw": 1},
            unknown_opcodes={},
            total=3,
            source="test.s",
            isa="rv32im",
        )
        assert stats.total == 3
        assert stats.category_counts["ALU"] == 2
        assert stats.category_counts["FP"] == 0

        # Attempt to mutate mapping should raise TypeError
        with pytest.raises(TypeError):
            stats.category_counts["ALU"] = 99  # type: ignore[index]

        # Total mismatch should raise ValueError
        with pytest.raises(ValueError):
            InstructionStats(
                category_counts=cat_counts,
                opcode_counts={"add": 2, "lw": 1},
                unknown_opcodes={},
                total=999,
            )

    def test_serialization_roundtrip(self):
        cat_counts = {c: 0 for c in _CATEGORY_ORDER}
        cat_counts["ALU"] = 1
        cat_counts["MISC"] = 1
        stats = InstructionStats(
            category_counts=cat_counts,
            opcode_counts={"add": 1, "custom_op": 1},
            unknown_opcodes={"custom_op": 1},
            total=2,
            source="custom.s",
        )
        d = stats.to_dict()
        assert d["kind"] == "single"
        assert d["schema_version"] == 1
        assert d["total"] == 2
        assert d["category_counts"]["ALU"] == 1
        assert d["unknown_opcodes"] == {"custom_op": 1}

        reconstructed = InstructionStats.from_dict(d)
        assert reconstructed.total == stats.total
        assert reconstructed.category_counts == stats.category_counts
        assert reconstructed.unknown_opcodes == stats.unknown_opcodes

    def test_from_emulator_counts(self):
        emu_opcodes = {"add": 100, "lw": 50, "custom_acc": 5}
        stats = from_emulator_counts(emu_opcodes, source="tinyfive", isa="rv32im")
        assert stats.count_mode == "dynamic"
        assert stats.total == 155
        assert stats.category_counts["ALU"] == 100
        assert stats.category_counts["MEM"] == 50
        assert stats.category_counts["MISC"] == 5
        assert stats.unknown_opcodes == {"custom_acc": 5}
        assert stats.source == "tinyfive"
        assert stats.isa == "rv32im"

    def test_from_emulator_counts_uppercase_and_misc(self):
        # Mixed casing and uppercase unknown opcodes
        emu_opcodes = {"ADD": 10, "AMOADD.W": 5, "CUSTOM_EXT": 2}
        stats = from_emulator_counts(emu_opcodes, source="spike", isa="rv32im")
        assert stats.total == 17
        assert stats.category_counts["ALU"] == 10
        assert stats.category_counts["ATOMIC"] == 5
        assert stats.category_counts["MISC"] == 2
        assert stats.unknown_opcodes == {"custom_ext": 2}
        assert "custom_ext" in stats.opcode_counts
        assert "amoadd.w" in stats.opcode_counts


class TestAnalyzeInstructions:
    """Tests for analysis API functions."""

    def test_empty_string(self):
        stats = analyze_instructions("")
        assert stats.total == 0
        for c in _CATEGORY_ORDER:
            assert stats.category_counts[c] == 0
        assert len(stats.opcode_counts) == 0

    def test_analyze_fixtures(self):
        # 1. basic_rv32im.s
        basic_path = FIXTURES_DIR / "basic_rv32im.s"
        stats = analyze_file(str(basic_path))
        assert stats.total == 15
        assert stats.category_counts["ALU"] == 7  # add, addi, sub, mul, div, srai, lui
        assert stats.category_counts["MEM"] == 2  # lw, sw
        assert stats.category_counts["BRANCH"] == 1  # beq
        assert stats.category_counts["JUMP"] == 2  # j, ret
        assert stats.category_counts["PSEUDO"] == 3  # li, mv, nop

        # 2. labels_and_directives.s
        labels_path = FIXTURES_DIR / "labels_and_directives.s"
        stats_labels = analyze_file(str(labels_path))
        # addi, bnez, nop
        assert stats_labels.total == 3
        assert stats_labels.category_counts["ALU"] == 1
        assert stats_labels.category_counts["BRANCH"] == 1
        assert stats_labels.category_counts["PSEUDO"] == 1

        # 3. extensions.s
        ext_path = FIXTURES_DIR / "extensions.s"
        stats_ext = analyze_file(str(ext_path))
        assert stats_ext.category_counts["FP"] == 5  # fadd.s, fmadd.d, fsqrt.s, fcvt.w.s, fmv.x.w
        assert stats_ext.category_counts["MEM"] == 3  # flw, fsd, c.lw
        assert stats_ext.category_counts["ATOMIC"] == 4  # lr.w, sc.w, amoadd.w.aq, amoswap.d.aqrl
        assert stats_ext.category_counts["SYSTEM"] == 5  # fence, fence.i, csrrw, ecall, ebreak
        assert stats_ext.category_counts["BRANCH"] == 1  # c.beqz
        assert stats_ext.category_counts["JUMP"] == 2  # c.j, ret
        assert stats_ext.category_counts["PSEUDO"] == 1  # c.nop
        assert stats_ext.category_counts["ALU"] == 1  # c.addi

        # 4. unknown.s
        unk_path = FIXTURES_DIR / "unknown.s"
        stats_unk = analyze_file(str(unk_path))
        assert stats_unk.category_counts["MISC"] == 2
        assert "custom_matrix_acc" in stats_unk.unknown_opcodes
        assert "zz_special_op" in stats_unk.unknown_opcodes


class TestComparison:
    """Tests for multi-file comparison logic."""

    def test_compare_before_and_after(self):
        before = str(FIXTURES_DIR / "compare_before.s")
        after = str(FIXTURES_DIR / "compare_after.s")

        comp = compare_files([before, after])
        assert isinstance(comp, ComparisonResult)
        assert comp.baseline_label == "compare_before.s"
        assert len(comp.files) == 2

        # In compare_after: one add removed, one sw removed, j .L2 removed, ret added
        before_s = comp.stats["compare_before.s"]
        after_s = comp.stats["compare_after.s"]
        assert after_s.total < before_s.total

        # Delta checks
        after_delta = comp.category_deltas["compare_after.s"]
        assert after_delta["ALU"].absolute == -1
        assert after_delta["MEM"].absolute == -1

        # Baseline delta must be all 0
        base_delta = comp.category_deltas["compare_before.s"]
        for cat in _CATEGORY_ORDER:
            assert base_delta[cat].absolute == 0
            assert base_delta[cat].percent == 0.0

        # Legacy diffs excludes baseline
        assert "compare_before.s" not in comp.diffs
        assert "compare_after.s" in comp.diffs
        assert comp.diffs["compare_after.s"]["ALU"] == -1

    def test_zero_baseline_division(self):
        cat_counts1 = {c: 0 for c in _CATEGORY_ORDER}
        cat_counts2 = {c: 0 for c in _CATEGORY_ORDER}
        cat_counts2["FP"] = 5
        s1 = InstructionStats(category_counts=cat_counts1, opcode_counts={}, unknown_opcodes={}, total=0)
        s2 = InstructionStats(category_counts=cat_counts2, opcode_counts={"fadd.s": 5}, unknown_opcodes={}, total=5)

        comp = compare_stats([("base", s1), ("cur", s2)])
        assert comp.category_deltas["cur"]["ALU"].percent == 0.0  # 0 -> 0
        assert comp.category_deltas["cur"]["FP"].percent is None  # 0 -> 5 (None / N/A)

    def test_incompatible_comparison_rejected(self):
        s1 = InstructionStats(category_counts={c: 0 for c in _CATEGORY_ORDER}, opcode_counts={}, unknown_opcodes={}, total=0, isa="rv32i")
        s2 = InstructionStats(category_counts={c: 0 for c in _CATEGORY_ORDER}, opcode_counts={}, unknown_opcodes={}, total=0, isa="rv64i")
        with pytest.raises(ValueError, match="not comparable"):
            compare_stats([("s1", s1), ("s2", s2)])

    def test_comparison_result_invalid_baseline(self):
        s1 = InstructionStats(category_counts={c: 0 for c in _CATEGORY_ORDER}, opcode_counts={}, unknown_opcodes={}, total=0)
        with pytest.raises(ValueError, match="baseline_label 'nonexistent' must be present in stats"):
            ComparisonResult(
                baseline_label="nonexistent",
                stats={"s1": s1},
                category_deltas={},
                opcode_deltas={},
            )


class TestRenderers:
    """Tests for text, JSON, markdown, and HTML report rendering."""

    @pytest.fixture
    def sample_stats(self):
        return analyze_instructions(
            ".text\n"
            "main:\n"
            "  add a0, a1, a2\n"
            "  lw a3, 0(sp)\n"
            "  custom_op t0, t1\n"
        )

    def test_render_text(self, sample_stats):
        txt = render_text(sample_stats, verbose=True)
        assert "RISC-V Instruction Statistics" in txt
        assert "ALU" in txt
        assert "add" in txt
        assert "WARNING: Unknown Instructions" in txt

    def test_render_json(self, sample_stats):
        out = render_json(sample_stats)
        data = json.loads(out)
        assert data["kind"] == "single"
        assert data["total"] == 3
        assert data["category_counts"]["ALU"] == 1
        assert data["unknown_opcodes"] == {"custom_op": 1}

    def test_render_markdown(self, sample_stats):
        md = render_markdown(sample_stats, verbose=True)
        assert "# RISC-V Instruction Statistics" in md
        assert "| ALU | 1 |" in md
        assert "> [!WARNING]" in md
        assert "`custom_op`: 1" in md

    def test_render_html_escaping(self, sample_stats):
        html_out = render_html(sample_stats, title="<Test & Title>")
        assert "&lt;Test &amp; Title&gt;" in html_out
        assert "<code>custom_op</code>" in html_out
        assert "<table" in html_out


class TestCLI:
    """Tests for CLI arguments, exit codes, and contracts."""

    def test_cli_single_file(self, capsys):
        target = str(FIXTURES_DIR / "basic_rv32im.s")
        ret = main([target])
        assert ret == 0
        captured = capsys.readouterr()
        assert "RISC-V Instruction Statistics" in captured.out

    def test_cli_compare_mode(self, capsys):
        b = str(FIXTURES_DIR / "compare_before.s")
        a = str(FIXTURES_DIR / "compare_after.s")
        ret = main([b, a, "--compare"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "RISC-V Instruction Count Comparison" in captured.out

    def test_cli_single_file_with_compare_flag_fails(self, capsys):
        target = str(FIXTURES_DIR / "basic_rv32im.s")
        ret = main([target, "--compare"])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_cli_fail_on_unknown(self, capsys):
        unk = str(FIXTURES_DIR / "unknown.s")
        ret = main([unk, "--fail-on-unknown"])
        assert ret == 2
        captured = capsys.readouterr()
        assert "Unknown opcodes encountered" in captured.err

    def test_cli_outputs_generation(self):
        b = str(FIXTURES_DIR / "compare_before.s")
        a = str(FIXTURES_DIR / "compare_after.s")

        with tempfile.TemporaryDirectory() as tmpdir:
            json_p = os.path.join(tmpdir, "report.json")
            md_p = os.path.join(tmpdir, "report.md")
            html_p = os.path.join(tmpdir, "report.html")

            ret = main([
                b, a, "--compare",
                "--json-output", json_p,
                "--markdown", md_p,
                "--html", html_p,
            ])
            assert ret == 0
            assert os.path.exists(json_p)
            assert os.path.exists(md_p)
            assert os.path.exists(html_p)

            with open(json_p, "r", encoding="utf-8") as f:
                data = json.load(f)
                assert data["kind"] == "comparison"

    def test_cli_single_file_json_stdout_purity(self, capsys):
        target = str(FIXTURES_DIR / "basic_rv32im.s")
        ret = main([target, "--json"])
        assert ret == 0
        captured = capsys.readouterr()
        # Verify stdout is strictly valid JSON without text table prepended
        data = json.loads(captured.out)
        assert data["kind"] == "single"
        assert data["total"] == 15

    def test_cli_compare_json_stdout_purity(self, capsys):
        b = str(FIXTURES_DIR / "compare_before.s")
        a = str(FIXTURES_DIR / "compare_after.s")
        ret = main([b, a, "--compare", "--json"])
        assert ret == 0
        captured = capsys.readouterr()
        # Verify stdout is strictly valid JSON without text table prepended
        data = json.loads(captured.out)
        assert data["kind"] == "comparison"


class TestCompilerDriverIntegration:
    """Tests for compiler driver integration with count_instr."""

    def test_count_instr_riscv(self):
        config = CompilerConfig(
            backend="riscv",
            count_instr=True,
        )
        driver = CompilerDriver(config)
        dsl_src = (
            "t1 = add(x, y)\n"
            "return t1\n"
        )
        res = driver.compile(input_path="", dsl_source=dsl_src)
        assert res.success
        assert "instruction_count" in res.stats
        assert any("Instruction count:" in w for w in res.warnings)

    def test_count_instr_llvm_rejected(self):
        config = CompilerConfig(
            backend="llvm",
            count_instr=True,
        )
        driver = CompilerDriver(config)
        res = driver.compile(input_path="", dsl_source="t1 = add(x, y)\nreturn t1\n")
        assert not res.success
        assert any("not supported for LLVM" in err for err in res.errors)


class TestLegacyCompatibility:
    """Tests for backwards compatibility with legacy tests."""

    def test_compare_two_files(self, tmp_path):
        f1_path, f2_path = tmp_path / "first.s", tmp_path / "second.s"
        f1_path.write_text(".text\nmain:\n  add x1, x2, x3\n  lw x4, 0(sp)\n  ret\n")
        f2_path.write_text(".text\nmain:\n  add x1, x2, x3\n  div x5, x6, x7\n  ret\n")

        result = compare_files([str(f1_path), str(f2_path)])
        assert isinstance(result, ComparisonResult)
        assert len(result.files) == 2
        assert len(result.counts) == 2
        assert result.counts["first.s"]["MEM"] == 1
        assert result.counts["second.s"]["MEM"] == 0

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
