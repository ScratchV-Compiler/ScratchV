"""Integration tests: asm peephole inside compiler pipeline."""

from __future__ import annotations

import pytest

from scratchv.backend.asm_emit import AsmEmitter
from scratchv.backend.asm_peephole import AsmPeepholeOptimizer
from scratchv.backend.instruction_select import InstructionSelector
from scratchv.backend.register_alloc import RegisterAllocator
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.frontend.dsl_parser import DSLParser


def _compile_dsl_to_asm(dsl: str) -> str:
    program = DSLParser().parse(dsl)
    instrs = InstructionSelector(program).run()
    allocated = RegisterAllocator(instrs, mode="greedy").run()
    return AsmEmitter(allocated).emit()


def _count_opcode_lines(asm: str, opcode: str) -> int:
    count = 0
    for line in asm.splitlines():
        stripped = line.strip()
        if not stripped or stripped.endswith(":"):
            continue
        # Strip leading label on same line
        if ":" in stripped.split()[0]:
            stripped = stripped.split(":", 1)[1].strip()
            if not stripped:
                continue
        parts = stripped.split()
        if parts and parts[0] == opcode:
            count += 1
    return count


@pytest.mark.integration
class TestCompilerPipelineIntegration:
    """Peephole via CompilerDriver._run_asm_passes."""

    def test_peephole_asm_flag_reduces_addi(self, tmp_path):
        dsl_path = "benchmarks/cases/017_while_sum.dsl"
        out_off = tmp_path / "off.s"
        out_on = tmp_path / "on.s"

        driver_off = CompilerDriver(CompilerConfig(peephole_asm=False))
        driver_on = CompilerDriver(CompilerConfig(peephole_asm=True))

        res_off = driver_off.compile(dsl_path, str(out_off))
        res_on = driver_on.compile(dsl_path, str(out_on))

        assert res_off.success and res_on.success
        addi_off = _count_opcode_lines(res_off.output_text, "addi")
        addi_on = _count_opcode_lines(res_on.output_text, "addi")
        assert addi_on <= addi_off
        peephole_warnings = [w for w in res_on.warnings if "Asm peephole" in w]
        if addi_on < addi_off:
            assert peephole_warnings
        # flag enabled must not break compilation even if zero opportunities

    def test_peephole_preserves_compilation_success(self, tmp_path):
        dsl_path = "benchmarks/cases/001_simple_add.dsl"
        out = tmp_path / "add.s"
        driver = CompilerDriver(CompilerConfig(peephole_asm=True))
        result = driver.compile(dsl_path, str(out))
        assert result.success
        assert "main:" in result.output_text
        assert "ret" in result.output_text

    def test_peephole_with_beautify_and_const_merge(self, tmp_path):
        asm = _compile_dsl_to_asm("y = add(a, b)\nreturn y")
        driver = CompilerDriver(CompilerConfig(
            peephole_asm=True,
            beautify_asm=True,
            const_merge=True,
        ))
        warnings: list[str] = []
        optimized = driver._run_asm_passes(asm, warnings)
        assert optimized
        assert isinstance(optimized, str)
        # Passes must leave a usable program skeleton
        assert "ret" in optimized or "jalr" in optimized or "add" in optimized

    def test_peephole_warning_when_changes_applied(self):
        driver = CompilerDriver(CompilerConfig(peephole_asm=True))
        asm = "  addi t0, t0, 1\n  addi t0, t0, 2\n  ret\n"
        warnings: list[str] = []
        out = driver._run_asm_passes(asm, warnings)
        assert "3" in out
        peephole_warnings = [w for w in warnings if "Asm peephole" in w]
        assert peephole_warnings
        assert "instr saved" in peephole_warnings[0]
        assert "->" in peephole_warnings[0]

    def test_peephole_disabled_leaves_fusible_pair(self):
        driver = CompilerDriver(CompilerConfig(peephole_asm=False))
        asm = "  addi t0, t0, 1\n  addi t0, t0, 2\n  ret\n"
        warnings: list[str] = []
        out = driver._run_asm_passes(asm, warnings)
        assert out.count("addi") == 2
        assert not any("Asm peephole" in w for w in warnings)


@pytest.mark.integration
class TestBackendChainIntegration:
    """DSL → codegen → peephole without full driver."""

    def test_emit_then_peephole_idempotent_on_clean_asm(self):
        asm = _compile_dsl_to_asm("y = add(a, b)\nreturn y")
        opt = AsmPeepholeOptimizer()
        first, c1 = opt.optimize(asm)
        second, c2 = opt.optimize(first)
        assert c2 == 0
        assert second == first

    def test_synthetic_fusible_sequence_through_pipeline(self):
        asm = (
            ".text\nmain:\n"
            "  li t0, 1\n  addi t0, t0, 2\n"
            "  beq x0, x0, main\n"
            "  ret\n"
        )
        result, changes = AsmPeepholeOptimizer().optimize(asm)
        assert changes >= 2
        assert "j main" in result
        assert "li t0, 3" in result or "li t0 3" in result

    def test_driver_pass_matches_direct_optimize(self):
        asm = (
            "  li t0, 10\n  addi t0, t0, 5\n"
            "  nop\n  mv t1, t1\n  ret\n"
        )
        direct, n = AsmPeepholeOptimizer().optimize(asm)
        warnings: list[str] = []
        via_driver = CompilerDriver(
            CompilerConfig(peephole_asm=True)
        )._run_asm_passes(asm, warnings)
        assert n >= 1
        assert via_driver == direct
        assert any("Asm peephole" in w for w in warnings)
