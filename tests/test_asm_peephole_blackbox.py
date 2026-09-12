"""Black-box tests: CLI and file I/O without importing internals."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "asm_peephole"


def _run_cli(args: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "scratchv.backend.asm_peephole", *args]
    return subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )


@pytest.mark.blackbox
class TestPeepholeCLI:
    """Subprocess-based CLI tests."""

    def test_cli_optimize_addi_fusion(self, tmp_path):
        inp = FIXTURES / "input_addi_fusion.s"
        out = tmp_path / "out.s"
        proc = _run_cli([str(inp), "-o", str(out), "--report"])
        assert proc.returncode == 0, proc.stderr
        text = out.read_text()
        assert "8" in text
        assert "peephole" in text.lower()
        assert "Total changes" in proc.stderr
        assert "Instructions saved" in proc.stderr
        assert "instruction(s) saved" in proc.stderr

    def test_cli_li_addi_fusion(self, tmp_path):
        inp = FIXTURES / "input_li_addi.s"
        out = tmp_path / "out.s"
        proc = _run_cli([str(inp), "-o", str(out)])
        assert proc.returncode == 0
        assert "15" in out.read_text()

    def test_cli_beq_to_jump(self, tmp_path):
        inp = FIXTURES / "input_beq_zero.s"
        out = tmp_path / "out.s"
        proc = _run_cli([str(inp), "-o", str(out)])
        assert proc.returncode == 0
        text = out.read_text()
        assert "j target" in text
        assert not any(
            line.strip().startswith("beq")
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        )

    def test_cli_no_change_passthrough(self, tmp_path):
        inp = FIXTURES / "input_no_change.s"
        out = tmp_path / "out.s"
        proc = _run_cli([str(inp), "-o", str(out), "--report"])
        assert proc.returncode == 0
        assert "No optimization" in proc.stderr or "Total changes: 0" in proc.stderr
        assert "add" in out.read_text()
        assert "sub" in out.read_text()

    def test_cli_hex_fusion(self, tmp_path):
        inp = FIXTURES / "input_hex_fusion.s"
        out = tmp_path / "out.s"
        proc = _run_cli([str(inp), "-o", str(out)])
        assert proc.returncode == 0
        text = out.read_text()
        assert "48" in text
        assert "(" not in text.split("#")[0]

    def test_cli_overflow_rejected(self, tmp_path):
        inp = FIXTURES / "input_addi_overflow.s"
        out = tmp_path / "out.s"
        proc = _run_cli([str(inp), "-o", str(out), "--report"])
        assert proc.returncode == 0
        text = out.read_text()
        assert text.count("addi") == 2
        assert "4000" not in text
        assert "Total changes: 0" in proc.stderr or "No optimization" in proc.stderr

    def test_cli_nop_and_mv_self(self, tmp_path):
        inp = FIXTURES / "input_nop_mv_self.s"
        out = tmp_path / "out.s"
        proc = _run_cli([str(inp), "-o", str(out), "--report"])
        assert proc.returncode == 0
        text = out.read_text()
        assert "nop" not in text
        assert "mv t0, t0" not in text
        assert "addi" in text

    def test_cli_mv_chain(self, tmp_path):
        inp = FIXTURES / "input_mv_chain.s"
        out = tmp_path / "out.s"
        proc = _run_cli([str(inp), "-o", str(out)])
        assert proc.returncode == 0
        assert "mv t2, t1" in out.read_text()

    def test_cli_missing_input_fails(self):
        proc = _run_cli(["/nonexistent/file.s"])
        assert proc.returncode != 0

    def test_cli_stdout_mode(self):
        inp = FIXTURES / "input_addi_fusion.s"
        proc = _run_cli([str(inp)])
        assert proc.returncode == 0
        assert proc.stdout.strip()
        assert "8" in proc.stdout

    def test_cli_list_rules(self):
        # argparse currently requires a positional input even for --list-rules
        proc = _run_cli(["--list-rules", str(FIXTURES / "input_no_change.s")])
        assert proc.returncode == 0, proc.stderr
        assert "addi+addi fusion" in proc.stdout
        assert "li+addi fusion" in proc.stdout
        assert "nop elimination" in proc.stdout
        assert "redundant mv pair elimination" not in proc.stdout


@pytest.mark.blackbox
class TestPeepholePublicAPIBlackbox:
    """Only public import path, treat as opaque box."""

    def test_optimize_returns_tuple(self):
        from scratchv.backend import AsmPeepholeOptimizer

        opt = AsmPeepholeOptimizer()
        asm = (FIXTURES / "input_addi_fusion.s").read_text()
        out, n = opt.optimize(asm)
        assert isinstance(out, str)
        assert isinstance(n, int)
        assert n >= 1

    def test_report_after_optimize(self):
        from scratchv.backend import AsmPeepholeOptimizer

        opt = AsmPeepholeOptimizer()
        opt.optimize((FIXTURES / "input_li_addi.s").read_text())
        report = opt.report()
        assert "Peephole Optimizer Report" in report
        assert "Instructions before:" in report
        assert "Instructions saved:" in report
        assert "Rule applications:" in report
        assert opt.total_matches
        assert opt.instructions_saved >= 1

    def test_public_api_idempotent(self):
        from scratchv.backend import AsmPeepholeOptimizer

        opt = AsmPeepholeOptimizer()
        asm = (FIXTURES / "input_hex_fusion.s").read_text()
        first, _ = opt.optimize(asm)
        second, n2 = opt.optimize(first)
        assert n2 == 0
        assert second == first
