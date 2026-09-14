"""Topic 24: portable Spike toolchain resolution and graceful degradation.

All tests are hermetic: they never require a real Spike installation.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from scratchv.standalone import spike_sim

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_fake_tool(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_text("#!/bin/sh\nexit 0\n")
    p.chmod(0o755)
    return p


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch, tmp_path):
    for var in ("SCRATCHV_SPIKE_BIN", "SCRATCHV_SPIKE_DASM",
                "SCRATCHV_SPIKE_LOG_PARSER", "SCRATCHV_SPIKE_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(spike_sim, "COMMON_SPIKE_DIRS", ())
    monkeypatch.setattr(spike_sim, "SPIKE", str(tmp_path / "legacy-spike"))
    monkeypatch.setattr(spike_sim, "SPIKE_DASM", str(tmp_path / "legacy-dasm"))
    monkeypatch.setattr(spike_sim, "SPIKE_LOG_PARSER",
                        str(tmp_path / "legacy-parser"))
    monkeypatch.setattr(shutil, "which", lambda name: None)


CANNED_STDERR = """\
Commited 1234 instructions
core   0: 0x80000000 (0x00000013) 1.5 MIPS
I$: 64 sets × 2 ways × 32 B
  hits: 10,000    misses: 25    miss rate: 0.25%
D$: 128 sets × 4 ways × 32 B
  hits: 20,000    misses: 50    miss rate: 0.25%
"""
CANNED_STDOUT = """\
PC histogram (number of commits per PC):
0x80000014: 123
0x80000018: 456

"""


# ── Import / CLI hygiene ────────────────────────────────────────────────────

def test_import_works_without_spike(tmp_path):
    env = {"PATH": str(tmp_path), "HOME": str(tmp_path),
           "PYTHONPATH": str(REPO_ROOT)}
    proc = subprocess.run(
        [sys.executable, "-c",
         "import scratchv.standalone.spike_sim as s; print(s.SPIKE)"],
        capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        spike_sim.main(["--help"])

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--spike-bin", "--spike-dasm", "--spike-log-parser",
                 "--require-spike"):
        assert flag in out


# ── Missing-spike degradation ───────────────────────────────────────────────

def test_missing_spike_skips_with_reason(tmp_path, capsys):
    binary = tmp_path / "output.bin"
    binary.write_bytes(b"\x00" * 64)

    rc = spike_sim.main(["--binary", str(binary), "--code-size", "64"])

    assert rc == spike_sim.EXIT_OK
    err = capsys.readouterr().err
    assert "SKIP:" in err
    assert "spike binary not found" in err
    assert "SCRATCHV_SPIKE_BIN" in err
    assert "SCRATCHV_SPIKE_HOME/bin/spike" in err
    assert "legacy" in err
    assert not (tmp_path / "output_spike.elf").exists()


def test_missing_spike_strict_returns_config_error(tmp_path, capsys):
    binary = tmp_path / "output.bin"
    binary.write_bytes(b"\x00" * 64)

    rc = spike_sim.main(["--binary", str(binary), "--code-size", "64",
                         "--require-spike"])

    assert rc == spike_sim.EXIT_CONFIG
    captured = capsys.readouterr()
    assert "ERROR:" in captured.err
    assert "--require-spike" in captured.err
    # Design doc §2.3 row 1: strict mode emits no report at all.
    assert captured.out == ""
    assert not (tmp_path / "output_spike.elf").exists()


def test_missing_spike_json_report_fields(tmp_path, capsys):
    binary = tmp_path / "output.bin"
    binary.write_bytes(b"\x00" * 64)

    rc = spike_sim.main(["--binary", str(binary), "--code-size", "64", "--json"])

    assert rc == spike_sim.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "skipped"
    assert report["skip_reason"] == "spike binary not found"
    assert report["spike_binary"] is None
    for key in ("binary", "code_size", "static_insns", "max_instr",
                "committed_insns", "wall_time_s", "exit_code", "icache",
                "dcache", "top_pcs", "stderr_tail", "parse_warnings",
                "tool_warnings", "spike_tools"):
        assert key in report
    assert report["spike_tools"]["spike"]["source"] == "missing"
    assert report["spike_tools"]["spike"]["path"] is None
    # F3: the CLI skip report uses the same -2 sentinel as the library path.
    assert report["exit_code"] == -2


# ── Resolution priority chain ───────────────────────────────────────────────

def test_resolution_cli_over_env(tmp_path, monkeypatch):
    fake_env = make_fake_tool(tmp_path, "spike-env")
    fake_cli = make_fake_tool(tmp_path, "spike-cli")
    monkeypatch.setenv("SCRATCHV_SPIKE_BIN", str(fake_env))

    tools_env = spike_sim.resolve_spike_tools()
    assert tools_env.spike == str(fake_env)
    assert tools_env.sources["spike"] == "env"

    tools_cli = spike_sim.resolve_spike_tools(cli_spike=str(fake_cli))
    assert tools_cli.spike == str(fake_cli)
    assert tools_cli.sources["spike"] == "cli"


def test_resolution_env_over_path(tmp_path, monkeypatch):
    fake_env = make_fake_tool(tmp_path, "spike-env")
    monkeypatch.setenv("SCRATCHV_SPIKE_BIN", str(fake_env))

    tools = spike_sim.resolve_spike_tools()
    assert tools.spike == str(fake_env)
    assert tools.sources["spike"] == "env"

    monkeypatch.delenv("SCRATCHV_SPIKE_BIN")
    fake_path = make_fake_tool(tmp_path, "spike-path")
    monkeypatch.setattr(
        shutil, "which",
        lambda name: str(fake_path) if name == "spike" else None)

    tools = spike_sim.resolve_spike_tools()
    assert tools.spike == str(fake_path)
    assert tools.sources["spike"] == "path"


def test_resolution_spike_home_and_common(tmp_path, monkeypatch):
    home = tmp_path / "spike-home"
    (home / "bin").mkdir(parents=True)
    fake_home_spike = make_fake_tool(home / "bin", "spike")
    monkeypatch.setenv("SCRATCHV_SPIKE_HOME", str(home))

    tools = spike_sim.resolve_spike_tools()
    assert tools.spike == str(fake_home_spike)
    assert tools.sources["spike"] == "spike_home"

    monkeypatch.delenv("SCRATCHV_SPIKE_HOME")
    common = tmp_path / "common"
    common.mkdir()
    fake_common_spike = make_fake_tool(common, "spike")
    monkeypatch.setattr(spike_sim, "COMMON_SPIKE_DIRS", (str(common),))

    tools = spike_sim.resolve_spike_tools()
    assert tools.spike == str(fake_common_spike)
    assert tools.sources["spike"] == "common"


def test_resolution_legacy_constant(tmp_path, monkeypatch):
    fake_legacy = make_fake_tool(tmp_path, "legacy-spike")
    monkeypatch.setattr(spike_sim, "SPIKE", str(fake_legacy))

    tools = spike_sim.resolve_spike_tools()

    assert tools.spike == str(fake_legacy)
    assert tools.sources["spike"] == "legacy"


# ── Invalid explicit paths ──────────────────────────────────────────────────

def test_cli_invalid_required_path_raises(tmp_path):
    with pytest.raises(spike_sim.SpikeConfigError) as ei:
        spike_sim.resolve_spike_tools(cli_spike=str(tmp_path / "nope"))
    assert "--spike-bin" in str(ei.value)


def test_optional_cli_invalid_path_warns_and_falls_through(
        tmp_path, monkeypatch):
    # F4: spike-dasm / spike-log-parser are optional; an invalid explicit
    # path must not abort resolution/execution.
    monkeypatch.setenv("SCRATCHV_SPIKE_DASM", str(tmp_path / "nope"))

    tools = spike_sim.resolve_spike_tools(
        cli_spike=str(make_fake_tool(tmp_path, "spike")),
        cli_dasm=str(tmp_path / "nope-dasm"),
        cli_log_parser=str(tmp_path / "nope-parser"),
    )

    assert tools.spike_dasm is None
    assert tools.spike_log_parser is None
    joined = "\n".join(tools.warnings)
    assert "--spike-dasm" in joined
    assert "--spike-log-parser" in joined


def test_cli_invalid_path_returns_config_error(tmp_path, capsys):
    binary = tmp_path / "output.bin"
    binary.write_bytes(b"\x00" * 64)

    rc = spike_sim.main(["--binary", str(binary), "--code-size", "64",
                         "--spike-bin", str(tmp_path / "nope"), "--json"])

    assert rc == spike_sim.EXIT_CONFIG
    captured = capsys.readouterr()
    assert "ERROR:" in captured.err
    # Design doc §2.3 row 4: no report is emitted for an invalid --spike-bin.
    assert captured.out == ""
    assert not (tmp_path / "output_spike.elf").exists()


def test_env_invalid_path_warns_and_falls_through(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRATCHV_SPIKE_BIN", str(tmp_path / "nope"))

    tools = spike_sim.resolve_spike_tools()

    assert tools.spike is None
    assert tools.sources["spike"] == "missing"
    assert any("SCRATCHV_SPIKE_BIN" in w for w in tools.warnings)


# ── Pure parsing helpers ────────────────────────────────────────────────────

def test_parse_commit_stats():
    committed, mips = spike_sim.parse_commit_stats(CANNED_STDERR)
    assert (committed, mips) == (1234, 1.5)

    corrected, corrected_mips = spike_sim.parse_commit_stats(
        "Committed 2,000,000 instructions\n42.0 MIPS")
    assert (corrected, corrected_mips) == (2_000_000, 42.0)

    assert spike_sim.parse_commit_stats("nothing here") == (0, 0.0)


def test_parse_cache_stats():
    stats = spike_sim.parse_cache_stats(CANNED_STDERR)
    assert stats["icache_hits"] == 10_000
    assert stats["icache_misses"] == 25
    assert stats["icache_miss_rate"] == 0.25
    assert stats["dcache_hits"] == 20_000
    assert stats["dcache_misses"] == 50
    assert stats["dcache_miss_rate"] == 0.25

    empty = spike_sim.parse_cache_stats("no stats here")
    assert set(empty) == {
        "icache_hits", "icache_misses", "icache_miss_rate",
        "dcache_hits", "dcache_misses", "dcache_miss_rate",
    }
    assert all(value == 0 for value in empty.values())


def test_parse_pc_histogram():
    hist = spike_sim.parse_pc_histogram(CANNED_STDOUT)
    assert hist == {0x80000014: 123, 0x80000018: 456}

    messy = (
        "PC histogram (number of commits per PC):\n"
        "0x80000010: 7\n"
        "garbage line\n"
        "0x80000020: 9\n"
        "\n"
        "0x80000030: 11\n"
    )
    assert spike_sim.parse_pc_histogram(messy) == {
        0x80000010: 7, 0x80000020: 9}

    assert spike_sim.parse_pc_histogram("no histogram here") == {}


def test_run_spike_canned_output_is_ok(tmp_path, monkeypatch):
    tools = spike_sim.SpikeTools(spike="/fake/spike")
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return types.SimpleNamespace(
            returncode=0, stdout=CANNED_STDOUT, stderr=CANNED_STDERR)

    monkeypatch.setattr(spike_sim.subprocess, "run", fake_run)

    result = spike_sim.run_spike(str(tmp_path / "x.elf"), tools=tools)

    assert captured["cmd"][0] == "/fake/spike"
    assert result.status == "ok"
    assert result.exit_code == 0
    assert result.spike_path == "/fake/spike"
    assert result.committed_insns == 1234
    assert result.icache_hits == 10_000
    assert result.dcache_misses == 50
    assert result.pc_histogram == {0x80000014: 123, 0x80000018: 456}
    # Every stats section is present: no parse warnings at all.
    assert result.parse_warnings == []


def test_run_spike_empty_output_marks_failed(tmp_path, monkeypatch):
    # F6: exit 0 without any recognizable Spike stats is not a success.
    tools = spike_sim.SpikeTools(spike="/fake/spike")

    monkeypatch.setattr(
        spike_sim.subprocess, "run",
        lambda cmd, **kwargs: types.SimpleNamespace(
            returncode=0, stdout="", stderr=""))

    result = spike_sim.run_spike(str(tmp_path / "x.elf"), tools=tools)

    assert result.status == "failed"
    assert result.exit_code == -2
    assert result.committed_insns == 0
    assert any("no Spike statistics" in w for w in result.parse_warnings)


def test_run_spike_oserror_maps_to_failed(tmp_path, monkeypatch):
    # F1: an existing executable the kernel refuses to exec (ENOEXEC).
    tools = spike_sim.SpikeTools(spike="/fake/spike")

    def fake_run(cmd, **kwargs):
        raise OSError(8, "Exec format error", cmd[0])

    monkeypatch.setattr(spike_sim.subprocess, "run", fake_run)

    result = spike_sim.run_spike(str(tmp_path / "x.elf"), tools=tools)

    assert result.status == "failed"
    assert result.exit_code == -2
    assert "failed to start Spike" in result.stderr
    assert "/fake/spike" in result.stderr
    assert "Exec format error" in result.stderr


def test_run_spike_with_log_oserror_maps_to_failed(tmp_path, monkeypatch):
    tools = spike_sim.SpikeTools(spike="/fake/spike")

    def fake_run(cmd, **kwargs):
        raise OSError(8, "Exec format error", cmd[0])

    monkeypatch.setattr(spike_sim.subprocess, "run", fake_run)

    result, log_content = spike_sim.run_spike_with_log(
        str(tmp_path / "x.elf"), tools=tools)

    assert result.status == "failed"
    assert result.exit_code == -2
    assert "failed to start Spike" in result.stderr
    assert log_content == ""


def test_run_spike_timeout_is_reported(tmp_path, monkeypatch):
    tools = spike_sim.SpikeTools(spike="/fake/spike")

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(spike_sim.subprocess, "run", fake_run)

    result = spike_sim.run_spike(str(tmp_path / "x.elf"), tools=tools)

    assert result.status == "timeout"
    assert result.exit_code == -1
    assert "TIMEOUT" in result.stderr


def test_run_spike_nonzero_exit_marks_failed(tmp_path, monkeypatch):
    tools = spike_sim.SpikeTools(spike="/fake/spike")

    monkeypatch.setattr(
        spike_sim.subprocess, "run",
        lambda cmd, **kwargs: types.SimpleNamespace(
            returncode=1, stdout="", stderr=CANNED_STDERR))

    result = spike_sim.run_spike(str(tmp_path / "x.elf"), tools=tools)

    assert result.status == "failed"
    assert result.exit_code == 1
    # Stats are still parsed from the partial output.
    assert result.committed_insns == 1234


def test_optional_tool_warnings_do_not_imply_missing_features():
    tools = spike_sim.SpikeTools(spike="/fake/spike")

    warnings = spike_sim._optional_tool_warnings(tools)

    assert len(warnings) == 2
    assert all("not used by this module" in w for w in warnings)
    assert not any("unavailable" in w for w in warnings)

    complete = spike_sim.SpikeTools(
        spike="s", spike_dasm="d", spike_log_parser="p")
    assert spike_sim._optional_tool_warnings(complete) == []


def test_run_spike_missing_tool_returns_skipped(tmp_path):
    tools = spike_sim.SpikeTools(
        warnings=("SCRATCHV_SPIKE_BIN=/old/spike is not executable; ignored",))

    result = spike_sim.run_spike(str(tmp_path / "x.elf"), tools=tools)

    assert result.status == "skipped"
    assert result.skip_reason == "spike binary not found"
    assert result.exit_code == -2
    assert list(result.tool_warnings) == list(tools.warnings)


# ── Report fields ───────────────────────────────────────────────────────────

def test_report_status_fields():
    result = spike_sim.SpikeResult(
        status="skipped", skip_reason="spike binary not found")

    text = spike_sim.generate_spike_report(
        result, "output.bin", 64, "64:2:32", "128:4:32", 50_000_000)
    assert "Status:" in text
    assert "skipped" in text
    assert "Skip reason:" in text
    assert "spike binary not found" in text

    tools = spike_sim.SpikeTools()
    report = spike_sim.build_json_report(
        result, "output.bin", 64, "64:2:32", "128:4:32", 50_000_000, tools)
    assert report["status"] == "skipped"
    assert report["skip_reason"] == "spike binary not found"
    assert report["spike_binary"] is None
    for key in ("binary", "code_size", "static_insns", "max_instr",
                "committed_insns", "wall_time_s", "exit_code", "icache",
                "dcache", "top_pcs", "stderr_tail"):
        assert key in report
    assert report["spike_tools"]["spike"]["path"] is None

    plain = spike_sim.build_json_report(
        result, "output.bin", 64, "64:2:32", "128:4:32", 50_000_000)
    assert "spike_tools" not in plain


# ── main() end-to-end (mock / real subprocess) ──────────────────────────────

def _write_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "output.bin"
    binary.write_bytes(b"\x00" * 64)
    return binary


def test_main_success_path_writes_and_cleans_elf(
        tmp_path, monkeypatch, capsys):
    fake_spike = make_fake_tool(tmp_path, "spike")
    binary = _write_binary(tmp_path)
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        elf = Path(cmd[-1])
        captured["cmd"] = list(cmd)
        captured["elf_magic"] = elf.read_bytes()[:4]
        return types.SimpleNamespace(
            returncode=0, stdout=CANNED_STDOUT, stderr=CANNED_STDERR)

    monkeypatch.setattr(spike_sim.subprocess, "run", fake_run)

    rc = spike_sim.main(["--binary", str(binary), "--code-size", "64",
                         "--spike-bin", str(fake_spike), "--json"])

    report = json.loads(capsys.readouterr().out)
    assert rc == spike_sim.EXIT_OK
    assert report["status"] == "ok"
    assert report["exit_code"] == 0
    assert report["committed_insns"] == 1234
    assert report["icache"]["hits"] == 10_000
    assert report["spike_binary"] == str(fake_spike)
    assert report["spike_tools"]["spike"]["source"] == "cli"
    # The resolved toolchain reached the runner.
    assert captured["cmd"][0] == str(fake_spike)
    assert "-g" in captured["cmd"]
    # A valid ELF32 was written next to the binary, then cleaned up.
    assert captured["elf_magic"] == b"\x7fELF"
    assert not (tmp_path / "output_spike.elf").exists()
    # F5: optional tools are reported as unused, not as degraded features.
    assert len(report["tool_warnings"]) == 2
    assert all("not used by this module" in w
               for w in report["tool_warnings"])


def test_main_unexecutable_spike_no_traceback(tmp_path, capsys):
    # F1 reproduction: chmod +x plain text file (no shebang) -> ENOEXEC.
    bad = tmp_path / "badspike"
    bad.write_text("plain text, no shebang\n")
    bad.chmod(0o755)
    binary = _write_binary(tmp_path)

    rc = spike_sim.main(["--binary", str(binary), "--code-size", "64",
                         "--spike-bin", str(bad), "--json"])

    captured = capsys.readouterr()
    assert rc == spike_sim.EXIT_RUN_FAIL
    assert "Traceback" not in captured.err
    report = json.loads(captured.out)
    assert report["status"] == "failed"
    assert report["exit_code"] == -2
    assert "failed to start Spike" in report["stderr_tail"]
    assert not (tmp_path / "output_spike.elf").exists()


def test_main_fake_zero_output_marks_failed(tmp_path, capsys):
    # F6 reproduction: /bin/true-like executable (exit 0, no output).
    fake_spike = make_fake_tool(tmp_path, "spike")
    binary = _write_binary(tmp_path)

    rc = spike_sim.main(["--binary", str(binary), "--code-size", "64",
                         "--spike-bin", str(fake_spike), "--json"])

    report = json.loads(capsys.readouterr().out)
    assert rc == spike_sim.EXIT_RUN_FAIL
    assert report["status"] == "failed"
    assert report["exit_code"] == -2
    assert any("no Spike statistics" in w for w in report["parse_warnings"])
    assert not (tmp_path / "output_spike.elf").exists()


def test_main_nonzero_spike_exit_returns_run_fail(tmp_path, monkeypatch, capsys):
    fake_spike = make_fake_tool(tmp_path, "spike")
    binary = _write_binary(tmp_path)

    monkeypatch.setattr(
        spike_sim.subprocess, "run",
        lambda cmd, **kwargs: types.SimpleNamespace(
            returncode=1, stdout="", stderr=CANNED_STDERR))

    rc = spike_sim.main(["--binary", str(binary), "--code-size", "64",
                         "--spike-bin", str(fake_spike), "--json"])

    report = json.loads(capsys.readouterr().out)
    assert rc == spike_sim.EXIT_RUN_FAIL
    assert report["status"] == "failed"
    assert report["exit_code"] == 1


def test_main_optional_invalid_cli_still_runs(tmp_path, monkeypatch, capsys):
    # F4 reproduction: a bad --spike-dasm must not abort the simulation.
    fake_spike = make_fake_tool(tmp_path, "spike")
    binary = _write_binary(tmp_path)

    monkeypatch.setattr(
        spike_sim.subprocess, "run",
        lambda cmd, **kwargs: types.SimpleNamespace(
            returncode=0, stdout=CANNED_STDOUT, stderr=CANNED_STDERR))

    rc = spike_sim.main(["--binary", str(binary), "--code-size", "64",
                         "--spike-bin", str(fake_spike),
                         "--spike-dasm", str(tmp_path / "nope"),
                         "--spike-log-parser", str(tmp_path / "nope2")])

    captured = capsys.readouterr()
    assert rc == spike_sim.EXIT_OK
    assert "WARNING:" in captured.err
    assert "--spike-dasm" in captured.err
    assert "Committed insns:" in captured.out


# ── run_spike_bench.py --probe-spike / JSON contract (F2) ───────────────────

def test_probe_spike_tools_reports_resolution(monkeypatch, capsys):
    from scratchv.standalone import run_spike_bench

    fake = spike_sim.SpikeTools(
        spike="/opt/riscv/bin/spike", sources={"spike": "cli"})
    monkeypatch.setattr(spike_sim, "resolve_spike_tools", lambda: fake)

    tools = run_spike_bench.probe_spike_tools()

    assert tools is fake
    err = capsys.readouterr().err
    assert "Spike tools:" in err
    assert "spike=/opt/riscv/bin/spike (cli)" in err
    assert "spike-dasm=NOT FOUND" in err
    assert "spike-log-parser=NOT FOUND" in err
    assert "hint:" not in err


def test_probe_spike_tools_missing_prints_hint(monkeypatch, capsys):
    from scratchv.standalone import run_spike_bench

    monkeypatch.setattr(spike_sim, "resolve_spike_tools",
                        lambda: spike_sim.SpikeTools())

    tools = run_spike_bench.probe_spike_tools()

    assert tools is not None
    assert tools.spike is None
    err = capsys.readouterr().err
    assert "spike=NOT FOUND" in err
    assert "built-in emulator backend" in err


def test_probe_spike_tools_config_error_returns_none(monkeypatch, capsys):
    from scratchv.standalone import run_spike_bench

    def boom():
        raise spike_sim.SpikeConfigError(
            "--spike-bin='/x' is not an executable file")

    monkeypatch.setattr(spike_sim, "resolve_spike_tools", boom)

    assert run_spike_bench.probe_spike_tools() is None
    assert "ERROR:" in capsys.readouterr().err


def test_run_spike_bench_json_backend_and_spike_tools():
    from scratchv.standalone import run_spike_bench

    result = run_spike_bench.SpikeStyleResult(
        code_size=3140, binary_path="output.bin", wall_time_s=0.5)

    plain = run_spike_bench.generate_json_report(result)
    assert plain["backend"] == {"kind": "emulator", "spike_style": True}
    assert "spike_tools" not in plain
    for key in ("summary", "instruction_mix", "memory", "cache",
                "branch_behavior", "cycle_estimates", "top_pcs",
                "per_layer"):
        assert key in plain
    assert plain["summary"]["binary_path"] == "output.bin"
    assert plain["summary"]["code_size"] == 3140

    tools = spike_sim.SpikeTools(
        spike="/opt/riscv/bin/spike",
        sources={"spike": "env"},
        warnings=("SCRATCHV_SPIKE_DASM=/old/dasm is not executable; ignored",))
    report = run_spike_bench.generate_json_report(result, spike_tools=tools)

    assert report["backend"] == plain["backend"]
    assert report["spike_tools"]["spike"] == {
        "path": "/opt/riscv/bin/spike",
        "source": "env",
        "candidates": [],
    }
    assert report["spike_tools"]["warnings"] == [
        "SCRATCHV_SPIKE_DASM=/old/dasm is not executable; ignored"]
