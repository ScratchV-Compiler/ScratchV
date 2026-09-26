"""The benchmark must exercise real checkpoints and fail on invalid IR."""

import json

import pytest

import scratchv.analysis.ir_verifier as verifier_module
from benchmarks.bench_ir_verifier import DEFAULT_MODEL, benchmark, main
from scratchv.compiler import CompilerDriver
from tests.test_ir_verifier import straight


@pytest.mark.parametrize("level,passes", [
    ("none", []),
    ("basic", ["constant-folding", "dead-code-elim"]),
    ("all", ["constant-folding", "dead-code-elim", "ir-peephole",
             "muladd-fusion", "licm"]),
])
def test_cnn_benchmark_runs_real_verification(monkeypatch, level, passes):
    events = []
    original = verifier_module.verify_ir

    def check(program, *, stage=None):
        events.append(stage)
        return original(program, stage=stage)

    monkeypatch.setattr(verifier_module, "verify_ir", check)
    report = benchmark(DEFAULT_MODEL, repeats=1, warmup=0, levels=(level,))
    expected = ["after-parse"]
    for name in passes:
        expected.extend([f"before:{name}", f"after:{name}"])
    expected.append("before-codegen")
    assert events == expected  # Disabled side must not invoke the verifier.
    row, = report["results"]
    assert row["output_equal"]
    assert len(row["disabled_samples_s"]) == len(row["enabled_samples_s"]) == 1


def test_benchmark_rejects_invalid_ir(monkeypatch):
    def parse(*args):
        program = straight()
        program.functions[0].params = []
        return program

    monkeypatch.setattr(CompilerDriver, "_parse", parse)
    monkeypatch.setattr(CompilerDriver, "_generate_code", lambda *args: "output\n")
    with pytest.raises(RuntimeError, match="verify_ir=True:.*after-parse"):
        benchmark(DEFAULT_MODEL, repeats=1, warmup=0, levels=("none",))


def test_benchmark_rejects_changed_codegen(monkeypatch):
    monkeypatch.setattr(
        CompilerDriver, "_generate_code",
        lambda driver, program: f"output verify_ir={driver.config.verify_ir}\n",
    )
    with pytest.raises(RuntimeError, match="assembly differs"):
        benchmark(DEFAULT_MODEL, repeats=1, warmup=0, levels=("none",))


def test_benchmark_cli_writes_reports(tmp_path):
    json_path = tmp_path / "reports" / "ir.json"
    markdown_path = tmp_path / "reports" / "ir.md"
    assert main([
        "--repeats", "1", "--warmup", "0",
        "--json-output", str(json_path), "--markdown", str(markdown_path),
    ]) == 0
    report = json.loads(json_path.read_text())
    assert [row["optimize"] for row in report["results"]] == ["none", "basic", "all"]
    assert all(row["output_equal"] for row in report["results"])
    assert "不是模型运行耗时" in markdown_path.read_text()
