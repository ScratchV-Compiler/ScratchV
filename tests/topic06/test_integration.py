"""Integration contracts for the Topic 06 benchmark suite."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = ROOT / "tests" / "topic06" / "cases"
RUNNER_PATH = ROOT / "scripts" / "run_topic06_benchmarks.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("topic06_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_topic06_case_manifest_is_complete():
    dsl_files = sorted(CASE_DIR.rglob("*.dsl"))
    meta_files = sorted(CASE_DIR.rglob("*.meta.json"))

    assert len(dsl_files) == 23
    assert len(meta_files) == 23
    assert {path.with_suffix("") for path in dsl_files} == {
        path.with_suffix("").with_suffix("") for path in meta_files
    }


def test_compiler_emits_register_map_for_topic06_case(tmp_path):
    from scratchv.compiler import CompilerConfig, CompilerDriver

    driver = CompilerDriver(CompilerConfig(
        backend="riscv",
        optimize_level="all",
        reg_alloc="greedy",
    ))
    source = CASE_DIR / "activation" / "relu_add.dsl"
    result = driver.compile(str(source), str(tmp_path / "relu_add.s"))

    assert result.success
    assert result.stats["register_map"]["input"] == "t0"
    assert result.stats["register_map"]["bias"] == "t1"


def test_runner_loads_compiler_register_map(tmp_path):
    runner = _load_runner()
    register_map_file = tmp_path / "registers.json"
    register_map_file.write_text(
        json.dumps({
            "schema_version": 1,
            "register_map": {"input": "t0", "bias": "t1"},
        }),
        encoding="utf-8",
    )

    initial = runner.load_initial_registers(
        register_map_file,
        {"input": -3, "bias": 7, "tensor": [1, 2]},
    )

    assert initial == {"t0": -3, "t1": 7}


def test_topic06_report_paths_are_portable():
    runner = _load_runner()
    case = runner.TEST_DIR / "activation" / "relu_only.dsl"

    assert runner._report_path(case) == "tests/topic06/cases/activation/relu_only.dsl"


def test_runner_uses_project_layout():
    runner = _load_runner()

    assert runner.PROJECT_ROOT == ROOT
    assert runner.TEST_DIR == ROOT / "tests" / "topic06" / "cases"
    assert runner.BUILD_DIR == ROOT / "build" / "topic06"
    assert runner.REPORT_DIR == ROOT / "benchmark_reports" / "topic06"
    assert runner.BASELINE_FILE == ROOT / "benchmarks" / "topic06" / "baseline.json"
