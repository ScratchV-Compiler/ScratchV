"""Exercise the Topic 01 benchmark through its public CLI."""

import json
import html
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_report(tmp_path: Path) -> tuple[subprocess.CompletedProcess, dict]:
    json_path = tmp_path / "topic01.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.bench_dsl_control_flow",
            "--repeats",
            "2",
            "--json-output",
            str(json_path),
            "--markdown",
            str(tmp_path / "topic01.md"),
            "--html",
            str(tmp_path / "topic01.html"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    report = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
    return result, report


def test_report_executes_all_topic01_examples(tmp_path):
    result, report = _run_report(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["status"] == "passed"
    assert report["repeats"] == 2
    assert [(case["name"], case["expected"], case["actual"]) for case in report["cases"]] == [
        ("if_else", 7.0, 7.0),
        ("while_sum", 15.0, 15.0),
        ("nested_loop", 6.0, 6.0),
    ]
    assert all(case["llvm_verified"] for case in report["cases"])
    assert all(case["riscv_generated"] for case in report["cases"])
    assert all(case["median_ms"] > 0 for case in report["cases"])


def test_report_collapses_each_case_log_by_default(tmp_path):
    result, report = _run_report(tmp_path)
    assert result.returncode == 0, result.stderr

    for filename in ("topic01.md", "topic01.html"):
        content = (tmp_path / filename).read_text(encoding="utf-8")
        assert "Topic 01 DSL frontend benchmark" in content
        assert content.count("<details>") == len(report["cases"])
        assert content.count("</details>") == len(report["cases"])
        assert "<details open" not in content
        for case in report["cases"]:
            assert f"<summary>{case['name']}:" in content
            expected_source = (
                html.escape(case["source"])
                if filename.endswith(".html")
                else case["source"]
            )
            assert expected_source in content
