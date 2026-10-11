"""Exercise human CI summaries without turning missing execution into success."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.summarize_w4_ci import full_summary, number, read_report, unit_summary


SHA = "a" * 40
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def full_report():
    return {"passed": True, "status": "PASS", "stage": "complete",
        "git": {"head": SHA, "dirty": False}, "elapsed_seconds": 3450.0,
        "build": {"bytes": 11378920}, "fma_conformance": {"passed": True, "status": "PASS"},
        "gates": {"build:riscv-full": True, "smoke:qemu-full": True, "numeric:qemu-full-qwen3": True},
        "cases": [{"name": "full_seed_0", "passed": True,
            "qemu": {"qemu_wall_s": 3351.12, "forward_wall_s": 3350, "exit_code": 0},
            "numeric": {"passed": True, "max_abs": 8.58e-5, "rmse": 5.39e-6,
                        "elements_compared": 38895616, "elements_at_or_above_atol": 0}}]}


def render(report, **kwargs):
    args = dict(probe_outcome="success", gate_outcome="success", audit_outcome="success",
                commit=SHA, case="full_seed_0")
    args.update(kwargs)
    return full_summary(report, {"passed": True, "status": "PASS"}, **args)


def test_full_summary_exposes_actual_metrics_and_folds_detail(full_report):
    text = render(full_report)
    assert text.startswith("## W4 完整 Qwen3 RV64 前向：PASS")
    visible, details = text.split("<details>", 1)
    for value in ("8.58e-05", "5.39e-06", "38,895,616", "11,378,920", "3351.12", "3450", "rtol=0"):
        assert value in visible
    assert "FMA 位模式 / FCSR" in details and "保留数组审计" in details
    assert "<summary>" in details and "</details>" in details
    assert len(text) < 5000


@pytest.mark.parametrize("outcome,label", [("skipped", "NOT RUN"), ("failure", "FAIL"),
    ("cancelled", "CANCELLED"), ("", "UNKNOWN")])
def test_outcome_overrides_a_passing_report(full_report, outcome, label):
    text = render(full_report, probe_outcome=outcome)
    assert label in text.splitlines()[0]
    if outcome == "skipped":
        assert "8.58e-05" not in text and "3351.12" not in text
        assert "未测" in text


@pytest.mark.parametrize("report", [{}, {"passed": False, "status": "FAIL", "stage": "parse"},
    {"passed": False, "status": "BUILD_ONLY", "gates": {"build:riscv-full": True}}])
def test_missing_failed_and_build_only_are_not_numeric_pass(report):
    text = render(report)
    assert not text.splitlines()[0].endswith("：PASS")
    assert "未测" in text and "NOT RUN" in text


@pytest.mark.parametrize("kwargs", [{"gate_outcome": "failure"}, {"gate_outcome": "skipped"},
    {"audit_outcome": "skipped"}, {"audit_outcome": "failure"}, {"audit_outcome": ""}])
def test_incomplete_validation_never_looks_accepted(full_report, kwargs):
    assert "未验收" in render(full_report, **kwargs).splitlines()[0]


def test_stale_identity_is_explicit(full_report):
    full_report["git"]["head"] = "b" * 40
    assert "INVALID" in render(full_report).splitlines()[0]


def test_failed_late_case_is_visible_and_markup_is_escaped(full_report):
    full_report["cases"] *= 100
    failed = copy.deepcopy(full_report["cases"][0])
    failed.update(passed=False, name="<details> | injected\nname")
    failed["numeric"].update(passed=False, max_abs=0.01)
    full_report["cases"].append(failed)
    full_report.update(passed=False, status="FAIL", error="<script>|" + "x" * 10000)
    text = render(full_report, probe_outcome="failure")
    assert "&lt;details&gt; &#124; injected name" in text
    assert "<script>" not in text and len(text) < 6000
    assert "0.01" in text


@pytest.mark.parametrize("value", [True, None, "0", float("nan"), float("inf"), -1, 10**400])
def test_invalid_number_is_not_formatted_as_a_real_measurement(value):
    assert number(value) in ("未记录", "无效")


def test_zero_error_is_preserved():
    assert number(0.0) == "0"


def test_negative_signal_exit_is_preserved(full_report):
    full_report["cases"][0]["qemu"]["exit_code"] = -9
    assert "| -9 |" in render(full_report, probe_outcome="failure")


@pytest.mark.parametrize("outcome", ["failure", "cancelled"])
@pytest.mark.parametrize("content", [None, "<bad xml>"])
def test_failed_unit_with_missing_or_corrupt_xml_does_not_invent_zero_counts(tmp_path, outcome, content):
    path = tmp_path / "tests.xml"
    if content is not None:
        path.write_text(content)
    text = unit_summary(path, outcome)
    assert "FAIL" in text if outcome == "failure" else "CANCELLED" in text
    assert "| — | — | — | — |" in text
    assert "JUnit 报告缺失或无效" in text


def test_unit_counts_failure_and_skips(tmp_path):
    path = tmp_path / "tests.xml"
    path.write_text('<testsuites><testsuite><testcase/><testcase><skipped/></testcase>'
                    '<testcase><failure/></testcase></testsuite></testsuites>')
    text = unit_summary(path, "failure")
    assert "FAIL" in text and "| 1 | 1 | 0 | 1 |" in text
    assert "不代表完整 28 层" in text


def test_missing_empty_or_all_skipped_unit_evidence_cannot_pass(tmp_path):
    path = tmp_path / "tests.xml"
    assert "MISSING" in unit_summary(path, "success")
    for xml in ("<testsuite/>", "<testsuite><testcase><skipped/></testcase></testsuite>"):
        path.write_text(xml)
        assert "：PASS" not in unit_summary(path, "success")
    assert "NOT RUN" in unit_summary(path, "skipped")


@pytest.mark.parametrize("payload", ["[]", "null", "{", '{"cases": []}'])
def test_report_loading_is_tolerant_of_partial_artifacts(tmp_path, payload):
    path = tmp_path / "report.json"
    path.write_text(payload)
    assert isinstance(read_report(path), dict)
    assert read_report(tmp_path / "missing.json") == {}


def test_cli_appends_summary_with_no_probe_dependencies(tmp_path, monkeypatch, full_report):
    folder = tmp_path / "probe"
    folder.mkdir()
    (folder / "report.json").write_text(json.dumps(full_report))
    audit = tmp_path / "probe-audit"
    audit.mkdir()
    (audit / "report.json").write_text('{"passed": true, "status": "PASS"}')
    summary = tmp_path / "summary.md"
    summary.write_text("Earlier summary\n")
    for key, value in {"GITHUB_STEP_SUMMARY": str(summary), "GITHUB_SHA": SHA,
        "SELECTED_CASE": "full_seed_0", "PROBE_OUTCOME": "success", "GATE_OUTCOME": "success",
        "AUDIT_OUTCOME": "success"}.items():
        monkeypatch.setenv(key, value)
    result = subprocess.run([sys.executable, "-I", str(ROOT / "scripts/summarize_w4_ci.py"),
        "--kind", "full", "--report-dir", str(folder), "--output", str(folder / "summary.md")],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert summary.read_text(encoding="utf-8").startswith("Earlier summary\n## W4")
    assert "3351.12" in (folder / "summary.md").read_text(encoding="utf-8")


def test_workflow_publishes_summaries_even_on_failure_and_runs_tests():
    workflow = yaml.load((ROOT / ".github/workflows/w4-full-numeric.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    for job, label in (("unit", "Summarize unit and small QEMU evidence"),
                       ("full-numeric", "Summarize complete RV64 numerical evidence")):
        step = next(item for item in workflow["jobs"][job]["steps"] if item.get("name") == label)
        assert step["if"] == "always()"
        assert "scripts/summarize_w4_ci.py" in step["run"]
        assert all(".outcome" in value for value in step["env"].values())
    assert "scripts/summarize_w4_ci.py" in workflow["on"]["pull_request"]["paths"]
    test_step = next(item for item in workflow["jobs"]["unit"]["steps"] if item.get("id") == "unit_tests")
    assert "tests/test_w4_ci_summary.py" in test_step["run"]
