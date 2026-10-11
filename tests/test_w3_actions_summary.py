"""W3 Actions pages must distinguish measured results, partial coverage and skips."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.summarize_w3_ci import GATES, main, render

ROOT = Path(__file__).resolve().parents[1]


def save(folder, report):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")


def metric(value, name):
    return {"name": name, "passed": True, "max_abs": value,
            "relative_l2": 2e-7, "cosine_similarity": 0.99999}


def full_report(**extra):
    return {"gate": "numeric:ir-full-qwen3", "status": "PASS", "passed": True,
            "coverage_complete": True, "fp32_mode": "reference", "atol": 1e-4,
            "elapsed_seconds": 75.125, "required_cases": ["one", "two"], "selected_cases": ["one", "two"],
            "cases": [{"name": name, "passed": True, "comparison": {
                "logits": [metric(value, "logits.npy"), metric(value * 2, "diagnostic_logits.npy")],
                "checkpoints": [metric(0.7, "layer_3.output")]}}
                for name, value in [("one", 1e-6), ("two", 3e-6)]], **extra}


def test_full_key_metrics_are_visible_and_checkpoint_details_are_folded(tmp_path):
    report = full_report()
    before = deepcopy(report)
    save(tmp_path, report)
    text = render("full", tmp_path, "success")
    visible = text.split("<details>", 1)[0]
    assert "**PASS**" in visible and "75.125 s" in visible
    assert "2/2 cases" in visible and "reference" in visible
    assert format(3e-6, ".16e") in visible and format(6e-6, ".16e") in visible
    assert format(0.7, ".16e") not in visible
    assert "No full-model QEMU result" in visible
    assert "w3-full-numeric-reports" in text and "<details>" in text
    assert json.loads((tmp_path / "report.json").read_text()) == before


@pytest.mark.parametrize("outcome,expected", [("skipped", "NOT RUN"), ("", "NOT RUN"), ("cancelled", "CANCELLED")])
def test_skipped_step_does_not_reuse_a_passing_report(tmp_path, outcome, expected):
    save(tmp_path, full_report())
    text = render("full", tmp_path, outcome)
    assert f"**{expected}**" in text and "PASS" not in text
    assert "最大绝对误差" not in text


@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_missing_report_does_not_claim_pass(tmp_path, outcome):
    text = render("full", tmp_path, outcome)
    assert "MISSING REPORT" in text and "PASS" not in text and outcome in text


@pytest.mark.parametrize("invalid", ["{", "[]", '{"gate":"wrong"}', '{"gate":"numeric:ir-full-qwen3","cases":null}'])
def test_invalid_reports_do_not_crash_summary(tmp_path, invalid):
    (tmp_path / "report.json").write_text(invalid)
    text = render("full", tmp_path, "failure")
    assert "INVALID REPORT" in text and "PASS" not in text


def test_failed_step_overrides_passing_report(tmp_path):
    save(tmp_path, full_report())
    text = render("full", tmp_path, "failure")
    assert "**FAIL (step did not succeed)**" in text
    assert "**PASS**" not in text


def test_partial_coverage_shows_unexecuted_case_and_missing_metrics(tmp_path):
    report = full_report(status="PARTIAL", passed=False, coverage_complete=False, selected_cases=["one"])
    report["cases"] = report["cases"][:1]
    save(tmp_path, report)
    text = render("full", tmp_path, "failure")
    assert "**PARTIAL (not acceptance)**" in text
    assert "1/2 cases recorded" in text and "NOT RUN" in text and "two" in text
    assert "未记录 1/2" in text and "**PASS**" not in text


def test_claimed_pass_with_incomplete_coverage_is_not_pass(tmp_path):
    save(tmp_path, full_report(coverage_complete=False))
    assert "**INCOMPLETE (not acceptance)**" in render("full", tmp_path, "success")


def test_preparation_lists_all_checks_and_never_claims_full_model_execution(tmp_path):
    report = {"gate": "preparation:w3", "status": "FAIL", "passed": False, "elapsed_seconds": 32,
              "gates": [{"name": "attention", "passed": False, "status": "FAIL", "elapsed_seconds": 1.5}]}
    save(tmp_path, report)
    save(tmp_path / "attention", {"gate": "unit:attention-backend", "status": "PASS", "passed": True,
         "cases": [{"name": "one", "executions": [{"optimization": level, "qemu_vs_ort": metric(3e-6, "logits")}
                                                    for level in ("none", "all")]}]})
    # A stale medium child without a corresponding parent invocation is ignored.
    save(tmp_path / "medium", {"gate": "w3-medium-host", "status": "PASS", "passed": True})
    text = render("preparation", tmp_path, "failure")
    visible = text.split("<details>", 1)[0]
    assert "NOT RUN by this job" in visible
    assert "FAIL (parent validation)" in visible and "1 cases / 2 target executions recorded" in visible
    for _, _, scope in GATES:
        assert scope in visible
    assert visible.count("NOT RUN") == 5
    assert "w3-preparation-reports" in text


def test_missing_child_report_cannot_leave_overall_pass(tmp_path):
    save(tmp_path, {"gate": "preparation:w3", "status": "PASS", "passed": True,
                   "gates": [{"name": "medium", "status": "PASS", "passed": True}]})
    text = render("preparation", tmp_path, "success")
    assert "**INCOMPLETE REPORTS (not acceptance)**" in text and "MISSING REPORT" in text
    assert "**PASS**" not in text


def test_errors_are_escaped_and_arbitrary_machine_evidence_is_not_dumped(tmp_path):
    report = full_report(status="FAIL", passed=False, error="<script>" + "x" * 100000)
    report["raw_source"] = "DO_NOT_DUMP_" * 100000
    save(tmp_path, report)
    text = render("full", tmp_path, "failure")
    assert "<script>" not in text and "&lt;script&gt;" in text
    assert "DO_NOT_DUMP" not in text and len(text.encode()) < 16000


def test_cli_appends_to_github_summary_without_dependencies_or_mutation(tmp_path, monkeypatch):
    destination = tmp_path / "summary.md"
    destination.write_text("Prior summary\n")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(destination))
    monkeypatch.setenv("PROBE_OUTCOME", "skipped")
    assert main(["--kind", "full", "--report-dir", str(tmp_path / "absent")]) == 0
    assert destination.read_text().startswith("Prior summary\n## W3")
    # -S excludes site-packages: a failed dependency install still gets a summary.
    process = subprocess.run([sys.executable, "-S", "-X", "utf8", str(ROOT / "scripts/summarize_w3_ci.py"),
                              "--kind", "full", "--report-dir", str(tmp_path / "absent"), "--outcome", "failure"],
                             cwd=tmp_path, env=os.environ.copy(), text=True, capture_output=True)
    assert process.returncode == 0, process.stderr
    assert "MISSING REPORT" in destination.read_text()
