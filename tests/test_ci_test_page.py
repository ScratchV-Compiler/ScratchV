"""Pages must publish this run's pytest results, including errors and skips."""

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from scratchv.ci import test_page

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def reports(tmp_path, monkeypatch):
    monkeypatch.setattr(test_page, "HIST_FILE", tmp_path / "test_history.json")
    return tmp_path / "test_results.xml", tmp_path / "tests.html"


@pytest.mark.parametrize("xml", [None, "", "<test", "<testsuites><testsuite/></testsuites>"],
                         ids=["missing", "blank", "malformed", "empty-suite"])
def test_invalid_junit_cannot_create_a_page_or_update_history(reports, xml):
    source, output = reports
    if xml is not None:
        source.write_text(xml)
    assert test_page.main(["--junit", str(source), "-o", str(output)]) == 1
    assert not output.exists() and not test_page.HIST_FILE.exists()


def test_real_junit_statuses_are_counted_once_and_skips_are_excluded_from_rate(reports):
    source, output = reports
    source.write_text("""<testsuites><testsuite><testsuite>
      <testcase classname="tests.test_backend.TestNormal" name="a"/>
      <testcase classname="tests.test_backend.TestNormal" name="b"><skipped/></testcase>
      <testcase classname="tests.test_ir.TestInvalid" name="c"><failure/></testcase>
      <testcase classname="tests.test_ir.TestInvalid" name="d"><error/></testcase>
    </testsuite></testsuite></testsuites>""")
    assert test_page.main(["--junit", str(source), "-o", str(output)]) == 0
    page = output.read_text()
    assert "2 个模块 · 4 个测试用例" in page
    assert "33%" in page and 'href="test_results.xml"' in page
    assert "1/1 已执行通过 · 1 跳过" in page
    entry, = json.loads(test_page.HIST_FILE.read_text())
    assert {key: entry[key] for key in ("pass", "fail", "errors", "skipped", "rate")} == {
        "pass": 1, "fail": 1, "errors": 1, "skipped": 1, "rate": 33.3,
    }


def test_only_skipped_is_reported_without_a_pass_rate(reports):
    source, output = reports
    source.write_text("<testsuite><testcase><skipped/></testcase></testsuite>")
    assert test_page.main(["--junit", str(source), "-o", str(output)]) == 0
    assert "无已执行用例" in output.read_text()
    entry, = json.loads(test_page.HIST_FILE.read_text())
    assert entry["skipped"] == 1 and entry["rate"] is None


def test_pages_publishes_test_job_evidence_only_after_ci_passes(tmp_path):
    config = yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
    jobs = config["jobs"]
    steps = jobs["deploy-pages"]["steps"]
    generators = [job for job, value in jobs.items() for step in value.get("steps", [])
                  if "scratchv/ci/test_page.py" in step.get("run", "")]
    assert generators == ["test"]
    assert jobs["deploy-pages"]["needs"] == "acceptance"
    assert jobs["deploy-pages"]["if"] == (
        "${{ !cancelled() && github.ref == 'refs/heads/main' && needs.acceptance.result == 'success' }}"
    )
    assert {"test", "benchmark"} <= set(jobs["acceptance"]["needs"])
    assert "test" not in jobs["benchmark"].get("needs", [])
    assert "benchmark" not in jobs["test"].get("needs", [])
    download = {step["with"]["name"]: step["with"]["path"] for step in steps
                if step.get("uses", "").startswith("actions/download-artifact@")}
    assert download == {"benchmark-reports": "benchmark_reports",
                        "peephole-benchmark-report": "benchmark_reports",
                        "test-reports": "test-reports"}
    copy = next(step["run"] for step in steps if step["name"] == "Include the tested page and its JUnit source")
    reports = tmp_path / download["test-reports"]
    site = tmp_path / download["benchmark-reports"]
    reports.mkdir()
    site.mkdir()
    (reports / "tests.html").write_text("current test page")
    (reports / "test_history.json").write_text("[]")
    assert subprocess.run(["bash", "-c", copy], cwd=tmp_path).returncode != 0
    (reports / "test_results.xml").write_text("<testsuite><testcase/></testsuite>")
    subprocess.run(["bash", "-c", copy], cwd=tmp_path, check=True)
    for name in ("tests.html", "test_results.xml", "test_history.json"):
        assert (site / name).read_bytes() == (reports / name).read_bytes()
