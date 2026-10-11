"""Execute the workflow summary shell against completed, skipped and absent evidence."""
import os
from pathlib import Path
import shutil
import subprocess
import textwrap

import pytest


WORKFLOW = Path(__file__).parents[1] / ".github/workflows/llm-deploy.yml"
SUMMARIES = (
    ("Summarize W2 unit gates", ("w2-backend-ops", "w2-runtime", "w2-runtime-model")),
    ("Summarize RISC-V numeric gate", ("qwen3-riscv",)),
    ("Summarize Qwen3 small probe", ("qwen3-small",)),
)


def shell_block(name):
    # Only read the literal run block: never evaluate Actions expressions.
    source = WORKFLOW.read_text(encoding="utf-8")
    step = source.split("      - name: " + name + "\n", 1)[1].split("\n      - ", 1)[0]
    return textwrap.dedent(step.split("        run: |\n", 1)[1])


def execute(tmp_path, name, directories, outcome, have_report):
    bash = os.environ.get("SCRATCHV_BASH") or shutil.which("bash")
    if not bash:
        pytest.skip("Bash is required to exercise GitHub Actions summary commands")
    for directory in directories:
        report = tmp_path / "output" / directory / "report.md"
        report.parent.mkdir(parents=True)
        if have_report:
            report.write_text("SENTINEL measured report\n", encoding="utf-8")
    summary = tmp_path / "summary.md"
    env = dict(os.environ, GITHUB_STEP_SUMMARY=summary.as_posix())
    for key in ("FRONTEND_OUTCOME", "BACKEND_OUTCOME", "RUNTIME_OUTCOME", "RUNTIME_MODEL_OUTCOME",
                "PROBE_OUTCOME", "TEST_OUTCOME", "ARTIFACT_TEST_OUTCOME"):
        env[key] = outcome
    result = subprocess.run([bash, "-eu", "-c", shell_block(name)], cwd=tmp_path,
                            env=env, capture_output=True, text=True, encoding="utf-8", timeout=10)
    assert result.returncode == 0, result.stderr
    return summary.read_text(encoding="utf-8")


@pytest.mark.parametrize("name,directories", SUMMARIES)
@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_ran_probe_keeps_available_report(tmp_path, name, directories, outcome):
    summary = execute(tmp_path, name, directories, outcome, True)
    assert summary.count("SENTINEL measured report") == len(directories)
    assert outcome in summary


@pytest.mark.parametrize("name,directories", SUMMARIES)
@pytest.mark.parametrize("outcome", ["skipped", ""])
def test_unrun_probe_never_reuses_stale_report(tmp_path, name, directories, outcome):
    summary = execute(tmp_path, name, directories, outcome, True)
    assert summary.count("Not run / 未运行") == len(directories)
    assert "SENTINEL" not in summary
    assert "No " in summary and "acceptance result" in summary


@pytest.mark.parametrize("name,directories", SUMMARIES)
@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
def test_absent_report_does_not_invent_acceptance(tmp_path, name, directories, outcome):
    summary = execute(tmp_path, name, directories, outcome, False)
    assert summary.count("Report unavailable / 未生成报告") == len(directories)
    assert "No " in summary and "acceptance result" in summary
    assert "SENTINEL" not in summary
