"""Exercise actual W3 workflow scripts and the opt-in/acceptance contracts.

This is local configuration validation, not evidence of a GitHub/Linux run.

The W3 gates used to live in three separate workflow files that llm-deploy.yml
called through ``workflow_call``. They are now jobs of llm-deploy.yml itself,
so every assertion below anchors on a job id instead of a file name. The
contracts themselves are unchanged: the gates stay opt-in, they stay hardened
against masked failures, and a partial or failed gate can never pass.
"""

import ast
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
PIPELINE = "llm-deploy.yml"
W3_JOBS = ("w3-preparation", "w3-full-numeric")


def workflow(name):
    # BaseLoader preserves the GitHub key "on" instead of YAML 1.1 boolean True.
    return yaml.load((WORKFLOWS / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def job(name, job_id):
    return workflow(name)["jobs"][job_id]


def step(name, job_id, label):
    return next(item for item in job(name, job_id).get("steps", [])
                if item.get("name") == label)


@pytest.fixture(scope="module")
def bash():
    candidates = [os.environ.get("SCRATCHV_BASH"), shutil.which("bash")]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            result = subprocess.run([candidate, "-c", "exit 0"], capture_output=True)
            if result.returncode == 0:
                return candidate
    pytest.skip("Bash is required to execute the Linux workflow shell snippets")


def run_script(bash, script, tmp_path, **values):
    env = os.environ.copy()
    env.update({key: str(value) for key, value in values.items()})
    return subprocess.run([bash, "--noprofile", "--norc", "-c", script],
                          cwd=tmp_path, env=env, capture_output=True, text=True)


def evaluate_guard(expression, event, enabled, w3_validation=False):
    """Evaluate only the tiny boolean subset used by the real job guards."""
    value = expression.removeprefix("${{").removesuffix("}}").strip()
    value = value.replace("github.event_name", repr(event))
    value = value.replace("vars.W3_NIGHTLY_ENABLED", repr(enabled))
    value = value.replace("inputs.run_w3_validation", repr(w3_validation))
    value = value.replace("always()", "True").replace("||", "or").replace("&&", "and")

    def visit(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BoolOp):
            values = [bool(visit(item)) for item in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
            # GitHub string equality is case insensitive.
            left, right = visit(node.left), visit(node.comparators[0])
            if isinstance(left, str) and isinstance(right, str):
                return left.casefold() == right.casefold()
            return left == right
        raise AssertionError(f"Unsupported guard node: {ast.dump(node)}")

    return bool(visit(ast.parse(value, mode="eval").body))


def test_one_registered_pipeline_owns_the_whole_llm_line():
    data = workflow(PIPELINE)
    assert data["name"] == "LLM Deploy v1.0"
    # One file, one schedule, one manual entry point. The folded files used to
    # schedule the same 23:19 UTC nightly run twice.
    assert set(data["on"]) == {"pull_request", "schedule", "workflow_dispatch"}
    assert data["on"]["schedule"] == [{"cron": "23 19 * * *"}]
    assert data["permissions"] == {"contents": "read"}
    for folded in ("w3-preparation.yml", "w3-full-numeric.yml", "w3-nightly.yml",
                   "linux-reproduction-docs.yml"):
        assert not (WORKFLOWS / folded).exists(), f"{folded} should be folded into {PIPELINE}"
    # The old indirection job is gone; the W3 gates are real jobs now.
    assert "w3-validation" not in data["jobs"]
    assert set(W3_JOBS) <= set(data["jobs"])


@pytest.mark.parametrize("job_id", W3_JOBS)
def test_w3_gates_are_hardened_and_publish_evidence(job_id):
    data = job(PIPELINE, job_id)
    assert data["runs-on"] == "ubuntu-24.04"
    assert data["concurrency"]["cancel-in-progress"] == "false"
    assert "continue-on-error" not in data
    for item in data["steps"]:
        assert "continue-on-error" not in item
    reports = [item for item in data["steps"] if item.get("uses", "").startswith("actions/upload-artifact")]
    assert all(item["if"] == "always()" for item in reports)
    assert any("w3-runner-resources.txt" in item["with"]["path"] for item in reports)


def test_preparation_fetches_history_required_by_real_onnx_benchmark_regression():
    steps = job(PIPELINE, "w3-preparation")["steps"]
    checkout = next(item for item in steps if item.get("uses", "").startswith("actions/checkout@"))
    # test_onnx_operator_benchmark exports an actual historical revision via
    # git archive. The default depth=1 checkout would omit that baseline.
    assert checkout["with"].get("fetch-depth") == "0"
    regression = next(item for item in steps if item.get("name") == "W3 and affected W2/IR regressions")
    assert "tests/test_onnx*.py" in regression["run"]


@pytest.mark.parametrize("event,enabled,w3_validation,expected", [
    # Pull requests never reach the W3 gates, whatever the opt-in says.
    ("pull_request", "true", True, False),
    ("pull_request", "", False, False),
    # The nightly schedule needs the repository opt-in variable.
    ("schedule", "", False, False), ("schedule", "false", False, False),
    ("schedule", "1", False, False), ("schedule", "true", False, True),
    ("schedule", "TRUE", False, True),
    # A manual dispatch needs the explicit flag, not the nightly variable.
    ("workflow_dispatch", "", False, False),
    ("workflow_dispatch", "true", False, False),
    ("workflow_dispatch", "", True, True),
    ("workflow_dispatch", "false", True, True),
])
def test_expensive_w3_jobs_require_opt_in_or_explicit_dispatch(event, enabled, w3_validation, expected):
    for job_id in W3_JOBS:
        assert evaluate_guard(job(PIPELINE, job_id)["if"], event, enabled, w3_validation) is expected
    acceptance = job(PIPELINE, "w3-acceptance")
    assert evaluate_guard(acceptance["if"], event, enabled, w3_validation) is expected
    assert set(acceptance["needs"]) == set(W3_JOBS)
    # Distinct named groups so the three jobs never block each other on one ref.
    groups = {job(PIPELINE, job_id)["concurrency"]["group"] for job_id in W3_JOBS}
    assert len(groups) == len(W3_JOBS)


def test_manual_dispatch_runs_both_gates_even_with_the_nightly_opt_in_disabled():
    for job_id in (*W3_JOBS, "w3-acceptance"):
        assert evaluate_guard(job(PIPELINE, job_id)["if"], "workflow_dispatch", "", True) is True


@pytest.mark.parametrize("selected,expected", [
    ("", 0), ("all", 0), ("short_17", 0), ("full_seed_0", 1),
    ("--help", 1), ("all; exit 0", 1),
])
def test_case_input_is_validated_before_install(bash, tmp_path, selected, expected):
    check = step(PIPELINE, "w3-full-numeric", "Validate requested coverage")["run"]
    result = run_script(bash, check, tmp_path, SELECTED_CASE=selected)
    assert result.returncode == expected, result.stderr


@pytest.mark.parametrize("selected", ["", "all", "short_17"])
@pytest.mark.parametrize("runner_status", [0, 1, 2])
def test_actual_full_command_preserves_arguments_and_failure(bash, tmp_path, selected, runner_status):
    command = step(PIPELINE, "w3-full-numeric", "Compare complete IR and ORT outputs")["run"]
    # Only replace the executable; execute the actual workflow shell body.
    wrapper = 'python() { printf "%s\\n" "$@" > "$ARG_LOG"; return "$MOCK_RETURN"; }\n'
    capture = tmp_path / "arguments.txt"
    result = run_script(bash, wrapper + command, tmp_path, SELECTED_CASE=selected,
                        ARG_LOG=capture.as_posix(), MOCK_RETURN=runner_status)
    assert result.returncode == runner_status, result.stderr
    args = capture.read_text().splitlines()
    assert args[args.index("--fp32-mode") + 1] == "reference"
    if selected == "short_17":
        assert args[-2:] == ["--case", "short_17"]
    else:
        assert "--case" not in args  # All seven cases remain selected by the runner.


@pytest.mark.parametrize("preparation,full,passed", [
    ("success", "success", True), ("failure", "success", False),
    ("success", "failure", False), ("success", "skipped", False),
    ("cancelled", "success", False), ("skipped", "skipped", False),
])
def test_acceptance_summary_cannot_turn_missing_or_failed_gate_green(bash, tmp_path, preparation, full, passed):
    command = step(PIPELINE, "w3-acceptance", "Require both complete gates")["run"]
    summary = tmp_path / "summary.md"
    result = run_script(bash, command, tmp_path, PREPARATION_RESULT=preparation,
                        FULL_NUMERIC_RESULT=full, GITHUB_SHA="a" * 40,
                        GITHUB_STEP_SUMMARY=summary.as_posix())
    assert (result.returncode == 0) is passed
    assert "不批准 W3 团队里程碑" in summary.read_text()


def test_full_coverage_defaults_and_raw_evidence_retention():
    data = workflow(PIPELINE)
    case = data["on"]["workflow_dispatch"]["inputs"]["w3_case"]
    assert case["default"] == "all"
    assert set(case["options"]) == {"all", "short_17"}
    assert data["jobs"]["w3-full-numeric"]["env"]["SELECTED_CASE"] == "${{ inputs.w3_case || 'all' }}"
    uploads = [item["with"] for item in data["jobs"]["w3-full-numeric"]["steps"]
               if item.get("uses", "").startswith("actions/upload-artifact")]
    raw = next(item for item in uploads if item["name"] == "w3-full-numeric-raw")
    assert raw["retention-days"] == "3"
    for required in ("inputs.npz", "logits.npy", "diagnostic_logits.npy", "checkpoints.npz", "report.json"):
        assert required in raw["path"]


@pytest.mark.parametrize("job_id,probe_name,summary_name,kind", [
    ("w3-preparation", "Five preparation checks (not full-model acceptance)", "Summarize W3 preparation evidence", "preparation"),
    ("w3-full-numeric", "Compare complete IR and ORT outputs", "Summarize W3 full numerical evidence", "full"),
])
def test_actual_numeric_summary_steps_run_after_failure_and_keep_original_exit_codes(bash, tmp_path, job_id, probe_name, summary_name, kind):
    probe = step(PIPELINE, job_id, probe_name)
    summary = step(PIPELINE, job_id, summary_name)
    assert probe["id"] == "probe"
    assert summary["if"] == "always()"
    assert summary["env"] == {"PROBE_OUTCOME": "${{ steps.probe.outcome }}"}
    assert "continue-on-error" not in probe
    wrapper = 'python() { printf "%s\\n" "$@" > "$ARG_LOG"; }\n'
    capture = tmp_path / "arguments.txt"
    result = run_script(bash, wrapper + summary["run"], tmp_path, ARG_LOG=capture.as_posix())
    assert result.returncode == 0, result.stderr
    args = capture.read_text().splitlines()
    assert "scripts/summarize_w3_ci.py" in args and args[args.index("--kind") + 1] == kind


@pytest.mark.parametrize("result,label", [("skipped", "NOT RUN"), ("success", "PASS"), ("failure", "FAIL"),
                                         ("cancelled", "CANCELLED"), ("", "UNCONFIRMED")])
def test_llm_page_reports_w3_skip_without_running_or_accepting_w3(bash, tmp_path, result, label):
    summary_job = job(PIPELINE, "w3-execution-summary")
    assert summary_job["if"] == "always()" and summary_job["needs"] == ["w3-acceptance"]
    assert not any("uses" in item for item in summary_job["steps"])
    command = summary_job["steps"][0]["run"]
    # Run the actual dependency-free Python body on hosts without python3 alias.
    body = command.split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    import sys
    destination = tmp_path / "summary.md"
    env = dict(os.environ, W3_RESULT=result, GITHUB_STEP_SUMMARY=str(destination))
    process = subprocess.run([sys.executable, "-X", "utf8", "-c", body], env=env,
                             capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    text = destination.read_text(encoding="utf-8")
    assert f"**{label}**" in text and "<details>" in text
    assert ("W1/W2 通过不等于 W3 验收" in text) is (result == "skipped")
    assert len(text) < 1600


@pytest.mark.parametrize("job_id", W3_JOBS)
def test_all_actual_bash_blocks_parse(bash, tmp_path, job_id):
    for item in job(PIPELINE, job_id).get("steps", []):
        if "run" in item:
            result = subprocess.run([bash, "-n"], input=item["run"],
                                    text=True, capture_output=True, cwd=tmp_path)
            assert result.returncode == 0, f"{job_id}: {item.get('name')}: {result.stderr}"
