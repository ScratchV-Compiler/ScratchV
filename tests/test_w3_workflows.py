"""Exercise actual W3 workflow scripts and the opt-in/acceptance contracts.

This is local configuration validation, not evidence of a GitHub/Linux run.
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


def workflow(name):
    # BaseLoader preserves the GitHub key "on" instead of YAML 1.1 boolean True.
    return yaml.load((WORKFLOWS / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def step(name, label):
    jobs = workflow(name)["jobs"]
    return next(item for job in jobs.values() for item in job.get("steps", [])
                if item.get("name") == label)


@pytest.fixture(scope="module")
def bash():
    candidates = [shutil.which("bash"), "C:/Program Files/Git/bin/bash.exe"]
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


@pytest.mark.parametrize("name", ["w3-preparation.yml", "w3-full-numeric.yml"])
def test_workflows_keep_manual_and_same_commit_reusable_entries(name):
    data = workflow(name)
    assert set(data["on"]) == {"workflow_dispatch", "workflow_call"}
    assert data["permissions"] == {"contents": "read"}
    assert data["concurrency"]["cancel-in-progress"] == "false"
    for job in data["jobs"].values():
        assert job["runs-on"] == "ubuntu-24.04"
        assert "continue-on-error" not in job
        for item in job["steps"]:
            assert "continue-on-error" not in item
        reports = [item for item in job["steps"] if item.get("uses", "").startswith("actions/upload-artifact")]
        assert all(item["if"] == "always()" for item in reports)
        assert any("w3-runner-resources.txt" in item["with"]["path"] for item in reports)


def test_preparation_fetches_history_required_by_real_onnx_benchmark_regression():
    steps = workflow("w3-preparation.yml")["jobs"]["preparation"]["steps"]
    checkout = next(item for item in steps if item.get("uses", "").startswith("actions/checkout@"))
    # test_onnx_operator_benchmark exports an actual historical revision via
    # git archive. The default depth=1 checkout would omit that baseline.
    assert checkout["with"].get("fetch-depth") == "0"
    regression = next(item for item in steps if item.get("name") == "W3 and affected W2/IR regressions")
    assert "tests/test_onnx*.py" in regression["run"]


@pytest.mark.parametrize("event,enabled,expected", [
    ("schedule", "", False), ("schedule", "false", False),
    ("schedule", "1", False), ("schedule", "true", True),
    ("schedule", "TRUE", True), ("workflow_dispatch", "", True),
    ("workflow_dispatch", "false", True),
])
def test_nightly_expensive_jobs_require_opt_in_or_explicit_dispatch(event, enabled, expected):
    data = workflow("w3-nightly.yml")
    assert set(data["on"]) == {"schedule", "workflow_dispatch", "workflow_call"}
    assert data["on"]["schedule"] == [{"cron": "23 19 * * *"}]
    for job in data["jobs"].values():
        assert evaluate_guard(job["if"], event, enabled) is expected
    assert data["jobs"]["preparation"]["uses"] == "./.github/workflows/w3-preparation.yml"
    assert data["jobs"]["full-numeric"]["uses"] == "./.github/workflows/w3-full-numeric.yml"
    assert data["jobs"]["full-numeric"]["with"] == {"case": "all"}
    assert set(data["jobs"]["acceptance"]["needs"]) == {"preparation", "full-numeric"}
    # Different caller/callee concurrency groups avoid self-blocking reuse.
    groups = {workflow(name)["concurrency"]["group"] for name in (
        "w3-nightly.yml", "w3-preparation.yml", "w3-full-numeric.yml")}
    assert len(groups) == 3


@pytest.mark.parametrize("event", ["pull_request", "schedule", "workflow_dispatch"])
@pytest.mark.parametrize("selected", [False, True])
def test_existing_manual_entry_runs_w3_only_when_explicitly_requested(event, selected):
    data = workflow("llm-deploy.yml")
    option = data["on"]["workflow_dispatch"]["inputs"]["run_w3_validation"]
    assert option["type"] == "boolean" and option["default"] == "false"
    job = data["jobs"]["w3-validation"]
    assert job["uses"] == "./.github/workflows/w3-nightly.yml"
    assert evaluate_guard(job["if"], event, "", selected) is (
        event == "workflow_dispatch" and selected)
    # The called workflow inherits the caller event. A manual call must run
    # both complete gates even while the scheduled nightly opt-in is disabled.
    if event == "workflow_dispatch" and selected:
        assert all(evaluate_guard(item["if"], event, "")
                   for item in workflow("w3-nightly.yml")["jobs"].values())


@pytest.mark.parametrize("selected,expected", [
    ("", 0), ("all", 0), ("short_17", 0), ("full_seed_0", 1),
    ("--help", 1), ("all; exit 0", 1),
])
def test_reusable_case_input_is_validated_before_install(bash, tmp_path, selected, expected):
    check = step("w3-full-numeric.yml", "Validate requested coverage")["run"]
    result = run_script(bash, check, tmp_path, SELECTED_CASE=selected)
    assert result.returncode == expected, result.stderr


@pytest.mark.parametrize("selected", ["", "all", "short_17"])
@pytest.mark.parametrize("runner_status", [0, 1, 2])
def test_actual_full_command_preserves_arguments_and_failure(bash, tmp_path, selected, runner_status):
    command = step("w3-full-numeric.yml", "Compare complete IR and ORT outputs")["run"]
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
def test_nightly_summary_cannot_turn_missing_or_failed_gate_green(bash, tmp_path, preparation, full, passed):
    command = step("w3-nightly.yml", "Require both complete gates")["run"]
    summary = tmp_path / "summary.md"
    result = run_script(bash, command, tmp_path, PREPARATION_RESULT=preparation,
                        FULL_NUMERIC_RESULT=full, GITHUB_SHA="a" * 40,
                        GITHUB_STEP_SUMMARY=summary.as_posix())
    assert (result.returncode == 0) is passed
    assert "does not approve the W3 team milestone" in summary.read_text()


def test_full_coverage_defaults_and_raw_evidence_retention():
    data = workflow("w3-full-numeric.yml")
    for trigger in ("workflow_dispatch", "workflow_call"):
        assert data["on"][trigger]["inputs"]["case"]["default"] == "all"
    assert data["jobs"]["full-numeric"]["env"]["SELECTED_CASE"] == "${{ inputs.case || 'all' }}"
    uploads = [item["with"] for item in data["jobs"]["full-numeric"]["steps"]
               if item.get("uses", "").startswith("actions/upload-artifact")]
    raw = next(item for item in uploads if item["name"] == "w3-full-numeric-raw")
    assert raw["retention-days"] == "3"
    for required in ("inputs.npz", "logits.npy", "diagnostic_logits.npy", "checkpoints.npz", "report.json"):
        assert required in raw["path"]


@pytest.mark.parametrize("name", ["w3-preparation.yml", "w3-full-numeric.yml", "w3-nightly.yml"])
def test_all_actual_bash_blocks_parse(bash, tmp_path, name):
    for job in workflow(name)["jobs"].values():
        for item in job.get("steps", []):
            if "run" in item:
                result = subprocess.run([bash, "-n"], input=item["run"],
                                        text=True, capture_output=True, cwd=tmp_path)
                assert result.returncode == 0, f"{name}: {item.get('name')}: {result.stderr}"
