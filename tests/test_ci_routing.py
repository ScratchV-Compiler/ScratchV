"""Exercise PR routing against real Git history and the final fail-closed gate."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scope = load_script("ci_scope")
summary = load_script("summarize_ci")
CONFIG = {"llm": ["scratchv/**", "probes/**"], "w4": []}


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout.strip()


def write(root, path, text="test\n"):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def commit(root, message):
    git(root, "add", "-A")
    git(root, "commit", "-qm", message)
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "CI routing test")
    git(root, "config", "user.email", "ci-routing@example.invalid")
    git(root, "config", "core.autocrlf", "false")
    write(root, "README.md")
    write(root, "docs/old.md")
    write(root, "scratchv/delete.py")
    base = commit(root, "base")
    config = tmp_path / "scopes.json"
    config.write_text(json.dumps(CONFIG), encoding="utf-8")
    event = tmp_path / "event.json"
    return root, base, config, event


def set_event(event, base, head):
    event.write_text(json.dumps({"pull_request": {"base": {"sha": base}, "head": {"sha": head}}}),
                     encoding="utf-8")


def test_real_diff_includes_rename_source_deletion_and_space_names(repository):
    root, base, config, event = repository
    (root / "scratchv/delete.py").unlink()
    (root / "docs/old.md").rename(root / "moved outside docs.md")
    write(root, "unrelated name with spaces.txt")
    head = commit(root, "rename and delete")
    set_event(event, base, head)
    paths = scope.changed_paths(root, base, head)
    assert set(paths) == {"scratchv/delete.py", "docs/old.md", "moved outside docs.md",
                          "unrelated name with spaces.txt"}
    assert scope.detect_scopes(root, "pull_request", event, config) == (
        {"llm": True, "w4": False}, 4)


def test_merge_base_diff_excludes_base_branch_only_changes(repository):
    root, base, config, event = repository
    write(root, "scratchv/base_only.py")
    advanced_base = commit(root, "advance base")
    git(root, "checkout", "--detach", base)
    write(root, "README.md", "head change\n")
    head = commit(root, "PR branch")
    set_event(event, advanced_base, head)
    assert scope.detect_scopes(root, "pull_request", event, config) == (
        {"llm": False, "w4": False}, 1)


@pytest.mark.parametrize("path,expected", [
    ("scratchv/backend/test.py", (True, False)),
    ("probes/model/run.py", (True, False)),
    ("docs/nested/readme.md", (False, False)),
    ("README.md", (False, False)),
    (".github/workflows/test.yml", (True, False)),
    ("scripts/ci_scopes.json", (True, False)),
    ("scripts/summarize_ci.py", (True, False)),
    ("tests/test_ci_routing.py", (True, False)),
])
def test_path_scopes_and_shared_configuration(path, expected):
    assert tuple(scope.select_scopes(CONFIG, [path]).values()) == expected


@pytest.mark.parametrize("path,pattern,expected", [
    ("probes/README.md", "probes/**/*.md", True),
    ("probes/model/README.md", "probes/**/*.md", True),
    ("probes/model/deeper/README.md", "probes/**/*.md", True),
    ("probes/model/README.md", "probes/*.md", False),
    ("README.md", "**/*.md", True),
    ("docs/README.md", "**/*.md", True),
    ("tests/nested/test_ci_routing.py", "tests/test_ci_*", False),
    ("tests/test_ci_routing.py", "tests/test_ci_*", True),
    ("probes/README.txt", "probes/**/*.md", False),
    ("probes/readme.MD", "probes/**/*.md", False),
])
def test_github_path_globs_keep_segment_boundaries(path, pattern, expected):
    assert scope.matches_path(path, pattern) is expected


def test_delivered_w4_scope_receives_matching_and_shared_changes():
    config = {**CONFIG, "w4": ["probes/w4_*/**"]}
    assert scope.select_scopes(config, ["probes/w4_full/run.py"])["w4"]
    assert scope.select_scopes(config, ["scripts/ci_scope.py"])["w4"]


@pytest.mark.parametrize("event_name", ["push", "schedule", "workflow_dispatch"])
def test_non_pr_never_activates_scopes_or_needs_event(repository, event_name):
    root, _, config, event = repository
    event.write_text("invalid JSON", encoding="utf-8")
    assert scope.detect_scopes(root, event_name, event, config) == (
        {"llm": False, "w4": False}, 0)


@pytest.mark.parametrize("event_name", ["pull_request_target", "", "unexpected", "pull_request\n"])
def test_unknown_or_missing_event_fails_closed(repository, event_name):
    root, _, config, event = repository
    with pytest.raises(ValueError):
        scope.detect_scopes(root, event_name, event, config)


@pytest.mark.parametrize("value", [None, [], {}, {"llm": []}, {**CONFIG, "unexpected": []},
                                 {**CONFIG, "llm": "scratchv/**"}, {**CONFIG, "llm": [""]},
                                 {**CONFIG, "llm": [None]}])
def test_invalid_scope_configuration_fails(tmp_path, value):
    path = tmp_path / "scope.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        scope.read_config(path)


@pytest.mark.parametrize("reader", [scope.read_config, scope.pr_commits])
def test_malformed_json_fails_before_routing(tmp_path, reader):
    path = tmp_path / "invalid.json"
    path.write_text("{invalid JSON", encoding="utf-8")
    with pytest.raises(ValueError):
        reader(path)


@pytest.mark.parametrize("value", [None, [], {}, {"pull_request": None},
                                 {"pull_request": {"base": {}, "head": {}}}])
def test_invalid_event_shape_fails(tmp_path, value):
    event = tmp_path / "event.json"
    event.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        scope.pr_commits(event)


@pytest.mark.parametrize("value", ["main", "a" * 39, "a" * 41, "g" * 40,
                                 "a" * 40 + "\n", "--help", None, True])
@pytest.mark.parametrize("side", ["base", "head"])
def test_invalid_sha_fails_without_git_execution(tmp_path, value, side):
    event = tmp_path / "event.json"
    payload = {"pull_request": {"base": {"sha": "a" * 40}, "head": {"sha": "b" * 40}}}
    payload["pull_request"][side]["sha"] = value
    event.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        scope.pr_commits(event)


def test_missing_history_and_event_fail(repository):
    root, base, config, event = repository
    set_event(event, base, "f" * 40)
    with pytest.raises(subprocess.CalledProcessError):
        scope.detect_scopes(root, "pull_request", event, config)
    with pytest.raises(ValueError):
        scope.detect_scopes(root, "pull_request", None, config)


def test_cli_writes_only_boolean_scopes_and_count(repository, monkeypatch, capsys):
    root, base, config, event = repository
    write(root, "scratchv/private filename.md")
    head = commit(root, "llm scope")
    set_event(event, base, head)
    write(root, "scripts/ci_scopes.json", config.read_text(encoding="utf-8"))
    monkeypatch.setattr(scope, "ROOT", root)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    output = root.parent / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert scope.main() == 0
    assert output.read_text(encoding="utf-8") == "llm=true\nw4=false\n"
    captured = capsys.readouterr()
    assert "Changed path count: 1" in captured.out
    assert "private" not in captured.out and not captured.err


def test_cli_failure_does_not_publish_false_success(repository, monkeypatch, capsys):
    root, base, config, event = repository
    write(root, "scripts/ci_scopes.json", config.read_text(encoding="utf-8"))
    set_event(event, base, "f" * 40)
    monkeypatch.setattr(scope, "ROOT", root)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    output = root.parent / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert scope.main() == 1
    captured = capsys.readouterr()
    assert not output.exists() and not captured.out
    assert "CI scope detection failed" in captured.err


def test_git_timeout_is_bounded_and_never_publishes_success(repository, monkeypatch, capsys):
    root, base, config, event = repository
    write(root, "scripts/ci_scopes.json", config.read_text(encoding="utf-8"))
    set_event(event, base, base)
    monkeypatch.setattr(scope, "ROOT", root)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    output = root.parent / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    def timeout(command, **kwargs):
        assert kwargs["timeout"] == 60
        raise subprocess.TimeoutExpired(command, 60)

    monkeypatch.setattr(scope.subprocess, "run", timeout)
    assert scope.main() == 1
    captured = capsys.readouterr()
    assert not output.exists() and not captured.out
    assert "CI scope detection failed" in captured.err


def successful_jobs(*, requested=True, w4=False):
    results = {name: {"result": "success"} for name in summary.REQUIRED}
    results["scope"]["outputs"] = {name: str(requested).lower() for name in ("llm", "w4")}
    if not w4:
        results["scope"]["outputs"]["w4"] = "false"
    for name in ("llm", *(("w4",) if w4 else ())):
        results[name] = {"result": "success" if requested else "skipped"}
    return results


@pytest.mark.parametrize("requested", [True, False])
@pytest.mark.parametrize("w4", [True, False])
def test_complete_summary_accepts_only_requested_success_or_unrequested_skip(requested, w4):
    text, passed = summary.render_summary(successful_jobs(requested=requested, w4=w4))
    assert passed and "**PASS**" in text
    assert ("NOT RUN" in text.split("Numerical evidence")[0]) is (not requested)
    assert "explicit manual validation entry" in text
    assert len(text) < 1600


@pytest.mark.parametrize("name", [*summary.REQUIRED, "llm", "w4"])
@pytest.mark.parametrize("outcome", ["failure", "cancelled", "skipped", "pending", None, True, {}])
def test_requested_jobs_never_accept_failure_missing_or_skip(name, outcome):
    results = successful_jobs(w4=True)
    results[name]["result"] = outcome
    text, passed = summary.render_summary(results)
    assert not passed and "**FAIL**" in text


@pytest.mark.parametrize("name", [*summary.REQUIRED, "llm"])
def test_missing_required_jobs_fail(name):
    results = successful_jobs()
    del results[name]
    assert not summary.render_summary(results)[1]


@pytest.mark.parametrize("flag", ["true", "", None, True, []])
def test_w4_cannot_disappear_from_needs_unless_explicitly_unrequested(flag):
    results = successful_jobs()
    results["scope"]["outputs"]["w4"] = flag
    text, passed = summary.render_summary(results)
    assert not passed and "| w4 | MISSING |" in text


@pytest.mark.parametrize("name", ["llm", "w4"])
@pytest.mark.parametrize("flag", [None, True, False, "TRUE", "", "1", []])
def test_missing_or_invalid_scope_outputs_fail_closed(name, flag):
    results = successful_jobs(w4=True)
    results["scope"]["outputs"][name] = flag
    text, passed = summary.render_summary(results)
    assert not passed and "Invalid/missing scope output" in text


@pytest.mark.parametrize("outputs", [None, [], "true", {}, {"llm": "true"}, {"w4": "false"}])
def test_missing_or_malformed_output_object_fails_closed(outputs):
    results = successful_jobs()
    results["scope"]["outputs"] = outputs
    assert not summary.render_summary(results)[1]


@pytest.mark.parametrize("name", ["llm", "w4"])
@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled", None])
def test_unrequested_jobs_must_really_be_skipped(name, outcome):
    results = successful_jobs(requested=False, w4=True)
    results[name]["result"] = outcome
    assert not summary.render_summary(results)[1]


@pytest.mark.parametrize("payload", ["invalid JSON", "null", "[]", "{}"])
def test_summary_cli_malformed_input_fails_with_table(payload):
    env = dict(os.environ, CI_JOB_RESULTS=payload)
    result = subprocess.run([sys.executable, "-B", "-X", "utf8", str(ROOT / "scripts/summarize_ci.py")],
                            env=env, capture_output=True, text=True, encoding="utf-8", timeout=10)
    assert result.returncode == 1 and "**FAIL**" in result.stdout
    assert "| Check | Result | Scope |" in result.stdout


def test_summary_cli_returns_success_only_for_complete_result():
    env = dict(os.environ, CI_JOB_RESULTS=json.dumps(successful_jobs(requested=False)))
    result = subprocess.run([sys.executable, "-B", "-X", "utf8", str(ROOT / "scripts/summarize_ci.py")],
                            env=env, capture_output=True, text=True, encoding="utf-8", timeout=10)
    assert result.returncode == 0 and "**PASS**" in result.stdout
