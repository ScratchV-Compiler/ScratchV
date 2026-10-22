"""Bounded AI review coverage and failure accounting; no external API calls."""
import importlib.util
import json
from pathlib import Path
import urllib.error

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ai_code_review", ROOT / "scripts/ai_code_review.py")
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def file(name="a.py", **overrides):
    return {"filename": name, "status": "modified", "patch": "@@ -1 +1 @@\n-old\n+new",
            "additions": 1, "deletions": 1, **overrides}


def plan(count=1):
    return review.make_plan({"repository": "owner/repo", "pr_number": 104, "head": "abc", "head_after": "abc",
                             "changed_files": count, "files": [file(f"{i}.py") for i in range(count)]})


def ok(*_args):
    return "No concrete issue found.", 200, ""


def reviewed_batch(number=0, names=("0.py",)):
    return {"head": "abc", "batch": number, "files": [
        {"filename": name, "status": "reviewed", "review": "No concrete issue found."} for name in names]}


def workflow():
    return yaml.load((ROOT / ".github/workflows/ai-code-review.yml").read_text(), Loader=yaml.BaseLoader)


def test_workflow_uses_paginated_api_and_trusted_checkout():
    jobs = workflow()["jobs"]
    assert "pull_request_target" in workflow()["on"]
    assert workflow()["on"]["workflow_dispatch"]["inputs"]["pr_number"]["required"] == "true"
    scripts = [s.get("with", {}).get("script", "") for s in jobs["prepare"]["steps"]]
    fetch = next(s for s in scripts if "pulls.listFiles" in s)
    assert "github.paginate(github.rest.pulls.listFiles" in fetch
    assert "per_page: 100" in fetch
    assert "before.changed_files" in fetch and "data.head_after = after.head.sha" in fetch
    for job in jobs.values():
        checkout = next(s for s in job["steps"] if s.get("uses") == "actions/checkout@v4")
        assert checkout["with"] == {"persist-credentials": "false"}
        assert "pull_request.head" not in json.dumps(job)
    assert jobs["review"]["strategy"]["max-parallel"] == "1"
    assert jobs["review"]["strategy"]["fail-fast"] == "false"


def test_workflow_keeps_failed_batch_artifacts_and_checks_staleness():
    jobs = workflow()["jobs"]
    for name in jobs:
        upload = next(s for s in jobs[name]["steps"] if s.get("uses") == "actions/upload-artifact@v4")
        assert upload["if"] == "always()"
        assert upload["with"]["overwrite"] == "true"
        assert upload["with"]["if-no-files-found"] == "error"
    summary = jobs["ai-code-review"]
    assert summary["if"] == "always()" and set(summary["needs"]) == {"prepare", "review"}
    post = next(s["with"]["script"] for s in summary["steps"] if s.get("uses") == "actions/github-script@v7")
    assert "github.paginate(github.rest.issues.listComments" in post
    assert "pr.head.sha !== summary.head" in post
    assert "summary.complete = false" in post
    assert summary["steps"][-1]["run"].endswith("gate --summary review-summary/summary.json")


def test_more_than_100_files_are_partitioned_without_global_truncation(tmp_path):
    p = plan(103)
    batches = [review.run_batch(p, i, "trusted prompt", tmp_path / f"batch-{i}.json", "test-key",
                               request=ok, sleep=lambda _: None) for i in range(p["batches"])]
    summary = review.summarize(p, batches)
    assert summary["complete"] and summary["counts"] == {"reviewed": 103, "failed": 0, "unreviewable": 0}
    assert [r["filename"] for r in summary["files"]] == [f"{i}.py" for i in range(103)]


@pytest.mark.parametrize("mutate", [
    lambda raw: raw.update(changed_files=3),
    lambda raw: raw.update(head_after="different"),
    lambda raw: raw.update(files=[file(), file()], changed_files=2),
    lambda raw: raw.update(files=[{"filename": None}], changed_files=1),
    lambda raw: raw.update(files="not a list"),
])
def test_incomplete_or_stale_fetch_fails_closed(mutate):
    raw = {"head": "abc", "head_after": "abc", "changed_files": 1, "files": [file()]}
    mutate(raw)
    p = review.make_plan(raw)
    assert p["errors"]
    assert not review.summarize(p, [])["complete"]


def test_github_3000_file_limit_is_not_reported_as_complete():
    raw = {"head": "abc", "head_after": "abc", "changed_files": 3001,
           "files": [file(f"{i}.py") for i in range(3000)]}
    assert review.make_plan(raw)["errors"]


@pytest.mark.parametrize("overrides", [{"patch": None}, {"patch": ""}, {"additions": 2}, {"deletions": 0}])
def test_missing_or_partial_patch_remains_explicitly_unreviewable(overrides):
    record = review.review_file(file(**overrides), "prompt", "test-key", request=lambda *_: pytest.fail("API called"))
    assert record["status"] == "unreviewable"
    assert record["attempts"] == 0


def test_filename_quotes_and_newlines_are_data():
    named = file('a "quoted"\nfile.py')
    calls = []
    def request(payload, _key):
        calls.append(payload)
        return ok()
    record = review.review_file(named, "prompt", "test-key", request=request)
    assert record["filename"] == named["filename"] and record["status"] == "reviewed"
    assert json.loads(calls[0]["messages"][1]["content"].split("\n", 1)[1])["filename"] == named["filename"]


@pytest.mark.parametrize("first", [(429, "999"), (503, ""), TimeoutError("secret must not appear")])
def test_transient_failure_retries_once_and_records_attempts(first):
    requests, sleeps = [], []
    def request(*_):
        requests.append(1)
        if len(requests) == 1:
            if isinstance(first, Exception):
                raise first
            return "", *first
        return ok()
    result = review.review_file(file(), "prompt", "test-key", request=request, sleep=sleeps.append)
    assert result["status"] == "reviewed" and result["attempts"] == 2
    assert len(sleeps) == 1 and 1 <= sleeps[0] <= 30


@pytest.mark.parametrize("status,expected_attempts", [(429, 2), (503, 2), (401, 1), (400, 1)])
def test_http_failure_is_not_silently_skipped(status, expected_attempts):
    result = review.review_file(file(), "prompt", "test-key",
                                request=lambda *_: ("", status, ""), sleep=lambda _: None)
    assert result["status"] == "failed" and result["attempts"] == expected_attempts
    assert result["reason"] == f"HTTP {status}"


def test_timeout_is_accounted_without_leaking_exception_text():
    def request(*_):
        raise TimeoutError("the-secret-key")
    result = review.review_file(file(), "prompt", "test-key", request=request, sleep=lambda _: None)
    assert result["status"] == "failed" and result["attempts"] == 2
    assert "the-secret-key" not in json.dumps(result)


def test_missing_api_key_is_a_failure_for_each_reviewable_file():
    result = review.review_file(file(), "prompt", "", request=lambda *_: pytest.fail("API called"))
    assert result["status"] == "failed" and result["attempts"] == 0


@pytest.mark.parametrize("body", [b"not json", b"{}", b"[]", b'{"choices":[{"finish_reason":"length","message":{"content":"partial review"}}]}', b'{"choices":[null]}', b'{"choices":[{"message":null}]}', b'{"choices":[]}',
                                 b'{"choices":[{"message":{"content":""}}]}',
                                 b'{"choices":[{"message":{"content":null}}]}'])
def test_malformed_or_empty_api_body_is_not_success(monkeypatch, body):
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def read(self, _): return body
    monkeypatch.setattr(review.urllib.request, "urlopen", lambda *_, **__: Response())
    result = review.review_file(file(), "prompt", "test-key", sleep=lambda _: None)
    assert result["status"] == "failed" and result["attempts"] == 2


def test_http_error_adapter_does_not_publish_api_body(monkeypatch):
    def fail(*_, **__):
        raise urllib.error.HTTPError(review.API_URL, 429, "secret", {"Retry-After": "3"}, None)
    monkeypatch.setattr(review.urllib.request, "urlopen", fail)
    assert review.request_review({}, "test-key") == ("", 429, "3")


def test_interrupted_batch_keeps_completed_and_pending_records(tmp_path):
    p = plan(2)
    calls = []
    def interrupted(*_):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return ok()
    output = tmp_path / "batch.json"
    with pytest.raises(KeyboardInterrupt):
        review.run_batch(p, 0, "prompt", output, "test-key", request=interrupted, sleep=lambda _: None)
    result = json.loads(output.read_text())
    assert [r["status"] for r in result["files"]] == ["reviewed", "pending"]
    summary = review.summarize(p, [result])
    assert not summary["complete"] and summary["counts"]["failed"] == 1


def test_missing_job_and_pending_record_fail_accounting():
    p = plan(11)
    batch = reviewed_batch(names=tuple(f"{i}.py" for i in range(9)))
    batch["files"].append({"filename": "9.py", "status": "pending"})
    summary = review.summarize(p, [batch])
    assert summary["counts"] == {"reviewed": 9, "failed": 2, "unreviewable": 0}
    assert not summary["complete"]


@pytest.mark.parametrize("batches", [
    [reviewed_batch(), reviewed_batch()],
    [dict(reviewed_batch(), head="different")],
    [dict(reviewed_batch(), batch=999)],
    [reviewed_batch(names=("not-in-plan.py",))],
])
def test_wrong_duplicate_or_foreign_batch_cannot_pass(batches):
    result = review.summarize(plan(), batches)
    assert not result["complete"] and result["errors"]


def test_current_head_must_match_plan():
    summary = review.summarize(plan(), [reviewed_batch()], current_head="new-head")
    assert not summary["complete"] and "stale" in summary["errors"][0]


def test_empty_success_text_cannot_count_as_a_review():
    batch = reviewed_batch()
    batch["files"][0]["review"] = " "
    summary = review.summarize(plan(), [batch])
    assert not summary["complete"] and summary["counts"]["failed"] == 1


def test_comment_is_bounded_and_full_report_is_not_truncated():
    p = plan(100)
    for item in p["files"]:
        item["filename"] += "文" * 1000
    summary = review.summarize(p, [])
    report, comment = review.render_report(summary, "https://example.test/run")
    assert len(comment.encode()) < 65000 and "List truncated" in comment
    assert summary["files"][-1]["filename"] in report
    assert "failed/missing: **100**" in comment


def test_cli_preserves_failure_report_before_gate(tmp_path):
    download = tmp_path / "download"
    review.write_json(download / "plan.json", plan(2))
    review.write_json(download / "batch-0.json", reviewed_batch())
    out = tmp_path / "summary"
    assert review.main(["summarize", "--directory", str(download), "--output", str(out),
                        "--run-url", "https://example.test/run"]) == 0
    assert (out / "review_output.md").exists() and (out / "review_comment.md").exists()
    assert review.main(["gate", "--summary", str(out / "summary.json")]) == 1


def test_cli_missing_plan_is_reported(tmp_path):
    out = tmp_path / "summary"
    assert review.main(["summarize", "--directory", str(tmp_path / "missing"),
                        "--output", str(out), "--run-url", "https://example.test/run"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert not summary["complete"] and summary["errors"]


def test_failed_fetch_still_writes_plan_and_matrix(tmp_path, monkeypatch):
    output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    result = review.main(["plan", "--input", str(tmp_path / "not-found.json"), "--output", str(tmp_path / "plan.json")])
    assert result == 1 and (tmp_path / "plan.json").exists()
    assert json.loads(output.read_text().split("=", 1)[1]) == {"include": [{"batch": 0}]}

def test_large_pr_reports_matrix_limit_without_generating_invalid_matrix():
    p = plan(2561)
    assert p["errors"] and p["batches"] == 1
    assert "256-job" in p["errors"][0]


@pytest.mark.parametrize("invalid", [[], None, {"head": "abc", "batch": 0, "files": [None]},
                                   {"head": "abc", "batch": 0, "files": [{"filename": {}}]}])
def test_malformed_batch_remains_an_explicit_failed_review(invalid):
    summary = review.summarize(plan(), [invalid])
    assert not summary["complete"] and summary["errors"]
    assert summary["counts"]["failed"] == 1


@pytest.mark.parametrize("text", ["bad json", "[]", '{"files": "invalid"}'])
def test_corrupt_plan_keeps_failure_summary(tmp_path, text):
    download = tmp_path / "download"
    download.mkdir()
    (download / "plan.json").write_text(text)
    out = tmp_path / "summary"
    assert review.main(["summarize", "--directory", str(download), "--output", str(out),
                        "--run-url", "https://example.test/run"]) == 0
    assert not json.loads((out / "summary.json").read_text())["complete"]
    assert (out / "review_output.md").exists()

@pytest.mark.parametrize("invalid", [[], {}])
def test_nonstring_status_keeps_failure_summary(invalid):
    batch = reviewed_batch()
    batch["files"][0]["status"] = invalid
    summary = review.summarize(plan(), [batch])
    assert not summary["complete"] and summary["counts"]["failed"] == 1


def test_one_failed_file_does_not_abandon_remaining_batch(tmp_path):
    requests = []
    def request(*_):
        requests.append(1)
        if len(requests) <= 2:
            raise ValueError("bad response")
        return ok()
    result = review.run_batch(plan(2), 0, "prompt", tmp_path / "batch.json", "test-key",
                              request=request, sleep=lambda _: None)
    assert [r["status"] for r in result["files"]] == ["failed", "reviewed"]
    summary = review.summarize(plan(2), [result])
    assert summary["counts"] == {"failed": 1, "reviewed": 1, "unreviewable": 0}
    assert not summary["complete"]
