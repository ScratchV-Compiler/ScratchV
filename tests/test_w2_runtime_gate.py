"""Runtime acceptance must reject missing/corrupt assets and reference drift."""
import copy
import hashlib
import json

import pytest

from probes.w2_runtime import run as gate


@pytest.fixture
def assets(tmp_path):
    data = {item["name"]: ("fixture:" + item["name"]).encode() for item in gate.MANIFEST["source_files"]}
    manifest = copy.deepcopy(gate.MANIFEST)
    for item in manifest["source_files"]:
        item["sha256"] = hashlib.sha256(data[item["name"]]).hexdigest()
    source = tmp_path / "source"
    source.mkdir()
    for name, payload in data.items():
        (source / name).write_bytes(payload)
    return source, data, manifest


@pytest.mark.parametrize("problem", ["missing", "corrupt", "lfs", "duplicate", "traversal", "missing_manifest_entry"])
def test_asset_verification_rejects_incomplete_or_altered_source(assets, problem):
    source, _, manifest = assets
    file = source / "tokenizer.json"
    if problem == "missing":
        file.unlink()
    elif problem == "corrupt":
        file.write_bytes(b"corrupted")
    elif problem == "lfs":
        file.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    elif problem == "duplicate":
        manifest["source_files"].append(manifest["source_files"][0])
    elif problem == "traversal":
        manifest["source_files"][0]["name"] = "../config.json"
    else:
        manifest["source_files"].pop()
    with pytest.raises((ValueError, FileNotFoundError)):
        gate.verify_assets(source, manifest)


def test_local_assets_match_manifest(assets):
    source, _, manifest = assets
    assert len(gate.verify_assets(source, manifest)) == 7


def test_corrupt_existing_assets_are_preserved(assets):
    source, _, manifest = assets
    (source / "tokenizer.json").write_bytes(b"keep this evidence")
    with pytest.raises(ValueError, match="SHA256"):
        gate.verify_assets(source, manifest)
    assert (source / "tokenizer.json").read_bytes() == b"keep this evidence"


@pytest.mark.parametrize("mode", ["download", "verify"])
def test_removed_mode_is_rejected_before_creating_output(tmp_path, mode):
    out = tmp_path / "report"
    with pytest.raises(SystemExit, match="2"):
        gate.main(["--mode", mode, "--output-dir", str(out)])
    assert not out.exists()


def test_missing_assets_produces_failed_report_with_source_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "require_environment", lambda: {"fixture": True})
    out = tmp_path / "report"
    assert gate.main(["--tokenizer-dir", str(tmp_path / "absent"), "--output-dir", str(out)]) == 1
    r = json.loads((out / "report.json").read_text())
    assert not r["passed"] and r["stage"] == "assets"
    assert "Missing tokenizer asset" in r["error"]
    assert not (tmp_path / "absent").exists()
    assert r["source_sha256"]["scratchv/runtime/llm_inputs.py"]
    assert "FAIL" in (out / "report.md").read_text()


def test_validation_failure_preserves_completed_cases_and_escapes_html(tmp_path, monkeypatch):
    def fail(args, report):
        report["stage"] = "tokenizer"
        report["tokenizer_cases"].append({"name": "first", "passed": True})
        raise ValueError("bad <script>oracle</script>")
    monkeypatch.setattr(gate, "run_checks", fail)
    out = tmp_path / "report"
    assert gate.main(["--tokenizer-dir", str(tmp_path / "fixture"), "--output-dir", str(out)]) == 1
    r = json.loads((out / "report.json").read_text())
    assert r["stage"] == "tokenizer" and len(r["tokenizer_cases"]) == 1 and not r["passed"]
    assert "<script>" not in (out / "report.html").read_text()


def test_prior_output_is_not_overwritten(tmp_path):
    out = tmp_path / "report"
    out.mkdir()
    (out / "report.json").write_text("prior report")
    with pytest.raises(SystemExit):
        gate.main(["--tokenizer-dir", str(tmp_path), "--output-dir", str(out)])
    assert (out / "report.json").read_text() == "prior report"


class TokenizerDouble:
    def encode(self, text, **kwargs):
        return [1, 2]
    def decode(self, ids, **kwargs):
        return "é"


@pytest.mark.parametrize("problem", ["fast", "slow", "golden_ids", "golden_text", "empty", "duplicate"])
def test_tokenizer_comparison_rejects_either_reference_or_golden_drift(problem):
    adapter, fast, slow = TokenizerDouble(), TokenizerDouble(), TokenizerDouble()
    case = {"name": "nfc", "text": "e\u0301", "ids": [1, 2], "decoded": "é"}
    corpus = {"cases": [case]}
    if problem in ("fast", "slow"):
        (fast if problem == "fast" else slow).encode = lambda *a, **k: [3]
    elif problem == "golden_ids":
        case["ids"] = [9]
    elif problem == "golden_text":
        case["decoded"] = "different"
    elif problem == "empty":
        corpus["cases"] = []
    else:
        corpus["cases"].append(case)
    with pytest.raises(ValueError):
        gate.tokenizer_cases(adapter, fast, slow, corpus, [])


def test_official_normalization_is_not_misclassified_as_decode_failure():
    tokenizer = TokenizerDouble()
    rows = []
    gate.tokenizer_cases(tokenizer, tokenizer, tokenizer,
                        {"cases": [{"name": "nfc", "text": "e\u0301", "ids": [1, 2], "decoded": "é"}]}, rows)
    assert rows[0]["passed"] and not rows[0]["raw_text_preserved"]


def test_source_archive_does_not_inherit_parent_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ROOT", tmp_path)

    def unexpected(*args, **kwargs):
        pytest.fail("An archive without .git must not discover an ancestor repository")

    monkeypatch.setattr(gate.subprocess, "check_output", unexpected)
    result = gate.checkout_evidence()
    assert result["head"] is None and result["clean"] is None
    assert "no .git entry" in result["unavailable"]


@pytest.mark.parametrize("failure", [FileNotFoundError("git missing"), gate.subprocess.TimeoutExpired("git", 10)])
def test_checkout_metadata_unavailable_is_explicit(tmp_path, monkeypatch, failure):
    (tmp_path / ".git").write_text("gitdir: fixture", encoding="utf-8")
    monkeypatch.setattr(gate, "ROOT", tmp_path)

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(gate.subprocess, "check_output", fail)
    result = gate.checkout_evidence()
    assert result["head"] is None and result["status"] is None and result["clean"] is None
    assert type(failure).__name__ in result["unavailable"]
