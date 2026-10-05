"""Offline audit rechecks saved arrays, not just producer PASS flags."""
import json
from pathlib import Path

import numpy as np
import pytest

from probes.w3_common import sha256_file
from probes.w3_qwen3_full import run as full
from probes.w3_qwen3_full.cases import CASE_NAMES, input_cases
from scratchv.verification.fp32_reference import profile
from scripts import verify_w3_evidence as audit


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def descriptor(path):
    return {"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    # The archive/JSON/hash/finite paths are real. Only tensor width/checkpoint
    # count shrink; no model is imported or executed to construct this fixture.
    monkeypatch.setattr(full, "LOGIT_SHAPE", (1, 256, 3))
    monkeypatch.setattr(full, "CHECKPOINTS", ("embedding",))
    sources = {"scratchv/example.py": "a" * 64}
    monkeypatch.setattr(audit, "source_evidence", lambda: {"source_sha256": sources.copy()})
    top = {"gate": "numeric:ir-full-qwen3", "stage": "complete", "passed": True, "status": "PASS",
           "coverage_complete": True, "selected_numerical_passed": True, "full_ir_executed": True,
           "w3_exit_accepted": False, "optimization_level": "none", "atol": 1e-4, "rtol": 0,
           "fp32_mode": "reference", "required_cases": list(CASE_NAMES), "selected_cases": list(CASE_NAMES),
           "files": audit.MANIFEST["files"], "source_sha256": sources, "cases": []}
    schema = [{"name": "embedding", "shape": [1, 256, 1024], "dtype": "float32", "sequence_axis": 1}]
    for name, valid, feed in input_cases():
        folder = tmp_path / name
        folder.mkdir()
        np.savez(folder / "inputs.npz", **feed)
        row = {"name": name, "valid_length": valid, "passed": True,
               "input_sha256": sha256_file(folder / "inputs.npz"), "workers": []}
        for backend in ("ort", "ir"):
            worker = folder / backend
            worker.mkdir()
            for filename in ("logits.npy", "diagnostic_logits.npy"):
                np.save(worker / filename, np.zeros((1, 256, 3), np.float32))
            np.savez_compressed(worker / "checkpoints.npz", embedding=np.zeros((1, 256, 1024), np.float32))
            write_json(worker / "checkpoint_schema.json", {"version": 1, "checkpoints": schema})
            files = {key: descriptor(worker / filename) for key, filename in (
                ("logits", "logits.npy"), ("diagnostic_logits", "diagnostic_logits.npy"),
                ("checkpoints", "checkpoints.npz"), ("checkpoint_schema", "checkpoint_schema.json"))}
            report = {"gate": "numeric:w3-full-worker", "schema_version": 1, "backend": backend,
                      "optimization_level": "none", "passed": True, "status": "PASS",
                      "full_model_executed": True, "full_ir_executed": backend == "ir",
                      "w3_exit_accepted": False, "source_sha256": sources, "files": top["files"],
                      "input": {**descriptor(folder / "inputs.npz"), "valid_length": valid},
                      "process_peak_rss_bytes": 4096,
                      "stages_seconds": {f"{backend}_{kind}_execute": 0.1 for kind in ("ordinary", "diagnostic")},
                      "ordinary_diagnostic": {"passed": True, "comparison": "exact", "max_abs": 0.0},
                      "artifacts": files, "checkpoints": schema, "fp32_mode": "reference"}
            if backend == "ir":
                report["ir"] = {"instruction_count": 31, "fp32_mode": "reference", "fp32_profile": profile()}
                report["executions"] = {kind: {"executed_steps": 31, "memory_stats": {"peak_numpy_storage_bytes": 4096}}
                                        for kind in ("ordinary", "diagnostic")}
            else:
                report["ort"] = {"provider": "CPUExecutionProvider", "graph_optimization": "disabled",
                                 "intra_op_threads": 1, "inter_op_threads": 1, "cpu_mem_arena": False}
            for filename in ("report.md", "report.html"):
                (worker / filename).write_text("fixture view", encoding="utf-8")
            write_json(worker / "report.json", report)
            row["workers"].append({"backend": backend, "passed": True, "returncode": 0,
                                   "report_sha256": sha256_file(worker / "report.json")})
        row["comparison"] = full.compare_case(folder, valid)
        top["cases"].append(row)
    top["invariants"] = full.invariants(tmp_path, set(CASE_NAMES))
    write_json(tmp_path / "report.json", top)
    return tmp_path, top


def update_worker(root, top, name, backend, modify):
    path = root / name / backend / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    modify(report, path.parent)
    write_json(path, report)
    row = next(row for row in top["cases"] if row["name"] == name)
    saved = next(w for w in row["workers"] if w["backend"] == backend)
    saved["report_sha256"] = sha256_file(path)
    write_json(root / "report.json", top)


def test_real_saved_arrays_rechecked_without_model_execution(evidence, monkeypatch):
    root, _ = evidence
    monkeypatch.setattr(full, "run_worker", lambda *a, **k: pytest.fail("Offline audit executed a worker"))
    digest = sha256_file(root / "report.json")
    result = audit.verify(root, expected_report_sha256=digest)
    assert result["passed"] and result["coverage_complete"] and len(result["cases"]) == 7
    assert len(result["invariants"]) == 4 and result["trusted_report_hash_checked"]
    assert result["source_comparison"]["matches_current"]
    assert not result["model_executed"] and not result["independent_reproduction"]
    assert not result["w3_exit_accepted"] and not result["pinned_model_files_checked"]


@pytest.mark.parametrize("change", ["missing_case", "duplicate_case", "partial", "wrong_asset", "relaxed_atol"])
def test_pass_flags_do_not_replace_coverage_and_contract(evidence, change):
    root, top = evidence
    if change == "missing_case":
        top["cases"].pop()
    elif change == "duplicate_case":
        top["cases"][-1] = top["cases"][0]
    elif change == "partial":
        top["selected_cases"].pop()
    elif change == "wrong_asset":
        top["files"] = []
    else:
        top["atol"] = 0.1
    write_json(root / "report.json", top)
    with pytest.raises(ValueError):
        audit.verify(root)


def test_untrusted_or_altered_report_anchor_is_rejected(evidence):
    root, _ = evidence
    with pytest.raises(ValueError, match="Trusted report SHA-256 differs"):
        audit.verify(root, expected_report_sha256="b" * 64)


def test_sources_differ_without_claiming_current_execution(evidence, monkeypatch):
    root, _ = evidence
    monkeypatch.setattr(audit, "source_evidence", lambda: {"source_sha256": {
        "scratchv/example.py": "b" * 64, "scripts/new_audit.py": "c" * 64}})
    result = audit.verify(root)
    difference = result["source_comparison"]
    assert result["passed"] and not difference["matches_current"]
    assert difference["changed"][0]["path"] == "scratchv/example.py"
    assert difference["added_since_recording"] == ["scripts/new_audit.py"]
    assert result["model_executed"] is False


def test_changed_inputs_with_updated_hash_still_fail_case_identity(evidence):
    root, top = evidence
    path = root / CASE_NAMES[0] / "inputs.npz"
    with np.load(path, allow_pickle=False) as source:
        feed = {name: source[name] for name in source.files}
    feed["input_ids"][0, 0] += 1
    np.savez(path, **feed)
    top["cases"][0]["input_sha256"] = sha256_file(path)
    write_json(root / "report.json", top)
    with pytest.raises(ValueError, match="deterministic case"):
        audit.verify(root)


@pytest.mark.parametrize("mutation", ["hash", "dtype", "nonfinite", "numeric", "profile"])
def test_array_and_profile_failures_even_with_pass_flags(evidence, mutation):
    root, top = evidence
    def modify(report, folder):
        if mutation == "profile":
            report["ir"]["fp32_profile"]["name"] = "numpy-fp32-reference-v1"
            return
        value = np.zeros((1, 256, 3), np.float64 if mutation == "dtype" else np.float32)
        value[0, -1, -1] = np.nan if mutation == "nonfinite" else 1
        for key, name in (("logits", "logits.npy"), ("diagnostic_logits", "diagnostic_logits.npy")):
            np.save(folder / name, value)
            if mutation != "hash":
                report["artifacts"][key] = descriptor(folder / name)
    update_worker(root, top, CASE_NAMES[0], "ir", modify)
    with pytest.raises(ValueError):
        audit.verify(root)


def test_nonfinite_checkpoint_rejected_after_updating_artifact_hash(evidence):
    root, top = evidence
    def modify(report, folder):
        array = np.zeros((1, 256, 1024), np.float32)
        array[0, 0, 0] = np.inf
        np.savez_compressed(folder / "checkpoints.npz", embedding=array)
        report["artifacts"]["checkpoints"] = descriptor(folder / "checkpoints.npz")
    update_worker(root, top, CASE_NAMES[0], "ir", modify)
    with pytest.raises(ValueError, match="nonfinite"):
        audit.verify(root)


def test_reported_error_must_match_actual_arrays(evidence):
    root, top = evidence
    top["cases"][0]["comparison"]["max_logits_abs"] = 1e-6
    write_json(root / "report.json", top)
    with pytest.raises(ValueError, match="Stored comparison differs"):
        audit.verify(root)


def test_invariants_recomputed_even_when_both_backends_agree(evidence):
    root, top = evidence
    def modify(report, folder):
        for key, name in (("logits", "logits.npy"), ("diagnostic_logits", "diagnostic_logits.npy")):
            np.save(folder / name, np.ones((1, 256, 3), np.float32))
            report["artifacts"][key] = descriptor(folder / name)
    for backend in ("ort", "ir"):
        update_worker(root, top, "changed_future", backend, modify)
    top["cases"][5]["comparison"] = full.compare_case(root / "changed_future", 256)
    write_json(root / "report.json", top)
    with pytest.raises(ValueError, match="Recomputed invariants fail"):
        audit.verify(root)


@pytest.mark.parametrize("text", ['{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}', '{"x": 1e999}', '[]'])
def test_strict_json_rejects_ambiguous_or_nonfinite_records(tmp_path, text):
    path = tmp_path / "report.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        audit.read_json(path)


def test_bad_evidence_cli_writes_fail_and_preserves_source(tmp_path):
    source = tmp_path / "saved"
    source.mkdir()
    write_json(source / "report.json", {"passed": True})
    before = sha256_file(source / "report.json")
    out = tmp_path / "audit"
    assert audit.main(["--evidence-dir", str(source), "--output-dir", str(out)]) == 1
    result = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert not result["passed"] and result["status"] == "FAIL" and "error" in result
    assert sha256_file(source / "report.json") == before
    with pytest.raises(FileExistsError):
        audit.main(["--evidence-dir", str(source), "--output-dir", str(out)])


def test_external_evidence_file_rejected(tmp_path):
    inner = tmp_path / "saved"
    inner.mkdir()
    (tmp_path / "report.json").write_text("{}")
    with pytest.raises(ValueError, match="Missing/external"):
        audit.safe_file(inner, "../report.json")


@pytest.mark.parametrize("key,digest", [("../source.py", "a" * 64), ("C:/source.py", "a" * 64),
                                        ("source.py", "not-a-hash")])
def test_source_manifest_is_validated_without_opening_its_paths(key, digest):
    with pytest.raises(ValueError, match="Invalid producer source hash"):
        audit.source_difference({key: digest}, {})
