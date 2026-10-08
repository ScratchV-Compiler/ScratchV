import hashlib
import io
import json
from pathlib import Path

import pytest

from scripts import collect_w3_ci_evidence as evidence
from scripts.collect_w3_ci_evidence import collect


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else value.encode("utf-8"))


@pytest.fixture
def complete_source(tmp_path):
    source = tmp_path / "source"
    for gate in ("medium", "subgraphs", "attention"):
        write(source / gate / "report.json", '{"passed":true}')
    write(source / "medium/trace_schema.json", b"{}")
    write(source / "medium/traces/short_17/ir_diagnostic_all.npz", b"abc")
    write(source / "medium/traces/short_17/ort_diagnostic.npz", b"def")
    return source


def manifest(destination):
    return json.loads((destination / "retained-evidence.json").read_text(encoding="utf-8"))

def test_ci_preserves_failure_arrays_and_selected_trace_not_full_weights(tmp_path):
    source=tmp_path/"source"
    (source/"subgraphs"/"broken").mkdir(parents=True)
    (source/"medium"/"traces"/"short_17").mkdir(parents=True)
    (source/"attention"/"case"/"none"/"run").mkdir(parents=True)
    (source/"subgraphs"/"report.json").write_text(json.dumps({"passed":False,"cases":[{"name":"broken","passed":False}]}))
    (source/"subgraphs"/"broken"/"ir_none.npz").write_bytes(b"failed diagnostic evidence")
    (source/"medium"/"trace_schema.json").write_text("{}")
    (source/"medium"/"traces"/"short_17"/"ort_diagnostic.npz").write_bytes(b"reference")
    (source/"attention"/"case"/"none"/"run"/"uart.bin").write_bytes(b"uart")
    (source/"model.safetensors").write_bytes(b"do not upload full weights")
    files=collect(source,tmp_path/"retained")
    assert str(Path("subgraphs/broken/ir_none.npz")) in files
    assert str(Path("attention/case/none/run/uart.bin")) in files
    assert "model.safetensors" not in files
    assert not (tmp_path/"retained"/"model.safetensors").exists()


def test_complete_inventory_hashes_bytes_without_claiming_numerical_revalidation(complete_source, tmp_path):
    destination = tmp_path / "retained"
    assert evidence.main(["--source", str(complete_source), "--destination", str(destination)]) == 0
    result = manifest(destination)
    assert result["selection_complete"] is True
    assert result["independent_full_revalidation"] is False
    assert result["omitted"] == []
    assert result["retained_bytes"] == 8
    for record in result["files"]:
        value = (destination / record["path"]).read_bytes()
        assert record["bytes"] == len(value)
        assert record["sha256"] == hashlib.sha256(value).hexdigest()


def test_large_valid_medium_report_does_not_mark_successful_collection_incomplete(complete_source, tmp_path):
    # The real seven-case medium PASS report is about 12.3 MiB. Preserve room
    # for that ordinary diagnostic metadata without removing the safety cap.
    report = {"passed": True, "diagnostics": "x" * (13 * evidence.MIB)}
    write(complete_source / "medium/report.json", json.dumps(report))
    destination = tmp_path / "retained"
    assert evidence.main(["--source", str(complete_source), "--destination", str(destination)]) == 0
    result = manifest(destination)
    assert result["selection_complete"] is True
    assert result["omitted"] == []
    assert result["retained_bytes"] == 8


def test_missing_trace_is_explicit_and_cli_fails_after_retaining_other_files(complete_source, tmp_path):
    missing = complete_source / "medium/traces/short_17/ort_diagnostic.npz"
    missing.unlink()
    destination = tmp_path / "retained"
    assert evidence.main(["--source", str(complete_source), "--destination", str(destination)]) == 1
    result = manifest(destination)
    assert result["selection_complete"] is False
    assert any(row["path"] == str(missing.relative_to(complete_source)) for row in result["omitted"])
    assert (destination / "medium/traces/short_17/ir_diagnostic_all.npz").read_bytes() == b"abc"


@pytest.mark.parametrize("contents", ["{broken", "[]", '{"cases":{}}', '{"cases":[1]}',
    '{"cases":[{"name":"../outside"}]}', '{"cases":[{"name":".."}]}',
    '{"cases":[{"name":"nested/path"}]}', '{"cases":[{}]}',
    json.dumps({"cases": [{"name": r"..\outside"}]}),
    json.dumps({"cases": [{"name": r"C:\secret"}]})])
def test_bad_case_report_retains_safe_partial_evidence(complete_source, tmp_path, contents):
    write(complete_source / "subgraphs/report.json", contents)
    write(complete_source / "subgraphs/broken/ir_none.npz", b"partial")
    write(tmp_path / "outside/secret.npy", b"secret")
    destination = tmp_path / "retained"
    collect(complete_source, destination)
    result = manifest(destination)
    assert result["selection_complete"] is False
    assert any("Cannot select cases" in row["reason"] for row in result["omitted"])
    assert (destination / "subgraphs/broken/ir_none.npz").read_bytes() == b"partial"
    assert not list(destination.rglob("secret.npy"))


@pytest.mark.parametrize("missing_report", [True, False])
def test_interrupted_or_invariant_failure_keeps_case_arrays(complete_source, tmp_path, missing_report):
    report_path = complete_source / "subgraphs/report.json"
    if missing_report:
        report_path.unlink()
    else:
        write(report_path, '{"passed":false,"cases":[{"name":"case","passed":true}]}')
    write(complete_source / "subgraphs/case/ir_none.npz", b"partial")
    destination = tmp_path / "retained"
    collect(complete_source, destination)
    assert (destination / "subgraphs/case/ir_none.npz").read_bytes() == b"partial"
    if missing_report:
        assert any("Missing gate report" in row["reason"] for row in manifest(destination)["omitted"])


def test_missing_failure_directory_is_not_reported_as_complete(complete_source, tmp_path):
    write(complete_source / "subgraphs/report.json", '{"passed":false,"cases":[{"name":"missing"}]}')
    destination = tmp_path / "retained"
    collect(complete_source, destination)
    assert any("Missing case evidence" in row["reason"] for row in manifest(destination)["omitted"])


@pytest.mark.parametrize("max_file,max_total,reason", [(8, 100, "per-file"), (100, 10, "total")])
def test_size_limits_preserve_priority_trace_and_later_small_files(complete_source, tmp_path, max_file, max_total, reason):
    write(complete_source / "attention/a.log", b"a" * 11)
    write(complete_source / "attention/b.log", b"ok")
    destination = tmp_path / "retained"
    collect(complete_source, destination, max_file_bytes=max_file, max_total_bytes=max_total)
    result = manifest(destination)
    assert result["retained_bytes"] == 10
    assert not (destination / "attention/a.log").exists()
    assert (destination / "attention/b.log").read_bytes() == b"ok"
    assert (destination / "medium/traces/short_17/ir_diagnostic_all.npz").read_bytes() == b"abc"
    assert any(reason in row["reason"] for row in result["omitted"])


def test_growing_file_cannot_exceed_budget_or_leave_partial_copy(complete_source, tmp_path, monkeypatch):
    growing = complete_source / "attention/a.log"
    write(growing, b"a")
    original = Path.open

    def open_file(path, mode="r", *args, **kwargs):
        if path == growing and mode == "rb":
            return io.BytesIO(b"a" * 20)
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)
    destination = tmp_path / "retained"
    collect(complete_source, destination, max_total_bytes=10)
    result = manifest(destination)
    assert result["retained_bytes"] == 8
    assert not (destination / "attention/a.log").exists()
    assert any("grew beyond" in row["reason"] for row in result["omitted"])


def test_oversized_report_does_not_prevent_other_diagnostics(complete_source, tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "MAX_REPORT_BYTES", 20)
    write(complete_source / "subgraphs/report.json", " " * 21)
    write(complete_source / "subgraphs/case/ir_none.npz", b"partial")
    destination = tmp_path / "retained"
    collect(complete_source, destination)
    assert (destination / "subgraphs/case/ir_none.npz").is_file()
    assert any("Report exceeds" in row["reason"] for row in manifest(destination)["omitted"])


def test_linked_external_evidence_is_not_copied(complete_source, tmp_path):
    external = tmp_path / "external.npy"
    external.write_bytes(b"private")
    link = complete_source / "attention/external.log"
    try:
        link.symlink_to(external)
    except OSError:
        pytest.skip("Creating symlinks is unavailable on this host")
    destination = tmp_path / "retained"
    collect(complete_source, destination)
    assert not (destination / "attention/external.log").exists()
    assert any("escapes source" in row["reason"] for row in manifest(destination)["omitted"])


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_invalid_budget_is_rejected_before_creating_destination(complete_source, tmp_path, limit):
    destination = tmp_path / "retained"
    with pytest.raises(ValueError, match="positive integers"):
        collect(complete_source, destination, max_total_bytes=limit)
    assert not destination.exists()


def test_destination_inside_source_is_rejected(complete_source):
    with pytest.raises(ValueError, match="outside its source"):
        collect(complete_source, complete_source / "retained")
