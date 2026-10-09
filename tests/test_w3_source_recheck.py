"""Standalone numerical gates must not publish PASS for mixed source revisions."""
import importlib
import json

import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from probes import w3_common as common
from probes.w3_attention import run as attention
from probes.w3_qwen3_medium import run as medium


@pytest.mark.parametrize("gate_name", ["attention", "medium", "subgraphs"])
@pytest.mark.parametrize("change", ["modified", "added", "removed", "unreadable", "unchanged"])
def test_standalone_gate_rechecks_sources_before_publishing_pass(tmp_path, monkeypatch, gate_name, change):
    if gate_name == "subgraphs":
        pytest.importorskip("torch")
        gate = importlib.import_module("probes.w3_qwen3_subgraphs.run")
    else:
        gate = attention if gate_name == "attention" else medium
    before = {"scratchv/compiler.py": "a" * 64}
    after = dict(before)
    if change == "modified":
        after["scratchv/compiler.py"] = "b" * 64
    elif change == "added":
        after["scratchv/new_op.py"] = "c" * 64
    elif change == "removed":
        after.clear()

    def finish_execution(*args, **kwargs):
        # Only numerical execution is stubbed. The real command entry, source
        # comparison and JSON/Markdown/HTML publication all remain exercised.
        report = args[2] if gate_name == "subgraphs" else args[1]
        sources = {"source_sha256": dict(before)}
        report.update({"provenance": sources} if gate is medium else sources)
        report.update(passed=True, status="PASS", stage="complete")

    def final_sources():
        if change == "unreadable":
            raise PermissionError("source became unreadable during execution")
        return {"source_sha256": after}

    monkeypatch.setattr(gate, "run_probe", finish_execution)
    if gate is attention:
        monkeypatch.setattr(gate, "source_evidence", lambda: {"source_sha256": dict(before)})
    monkeypatch.setattr(common, "source_evidence", final_sources)
    out = tmp_path / "evidence"
    arguments = ["--output-dir", str(out)]
    if gate_name == "subgraphs":
        arguments += ["--source-dir", str(tmp_path / "source")]
    assert gate.main(arguments) == (0 if change == "unchanged" else 1)
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["passed"] is (change == "unchanged")
    assert report["status"] == ("PASS" if change == "unchanged" else "FAIL")
    if change in ("modified", "added", "removed"):
        assert report["stage"] == "source_recheck"
        assert report["source_recheck"]["passed"] is False
        name = "scratchv/new_op.py" if change == "added" else "scratchv/compiler.py"
        assert report["source_recheck"]["changed"] == {
            name: {"before_sha256": before.get(name), "after_sha256": after.get(name)}}
        assert "sources changed" in report["error"]
    elif change == "unreadable":
        assert "PermissionError" in report["error"]
    else:
        assert report["source_recheck"] == {"passed": True, "changed": {}}
    for filename in ("report.md", "report.html"):
        assert report["status"] in (out / filename).read_text(encoding="utf-8")


def test_recheck_rejects_missing_fingerprints():
    with pytest.raises(ValueError, match="Missing execution source fingerprints"):
        common.recheck_sources({"passed": True})
