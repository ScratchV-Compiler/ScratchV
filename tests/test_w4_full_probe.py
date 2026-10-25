"""W4 full-output gate and evidence publication failure regressions."""
import json
import hashlib
import zipfile

import numpy as np
import pytest

from probes.w4_qwen3_full import run


SHAPE = (1, 256, 151936)


def report():
    return {"passed": True, "status": "PASS", "stage": "complete", "gates": {"numeric": True}}


def test_all_output_positions_are_compared():
    expected = np.broadcast_to(np.float32(0), SHAPE)
    actual = np.zeros(SHAPE, dtype=np.float32)
    actual[0, -1, -1] = np.float32(0.001)
    result = run.compare_logits(actual, expected)
    assert not result["passed"]
    assert result["worst_index"] == [0, 255, 151935]
    assert result["elements_at_or_above_atol"] == 1
    assert result["elements_compared"] == 38895616
    actual[0, -1, -1] = np.float32(0.0009)
    assert run.compare_logits(actual, expected)["passed"]


def test_nonfinite_and_wrong_shape_never_pass():
    zero = np.broadcast_to(np.float32(0), SHAPE)
    nonfinite = np.broadcast_to(np.float32(np.nan), SHAPE)
    with pytest.raises(ValueError, match="Nonfinite"):
        run.compare_logits(nonfinite, zero)
    with pytest.raises(ValueError, match="requires"):
        run.compare_logits(zero[:, :1], zero[:, :1])
    with pytest.raises(ValueError, match="requires"):
        run.compare_logits(np.broadcast_to(np.float64(0), SHAPE), zero)


def test_required_reports_before_pass(tmp_path, monkeypatch):
    real_write = run.atomic_text
    def failing(path, text):
        if path.suffix == ".html":
            raise OSError("simulated disk error")
        return real_write(path, text)
    monkeypatch.setattr(run, "atomic_text", failing)
    with pytest.raises(OSError, match="disk error"):
        run.write_reports(tmp_path, report())
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["status"] == "FAIL" and not saved["passed"]
    assert not (tmp_path / "report.md").exists()


def test_report_views_and_explicit_team_boundary(tmp_path):
    data = report()
    run.write_reports(tmp_path, data)
    assert json.loads((tmp_path / "report.json").read_text()) == data
    assert "pending" in (tmp_path / "report.md").read_text()
    assert (tmp_path / "report.html").is_file()


def test_missing_pinned_model_records_fail(tmp_path):
    out = tmp_path / "evidence"
    assert run.main(["--model-dir", str(tmp_path / "absent"), "--output-dir", str(out)]) == 1
    saved = json.loads((out / "report.json").read_text())
    assert not saved["passed"] and not any(saved["gates"].values())
    assert saved["stage"] == "asset_hashes"


def test_source_snapshot_keeps_verified_contents(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_bytes(b"verified source\n")
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(run, "ROOT", source)
    digest = hashlib.sha256(b"verified source\n").hexdigest()
    metadata = run.save_source_snapshot(out, {"module.py": digest})
    with zipfile.ZipFile(out / metadata["file"]) as archive:
        assert archive.read("module.py") == b"verified source\n"
    bad = tmp_path / "bad"
    bad.mkdir()
    (source / "module.py").write_bytes(b"changed source\n")
    with pytest.raises(ValueError, match="Source changed"):
        run.save_source_snapshot(bad, {"module.py": digest})
