"""W1 full-export gate contracts using tiny artifacts, never pretrained weights."""

import copy
import json
import stat
import zipfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper, numpy_helper
import pytest

from probes.w1_qwen3_export import run as probe
import export_qwen3_onnx as exporter


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "SEQ", 3)
    monkeypatch.setattr(probe, "VOCAB", 4)
    directory = tmp_path / "model"
    directory.mkdir()
    weights = numpy_helper.from_array(np.arange(16, dtype=np.float32).reshape(4, 4), "weights")
    model = helper.make_model(helper.make_graph(
        [helper.make_node("Gather", ["weights", "input_ids"], ["logits"], axis=0)],
        "tiny-export", [helper.make_tensor_value_info("input_ids", T.INT64, [1, 3]),
                        helper.make_tensor_value_info("attention_mask", T.FLOAT, [1, 1, 3, 3])],
        [helper.make_tensor_value_info("logits", T.FLOAT, [1, 3, 4])], [weights]),
        opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.save_model(model, directory / "model.onnx", save_as_external_data=True,
                    all_tensors_to_one_file=True, location="weights.data", size_threshold=0)
    manifest = copy.deepcopy(probe.MANIFEST)
    manifest["files"] = [{"name": name, "bytes": (directory / name).stat().st_size,
                          "sha256": probe.sha256(directory / name)}
                         for name in ("model.onnx", "weights.data")]
    return directory, manifest


def refresh_manifest(directory, manifest):
    for row in manifest["files"]:
        path = directory / row["name"]
        row.update(bytes=path.stat().st_size, sha256=probe.sha256(path))


def test_verify_tiny_external_model_and_execute_real_ort(artifact):
    directory, manifest = artifact
    files = probe.verify_files(directory, manifest)
    structure = probe.inspect_model(directory, files)
    assert structure["external_tensors"] == 1
    assert structure["tensor_dtypes"] == {"FLOAT": 1}
    report = probe.execute_ort(directory, 1)
    assert report["provider"] == "CPUExecutionProvider"
    assert len(report["cases"]) == 2
    assert all(row["passed"] and row["shape"] == [1, 3, 4] for row in report["cases"])


@pytest.mark.parametrize("fault", ["missing", "pointer", "hash", "size"])
def test_untrusted_or_missing_weights_fail(artifact, fault):
    directory, manifest = artifact
    path = directory / "weights.data"
    if fault == "missing":
        path.unlink()
    elif fault == "pointer":
        path.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    elif fault == "hash":
        path.write_bytes(bytes(path.stat().st_size))
    else:
        path.write_bytes(b"short")
    with pytest.raises((ValueError, FileNotFoundError)):
        probe.verify_files(directory, manifest)


@pytest.mark.parametrize("name", ["../weights", "/tmp/model", "C:/model", "a\\b", "a/../b", "./model", "a//b"])
def test_paths_cannot_escape_artifact_root(name):
    with pytest.raises(ValueError, match="Unsafe"):
        probe.safe_member(name)


@pytest.mark.parametrize("fault", ["offset", "length", "truncated", "unverified", "duplicate", "dtype", "shape", "opset", "extra", "cast"])
def test_structure_and_external_ranges_fail_closed(artifact, fault):
    directory, manifest = artifact
    path = directory / "model.onnx"
    model = onnx.load(path, load_external_data=False)
    tensor = model.graph.initializer[0]
    if fault in ("offset", "length", "truncated", "unverified"):
        key = {"offset": "offset", "length": "length", "truncated": "offset", "unverified": "location"}[fault]
        value = {"offset": "-1", "length": "4", "truncated": "1", "unverified": "other.data"}[fault]
        next(row for row in tensor.external_data if row.key == key).value = value
    elif fault == "duplicate":
        tensor.external_data.add(key="length", value="64")
    elif fault == "dtype":
        tensor.data_type = T.FLOAT16
    elif fault == "shape":
        model.graph.input[0].type.tensor_type.shape.dim[1].dim_param = "dynamic"
    elif fault == "opset":
        model.opset_import[0].version = 17
    elif fault == "cast":
        model.graph.node.append(helper.make_node("Cast", ["weights"], ["unused_fp16"], to=T.FLOAT16))
    else:
        manifest["files"].append({"name": "unused.data", "bytes": 1, "sha256": "0" * 64})
    onnx.save_model(model, path)
    refresh_manifest(directory, {"files": manifest["files"][:2]})
    with pytest.raises(ValueError):
        probe.inspect_model(directory, manifest["files"])


def archive_for(directory, manifest, path, prefix="wrapper/model"):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("wrapper/README.txt", "model instructions")
        for row in manifest["files"]:
            archive.write(directory / row["name"], prefix + "/" + row["name"])
    manifest["release"] = {"url": "https://example.invalid/pinned.zip", "bytes": path.stat().st_size,
                           "sha256": probe.sha256(path)}


def test_local_release_archive_uses_unique_model_root_and_hashes(artifact, tmp_path):
    directory, manifest = artifact
    archive = tmp_path / "release.zip"
    archive_for(directory, manifest, archive)
    destination = tmp_path / "acquired"
    result = probe.acquire_model(destination, archive, manifest)
    assert result["reused"] is False
    assert probe.verify_files(destination, manifest) == manifest["files"]
    assert probe.acquire_model(destination, archive, manifest)["reused"] is True


@pytest.mark.parametrize("fault", ["hash", "size", "traversal", "symlink", "duplicate_model", "wrong_weight"])
def test_release_corruption_never_installs_model(artifact, tmp_path, fault):
    directory, manifest = artifact
    archive = tmp_path / "release.zip"
    archive_for(directory, manifest, archive)
    if fault in ("hash", "size"):
        manifest["release"]["sha256" if fault == "hash" else "bytes"] = "0" * 64 if fault == "hash" else 1
    else:
        with zipfile.ZipFile(archive, "a") as target:
            if fault == "traversal":
                target.writestr("../outside", b"bad")
            elif fault == "symlink":
                link = zipfile.ZipInfo("link")
                link.external_attr = (stat.S_IFLNK | 0o777) << 16
                target.writestr(link, "outside")
            elif fault == "duplicate_model":
                target.writestr("another/model.onnx", b"bad")
            else:
                target.writestr("wrapper/model/unused.data", b"bad")
                manifest["files"][1]["sha256"] = "0" * 64
        manifest["release"].update(bytes=archive.stat().st_size, sha256=probe.sha256(archive))
    destination = tmp_path / "acquired"
    with pytest.raises(ValueError):
        probe.acquire_model(destination, archive, manifest)
    assert not destination.exists()
    assert not (tmp_path / "outside").exists()


def test_http_error_is_not_a_skipped_probe(artifact, tmp_path, monkeypatch):
    _, manifest = artifact
    def fail(*args, **kwargs):
        raise OSError("network unavailable")
    monkeypatch.setattr(probe.urllib.request, "urlopen", fail)
    with pytest.raises(OSError, match="network unavailable"):
        probe.acquire_model(tmp_path / "downloaded", manifest=manifest)


def test_input_mask_is_fp32_causal_and_padding_is_masked(monkeypatch):
    monkeypatch.setattr(probe, "SEQ", 8)
    feed = probe.make_inputs(3, 123)
    assert feed["input_ids"].dtype == np.int64
    assert feed["attention_mask"].dtype == np.float32
    allowed = feed["attention_mask"][0, 0] == 0
    assert allowed.sum(axis=1).tolist() == [1, 2, 3, 3, 3, 3, 3, 3]
    assert np.all(feed["attention_mask"][0, 0, :, 3:] == np.finfo(np.float32).min)


def test_nonfinite_output_cannot_pass_ort_gate(artifact):
    directory, _ = artifact
    path = directory / "weights.data"
    weights = np.frombuffer(path.read_bytes(), dtype=np.float32).copy()
    weights[:] = np.nan
    path.write_bytes(weights.tobytes())
    with pytest.raises(ValueError, match="nonfinite"):
        probe.execute_ort(directory, 1)


@pytest.mark.parametrize("fault", [None, "missing", "ort", "environment"])
def test_cli_writes_fresh_failure_or_success_report(artifact, tmp_path, monkeypatch, fault):
    directory, manifest = artifact
    monkeypatch.setattr(probe, "MANIFEST", manifest)
    monkeypatch.setattr(probe, "environment", lambda export=False: {"unit_test": True})
    output = tmp_path / "report"
    output.mkdir()
    (output / "report.json").write_text('{"passed": true}', encoding="utf-8")
    if fault == "missing":
        (directory / "weights.data").unlink()
    elif fault in ("ort", "environment"):
        def fail(*args, **kwargs):
            raise RuntimeError("injected " + fault)
        monkeypatch.setattr(probe, "execute_ort" if fault == "ort" else "environment", fail)
    status = probe.main(["--model-dir", str(directory), "--output-dir", str(output)])
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert status == (0 if fault is None else 1)
    assert report["passed"] is (fault is None)
    fingerprints = report["source_fingerprints"]
    assert set(fingerprints) == {"probes/w1_qwen3_export/run.py", "export_qwen3_onnx.py",
                                 "probes/w1_qwen3_export/manifest.json"}
    assert all(len(row["sha256"]) == 64 for row in fingerprints.values())
    assert fingerprints["export_qwen3_onnx.py"]["sha256"] in (output / "report.md").read_text(encoding="utf-8")
    if fault:
        assert "error" in report
    else:
        assert len(report["ort"]["cases"]) == 2
        assert "do not rerun export" in report["scope"]


def test_dependency_mismatch_fails_before_model_load(monkeypatch):
    monkeypatch.setattr(probe.importlib.metadata, "version", lambda name: "0.0.0")
    with pytest.raises(RuntimeError, match="version mismatches|mismatches"):
        probe.environment()


@pytest.mark.parametrize("name", ["report.md", "report.json"])
@pytest.mark.parametrize("failure", ["write", "close", "publish"])
@pytest.mark.parametrize("numeric_passed", [False, True])
def test_report_io_failure_cannot_leave_pass(artifact, tmp_path, monkeypatch,
                                           name, failure, numeric_passed):
    directory, manifest = artifact
    monkeypatch.setattr(probe, "MANIFEST", manifest)
    monkeypatch.setattr(probe, "environment", lambda export=False: {"unit_test": True})
    if not numeric_passed:
        def fail_ort(*args, **kwargs):
            raise RuntimeError("original ORT failure")
        monkeypatch.setattr(probe, "execute_ort", fail_ort)
    output = tmp_path / "report-fault"
    output.mkdir()
    # Reusing this CLI's output directory must not retain an older success.
    (output / "report.json").write_text('{"passed": true}', encoding="utf-8")
    original_write, original_replace = Path.write_text, Path.replace

    def broken_write(path, content, *args, **kwargs):
        if path == output / f".{name}.tmp" and failure in ("write", "close"):
            if failure == "close":
                original_write(path, content, *args, **kwargs)
            raise OSError(f"injected {failure} failure")
        return original_write(path, content, *args, **kwargs)

    def broken_replace(path, target):
        if Path(target) == output / name and failure == "publish":
            raise OSError("injected publish failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "write_text", broken_write)
    monkeypatch.setattr(Path, "replace", broken_replace)
    assert probe.main(["--model-dir", str(directory), "--output-dir", str(output)]) == 1
    if name == "report.json":
        assert not (output / "report.json").exists()
        text = (output / "report.md").read_text(encoding="utf-8")
        assert "Result: FAIL" in text
        if not numeric_passed:
            assert "original ORT failure" in text
    else:
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        assert report["passed"] is False and report["report_write_errors"]
        if numeric_passed:
            assert report["failed_stage"] == "report-write"
        else:
            assert report["failed_stage"] == "ort"
            assert report["error"] == "RuntimeError: original ORT failure"
    assert not list(output.glob(".report.*.tmp"))


def test_report_json_is_published_after_required_view(artifact, tmp_path, monkeypatch):
    directory, manifest = artifact
    monkeypatch.setattr(probe, "MANIFEST", manifest)
    monkeypatch.setattr(probe, "environment", lambda export=False: {"unit_test": True})
    output = tmp_path / "report-order"
    original_replace = Path.replace
    names = []

    def record(path, target):
        names.append(Path(target).name)
        if Path(target).name == "report.json":
            assert (output / "report.md").is_file()
            assert not (output / "report.json").exists()
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", record)
    assert probe.main(["--model-dir", str(directory), "--output-dir", str(output)]) == 0
    assert names == ["report.md", "report.json"]


def test_transient_json_publish_error_retries_only_as_fail(artifact, tmp_path, monkeypatch):
    directory, manifest = artifact
    monkeypatch.setattr(probe, "MANIFEST", manifest)
    monkeypatch.setattr(probe, "environment", lambda export=False: {"unit_test": True})
    output = tmp_path / "report-retry"
    original_replace = Path.replace
    attempts = []

    def fail_once(path, target):
        if Path(target).name == "report.json":
            attempts.append(json.loads(path.read_text(encoding="utf-8"))["passed"])
            if len(attempts) == 1:
                raise OSError("transient publish failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_once)
    assert probe.main(["--model-dir", str(directory), "--output-dir", str(output)]) == 1
    assert attempts == [True, False]
    assert json.loads((output / "report.json").read_text(encoding="utf-8"))["passed"] is False
    assert "Result: FAIL" in (output / "report.md").read_text(encoding="utf-8")


def test_interruption_does_not_leave_previous_success(artifact, tmp_path, monkeypatch):
    directory, manifest = artifact
    monkeypatch.setattr(probe, "MANIFEST", manifest)
    monkeypatch.setattr(probe, "environment", lambda export=False: {"unit_test": True})
    output = tmp_path / "report-interrupted"
    output.mkdir()
    (output / "report.json").write_text('{"passed": true}', encoding="utf-8")
    (output / "report.md").write_text("Result: PASS", encoding="utf-8")

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(probe, "execute_ort", interrupted)
    with pytest.raises(KeyboardInterrupt):
        probe.main(["--model-dir", str(directory), "--output-dir", str(output)])
    assert not (output / "report.json").exists()
    assert not (output / "report.md").exists()


def test_source_fingerprints_report_missing_sources_honestly(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    result = probe.source_fingerprints()
    assert len(result) == 3
    assert all(row["sha256"] is None and "FileNotFoundError" in row["error"]
               for row in result.values())


def test_export_mode_runs_existing_script_and_checks_fresh_verification(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    (root / "export_qwen3_onnx.py").write_text("# existing exporter", encoding="utf-8")
    source, model, report = (tmp_path / name for name in ("source", "model", "report"))
    source.mkdir(); report.mkdir()
    (source / "model.safetensors").write_bytes(b"pinned checkpoint")
    manifest = copy.deepcopy(probe.MANIFEST)
    manifest["source_checkpoint_sha256"] = probe.sha256(source / "model.safetensors")
    manifest["source_files"] = []
    monkeypatch.setattr(probe, "ROOT", root)
    monkeypatch.setattr(probe, "MANIFEST", manifest)
    calls = []
    def export(command, **kwargs):
        calls.append(command)
        assert kwargs["check"] is True
        assert command[command.index("--exporter") + 1] == "dynamo"
        model.mkdir()
        (model / "verification.json").write_text(json.dumps({**manifest, "passed": True}), encoding="utf-8")
    monkeypatch.setattr(probe.subprocess, "run", export)
    args = probe.argparse.Namespace(source_dir=source, model_dir=model, output_dir=report, threads=2)
    result = probe.export_model(args)
    assert result["passed"] is True and len(calls) == 1
    assert str(root / "export_qwen3_onnx.py") in calls[0]
    with pytest.raises(ValueError, match="empty"):
        probe.export_model(args)


def test_wrong_source_checkpoint_fails_before_export(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"bad source")
    args = probe.argparse.Namespace(source_dir=source, model_dir=tmp_path / "model", output_dir=tmp_path, threads=2)
    with pytest.raises(ValueError, match="Source checkpoint"):
        probe.export_model(args)


def test_changed_source_config_fails_before_export(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"pinned checkpoint")
    (source / "config.json").write_text('{"rope_theta": 1}', encoding="utf-8")
    manifest = copy.deepcopy(probe.MANIFEST)
    manifest["source_checkpoint_sha256"] = probe.sha256(source / "model.safetensors")
    manifest["source_files"] = [{"name": "config.json", "sha256": "0" * 64}]
    monkeypatch.setattr(probe, "MANIFEST", manifest)
    args = probe.argparse.Namespace(source_dir=source, model_dir=tmp_path / "model", output_dir=tmp_path, threads=2)
    with pytest.raises(ValueError, match="configuration/tokenizer"):
        probe.export_model(args)


@pytest.fixture
def exporter_case(artifact, tmp_path, monkeypatch):
    """Real tiny ORT model and a file-backed oracle, without torch/weights."""
    directory, _ = artifact
    monkeypatch.setattr(exporter, "SEQ", 3)
    monkeypatch.setattr(exporter, "VOCAB", 4)
    monkeypatch.setattr(exporter.importlib.metadata, "version", lambda name: "unit-test")
    work = tmp_path / "reference"
    work.mkdir()
    reference = np.tile(np.arange(4, dtype=np.float32), (1, 3, 1))
    for index in range(2):
        np.savez(work / f"input_{index}.npz", input_ids=np.zeros((1, 3), np.int64),
                 attention_mask=np.zeros((1, 1, 3, 3), np.float32), valid_length=np.int64(3))
        np.save(work / f"reference_{index}.npy", reference)
    return Namespace(output_dir=directory, work_dir=work, threads=1,
                     require_full_allclose=True, exporter="dynamo")


def test_exporter_verifier_accepts_exact_finite_reference(exporter_case):
    exporter.verify(exporter_case)
    report = json.loads((exporter_case.output_dir / "verification.json").read_text(encoding="utf-8"))
    assert report["passed"] is True and report["full_tensor_allclose"] is True
    assert all(case["max_abs_error"] == 0 for case in report["cases"])


@pytest.mark.parametrize("fault", ["broadcast_shape", "dtype", "nan", "inf"])
def test_exporter_verifier_rejects_invalid_oracle(exporter_case, fault):
    path = exporter_case.work_dir / "reference_0.npy"
    reference = np.load(path)
    if fault == "broadcast_shape":
        # The old verifier broadcast this singleton dimension and accepted it
        # when the ORT output was constant across the vocabulary dimension.
        reference = reference[:, :, :1]
        (exporter_case.output_dir / "weights.data").write_bytes(np.zeros(16, np.float32).tobytes())
    elif fault == "dtype":
        reference = reference.astype(np.float64)
    else:
        reference[:] = np.nan if fault == "nan" else np.inf
    np.save(path, reference)
    with pytest.raises(ValueError, match="reference"):
        exporter.verify(exporter_case)
    assert not (exporter_case.output_dir / "verification.json").exists()


@pytest.mark.parametrize("length", [np.int64(0), np.int64(-1), np.int64(4),
                                   np.float32(2.5), np.array([2], np.int64)])
def test_exporter_verifier_cannot_skip_valid_token_gate(exporter_case, length):
    path = exporter_case.work_dir / "input_0.npz"
    with np.load(path) as data:
        feed = {name: data[name] for name in ("input_ids", "attention_mask")}
    np.savez(path, **feed, valid_length=length)
    exporter_case.require_full_allclose = False
    with pytest.raises(ValueError, match="valid_length"):
        exporter.verify(exporter_case)


def test_exporter_extreme_finite_error_is_reported_without_infinity(exporter_case):
    weights = exporter_case.output_dir / "weights.data"
    weights.write_bytes(np.full(16, np.finfo(np.float32).max, np.float32).tobytes())
    for index in range(2):
        np.save(exporter_case.work_dir / f"reference_{index}.npy",
                np.full((1, 3, 4), -np.finfo(np.float32).max, np.float32))
    with pytest.raises(RuntimeError, match="numeric comparison failed"):
        exporter.verify(exporter_case)
    text = (exporter_case.output_dir / "verification.json").read_text(encoding="utf-8")
    assert "Infinity" not in text and "NaN" not in text
    report = json.loads(text)
    assert report["passed"] is False
    assert all(np.isfinite(case["max_abs_error"]) for case in report["cases"])
