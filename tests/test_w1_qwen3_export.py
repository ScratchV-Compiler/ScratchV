"""W1 full-export gate contracts using tiny artifacts, never pretrained weights."""

import copy
import json
import stat
import zipfile

import numpy as np
import onnx
from onnx import TensorProto as T, helper, numpy_helper
import pytest

from probes.w1_qwen3_export import run as probe


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
