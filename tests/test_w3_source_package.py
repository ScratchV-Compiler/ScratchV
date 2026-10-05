"""Portable source handoff must preserve bytes and reject misleading inventories."""
import json
from pathlib import Path
import zipfile

import pytest

from scripts import package_w3_repro as pack


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "source"
    for name in pack.REQUIRED | {"scratchv/example.py", "docs/llm-deploy-v1.0/W3/说明.md"}:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture: {name}\n", encoding="utf-8")
    def fake_git(root, *args):
        if args[0] == "rev-parse":
            return b"a" * 40 + b"\n"
        return b"\0".join(p.relative_to(root).as_posix().encode("utf-8")
                          for p in root.rglob("*") if p.is_file()) + b"\0"
    monkeypatch.setattr(pack, "git", fake_git)
    return root


def unpack(root, tmp_path):
    archive = tmp_path / "handoff.zip"
    result = pack.create(root, archive)
    target = tmp_path / "extracted"
    with zipfile.ZipFile(archive) as stream:
        stream.extractall(target)
    return target, result


def test_roundtrip_deterministic_and_excludes_external_assets(source, tmp_path):
    for name in ("probes/x/out/model.onnx", "scripts/.env", "output/report.json"):
        p = source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("excluded")
    extracted, result = unpack(source, tmp_path)
    verified = pack.verify(extracted, result["snapshot_id"])
    assert verified["passed"] and verified["files"] == len(pack.REQUIRED) + 2
    assert (extracted / "examples/run_ir_interpreter.py").read_bytes() == (
        source / "examples/run_ir_interpreter.py").read_bytes()
    assert not (extracted / "output").exists()
    second = pack.create(source, tmp_path / "second.zip")
    assert second["archive_sha256"] == result["archive_sha256"]
    with pytest.raises(ValueError, match="fresh paths"):
        pack.create(source, tmp_path / "second.zip")


def test_changed_source_and_wrong_identity_are_rejected(source, tmp_path):
    extracted, result = unpack(source, tmp_path)
    with pytest.raises(ValueError, match="identity"):
        pack.verify(extracted, "0" * 64)
    (extracted / "scratchv/example.py").write_text("changed")
    with pytest.raises(ValueError, match="bytes differ"):
        pack.verify(extracted, result["snapshot_id"])


def test_added_code_rejected_outputs_allowed(source, tmp_path):
    extracted, result = unpack(source, tmp_path)
    (extracted / "output").mkdir()
    (extracted / "output/result.json").write_text("{}")
    assert pack.verify(extracted, result["snapshot_id"])["passed"]
    (extracted / "scratchv/new.py").write_text("new")
    with pytest.raises(ValueError, match="Unlisted"):
        pack.verify(extracted)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/escape", "a\\b", "a//b", 3])
def test_unsafe_names(name):
    with pytest.raises(ValueError, match="Unsafe"):
        pack.safe_name(name)


def test_duplicate_manifest_path_rejected(source, tmp_path):
    extracted, _ = unpack(source, tmp_path)
    path = extracted / pack.MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files"].append(manifest["files"][0])
    manifest["snapshot_id"] = pack.snapshot_id(manifest["files"])
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        pack.verify(extracted)


def test_empty_manifest_does_not_pass(tmp_path):
    (tmp_path / pack.MANIFEST).write_text(json.dumps(
        {"schema_version": 1, "files": [], "snapshot_id": pack.snapshot_id([])}))
    with pytest.raises(ValueError, match="Empty"):
        pack.verify(tmp_path)


@pytest.mark.parametrize("value", [[], None, {"schema_version": 1, "files": {}}])
def test_malformed_manifest_is_explicit(tmp_path, value):
    (tmp_path / pack.MANIFEST).write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Unsupported"):
        pack.verify(tmp_path)


def test_file_limit_is_enforced(source, tmp_path, monkeypatch):
    monkeypatch.setattr(pack, "MAX_FILE_BYTES", 1)
    with pytest.raises(ValueError, match="limit"):
        pack.create(source, tmp_path / "too-big.zip")
    assert not (tmp_path / "too-big.zip").exists()


def test_changes_during_packaging_abort(source, tmp_path, monkeypatch):
    original = pack.read_source
    reads = {}
    def changing(root, name):
        reads[name] = reads.get(name, 0) + 1
        value = original(root, name)
        return value if reads[name] == 1 else value + b"changed"
    monkeypatch.setattr(pack, "read_source", changing)
    with pytest.raises(ValueError, match="changed during packaging"):
        pack.create(source, tmp_path / "unstable.zip")
    assert not (tmp_path / "unstable.zip").exists()
