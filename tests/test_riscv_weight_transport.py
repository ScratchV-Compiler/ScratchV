"""Small-threshold tests exercise the same bounded loader route as full weights."""
import hashlib
import json
import os
from pathlib import Path
import stat

import numpy as np
import pytest

from scratchv.backend.tensor_c_codegen import TensorCCodegen
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import Value
from scratchv.runtime import riscv_external as runtime
from scratchv.runtime import weight_transport as transport
from scratchv.runtime.riscv_tensor import discover_toolchain
from scratchv.runtime.weight_bundle import write_weight_bundle


def source_bundle(tmp_path, data):
    source = tmp_path / "weights.bin"
    source.write_bytes(data)
    return source, {"total_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


@pytest.mark.parametrize("size", [0, 1, 31, 32, 33, 64, 65])
def test_chunks_cover_exact_contiguous_bytes_and_hashes(tmp_path, size):
    data = bytes(range(size))
    source, manifest = source_bundle(tmp_path, data)
    report, out = {}, tmp_path / "transport"
    transport.stage_transport(source, manifest, out, report, chunk_bytes=32)
    assert report["status"] == "PASS" and report["total_bytes"] == size
    assert report["copied_bytes"] == report["retained_bytes"] == size
    assert len(report["files"]) == (size + 31) // 32
    assert b"".join((out / row["file"]).read_bytes() for row in report["files"]) == data
    assert [row["offset"] for row in report["files"]] == list(range(0, size, 32))
    assert all(row["nbytes"] <= 32 for row in report["files"])
    transport.validate_transport(out, report)
    transport.remove_transport(out, report)
    assert not list(out.glob("*.bin"))
    saved = json.loads((out / "transport.json").read_text())
    assert not saved["retained"] and saved["retained_bytes"] == 0
    assert saved["sha256"] == manifest["sha256"]
    assert source.read_bytes() == data


@pytest.mark.parametrize("limit", [0, -1, True, 512*1024*1024+1])
def test_chunk_bound_cannot_exceed_safe_transport_limit(tmp_path, limit):
    source, manifest = source_bundle(tmp_path, b"data")
    with pytest.raises(ValueError, match="512 MiB"):
        transport.stage_transport(source, manifest, tmp_path / "out", {}, chunk_bytes=limit)


def test_source_hash_mismatch_keeps_copy_and_fail_manifest(tmp_path):
    source, manifest = source_bundle(tmp_path, b"abcdefgh")
    manifest["sha256"] = "0" * 64
    report, out = {}, tmp_path / "out"
    with pytest.raises(ValueError, match="Source weight bundle hash"):
        transport.stage_transport(source, manifest, out, report, chunk_bytes=4)
    saved = json.loads((out / "transport.json").read_text())
    assert saved["status"] == "FAIL" and saved["retained_bytes"] == 8
    assert len(list(out.glob("*.bin"))) == 2


@pytest.mark.parametrize("damage", ["same-size", "truncate", "extra", "reorder", "whole-hash"])
def test_transport_damage_is_rejected(tmp_path, damage):
    source, manifest = source_bundle(tmp_path, b"abcdefgh")
    report, out = {}, tmp_path / "out"
    transport.stage_transport(source, manifest, out, report, chunk_bytes=4)
    first = out / report["files"][0]["file"]
    if damage == "same-size":
        first.write_bytes(b"wxyz")
    elif damage == "truncate":
        first.write_bytes(b"abc")
    elif damage == "extra":
        first.write_bytes(b"abcde")
    elif damage == "reorder":
        report["files"].reverse()
    else:
        report["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        transport.validate_transport(out, report)


def test_partial_copy_failure_retains_bounded_progress(tmp_path, monkeypatch):
    source, manifest = source_bundle(tmp_path, b"abcdefgh")
    original_open = Path.open
    monkeypatch.setattr(transport, "COPY_BUFFER_BYTES", 4)
    class FailingWriter:
        def __init__(self, stream):
            self.stream, self.writes = stream, 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.close()
        def fileno(self):
            return self.stream.fileno()
        def write(self, value):
            self.writes += 1
            if self.writes == 2:
                raise OSError("injected disk full")
            return self.stream.write(value)
    def opened(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return FailingWriter(stream) if path.name == "weights-00000.bin" else stream
    monkeypatch.setattr(Path, "open", opened)
    report, out = {}, tmp_path / "out"
    with pytest.raises(OSError, match="disk full"):
        transport.stage_transport(source, manifest, out, report, chunk_bytes=8)
    saved = json.loads((out / "transport.json").read_text())
    assert saved["status"] == "FAIL" and saved["copied_bytes"] == 4
    assert saved["files"][0]["written_bytes"] == 4
    assert (out / "weights-00000.bin").stat().st_size == 4


def partial_writer(monkeypatch, *, raise_after_write, fail_observation=False):
    """Actually persist a prefix, then return a short count or raise the cause."""
    original_open, original_lstat = Path.open, Path.lstat
    cause = OSError("injected error after partial write")
    class PartialWriter:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.close()
        def fileno(self):
            return self.stream.fileno()
        def write(self, value):
            count = self.stream.write(value[:3])
            if raise_after_write:
                raise cause
            return count
    def opened(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path.name == "weights-00000.bin" and args and args[0] == "xb":
            return PartialWriter(stream)
        return stream
    monkeypatch.setattr(Path, "open", opened)
    if fail_observation:
        def lstat(path, *args, **kwargs):
            if path.name == "weights-00000.bin":
                raise PermissionError("injected residual stat failure")
            return original_lstat(path, *args, **kwargs)
        monkeypatch.setattr(Path, "lstat", lstat)
    return cause


@pytest.mark.parametrize("raise_after_write", [False, True], ids=["short-count", "write-raises"])
def test_actual_partial_write_residual_is_measured(tmp_path, monkeypatch, raise_after_write):
    source, manifest = source_bundle(tmp_path, b"abcdefgh")
    cause = partial_writer(monkeypatch, raise_after_write=raise_after_write)
    report, out = {}, tmp_path / "out"
    with pytest.raises(OSError) as caught:
        transport.stage_transport(source, manifest, out, report, chunk_bytes=8)
    if raise_after_write:
        assert caught.value is cause
    else:
        assert "Short weight transport write" in str(caught.value)
    assert (out / "weights-00000.bin").read_bytes() == b"abc"
    saved = json.loads((out / "transport.json").read_text())
    assert saved["status"] == "FAIL" and saved["copied_bytes"] == 0
    assert saved["retained_bytes"] == 3 and saved["retained"] is True
    assert saved["retained_observation"] == "complete"
    assert saved["retained_files"] == [{"file": "weights-00000.bin", "nbytes": 3}]
    assert source.read_bytes() == b"abcdefgh"


def test_residual_stat_failure_is_unknown_and_preserves_write_error(tmp_path, monkeypatch):
    source, manifest = source_bundle(tmp_path, b"abcdefgh")
    cause = partial_writer(monkeypatch, raise_after_write=True, fail_observation=True)
    report, out = {}, tmp_path / "out"
    with pytest.raises(OSError) as caught:
        transport.stage_transport(source, manifest, out, report, chunk_bytes=8)
    assert caught.value is cause
    assert report["status"] == "FAIL" and report["retained_bytes"] is None
    assert report["retained"] is None and report["retained_observed_bytes"] == 0
    assert report["retained_observation"] == "unavailable"
    assert any("residual stat failure" in error for error in report["retained_observation_errors"])
    assert any("residual stat failure" in note for note in cause.__notes__)
    assert json.loads((out / "transport.json").read_text())["retained_bytes"] is None
    assert (out / "weights-00000.bin").read_bytes() == b"abc"


def test_residual_observation_failure_after_copy_is_not_pass(tmp_path, monkeypatch):
    source, manifest = source_bundle(tmp_path, b"abcdefgh")
    validate, lstat = transport.validate_transport, Path.lstat
    ready = False
    def validated(directory, report):
        nonlocal ready
        elapsed = validate(directory, report)
        ready = True
        return elapsed
    def observed(path, *args, **kwargs):
        if ready and path.name == "weights-00000.bin":
            raise PermissionError("injected final observation failure")
        return lstat(path, *args, **kwargs)
    monkeypatch.setattr(transport, "validate_transport", validated)
    monkeypatch.setattr(Path, "lstat", observed)
    report, out = {}, tmp_path / "out"
    with pytest.raises(OSError, match="observation"):
        transport.stage_transport(source, manifest, out, report, chunk_bytes=8)
    saved = json.loads((out / "transport.json").read_text())
    assert saved["status"] == "FAIL" and saved["retained_bytes"] is None


def test_uncreated_foreign_file_is_neither_counted_nor_removed(tmp_path, monkeypatch):
    source, manifest = source_bundle(tmp_path, b"abcdefgh")
    opened = Path.open
    def collision(path, *args, **kwargs):
        if path.name == "weights-00000.bin" and args and args[0] == "xb":
            with opened(path, "xb") as stream:
                stream.write(b"foreign")
        return opened(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", collision)
    report, out = {}, tmp_path / "out"
    with pytest.raises(FileExistsError):
        transport.stage_transport(source, manifest, out, report, chunk_bytes=8)
    assert report["retained_bytes"] == 0 and report["retained"] is False
    with pytest.raises(ValueError, match="created|owned"):
        transport.remove_transport(out, report)
    assert (out / "weights-00000.bin").read_bytes() == b"foreign"


def test_replaced_file_is_not_removed_as_owned_copy(tmp_path):
    source, manifest = source_bundle(tmp_path, b"abcdefgh")
    report, out = {}, tmp_path / "out"
    transport.stage_transport(source, manifest, out, report, chunk_bytes=4)
    replaced = out / "weights-00001.bin"
    replaced.rename(out / "original-owned.bin")
    replaced.write_bytes(b"foreign")
    with pytest.raises(ValueError, match="identity|owned"):
        transport.remove_transport(out, report)
    assert replaced.read_bytes() == b"foreign"
    assert (out / "weights-00000.bin").read_bytes() == b"abcd"


def test_partial_residual_observation_reports_known_bytes_separately(tmp_path, monkeypatch):
    source, manifest = source_bundle(tmp_path, b"abcdefgh")
    report, out = {}, tmp_path / "out"
    transport.stage_transport(source, manifest, out, report, chunk_bytes=4)
    original = Path.lstat
    def lstat(path, *args, **kwargs):
        if path.name == "weights-00000.bin":
            raise PermissionError("one file unavailable")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(OSError, match="observation"):
        transport._observe_retained(out, report)
    assert report["retained_bytes"] is None and report["retained"] is True
    assert report["retained_observed_bytes"] == 4
    assert report["retained_observation"] == "partial"
    assert report["retained_files"][0]["nbytes"] is None
    assert report["retained_files"][1] == {"file": "weights-00001.bin", "nbytes": 4}


def test_report_write_failure_does_not_replace_partial_write_error(tmp_path, monkeypatch):
    source, manifest = source_bundle(tmp_path, b"abcdefgh")
    cause = partial_writer(monkeypatch, raise_after_write=True)
    def failed_report(*args, **kwargs):
        raise OSError("injected evidence write failure")
    monkeypatch.setattr(transport, "save_transport_report", failed_report)
    report, out = {}, tmp_path / "out"
    with pytest.raises(OSError) as caught:
        transport.stage_transport(source, manifest, out, report, chunk_bytes=8)
    assert caught.value is cause
    assert report["retained_bytes"] == 3
    assert any("evidence write failure" in note for note in cause.__notes__)
    assert "evidence write failure" in report["report_error"]


def test_link_replacement_is_not_followed_or_removed(tmp_path):
    source, manifest = source_bundle(tmp_path, b"abcdefgh")
    report, out = {}, tmp_path / "out"
    transport.stage_transport(source, manifest, out, report, chunk_bytes=8)
    path = out / "weights-00000.bin"
    path.rename(out / "original-owned.bin")
    target = tmp_path / "foreign.bin"
    target.write_bytes(b"foreign")
    try:
        path.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"Host cannot create test symlink: {exc}")
    with pytest.raises(ValueError, match="escaped|link"):
        transport.remove_transport(out, report)
    assert path.is_symlink() and target.read_bytes() == b"foreign"
    assert report["retained_bytes"] is None
    assert report["retained_observation"] == "unavailable"


def test_link_observation_refuses_cleanup_without_symlink_privilege(tmp_path, monkeypatch):
    source, manifest = source_bundle(tmp_path, b"abcdefgh")
    report, out = {}, tmp_path / "out"
    transport.stage_transport(source, manifest, out, report, chunk_bytes=8)
    lstat, unlink = Path.lstat, Path.unlink
    def observed(path, *args, **kwargs):
        info = lstat(path, *args, **kwargs)
        if path.name == "weights-00000.bin":
            fields = list(info)
            fields[0] = stat.S_IFLNK | 0o777
            return os.stat_result(fields)
        return info
    def deleted(path, *args, **kwargs):
        if path.name == "weights-00000.bin":
            pytest.fail("A link observation must not reach unlink")
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "lstat", observed)
    monkeypatch.setattr(Path, "unlink", deleted)
    with pytest.raises(ValueError, match="escaped|link"):
        transport.remove_transport(out, report)
    assert report["retained_bytes"] is None
    assert report["retained_observation"] == "unavailable"
    assert (out / "weights-00000.bin").read_bytes() == b"abcdefgh"


def test_cleanup_directory_observation_failure_marks_residual_unknown(tmp_path, monkeypatch):
    source, manifest = source_bundle(tmp_path, b"abcdefgh")
    report, out = {}, tmp_path / "out"
    transport.stage_transport(source, manifest, out, report, chunk_bytes=8)
    reject = transport._reject_links
    cause = PermissionError("injected directory observation failure")
    def inaccessible(path):
        if Path(path) == out:
            raise cause
        return reject(path)
    monkeypatch.setattr(transport, "_reject_links", inaccessible)
    with pytest.raises(PermissionError) as caught:
        transport.remove_transport(out, report)
    assert caught.value is cause
    assert report["status"] == "FAIL" and report["retained_bytes"] is None
    assert report["retained_observation"] == "unavailable"
    assert (out / "weights-00000.bin").read_bytes() == b"abcdefgh"


def test_staging_error_is_inside_runtime_failure_report(tmp_path, monkeypatch):
    from tests.test_riscv_external import _stub_executable
    executable = _stub_executable(tmp_path)
    def failure(source, manifest, directory, report):
        report.update(status="FAIL", stage="copy", copied_bytes=0)
        raise OSError("injected staging error")
    monkeypatch.setattr(runtime, "stage_transport", failure)
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must not launch"))
    out = tmp_path / "run"
    with pytest.raises(OSError, match="staging error"):
        runtime.run_external(executable, {"x": np.zeros(1, np.float32)}, out)
    report = json.loads((out / "run.json").read_text())
    assert not report["passed"] and not report["guest_completed"]
    assert report["stage"] == "weight_transport" and report["qemu_wall_s"] is None
    assert report["weight_transport"]["copied_bytes"] == 0


def executable(tmp_path, *, columns):
    builder = IRBuilder()
    x, w = Value("x", shape=(2, 4)), Value("w", shape=(4, columns))
    builder.new_function("main", [x])
    builder.new_block("entry")
    builder.program.global_values.append(w)
    builder.ret(builder.matmul(x, w))
    data = np.arange(4 * columns, dtype=np.float32).reshape(4, columns) / 16
    artifact = TensorCCodegen(builder.program, {"w": data}, constant_storage="external").generate()
    write_weight_bundle(artifact.external_weights, artifact.external_initializers, tmp_path / "weights")
    return runtime.build_external(artifact, tmp_path / "weights", tmp_path / "build", discover_toolchain()), data


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="requires explicit SCRATCHV_QEMU/CC")
@pytest.mark.parametrize("columns", [0, 2, 3, 4])
@pytest.mark.parametrize("chunk_bytes", [31, 32], ids=["chunk31", "chunk32"])
def test_real_qemu_contiguous_loader_segments_and_success_cleanup(tmp_path, monkeypatch, columns, chunk_bytes):
    exe, weights = executable(tmp_path, columns=columns)
    monkeypatch.setattr(transport, "CHUNK_BYTES", chunk_bytes)
    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    out = tmp_path / "run"
    output, report = runtime.run_external(exe, {"x": x}, out, timeout=30)
    np.testing.assert_array_equal(output, x @ weights)
    assert report["passed"]
    evidence = report["weight_transport"]
    assert len(evidence["files"]) == (weights.nbytes + chunk_bytes - 1) // chunk_bytes
    assert not evidence["retained"] and evidence["retained_bytes"] == 0
    assert evidence["copy_elapsed_s"] >= 0 and evidence["post_validation_elapsed_s"] >= 0
    commands = [part for part in report["command"] if "weight-transport" in part]
    assert len(commands) == len(evidence["files"])
    for row, command in zip(evidence["files"], commands):
        assert f"addr=0x{exe.layout['weights_base'] + row['offset']:x}" in command
    assert not list((out / "weight-transport").glob("*.bin"))
    assert (exe.bundle_dir / "weights.bin").exists()
    assert json.loads((out / "weight-transport/transport.json").read_text())["stage"] == "copies_removed"


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="requires explicit SCRATCHV_QEMU/CC")
def test_post_execution_transport_corruption_fails_and_retains_copies(tmp_path, monkeypatch):
    exe, _ = executable(tmp_path, columns=3)
    monkeypatch.setattr(transport, "CHUNK_BYTES", 32)
    verify = runtime.validate_transport
    def corrupt(directory, report):
        path = directory / report["files"][0]["file"]
        value = bytearray(path.read_bytes())
        value[0] ^= 1
        path.write_bytes(value)
        return verify(directory, report)
    monkeypatch.setattr(runtime, "validate_transport", corrupt)
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="segment hash"):
        runtime.run_external(exe, {"x": np.ones((2, 4), np.float32)}, out, timeout=30)
    report = json.loads((out / "run.json").read_text())
    assert not report["passed"] and report["guest_completed"]
    assert report["weight_transport"]["status"] == "FAIL"
    assert len(list((out / "weight-transport").glob("*.bin"))) == 2


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="requires explicit SCRATCHV_QEMU/CC")
def test_success_copy_cleanup_failure_is_not_a_pass(tmp_path, monkeypatch):
    exe, _ = executable(tmp_path, columns=3)
    monkeypatch.setattr(transport, "CHUNK_BYTES", 32)
    unlink = Path.unlink
    def failure(path, *args, **kwargs):
        if path.name == "weights-00000.bin":
            raise OSError("injected locked transport copy")
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", failure)
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="cleanup failed"):
        runtime.run_external(exe, {"x": np.ones((2, 4), np.float32)}, out, timeout=30)
    report = json.loads((out / "run.json").read_text())
    assert not report["passed"] and report["guest_completed"]
    assert any("locked transport" in error for error in report["cleanup_errors"])
    assert report["weight_transport"]["retained_bytes"] == 32
    assert len(list((out / "weight-transport").glob("*.bin"))) == 1
