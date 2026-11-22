"""Private file-backed RAM carries the verified bytes through the real RV64 path."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scratchv.backend.tensor_c_codegen import TensorCCodegen
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import Value
from scratchv.runtime import file_ram_transport as ram
from scratchv.runtime.generation import GenerationCancelledError
from scratchv.runtime.riscv_external import build_external, run_external
from scratchv.runtime.riscv_tensor import discover_toolchain
from scratchv.runtime.weight_bundle import plan_guest_memory, write_weight_bundle
from scratchv.runtime.weight_transport import remove_transport
from tests.test_riscv_external import artifact


pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux RAM file descriptors")


def stage(tmp_path, data, *, large_offset=False, check=None):
    source = tmp_path / "weights.bin"
    source.write_bytes(data)
    layout = plan_guest_memory(workspace_bytes=4 * 1024**3 if large_offset else 0,
                               weight_bytes=len(data), cores=4)
    manifest = {"total_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    out, report = tmp_path / "staging", {}
    ram.stage_ram_image(source, manifest, layout, out, report, check)
    return source, out, report, layout


@pytest.mark.parametrize("data", [b"", bytes(range(251)) * 5001], ids=["empty", "multi-chunk"])
def test_sparse_extent_readback_and_cleanup_above_four_gib(tmp_path, data):
    source, out, report, layout = stage(tmp_path, data, large_offset=True)
    assert report["weight_offset"] > 4 * 1024**3
    assert report["copied_bytes"] == len(data)
    assert report["allocated_disk_bytes"] < report["ram_bytes"]
    assert report["retained_allocated_bytes"] == report["allocated_disk_bytes"]
    assert report["retained_bytes"] == layout["ram_bytes"]
    image, _ = ram.open_ram_image(out, report)
    with image:
        assert (out / "ram.bin").stat().st_mode & 0o222 == 0
        image.seek(report["weight_offset"] - 1)
        assert image.read(len(data) + 2) == b"\0" + data + b"\0"
    ram.validate_ram_image(out, report)
    remove_transport(out, report)
    assert report["retained_allocated_bytes"] == report["retained_bytes"] == 0
    assert not (out / "ram.bin").exists() and source.read_bytes() == data


def test_replaced_image_cannot_be_validated_or_removed(tmp_path):
    _, out, report, _ = stage(tmp_path, b"source weights")
    pinned, _ = ram.open_ram_image(out, report)
    with pinned:
        (out / "ram.bin").rename(out / "original")
        (out / "ram.bin").symlink_to(out / "original")
        with pytest.raises((ValueError, OSError)):
            ram.validate_ram_image(out, report)
        with pytest.raises((ValueError, OSError)):
            remove_transport(out, report)
        assert (out / "ram.bin").is_symlink() and (out / "original").exists()
    assert report["retained_observation"] == "unavailable"


def test_hash_mismatch_preserves_staged_image_and_manifest(tmp_path):
    src = tmp_path / "src"
    src.write_bytes(b"ABCD")
    layout = plan_guest_memory(weight_bytes=4)
    report = {}
    with pytest.raises(ValueError, match="differs from bundle"):
        ram.stage_ram_image(src, {"total_bytes": 4, "sha256": "0" * 64},
                            layout, tmp_path / "staging", report)
    assert report["status"] == "FAIL"
    assert report["retained"] and report["copied_bytes"] == 4
    assert json.loads((tmp_path / "staging/transport.json").read_text())["status"] == "FAIL"


def test_copy_cancellation_and_short_disk_fail_without_deleting_evidence(tmp_path, monkeypatch):
    src = tmp_path / "src"
    data = b"x" * 40
    src.write_bytes(data)
    manifest = {"total_bytes": 40, "sha256": hashlib.sha256(data).hexdigest()}
    layout = plan_guest_memory(weight_bytes=40)

    def interrupt():
        raise GenerationCancelledError("request cancelled")
    report = {}
    with pytest.raises(GenerationCancelledError):
        ram.stage_ram_image(src, manifest, layout, tmp_path / "cancel", report, interrupt)
    assert report["retained"] and report["copied_bytes"] == 0
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=32))
    report = {}
    with pytest.raises(OSError, match="Insufficient space"):
        ram.stage_ram_image(src, manifest, layout, tmp_path / "full", report)
    assert report["retained_bytes"] == 0
    assert not (tmp_path / "full/ram.bin").exists()


def test_wrong_region_and_modified_backing_are_rejected(tmp_path):
    src, out, report, layout = stage(tmp_path, b"keeporiginal")
    with pytest.raises(ValueError, match="extent"):
        ram.stage_ram_image(src, {"total_bytes": 12, "sha256": report["sha256"]},
                            {**layout, "weights_base": layout["ram_end"]}, tmp_path / "no", {})
    image = out / "ram.bin"
    image.chmod(0o600)
    with image.open("r+b") as f:
        f.seek(report["weight_offset"])
        f.write(b"changed!")
    with pytest.raises(ValueError, match="changed"):
        ram.validate_ram_image(out, report)
    assert src.read_bytes() == b"keeporiginal"


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="requires configured RV64 tools")
def test_real_same_elf_ab_and_cancelled_guest_cleanup(tmp_path):
    model, weights = artifact()
    write_weight_bundle(model.external_weights, model.external_initializers, tmp_path / "weights")
    exe = build_external(model, tmp_path / "weights", tmp_path / "build", discover_toolchain(), cores=4)
    x = np.arange(16, dtype=np.float32).reshape(4, 4)
    expected = x @ weights
    for mode in ("loader", "file-backed-ram"):
        dest = tmp_path / mode
        out, report = run_external(exe, {"x": x}, dest, timeout=30, qemu_weight_transport=mode)
        np.testing.assert_array_equal(out, expected)
        assert report["passed"] and report["qemu_reaped"] and report["qemu_weight_transport"] == mode
        assert report["weight_transport"]["retained_bytes"] == 0
        if mode == "file-backed-ram":
            assert report["weight_transport"]["boot_probe"]["restore_bytes"] == 0
            assert report["weight_transport"]["boot_probe"]["elapsed_s"] > 0
            assert "virt,memory-backend=svram" in report["command"]
            assert any("readonly=on,rom=off,share=off" in flag for flag in report["command"])
            assert not any(flag.startswith("loader,file=") and "weights-" in flag for flag in report["command"])

    def cancel_after_launch():
        children = Path(f"/proc/self/task/{os.getpid()}/children").read_text().split()
        return any((Path("/proc") / pid / "comm").read_text().startswith("qemu-system")
                   and (Path("/proc") / pid / "cwd").resolve() == dest for pid in children)
    dest = tmp_path / "cancel"
    with pytest.raises(GenerationCancelledError):
        run_external(exe, {"x": x}, dest, timeout=30, cancelled=cancel_after_launch,
                     qemu_weight_transport="file-backed-ram")
    report = json.loads((dest / "run.json").read_text())
    assert report["qemu_reaped"] and not report["passed"]
    assert report["weight_transport"]["retained"] and (dest / "weight-transport/ram.bin").exists()
    assert not Path(f"/proc/{report['qemu_pid']}").exists()


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="requires configured RV64 tools")
def test_real_weightless_guest_and_bad_mode(tmp_path):
    builder = IRBuilder()
    x = Value("x", shape=(3,))
    builder.new_function("main", [x])
    builder.new_block("entry")
    builder.ret(builder.neg(x))
    model = TensorCCodegen(builder.program, constant_storage="external").generate()
    write_weight_bundle(model.external_weights, model.external_initializers, tmp_path / "weights")
    exe = build_external(model, tmp_path / "weights", tmp_path / "build", discover_toolchain(), cores=4)
    data = np.array([1., -2., 3.], np.float32)
    out, report = run_external(exe, {"x": data}, tmp_path / "run", timeout=30,
                               qemu_weight_transport="file-backed-ram")
    np.testing.assert_array_equal(out, -data)
    assert report["weight_transport"]["total_bytes"] == report["weight_transport"]["allocated_disk_bytes"] == 0
    with pytest.raises(ValueError, match="Unknown QEMU weight"):
        run_external(exe, {"x": data}, tmp_path / "invalid", qemu_weight_transport="typo")
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("start, size, clipped", [
    (100, 30, (100, 30)), (94, 12, (100, 6)), (124, 16, (124, 6)),
    (105, 2, (105, 2)), (75, 20, (None, 0)), (130, 4, (None, 0)), (110, 0, (None, 0)),
])
def test_reset_restore_touches_only_weight_dtb_intersection(start, size, clipped):
    result = ram._boot_weight_intersection('addr=00000064 size=0x001e mem=ram name="fdt"', start, size)
    assert (result["restore_address"], result["restore_bytes"]) == clipped


def test_boot_map_rejects_unknown_rom_overlap_and_missing_device_tree():
    table = 'addr=00000064 size=0x001e mem=ram name="fdt"'
    other = 'addr=00000060 size=0x0005 mem=ram name="other"'
    with pytest.raises(ValueError, match="Unexpected"):
        ram._boot_weight_intersection(table + "\n" + other, 90, 12)
    with pytest.raises(ValueError, match="did not report"):
        ram._boot_weight_intersection(other, 0, 1)
    with pytest.raises(ValueError, match="Unrecognized"):
        ram._boot_weight_intersection(table + "\nunrecognized ROM address", 100, 1)

@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="requires configured RV64 QEMU")
def test_real_reset_overlay_repairs_only_the_overlap(tmp_path):
    # The sparse guest is 32 MiB. The upper half contains the boot DTB on this
    # virt machine; the protocol discovers its byte address from QEMU itself.
    size = 32 * 1024**2
    base = 0x80000000
    directory = tmp_path / "weight-transport"
    directory.mkdir()
    (tmp_path / "inputs.bin").write_bytes(b"x")
    image = directory / "ram.bin"
    with image.open("wb") as sparse:
        sparse.truncate(size)
    with image.open("rb") as backing:
        fd = backing.fileno()
        command = [os.environ["SCRATCHV_QEMU"], "-machine", "virt,memory-backend=svram",
                   "-object", f"memory-backend-file,id=svram,size={size},mem-path=/proc/self/fd/{fd},"
                              "readonly=on,rom=off,share=off",
                   "-display", "none", "-cpu", "rv64", "-smp", "4", "-m", "32",
                   "-bios", "none", "-monitor", "none", "-serial", "null", "-qmp", "stdio",
                   "-device", f"loader,file=inputs.bin,addr={base + 65536:#x},force-raw=on"]
        layout = {"ram_base": base, "weights_base": base + size // 2}
        broad = {"total_bytes": size // 2}
        with ram.boot_weight_overlay(command, backing, layout, directory, broad, 10):
            fdt = broad["boot_probe"]["fdt_address"]
        start, count = fdt + 8, 4
        part = {"total_bytes": count}
        with ram.boot_weight_overlay(command, backing, {**layout, "weights_base": start},
                                     directory, part, 10) as patch:
            assert (part["boot_probe"]["restore_address"], part["boot_probe"]["restore_bytes"]) == (start, count)

            def dump(extra, fds):
                commands = [{"execute": "qmp_capabilities"},
                            {"execute": "human-monitor-command", "arguments": {
                                "command-line": f"pmemsave {fdt:#x} 32 head.bin"}},
                            {"execute": "quit"}]
                data = b"".join((json.dumps(cmd) + "\n").encode() for cmd in commands)
                subprocess.run([*command, "-S", *extra], cwd=tmp_path, pass_fds=fds,
                               input=data, capture_output=True, timeout=10, check=True)
                return (tmp_path / "head.bin").read_bytes()

            reset = dump([], (fd,))
            assert reset[:4] == bytes.fromhex("d00dfeed") and reset[8:12] != bytes(4)
            handle = patch.fileno()
            restored = dump(["-device", f"loader,file=/proc/self/fd/{handle},addr={start:#x},force-raw=on"],
                            (fd, handle))
            assert restored == reset[:8] + bytes(count) + reset[12:]
            assert os.pread(fd, len(reset), fdt - base) == bytes(len(reset))
