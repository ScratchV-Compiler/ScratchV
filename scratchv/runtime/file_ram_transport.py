"""Stage the existing RV64 bundle into a private, file-backed QEMU RAM image."""
import errno
import hashlib
import json
import os
import re
import shutil
import subprocess
import time

from .weight_bundle import _check_unchanged, _fingerprint, _open_regular, _reject_links
from .weight_transport import COPY_BUFFER_BYTES, _finish_report, _owned_stat


def open_ram_image(directory, report):
    """Return an O_RDONLY handle pinned to the exact verified sparse image."""
    path = _reject_links(directory / "ram.bin")
    _owned_stat(path, report["files"][0])
    stream, before = _open_regular(path)
    if list(_fingerprint(before)) != report["image_stat"]:
        stream.close()
        raise ValueError("Staged RAM image changed")
    return stream, before


def validate_ram_image(directory, report, check_cancelled=None):
    """Check the standard bundle's existing digest over its exact guest extent."""
    started = time.perf_counter()
    image, before = open_ram_image(directory, report)
    with image:
        if before.st_size != report["ram_bytes"]:
            raise ValueError("Staged RAM image size mismatch")
        image.seek(report["weight_offset"])
        left = report["total_bytes"]
        digest = hashlib.sha256()
        while left:
            if check_cancelled is not None:
                check_cancelled()
            block = image.read(min(left, COPY_BUFFER_BYTES))
            if not block:
                raise ValueError("Truncated RAM image weight region")
            digest.update(block)
            left -= len(block)
        _check_unchanged(image, directory / "ram.bin", before)
    if digest.hexdigest() != report["sha256"]:
        raise ValueError("RAM image weight region differs from bundle")
    return time.perf_counter() - started


def stage_ram_image(source, manifest, layout, directory, report, check_cancelled=None):
    """Create an exclusive sparse image. Only weight pages consume disk blocks."""
    ram_bytes = layout["ram_bytes"]
    offset = layout["weights_base"] - layout["ram_base"]
    size = manifest["total_bytes"]
    if (ram_bytes != layout["memory_mib"] * 1024**2 or offset < 0
            or size < 0 or offset + size > ram_bytes
            or layout["regions"]["weights"]["nbytes"] != size):
        raise ValueError("Weight extent does not fit the guest RAM layout")
    source, directory = _reject_links(source), _reject_links(directory)
    report.update(schema="scratchv.qemu-file-backed-ram", version=1, mode="file-backed-ram",
                  status="FAIL", stage="prepare", files=[], ram_bytes=ram_bytes,
                  weight_offset=offset, total_bytes=size, copied_bytes=0,
                  sha256=manifest["sha256"], retained=False, retained_bytes=0)
    started = time.perf_counter()
    directory.mkdir(mode=0o700, exist_ok=False)
    primary = None
    try:
        report["available_disk_bytes"] = shutil.disk_usage(directory).free
        if size > report["available_disk_bytes"]:
            raise OSError(errno.ENOSPC, "Insufficient space for RAM image weight payload")
        report["stage"] = "copy"
        path = directory / "ram.bin"
        row = {"file": path.name, "created": False, "nbytes": ram_bytes, "written_bytes": 0}
        report["files"].append(row)
        origin, before = _open_regular(source)
        with origin:
            if before.st_size != size:
                raise ValueError("Source weight bundle size mismatch")
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as dest:
                info = os.fstat(fd)
                row.update(created=True, file_identity={"device": info.st_dev, "inode": info.st_ino})
                dest.truncate(ram_bytes)
                dest.seek(offset)
                while row["written_bytes"] < size:
                    if check_cancelled is not None:
                        check_cancelled()
                    block = origin.read(min(size - row["written_bytes"], COPY_BUFFER_BYTES))
                    if not block:
                        raise ValueError("Source weight bundle truncated")
                    if dest.write(block) != len(block):
                        raise OSError("Short RAM image weight write")
                    row["written_bytes"] += len(block)
                    report["copied_bytes"] += len(block)
                if origin.read(1):
                    raise ValueError("Source weight bundle grew")
                _check_unchanged(origin, source, before)
                dest.flush()
                os.fchmod(fd, 0o400)
                os.fsync(fd)
                info = os.fstat(fd)
                report.update(image_stat=list(_fingerprint(info)), allocated_disk_bytes=info.st_blocks*512)
        report["copy_elapsed_s"] = time.perf_counter() - started
        report["stage"] = "validate"
        report["validation_elapsed_s"] = validate_ram_image(directory, report, check_cancelled)
        report.update(stage="ready", status="PASS")
    except BaseException as exc:
        primary = exc
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["prepare_elapsed_s"] = time.perf_counter() - started
        _finish_report(directory, report, primary)


def _boot_weight_intersection(roms, weights_base, weight_bytes):
    """Read actual reset ROM placement from this QEMU; reject unknown overlaps."""
    fdt = None
    for line in roms.splitlines():
        if not line or line.startswith("/rom@"):
            continue  # Firmware files without a physical ROM address.
        entry = re.fullmatch(r'addr=([0-9a-fA-F]+) size=(0x[0-9a-fA-F]+) mem=(?:ram|rom) name="(.+)"', line)
        if entry is None:
            raise ValueError(f"Unrecognized QEMU boot ROM entry: {line}")
        start, size = int(entry[1], 16), int(entry[2], 16)
        first, last = max(start, weights_base), min(start + size, weights_base + weight_bytes)
        if entry[3] == "fdt":
            if fdt is not None or size == 0:
                raise ValueError("Ambiguous or empty QEMU boot device tree")
            fdt = {"fdt_address": start, "fdt_bytes": size,
                   "restore_address": first if first < last else None,
                   "restore_bytes": max(0, last - first)}
        elif first < last:
            raise ValueError(f"Unexpected QEMU boot ROM in weight region: {entry[3]}")
    if fdt is None:
        raise ValueError("QEMU did not report its boot device tree")
    return fdt


def boot_weight_overlay(command, image, layout, directory, report, timeout, check_cancelled=None):
    """At reset virt can overlay its DTB on an already populated RAM file.

    Discover the exact ROM range with CPUs paused. The same late raw-loader
    ordering used by the default transport restores only borrowed weight bytes.
    """
    import fcntl
    import resource

    boot = report["boot_probe"] = {"elapsed_s": 0.0, "restore_bytes": 0}
    if not report["total_bytes"]:
        report["children_maxrss_before_model_bytes"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
        return None
    probe = list(command)
    probe[probe.index("-qmp") + 1] = "stdio"
    probe[probe.index("-serial") + 1] = "null"
    # Preserve the real board, ELF, input loader and input bytes in the probe.
    idx = next(i for i, value in enumerate(probe) if value.startswith("loader,file=inputs.bin,"))
    probe[idx] = probe[idx].replace("file=inputs.bin,", "file=../inputs.bin,", 1)
    requests = ({"execute": "qmp_capabilities", "id": 1},
                {"execute": "human-monitor-command", "arguments": {"command-line": "info roms"}, "id": 2},
                {"execute": "quit", "id": 3})
    data = b"".join((json.dumps(req) + "\n").encode() for req in requests)
    started = time.monotonic()
    deadline = started + min(timeout, 30.0)
    with subprocess.Popen([*probe, "-S"], cwd=directory, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, pass_fds=(image.fileno(),)) as process:
        try:
            while True:
                if check_cancelled:
                    check_cancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("QEMU boot ROM discovery timed out")
                try:
                    out, err = process.communicate(data, timeout=min(remaining, 0.05))
                    break
                except subprocess.TimeoutExpired:
                    data = None  # communicate() retains the pending write.
            if process.returncode:
                raise ValueError("QEMU boot ROM discovery failed: " + err.decode(errors="replace")[-1000:])
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            boot["elapsed_s"] = time.monotonic() - started
            report["children_maxrss_before_model_bytes"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
    replies = {r.get("id"): r for r in (json.loads(line) for line in out.splitlines())}
    if not all("return" in replies.get(i, {}) for i in (1, 2, 3)):
        raise ValueError("QEMU did not return a complete boot ROM map")
    boot.update(_boot_weight_intersection(replies[2]["return"], layout["weights_base"], report["total_bytes"]))
    if not boot["restore_bytes"]:
        return None
    patch = os.fdopen(os.memfd_create("scratchv-reset-weights", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING), "w+b")
    try:
        offset = boot["restore_address"] - layout["ram_base"]
        count = boot["restore_bytes"]
        while count:
            if check_cancelled:
                check_cancelled()
            block = os.pread(image.fileno(), min(count, COPY_BUFFER_BYTES), offset)
            if not block or patch.write(block) != len(block):
                raise OSError("Incomplete reset-overlap weight copy")
            offset += len(block)
            count -= len(block)
        patch.flush()
        fcntl.fcntl(patch, fcntl.F_ADD_SEALS, fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW |
                    fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
        return patch
    except BaseException:
        patch.close()
        raise
