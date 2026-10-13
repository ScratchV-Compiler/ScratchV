"""Bounded QEMU raw-loader copies of an unchanged standard weight bundle.

Some QEMU releases issue a single host read per raw file. Transport files stay
well below Linux's single-read limit; the public bundle remains weights.bin.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import time

from .weight_bundle import _open_regular, _check_unchanged, _reject_links

CHUNK_BYTES = 512 * 1024 * 1024
COPY_BUFFER_BYTES = 1024 * 1024


def save_transport_report(directory, report):
    directory = _reject_links(directory)
    pending = _reject_links(directory / "transport.pending.json")
    pending.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(pending, directory / "transport.json")


def _files(directory, report):
    directory = _reject_links(directory)
    for index, row in enumerate(report["files"]):
        name = f"weights-{index:05d}.bin"
        if row["file"] != name:
            raise ValueError("Invalid weight transport filename/order")
        path = directory / name
        # Exact files created by this stage only; no glob or recursive cleanup.
        if path.parent.resolve() != directory.resolve() or path.is_symlink():
            raise ValueError("Weight transport path escaped its directory")
        yield row, path


def _owned_stat(path, row):
    """Inspect an owned pathname without following or accepting a replacement."""
    if row.get("created") is not True:
        raise ValueError("Weight transport file was not created by this stage")
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
        raise ValueError("Owned weight transport path is a link or nonregular file")
    expected = row.get("file_identity")
    actual = {"device": info.st_dev, "inode": info.st_ino}
    if expected != actual:
        raise ValueError("Owned weight transport file identity changed")
    return info


def _observe_retained(directory, report):
    """Measure owned file lengths; unknown observations are never reported as 0."""
    report.update(retained=None, retained_bytes=None, retained_files=[],
                  retained_observed_bytes=0, retained_observation="unavailable",
                  retained_observation_errors=[])
    remaining, errors, observed = [], [], 0
    owned = [(index, row) for index, row in enumerate(report["files"]) if row.get("created") is True]
    try:
        directory = _reject_links(directory)
    except (OSError, ValueError) as exc:
        errors.append(f"directory: {type(exc).__name__}: {exc}")
        directory = None
    for index, row in owned:
        name = f"weights-{index:05d}.bin"
        try:
            if row["file"] != name:
                raise ValueError("Invalid owned weight transport filename/order")
            if directory is None:
                raise ValueError("Owned transport directory is not safely observable")
            info = _owned_stat(directory / name, row)
        except FileNotFoundError:
            observed += 1
        except (OSError, ValueError) as exc:
            detail = f"{name}: {type(exc).__name__}: {exc}"
            errors.append(detail)
            remaining.append({"file": name, "nbytes": None, "error": detail})
        else:
            observed += 1
            remaining.append({"file": name, "nbytes": info.st_size})
    known = [row for row in remaining if row["nbytes"] is not None]
    known_bytes = sum(row["nbytes"] for row in known)
    report.update(retained=True if known else None if errors else False,
                  retained_files=remaining, retained_bytes=None if errors else known_bytes,
                  retained_observed_bytes=known_bytes,
                  retained_observation="partial" if errors and observed else "unavailable" if errors else "complete",
                  retained_observation_errors=errors)
    if errors:
        raise OSError("Weight transport residual observation failed: " + "; ".join(errors))


def refresh_transport_residuals(directory, report):
    """Refresh owned-file evidence without modifying or deleting any file.

    Observation failures leave unknown lengths explicit, mark the transport as
    failed and propagate to the caller, which may already have a primary error.
    """
    try:
        _observe_retained(directory, report)
    except BaseException:
        report["status"] = "FAIL"
        raise


def _finish_report(directory, report, primary):
    """Publish residual evidence without replacing an existing operation error."""
    deferred = None
    try:
        refresh_transport_residuals(directory, report)
    except BaseException as exc:
        report["status"] = "FAIL"
        if not report["retained_observation_errors"]:
            report["retained_observation_errors"] = [f"{type(exc).__name__}: {exc}"]
        if primary is None:
            primary = deferred = exc
            report["error"] = f"{type(exc).__name__}: {exc}"
        elif hasattr(primary, "add_note"):
            primary.add_note(f"Could not fully observe transport residuals: {exc}")
    try:
        save_transport_report(directory, report)
    except BaseException as exc:
        report.update(status="FAIL", report_error=f"{type(exc).__name__}: {exc}")
        if primary is None:
            primary = deferred = exc
        elif hasattr(primary, "add_note"):
            primary.add_note(f"Could not preserve transport.json: {exc}")
    if deferred is not None:
        raise deferred


def validate_transport(directory, report):
    """Re-read actual copies, checking each segment and concatenated byte hash."""
    started = time.perf_counter()
    whole, cursor = hashlib.sha256(), 0
    for row, path in _files(directory, report):
        if row["offset"] != cursor or not 0 < row["nbytes"] <= report["chunk_bytes"]:
            raise ValueError("Invalid weight transport extent")
        stream, before = _open_regular(path)
        digest, count = hashlib.sha256(), 0
        with stream:
            if before.st_size != row["nbytes"]:
                raise ValueError("Weight transport size mismatch")
            while count < row["nbytes"]:
                block = stream.read(min(COPY_BUFFER_BYTES, row["nbytes"] - count))
                if not block:
                    break
                count += len(block)
                digest.update(block)
                whole.update(block)
            if stream.read(1):
                raise ValueError("Weight transport grew during validation")
            _check_unchanged(stream, path, before)
        if count != row["nbytes"] or digest.hexdigest() != row["sha256"]:
            raise ValueError("Weight transport segment hash mismatch")
        cursor += count
    if cursor != report["total_bytes"] or whole.hexdigest() != report["sha256"]:
        raise ValueError("Weight transport total length/hash mismatch")
    return time.perf_counter() - started


def stage_transport(source, manifest, directory, report, *, chunk_bytes=None):
    """Create and verify new copies; failures retain owned files for inspection.

    ``report`` is also held by the caller, so partial failure evidence survives
    even if writing transport.json itself fails (e.g. a full output volume).
    """
    limit = CHUNK_BYTES if chunk_bytes is None else chunk_bytes
    if type(limit) is not int or not 0 < limit <= 512 * 1024 * 1024:
        raise ValueError("Weight transport chunks must be between 1 byte and 512 MiB")
    report.update(schema="scratchv.weight-transport", version=1, status="FAIL", stage="prepare",
        chunk_bytes=limit, total_bytes=manifest["total_bytes"], sha256=manifest["sha256"],
        copied_bytes=0, files=[], retained=False, retained_bytes=0, retained_files=[])
    source, directory = _reject_links(source), _reject_links(directory)
    owned, primary = False, None
    started = time.perf_counter()
    try:
        directory.mkdir(exist_ok=False)
        owned = True
        report["stage"] = "copy"
        stream, before = _open_regular(source)
        whole, cursor = hashlib.sha256(), 0
        with stream:
            if before.st_size != report["total_bytes"]:
                raise ValueError("Source weight bundle length changed before transport")
            while cursor < report["total_bytes"]:
                size = min(limit, report["total_bytes"] - cursor)
                row = {"file": f"weights-{len(report['files']):05d}.bin", "offset": cursor,
                       "nbytes": size, "written_bytes": 0, "sha256": None, "created": False}
                report["files"].append(row)
                digest = hashlib.sha256()
                output = (directory / row["file"]).open("xb")
                row["created"] = True
                with output:
                    info = os.fstat(output.fileno())
                    row["file_identity"] = {"device": info.st_dev, "inode": info.st_ino}
                    while row["written_bytes"] < size:
                        block = stream.read(min(COPY_BUFFER_BYTES, size - row["written_bytes"]))
                        if not block:
                            raise ValueError("Source weight bundle truncated during transport")
                        if output.write(block) != len(block):
                            raise OSError("Short weight transport write")
                        digest.update(block)
                        whole.update(block)
                        row["written_bytes"] += len(block)
                        report["copied_bytes"] += len(block)
                    output.flush()
                    os.fsync(output.fileno())
                row["sha256"] = digest.hexdigest()
                cursor += size
            if stream.read(1):
                raise ValueError("Source weight bundle grew during transport")
            _check_unchanged(stream, source, before)
        if whole.hexdigest() != report["sha256"]:
            raise ValueError("Source weight bundle hash changed before/during transport")
        report["copy_elapsed_s"] = time.perf_counter() - started
        report["stage"] = "validate"
        report["validation_elapsed_s"] = validate_transport(directory, report)
        report.update(status="PASS", stage="ready")
        return report
    except BaseException as exc:
        primary = exc
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["prepare_elapsed_s"] = time.perf_counter() - started
        if owned:
            _finish_report(directory, report, primary)


def remove_transport(directory, report):
    """After successful execution only, unlink exactly this run's owned copies."""
    errors, primary = [], None
    try:
        directory = _reject_links(directory)
        # Preflight every path before deleting any file. Planned but never
        # created files, replacements and links are not this invocation's data.
        paths = list(_files(directory, report))
        for row, path in paths:
            _owned_stat(path, row)
        for row, path in paths:
            try:
                _owned_stat(path, row)
                path.unlink()
            except (OSError, ValueError) as exc:
                errors.append(f"{row['file']}: {type(exc).__name__}: {exc}")
                if primary is None:
                    primary = exc
        if primary is not None:
            raise primary
        report["stage"] = "copies_removed"
    except BaseException as exc:
        primary = exc
        report.update(status="FAIL", stage="cleanup",
                      cleanup_errors=errors or [f"{type(exc).__name__}: {exc}"])
        raise
    finally:
        _finish_report(directory, report, primary)
