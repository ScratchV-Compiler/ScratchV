"""Retain bounded diagnostic evidence, with an explicit inventory of omissions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat


MIB = 1024**2
MAX_FILE_BYTES = 256 * MIB
MAX_TOTAL_BYTES = 512 * MIB
MAX_REPORT_BYTES = 32 * MIB
FAILURE_SUFFIXES = {".npz", ".npy", ".onnx", ".log", ".stdout", ".stderr"}


def collect(source, destination, *, max_file_bytes=MAX_FILE_BYTES, max_total_bytes=MAX_TOTAL_BYTES):
    """Copy selected evidence; omissions remain visible even after raw artifacts expire.

    This inventories copied bytes, but does not rerun any numerical gate. The CLI
    exits unsuccessfully when selection is incomplete; other safe files survive.
    """
    for limit in (max_file_bytes, max_total_bytes):
        if type(limit) is not int or limit <= 0:
            raise ValueError("Evidence byte limits must be positive integers")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.is_relative_to(source):
        raise ValueError("Evidence destination must be outside its source directory")
    destination.mkdir(parents=True, exist_ok=False)
    selected, omitted = {}, []

    def omit(path, reason, **details):
        omitted.append({"path": str(path.relative_to(source)), "reason": reason, **details})

    def safe(path):
        if not path.resolve().is_relative_to(source):
            raise ValueError("Evidence path escapes source directory")

    def scan(folder, suffixes, names=(), *, required=False):
        try:
            safe(folder)
            if not folder.exists():
                if required:
                    omit(folder, "Missing case evidence directory")
                return
            # Never follow linked directories while searching for failure evidence.
            def walk_error(error):
                omit(folder, f"Cannot scan evidence: {error}")

            for root, dirs, files in os.walk(folder, followlinks=False, onerror=walk_error):
                root = Path(root)
                for name in list(dirs):
                    child = root / name
                    if child.is_symlink() or not child.resolve().is_relative_to(source):
                        dirs.remove(name)
                        omit(child, "Linked or external evidence directory")
                dirs.sort()
                for name in sorted(files):
                    path = root / name
                    if path.suffix in suffixes or name in names:
                        selected[path] = None
        except (OSError, ValueError) as exc:
            omit(folder, str(exc))

    # First priority: the exact trace selected by the independent layer-diff gate.
    for name in ("trace_schema.json", "traces/short_17/ir_diagnostic_all.npz",
                 "traces/short_17/ort_diagnostic.npz"):
        selected[source / "medium" / name] = None
    # Original target logs permit inspection of UART/protocol failures.
    scan(source / "attention", {".log", ".stdout", ".stderr"}, {"uart.bin"})
    for gate in ("medium", "subgraphs", "attention"):
        folder = source / gate
        report_path = folder / "report.json"
        if not report_path.exists():
            # Interrupted runs may have raw evidence but no final report.
            omit(report_path, "Missing gate report; collected available diagnostics")
            scan(folder, FAILURE_SUFFIXES, {"uart.bin"})
            continue
        try:
            safe(report_path)
            info = report_path.stat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("Report is not a regular file")
            if info.st_size > MAX_REPORT_BYTES:
                raise ValueError("Report exceeds size limit")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                raise ValueError("Report must be an object")
            if report.get("passed") is True:
                continue
            rows = report.get("cases", [])
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise ValueError("Report cases must be a list of objects")
            failed = [row for row in rows if row.get("passed") is not True]
            # An invariant can fail after all individual comparisons have passed.
            for row in failed or rows:
                name = row.get("name")
                if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
                    raise ValueError("Invalid case evidence directory name")
                case_folder = folder / ("traces" if gate == "medium" else "") / name
                if not case_folder.resolve().is_relative_to(folder.resolve()):
                    raise ValueError("Case evidence escapes its gate directory")
                scan(case_folder, FAILURE_SUFFIXES, {"uart.bin"}, required=True)
            if not rows:
                scan(folder, FAILURE_SUFFIXES, {"uart.bin"})
        except (OSError, ValueError) as exc:
            omit(report_path, f"Cannot select cases from report: {exc}")
            scan(folder, FAILURE_SUFFIXES, {"uart.bin"})

    records, total = [], 0
    for path in selected:
        relative = path.relative_to(source)
        target = destination / relative
        try:
            safe(path)
            info = path.stat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("Evidence is not a regular file")
            if info.st_size > max_file_bytes:
                raise ValueError("Evidence exceeds per-file byte limit")
            if info.st_size > max_total_bytes - total:
                raise ValueError("Evidence exceeds remaining total byte limit")
            target.parent.mkdir(parents=True, exist_ok=True)
            digest, copied = hashlib.sha256(), 0
            with path.open("rb") as src, target.open("xb") as dst:
                while chunk := src.read(MIB):
                    copied += len(chunk)
                    if copied > min(max_file_bytes, max_total_bytes - total):
                        raise ValueError("Evidence grew beyond byte limit during collection")
                    dst.write(chunk)
                    digest.update(chunk)
            total += copied
            records.append({"path": str(relative), "bytes": copied, "sha256": digest.hexdigest()})
        except (OSError, ValueError) as exc:
            target.unlink(missing_ok=True)
            omit(path, str(exc))

    files = [record["path"] for record in records]
    manifest = {
        "schema_version": 1,
        "scope": "Selected preparation trace and failure diagnostics only; not complete W3 raw evidence.",
        "independent_full_revalidation": False,
        "validation": "Copy inventory and SHA256 only; this collector does not perform numerical validation.",
        "selection_complete": not omitted,
        "limits": {"max_file_bytes": max_file_bytes, "max_total_bytes": max_total_bytes},
        "retained_bytes": total,
        "files": records,
        "omitted": omitted,
    }
    (destination / "retained-files.json").write_text(json.dumps(files, indent=2) + "\n", encoding="utf-8")
    (destination / "retained-evidence.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    collect(args.source, args.destination)
    manifest = json.loads((args.destination / "retained-evidence.json").read_text(encoding="utf-8"))
    if not manifest["selection_complete"]:
        print(f"Evidence selection incomplete: {len(manifest['omitted'])} omissions; see retained-evidence.json")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
