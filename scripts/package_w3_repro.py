"""Create/verify a bounded, content-addressed W3 source handoff without committing.

This preserves source bytes, not Git history, model assets or execution results.
The manifest is an integrity inventory, not a signature or numerical acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "W3_SOURCE_MANIFEST.json"
ROOT_FILES = {"pyproject.toml", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.txt",
              ".gitignore", ".gitattributes"}
TREES = {"scratchv", "scratchv_dag", "benchmarks", "examples", "probes", "scripts", "requirements",
         "tests", "LICENSES"}
EXCLUDED_PARTS = {".git", "__pycache__", "output", "out", "build", "dist", ".venv",
                  "venv", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
EXCLUDED_SUFFIXES = {".onnx", ".data", ".safetensors", ".npy", ".npz", ".pyc",
                     ".elf", ".bin", ".zip", ".pt", ".pth", ".exe", ".dll"}
MAX_FILE_BYTES = 8 * 1024**2
MAX_TOTAL_BYTES = 64 * 1024**2
REQUIRED = {"pyproject.toml", "LICENSE", "scripts/package_w3_repro.py",
            "probes/w3_qwen3_full/run.py", "requirements/qwen3-small-probe.txt",
            "examples/run_ir_interpreter.py"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(name):
    if not isinstance(name, str):
        raise ValueError("Unsafe manifest path")
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name
            or path.is_absolute() or any(x in ("", ".", "..") for x in name.split("/"))):
        raise ValueError("Unsafe manifest path")
    return path


def included(name):
    path = safe_name(name)
    if any(x in EXCLUDED_PARTS or x.startswith(".env") for x in path.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return (name in ROOT_FILES or path.parts[0] in TREES
            or name.startswith("docs/llm-deploy-v1.0/")
            or (name.startswith(".github/workflows/w3-") and path.suffix in (".yml", ".yaml")))


def git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], check=True,
                            capture_output=True, timeout=30)
    return result.stdout


def candidates(root):
    raw = git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    return sorted({name for value in raw.split(b"\0") if value
                   if included(name := value.decode("utf-8"))
                   and ((root/name).exists() or (root/name).is_symlink())})


def read_source(root, name):
    path = root / name
    # Reject links including links in parents, even when pointing inside the tree.
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError(f"Linked source is not supported: {name}")
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or external source: {name}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"Source exceeds file limit: {name}")
    with path.open("rb") as stream:
        value = stream.read(MAX_FILE_BYTES + 1)
    if len(value) > MAX_FILE_BYTES:
        raise ValueError(f"Source grew beyond file limit: {name}")
    return value


def snapshot_id(files):
    return digest(json.dumps(files, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8"))


def create(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    checksum = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or checksum.exists():
        raise ValueError("Archive and checksum must use fresh paths")
    payload, total = {}, 0
    for name in candidates(root):
        value = read_source(root, name)
        total += len(value)
        if total > MAX_TOTAL_BYTES:
            raise ValueError("Source snapshot exceeds total size limit")
        payload[name] = value
    for required in REQUIRED:
        if required not in payload:
            raise ValueError(f"Missing required source: {required}")
    files = [{"path": name, "bytes": len(value), "sha256": digest(value)}
             for name, value in payload.items()]
    manifest = {"schema_version": 1, "snapshot_id": snapshot_id(files),
                "base_commit": git(root, "rev-parse", "HEAD").decode().strip(),
                "scope": "Selected current working-tree source bytes; includes uncommitted W3 files.",
                "excludes": "Git history, model weights/binaries, environments and run evidence.",
                "is_commit": False, "numerical_acceptance": False, "files": files}
    if candidates(root) != list(payload):
        raise ValueError("Source inventory changed during packaging; retry after edits finish")
    for name, value in payload.items():
        if digest(read_source(root, name)) != digest(value):
            raise ValueError(f"Source changed during packaging: {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in [*payload.items(), (MANIFEST, json.dumps(
                manifest, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")]:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, value)
    archive_hash = digest(output.read_bytes())
    with checksum.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{archive_hash}  {output.name}\n")
    return {"snapshot_id": manifest["snapshot_id"], "files": len(files),
            "source_bytes": total, "archive": str(output), "archive_sha256": archive_hash,
            "base_commit": manifest["base_commit"], "is_commit": False}


def verify(root, expected_snapshot_id=None):
    root = Path(root).resolve()
    manifest = json.loads(read_source(root, MANIFEST))
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or not isinstance(manifest.get("files"), list)):
        raise ValueError("Unsupported source manifest")
    files = manifest["files"]
    if not files:
        raise ValueError("Empty source manifest")
    actual_id = snapshot_id(files)
    if actual_id != manifest.get("snapshot_id") or (expected_snapshot_id is not None
                                                    and actual_id != expected_snapshot_id):
        raise ValueError("Snapshot identity mismatch")
    seen, total = set(), 0
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
            raise ValueError("Invalid source file descriptor")
        name = entry["path"]
        if not isinstance(name, str) or not included(name) or name in seen:
            raise ValueError("Unexpected or duplicate source path")
        seen.add(name)
        value = read_source(root, name)
        total += len(value)
        if total > MAX_TOTAL_BYTES or type(entry["bytes"]) is not int:
            raise ValueError("Invalid source byte inventory")
        if len(value) != entry["bytes"] or digest(value) != entry["sha256"]:
            raise ValueError(f"Source bytes differ: {name}")
    if not REQUIRED.issubset(seen):
        raise ValueError("Manifest is missing required source files")
    # Output directories and external models are allowed; added source is not.
    for folder in [*(root/x for x in TREES), root/"docs/llm-deploy-v1.0", root/".github/workflows"]:
        if folder.exists():
            for directory, dirs, names in os.walk(folder, followlinks=False):
                dirs[:] = [name for name in dirs if name not in EXCLUDED_PARTS]
                if any((Path(directory)/name).is_symlink() for name in dirs):
                    raise ValueError("Linked source directory is not supported")
                for filename in names:
                    path = Path(directory) / filename
                    name = path.relative_to(root).as_posix()
                    if included(name) and name not in seen:
                        raise ValueError(f"Unlisted source file: {name}")
    for name in ROOT_FILES:
        if (root/name).exists() and name not in seen:
            raise ValueError(f"Unlisted root source: {name}")
    return {"passed": True, "snapshot_id": actual_id, "files": len(seen),
            "source_bytes": total, "claim": "Source integrity only; no numerical or team acceptance."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("create")
    make.add_argument("--source-root", type=Path, default=ROOT)
    make.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--root", type=Path, default=ROOT)
    check.add_argument("--expected-snapshot-id")
    args = parser.parse_args(argv)
    try:
        result = create(args.source_root, args.output) if args.command == "create" else verify(
            args.root, args.expected_snapshot_id)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"W3 source snapshot failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
