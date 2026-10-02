#!/usr/bin/env python3
"""Record W1 reproduction prerequisites without downloading or running models.

An exit code of zero means only that the environment is ready. It is neither a
numeric gate nor evidence of a second person's independent reproduction.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL = "probes/w1_tiny_transformer/out/tiny_transformer_2l.onnx"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], root: Path) -> str:
    result = subprocess.run(command, cwd=root, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=30)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip()[:4000]
                           or f"Command exited {result.returncode}: {command[0]}")
    return result.stdout.rstrip("\r\n")


def check_packages(root: Path) -> dict:
    """Read the same exact pins as the probe; do not maintain a second version list."""
    packages = {}
    for line in (root / "requirements/qwen3-small-probe.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if not match:
            raise ValueError(f"Unsupported requirement; review the preflight parser: {line}")
        name, expected = match.groups()
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        # A local wheel tag such as torch 2.7.1+cpu satisfies the pinned release.
        packages[name] = {"expected": expected, "installed": installed,
                          "ok": installed is not None and installed.split("+", 1)[0] == expected}
    if not packages:
        raise ValueError("The pinned requirements file is empty")
    return {"ok": all(p["ok"] for p in packages.values()), "packages": packages}


def check_model(root: Path) -> dict:
    """Verify that the model is the real LFS object named by this checkout."""
    pointer = run(["git", "show", f"HEAD:{MODEL}"], root)
    oid = re.search(r"^oid sha256:([0-9a-f]{64})$", pointer, re.M)
    size = re.search(r"^size (\d+)$", pointer, re.M)
    if not pointer.startswith("version https://git-lfs.github.com/spec/v1\n") or not oid or not size:
        raise ValueError("Committed probe is not a valid LFS pointer; review the model source")
    path = root / MODEL
    if not path.is_file():
        raise FileNotFoundError(f"Missing {MODEL}; run git lfs pull for this file")
    with path.open("rb") as stream:
        if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError(f"{MODEL} is still an LFS pointer; run git lfs pull")
    actual_size = path.stat().st_size
    actual_hash = sha256(path)
    return {"ok": actual_size == int(size.group(1)) and actual_hash == oid.group(1),
            "path": MODEL, "bytes": actual_size, "sha256": actual_hash,
            "expected_bytes": int(size.group(1)), "expected_sha256": oid.group(1)}


def check_tools(root: Path, cc: str | None, qemu: str | None) -> dict:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scratchv.runtime.riscv_tensor import discover_toolchain, toolchain_versions
    tools = discover_toolchain(cc=cc, qemu=qemu)
    versions = toolchain_versions(tools)
    compiler_version = versions["compiler"].strip()
    machines = run([tools.qemu, "-machine", "help"], root)
    cpus = run([tools.qemu, "-cpu", "help"], root)
    has_virt = re.search(r"^\s*virt(?:\s|$)", machines, re.M) is not None
    has_rv64 = re.search(r"^\s*rv64(?:\s|$)", cpus, re.M) is not None
    return {"ok": tools.is_zig and compiler_version == "0.14.1" and has_virt and has_rv64,
            "compiler": list(tools.cc), "qemu": tools.qemu, "versions": versions,
            "virt_machine": has_virt, "rv64_cpu": has_rv64,
            "expected": "Zig 0.14.1 with bundled musl; qemu-system-riscv64",
            "note": "Other compilers need a separately validated freestanding math library."}


def check_cpu_torch() -> dict:
    torch = importlib.import_module("torch")
    hip = getattr(torch.version, "hip", None)
    return {"ok": torch.version.cuda is None and hip is None, "version": torch.__version__,
            "cuda_build": torch.version.cuda, "hip_build": hip}


def check_native_imports() -> dict:
    modules = {name: importlib.import_module(name) for name in ("numpy", "onnx", "onnxruntime")}
    providers = modules["onnxruntime"].get_available_providers()
    return {"ok": "CPUExecutionProvider" in providers,
            "versions": {name: module.__version__ for name, module in modules.items()},
            "ort_providers": providers}


def collect(root: Path, *, cc=None, qemu=None, require_clean=False) -> dict:
    report = {"schema_version": 1, "ready": False,
              "scope": "Environment prerequisites only; no numeric or human acceptance is performed.",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "checks": {}, "warnings": []}

    def record(name, operation):
        try:
            report["checks"][name] = operation()
        except Exception as exc:
            report["checks"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    record("python", lambda: {"ok": sys.version_info[:2] == (3, 12), "expected": "3.12"})

    def repository():
        head = run(["git", "rev-parse", "HEAD"], root)
        status = run(["git", "status", "--porcelain=v1", "--untracked-files=all"], root)
        if status:
            report["warnings"].append("Checkout contains local changes. Preserve the diff; HEAD alone does not identify this run.")
        return {"ok": not (require_clean and status), "head": head,
                "clean": not bool(status), "status": status.splitlines(), "require_clean": require_clean}

    record("repository", repository)
    record("packages", lambda: check_packages(root))
    record("cpu_torch", check_cpu_torch)
    record("native_imports", check_native_imports)
    record("pip_check", lambda: {"ok": True, "output": run([sys.executable, "-m", "pip", "check"], root)})
    record("toolchain", lambda: check_tools(root, cc, qemu))
    record("synthetic_model", lambda: check_model(root))
    report["ready"] = all(check["ok"] for check in report["checks"].values())
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/w1-repro-env")
    parser.add_argument("--cc", help="Zig executable; discovery matches the real runtime")
    parser.add_argument("--qemu", help="qemu-system-riscv64 executable")
    parser.add_argument("--require-clean", action="store_true", help="Fail on tracked or untracked local changes")
    args = parser.parse_args(argv)
    out = args.output_dir.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error("--output-dir must be absent or empty; preserve previous evidence")
    out.mkdir(parents=True, exist_ok=True)
    report = collect(ROOT, cc=args.cc, qemu=args.qemu, require_clean=args.require_clean)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# W1 reproduction environment", "", f"Ready: {report['ready']}", "", report["scope"], ""]
    for name, check in report["checks"].items():
        lines.append(f"- {name}: {'OK' if check['ok'] else 'FAIL'}")
        print(f"{'OK' if check['ok'] else 'FAIL'} {name}")
        if "error" in check:
            lines.append(f"  - {check['error']}")
            print(f"  {check['error']}")
    lines.extend(["", *report["warnings"], "", "Exact versions, paths, checkout and LFS hash: report.json."])
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {out / 'report.json'}")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
