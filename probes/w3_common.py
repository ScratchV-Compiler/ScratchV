"""Evidence utilities for W3 preparation; no downloads or global runtime changes."""
from __future__ import annotations
import ctypes
from contextlib import redirect_stdout
import hashlib
from html import escape
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def new_output_dir(path):
    out = Path(path).resolve()
    out.mkdir(parents=True, exist_ok=False)
    return out

def numpy_runtime_evidence():
    """Record host math details without importing the ORT backend.

    Package versions alone do not identify BLAS or SIMD implementations.
    These public NumPy diagnostics are provenance, not proof of which ORT
    kernel executed. Missing diagnostics never fabricate a backend.
    """
    import numpy as np

    result = {"machine": platform.machine(), "processor": platform.processor(),
              "thread_environment": {name: os.environ.get(name) for name in (
                  "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    for field, function in (("numpy_build", "show_config"), ("numpy_runtime", "show_runtime")):
        stream = io.StringIO()
        try:
            with redirect_stdout(stream):
                getattr(np, function)()
            result[field] = stream.getvalue().strip()
        except (AttributeError, RuntimeError, OSError) as exc:
            result[field] = None
            result[field + "_error"] = f"{type(exc).__name__}: {exc}"
    return result


def source_evidence():
    """Record this checkout's identity and content, including untracked W3 code."""
    versions = {}
    for name in ("numpy", "onnx", "onnxruntime", "torch", "transformers", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    git = {"head": None, "dirty": None, "root": str(ROOT),
           "status_scope": "source/docs; excludes LFS model binaries"}
    if (ROOT / ".git").exists():
        for field, args in (("head", ["rev-parse", "HEAD"]),
                            ("dirty", ["status", "--porcelain", "--untracked-files=normal", "--", ".",
                                       ":(exclude)**/*.onnx", ":(exclude)**/*.data",
                                       ":(exclude)**/*.safetensors"])):
            result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=15)
            if result.returncode == 0:
                git[field] = result.stdout.strip() if field == "head" else bool(result.stdout.strip())
    sources = []
    for folder in ("scratchv", "probes", "scripts"):
        sources.extend((ROOT / folder).rglob("*.py"))
    sources.extend((ROOT / "requirements").glob("*qwen*.txt"))
    sources.extend((ROOT / "probes").rglob("manifest.json"))
    sources = sorted(set(p for p in sources if "__pycache__" not in p.parts and "output" not in p.relative_to(ROOT).parts))
    return {
        "git": git,
        "environment": {"python": platform.python_version(), "executable": sys.executable,
                        "platform": platform.platform(), "versions": versions,
                        "numeric_runtime": numpy_runtime_evidence()},
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sources},
    }

def process_peak_rss_bytes():
    """Peak resident memory of this process, not IR workspace or an interval peak."""
    if os.name == "nt":
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
        return None
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except (ImportError, OSError):
        return None

def atomic_text(path, text):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)

def write_reports(out, report):
    """Publish required views before JSON. A write failure can never report PASS."""
    out = Path(out)
    try:
        payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        title = str(report.get("gate", "W3 preparation"))
        status = str(report.get("status", "PASS" if report.get("passed") else "FAIL"))
        markdown = f"# {title}\n\nResult: **{status}**\n\n"
        markdown += "See the structured evidence below for scope, source identity, cases, errors and measurements.\n\n"
        fence = chr(96) * 3
        markdown += fence + "json\n" + payload + fence + "\n"
        html = ('<!doctype html><meta charset="utf-8"><title>' + escape(title) + '</title>'
                '<style>body{max-width:1100px;margin:30px auto;padding:0 20px;font:16px system-ui}'
                'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;padding:20px}</style>'
                '<h1>' + escape(title) + '</h1><p>Result: <strong>' + escape(status) +
                '</strong></p><pre>' + escape(payload) + '</pre>')
        atomic_text(out / "report.md", markdown)
        atomic_text(out / "report.html", html)
        atomic_text(out / "report.json", payload)
    except Exception as exc:
        report["passed"] = False
        report["status"] = "FAIL"
        report.setdefault("report_errors", []).append(f"{type(exc).__name__}: {exc}")
        # A human may open a view directly; remove already-written PASS views
        # as well as JSON when any required report fails to publish.
        for filename in ("report.md", "report.html", "report.json"):
            try:
                (out / filename).unlink(missing_ok=True)
            except OSError as cleanup_error:
                report["report_errors"].append(f"cleanup {filename}: {cleanup_error}")
        target = out / "report.json"
        try:
            target.unlink(missing_ok=True)
            atomic_text(target, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        except Exception:
            pass
        raise
