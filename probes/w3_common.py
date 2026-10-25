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
import time

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

def _already_loaded_library(path):
    """Get an existing module only; never load a candidate BLAS library."""
    if sys.platform == "win32":
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        get_module = kernel.GetModuleHandleW
        get_module.argtypes = [wintypes.LPCWSTR]
        get_module.restype = wintypes.HMODULE
        handle = get_module(str(path))
        if not handle:
            raise OSError("NumPy library is not already loaded")
        return ctypes.CDLL(str(path), handle=handle)
    if sys.platform not in ("linux", "darwin") or not hasattr(os, "RTLD_NOLOAD"):
        raise OSError("Loaded-only library lookup is unavailable on this platform")
    return ctypes.CDLL(str(path), mode=os.RTLD_NOLOAD | getattr(os, "RTLD_LOCAL", 0))


def _openblas_text(library, name):
    for prefix in ("scipy_", ""):
        for suffix in ("64_", "_64", ""):
            symbol = f"{prefix}openblas_get_{name}{suffix}"
            try:
                function = getattr(library, symbol)
            except AttributeError:
                continue
            function.argtypes = []
            function.restype = ctypes.c_char_p
            value = function()
            if not value:
                raise ValueError(f"{symbol} returned no metadata")
            return value.decode("utf-8", errors="replace")
    raise AttributeError(f"OpenBLAS get_{name} symbol is unavailable")


def numpy_openblas_evidence(np):
    """Optional loaded NumPy-wheel BLAS metadata, separate from SIMD support.

    Only NumPy's private library directories are inspected. Other builds (for
    example system BLAS or MKL) remain explicitly unavailable rather than
    loading a library or inferring an active kernel from CPU capabilities.
    """
    result = {"status": "unavailable", "libraries": []}
    try:
        package = Path(np.__file__).resolve().parent
        candidates = set()
        for directory in (package.parent / "numpy.libs", package / ".libs", package / ".dylibs"):
            if directory.is_dir():
                candidates.update(path for path in directory.iterdir()
                                  if "openblas" in path.name.lower() and path.is_file()
                                  and (path.name.lower().endswith((".dll", ".dylib", ".so"))
                                       or ".so." in path.name.lower()))
        for path in sorted(candidates):
            row = {"path": str(path), "status": "unavailable"}
            result["libraries"].append(row)
            try:
                library = _already_loaded_library(path)
                row["core_name"] = _openblas_text(library, "corename")
                row["config"] = _openblas_text(library, "config")
                row["status"] = result["status"] = "available"
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
        if not candidates:
            result["reason"] = "No private NumPy OpenBLAS library found"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def numpy_runtime_evidence():
    """Record host math details without importing the ORT backend.

    Package versions alone do not identify BLAS or SIMD implementations.
    These public NumPy diagnostics are provenance, not proof of which ORT
    kernel executed. Missing diagnostics never fabricate a backend.
    """
    import numpy as np

    result = {"machine": platform.machine(), "processor": platform.processor(),
              "thread_environment": {name: os.environ.get(name) for name in (
                  "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
              "blas_environment": {"OPENBLAS_CORETYPE": os.environ.get("OPENBLAS_CORETYPE")},
              "numpy_openblas": numpy_openblas_evidence(np)}
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

def recheck_sources(report, *, provenance_key=None):
    """Reject mixed-source runs while retaining the changed file identities."""
    recorded = report if provenance_key is None else report.get(provenance_key, {})
    before = recorded.get("source_sha256")
    if not isinstance(before, dict) or not before:
        raise ValueError("Missing execution source fingerprints")
    after = source_evidence()["source_sha256"]
    changed = {name: {"before_sha256": before.get(name), "after_sha256": after.get(name)}
               for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)}
    report["source_recheck"] = {"passed": not changed, "changed": changed}
    if changed:
        raise ValueError("Production sources changed during numerical execution")


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
    primary_error = None
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        # File scanners/readers can briefly deny replacement of a closed report
        # on Windows. Retry only these OS errors, with a bounded total delay;
        # permanent permission failures must still fail report publication.
        delays = (0.01, 0.03, 0.1, 0.3)
        for attempt in range(len(delays) + 1):
            try:
                os.replace(name, path)
                break
            except PermissionError as exc:
                if (sys.platform != "win32" or getattr(exc, "winerror", None) not in (5, 32, 33)
                        or attempt == len(delays)):
                    raise
                time.sleep(delays[attempt])
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        except OSError as exc:
            if primary_error is None:
                raise
            if hasattr(primary_error, "add_note"):
                primary_error.add_note(f"Could not remove pending report: {type(exc).__name__}: {exc}")

def write_reports(out, report):
    """Publish required views before JSON. A write failure can never report PASS."""
    out = Path(out)
    try:
        payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        title = str(report.get("gate", "W3 preparation"))
        status = str(report.get("status", "PASS" if report.get("passed") else "FAIL"))
        markdown = f"# {title}\n\nResult: **{status}**\n\n"
        from probes.w3_summary import evidence_index_views, summary_views
        summary_md, summary_html = summary_views(report)
        index_md, index_html = evidence_index_views(report)
        markdown += summary_md + index_md
        html = ('<!doctype html><meta charset="utf-8"><title>' + escape(title) + '</title>'
                '<style>body{max-width:1100px;margin:30px auto;padding:0 20px;font:16px system-ui}'
                'table{width:100%;border-collapse:collapse;margin:16px 0}'
                'th,td{border:1px solid #ccc;padding:8px 12px;text-align:left;overflow-wrap:anywhere}'
                'th{background:#f3f5f7}summary{cursor:pointer}details{margin-top:20px}</style>'
                '<h1>' + escape(title) + '</h1><p>Result: <strong>' + escape(status) +
                '</strong></p>' + summary_html + index_html)
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
