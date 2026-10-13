#!/usr/bin/env python3
"""Verify the complete pinned Qwen3 FP32 export; missing artifacts fail the gate."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads(Path(__file__).with_name("manifest.json").read_text(encoding="utf-8"))
SEQ, VOCAB = 256, 151936
PINNED = {"numpy": "2.2.6", "onnx": "1.18.0", "onnxruntime": "1.22.1",
          "protobuf": "5.29.5"}
EXPORT_PINNED = {**PINNED, "torch": "2.7.1+cpu", "transformers": "4.51.3",
                 "onnxscript": "0.2.7", "onnx-ir": "0.1.7",
                 "huggingface-hub": "0.30.2", "safetensors": "0.5.3",
                 "tokenizers": "0.21.4"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprints():
    result = {}
    for name in ("probes/w1_qwen3_export/run.py", "export_qwen3_onnx.py",
                 "probes/w1_qwen3_export/manifest.json"):
        try:
            result[name] = {"sha256": sha256(ROOT / name)}
        except OSError as exc:
            result[name] = {"sha256": None, "error": f"{type(exc).__name__}: {exc}"}
    return result


def environment(export=False):
    expected = EXPORT_PINNED if export else PINNED
    actual = {}
    for name in expected:
        try:
            actual[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual[name] = None
    mismatch = {name: version for name, version in actual.items()
                if version != expected[name]}
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "packages": actual, "expected_packages": expected}
    if sys.version_info[:2] != (3, 12) or mismatch:
        raise RuntimeError("Use Python 3.12 and requirements/qwen3-export.txt "
                           f"(CPU torch for export); mismatches: {mismatch}")
    return result


def safe_member(name):
    """Only portable relative paths, including on Windows."""
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or path.is_absolute()
            or any(part in ("", ".", "..") for part in name.rstrip("/").split("/"))):
        raise ValueError(f"Unsafe artifact path: {name!r}")
    return path


def verify_files(directory, manifest=MANIFEST):
    directory = Path(directory).resolve()
    rows, seen = [], set()
    for item in manifest["files"]:
        name = item["name"]
        safe_member(name)
        if "/" in name or name in seen:
            raise ValueError(f"Duplicate or non-flat manifest file: {name}")
        seen.add(name)
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise FileNotFoundError(f"Missing model artifact: {name}")
        with path.open("rb") as stream:
            if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"Unresolved Git LFS pointer: {name}")
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Artifact size/SHA-256 mismatch: {name}")
        rows.append(dict(item))
    if "model.onnx" not in seen:
        raise ValueError("Manifest must include model.onnx")
    return rows


def extract_archive(archive, destination, manifest=MANIFEST):
    """Validate every ZIP member, then extract only the one complete model root."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Archive destination must not exist: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    required = {item["name"]: item["bytes"] for item in manifest["files"]}
    with zipfile.ZipFile(archive) as source:
        members = {}
        for item in source.infolist():
            safe_member(item.filename)
            if item.filename in members:
                raise ValueError(f"Duplicate ZIP member: {item.filename}")
            mode = item.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise ValueError(f"Non-regular ZIP member: {item.filename}")
            members[item.filename] = item
        roots = [PurePosixPath(name).parent for name, item in members.items()
                 if not item.is_dir() and PurePosixPath(name).name == "model.onnx"]
        if len(roots) != 1:
            raise ValueError("ZIP must contain exactly one model.onnx")
        prefix = roots[0]
        selected = {}
        for name, size in required.items():
            member_name = str(prefix / name)
            item = members.get(member_name)
            if item is None or item.is_dir() or item.file_size != size:
                raise ValueError(f"Missing/wrong-size ZIP model artifact: {name}")
            selected[name] = item
        for name in ("LICENSE", "verification.json"):
            item = members.get(str(prefix / name))
            if item is not None:
                if item.is_dir() or item.file_size > 1024 * 1024:
                    raise ValueError(f"Unexpected auxiliary ZIP file: {name}")
                selected[name] = item
        with tempfile.TemporaryDirectory(prefix="qwen3-extract-", dir=destination.parent) as temp:
            stage = Path(temp) / "model"
            stage.mkdir()
            for name, item in selected.items():
                with source.open(item) as incoming, (stage / name).open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, 8 * 1024 * 1024)
            verify_files(stage, manifest)
            stage.rename(destination)


def acquire_model(destination, archive=None, manifest=MANIFEST):
    """Re-use only hash-verified data; release tag and archive hash are pinned."""
    destination = Path(destination)
    if destination.exists():
        verify_files(destination, manifest)
        return {"reused": True, "release": manifest["release"]}
    release = manifest["release"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qwen3-download-", dir=destination.parent) as temp:
        path = Path(archive) if archive else Path(temp) / "model.zip"
        if archive is None:
            request = urllib.request.Request(release["url"], headers={"User-Agent": "ScratchV-W1-probe"})
            with urllib.request.urlopen(request, timeout=120) as incoming, path.open("xb") as outgoing:
                count = 0
                while chunk := incoming.read(8 * 1024 * 1024):
                    count += len(chunk)
                    if count > release["bytes"]:
                        raise ValueError("Release download exceeds pinned size")
                    outgoing.write(chunk)
        if path.stat().st_size != release["bytes"] or sha256(path) != release["sha256"]:
            raise ValueError("Release ZIP size/SHA-256 mismatch")
        extract_archive(path, destination, manifest)
    return {"reused": False, "release": release}


def tensors(message):
    if message.DESCRIPTOR.full_name == "onnx.TensorProto":
        yield message
        return
    for field, value in message.ListFields():
        if field.message_type is not None:
            repeated = (field.is_repeated if hasattr(field, "is_repeated")
                        else field.label == field.LABEL_REPEATED)
            if repeated:
                for child in value:
                    yield from tensors(child)
            else:
                yield from tensors(value)


def expected_io():
    return ({"input_ids": (7, [1, SEQ]), "attention_mask": (1, [1, 1, SEQ, SEQ])},
            {"logits": (1, [1, SEQ, VOCAB])})


def inspect_model(directory, files):
    import onnx
    from onnx import TensorProto as T

    directory = Path(directory).resolve()
    model = onnx.load(str(directory / "model.onnx"), load_external_data=False)
    if {x.domain: x.version for x in model.opset_import} != {"": 18} or model.ir_version != 10:
        raise ValueError("Expected ONNX opset 18 and IR version 10")
    if model.functions:
        raise ValueError("Expected inlined model without local functions")
    for values, expected in zip((model.graph.input, model.graph.output), expected_io()):
        actual = {}
        for value in values:
            tensor = value.type.tensor_type
            if any(not dim.HasField("dim_value") or dim.dim_param for dim in tensor.shape.dim):
                raise ValueError(f"Dynamic shape: {value.name}")
            actual[value.name] = (tensor.elem_type, [dim.dim_value for dim in tensor.shape.dim])
        if len(values) != len(actual) or actual != expected:
            raise ValueError(f"Unexpected static FP32 model interface: {actual}")
    allowed = {T.FLOAT: 4, T.INT64: 8, T.INT32: 4, T.BOOL: 1}
    for node in model.graph.node:
        if node.op_type == "Cast":
            targets = [attribute.i for attribute in node.attribute if attribute.name == "to"]
            if len(targets) != 1 or targets[0] not in allowed:
                raise ValueError(f"Non-FP32 floating/unsupported Cast target: {node.name}")
    external_files, count, dtype_counts = set(), 0, Counter()
    available = {item["name"]: item["bytes"] for item in files}
    for tensor in tensors(model):
        dtype_counts[T.DataType.Name(tensor.data_type)] += 1
        if tensor.data_type not in allowed:
            raise ValueError(f"Non-FP32 floating/unsupported tensor dtype: {tensor.name}")
        if tensor.data_location == T.EXTERNAL or tensor.external_data:
            info = {item.key: item.value for item in tensor.external_data}
            if (len(info) != len(tensor.external_data) or set(info) != {"location", "offset", "length"}
                    or tensor.data_location != T.EXTERNAL or tensor.raw_data):
                raise ValueError(f"Malformed external tensor: {tensor.name}")
            name = info["location"]
            safe_member(name)
            if "/" in name or name == "model.onnx" or name not in available:
                raise ValueError(f"Unverified external file: {name}")
            offset, length = int(info["offset"]), int(info["length"])
            required = math.prod(tensor.dims) * allowed[tensor.data_type]
            if any(dim < 0 for dim in tensor.dims) or offset < 0 or length != required:
                raise ValueError(f"Invalid external tensor range/size: {tensor.name}")
            if offset + length > available[name]:
                raise ValueError(f"Truncated external tensor: {tensor.name}")
            external_files.add(name)
            count += 1
    if external_files != set(available) - {"model.onnx"}:
        raise ValueError("Manifest files and referenced external data do not match")
    # Path-based checking handles the >2 GiB model without protobuf serialization.
    onnx.checker.check_model(str(directory / "model.onnx"), full_check=True)
    return {"opset": 18, "ir_version": 10, "external_tensors": count,
            "external_files": sorted(external_files), "tensor_dtypes": dict(dtype_counts),
            "operators": dict(sorted(Counter(n.op_type for n in model.graph.node).items()))}


def make_inputs(length, seed):
    import numpy as np

    if not 1 <= length <= SEQ:
        raise ValueError("Invalid valid-token length")
    ids = np.zeros((1, SEQ), np.int64)
    ids[:, :length] = np.random.default_rng(seed).integers(0, VOCAB, (1, length), dtype=np.int64)
    allowed = np.arange(SEQ)[None, :] <= np.arange(SEQ)[:, None]
    allowed &= np.arange(SEQ)[None, :] < length
    mask = np.where(allowed, np.float32(0), np.finfo(np.float32).min)
    return {"input_ids": ids, "attention_mask": mask.reshape(1, 1, SEQ, SEQ)}


def execute_ort(directory, threads):
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    begin = time.perf_counter()
    session = ort.InferenceSession(str(Path(directory) / "model.onnx"), options,
                                   providers=["CPUExecutionProvider"])
    session_seconds = time.perf_counter() - begin
    expected = expected_io()
    for values, schema in zip((session.get_inputs(), session.get_outputs()), expected):
        actual = {value.name: (value.type, value.shape) for value in values}
        wanted = {name: ("tensor(int64)" if dtype == 7 else "tensor(float)", shape)
                  for name, (dtype, shape) in schema.items()}
        if len(values) != len(actual) or actual != wanted:
            raise ValueError(f"Unexpected ORT interface: {actual}")
    cases = []
    for index, length in enumerate((min(5, SEQ), SEQ)):
        feed = make_inputs(length, 20260929 + index)
        begin = time.perf_counter()
        result = session.run(["logits"], feed)[0]
        elapsed = time.perf_counter() - begin
        if result.shape != (1, SEQ, VOCAB) or result.dtype != np.float32:
            raise ValueError(f"Unexpected logits shape/dtype: {result.shape}/{result.dtype}")
        # Small chunks avoid allocating another full 148 MiB logits tensor.
        if not all(np.isfinite(result[:, start:start + 8]).all() for start in range(0, SEQ, 8)):
            raise ValueError("ORT produced nonfinite logits")
        cases.append({"valid_length": length, "seed": 20260929 + index,
                      "shape": list(result.shape), "dtype": str(result.dtype),
                      "all_finite": True, "seconds": elapsed,
                      "last_valid_top1": int(result[0, length - 1].argmax()), "passed": True})
        del result
    return {"provider": "CPUExecutionProvider", "threads": threads,
            "graph_optimization": "disabled", "session_seconds": session_seconds, "cases": cases}


def peak_memory():
    """Peak RSS of this verifier process, explicitly excluding export children."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            current = ctypes.windll.kernel32.GetCurrentProcess
            current.restype = wintypes.HANDLE
            get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
            get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            if not get_memory(current(), ctypes.byref(counters), counters.cb):
                raise OSError("GetProcessMemoryInfo failed")
            value, method = counters.PeakWorkingSetSize, "Windows PeakWorkingSetSize"
        else:
            import resource
            value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            value *= 1 if sys.platform == "darwin" else 1024
            method = "getrusage(RUSAGE_SELF).ru_maxrss"
        return {"bytes": int(value), "method": method, "scope": "verifier process; excludes export subprocesses"}
    except (ImportError, OSError, AttributeError) as exc:
        return {"bytes": None, "method": "unavailable", "reason": str(exc)}


def export_model(args):
    source = Path(args.source_dir).resolve() if args.source_dir else None
    if source is None or not source.is_dir():
        raise FileNotFoundError("Export requires --source-dir containing the pinned Hugging Face snapshot")
    if sha256(source / "model.safetensors") != MANIFEST["source_checkpoint_sha256"]:
        raise ValueError("Source checkpoint SHA-256 mismatch")
    for item in MANIFEST["source_files"]:
        if sha256(source / item["name"]) != item["sha256"]:
            raise ValueError(f"Pinned source configuration/tokenizer mismatch: {item['name']}")
    output = args.model_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Export requires an empty model directory")
    script = ROOT / "export_qwen3_onnx.py"
    if not script.is_file():
        raise FileNotFoundError(f"Missing existing exporter: {script}")
    command = [sys.executable, "-X", "utf8", "-B", str(script), "--model-dir", str(source),
               "--output-dir", str(output), "--work-dir", str(args.output_dir / "export-work"),
               "--threads", str(args.threads), "--exporter", "dynamo"]
    with (args.output_dir / "export.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1"))
    manifest = json.loads((output / "verification.json").read_text(encoding="utf-8"))
    for key in ("model_id", "revision", "source_checkpoint_sha256", "dtype", "use_cache"):
        if manifest.get(key) != MANIFEST[key]:
            raise ValueError(f"Export provenance mismatch: {key}")
    if manifest.get("passed") is not True:
        raise ValueError("Existing exporter's PyTorch/ORT validation failed")
    return manifest


def _report_markdown(report):
    summary = ["# probe:qwen3-onnx", "", f"Result: {'PASS' if report['passed'] else 'FAIL'}",
               f"Mode: {report['mode']}; revision: {report['revision']}", "", report["scope"], "",
               f"Duration: {report['seconds']:.3f} s; verifier peak RSS: {report['peak_memory']['bytes']} bytes."]
    if "error" in report:
        summary += ["", f"Failed stage: {report['failed_stage']}", report["error"]]
    if report.get("report_write_errors"):
        summary += ["", "Report write errors:", *[f"- {item}" for item in report["report_write_errors"]]]
    summary += ["", "| Probe source | SHA-256 |", "|---|---|"]
    for name, fingerprint in report["source_fingerprints"].items():
        summary.append(f"| `{name}` | `{fingerprint['sha256'] or 'unavailable'}` |")
    return "\n".join(summary) + "\n"


def write_reports(out, report):
    """Publish complete JSON last and keep evidence failures non-successful."""
    failures = []

    def write(name, content):
        temporary = out / f".{name}.tmp"
        try:
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(out / name)
            return True
        except OSError as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                failures.append(f"{temporary.name}: cleanup failed: {cleanup_error}")
            return False

    def fail():
        if report["passed"]:
            report.update(passed=False, failed_stage="report-write",
                          error="Incomplete full ONNX gate evidence")
        # Keep the original environment/model/ORT error when the probe failed.
        report["report_write_errors"] = list(failures)

    def recover_view():
        write("report.md", _report_markdown(report))
        report["report_write_errors"] = list(failures)

    if not write("report.md", _report_markdown(report)):
        fail()
        recover_view()
    if not write("report.json", json.dumps(report, indent=2, allow_nan=False) + "\n"):
        fail()
        recover_view()
        # A retry may publish only FAIL, even if a failed close wrote all bytes.
        write("report.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
        report["report_write_errors"] = list(failures)
    if failures:
        print("[gate] FAIL; report write errors: " + "; ".join(failures), file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("verify", "download", "export"), default="verify")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/qwen3-0.6b-onnx")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/qwen3-export-probe")
    parser.add_argument("--archive", type=Path, help="Already downloaded pinned release ZIP (download mode)")
    parser.add_argument("--source-dir", type=Path, help="Pinned official HF snapshot (export mode)")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.threads < 1 or (args.archive and args.mode != "download") or (args.source_dir and args.mode != "export"):
        parser.error("threads must be positive; --archive is download-only and --source-dir export-only")
    args.output_dir = args.output_dir.resolve()
    args.model_dir = args.model_dir.resolve()
    if args.output_dir == args.model_dir or args.output_dir.is_relative_to(args.model_dir):
        parser.error("Report output must be outside the model directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # This entry point permits output-directory reuse. Invalidate the previous
    # result before any work, so interruption or a failed atomic publish cannot
    # leave an old successful JSON attributed to the current attempt.
    for name in ("report.json", "report.md"):
        (args.output_dir / name).unlink(missing_ok=True)
    started = time.perf_counter()
    report = {"schema_version": 1, "gate": "probe:qwen3-onnx", "passed": False,
              "created_at": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
              "model_id": MANIFEST["model_id"], "revision": MANIFEST["revision"],
              "environment": {"python": platform.python_version(), "platform": platform.platform()},
              "source_fingerprints": source_fingerprints(),
              "model_dir": str(args.model_dir), "stages_seconds": {},
              "scope": "Complete FP32 ONNX artifacts, fixed I/O and fresh ORT execution; "
                       "verify/download do not rerun export or compare PyTorch numerics."}
    stage = "environment"
    try:
        report["environment"] = environment(args.mode == "export")
        manifest = MANIFEST
        stage = args.mode
        before = time.perf_counter()
        if args.mode == "download":
            report["download"] = acquire_model(args.model_dir, args.archive)
        elif args.mode == "export":
            manifest = export_model(args)
            report["export_validation"] = {key: manifest[key] for key in ("passed", "cases", "full_tensor_allclose")}
        report["stages_seconds"][stage] = time.perf_counter() - before
        for stage, operation in (
            ("hashes", lambda: verify_files(args.model_dir, manifest)),
            ("structure", lambda: inspect_model(args.model_dir, report["hashes"])),
            ("ort", lambda: execute_ort(args.model_dir, args.threads)),
        ):
            before = time.perf_counter()
            report[stage] = operation()
            report["stages_seconds"][stage] = time.perf_counter() - before
        report["passed"] = True
    except Exception as exc:
        report["failed_stage"] = stage
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["seconds"] = time.perf_counter() - started
    report["peak_memory"] = peak_memory()
    write_reports(args.output_dir, report)
    print(json.dumps({key: report[key] for key in ("gate", "passed", "mode", "seconds")}), flush=True)
    if not report["passed"]:
        print(report["error"], file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
