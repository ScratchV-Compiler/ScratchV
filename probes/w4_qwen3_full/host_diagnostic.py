"""Optional full-graph Host C diagnostic; never an RV64/QEMU acceptance gate."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import gc
from html import escape
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from probes.w1_qwen3_export import run as assets
from probes.w2_qwen3_parse.validation import audit_graph_structure
from probes.w3_common import atomic_text, process_peak_rss_bytes, sha256_file, source_evidence
from probes.w3_qwen3_full.cases import CASE_NAMES, input_cases
from probes.w4_qwen3_full.run import compare_logits, ort_reference
from scratchv.backend.tensor_c_codegen import TensorCCodegen
from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.runtime.riscv_tensor import _run_process, _save_process_logs


ROOT = Path(__file__).resolve().parents[2]
POINTERS = ctypes.POINTER(ctypes.c_void_p)


def new_report(case, policy):
    return {"schema_version": 1, "diagnostic_kind": "host_c_full_graph", "host_only": True,
            "status": "RUNNING", "passed": False, "host_diagnostic_passed": False,
            "full_host_forward_executed": False, "full_qemu_forward_executed": False,
            "w4_team_accepted": False,
            "gates": {"build:riscv-full": False, "smoke:qemu-full": False,
                      "numeric:qemu-full-qwen3": False},
            "scope": "Native host shared-library diagnostic only; no RV64 code or QEMU forward",
            "case": case, "matmul_policy": policy, "stage": "initialization",
            "stages_seconds": {}, "host_process_peak_rss_bytes": {}}


def publish(out, report):
    """Publish explicit Host C views, clearing stale success on any failure."""
    try:
        payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        text = ("# Host C full-graph diagnostic\n\n**This is not an RV64/QEMU gate.**\n\n"
                f"Status: **{report['status']}**; stage: `{report['stage']}`.\n\n"
                "```json\n" + payload + "\n```\n")
        atomic_text(out / "report.md", text)
        atomic_text(out / "report.html", '<!doctype html><meta charset="utf-8">'
                    '<title>Host C diagnostic, not RV64 acceptance</title>'
                    '<h1>Host C diagnostic — no RV64/QEMU claim</h1><pre>' + escape(payload) + '</pre>')
        atomic_text(out / "report.json", payload + "\n")
    except Exception as exc:
        report.update(status="HOST_FAIL", passed=False, host_diagnostic_passed=False)
        for name in ("report.json", "report.md", "report.html"):
            try:
                (out / name).unlink(missing_ok=True)
            except OSError:
                pass
        try:
            fallback = new_report(report.get("case"), report.get("matmul_policy"))
            fallback.update(status="HOST_FAIL", report_error=f"{type(exc).__name__}: {exc}")
            atomic_text(out / "report.json", json.dumps(fallback) + "\n")
        except Exception:
            pass
        raise


def compiler_command(cc=None):
    candidates = [cc] if cc else [os.environ.get("SCRATCHV_ZIG"), os.environ.get("SCRATCHV_CC"),
                                "zig", "clang", "gcc", "cc"]
    for candidate in candidates:
        if not candidate:
            continue
        found = str(Path(candidate).resolve()) if Path(candidate).is_file() else shutil.which(candidate)
        if found:
            return [found, "cc"] if Path(found).stem.lower() == "zig" else [found]
    raise FileNotFoundError("Host C diagnostic requires an explicit Zig/clang/gcc compiler")


def read_worker_report(path):
    """Read local worker evidence without accepting ambiguous JSON fields."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate Host C report key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Nonfinite Host C report value: {value}")

    if path.stat().st_size > 16 * 1024**2:
        raise ValueError("Host C worker report exceeds 16 MiB")
    report = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                        parse_constant=reject_constant)
    if not isinstance(report, dict):
        raise ValueError("Host C worker report must be an object")
    return report


def build_library(artifact, directory, compiler, timeout=900.0):
    if artifact.constant_storage != "external":
        raise ValueError("Host diagnostic requires the external tensor ABI")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / "model.c"
    source.write_text(artifact.source, encoding="utf-8")
    library = directory / ("model.dll" if os.name == "nt" else "model.so")
    flags = list(dict.fromkeys(["-fno-fast-math", "-ffp-contract=off", "-fno-strict-aliasing",
                              *artifact.compile_flags]))
    command = [*compiler, "-shared", "-O2", "-std=c11", *flags]
    if os.name != "nt":
        command.append("-fPIC")
    command += [str(source), "-o", str(library), "-lm"]
    environment = dict(os.environ)
    environment["ZIG_GLOBAL_CACHE_DIR"] = str(ROOT / "output/zig-global-cache")
    environment["ZIG_LOCAL_CACHE_DIR"] = str(directory / "zig-cache")
    try:
        result = _run_process(command, cwd=directory, timeout=timeout, env=environment)
    except subprocess.TimeoutExpired as exc:
        _save_process_logs(directory, "compiler", exc.stdout, exc.stderr)
        raise TimeoutError(f"Host C compilation exceeded {timeout}s") from exc
    _save_process_logs(directory, "compiler", result.stdout, result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Host C compiler failed ({result.returncode}): "
                           + result.stderr.decode(errors="replace")[-8000:])
    version_command = [compiler[0], "version"] if len(compiler) == 2 else [compiler[0], "--version"]
    version = _run_process(version_command, cwd=directory, timeout=30, env=environment)
    if version.returncode:
        raise RuntimeError("Could not record the host compiler version")
    return library, {"command": command, "compiler_version": version.stdout.decode(errors="replace").strip(),
                     "library_sha256": sha256_file(library), "library_bytes": library.stat().st_size,
                     "source_sha256": sha256_file(source), "compile_flags": flags,
                     "implicit_fma_contraction": False, "kernel_calls": artifact.kernel_calls,
                     "matmul_policy": artifact.matmul_policy}


def _aligned_bytes(nbytes):
    if type(nbytes) is not int or nbytes < 0:
        raise ValueError("Storage size must be a nonnegative integer")
    owner = np.empty(max(1, nbytes) + 63, dtype=np.uint8)
    offset = (-owner.ctypes.data) % 64
    return owner, owner[offset:offset + nbytes]


def _array_for_spec(array, spec):
    if (not isinstance(array, np.ndarray) or array.shape != tuple(spec.shape)
            or array.dtype != np.dtype(spec.numpy_dtype)):
        raise ValueError(f"Host tensor {spec.name!r} shape/dtype mismatch")
    result = array if array.flags.c_contiguous else np.ascontiguousarray(array)
    if not result.flags.aligned:
        raise ValueError(f"Host tensor {spec.name!r} must have aligned element storage")
    return result


@dataclass
class NativeBuffers:
    """Own every Python object whose address the synchronous C call may use."""
    inputs: tuple
    weights: tuple
    workspace_owner: np.ndarray
    workspace: np.ndarray
    output_owner: np.ndarray
    output: np.ndarray
    input_pointers: object
    weight_pointers: object
    copied_weight_bytes: int

    def arguments(self, workspace_bytes):
        return (self.input_pointers, self.weight_pointers, self.workspace.ctypes.data,
                workspace_bytes, self.output.ctypes.data)


def prepare_buffers(artifact, feed):
    if set(feed) != {spec.name for spec in artifact.inputs}:
        raise ValueError("Host input names must exactly match the generated ABI")
    if len(artifact.external_weights) != len(artifact.external_initializers):
        raise ValueError("External tensor specs and arrays differ in length")
    inputs = tuple(_array_for_spec(feed[spec.name], spec) for spec in artifact.inputs)
    weights = tuple(_array_for_spec(array, spec)
                    for spec, array in zip(artifact.external_weights, artifact.external_initializers))
    copied = sum(array.nbytes for array, original in zip(weights, artifact.external_initializers)
                 if array is not original)
    workspace_owner, workspace = _aligned_bytes(artifact.workspace_bytes)
    output_owner, output_bytes = _aligned_bytes(artifact.output.nbytes)
    output = output_bytes.view(artifact.output.numpy_dtype).reshape(artifact.output.shape)
    if output.dtype.kind == "f":
        output.fill(np.nan)  # An unwritten output element must not look valid.
    input_pointers = (ctypes.c_void_p * len(inputs))(*(array.ctypes.data for array in inputs))
    weight_pointers = (ctypes.c_void_p * len(weights))(*(array.ctypes.data for array in weights))
    return NativeBuffers(inputs, weights, workspace_owner, workspace, output_owner, output,
                         input_pointers, weight_pointers, copied)


def invoke_library(artifact, buffers, library):
    loaded = ctypes.CDLL(str(library))
    function = None
    try:
        function = getattr(loaded, artifact.function_name)
        function.restype = ctypes.c_int
        function.argtypes = [POINTERS, POINTERS, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        status = function(*buffers.arguments(artifact.workspace_bytes))
        if status != 0:
            raise RuntimeError(f"Host C returned failure status {status}")
        return buffers.output
    finally:
        del function
        import _ctypes
        if os.name == "nt":
            _ctypes.FreeLibrary(loaded._handle)
        else:
            _ctypes.dlclose(loaded._handle)
        loaded._handle = 0


def worker(args):
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = new_report(args.case, args.matmul_policy)
    started = time.perf_counter()
    publish(out, report)

    def stage(name, operation):
        report["stage"] = name
        atomic_text(out / "progress.json", json.dumps({"stage": name,
                    "elapsed_s": time.perf_counter() - started}) + "\n")
        print(f"[Host C only] {name}", flush=True)
        begin = time.perf_counter()
        try:
            return operation()
        finally:
            report["stages_seconds"][name] = time.perf_counter() - begin
            report["host_process_peak_rss_bytes"][name] = process_peak_rss_bytes()
            publish(out, report)

    try:
        report.update(stage("source_evidence", source_evidence))
        model_dir = args.model_dir.resolve()
        report["model_dir"] = str(model_dir)
        report["model_files"] = stage("asset_hashes", lambda: assets.verify_files(model_dir))
        import onnx
        metadata = stage("metadata_load", lambda: onnx.load(str(model_dir / "model.onnx"), load_external_data=False))
        report["graph_audit"] = stage("28_layer_audit", lambda: audit_graph_structure(metadata))
        if report["graph_audit"].get("passed") is not True or report["graph_audit"].get("layer_count") != 28:
            raise ValueError("Host diagnostic requires the fixed complete 28-layer model")
        selected = [case for case in input_cases() if case[0] == args.case]
        if len(selected) != 1:
            raise ValueError("Host diagnostic requires exactly one selected full input")
        _, valid_length, feed = selected[0]
        report["valid_length"] = valid_length
        np.savez(out / "inputs.npz", **feed)
        report["input_sha256"] = sha256_file(out / "inputs.npz")
        del metadata
        # Keep ORT's session/weights out of the native C allocation phase.
        stage("ort_reference", lambda: ort_reference(model_dir, feed, out / "ort.npy"))
        gc.collect()
        report["ort_sha256"] = sha256_file(out / "ort.npy")
        parser = ONNXParser()
        program = stage("parse", lambda: parser.parse(str(model_dir / "model.onnx"), mmap_external_data=True))
        artifact = stage("external_codegen", lambda: TensorCCodegen(program, parser.initializers,
            constant_storage="external", kernel_calls=True, matmul_policy=args.matmul_policy,
            max_constant_bytes=3 * 1024**3, max_workspace_bytes=1024**3).generate())
        report["codegen"] = {"workspace_bytes": artifact.workspace_bytes,
                             "weight_bytes": artifact.constant_bytes,
                             "weights": len(artifact.external_weights),
                             "source_bytes": len(artifact.source.encode()),
                             "output_shape": list(artifact.output.shape), "kernel_calls": True}
        compiler = compiler_command(args.cc)
        library, report["build"] = stage("host_shared_library", lambda:
            build_library(artifact, out / "build", compiler, args.compile_timeout))
        buffers = stage("allocate_native_buffers", lambda: prepare_buffers(artifact, feed))
        report["buffer_policy"] = {"workspace_alignment": 64, "copied_weight_bytes": buffers.copied_weight_bytes,
                                   "borrowed_weight_count": sum(array is original for array, original in
                                       zip(buffers.weights, artifact.external_initializers)),
                                   "weight_lifetime": "Strong Python references retained through the native call"}
        del program, parser
        gc.collect()
        actual = stage("host_c_forward", lambda: invoke_library(artifact, buffers, library))
        report["full_host_forward_executed"] = True
        np.save(out / "host.npy", actual, allow_pickle=False)
        report["host_output_sha256"] = sha256_file(out / "host.npy")
        expected = np.load(out / "ort.npy", mmap_mode="r", allow_pickle=False)
        report["numeric"] = stage("full_logits_comparison", lambda: compare_logits(actual, expected))
        report.update(passed=report["numeric"]["passed"], host_diagnostic_passed=report["numeric"]["passed"],
                      status="HOST_PASS" if report["numeric"]["passed"] else "HOST_FAIL", stage="complete")
    except BaseException as exc:
        report.update(passed=False, host_diagnostic_passed=False, status="HOST_FAIL",
                      error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    report["elapsed_seconds"] = time.perf_counter() - started
    publish(out, report)
    return 0 if report["host_diagnostic_passed"] else 1


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model-dir", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path, required=True)
    cli.add_argument("--case", choices=CASE_NAMES, default="full_seed_0")
    cli.add_argument("--cc")
    cli.add_argument("--matmul-policy", choices=("sequential", "blocked_fma"), default="sequential")
    cli.add_argument("--timeout", type=float, default=3600.0)
    cli.add_argument("--compile-timeout", type=float, default=900.0)
    cli.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = cli.parse_args(argv)
    if any(not np.isfinite(value) or value <= 0 for value in (args.timeout, args.compile_timeout)):
        raise ValueError("Diagnostic timeouts must be positive and finite")
    if args.worker:
        return worker(args)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = new_report(args.case, args.matmul_policy)
    report["stage"] = "isolated_worker"
    publish(out, report)
    command = [sys.executable, "-B", "-X", "utf8", "-m", "probes.w4_qwen3_full.host_diagnostic",
               "--worker", "--model-dir", str(args.model_dir.resolve()), "--output-dir", str(out / "worker"),
               "--case", args.case, "--matmul-policy", args.matmul_policy,
               "--compile-timeout", str(args.compile_timeout)]
    if args.cc:
        command += ["--cc", args.cc]
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    start = time.perf_counter()
    try:
        result = _run_process(command, cwd=ROOT, timeout=args.timeout, env=environment)
        _save_process_logs(out, "worker", result.stdout, result.stderr)
        child = read_worker_report(out / "worker/report.json")
        if (type(child.get("schema_version")) is not int or child["schema_version"] != 1
                or child.get("diagnostic_kind") != "host_c_full_graph"
                or child.get("w4_team_accepted") is not False
                or child.get("host_only") is not True or child.get("full_qemu_forward_executed") is not False
                or child.get("case") != args.case or child.get("matmul_policy") != args.matmul_policy
                or set(child.get("gates", {})) != set(report["gates"])
                or any(value is not False for value in child.get("gates", {}).values())):
            raise ValueError("Worker report crossed the Host C diagnostic boundary")
        report = child
        if result.returncode or report.get("status") != "HOST_PASS" or report.get("host_diagnostic_passed") is not True:
            report.update(passed=False, host_diagnostic_passed=False, status="HOST_FAIL")
        else:
            numeric = report.get("numeric", {})
            error = numeric.get("max_abs")
            if (report.get("full_host_forward_executed") is not True or report.get("passed") is not True
                    or numeric.get("passed") is not True or numeric.get("atol") != 1e-3
                    or numeric.get("rtol") != 0.0 or numeric.get("elements_compared") != 256 * 151936
                    or isinstance(error, bool) or not isinstance(error, (int, float))
                    or not np.isfinite(error) or not 0 <= error < 1e-3):
                raise ValueError("Incomplete or invalid Host C full-logits numerical evidence")
        report["worker_exit_code"] = result.returncode
    except BaseException as exc:
        if isinstance(exc, subprocess.TimeoutExpired):
            _save_process_logs(out, "worker", exc.stdout, exc.stderr)
        report.update(passed=False, host_diagnostic_passed=False, status="HOST_FAIL",
                      error=f"{type(exc).__name__}: {exc}")
    report["worker_command"] = command
    report["parent_elapsed_seconds"] = time.perf_counter() - start
    report["worker_timeout_seconds"] = args.timeout
    publish(out, report)
    print(f"[Host C only] {report['status']}: {out / 'report.json'}", flush=True)
    return 0 if report["host_diagnostic_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
