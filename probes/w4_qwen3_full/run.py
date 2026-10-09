"""Build and execute the pinned complete Qwen3 model on bare-metal RV64 QEMU."""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
from html import escape
import json
from pathlib import Path
import sys
import time
import traceback
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probes.w1_qwen3_export import run as assets
from probes.w2_qwen3_parse.validation import audit_graph_structure
from probes.w3_common import atomic_text, source_evidence, sha256_file
from probes.w3_qwen3_full.cases import input_cases, CASE_NAMES
from probes.w4_qwen3_full.fma_conformance import run_probe as run_fma_probe
from scratchv.backend.tensor_c_codegen import TensorCCodegen, TensorSpec, _shape
from scratchv.ir.types import DataType
from scratchv.runtime.riscv_external import build_external, run_external, ExternalExecutable, validate_external_elf
from scratchv.runtime.riscv_tensor import discover_toolchain, RiscVToolchain, input_layout, toolchain_versions
from scratchv.runtime.weight_bundle import write_weight_bundle, validate_weight_bundle, plan_guest_memory

ATOL = 1e-3


def save_source_snapshot(out, hashes):
    """Retain executed source contents as well as hashes; never a clean git claim."""
    target = out / "source-snapshot.zip"
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, digest in sorted(hashes.items()):
            path = ROOT / name
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Source changed while taking snapshot: {name}")
            archive.writestr(name, data)
    return {"file": target.name, "sha256": sha256_file(target),
            "scope": "Hashed production/probe/script sources and fixed requirements/manifests; "
                     "restore over recorded Git base, not a complete standalone repository"}


def write_reports(out, report):
    try:
        payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        lines = ["# W4 complete Qwen3 RV64 probe", "", f"Status: **{report['status']}**",
                 f"Stage: `{report['stage']}`", "", "| Gate | Passed |", "|---|---|"]
        lines += [f"| `{name}` | {value} |" for name, value in report["gates"].items()]
        lines += ["", "External team reproduction and remote nightly remain pending.", "",
                  "```json", payload, "```", ""]
        atomic_text(out / "report.md", "\n".join(lines))
        atomic_text(out / "report.html", '<!doctype html><meta charset="utf-8"><title>W4 evidence</title>'
                    '<h1>W4 complete Qwen3 RV64 probe</h1><pre>' + escape(payload) + '</pre>')
        atomic_text(out / "report.json", payload + "\n")
    except Exception as exc:
        report.update(passed=False, status="FAIL", report_error=f"{type(exc).__name__}: {exc}")
        for name in ("report.json", "report.md", "report.html"):
            try:
                (out / name).unlink(missing_ok=True)
            except OSError:
                pass
        try:
            fallback = {"passed": False, "status": "FAIL", "stage": str(report.get("stage")),
                        "report_error": report["report_error"]}
            atomic_text(out / "report.json", json.dumps(fallback, indent=2, allow_nan=False) + "\n")
        except Exception:
            pass
        raise


def compare_logits(actual, expected):
    shape = (1, 256, 151936)
    if actual.shape != shape or expected.shape != shape or actual.dtype != np.float32 or expected.dtype != np.float32:
        raise ValueError("Full W4 comparison requires FP32 [1,256,151936] logits")
    maximum, worst, bad, square_sum = 0.0, None, 0, 0.0
    for pos in range(256):
        a, b = actual[0, pos], expected[0, pos]
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Nonfinite logits cannot pass numeric acceptance")
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        index = int(diff.argmax())
        if diff[index] > maximum:
            maximum, worst = float(diff[index]), [0, pos, index]
        bad += int(np.count_nonzero(diff >= ATOL))
        square_sum += float(np.dot(diff, diff))
    return {"passed": maximum < ATOL, "atol": ATOL, "rtol": 0.0,
            "criterion": "strict max_abs < atol across every output element",
            "max_abs": maximum, "worst_index": worst, "elements_at_or_above_atol": bad,
            "rmse": float(np.sqrt(square_sum / actual.size)), "elements_compared": int(actual.size)}


def spec_json(spec):
    return {"name": spec.name, "dtype": spec.dtype.value, "shape": list(spec.shape)}


def save_executable(exe, out):
    record = {"elf": str(exe.elf_path), "bundle_dir": str(exe.bundle_dir), "manifest": exe.manifest,
              "layout": exe.layout, "inputs": [spec_json(s) for s in exe.inputs],
              "weights": [spec_json(s) for s in exe.weights], "output": spec_json(exe.output),
              "cc": list(exe.toolchain.cc), "qemu": exe.toolchain.qemu, "evidence": exe.evidence}
    atomic_text(out / "executable.json", json.dumps(record, indent=2) + "\n")
    return sha256_file(out / "executable.json")


def load_executable(path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate executable metadata key: {key}")
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError(f"Nonfinite executable metadata: {value}")
    path = Path(path)
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError("Executable metadata exceeds 16 MiB")
    record = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                        parse_constant=reject_constant)
    def spec(value):
        if (set(value) != {"name", "dtype", "shape"} or not isinstance(value["name"], str)
                or not value["name"] or "\0" in value["name"]):
            raise ValueError("Invalid executable tensor spec")
        dtype = DataType(value["dtype"])
        if dtype not in (DataType.FLOAT32, DataType.INT32, DataType.INT64):
            raise ValueError("Unsupported external tensor dtype")
        return TensorSpec(value["name"], dtype, _shape(value["shape"]))
    exe = ExternalExecutable(Path(record["elf"]), Path(record["bundle_dir"]), record["manifest"],
        record["layout"], tuple(map(spec, record["inputs"])), spec(record["output"]),
        tuple(map(spec, record["weights"])), RiscVToolchain(tuple(record["cc"]), record["qemu"]),
        record["evidence"])
    manifest = validate_weight_bundle(exe.bundle_dir, exe.weights)
    if manifest != exe.manifest:
        raise ValueError("Reused weights do not match build manifest")
    _, input_bytes = input_layout(exe.inputs)
    layout = plan_guest_memory(workspace_bytes=exe.layout["regions"]["workspace"]["nbytes"],
        weight_bytes=manifest["total_bytes"], input_bytes=input_bytes, output_bytes=exe.output.nbytes)
    if layout != exe.layout:
        raise ValueError("Reused memory layout is not the canonical non-overlapping plan")
    evidence = validate_external_elf(exe.elf_path, layout)
    if any(evidence[key] != exe.evidence.get(key) for key in evidence):
        raise ValueError("Reused ELF changed after build")
    if set(exe.evidence["sources"]) != {"model.c", "guest.c", "start.S", "link.ld"}:
        raise ValueError("Reused generated source manifest must contain all four source files")
    for name, digest in exe.evidence["sources"].items():
        if Path(name).name != name or sha256_file(exe.elf_path.parent / name) != digest:
            raise ValueError("Reused generated source changed after build")
    return exe


def ort_reference(model_dir, feed, output):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_cpu_mem_arena = False
    session = ort.InferenceSession(str(model_dir / "model.onnx"), options,
                                  providers=["CPUExecutionProvider"])
    logits = session.run(["logits"], feed)[0]
    if logits.shape != (1, 256, 151936) or logits.dtype != np.float32 or not np.isfinite(logits).all():
        raise ValueError("Invalid complete ORT reference output")
    np.save(output, logits, allow_pickle=False)
    del logits, session
    gc.collect()


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model-dir", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path, required=True)
    cli.add_argument("--case", choices=(*CASE_NAMES, "all"), default="full_seed_0")
    cli.add_argument("--cc")
    cli.add_argument("--qemu")
    cli.add_argument("--timeout", type=float, default=10800)
    cli.add_argument("--build-timeout", type=float, default=900)
    cli.add_argument("--build-only", action="store_true")
    cli.add_argument("--matmul-policy", choices=("sequential", "blocked_fma"), default="sequential")
    cli.add_argument("--reuse-build", type=Path, help="Local executable.json; verified before every run")
    args = cli.parse_args(argv)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = {"schema_version": 1, "gate": "numeric:qemu-full-qwen3", "passed": False,
              "status": "FAIL", "stage": "initialization", "stages_seconds": {}, "cases": [],
              "gates": {"build:riscv-full": False, "smoke:qemu-full": False,
                        "numeric:qemu-full-qwen3": False},
              "full_qemu_forward_executed": False, "team_reproduction": "PENDING",
              "remote_nightly": "NOT_RUN", "w4_team_accepted": False,
              "scope": "Local bare-metal RV64 full single-forward evidence; no Linux guest/mmap or generation loop"}

    def stage(name, operation):
        report["stage"] = name
        atomic_text(out / "progress.json", json.dumps({"stage": name,
                    "elapsed_s": time.perf_counter() - start}) + "\n")
        print(f"[W4] {name}", flush=True)
        begin = time.perf_counter()
        try:
            return operation()
        finally:
            report["stages_seconds"][name] = time.perf_counter() - begin
            write_reports(out, report)

    try:
        if any(not np.isfinite(value) or value <= 0 for value in (args.timeout, args.build_timeout)):
            raise ValueError("Timeouts must be positive and finite")
        report.update(stage("source_evidence", source_evidence))
        report["source_snapshot"] = stage("source_snapshot", lambda: save_source_snapshot(out, report["source_sha256"]))
        model_dir = args.model_dir.resolve()
        report["model_dir"] = str(model_dir)
        report["matmul_policy"] = args.matmul_policy
        report["model_files"] = stage("asset_hashes", lambda: assets.verify_files(model_dir))
        import onnx
        model = stage("metadata_load", lambda: onnx.load(str(model_dir / "model.onnx"), load_external_data=False))
        report["graph_audit"] = stage("28_layer_audit", lambda: audit_graph_structure(model))
        if not report["graph_audit"]["passed"] or report["graph_audit"]["layer_count"] != 28:
            raise ValueError("Complete 28-layer Qwen3 structural audit failed")
        if args.reuse_build:
            previous = json.loads(args.reuse_build.with_name("report.json").read_text(encoding="utf-8"))
            if sha256_file(args.reuse_build) != previous.get("executable_sha256"):
                raise ValueError("Reused executable metadata hash is missing or changed")
            exe = stage("load_existing_build", lambda: load_executable(args.reuse_build))
            if args.cc:
                raise ValueError("Do not change --cc while reusing an ELF; build a new executable instead")
            if args.qemu:
                replacement = discover_toolchain(exe.toolchain.cc, args.qemu)
                report["qemu_override"] = {"built_with": exe.toolchain.qemu, "execute_with": replacement.qemu}
                exe = replace(exe, toolchain=replacement)
            if previous["model_files"] != report["model_files"]:
                raise ValueError("Reused build model identity mismatch")
            if previous.get("matmul_policy") != args.matmul_policy:
                raise ValueError("Reused build arithmetic policy differs from requested policy")
            # Relevant production files must still have exactly the build's contents.
            production = lambda hashes: {key: value for key, value in hashes.items() if key.startswith("scratchv/")}
            if production(previous["source_sha256"]) != production(report["source_sha256"]):
                raise ValueError("Reused build production source inventory or contents changed")
        else:
            from scratchv.frontend.onnx_parser import ONNXParser
            parser = ONNXParser()
            program = stage("parse", lambda: parser.parse(str(model_dir / "model.onnx"), mmap_external_data=True))
            artifact = stage("external_codegen", lambda: TensorCCodegen(program,
                initializers=parser.initializers, constant_storage="external",
                max_constant_bytes=3 * 1024**3, max_workspace_bytes=1024**3,
                kernel_calls=True, matmul_policy=args.matmul_policy).generate())
            report["codegen"] = {"source_bytes": len(artifact.source.encode()),
                                  "constant_bytes": artifact.constant_bytes,
                                  "workspace_bytes": artifact.workspace_bytes,
                                  "weights": len(artifact.external_weights),
                                  "kernel_calls": artifact.kernel_calls,
                                  "output": spec_json(artifact.output)}
            stage("write_weight_bundle", lambda: write_weight_bundle(artifact.external_weights,
                artifact.external_initializers, out / "weights"))
            toolchain = discover_toolchain(args.cc, args.qemu)
            exe = stage("build_riscv_full", lambda: build_external(artifact, out / "weights",
                out / "build", toolchain, timeout=args.build_timeout))
            del artifact, program, parser
            gc.collect()
        report["executable_sha256"] = save_executable(exe, out)
        report.update(build=exe.evidence, memory_layout=exe.layout, weight_sha256=exe.manifest["sha256"])
        report["gates"]["build:riscv-full"] = True
        report["execution_tools"] = stage("current_tool_versions", lambda: toolchain_versions(exe.toolchain))
        report["execution_tool_binary_sha256"] = {
            "cc": sha256_file(exe.toolchain.cc[0]), "qemu": sha256_file(exe.toolchain.qemu)}
        if args.build_only:
            report.update(status="BUILD_ONLY", stage="build_complete")
        else:
            report["fma_conformance"] = stage("fma_conformance", lambda: run_fma_probe(
                out / "fma-conformance", cc=exe.toolchain.cc, qemu=exe.toolchain.qemu))
            conformance = report["fma_conformance"]
            if (conformance.get("passed") is not True or conformance.get("status") != "PASS"
                    or conformance.get("tool_binary_sha256") != report["execution_tool_binary_sha256"]):
                raise ValueError("Actual execution toolchain failed strict RV64 FMA conformance")
            cases = input_cases()
            if args.case != "all":
                cases = [case for case in cases if case[0] == args.case]
            requested = list(CASE_NAMES) if args.case == "all" else [args.case]
            if [case[0] for case in cases] != requested:
                raise ValueError("Selected cases must exactly match requested nonempty coverage")
            for name, valid_length, feed in cases:
                case_dir = out / name
                case_dir.mkdir()
                np.savez(case_dir / "inputs.npz", **feed)
                row = {"name": name, "valid_length": valid_length, "passed": False,
                       "input_sha256": sha256_file(case_dir / "inputs.npz")}
                report["cases"].append(row)
                stage(f"{name}_ort", lambda: ort_reference(model_dir, feed, case_dir / "ort.npy"))
                row["ort_sha256"] = sha256_file(case_dir / "ort.npy")
                actual, run = stage(f"{name}_qemu", lambda: run_external(exe, feed, case_dir / "qemu", timeout=args.timeout))
                if sha256_file(exe.toolchain.qemu) != report["execution_tool_binary_sha256"]["qemu"]:
                    raise ValueError("QEMU executable changed after FMA preflight")
                row["qemu"] = run
                report["full_qemu_forward_executed"] = True
                report["gates"]["smoke:qemu-full"] = True
                expected = np.load(case_dir / "ort.npy", mmap_mode="r", allow_pickle=False)
                row["numeric"] = stage(f"{name}_comparison", lambda: compare_logits(actual, expected))
                row["passed"] = row["numeric"]["passed"]
                del actual, expected
                if not row["passed"]:
                    raise ValueError(f"{name}: full QEMU error {row['numeric']['max_abs']} exceeds strict {ATOL}")
            report["gates"]["numeric:qemu-full-qwen3"] = all(row["passed"] for row in report["cases"])
            report.update(passed=True, status="PASS", stage="complete", coverage=args.case,
                          cases_passed=len(report["cases"]), seven_case_coverage=args.case == "all")
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        if isinstance(exc, KeyboardInterrupt):
            report["interrupted"] = True
    report["elapsed_seconds"] = time.perf_counter() - start
    write_reports(out, report)
    print(f"[W4] {report['status']}: {out / 'report.json'}", flush=True)
    return 0 if report["passed"] or (args.build_only and report["status"] == "BUILD_ONLY") else 1


if __name__ == "__main__":
    raise SystemExit(main())
