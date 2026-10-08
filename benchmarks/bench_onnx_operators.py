"""Compare a pinned Git baseline with the current ONNX/IR implementation.

No checkout, reset, commit or Git worktree mutation. Baseline sources come
from git archive; the current dirty sources are copied into a second snapshot.
Both execute identical ONNX files and ORT references in isolated subprocesses.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "20b105e80ba3dbe13cb01a3d4ca18c32b6c31ed6"
THREADS = {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
TIMING_SCOPE = (
    "parse: fresh ONNXParser.parse per sample; run: IRInterpreter.run including verification, "
    "planning, binding copies, NumPy computation and return copy. ORT reference generation, "
    "common initializer loading, external verification and comparisons are excluded. "
    "Each round starts with one untimed correctness qualification, then the configured warmups "
    "and timed repetitions. Sequential ABBA order, two rounds per version; warmup is per round. "
    "Host CPU, not QEMU."
)
BINDING_POLICY = (
    "Both versions receive original ONNX initializer arrays without reshaping. The current "
    "parser's arrays must match dtype/shape/bytes, and its extra Constant globals are added. "
    "The baseline parser has no initializers property; supplying this existing interpreter "
    "argument is a harness adapter, not a baseline implementation change."
)


def git(*arguments, binary=False):
    # A source handoff may sit inside another checkout. Never let Git discover
    # that unrelated parent and manufacture a baseline for the extracted copy.
    if not (ROOT / ".git").exists():
        raise RuntimeError("Historical baseline comparison requires a Git checkout at "
                           f"{ROOT}; an extracted source archive has no Git history")
    result = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True,
                            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}, check=True)
    return result.stdout if binary else result.stdout.decode("utf-8").strip()


def snapshots(work, baseline_ref):
    from benchmarks.onnx_operator_worker import source_hash
    commit = git("rev-parse", "--verify", "--end-of-options", baseline_ref + "^{commit}")
    before, after = work / "before", work / "after"
    before.mkdir()
    archive = git("archive", "--format=zip", commit, "scratchv", binary=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        for member in source.infolist():
            if not (before / member.filename).resolve().is_relative_to(before.resolve()):
                raise ValueError("archive path outside baseline snapshot")
        source.extractall(before)
    shutil.copytree(ROOT / "scratchv", after / "scratchv",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    baseline = dict(commit=commit, source_sha256=source_hash(before), source_path=str(before))
    current = dict(head=git("rev-parse", "HEAD"),
                   dirty=bool(git("status", "--porcelain", "--untracked-files=all", "--", "scratchv")),
                   source_sha256=source_hash(after), source_path=str(after))
    return baseline, current


def add_probe_case(cases, fixtures, model_path):
    import numpy as np
    import onnx
    if not model_path.is_file():
        raise ValueError(f"PR #89 model missing: {model_path}; fetch its Git LFS data first")
    with model_path.open("rb") as stream:
        if stream.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError("PR #89 model is an LFS pointer; run git lfs pull first")
    name = "pr89_transformer_2l"
    model = fixtures / (name + ".onnx")
    shutil.copyfile(model_path, model)
    graph = onnx.load(model, load_external_data=False).graph
    feed = fixtures / (name + ".inputs.npz")
    np.savez(feed, input_ids=np.random.default_rng(42).integers(0, 128, (1, 256), dtype=np.int64),
             attention_mask=np.triu(np.full((256, 256), -1e9, dtype=np.float32), 1).reshape(1, 1, 256, 256))
    cases.append(dict(name=name, group="composite", operators=sorted({node.op_type for node in graph.node}),
                      description="PR #89 原始两层合成 Transformer，FP32，输出 [1,256,128]；非完整 Qwen3",
                      model=str(model.resolve()), inputs=str(feed.resolve()), atol=1e-5, rtol=0))


def prepare_references(cases, fixtures):
    import numpy as np
    import onnx
    import onnxruntime as ort
    from benchmarks.onnx_operator_worker import file_hash
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    for case in cases:
        try:
            case["model_sha256"] = file_hash(case["model"])
            case["inputs_sha256"] = file_hash(case["inputs"])
            model = onnx.load(case["model"])
            onnx.checker.check_model(model, full_check=True)
            arrays = {value.name: onnx.numpy_helper.to_array(value) for value in model.graph.initializer}
            initializers = fixtures / (case["name"] + ".initializers.npz")
            np.savez(initializers, **arrays)
            with np.load(case["inputs"], allow_pickle=False) as data:
                feed = {key: data[key] for key in data.files}
            session = ort.InferenceSession(case["model"], options, providers=["CPUExecutionProvider"])
            expected = session.run(None, feed)[0]
            if not np.isfinite(expected).all():
                raise ValueError("ORT reference is nonfinite")
            expected_path = fixtures / (case["name"] + ".expected.npy")
            np.save(expected_path, expected, allow_pickle=False)
            case.update(expected=str(expected_path), initializers=str(initializers),
                        expected_sha256=file_hash(expected_path), initializers_sha256=file_hash(initializers))
        except Exception as exc:
            case["reference_error"] = f"{type(exc).__name__}: {exc}"


def merge_rounds(rows):
    """One failure in any round invalidates performance comparisons."""
    result = dict(next((row for row in rows if row["status"] != "PASS"), rows[0]))
    for key in ("parse", "run"):
        samples = [sample for row in rows for sample in row[key + "_samples_s"]]
        result[key + "_samples_s"] = samples
        result[key + "_median_s"] = statistics.median(samples) if result["status"] == "PASS" else None
        result[key + "_min_s"] = min(samples) if samples and result["status"] == "PASS" else None
        result[key + "_max_s"] = max(samples) if samples and result["status"] == "PASS" else None
    if result["status"] == "PASS":
        for key in ("max_abs_error", "max_rel_error"):
            result[key] = max(row[key] for row in rows)
    return result


def run_comparison(args):
    os.environ.update(THREADS)
    import numpy as np
    import onnx
    import onnxruntime as ort
    from benchmarks.onnx_operator_cases import build_cases
    from benchmarks.onnx_operator_worker import empty_row, source_hash

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="run-", dir=output))
    fixtures = work / "fixtures"
    fixtures.mkdir()
    baseline, current = snapshots(work, args.baseline_ref)
    cases = build_cases(fixtures)
    if not args.case or "pr89_transformer_2l" in args.case:
        add_probe_case(cases, fixtures, args.probe_model.resolve())
    if args.case:
        unknown = set(args.case) - {case["name"] for case in cases}
        if unknown:
            raise ValueError(f"unknown cases: {sorted(unknown)}")
        cases = [case for case in cases if case["name"] in args.case]
    prepare_references(cases, fixtures)
    manifest = work / "manifest.json"
    manifest.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rounds = {"before": [], "after": []}
    versions = {"before": baseline, "after": current}
    for index, side in enumerate(("before", "after", "after", "before")):
        metadata = versions[side]
        destination = work / f"{index}-{side}.json"
        command = [sys.executable, "-I", "-B", "-X", "utf8", str(ROOT / "benchmarks/onnx_operator_worker.py"),
                   "--source", metadata["source_path"], "--source-sha256", metadata["source_sha256"],
                   "--manifest", str(manifest), "--output", str(destination),
                   "--warmup", str(args.warmup), "--repeats", str(args.repeats)]
        try:
            completed = subprocess.run(command, cwd=metadata["source_path"], capture_output=True,
                                       text=True, encoding="utf-8", env={**os.environ, **THREADS}, timeout=60)
            message = ((completed.stderr or completed.stdout or "worker produced no report")[-6000:]
                       if completed.returncode or not destination.is_file() else None)
        except subprocess.TimeoutExpired:
            message = f"worker timed out after 60 seconds: round={index}, version={side}"
        if message is not None:
            rows = {case["name"]: dict(empty_row(), status="WORKER_ERROR", phase="process", error=message)
                    for case in cases}
        else:
            rows = json.loads(destination.read_text(encoding="utf-8"))
        rounds[side].append(rows)
    for metadata in versions.values():
        if source_hash(Path(metadata["source_path"])) != metadata["source_sha256"]:
            raise RuntimeError("source snapshot changed during comparison")
    results = [dict(case, before=merge_rounds([data[case["name"]] for data in rounds["before"]]),
                    after=merge_rounds([data[case["name"]] for data in rounds["after"]])) for case in cases]
    blas = io.StringIO()
    with contextlib.redirect_stdout(blas):
        np.show_config()
    infrastructure_ok = all(row[side]["status"] not in ("REFERENCE_ERROR", "WORKER_ERROR")
                            for row in results for side in ("before", "after"))
    return dict(schema_version=1, baseline=baseline, current=current,
                environment=dict(python=platform.python_version(), executable=sys.executable,
                                 platform=platform.platform(), numpy=np.__version__, onnx=onnx.__version__,
                                 onnxruntime=ort.__version__, threads=THREADS, blas=blas.getvalue(),
                                 ort="CPUExecutionProvider, sequential, one thread, optimizations disabled"),
                warmup=args.warmup, repeats=args.repeats * 2, repeats_per_round=args.repeats,
                rounds_per_variant=2, execution_order=["before", "after", "after", "before"],
                timing_scope=TIMING_SCOPE, binding_policy=BINDING_POLICY,
                manifest=str(manifest), results=results,
                passed=infrastructure_ok and all(row["after"]["status"] == "PASS" for row in results))


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--baseline-ref", default=BASELINE)
    cli.add_argument("--output-dir", type=Path, default=Path("benchmark_reports/onnx_operators"))
    cli.add_argument("--probe-model", type=Path,
                     default=ROOT / "probes/w1_tiny_transformer/out/tiny_transformer_2l.onnx")
    cli.add_argument("--case", action="append", help="Run a named case (repeatable); default: all 24")
    cli.add_argument("--warmup", type=int, default=1)
    cli.add_argument("--repeats", type=int, default=5, help="Timed samples per ABBA round")
    args = cli.parse_args(argv)
    if args.warmup < 0 or args.repeats < 1:
        cli.error("warmup >= 0 and repeats >= 1 are required")
    from benchmarks.onnx_operator_report import render_html, render_markdown
    try:
        report = run_comparison(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        # Overwrite any previous success report, so a failed rerun cannot leave
        # a stale PASS as the visible result of this invocation.
        report = dict(schema_version=1, baseline={"ref": args.baseline_ref}, current={},
                      environment={}, results=[], passed=False, runner_error=f"{type(exc).__name__}: {exc}",
                      warmup=args.warmup, repeats=args.repeats * 2, timing_scope=TIMING_SCOPE,
                      binding_policy=BINDING_POLICY)
        print(f"benchmark failed: {exc}", file=sys.stderr)
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for suffix, content in (("json", json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"),
                                ("md", render_markdown(report)), ("html", render_html(report))):
            (args.output_dir / ("report." + suffix)).write_text(content, encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"report writing failed: {exc}", file=sys.stderr)
        return 1
    for side in ("before", "after"):
        count = sum(row[side]["status"] == "PASS" for row in report["results"])
        print(f"{side}: {count}/{len(report['results'])} PASS")
    print(f"Report: {(args.output_dir / 'report.html').resolve()}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
