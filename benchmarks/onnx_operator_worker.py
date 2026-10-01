"""Isolated execution worker for the ONNX before/after benchmark.

Run by filename with Python -I; no current-project imports precede selecting
the requested scratchv snapshot. This is a host NumPy benchmark, not QEMU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import onnx  # Preload the same dependencies for both versions, outside timing.


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash(root):
    digest = hashlib.sha256()
    for path in sorted((Path(root) / "scratchv").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def assert_import_source(root):
    root = Path(root).resolve() / "scratchv"
    for name, module in list(sys.modules.items()):
        if name == "scratchv" or name.startswith("scratchv."):
            path = getattr(module, "__file__", None)
            if path is None or not Path(path).resolve().is_relative_to(root):
                raise RuntimeError(f"mixed scratchv imports: {name} from {path}")


class NumericMismatch(ValueError):
    def __init__(self, message, metrics):
        super().__init__(message)
        self.metrics = metrics


def compare_output(actual, expected, atol, rtol):
    metrics = {"actual_shape": list(actual.shape) if isinstance(actual, np.ndarray) else None,
               "actual_dtype": str(actual.dtype) if isinstance(actual, np.ndarray) else None,
               "expected_shape": list(expected.shape), "expected_dtype": str(expected.dtype),
               "max_abs_error": None, "max_rel_error": None}
    if not isinstance(actual, np.ndarray) or actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise NumericMismatch("output shape/dtype differs from ORT", metrics)
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise NumericMismatch("nonfinite output or reference", metrics)
    if actual.dtype.kind in "iu":
        # Compare integers before float conversion, including values above 2**53.
        difference = np.abs(actual.astype(object) - expected.astype(object))
        maximum = float(max(difference.flat, default=0))
        correct = np.array_equal(actual, expected)
    else:
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        maximum = float(difference.max(initial=0))
        correct = bool(np.all(difference <= atol + rtol * np.abs(expected.astype(np.float64))))
    denominator = np.maximum(np.abs(expected.astype(np.float64)), 1e-12)
    relative = float((np.asarray(difference, dtype=np.float64) / denominator).max(initial=0))
    metrics.update(max_abs_error=maximum, max_rel_error=relative)
    if not correct:
        raise NumericMismatch(f"output differs from ORT: max_abs_error={maximum:g}", metrics)
    return metrics


def empty_row():
    return dict(status="EXECUTION_ERROR", phase="setup", error=None,
                max_abs_error=None, max_rel_error=None, parse_samples_s=[], run_samples_s=[],
                parse_median_s=None, run_median_s=None, executed_steps=None,
                actual_shape=None, actual_dtype=None, expected_shape=None, expected_dtype=None)


def array_signature(arrays):
    return {name: (array.dtype.str, array.shape, hashlib.sha256(array.tobytes()).hexdigest())
            for name, array in arrays.items()}


def bindings(parser, program, common):
    """Supply unchanged ONNX initializers even to the older parser API.

    The current parser must independently agree on their dtype/shape/bytes;
    its extra Constant globals are then supplied as well. No squeeze/repair.
    """
    supplied = dict(common)
    if hasattr(parser, "initializers"):
        parsed = parser.initializers
        if not set(common) <= set(parsed):
            raise ValueError("parser omitted original ONNX initializer data")
        if array_signature(common) != array_signature({key: parsed[key] for key in common}):
            raise ValueError("parser initializer data differs from original ONNX data")
        supplied.update(parsed)
    globals_ = {value.name for value in program.global_values}
    if not set(supplied) <= globals_:
        raise ValueError("initializer data has no matching IR global")
    return supplied


def run_case(case, *, warmup, repeats, parser_cls, interpreter_cls, verifier):
    row = empty_row()
    if case.get("reference_error"):
        row.update(status="REFERENCE_ERROR", phase="reference", error=case["reference_error"])
        return row
    try:
        row["phase"] = "reference"
        for key in ("model", "inputs", "expected", "initializers"):
            if file_hash(case[key]) != case[key + "_sha256"]:
                raise ValueError(f"fixture fingerprint mismatch: {key}")
        with np.load(case["inputs"], allow_pickle=False) as data:
            feed = {key: data[key] for key in data.files}
        with np.load(case["initializers"], allow_pickle=False) as data:
            common = {key: data[key] for key in data.files}
        expected = np.load(case["expected"], allow_pickle=False)
        if not np.isfinite(expected).all():
            raise ValueError("nonfinite ORT reference")
        row.update(expected_shape=list(expected.shape), expected_dtype=str(expected.dtype))
        original_feed, original_common = array_signature(feed), array_signature(common)

        # Fresh parsers avoid the baseline's parser-reuse accumulation. Each
        # resulting Program is executed, including warmups and timed samples.
        for iteration in range(1 + warmup + repeats):
            row["phase"] = "parse"
            parser = parser_cls()
            start = time.perf_counter()
            program = parser.parse(case["model"])
            parse_seconds = time.perf_counter() - start
            row["phase"] = "verify"
            valid, issues = verifier(program)
            if not valid:
                raise ValueError("; ".join(str(issue) for issue in issues[:3]))
            row["phase"] = "bindings"
            weights = bindings(parser, program, common)
            original_weights = array_signature(weights)
            interpreter = interpreter_cls(program)
            row["phase"] = "execute"
            start = time.perf_counter()
            result = interpreter.run(feed, initializers=weights)
            run_seconds = time.perf_counter() - start
            row["phase"] = "compare"
            metrics = compare_output(result.return_value, expected, case["atol"], case["rtol"])
            if (array_signature(feed) != original_feed or array_signature(common) != original_common
                    or array_signature(weights) != original_weights):
                raise NumericMismatch("execution mutated inputs or initializer data", metrics)
            for key in ("max_abs_error", "max_rel_error"):
                metrics[key] = max(metrics[key], row[key] or 0)
            row.update(metrics, executed_steps=result.executed_steps)
            if iteration > warmup:
                row["parse_samples_s"].append(parse_seconds)
                row["run_samples_s"].append(run_seconds)
        row["phase"] = "reference"
        for key in ("model", "inputs", "expected", "initializers"):
            if file_hash(case[key]) != case[key + "_sha256"]:
                raise ValueError(f"fixture changed during execution: {key}")
        row.update(status="PASS", phase="complete",
                   parse_median_s=statistics.median(row["parse_samples_s"]),
                   run_median_s=statistics.median(row["run_samples_s"]))
    except Exception as exc:
        phase = row["phase"]
        status = {"reference": "REFERENCE_ERROR", "parse": "PARSE_ERROR",
                  "verify": "VERIFY_ERROR", "compare": "NUMERIC_ERROR"}.get(phase, "EXECUTION_ERROR")
        if phase == "parse" and "Unsupported ONNX op" in str(exc):
            status = "UNSUPPORTED"
        if isinstance(exc, NumericMismatch):
            row.update(exc.metrics)
        row.update(status=status, error=f"{type(exc).__name__}: {exc}")
    return row


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", type=Path, required=True)
    cli.add_argument("--source-sha256", required=True)
    cli.add_argument("--manifest", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--warmup", type=int, required=True)
    cli.add_argument("--repeats", type=int, required=True)
    args = cli.parse_args(argv)
    if args.warmup < 0 or args.repeats < 1:
        cli.error("warmup >= 0 and repeats >= 1 are required")
    if source_hash(args.source) != args.source_sha256:
        raise RuntimeError("source snapshot fingerprint mismatch")
    sys.path.insert(0, str(args.source.resolve()))
    from scratchv.frontend.onnx_parser import ONNXParser
    from scratchv.analysis.ir_verifier import verify_ir
    from scratchv.verification.ir_interpreter import IRInterpreter
    assert_import_source(args.source)
    cases = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = {case["name"]: run_case(case, warmup=args.warmup, repeats=args.repeats,
                                 parser_cls=ONNXParser, interpreter_cls=IRInterpreter,
                                 verifier=verify_ir) for case in cases}
    assert_import_source(args.source)
    if source_hash(args.source) != args.source_sha256:
        raise RuntimeError("source snapshot changed while running")
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
