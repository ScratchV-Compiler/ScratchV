#!/usr/bin/env python3
"""Compare recorded checkpoint arrays without executing a model or loading pickle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probes.w2_qwen3_small.diagnostics import compare_outputs, tensor_diff
from probes.w3_common import source_evidence, write_reports

DEFAULT_MAX_BYTES = 2 * 1024**3


class EvidenceError(ValueError):
    def __init__(self, message, checkpoint=None, details=None):
        super().__init__(message)
        self.checkpoint = checkpoint
        self.details = details


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def load_schema(path: Path) -> list[dict]:
    if path.stat().st_size > 8 * 1024**2:
        raise EvidenceError("schema exceeds 8 MiB limit")
    document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              EvidenceError(f"nonfinite JSON constant: {value}")))
    if (not isinstance(document, dict) or not _integer(document.get("version"))
            or document["version"] != 1):
        raise EvidenceError("schema must be an object with version=1")
    entries = document.get("checkpoints")
    if not isinstance(entries, list) or not entries:
        raise EvidenceError("schema checkpoints must be a nonempty ordered list")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or not {"name", "shape", "dtype"} <= set(entry):
            raise EvidenceError("each checkpoint requires name, shape and dtype")
        name = entry["name"]
        if not isinstance(name, str) or not name or name in seen:
            raise EvidenceError(f"empty, invalid or duplicate checkpoint name: {name!r}")
        seen.add(name)
        shape = entry["shape"]
        if (not isinstance(shape, list)
                or any(not _integer(dim) or dim < 0 for dim in shape)):
            raise EvidenceError("shape must contain nonnegative integer dimensions", name)
        try:
            if not isinstance(entry["dtype"], str):
                raise TypeError("dtype must be a string")
            dtype = np.dtype(entry["dtype"])
        except (TypeError, ValueError) as exc:
            raise EvidenceError(f"invalid dtype: {entry['dtype']!r}", name) from exc
        if dtype.kind not in "biuf" or dtype.hasobject:
            raise EvidenceError("dtype must be real numeric or boolean, without objects", name)
        if "layer" in entry and (not _integer(entry["layer"]) or entry["layer"] < 0):
            raise EvidenceError("layer must be a nonnegative integer", name)
        if "checkpoint" in entry and (not isinstance(entry["checkpoint"], str)
                                      or not entry["checkpoint"]):
            raise EvidenceError("checkpoint must be a nonempty string", name)
        axis = entry.get("sequence_axis")
        if axis is not None and (not _integer(axis) or not -len(shape) <= axis < len(shape)):
            raise EvidenceError("sequence_axis is outside checkpoint rank", name)
    return entries


def load_arrays(path: Path, entries: list[dict], max_bytes=DEFAULT_MAX_BYTES) -> dict:
    """Preflight ZIP membership/size, then load numeric NPY arrays with no pickle."""
    if not _integer(max_bytes) or max_bytes < 1:
        raise EvidenceError("max_bytes must be a positive integer")
    names = [entry["name"] for entry in entries]
    by_name = {entry["name"]: entry for entry in entries}
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        raw_names = [member.filename for member in members]
        if len(set(raw_names)) != len(raw_names):
            raise EvidenceError("NPZ contains duplicate ZIP members")
        if any(not name.endswith(".npy") or name == ".npy" for name in raw_names):
            raise EvidenceError("NPZ must contain only named .npy array members")
        keys = [name[:-4] for name in raw_names]
        # NpzFile accepts both member paths and names without the .npy suffix;
        # keys such as x and x.npy would otherwise resolve to the same member.
        if set(keys) & set(raw_names):
            raise EvidenceError("NPZ contains ambiguous checkpoint/member name aliases")
        if len(set(keys)) != len(keys) or set(keys) != set(names):
            missing, extra = sorted(set(names) - set(keys)), sorted(set(keys) - set(names))
            first = next((name for name in names if name not in keys), None)
            raise EvidenceError(f"NPZ names must exactly match schema; missing={missing}, extra={extra}", first)
        if sum(member.file_size for member in members) > max_bytes:
            raise EvidenceError("NPZ expanded data exceeds max_bytes")
        if sum(math.prod(entry["shape"]) * np.dtype(entry["dtype"]).itemsize
               for entry in entries) > max_bytes:
            raise EvidenceError("schema logical data exceeds max_bytes")
        # Inspect headers before NumPy can allocate arrays. A tiny forged member
        # claiming a huge shape must not bypass the expanded ZIP size limit.
        for member in members:
            name = member.filename[:-4]
            entry = by_name[name]
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=10000)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=10000)
                else:
                    raise EvidenceError(f"unsupported NPY version: {version}", name)
                if list(shape) != entry["shape"] or dtype != np.dtype(entry["dtype"]):
                    raise EvidenceError(f"{name}: NPY header shape/dtype disagrees with schema", name)
                if dtype.hasobject or dtype.kind not in "biuf":
                    raise EvidenceError(f"{name}: NPY dtype is not safe numeric data", name)
                payload_bytes = math.prod(shape) * dtype.itemsize
                if stream.tell() + payload_bytes != member.file_size:
                    raise EvidenceError(f"{name}: NPY payload size disagrees with header", name)
    result = {}
    with np.load(path, allow_pickle=False, max_header_size=10000) as archive:
        for entry in entries:
            name = entry["name"]
            try:
                array = archive[name]
            except (ValueError, TypeError, EOFError) as exc:
                raise EvidenceError(f"cannot safely load checkpoint {name}: {exc}", name) from exc
            if list(array.shape) != entry["shape"]:
                raise EvidenceError(f"{name}: array shape {list(array.shape)} disagrees with schema {entry['shape']}", name)
            if array.dtype != np.dtype(entry["dtype"]):
                raise EvidenceError(f"{name}: array dtype {array.dtype} disagrees with schema {entry['dtype']}", name)
            if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
                diagnostic = tensor_diff(array, array)
                raise EvidenceError(f"{name}: nonfinite or unsupported array", name,
                                    dict(name=name, evidence_path=str(path.resolve()),
                                         worst_index=diagnostic["worst_index"],
                                         value=diagnostic["actual_value"],
                                         reason=diagnostic["reason"]))
            result[name] = array
    return result


def select_names(entries: list[dict], layers=(), checkpoints=()) -> list[str]:
    if len(set(layers)) != len(layers) or len(set(checkpoints)) != len(checkpoints):
        raise EvidenceError("duplicate layer/checkpoint selections are not allowed")
    known_layers = {entry["layer"] for entry in entries if "layer" in entry}
    if any(not _integer(layer) or layer < 0 or layer not in known_layers for layer in layers):
        raise EvidenceError(f"layer selection is outside schema layers {sorted(known_layers)}")
    known_checkpoints = {entry["name"] for entry in entries}
    known_checkpoints.update(entry["checkpoint"] for entry in entries if "checkpoint" in entry)
    if set(checkpoints) - known_checkpoints:
        raise EvidenceError(f"unknown checkpoint selections: {sorted(set(checkpoints) - known_checkpoints)}")
    names = [entry["name"] for entry in entries
             if (not layers or entry.get("layer") in layers)
             and (not checkpoints or entry["name"] in checkpoints
                  or entry.get("checkpoint") in checkpoints)]
    if not names:
        raise EvidenceError("layer/checkpoint selection has an empty intersection")
    return names


def compare_evidence(actual_path, reference_path, schema_path, *, reference_schema_path=None,
                     layers=(), checkpoints=(), atol=1e-5, max_bytes=DEFAULT_MAX_BYTES):
    # tensor_diff owns the W2 finite positive tolerance contract, even for bad inputs.
    tensor_diff(np.array([], dtype="float32"), np.array([], dtype="float32"), atol)
    entries = load_schema(Path(schema_path))
    reference_entries = load_schema(Path(reference_schema_path)) if reference_schema_path else entries
    if entries != reference_entries:
        raise EvidenceError("actual/reference schemas must match exactly in checkpoint order and metadata")
    names = select_names(entries, layers, checkpoints)
    # Always validate every member, including unselected evidence, before comparison.
    actual = load_arrays(Path(actual_path), entries, max_bytes)
    expected = load_arrays(Path(reference_path), reference_entries, max_bytes)
    comparison = compare_outputs({name: actual[name] for name in names},
                                 {name: expected[name] for name in names}, names, atol=atol)
    partial = bool(layers or checkpoints)
    passing = comparison["passed"]
    worst = max(comparison["checkpoints"],
                key=lambda row: (math.inf if row["reason"] == "absolute error overflow"
                                 else row["max_abs"] if row["max_abs"] is not None else -1),
                default=None)
    return dict(gate="w3_layer_diff", status="FAIL" if not passing else ("PARTIAL" if partial else "PASS"),
                passed=passing and not partial, selected_passed=passing, partial=partial,
                coverage=dict(total_checkpoints=len(entries), compared_checkpoints=len(names),
                              complete=not partial, selected_names=names,
                              evidence_validated_checkpoints=len(entries)),
                atol=float(atol), first_divergence=comparison["first_divergence"],
                worst_element=worst, checkpoints=comparison["checkpoints"],
                claim="Comparison covers recorded checkpoints only; no whole-model or backend acceptance is implied.")


def _file_evidence(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return dict(path=str(path.resolve()), size_bytes=path.stat().st_size, sha256=digest.hexdigest())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--reference-schema", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--layer", type=int, action="append", default=[])
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    args = parser.parse_args(argv)
    report = dict(gate="w3_layer_diff", status="FAIL", passed=False)
    try:
        # Existing evidence is never overwritten, including on report errors.
        args.out.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        print(f"FAIL: cannot create fresh output directory: {exc}", file=sys.stderr)
        return 1
    try:
        report.update(source_evidence())
        report["artifacts"] = {name: _file_evidence(path) for name, path in
                               (("actual", args.actual), ("reference", args.reference),
                                ("schema", args.schema), ("reference_schema", args.reference_schema))
                               if path is not None}
        report.update(compare_evidence(args.actual, args.reference, args.schema,
                                       reference_schema_path=args.reference_schema,
                                       layers=args.layer, checkpoints=args.checkpoint,
                                       atol=args.atol, max_bytes=args.max_bytes))
    except Exception as exc:
        report.update(status="FAIL", passed=False, error=f"{type(exc).__name__}: {exc}",
                      first_divergence=getattr(exc, "checkpoint", None),
                      worst_element=getattr(exc, "details", None))
    try:
        write_reports(args.out, report)
    except Exception as exc:
        print(f"FAIL: report publication failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"{report['status']}: {args.out / 'report.json'}")
    return 0 if report["passed"] else (2 if report["status"] == "PARTIAL" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
