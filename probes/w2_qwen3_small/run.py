#!/usr/bin/env python3
"""Reproducible official Qwen3 two-layer PyTorch -> ONNX -> ScratchV probe.

Random weights test architecture and numerical semantics, not language quality.
Both the ordinary logits-only model and a diagnostic copy run through the real
ScratchV parser/interpreter. A failed comparison or execution is a failing gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from html import escape
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probes.w2_qwen3_small.diagnostics import (  # noqa: E402
    build_diagnostic_model, compare_outputs, tensor_diff, unpack_trace,
)

ATOL = 1e-5
SEQ = 256
VOCAB = 128
PINNED = {
    "torch": "2.7.1", "transformers": "4.51.3", "onnx": "1.18.0",
    "onnxruntime": "1.22.1", "numpy": "2.2.6", "onnxscript": "0.2.7",
    "onnx-ir": "0.1.7", "huggingface-hub": "0.30.2",
    "safetensors": "0.5.3", "protobuf": "5.29.5",
}


def environment() -> dict:
    versions = {name: importlib.metadata.version(name) for name in PINNED}
    mismatches = {name: value for name, value in versions.items()
                  if value.split("+")[0] != PINNED[name]}
    if sys.version_info[:2] != (3, 12) or mismatches:
        raise RuntimeError(f"Use Python 3.12 and requirements/qwen3-small-probe.txt; "
                           f"version mismatches: {mismatches}")
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "device": "cpu"}


def arrays_sha256(arrays: dict[str, np.ndarray]) -> str:
    """Hash data and schema, independently of NPZ container timestamps."""
    digest = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        digest.update(json.dumps([name, array.dtype.str, list(array.shape)]).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def make_inputs(seed: int, valid_length: int) -> dict[str, np.ndarray]:
    if not 1 <= valid_length <= SEQ:
        raise ValueError("valid_length must be in [1, 256]")
    ids = np.zeros((1, SEQ), dtype=np.int64)
    ids[:, :valid_length] = np.random.default_rng(seed).integers(
        1, VOCAB, (1, valid_length), dtype=np.int64)
    allowed = np.arange(SEQ)[None, :] <= np.arange(SEQ)[:, None]
    allowed &= np.arange(SEQ)[None, :] < valid_length
    mask = np.where(allowed, np.float32(0), np.finfo(np.float32).min)
    return {"input_ids": ids, "attention_mask": mask.reshape(1, 1, SEQ, SEQ)}


def input_cases() -> list[tuple[str, int, dict[str, np.ndarray]]]:
    cases = [("full_seed_0", SEQ, make_inputs(0, SEQ)),
             ("full_seed_42", SEQ, make_inputs(42, SEQ)),
             ("one_token", 1, make_inputs(7, 1)),
             ("short_17", 17, make_inputs(7, 17)),
             ("short_255", 255, make_inputs(7, 255))]
    causal = {name: array.copy() for name, array in cases[0][2].items()}
    causal["input_ids"][:, 64:] = (causal["input_ids"][:, 64:] % (VOCAB - 1)) + 1
    cases.append(("changed_future", SEQ, causal))
    padded = {name: array.copy() for name, array in cases[3][2].items()}
    padded["input_ids"][:, 17:] = np.random.default_rng(99).integers(
        1, VOCAB, (1, SEQ - 17), dtype=np.int64)
    cases.append(("changed_padding", 17, padded))
    return cases


def split_positions(array: np.ndarray, axis: int, start: int, stop: int):
    slices = [slice(None)] * array.ndim
    slices[axis] = slice(start, stop)
    return array[tuple(slices)]


def position_comparison(actual, expected, names, metadata, valid_length):
    result = compare_outputs(actual, expected, names, atol=ATOL)
    for row in result["checkpoints"]:
        name = row["name"]
        if not row["shape_matches"]:
            continue
        axis = metadata[name]["sequence_axis"]
        row["valid_tokens"] = tensor_diff(
            split_positions(actual[name], axis, 0, valid_length),
            split_positions(expected[name], axis, 0, valid_length), atol=ATOL)
        if valid_length < SEQ:
            row["padding_queries"] = tensor_diff(
                split_positions(actual[name], axis, valid_length, SEQ),
                split_positions(expected[name], axis, valid_length, SEQ), atol=ATOL)
    return result


def attention_checks(reference, feed, config) -> list[dict]:
    """Independent NumPy repeat/attention checks catch GQA head-order errors."""
    checks = []
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    width = config["head_dim"]
    for layer in range(config["num_hidden_layers"]):
        prefix = f"layer_{layer}."
        q, k = reference[prefix + "rope_q"], reference[prefix + "rope_k"]
        v = reference[prefix + "v_proj"].reshape(1, SEQ, kv_heads, width)
        v = np.repeat(v.transpose(0, 2, 1, 3), heads // kv_heads, axis=1)
        k = np.repeat(k, heads // kv_heads, axis=1)
        scores = np.matmul(q, k.swapaxes(-1, -2)) * np.float32(width ** -0.5)
        scores += feed["attention_mask"]
        probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        context = np.matmul(probabilities, v).transpose(0, 2, 1, 3)
        context = context.reshape(1, SEQ, heads * width)
        for suffix, actual, expected in [
            ("gqa_probabilities", reference[prefix + "attn_probs"], probabilities),
            ("gqa_context", reference[prefix + "attn_context"], context),
        ]:
            checks.append({"name": prefix + suffix, **tensor_diff(actual, expected, ATOL)})
        blocked = feed["attention_mask"][0, 0] < 0
        leaked = reference[prefix + "attn_probs"][..., blocked]
        checks.append({"name": prefix + "blocked_attention",
                       **tensor_diff(leaked, np.zeros_like(leaked), ATOL)})
    return checks


def provenance() -> dict:
    result = {}
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        result["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
    result["source_sha256"] = {
        str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [*sorted(Path(__file__).parent.glob("*.py")),
                     ROOT / "scratchv/frontend/onnx_parser.py",
                     ROOT / "scratchv/verification/ir_interpreter.py",
                     ROOT / "scratchv/verification/ir_numpy_ops.py"]}
    return result


def run_probe(out: Path, report: dict, model_seed: int = 0) -> None:
    report["stage"] = "environment"
    report["environment"] = environment()
    import onnx
    import onnxruntime as ort
    import torch
    from probes.w2_qwen3_small.model import build_model, export_onnx
    from scratchv.frontend.onnx_parser import ONNXParser
    from scratchv.analysis.ir_verifier import verify_ir
    from scratchv.verification.ir_interpreter import IRInterpreter

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    report["stage"] = "model"
    wrapper = build_model(seed=model_seed)
    names = list(wrapper.output_names)
    metadata = wrapper.checkpoint_metadata
    config = wrapper.config_metadata
    report["config"] = config
    report["model_seed"] = model_seed
    report["provenance"] = provenance()
    report["weights_sha256"] = arrays_sha256({
        name: tensor.detach().cpu().numpy() for name, tensor in wrapper.model.state_dict().items()})
    (out / "config.json").write_text(wrapper.model.config.to_json_string(use_diff=False), encoding="utf-8")
    cases = input_cases()

    def reference(feed):
        args = [torch.from_numpy(feed[name]) for name in ("input_ids", "attention_mask")]
        with torch.inference_mode(), wrapper.capture():
            tensors = wrapper(*args)
        if len(tensors) != len(names):
            raise ValueError("PyTorch checkpoint count changed")
        return {name: tensor.detach().cpu().numpy().copy()
                for name, tensor in zip(names, tensors)}

    first_reference = reference(cases[0][2])
    report["stage"] = "export"
    tensors = [torch.from_numpy(cases[0][2][name]) for name in ("input_ids", "attention_mask")]
    report["export"] = export_onnx(wrapper, out / "checkpoints.onnx", *tensors)
    exported = onnx.load(out / "checkpoints.onnx")
    onnx.checker.check_model(exported, full_check=True)
    ordinary = onnx.ModelProto()
    ordinary.CopyFrom(exported)
    logits_info = next(output for output in ordinary.graph.output if output.name == "logits")
    del ordinary.graph.output[:]
    ordinary.graph.output.append(logits_info)
    onnx.save(ordinary, out / "model.onnx")
    diagnostic, schema = build_diagnostic_model(exported, first_reference)
    onnx.save(diagnostic, out / "diagnostics.onnx")
    report["checkpoints"] = schema
    report["checkpoint_metadata"] = metadata
    (out / "checkpoints.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    report["onnx"] = {"opset": {o.domain: o.version for o in ordinary.opset_import},
                      "nodes": len(ordinary.graph.node),
                      "operators": dict(sorted(Counter(n.op_type for n in ordinary.graph.node).items())),
                      "model_sha256": hashlib.sha256((out / "model.onnx").read_bytes()).hexdigest(),
                      "diagnostics_sha256": hashlib.sha256((out / "diagnostics.onnx").read_bytes()).hexdigest()}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    normal_session = ort.InferenceSession(str(out / "model.onnx"), options,
                                         providers=["CPUExecutionProvider"])
    diagnostic_session = ort.InferenceSession(str(out / "diagnostics.onnx"), options,
                                             providers=["CPUExecutionProvider"])
    report["stage"] = "parse"
    parsers, programs = {}, {}
    for kind, filename in [("normal", "model.onnx"), ("diagnostic", "diagnostics.onnx")]:
        parsers[kind] = ONNXParser()
        programs[kind] = parsers[kind].parse(str(out / filename))
        valid, errors = verify_ir(programs[kind])
        if not valid:
            raise ValueError(f"{kind} IR verification failed: {errors}")
    report["stage"] = "numeric"
    report["cases"] = []
    cached_logits = {}
    for case_name, valid_length, feed in cases:
        report["current_case"] = case_name
        started = time.perf_counter()
        case = {"name": case_name, "valid_length": valid_length,
                "input_sha256": arrays_sha256(feed), "passed": False}
        report["cases"].append(case)
        np.savez(out / f"inputs_{case_name}.npz", **feed)
        pt = first_reference if case_name == cases[0][0] else reference(feed)
        with torch.inference_mode():
            plain_logits = wrapper.model(
                input_ids=torch.from_numpy(feed["input_ids"]),
                attention_mask=torch.from_numpy(feed["attention_mask"]),
                position_ids=torch.arange(SEQ).reshape(1, SEQ), use_cache=False,
                return_dict=False, logits_to_keep=0,
            )[0].numpy()
        case["capture_preserves_logits"] = tensor_diff(pt["logits"], plain_logits, ATOL)
        ort_normal = normal_session.run(["logits"], feed)[0]
        ort_values = diagnostic_session.run(None, feed)
        if len(ort_values) != len(names) + 1:
            raise ValueError("Unexpected diagnostic ONNX output count")
        ort_named = dict(zip([o.name for o in diagnostic_session.get_outputs()][1:], ort_values[1:]))
        ort_packed = unpack_trace(ort_values[0], schema)
        case["ort_pack_layout"] = compare_outputs(ort_packed, ort_named, names, ATOL)
        case["pytorch_vs_ort"] = position_comparison(ort_named, pt, names, metadata, valid_length)
        result = IRInterpreter(programs["diagnostic"]).run(
            feed, initializers=parsers["diagnostic"].initializers)
        ir_named = unpack_trace(result.return_value, schema)
        case["ir_vs_ort"] = position_comparison(ir_named, ort_named, names, metadata, valid_length)
        ordinary_result = IRInterpreter(programs["normal"]).run(
            feed, initializers=parsers["normal"].initializers)
        ir_normal = ordinary_result.return_value
        case["ordinary_logits"] = {
            "pytorch_vs_ort": tensor_diff(ort_normal, pt["logits"], ATOL),
            "ir_vs_ort": tensor_diff(ir_normal, ort_normal, ATOL),
            "ort_diagnostic_vs_ordinary": tensor_diff(ort_named["logits"], ort_normal, ATOL),
            "ir_diagnostic_vs_ordinary": tensor_diff(ir_named["logits"], ir_normal, ATOL),
        }
        case["attention_checks"] = attention_checks(pt, feed, config)
        case["executed_steps"] = {"normal": ordinary_result.executed_steps,
                                  "diagnostic": result.executed_steps}
        case["passed"] = all([
            case["capture_preserves_logits"]["passed"], case["ort_pack_layout"]["passed"],
            case["pytorch_vs_ort"]["passed"], case["ir_vs_ort"]["passed"],
            *[d["passed"] for d in case["ordinary_logits"].values()],
            *[d["passed"] for d in case["attention_checks"]],
        ])
        case["seconds"] = time.perf_counter() - started
        cached_logits[case_name] = {"pytorch": pt["logits"], "ort": ort_normal, "ir": ir_normal}
        for backend, logits in cached_logits[case_name].items():
            np.save(out / f"logits_{case_name}_{backend}.npy", logits)
        print(f"[{case_name}] {'PASS' if case['passed'] else 'FAIL'} "
              f"first IR divergence={case['ir_vs_ort']['first_divergence']}", flush=True)
    report["invariants"] = []
    for name, left, right, length in [
        ("causality", "full_seed_0", "changed_future", 64),
        ("padding_isolation", "short_17", "changed_padding", 17),
    ]:
        for backend in ("pytorch", "ort", "ir"):
            report["invariants"].append({"name": name, "backend": backend,
                **tensor_diff(cached_logits[left][backend][:, :length],
                              cached_logits[right][backend][:, :length], ATOL)})
    report["passed"] = (all(case["passed"] for case in report["cases"])
                        and all(check["passed"] for check in report["invariants"]))
    report["stage"] = "complete"
    report.pop("current_case", None)


def write_reports(out: Path, report: dict) -> None:
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                     encoding="utf-8")
    lines = ["# Real two-layer Qwen3 numerical probe", "",
             f"Result: **{'PASS' if report['passed'] else 'FAIL'}**", "",
             "Official transformers 4.51.3 Qwen3, seeded random weights, FP32, L=256, no KV cache.",
             "All checkpoint positions must pass max absolute error < 1e-5 (rtol=0).",
             "Padding queries are also checked and are reported separately in JSON.",
             "Diagnostic outputs observe the same forward; ordinary logits-only IR is checked separately.",
             "This probe does not validate pretrained language quality or RISC-V execution.", ""]
    if report.get("error"):
        lines += [f"Failed stage: `{report['stage']}`", "", "```text", report["error"], "```", ""]
    lines += ["| Case | Valid tokens | PyTorch / ORT max error | IR / ORT max error | First divergence | Result |",
              "|---|---:|---:|---:|---|---|"]
    for case in report.get("cases", []):
        pairs = [case.get(name, {}) for name in ("pytorch_vs_ort", "ir_vs_ort")]
        maxima = [max((row["max_abs"] for row in pair.get("checkpoints", [])
                       if row.get("max_abs") is not None), default=None) for pair in pairs]
        values = [f"{value:.3e}" if value is not None else "n/a" for value in maxima]
        first = next((pair.get("first_divergence") for pair in pairs if pair.get("first_divergence")), "none")
        lines.append(f"| {case['name']} | {case['valid_length']} | {values[0]} | {values[1]} | "
                     f"{first} | {'PASS' if case['passed'] else 'FAIL'} |")
    for case in report.get("cases", []):
        lines += ["", f"<details><summary>{case['name']}: checkpoint comparisons</summary>", "",
                  "| Comparison | Checkpoint | Max absolute error | Result |", "|---|---|---:|---|"]
        for pair in ("pytorch_vs_ort", "ir_vs_ort"):
            for row in case.get(pair, {}).get("checkpoints", []):
                lines.append(f"| {pair} | {row['name']} | {row.get('max_abs')} | "
                             f"{'PASS' if row['passed'] else 'FAIL'} |")
        lines += ["", "</details>"]
    lines += ["", "| Invariant | Backend | Max absolute error | Result |", "|---|---|---:|---|"]
    for check in report.get("invariants", []):
        lines.append(f"| {check['name']} | {check['backend']} | {check.get('max_abs')} | "
                     f"{'PASS' if check['passed'] else 'FAIL'} |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_html(out, report)


def write_html(out: Path, report: dict) -> None:
    """A portable, dependency-free view of the same numeric evidence."""
    def cell(value):
        return escape(str(value))

    def table(rows, columns):
        header = "".join(f"<th>{cell(column)}</th>" for column in columns)
        body = "".join("<tr>" + "".join(f"<td>{cell(value)}</td>" for value in row) + "</tr>"
                       for row in rows)
        return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"

    status = "PASS" if report["passed"] else "FAIL"
    sections = [f"<h1>Real two-layer Qwen3 probe</h1><p class='{status.lower()}'>{status}</p>",
                "<p>FP32 · L=256 · 29 checkpoints · max absolute error &lt; 1e-5</p>",
                "<p>Official Qwen3 with random weights. All positions, including padding queries, are checked. "
                "Ordinary logits and diagnostic graphs must both pass. This is not a RISC-V or pretrained quality test.</p>"]
    if report.get("error"):
        sections.append(f"<pre>{cell(report['stage'])}: {cell(report['error'])}</pre>")
    for case in report.get("cases", []):
        case_status = "PASS" if case["passed"] else "FAIL"
        details = []
        for pair in ("pytorch_vs_ort", "ir_vs_ort"):
            comparison = case.get(pair, {})
            details.append(f"<h3>{cell(pair)}</h3><p>First divergence: "
                           f"{cell(comparison.get('first_divergence') or 'none')}</p>")
            rows = [[row["name"], row.get("max_abs"), row.get("worst_index"),
                     row.get("actual_value"), row.get("expected_value"),
                     "PASS" if row["passed"] else row.get("reason")]
                    for row in comparison.get("checkpoints", [])]
            details.append(table(rows, ["Checkpoint", "Max abs", "Worst index", "Actual", "Reference", "Result"]))
        auxiliary = {"capture_preserves_logits": case.get("capture_preserves_logits"),
                     "ordinary_logits": case.get("ordinary_logits"),
                     "attention_checks": case.get("attention_checks")}
        details.append(f"<h3>Other checks</h3><pre>{cell(json.dumps(auxiliary, indent=2))}</pre>")
        opened = " open" if not case["passed"] else ""
        sections.append(f"<details{opened}><summary>{cell(case['name'])} · "
                        f"{case['valid_length']} valid tokens · {case_status}</summary>{''.join(details)}</details>")
    rows = [[check["name"], check["backend"], check.get("max_abs"),
             "PASS" if check["passed"] else "FAIL"] for check in report.get("invariants", [])]
    sections.append("<h2>Metamorphic checks</h2>" + table(rows, ["Invariant", "Backend", "Max abs", "Result"]))
    sections.append("<p>Full environment, schemas, hashes and valid/padding breakdowns are in report.json.</p>")
    css = ("body{font:16px/1.5 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#182536;background:#f7f9fc}"
           "h1{font-size:30px}h3{font-size:18px}details{background:white;margin:16px 0;padding:16px;border:1px solid #dce3eb;border-radius:8px;overflow:auto}"
           "summary{cursor:pointer;font-weight:650}table{border-collapse:collapse;width:100%;font-size:13px;background:white}"
           "th,td{text-align:left;padding:8px;border-bottom:1px solid #dce3eb}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}"
           ".pass{color:#126538;font-weight:bold}.fail{color:#a42020;font-weight:bold}")
    document = ("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
                f"<title>Qwen3 small probe · {status}</title><style>{css}</style><body>{''.join(sections)}</body></html>")
    (out / "report.html").write_text(document, encoding="utf-8")


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", type=Path, default=ROOT / "output/qwen3-small")
    cli.add_argument("--model-seed", type=int, default=0)
    args = cli.parse_args(argv)
    out = args.output_dir.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        cli.error("--output-dir must be empty; choose a fresh directory to preserve prior evidence")
    out.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "stage": "initialization", "atol": ATOL,
              "rtol": 0, "acceptance": "all positions including padding queries"}
    started = time.perf_counter()
    try:
        run_probe(out, report, args.model_seed)
    except Exception as exc:
        report["passed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    report["seconds"] = time.perf_counter() - started
    write_reports(out, report)
    print(f"[gate] {'PASS' if report['passed'] else 'FAIL'}; report: {out / 'report.json'}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
