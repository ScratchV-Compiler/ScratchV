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
from probes.numeric_summary import (  # noqa: E402
    METRIC_EXPLANATION, compact_metric_table, compact_metric_table_html,
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
    result = {"git_commit": None, "git_dirty": None}
    if not (ROOT / ".git").exists():
        # A source archive can be nested beneath another checkout. Do not let
        # Git's upward search attribute this source tree to that parent.
        result["git_reason"] = "Source directory has no .git metadata"
    else:
        try:
            result["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                stderr=subprocess.DEVNULL, timeout=10).strip()
            result["git_dirty"] = bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True,
                stderr=subprocess.DEVNULL, timeout=10).strip())
        except (OSError, subprocess.SubprocessError) as exc:
            result.update(git_commit=None, git_dirty=None,
                          git_reason=f"Checkout metadata unavailable: {type(exc).__name__}: {exc}")
    result["source_sha256"] = {
        str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [*sorted(Path(__file__).parent.glob("*.py")),
                     ROOT / "scratchv/frontend/onnx_parser.py",
                     ROOT / "scratchv/verification/ir_interpreter.py",
                     ROOT / "scratchv/verification/ir_numpy_ops.py",
                     ROOT / "scratchv/verification/numeric_metrics.py",
                     ROOT / "probes/numeric_summary.py"]}
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


def _table_views(headers, rows):
    """Render the same small, escaped table in both report formats."""
    def markdown_cell(value):
        return escape(str(value)).replace("|", "&#124;").replace("\n", "<br>")

    markdown = ["| " + " | ".join(map(markdown_cell, headers)) + " |",
                "|" + "---|" * len(headers)]
    markdown += ["| " + " | ".join(map(markdown_cell, row)) + " |" for row in rows]
    header = "".join(f"<th>{escape(str(value))}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row)
                   + "</tr>" for row in rows)
    return "\n".join(markdown), f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _details_views(title, markdown, html, *, opened=False):
    attribute = " open" if opened else ""
    heading = f"<details{attribute}><summary>{escape(title)}</summary>"
    return f"{heading}\n\n{markdown}\n\n</details>", f"{heading}{html}</details>"


def _page_html(title, content):
    css = ("body{font:16px/1.5 system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#182536}"
           "details{margin:16px 0;padding:12px;border:1px solid #dce3eb;overflow:auto}"
           "summary{cursor:pointer;font-weight:650}table{border-collapse:collapse;width:100%;font-size:14px}"
           "th,td{text-align:left;padding:8px;border:1px solid #dce3eb}"
           "pre{white-space:pre-wrap;overflow-wrap:anywhere}.failure{color:#a42020;font-weight:bold}")
    return ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{escape(title)}</title><style>{css}</style><body>{content}</body></html>")


def _host_report_views(report):
    cases = report.get("cases", [])
    comparisons = []
    for name, label in (("pytorch_vs_ort", "ORT / PyTorch（Transformer 导出）"),
                        ("ir_vs_ort", "IR 原生 NumPy / ORT")):
        rows = [case.get("ordinary_logits", {}).get(name, {}) for case in cases]
        # Missing cases or old metric fields must not look like full coverage.
        rows += [{}] * max(0, 7 - len(cases))
        comparisons.append((label, rows))
    description = ("官方 Qwen3 结构的两层缩小模型，固定随机权重，FP32、L=256、无 KV cache。"
                   "同一模型、同一权重和输入分别执行 PyTorch、ONNX Runtime（ORT）与 ScratchV IR 解释器。")
    scope = ("核心表仅汇总普通图的最终 logits，覆盖全部位置（含 padding）。"
             "比较名称按“被测 / 参考”排列；七个输入场景包括两份满长输入、1/17/255 token 边界、因果与补齐隔离检查。"
             "验收仍要求所有检查点及普通/诊断图一致性通过严格 MaxAbs < 1e-5（rtol=0）。")
    limitation = "本报告验证数值计算，不运行 C/RISC-V，也不评估随机权重模型的语言能力（PPL、zero-shot 未启用）。"
    status = "PASS" if report.get("passed") else "FAIL"
    title = "两层 Qwen3：Transformer 导出与 IR 精度"
    markdown = [f"# {title}", "", f"Result: **{status}**", "", description, "", scope, "",
                compact_metric_table(comparisons), "", METRIC_EXPLANATION, "", limitation]
    html = [f"<h1>{escape(title)}</h1><p><strong>Result: {status}</strong></p>",
            f"<p>{escape(description)}</p><p>{escape(scope)}</p>", compact_metric_table_html(comparisons),
            f"<p>{escape(METRIC_EXPLANATION)}</p><p>{escape(limitation)}</p>"]
    failures = []
    if report.get("error"):
        failures.append(f"阶段 {report.get('stage', 'unknown')}: {report['error']}")
    for case in cases:
        if case.get("passed") is not True:
            first = next((case.get(pair, {}).get("first_divergence") for pair in
                          ("pytorch_vs_ort", "ir_vs_ort") if case.get(pair, {}).get("first_divergence")), None)
            failures.append(f"{case.get('name', 'unknown')}: FAIL；"
                            + (f"首次超限检查点 {first}" if first else "有检查失败或尚未完成，请展开明细"))
    for check in report.get("invariants", []):
        if check.get("passed") is not True:
            failures.append(f"{check.get('name')} / {check.get('backend')}: 隔离检查 FAIL")
    if not report.get("passed") and not failures:
        failures.append(f"阶段 {report.get('stage', 'unknown')}: 尚未完成全部验收")
    if failures:
        markdown += ["", "**失败／未完成项**", "", *[f"- {escape(item)}" for item in failures]]
        html.append("<div class='failure'><p>失败／未完成项</p><ul>"
                    + "".join(f"<li>{escape(item)}</li>" for item in failures) + "</ul></div>")

    def append_detail(title, headers, rows, note="", *, opened=False):
        md, body = _table_views(headers, rows)
        if note:
            md += "\n\n" + note
            body += f"<p>{escape(note)}</p>"
        md, body = _details_views(title, md, body, opened=opened)
        markdown.extend(["", md])
        html.append(body)

    case_rows = []
    for case in cases:
        ordinary = case.get("ordinary_logits", {})
        case_rows.append([case.get("name"), case.get("valid_length"),
                          ordinary.get("pytorch_vs_ort", {}).get("max_abs", "未统计"),
                          ordinary.get("ir_vs_ort", {}).get("max_abs", "未统计"),
                          "PASS" if case.get("passed") else "FAIL"])
    append_detail("输入场景明细", ["场景", "有效 token", "ORT / PyTorch MaxAbs", "IR / ORT MaxAbs", "全部检查"], case_rows)
    checkpoint_rows = []
    for case in cases:
        for pair in ("pytorch_vs_ort", "ir_vs_ort"):
            for row in case.get(pair, {}).get("checkpoints", []):
                checkpoint_rows.append([case.get("name"), pair, row.get("name"), row.get("max_abs"),
                                        "PASS" if row.get("passed") else row.get("reason") or "FAIL"])
        for name, row in case.get("ordinary_logits", {}).items():
            if "diagnostic_vs_ordinary" in name:
                checkpoint_rows.append([case.get("name"), name, "logits", row.get("max_abs"),
                                        "PASS" if row.get("passed") else row.get("reason") or "FAIL"])
    append_detail("检查点与普通/诊断图一致性", ["场景", "比较", "检查点", "MaxAbs", "结果"], checkpoint_rows,
                  "normal 为只输出最终 logits 的普通图；diagnostic 额外输出中间检查点以定位误差。两种图都参与验收。")
    append_detail("因果与补齐隔离检查", ["检查", "执行路径", "MaxAbs", "结果"],
                  [[row.get("name"), row.get("backend"), row.get("max_abs"),
                    "PASS" if row.get("passed") else "FAIL"] for row in report.get("invariants", [])])
    footer = "完整诊断、位置明细、命令、环境与源码指纹保存在 report.json；原始数组随 CI artifact 提供。"
    markdown.extend(["", footer])
    html.append(f"<p>{escape(footer)}</p>")
    return "\n".join(markdown) + "\n", _page_html(title, "".join(html))


def _report_markdown(report: dict) -> str:
    return _host_report_views(report)[0]


def _report_html(report: dict) -> str:
    return _host_report_views(report)[1]


def write_reports(out: Path, report: dict) -> None:
    """Publish complete JSON last; losing required evidence fails the gate."""
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
        # Reporting must not replace the original export/numeric error.
        if report["passed"]:
            report.update(passed=False, stage="report-write",
                          error="Incomplete two-layer IR gate evidence")
        report["report_write_errors"] = list(failures)

    def recover_views():
        summary = ("# Real two-layer Qwen3 numerical probe\n\nResult: **FAIL**\n\n"
                   f"Stage: {report['stage']}\n\nError: {report.get('error', 'Gate failed')}\n\n"
                   + "Report write errors:\n" + "\n".join(f"- {item}" for item in failures) + "\n")
        write("report.md", summary)
        write("report.html", '<!doctype html><meta charset="utf-8"><pre>'
              + escape(summary) + "</pre>")
        report["report_write_errors"] = list(failures)

    write("report.md", _report_markdown(report))
    write("report.html", _report_html(report))
    if failures:
        fail()
        recover_views()
    if not write("report.json", json.dumps(report, indent=2, allow_nan=False) + "\n"):
        fail()
        recover_views()
        # Retry only as FAIL. A permanent write/publish failure exposes no
        # final JSON, including when a failed close left complete temp bytes.
        write("report.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
        report["report_write_errors"] = list(failures)
    if failures:
        print("[gate] FAIL; report write errors: " + "; ".join(failures), file=sys.stderr)


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
