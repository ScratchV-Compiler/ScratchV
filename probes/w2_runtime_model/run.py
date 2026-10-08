#!/usr/bin/env python3
"""Text -> official tokenizer -> static Qwen3 -> shared IR -> one greedy ID.

This is an interface integration probe with seeded random weights, not a
pretrained 0.6B model or a generation loop. The complete vocabulary is retained.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
import html
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probes.w2_runtime.run import MANIFEST, checkout_evidence, require_environment, sha256, verify_assets
from probes.w2_qwen3_small.run import arrays_sha256, environment
from scratchv.runtime.llm_inputs import last_valid_logits, greedy_token, prepare_inputs

SEQ, VOCAB, ATOL = 256, 151936, 1e-5
LEVELS = ("none", "all")
CASES = (("chinese", "你好，ScratchV！"),
         ("special_pad_in_prompt", "<|im_start|>user\nTest<|endoftext|>!<|im_end|>\n"))


def checked_inputs(ids, prepared, pad_token_id):
    """Independent ABI oracle: never infer valid length from pad-ID values."""
    length = len(ids)
    if prepared.valid_length != length or not 1 <= length <= SEQ:
        raise ValueError("Input preparation changed the prompt length")
    actual = prepared.as_feed()
    expected_ids = np.full((1, SEQ), pad_token_id, dtype=np.int64)
    expected_ids[0, :length] = ids
    expected_mask = np.full((1, 1, SEQ, SEQ), np.finfo(np.float32).min, dtype=np.float32)
    for query in range(SEQ):
        for key in range(min(query + 1, length)):
            expected_mask[0, 0, query, key] = 0
    for name, expected in (("input_ids", expected_ids), ("attention_mask", expected_mask)):
        value = actual[name]
        if value.shape != expected.shape or value.dtype != expected.dtype or not np.array_equal(value, expected):
            raise ValueError(f"Input preparation changed {name} ABI, IDs, causal mask or padding")
    return actual


def compare_logits(actual, expected, valid_length, *, vocab_size=VOCAB):
    """Compare every element in bounded FP64 chunks, including padding rows."""
    shape = (1, SEQ, vocab_size)
    for name, array in (("actual", actual), ("expected", expected)):
        if not isinstance(array, np.ndarray) or np.ma.isMaskedArray(array):
            raise TypeError(f"{name} logits must be an ordinary ndarray")
        if array.shape != shape or array.dtype != np.dtype(np.float32):
            raise ValueError(f"{name} logits require FP32 {shape}; got {array.dtype} {array.shape}")
    if not 1 <= valid_length <= SEQ:
        raise ValueError("Invalid prompt length")
    maxima = {"max_abs": 0.0, "valid_max_abs": 0.0, "padding_max_abs": 0.0}
    worst = None
    for start in range(0, SEQ, 8):
        stop = min(start + 8, SEQ)
        left, right = actual[0, start:stop], expected[0, start:stop]
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f"Nonfinite logits in positions {start}:{stop}")
        difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
        maximum = float(difference.max())
        if maximum > maxima["max_abs"]:
            position, token = np.unravel_index(int(difference.argmax()), difference.shape)
            worst = [0, start + int(position), int(token)]
        maxima["max_abs"] = max(maxima["max_abs"], maximum)
        boundary = min(max(valid_length - start, 0), stop - start)
        if boundary:
            maxima["valid_max_abs"] = max(maxima["valid_max_abs"], float(difference[:boundary].max()))
        if boundary < stop - start:
            maxima["padding_max_abs"] = max(maxima["padding_max_abs"], float(difference[boundary:].max()))
    return {**maxima, "worst_index": worst, "elements": int(actual.size),
            "passed": maxima["max_abs"] < ATOL, "atol": ATOL, "rtol": 0}


def sample_checked(logits, reference, length, adapter, references, *, vocab_size=VOCAB):
    vector = last_valid_logits(logits, length, vocab_size=vocab_size)
    expected = reference[0, length - 1, :]
    # Independently index the reference to catch an incorrect host row selector.
    if vector.shape != expected.shape or not np.array_equal(vector, logits[0, length - 1, :]):
        raise ValueError("Host selected the wrong logits row")
    actual_id, expected_id = greedy_token(vector), int(np.argmax(expected))
    if actual_id != expected_id:
        raise ValueError(f"Greedy ID differs: actual={actual_id}, reference={expected_id}")
    adapter.validate_token_ids([actual_id])  # Never hide invalid padded model rows.
    text = adapter.decode([actual_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    for tokenizer in references:
        if text != tokenizer.decode([actual_id], skip_special_tokens=False, clean_up_tokenization_spaces=False):
            raise ValueError("Sampled token decode differs from official tokenizer")
    ordered = np.partition(expected, -2)[-2:]
    return {"id": actual_id, "decoded": text, "position": length - 1,
            "reference_top2_margin": float(ordered[-1] - ordered[-2]), "passed": True}


def build_model(adapter):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from probes.w2_qwen3_small.model import MODEL_CONFIG

    config_values = {**MODEL_CONFIG, "vocab_size": VOCAB,
                     "pad_token_id": adapter.pad_token_id,
                     "bos_token_id": None, "eos_token_id": list(adapter.generation_eos_token_ids)}
    config = Qwen3Config(**config_values)
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(config).float().eval()
    if model.lm_head.weight is not model.model.embed_tokens.weight:
        raise ValueError("Expected official tied embedding and LM head")

    class NormalModel(torch.nn.Module):
        def __init__(self, qwen):
            super().__init__()
            self.model = qwen
            self.register_buffer("positions", torch.arange(SEQ).reshape(1, SEQ), persistent=False)

        def forward(self, input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask,
                              position_ids=self.positions, use_cache=False,
                              output_attentions=False, output_hidden_states=False,
                              return_dict=False, logits_to_keep=0)[0]

    return NormalModel(model).eval(), config_values


def check_onnx_contract(model):
    import onnx
    onnx.checker.check_model(model, full_check=True)
    standard_opsets = [item.version for item in model.opset_import if item.domain in ("", "ai.onnx")]
    if standard_opsets != [18]:
        raise ValueError(f"Expected exactly one standard ONNX opset 18: {standard_opsets}")
    expected = {"input_ids": (onnx.TensorProto.INT64, [1, SEQ]),
                "attention_mask": (onnx.TensorProto.FLOAT, [1, 1, SEQ, SEQ]),
                "logits": (onnx.TensorProto.FLOAT, [1, SEQ, VOCAB])}
    if [v.name for v in model.graph.input] != ["input_ids", "attention_mask"] or [v.name for v in model.graph.output] != ["logits"]:
        raise ValueError("ONNX must have two ordinary inputs and only logits output")
    for value in [*model.graph.input, *model.graph.output]:
        element = value.type.tensor_type
        actual = element.elem_type, [d.dim_value for d in element.shape.dim]
        if actual != expected[value.name]:
            raise ValueError(f"Unexpected static ONNX ABI: {value.name}: {actual}")
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer):
        raise ValueError("This small model must embed its weights")


def source_evidence():
    paths = [Path(__file__), ROOT / "probes/w2_runtime/run.py",
             ROOT / "probes/w2_qwen3_small/model.py", ROOT / "probes/w2_qwen3_small/run.py",
             ROOT / "probes/w1_qwen3_export/manifest.json",
             ROOT / "scratchv/runtime/qwen3_tokenizer.py", ROOT / "scratchv/runtime/llm_inputs.py",
             ROOT / "scratchv/frontend/onnx_parser.py", ROOT / "scratchv/verification/ir_interpreter.py",
             ROOT / "scratchv/verification/ir_numpy_ops.py", ROOT / "scratchv/pass_manager.py",
             ROOT / "scratchv/pass_interface.py", ROOT / "scratchv/ir/types.py", ROOT / "scratchv/ir/builder.py",
             ROOT / "requirements/qwen3-small-probe.txt",
             *sorted((ROOT / "scratchv/optimizer").glob("*.py")),
             *sorted((ROOT / "scratchv/analysis").glob("*.py"))]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in paths}


def run_probe(directory, out, report):
    report.update(stage="environment", environment=environment(), tokenizer_environment=require_environment())
    import onnx
    import onnxruntime as ort
    import torch
    from transformers import AutoTokenizer
    from scratchv.analysis.ir_verifier import verify_ir
    from scratchv.frontend.onnx_parser import ONNXParser
    from scratchv.pass_manager import create_optimization_pass_manager
    from scratchv.runtime.qwen3_tokenizer import Qwen3Tokenizer
    from scratchv.verification.ir_interpreter import IRInterpreter

    def require_valid(program):
        valid, errors = verify_ir(program)
        if not valid:
            raise ValueError(f"Invalid shared IR: {errors}")

    from scratchv.runtime import llm_inputs
    modules = {"onnx_parser": sys.modules[ONNXParser.__module__],
               "ir_interpreter": sys.modules[IRInterpreter.__module__],
               "llm_inputs": llm_inputs,
               "qwen3_tokenizer": sys.modules[Qwen3Tokenizer.__module__]}
    report["module_files"] = {name: str(Path(module.__file__).resolve()) for name, module in modules.items()}
    for name, path in report["module_files"].items():
        if not Path(path).is_relative_to(ROOT):
            raise ValueError(f"{name} imported outside the current source tree: {path}")
    report.update(stage="tokenizer", assets=verify_assets(directory))
    adapter = Qwen3Tokenizer.from_directory(directory)
    if adapter.model_vocab_size != VOCAB:
        raise ValueError("The complete official model vocabulary is mandatory")
    references = [AutoTokenizer.from_pretrained(directory, use_fast=fast, local_files_only=True,
                                               trust_remote_code=False) for fast in (True, False)]
    if [type(t).__name__ for t in references] != ["Qwen2TokenizerFast", "Qwen2Tokenizer"]:
        raise ValueError("Require two distinct official tokenizer references")
    report["cases"] = []
    feeds = []
    for name, text in CASES:
        ids = adapter.encode(text)
        if any(ids != reference.encode(text) for reference in references):
            raise ValueError(f"Official prompt encoding differs: {name}")
        prepared = prepare_inputs(ids, pad_token_id=adapter.pad_token_id)
        feed = checked_inputs(ids, prepared, adapter.pad_token_id)
        np.savez(out / f"{name}.inputs.npz", **feed)
        feeds.append((prepared.valid_length, feed))
        report["cases"].append({"name": name, "text": text, "token_ids": ids,
                                "valid_length": prepared.valid_length, "input_sha256": arrays_sha256(feed),
                                "pad_id_inside_prompt": adapter.pad_token_id in ids,
                                "passed": False, "executions": []})
    if len(feeds) != 2 or not report["cases"][1]["pad_id_inside_prompt"]:
        raise ValueError("Require two original-text cases including an in-prompt pad ID")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    report["stage"] = "model"
    started = time.perf_counter()
    wrapper, config = build_model(adapter)
    report.update(config=config, model_seed=0, model_seconds=time.perf_counter() - started,
                  model_class=type(wrapper.model).__module__ + "." + type(wrapper.model).__name__,
                  weights="seed 0 random initialization; no pretrained weights",
                  weights_sha256=arrays_sha256({name: value.detach().numpy()
                                               for name, value in wrapper.model.state_dict().items()}))
    (out / "config.json").write_text(wrapper.model.config.to_json_string(use_diff=False), encoding="utf-8")
    # Persist references and release their complete tensors before ORT/IR runs.
    for case, (length, feed) in zip(report["cases"], feeds):
        report.update(stage="torch", current_case=case["name"])
        started = time.perf_counter()
        with torch.inference_mode():
            value = wrapper(*(torch.from_numpy(feed[n]) for n in ("input_ids", "attention_mask"))).numpy()
        case["torch_seconds"] = time.perf_counter() - started
        check = compare_logits(value, value, length)
        case["torch"] = {"contract": check,
                         "sampling": sample_checked(value, value, length, adapter, references)}
        np.save(out / f"{case['name']}.torch.npy", value)
        del value
    gc.collect()
    report["stage"] = "export"
    started = time.perf_counter()
    with torch.inference_mode():
        exported = torch.onnx.export(wrapper, tuple(torch.from_numpy(feeds[0][1][n]) for n in ("input_ids", "attention_mask")),
                                     dynamo=True, opset_version=18, input_names=["input_ids", "attention_mask"],
                                     output_names=["logits"], dynamic_shapes=None, optimize=False, fallback=False)
        exported.save(str(out / "model.onnx"), external_data=False)
    report["export_seconds"] = time.perf_counter() - started
    del exported, wrapper
    gc.collect()
    exported = onnx.load(out / "model.onnx")
    check_onnx_contract(exported)
    report["onnx"] = {"sha256": sha256(out / "model.onnx"), "bytes": (out / "model.onnx").stat().st_size,
                      "nodes": len(exported.graph.node), "opset": 18,
                      "exporter": "torch.onnx.export(dynamo=True, optimize=False, fallback=False)",
                      "inputs": {"input_ids": "INT64[1,256]", "attention_mask": "FP32[1,1,256,256]"},
                      "output": "FP32[1,256,151936]", "output_bytes": SEQ * VOCAB * 4}
    del exported
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    report["stage"] = "ort"
    started = time.perf_counter()
    session = ort.InferenceSession(str(out / "model.onnx"), options, providers=["CPUExecutionProvider"])
    report["ort_load_seconds"] = time.perf_counter() - started
    for case, (length, feed) in zip(report["cases"], feeds):
        report["current_case"] = case["name"]
        started = time.perf_counter()
        actual = session.run(["logits"], feed)[0]
        row = {"engine": "ort", "seconds": time.perf_counter() - started, "passed": False}
        case["executions"].append(row)
        expected = np.load(out / f"{case['name']}.torch.npy", mmap_mode="r")
        row["vs_torch"] = compare_logits(actual, expected, length)
        row["sampling"] = sample_checked(actual, expected, length, adapter, references)
        row["passed"] = row["vs_torch"]["passed"]
        if not row["passed"]:
            raise ValueError(f"ORT numeric error for {case['name']}")
        # IR is compared independently to both references without retaining a
        # second live full-size output array from another execution engine.
        np.save(out / f"{case['name']}.ort.npy", actual)
        del actual, expected
    del session
    gc.collect()
    report["stage"] = "parse"
    started = time.perf_counter()
    parser = ONNXParser()
    program = parser.parse(str(out / "model.onnx"))
    require_valid(program)
    report["parse_seconds"] = time.perf_counter() - started
    report["ir_optimization_seconds"] = {}
    for level in LEVELS:
        report.update(stage="ir", current_level=level)
        started = time.perf_counter()
        manager = create_optimization_pass_manager(level)
        manager.before_pass = lambda p, value: require_valid(value)
        manager.after_pass = lambda p, value: require_valid(value)
        optimized = manager.run_pipeline(copy.deepcopy(program)).data
        require_valid(optimized)
        report["ir_optimization_seconds"][level] = time.perf_counter() - started
        for case, (length, feed) in zip(report["cases"], feeds):
            report["current_case"] = case["name"]
            row = {"engine": f"ir-{level}", "passed": False}
            case["executions"].append(row)
            started = time.perf_counter()
            result = IRInterpreter(optimized).run(feed, initializers=parser.initializers)
            row.update(seconds=time.perf_counter() - started, executed_steps=result.executed_steps)
            actual = result.return_value
            expected = np.load(out / f"{case['name']}.torch.npy", mmap_mode="r")
            row["vs_torch"] = compare_logits(actual, expected, length)
            row["sampling"] = sample_checked(actual, expected, length, adapter, references)
            del expected
            expected = np.load(out / f"{case['name']}.ort.npy", mmap_mode="r")
            row["vs_ort"] = compare_logits(actual, expected, length)
            row["passed"] = row["vs_torch"]["passed"] and row["vs_ort"]["passed"]
            del result, actual, expected
            gc.collect()
            if not row["passed"]:
                raise ValueError(f"IR {level} numeric error for {case['name']}")
            print(f"[{case['name']}/ir-{level}] PASS max_abs={row['vs_torch']['max_abs']:.9g}", flush=True)
        del optimized
    for case in report["cases"]:
        case["passed"] = ([row["engine"] for row in case["executions"]] == ["ort", "ir-none", "ir-all"]
                          and all(row["passed"] for row in case["executions"]))
    report.update(passed=bool(report["cases"]) and all(case["passed"] for case in report["cases"]), stage="complete")
    report["completed_model_invocations"] = len(report["cases"]) + sum(len(case["executions"]) for case in report["cases"])
    report.pop("current_case", None)
    report.pop("current_level", None)


def save_reports(out, report):
    failures = []

    def views():
        text = ("# W2 runtime/model integration\n\nResult: **" + ("PASS" if report["passed"] else "FAIL")
                + "**\n\nOfficial two-layer Qwen3, random weights, full vocabulary; one forward and greedy token.\n"
                + "Not pretrained inference or a generation loop. All positions use strict max_abs < 1e-5.\n\n```json\n"
                + json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n```\n")
        return {"report.md": text, "report.html": '<!doctype html><meta charset="utf-8"><pre>' + html.escape(text) + "</pre>"}

    def failed():
        report["report_write_errors"] = failures
        if report["passed"]:
            report.update(passed=False, stage="report-write", error="Incomplete integration evidence")

    def write_views():
        for name, value in views().items():
            try:
                (out / name).write_text(value, encoding="utf-8")
            except OSError as exc:
                failures.append(f"{name}: {type(exc).__name__}: {exc}")

    write_views()
    if failures:
        failed()
        write_views()
    pending = out / ".report.json.tmp"
    try:
        pending.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        pending.replace(out / "report.json")
    except OSError as exc:
        failures.append(f"report.json: {type(exc).__name__}: {exc}")
        failed()
        write_views()
    finally:
        try:
            pending.unlink(missing_ok=True)
        except OSError:
            pass
    if failures:
        print("; ".join(failures), file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("output/w2-runtime-model"), type=Path)
    args = parser.parse_args(argv)
    out = args.output_dir.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error("--output-dir must be absent or empty; preserve earlier evidence")
    out.mkdir(parents=True, exist_ok=True)
    report = {"gate": "integration:runtime-model", "passed": False, "stage": "initialize",
              "created_at": datetime.now(timezone.utc).isoformat(), "model_id": MANIFEST["model_id"],
              "tokenizer_revision": MANIFEST["revision"], "cases": []}
    started = time.perf_counter()
    try:
        report["source_sha256"] = source_evidence()
        report["checkout"] = checkout_evidence()
        run_probe(args.tokenizer_dir.resolve(), out, report)
    except Exception as exc:
        report.update(passed=False, error=f"{type(exc).__name__}: {exc}")
    report["seconds"] = time.perf_counter() - started
    save_reports(out, report)
    print(f"[integration:runtime-model] {'PASS' if report['passed'] else 'FAIL'}; stage={report['stage']}; report={out / 'report.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
