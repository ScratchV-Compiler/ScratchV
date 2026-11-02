#!/usr/bin/env python3
"""Run three fixed W5 Host generation references with the pinned full ORT model.

Default: compare token IDs/text/stop reasons with references.json.
--record: collect a candidate reference in output only; never rewrite the fixture.
This is an ORT/Host test, not the QEMU W5 end-to-end gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probes.w2_runtime_model.run import checked_inputs
from scratchv.runtime.generation import generate
from scratchv.runtime.llm_inputs import SEQ_LENGTH
from scratchv.runtime.qwen3_tokenizer import Qwen3Tokenizer

FIXTURE = Path(__file__).with_name("references.json")


def reference_stop(length, generated, budget, eos_ids):
    """Independent stop oracle; the prompt EOS is deliberately ignored."""
    if generated and generated[-1] in eos_ids:
        return "eos"
    if length == SEQ_LENGTH:
        return "capacity"
    if len(generated) >= budget:
        return "max_new_tokens"
    return None


def session_for(assets, manifest):
    import onnxruntime as ort

    # Reuse the existing artifact identity/size manifest, without rehashing GBs.
    # Size is an availability/LFS-pointer check, not a replacement integrity hash.
    for item in manifest["files"]:
        path = assets / "onnx" / item["name"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Missing/incomplete fixed ONNX asset: {item['name']}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_cpu_mem_arena = False
    session = ort.InferenceSession(
        str(assets / "onnx/model.onnx"), options, providers=["CPUExecutionProvider"])
    inputs = [(item.name, item.type, item.shape) for item in session.get_inputs()]
    outputs = [(item.name, item.type, item.shape) for item in session.get_outputs()]
    if inputs != [("input_ids", "tensor(int64)", [1, 256]),
                  ("attention_mask", "tensor(float)", [1, 1, 256, 256])]:
        raise ValueError(f"Unexpected model input ABI: {inputs}")
    if outputs != [("logits", "tensor(float)", [1, 256, 151936])]:
        raise ValueError(f"Unexpected model output ABI: {outputs}")
    return session


def run_case(case, tokenizer, session):
    """Exercise Host generate, checked by independent per-call and final oracles."""
    prompt = case["prompt"]
    budget = case["max_new_tokens"]
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    history = list(prompt_ids)
    independently_generated = []
    steps = []
    eos_ids = tokenizer.generation_eos_token_ids
    start_wall, start_cpu = time.perf_counter(), time.process_time()

    def forward(prepared):
        # A wrong append, stale input, padding value, mask, or extra call fails.
        if reference_stop(len(history), independently_generated, budget, eos_ids):
            raise AssertionError("Host called forward after the independent stop condition")
        feed = checked_inputs(history, prepared, tokenizer.pad_token_id)
        begin_wall, begin_cpu = time.perf_counter(), time.process_time()
        logits = session.run(["logits"], feed)[0]
        elapsed_wall = time.perf_counter() - begin_wall
        elapsed_cpu = time.process_time() - begin_cpu
        if (logits.shape != (1, SEQ_LENGTH, tokenizer.model_vocab_size)
                or logits.dtype != np.float32 or not np.isfinite(logits).all()):
            raise ValueError("ORT returned invalid complete FP32 logits")
        # Deliberately independent of last_valid_logits/greedy_token.
        vector = logits[0, len(history) - 1, :]
        token = int(np.argmax(vector))
        tokenizer.validate_token_ids([token])
        top_two = np.partition(vector, -2)[-2:]
        steps.append({
            "step": len(steps), "valid_length": len(history),
            "token_id": token,
            "decoded_token": tokenizer.decode([token], skip_special_tokens=False,
                                               clean_up_tokenization_spaces=False),
            "top1_logit": float(vector[token]),
            "top2_logit": float(top_two[0]),
            "top1_top2_margin": float(vector[token]) - float(top_two[0]),
            "forward_wall_seconds": elapsed_wall, "forward_cpu_seconds": elapsed_cpu,
        })
        independently_generated.append(token)
        history.append(token)
        return logits

    actual = generate(prompt, tokenizer=tokenizer, forward=forward, max_new_tokens=budget)
    wall_seconds = time.perf_counter() - start_wall
    cpu_seconds = time.process_time() - start_cpu
    expected_stop = reference_stop(len(history), independently_generated, budget, eos_ids)
    expected_text = tokenizer.decode(independently_generated, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
    if tuple(actual.prompt_token_ids) != tuple(prompt_ids):
        raise AssertionError("Host prompt IDs differ from the independent tokenizer result")
    if tuple(actual.generated_token_ids) != tuple(independently_generated):
        raise AssertionError("Host sampled IDs differ from independent ORT argmax")
    if actual.stop_reason != expected_stop or actual.generated_text != expected_text:
        raise AssertionError("Host stop reason/text differs from the independent oracle")
    if len(steps) != len(independently_generated):
        raise AssertionError("Forward count differs from generated token count")
    return {
        "name": case["name"], "prompt": prompt,
        "prompt_utf8_hex": prompt.encode("utf-8").hex(),
        "required_substring": case["required_substring"], "max_new_tokens": budget,
        "prompt_token_ids": prompt_ids, "generated_token_ids": independently_generated,
        "generated_text": expected_text, "stop_reason": expected_stop,
        "semantic_passed": case["required_substring"] in expected_text,
        "host_oracle_passed": True, "steps": steps,
        "wall_seconds": wall_seconds, "cpu_seconds": cpu_seconds,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def compare_frozen(actual, expected):
    keys = ("prompt", "prompt_utf8_hex", "required_substring", "max_new_tokens",
            "prompt_token_ids", "generated_token_ids", "generated_text", "stop_reason")
    differences = [key for key in keys if actual[key] != expected[key]]
    return {"passed": not differences, "different_fields": differences}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--assets", type=Path, default=ROOT / "assets/qwen3-0.6b")
    cli.add_argument("--output-dir", type=Path, required=True)
    cli.add_argument("--record", action="store_true",
                     help="Record a candidate reference in output; do not edit frozen references")
    args = cli.parse_args(argv)
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / fixture["model_manifest"]).read_text(encoding="utf-8"))
    if (manifest["revision"] != fixture["revision"]
            or manifest["model_id"] != fixture["model_id"]):
        raise ValueError("Reference fixture does not match the fixed model manifest")
    if not args.record and fixture["status"] != "frozen":
        raise ValueError("Freeze and review the initial reference before verification mode")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "reference-run.json"
    report = {
        "schema_version": 1, "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "record_candidate" if args.record else "verify_frozen",
        "model_manifest": fixture["model_manifest"],
        "model_id": manifest["model_id"], "revision": manifest["revision"],
        "asset_identity": "existing manifest/revision; files checked for expected size, not rehashed",
        "contract": fixture["contract"],
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("numpy", "onnxruntime", "tokenizers")},
            "provider": "CPUExecutionProvider", "intra_op_threads": 1,
            "inter_op_threads": 1, "graph_optimization": "ORT_DISABLE_ALL",
            "cpu_memory_arena": False,
        },
        "memory_note": "Linux ru_maxrss is cumulative process peak, not per-case allocation.",
        "qemu_e2e": "NOT_RUN", "cases": [],
    }
    started_wall, started_cpu = time.perf_counter(), time.process_time()

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        tokenizer = Qwen3Tokenizer.from_directory(args.assets / "tokenizer")
        report["tokenizer"] = {
            "pad_token_id": tokenizer.pad_token_id,
            "generation_eos_token_ids": list(tokenizer.generation_eos_token_ids),
            "model_vocab_size": tokenizer.model_vocab_size,
            "tokenizer_vocab_size": tokenizer.tokenizer_vocab_size,
        }
        if not args.record and report["tokenizer"] != fixture["tokenizer"]:
            raise ValueError("Tokenizer IDs/vocabulary changed from the frozen reference")
        begin_wall, begin_cpu = time.perf_counter(), time.process_time()
        session = session_for(args.assets, manifest)
        report["session_load_wall_seconds"] = time.perf_counter() - begin_wall
        report["session_load_cpu_seconds"] = time.process_time() - begin_cpu
        for case in fixture["cases"]:
            print(f"Running {case['name']}: {case['max_new_tokens']} token budget", flush=True)
            result = run_case(case, tokenizer, session)
            result["frozen_comparison"] = None if args.record else compare_frozen(result, case)
            report["cases"].append(result)
            save()
            print(json.dumps({key: result[key] for key in
                              ("name", "generated_text", "stop_reason", "wall_seconds",
                               "cpu_seconds", "semantic_passed")}, ensure_ascii=False), flush=True)
        report["status"] = "passed" if all(
            case["semantic_passed"] and case["host_oracle_passed"]
            and (args.record or case["frozen_comparison"]["passed"])
            for case in report["cases"]) else "failed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started_wall
        report["cpu_seconds"] = time.process_time() - started_cpu
        report["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save()
    print(f"{report['status'].upper()}: {output}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
