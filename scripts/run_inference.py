#!/usr/bin/env python3
"""Build/reuse one RV64 ELF and weight bundle for Host-driven QEMU generation.

Raw completion is the default; chat and the frozen reference suite are explicit.
The ordinary path restarts QEMU per forward; --kv-cache keeps one QEMU session.
Both paths reuse their ELF and weight bundle throughout generation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probes.w3_common import atomic_text
from probes.w2_runtime_model.run import checked_inputs
from probes.w4_qwen3_full.run import CoreNamespace, add_core_argument, load_executable, save_executable, ATOL
from probes.w4_qwen3_full.fma_conformance import run_probe as run_fma_probe
from scratchv.backend.tensor_c_codegen import TensorCCodegen
from scratchv.runtime.generation import GenerationCancelledError, generate
from scratchv.runtime.llm_inputs import last_valid_logits, greedy_token
from scratchv.runtime.qemu_generation import QemuForward
from scratchv.runtime.qwen3_tokenizer import Qwen3Tokenizer
from scratchv.runtime.riscv_external import build_external, sha256_file
from scratchv.runtime.riscv_harts import UNSET, read_cores, resolve_cores, select_cores
from scratchv.runtime.riscv_tensor import discover_toolchain
from scratchv.runtime.weight_bundle import write_weight_bundle


def parse_args(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    source = cli.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path, help="Static ONNX graph; compile once")
    source.add_argument("--reuse-build", type=Path, help="Existing W4/W5 executable.json")
    cli.add_argument("--kv-cache", action="store_true", help="Resident KV session: --model is Prefill ONNX")
    cli.add_argument("--decode-model", type=Path, help="Static Decode ONNX (requires --kv-cache)")
    cli.add_argument("--kv-capacity", type=int, choices=(256,), default=256,
                     help="KV ABI v1 capacity (256)")
    cli.add_argument("--tokenizer-dir", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path, required=True, help="New output directory")
    cli.add_argument("--mode", choices=("raw", "chat"), default="raw")
    prompt = cli.add_mutually_exclusive_group()
    prompt.add_argument("--prompt")
    prompt.add_argument("--chat-turn", action="append", help="Repeat for ordered user turns")
    prompt.add_argument("--reference-suite", action="store_true", help="Run all 3 frozen W5 prompts")
    cli.add_argument("--system-prompt")
    cli.add_argument("--max-new-tokens", type=int)
    add_core_argument(cli)
    cli.add_argument("--timeout", type=float, default=3600, help="Per-forward QEMU timeout, seconds")
    cli.add_argument("--total-timeout", type=float, default=10800, help="Generation/suite timeout, seconds")
    cli.add_argument("--build-timeout", type=float, default=900)
    cli.add_argument("--build-only", action="store_true")
    cli.add_argument("--cc")
    cli.add_argument("--qemu")
    cli.add_argument("--matmul-policy", choices=("sequential", "blocked_fma"), default="sequential")
    cli.add_argument("--mean-policy", choices=("sequential", "two-four-lane-last-axis"), default="sequential",
                     help="FP32 last-axis Mean order; default preserves sequential reduction")
    cli.add_argument("--nonlinear-policy", choices=("libm", "fp32-reference"), default="libm",
                     help="Explicit FP32 Trig/Sigmoid/Softmax order; default preserves platform libm")
    cli.add_argument("--matmul-tile", choices=("1x4", "2x4", "1x8"), default="1x4")
    cli.add_argument("--matmul-rhs-layout", choices=("original", "column-panel"), default="original")
    cli.add_argument("--lm-head-policy", choices=("original", "vocab-blocked"), default="original")
    cli.add_argument("--attention-decode-policy", choices=("original", "kv-heads"), default="original",
                     help="Optional KV Decode Attention; Prefill retains the original implementation")
    cli.add_argument("--verify-onnx", type=Path, help="Optional ORT numeric/greedy oracle for every step")
    args = cli.parse_args(argv, namespace=CoreNamespace())
    if any(not np.isfinite(t) or t <= 0 for t in (args.timeout, args.total_timeout, args.build_timeout)):
        cli.error("timeouts must be positive and finite")
    if args.max_new_tokens is not None and args.max_new_tokens < 0:
        cli.error("--max-new-tokens must be nonnegative")
    if not args.build_only and args.prompt is None and not args.chat_turn and not args.reference_suite:
        cli.error("generation requires --prompt, --chat-turn, or --reference-suite")
    if (args.chat_turn or args.system_prompt is not None) and args.mode != "chat":
        cli.error("--chat-turn/--system-prompt require --mode chat")
    if args.reference_suite and (args.mode != "raw" or args.max_new_tokens is not None or not args.verify_onnx):
        cli.error("--reference-suite requires raw mode, --verify-onnx and its frozen token budgets")
    if (args.matmul_tile != "1x4" or args.matmul_rhs_layout != "original") and args.matmul_policy != "blocked_fma":
        cli.error("nondefault tile/RHS layout requires --matmul-policy blocked_fma")
    if args.reuse_build and args.cc:
        cli.error("--cc cannot change a reused build")
    if args.lm_head_policy != "original" and (args.matmul_policy != "blocked_fma" or args.matmul_tile != "2x4"):
        cli.error("--lm-head-policy vocab-blocked requires --matmul-policy blocked_fma --matmul-tile 2x4")
    if args.attention_decode_policy != "original" and (
            not args.kv_cache or args.matmul_policy != "blocked_fma"
            or args.matmul_tile != "2x4" or args.nonlinear_policy != "fp32-reference"):
        cli.error("--attention-decode-policy kv-heads requires --kv-cache, blocked_fma/2x4 and fp32-reference")
    if args.kv_cache:
        if args.mode != "raw":
            cli.error("--kv-cache currently supports raw completion only")
        if args.model and not args.decode_model:
            cli.error("--kv-cache with --model requires --decode-model")
        if args.reuse_build and args.decode_model:
            cli.error("--decode-model cannot change a reused KV build")
    elif args.decode_model:
        cli.error("--decode-model requires --kv-cache")
    return args


def prepare_executable(args, out, cores=UNSET, *, harts=UNSET):
    """Reuse existing metadata validation, or compile/write weights exactly once."""
    cores = select_cores(cores, harts=harts)
    if args.reuse_build:
        previous = args.reuse_build.with_name("report.json")
        record = json.loads(previous.read_text(encoding="utf-8"))
        if record.get("executable_sha256") != sha256_file(args.reuse_build):
            raise ValueError("Reused executable metadata identity is missing or changed")
        exe = load_executable(args.reuse_build)
        if (("cores" in record or "harts" in record)
                and read_cores(record) != exe.cores):
            raise ValueError("Reused report core count differs from executable")
        if (exe.evidence.get("lm_head_policy", "original") != args.lm_head_policy
                or record.get("lm_head_policy", "original") != args.lm_head_policy):
            raise ValueError("Reused build LM Head policy differs from requested --lm-head-policy")
        # W4 stored arithmetic policy in the sibling report; W5 also keeps it
        # in the validated executable metadata.
        policy = exe.evidence.get("matmul_policy", record.get("matmul_policy"))
        expected = {"matmul_tile": args.matmul_tile, "matmul_rhs_layout": args.matmul_rhs_layout,
                    "qemu_workspace_layout": "none"}
        defaults = {"matmul_tile": "1x4", "matmul_rhs_layout": "original", "qemu_workspace_layout": "none"}
        if exe.cores != cores or policy != args.matmul_policy:
            raise ValueError("Reused build core count/arithmetic policy differs from requested configuration")
        if (exe.evidence.get("mean_policy", "sequential") != args.mean_policy
                or record.get("mean_policy", "sequential") != args.mean_policy):
            raise ValueError("Reused build Mean policy differs from requested --mean-policy")
        if (exe.evidence.get("nonlinear_policy", "libm") != args.nonlinear_policy
                or record.get("nonlinear_policy", "libm") != args.nonlinear_policy):
            raise ValueError("Reused build nonlinear policy differs from requested --nonlinear-policy")
        if any(exe.evidence.get(k, defaults[k]) != value for k, value in expected.items()):
            raise ValueError("Reused build layout/tile differs from requested configuration; padding must be none")
        # Preserve validated W4 evidence when this executable is saved as W5
        # metadata. A later reuse must not depend on the original W4 report.
        evidence = dict(exe.evidence)
        evidence.setdefault("matmul_policy", policy)
        if "w5_graph_audit" not in evidence and isinstance(record.get("graph_audit"), dict):
            evidence["w5_graph_audit"] = dict(record["graph_audit"])
        if evidence != exe.evidence:
            exe = replace(exe, evidence=evidence)
        if args.qemu:
            exe = replace(exe, toolchain=discover_toolchain(exe.toolchain.cc, args.qemu))
        return exe
    from scratchv.frontend.onnx_parser import ONNXParser
    parser = ONNXParser()
    program = parser.parse(str(args.model.resolve()), mmap_external_data=True)
    artifact = TensorCCodegen(
        program, initializers=parser.initializers, constant_storage="external",
        max_constant_bytes=3 * 1024**3, max_workspace_bytes=1024**3, kernel_calls=True,
        matmul_policy=args.matmul_policy, matmul_tile=args.matmul_tile, mean_policy=args.mean_policy,
        nonlinear_policy=args.nonlinear_policy,
        lm_head_policy=args.lm_head_policy,
        matmul_rhs_layout=args.matmul_rhs_layout, qemu_workspace_layout="none",
    ).generate()
    write_weight_bundle(artifact.external_weights, artifact.external_initializers, out / "weights")
    exe = build_external(artifact, out / "weights", out / "build",
                         discover_toolchain(args.cc, args.qemu), timeout=args.build_timeout, cores=cores)
    audit = {"passed": False, "layer_count": None}
    if tuple(exe.output.shape) == (1, 256, 151936):
        import onnx
        from probes.w2_qwen3_parse.validation import audit_graph_structure, Qwen3StructureError
        try:
            found = audit_graph_structure(onnx.load(str(args.model), load_external_data=False))
            audit = {key: found[key] for key in ("passed", "layer_count", "graph", "tied_embeddings")}
        except Qwen3StructureError as exc:
            audit["error"] = str(exc)
    return replace(exe, evidence={**exe.evidence, "matmul_policy": args.matmul_policy,
                                  "w5_graph_audit": audit})


def require_full_executable(exe):
    """Check executable evidence already validated/promoted by preparation."""
    audit = exe.evidence.get("w5_graph_audit")
    if (not isinstance(audit, dict) or audit.get("passed") is not True
            or audit.get("layer_count") != 28 or tuple(exe.output.shape) != (1, 256, 151936)):
        raise ValueError("Full reference suite requires complete graph evidence for this executable")


def oracle_session(model_path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_cpu_mem_arena = False
    return ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])


def reference_session(model_path, fixture):
    """Require the fixed complete Qwen3 graph before claiming a W5 full gate."""
    import onnx
    from probes.w2_qwen3_parse.validation import audit_graph_structure
    from probes.w5_generation.run import session_for
    manifest = json.loads((ROOT / fixture["model_manifest"]).read_text(encoding="utf-8"))
    if (manifest["model_id"] != fixture["model_id"] or manifest["revision"] != fixture["revision"]
            or model_path.name != "model.onnx" or model_path.parent.name != "onnx"):
        raise ValueError("Reference suite requires the fixed W1 ONNX asset layout/revision")
    audit = audit_graph_structure(onnx.load(str(model_path), load_external_data=False))
    if not audit["passed"] or audit["layer_count"] != 28:
        raise ValueError("Reference suite requires the complete 28-layer Qwen3 graph")
    return session_for(model_path.parent.parent, manifest)


def compare_step(actual, expected, valid_length, vocab_size):
    """Full finite-output check with the existing W4 absolute tolerance."""
    left = last_valid_logits(actual, valid_length, vocab_size=vocab_size)
    right = last_valid_logits(expected, valid_length, vocab_size=vocab_size)
    maximum = 0.0
    for row in range(actual.shape[1]):
        delta = np.abs(actual[0, row].astype(np.float64) - expected[0, row].astype(np.float64))
        maximum = max(maximum, float(delta.max()))
    ids = greedy_token(left), greedy_token(right)
    selected_row_passed = (np.array_equal(left, actual[0, valid_length - 1])
                           and np.array_equal(right, expected[0, valid_length - 1]))
    top_two = np.partition(right, -2)[-2:] if right.size > 1 else None
    # Finite FP32 endpoints can have a difference outside FP32 range.
    return {"passed": maximum < ATOL and ids[0] == ids[1] and selected_row_passed,
            "selected_row_passed": selected_row_passed, "max_abs": maximum,
            "atol": ATOL, "rtol": 0.0, "qemu_token_id": ids[0], "ort_token_id": ids[1],
            "ort_top2_margin": (float(top_two[1]) - float(top_two[0])) if top_two is not None else None,
            "elements_compared": int(actual.size)}


def load_reference_suite(tokenizer):
    fixture = json.loads((ROOT / "probes/w5_generation/references.json").read_text(encoding="utf-8"))
    meta = {"pad_token_id": tokenizer.pad_token_id,
            "generation_eos_token_ids": list(tokenizer.generation_eos_token_ids),
            "model_vocab_size": tokenizer.model_vocab_size,
            "tokenizer_vocab_size": tokenizer.tokenizer_vocab_size}
    if fixture["status"] != "frozen" or fixture["tokenizer"] != meta or len(fixture["cases"]) != 3:
        raise ValueError("Reference suite/tokenizer differs from the frozen contract")
    return fixture


def require_full_cached_executable(exe):
    from scratchv.runtime.kv_cache import KVCacheLayout
    if exe.cache_layout != KVCacheLayout(28, 8, 128, 151936, pad_token_id=151643):
        raise ValueError("Reference suite requires the complete 28-layer Qwen3 KV executable")


def cached_main(args):
    from scratchv.frontend.onnx_parser import ONNXParser
    from scratchv.runtime.cached_generation import generate_cached
    from scratchv.runtime.kv_cache import KVCacheLayout, cached_logits
    from scratchv.runtime.llm_inputs import prepare_inputs
    from scratchv.runtime.riscv_session import (
        build_session, QemuSession, save_session_executable, load_session_executable)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    requested = args.cores if args.cores == "auto" else int(args.cores)
    cores = resolve_cores(requested)
    started = time.perf_counter()
    policies = {"matmul_policy": args.matmul_policy, "matmul_tile": args.matmul_tile,
                "matmul_rhs_layout": args.matmul_rhs_layout, "mean_policy": args.mean_policy,
                "nonlinear_policy": args.nonlinear_policy, "lm_head_policy": args.lm_head_policy,
                "attention_decode_policy": args.attention_decode_policy}
    report = {"schema_version": 1, "passed": False, "status": "FAIL", "stage": "prepare",
              "kv_cache": True, "mode": "raw", "cores": cores, "cores_requested": requested,
              "harts": cores, "harts_requested": requested, "capacity": args.kv_capacity,
              "qemu_workspace_layout": "none", **policies,
              "execution": "one resident QEMU and one ELF/bundle; each case resets KV, then one Prefill and N-1 Decode calls",
              "numeric_scope": "all vocabulary logits at the current token; no cache tensor dump",
              "full_model_gate": "NOT_RUN", "cases": [], "numeric_steps": [],
              "generation_timings": {}, "timings": {"checkpoint_save_s": 0.0},
              "timing_scope": "case generation_and_validation_s includes ORT, numeric checks and final text/stop checks; excludes build, session startup, report saving and terminal display; checkpoint_save_s excludes the final report write"}
    session = None
    previous = {}
    exit_code = 1

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"generation interrupted by signal {signum}")

    def save():
        if session is not None:
            report["session"] = session.report
            report["runs"] = session.runs
        report["elapsed_seconds"] = time.perf_counter() - started
        atomic_text(out / "report.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    def checkpoint():
        begin = time.perf_counter()
        try:
            save()
        finally:
            report["timings"]["checkpoint_save_s"] += time.perf_counter() - begin

    try:
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.getsignal(sig)
                signal.signal(sig, interrupted)
        tokenizer = Qwen3Tokenizer.from_directory(args.tokenizer_dir)
        if args.reuse_build:
            prior = json.loads(args.reuse_build.with_name("report.json").read_text())
            if prior.get("executable_sha256") != sha256_file(args.reuse_build):
                raise ValueError("Reused KV executable metadata changed")
            exe = load_session_executable(args.reuse_build)
            if exe.cores != cores:
                raise ValueError("Reused KV core count differs from requested configuration")
            stored_policies = dict(exe.evidence.get("policies", {}))
            for key in ("lm_head_policy", "attention_decode_policy"):
                stored_policies.setdefault(key, "original")
            if stored_policies != policies:
                raise ValueError("Reused KV arithmetic policies differ from requested configuration")
            if args.qemu:
                exe = replace(exe, toolchain=discover_toolchain(exe.toolchain.cc, args.qemu))
        else:
            artifacts = []
            for mode, path in (("prefill", args.model), ("decode", args.decode_model)):
                parser = ONNXParser()
                program = parser.parse(str(path.resolve()), mmap_external_data=True)
                artifacts.append(TensorCCodegen(program, parser.initializers,
                    constant_storage="external", abi_version=2, function_name=f"sv_{mode}_v2",
                    max_constant_bytes=3*1024**3, max_workspace_bytes=1024**3, kernel_calls=True,
                    matmul_policy=args.matmul_policy, matmul_tile=args.matmul_tile,
                    matmul_rhs_layout=args.matmul_rhs_layout, mean_policy=args.mean_policy,
                    nonlinear_policy=args.nonlinear_policy, lm_head_policy=args.lm_head_policy,
                    attention_decode_policy=args.attention_decode_policy if mode == "decode" else "original").generate())
            outputs = artifacts[0].outputs
            if len(outputs) < 3 or len(outputs) % 2 != 1 or len(outputs[1].shape) != 4:
                raise ValueError("Invalid KV output signature")
            cache = KVCacheLayout((len(outputs)-1)//2, outputs[1].shape[1], outputs[1].shape[3],
                                  outputs[0].shape[2], pad_token_id=tokenizer.pad_token_id)
            exe = build_session(*artifacts, cache, out / "build", discover_toolchain(args.cc, args.qemu),
                                cores=cores, timeout=args.build_timeout)
            exe = replace(exe, evidence={**exe.evidence, "policies": policies})
        if (exe.cache_layout.vocab_size != tokenizer.model_vocab_size
                or exe.cache_layout.pad_token_id != tokenizer.pad_token_id):
            raise ValueError("Tokenizer differs from the KV executable vocabulary/padding")
        if args.reference_suite:
            require_full_cached_executable(exe)
        report["build"] = exe.evidence
        report["executable_sha256"] = save_session_executable(exe, out)
        report["timings"]["prepare_s"] = time.perf_counter() - started
        if args.build_only:
            report.update(status="BUILD_ONLY", stage="build_complete")
        else:
            report["stage"] = "reference_contract"
            fixture = load_reference_suite(tokenizer) if args.reference_suite else None
            if fixture is not None:
                cases = fixture["cases"]
                if (len({case["name"] for case in cases}) != 3
                        or any(case["max_new_tokens"] != 8 or len(case["generated_token_ids"]) != 8
                               for case in cases)):
                    raise ValueError("KV reference suite requires three distinct frozen 8-token cases")
            else:
                cases = [{"name": "turn-0", "prompt": args.prompt,
                          "max_new_tokens": 8 if args.max_new_tokens is None else args.max_new_tokens}]
            report["stage"] = "oracle_prepare"
            begin = time.perf_counter()
            oracle = (reference_session(args.verify_onnx, fixture) if fixture is not None else
                      oracle_session(args.verify_onnx) if args.verify_onnx else None)
            report["timings"]["oracle_prepare_s"] = time.perf_counter() - begin
            deadline = time.monotonic() + args.total_timeout
            session = QemuSession(exe, out / "session", timeout=args.timeout, deadline=deadline)

            class CheckedSession:
                def __init__(self, case, timings):
                    self.case, self.timings = case, timings
                    self.history = []
                    self.step = 0
                    self.prefill_count = self.decode_count = 0
                    self.reference_history = list(case["prompt_token_ids"]) if fixture is not None else None

                def reset(self):
                    self.history = []
                    return session.reset()

                def check(self, actual):
                    if oracle is not None:
                        begin = time.perf_counter()
                        prepared = prepare_inputs(self.history, pad_token_id=tokenizer.pad_token_id,
                                                  vocab_size=tokenizer.model_vocab_size)
                        feed = (checked_inputs(self.reference_history, prepared, tokenizer.pad_token_id)
                                if fixture is not None else prepared.as_feed())
                        expected = oracle.run(["logits"], feed)[0]
                        self.timings["ort_s"] += time.perf_counter() - begin
                        begin = time.perf_counter()
                        expected = last_valid_logits(expected, len(self.history), vocab_size=tokenizer.model_vocab_size)
                        observed = cached_logits(actual, tokenizer.model_vocab_size)
                        error = float(np.max(np.abs(observed.astype(np.float64) - expected.astype(np.float64))))
                        qemu_id, ort_id = greedy_token(observed), greedy_token(expected)
                        comparison = {"case": self.case["name"], "step": self.step,
                            "phase": "prefill" if self.step == 0 else "decode",
                            "valid_length": len(self.history), "max_abs_error": error, "max_abs": error,
                            "atol": ATOL, "rtol": 0.0, "greedy_equal": qemu_id == ort_id,
                            "qemu_token_id": qemu_id, "ort_token_id": ort_id,
                            "elements_compared": int(observed.size), "passed": error < ATOL and qemu_id == ort_id}
                        if fixture is not None:
                            frozen_id = self.case["generated_token_ids"][self.step]
                            comparison.update(reference_token_id=frozen_id,
                                              reference_token_passed=qemu_id == ort_id == frozen_id)
                            comparison["passed"] &= comparison["reference_token_passed"]
                        report["numeric_steps"].append(comparison)
                        self.timings["numeric_comparison_s"] += time.perf_counter() - begin
                        if not comparison["passed"]:
                            raise ValueError("Cached logits, greedy token or frozen token differ from the ORT reference")
                        if fixture is not None:
                            self.reference_history.append(frozen_id)
                    self.step += 1
                    return actual

                def forward(self, phase, value):
                    if fixture is not None:
                        if self.step >= len(self.case["generated_token_ids"]):
                            raise ValueError("Cached forward called after the frozen reference ended")
                        if self.history != self.reference_history:
                            raise ValueError("Cached input history differs from the frozen reference")
                    run_start = len(session.runs)
                    try:
                        return self.check(getattr(session, phase)(value))
                    finally:
                        for run in session.runs[run_start:]:
                            run.update(case=self.case["name"], phase=phase)

                def prefill(self, ids):
                    self.history = list(ids)
                    self.prefill_count += 1
                    return self.forward("prefill", ids)

                def decode(self, token):
                    self.history.append(token)
                    self.decode_count += 1
                    return self.forward("decode", token)

            report["stage"] = "session_start"
            session_start = time.perf_counter()
            with session:
                report["timings"]["session_start_s"] = time.perf_counter() - session_start
                report["stage"] = "generation"
                if fixture is not None:
                    report["full_model_gate"] = "RUNNING"
                    checkpoint()
                for case in cases:
                    report["active_case"] = case["name"]
                    numeric_start = len(report["numeric_steps"])
                    timings = {"ort_s": 0.0, "numeric_comparison_s": 0.0}
                    report["generation_timings"] = timings
                    checked = CheckedSession(case, timings)
                    begin = time.perf_counter()
                    result = generate_cached(case["prompt"], tokenizer=tokenizer, session=checked,
                        max_new_tokens=case["max_new_tokens"],
                        timeout_seconds=max(0.0, deadline-time.monotonic()), timings=timings)
                    timings["generation_call_s"] = time.perf_counter() - begin
                    check_start = time.perf_counter()
                    row = {"name": case["name"], "prompt": case["prompt"],
                           "max_new_tokens": case["max_new_tokens"], **asdict(result),
                           "numeric_step_count": len(report["numeric_steps"]) - numeric_start,
                           "prefill_count": checked.prefill_count, "decode_count": checked.decode_count,
                           "forward_count": checked.prefill_count + checked.decode_count,
                           "timings": timings}
                    if fixture is not None:
                        row["reference_passed"] = (
                            row["numeric_step_count"] == row["forward_count"] == 8
                            and row["prefill_count"] == 1 and row["decode_count"] == 7
                            and list(result.prompt_token_ids) == case["prompt_token_ids"]
                            and list(result.generated_token_ids) == case["generated_token_ids"]
                            and result.generated_text == case["generated_text"]
                            and result.stop_reason == case["stop_reason"]
                            and case["required_substring"] in result.generated_text)
                    timings["result_validation_s"] = time.perf_counter() - check_start
                    timings["generation_and_validation_s"] = time.perf_counter() - begin
                    if not report["cases"]:
                        timings["first_case_with_session_start_s"] = time.perf_counter() - session_start
                    report["cases"].append(row)
                    if fixture is None:
                        report["result"] = asdict(result)
                        report["generation_timings"] = timings
                    if fixture is not None and not row["reference_passed"]:
                        raise ValueError(f"Frozen reference mismatch: {case['name']}")
                    checkpoint()
                    print(result.generated_text, flush=True)
            report["timings"]["resident_session_s"] = time.perf_counter() - session_start
            if fixture is not None:
                if len(report["cases"]) != 3 or not all(row["reference_passed"] for row in report["cases"]):
                    raise ValueError("All three frozen cases must pass")
                report["full_model_gate"] = "PASS"
            report.update(passed=True, status="PASS", stage="complete")
        exit_code = 0
    except (GenerationCancelledError, KeyboardInterrupt, SystemExit) as exc:
        if isinstance(exc, SystemExit) and exc.code not in (130, 143):
            raise
        report.update(status="CANCELLED", error=f"{type(exc).__name__}: {exc}")
        if report["full_model_gate"] == "RUNNING":
            report["full_model_gate"] = "CANCELLED"
        exit_code = exc.code if isinstance(exc, SystemExit) else 130
        print(report["error"], file=sys.stderr, flush=True)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        if report["full_model_gate"] == "RUNNING":
            report["full_model_gate"] = "FAIL"
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        try:
            save()
        except Exception as exc:
            exit_code = exit_code or 1
            print(f"Report save failed: {type(exc).__name__}: {exc}; "
                  f"{out / 'report.json'} was not updated and may be missing or stale.",
                  file=sys.stderr, flush=True)
        finally:
            gc.collect()
    return exit_code


def main(argv=None):
    args = parse_args(argv)
    if args.kv_cache:
        return cached_main(args)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    requested = args.cores if args.cores == "auto" else int(args.cores)
    cores = resolve_cores(requested)
    report = {"schema_version": 1, "status": "FAIL", "passed": False, "stage": "prepare",
              "mode": args.mode, "cores_requested": requested, "cores": cores,
              "harts_requested": requested, "harts": cores,
              "qemu_workspace_layout": "none", "matmul_policy": args.matmul_policy,
              "mean_policy": args.mean_policy, "nonlinear_policy": args.nonlinear_policy,
              "matmul_tile": args.matmul_tile, "matmul_rhs_layout": args.matmul_rhs_layout,
              "lm_head_policy": args.lm_head_policy, "attention_decode_policy": args.attention_decode_policy,
              "cases": [], "runs": [], "numeric_steps": [], "qemu_forwards": 0,
              "full_model_gate": "NOT_RUN", "container_reproduction": "NOT_ASSERTED",
              "execution": "one ELF/bundle; a fresh QEMU process per forward; no KV cache"}
    started = time.perf_counter()
    adapter = None
    previous = {}
    exit_code = 1

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"generation interrupted by signal {signum}")

    def save():
        if adapter is not None:
            report["runs"] = adapter.runs
            report["forward_attempts"] = len(adapter.runs)
            report["qemu_forwards"] = sum(bool(run.get("guest_completed")) for run in adapter.runs)
        report["elapsed_seconds"] = time.perf_counter() - started
        atomic_text(out / "report.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    try:
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.getsignal(sig)
                signal.signal(sig, interrupted)
        tokenizer = Qwen3Tokenizer.from_directory(args.tokenizer_dir)
        exe = prepare_executable(args, out, cores)
        if args.reference_suite:
            require_full_executable(exe)
        report["executable_sha256"] = save_executable(exe, out)
        adapter = QemuForward(exe, out / "steps", timeout=args.timeout)
        if adapter.vocab_size != tokenizer.model_vocab_size:
            raise ValueError("Tokenizer model vocabulary differs from the executable output")
        report["build"] = exe.evidence
        if args.build_only:
            report.update(status="BUILD_ONLY", stage="build_complete")
        else:
            report["stage"] = "fma_preflight"
            fma = run_fma_probe(out / "fma-conformance", cc=exe.toolchain.cc,
                                qemu=exe.toolchain.qemu, timeout=min(args.build_timeout, 120))
            report["fma_conformance"] = fma
            if not fma.get("passed") or fma.get("status") != "PASS":
                raise ValueError("Actual RV64 execution toolchain failed FMA conformance")
            fixture = None
            if args.reference_suite:
                report["stage"] = "reference_contract"
                fixture = load_reference_suite(tokenizer)
                cases = fixture["cases"]
            else:
                texts = args.chat_turn or [args.prompt]
                cases = [{"name": f"turn-{i}", "prompt": text,
                          "max_new_tokens": 8 if args.max_new_tokens is None else args.max_new_tokens}
                         for i, text in enumerate(texts)]
            report["stage"] = "oracle_prepare"
            session = (reference_session(args.verify_onnx, fixture) if fixture is not None else
                       oracle_session(args.verify_onnx) if args.verify_onnx else None)
            conversation = None
            if args.mode == "chat":
                report["stage"] = "conversation_prepare"
                from scratchv.runtime.conversation import Conversation
                conversation = Conversation.from_directory(args.tokenizer_dir, system_prompt=args.system_prompt)
            deadline = time.monotonic() + args.total_timeout
            adapter.deadline = deadline
            report["stage"] = "generation"
            if fixture is not None:
                report["full_model_gate"] = "RUNNING"
                save()

            def forward(prepared):
                step = len(report["numeric_steps"]) - numeric_start
                if fixture is not None:
                    if step >= len(case["generated_token_ids"]):
                        raise ValueError("Host called forward after the frozen reference ended")
                    feed = checked_inputs(reference_history, prepared, tokenizer.pad_token_id)
                else:
                    feed = prepared.as_feed()
                ort_start = time.perf_counter()
                expected = session.run(["logits"], feed)[0] if session is not None else None
                if session is not None:
                    timings["ort_s"] += time.perf_counter() - ort_start
                actual = adapter(prepared)
                if expected is not None:
                    comparison_start = time.perf_counter()
                    comparison = compare_step(actual, expected, prepared.valid_length, adapter.vocab_size)
                    comparison.update(case=case["name"], step=step, valid_length=prepared.valid_length)
                    if fixture is not None:
                        # Select independently from frozen history, never from a
                        # caller-declared row or the generation sampling helper.
                        position = len(reference_history) - 1
                        qemu_id = int(np.argmax(actual[0, position]))
                        ort_id = int(np.argmax(expected[0, position]))
                        frozen_id = case["generated_token_ids"][step]
                        comparison["reference_token_id"] = frozen_id
                        comparison["selected_row_passed"] = (
                            comparison["selected_row_passed"] and comparison["qemu_token_id"] == qemu_id
                            and comparison["ort_token_id"] == ort_id)
                        comparison["reference_token_passed"] = qemu_id == ort_id == frozen_id
                        comparison["passed"] = (comparison["passed"]
                            and comparison["selected_row_passed"] and comparison["reference_token_passed"])
                    report["numeric_steps"].append(comparison)
                    timings["numeric_comparison_s"] += time.perf_counter() - comparison_start
                    if not comparison["passed"]:
                        # The outer handler saves the failure after preserving its cause.
                        raise ValueError("QEMU/ORT full logits, selected row or frozen token differ; see numeric_steps")
                    if fixture is not None:
                        reference_history.append(frozen_id)
                return actual

            for case in cases:
                numeric_start, run_start = len(report["numeric_steps"]), len(adapter.runs)
                timings = {"ort_s": 0.0, "numeric_comparison_s": 0.0}
                generation_start = time.perf_counter()
                reference_history = list(case["prompt_token_ids"]) if fixture is not None else None
                remaining = max(0.0, deadline - time.monotonic())
                kwargs = {"forward": forward, "max_new_tokens": case["max_new_tokens"],
                          "timeout_seconds": remaining}
                result = (conversation.turn(case["prompt"], **kwargs) if conversation else
                          generate(case["prompt"], tokenizer=tokenizer, **kwargs))
                timings["generation_call_s"] = time.perf_counter() - generation_start
                validation_start = time.perf_counter()
                row = {"name": case["name"], "prompt": case["prompt"],
                       "max_new_tokens": case["max_new_tokens"], **asdict(result), "timings": timings}
                if fixture is not None:
                    row["numeric_step_count"] = len(report["numeric_steps"]) - numeric_start
                    row["forward_count"] = len(adapter.runs) - run_start
                    row["reference_passed"] = (
                        row["numeric_step_count"] == row["forward_count"] == len(case["generated_token_ids"])
                        and list(result.prompt_token_ids) == case["prompt_token_ids"]
                        and list(result.generated_token_ids) == case["generated_token_ids"]
                        and result.generated_text == case["generated_text"]
                        and result.stop_reason == case["stop_reason"]
                        and case["required_substring"] in result.generated_text)
                timings["result_validation_s"] = time.perf_counter() - validation_start
                timings["generation_and_validation_s"] = time.perf_counter() - generation_start
                report["cases"].append(row)
                if fixture is not None and not row["reference_passed"]:
                    raise ValueError(f"Frozen reference mismatch: {case['name']}")
                save()
                print(result.generated_text, flush=True)
            if conversation:
                report["history"] = [asdict(message) for message in conversation.history]
            if fixture is not None:
                report["full_model_gate"] = "PASS"
            report.update(passed=True, status="PASS", stage="complete")
        exit_code = 0
    except (GenerationCancelledError, KeyboardInterrupt, SystemExit) as exc:
        # The runner's temporary signal handler exits with 128 + signal.
        # Other SystemExit exceptions retain their original propagation.
        if isinstance(exc, SystemExit) and exc.code not in (130, 143):
            raise
        report.update(status="CANCELLED", error=f"{type(exc).__name__}: {exc}")
        if report["full_model_gate"] == "RUNNING":
            report["full_model_gate"] = "CANCELLED"
        exit_code = exc.code if isinstance(exc, SystemExit) else 130
        print(report["error"], file=sys.stderr, flush=True)
    except Exception as exc:
        report.update(error=f"{type(exc).__name__}: {exc}")
        if report["full_model_gate"] == "RUNNING":
            report["full_model_gate"] = "FAIL"
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        try:
            save()
        except Exception as exc:
            # A secondary persistence error must not hide generation/cancellation.
            # Even successful generation or build-only needs a usable final report.
            exit_code = exit_code or 1
            print(f"Report save failed: {type(exc).__name__}: {exc}; "
                  f"{out / 'report.json'} was not updated and may be missing or stale.",
                  file=sys.stderr, flush=True)
        else:
            print(f"{report['status']}: {out / 'report.json'}", flush=True)
        finally:
            gc.collect()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
