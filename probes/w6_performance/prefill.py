#!/usr/bin/env python3
"""Compare static Prefill lengths 256 and 5 on the same seeded small Qwen3.

This probe deliberately rejects pretrained/full-model checkpoints. Both resident
four-core QEMU sessions use the same Decode256 graph, weights and FP32 policies.
Run from the repository root with --output-dir pointing to a new output folder.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratchv.runtime.kv_cache import prepare_decode, prepare_prefill

ATOL = 1e-3
PROMPT = [1, 2, 3, 4, 5]
VARIANTS = ("prefill-256", "prefill-5")
POLICIES = dict(
    matmul_policy="blocked_fma", matmul_tile="2x4",
    matmul_rhs_layout="column-panel", mean_policy="two-four-lane-last-axis",
    nonlinear_policy="fp32-reference", qemu_workspace_layout="none",
    lm_head_policy="original", attention_decode_policy="original",
)


def save_report(directory, report):
    path = directory / "report.json"
    temporary = path.with_suffix(".json.pending")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def compile_graph(path, mode):
    from scratchv.frontend.onnx_parser import ONNXParser
    from scratchv.backend.tensor_c_codegen import TensorCCodegen

    parser = ONNXParser()
    program = parser.parse(str(path), mmap_external_data=True)
    return TensorCCodegen(
        program, parser.initializers, constant_storage="external",
        function_name=f"sv_{mode}_v2", kernel_calls=True,
        max_constant_bytes=64 * 1024**2, max_workspace_bytes=128 * 1024**2,
        **POLICIES,
    ).generate()


def compare_outputs(actual, expected, layout, length, *, require_zero_tail=True):
    """Compare every vocabulary entry and every valid K/V element per layer."""
    if len(actual) != len(layout.output_names) or len(expected) != len(actual):
        raise AssertionError("All logits and layer K/V outputs must be compared")
    rows = []
    for index, (name, value, reference) in enumerate(zip(layout.output_names, actual, expected)):
        shape = (1, 1, layout.vocab_size) if index == 0 else layout.shape
        if (value.shape != shape or reference.shape != shape
                or value.dtype != np.float32 or reference.dtype != np.float32
                or not np.isfinite(value).all() or not np.isfinite(reference).all()):
            raise AssertionError(f"{name}: invalid shape, dtype or nonfinite value")
        selected = value if index == 0 else value[:, :, :length, :]
        target = reference if index == 0 else reference[:, :, :length, :]
        error = float(np.max(np.abs(selected.astype(np.float64) - target.astype(np.float64))))
        zero_tail = bool(np.all(value[:, :, length:, :] == 0)) if index else None
        if error >= ATOL or (index and require_zero_tail and not zero_tail):
            raise AssertionError(f"{name}: max_abs={error}, future_cache_zero={zero_tail}")
        rows.append(dict(name=name, max_abs=error, elements=int(selected.size),
                         future_cache_zero=zero_tail, passed=True))
    token = int(actual[0][0, 0].argmax())
    reference_token = int(expected[0][0, 0].argmax())
    if token != reference_token:
        raise AssertionError(f"Greedy token differs: {token} != {reference_token}")
    return dict(passed=True, valid_length=length, logits_max_abs=rows[0]["max_abs"],
                valid_kv_max_abs=max(row["max_abs"] for row in rows[1:]),
                greedy_token=token, reference_greedy_token=reference_token,
                outputs=rows)


def reference_cases(directory, layout, report):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    started = time.perf_counter()
    oracle = ort.InferenceSession(str(directory / "prefill-256" / "prefill.onnx"),
                                  options, providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(str(directory / "prefill-256" / "decode.onnx"),
                                   options, providers=["CPUExecutionProvider"])
    report["reference"]["session_start_s"] = time.perf_counter() - started
    references = {}
    traces = directory / "ort"
    traces.mkdir()
    for length in range(1, 6):
        started = time.perf_counter()
        outputs = oracle.run(None, prepare_prefill(PROMPT[:length], layout))
        seconds = time.perf_counter() - started
        references[length] = outputs
        path = traces / f"prefill-{length}.npz"
        np.savez(path, **dict(zip(layout.output_names, outputs)))
        report["reference"]["steps"].append(dict(
            kind="prefill", valid_length=length, ort_s=seconds, outputs=relative(path)))
    history = list(PROMPT)
    previous = [value.copy() for value in references[5]]
    # QEMU clears invalid Prefill cache slots before committing them. Match that
    # contract for the independent Decode oracle, whose input capacity stays 256.
    for value in previous[1:]:
        value[:, :, len(history):, :] = 0
    decode_references = []
    for step in range(2):
        token = int(previous[0][0, 0].argmax())
        started = time.perf_counter()
        decoded = decoder.run(None, prepare_decode(token, len(history), layout, previous[1:]))
        decode_s = time.perf_counter() - started
        history.append(token)
        started = time.perf_counter()
        fresh = oracle.run(None, prepare_prefill(history, layout))
        prefill_s = time.perf_counter() - started
        comparison = compare_outputs(decoded, fresh, layout, len(history))
        path = traces / f"decode-{step + 1}.npz"
        np.savez(path, **dict(zip(layout.output_names, fresh)))
        report["reference"]["steps"].append(dict(
            kind="decode", step=step + 1, input_token=token, valid_length=len(history),
            ort_decode_s=decode_s, ort_full_prefill_s=prefill_s,
            decoded_vs_full_prefill=comparison, outputs=relative(path)))
        decode_references.append((token, fresh))
        previous = decoded
    return references, decode_references


def request_record(session):
    # Deliberately do not copy runtime commands, machine paths or build identity.
    row = session.runs[-1]
    return {name: row[name] for name in (
        "kind", "request", "committed_before", "committed_after", "passed",
        "forward_s", "output_transfer_s", "total_s") if name in row}


def verify_request(session, logits, expected, layout, length):
    started = time.perf_counter()
    cache = session.dump_cache()
    dump_s = time.perf_counter() - started
    started = time.perf_counter()
    comparison = compare_outputs([logits, *cache], expected, layout, length)
    comparison_s = time.perf_counter() - started
    if session.valid_length != length:
        raise AssertionError("Committed cache length differs from the checked prefix")
    return dict(cache_dump_s=dump_s, comparison_s=comparison_s, comparison=comparison)


def prefill_request(session, tokens, expected, layout):
    session.reset()
    reset = request_record(session)
    started = time.perf_counter()
    logits = session.prefill(tokens)
    call_wall_s = time.perf_counter() - started
    row = dict(reset=reset, call_wall_s=call_wall_s, request=request_record(session))
    row.update(verify_request(session, logits, expected, layout, len(tokens)))
    return row


def statistics_report(pairs):
    result = {}
    for variant in VARIANTS:
        result[variant] = {}
        values = [pair["runs"][variant]["call_wall_s"] for pair in pairs]
        result[variant]["call_wall_s"] = dict(median=statistics.median(values),
                                             minimum=min(values), maximum=max(values))
        for metric in ("total_s", "forward_s", "output_transfer_s"):
            values = [pair["runs"][variant]["request"][metric] for pair in pairs]
            result[variant][metric] = dict(median=statistics.median(values),
                                          minimum=min(values), maximum=max(values))
    for metric in ("call_wall_s", "total_s", "forward_s"):
        def value(pair, variant):
            row = pair["runs"][variant]
            return row[metric] if metric == "call_wall_s" else row["request"][metric]
        ratios = [value(pair, VARIANTS[0]) / value(pair, VARIANTS[1]) for pair in pairs]
        result[f"{metric}_speedup"] = dict(
            ratio_of_medians=result[VARIANTS[0]][metric]["median"]
            / result[VARIANTS[1]][metric]["median"],
            paired_ratios=ratios, median_paired_ratio=statistics.median(ratios))
    return result


def run(args, report):
    import torch
    from scripts.export_qwen3_kv_onnx import export_models
    from scratchv.runtime.riscv_session import build_session, QemuSession
    from scratchv.runtime.riscv_tensor import discover_toolchain

    torch.set_num_threads(1)
    directory = args.output_dir
    started = time.perf_counter()
    model, wrappers, layout, exported = export_models(
        directory / VARIANTS[0], seed=args.seed, prefill_length=256)
    short, short_wrappers, short_layout, _ = export_models(
        directory / VARIANTS[1], seed=args.seed, prefill_length=5, mode="prefill")
    if (layout.layers != 2 or layout.vocab_size != 128 or short_layout != layout
            or model.config.hidden_size != 32):
        raise ValueError("This probe accepts only the existing two-layer seeded small Qwen3")
    original_state, short_state = model.state_dict(), short.state_dict()
    if original_state.keys() != short_state.keys() or any(
            not torch.equal(value, short_state[name]) for name, value in original_state.items()):
        raise AssertionError("Prefill exports must use identical seeded weights")
    report["model"] = dict(name="seeded-small-Qwen3", seed=args.seed,
                           layout=layout.to_dict(), hidden_size=32,
                           same_weights=True, packages=exported["packages"])
    report["setup"]["export_s"] = time.perf_counter() - started
    del model, short, wrappers, short_wrappers, original_state, short_state
    references, decode_references = reference_cases(directory, layout, report)
    started = time.perf_counter()
    decoder_path = directory / VARIANTS[0] / "decode.onnx"
    decode = compile_graph(decoder_path, "decode")
    prefills = {variant: compile_graph(directory / variant / "prefill.onnx", "prefill")
                for variant in VARIANTS}
    report["setup"]["codegen_s"] = time.perf_counter() - started
    tools = discover_toolchain()
    executables = {}
    for variant in VARIANTS:
        artifact = prefills[variant]
        executable = build_session(artifact, decode, layout, directory / variant / "build",
                                   tools, cores=4, timeout=args.build_timeout)
        executables[variant] = executable
        report["builds"][variant] = dict(
            prefill_length=executable.prefill_length,
            prefill_graph=relative(directory / variant / "prefill.onnx"),
            decode_graph=relative(decoder_path), shared_decode_artifact=True,
            prefill_workspace_bytes=artifact.workspace_bytes,
            decode_workspace_bytes=decode.workspace_bytes, cache_bank_bytes=layout.bank_bytes,
            allocated_workspace_bytes=executable.layout["regions"]["workspace"]["nbytes"],
            weight_bytes=executable.manifest["total_bytes"],
            compile_s=executable.evidence["compile_seconds"],
            versions=executable.evidence["versions"],
            raw_build_report=relative(directory / variant / "build" / "build.json"))
        save_report(directory, report)
    sessions = {}
    try:
        with ExitStack() as stack:
            for variant in VARIANTS:
                session = QemuSession(executables[variant], directory / variant / "qemu",
                                      timeout=args.timeout)
                sessions[variant] = session
                stack.enter_context(session)
                report["sessions"][variant] = dict(
                    startup_s=session.report["startup_s"],
                    weight_staging_s=session.report["weight_staging_s"],
                    raw_report=relative(session.run_dir / "session.json"))
            # Prefixes 1..5 check every valid token position's complete logits
            # vector against the same 256-position ORT graph. They are excluded
            # from timing statistics, as are the two subsequent Decode requests.
            for variant in VARIANTS:
                session = sessions[variant]
                for length in range(1, 6):
                    row = prefill_request(session, PROMPT[:length], references[length], layout)
                    report["correctness"].append(dict(variant=variant, kind="prefill",
                                                       valid_length=length, **row))
                for step, (token, expected) in enumerate(decode_references, 1):
                    started = time.perf_counter()
                    logits = session.decode(token)
                    call_wall_s = time.perf_counter() - started
                    row = dict(variant=variant, kind="decode", step=step, input_token=token,
                               valid_length=5 + step, call_wall_s=call_wall_s,
                               request=request_record(session))
                    row.update(verify_request(session, logits, expected, layout, 5 + step))
                    report["correctness"].append(row)
            for variant in VARIANTS:
                row = prefill_request(sessions[variant], PROMPT, references[5], layout)
                report["warmups"].append(dict(variant=variant, **row))
            for index in range(args.pairs):
                order = VARIANTS if index % 2 == 0 else tuple(reversed(VARIANTS))
                pair = dict(pair=index + 1, order=list(order), runs={})
                report["pairs"].append(pair)
                for variant in order:
                    pair["runs"][variant] = prefill_request(
                        sessions[variant], PROMPT, references[5], layout)
                save_report(directory, report)
                print(f"Pair {index + 1}/{args.pairs}: " + ", ".join(
                    f"{variant}={pair['runs'][variant]['request']['total_s']:.6f}s"
                    for variant in VARIANTS), flush=True)
    finally:
        for variant, session in sessions.items():
            report["sessions"].setdefault(variant, {}).update(
                closed=session.state == "CLOSED", qemu_reaped=session.report.get("qemu_reaped", False),
                passed=session.report.get("passed", False))
    if not all(row["passed"] for row in report["sessions"].values()):
        raise AssertionError("Both QEMU processes must exit successfully and be reaped")
    report["statistics"] = statistics_report(report["pairs"])
    comparisons = [row["comparison"] for row in report["correctness"] + report["warmups"]]
    comparisons += [row["comparison"] for pair in report["pairs"] for row in pair["runs"].values()]
    report["numeric_summary"] = dict(
        checked_requests=len(comparisons), logits_max_abs=max(c["logits_max_abs"] for c in comparisons),
        valid_kv_max_abs=max(c["valid_kv_max_abs"] for c in comparisons),
        all_greedy_tokens_equal=True, all_future_cache_zero=True, passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path, help="Rejected: this probe is small-model only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cores", type=int, choices=(4,), default=4)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--build-timeout", type=float, default=600.0)
    args = parser.parse_args()
    if args.model_dir is not None:
        parser.error("Full-model/checkpoint inputs are disabled; use the seeded two-layer model")
    if args.pairs < 6 or args.pairs % 2:
        parser.error("--pairs must be even and at least 6 to balance AB/BA order")
    if any(not np.isfinite(value) or value <= 0 for value in (args.timeout, args.build_timeout)):
        parser.error("Timeouts must be finite and positive")
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT / "output") or args.output_dir == ROOT / "output":
        parser.error("--output-dir must be a new directory below this checkout's output/")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(
        schema="scratchv.w6-prefill-short.v1", status="RUNNING", passed=False,
        scope="Two-layer random-weight Qwen3; QEMU/TCG wall time, not full-model or hardware performance",
        configuration=dict(cores=4, dtype="FP32", policies=POLICIES,
                           prompt_token_ids=PROMPT, measured_pairs=args.pairs, warmups_per_variant=1,
                           correctness_prefill_lengths=list(range(1, 6)), decode_steps_after_length_5=2,
                           atol=ATOL, numeric_rule="All finite valid elements: max_abs < atol; greedy tokens equal"),
        timing_scope=dict(
            call_wall_s="Outer Prefill/Decode call wall time including Host input preparation; excludes subsequent K/V dump and comparison",
            total_s="Runtime request wall time: command send, forward, logits QMP transfer and validation; excludes Host input preparation",
            forward_s="Runtime command send to UART response, including guest validation and forward",
            output_transfer_s="Part of total_s: QMP stop, complete logits read/validation and resume",
            excluded="All call/request/forward timing columns exclude export/build, session startup, RESET, debug K/V dump, ORT and comparisons. Host input preparation is included in call_wall_s, but excluded from total_s and forward_s",
            pairing="Two resident sessions; one executes at a time; one warmup each; alternating AB/BA",
            units="Elapsed wall seconds; concurrent threads' CPU times are not added"),
        setup={}, reference=dict(session_start_s=None, graph_optimization="disabled", steps=[]),
        builds={}, sessions={},
        correctness=[], warmups=[], pairs=[],
        raw_directory=relative(args.output_dir),
    )
    started = time.perf_counter()
    try:
        run(args, report)
        report.update(status="PASS", passed=True)
    except BaseException as exc:
        report.update(status="FAIL", passed=False,
                      error=f"{type(exc).__name__}: {str(exc).replace(str(ROOT) + '/', '')}")
        raise
    finally:
        report["probe_elapsed_s"] = time.perf_counter() - started
        save_report(args.output_dir, report)
    print(f"PASS: {relative(args.output_dir / 'report.json')}")


if __name__ == "__main__":
    main()
