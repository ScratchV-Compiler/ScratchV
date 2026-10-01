#!/usr/bin/env python3
"""W1 probe: can ScratchV's IR express a 2-layer Transformer?

Checks the model and the actual ScratchV execution path:

  1. Is the ONNX model itself correct?      ONNX Runtime vs an independent
                                            numpy implementation.
  2. Can ScratchV consume it?               ONNX -> IR -> IRInterpreter vs ORT.

(1) passing is a prerequisite for (2) meaning anything. Either failing
returns a nonzero exit code. --model consumes an existing ONNX artifact;
otherwise the deterministic probe model is generated as before.

Writes artifacts to out/ for the CI job to upload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from tiny_transformer import Config, build, causal_mask          # noqa: E402
from reference import forward                                    # noqa: E402

OUT = HERE / "out"
ATOL = 1e-5


def unsupported_ops(model, parser_cls) -> list[str]:
    """Every ONNX op in the graph with no `_handle_<op>` on the parser."""
    return sorted({
        n.op_type for n in model.graph.node
        if not hasattr(parser_cls, f"_handle_{n.op_type.lower()}")
    })


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", type=Path, help="Validate an existing 2-layer probe ONNX")
    cli.add_argument("--output-dir", type=Path, default=OUT)
    cli.add_argument("--seed", type=int, default=1, help="Input token RNG seed")
    args = cli.parse_args(argv)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    print(f"[cfg] layers={cfg.layers} hidden={cfg.hidden} heads={cfg.heads} "
          f"kv_heads={cfg.kv_heads} head_dim={cfg.head_dim} "
          f"rotary_dim={cfg.rotary_dim} seq={cfg.seq} vocab={cfg.vocab_size}")

    # ── build ──────────────────────────────────────────────────────────────
    import onnx
    model, weights = build(cfg)                       # runs onnx.checker
    if args.model is None:
        onnx_path = out / "tiny_transformer_2l.onnx"
        onnx_path.write_bytes(model.SerializeToString())
    else:
        onnx_path = args.model.resolve()
        with onnx_path.open("rb") as stream:
            if stream.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
                cli.error("--model is an LFS pointer; run git lfs pull first")
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model, full_check=True)
    ops: dict[str, int] = {}
    for n in model.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"[onnx] {len(model.graph.node)} nodes, {len(ops)} distinct ops -> {onnx_path.name}")
    print(f"[onnx] ops: {dict(sorted(ops.items()))}")

    ids = np.random.default_rng(args.seed).integers(0, cfg.vocab_size,
                                            size=(1, cfg.seq)).astype(np.int64)
    mask = causal_mask(cfg)

    # ── (1) is the model correct? ──────────────────────────────────────────
    import onnxruntime as ort
    ort_logits = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"input_ids": ids, "attention_mask": mask})[0]
    ref_logits = forward(cfg, ids, mask, weights)

    max_abs = float(np.abs(ort_logits - ref_logits).max())
    np.save(out / "logits_ort.npy", ort_logits)
    np.save(out / "logits_reference.npy", ref_logits)
    model_ok = (ort_logits.shape == ref_logits.shape
                and np.isfinite(ort_logits).all()
                and np.isfinite(ref_logits).all() and max_abs < ATOL)
    print(f"[model] ORT vs numpy reference: max |diff| = {max_abs:.3e} "
          f"({'PASS' if model_ok else 'FAIL'}, atol {ATOL})")

    # ── (2) can ScratchV consume it? ───────────────────────────────────────
    from scratchv.frontend.onnx_parser import ONNXParser, ONNXParseError

    missing = unsupported_ops(model, ONNXParser)
    parse_error: str | None = None
    ir_error: str | None = None
    program = None
    parser = ONNXParser()
    ir_ok = False
    ir_max_abs = None
    executed_steps = None
    try:
        program = parser.parse(str(onnx_path))
        print("[ir]    ONNXParser.parse() succeeded")
    except ONNXParseError as exc:
        parse_error = str(exc)
        print(f"[ir]    ONNXParser.parse() failed: {parse_error}")
    except Exception as exc:                          # noqa: BLE001
        parse_error = f"{type(exc).__name__}: {exc}"
        print(f"[ir]    ONNXParser.parse() raised unexpectedly: {parse_error}")
        traceback.print_exc()

    if missing:
        print(f"[ir]    ops with no handler ({len(missing)}): {missing}")

    if program is not None:
        from scratchv.verification.ir_interpreter import IRInterpreter
        try:
            result = IRInterpreter(program).run(
                {"input_ids": ids, "attention_mask": mask},
                initializers=parser.initializers,
            )
            ir_logits = result.return_value
            executed_steps = result.executed_steps
            if (ir_logits is None or ir_logits.shape != ort_logits.shape
                    or ir_logits.dtype != ort_logits.dtype):
                raise ValueError("IR output shape/dtype disagrees with ORT")
            ir_max_abs = float(np.abs(ir_logits - ort_logits).max())
            ir_ok = bool(np.isfinite(ir_logits).all() and ir_max_abs < ATOL)
            np.save(out / "logits_ir.npy", ir_logits)
            print(f"[ir]    IRInterpreter vs ORT: max |diff| = {ir_max_abs:.3e} "
                  f"({'PASS' if ir_ok else 'FAIL'}, steps={executed_steps})")
        except Exception as exc:                      # noqa: BLE001
            ir_error = f"{type(exc).__name__}: {exc}"
            print(f"[ir]    IRInterpreter failed: {ir_error}")

    passed = bool(model_ok and parse_error is None and ir_ok)
    summary = {
        "model_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "input_seed": args.seed, "nodes": len(model.graph.node),
        "operators": dict(sorted(ops.items())), "unsupported_ops": missing,
        "model_ok": bool(model_ok), "model_max_abs": max_abs,
        "parse_error": parse_error, "ir_error": ir_error,
        "ir_ok": ir_ok, "ir_max_abs": ir_max_abs,
        "executed_steps": executed_steps, "atol": ATOL, "passed": passed,
    }
    (out / "report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = out / "report.md"
    report.write_text(
        "# probe:small-transformer\n\n"
        f"- model: {cfg.layers} layers, hidden {cfg.hidden}, "
        f"{cfg.heads} heads / {cfg.kv_heads} kv_heads, head_dim {cfg.head_dim}\n"
        f"- graph: {len(model.graph.node)} nodes, {len(ops)} distinct ops\n"
        f"- ONNX vs numpy reference: max |diff| = {max_abs:.3e} "
        f"({'PASS' if model_ok else 'FAIL'})\n"
        f"- ScratchV ONNXParser: {'PASS' if parse_error is None else 'FAIL'}"
        f"{'' if parse_error is None else ' — ' + parse_error}\n"
        f"- unsupported ops ({len(missing)}): {', '.join(missing) or 'none'}\n",
        encoding="utf-8",
    )

    with report.open("a", encoding="utf-8") as stream:
        stream.write(
            f"- IRInterpreter vs ORT: {'PASS' if ir_ok else 'FAIL'}, "
            f"max |diff| = {ir_max_abs}, executed steps = {executed_steps}\n"
            f"- interpreter error: {ir_error or 'none'}\n"
            f"- input seed: {args.seed}\n"
            f"- overall: {'PASS' if passed else 'FAIL'}\n"
            "- scope: fixed synthetic probe; no Q/K RMSNorm, partial RoPE; "
            "not a full Qwen3 or RISC-V execution test\n"
        )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
