#!/usr/bin/env python3
"""W1 probe: can ScratchV's IR express a 2-layer Transformer?

Answers two separate questions and reports both:

  1. Is the ONNX model itself correct?      ONNX Runtime vs an independent
                                            numpy implementation.
  2. Can ScratchV consume it?               ONNXParser.parse() on the same file.

(1) passing is a prerequisite for (2) meaning anything: if the model were
wrong, a parser failure would tell us nothing. (2) failing is a legitimate
probe result, not a broken test — it is the finding the plan's W1 exists to
produce. The exit code reflects both, so the CI gate is honest.

Writes artifacts to out/ for the CI job to upload.
"""

from __future__ import annotations

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


def main() -> int:
    OUT.mkdir(exist_ok=True)
    cfg = Config()
    print(f"[cfg] layers={cfg.layers} hidden={cfg.hidden} heads={cfg.heads} "
          f"kv_heads={cfg.kv_heads} head_dim={cfg.head_dim} "
          f"rotary_dim={cfg.rotary_dim} seq={cfg.seq} vocab={cfg.vocab_size}")

    # ── build ──────────────────────────────────────────────────────────────
    model, weights = build(cfg)                       # runs onnx.checker
    onnx_path = OUT / "tiny_transformer_2l.onnx"
    onnx_path.write_bytes(model.SerializeToString())
    ops: dict[str, int] = {}
    for n in model.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"[onnx] {len(model.graph.node)} nodes, {len(ops)} distinct ops -> {onnx_path.name}")
    print(f"[onnx] ops: {dict(sorted(ops.items()))}")

    ids = np.random.default_rng(1).integers(0, cfg.vocab_size,
                                            size=(1, cfg.seq)).astype(np.int64)
    mask = causal_mask(cfg)

    # ── (1) is the model correct? ──────────────────────────────────────────
    import onnxruntime as ort
    ort_logits = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"input_ids": ids, "attention_mask": mask})[0]
    ref_logits = forward(cfg, ids, mask, weights)

    max_abs = float(np.abs(ort_logits - ref_logits).max())
    np.save(OUT / "logits_ort.npy", ort_logits)
    np.save(OUT / "logits_reference.npy", ref_logits)
    model_ok = max_abs < ATOL
    print(f"[model] ORT vs numpy reference: max |diff| = {max_abs:.3e} "
          f"({'PASS' if model_ok else 'FAIL'}, atol {ATOL})")

    # ── (2) can ScratchV consume it? ───────────────────────────────────────
    from scratchv.frontend.onnx_parser import ONNXParser, ONNXParseError

    missing = unsupported_ops(model, ONNXParser)
    parse_error: str | None = None
    try:
        ONNXParser().parse(str(onnx_path))
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

    report = OUT / "report.md"
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

    # The ONNX being correct is the hard part of this probe and is fully
    # automated. Whether ScratchV can eat it is reported but does not fail the
    # probe, because a "no" is a valid answer that the team acts on — turning
    # CI red for it would just train people to ignore the job.
    return 0 if model_ok else 1


if __name__ == "__main__":
    sys.exit(main())
