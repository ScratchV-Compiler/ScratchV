"""Export the pinned Qwen3-0.6B to FP32 ONNX and verify it on CPU.

Run with the packages in requirements/qwen3-export.txt. Model downloads are
handled separately; --model-dir must contain the complete pinned snapshot.
Each expensive stage runs in its own process to release model memory.
Export into an empty --output-dir; existing artifacts are never overwritten.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
CHECKPOINT_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
SEQ = 256
VOCAB = 151936


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path,
                        default=ROOT / "models" / "qwen3-source" / REVISION)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "models" / "qwen3-0.6b-onnx")
    parser.add_argument("--work-dir", type=Path,
                        default=ROOT / "output" / "qwen3-export")
    parser.add_argument("--stage", choices=("all", "reference", "export", "pack", "verify"),
                        default="all")
    parser.add_argument("--exporter", choices=("dynamo", "legacy"), default="dynamo")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--require-full-allclose", action="store_true",
                        help="Also require padded logits to meet atol=rtol=1e-4")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    for name in ("model_dir", "output_dir", "work_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.stage in ("all", "export") and args.output_dir.exists():
        if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
            parser.error("Export requires an empty --output-dir; choose a new directory "
                         "to preserve existing model files and verification results")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    return args


def load_wrapper(args):
    import torch
    from transformers import AutoModelForCausalLM

    if sha256(args.model_dir / "model.safetensors") != CHECKPOINT_SHA256:
        raise ValueError("Source weights do not match the pinned official checkpoint")
    torch.set_num_threads(args.threads)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.float32,
        attn_implementation="eager", local_files_only=True,
    ).eval()
    expected = {"num_hidden_layers": 28, "hidden_size": 1024,
                "num_attention_heads": 16, "num_key_value_heads": 8,
                "head_dim": 128, "vocab_size": VOCAB}
    for key, value in expected.items():
        if getattr(model.config, key) != value:
            raise ValueError(f"Unexpected model config: {key}")
    model.config.use_cache = False

    class Forward(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            self.register_buffer("positions", torch.arange(SEQ).reshape(1, SEQ),
                                 persistent=False)

        def forward(self, input_ids, attention_mask):
            return self.model(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=self.positions, use_cache=False,
                output_attentions=False, output_hidden_states=False,
                return_dict=False, logits_to_keep=0,
            )[0]

    return Forward().eval()


def reference(args):
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        raise ValueError("The pinned tokenizer must define a padding token")
    prompt = "The capital of France is"
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    cases = [tokens, np.random.default_rng(20260929).integers(0, VOCAB, SEQ).tolist()]
    wrapper = load_wrapper(args)
    for index, values in enumerate(cases):
        n = len(values)
        ids = np.full((1, SEQ), tokenizer.pad_token_id, dtype=np.int64)
        ids[0, :n] = values
        allowed = (np.arange(SEQ)[None, :] <= np.arange(SEQ)[:, None])
        allowed &= np.arange(SEQ)[None, :] < n
        mask = np.where(allowed, np.float32(0), np.finfo(np.float32).min)
        mask = mask.reshape(1, 1, SEQ, SEQ)
        np.savez(args.work_dir / f"input_{index}.npz", input_ids=ids,
                 attention_mask=mask, valid_length=np.int64(n))
        with torch.inference_mode():
            logits = wrapper(torch.from_numpy(ids), torch.from_numpy(mask)).numpy()
        if logits.shape != (1, SEQ, VOCAB) or not np.isfinite(logits).all():
            raise ValueError("Invalid PyTorch reference output")
        np.save(args.work_dir / f"reference_{index}.npy", logits)
        print(f"reference case={index} valid_length={n} shape={logits.shape}", flush=True)


def export(args):
    import numpy as np
    import torch

    wrapper = load_wrapper(args)
    with np.load(args.work_dir / "input_0.npz") as inputs:
        tensors = tuple(torch.from_numpy(inputs[name].copy())
                        for name in ("input_ids", "attention_mask"))
    path = args.output_dir / "model.onnx"
    with torch.inference_mode():
        if args.exporter == "dynamo":
            program = torch.onnx.export(
                wrapper, tensors, dynamo=True, opset_version=18,
                input_names=["input_ids", "attention_mask"], output_names=["logits"],
                dynamic_shapes=None, report=True, optimize=False,
                artifacts_dir=str(args.work_dir),
            )
            program.save(str(path), external_data=True)
        else:
            torch.onnx.export(
                wrapper, tensors, str(path), dynamo=False, opset_version=18,
                input_names=["input_ids", "attention_mask"], output_names=["logits"],
                dynamic_axes=None, external_data=True,
            )
    print(f"exported {path}", flush=True)


def external_tensors(message):
    """Walk tensors in graphs, attributes and local functions without weights."""
    if message.DESCRIPTOR.full_name == "onnx.TensorProto":
        if message.external_data:
            yield message
        return
    for field, value in message.ListFields():
        if field.message_type is not None:
            repeated = (field.is_repeated if hasattr(field, "is_repeated")
                        else field.label == field.LABEL_REPEATED)
            if repeated:
                for item in value:
                    yield from external_tensors(item)
            else:
                yield from external_tensors(value)


def pack(args):
    """Stream weights into <=1 GiB LFS files, without loading them into RAM."""
    import onnx

    path = args.output_dir / "model.onnx"
    model = onnx.load(str(path), load_external_data=False)
    limit = 1024 ** 3
    originals, packed, reused = set(), set(), {}
    destination, name, size, shard = None, None, 0, 0
    try:
        for tensor in external_tensors(model):
            info = {entry.key: entry.value for entry in tensor.external_data}
            source = (path.parent / info["location"]).resolve()
            if not source.is_relative_to(path.parent) or not source.is_file():
                raise ValueError(f"Invalid source data path: {source}")
            start = int(info.get("offset", "0"))
            length = int(info.get("length", source.stat().st_size - start))
            if length > limit or start < 0 or length < 0:
                raise ValueError(f"Tensor cannot fit in one LFS shard: {tensor.name}")
            originals.add(source)
            key = (source, start, length)
            if key not in reused:
                if destination is None or size + length > limit:
                    if destination is not None:
                        destination.close()
                    shard += 1
                    name = f"weights-{shard:05d}.data"
                    target = path.parent / name
                    # Exclusive creation avoids overwriting existing artifacts.
                    destination = target.open("xb")
                    packed.add(target)
                    size = 0
                reused[key] = (name, size, length)
                with source.open("rb") as stream:
                    stream.seek(start)
                    remaining = length
                    while remaining:
                        chunk = stream.read(min(8 * 1024 * 1024, remaining))
                        if not chunk:
                            raise EOFError(f"Truncated external data: {source}")
                        destination.write(chunk)
                        remaining -= len(chunk)
                size += length
            filename, offset, count = reused[key]
            del tensor.external_data[:]
            for field, value in (("location", filename), ("offset", offset), ("length", count)):
                entry = tensor.external_data.add()
                entry.key, entry.value = field, str(value)
    finally:
        if destination is not None:
            destination.close()
    temporary = path.with_name("model.packed.onnx")
    onnx.save_model(model, str(temporary))
    onnx.checker.check_model(str(temporary))
    temporary.replace(path)
    # Sources are generated export artifacts inside this exact output directory.
    for source in originals - packed:
        source.unlink()
    print(f"packed external weights into {shard} LFS shards", flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(args):
    from collections import Counter
    import numpy as np
    import onnx
    import onnxruntime as ort

    path = args.output_dir / "model.onnx"
    onnx.checker.check_model(str(path))
    model = onnx.load(str(path), load_external_data=False)
    external = sorted({item.value for tensor in external_tensors(model)
                       for item in tensor.external_data if item.key == "location"})
    files = [path]
    for name in external:
        data_path = (path.parent / name).resolve()
        if not data_path.is_relative_to(path.parent) or not data_path.is_file():
            raise ValueError(f"Invalid external data file: {name}")
        # All generated large artifacts must match the repository's LFS rules.
        if data_path.suffix != ".data":
            raise ValueError(f"External data needs a .data filename for LFS: {name}")
        files.append(data_path)

    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    expected = {"input_ids": ([1, SEQ], "tensor(int64)"),
                "attention_mask": ([1, 1, SEQ, SEQ], "tensor(float)")}
    actual = {x.name: (x.shape, x.type) for x in session.get_inputs()}
    if actual != expected:
        raise ValueError(f"Unexpected ONNX inputs: {actual}")
    outputs = session.get_outputs()
    if len(outputs) != 1 or outputs[0].name != "logits":
        raise ValueError("Expected logits as the only ONNX output")
    cases = []
    for index in range(2):
        with np.load(args.work_dir / f"input_{index}.npz") as inputs:
            feed = {name: inputs[name] for name in expected}
            valid_length = inputs["valid_length"]
            if (valid_length.ndim != 0 or valid_length.dtype.kind not in "iu"
                    or not 1 <= int(valid_length) <= SEQ):
                raise ValueError("Reference valid_length must be an integer scalar in [1, sequence length]")
            n = int(valid_length)
        start = time.perf_counter()
        result = session.run(["logits"], feed)[0]
        elapsed = time.perf_counter() - start
        reference = np.load(args.work_dir / f"reference_{index}.npy", mmap_mode="r")
        if result.shape != (1, SEQ, VOCAB) or result.dtype != np.float32:
            raise ValueError(f"Unexpected output: {result.shape}/{result.dtype}")
        if reference.shape != result.shape or reference.dtype != np.float32:
            raise ValueError(f"Unexpected PyTorch reference: {reference.shape}/{reference.dtype}")
        max_abs, total, full_passed = 0.0, 0.0, True
        valid_max, valid_passed, finite = 0.0, True, True
        for begin in range(0, SEQ, 8):
            a = result[:, begin:begin + 8]
            b = reference[:, begin:begin + 8]
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("ONNX output and PyTorch reference must both be finite")
            # A nonfinite oracle can otherwise satisfy inf <= inf. Compute
            # metrics in small FP64 chunks so opposite finite FP32 extrema do
            # not overflow the subtraction or produce invalid JSON evidence.
            difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
            max_abs = max(max_abs, float(difference.max()))
            total += float(difference.sum(dtype=np.float64))
            finite &= bool(np.isfinite(a).all())
            full_passed &= bool(np.all(difference <= 1e-4 + 1e-4 * np.abs(b)))
            if begin < n:
                count = min(8, n - begin)
                valid_diff = difference[:, :count]
                valid_max = max(valid_max, float(valid_diff.max()))
                valid_passed &= bool(np.all(valid_diff <= 1e-4 + 1e-4 * np.abs(b[:, :count])))
        passed = finite and valid_passed and (full_passed or not args.require_full_allclose)
        cases.append({"case": index, "valid_length": n, "shape": list(result.shape),
                      "max_abs_error": max_abs, "mean_abs_error": total / result.size,
                      "valid_token_max_abs_error": valid_max,
                      "full_tensor_allclose": full_passed,
                      "valid_token_allclose": valid_passed, "all_outputs_finite": finite,
                      "allclose_atol": 1e-4, "allclose_rtol": 1e-4, "passed": passed,
                      "seconds": elapsed,
                      "last_valid_top1_ort": int(result[0, n - 1].argmax()),
                      "last_valid_top1_pytorch": int(reference[0, n - 1].argmax())})
        print(json.dumps(cases[-1]), flush=True)
        del result, reference
    packages = ["torch", "transformers", "onnx", "onnxruntime", "onnxscript", "onnx-ir",
                "numpy", "protobuf", "huggingface-hub", "safetensors", "tokenizers"]
    report = {
        "model_id": "Qwen/Qwen3-0.6B", "revision": REVISION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint_sha256": CHECKPOINT_SHA256,
        "dtype": "float32", "use_cache": False, "exporter": args.exporter,
        "attention": "eager", "padding": "right", "position_ids": "0..255",
        "mask": "4D additive causal + padding; 0 / finfo(float32).min",
        "provider": "CPUExecutionProvider", "threads": args.threads,
        "validation_scope": "All output shapes/finite values; numerical gate on valid tokens. "
                            "Full-tensor allclose is independently reported, including padding.",
        "require_full_allclose": args.require_full_allclose,
        "packages": {p: importlib.metadata.version(p) for p in packages},
        "opsets": {p.domain: p.version for p in model.opset_import},
        "ir_version": model.ir_version,
        "operators": dict(sorted(Counter(n.domain + ":" + n.op_type
                                          for n in model.graph.node).items())),
        "local_functions": len(model.functions), "cases": cases,
        "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)}
                  for p in files],
        "passed": all(c["passed"] for c in cases),
        "full_tensor_allclose": all(c["full_tensor_allclose"] for c in cases),
    }
    (args.output_dir / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError("ONNX/PyTorch numeric comparison failed; see verification.json")


def main():
    args = arguments()
    if args.stage != "all":
        globals()[args.stage](args)
        return
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    for stage in ("reference", "export", "pack", "verify"):
        command = [sys.executable, str(Path(__file__).resolve()), "--stage", stage,
                   "--model-dir", str(args.model_dir), "--output-dir", str(args.output_dir),
                   "--work-dir", str(args.work_dir), "--exporter", args.exporter,
                   "--threads", str(args.threads)]
        if args.require_full_allclose:
            command.append("--require-full-allclose")
        print(f"=== {stage} ===", flush=True)
        subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    main()
