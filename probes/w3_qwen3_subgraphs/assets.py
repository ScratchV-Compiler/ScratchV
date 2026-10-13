"""Authenticate an existing snapshot before reading individual tensors.

No model constructor or download is used. Checkpoint hashing is streamed and
safetensors only materializes tensors explicitly requested by each subgraph.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "probes/w1_qwen3_export/manifest.json"
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
DIMENSIONS = {"hidden_size": 1024, "intermediate_size": 3072,
              "num_attention_heads": 16, "num_key_value_heads": 8,
              "head_dim": 128, "num_hidden_layers": 28,
              "rms_norm_eps": 1e-6, "rope_theta": 1000000}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(directory, manifest=MANIFEST):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Pinned snapshot directory missing: {directory}")
    rows = []
    for item in [{"name": "model.safetensors", "sha256": manifest["source_checkpoint_sha256"]},
                 *manifest["source_files"]]:
        name = item["name"]
        if not name or name in (".", "..") or any(c in name for c in "/\\:"):
            raise ValueError(f"Non-flat snapshot asset name: {name}")
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise FileNotFoundError(f"Missing snapshot asset: {name}")
        actual = sha256(path)
        if actual != item["sha256"]:
            raise ValueError(f"Pinned snapshot SHA256 mismatch: {name}: {actual}")
        rows.append({"name": name, "sha256": actual, "bytes": path.stat().st_size})
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    for key, expected in DIMENSIONS.items():
        if config.get(key) != expected:
            raise ValueError(f"Pinned Qwen3 configuration mismatch: {key}")
    return {"directory": str(directory), "model_id": manifest["model_id"],
            "revision": manifest["revision"], "manifest_sha256": sha256(MANIFEST_PATH),
            "verification": "W1 checkpoint and all seven source-file SHA256 values match; offline/read-only",
            "files": rows, "config": config}


class Weights:
    """Read selected first-layer weights, recording original dtype and FP32 bytes."""

    def __init__(self, source):
        self.path = Path(source) / "model.safetensors"
        self.evidence = {}

    def get(self, suffix, shape):
        from safetensors import safe_open
        name = "model.layers.0." + suffix
        with safe_open(str(self.path), framework="pt", device="cpu") as checkpoint:
            tensor = checkpoint.get_tensor(name)
            original_dtype = str(tensor.dtype)
            array = tensor.float().numpy().copy()
        if array.shape != tuple(shape) or not np.isfinite(array).all():
            raise ValueError(f"Unexpected/nonfinite source weight: {name}: {array.shape}")
        self.evidence[name] = {"shape": list(array.shape), "source_dtype": original_dtype,
                               "execution_dtype": str(array.dtype), "fp32_sha256":
                               hashlib.sha256(array.tobytes()).hexdigest(), "bytes": array.nbytes}
        return array
