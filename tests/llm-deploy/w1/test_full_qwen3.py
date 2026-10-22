"""Pinned complete FP32 Qwen3 assets and the original two ORT forwards."""
import numpy as np
import pytest

from probes.w1_qwen3_export import run as artifacts


def test_qwen3_assets_and_onnx_structure(qwen3_model_dir, qwen3_model_files):
    artifacts.environment()
    result = artifacts.inspect_model(qwen3_model_dir, qwen3_model_files)
    assert result["external_files"] == [f"weights-{i:05d}.data" for i in (1, 2, 3)]
    assert result["external_tensors"] > 0


@pytest.fixture(scope="module")
def complete_session(qwen3_model_path, ort_session):
    return ort_session(qwen3_model_path, disable_optimizations=True)


def test_qwen3_static_interface(complete_session):
    assert {v.name: (v.type, v.shape) for v in complete_session.get_inputs()} == {
        "input_ids": ("tensor(int64)", [1, 256]),
        "attention_mask": ("tensor(float)", [1, 1, 256, 256]),
    }
    assert {v.name: (v.type, v.shape) for v in complete_session.get_outputs()} == {
        "logits": ("tensor(float)", [1, 256, 151936]),
    }


@pytest.mark.parametrize("length,seed", [(5, 20260929), (256, 20260930)], ids=["short", "full"])
def test_qwen3_ort_forward(complete_session, length, seed):
    logits = complete_session.run(["logits"], artifacts.make_inputs(length, seed))[0]
    assert logits.shape == (1, 256, 151936)
    assert logits.dtype == np.float32
    for start in range(0, 256, 8):
        assert np.isfinite(logits[:, start:start + 8]).all(), start
