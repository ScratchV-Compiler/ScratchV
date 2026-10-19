"""W1 synthetic two-layer Transformer: independent NumPy, ORT and IR."""
import numpy as np
import pytest

from probes.w1_tiny_transformer.tiny_transformer import causal_mask


@pytest.fixture(scope="module")
def tiny_results(two_layer_transformer, two_layer_transformer_path, ort_session):
    config = two_layer_transformer["config"]
    ids = np.random.default_rng(1).integers(
        0, config.vocab_size, size=(1, config.seq),
    ).astype(np.int64)
    feed = {"input_ids": ids, "attention_mask": causal_mask(config)}
    return feed, ort_session(two_layer_transformer_path).run(None, feed)[0]


def test_tiny_transformer_against_numpy(two_layer_transformer, tiny_results, monkeypatch,
                                      repository_root):
    # The preserved standalone reference imports its sibling by its short name.
    monkeypatch.syspath_prepend(str(repository_root / "probes/w1_tiny_transformer"))
    from probes.w1_tiny_transformer.reference import forward

    feed, actual = tiny_results
    expected = forward(two_layer_transformer["config"], feed["input_ids"],
                       feed["attention_mask"], two_layer_transformer["weights"])
    # The original independent NumPy reference uses FP64 intermediates.
    assert actual.dtype == np.float32
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    assert np.max(np.abs(actual.astype(np.float64) - expected)) < 1e-5


def test_tiny_transformer_ir(two_layer_transformer_path, tiny_results, run_ir, assert_tensor):
    feed, expected = tiny_results
    assert_tensor(run_ir(two_layer_transformer_path, feed), expected)
