"""Independent metric formulas, extreme ranges and unchanged gate decisions."""
import json
import math

import numpy as np
import pytest

from scratchv.verification.numeric_metrics import numeric_metrics


@pytest.mark.parametrize("actual,reference,cosine,relative", [
    ([3, 4], [3, 4], 1, 0),
    ([6, 8], [3, 4], 1, 1),
    ([-3, -4], [3, 4], -1, 2),
    ([1, 0], [0, 1], 0, math.sqrt(2)),
    ([0, 0], [3, 4], None, 1),
    ([0, 0], [0, 0], None, 0),
    ([3, 4], [0, 0], None, None),
])
def test_analytic_vectors(actual, reference, cosine, relative):
    result = numeric_metrics(np.array(actual, dtype=np.float64),
                             np.array(reference, dtype=np.float64), block_elements=1)
    for key, value in (("cosine_similarity", cosine), ("relative_l2", relative)):
        if value is None:
            assert result[key] is None and result[key + "_reason"]
        else:
            assert result[key] == pytest.approx(value, rel=2e-15, abs=1e-15)
            assert result[key + "_reason"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("scale", [np.finfo(np.float64).max, 1e-300,
                                    np.nextafter(np.float64(0), np.float64(1))])
def test_squares_and_differences_do_not_overflow_or_erase_tiny_values(scale):
    reference = np.array([scale, scale], dtype=np.float64)
    opposite = numeric_metrics(-reference, reference, block_elements=1)
    assert opposite["relative_l2"] == pytest.approx(2)
    assert opposite["cosine_similarity"] == pytest.approx(-1)
    orthogonal = numeric_metrics(np.array([scale, -scale]), reference)
    assert orthogonal["relative_l2"] == pytest.approx(math.sqrt(2))
    assert orthogonal["cosine_similarity"] == 0


def test_each_vector_has_its_own_scale_for_cosine():
    maximum = np.finfo(np.float64).max
    minimum = np.nextafter(np.float64(0), np.float64(1))
    result = numeric_metrics(np.array([maximum]), np.array([minimum]))
    assert result["cosine_similarity"] == 1
    assert result["relative_l2"] is None
    assert "exceeds" in result["relative_l2_reason"]
    result = numeric_metrics(np.array([maximum, minimum]), np.array([maximum, 0]))
    assert result["cosine_similarity"] == 1
    assert result["relative_l2"] is None
    assert "below" in result["relative_l2_reason"]


@pytest.mark.parametrize("dtype,base", [(np.int64, 2**63 - 2), (np.uint64, 2**64 - 2)])
def test_integer_neighbours_keep_nonzero_difference_above_fp64_exact_range(dtype, base):
    actual, expected = np.array([base + 1], dtype), np.array([base], dtype)
    result = numeric_metrics(actual, expected)
    assert result["relative_l2"] == pytest.approx(1 / base, rel=2e-15, abs=0)
    assert result["relative_l2"] > 0
    assert result["cosine_similarity"] == 1


def test_signed_integer_subtraction_cannot_wrap():
    limits = np.iinfo(np.int64)
    result = numeric_metrics(np.array([limits.min]), np.array([limits.max]))
    assert result["relative_l2"] == pytest.approx((2**64 - 1) / (2**63 - 1))
    assert result["cosine_similarity"] == -1


@pytest.mark.parametrize("actual,expected,reason", [
    (None, np.ones(1), "NumPy arrays"),
    (np.ones(2), np.ones(1), "shape mismatch"),
    (np.ones(1, np.float32), np.ones(1, np.float64), "dtype mismatch"),
    (np.ones(1, complex), np.ones(1, complex), "real numeric"),
    (np.empty((2, 0)), np.empty((2, 0)), "empty"),
    (np.array([1, np.nan]), np.ones(2), "nonfinite"),
    (np.array([1, np.inf]), np.array([1, np.inf]), "nonfinite"),
])
def test_undefined_inputs_have_explicit_reasons(actual, expected, reason):
    result = numeric_metrics(actual, expected)
    for key in ("relative_l2", "cosine_similarity"):
        assert result[key] is None and reason in result[key + "_reason"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("block", [0, -1, True, 1.5])
def test_bad_block_size_is_rejected(block):
    with pytest.raises(ValueError, match="positive integer"):
        numeric_metrics(np.ones(1), np.ones(1), block_elements=block)


@pytest.mark.parametrize("block", [1, 7, 262144])
def test_strided_memmap_matches_independent_vector_formula(tmp_path, block):
    source = np.lib.format.open_memmap(tmp_path / "arrays.npy", mode="w+",
                                       dtype=np.float32, shape=(2, 23, 37))
    source[:] = np.random.default_rng(487).normal(size=source.shape)
    actual, expected = source[0, ::2, ::-3].T, source[1, ::2, ::-3].T
    assert not actual.flags.c_contiguous
    a, b = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
    result = numeric_metrics(actual, expected, block_elements=block)
    assert result["relative_l2"] == pytest.approx(np.linalg.norm(a - b) / np.linalg.norm(b), rel=3e-15)
    assert result["cosine_similarity"] == pytest.approx(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b),
                                                       rel=3e-14, abs=1e-16)


def test_iteration_never_exceeds_requested_buffer(monkeypatch):
    import scratchv.verification.numeric_metrics as module
    original = module._chunks
    lengths = []

    def observed(*args):
        for a, b in original(*args):
            lengths.append(a.size)
            yield a, b

    monkeypatch.setattr(module, "_chunks", observed)
    array = np.arange(20000, dtype=np.float32).reshape(100, 200).T[:, ::2]
    result = numeric_metrics(array, array + 1, block_elements=127)
    assert result["relative_l2"] > 0
    assert len(lengths) > 2 and max(lengths) <= 127


@pytest.mark.parametrize("gate", ["w2", "w3"])
def test_new_metrics_do_not_weaken_absolute_gate(gate):
    from probes.w2_qwen3_small.diagnostics import tensor_diff
    from probes.w3_qwen3_full.comparison import compare_tensor
    compare = tensor_diff if gate == "w2" else compare_tensor
    actual = np.array([2, 4], dtype=np.float32)
    expected = np.array([1, 2], dtype=np.float32)
    result = compare(actual, expected)
    assert not result["passed"] and result["max_abs"] == 2
    assert result["cosine_similarity"] == 1
    assert result["relative_l2"] == 1
    passed = compare(actual, actual)
    assert passed["passed"] and passed["max_abs"] == 0
    assert passed["cosine_similarity"] == 1 and passed["relative_l2"] == 0
    invalid = compare(actual, expected.astype(np.float64))
    assert not invalid["passed"]
    for key in ("relative_l2", "cosine_similarity"):
        assert invalid[key] is None and invalid[key + "_reason"]


def test_empty_gate_policies_are_preserved():
    from probes.w2_qwen3_small.diagnostics import tensor_diff
    from probes.w3_qwen3_full.comparison import compare_tensor
    array = np.empty(0, dtype=np.float32)
    w2, w3 = tensor_diff(array, array), compare_tensor(array, array)
    assert w2["passed"] and not w3["passed"]
    for row in (w2, w3):
        assert row["relative_l2"] is None and row["cosine_similarity"] is None
        assert row["relative_l2_reason"] and row["cosine_similarity_reason"]
