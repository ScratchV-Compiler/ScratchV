"""Independent numerical and shape checks for deterministic FP32 kernels."""

import numpy as np
import pytest

from scratchv.verification.fp32_eigen import mean_last, trig


@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17, 31, 32, 127, 128, 255, 1024, 3072])
@pytest.mark.parametrize("keepdims", [True, False])
def test_mean_accuracy_including_packet_prefixes_and_tails(width, keepdims):
    rng = np.random.default_rng(2201 + width)
    values = rng.uniform(-3, 5, (2, 7, width)).astype(np.float32)
    actual = mean_last(values, keepdims=keepdims)
    reference = np.mean(values.astype(np.float64), axis=-1, keepdims=keepdims)
    assert actual.dtype == np.float32
    assert actual.shape == reference.shape
    np.testing.assert_allclose(actual, reference, atol=6e-7, rtol=6e-7)


@pytest.mark.parametrize("width", [1, 3, 5, 8, 17])
def test_mean_scalar_result_and_empty_leading_dimensions(width):
    values = np.arange(width, dtype=np.float32)
    actual = mean_last(values, keepdims=False)
    assert isinstance(actual, np.ndarray)
    assert actual.shape == ()
    assert actual == np.float32((width - 1) / 2)
    assert mean_last(values).shape == (1,)
    for leading in [(0,), (2, 0)]:
        empty = np.empty(leading + (width,), dtype=np.float32)
        assert mean_last(empty).shape == leading + (1,)
        assert mean_last(empty, keepdims=False).shape == leading


@pytest.mark.parametrize("shape", [(), (0,), (2, 0)])
def test_mean_rejects_missing_or_empty_reduction_dimension(shape):
    with pytest.raises(ValueError, match="nonempty last dimension"):
        mean_last(np.empty(shape, dtype=np.float32))


@pytest.mark.parametrize("width", [3, 5, 8, 17])
def test_mean_does_not_depend_on_input_alignment_or_strides(width):
    backing = np.arange(1 + 6 * width * 2, dtype=np.float32)
    values = backing[1:].reshape(6, width * 2)[:, ::-2]
    assert not values.flags.c_contiguous
    values.setflags(write=False)
    original = values.copy()
    actual = mean_last(values)
    np.testing.assert_array_equal(actual, mean_last(original))
    np.testing.assert_allclose(actual, np.mean(original.astype(np.float64), axis=-1, keepdims=True))
    np.testing.assert_array_equal(values, original)
    assert not np.shares_memory(actual, values)


@pytest.mark.parametrize("width", [1, 3, 4, 5, 17])
def test_mean_preserves_negative_zero(width):
    actual = mean_last(np.full((4, width), -0.0, dtype=np.float32))
    assert np.all(actual == 0)
    assert np.all(np.signbit(actual))


@pytest.mark.parametrize("sine", [True, False])
@pytest.mark.parametrize("bound", [1e-5, 255, 17999])
def test_trig_accuracy_against_independent_double_reference(sine, bound):
    values = np.random.default_rng(4101).uniform(-bound, bound, 32768).astype(np.float32)
    function = np.sin if sine else np.cos
    reference = function(values.astype(np.float64))
    actual = trig(values, sine=sine)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, reference, rtol=0, atol=1.25e-7)
    assert np.all(np.abs(actual) <= np.float32(1))


@pytest.mark.parametrize("sine", [True, False])
def test_trig_scalar_empty_and_signed_zero(sine):
    for scalar in [np.float32(0), np.float32(-0.0), np.float32(1)]:
        actual = trig(scalar, sine=sine)
        assert isinstance(actual, np.ndarray)
        assert actual.shape == ()
        assert actual.dtype == np.float32
    zeros = trig(np.array([0.0, -0.0], dtype=np.float32), sine=sine)
    np.testing.assert_array_equal(zeros, [0.0, 0.0] if sine else [1.0, 1.0])
    np.testing.assert_array_equal(np.signbit(zeros), [False, True] if sine else [False, False])
    for shape in [(0,), (2, 0), (0, 3)]:
        actual = trig(np.empty(shape, dtype=np.float32), sine=sine)
        assert actual.shape == shape
        assert actual.dtype == np.float32


@pytest.mark.parametrize("sine", [True, False])
def test_trig_readonly_strided_input_is_not_mutated(sine):
    values = np.linspace(-100, 100, 187, dtype=np.float32).reshape(11, 17).T[:, ::-2]
    assert not values.flags.c_contiguous
    values.setflags(write=False)
    original = values.copy()
    actual = trig(values, sine=sine)
    np.testing.assert_array_equal(actual, trig(original, sine=sine))
    np.testing.assert_array_equal(values, original)
    assert not np.shares_memory(actual, values)


@pytest.mark.parametrize("sine", [True, False])
def test_trig_interval_boundary_and_numpy_fallback(sine):
    edge = np.float32(18000)
    interior = np.nextafter(edge, np.float32(0))
    values = np.array(
        [-np.finfo(np.float32).max, -1e20, -edge, -interior, interior, edge, 1e20,
         np.finfo(np.float32).max, np.inf, -np.inf, np.nan], dtype=np.float32
    )
    function = np.sin if sine else np.cos
    with np.errstate(invalid="ignore"):
        actual = trig(values, sine=sine)
        fallback_reference = function(values)
        exact_reference = function(values.astype(np.float64))
    fallback = ~np.isfinite(values) | (np.abs(values) >= edge)
    np.testing.assert_array_equal(actual[fallback], fallback_reference[fallback])
    np.testing.assert_allclose(actual[~fallback], exact_reference[~fallback], rtol=0, atol=1.25e-7)


@pytest.mark.parametrize("dtype", [np.float64, np.float16, np.int32, np.int64, np.bool_])
def test_fp32_functions_reject_implicit_precision_conversion(dtype):
    values = np.ones(4, dtype=dtype)
    with pytest.raises(TypeError, match="float32"):
        mean_last(values)
    with pytest.raises(TypeError, match="float32"):
        trig(values, sine=True)


def test_boolean_parameters_reject_ambiguous_values():
    values = np.ones(4, dtype=np.float32)
    with pytest.raises(TypeError, match="keepdims"):
        mean_last(values, keepdims=1)
    with pytest.raises(TypeError, match="sine"):
        trig(values, sine="sin")
