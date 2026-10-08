"""Portable bit-exact FMA checks against a Python arbitrary-integer oracle."""
import itertools

import numpy as np
import pytest

from scratchv.verification.fp32_fma import fma


def exact_fma_bits(a_bits, b_bits, c_bits):
    """Independent exact dyadic arithmetic, no host floating-point reference."""
    def decode(bits):
        sign = bits >> 31
        exponent = (bits >> 23) & 255
        fraction = bits & 0x7fffff
        assert exponent != 255
        return sign, fraction | ((1 << 23) if exponent else 0), exponent - 150 if exponent else -149

    sa, a, ea = decode(int(a_bits))
    sb, b, eb = decode(int(b_bits))
    sc, c, ec = decode(int(c_bits))
    ep, sp = ea + eb, sa ^ sb
    base = min(ep, ec)
    exact = ((-1 if sp else 1) * ((a * b) << (ep - base))
             + (-1 if sc else 1) * (c << (ec - base)))
    if exact == 0:
        return (sp & sc) << 31
    sign, magnitude = int(exact < 0), abs(exact)
    quantum = max(magnitude.bit_length() - 1 + base - 23, -149)
    shift = quantum - base
    if shift < 0:
        quotient = magnitude << -shift
    else:
        quotient, remainder = divmod(magnitude, 1 << shift)
        midpoint_twice = 1 << shift
        if 2 * remainder > midpoint_twice or (2 * remainder == midpoint_twice and quotient & 1):
            quotient += 1
    if quotient == 1 << 24:
        quotient >>= 1
        quantum += 1
    exponent = quantum + 150 if quotient >= 1 << 23 else 0
    if exponent >= 255:
        raise OverflowError("exact result rounds to infinity")
    return sign << 31 | exponent << 23 | (quotient & 0x7fffff)


def check_bits(rows):
    finite, expected, overflowing = [], [], []
    for row in rows:
        try:
            answer = exact_fma_bits(*row)
        except OverflowError:
            overflowing.append(row)
        else:
            finite.append(row)
            expected.append(answer)
    if finite:
        data = np.asarray(finite, dtype=np.uint32).view(np.float32)
        actual = fma(*data.T)
        np.testing.assert_array_equal(actual.view(np.uint32), np.asarray(expected, dtype=np.uint32))
    if overflowing:
        data = np.asarray(overflowing, dtype=np.uint32).view(np.float32)
        with pytest.raises(FloatingPointError, match="overflows"):
            fma(*data.T)


def test_float32_fma_boundaries_signed_zero_subnormal_cancellation_and_overflow():
    bits = [0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
            0x00800000, 0x80800000, 0x3f000000, 0xbf000000, 0x3f800000,
            0xbf800000, 0x3f800001, 0x3f7fffff, 0x7f7fffff, 0xff7fffff]
    check_bits(itertools.product(bits, repeat=3))


def test_compensated_double_rounding_midpoint_uses_exact_result():
    values = np.array([0x33800001, 0x3f7fffff, 0x3f800000], dtype=np.uint32).view(np.float32)
    result = fma(*values)
    assert result.view(np.uint32) == 0x3f800001
    assert exact_fma_bits(*values.view(np.uint32)) == 0x3f800001


@pytest.mark.parametrize("bounded", [False, True])
def test_random_finite_inputs_against_exact_integer_reference(bounded):
    rng = np.random.default_rng(20261005)
    rows = rng.integers(0, 2**32, size=(10_000, 3), dtype=np.uint32)
    if bounded:
        rows &= np.uint32(0x807fffff)
        rows |= rng.integers(86, 169, size=rows.shape, dtype=np.uint32) << np.uint32(23)
    rows = rows[np.all(((rows >> 23) & 255) != 255, axis=1)]
    check_bits(rows)


def test_range_endpoints_and_adjacent_values_against_exact_reference():
    values = [np.float32(2**-40), np.float32(2**40)]
    choices = [item for value in values for item in (
        np.nextafter(value, np.float32(0)), value, np.nextafter(value, np.float32(np.inf)))]
    choices += [-value for value in choices]
    bits = np.asarray(choices, dtype=np.float32).view(np.uint32)
    check_bits(itertools.product(bits, repeat=3))


def test_product_cancellation_and_adjacent_addends_against_exact_reference():
    # Uniform random triples rarely almost cancel. Place c at the negated
    # rounded product and its immediate neighbors to exercise that boundary.
    rng = np.random.default_rng(6301)
    operands = rng.integers(0, 2**32, size=(4096, 2), dtype=np.uint32)
    operands &= np.uint32(0x807fffff)
    operands |= rng.integers(100, 151, size=operands.shape,
                             dtype=np.uint32) << np.uint32(23)
    floats = operands.view(np.float32)
    canceled = (-(floats[:, 0] * floats[:, 1])).view(np.uint32)
    rows = [np.column_stack((operands, canceled + offset))
            for offset in (np.uint32(0), np.uint32(1))]
    rows.append(np.column_stack((operands, canceled - np.uint32(1))))
    check_bits(np.concatenate(rows))


def test_mixed_fast_and_integer_fallback_and_input_immutability(monkeypatch):
    import scratchv.verification.fp32_fma as module

    original, observed = module._integer_fma, []
    def capture(a, b, c):
        observed.append(len(a))
        return original(a, b, c)
    monkeypatch.setattr(module, "_integer_fma", capture)
    values = np.array([[0.75, 0.25, 0.125], [0, 1, 1], [2, 0.5, 1],
                       [0.5, 0.5, -0.25]], dtype=np.float32)
    before = values.copy()
    values.setflags(write=False)
    result = fma(*values.T)
    expected = [exact_fma_bits(*row) for row in values.view(np.uint32)]
    np.testing.assert_array_equal(result.view(np.uint32), expected)
    np.testing.assert_array_equal(values, before)
    assert observed == [3]


def test_broadcast_scalar_and_empty_shapes():
    result = fma(np.array([[1], [-1]], dtype=np.float32),
                 np.array([[2, 3, 4]], dtype=np.float32), np.float32(1))
    np.testing.assert_array_equal(result, [[3, 4, 5], [-1, -2, -3]])
    assert result.dtype == np.float32
    assert fma(np.float32(1), np.float32(2), np.float32(3)).shape == ()
    assert fma(np.empty((0, 3), dtype=np.float32), np.float32(1), np.float32(2)).shape == (0, 3)


@pytest.mark.parametrize("bad", [np.float32(np.nan), np.float32(np.inf), np.float32(-np.inf)])
def test_nonfinite_inputs_are_rejected(bad):
    with pytest.raises(ValueError, match="finite float32"):
        fma(np.float32(1), bad, np.float32(1))


@pytest.mark.parametrize("bad", [1.0, 1, np.float64(1), np.array([1], dtype=">f4")])
def test_precision_promotion_is_rejected(bad):
    with pytest.raises(TypeError, match="float32"):
        fma(bad, np.float32(1), np.float32(1))
