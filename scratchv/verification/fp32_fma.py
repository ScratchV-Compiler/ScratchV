"""Correctly rounded finite FP32 fused multiply-add using NumPy only.

No floating-point precision promotion is used. The integer fallback aligns
exact significands in uint64 (or nine uint64 limbs), then performs one binary32
round-to-nearest-even operation. It supports broadcasting, normals,
subnormals and signed zero; NaN/Inf inputs and finite-result overflow raise.

The FP32 error-free-transform fast path requires all three absolute inputs
in [2**-40, 2**40], hence a/b low-part products cannot be smaller than 2**-126
and no intermediate can overflow. Dekker TwoProduct and Knuth TwoSum give
exact product_error and sum_error. Let s=fl(fl(a*b)+c), E=product_error+sum_error
(exact), and r=fl(E). Only cases abs(fl(a*b))<=abs(s), with s not a power of two,
are accepted. Then s has equal adjacent spacing u, abs(E)<=u, and abs(r)<=u.
The only rounding boundaries reachable from s are s +/- u/2. If abs(r)==u/2,
use the integer fallback. Otherwise r's distance from either boundary is at
least one ulp(r), larger than abs(E-r)<=0.5 ulp(r); fl(s+r) is therefore the
correct single-rounded result. Excluding powers of two avoids asymmetric
binade spacing. Cancellation, zero, range boundaries not satisfying these
conditions, midpoint ambiguity and all other cases use exact integers.

The implementation assumes normal NumPy IEEE binary32 round-to-nearest-even
operations. It does not change the host rounding mode or enable FTZ/DAZ.
"""
import numpy as np

_U = np.uint64
_I = np.int64


def _topbit(x):
    """floor(log2(x)), choosing zero for x=0; only integer shifts."""
    x = x.copy()
    result = np.zeros(x.shape, dtype=np.int64)
    for shift in (32, 16, 8, 4, 2, 1):
        high = x >> _U(shift)
        selected = high != 0
        result += np.where(selected, shift, 0)
        x = np.where(selected, high, x)
    return result


def _unpack(array):
    bits = array.view(np.uint32).reshape(-1)
    exponent = (bits >> np.uint32(23)) & np.uint32(255)
    if np.any(exponent == 255):
        raise ValueError("finite float32 inputs required")
    mantissa = (bits & np.uint32(0x7fffff)).astype(np.uint64)
    mantissa |= np.where(exponent != 0, _U(1 << 23), _U(0))
    scale = np.where(exponent != 0, exponent.astype(np.int64) - _I(150), _I(-149))
    return bits >> np.uint32(31), mantissa, scale


def _pack(sign, q, quantum):
    carry = q == _U(1 << 24)
    q = np.where(carry, q >> _U(1), q)
    quantum = quantum + carry.astype(np.int64)
    exponent = np.where(q >= _U(1 << 23), quantum + _I(150), _I(0))
    if np.any(exponent >= 255):
        raise FloatingPointError("fused result overflows finite float32")
    bits = ((sign.astype(np.uint32) << np.uint32(31))
            | (exponent.astype(np.uint32) << np.uint32(23))
            | (q.astype(np.uint32) & np.uint32(0x7fffff)))
    return bits.view(np.float32)


def _round64(sign, magnitude, scale):
    quantum = np.maximum(_topbit(magnitude) + scale - _I(23), _I(-149))
    shift = quantum - scale
    right = np.maximum(shift, _I(0)).astype(np.uint64)
    left = np.maximum(-shift, _I(0)).astype(np.uint64)
    q = (magnitude >> right) << left
    guard_shift = np.maximum(shift - _I(1), _I(0)).astype(np.uint64)
    guard = (shift > 0) & (((magnitude >> guard_shift) & _U(1)) != 0)
    # For shift >=65 all magnitude bits are sticky, but the guard is zero.
    mask = np.where(guard_shift >= _U(64), _U(0xffffffffffffffff),
                    (_U(1) << np.minimum(guard_shift, _U(63))) - _U(1))
    sticky = (magnitude & mask) != 0
    q += (guard & (sticky | ((q & _U(1)) != 0))).astype(np.uint64)
    return _pack(sign, q, quantum)


def _limbs(magnitude, scale):
    # Smallest exact product quantum is 2^-298. Highest possible product bit
    # index is 553; 9 _limbs (576 bits) also have room for the final addition.
    result = np.zeros((9, magnitude.size), dtype=np.uint64)
    shifts = scale + _I(298)
    indices = np.arange(magnitude.size)
    low = shifts // _I(64)
    rem = (shifts % _I(64)).astype(np.uint64)
    result[low, indices] = magnitude << rem
    high = magnitude >> (_U(64) - rem)
    selected = (low + 1 < 9) & (high != 0)
    result[low[selected] + 1, indices[selected]] = high[selected]
    return result


def _slow(sign_p, product, ep, sign_c, cm, ec):
    p, c = _limbs(product, ep), _limbs(cm, ec)
    greater, less = np.zeros(product.size, dtype=bool), np.zeros(product.size, dtype=bool)
    for index in range(8, -1, -1):
        undecided = ~(greater | less)
        greater |= undecided & (p[index] > c[index])
        less |= undecided & (p[index] < c[index])
    p_ge = ~less
    same = sign_p == sign_c
    sign = np.where(same | p_ge, sign_p, sign_c)
    mag = np.zeros_like(p)
    carry, borrow = np.zeros(product.size, dtype=np.uint64), np.zeros(product.size, dtype=np.uint64)
    for index in range(9):
        total = p[index] + c[index]
        addition = total + carry
        carry = ((total < p[index]) | (addition < total)).astype(np.uint64)
        hi, lo = np.where(p_ge, p[index], c[index]), np.where(p_ge, c[index], p[index])
        diff = hi - lo
        subtraction = diff - borrow
        borrow = ((hi < lo) | (subtraction > diff)).astype(np.uint64)
        mag[index] = np.where(same, addition, subtraction)
    exact_zero = ~np.any(mag != 0, axis=0)
    sign = np.where(exact_zero & ~same, np.uint32(0), sign)
    high_index = np.zeros(product.size, dtype=np.int64)
    for index in range(9):
        high_index = np.where(mag[index] != 0, index, high_index)
    columns = np.arange(product.size)
    highest = high_index * _I(64) + _topbit(mag[high_index, columns])
    shift = np.maximum(highest - _I(23), _I(149))
    index = shift // _I(64)
    rem = (shift % _I(64)).astype(np.uint64)
    padded = np.vstack((mag, np.zeros((1, product.size), dtype=np.uint64)))
    q = (padded[index, columns] >> rem) | (padded[index + 1, columns] << (_U(64) - rem))
    guard_index = (shift - _I(1)) // _I(64)
    guard_rem = ((shift - _I(1)) % _I(64)).astype(np.uint64)
    guard = ((mag[guard_index, columns] >> guard_rem) & _U(1)) != 0
    mask = (_U(1) << guard_rem) - _U(1)
    sticky = np.zeros(product.size, dtype=bool)
    for index in range(9):
        sticky |= ((index < guard_index) & (mag[index] != 0))
        sticky |= ((index == guard_index) & ((mag[index] & mask) != 0))
    q += (guard & (sticky | ((q & _U(1)) != 0))).astype(np.uint64)
    return _pack(sign, q, shift - _I(298))


def _integer_fma(a, b, c):
    arrays = [np.asarray(x) for x in (a, b, c)]
    if any(x.dtype != np.dtype("float32") for x in arrays):
        raise TypeError("FMA requires native float32 inputs")
    a, b, c = np.broadcast_arrays(*arrays)
    sa, am, ae = _unpack(a)
    sb, bm, be = _unpack(b)
    sc, cm, ce = _unpack(c)
    product, pe, sp = am * bm, ae + be, sa ^ sb
    scale = np.where(product == 0, ce, np.where(cm == 0, pe, np.minimum(pe, ce)))
    ps, cs = pe - scale, ce - scale
    # Each aligned operand has <=63 significant bits. Their sum fits uint64
    # (max 2^64-2), and opposite-sign subtraction is exact magnitude arithmetic.
    fast = (((product == 0) | (_topbit(product) + ps < 63))
            & ((cm == 0) | (_topbit(cm) + cs < 63)))
    result = np.empty(product.shape, dtype=np.float32)
    if np.any(fast):
        p = product[fast] << np.maximum(ps[fast], _I(0)).astype(np.uint64)
        cv = cm[fast] << np.maximum(cs[fast], _I(0)).astype(np.uint64)
        same = sp[fast] == sc[fast]
        p_ge = p >= cv
        magnitude = np.where(same, p + cv, np.where(p_ge, p - cv, cv - p))
        sign = np.where(same | p_ge, sp[fast], sc[fast])
        sign = np.where((magnitude == 0) & ~same, np.uint32(0), sign)
        result[fast] = _round64(sign, magnitude, scale[fast])
    if np.any(~fast):
        result[~fast] = _slow(sp[~fast], product[~fast], pe[~fast], sc[~fast], cm[~fast], ce[~fast])
    return result.reshape(a.shape)


def fma(a, b, c):
    """Return round32(a*b+c), rejecting non-f32/nonfinite inputs and overflow."""
    values = [np.asarray(value) for value in (a, b, c)]
    if any(value.dtype != np.dtype("float32") for value in values):
        raise TypeError("FMA requires native float32 inputs")
    arrays = np.broadcast_arrays(*values)
    shape = arrays[0].shape
    a, b, c = [value.reshape(-1) for value in arrays]
    lower = np.array(87 << 23, dtype=np.uint32).view(np.float32)
    upper = np.array(167 << 23, dtype=np.uint32).view(np.float32)
    bounded = np.ones(a.shape, dtype=bool)
    for value in (a, b, c):
        bounded &= (np.abs(value) >= lower) & (np.abs(value) <= upper)
    indices = np.flatnonzero(bounded)
    result = np.empty(a.shape, dtype=np.float32)
    accepted = np.zeros(a.shape, dtype=bool)
    if indices.size:
        av, bv, cv = a[indices], b[indices], c[indices]
        product = av * bv
        split_a = np.float32(4097) * av
        high_a = split_a - (split_a - av)
        low_a = av - high_a
        split_b = np.float32(4097) * bv
        high_b = split_b - (split_b - bv)
        low_b = bv - high_b
        product_error = low_a * low_b - (
            ((product - high_a * high_b) - low_a * high_b) - high_a * low_b)
        total = product + cv
        virtual_c = total - product
        sum_error = (product - (total - virtual_c)) + (cv - virtual_c)
        residual = product_error + sum_error
        total_bits = total.view(np.uint32)
        exponent = (total_bits >> np.uint32(23)) & np.uint32(255)
        half_ulp = ((np.maximum(exponent, np.uint32(24)) - np.uint32(24))
                    << np.uint32(23)).view(np.float32)
        safe = ((np.abs(product) <= np.abs(total))
                & ((total_bits & np.uint32(0x7fffff)) != 0)
                & (np.abs(residual) != half_ulp))
        accepted[indices[safe]] = True
        result[indices[safe]] = (total + residual)[safe]
    if np.any(~accepted):
        result[~accepted] = _integer_fma(a[~accepted], b[~accepted], c[~accepted])
    return result.reshape(shape)
