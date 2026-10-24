"""Tests for verification module."""

import numpy as np
import pytest
from scratchv.frontend.dsl_parser import DSLParser
from scratchv.simulator.rv32_emulator import IRTraceExecutor
from scratchv.verification.verifier import (
    DSLInterpreter,
    numpy_reference,
)


class TestNumpyReference:
    def test_add(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = numpy_reference("Add", a, b)
        np.testing.assert_array_equal(result, a + b)

    def test_mul(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = numpy_reference("Mul", a, b)
        np.testing.assert_array_equal(result, a * b)

    def test_relu(self):
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        result = numpy_reference("Relu", x)
        np.testing.assert_array_equal(result, np.array([0.0, 0.0, 1.0, 2.0]))

    def test_gelu(self):
        x = np.array([0.0, 1.0, -1.0])
        result = numpy_reference("Gelu", x)
        # GELU(0) = 0
        assert abs(result[0]) < 1e-6
        # GELU(1) ≈ 0.8413
        assert abs(result[1] - 0.8413) < 0.01

    def test_matmul(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = numpy_reference("MatMul", a, b)
        expected = a @ b
        np.testing.assert_array_almost_equal(result, expected)

    def test_exp(self):
        x = np.array([0.0, 1.0, 2.0])
        result = numpy_reference("Exp", x)
        np.testing.assert_array_almost_equal(result, np.exp(x))

    def test_neg(self):
        x = np.array([1.0, -2.0, 3.0])
        result = numpy_reference("Neg", x)
        np.testing.assert_array_equal(result, -x)

    def test_softmax(self):
        x = np.array([1.0, 2.0, 3.0])
        result = numpy_reference("Softmax", x)
        # Sum should be ~1.0
        assert abs(result.sum() - 1.0) < 1e-5
        # All positive
        assert (result > 0).all()


class TestDSLInterpreter:
    def test_simple_add(self):
        dsl = "y = add(a, b)\nreturn y"
        interpreter = DSLInterpreter()
        result = interpreter.run(
            dsl,
            {
                "a": np.array([1.0, 2.0]),
                "b": np.array([3.0, 4.0]),
            },
        )
        np.testing.assert_array_equal(result, np.array([4.0, 6.0]))

    def test_mul_then_add(self):
        dsl = "t = mul(a, b)\ny = add(t, c)\nreturn y"
        interpreter = DSLInterpreter()
        result = interpreter.run(
            dsl,
            {
                "a": np.float64(2.0),
                "b": np.float64(3.0),
                "c": np.float64(1.0),
            },
        )
        assert abs(result - 7.0) < 1e-6

    def test_relu(self):
        dsl = "y = relu(x)\nreturn y"
        interpreter = DSLInterpreter()
        result = interpreter.run(dsl, {"x": np.array([-1.0, 0.0, 2.0])})
        np.testing.assert_array_equal(result, np.array([0.0, 0.0, 2.0]))

    def test_matmul_dsl(self):
        dsl = "c = matmul(A, B, m:2, n:2, k:2)\nreturn c"
        interpreter = DSLInterpreter()
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = interpreter.run(dsl, {"A": A, "B": B})
        np.testing.assert_array_almost_equal(result, A @ B)

    def test_multi_op_chain(self):
        dsl = """
t1 = mul(x, w)
t2 = add(t1, b)
y = relu(t2)
return y
"""
        interpreter = DSLInterpreter()
        x = np.array([1.0, -2.0])
        w = np.array([0.5, 1.5])
        b = np.array([0.1, -0.2])
        result = interpreter.run(dsl, {"x": x, "w": w, "b": b})
        expected = np.maximum(x * w + b, 0.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_extended_integer_operations(self):
        dsl = """
a = const(-17, dtype:i32)
b = const(5, dtype:i32)
r = rem(a, b)
lo = min(r, b)
hi = max(lo, a)
y = abs(hi)
return y
"""

        result = DSLInterpreter().run(dsl, {})

        assert result.dtype == np.int32
        assert int(result) == 2

    def test_extended_remainder_divide_by_zero_matches_riscv(self):
        dsl = """
a = const(-17, dtype:i32)
z = const(0, dtype:i32)
y = rem(a, z)
return y
"""

        result = DSLInterpreter().run(dsl, {})

        assert int(result) == -17

    @pytest.mark.parametrize(
        ("dtype", "numpy_dtype"),
        [("f32", np.float32), ("f64", np.float64)],
    )
    def test_extended_float_min_max_abs_preserve_dtype(
        self,
        dtype,
        numpy_dtype,
    ):
        dsl = f"""
a = const(-2.5, dtype:{dtype})
b = const(1.25, dtype:{dtype})
lo = min(a, b)
hi = max(lo, b)
y = abs(hi)
return y
"""

        result = DSLInterpreter().run(dsl, {})

        assert result.dtype == numpy_dtype
        assert float(result) == 1.25


class TestIRTraceExtendedOperations:
    def test_ir_trace_executor_supports_extended_integer_operations(self):
        program = DSLParser().parse(
            """
a = const(-17, dtype:i32)
b = const(5, dtype:i32)
r = rem(a, b)
lo = min(r, b)
hi = max(lo, a)
y = abs(hi)
return y
"""
        )

        result = IRTraceExecutor(program).run({})

        assert result.dtype == np.int32
        assert int(result) == 2

    @pytest.mark.parametrize(
        ("dtype", "numpy_dtype"),
        [("f32", np.float32), ("f64", np.float64)],
    )
    def test_ir_trace_executor_supports_extended_float_operations(
        self,
        dtype,
        numpy_dtype,
    ):
        program = DSLParser().parse(
            f"""
a = const(-2.5, dtype:{dtype})
b = const(1.25, dtype:{dtype})
lo = min(a, b)
hi = max(lo, b)
y = abs(hi)
return y
"""
        )

        result = IRTraceExecutor(program).run({})

        assert result.dtype == numpy_dtype
        assert float(result) == 1.25
