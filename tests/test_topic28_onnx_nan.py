"""ONNX Min/Max must propagate NaN rather than select the numeric input."""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.simulator.rv32_emulator import IRTraceExecutor
from tests.topic28_runtime import execute_fp


onnx = pytest.importorskip("onnx")
from onnx.reference import ReferenceEvaluator


def _model(tmp_path, operator, width, values, *, parameter=False):
    dtype = np.float32 if width == 32 else np.float64
    tensor_type = onnx.TensorProto.FLOAT if width == 32 else onnx.TensorProto.DOUBLE
    names = [f"input_{index}" for index in range(len(values))]
    initializers = [
        onnx.numpy_helper.from_array(np.asarray(value, dtype=dtype), name)
        for index, (name, value) in enumerate(zip(names, values))
        if not (parameter and index == 0)
    ]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node(operator, names, ["output"])],
        "main",
        (
            [onnx.helper.make_tensor_value_info(names[0], tensor_type, [])]
            if parameter
            else []
        ),
        [onnx.helper.make_tensor_value_info("output", tensor_type, [])],
        initializers,
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 13)],
        ir_version=9,
    )
    onnx.checker.check_model(model)
    path = tmp_path / "nan.onnx"
    onnx.save(model, path)
    return model, path


def _compile(
    path,
    tmp_path,
    allocator="linear",
    optimization="none",
    *,
    isa="rv32imfd",
    abi="ilp32d",
):
    result = CompilerDriver(
        CompilerConfig(
            use_extended_isel=True,
            reg_alloc=allocator,
            verify_ir=True,
            target_isa=isa,
            target_abi=abi,
            optimize_level=optimization,
        )
    ).compile(str(path), str(tmp_path / "nan.s"))
    assert result.success, result.errors
    return result.output_text


def _float(bits, width):
    return struct.unpack(
        "<f" if width == 32 else "<d", bits.to_bytes(width // 8, "little")
    )[0]


class TestONNXMinMaxNaNPropagation:
    @pytest.mark.parametrize("operator", ("Min", "Max"))
    @pytest.mark.parametrize("width", (32, 64))
    @pytest.mark.parametrize("values", ((float("nan"), 2.0), (2.0, float("nan"))))
    @pytest.mark.parametrize("optimization", ("none", "basic", "all"))
    def test_onnx_min_max_propagate_nan(
        self, tmp_path, operator, width, values, optimization
    ):
        # NaN propagation is a frontend/optimizer rule; allocator coverage
        # lives in test_onnx_nan_survives_variadic_register_pressure.
        model, path = _model(tmp_path, operator, width, values)
        expected = ReferenceEvaluator(model).run(None, {})[0].item()
        assert math.isnan(expected)
        bits, flags = execute_fp(
            _compile(path, tmp_path, "greedy", optimization), tmp_path, width
        )
        assert math.isnan(
            _float(bits, width)
        ), f"expected ONNX NaN, got {_float(bits, width)}"
        assert flags == 0  # Quiet NaN must not introduce an invalid-operation flag.

    @pytest.mark.parametrize("operator", ("Min", "Max"))
    @pytest.mark.parametrize("width", (32, 64))
    @pytest.mark.parametrize("values", ((float("nan"), 2.0), (2.0, float("nan"))))
    def test_ir_trace_preserves_onnx_nan_rule(self, tmp_path, operator, width, values):
        model, path = _model(tmp_path, operator, width, values)
        expected = ReferenceEvaluator(model).run(None, {})[0]
        actual = IRTraceExecutor(ONNXParser().parse(str(path))).run({})
        assert actual.dtype == expected.dtype
        assert math.isnan(actual.item()) and math.isnan(expected.item())

    @pytest.mark.parametrize("operator", ("Min", "Max"))
    @pytest.mark.parametrize("width", (32, 64))
    @pytest.mark.parametrize(
        "values",
        (
            (float("nan"),),
            (float("nan"), float("nan")),
            (float("nan"), 2.0, -4.0),
            (2.0, float("nan"), -4.0),
            (2.0, -4.0, float("nan")),
        ),
    )
    def test_variadic_onnx_min_max_keep_nan_at_every_position(
        self, tmp_path, operator, width, values
    ):
        model, path = _model(tmp_path, operator, width, values)
        expected = ReferenceEvaluator(model).run(None, {})[0]
        bits, flags = execute_fp(
            _compile(path, tmp_path, "greedy", "all"), tmp_path, width
        )
        assert math.isnan(expected.item()) and math.isnan(_float(bits, width))
        assert math.isnan(IRTraceExecutor(ONNXParser().parse(str(path))).run({}).item())
        assert flags == 0

    @pytest.mark.parametrize("operator", ("Min", "Max"))
    @pytest.mark.parametrize("width", (32, 64))
    @pytest.mark.parametrize(
        "argument,constant",
        (
            (float("nan"), 2.0),
            (2.0, float("nan")),
            (7.0, 2.0),
        ),
    )
    @pytest.mark.parametrize("optimization", ("none", "basic", "all"))
    def test_onnx_min_max_runtime_parameter(
        self, tmp_path, operator, width, argument, constant, optimization
    ):
        model, path = _model(
            tmp_path, operator, width, (argument, constant), parameter=True
        )
        dtype = np.float32 if width == 32 else np.float64
        inputs = {"input_0": np.asarray(argument, dtype=dtype)}
        expected = ReferenceEvaluator(model).run(None, inputs)[0].item()
        bits, flags = execute_fp(
            _compile(path, tmp_path, "linear", optimization),
            tmp_path,
            width,
            argument=argument,
        )
        actual = _float(bits, width)
        trace = IRTraceExecutor(ONNXParser().parse(str(path))).run(inputs).item()
        if math.isnan(expected):
            assert math.isnan(actual) and math.isnan(trace)
        else:
            assert actual == expected and trace == expected
        assert flags == 0

    @pytest.mark.parametrize("operator", ("Min", "Max"))
    @pytest.mark.parametrize("width", (32, 64))
    @pytest.mark.parametrize("signaling", (False, True))
    @pytest.mark.parametrize("reverse", (False, True))
    def test_onnx_min_max_nan_encodings_and_flags(
        self, tmp_path, operator, width, signaling, reverse
    ):
        bits = (
            (0x7F812345 if signaling else 0x7FC12345)
            if width == 32
            else (0x7FF0000000001234 if signaling else 0x7FF8000000001234)
        )
        dtype = "<f4" if width == 32 else "<f8"
        nan = np.frombuffer(bits.to_bytes(width // 8, "little"), dtype=dtype)[0]
        values = (2.0, nan) if reverse else (nan, 2.0)
        model, path = _model(tmp_path, operator, width, values)
        with np.errstate(invalid="ignore"):
            expected = ReferenceEvaluator(model).run(None, {})[0].item()
        actual, flags = execute_fp(
            _compile(path, tmp_path, "naive", "all"), tmp_path, width
        )
        assert math.isnan(expected) and math.isnan(_float(actual, width))
        assert actual == (0x7FC00000 if width == 32 else 0x7FF8000000000000)
        assert flags == (16 if signaling else 0)

    @pytest.mark.parametrize("operator", ("Min", "Max"))
    @pytest.mark.parametrize("width", (32, 64))
    @pytest.mark.parametrize(
        "values",
        (
            (7.0, 2.0),
            (float("inf"), -float("inf")),
            (float("inf"), 2.0),
            (2.0, -float("inf")),
            (3e38, 3e38),
            (-3e38, -3e38),
        ),
    )
    def test_onnx_min_max_without_nan_do_not_add_arithmetic_exceptions(
        self, tmp_path, operator, width, values
    ):
        model, path = _model(tmp_path, operator, width, values)
        expected = ReferenceEvaluator(model).run(None, {})[0].item()
        bits, flags = execute_fp(
            _compile(path, tmp_path, "greedy", "all"), tmp_path, width
        )
        assert _float(bits, width) == expected
        assert flags == 0  # In particular, +inf and -inf must not be added together.

    @pytest.mark.parametrize("operator", ("min", "max"))
    @pytest.mark.parametrize("width", (32, 64))
    @pytest.mark.parametrize("reverse", (False, True))
    def test_dsl_min_max_still_select_numeric_operand(
        self, tmp_path, operator, width, reverse
    ):
        values = ("2.0", "nan") if reverse else ("nan", "2.0")
        source = (
            f"a = const({values[0]},dtype:f{width})\n"
            f"b = const({values[1]},dtype:f{width})\n"
            f"r = {operator}(a,b)\nreturn r\n"
        )
        path = tmp_path / "number.dsl"
        path.write_text(source)
        bits, flags = execute_fp(
            _compile(path, tmp_path, "greedy", "all"), tmp_path, width
        )
        assert _float(bits, width) == 2.0 and flags == 0

    @pytest.mark.parametrize("operator", ("Min", "Max"))
    @pytest.mark.parametrize("width", (32, 64))
    @pytest.mark.parametrize("allocator", ("naive", "greedy", "linear"))
    def test_onnx_nan_survives_variadic_register_pressure(
        self, tmp_path, operator, width, allocator
    ):
        values = list(range(40))
        values[19] = float("nan")  # Later reductions must not erase this NaN.
        model, path = _model(tmp_path, operator, width, values)
        assert math.isnan(ReferenceEvaluator(model).run(None, {})[0].item())
        bits, flags = execute_fp(
            _compile(path, tmp_path, allocator, "all"), tmp_path, width
        )
        assert math.isnan(_float(bits, width)) and flags == 0

    @pytest.mark.parametrize("operator", ("Min", "Max"))
    @pytest.mark.parametrize("reverse", (False, True))
    @pytest.mark.parametrize("allocator", ("naive", "greedy", "linear"))
    def test_onnx_nan_on_single_precision_only_target(
        self, tmp_path, operator, reverse, allocator
    ):
        from scratchv.ir.types import DataType
        from tests.topic28_runtime import observe

        values = (2.0, float("nan")) if reverse else (float("nan"), 2.0)
        model, path = _model(tmp_path, operator, 32, values)
        assembly = _compile(
            path, tmp_path, allocator, "all", isa="rv32imf", abi="ilp32f"
        )
        bits, flags = observe(
            assembly, tmp_path, DataType.FLOAT32, isa="rv32imf", abi="ilp32f"
        )
        assert math.isnan(ReferenceEvaluator(model).run(None, {})[0].item())
        assert math.isnan(_float(bits, 32)) and flags == 0

    @pytest.mark.parametrize("operator", ("Min", "Max"))
    @pytest.mark.parametrize("width", (32, 64))
    @pytest.mark.parametrize("reverse", (False, True))
    def test_nan_propagation_keeps_existing_riscv_signed_zero_policy(
        self, tmp_path, operator, width, reverse
    ):
        # This NaN fix preserves the documented target zero ordering. The trace
        # must agree with the hardware path even where NumPy's zero tie differs.
        values = (0.0, -0.0) if reverse else (-0.0, 0.0)
        _, path = _model(tmp_path, operator, width, values)
        expected_bits = 1 << (width - 1) if operator == "Min" else 0
        actual, flags = execute_fp(
            _compile(path, tmp_path, "greedy", "all"), tmp_path, width
        )
        trace = IRTraceExecutor(ONNXParser().parse(str(path))).run({})
        assert actual == expected_bits and flags == 0
        assert int.from_bytes(trace.tobytes(), "little") == expected_bits
