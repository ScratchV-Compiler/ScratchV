"""Reference division is typed and agrees with actual RV32 execution."""

import numpy as np
import pytest

from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.frontend.dsl_parser import DSLParser
from scratchv.simulator.rv32_emulator import IRTraceExecutor
from scratchv.verification.verifier import DSLInterpreter
from tests.topic28_runtime import execute_fp, execute_i32


def references(source):
    return [
        DSLInterpreter().run(source, {}),
        IRTraceExecutor(DSLParser().parse(source)).run({}),
    ]


class TestDivisionReferenceSemantics:
    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (7, 3, 2),
            (-7, 3, -2),
            (7, -3, -2),
            (-7, -3, 2),
            (7, 0, -1),
            (0, 0, -1),
            (-2147483648, -1, -2147483648),
            (-2147483648, 1, -2147483648),
            (2147483647, 3, 715827882),
        ],
    )
    def test_i32_division_reference_matches_qemu(self, a, b, expected, tmp_path):
        source = f"a = const({a},dtype:i32)\nb = const({b},dtype:i32)\nr = div(a,b)\nreturn r"
        result = CompilerDriver(
            CompilerConfig(use_extended_isel=True, reg_alloc="greedy", verify_ir=True)
        ).compile("", str(tmp_path / "out.s"), dsl_source=source)
        assert result.success, result.errors
        assert execute_i32(result.output_text, tmp_path) == expected
        for reference in references(source):
            assert reference.dtype == np.int32
            assert int(reference) == expected

    @pytest.mark.parametrize("width", [32, 64])
    @pytest.mark.parametrize(
        "a,b",
        [
            ("7", "3"),
            ("1", "0"),
            ("1", "-0.0"),
            ("0", "0"),
            ("-0.0", "3"),
            ("1e-20", "1e-20"),
            ("inf", "inf"),
            ("nan", "4"),
        ],
    )
    def test_float_division_reference_keeps_denominator_and_dtype(
        self, width, a, b, tmp_path
    ):
        source = f"a = const({a},dtype:f{width})\nb = const({b},dtype:f{width})\nr = div(a,b)\nreturn r"
        result = CompilerDriver(
            CompilerConfig(
                use_extended_isel=True,
                reg_alloc="greedy",
                target_isa="rv32imfd",
                target_abi="ilp32d",
                verify_ir=True,
            )
        ).compile("", str(tmp_path / "out.s"), dsl_source=source)
        assert result.success, result.errors
        bits, _ = execute_fp(result.output_text, tmp_path, width)
        unsigned = np.uint32 if width == 32 else np.uint64
        dtype = np.float32 if width == 32 else np.float64
        for reference in references(source):
            assert reference.dtype == dtype
            assert int(np.asarray(reference).view(unsigned)) == bits

    @pytest.mark.parametrize("executor", ["dsl", "ir"])
    def test_integer_division_reference_broadcasts_arrays(self, executor):
        left = np.asarray([[7], [-7], [-2147483648]], dtype=np.int32)
        right = np.asarray([3, -1, 0], dtype=np.int32)
        if executor == "dsl":
            result = DSLInterpreter().run(
                "r = div(a,b)\nreturn r", {"a": left, "b": right}
            )
        else:
            result = IRTraceExecutor(DSLParser().parse("r = div(a,b)\nreturn r")).run(
                {"a": left, "b": right}
            )
        assert result.dtype == np.int32
        assert np.array_equal(
            result,
            np.asarray(
                [[2, -7, -1], [-2, 7, -1], [-715827882, -2147483648, -1]],
                dtype=np.int32,
            ),
        )
