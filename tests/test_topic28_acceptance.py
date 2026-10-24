"""Original Topic 28 deliverables, executed by an independent RV32 caller.

This is the bounded acceptance entry point. Defect regressions for introduced
or affected behavior are collected by the complete suite.
"""

from pathlib import Path
import struct

import onnx
import pytest
from onnx import helper

from scratchv.compiler import CompilerDriver
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType as T
from scratchv.main import args_to_config, build_arg_parser
from tests.topic28_runtime import observe_abi
from tests.topic28_support import MODES, TARGETS, compile_ir, configuration


EXAMPLES = Path(__file__).resolve().parents[1] / "benchmarks" / "topic28"


def bits(value, dtype):
    if dtype is T.INT32:
        return value
    return int.from_bytes(
        struct.pack("<f" if dtype is T.FLOAT32 else "<d", value), "little"
    )


class TestTopic28Acceptance:
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize(
        "name,dtype,isa,abi,expected",
        [
            ("i32_div_rem", T.INT32, "rv32im", "ilp32", -5),
            ("f32_sum", T.FLOAT32, "rv32imf", "ilp32f", 3.5),
            ("f64_sum", T.FLOAT64, "rv32imfd", "ilp32d", 6.75),
            ("f64_dot", T.FLOAT64, "rv32imfd", "ilp32d", 15.0),
        ],
    )
    def test_delivered_programs(self, mode, name, dtype, isa, abi, expected, tmp_path):
        source = EXAMPLES / f"{name}.dsl"
        args = build_arg_parser().parse_args(
            [
                str(source),
                "--extended-isel",
                "--target-isa",
                isa,
                "--target-abi",
                abi,
                "--reg-alloc",
                mode,
                "--verify-ir",
            ]
        )
        result = CompilerDriver(args_to_config(args)).compile(
            str(source), str(tmp_path / "out.s")
        )
        assert result.success, result.errors
        assert observe_abi(result.output_text, tmp_path, dtype, isa=isa, abi=abi) == (
            bits(expected, dtype),
            0,
        )

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize(
        "expression,a,b,expected",
        [
            ("div(a,b)", -17, 5, -3),
            ("rem(a,b)", -17, 5, -2),
            ("mod(a,b)", -17, 5, -2),
            ("div(a,b)", 17, -5, -3),
            ("rem(a,b)", 17, -5, 2),
            ("div(a,b)", 7, 0, -1),
            ("rem(a,b)", 7, 0, 7),
            ("div(a,b)", -(1 << 31), -1, -(1 << 31)),
            ("rem(a,b)", -(1 << 31), -1, 0),
            ("min(a,b)", -17, 5, -17),
            ("max(a,b)", -17, 5, 5),
        ],
    )
    def test_integer_operators_and_division_contract(
        self, mode, expression, a, b, expected, tmp_path
    ):
        result = CompilerDriver(configuration(mode, isa="rv32im", abi="ilp32")).compile(
            "",
            str(tmp_path / "out.s"),
            dsl_source=f"a = const({a},dtype:i32)\nb = const({b},dtype:i32)\nr = {expression}\nreturn r",
        )
        assert result.success, result.errors
        assert observe_abi(
            result.output_text, tmp_path, T.INT32, isa="rv32im", abi="ilp32"
        ) == (expected, 0)

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("target", TARGETS)
    def test_dsl_float_arithmetic_and_math(self, mode, target, tmp_path):
        dtype, _, isa, abi = target
        result = CompilerDriver(configuration(mode, isa=isa, abi=abi)).compile(
            "",
            str(tmp_path / "out.s"),
            dsl_source=(
                f"a = const(4.0,dtype:{dtype.value})\nb = const(2.0,dtype:{dtype.value})\n"
                "c = add(a,b)\nd = sub(c,b)\ne = mul(d,b)\nf = div(e,b)\n"
                "g = neg(f)\nh = abs(g)\ni = sqrt(h)\nj = min(i,a)\nr = max(j,b)\nreturn r"
            ),
        )
        assert result.success, result.errors
        suffix = "s" if dtype is T.FLOAT32 else "d"
        assert f"fmv.{suffix} fa0," in result.output_text
        assert "rne" in result.output_text and "li.d" not in result.output_text
        for mnemonic in ("fadd", "fsub", "fmul", "fdiv", "fsqrt", "fmin", "fmax"):
            assert f"{mnemonic}.{suffix} " in result.output_text
        assert observe_abi(result.output_text, tmp_path, dtype, isa=isa, abi=abi) == (
            bits(2.0, dtype),
            0,
        )

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("target", TARGETS)
    def test_onnx_scalar_math_and_dtype(self, mode, target, tmp_path):
        dtype, nt, isa, abi = target
        nodes = [
            helper.make_node(op, inputs, [output])
            for op, inputs, output in [
                ("Add", ["a", "b"], "sum"),
                ("Sub", ["sum", "b"], "difference"),
                ("Mul", ["difference", "b"], "product"),
                ("Div", ["product", "b"], "quotient"),
                ("Sqrt", ["quotient"], "root"),
                ("Min", ["root", "a"], "minimum"),
                ("Max", ["minimum", "b"], "out"),
            ]
        ]
        graph = helper.make_graph(
            nodes,
            "main",
            [],
            [helper.make_tensor_value_info("out", nt, [])],
            [
                helper.make_tensor("a", nt, [], [4.0]),
                helper.make_tensor("b", nt, [], [2.0]),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        onnx.checker.check_model(model)
        path = tmp_path / "input.onnx"
        onnx.save(model, path)
        result = CompilerDriver(configuration(mode, isa=isa, abi=abi)).compile(
            str(path), str(tmp_path / "out.s")
        )
        assert result.success, result.errors
        assert observe_abi(result.output_text, tmp_path, dtype, isa=isa, abi=abi) == (
            bits(2.0, dtype),
            0,
        )

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize(
        "operator,a,b,condition",
        [
            ("==", 2.0, 2.0, True),
            ("==", 2.0, 3.0, False),
            ("!=", 2.0, 3.0, True),
            ("!=", 2.0, 2.0, False),
            ("<", 2.0, 3.0, True),
            ("<", 3.0, 2.0, False),
            (">", 3.0, 2.0, True),
            (">", 2.0, 3.0, False),
            ("<=", 2.0, 2.0, True),
            ("<=", 3.0, 2.0, False),
            (">=", 2.0, 2.0, True),
            (">=", 2.0, 3.0, False),
        ],
    )
    def test_f64_memory_comparison_and_branches(
        self, mode, operator, a, b, condition, tmp_path
    ):
        builder = IRBuilder()
        builder.new_function("main")
        builder.new_block("entry")
        slot = builder.alloca(1, T.FLOAT64)
        builder.store(slot, builder.load_const(a, T.FLOAT64))
        loaded = builder.load(slot)
        assert loaded.dtype is T.FLOAT64
        builder.br_compare(
            loaded, operator, builder.load_const(b, T.FLOAT64), "yes", "no"
        )
        builder.new_block("yes")
        builder.ret(builder.load_const(7.0, T.FLOAT64))
        builder.new_block("no")
        builder.ret(builder.load_const(9.0, T.FLOAT64))
        assembly = compile_ir(builder.program, tmp_path, configuration(mode))
        assert "fld " in assembly and "fsd " in assembly
        assert observe_abi(assembly, tmp_path, T.FLOAT64) == (
            bits(7.0 if condition else 9.0, T.FLOAT64),
            0,
        )
