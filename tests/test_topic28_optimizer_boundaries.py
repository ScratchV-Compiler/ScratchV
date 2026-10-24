"""Optimizer and compilation-unit contracts with real execution.

Covers cross-block DCE, unused FP exceptions, per-function label scoping and
unsupported-type rejection.
"""

import struct

import numpy as np
import pytest

from scratchv.analysis.ir_verifier import verify_ir
from scratchv.backend.inst_select_ext import ExtendedInstructionSelector
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType as T, OpCode
from scratchv.optimizer.dead_code import DeadCodeEliminator
from tests.topic28_runtime import execute_fp, execute_i32


MODES = ("naive", "greedy", "linear")
LEVELS = ("none", "basic", "all")
FLOATS = ((32, T.FLOAT32), (64, T.FLOAT64))


def config(mode="greedy", level="none", verify=True, **kwargs):
    return CompilerConfig(
        use_extended_isel=True,
        reg_alloc=mode,
        optimize_level=level,
        verify_ir=verify,
        target_isa="rv32imfd",
        target_abi="ilp32d",
        **kwargs,
    )


def compile_ir(program, tmp_path, configuration):
    driver = CompilerDriver(configuration)
    driver._parse = lambda *args: program
    return driver.compile("", str(tmp_path / "out.s"), dsl_source="")


def bits(value, width):
    return int.from_bytes(struct.pack("<f" if width == 32 else "<d", value), "little")


def onnx_div_model(tmp_path, dtype, *, parameter=False):
    onnx = pytest.importorskip("onnx")
    tensor_type = onnx.TensorProto.INT32 if dtype is T.INT32 else onnx.TensorProto.INT64
    array_type = np.int32 if dtype is T.INT32 else np.int64
    if parameter:
        inputs = [onnx.helper.make_tensor_value_info("a", tensor_type, [])]
        nodes = [onnx.helper.make_node("Div", ["a", "b"], ["out"])]
        values = {"b": 3}
    else:
        inputs = []
        nodes = [
            onnx.helper.make_node("Add", ["a", "b"], ["sum"]),
            onnx.helper.make_node("Div", ["sum", "c"], ["out"]),
        ]
        values = {"a": 2147483647, "b": 1, "c": 2}
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes,
            "main",
            inputs,
            [onnx.helper.make_tensor_value_info("out", tensor_type, [])],
            [
                onnx.numpy_helper.from_array(np.asarray(v, dtype=array_type), name=k)
                for k, v in values.items()
            ],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)
    path = tmp_path / "input.onnx"
    onnx.save(model, path)
    return path, model


def cross_block_program(dtype):
    b = IRBuilder()
    b.new_function("main")
    b.new_block("entry")
    if dtype is T.INT32:
        p = b.alloca(1, T.INT32)
        b.store(p, b.load_const(7, T.INT32))
        value = b.rem(b.load(p, T.INT32), b.load_const(5, T.INT32))
    else:
        value = b.sqrt(b.load_const(4.0, dtype))
    b.br("next")
    b.new_block("next")
    b.ret(value)
    assert verify_ir(b.program) == (True, [])
    return b.program


def exception_source(width, case):
    if case == "sqrt":
        return (
            f"a = const(-1,dtype:f{width})\nu = sqrt(a)\nr = const(4,dtype:f{width})\nreturn r",
            16,
        )
    if case == "divide":
        expression, a, b, flags = "div(a,b)", "1", "0", 8
    else:
        expression, a, b, flags = (
            "add(a,b)",
            "1",
            str(2.0 ** (-24 if width == 32 else -53)),
            1,
        )
    return (
        f"a = const({a},dtype:f{width})\nb = const({b},dtype:f{width})\nu = {expression}\n"
        f"r = const(4,dtype:f{width})\nreturn r"
    ), flags


def multi_function_program(dtype, takes_true):
    b = IRBuilder()
    for index, function in enumerate(("main", "helper")):
        b.new_function(function)
        b.new_block("entry")
        a = b.load_const(1 if takes_true else 3, dtype)
        c = b.load_const(2, dtype)
        b.br_compare(a, "<", c, "a_b", ".Lelse")
        b.new_block("a_b")
        b.ret(b.load_const(11 + 10 * index, dtype))
        b.new_block(".Lelse")
        b.ret(b.load_const(12 + 10 * index, dtype))
    assert verify_ir(b.program) == (True, [])
    return b.program


def unsupported_program(position):
    b = IRBuilder()
    function = b.new_function("main")
    b.new_block("entry")
    unsupported = b.make_value(
        "outside_scope", dtype=T.INT64, is_constant=True, const_value=1
    )
    if position == "global":
        b.program.global_values.append(unsupported)
    elif position == "parameter":
        function.params.append(unsupported)
    elif position == "local":
        function.locals.append(unsupported)
    elif position == "alloca":
        b.alloca(1, T.INT64)
    elif position == "operand":
        b.add(unsupported, unsupported)
    elif position == "return":
        b.ret(unsupported)
        return b.program
    b.ret(b.load_const(7, T.INT32))
    return b.program


class TestOptimizerAndTypeBoundaries:
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("argument", [7, -7, 0, -2147483648])
    def test_i32_onnx_parameter_is_materialized(self, mode, argument, tmp_path):
        from onnx.reference import ReferenceEvaluator

        path, model = onnx_div_model(tmp_path, T.INT32, parameter=True)
        expected = int(
            ReferenceEvaluator(model).run(
                None, {"a": np.asarray(argument, dtype=np.int32)}
            )[0]
        )
        result = CompilerDriver(config(mode, "all", True)).compile(
            str(path), str(tmp_path / "out.s")
        )
        assert result.success, result.errors
        assert execute_i32(result.output_text, tmp_path, argument=argument) == expected

    @pytest.mark.parametrize("verify", [False, True])
    @pytest.mark.parametrize("case", ["two", "nine", "array", "unknown"])
    def test_unsupported_integer_parameter_layout_is_rejected(
        self, verify, case, tmp_path
    ):
        b = IRBuilder()
        function = b.new_function("main")
        b.new_block("entry")
        count = {"two": 2, "nine": 9}.get(case, 1)
        function.params = [b.make_value(f"arg{i}", dtype=T.INT32) for i in range(count)]
        if case in {"array", "unknown"}:
            function.params[0].shape = (2,) if case == "array" else (0,)
        b.ret(b.div(function.params[0], b.load_const(3, T.INT32)))
        assert verify_ir(b.program) == (True, [])
        result = compile_ir(b.program, tmp_path, config("greedy", verify=verify))
        assert not result.success
        assert any(
            "scalar parameter" in error for error in result.errors
        ), result.errors
        assert not (tmp_path / "out.s").exists()

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("level", LEVELS)
    @pytest.mark.parametrize("dtype", [T.INT32, T.FLOAT32, T.FLOAT64])
    def test_dce_keeps_cross_block_definitions(self, mode, level, dtype, tmp_path):
        program = cross_block_program(dtype)
        result = compile_ir(program, tmp_path, config(mode, level, True))
        assert result.success, result.errors
        assert verify_ir(program) == (True, [])
        if dtype is T.INT32:
            assert execute_i32(result.output_text, tmp_path) == 2
        else:
            width = 32 if dtype is T.FLOAT32 else 64
            assert execute_fp(result.output_text, tmp_path, width) == (
                bits(2.0, width),
                0,
            )

    @pytest.mark.parametrize("dtype", [T.INT32, T.FLOAT32, T.FLOAT64])
    @pytest.mark.parametrize(
        "passes", [("dead-code-elim",), ("dead-code-elim", "dead-code-elim")]
    )
    def test_explicit_repeated_dce_keeps_valid_ir(self, dtype, passes, tmp_path):
        program = cross_block_program(dtype)
        result = compile_ir(program, tmp_path, config(passes=passes))
        assert result.success, result.errors
        assert verify_ir(program) == (True, [])

    @pytest.mark.parametrize("level", LEVELS)
    @pytest.mark.parametrize("width", [32, 64])
    @pytest.mark.parametrize("case", ["sqrt", "divide", "inexact"])
    def test_dce_preserves_unused_fp_exceptions(self, level, width, case, tmp_path):
        source, flags = exception_source(width, case)
        result = CompilerDriver(config("greedy", level, True)).compile(
            "", str(tmp_path / "out.s"), dsl_source=source
        )
        assert result.success, result.errors
        assert execute_fp(result.output_text, tmp_path, width) == (
            bits(4.0, width),
            flags,
        )

    @pytest.mark.parametrize("level", LEVELS)
    @pytest.mark.parametrize("width,dtype", FLOATS)
    def test_unused_min_with_signaling_nan_still_sets_invalid(
        self, level, width, dtype, tmp_path
    ):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        a = b.load_const(1.0, dtype)
        b.current_block.instructions[-1].attrs["bit_pattern"] = (
            0x7F800001 if width == 32 else 0x7FF0000000000001
        )
        b.min(a, b.load_const(2.0, dtype))
        b.ret(b.load_const(4.0, dtype))
        assert verify_ir(b.program) == (True, [])
        result = compile_ir(b.program, tmp_path, config("greedy", level))
        assert result.success, result.errors
        assert execute_fp(result.output_text, tmp_path, width) == (bits(4.0, width), 16)

    @pytest.mark.parametrize("width", [32, 64])
    @pytest.mark.parametrize(
        "passes", [("dead-code-elim",), ("dead-code-elim", "dead-code-elim")]
    )
    def test_explicit_repeated_dce_uses_same_fp_policy(
        self, width, passes, tmp_path
    ):
        source, flags = exception_source(width, "sqrt")
        result = CompilerDriver(config(passes=passes)).compile(
            "", str(tmp_path / "out.s"), dsl_source=source
        )
        assert result.success, result.errors
        assert execute_fp(result.output_text, tmp_path, width) == (
            bits(4.0, width),
            flags,
        )

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("dtype", [T.INT32, T.FLOAT32, T.FLOAT64])
    @pytest.mark.parametrize("takes_true", [False, True])
    def test_multifunction_labels_and_branch_targets_are_scoped(
        self, mode, dtype, takes_true, tmp_path
    ):
        program = multi_function_program(dtype, takes_true)
        result = compile_ir(program, tmp_path, config(mode, "all"))
        assert result.success, result.errors
        assert verify_ir(program) == (True, [])
        labels = [
            line.strip()
            for line in result.output_text.splitlines()
            if line.strip().endswith(":")
        ]
        assert len(labels) == len(set(labels))
        for index, function in enumerate(("main", "helper")):
            expected = (11 if takes_true else 12) + 10 * index
            if dtype is T.INT32:
                assert (
                    execute_i32(result.output_text, tmp_path, function=function)
                    == expected
                )
            else:
                width = 32 if dtype is T.FLOAT32 else 64
                assert execute_fp(
                    result.output_text, tmp_path, width, function=function
                ) == (bits(expected, width), 0)

    def test_selector_rerun_has_stable_namespaced_labels(self):
        selector = ExtendedInstructionSelector(multi_function_program(T.INT32, True))
        first = selector.run()
        second = selector.run()
        assert [
            i.target for i in first if i.op is not None and i.op.value == "label"
        ] == [i.target for i in second if i.op.value == "label"]

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("extended", [False, True])
    def test_unconditional_branch_uses_scoped_labels_in_both_selectors(
        self, mode, extended, tmp_path
    ):
        b = IRBuilder()
        for index, function in enumerate(("main", "helper")):
            b.new_function(function)
            b.new_block("entry")
            b.br(".Lexit")
            b.new_block(".Lexit")
            b.ret(b.load_const(31 + index, T.INT32))
        assert verify_ir(b.program) == (True, [])
        configuration = CompilerConfig(
            use_extended_isel=extended,
            reg_alloc=mode,
            optimize_level="all",
            verify_ir=True,
            target_isa="rv32im",
            target_abi="ilp32",
        )
        result = compile_ir(b.program, tmp_path, configuration)
        assert result.success, result.errors
        for index, function in enumerate(("main", "helper")):
            assert (
                execute_i32(result.output_text, tmp_path, function=function)
                == 31 + index
            )

    @pytest.mark.parametrize("verify", [False, True])
    def test_onnx_i64_is_rejected_before_optimization(self, verify, tmp_path):
        from onnx.reference import ReferenceEvaluator

        path, model = onnx_div_model(tmp_path, T.INT64)
        assert int(ReferenceEvaluator(model).run(None, {})[0]) == 1073741824
        output = tmp_path / "out.s"
        output.write_text("preserve previous output")
        result = CompilerDriver(config("greedy", "all", verify)).compile(
            str(path), str(output)
        )
        assert not result.success
        assert any(
            "i64" in error and "extended" in error for error in result.errors
        ), result.errors
        assert output.read_text() == "preserve previous output"

    @pytest.mark.parametrize(
        "position", ["global", "parameter", "local", "alloca", "operand", "return"]
    )
    @pytest.mark.parametrize("verify", [False, True])
    def test_unsupported_unused_or_referenced_types_are_not_hidden_by_dce(
        self, position, verify, tmp_path
    ):
        result = compile_ir(
            unsupported_program(position), tmp_path, config(level="all", verify=verify)
        )
        assert not result.success
        assert any(
            "i64" in error and "extended" in error for error in result.errors
        ), result.errors

    @pytest.mark.parametrize(
        "position", ["global", "parameter", "local", "alloca", "operand", "return"]
    )
    def test_direct_selector_has_same_type_boundary(self, position):
        with pytest.raises(ValueError, match="extended.*i64|i64.*extended"):
            ExtendedInstructionSelector(unsupported_program(position)).run()

    def test_dce_still_removes_unused_integer_arithmetic(self):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        b.add(b.load_const(1, T.INT32), b.load_const(2, T.INT32))
        b.ret(b.load_const(7, T.INT32))
        assert DeadCodeEliminator().optimize(b.program) == 1
        assert not any(i.opcode is OpCode.ADD for i in b.current_block.instructions)
        assert verify_ir(b.program) == (True, [])

    @pytest.mark.parametrize("dtype", [T.FLOAT32, T.FLOAT64])
    @pytest.mark.parametrize("preserve", [False, True])
    def test_float_policy_does_not_disable_constant_dce_or_change_default(
        self, dtype, preserve
    ):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        b.sqrt(b.load_const(-1.0, dtype))
        unused = b.load_const(9.0, dtype)
        b.ret(b.load_const(4.0, dtype))
        pass_ = (
            DeadCodeEliminator(preserve_float_semantics=True)
            if preserve
            else DeadCodeEliminator()
        )
        assert pass_.optimize(b.program) == (1 if preserve else 2)
        assert not any(i.dest is unused for i in b.current_block.instructions)
        assert (
            any(i.opcode is OpCode.SQRT for i in b.current_block.instructions)
            is preserve
        )
        assert verify_ir(b.program) == (True, [])
