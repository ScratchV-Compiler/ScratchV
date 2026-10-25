"""Rejected and boundary typed inputs must fail before producing an executable."""

import struct

import numpy as np
import pytest

from scratchv.analysis.ir_verifier import verify_ir
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.frontend.dsl_extended import ExtendedDSLParser
from scratchv.frontend.dsl_parser import DSLParseError, DSLParser
from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType as T
from tests.topic28_runtime import execute_fp, execute_i32


MODES = ("naive", "greedy", "linear")
FLOATS = ((32, T.FLOAT32), (64, T.FLOAT64))


def config(mode="greedy", verify=True, **kwargs):
    return CompilerConfig(
        use_extended_isel=True,
        reg_alloc=mode,
        verify_ir=verify,
        target_isa="rv32imfd",
        target_abi="ilp32d",
        **kwargs,
    )


def float_model(tmp_path, width, shape, *, output_shape=None):
    onnx = pytest.importorskip("onnx")
    dtype = onnx.TensorProto.FLOAT if width == 32 else onnx.TensorProto.DOUBLE
    array_type = np.float32 if width == 32 else np.float64
    data = np.full(shape, 4.0, dtype=array_type)
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Sqrt", ["a"], ["out"])],
        "main",
        [],
        [
            onnx.helper.make_tensor_value_info(
                "out", dtype, shape if output_shape is None else output_shape
            )
        ],
        [onnx.numpy_helper.from_array(data, name="a")],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
    )
    onnx.checker.check_model(model)
    path = tmp_path / "shape.onnx"
    onnx.save(model, path)
    return path


class TestTypedInputBoundaries:
    @pytest.mark.parametrize("verify", [False, True])
    @pytest.mark.parametrize("width", [32, 64])
    @pytest.mark.parametrize("shape", [(2,), (1, 2), (0,)])
    def test_non_scalar_initializer_is_rejected(
        self, verify, width, shape, tmp_path
    ):
        path = float_model(tmp_path, width, shape)
        output = tmp_path / "out.s"
        result = CompilerDriver(config("greedy", verify)).compile(
            str(path), str(output)
        )
        assert not result.success, result.output_text
        assert any(
            "scalar" in error and "shape" in error for error in result.errors
        ), result.errors
        assert not output.exists()

    @pytest.mark.parametrize("verify", [False, True])
    @pytest.mark.parametrize("width", [32, 64])
    def test_non_scalar_declared_output_is_rejected(self, verify, width, tmp_path):
        path = float_model(tmp_path, width, (), output_shape=(2,))
        result = CompilerDriver(config("greedy", verify)).compile(
            str(path), str(tmp_path / "out.s")
        )
        assert not result.success
        assert any(
            "ONNX shape inference failed" in error for error in result.errors
        ), result.errors

    @pytest.mark.parametrize("width", [32, 64])
    @pytest.mark.parametrize(
        "case", ["direct_return", "parameter_alias", "initializer_alias"]
    )
    def test_declared_shape_cannot_hide_source_shape(self, width, case, tmp_path):
        onnx = pytest.importorskip("onnx")
        tensor_type = onnx.TensorProto.FLOAT if width == 32 else onnx.TensorProto.DOUBLE
        array_type = np.float32 if width == 32 else np.float64
        inputs, nodes, initializers = [], [], []
        source_shape = () if case == "direct_return" else (2,)
        if case == "parameter_alias":
            inputs = [onnx.helper.make_tensor_value_info("a", tensor_type, [2])]
        else:
            initializers = [
                onnx.numpy_helper.from_array(
                    np.full(source_shape, 4.0, dtype=array_type), name="a"
                )
            ]
        if case != "direct_return":
            nodes = [onnx.helper.make_node("Min", ["a"], ["out"])]
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                nodes,
                "main",
                inputs,
                [
                    onnx.helper.make_tensor_value_info(
                        "a" if case == "direct_return" else "out",
                        tensor_type,
                        [2] if case == "direct_return" else [1],
                    )
                ],
                initializers,
            ),
            opset_imports=[onnx.helper.make_opsetid("", 13)],
        )
        onnx.checker.check_model(model)
        path = tmp_path / "alias.onnx"
        onnx.save(model, path)
        # Upstream shape inference now rejects conflicting declarations before
        # either storage or ABI metadata can be overwritten by an alias.
        from scratchv.frontend.onnx_parser import ONNXParseError

        with pytest.raises(ONNXParseError, match="ONNX shape inference failed"):
            ONNXParser().parse(str(path))
        result = CompilerDriver(config("greedy")).compile(
            str(path), str(tmp_path / "out.s")
        )
        assert not result.success
        assert any(
            "ONNX shape inference failed" in error for error in result.errors
        ), result.errors

    @pytest.mark.parametrize("verify", [False, True])
    @pytest.mark.parametrize("width,dtype", FLOATS)
    @pytest.mark.parametrize("consumer", ["sqrt", "add", "return", "store"])
    def test_inline_fp_constant_without_definition_is_rejected(
        self, verify, width, dtype, consumer, tmp_path
    ):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        a = b.make_value("inline_four", dtype=dtype, is_constant=True, const_value=4.0)
        if consumer == "sqrt":
            b.ret(b.sqrt(a))
        elif consumer == "add":
            b.ret(b.add(b.load_const(1.0, dtype), a))
        elif consumer == "store":
            p = b.alloca(1, dtype)
            b.store(p, a)
            b.ret(b.load(p, dtype))
        else:
            b.ret(a)
        assert verify_ir(b.program) == (True, [])
        driver = CompilerDriver(config("greedy", verify))
        driver._parse = lambda *args: b.program
        result = driver.compile("", str(tmp_path / "out.s"), dsl_source="")
        assert not result.success
        assert any(
            "inline_four" in error and "definition" in error for error in result.errors
        ), result.errors
        assert not (tmp_path / "out.s").exists()

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("width,dtype", FLOATS)
    @pytest.mark.parametrize(
        "source", ["load_const", "global", "parameter", "arithmetic"]
    )
    def test_defined_fp_sources_still_execute(
        self, mode, width, dtype, source, tmp_path
    ):
        b = IRBuilder()
        function = b.new_function("main")
        b.new_block("entry")
        if source == "global":
            a = b.make_value(
                "global_four", dtype=dtype, is_constant=True, const_value=4.0
            )
            b.program.global_values.append(a)
        elif source == "parameter":
            a = b.make_value("input", dtype=dtype)
            function.params.append(a)
        elif source == "arithmetic":
            two = b.load_const(2.0, dtype)
            a = b.add(two, two)
        else:
            a = b.load_const(4.0, dtype)
        b.ret(b.sqrt(a))
        driver = CompilerDriver(config(mode))
        driver._parse = lambda *args: b.program
        result = driver.compile("", str(tmp_path / "out.s"), dsl_source="")
        assert result.success, result.errors
        bits = int.from_bytes(struct.pack("<f" if width == 32 else "<d", 2.0), "little")
        assert execute_fp(
            result.output_text,
            tmp_path,
            width,
            argument=4.0 if source == "parameter" else None,
        ) == (bits, 0)

    @pytest.mark.parametrize("verify", [False, True])
    @pytest.mark.parametrize(
        "literal",
        ["2147483648", "-2147483649", "99999999999999999999", "-99999999999999999999"],
    )
    def test_out_of_range_i32_constant_is_a_frontend_error(
        self, verify, literal, tmp_path
    ):
        source = f"a = const({literal},dtype:i32)\nb = const(2,dtype:i32)\nr = div(a,b)\nreturn r"
        parser = ExtendedDSLParser()
        assert parser.validate(source).has_errors
        with pytest.raises(DSLParseError, match="i32.*range"):
            parser.parse(source)
        output = tmp_path / "out.s"
        output.write_text("preserve previous output")
        result = CompilerDriver(
            CompilerConfig(
                use_extended_isel=True,
                reg_alloc="greedy",
                verify_ir=verify,
                optimize_level="all",
            )
        ).compile("", str(output), dsl_source=source)
        assert not result.success
        assert any(
            "i32" in error and "range" in error for error in result.errors
        ), result.errors
        assert output.read_text() == "preserve previous output"

    @pytest.mark.parametrize("value", [-2147483648, 2147483647])
    def test_i32_boundary_constants_execute(self, value, tmp_path):
        result = CompilerDriver(
            CompilerConfig(
                use_extended_isel=True,
                reg_alloc="greedy",
                verify_ir=True,
                optimize_level="all",
            )
        ).compile(
            "",
            str(tmp_path / "out.s"),
            dsl_source=f"a = const({value},dtype:i32)\nreturn a",
        )
        assert result.success, result.errors
        assert execute_i32(result.output_text, tmp_path) == value

    @pytest.mark.parametrize("literal", ["2147483648", "-2147483649", "9" * 5000])
    def test_parser_dispatch_enforces_range_without_validator(self, literal):
        with pytest.raises(DSLParseError, match="i32.*range"):
            DSLParser()._dispatch_op("const", [literal, "dtype:i32"])

    def test_long_zero_prefixed_integer_has_a_deterministic_range_contract(self):
        literal = "+" + "0" * 5000 + "1"
        source = f"a = const({literal},dtype:i32)\nreturn a"
        program = DSLParser().parse(source)
        assert verify_ir(program) == (True, [])
        assert program.functions[0].blocks[0].instructions[0].attrs["value"] == 1

    @pytest.mark.parametrize("dtype", [T.FLOAT32, T.FLOAT64])
    def test_machine_definitions_are_scoped_to_each_function(self, dtype, tmp_path):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        first = b.load_const(4.0, dtype)
        b.ret(first)
        b.new_function("other")
        b.new_block("entry")
        unresolved = b.make_value(
            first.name, dtype=dtype, is_constant=True, const_value=4.0
        )
        b.ret(b.sqrt(unresolved))
        assert verify_ir(b.program) == (True, [])
        driver = CompilerDriver(config())
        driver._parse = lambda *args: b.program
        result = driver.compile("", str(tmp_path / "out.s"), dsl_source="")
        assert not result.success
        assert any(
            "definition" in error and first.name in error for error in result.errors
        )
