"""LLVM scalar constants must parse and retain their storage bits (R7)."""

import ctypes

import onnx
import pytest

from scratchv.analysis.ir_verifier import verify_ir
from scratchv.backend.llvm_codegen import LLVMCodegen
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType


llvm = pytest.importorskip("llvmlite.binding")


def _return_bits(assembly, width):
    """Observe raw bits before Python can convert or quiet a floating result."""
    floating_type = "float" if width == 32 else "double"
    integer_type = f"i{width}"
    wrapper = f"""
define {integer_type} @r7_return_bits() {{
entry:
  %value = call {floating_type} @main()
  %bits = bitcast {floating_type} %value to {integer_type}
  ret {integer_type} %bits
}}
"""
    module = llvm.parse_assembly(assembly + wrapper)
    module.verify()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    # Adapt only the execution target; keep the generated function unchanged.
    module.triple = llvm.get_default_triple()
    target = llvm.Target.from_default_triple().create_target_machine()
    module.data_layout = str(target.target_data)
    with llvm.create_mcjit_compiler(module, target) as engine:
        engine.finalize_object()
        return_type = ctypes.c_uint32 if width == 32 else ctypes.c_uint64
        function = ctypes.CFUNCTYPE(return_type)(
            engine.get_function_address("r7_return_bits")
        )
        return function()


class TestLLVMScalarConstants:
    @pytest.mark.parametrize("level", ("none", "basic", "all"))
    @pytest.mark.parametrize(
        "dtype,token,expected",
        (
            ("f32", "1.2", 0x3F99999A),
            ("f64", "1.0000000000000002", 0x3FF0000000000001),
            ("f64", "-0.0", 0x8000000000000000),
        ),
    )
    def test_typed_dsl_constant_parses_and_returns_exact_bits(
        self, tmp_path, level, dtype, token, expected
    ):
        output = tmp_path / "constant.ll"
        result = CompilerDriver(
            CompilerConfig(backend="llvm", optimize_level=level)
        ).compile(
            "",
            str(output),
            dsl_source=f"a = const({token},dtype:{dtype})\nreturn a\n",
        )
        assert result.success, result.errors
        assert output.read_text() == result.output_text
        assert _return_bits(result.output_text, int(dtype[1:])) == expected

    @pytest.mark.parametrize("width", (32, 64))
    @pytest.mark.parametrize("source", ("instruction", "value", "both", "bare"))
    @pytest.mark.parametrize(
        "case", ("one", "negative_zero", "quiet_nan", "signaling_nan")
    )
    def test_ir_constant_encoding_is_authoritative(self, width, source, case):
        patterns = {
            "one": (0x3F800000, 0x3FF0000000000000),
            "negative_zero": (0x80000000, 0x8000000000000000),
            "quiet_nan": (0x7FC01234, 0x7FF8000000001234),
            "signaling_nan": (0x7F801234, 0x7FF0000000001234),
        }
        bits = patterns[case][width == 64]
        dtype = DataType.FLOAT32 if width == 32 else DataType.FLOAT64
        builder = IRBuilder()
        builder.new_function("main")
        builder.new_block("entry")
        if source == "bare":
            value = builder.make_const(0.0, dtype)
            value.bit_pattern = bits
        else:
            value = builder.load_const(0.0, dtype, bit_pattern=bits)
            definition = builder.current_block.instructions[-1]
            if source == "instruction":
                value.bit_pattern = None
            elif source == "value":
                definition.attrs.pop("bit_pattern")
        builder.ret(value)
        assert verify_ir(builder.program) == (True, [])
        assert _return_bits(LLVMCodegen(builder.program).emit(), width) == bits

    @pytest.mark.parametrize("level", ("none", "basic", "all"))
    def test_f32_operands_are_rounded_before_constant_arithmetic(self, tmp_path, level):
        # 16777217 rounds to 16777216 in binary32 before the subtraction.
        source = (
            "a = const(16777217.0,dtype:f32)\n"
            "b = const(16777216.0,dtype:f32)\n"
            "r = sub(a,b)\nreturn r\n"
        )
        result = CompilerDriver(
            CompilerConfig(backend="llvm", optimize_level=level)
        ).compile(
            "",
            str(tmp_path / "rounding.ll"),
            dsl_source=source,
        )
        assert result.success, result.errors
        assert _return_bits(result.output_text, 32) == 0x00000000

    @pytest.mark.parametrize("level", ("none", "basic", "all"))
    @pytest.mark.parametrize(
        "dtype,token,expected",
        (
            ("f32", "0.1", 0x3DCCCCCD),
            ("f32", "-0.0", 0x80000000),
            ("f32", "0.0", 0x00000000),
            ("f32", "inf", 0x7F800000),
            ("f32", "-inf", 0xFF800000),
            ("f32", "nan", 0x7FC00000),
            ("f32", "1.401298464324817e-45", 0x00000001),
            ("f32", "1.1754943508222875e-38", 0x00800000),
            ("f32", "3.4028234663852886e38", 0x7F7FFFFF),
            ("f32", "1.0000000596046448", 0x3F800000),
            ("f32", "1.000000059604645", 0x3F800001),
            ("f64", "0.1", 0x3FB999999999999A),
            ("f64", "0.0", 0x0000000000000000),
            ("f64", "inf", 0x7FF0000000000000),
            ("f64", "-inf", 0xFFF0000000000000),
            ("f64", "nan", 0x7FF8000000000000),
            ("f64", "5e-324", 0x0000000000000001),
            ("f64", "2.2250738585072014e-308", 0x0010000000000000),
            ("f64", "1.7976931348623157e308", 0x7FEFFFFFFFFFFFFF),
        ),
    )
    def test_dsl_boundary_constant_returns_exact_bits(
        self, tmp_path, level, dtype, token, expected
    ):
        result = CompilerDriver(
            CompilerConfig(backend="llvm", optimize_level=level)
        ).compile(
            "",
            str(tmp_path / "boundary.ll"),
            dsl_source=f"a = const({token},dtype:{dtype})\nreturn a\n",
        )
        assert result.success, result.errors
        assert _return_bits(result.output_text, int(dtype[1:])) == expected

    @pytest.mark.parametrize("level", ("none", "basic", "all"))
    @pytest.mark.parametrize("width", (32, 64))
    @pytest.mark.parametrize(
        "case",
        (
            "ordinary",
            "positive_zero",
            "negative_zero",
            "infinity",
            "negative_infinity",
            "subnormal",
            "quiet_nan",
            "signaling_nan",
        ),
    )
    def test_onnx_scalar_initializer_keeps_original_bits(
        self, tmp_path, level, width, case
    ):
        patterns = {
            "ordinary": (0x3F99999A, 0x3FF0000000000001),
            "positive_zero": (0x00000000, 0x0000000000000000),
            "negative_zero": (0x80000000, 0x8000000000000000),
            "infinity": (0x7F800000, 0x7FF0000000000000),
            "negative_infinity": (0xFF800000, 0xFFF0000000000000),
            "subnormal": (0x00000001, 0x0000000000000001),
            "quiet_nan": (0x7FC01234, 0x7FF8000000001234),
            "signaling_nan": (0x7F801234, 0x7FF0000000001234),
        }
        bits = patterns[case][width == 64]
        tensor_type = onnx.TensorProto.FLOAT if width == 32 else onnx.TensorProto.DOUBLE
        initializer = onnx.TensorProto(
            name="a",
            data_type=tensor_type,
            raw_data=bits.to_bytes(width // 8, "little"),
        )
        graph = onnx.helper.make_graph(
            [],
            "main",
            [],
            [onnx.helper.make_tensor_value_info("a", tensor_type, [])],
            [initializer],
        )
        model = onnx.helper.make_model(
            graph,
            opset_imports=[onnx.helper.make_opsetid("", 13)],
            ir_version=9,
        )
        onnx.checker.check_model(model)
        path = tmp_path / "constant.onnx"
        onnx.save(model, path)
        result = CompilerDriver(
            CompilerConfig(backend="llvm", optimize_level=level)
        ).compile(
            str(path),
            str(tmp_path / "constant.ll"),
        )
        assert result.success, result.errors
        assert _return_bits(result.output_text, width) == bits

    @pytest.mark.parametrize(
        "dtype,expected",
        (
            ("f32", 0x80000000),
            ("f64", 0x8000000000000000),
        ),
    )
    @pytest.mark.parametrize(
        "passes",
        (
            ("ir-peephole",),
            ("constant-folding", "ir-peephole", "constant-folding"),
        ),
    )
    def test_explicit_passes_do_not_erase_negative_zero(
        self, tmp_path, dtype, expected, passes
    ):
        source = (
            f"a = const(-0.0,dtype:{dtype})\n"
            f"b = const(0.0,dtype:{dtype})\n"
            "r = mul(a,b)\nreturn r\n"
        )
        result = CompilerDriver(
            CompilerConfig(
                backend="llvm",
                optimize_level="none",
                passes=passes,
                verify_ir=True,
            )
        ).compile("", str(tmp_path / "passes.ll"), dsl_source=source)
        assert result.success, result.errors
        assert _return_bits(result.output_text, int(dtype[1:])) == expected

    @pytest.mark.parametrize("width", (32, 64))
    @pytest.mark.parametrize("source", ("instruction", "value", "bare"))
    @pytest.mark.parametrize("invalid", ("negative", "too_wide", "boolean", "fraction"))
    def test_invalid_encoding_is_rejected_without_overwriting_output(
        self, tmp_path, width, source, invalid
    ):
        bits = {
            "negative": -1,
            "too_wide": 1 << width,
            "boolean": True,
            "fraction": 1.5,
        }[invalid]
        dtype = DataType.FLOAT32 if width == 32 else DataType.FLOAT64
        builder = IRBuilder()
        builder.new_function("main")
        builder.new_block("entry")
        value = (
            builder.make_const(0.0, dtype)
            if source == "bare"
            else builder.load_const(0.0, dtype)
        )
        if source == "instruction":
            builder.current_block.instructions[-1].attrs["bit_pattern"] = bits
        else:
            value.bit_pattern = bits
        builder.ret(value)
        output = tmp_path / "existing.ll"
        output.write_text("previous output")
        with pytest.raises(ValueError, match="bit_pattern"):
            LLVMCodegen(builder.program).save(str(output))
        assert output.read_text() == "previous output"

    @pytest.mark.parametrize("width", (32, 64))
    def test_conflicting_encodings_are_rejected_without_overwriting_output(
        self, tmp_path, width
    ):
        dtype = DataType.FLOAT32 if width == 32 else DataType.FLOAT64
        builder = IRBuilder()
        builder.new_function("main")
        builder.new_block("entry")
        value = builder.load_const(0.0, dtype, bit_pattern=0)
        builder.current_block.instructions[-1].attrs["bit_pattern"] = 1 << (width - 1)
        builder.ret(value)
        output = tmp_path / "existing.ll"
        output.write_text("previous output")
        with pytest.raises(ValueError, match="bit_pattern differs"):
            LLVMCodegen(builder.program).save(str(output))
        assert output.read_text() == "previous output"
