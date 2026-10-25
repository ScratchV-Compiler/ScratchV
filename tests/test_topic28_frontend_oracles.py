"""Typed guards and honest FP references.

The DSL interpreter, straight-line IR trace and QEMU are kept as independent
oracles; reference-only results are never treated as binary evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.frontend.dsl_extended import ExtendedDSLParser
from scratchv.frontend.dsl_parser import DSLParseError, DSLParser
from scratchv.ir.types import DataType, Instruction, OpCode, Value
from scratchv.simulator.rv32_emulator import IRTraceExecutor
from scratchv.verification.verifier import DSLInterpreter, verify_dsl
from tests.topic28_runtime import execute_fp, execute_i32


def guard_source(kind, condition, dtype="i32"):
    if kind == "if":
        body = " return a\nelse:\n return a\nendif"
    else:
        # False guards still exercise both parse/verify and binary lowering.
        body = " return a\nendwhile\nreturn a"
    return f"a = const(1,dtype:{dtype})\n{kind} ({condition}):\n{body}\n"


def run_reference(source, implementation):
    if implementation == "dsl":
        return DSLInterpreter().run(source, {})
    return IRTraceExecutor(DSLParser().parse(source)).run({})


FP_MIN_MAX_CASES = [
    ("min", "nan", "4", 4.0),
    ("min", "4", "nan", 4.0),
    ("max", "nan", "4", 4.0),
    ("max", "4", "nan", 4.0),
    ("min", "-0.0", "0.0", -0.0),
    ("min", "0.0", "-0.0", -0.0),
    ("max", "-0.0", "0.0", 0.0),
    ("max", "0.0", "-0.0", 0.0),
    ("min", "nan", "nan", float("nan")),
    ("max", "nan", "nan", float("nan")),
    ("min", "1.0000000000000002", "1.0000000000000004", 1.0000000000000002),
    ("max", "1.0000000000000002", "1.0000000000000004", 1.0000000000000004),
]


def compile_source(source, tmp_path, width, mode):
    path = tmp_path / "review.dsl"
    path.write_text(source)
    result = CompilerDriver(
        CompilerConfig(
            use_extended_isel=True,
            verify_ir=True,
            reg_alloc=mode,
            target_isa="rv32im" if width == 0 else "rv32imfd",
            target_abi="ilp32" if width == 0 else "ilp32d",
        )
    ).compile(str(path))
    assert result.success, result.errors
    return result.output_text


class TestFrontendReferenceSemantics:
    @pytest.mark.parametrize("kind", ["if", "while"])
    @pytest.mark.parametrize(
        "condition",
        [
            "a < 3",
            "3 > a",
            "a > -2147483648",
            "2147483647 > a",
            "a < 3e0",
            "3E+0 > a",
        ],
    )
    def test_integer_guards_emit_actual_int_constants(
        self, kind, condition, tmp_path
    ):
        source = guard_source(kind, condition)
        program = ExtendedDSLParser().parse(source)
        constants = [
            instr
            for func in program.functions
            for block in func.blocks
            for instr in block.instructions
            if instr.opcode is OpCode.LOAD_CONST
        ]
        assert all(instr.dest.dtype is DataType.INT32 for instr in constants)
        assert all(type(instr.attrs["value"]) is int for instr in constants)
        path = tmp_path / "r6.dsl"
        path.write_text(source)
        result = CompilerDriver(
            CompilerConfig(
                use_extended_isel=True,
                verify_ir=True,
            )
        ).compile(str(path))
        assert result.success, result.errors

    @pytest.mark.parametrize("kind", ["if", "while"])
    @pytest.mark.parametrize(
        "literal",
        [
            "1.5",
            "-1.5",
            "2147483648",
            "-2147483649",
            "1.0000000000000000001",
            "1e10000",
        ],
    )
    @pytest.mark.parametrize("left_literal", [False, True])
    def test_integer_guards_reject_fraction_or_out_of_range(
        self, kind, literal, left_literal
    ):
        condition = f"{literal} < a" if left_literal else f"a < {literal}"
        with pytest.raises(DSLParseError, match="i32.*literal|i32.*range"):
            ExtendedDSLParser().parse(guard_source(kind, condition))

    @pytest.mark.parametrize("kind", ["if", "while"])
    @pytest.mark.parametrize("literal", ["1e-3", "1e+3", "-1E-3", "+1E+3"])
    @pytest.mark.parametrize("left_literal", [False, True])
    def test_scientific_conditions_validator_and_parser_agree(
        self, kind, literal, left_literal, tmp_path
    ):
        condition = f"{literal} < a" if left_literal else f"a < {literal}"
        source = guard_source(kind, condition, "f64")
        parser = ExtendedDSLParser()
        assert not parser.validate(source).has_errors
        parser.parse(source)
        path = tmp_path / "r9.dsl"
        path.write_text(source)
        result = CompilerDriver(
            CompilerConfig(
                use_extended_isel=True,
                verify_ir=True,
                target_isa="rv32imfd",
                target_abi="ilp32d",
            )
        ).compile(str(path))
        assert result.success, result.errors

    @pytest.mark.parametrize("kind", ["if", "while"])
    @pytest.mark.parametrize("literal", ["1e", "1e+", "--1e-3", "1.2.3"])
    def test_invalid_scientific_conditions_rejected_by_both(self, kind, literal):
        source = guard_source(kind, f"a < {literal}", "f64")
        parser = ExtendedDSLParser()
        assert parser.validate(source).has_errors
        with pytest.raises(DSLParseError):
            parser.parse(source)

    @pytest.mark.parametrize("implementation", ["dsl", "ir"])
    @pytest.mark.parametrize("dtype", ["f32", "f64"])
    def test_sqrt_reference_is_defined_and_preserves_precision(
        self, implementation, dtype
    ):
        source = f"a = const(4,dtype:{dtype})\nr = sqrt(a)\nreturn r"
        result = run_reference(source, implementation)
        assert result.dtype == np.dtype(dtype.replace("f", "float"))
        assert float(result) == 2.0

    @pytest.mark.parametrize("implementation", ["dsl", "ir"])
    @pytest.mark.parametrize("dtype", ["f32", "f64"])
    @pytest.mark.parametrize("op,left,right,expected", FP_MIN_MAX_CASES)
    def test_min_max_reference_number_nan_zero_and_precision(
        self,
        implementation,
        dtype,
        op,
        left,
        right,
        expected,
    ):
        source = f"a = const({left},dtype:{dtype})\nb = const({right},dtype:{dtype})\nr = {op}(a,b)\nreturn r"
        result = run_reference(source, implementation)
        numpy_dtype = np.float32 if dtype == "f32" else np.float64
        assert result.dtype == numpy_dtype
        if np.isnan(expected):
            assert np.isnan(result)
        else:
            assert float(result) == float(numpy_dtype(expected))
            if expected == 0:
                assert bool(np.signbit(result)) == bool(np.signbit(expected))

    @pytest.mark.parametrize("implementation", ["dsl", "ir"])
    def test_undefined_values_raise_instead_of_defaulting_to_zero(
        self, implementation
    ):
        with pytest.raises(ValueError, match="[Uu]ndefined.*missing"):
            run_reference("return missing", implementation)

    @pytest.mark.parametrize("implementation", ["dsl", "ir"])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("op", ["min", "max"])
    def test_min_max_reference_broadcast_arrays_preserve_all_element_bits(
        self, implementation, dtype, op
    ):
        left = np.asarray([np.nan, 4, -0.0, 0.0, -0.0, 0.0], dtype=dtype)
        right = np.asarray([4, np.nan, 0.0, -0.0, -0.0, 0.0], dtype=dtype)
        expected = np.asarray(
            (
                [4, 4, -0.0, -0.0, -0.0, 0.0]
                if op == "min"
                else [4, 4, 0.0, 0.0, -0.0, 0.0]
            ),
            dtype=dtype,
        )
        source = f"r = {op}(a,b)\nreturn r"
        if implementation == "dsl":
            result = DSLInterpreter().run(source, {"a": left, "b": right})
        else:
            result = IRTraceExecutor(DSLParser().parse(source)).run(
                {"a": left, "b": right}
            )
        unsigned = np.uint32 if dtype is np.float32 else np.uint64
        np.testing.assert_array_equal(
            np.asarray(result).view(unsigned), expected.view(unsigned)
        )

    def test_unknown_ir_opcode_raises_instead_of_silent_none(self):
        program = DSLParser().parse("a = const(4)\nreturn a")
        program.functions[0].blocks[0].instructions.insert(
            1,
            Instruction(
                OpCode.TRANSPOSE,
                Value("unhandled"),
                [Value("a")],
            ),
        )
        with pytest.raises(ValueError, match="Unsupported.*transpose"):
            IRTraceExecutor(program).run({})

    @pytest.mark.parametrize("opcode", [OpCode.BR, OpCode.BR_IF])
    def test_straight_line_trace_rejects_cfg_instead_of_ignoring_branches(
        self, opcode
    ):
        program = DSLParser().parse("a = const(4)\nreturn a")
        with pytest.raises(ValueError, match="Unsupported control flow"):
            IRTraceExecutor(program)._exec_instr(Instruction(opcode, target="next"))

    def test_verify_dsl_is_still_only_reference_not_binary_evidence(self):
        result = verify_dsl("a = const(4)\nr = sqrt(a)\nreturn r", {})
        assert float(result["expected"]) == 2.0
        # The existing API is a placeholder. It must not be cited as a QEMU run.
        assert result["got"] is result["expected"]

    @pytest.mark.parametrize("width", [32, 64])
    @pytest.mark.parametrize("op,left,right,expected", FP_MIN_MAX_CASES)
    def test_min_max_reference_bits_match_qemu(
        self, width, op, left, right, expected, tmp_path
    ):
        source = f"a = const({left},dtype:f{width})\nb = const({right},dtype:f{width})\nr = {op}(a,b)\nreturn r"
        assembly = compile_source(source, tmp_path, width, "greedy")
        bits, flags = execute_fp(assembly, tmp_path, width)
        unsigned = np.uint32 if width == 32 else np.uint64
        for implementation in ("dsl", "ir"):
            reference = np.asarray(run_reference(source, implementation))
            assert int(reference.view(unsigned)) == bits
        # All NaNs in these source spellings are quiet; min/max raises no flags.
        assert flags == 0

    @pytest.mark.parametrize("width", [32, 64])
    @pytest.mark.parametrize("literal", ["4.0", "-0.0", "inf"])
    def test_sqrt_reference_bits_match_qemu(self, width, literal, tmp_path):
        source = f"a = const({literal},dtype:f{width})\nr = sqrt(a)\nreturn r"
        bits, flags = execute_fp(
            compile_source(source, tmp_path, width, "greedy"), tmp_path, width
        )
        unsigned = np.uint32 if width == 32 else np.uint64
        for implementation in ("dsl", "ir"):
            assert (
                int(np.asarray(run_reference(source, implementation)).view(unsigned))
                == bits
            )
        assert flags == 0

    @pytest.mark.parametrize("kind", ["if", "while"])
    @pytest.mark.parametrize("left_literal", [False, True])
    def test_verified_integer_control_flow_executes_on_qemu(
        self, kind, left_literal, tmp_path
    ):
        condition = "3 > a" if left_literal else "a < 3"
        if kind == "if":
            initial = 4 if left_literal else 1
            source = (
                f"a = const({initial},dtype:i32)\n"
                "then_value = const(2,dtype:i32)\nelse_value = const(3,dtype:i32)\n"
                f"if ({condition}):\n return then_value\nelse:\n return else_value\nendif\n"
            )
            # Exercise both the taken and untaken path, with different return
            # values so a reversed/ignored predicate cannot pass this assertion.
            expected = 2 if initial < 3 else 3
        else:
            source = "a = const(1,dtype:i32)\nstep = const(1,dtype:i32)\n"
            source += f"while ({condition}):\n a = add(a,step)\nendwhile\nreturn a\n"
            expected = 3
        assert (
            execute_i32(compile_source(source, tmp_path, 0, "greedy"), tmp_path)
            == expected
        )

    @pytest.mark.parametrize("kind", ["if", "while"])
    @pytest.mark.parametrize("literal", ["1e-3", "1e+3", "-1E-3", "+1E+3"])
    @pytest.mark.parametrize("left_literal", [False, True])
    def test_verified_scientific_conditions_execute_on_qemu(
        self, kind, literal, left_literal, tmp_path
    ):
        condition = f"{literal} < a" if left_literal else f"a < {literal}"
        source = (
            "a = const(1,dtype:f64)\n"
            "then_value = const(2,dtype:f64)\nelse_value = const(3,dtype:f64)\n"
        )
        if kind == "if":
            source += f"if ({condition}):\n return then_value\nelse:\n return else_value\nendif\n"
        else:
            # True guards enter the body and return 2; false guards must skip it
            # and return 3. This tests actual control flow rather than parser-only
            # acceptance or two branches which return the same value.
            source += f"while ({condition}):\n return then_value\nendwhile\nreturn else_value\n"
        takes_body = float(literal) < 1.0 if left_literal else 1.0 < float(literal)
        expected_bits = 0x4000000000000000 if takes_body else 0x4008000000000000
        bits, flags = execute_fp(
            compile_source(source, tmp_path, 64, "greedy"), tmp_path, 64
        )
        assert bits == expected_bits
        assert flags == 0
