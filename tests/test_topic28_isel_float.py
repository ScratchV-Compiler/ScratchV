"""Floating-point instruction selection: typed IR, constants, memory, comparisons."""

from __future__ import annotations

import math

import pytest

from scratchv.analysis.ir_verifier import verify_ir
from scratchv.backend.asm_emit import AsmEmitter
from scratchv.backend.target_features import TargetFeatures
from scratchv.backend.inst_select_ext import ExtendedInstructionSelector
from scratchv.backend.machine_semantics import validate_register_classes
from scratchv.backend.machine_types import (
    MachineInstr,
    MachineOp,
    MachineOperand,
    RegisterClass,
)
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.frontend.dsl_parser import DSLParser
from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, Instruction, OpCode
from scratchv.main import args_to_config, build_arg_parser


def _features(dtype: DataType) -> TargetFeatures:
    if dtype is DataType.FLOAT32:
        return TargetFeatures("rv32imf", "ilp32f")
    return TargetFeatures("rv32imfd", "ilp32d")


def _binary_selection(method: str, dtype: DataType):
    builder = IRBuilder()
    builder.new_function("test")
    builder.new_block("entry")
    lhs = builder.load_const(4.0, dtype)
    rhs = builder.load_const(2.0, dtype)
    result = getattr(builder, method)(lhs, rhs)
    instructions = ExtendedInstructionSelector(
        builder.program,
        features=_features(dtype),
    ).run()
    return result, next(
        instruction
        for instruction in instructions
        if instruction.dst is not None and instruction.dst.value == result.name
    )


class TestFloatingPointInstructionSelection:
    def test_sqrt_reaches_typed_ir_from_builder_and_dsl(self):
        program = DSLParser().parse("x = const(4.0, dtype:f64)\ny = sqrt(x)\nreturn y")
        sqrt = next(
            instruction
            for instruction in program.functions[0].blocks[0].instructions
            if instruction.opcode is OpCode.SQRT
        )
        assert sqrt.dest.dtype is DataType.FLOAT64
        assert verify_ir(program) == (True, [])

        builder = IRBuilder()
        with pytest.raises(ValueError, match="sqrt does not support i32"):
            builder.sqrt(builder.make_value("x", dtype=DataType.INT32))

    def test_dsl_accepts_scientific_and_ieee_special_float_constants(self):
        program = DSLParser().parse(
            "a = const(1e-45, dtype:f32)\n"
            "b = const(inf, dtype:f32)\n"
            "c = const(nan, dtype:f64)\n"
            "return a"
        )
        values = [
            instruction.attrs["value"]
            for instruction in program.functions[0].blocks[0].instructions
            if instruction.opcode is OpCode.LOAD_CONST
        ]
        assert values[0] == 1e-45
        assert math.isinf(values[1])
        assert math.isnan(values[2])

    @pytest.mark.parametrize(
        ("method", "single", "double"),
        [
            ("add", MachineOp.FADD_S, MachineOp.FADD_D),
            ("sub", MachineOp.FSUB_S, MachineOp.FSUB_D),
            ("mul", MachineOp.FMUL_S, MachineOp.FMUL_D),
            ("div", MachineOp.FDIV_S, MachineOp.FDIV_D),
        ],
    )
    @pytest.mark.parametrize("dtype", [DataType.FLOAT32, DataType.FLOAT64])
    def test_float_arithmetic_is_typed_and_uses_static_rne(
        self,
        method,
        single,
        double,
        dtype,
    ):
        _, selected = _binary_selection(method, dtype)
        assert selected.op is (single if dtype is DataType.FLOAT32 else double)
        assert selected.rounding_mode == "rne"
        assert {
            selected.dst.reg_class,
            selected.src1.reg_class,
            selected.src2.reg_class,
        } == {RegisterClass.FPR}

    @pytest.mark.parametrize(
        ("method", "single", "double"),
        [
            ("min", MachineOp.FMIN_S, MachineOp.FMIN_D),
            ("max", MachineOp.FMAX_S, MachineOp.FMAX_D),
        ],
    )
    @pytest.mark.parametrize("dtype", [DataType.FLOAT32, DataType.FLOAT64])
    def test_float_min_max_have_no_rounding_operand(
        self, method, single, double, dtype
    ):
        _, selected = _binary_selection(method, dtype)
        assert selected.op is (single if dtype is DataType.FLOAT32 else double)
        assert selected.rounding_mode is None

    @pytest.mark.parametrize(
        ("method", "single", "double", "has_rounding"),
        [
            ("neg", MachineOp.FNEG_S, MachineOp.FNEG_D, False),
            ("abs", MachineOp.FABS_S, MachineOp.FABS_D, False),
            ("sqrt", MachineOp.SQRT_S, MachineOp.SQRT_D, True),
        ],
    )
    @pytest.mark.parametrize("dtype", [DataType.FLOAT32, DataType.FLOAT64])
    def test_float_unary_selection(self, method, single, double, has_rounding, dtype):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        source = builder.load_const(4.0, dtype)
        result = getattr(builder, method)(source)
        instructions = ExtendedInstructionSelector(
            builder.program,
            features=_features(dtype),
        ).run()
        selected = next(
            instruction
            for instruction in instructions
            if instruction.dst is not None and instruction.dst.value == result.name
        )
        assert selected.op is (single if dtype is DataType.FLOAT32 else double)
        assert selected.dst.reg_class is RegisterClass.FPR
        assert selected.src1.reg_class is RegisterClass.FPR
        assert (selected.rounding_mode == "rne") is has_rounding

    @pytest.mark.parametrize(
        ("dtype", "load_op", "store_op"),
        [
            (DataType.FLOAT32, MachineOp.FLW, MachineOp.FSW),
            (DataType.FLOAT64, MachineOp.FLD, MachineOp.FSD),
        ],
    )
    def test_float_load_store_preserve_ir_store_order(self, dtype, load_op, store_op):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        address = builder.make_value("address", dtype=dtype)
        loaded = builder.load(address, dtype=dtype)
        builder.store(address, loaded)
        instructions = ExtendedInstructionSelector(
            builder.program,
            features=_features(dtype),
        ).run()
        load = next(item for item in instructions if item.op is load_op)
        store = next(item for item in instructions if item.op is store_op)
        assert load.dst.reg_class is RegisterClass.FPR
        assert load.src1.reg_class is RegisterClass.GPR
        assert store.dst.value == loaded.name
        assert store.dst.reg_class is RegisterClass.FPR
        assert store.src1.value == address.name
        assert store.src1.reg_class is RegisterClass.GPR

    def test_float_memory_assembly_uses_offset_base_syntax(self):
        assembly = AsmEmitter(
            [
                MachineInstr(
                    MachineOp.FLW,
                    MachineOperand.reg("ft0", RegisterClass.FPR),
                    MachineOperand.reg("t0", RegisterClass.GPR),
                ),
                MachineInstr(
                    MachineOp.FSD,
                    MachineOperand.reg("ft1", RegisterClass.FPR),
                    MachineOperand.reg("sp", RegisterClass.GPR),
                ),
            ]
        ).emit()
        assert "flw ft0, 0(t0)" in assembly
        assert "fsd ft1, 0(sp)" in assembly

    @pytest.mark.parametrize("predicate", ["==", "!=", "<", ">", "<=", ">="])
    @pytest.mark.parametrize("dtype", [DataType.FLOAT32, DataType.FLOAT64])
    def test_float_comparison_has_fpr_inputs_and_gpr_result(self, predicate, dtype):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        lhs = builder.load_const(4.0, dtype)
        rhs = builder.load_const(2.0, dtype)
        builder.br_compare(lhs, predicate, rhs, "yes", "no")
        instructions = ExtendedInstructionSelector(
            builder.program,
            features=_features(dtype),
        ).run()
        comparisons = {
            MachineOp.FEQ_S,
            MachineOp.FEQ_D,
            MachineOp.FLT_S,
            MachineOp.FLT_D,
            MachineOp.FLE_S,
            MachineOp.FLE_D,
        }
        compare = next(item for item in instructions if item.op in comparisons)
        assert compare.dst.reg_class is RegisterClass.GPR
        assert compare.src1.reg_class is RegisterClass.FPR
        assert compare.src2.reg_class is RegisterClass.FPR
        assert compare.rounding_mode is None

    def test_constant_pool_is_bit_exact_and_distinguishes_signed_zero(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        builder.load_const(0.0, DataType.FLOAT32)
        builder.load_const(-0.0, DataType.FLOAT32)
        builder.load_const(0.0, DataType.FLOAT32)
        builder.load_const(math.inf, DataType.FLOAT64)
        selector = ExtendedInstructionSelector(
            builder.program,
            features=TargetFeatures("rv32imfd", "ilp32d"),
        )
        instructions = selector.run()
        assert [entry.bit_pattern for entry in selector.constant_pool] == [
            0x00000000,
            0x80000000,
            0x7FF0000000000000,
        ]
        assembly = AsmEmitter(
            instructions,
            constant_pool=selector.constant_pool,
        ).emit()
        assert ".word 0x00000000" in assembly
        assert ".word 0x80000000" in assembly
        assert ".quad 0x7ff0000000000000" in assembly
        assert "li.d" not in assembly

    def test_constant_pool_preserves_ieee_boundaries_and_nan_payloads(self):
        builder = IRBuilder()
        builder.new_function("test")
        block = builder.new_block("entry")
        patterns = (
            0x00000001,  # smallest positive subnormal
            0x00800000,  # smallest positive normal
            0x7F7FFFFF,  # largest finite value
            0x7F800000,  # +Inf
            0xFF800000,  # -Inf
            0x7FC00001,  # quiet NaN payload
            0x7FA00002,  # signaling NaN payload
        )
        for index, bits in enumerate(patterns):
            destination = builder.make_value(
                f"nan{index}",
                dtype=DataType.FLOAT32,
                is_constant=True,
                const_value=math.nan,
            )
            block.add(
                Instruction(
                    OpCode.LOAD_CONST,
                    destination,
                    attrs={"value": math.nan, "bit_pattern": bits},
                )
            )
        selector = ExtendedInstructionSelector(
            builder.program,
            features=TargetFeatures("rv32imf", "ilp32f"),
        )
        selector.run()
        assert tuple(entry.bit_pattern for entry in selector.constant_pool) == patterns

    @pytest.mark.parametrize(
        ("onnx_dtype", "ir_dtype"),
        [("FLOAT", DataType.FLOAT32), ("DOUBLE", DataType.FLOAT64)],
    )
    def test_onnx_sqrt_preserves_float_dtype(self, tmp_path, onnx_dtype, ir_dtype):
        onnx = pytest.importorskip("onnx")
        tensor_type = getattr(onnx.TensorProto, onnx_dtype)
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("Sqrt", ["x"], ["y"])],
            "sqrt",
            [onnx.helper.make_tensor_value_info("x", tensor_type, [1])],
            [onnx.helper.make_tensor_value_info("y", tensor_type, [1])],
        )
        model = onnx.helper.make_model(graph)
        path = tmp_path / "sqrt.onnx"
        onnx.save(model, path)
        program = ONNXParser().parse(str(path))
        sqrt = next(
            instruction
            for instruction in program.functions[0].blocks[0].instructions
            if instruction.opcode is OpCode.SQRT
        )
        assert sqrt.dest.dtype is ir_dtype
        assert verify_ir(program) == (True, [])

    def test_target_isa_abi_combination_is_validated(self):
        with pytest.raises(ValueError, match="ISA/ABI combination"):
            TargetFeatures("rv32im", "ilp32d")

    def test_missing_target_extension_fails_before_selection(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        lhs = builder.make_value("lhs", dtype=DataType.FLOAT32)
        rhs = builder.make_value("rhs", dtype=DataType.FLOAT32)
        builder.add(lhs, rhs)
        with pytest.raises(ValueError, match="requires the RISC-V F extension"):
            ExtendedInstructionSelector(
                builder.program,
                features=TargetFeatures("rv32im", "ilp32"),
            ).run()

    def test_unsupported_float_op_cannot_fall_back_to_integer_selection(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        builder.relu(builder.make_value("x", dtype=DataType.FLOAT32))
        with pytest.raises(ValueError, match="no typed floating-point selection"):
            ExtendedInstructionSelector(
                builder.program,
                features=TargetFeatures("rv32imf", "ilp32f"),
            ).run()

    def test_machine_register_class_conflict_is_rejected(self):
        instructions = [
            MachineInstr(
                MachineOp.ADD,
                MachineOperand.vreg("same", RegisterClass.GPR),
                MachineOperand.vreg("lhs", RegisterClass.GPR),
                MachineOperand.vreg("rhs", RegisterClass.GPR),
            ),
            MachineInstr(
                MachineOp.FNEG_S,
                MachineOperand.vreg("out", RegisterClass.FPR),
                MachineOperand.vreg("same", RegisterClass.FPR),
            ),
        ]
        with pytest.raises(ValueError, match="requires both gpr and fpr"):
            validate_register_classes(instructions)

    def test_compiler_allocates_fpr_program_after_phase3(self, tmp_path):
        result = CompilerDriver(
            CompilerConfig(
                use_extended_isel=True,
                target_isa="rv32imf",
                target_abi="ilp32f",
            )
        ).compile(
            "",
            str(tmp_path / "out.s"),
            dsl_source=(
                "a = const(1.5, dtype:f32)\n"
                "b = const(2.0, dtype:f32)\n"
                "r = add(a, b)\n"
                "return r"
            ),
        )
        assert result.success, result.errors
        assert "fadd.s ft" in result.output_text
        assert "fmv.s fa0" in result.output_text
        assert (tmp_path / "out.s").exists()

    def test_compiler_lowers_float_return_even_without_fp_arithmetic(self, tmp_path):
        result = CompilerDriver(
            CompilerConfig(
                use_extended_isel=True,
                target_isa="rv32imf",
                target_abi="ilp32f",
            )
        ).compile(
            "",
            str(tmp_path / "out.s"),
            dsl_source="x = const(1.5,dtype:f32)\nreturn x",
        )
        assert result.success, result.errors
        assert "fmv.s fa0" in result.output_text

    def test_float_return_requires_target_extension_before_phase3_boundary(
        self, tmp_path
    ):
        result = CompilerDriver(
            CompilerConfig(
                use_extended_isel=True,
                target_isa="rv32im",
                target_abi="ilp32",
            )
        ).compile(
            "",
            str(tmp_path / "out.s"),
            dsl_source="return input_value",
        )
        assert not result.success
        assert "requires the RISC-V F extension" in result.errors[0]

    def test_target_cli_options_reach_compiler_config(self):
        args = build_arg_parser().parse_args(
            [
                "input.dsl",
                "--extended-isel",
                "--target-isa",
                "rv32imfd",
                "--target-abi",
                "ilp32d",
            ]
        )
        config = args_to_config(args)
        assert (config.target_isa, config.target_abi) == (
            "rv32imfd",
            "ilp32d",
        )

    def test_rounding_mode_is_rejected_on_non_rounding_operation(self):
        with pytest.raises(ValueError, match="does not take a rounding mode"):
            MachineInstr(MachineOp.FMIN_S, rounding_mode="rne")

    def test_assembly_emits_static_rounding_only_when_present(self):
        f0 = MachineOperand.reg("ft0", RegisterClass.FPR)
        f1 = MachineOperand.reg("ft1", RegisterClass.FPR)
        f2 = MachineOperand.reg("ft2", RegisterClass.FPR)
        assembly = AsmEmitter(
            [
                MachineInstr(MachineOp.FADD_S, f0, f1, f2, rounding_mode="rne"),
                MachineInstr(MachineOp.FMIN_S, f0, f1, f2),
            ]
        ).emit()
        assert "fadd.s ft0, ft1, ft2, rne" in assembly
        assert "fmin.s ft0, ft1, ft2\n" in assembly

    def test_legacy_li_d_cannot_reach_final_assembly(self):
        with pytest.raises(ValueError, match="not a standard RISC-V instruction"):
            AsmEmitter(
                [
                    MachineInstr(
                        MachineOp.LI_D,
                        MachineOperand.reg("f0", RegisterClass.FPR),
                        MachineOperand.immediate(1),
                    )
                ]
            ).emit()
