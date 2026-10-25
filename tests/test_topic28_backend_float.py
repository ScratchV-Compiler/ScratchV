"""Floating-point backend integration: allocation, spills, frames, execution."""

from __future__ import annotations

import re
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scratchv.backend.abi_frame import apply_abi_frames
from scratchv.backend.allocation import collect_register_types, validate_allocated
from scratchv.backend.asm_emit import AsmEmitter
from scratchv.backend.machine_types import (
    MachineInstr as MI,
    MachineOp as O,
    MachineOperand as P,
    RegisterClass as RC,
)
from scratchv.backend.register_alloc import RegisterAllocator
from scratchv.backend.regalloc_linear import LinearScanAllocator
from scratchv.backend.riscv_encoder import RISCVAEncoder
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.simulator.rv32_emulator import RV32Emulator


def allocate(instructions, mode):
    allocator = (
        LinearScanAllocator()
        if mode == "linear"
        else RegisterAllocator(instructions, mode)
    )
    return (
        allocator.allocate_result(instructions)
        if mode == "linear"
        else allocator.allocate_result()
    )


def fp(name, width):
    return P.vreg(name, RC.FPR, width)


def cross_assemble(assembly, tmp_path, width=64):
    clang = shutil.which("clang")
    assert clang, "Topic 28 acceptance requires clang with the RISC-V assembler"
    source = tmp_path / "out.s"
    source.write_text(assembly)
    result = subprocess.run(
        [
            clang,
            "--target=riscv32",
            f"-march=rv32im{'f' if width == 32 else 'fd'}",
            f"-mabi=ilp32{'f' if width == 32 else 'd'}",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "out.o"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + assembly


def execute_allocated(instructions, memory):
    registers = {"zero": 0, "x0": 0, "sp": 4096}
    memory = dict(memory)

    def read(operand):
        return operand.value if operand.kind == "imm" else registers[str(operand.value)]

    def address(operand):
        match = re.fullmatch(r"(-?\d+)\((\w+)\)", str(operand.value))
        return registers[match[2]] + int(match[1]) if match else read(operand)

    for instr in instructions:
        if instr.op is O.LI:
            registers[str(instr.dst.value)] = read(instr.src1)
        elif instr.op in {O.FLW, O.FLD, O.LW}:
            registers[str(instr.dst.value)] = memory[address(instr.src1)]
        elif instr.op in {O.FSW, O.FSD, O.SW}:
            memory[address(instr.src1)] = read(instr.dst)
        elif instr.op in {O.FADD_S, O.FADD_D, O.ADD}:
            registers[str(instr.dst.value)] = read(instr.src1) + read(instr.src2)
        elif instr.op in {O.FMV_S, O.FMV_D, O.MV}:
            registers[str(instr.dst.value)] = read(instr.src1)
    return registers.get("fa0", registers.get("a0"))


class TestFloatingPointBackendIntegration:
    @pytest.mark.parametrize("mode", ["naive", "greedy", "linear"])
    @pytest.mark.parametrize("width", [32, 64])
    @pytest.mark.parametrize("count", [12, 16])
    def test_pressure_spills_use_correct_width_and_preserve_values(
        self, mode, width, count, tmp_path
    ):
        load = O.FLW if width == 32 else O.FLD
        add = O.FADD_S if width == 32 else O.FADD_D
        instructions = [MI(O.LABEL, target="main")]
        for index in range(count):
            instructions.extend(
                [
                    MI(O.LI, P.vreg(f"address{index}"), P.immediate(256 + index * 8)),
                    MI(load, fp(f"v{index}", width), P.vreg(f"address{index}")),
                ]
            )
        for index in range(1, count):
            instructions.append(
                MI(
                    add,
                    fp("v0", width),
                    fp("v0", width),
                    fp(f"v{index}", width),
                    rounding_mode="rne",
                )
            )
        instructions.extend(
            [
                MI(
                    O.FMV_S if width == 32 else O.FMV_D,
                    P.reg("fa0", RC.FPR, width),
                    fp("v0", width),
                ),
                MI(O.JALR, P.reg("zero"), P.reg("ra")),
            ]
        )
        allocation = allocate(instructions, mode)
        validate_allocated(allocation.instructions)
        fp_slots = [
            slot for slot in allocation.spill_slots.values() if slot.reg_class is RC.FPR
        ]
        assert all(
            slot.size == width // 8 and slot.offset % slot.alignment == 0
            for slot in fp_slots
        )
        if count > 12 or mode == "naive":
            assert fp_slots
            assert any(
                instr.op is (O.FSW if width == 32 else O.FSD)
                for instr in allocation.instructions
            )
            assert any(
                instr.op is load and "[regalloc:reload]" in instr.comment
                for instr in allocation.instructions
            )
        assembly = apply_abi_frames(
            AsmEmitter(allocation.instructions).emit(), allocation
        )
        assert all(
            int(size) % 16 == 0 for size in re.findall(r"addi sp, sp, -(\d+)", assembly)
        )
        cross_assemble(assembly, tmp_path, width)
        # Execute the allocated stream with distinct memory values. This checks
        # ownership/reloads, not just the shape of instructions in the output.
        assert execute_allocated(
            allocation.instructions, {256 + i * 8: float(i + 1) for i in range(count)}
        ) == sum(range(1, count + 1))

    def test_metadata_missing_width_conflict_and_wrong_physical_class_rejected(self):
        with pytest.raises(ValueError, match="missing bit width"):
            collect_register_types(
                [MI(O.FNEG_S, P.vreg("out", RC.FPR), P.vreg("x", RC.FPR))]
            )
        with pytest.raises(ValueError, match="conflicting width"):
            collect_register_types(
                [
                    MI(O.FNEG_S, fp("x", 32), fp("a", 32)),
                    MI(O.FNEG_D, fp("x", 64), fp("b", 64)),
                ]
            )
        with pytest.raises(ValueError, match="wrong register class"):
            validate_allocated([MI(O.FMV_S, P.reg("t0", RC.FPR), P.reg("ft0", RC.FPR))])

    @pytest.mark.parametrize(
        "mnemonic,extension",
        [
            ("fadd.s", "F"),
            ("fadd.d", "D"),
            ("flw", "F"),
            ("fld", "D"),
            ("fsw", "F"),
            ("fsd", "D"),
            ("fmv.d", "D"),
            ("fsqrt.s", "F"),
            ("fcvt.s.d", "D"),
            ("fcvt.d.s", "D"),
        ],
    )
    def test_builtin_encoder_rejects_fp_explicitly(self, mnemonic, extension):
        with pytest.raises(ValueError, match=f"RISC-V {extension} extension"):
            RISCVAEncoder().assemble(f"{mnemonic} ft0, ft1, ft2")

    def test_non_fp_fence_is_not_misreported_as_float_extension(self):
        with pytest.raises(ValueError, match="Unknown instruction: fence"):
            RISCVAEncoder().assemble("fence rw,rw")

    def test_builtin_emulator_rejects_floating_binary(self):
        emulator = RV32Emulator(mem_size=4096)
        emulator.load_code((0x53).to_bytes(4, "little"))
        with pytest.raises(ValueError, match="F/D extension"):
            emulator.run(max_instr=1)

    def test_function_sizes_use_previous_function_even_with_local_labels(self):
        assembly = AsmEmitter(
            [
                MI(O.LABEL, target="first"),
                MI(O.LABEL, target=".exit"),
                MI(O.JALR, P.reg("zero"), P.reg("ra")),
                MI(O.LABEL, target="second"),
                MI(O.JALR, P.reg("zero"), P.reg("ra")),
            ]
        ).emit()
        assert assembly.count(".size first, .-first") == 1
        assert assembly.count(".size second, .-second") == 1

    @pytest.mark.parametrize("dtype", ["f32", "f64"])
    @pytest.mark.parametrize(
        "case,expression,expected,flags",
        [
            ("normal", "sqrt(a)", 0x40000000, 0),
            ("signed_zero", "min(negative_zero,positive_zero)", 0x80000000, 0),
            ("nan_number", "max(not_a_number,a)", 0x40800000, 0),
            ("infinity", "div(a,positive_zero)", 0x7F800000, 8),
            ("subnormal", "min(subnormal,a)", 1, 0),
            ("rounding", "add(one,half_ulp)", 0x3F800000, 1),
        ],
    )
    def test_qemu_fp_binary_values_and_exception_flags(
        self, dtype, case, expression, expected, flags, tmp_path
    ):
        qemu = shutil.which("qemu-riscv32")
        linker = shutil.which("ld.lld")
        if os.environ.get("SCRATCHV_REQUIRE_RISCV_EXECUTION") == "1" and (
            not qemu or not linker
        ):
            pytest.fail("required FP execution tools missing: qemu-riscv32 / ld.lld")
        if not linker:
            pytest.skip(
                f"FP binary execution requires qemu-riscv32 and ld.lld; found qemu={qemu}, linker={linker}"
            )
        is_double = dtype == "f64"
        subnormal = "4.9406564584124654e-324" if is_double else "1.401298464324817e-45"
        half_ulp = "1.1102230246251565e-16" if is_double else "5.960464477539063e-8"
        source = (
            f"a = const(4.0,dtype:{dtype})\nnegative_zero = const(-0.0,dtype:{dtype})\n"
            f"positive_zero = const(0.0,dtype:{dtype})\nnot_a_number = const(nan,dtype:{dtype})\n"
            f"subnormal = const({subnormal},dtype:{dtype})\none = const(1.0,dtype:{dtype})\n"
            f"half_ulp = const({half_ulp},dtype:{dtype})\nr = {expression}\nreturn r"
        )
        result = CompilerDriver(
            CompilerConfig(
                use_extended_isel=True,
                reg_alloc="greedy",
                target_isa="rv32imfd",
                target_abi="ilp32d",
            )
        ).compile("", str(tmp_path / "binary.s"), dsl_source=source)
        assert result.success, result.errors
        if is_double:
            expected = {
                "normal": 0x4000000000000000,
                "signed_zero": 0x8000000000000000,
                "nan_number": 0x4010000000000000,
                "infinity": 0x7FF0000000000000,
                "subnormal": 1,
                "rounding": 0x3FF0000000000000,
            }[case]
        bits_check = (
            f"fsd fa0,0(sp)\nlw t0,4(sp)\nli t1,{(expected >> 32) & 0xFFFFFFFF}\n"
            "bne t0,t1,.fail\nlw t0,0(sp)"
            if is_double
            else "fmv.x.w t0,fa0"
        )
        harness = (
            ".text\n.globl _start\n_start:\naddi sp,sp,-16\ncsrw fflags,zero\ncall main\n"
            f"{bits_check}\nli t1,{expected & 0xFFFFFFFF}\nbne t0,t1,.fail\n"
            f"csrr t0,fflags\nli t1,{flags}\nbne t0,t1,.fail\nli a0,0\nj .exit\n"
            ".fail:\nli a0,1\n.exit:\nli a7,93\necall\n"
        )
        assembly = tmp_path / "execution.s"
        assembly.write_text(result.output_text + harness)
        binary = tmp_path / "execution"
        linked = subprocess.run(
            [
                shutil.which("clang"),
                "--target=riscv32",
                "-march=rv32imfd",
                "-mabi=ilp32d",
                "-nostdlib",
                "-static",
                "-fuse-ld=lld",
                "-Wl,-e,_start",
                str(assembly),
                "-o",
                str(binary),
            ],
            capture_output=True,
            text=True,
        )
        assert linked.returncode == 0, linked.stderr
        if not qemu:
            pytest.skip(
                "FP executable linked successfully; execution requires qemu-riscv32 (not found)"
            )
        execution = subprocess.run([qemu, str(binary)], capture_output=True, timeout=10)
        assert execution.returncode == 0, f"{dtype}/{case}: {execution.stderr}"

    @pytest.mark.parametrize("dtype", ["FLOAT", "DOUBLE"])
    def test_onnx_scalar_parameter_and_initializer_reach_assembly(
        self, dtype, tmp_path
    ):
        onnx = pytest.importorskip("onnx")
        tensor_type = getattr(onnx.TensorProto, dtype)
        graph = onnx.helper.make_graph(
            [
                onnx.helper.make_node("Add", ["x", "c"], ["sum"]),
                onnx.helper.make_node("Sqrt", ["sum"], ["y"]),
            ],
            "scalar",
            [onnx.helper.make_tensor_value_info("x", tensor_type, [1])],
            [onnx.helper.make_tensor_value_info("y", tensor_type, [1])],
            [onnx.helper.make_tensor("c", tensor_type, [1], [4.0])],
        )
        path = tmp_path / "scalar.onnx"
        onnx.save(onnx.helper.make_model(graph), path)
        result = CompilerDriver(
            CompilerConfig(
                use_extended_isel=True,
                reg_alloc="greedy",
                target_isa="rv32imfd",
                target_abi="ilp32d",
                verify_ir=True,
            )
        ).compile(str(path), str(tmp_path / "scalar.s"))
        assert result.success, result.errors
        suffix = "s" if dtype == "FLOAT" else "d"
        assert (
            f"fadd.{suffix}" in result.output_text
            and f"fsqrt.{suffix}" in result.output_text
        )
        assert "scalar hard-float ABI parameter" in result.output_text
        assert "bit-exact" in result.output_text
        cross_assemble(result.output_text, tmp_path)

    @pytest.mark.parametrize(
        "option",
        ["peephole_asm", "const_merge", "beautify_asm", "count_instr", "schedule"],
    )
    def test_optional_assembly_passes_preserve_fp_and_constant_pool(
        self, option, tmp_path
    ):
        result = CompilerDriver(
            CompilerConfig(
                use_extended_isel=True,
                target_isa="rv32imfd",
                target_abi="ilp32d",
                **{option: True},
            )
        ).compile(
            "",
            str(tmp_path / "pass.s"),
            dsl_source=(
                "a = const(-0.0,dtype:f64)\nb = const(2.0,dtype:f64)\nr = add(a,b)\nreturn r"
            ),
        )
        assert result.success, result.errors
        assert ".quad 0x8000000000000000" in result.output_text
        assert re.search(r"fadd.d\s+ft\d+,\s*ft\d+,\s*ft\d+,\s*rne", result.output_text)
        assert re.search(r"fmv.d\s+fa0", result.output_text)
        cross_assemble(result.output_text, tmp_path)

    def test_mixed_frame_restores_every_return_and_checks_layout(self, tmp_path):
        from scratchv.backend.allocation import (
            AllocationResult,
            RegisterType,
            SpillLayout,
            SpillSlot,
        )

        layout = SpillLayout()
        a = layout.reserve("a", RegisterType(RC.FPR, 32))
        b = layout.reserve("b", RegisterType(RC.FPR, 64))
        assert layout.reserve("b", RegisterType(RC.FPR, 64)) is b
        allocation = AllocationResult([], {}, layout.slots, layout.size)
        assembly = apply_abi_frames(
            f"main:\n  fsw ft0,{a.offset}(sp) # [regalloc:spill]\n"
            f"  fsd ft1,{b.offset}(sp) # [regalloc:spill]\n"
            "  beq a0,zero,.exit\n  ret\n.exit:\n  ret\n",
            allocation,
        )
        assert assembly.count("destroy stack frame") == 2
        cross_assemble(assembly, tmp_path)
        with pytest.raises(ValueError, match="overlapping"):
            apply_abi_frames(
                "main:\nret",
                AllocationResult(
                    [],
                    {},
                    {
                        "a": SpillSlot(-8, 8, 8, RC.FPR),
                        "b": SpillSlot(-4, 4, 4, RC.FPR),
                    },
                    8,
                ),
            )
        with pytest.raises(ValueError, match="no layout entry"):
            apply_abi_frames(
                "main:\nfsw ft0,-12(sp) # [regalloc:spill]\nret", allocation
            )

    @pytest.mark.parametrize("dtype", ["f32", "f64"])
    @pytest.mark.parametrize("predicate", ["==", "!=", "<", ">", "<=", ">="])
    def test_comparison_cfg_and_both_returns_cross_assemble(
        self, dtype, predicate, tmp_path
    ):
        result = CompilerDriver(
            CompilerConfig(
                use_extended_isel=True,
                reg_alloc="greedy",
                target_isa="rv32imfd",
                target_abi="ilp32d",
            )
        ).compile(
            "",
            str(tmp_path / "branch.s"),
            dsl_source=(
                f"a = const(1.0,dtype:{dtype})\nb = const(2.0,dtype:{dtype})\n"
                f"if (a {predicate} b):\n return a\nelse:\n return b\nendif"
            ),
        )
        assert result.success, result.errors
        suffix = "s" if dtype == "f32" else "d"
        assert re.search(
            rf"f(?:eq|lt|le)\.{suffix}\s+[ts]\d+,\s*ft\d+,\s*ft\d+", result.output_text
        )
        assert result.output_text.count(f"fmv.{suffix} fa0") == 2
        cross_assemble(result.output_text, tmp_path)

    @pytest.mark.parametrize("mode", ["naive", "greedy", "linear"])
    @pytest.mark.parametrize("width", [32, 64])
    def test_fp_user_store_keeps_gpr_address_and_fp_value(self, mode, width, tmp_path):
        load, store = (O.FLW, O.FSW) if width == 32 else (O.FLD, O.FSD)
        stream = [
            MI(O.LABEL, target="main"),
            MI(O.LI, P.vreg("address"), P.immediate(256)),
            MI(load, fp("value", width), P.vreg("address")),
            MI(store, fp("value", width), P.vreg("address")),
            MI(O.JALR, P.reg("zero"), P.reg("ra")),
        ]
        allocation = allocate(stream, mode)
        stores = [
            instr
            for instr in allocation.instructions
            if instr.op is store and not instr.comment
        ]
        assert len(stores) == 1
        assert stores[0].dst.reg_class is RC.FPR and stores[0].src1.reg_class is RC.GPR
        cross_assemble(
            apply_abi_frames(AsmEmitter(allocation.instructions).emit(), allocation),
            tmp_path,
        )

    @pytest.mark.parametrize(
        "case,isa,abi",
        [
            ("f32_sum", "rv32imf", "ilp32f"),
            ("f64_dot", "rv32imfd", "ilp32d"),
            ("i32_div_rem", "rv32im", "ilp32"),
        ],
    )
    def test_actual_cli_compiles_examples(self, case, isa, abi, tmp_path):
        path = (
            Path(__file__).resolve().parents[1]
            / "benchmarks"
            / "topic28"
            / f"{case}.dsl"
        )
        output = tmp_path / "cli.s"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scratchv.main",
                str(path),
                "--extended-isel",
                "--reg-alloc",
                "greedy",
                "--target-isa",
                isa,
                "--target-abi",
                abi,
                "-o",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        cross_assemble(output.read_text(), tmp_path, 32 if isa == "rv32imf" else 64)

    def test_experimental_linear_allocator_explicitly_rejects_fpr(self):
        from scratchv.backend.regalloc_linear_v1_5 import block_from_machine_instrs

        with pytest.raises(ValueError, match="GPR-only"):
            block_from_machine_instrs([MI(O.FNEG_S, fp("out", 32), fp("in", 32))])

    @pytest.mark.parametrize("dtype", ["f32", "f64"])
    @pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
    @pytest.mark.parametrize("expression", ["min(a,b)", "max(a,b)", "abs(a)"])
    def test_fp_special_operands_never_take_integer_materialization(
        self, dtype, value, expression, tmp_path
    ):
        result = CompilerDriver(
            CompilerConfig(
                use_extended_isel=True,
                reg_alloc="greedy",
                target_isa="rv32imfd",
                target_abi="ilp32d",
            )
        ).compile(
            "",
            str(tmp_path / "special.s"),
            dsl_source=(
                f"a = const({value},dtype:{dtype})\nb = const(4.0,dtype:{dtype})\nr = {expression}\nreturn r"
            ),
        )
        assert result.success, result.errors
        cross_assemble(result.output_text, tmp_path)

    @pytest.mark.parametrize("width", [32, 64])
    @pytest.mark.parametrize("renamed", [False, True])
    def test_linear_conversion_round_trip_preserves_fp_metadata_and_rounding(
        self, width, renamed
    ):
        from scratchv.backend.regalloc_linear import (
            block_from_machine_instrs,
            machine_instrs_from_block,
        )

        original = MI(
            O.FADD_S if width == 32 else O.FADD_D,
            fp("result", width),
            fp("left", width),
            fp("right", width),
            rounding_mode="rne",
        )
        block = block_from_machine_instrs([original])
        if renamed:
            block[0].operands = ["ft0", "ft1", "ft2"]
        result = machine_instrs_from_block(block)[0]
        assert result.rounding_mode == "rne"
        assert all(
            operand.reg_class is RC.FPR and operand.bit_width == width
            for operand in (result.dst, result.src1, result.src2)
        )
        if renamed:
            validate_allocated([result])
        else:
            assert result == original
