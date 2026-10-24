"""Calling-convention and source-bit contracts.

Covers returns across real calls, raw ONNX bits, peephole symbol scoping and
selector roles. Product objects are linked from an independent caller.
"""

import struct

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from scratchv.analysis.ir_verifier import verify_ir
from scratchv.backend.abi_frame import apply_abi_frames
from scratchv.backend.asm_emit import AsmEmitter
from scratchv.backend.inst_select_ext import ExtendedInstructionSelector
from scratchv.backend.machine_types import MachineInstr as MI, MachineOp as O
from scratchv.backend.machine_types import MachineOperand as P, RegisterClass as RC
from scratchv.backend.regalloc_linear import LinearScanAllocator
from scratchv.backend.register_alloc import RegisterAllocator
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType as T, Value
from scratchv.optimizer.peephole import IRPeepholeOptimizer
from tests.topic28_runtime import (
    _required_tool,
    caller_start,
    check_caller,
    execute,
    observe,
    write_exit,
)
from tests.topic28_onnx import save_graph as save_model
from tests.topic28_support import compile_ir


MODES = ("naive", "greedy", "linear")
TARGETS = (
    (T.FLOAT32, "rv32imf", "ilp32f"),
    (T.FLOAT32, "rv32imfd", "ilp32d"),
    (T.FLOAT64, "rv32imfd", "ilp32d"),
)


def config(mode, level, verify, *, extended=True, isa=None, abi=None, **kw):
    return CompilerConfig(
        reg_alloc=mode,
        optimize_level=level,
        verify_ir=verify,
        use_extended_isel=extended,
        target_isa=isa or ("rv32imfd" if extended else "rv32im"),
        target_abi=abi or ("ilp32d" if extended else "ilp32"),
        **kw,
    )


def float_bits(value, dtype):
    return int.from_bytes(
        struct.pack("<d" if dtype is T.FLOAT64 else "<f", value), "little"
    )


class TestCallingConventionAndSourceBits:
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize(
        "dtype,extended",
        [(T.INT32, False), (T.INT32, True), (T.FLOAT32, True), (T.FLOAT64, True)],
    )
    @pytest.mark.parametrize("path", [0, 1])
    def test_product_returns_restore_caller_state(
        self, mode, dtype, extended, path, tmp_path
    ):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        initial = 7 + path if dtype is T.INT32 else 3.25 + path
        p = b.alloca(4 if not extended else 1, dtype)
        b.store(p, b.load_const(initial, dtype))
        x = b.load(p, dtype)
        count = 22 if dtype is T.INT32 else 16
        values = [
            b.add(x, b.load_const(i if dtype is T.INT32 else float(i), dtype))
            for i in range(1, count + 1)
        ]
        total = values[0]
        for value in values[1:]:
            total = b.add(total, value)
        if dtype is T.INT32:
            b.br_if(b.sub(x, b.load_const(7, dtype)), "negative", "positive")
        else:
            b.br_compare(x, "==", b.load_const(3.25, dtype), "positive", "negative")
        b.new_block("positive")
        b.ret(total)
        b.new_block("negative")
        b.ret(b.neg(total))
        assembly = compile_ir(
            b.program, tmp_path, config(mode, "all", True, extended=extended)
        )
        expected = (count * initial + count * (count + 1) // 2) * (-1 if path else 1)
        wanted = expected if dtype is T.INT32 else float_bits(expected, dtype)
        assert observe(
            assembly,
            tmp_path,
            dtype,
            isa="rv32imfd" if extended else "rv32im",
            abi="ilp32d" if extended else "ilp32",
        ) == (wanted, 0)

    @pytest.mark.parametrize("mode", MODES)
    def test_real_nested_call_preserves_mixed_live_values(self, mode, tmp_path):
        fp = lambda name, width: P.vreg(name, RC.FPR, width)
        ins = [
            MI(O.LABEL, target="main"),
            MI(O.LA, P.vreg("a"), P.symbol(".data_s")),
            MI(O.FLW, fp("s", 32), P.vreg("a")),
            MI(O.LA, P.vreg("b"), P.symbol(".data_d")),
            MI(O.FLD, fp("d", 64), P.vreg("b")),
            MI(O.LI, P.vreg("i"), P.immediate(7)),
            MI(O.CALL, target="helper"),
            MI(O.J, target=".next"),
            MI(O.LABEL, target=".next"),
            MI(O.FADD_S, fp("ss", 32), fp("s", 32), fp("s", 32), rounding_mode="rne"),
            MI(O.FADD_D, fp("dd", 64), fp("d", 64), fp("d", 64), rounding_mode="rne"),
            MI(O.MV, P.reg("a0"), P.vreg("i")),
            MI(O.FMV_S, P.reg("fa1", RC.FPR, 32), fp("ss", 32)),
            MI(O.FMV_D, P.reg("fa0", RC.FPR, 64), fp("dd", 64)),
            MI(O.JALR, P.reg("zero"), P.reg("ra"), comment="return"),
        ]
        allocator = (
            LinearScanAllocator() if mode == "linear" else RegisterAllocator(ins, mode)
        )
        allocation = (
            allocator.allocate_result(ins)
            if mode == "linear"
            else allocator.allocate_result()
        )
        call = next(
            index
            for index, instr in enumerate(allocation.instructions)
            if instr.op is O.CALL
        )
        assert {O.SW, O.FSW, O.FSD} <= {
            instr.op for instr in allocation.instructions[:call]
        }
        assert any(
            instr.op is O.FLD and "[regalloc:reload]" in instr.comment
            for instr in allocation.instructions[call + 1 :]
        )
        product = apply_abi_frames(
            AsmEmitter(allocation.instructions).emit(), allocation
        )
        helper_asm = (
            ".text\n.globl helper\nhelper:\n"
            + "".join(f"li t{i},-99\n" for i in range(7))
            + "".join(f"fmv.s.x ft{i},zero\n" for i in range(12))
            + "ret\n"
        )
        harness = (
            ".section .rodata\n.p2align 3\n.data_s:\n.word 0x40600000\n"
            ".p2align 3\n.data_d:\n.quad 0x4011000000000000\n"
            + helper_asm
            + caller_start()
            + "call main\n"
            + check_caller()
            + "sw a0,0(sp)\nfsw fa1,4(sp)\nfsd fa0,8(sp)\n"
            + write_exit(16)
        )
        # This observes backend live values, not a supported multi-return IR ABI.
        assert execute(product, harness, tmp_path) == struct.pack("<Ifd", 7, 7.0, 8.5)

    @pytest.mark.parametrize("extended", [False, True])
    @pytest.mark.parametrize("case", ["step2", "large", "zero", "nested", "overflow"])
    def test_positive_steps_execute_as_declared(self, extended, case, tmp_path):
        start, end, step = {
            "step2": (0, 5, 2),
            "large": (0, 7000, 3000),
            "zero": (2, 2, 3),
            "nested": (0, 5, 2),
            "overflow": (2147483646, 2147483647, 2),
        }[case]
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        p = b.alloca(1 if extended else 4, T.INT32)
        b.store(p, b.load_const(0, T.INT32))
        iv = b.for_loop(start, end, step)
        if case == "nested":
            jv = b.for_loop(1, 6, 2)
            term = b.add(iv, jv)
            expected = sum(
                i + j for i in range(start, end, step) for j in range(1, 6, 2)
            )
        else:
            term = b.load_const(1, T.INT32) if case == "overflow" else iv
            expected = (
                len(range(start, end, step))
                if case == "overflow"
                else sum(range(start, end, step))
            )
        b.store(p, b.add(b.load(p, T.INT32), term))
        if case == "nested":
            b.endfor()
        b.endfor()
        b.ret(b.load(p, T.INT32))
        assembly = compile_ir(
            b.program, tmp_path, config("greedy", "all", True, extended=extended)
        )
        assert observe(assembly, tmp_path, isa="rv32im", abi="ilp32") == (expected, 0)

    @pytest.mark.parametrize("dtype,isa,abi", TARGETS)
    @pytest.mark.parametrize(
        "pattern",
        ["snan", "negative_snan", "qnan", "negative_zero", "infinity", "subnormal"],
    )
    def test_onnx_returns_original_bits(
        self, dtype, isa, abi, pattern, tmp_path
    ):
        # Source bits are preserved before allocation; allocator behavior is
        # covered by the ABI/caller-state tests.
        raw = (
            {
                "snan": 0x7F800001,
                "negative_snan": 0xFF800123,
                "qnan": 0x7FC00123,
                "negative_zero": 0x80000000,
                "infinity": 0x7F800000,
                "subnormal": 1,
            }
            if dtype is T.FLOAT32
            else {
                "snan": 0x7FF0000000000001,
                "negative_snan": 0xFFF0000000000123,
                "qnan": 0x7FF8000000000123,
                "negative_zero": 0x8000000000000000,
                "infinity": 0x7FF0000000000000,
                "subnormal": 1,
            }
        )[pattern]
        array = np.asarray(
            raw, dtype=np.uint32 if dtype is T.FLOAT32 else np.uint64
        ).view(np.float32 if dtype is T.FLOAT32 else np.float64)
        path, model = save_model(
            helper.make_graph(
                [],
                "main",
                [],
                [
                    helper.make_tensor_value_info(
                        "a",
                        TensorProto.FLOAT if dtype is T.FLOAT32 else TensorProto.DOUBLE,
                        [],
                    )
                ],
                [numpy_helper.from_array(array, name="a")],
            ),
            tmp_path,
        )
        assert ReferenceEvaluator(model).run(None, {})[0].tobytes() == array.tobytes()
        result = CompilerDriver(config("greedy", "all", True, isa=isa, abi=abi)).compile(
            str(path), str(tmp_path / "out.s")
        )
        assert result.success, result.errors
        assert observe(result.output_text, tmp_path, dtype, isa=isa, abi=abi) == (
            raw,
            0,
        )

    @pytest.mark.parametrize("dtype,isa,abi", TARGETS)
    @pytest.mark.parametrize("operation", ["Min", "Max"])
    def test_onnx_snan_arithmetic_sets_invalid(
        self, dtype, isa, abi, operation, tmp_path
    ):
        raw = 0x7F800001 if dtype is T.FLOAT32 else 0x7FF0000000000001
        array = np.asarray(
            raw, dtype=np.uint32 if dtype is T.FLOAT32 else np.uint64
        ).view(np.float32 if dtype is T.FLOAT32 else np.float64)
        nt = TensorProto.FLOAT if dtype is T.FLOAT32 else TensorProto.DOUBLE
        path, model = save_model(
            helper.make_graph(
                [helper.make_node(operation, ["a", "b"], ["out"])],
                "main",
                [],
                [helper.make_tensor_value_info("out", nt, [])],
                [
                    numpy_helper.from_array(array, name="a"),
                    numpy_helper.from_array(
                        np.asarray(2.0, dtype=array.dtype), name="b"
                    ),
                ],
            ),
            tmp_path,
        )
        with np.errstate(invalid="ignore"):
            assert np.isnan(ReferenceEvaluator(model).run(None, {})[0]).all()
        result = CompilerDriver(
            config("greedy", "all", True, isa=isa, abi=abi)
        ).compile(str(path), str(tmp_path / "out.s"))
        assert result.success, result.errors
        canonical_nan = 0x7FC00000 if dtype is T.FLOAT32 else 0x7FF8000000000000
        assert observe(result.output_text, tmp_path, dtype, isa=isa, abi=abi) == (
            canonical_nan,
            0x10,
        )

    @pytest.mark.parametrize(
        "name", ["_zero", "__scratchv_peephole_zero_1", "__scratchv_peephole_zero_2"]
    )
    def test_peephole_names_do_not_alias_globals(self, name, tmp_path):
        b = IRBuilder()
        b.program.global_values.append(Value(name, T.FLOAT32, True, 3.0))
        x = b.make_value("x", T.INT32)
        b.new_function("main", [x])
        b.new_block("entry")
        b.ret(b.mul(x, b.load_const(1, T.INT32)))
        assembly = compile_ir(b.program, tmp_path, config("greedy", "all", True))
        assert observe(assembly, tmp_path, argument=7) == (7, 0)

    def test_repeated_peephole_reserves_future_definitions(self):
        b = IRBuilder()
        x = b.make_value("x", T.INT32)
        b.new_function("main", [x])
        b.new_block("entry")
        product = b.mul(x, b.load_const(1, T.INT32))
        future = b.load_const(4, T.INT32)
        future.name = "__scratchv_peephole_zero_1"
        b.ret(b.add(product, future))
        opt = IRPeepholeOptimizer()
        opt.optimize(b.program)
        assert verify_ir(b.program) == (True, [])
        opt.optimize(b.program)
        assert verify_ir(b.program) == (True, [])

    @pytest.mark.parametrize("verify", [False, True])
    @pytest.mark.parametrize(
        "name",
        [
            "graph with space",
            "2model",
            "graph\n.text",
            ".",
            "模型",
            ".LC_f32_0",
            "$model",
        ],
    )
    def test_unsafe_graph_symbols_rejected_before_output(
        self, verify, name, tmp_path
    ):
        # Function names are checked by selection, before register allocation.
        path, _ = save_model(
            helper.make_graph(
                [],
                name,
                [],
                [helper.make_tensor_value_info("a", TensorProto.INT32, [])],
                [numpy_helper.from_array(np.asarray(5, dtype=np.int32), name="a")],
            ),
            tmp_path,
        )
        output = tmp_path / "out.s"
        output.write_text("previous output")
        result = CompilerDriver(config("greedy", "all", verify)).compile(
            str(path), str(output)
        )
        assert not result.success
        assert any("function symbol" in error for error in result.errors), result.errors
        assert output.read_text() == "previous output"

    @pytest.mark.parametrize("dtype", [T.INT32, T.FLOAT32, T.FLOAT64])
    @pytest.mark.parametrize(
        "name", [".local_model", "ordinary_model", "_private_model"]
    )
    def test_function_exports_link_from_separate_caller(
        self, dtype, name, tmp_path
    ):
        b = IRBuilder()
        b.new_function(name)
        b.new_block("entry")
        p = b.alloca(1, dtype)
        val = 5 if dtype is T.INT32 else 3.25
        b.store(p, b.load_const(val, dtype))
        b.ret(b.load(p, dtype))
        assembly = compile_ir(b.program, tmp_path, config("greedy", "all", True))
        expected = val if dtype is T.INT32 else float_bits(val, dtype)
        assert observe(assembly, tmp_path, dtype, function=name, separate=True) == (
            expected,
            0,
        )

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("dtype", [T.INT32, T.FLOAT32, T.FLOAT64])
    def test_direct_selector_keeps_two_function_roles(self, mode, dtype, tmp_path):
        b = IRBuilder()
        for name, value in [(".first_model", 5), (".second_model", 9)]:
            b.new_function(name)
            b.new_block("entry")
            b.ret(b.load_const(value if dtype is T.INT32 else float(value), dtype))
        assert verify_ir(b.program) == (True, [])
        selector = ExtendedInstructionSelector(b.program)
        selected = selector.run()
        allocator = (
            LinearScanAllocator()
            if mode == "linear"
            else RegisterAllocator(selected, mode)
        )
        allocation = (
            allocator.allocate_result(selected)
            if mode == "linear"
            else allocator.allocate_result()
        )
        assembly = apply_abi_frames(
            AsmEmitter(allocation.instructions, selector.constant_pool).emit(),
            allocation,
        )
        for name, value in [(".first_model", 5), (".second_model", 9)]:
            path = tmp_path / name
            path.mkdir()
            wanted = value if dtype is T.INT32 else float_bits(value, dtype)
            assert observe(assembly, path, dtype, function=name, separate=True) == (
                wanted,
                0,
            )

    @pytest.mark.parametrize(
        "dtype,raw",
        [
            (T.INT32, 1),
            (T.FLOAT32, -1),
            (T.FLOAT32, 1 << 32),
            (T.FLOAT64, 1 << 64),
            (T.FLOAT32, True),
        ],
    )
    @pytest.mark.parametrize("location", ["unused_global", "load_const"])
    def test_invalid_bit_metadata_rejected_even_when_unused(
        self, dtype, raw, location, tmp_path
    ):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        if location == "unused_global":
            b.program.global_values.append(
                Value("unused", dtype, True, 1, bit_pattern=raw)
            )
        else:
            value = b.load_const(1, dtype)
            value.bit_pattern = raw
            b.current_block.instructions[-1].attrs["bit_pattern"] = raw
        b.ret(b.load_const(7, T.INT32))
        assert not verify_ir(b.program)[0]
        with pytest.raises(ValueError, match="bit_pattern"):
            ExtendedInstructionSelector(b.program).run()
        driver = CompilerDriver(config("greedy", "all", False))
        driver._parse = lambda *args: b.program
        result = driver.compile("", str(tmp_path / "out.s"), dsl_source="")
        assert not result.success and any("bit_pattern" in e for e in result.errors)

    def test_global_and_load_metadata_survive_alias_and_selector_reuse(
        self, tmp_path
    ):
        raw = 0x7F800123
        array = np.asarray(raw, dtype=np.uint32).view(np.float32)
        path, _ = save_model(
            helper.make_graph(
                [helper.make_node("Min", ["a"], ["out"])],
                "main",
                [],
                [helper.make_tensor_value_info("out", TensorProto.FLOAT, [])],
                [numpy_helper.from_array(array, name="a")],
            ),
            tmp_path,
        )
        program = ONNXParser().parse(str(path))
        assert verify_ir(program) == (True, [])
        assert program.global_values[0].bit_pattern == raw
        selector = ExtendedInstructionSelector(program)
        first = selector.run()
        pool = selector.constant_pool
        assert first == selector.run() and pool == selector.constant_pool
        assert any(entry.bit_pattern == raw for entry in pool)
        result = CompilerDriver(config("greedy", "all", True)).compile(
            str(path), str(tmp_path / "out.s")
        )
        assert result.success, result.errors
        assert observe(result.output_text, tmp_path, T.FLOAT32) == (raw, 0)

    def test_conflicting_load_bit_metadata_is_diagnosed(self, tmp_path):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        value = b.load_const(1.0, T.FLOAT32, bit_pattern=0x3F800000)
        b.current_block.instructions[-1].attrs["bit_pattern"] = 0x40000000
        b.ret(value)
        assert not verify_ir(b.program)[0]
        with pytest.raises(ValueError, match="differs from result metadata"):
            ExtendedInstructionSelector(b.program).run()

    @pytest.mark.parametrize(
        "dtype,raw",
        [
            (T.INT32, 1),
            (T.FLOAT32, -1),
            (T.FLOAT32, 1 << 32),
            (T.FLOAT64, 1 << 64),
            (T.FLOAT32, True),
        ],
    )
    def test_builder_rejects_invalid_explicit_encoding(self, dtype, raw):
        with pytest.raises(ValueError, match="bit_pattern"):
            IRBuilder().load_const(1, dtype, bit_pattern=raw)

    def test_unsupported_call_symbol_is_rejected(self):
        ins = [
            MI(O.LABEL, target="main", label_role="function"),
            MI(O.CALL, target="$helper"),
            MI(O.JALR, P.reg("zero"), P.reg("ra"), comment="return"),
        ]
        allocation = RegisterAllocator(ins, "greedy").allocate_result()
        with pytest.raises(ValueError, match="call symbol"):
            AsmEmitter(allocation.instructions).emit()
