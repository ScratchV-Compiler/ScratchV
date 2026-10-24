"""Pipeline regressions: name collisions, CFG definition order, LICM and loops.

Every contract here is checked against real RV32 assembly/execution rather
than opcode shape.
"""

import struct
import subprocess

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from scratchv.analysis.ir_verifier import verify_ir
from scratchv.backend.inst_select_ext import ExtendedInstructionSelector
from scratchv.backend.machine_types import MachineOp
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType as T, OpCode, Value
from scratchv.optimizer.licm import LICM
from tests.topic28_onnx import save_graph as save_model
from tests.topic28_runtime import _qemu, _required_tool, execute_fp, execute_i32


MODES = ("naive", "greedy", "linear")
LEVELS = ("none", "basic", "all")
TYPES = (T.INT32, T.FLOAT32, T.FLOAT64)
ONNX_TYPES = {
    T.INT32: TensorProto.INT32,
    T.FLOAT32: TensorProto.FLOAT,
    T.FLOAT64: TensorProto.DOUBLE,
}
ARRAY_TYPES = {T.INT32: np.int32, T.FLOAT32: np.float32, T.FLOAT64: np.float64}


def configuration(mode="greedy", level="none", verify=True, *, extended=True, **kwargs):
    return CompilerConfig(
        use_extended_isel=extended,
        reg_alloc=mode,
        optimize_level=level,
        verify_ir=verify,
        target_isa="rv32imfd" if extended else "rv32im",
        target_abi="ilp32d" if extended else "ilp32",
        **kwargs,
    )


def compile_ir(program, tmp_path, config):
    driver = CompilerDriver(config)
    driver._parse = lambda *args: program
    return driver.compile("", str(tmp_path / "out.s"), dsl_source="")


def bits(value, width):
    return int.from_bytes(struct.pack("<f" if width == 32 else "<d", value), "little")


def assert_return(assembly, tmp_path, dtype, expected, argument=None):
    if dtype is T.INT32:
        assert execute_i32(assembly, tmp_path, argument=argument) == expected
    else:
        width = 32 if dtype is T.FLOAT32 else 64
        assert execute_fp(assembly, tmp_path, width, argument=argument) == (
            bits(expected, width),
            0,
        )


def execute_fixed_address(assembly, tmp_path, dtype):
    """Place known data at an explicit address; test LOAD itself, not unmapped memory."""
    width = 64 if dtype is T.FLOAT64 else 32
    payload = 37 if dtype is T.INT32 else bits(3.25, width)
    store = {T.INT32: "sw a0", T.FLOAT32: "fsw fa0", T.FLOAT64: "fsd fa0"}[dtype]
    source = tmp_path / "execution.s"
    source.write_text(
        assembly
        + (
            f'\n.section .review_data,"aw",@progbits\n.p2align 3\n'
            f'.review_payload:\n{ ".quad" if width == 64 else ".word" } {payload}\n'
            ".text\n.globl _start\n_start:\naddi sp,sp,-16\ncsrw fflags,zero\ncall main\n"
            f"{store},0(sp)\n"
            + ("sw zero,4(sp)\n" if width == 32 else "")
            + "csrr t0,fflags\nsw t0,8(sp)\nli a0,1\nmv a1,sp\nli a2,12\nli a7,64\necall\n"
            "li a0,0\nli a7,93\necall\n"
        )
    )
    binary = tmp_path / "execution"
    linked = subprocess.run(
        [
            _required_tool("clang"),
            "--target=riscv32",
            "-march=rv32imfd",
            "-mabi=ilp32d",
            "-nostdlib",
            "-static",
            "-fuse-ld=lld",
            "-Wl,-e,_start",
            "-Wl,--section-start=.review_data=0x20000",
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert linked.returncode == 0, linked.stderr + "\n" + source.read_text()
    executed = subprocess.run([_qemu(), str(binary)], capture_output=True, timeout=10)
    assert executed.returncode == 0, (executed.returncode, executed.stderr)
    assert len(executed.stdout) == 12
    assert int.from_bytes(executed.stdout[: width // 8], "little") == payload
    assert int.from_bytes(executed.stdout[8:12], "little") == 0


PIPELINES = [
    ("none", None),
    ("basic", None),
    ("all", None),
    ("none", ("licm",)),
    ("none", ("licm", "licm")),
    ("all", ("dead-code-elim", "licm", "dead-code-elim", "licm")),
]


class TestPipelineRegressions:
    @pytest.mark.parametrize("dtype", TYPES)
    @pytest.mark.parametrize(
        "collision", ["parameter", "later_parameter", "initializer"]
    )
    def test_onnx_names_cannot_alias_generated_values(
        self, dtype, collision, tmp_path
    ):
        parameter = {"parameter": "v_2", "later_parameter": "v_3"}.get(
            collision, "input"
        )
        constant = "v_2" if collision == "initializer" else "b"
        nodes = [helper.make_node("Mul", [parameter, constant], ["p"])]
        if collision == "later_parameter":
            nodes.extend(
                [
                    helper.make_node("Add", ["p", parameter], ["q"]),
                    helper.make_node("Add", ["q", parameter], ["out"]),
                ]
            )
        else:
            nodes.append(
                helper.make_node(
                    "Add",
                    ["p", constant if collision == "initializer" else parameter],
                    ["out"],
                )
            )
        graph = helper.make_graph(
            nodes,
            "main",
            [helper.make_tensor_value_info(parameter, ONNX_TYPES[dtype], [])],
            [helper.make_tensor_value_info("out", ONNX_TYPES[dtype], [])],
            [
                numpy_helper.from_array(
                    np.asarray(3, dtype=ARRAY_TYPES[dtype]), name=constant
                )
            ],
        )
        path, model = save_model(graph, tmp_path)
        expected = (
            ReferenceEvaluator(model)
            .run(None, {parameter: np.asarray(2, dtype=ARRAY_TYPES[dtype])})[0]
            .item()
        )
        program = ONNXParser().parse(str(path))
        assert verify_ir(program) == (True, [])
        result = CompilerDriver(configuration("greedy", "all", True)).compile(
            str(path), str(tmp_path / "out.s")
        )
        assert result.success, result.errors
        assert_return(result.output_text, tmp_path, dtype, expected, 2)

    def test_builder_reserves_explicit_and_external_operand_names(self):
        b = IRBuilder()
        b.new_function("main", [Value("v_1", T.INT32)])
        b.new_block("entry")
        b.make_value("v_2", T.INT32)
        one = b.load_const(1, T.INT32)
        external = Value("v_4", T.INT32, True, 3)
        result = b.add(one, external)
        b.ret(result)
        assert len({one.name, result.name, "v_1", "v_2", "v_4"}) == 5
        assert verify_ir(b.program) == (True, [])

    @pytest.mark.parametrize("verify", [False, True])
    @pytest.mark.parametrize("shape", [(2,), (1, 2), (0,)])
    def test_integer_initializer_tensor_is_rejected_before_output(
        self, verify, shape, tmp_path
    ):
        initializers = [
            numpy_helper.from_array(np.full(shape, value, dtype=np.int32), name=name)
            for name, value in [("a", 1), ("b", 3)]
        ]
        path, model = save_model(
            helper.make_graph(
                [helper.make_node("Add", ["a", "b"], ["out"])],
                "main",
                [],
                [helper.make_tensor_value_info("out", TensorProto.INT32, shape)],
                initializers,
            ),
            tmp_path,
        )
        np.testing.assert_array_equal(
            ReferenceEvaluator(model).run(None, {})[0],
            np.full(shape, 4, dtype=np.int32),
        )
        output = tmp_path / "out.s"
        output.write_text("previous output")
        result = CompilerDriver(configuration("greedy", "all", verify)).compile(
            str(path), str(output)
        )
        assert not result.success
        assert any(
            "scalar" in error and "shape" in error for error in result.errors
        ), result.errors
        assert output.read_text() == "previous output"
        with pytest.raises(ValueError, match="scalar.*shape"):
            ExtendedInstructionSelector(ONNXParser().parse(str(path))).run()

    @pytest.mark.parametrize("shape", [(), (1,), (1, 1)])
    def test_integer_scalar_shapes_are_not_rejected(self, shape, tmp_path):
        path, model = save_model(
            helper.make_graph(
                [helper.make_node("Add", ["a", "b"], ["out"])],
                "main",
                [],
                [helper.make_tensor_value_info("out", TensorProto.INT32, shape)],
                [
                    numpy_helper.from_array(
                        np.full(shape, value, dtype=np.int32), name=name
                    )
                    for name, value in [("a", 1), ("b", 3)]
                ],
            ),
            tmp_path,
        )
        assert ReferenceEvaluator(model).run(None, {})[0].item() == 4
        result = CompilerDriver(configuration("greedy", "all")).compile(
            str(path), str(tmp_path / "out.s")
        )
        assert result.success, result.errors
        assert execute_i32(result.output_text, tmp_path) == 4

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize(
        "dtype,extended",
        [(T.INT32, False), (T.INT32, True), (T.FLOAT32, True), (T.FLOAT64, True)],
    )
    @pytest.mark.parametrize("source", ["load_const", "inline"])
    def test_constant_address_load_executes(
        self, mode, dtype, extended, source, tmp_path
    ):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        address = (
            b.load_const(0x20000, T.INT32)
            if source == "load_const"
            else b.make_value("address", T.INT32, True, 0x20000)
        )
        b.ret(b.load(address, dtype))
        assert verify_ir(b.program) == (True, [])
        result = compile_ir(
            b.program, tmp_path, configuration(mode, "all", True, extended=extended)
        )
        assert result.success, result.errors
        assert verify_ir(b.program) == (True, [])
        execute_fixed_address(result.output_text, tmp_path, dtype)

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("dtype", [T.FLOAT32, T.FLOAT64])
    @pytest.mark.parametrize("consumer", ["return", "arithmetic", "store", "branch"])
    def test_fp_definition_uses_cfg_not_block_layout(
        self, mode, dtype, consumer, tmp_path
    ):
        b = IRBuilder()
        f = b.new_function("main")
        entry = b.new_block("entry")
        b.br("define")
        define = b.new_block("define")
        value = b.sqrt(b.load_const(16.0, dtype))
        b.br("use")
        use = b.new_block("use")
        if consumer == "arithmetic":
            b.ret(b.add(value, value))
            expected = 8
        elif consumer == "store":
            ptr = b.alloca(1, dtype)
            b.store(ptr, value)
            b.ret(b.load(ptr, dtype))
            expected = 4
        elif consumer == "branch":
            b.br_compare(value, "<", b.load_const(5.0, dtype), "yes", "no")
            yes = b.new_block("yes")
            b.ret(b.load_const(7.0, dtype))
            no = b.new_block("no")
            b.ret(b.load_const(11.0, dtype))
            expected = 7
        else:
            b.ret(value)
            expected = 4
        f.blocks = [entry, use, *([yes, no] if consumer == "branch" else []), define]
        assert verify_ir(b.program) == (True, [])
        result = compile_ir(b.program, tmp_path, configuration(mode, "all", True))
        assert result.success, result.errors
        assert verify_ir(b.program) == (True, [])
        assert_return(result.output_text, tmp_path, dtype, expected)

    @pytest.mark.parametrize("dtype", [T.FLOAT32, T.FLOAT64])
    @pytest.mark.parametrize(
        "case", ["use_before_definition", "not_dominating", "address_as_data"]
    )
    def test_forward_source_inventory_does_not_accept_invalid_definitions(
        self, dtype, case, tmp_path
    ):
        b = IRBuilder()
        f = b.new_function("main")
        b.new_block("entry")
        if case == "address_as_data":
            b.ret(b.sqrt(b.alloca(1, dtype)))
        elif case == "not_dominating":
            b.br_if(b.load_const(1, T.INT32), "define", "use")
            b.new_block("define")
            value = b.sqrt(b.load_const(16.0, dtype))
            b.br("use")
            b.new_block("use")
            b.ret(value)
        else:
            value = b.sqrt(b.load_const(16.0, dtype))
            b.ret(value)
            instructions = f.blocks[0].instructions
            instructions[0], instructions[1] = instructions[1], instructions[0]
        result = compile_ir(b.program, tmp_path, configuration(verify=False))
        assert not result.success
        assert any(
            "definition" in error or "dominate" in error for error in result.errors
        ), result.errors
        with pytest.raises(ValueError):
            ExtendedInstructionSelector(b.program).run()

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize(
        "dtype,extended",
        [(T.INT32, False), (T.INT32, True), (T.FLOAT32, True), (T.FLOAT64, True)],
    )
    def test_layout_forward_definition_is_not_a_function_live_in(
        self, mode, dtype, extended, tmp_path
    ):
        b = IRBuilder()
        f = b.new_function("main")
        entry = b.new_block("entry")
        b.br("define")
        define = b.new_block("define")
        ptr = b.alloca(1, dtype)
        b.store(ptr, b.load_const(7 if dtype is T.INT32 else 4.0, dtype))
        value = b.load(ptr, dtype)
        b.br("use")
        use = b.new_block("use")
        b.ret(value)
        f.blocks = [entry, use, define]
        assert verify_ir(b.program) == (True, [])
        result = compile_ir(
            b.program, tmp_path, configuration(mode, "all", True, extended=extended)
        )
        assert result.success, result.errors
        assert_return(result.output_text, tmp_path, dtype, 7 if dtype is T.INT32 else 4)

    def test_licm_processes_adjacent_loops_and_keeps_memory_reads_inside(self):
        b = IRBuilder()
        parameter = b.make_value("input", T.INT32)
        b.new_function("main", [parameter])
        b.new_block("entry")
        ptr = b.alloca(1, T.INT32)
        b.store(ptr, b.load_const(0, T.INT32))
        two = b.load_const(2, T.INT32)
        for _ in range(2):
            b.for_loop(0, 1)
            product = b.mul(parameter, two)
            b.store(ptr, b.add(b.load(ptr, T.INT32), product))
            b.endfor()
        b.ret(b.load(ptr, T.INT32))
        assert verify_ir(b.program) == (True, [])
        assert LICM(preserve_float_semantics=True).optimize(b.program) == 2
        assert verify_ir(b.program) == (True, [])
        instructions = b.current_block.instructions
        depth = 0
        for instr in instructions:
            if instr.opcode is OpCode.FOR:
                depth += 1
            elif instr.opcode is OpCode.ENDFOR:
                depth -= 1
            elif instr.opcode is OpCode.MUL:
                assert depth == 0
            elif instr.opcode is OpCode.ADD:
                assert depth == 1

    @pytest.mark.parametrize("level", LEVELS)
    def test_licm_multiple_invariants_preserve_loop_and_instruction_order(
        self, level, tmp_path
    ):
        b = IRBuilder()
        parameter = b.make_value("input", T.INT32)
        b.new_function("main", [parameter])
        b.new_block("entry")
        ptr = b.alloca(1, T.INT32)
        b.store(ptr, b.load_const(0, T.INT32))
        two = b.load_const(2, T.INT32)
        three = b.load_const(3, T.INT32)
        b.for_loop(0, 2)
        product = b.mul(parameter, two)
        offset = b.add(parameter, three)
        total = b.add(product, offset)
        b.store(ptr, b.add(b.load(ptr, T.INT32), total))
        b.endfor()
        b.ret(b.load(ptr, T.INT32))
        assert verify_ir(b.program) == (True, [])
        result = compile_ir(b.program, tmp_path, configuration("greedy", level, True))
        assert result.success, result.errors
        assert verify_ir(b.program) == (True, [])
        instructions = b.current_block.instructions
        assert sum(instr.opcode is OpCode.FOR for instr in instructions) == 1
        assert sum(instr.opcode is OpCode.ENDFOR for instr in instructions) == 1
        assert_return(result.output_text, tmp_path, T.INT32, 36, argument=5)

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("dtype", [T.FLOAT32, T.FLOAT64])
    def test_block_and_pool_symbols_and_br_targets_do_not_collide(
        self, mode, dtype, tmp_path
    ):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        target = "LC_" + dtype.value + "_0"
        b.br(target)
        b.new_block(target)
        b.ret(b.load_const(2.0, dtype))
        assert verify_ir(b.program) == (True, [])
        result = compile_ir(b.program, tmp_path, configuration(mode, "all"))
        assert result.success, result.errors
        labels = [
            line.strip()
            for line in result.output_text.splitlines()
            if line.strip().endswith(":")
        ]
        assert len(labels) == len(set(labels))
        assert_return(result.output_text, tmp_path, dtype, 2)
        selector = ExtendedInstructionSelector(b.program)
        first = selector.run()
        first_pool = selector.constant_pool
        second = selector.run()
        assert [i.target for i in first if i.op is MachineOp.LABEL] == [
            i.target for i in second if i.op is MachineOp.LABEL
        ]
        assert first_pool == selector.constant_pool

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("extended", [False, True])
    @pytest.mark.parametrize("case", ["zero", "sum", "nested", "label_collision"])
    def test_for_loops_cross_assemble_and_execute(
        self, mode, extended, case, tmp_path
    ):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("Lloop_header_1" if case == "label_collision" else "entry")
        acc = b.alloca(1, T.INT32)
        b.store(acc, b.load_const(0, T.INT32))
        iv = b.for_loop(0, 0 if case == "zero" else (2 if case == "nested" else 3))
        if case == "nested":
            jv = b.for_loop(0, 2)
            term = b.add(iv, jv)
        else:
            term = iv
        b.store(acc, b.add(b.load(acc, T.INT32), term))
        if case == "nested":
            b.endfor()
        b.endfor()
        b.ret(b.load(acc, T.INT32))
        assert verify_ir(b.program) == (True, [])
        result = compile_ir(
            b.program, tmp_path, configuration(mode, "all", True, extended=extended)
        )
        assert result.success, result.errors
        assert verify_ir(b.program) == (True, [])
        assert execute_i32(result.output_text, tmp_path) == (
            0 if case == "zero" else 4 if case == "nested" else 3
        )

    @pytest.mark.parametrize("pipeline", PIPELINES)
    @pytest.mark.parametrize("dtype", [T.FLOAT32, T.FLOAT64])
    @pytest.mark.parametrize("count", [0, 2])
    @pytest.mark.parametrize("operation", ["sqrt", "divide", "snan_min"])
    def test_licm_keeps_fp_exceptions_inside_loop(
        self, pipeline, dtype, count, operation, tmp_path
    ):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        a = b.load_const(-1.0 if operation == "sqrt" else 1.0, dtype)
        if operation == "snan_min":
            b.current_block.instructions[-1].attrs["bit_pattern"] = (
                0x7F800001 if dtype is T.FLOAT32 else 0x7FF0000000000001
            )
        rhs = b.load_const(0.0 if operation == "divide" else 2.0, dtype)
        b.for_loop(0, count)
        if operation == "sqrt":
            b.sqrt(a)
        elif operation == "divide":
            b.div(a, rhs)
        else:
            b.min(a, rhs)
        b.endfor()
        b.ret(b.load_const(4.0, dtype))
        assert verify_ir(b.program) == (True, [])
        level, passes = pipeline
        result = compile_ir(
            b.program, tmp_path, configuration("greedy", level, True, passes=passes)
        )
        assert result.success, result.errors
        assert verify_ir(b.program) == (True, [])
        width = 32 if dtype is T.FLOAT32 else 64
        assert execute_fp(result.output_text, tmp_path, width) == (
            bits(4.0, width),
            0 if count == 0 else 8 if operation == "divide" else 16,
        )
