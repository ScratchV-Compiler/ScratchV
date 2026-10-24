"""ONNX wire encoding, symbol/ELF metadata and per-function ABI frames."""

import re
import struct
import subprocess

import onnx
import numpy as np
import pytest
from onnx import TensorProto as TP, helper
from onnx.reference import ReferenceEvaluator

from scratchv.analysis.ir_verifier import verify_ir
from scratchv.backend.abi_frame import apply_abi_frames
from scratchv.backend.asm_emit import AsmEmitter
from scratchv.backend.machine_types import (
    FloatConstant,
    MachineInstr as MI,
    MachineOp as O,
    MachineOperand as P,
    RegisterClass as RC,
)
from scratchv.compiler import CompilerDriver
from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType as T
from tests.topic28_runtime import observe
from tests.topic28_onnx import length_field
from tests.topic28_support import MODES, TARGETS, allocate, compile_ir, configuration


def typed_model(path, dtype, nt, raw, packed, operation, unknown_group_at=None):
    width = 4 if dtype is T.FLOAT32 else 8
    tensor = TP(name="a", data_type=nt).SerializeToString()
    field, wire = (4, 5) if width == 4 else (10, 1)
    tensor += (
        length_field(field, raw.to_bytes(width, "little"))
        if packed
        else bytes([field * 8 + wire]) + raw.to_bytes(width, "little")
    )
    # Unknown field 100 contains a nested group 101 and a varint value.
    group = b"\xa3\x06\xab\x06\x08\x01\xac\x06\xa4\x06"
    if unknown_group_at == "tensor":
        tensor += group
    output = "a" if operation == "return" else "out"
    nodes = (
        [] if operation == "return" else [helper.make_node("Min", ["a", "b"], [output])]
    )
    graph = helper.make_graph(
        nodes, ".wire_model", [], [helper.make_tensor_value_info(output, nt, [])]
    ).SerializeToString()
    graph += length_field(5, tensor)
    if operation != "return":
        two = 0x40000000 if width == 4 else 0x4000000000000000
        graph += length_field(
            5,
            TP(
                name="b", data_type=nt, raw_data=two.to_bytes(width, "little")
            ).SerializeToString(),
        )
    if unknown_group_at == "graph":
        graph += group
    model = helper.make_model(
        helper.make_graph([], "unused", [], []),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ClearField("graph")
    path.write_bytes(
        model.SerializeToString()
        + length_field(7, graph)
        + (group if unknown_group_at == "model" else b"")
    )
    onnx.checker.check_model(str(path))
    assert raw.to_bytes(width, "little") in path.read_bytes()


class TestWireSymbolsAndFrames:
    @pytest.mark.parametrize("target", TARGETS)
    @pytest.mark.parametrize("packed", [False, True])
    @pytest.mark.parametrize("pattern", ["snan", "qnan", "negative_zero"])
    @pytest.mark.parametrize("operation", ["return", "Min"])
    def test_typed_wire_bits_and_flags(
        self, target, packed, pattern, operation, tmp_path
    ):
        dtype, nt, isa, abi = target
        raw = (
            {"snan": 0xFF800123, "qnan": 0xFFC00123, "negative_zero": 0x80000000}
            if dtype is T.FLOAT32
            else {
                "snan": 0xFFF0000000000123,
                "qnan": 0xFFF8000000000123,
                "negative_zero": 0x8000000000000000,
            }
        )[pattern]
        path = tmp_path / "input.onnx"
        typed_model(path, dtype, nt, raw, packed, operation)
        program = ONNXParser().parse(str(path))
        assert verify_ir(program) == (True, [])
        assert program.global_values[0].bit_pattern == raw
        result = CompilerDriver(
            configuration("greedy", "all", True, isa=isa, abi=abi)
        ).compile(str(path), str(tmp_path / "out.s"))
        assert result.success, result.errors
        canonical_nan = 0x7FC00000 if dtype is T.FLOAT32 else 0x7FF8000000000000
        if operation == "Min" and pattern != "negative_zero":
            with np.errstate(invalid="ignore"):
                assert np.isnan(
                    ReferenceEvaluator(onnx.load(path)).run(None, {})[0]
                ).all()
        expected = (
            (raw, 0)
            if operation == "return" or pattern == "negative_zero"
            else (canonical_nan, 0x10 if pattern == "snan" else 0)
        )
        assert (
            observe(
                result.output_text,
                tmp_path,
                dtype,
                function=".wire_model",
                isa=isa,
                abi=abi,
                separate=True,
            )
            == expected
        )

    @pytest.mark.parametrize("target", [TARGETS[0], TARGETS[2]])
    @pytest.mark.parametrize("packed", [False, True])
    @pytest.mark.parametrize("location", ["model", "graph", "tensor"])
    def test_unknown_protobuf_fields_preserve_source_bits(
        self, target, packed, location, tmp_path
    ):
        dtype, nt, isa, abi = target
        raw = 0x7F800123 if dtype is T.FLOAT32 else 0x7FF0000000000123
        path = tmp_path / "input.onnx"
        typed_model(path, dtype, nt, raw, packed, "return", location)
        result = CompilerDriver(configuration("greedy", isa=isa, abi=abi)).compile(
            str(path), str(tmp_path / "out.s")
        )
        assert result.success, result.errors
        assert observe(
            result.output_text,
            tmp_path,
            dtype,
            function=".wire_model",
            isa=isa,
            abi=abi,
            separate=True,
        ) == (raw, 0)

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("count", [1, 22])
    @pytest.mark.parametrize("block", ["next", ".next"])
    @pytest.mark.parametrize("function", ["main", ".role_model"])
    def test_structured_blocks_do_not_create_frames(
        self, mode, count, block, function, tmp_path
    ):
        instructions = [
            MI(O.LABEL, target=function, label_role="function"),
            MI(O.LI, P.reg("s0"), P.immediate(99)),
        ]
        for index in range(count):
            instructions.append(MI(O.LI, P.vreg(f"x{index}"), P.immediate(index + 1)))
        instructions += [
            MI(O.J, target=block),
            MI(O.LABEL, target=block, label_role="block"),
        ]
        value = P.vreg("x0")
        for index in range(1, count):
            dest = P.vreg(f"sum{index}")
            instructions.append(MI(O.ADD, dest, value, P.vreg(f"x{index}")))
            value = dest
        instructions += [
            MI(O.MV, P.reg("a0"), value),
            MI(O.JALR, P.reg("zero"), P.reg("ra"), comment="return"),
        ]
        allocation = allocate(instructions, mode)
        assert [
            (i.target, i.label_role) for i in allocation.instructions if i.op is O.LABEL
        ] == [(function, "function"), (block, "block")]
        assembly = apply_abi_frames(
            AsmEmitter(allocation.instructions).emit(), allocation
        )
        assert observe(
            assembly,
            tmp_path,
            T.INT32,
            function=function,
            isa="rv32im",
            abi="ilp32",
            separate=True,
        ) == (count * (count + 1) // 2, 0)

    @pytest.mark.parametrize("verify", [False, True])
    @pytest.mark.parametrize("extended", [False, True])
    @pytest.mark.parametrize("name", [".text", ".data", ".rodata", ".bss"])
    def test_reserved_function_names_rejected_before_output(
        self, verify, extended, name, tmp_path
    ):
        # Keep every name, selector and verifier path; allocation is never reached.
        graph = helper.make_graph(
            [],
            name,
            [],
            [helper.make_tensor_value_info("a", TP.INT32, [])],
            [TP(name="a", data_type=TP.INT32, raw_data=(7).to_bytes(4, "little"))],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        onnx.checker.check_model(model)
        path = tmp_path / "input.onnx"
        onnx.save(model, path)
        output = tmp_path / "out.s"
        output.write_text("preserve")
        result = CompilerDriver(
            configuration("greedy", "all", verify, extended=extended)
        ).compile(str(path), str(output))
        assert not result.success and any(
            "function symbol" in error for error in result.errors
        )
        assert output.read_text() == "preserve"

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("extended", [False, True])
    @pytest.mark.parametrize("verify", [False, True])
    @pytest.mark.parametrize("name", ["branch with space", "模型", "a,b"])
    def test_unsafe_blocks_rejected_before_output(
        self, mode, extended, verify, name, tmp_path
    ):
        b = IRBuilder()
        b.new_function("main")
        b.new_block(name)
        b.ret(b.load_const(7, T.INT32))
        assert verify_ir(b.program) == (True, [])
        output = tmp_path / "out.s"
        output.write_text("preserve")
        driver = CompilerDriver(configuration(mode, verify=verify, extended=extended))
        driver._parse = lambda *args: b.program
        result = driver.compile("", str(output), dsl_source="")
        assert not result.success and any(
            "label symbol" in error for error in result.errors
        )
        assert output.read_text() == "preserve"

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("extended", [False, True])
    @pytest.mark.parametrize("name", ["2branch", "x$y", ".Luser"])
    def test_supported_internal_symbols_execute(
        self, mode, extended, name, tmp_path
    ):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        b.br(name)
        b.new_block(name)
        b.ret(b.load_const(7, T.INT32))
        assembly = compile_ir(
            b.program, tmp_path, configuration(mode, extended=extended)
        )
        assert observe(assembly, tmp_path, separate=True) == (7, 0)

    @pytest.mark.parametrize("name", [".text", ".data", ".bss", ".rodata"])
    def test_reserved_call_targets_rejected(self, name):
        allocation = allocate(
            [
                MI(O.LABEL, target="main", label_role="function"),
                MI(O.CALL, target=name),
                MI(O.JALR, P.reg("zero"), P.reg("ra")),
            ],
            "greedy",
        )
        with pytest.raises(ValueError, match="call symbol"):
            AsmEmitter(allocation.instructions).emit()

    def test_custom_section_cannot_be_a_label(self):
        with pytest.raises(ValueError, match="label symbol"):
            AsmEmitter(
                [
                    MI(O.SECTION, comment="custom"),
                    MI(O.LABEL, target=".custom", label_role="function"),
                ]
            ).emit()

    @pytest.mark.parametrize("extended", [False, True])
    def test_elf_sizes_match_distinct_functions(self, extended, tmp_path):
        b = IRBuilder()
        for name, value in [("first", 5), ("second", 9)]:
            b.new_function(name)
            b.new_block("entry")
            b.ret(b.load_const(value, T.INT32))
        assembly = compile_ir(
            b.program, tmp_path, configuration("greedy", "all", True, extended=extended)
        )
        for name, value in [("first", 5), ("second", 9)]:
            path = tmp_path / name
            path.mkdir()
            assert observe(assembly, path, function=name, separate=True) == (value, 0)
        obj = tmp_path / "first" / "product.o"
        symbols = subprocess.check_output(["readelf", "-sW", str(obj)], text=True)
        parsed = {
            parts[-1]: (int(parts[1], 16), int(parts[2]))
            for line in symbols.splitlines()
            if len(parts := line.split()) >= 8 and parts[-1] in {"first", "second"}
        }
        sections = subprocess.check_output(["readelf", "-SW", str(obj)], text=True)
        text = next(
            line.split()
            for line in sections.splitlines()
            if re.search(r"\]\s+\.text\s", line)
        )
        text_size = int(text[text.index(".text") + 4], 16)
        assert parsed["first"][1] == parsed["second"][0] - parsed["first"][0] > 0
        assert parsed["second"][1] == text_size - parsed["second"][0] > 0

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("dtype", [T.INT32, T.FLOAT32, T.FLOAT64])
    def test_local_frames_do_not_inherit_other_functions(
        self, mode, dtype, tmp_path
    ):
        b = IRBuilder()
        for index in range(3):
            b.new_function(f".frame{index}")
            b.new_block("entry")
            ptr = b.alloca(100 if dtype is T.FLOAT64 else 200, dtype)
            b.store(ptr, b.load_const(5 + 4 * index, dtype))
            values = [b.load_const(value, dtype) for value in range(1, 25)]
            b.br("use")
            b.new_block("use")
            total = values[0]
            for value in values[1:]:
                total = b.add(total, value)
            b.ret(b.add(b.load(ptr, dtype), total))
        assembly = compile_ir(b.program, tmp_path, configuration(mode, "all", True))
        assert all(
            int(size) <= 2032 for size in re.findall(r"addi sp, sp, -(\d+)", assembly)
        )
        for index in range(3):
            path = tmp_path / str(index)
            path.mkdir()
            value = 305 + 4 * index
            expected = (
                value
                if dtype is T.INT32
                else int.from_bytes(
                    struct.pack("<f" if dtype is T.FLOAT32 else "<d", value), "little"
                )
            )
            assert observe(
                assembly, path, dtype, function=f".frame{index}", separate=True
            ) == (expected, 0)

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize(
        "dtype,count", [(T.INT32, 200), (T.FLOAT32, 200), (T.FLOAT64, 100)]
    )
    def test_spill_only_frames_are_per_function(self, mode, dtype, count, tmp_path):
        width = 64 if dtype is T.FLOAT64 else 32
        kind = RC.GPR if dtype is T.INT32 else RC.FPR
        pool = []
        instructions = []
        for index in range(3):
            function = f"spill{index}"
            instructions.append(MI(O.LABEL, target=function, label_role="function"))
            for value in range(count):
                dest = P.vreg(f"{function}_v{value}", kind, width)
                if dtype is T.INT32:
                    instructions.append(MI(O.LI, dest, P.immediate(value + 1 + index)))
                else:
                    label = f".pool_{index}_{value}"
                    raw = int.from_bytes(
                        struct.pack("<f" if width == 32 else "<d", value + 1 + index),
                        "little",
                    )
                    pool.append(FloatConstant(label, width, raw))
                    instructions += [
                        MI(O.LA, P.reg("a1"), P.symbol(label)),
                        MI(O.FLW if width == 32 else O.FLD, dest, P.reg("a1")),
                    ]
            label = f"use{index}"
            instructions += [
                MI(O.J, target=label),
                MI(O.LABEL, target=label, label_role="block"),
            ]
            total = P.vreg(f"{function}_v0", kind, width)
            for value in range(1, count):
                dest = P.vreg(f"{function}_sum{value}", kind, width)
                instructions.append(
                    MI(
                        (
                            O.ADD
                            if dtype is T.INT32
                            else O.FADD_S if width == 32 else O.FADD_D
                        ),
                        dest,
                        total,
                        P.vreg(f"{function}_v{value}", kind, width),
                        rounding_mode=None if dtype is T.INT32 else "rne",
                    )
                )
                total = dest
            instructions += [
                MI(
                    O.MV if dtype is T.INT32 else O.FMV_S if width == 32 else O.FMV_D,
                    P.reg("a0" if dtype is T.INT32 else "fa0", kind, width),
                    total,
                ),
                MI(O.JALR, P.reg("zero"), P.reg("ra"), comment="return"),
            ]
        allocation = allocate(instructions, mode)
        assert allocation.spill_bytes > 2032
        assembly = apply_abi_frames(
            AsmEmitter(allocation.instructions, pool).emit(), allocation
        )
        for index in range(3):
            path = tmp_path / str(index)
            path.mkdir()
            expected = count * (count + 1) // 2 + count * index
            if dtype is not T.INT32:
                expected = int.from_bytes(
                    struct.pack("<f" if width == 32 else "<d", expected), "little"
                )
            assert observe(
                assembly, path, dtype, function=f"spill{index}", separate=True
            ) == (expected, 0)

    def test_single_oversized_local_frame_still_rejected(self, tmp_path):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        ptr = b.alloca(600, T.INT32)
        b.store(ptr, b.load_const(7, T.INT32))
        b.ret(b.load(ptr, T.INT32))
        driver = CompilerDriver(configuration("greedy"))
        driver._parse = lambda *args: b.program
        output = tmp_path / "out.s"
        output.write_text("preserve")
        result = driver.compile("", str(output), dsl_source="")
        assert not result.success and any(
            "stack frame exceeds" in error for error in result.errors
        )
        assert output.read_text() == "preserve"
