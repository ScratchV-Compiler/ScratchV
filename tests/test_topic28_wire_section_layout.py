"""Protobuf merge semantics, function section sizes and dollar-label references."""

import re
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
from scratchv.ir.types import DataType as T
from tests.topic28_onnx import integer_field, length_field, typed_tensor
from tests.topic28_support import MODES, TARGETS, allocate, configuration
from tests.topic28_runtime import _required_tool, observe_abi as observe


def write_model(path, nt, tensors, operation, *, split_graph=False):
    output = "a" if operation == "return" else "out"
    nodes = (
        []
        if operation == "return"
        else [
            helper.make_node(
                operation, ["a"] if operation == "Sqrt" else ["a", "b"], [output]
            )
        ]
    )
    graph = helper.make_graph(
        nodes, "main", [], [helper.make_tensor_value_info(output, nt, [])]
    ).SerializeToString()
    data = b"".join(length_field(5, tensor) for tensor in tensors)
    if operation == "Min":
        two = 0x40000000 if nt == TP.FLOAT else 0x4000000000000000
        data += length_field(
            5,
            TP(
                name="b",
                data_type=nt,
                raw_data=two.to_bytes(4 if nt == TP.FLOAT else 8, "little"),
            ).SerializeToString(),
        )
    model = helper.make_model(
        helper.make_graph([], "unused", [], []),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ClearField("graph")
    path.write_bytes(
        model.SerializeToString()
        + (
            length_field(7, graph) + length_field(7, data)
            if split_graph
            else length_field(7, graph + data)
        )
    )
    onnx.checker.check_model(str(path))


def compile_model(path, tmp_path, mode, level, verify, target):
    dtype, _, isa, abi = target
    program = ONNXParser().parse(str(path))
    assert verify_ir(program) == (True, [])
    result = CompilerDriver(
        configuration(mode, level, verify, isa=isa, abi=abi)
    ).compile(str(path), str(tmp_path / "out.s"))
    assert result.success, result.errors
    return program, observe(result.output_text, tmp_path, dtype, isa=isa, abi=abi)


def function_body(name, count, value, role):
    result = [
        MI(O.LABEL, target=name, label_role=role),
        MI(O.LI, P.reg("s0"), P.immediate(99)),
    ]
    result += [
        MI(O.LI, P.vreg(f"{name}_v{i}"), P.immediate(i + 1)) for i in range(count)
    ]
    total = P.vreg(f"{name}_v0")
    for i in range(1, count):
        dest = P.vreg(f"{name}_sum{i}")
        result.append(MI(O.ADD, dest, total, P.vreg(f"{name}_v{i}")))
        total = dest
    result += [
        MI(O.ADDI, P.reg("a0"), total, P.immediate(value)),
        MI(O.JALR, P.reg("zero"), P.reg("ra"), comment="return"),
    ]
    return result


class TestWireAndSectionLayout:
    @pytest.mark.parametrize("target", TARGETS)
    @pytest.mark.parametrize("packed", [False, True])
    @pytest.mark.parametrize("operation", ["return", "Sqrt"])
    def test_repeated_name_uses_final_initializer_identity(
        self, target, packed, operation, tmp_path
    ):
        # Wire decoding precedes allocation. Keep both encodings and all targets;
        # allocator and optimization coverage lives in the respective suites.
        _, nt, _, _ = target
        four, nine, two = (
            (0x40800000, 0x41100000, 0x40000000)
            if nt == TP.FLOAT
            else (0x4010000000000000, 0x4022000000000000, 0x4000000000000000)
        )
        path = tmp_path / "input.onnx"
        write_model(
            path,
            nt,
            [
                typed_tensor(nt, four, packed),
                typed_tensor(nt, nine, packed) + length_field(8, b"b"),
            ],
            operation,
        )
        assert [t.name for t in onnx.load(path).graph.initializer] == ["a", "b"]
        program, actual = compile_model(path, tmp_path, "greedy", "all", True, target)
        assert {v.name: v.bit_pattern for v in program.global_values} == {
            "a": four,
            "b": nine,
        }
        assert actual == (four if operation == "return" else two, 0)

    @pytest.mark.parametrize("target", TARGETS)
    @pytest.mark.parametrize("packed", [False, True])
    @pytest.mark.parametrize("operation", ["return", "Min"])
    @pytest.mark.parametrize("variant", ["default_metadata", "unknown_raw_wire"])
    def test_storage_discriminator_preserves_typed_bits(
        self, target, packed, operation, variant, tmp_path
    ):
        _, nt, _, _ = target
        raw = 0x7F800123 if nt == TP.FLOAT else 0x7FF0000000000123
        tensor = typed_tensor(nt, raw, packed) + integer_field(14, TP.DEFAULT)
        if variant == "default_metadata":
            tensor += length_field(
                13,
                onnx.StringStringEntryProto(
                    key="location", value="ignored.bin"
                ).SerializeToString(),
            )
        else:
            tensor += integer_field(
                9, 1
            )  # Unknown varint, not length-delimited raw_data.
        path = tmp_path / "input.onnx"
        write_model(path, nt, [tensor], operation)
        program, actual = compile_model(path, tmp_path, "greedy", "all", True, target)
        assert program.global_values[0].bit_pattern == raw
        canonical_nan = 0x7FC00000 if nt == TP.FLOAT else 0x7FF8000000000000
        if operation == "Min":
            with np.errstate(invalid="ignore"):
                assert np.isnan(
                    ReferenceEvaluator(onnx.load(path)).run(None, {})[0]
                ).all()
        assert actual == ((raw, 0) if operation == "return" else (canonical_nan, 0x10))

    @pytest.mark.parametrize("target", TARGETS)
    @pytest.mark.parametrize(
        "variant",
        [
            "renamed",
            "unknown_name_wire",
            "last_dtype",
            "last_location",
            "unknown_location_enum",
            "unknown_location_wire",
            "unknown_external_wire",
            "merged_graph",
            "wrapped_dtype",
            "wrapped_location",
        ],
    )
    def test_singular_wire_and_graph_merge_semantics(
        self, target, variant, tmp_path
    ):
        _, nt, _, _ = target
        raw = 0xFF800123 if nt == TP.FLOAT else 0xFFF0000000000123
        tensor = typed_tensor(nt, raw, True, "old" if variant == "renamed" else "a")
        if variant == "renamed":
            tensor += length_field(8, b"a")
        elif variant == "unknown_name_wire":
            tensor += integer_field(8, 1)
        elif variant == "last_dtype":
            tensor = integer_field(2, TP.INT32) + tensor
        elif variant == "last_location":
            tensor += integer_field(14, TP.EXTERNAL) + integer_field(14, TP.DEFAULT)
            tensor += length_field(
                13,
                onnx.StringStringEntryProto(
                    key="location", value="ignored.bin"
                ).SerializeToString(),
            )
        elif variant == "unknown_location_enum":
            tensor += integer_field(14, TP.DEFAULT) + integer_field(14, 99)
        elif variant == "unknown_location_wire":
            tensor += length_field(14, b"\x01")
        elif variant == "unknown_external_wire":
            tensor += integer_field(13, 1)
        elif variant == "wrapped_dtype":
            tensor += integer_field(2, (1 << 32) + nt)
        elif variant == "wrapped_location":
            tensor += integer_field(14, (1 << 32) + TP.DEFAULT)
            tensor += length_field(
                13,
                onnx.StringStringEntryProto(
                    key="location", value="ignored.bin"
                ).SerializeToString(),
            )
        path = tmp_path / "input.onnx"
        write_model(path, nt, [tensor], "Min", split_graph=variant == "merged_graph")
        program, actual = compile_model(path, tmp_path, "greedy", "all", True, target)
        assert program.global_values[0].bit_pattern == raw
        with np.errstate(invalid="ignore"):
            assert np.isnan(ReferenceEvaluator(onnx.load(path)).run(None, {})[0]).all()
        assert actual == (0x7FC00000 if nt == TP.FLOAT else 0x7FF8000000000000, 0x10)

    @pytest.mark.parametrize("target", TARGETS)
    @pytest.mark.parametrize("storage", ["raw", "external"])
    def test_actual_raw_and_external_storage_remain_bit_exact(
        self, target, storage, tmp_path
    ):
        _, nt, _, _ = target
        raw = 0xFF800123 if nt == TP.FLOAT else 0xFFF0000000000123
        data = raw.to_bytes(4 if nt == TP.FLOAT else 8, "little")
        tensor = TP(name="a", data_type=nt)
        if storage == "raw":
            tensor.raw_data = data
            tensor.data_location = TP.DEFAULT
            tensor.external_data.add(key="location", value="ignored.bin")
        else:
            (tmp_path / "weights.bin").write_bytes(data)
            tensor.data_location = TP.EXTERNAL
            tensor.external_data.add(key="location", value="weights.bin")
        # The real storage field must win over a subsequent unknown same-number
        # field of a different wire type; this fixture has no typed numeric data.
        wire = tensor.SerializeToString() + integer_field(9, 1)
        path = tmp_path / "input.onnx"
        write_model(path, nt, [wire], "return")
        program, actual = compile_model(path, tmp_path, "greedy", "all", True, target)
        assert program.global_values[0].bit_pattern == raw
        assert actual == (raw, 0)

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize(
        "roles", ["explicit_normal", "explicit_dot", "legacy_normal"]
    )
    @pytest.mark.parametrize("count", [1, 24])
    @pytest.mark.parametrize("sections", ["between", "trailing", "both", "same"])
    def test_function_sizes_survive_section_switches_and_abi(
        self, mode, roles, count, sections, tmp_path
    ):
        first, second = (
            (".first", ".second") if roles == "explicit_dot" else ("first", "second")
        )
        role = None if roles == "legacy_normal" else "function"
        instructions = function_body(first, count, 5, role)
        if sections in {"between", "both", "same"}:
            instructions.append(
                MI(O.SECTION, comment="text" if sections == "same" else "text.hot")
            )
        instructions += function_body(second, count, 9, role)
        if sections in {"trailing", "both"}:
            instructions.append(MI(O.SECTION, comment="text.unused"))
        allocation = allocate(instructions, mode)
        assembly = apply_abi_frames(
            AsmEmitter(allocation.instructions).emit(), allocation
        )
        for name, value in [(first, 5), (second, 9)]:
            path = tmp_path / name
            path.mkdir()
            assert observe(assembly, path, T.INT32, name, "rv32im", "ilp32") == (
                count * (count + 1) // 2 + value,
                0,
            )
        obj = tmp_path / first / "product.o"
        symbols = subprocess.check_output(
            [_required_tool("readelf"), "-sW", str(obj)], text=True
        )
        parsed = {
            parts[-1]: parts
            for line in symbols.splitlines()
            if len(parts := line.split()) >= 8 and parts[-1] in {first, second}
        }
        section_text = subprocess.check_output(
            [_required_tool("readelf"), "-SW", str(obj)], text=True
        )
        section_sizes = {
            int(match[1]): int(match[2], 16)
            for line in section_text.splitlines()
            if (
                match := re.match(
                    r"\s*\[\s*(\d+)\]\s+\S+\s+PROGBITS\s+[0-9a-f]+\s+[0-9a-f]+\s+([0-9a-f]+)",
                    line,
                )
            )
        }
        for row in parsed.values():
            assert row[3:5] == ["FUNC", "GLOBAL"] and int(row[2]) > 0
        assert set(parsed) == {first, second}
        a, b = parsed[first], parsed[second]
        first_end = int(b[1], 16) if a[6] == b[6] else section_sizes[int(a[6])]
        assert int(a[2]) == first_end - int(a[1], 16)
        assert int(b[2]) == section_sizes[int(b[6])] - int(b[1], 16)

    def test_function_body_cannot_silently_span_sections(self):
        instructions = [
            MI(O.LABEL, target="main", label_role="function"),
            MI(O.LI, P.reg("a0"), P.immediate(7)),
            MI(O.SECTION, comment="text.hot"),
            MI(O.LABEL, target=".next", label_role="block"),
            MI(O.JALR, P.reg("zero"), P.reg("ra")),
        ]
        allocation = allocate(instructions, "greedy")
        with pytest.raises(ValueError, match="cannot continue across output sections"):
            AsmEmitter(allocation.instructions).emit()

    @pytest.mark.parametrize("label", ["$foo", "foo$bar", ".Lfoo", ".$foo"])
    @pytest.mark.parametrize("fallback", [False, True])
    @pytest.mark.parametrize("op", [O.J, O.JAL, O.BNEZ, O.BEQ, O.BNE, O.BLT, O.BGE])
    def test_internal_dollar_branch_references_execute(
        self, label, fallback, op, tmp_path
    ):
        operands = {
            O.J: (),
            O.JAL: (P.reg("zero"),),
            O.BNEZ: (P.reg("t2"),),
            O.BEQ: (P.reg("zero"), P.reg("zero")),
            O.BNE: (P.reg("t2"), P.reg("zero")),
            O.BLT: (P.reg("zero"), P.reg("t2")),
            O.BGE: (P.reg("t2"), P.reg("zero")),
        }[op]
        jump = MI(
            op, *operands, **({"comment": label} if fallback else {"target": label})
        )
        instructions = [
            MI(O.LABEL, target=".dollar", label_role="function"),
            MI(O.LI, P.reg("t2"), P.immediate(1)),
            jump,
            MI(O.LI, P.reg("a0"), P.immediate(-1)),
            MI(O.JALR, P.reg("zero"), P.reg("ra")),
            MI(O.LABEL, target=label, label_role="block"),
            MI(O.LI, P.reg("a0"), P.immediate(7)),
            MI(O.JALR, P.reg("zero"), P.reg("ra")),
        ]
        allocation = allocate(instructions, "greedy")
        assembly = apply_abi_frames(
            AsmEmitter(allocation.instructions).emit(), allocation
        )
        assert observe(assembly, tmp_path, T.INT32, ".dollar", "rv32im", "ilp32") == (
            7,
            0,
        )

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("label", ["$pool", "pool$bits", ".Lpool", ".$pool"])
    @pytest.mark.parametrize("target", TARGETS)
    def test_internal_dollar_address_references_execute(
        self, mode, label, target, tmp_path
    ):
        dtype, nt, isa, abi = target
        width = 32 if nt == TP.FLOAT else 64
        raw = 0x7F800123 if width == 32 else 0x7FF0000000000123
        instructions = [
            MI(O.LABEL, target="main", label_role="function"),
            MI(O.LA, P.vreg("address"), P.symbol(label)),
            MI(
                O.FLW if width == 32 else O.FLD,
                P.reg("fa0", RC.FPR, width),
                P.vreg("address"),
            ),
            MI(O.JALR, P.reg("zero"), P.reg("ra")),
        ]
        allocation = allocate(instructions, mode)
        assembly = apply_abi_frames(
            AsmEmitter(
                allocation.instructions, [FloatConstant(label, width, raw)]
            ).emit(),
            allocation,
        )
        assert observe(assembly, tmp_path, dtype, isa=isa, abi=abi) == (raw, 0)
