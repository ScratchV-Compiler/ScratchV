"""Register allocation and emitter edge cases.

Covers integer constant legality, physical clobber on CFG edges and
terminator-defined virtual registers.
"""

import struct

import pytest

from scratchv.backend.abi_frame import apply_abi_frames
from scratchv.backend.asm_emit import AsmEmitter
from scratchv.backend.machine_types import (
    MachineInstr as MI,
    MachineOp as O,
    MachineOperand as P,
    RegisterClass as RC,
)
from scratchv.backend.regalloc_linear import LinearScanAllocator
from scratchv.compiler import CompilerConfig, CompilerDriver
from tests.topic28_runtime import execute_fp, execute_i32


def _fp(name, width):
    return P.vreg(name, RC.FPR, width)


class TestRegallocEdgeCases:
    @pytest.mark.parametrize("extended", [False, True])
    @pytest.mark.parametrize(
        "a,b,expression,expected",
        [
            (7, 3, "add(a,b)", 10),
            (7, 3, "sub(a,b)", 4),
            (7, 3, "mul(a,b)", 21),
            (7, 3, "neg(a)", -7),
            (2147483647, 1, "add(a,b)", -2147483648),
            (-2147483648, 1, "sub(a,b)", 2147483647),
            (-7, 3, "mul(a,b)", -21),
            (-2147483648, 0, "neg(a)", -2147483648),
        ],
    )
    def test_integer_constant_operations_are_legal_and_execute(
        self, extended, a, b, expression, expected, tmp_path
    ):
        source = f"a = const({a},dtype:i32)\nb = const({b},dtype:i32)\nr = {expression}\nreturn r"
        result = CompilerDriver(
            CompilerConfig(
                reg_alloc="greedy",
                use_extended_isel=extended,
                verify_ir=True,
            )
        ).compile("", str(tmp_path / "integer.s"), dsl_source=source)
        assert result.success, result.errors
        assert execute_i32(result.output_text, tmp_path) == expected

    @pytest.mark.parametrize("extended", [False, True])
    @pytest.mark.parametrize("condition,expected", [("a < 3", 7), ("a > 3", 9)])
    def test_integer_control_flow_constant_stores_are_legal(
        self, extended, condition, expected, tmp_path
    ):
        source = f"""a = const(1,dtype:i32)
r = const(0,dtype:i32)
if ({condition}):
 r = const(7,dtype:i32)
else:
 r = const(9,dtype:i32)
endif
return r
"""
        result = CompilerDriver(
            CompilerConfig(
                reg_alloc="greedy",
                use_extended_isel=extended,
                verify_ir=True,
            )
        ).compile("", str(tmp_path / "control.s"), dsl_source=source)
        assert result.success, result.errors
        assert execute_i32(result.output_text, tmp_path) == expected

    @pytest.mark.parametrize("alias", ["ft0", "f0"])
    def test_report_literal_stream_allocates_after_physical_clobber(self, alias):
        stream = [
            MI(O.LABEL, target="main"),
            MI(O.LI, P.vreg("addr"), P.immediate(256)),
            MI(O.FLD, _fp("value", 64), P.vreg("addr")),
            MI(O.FMV_D, P.reg(alias, RC.FPR, 64), P.reg("fa0", RC.FPR, 64)),
            MI(O.J, target=".next"),
            MI(O.LABEL, target=".next"),
            MI(O.FMV_D, P.reg("fa0", RC.FPR, 64), _fp("value", 64)),
            MI(O.JALR, P.reg("zero"), P.reg("ra")),
        ]
        result = LinearScanAllocator().allocate_result(stream)
        assert any(
            "[regalloc:reload]" in instr.comment for instr in result.instructions
        )

    @pytest.mark.parametrize(
        "kind,width,alias",
        [
            ("integer", 32, "t1"),
            ("integer", 32, "x6"),
            ("float", 32, "ft0"),
            ("float", 32, "f0"),
            ("float", 64, "ft0"),
            ("float", 64, "f0"),
        ],
    )
    @pytest.mark.parametrize("cfg", ["jump", "diamond_taken", "diamond_not_taken"])
    def test_physical_definition_preserves_edge_value_bits(
        self, kind, width, alias, cfg, tmp_path
    ):
        is_float = kind == "float"
        operand = _fp("value", width) if is_float else P.vreg("value")
        load = (O.FLW if width == 32 else O.FLD) if is_float else O.LW
        move = (O.FMV_S if width == 32 else O.FMV_D) if is_float else O.MV
        target = P.reg("fa0", RC.FPR, width) if is_float else P.reg("a0")
        clobber = (
            MI(move, P.reg(alias, RC.FPR, width), P.reg("fa0", RC.FPR, width))
            if is_float
            else MI(O.LI, P.reg(alias), P.immediate(99))
        )
        stream = [
            MI(O.LABEL, target="main"),
            MI(O.LA, P.vreg("addr"), P.symbol("payload")),
            MI(load, operand, P.vreg("addr")),
        ]
        if cfg == "jump":
            stream.extend([clobber, MI(O.J, target=".join")])
        else:
            stream.extend(
                [
                    MI(
                        O.BEQ if cfg == "diamond_taken" else O.BNE,
                        P.reg("zero"),
                        P.reg("zero"),
                        target=".overwrite",
                    ),
                    MI(O.J, target=".join"),
                    MI(O.LABEL, target=".overwrite"),
                    clobber,
                    MI(O.J, target=".join"),
                ]
            )
        stream.extend(
            [
                MI(O.LABEL, target=".join"),
                MI(move, target, operand),
                MI(O.JALR, P.reg("zero"), P.reg("ra")),
            ]
        )
        allocator = LinearScanAllocator()
        result = allocator.allocate_result(stream)
        # Ensure this actually tests the explicitly clobbered allocation, not a
        # value coincidentally living in another register.
        assert allocator.alloc_map["value"] == ("ft0" if is_float else "t1")
        bits = (
            int.from_bytes(struct.pack("<f" if width == 32 else "<d", 7.0), "little")
            if is_float
            else 7
        )
        data = f".section .rodata\n.p2align 3\npayload:\n{' .word' if width == 32 else ' .quad'} {bits}\n"
        assembly = data + apply_abi_frames(
            AsmEmitter(result.instructions).emit(), result
        )
        if is_float:
            assert execute_fp(assembly, tmp_path, width) == (bits, 0)
        else:
            assert execute_i32(assembly, tmp_path) == 7

    @pytest.mark.parametrize("opcode", [O.JAL, O.JALR])
    def test_defining_virtual_terminator_is_rejected_instead_of_storing_before_definition(
        self, opcode
    ):
        # A link result is defined by the jump itself. Saving it before the jump
        # reads an old value, while a store after the jump cannot reach its edge.
        # This unsupported interface must fail explicitly, as greedy already does.
        stream = [
            MI(O.LABEL, target="main"),
            MI(O.LI, P.reg("t0"), P.immediate(99)),
            MI(
                opcode,
                P.vreg("link"),
                P.reg("ra") if opcode is O.JALR else None,
                target=".next",
            ),
            MI(O.LABEL, target=".next"),
            MI(O.MV, P.reg("a0"), P.vreg("link")),
            MI(O.JALR, P.reg("zero"), P.reg("ra")),
        ]
        with pytest.raises(
            RuntimeError, match="terminator defining a virtual register"
        ):
            LinearScanAllocator().allocate_result(stream)
