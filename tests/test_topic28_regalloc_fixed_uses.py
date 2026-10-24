"""Fixed physical sources can relocate a virtual definition across CFG edges.

The returned virtual result must retain the bits actually computed before a
CFG edge, even when its preferred physical register is a fixed source operand.
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
from tests.topic28_runtime import execute_fp


class TestFixedRegisterUses:
    @pytest.mark.parametrize("width", (32, 64))
    @pytest.mark.parametrize("alias", ("ft0", "f0"))
    @pytest.mark.parametrize("cfg", ("jump", "diamond_taken", "diamond_not_taken"))
    def test_fixed_source_relocation_keeps_computed_result_across_edges(
        self,
        width,
        alias,
        cfg,
        tmp_path,
    ):
        fp = lambda name: P.vreg(name, RC.FPR, width)
        load = O.FLW if width == 32 else O.FLD
        negate = O.FNEG_S if width == 32 else O.FNEG_D
        move = O.FMV_S if width == 32 else O.FMV_D
        stream = [
            MI(O.LABEL, target="main"),
            MI(O.LA, P.vreg("addr"), P.symbol("payload")),
        ]
        # These dead, sequential definitions cycle the preferred register bank.
        # All FPRs then hold 7.0, and the next virtual result is assigned ft0.
        # There is no pressure spill, CALL or explicit allocatable physical def.
        stream.extend(
            MI(load, fp(f"seed{index}"), P.vreg("addr")) for index in range(12)
        )
        stream.append(MI(negate, fp("result"), P.reg(alias, RC.FPR, width)))
        if cfg == "jump":
            stream.append(MI(O.J, target=".join"))
        else:
            stream.extend(
                [
                    MI(
                        O.BEQ if cfg == "diamond_taken" else O.BNE,
                        P.reg("zero"),
                        P.reg("zero"),
                        target=".alternate",
                    ),
                    MI(O.J, target=".join"),
                    MI(O.LABEL, target=".alternate"),
                    MI(O.J, target=".join"),
                ]
            )
        stream.extend(
            [
                MI(O.LABEL, target=".join"),
                MI(move, P.reg("fa0", RC.FPR, width), fp("result")),
                MI(O.JALR, P.reg("zero"), P.reg("ra")),
            ]
        )
        allocator = LinearScanAllocator()
        allocation = allocator.allocate_result(stream)
        assert allocator.alloc_map["result"] == "ft0"
        instruction = next(
            insn for insn in allocation.instructions if insn.op is negate
        )
        assert (
            str(instruction.dst.value) != "ft0"
        ), "the fixed source must force relocation"
        encode = lambda value: int.from_bytes(
            struct.pack("<f" if width == 32 else "<d", value), "little"
        )
        data = (
            ".section .rodata\n.p2align 3\npayload:\n"
            f"{'.word' if width == 32 else '.quad'} {encode(7.0)}\n"
        )
        assembly = data + apply_abi_frames(
            AsmEmitter(allocation.instructions).emit(), allocation
        )
        assert execute_fp(assembly, tmp_path, width) == (encode(-7.0), 0)
