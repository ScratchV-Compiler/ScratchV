"""Safety tests for standalone symbolic peephole finalization."""

from __future__ import annotations

import copy
import struct
from pathlib import Path

import pytest

from scratchv.standalone.onnx_to_riscv_standalone import (
    _R_GP,
    _R_T0,
    _R_T1,
    _R_T2,
    CNNRISCVGenerator,
    MemoryPlan,
    ONNXModel,
    RISCVEmitter,
    patch_gp_data_base,
    rv_add,
    rv_addi,
    rv_auipc,
    rv_beq,
    rv_nop,
    rv_ret,
)
from scratchv.standalone.peephole_relocator import PeepholeFinalizer

MODEL = Path(__file__).resolve().parents[1] / "models" / "graph" / "cnn.onnx"


def _finalize(emitter: RISCVEmitter, enabled: bool = True) -> PeepholeFinalizer:
    finalizer = PeepholeFinalizer(enabled=enabled)
    emitter.finalize(finalizer)
    return finalizer


def test_default_emitter_path_does_not_enable_peephole_and_matches_disabled_finalizer():
    default = RISCVEmitter()
    default.emit(rv_addi(_R_T0, _R_T1, 0))
    default.emit(rv_ret())
    default.resolve_fixups()

    disabled = RISCVEmitter()
    disabled.emit(rv_addi(_R_T0, _R_T1, 0))
    disabled.emit(rv_ret())
    _finalize(disabled, enabled=False)

    assert disabled.code == default.code


def test_default_standalone_generator_matches_disabled_injected_path():
    model = ONNXModel.from_file(str(MODEL))
    memory = MemoryPlan()
    memory.layout_weights(model.initializers)
    if model.inputs:
        input_tensor = model.inputs[0]
        memory.alloc_workspace(input_tensor.name, input_tensor.num_elements)

    default_code = CNNRISCVGenerator(model, copy.deepcopy(memory)).generate()
    disabled_code = CNNRISCVGenerator(
        model,
        copy.deepcopy(memory),
        finalize_strategy=PeepholeFinalizer(enabled=False),
    ).generate()

    assert disabled_code == default_code


def test_slash_label_is_mapped_during_optimization_and_relocated_after_deletion():
    emitter = RISCVEmitter()
    emitter.label("node/with-slash")
    emitter.emit_branch(rv_beq, 0, 0, "node/with-slash")
    emitter.emit(rv_nop())
    emitter.label("target/with-slash")
    emitter.emit(rv_ret())

    finalizer = _finalize(emitter)

    assert "node/with-slash" not in finalizer.input_assembly
    assert "target/with-slash" not in finalizer.input_assembly
    assert "L0000" in finalizer.input_assembly
    assert "L0001" in finalizer.input_assembly
    assert emitter.labels["target/with-slash"] == 1
    assert len(emitter.code) == 2
    assert finalizer.optimizer.total_matches["beq zero-zero to jump"] == 1
    assert finalizer.optimizer.total_matches["nop elimination"] == 1


def test_equal_length_branch_rewrite_keeps_target_and_machine_encoding():
    emitter = RISCVEmitter()
    emitter.emit_branch(rv_beq, 0, 0, "target")
    emitter.label("target")
    emitter.emit(rv_ret())

    finalizer = _finalize(emitter)

    assert len(emitter.code) == 2
    assert finalizer.optimizer.instructions_saved == 0
    assert finalizer.optimizer.total_matches["beq zero-zero to jump"] == 1
    assert (emitter.code[0] & 0x7F) == 0b1101111
    assert emitter.code[0] != rv_addi(0, 0, 0)


def test_init_data_base_pair_is_protected_from_self_move_elimination():
    emitter = RISCVEmitter()
    emitter.label("_init_data_base")
    emitter.emit(rv_auipc(_R_GP, 0))
    emitter.emit(rv_addi(_R_GP, _R_GP, 0))
    emitter.emit(rv_ret())
    emitter.protected_indices.update({0, 1})

    finalizer = _finalize(emitter)

    assert len(emitter.code) == 3
    assert finalizer.optimizer.total_matches["addi-zero self elimination"] == 0


def test_undefined_label_and_branch_overflow_fail_loudly():
    undefined = RISCVEmitter()
    undefined.emit_branch(rv_beq, 0, 0, "missing")
    with pytest.raises(ValueError, match="undefined label"):
        _finalize(undefined)

    overflow = RISCVEmitter()
    overflow.emit_branch(rv_beq, _R_T0, _R_T1, "far")
    for _ in range(1025):
        overflow.emit(rv_add(_R_T0, _R_T1, _R_T2))
    overflow.label("far")
    with pytest.raises(ValueError, match="branch.*range"):
        _finalize(overflow)

    duplicate = RISCVEmitter()
    duplicate.label("same")
    with pytest.raises(ValueError, match="Duplicate label"):
        duplicate.label("same")


def test_gp_patch_uses_final_code_size():
    emitter = RISCVEmitter()
    emitter.label("_init_data_base")
    emitter.emit(rv_auipc(_R_GP, 0))
    emitter.emit(rv_addi(_R_GP, _R_GP, 0))
    emitter.emit(rv_ret())

    code = struct.pack("<3I", *emitter.code)
    patched, report = patch_gp_data_base(emitter, code, sync_listing=True)

    assert report["code_size"] == 12
    assert report["validation"] is True
    assert len(patched) == 12
    assert emitter.code[1] != rv_addi(_R_GP, _R_GP, 0)
