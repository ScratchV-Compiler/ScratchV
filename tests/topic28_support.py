"""Shared compilation and allocation setup for Topic 28 tests."""

from onnx import TensorProto as TP

from scratchv.analysis.ir_verifier import verify_ir
from scratchv.backend.regalloc_linear import LinearScanAllocator
from scratchv.backend.register_alloc import RegisterAllocator
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.ir.types import DataType as T

MODES = ("naive", "greedy", "linear")
LEVELS = ("none", "basic", "all")
TARGETS = ((T.FLOAT32, TP.FLOAT, "rv32imf", "ilp32f"),
           (T.FLOAT32, TP.FLOAT, "rv32imfd", "ilp32d"),
           (T.FLOAT64, TP.DOUBLE, "rv32imfd", "ilp32d"))


def configuration(mode, level="none", verify=True, *, extended=True, isa="rv32imfd", abi="ilp32d"):
    return CompilerConfig(reg_alloc=mode, optimize_level=level, verify_ir=verify,
        use_extended_isel=extended, target_isa=isa, target_abi=abi)


def compile_ir(program, path, config):
    assert verify_ir(program) == (True, [])
    driver = CompilerDriver(config)
    driver._parse = lambda *args: program
    result = driver.compile("", str(path / "out.s"), dsl_source="")
    assert result.success, result.errors
    assert verify_ir(program) == (True, [])
    return result.output_text


def allocate(instructions, mode):
    allocator = LinearScanAllocator() if mode == "linear" else RegisterAllocator(instructions, mode)
    return allocator.allocate_result(instructions) if mode == "linear" else allocator.allocate_result()
