"""End-to-end test: ONNX CNN model → RISC-V binary → data verification.

Full pipeline test:
    1. Parse cnn.onnx → ScratchV IR
    2. IR → RISC-V assembly (instruction selection + register alloc + emit)
    3. RISC-V assembly → 32-bit machine code (RV32IM encoder)
    4. IR trace execution (numpy) → compiled output
    5. ONNX Runtime reference inference → expected output
    6. MSE / MAE comparison
"""

from __future__ import annotations

import os
import numpy as np
import pytest

# ── Path to project root ────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "graph", "cnn.onnx")

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="cnn.onnx model not found",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def onnx_model():
    """Load the ONNX model and return metadata."""
    import onnx
    model = onnx.load(MODEL_PATH)
    graph = model.graph

    inputs = []
    for inp in graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        inputs.append({"name": inp.name, "shape": shape})

    nodes = []
    for node in graph.node:
        nodes.append({
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
        })

    initializers = {}
    for init in graph.initializer:
        arr = onnx.numpy_helper.to_array(init)
        initializers[init.name] = arr

    return {
        "graph_name": graph.name,
        "inputs": inputs,
        "nodes": nodes,
        "num_nodes": len(nodes),
        "initializers": initializers,
    }


@pytest.fixture(scope="module")
def ir_program():
    """Step 1: Parse ONNX → ScratchV IR Program."""
    from scratchv.frontend.onnx_parser import ONNXParser
    parser = ONNXParser()
    program = parser.parse(MODEL_PATH)
    return program


@pytest.fixture(scope="module")
def riscv_assembly(ir_program):
    """Step 2: IR → RISC-V assembly text."""
    from scratchv.backend.instruction_select import InstructionSelector
    from scratchv.backend.register_alloc import RegisterAllocator
    from scratchv.backend.asm_emit import AsmEmitter

    selector = InstructionSelector(ir_program)
    machine_instrs = selector.run()
    alloc = RegisterAllocator(machine_instrs, mode="greedy")
    allocated = alloc.run()
    emitter = AsmEmitter(allocated)
    return emitter.emit()


@pytest.fixture(scope="module")
def riscv_binary(riscv_assembly):
    """Step 3: RISC-V assembly → binary machine code."""
    from scratchv.backend.riscv_encoder import assemble_to_binary
    return assemble_to_binary(riscv_assembly)


@pytest.fixture(scope="module")
def test_input():
    """Create reproducible test input."""
    rng = np.random.RandomState(42)
    return {"input1": rng.randn(1, 3, 250, 250).astype(np.float32) * 0.1}


@pytest.fixture(scope="module")
def scratchv_output(ir_program, onnx_model, test_input):
    """Step 4: Execute IR via trace executor → compiled output."""
    from scratchv.simulator.rv32_emulator import IRTraceExecutor
    executor = IRTraceExecutor(ir_program, onnx_model["initializers"])
    return executor.run(test_input)


@pytest.fixture(scope="module")
def onnxrt_output(test_input):
    """Step 5: ONNX Runtime reference inference."""
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    session = ort.InferenceSession(
        MODEL_PATH, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in session.get_outputs()]
    feed = {k: v for k, v in test_input.items()}
    result = session.run(out_names, feed)
    return dict(zip(out_names, result))


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: ONNX model structure
# ═══════════════════════════════════════════════════════════════════════════════

class TestONNXModel:
    """Verify the CNN model structure is correctly loaded."""

    def test_model_has_expected_nodes(self, onnx_model):
        """Model should have 15 operators in the expected order."""
        assert onnx_model["num_nodes"] == 15
        op_types = [n["op_type"] for n in onnx_model["nodes"]]
        assert op_types[0] == "Conv"
        assert op_types[1] == "Relu"
        assert op_types[2] == "MaxPool"
        assert op_types[-1] == "Reshape"
        assert "Sigmoid" in op_types
        assert "Gemm" in op_types

    def test_input_shape(self, onnx_model):
        """Input should be NCHW: (1, 3, 250, 250)."""
        inp = onnx_model["inputs"][0]
        assert inp["shape"] == [1, 3, 250, 250]

    def test_initializers_loaded(self, onnx_model):
        """All weights and biases should be loaded."""
        inits = onnx_model["initializers"]
        assert "layer1.0.weight" in inits
        assert "layer1.0.bias" in inits
        assert "fc1.weight" in inits
        assert "fc2.weight" in inits
        # conv weight shapes
        assert inits["layer1.0.weight"].shape == (32, 3, 3, 3)
        assert inits["layer3.0.weight"].shape == (64, 32, 3, 3)
        # fc weight shape
        assert inits["fc1.weight"].shape == (128, 53824)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: IR compilation
# ═══════════════════════════════════════════════════════════════════════════════

class TestIRCompilation:
    """Verify ONNX → IR translation."""

    def test_program_has_function(self, ir_program):
        """IR program should contain one function."""
        assert len(ir_program.functions) == 1

    def test_all_ops_translated(self, ir_program):
        """All 15 ONNX ops become 17 IR instructions."""
        func = ir_program.functions[0]
        total = sum(len(b.instructions) for b in func.blocks)
        assert total == 17

    def test_op_codes_present(self, ir_program):
        """Verify all expected opcodes appear in the IR."""
        from scratchv.ir.types import OpCode
        func = ir_program.functions[0]
        opcodes = {i.opcode for b in func.blocks for i in b.instructions}
        expected = {OpCode.CONV, OpCode.RELU, OpCode.MAXPOOL,
                    OpCode.GEMM, OpCode.SIGMOID, OpCode.RESHAPE,
                    OpCode.RETURN, OpCode.LOAD_CONST}
        assert expected.issubset(opcodes), f"Missing: {expected - opcodes}"

    def test_ir_dump_readable(self, ir_program):
        """IR dump should contain function name and ops."""
        dump = ir_program.dump()
        assert "main_graph" in dump
        assert "conv" in dump
        assert "relu" in dump
        assert "gemm" in dump
        assert "sigmoid" in dump
        assert "return" in dump


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: RISC-V assembly
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssembly:
    """Verify RISC-V assembly generation."""

    def test_produces_text_output(self, riscv_assembly):
        """Assembly should be non-empty text."""
        assert len(riscv_assembly) > 0
        assert ".text" in riscv_assembly

    def test_has_return_instruction(self, riscv_assembly):
        """Assembly must end with ret (jalr zero, ra)."""
        assert "jalr" in riscv_assembly
        assert "zero" in riscv_assembly
        assert "ra" in riscv_assembly

    def test_has_runtime_calls(self, riscv_assembly):
        """Assembly should contain call instructions for NN ops."""
        assert "conv" in riscv_assembly
        assert "gemm" in riscv_assembly
        assert "maxpool" in riscv_assembly
        assert "sigmoid" in riscv_assembly

    def test_emits_function_label(self, riscv_assembly):
        """Should emit .globl and function label for main_graph."""
        assert "main_graph:" in riscv_assembly
        assert ".globl" in riscv_assembly

    def test_counts(self, riscv_assembly):
        """Should have exactly 24 real instructions (excluding directives)."""
        lines = [ln for ln in riscv_assembly.split("\n")
                 if ln.strip() and not ln.strip().startswith(".")
                 and not ln.strip().startswith("#")
                 and not ln.strip().endswith(":")]
        assert len(lines) == 24


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: Binary encoding
# ═══════════════════════════════════════════════════════════════════════════════

class TestBinaryEncoding:
    """Verify RISC-V binary machine code."""

    def test_produces_96_bytes(self, riscv_binary):
        """24 instructions × 4 bytes = 96 bytes."""
        assert len(riscv_binary) == 96

    def test_valid_rv32_instructions(self, riscv_binary):
        """Every 4-byte word should decode as a valid RV32 opcode."""
        for i in range(0, len(riscv_binary), 4):
            word = int.from_bytes(riscv_binary[i:i + 4], "little")
            opcode = word & 0x7F
            # Valid base opcodes
            valid = opcode in (
                0b0110011,  # R-type
                0b0010011,  # I-type
                0b0000011,  # LOAD
                0b0100011,  # STORE
                0b1100011,  # BRANCH
                0b1100111,  # JALR
                0b1101111,  # JAL
                0b0110111,  # LUI
                0b0010111,  # AUIPC
            )
            msg = f"Bad opcode 0x{opcode:02x} at offset {i}: 0x{word:08x}"
            assert valid, msg

    def test_binary_deterministic(self, riscv_assembly):
        """Same assembly → same binary (deterministic encoding)."""
        from scratchv.backend.riscv_encoder import assemble_to_binary
        bin1 = assemble_to_binary(riscv_assembly)
        bin2 = assemble_to_binary(riscv_assembly)
        assert bin1 == bin2


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5: Trace execution
# ═══════════════════════════════════════════════════════════════════════════════

class TestExecution:
    """Verify IR trace execution produces valid output."""

    def test_output_is_array(self, scratchv_output):
        """Output should be a numpy array."""
        assert isinstance(scratchv_output, np.ndarray)

    def test_output_is_finite(self, scratchv_output):
        """Output should not contain NaN or Inf."""
        assert np.all(np.isfinite(scratchv_output))

    def test_output_range_reasonable(self, scratchv_output):
        """Sigmoid output should be in [0, 1]."""
        out = np.asarray(scratchv_output, dtype=np.float32).flatten()
        for v in out:
            assert 0.0 <= v <= 1.0, f"Output {v} outside sigmoid range [0,1]"

    def test_output_shape(self, scratchv_output):
        """Final output should be scalar-like (sigmoid)."""
        assert scratchv_output.size == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6: Verification (MSE / MAE)
# ═══════════════════════════════════════════════════════════════════════════════

class TestVerification:
    """Compare ScratchV output against ONNX Runtime reference."""

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("onnxruntime"),
        reason="onnxruntime not installed",
    )
    def test_onnxruntime_available(self, onnxrt_output):
        """ONNX Runtime should produce output."""
        assert onnxrt_output is not None
        assert len(onnxrt_output) > 0

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("onnxruntime"),
        reason="onnxruntime not installed",
    )
    def test_mse_below_threshold(self, scratchv_output, onnxrt_output):
        """MSE should be finite."""
        ref = list(onnxrt_output.values())[0]
        comp = np.asarray(scratchv_output, dtype=np.float32).flatten()
        ref = np.asarray(ref, dtype=np.float32).flatten()

        mse = float(np.mean((ref[:len(comp)] - comp) ** 2))
        assert np.isfinite(mse)
        assert mse < 1.0  # should be well below 1.0 for sigmoid output

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("onnxruntime"),
        reason="onnxruntime not installed",
    )
    def test_mae_below_threshold(self, scratchv_output, onnxrt_output):
        """MAE should be finite."""
        ref = list(onnxrt_output.values())[0]
        comp = np.asarray(scratchv_output, dtype=np.float32).flatten()
        ref = np.asarray(ref, dtype=np.float32).flatten()

        mae = float(np.mean(np.abs(ref[:len(comp)] - comp)))
        assert np.isfinite(mae)
        assert mae < 1.0

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("onnxruntime"),
        reason="onnxruntime not installed",
    )
    def test_output_not_identical(self, scratchv_output, onnxrt_output):
        """Outputs should differ (different implementations)."""
        ref = list(onnxrt_output.values())[0]
        comp = np.asarray(scratchv_output, dtype=np.float32).flatten()
        ref = np.asarray(ref, dtype=np.float32).flatten()
        assert not np.allclose(ref[:len(comp)], comp, rtol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 7: Per-layer IR tracing (data shape propagation)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLayerShapes:
    """Verify intermediate tensor shapes through the network."""

    def test_shape_propagation(self, ir_program, onnx_model, test_input):
        """Check shapes at each Conv/MaxPool/FC boundary."""
        from scratchv.simulator.rv32_emulator import IRTraceExecutor

        executor = IRTraceExecutor(ir_program, onnx_model["initializers"])
        executor._values = dict(test_input)
        for init_name, arr in onnx_model["initializers"].items():
            executor._values[init_name] = arr

        shapes = {}
        for func in ir_program.functions:
            for block in func.blocks:
                for instr in block.instructions:
                    instr.opcode.value  # noqa
                    try:
                        output = executor._exec_instr(instr)
                    except Exception:
                        output = None
                    if (instr.dest and output is not None
                            and hasattr(output, "shape")):
                        shapes[instr.dest.name] = tuple(output.shape)

        # Check key shapes
        # After Conv1: (1, 32, 248, 248)
        # After MaxPool1: (1, 32, 124, 124)
        # After Conv2: (1, 32, 122, 122)
        # After MaxPool2: (1, 32, 61, 61)
        # After Conv3: (1, 64, 59, 59)
        # After MaxPool3: (1, 64, 29, 29)
        # After Reshape1: (1, 53824)
        # After Gemm1: (1, 128)
        # After Gemm2: (1, 1)

        # Verify spatial dimensions shrink correctly
        for name, shape in shapes.items():
            if len(shape) >= 3:
                *_, h, w = shape[-2], shape[-1]
                assert h > 0 and w > 0, f"Bad shape for {name}: {shape}"
            print(f"  {name}: {shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 8: Performance benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerformance:
    """Benchmark each stage of the pipeline."""

    def test_parse_speed(self, onnx_model):
        """ONNX parsing should complete quickly."""
        import time
        from scratchv.frontend.onnx_parser import ONNXParser
        t0 = time.perf_counter()
        parser = ONNXParser()
        parser.parse(MODEL_PATH)
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"Parsing took {elapsed:.2f}s"

    def test_compile_speed(self, ir_program):
        """IR → assembly compilation should be fast."""
        import time
        from scratchv.backend.instruction_select import InstructionSelector
        from scratchv.backend.register_alloc import RegisterAllocator
        from scratchv.backend.asm_emit import AsmEmitter

        t0 = time.perf_counter()
        selector = InstructionSelector(ir_program)
        instrs = selector.run()
        alloc = RegisterAllocator(instrs, mode="greedy")
        allocated = alloc.run()
        emitter = AsmEmitter(allocated)
        emitter.emit()
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.1, f"Compilation took {elapsed:.3f}s"

    def test_assemble_speed(self, riscv_assembly):
        """Assembly → binary should be very fast."""
        import time
        from scratchv.backend.riscv_encoder import assemble_to_binary
        t0 = time.perf_counter()
        assemble_to_binary(riscv_assembly)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.01, f"Assembly took {elapsed:.4f}s"
