"""Tests for backend (instruction selection, reg alloc, assembly emission)."""

import re

from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType
from scratchv.backend.instruction_select import InstructionSelector
from scratchv.backend.register_alloc import RegisterAllocator, MachineOp
from scratchv.backend.asm_emit import AsmEmitter
from scratchv.backend.riscv_encoder import assemble_to_binary
from scratchv.frontend.dsl_parser import DSLParser
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.main import args_to_config, build_arg_parser


class TestInstructionSelect:
    def test_select_add(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a")
        b = builder.make_value(name="b")
        c = builder.add(a, b)
        builder.ret(c)

        selector = InstructionSelector(builder.program)
        instrs = selector.run()
        # Expect: label, add, mv a0, ret
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.ADD in ops

    def test_select_relu(self):
        dsl = "y = relu(x)\nreturn y"
        parser = DSLParser()
        program = parser.parse(dsl)

        selector = InstructionSelector(program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.MAX in ops  # relu → max


class TestRegisterAlloc:
    def test_alloc_greedy(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a")
        b = builder.make_value(name="b")
        c = builder.add(a, b)
        builder.ret(c)

        selector = InstructionSelector(builder.program)
        instrs = selector.run()

        alloc = RegisterAllocator(instrs, mode="greedy")
        result = alloc.run()

        # Should produce valid instructions
        assert len(result) > 0
        # All vregs should be resolved
        for instr in result:
            for op in (instr.dst, instr.src1, instr.src2):
                if op is not None:
                    if hasattr(op, 'kind'):
                        assert op.kind != 'vreg', f"Unresolved vreg in {instr}"

    def test_alloc_naive(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a")
        b = builder.make_value(name="b")
        c = builder.add(a, b)
        builder.ret(c)

        selector = InstructionSelector(builder.program)
        instrs = selector.run()

        alloc = RegisterAllocator(instrs, mode="naive")
        result = alloc.run()
        assert len(result) > 0


class TestAsmEmitter:
    def test_emit_assembly(self):
        dsl = "y = add(a, b)\nreturn y"
        parser = DSLParser()
        program = parser.parse(dsl)

        selector = InstructionSelector(program)
        instrs = selector.run()
        alloc = RegisterAllocator(instrs, mode="greedy")
        allocated = alloc.run()

        emitter = AsmEmitter(allocated)
        asm = emitter.emit()

        assert ".text" in asm
        assert "main:" in asm
        assert "add" in asm

    def test_emit_relu(self):
        dsl = "y = relu(x)\nreturn y"
        parser = DSLParser()
        program = parser.parse(dsl)

        selector = InstructionSelector(program)
        instrs = selector.run()
        alloc = RegisterAllocator(instrs, mode="greedy")
        allocated = alloc.run()

        emitter = AsmEmitter(allocated)
        asm = emitter.emit()

        assert "max" in asm


class TestConstMergeIntegration:
    def test_main_cli_flag_enables_compiler_config(self):
        args = build_arg_parser().parse_args(["input.dsl", "--const-merge"])
        config = args_to_config(args)
        assert config.const_merge is True

    def test_compiler_driver_runs_const_merge_post_pass(self):
        driver = CompilerDriver(CompilerConfig(const_merge=True))
        warnings: list[str] = []
        result = driver._run_asm_passes(
            "  lui t0, 1\n  addi t0, t0, 2\n",
            warnings,
        )
        assert "li t0, 4098" in result
        assert len(warnings) == 1
        assert "1 changes" in warnings[0]
        assert "1 pairs" in warnings[0]

    def test_disabled_config_leaves_assembly_unchanged(self):
        driver = CompilerDriver(CompilerConfig(const_merge=False))
        source = "  lui t0, 1\n  addi t0, t0, 2\n"
        warnings: list[str] = []
        result = driver._run_asm_passes(source, warnings)
        assert result == source
        assert warnings == []


# ── Topic 29: vector CLI wiring ─────────────────────────────────────────

class TestVectorCLIWiring:
    def test_vector_flags_parse_and_convert(self):
        args = build_arg_parser().parse_args([
            "m.onnx", "--vectorize", "--vector-width", "2",
            "--vector-isa", "scalar",
        ])
        config = args_to_config(args)
        assert config.vectorize is True
        assert config.vector_width == 2
        assert config.vector_isa == "scalar"

    def test_vector_defaults(self):
        args = build_arg_parser().parse_args(["m.onnx"])
        config = args_to_config(args)
        assert config.vectorize is False
        assert config.vector_width == 4
        assert config.vector_isa == "scalar"

    def test_vector_isa_v_parses(self):
        args = build_arg_parser().parse_args(["m.onnx", "--vector-isa", "v"])
        assert args_to_config(args).vector_isa == "v"


# ── Topic 29: vector driver integration ─────────────────────────────────

def _vector_ir(n: int = 16):
    b = IRBuilder()
    b.new_function("main")
    b.new_block("entry")
    a = b.load_const(0x400000, dtype=DataType.INT32)
    bb = b.load_const(0x410000, dtype=DataType.INT32)
    o = b.load_const(0x420000, dtype=DataType.INT32)
    iv = b.for_loop(0, n)
    c4 = b.load_const(4, dtype=DataType.INT32)
    off = b.mul(iv, c4)
    va = b.load(b.add(a, off))
    vb = b.load(b.add(bb, off))
    r = b.relu(b.mul(va, vb))
    b.store(b.add(o, off), r)
    b.endfor()
    b.ret()
    return b.program


class TestVectorDriverIntegration:
    def test_compile_vectorized_program(self, monkeypatch, tmp_path):
        driver = CompilerDriver(CompilerConfig(
            vectorize=True, vector_width=2, reg_alloc="greedy"))
        program = _vector_ir()
        monkeypatch.setattr(driver, "_parse", lambda *a, **k: program)
        source = tmp_path / "input.dsl"
        source.write_text("x = add(a, b)\n", encoding="utf-8")

        result = driver.compile(str(source), str(tmp_path / "out.s"))

        assert result.success
        assert "vectorized 1/1 loop(s), width=2" in result.stats["opt_message"]
        assert not re.search(r"^\s*v[a-z]", result.output_text, re.MULTILINE)
        assert "vload" not in result.output_text
        assert "vstore" not in result.output_text

    def test_vector_isa_v_rejected(self, monkeypatch, tmp_path):
        driver = CompilerDriver(CompilerConfig(
            vectorize=True, vector_isa="v", reg_alloc="greedy"))
        monkeypatch.setattr(driver, "_parse", lambda *a, **k: _vector_ir())
        source = tmp_path / "input.dsl"
        source.write_text("x = add(a, b)\n", encoding="utf-8")

        result = driver.compile(str(source), str(tmp_path / "out.s"))

        assert result.success is False
        assert any("phase 2" in err for err in result.errors)

    def test_vectorize_disabled_matches_baseline(self, monkeypatch, tmp_path):
        config = CompilerConfig(vectorize=False, reg_alloc="greedy")
        driver = CompilerDriver(config)
        monkeypatch.setattr(driver, "_parse", lambda *a, **k: _vector_ir())
        source = tmp_path / "input.dsl"
        source.write_text("x = add(a, b)\n", encoding="utf-8")
        result = driver.compile(str(source), str(tmp_path / "out.s"))

        assert result.success
        assert result.stats["opt_message"] == ""
        assert not re.search(r"^\s*v[a-z]", result.output_text, re.MULTILINE)

    def test_vectorize_llvm_backend_rejected(self, monkeypatch, tmp_path):
        """Review F5: the LLVM backend used to drop vector ops silently."""
        driver = CompilerDriver(CompilerConfig(
            vectorize=True, backend="llvm", reg_alloc="greedy"))
        monkeypatch.setattr(driver, "_parse", lambda *a, **k: _vector_ir())
        source = tmp_path / "input.dsl"
        source.write_text("x = add(a, b)\n", encoding="utf-8")

        result = driver.compile(str(source), str(tmp_path / "out.ll"))

        assert result.success is False
        assert any("llvm" in err.lower() for err in result.errors)
        assert not (tmp_path / "out.ll").exists()

    def test_vectorize_invalid_width_rejected(self, monkeypatch, tmp_path):
        """Review F6: vector_width=0 used to raise ZeroDivisionError."""
        driver = CompilerDriver(CompilerConfig(
            vectorize=True, vector_width=0, reg_alloc="greedy"))
        monkeypatch.setattr(driver, "_parse", lambda *a, **k: _vector_ir())
        source = tmp_path / "input.dsl"
        source.write_text("x = add(a, b)\n", encoding="utf-8")

        result = driver.compile(str(source), str(tmp_path / "out.s"))

        assert result.success is False
        assert any("width" in err for err in result.errors)

    def test_vectorize_linear_regalloc_falls_back_to_greedy(
            self, monkeypatch, tmp_path):
        """Review F7: the default linear allocator cannot emit labels."""
        driver = CompilerDriver(CompilerConfig(
            vectorize=True, vector_width=2))  # reg_alloc defaults to linear
        monkeypatch.setattr(driver, "_parse", lambda *a, **k: _vector_ir())
        source = tmp_path / "input.dsl"
        source.write_text("x = add(a, b)\n", encoding="utf-8")

        result = driver.compile(str(source), str(tmp_path / "out.s"))

        assert result.success
        assert driver.config.reg_alloc == "greedy"
        assert any("greedy" in warning for warning in result.warnings)
        assert ".label" not in result.output_text
        assert len(assemble_to_binary(result.output_text)) > 0


# ── Topic 29 P0: constant operands in R-type instructions ───────────────

class TestConstantOperandMaterialization:
    def test_no_rtype_immediate(self):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        x = b.make_value(name="x", dtype=DataType.INT32)
        c = b.load_const(4, dtype=DataType.INT32)
        y = b.mul(x, c)
        z = b.add(y, c)
        w = b.sub(z, c)
        v = b.div(w, c)
        b.ret(v)

        machine = InstructionSelector(b.program).run()
        allocated = RegisterAllocator(machine, mode="greedy").run()
        asm = AsmEmitter(allocated).emit()

        assert "li" in asm
        assert not re.search(
            r"^\s*(add|sub|mul|div)\s+\w+,\s*\w+,\s*-?\d+\s*$",
            asm, re.MULTILINE)
