"""Tests for backend (instruction selection, reg alloc, assembly emission)."""

from scratchv.ir.builder import IRBuilder
from scratchv.backend.instruction_select import InstructionSelector
from scratchv.backend.register_alloc import RegisterAllocator, MachineOp
from scratchv.backend.asm_emit import AsmEmitter
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
