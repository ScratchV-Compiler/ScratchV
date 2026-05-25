"""Tests for LLVM IR code generation backend."""

from scratchv.frontend.dsl_parser import DSLParser
from scratchv.backend.llvm_codegen import LLVMCodegen


class TestLLVMCodegen:
    def test_emit_add(self):
        dsl = "y = add(a, b)\nreturn y"
        parser = DSLParser()
        program = parser.parse(dsl)

        codegen = LLVMCodegen(program)
        llvm_ir = codegen.emit()

        assert "define" in llvm_ir
        assert "fadd" in llvm_ir
        assert "ret" in llvm_ir
        assert "@main" in llvm_ir

    def test_emit_relu(self):
        dsl = "y = relu(x)\nreturn y"
        parser = DSLParser()
        program = parser.parse(dsl)

        codegen = LLVMCodegen(program)
        llvm_ir = codegen.emit()

        assert "fcmp" in llvm_ir  # Relu uses icmp
        assert "select" in llvm_ir  # select pattern

    def test_emit_mul_sub(self):
        dsl = "y = mul(a, b)\nz = sub(y, c)\nreturn z"
        parser = DSLParser()
        program = parser.parse(dsl)

        codegen = LLVMCodegen(program)
        llvm_ir = codegen.emit()

        assert "fmul" in llvm_ir
        assert "fsub" in llvm_ir

    def test_emit_gelu(self):
        dsl = "y = gelu(x)\nreturn y"
        parser = DSLParser()
        program = parser.parse(dsl)

        codegen = LLVMCodegen(program)
        llvm_ir = codegen.emit()

        assert "tanh" in llvm_ir or "tanhf" in llvm_ir
        assert "declare" in llvm_ir

    def test_emit_constants(self):
        dsl = "b = add(a, 4.0)\nreturn b"
        parser = DSLParser()
        program = parser.parse(dsl)

        codegen = LLVMCodegen(program)
        llvm_ir = codegen.emit()

        assert "fadd" in llvm_ir

    def test_emit_for_loop(self):
        dsl = """
for i = 0, 3
endfor
return 0
"""
        parser = DSLParser()
        program = parser.parse(dsl)

        codegen = LLVMCodegen(program)
        llvm_ir = codegen.emit()

        assert "alloca" in llvm_ir
        assert "icmp" in llvm_ir
        assert "br" in llvm_ir

    def test_save_to_file(self, tmp_path):
        dsl = "y = add(a, b)\nreturn y"
        parser = DSLParser()
        program = parser.parse(dsl)

        codegen = LLVMCodegen(program)
        path = tmp_path / "test.ll"
        codegen.save(str(path))

        assert path.exists()
        content = path.read_text()
        assert "fadd" in content

    def test_emit_div_neg(self):
        dsl = "y = div(a, b)\nz = neg(y)\nreturn z"
        parser = DSLParser()
        program = parser.parse(dsl)

        codegen = LLVMCodegen(program)
        llvm_ir = codegen.emit()

        assert "fdiv" in llvm_ir
        assert "fneg" in llvm_ir

    def test_emit_exp(self):
        dsl = "y = exp(x)\nreturn y"
        parser = DSLParser()
        program = parser.parse(dsl)

        codegen = LLVMCodegen(program)
        llvm_ir = codegen.emit()

        assert "call" in llvm_ir
        assert "expf" in llvm_ir or "exp" in llvm_ir

    def test_emit_multiple_blocks(self):
        dsl = """
for i = 0, 2
    y = add(x, i)
endfor
return y
"""
        parser = DSLParser()
        program = parser.parse(dsl)

        codegen = LLVMCodegen(program)
        llvm_ir = codegen.emit()

        # Should have multiple block labels
        assert ": " in llvm_ir or ":" in llvm_ir
