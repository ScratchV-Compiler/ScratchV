"""Topic21 I01–I10, A07/A08/A10 and coexistence with DSL diagnostics."""

import copy

import pytest

import scratchv.analysis.ir_verifier as verifier_module
from scratchv.compiler import CompilerConfig, CompilerDriver, PassManager, _PassAdapter
from scratchv.ir.types import Instruction, OpCode, Value
from scratchv.main import args_to_config, build_arg_parser, main
from scratchv.pass_interface import PassResult
from tests.test_ir_verifier import straight


def harness(monkeypatch, *, enabled=True, optimization="basic", p=None):
    events = []
    current = p if p is not None else straight()
    driver = CompilerDriver(CompilerConfig(verify_ir=enabled, optimize_level=optimization))
    monkeypatch.setattr(driver, "_parse", lambda *args: current)
    original = verifier_module.verify_ir

    def check(program, *, stage=None):
        events.append(stage)
        return original(program, stage=stage)

    def codegen(program):
        events.append("codegen")
        return "generated output\n"

    monkeypatch.setattr(verifier_module, "verify_ir", check)
    monkeypatch.setattr(driver, "_generate_code", codegen)
    return driver, events


@pytest.mark.parametrize("optimization,expected", [
    ("none", ["after-parse", "before-codegen", "codegen"]),
    ("basic", ["after-parse", "before:constant-folding", "after:constant-folding",
               "before:dead-code-elim", "after:dead-code-elim", "before-codegen", "codegen"]),
])
def test_i01_i02_exact_checkpoints(monkeypatch, tmp_path, optimization, expected):
    driver, events = harness(monkeypatch, optimization=optimization)
    target = tmp_path / "out.s"
    result = driver.compile("model.onnx", str(target))
    assert result.success, result.errors
    assert events == expected
    assert target.read_text() == "generated output\n"


@pytest.mark.parametrize("existing", [False, True])
def test_i03_i07_parse_error_stops_and_preserves_bytes(monkeypatch, tmp_path, existing):
    p = straight()
    p.functions[0].params = []
    driver, events = harness(monkeypatch, p=p)
    target = tmp_path / "out.s"
    if existing:
        target.write_bytes(b"\x00original\xff")
    result = driver.compile("model.onnx", str(target))
    assert not result.success
    assert events == ["after-parse"]
    assert result.ir_diagnostics and not result.diagnostics
    assert "stage=after-parse" in result.errors[0]
    if existing:
        assert target.read_bytes() == b"\x00original\xff"
    else:
        assert not target.exists()


@pytest.mark.parametrize("mutation,rule", [("delete", "def-before-use"),
                                           ("duplicate", "ssa-validity"),
                                           ("target", "label-existence")])
def test_i04_i05_a07_replacement_pass_failure(monkeypatch, tmp_path, mutation, rule):
    original = straight()
    driver, events = harness(monkeypatch, p=original)

    def run(pass_, data):
        events.append("run:" + pass_.name)
        replacement = copy.deepcopy(data)
        block = replacement.functions[0].blocks[0]
        if mutation == "delete":
            block.instructions.pop(0)
        elif mutation == "duplicate":
            block.instructions.insert(1, copy.deepcopy(block.instructions[0]))
        else:
            block.instructions[-1] = Instruction(OpCode.BR, target="missing")
        return PassResult(replacement)

    monkeypatch.setattr(_PassAdapter, "run", run)
    target = tmp_path / "out.s"
    target.write_bytes(b"original")
    result = driver.compile("model.onnx", str(target))
    assert not result.success
    assert events == ["after-parse", "before:constant-folding", "run:constant-folding", "after:constant-folding"]
    assert any(i.rule == rule and i.stage == "after:constant-folding" for i in result.ir_diagnostics)
    assert len(original.functions[0].blocks[0].instructions) == 3
    assert target.read_bytes() == b"original"


@pytest.mark.parametrize("bad_result", [None, PassResult(None), PassResult("assembly")])
@pytest.mark.parametrize("enabled", [False, True])
def test_a08_pass_contract_stops_before_backend(monkeypatch, tmp_path, bad_result, enabled):
    driver, events = harness(monkeypatch, enabled=enabled)
    monkeypatch.setattr(_PassAdapter, "run", lambda *args: bad_result)
    result = driver.compile("model.onnx", str(tmp_path / "out.s"))
    assert not result.success
    assert "codegen" not in events
    assert not any(str(e).startswith("after:constant") for e in events)
    assert not (tmp_path / "out.s").exists()


def test_a07_next_adapter_and_codegen_receive_replacement(monkeypatch, tmp_path):
    driver, events = harness(monkeypatch)
    original_run = _PassAdapter.run
    replacement = straight()
    replacement.functions[0].name = "replacement"

    def run(pass_, data):
        if pass_.name == "constant-folding":
            return PassResult(replacement)
        assert data is replacement
        return original_run(pass_, data)

    class DeadCode:
        def __init__(self, program):
            assert program is replacement

        def run(self):
            return 0

    monkeypatch.setattr(_PassAdapter, "run", run)
    monkeypatch.setattr("scratchv.optimizer.dead_code.DeadCodeEliminator", DeadCode)
    monkeypatch.setattr(driver, "_generate_code", lambda p: p.functions[0].name)
    result = driver.compile("model.onnx", str(tmp_path / "out.s"))
    assert result.success and result.output_text == "replacement"


def test_i06_warning_only_continues(monkeypatch, tmp_path):
    p = straight()
    p.functions[0].new_block("dead").add(Instruction(OpCode.RETURN))
    driver, events = harness(monkeypatch, p=p, optimization="none")
    result = driver.compile("model.onnx", str(tmp_path / "out.s"))
    assert result.success and not result.errors
    assert len(result.warnings) == 2
    assert [i.stage for i in result.ir_diagnostics] == ["after-parse", "before-codegen"]
    assert events[-1] == "codegen"


@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize("verify_ir", [False, True])
@pytest.mark.parametrize("logging", [False, True])
def test_i09_a10_flags_independent(monkeypatch, tmp_path, verify, verify_ir, logging):
    driver, events = harness(monkeypatch, enabled=verify_ir)
    driver.config.verify = verify
    driver.config.use_logger = logging
    driver.config.log_level = "DEBUG"
    assert driver.compile("model.onnx", str(tmp_path / "out.s")).success
    assert len(events) == (7 if verify_ir else 1)
    args = build_arg_parser().parse_args(["test.onnx"] + (["--verify"] if verify else []) + (["--verify-ir"] if verify_ir else []))
    config = args_to_config(args)
    assert (config.verify, config.verify_ir) == (verify, verify_ir)


def test_i08_cli_ir_error_once_with_stage(monkeypatch, tmp_path, capsys):
    p = straight()
    p.functions[0].params = []
    monkeypatch.setattr(CompilerDriver, "_parse", lambda *args: p)
    target = tmp_path / "out.s"
    assert main(["test.onnx", "--verify-ir", "-o", str(target)]) == 1
    stderr = capsys.readouterr().err
    assert "[def-before-use]" in stderr
    assert "stage=after-parse function=main block=entry instruction=0" in stderr
    assert "\033[" not in stderr and not target.exists()


@pytest.mark.parametrize("source", ["x = add(1, 2)\nreturn x\n",
                                    "for i = 0, 3\nx = add(1, 2)\nendfor\nreturn 1\n"])
def test_i10_real_dsl_legal_shared_ir(monkeypatch, tmp_path, source):
    seen = []
    driver = CompilerDriver(CompilerConfig(verify_ir=True))
    monkeypatch.setattr(driver, "_generate_code", lambda p: seen.append(p) or "code")
    result = driver.compile("demo.dsl", str(tmp_path / "out.s"), dsl_source=source)
    assert result.success, result.errors
    if source.startswith("for"):
        assert any(i.opcode == OpCode.FOR for i in seen[0].functions[0].blocks[0].instructions)


def test_i10_real_onnx_registered_inputs(monkeypatch, tmp_path):
    import onnx
    from onnx import TensorProto, helper
    model = helper.make_model(helper.make_graph(
        [helper.make_node("Add", ["a", "b"], ["y"])], "add",
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1]) for n in ("a", "b")],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    ))
    source = tmp_path / "add.onnx"
    onnx.save(model, source)
    driver = CompilerDriver(CompilerConfig(verify_ir=True))
    monkeypatch.setattr(driver, "_generate_code", lambda p: "code")
    result = driver.compile(str(source), str(tmp_path / "out.s"))
    assert result.success, result.errors
    assert result.ir_diagnostics == []


def test_dsl_syntax_recovery_stops_before_ir(monkeypatch, tmp_path, capsys):
    source = tmp_path / "bad.dsl"
    source.write_text("retrun x\ny = ad(a, b)\nz = relu(a, b)\n")
    target = tmp_path / "out.s"
    target.write_bytes(b"original")

    def forbidden(*args, **kwargs):
        pytest.fail("syntax errors must stop before IR construction/verification")

    monkeypatch.setattr(CompilerDriver, "_parse", forbidden)
    monkeypatch.setattr(verifier_module, "verify_ir", forbidden)
    result = CompilerDriver(CompilerConfig(verify_ir=True)).compile(str(source), str(target))
    assert len(result.diagnostics) == 3 and not result.ir_diagnostics
    assert [d.line for d in result.diagnostics] == [1, 2, 3]
    assert main([str(source), "--verify-ir", "-o", str(target)]) == 1
    stderr = capsys.readouterr().err
    assert stderr.count(": error[") == 3
    assert "did you mean" in stderr and "\033[" not in stderr
    assert target.read_bytes() == b"original"


def test_dsl_implicit_inputs_strict_only_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(CompilerDriver, "_generate_code", lambda *args: "code")
    source = "c = add(a, b)\nreturn c\n"
    for enabled in (False, True):
        driver = CompilerDriver(CompilerConfig(verify_ir=enabled))
        result = driver.compile("demo.dsl", str(tmp_path / "out.s"), dsl_source=source)
        assert result.success is not enabled
        assert not result.diagnostics
        if enabled:
            assert {i.value_name for i in result.ir_diagnostics} == {"a", "b"}


def test_generic_pass_manager_has_no_ir_dependency():
    class TextPass:
        name = "text"

        def run(self, text):
            return PassResult(text + "!")

    manager = PassManager().add(TextPass())
    assert manager.run("text").data == "text!"


@pytest.mark.parametrize("source,missing", [
    ("if (a > b):\nc = add(a, b)\nelse:\nc = mul(a, b)\nendif\nreturn c\n", {"a", "b"}),
    ("while (i < 10):\nacc = add(acc, x)\nendwhile\nreturn acc\n", {"i", "acc", "x"}),
])
def test_extended_dsl_keeps_real_external_input_diagnostics(source, missing):
    from scratchv.frontend.dsl_extended import ExtendedDSLParser
    program = ExtendedDSLParser().parse(source)
    passed, diagnostics = verifier_module.verify_ir(program)
    assert not passed
    assert {d.rule for d in diagnostics} == {"def-before-use"}
    assert {d.value_name for d in diagnostics} == missing


def test_dsl_error_limit_still_reported_with_ir_enabled(tmp_path, capsys):
    source = tmp_path / "many.dsl"
    source.write_text("\n".join(f"bad statement {i}" for i in range(25)))
    target = tmp_path / "out.s"
    assert main([str(source), "--verify-ir", "-o", str(target)]) == 1
    stderr = capsys.readouterr().err
    assert stderr.count(": error[") == 20
    assert stderr.count("further errors suppressed") == 1
    assert not target.exists()


def test_driver_reuse_clears_prior_ir_diagnostics(monkeypatch, tmp_path):
    p = straight()
    p.functions[0].params = []
    driver, events = harness(monkeypatch, p=p, optimization="none")
    assert not driver.compile("test.onnx", str(tmp_path / "out.s")).success
    p.functions[0].params = [Value("a")]
    result = driver.compile("test.onnx", str(tmp_path / "out.s"))
    assert result.success and result.ir_diagnostics == [] and result.warnings == []
