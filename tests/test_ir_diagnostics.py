"""Readable IR reports preserve exact locations and stage-time context."""

import io

import pytest

from scratchv.analysis import format_ir_error, render_ir_error, verify_ir
from scratchv.analysis.ir_verifier import ErrorLevel, VerificationError
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.ir.types import BasicBlock, DataType, Function, Instruction, OpCode, Program, Value
from scratchv.main import main


def make_example(case):
    """Small IR fixtures independent of optional developer scripts."""
    program = Program()
    function = Function("main")
    program.add_function(function)
    entry = function.new_block("entry")
    one = Value("one", is_constant=True, const_value=1.0)
    if case == "undefined":
        entry.add(Instruction(OpCode.RETURN, operands=[Value("missing")]))
    elif case == "use-before-def":
        entry.add(Instruction(OpCode.NEG, Value("y"), [Value("x")]))
        entry.add(Instruction(OpCode.ADD, Value("x"), [one, one]))
        entry.add(Instruction(OpCode.RETURN, operands=[Value("y")]))
    elif case == "branch":
        condition = Value("condition", DataType.INT32)
        function.params = [condition]
        entry.add(Instruction(OpCode.BR_IF, operands=[condition], target="left,right"))
        left = function.new_block("left")
        left.add(Instruction(OpCode.NEG, Value("x"), [one]))
        left.add(Instruction(OpCode.BR, target="merge"))
        function.new_block("right").add(Instruction(OpCode.BR, target="merge"))
        function.new_block("merge").add(Instruction(OpCode.RETURN, operands=[Value("x")]))
    elif case == "unreachable":
        entry.add(Instruction(OpCode.RETURN))
        function.new_block("dead").add(Instruction(OpCode.RETURN))
    elif case == "valid":
        entry.add(Instruction(OpCode.NEG, Value("x"), [one]))
        entry.add(Instruction(OpCode.RETURN, operands=[Value("x")]))
    else:
        raise ValueError(case)
    return program


def diagnostic(case):
    return verify_ir(make_example(case), stage="after-parse")[1][0]


def test_context_and_marker_identify_operand_not_destination():
    issue = diagnostic("use-before-def")
    rendered = format_ir_error(issue)
    assert "--> 0 | $y: f32 = neg $x: f32" in rendered
    assert "1 | $x: f32 = add" in rendered
    assert "2 | return $y: f32" in rendered
    marked = next(line for line in issue.context if line.marked)
    assert marked.text[marked.column:marked.column + marked.length] == "$x"
    lines = rendered.splitlines()
    line_index = next(i for i, line in enumerate(lines) if "--> 0" in line)
    assert lines[line_index + 1].index("^") == lines[line_index].index("$x")
    assert "note: Move the definition before this use" in rendered
    assert "stage=after-parse" in rendered and "\033[" not in rendered
    assert "note:" not in str(issue)  # Compact API remains compatible.


def test_context_is_bounded_and_preserves_original_indices():
    program = make_example("valid")
    block = program.functions[0].blocks[0]
    block.instructions = [Instruction(OpCode.ALLOCA, Value(f"x{i}")) for i in range(12)]
    block.instructions[6] = Instruction(OpCode.NEG, Value("bad"), [Value("missing")])
    block.add(Instruction(OpCode.RETURN))
    issue = verify_ir(program)[1][0]
    indices = [line.label for line in issue.context if line.label.isdigit()]
    assert indices == ["4", "5", "6", "7", "8"]
    assert sum(line.label == "..." for line in issue.context) == 2
    assert issue.instruction_index == 6


def test_context_survives_later_ir_mutation():
    program = make_example("branch")
    issue = verify_ir(program)[1][0]
    original = format_ir_error(issue)
    program.functions[0].blocks[-1].instructions.clear()
    program.functions[0].name = "changed"
    assert format_ir_error(issue) == original
    assert "block=merge instruction=0" in original


def test_duplicate_block_names_use_ordinals_for_context():
    program = make_example("undefined")
    block = BasicBlock("entry")
    block.add(Instruction(OpCode.NEG, Value("other"), [Value("missing")]))
    block.add(Instruction(OpCode.RETURN))
    program.functions[0].blocks.append(block)
    issues = [i for i in verify_ir(program)[1] if i.rule == "def-before-use"]
    assert len(issues) == 2
    assert "block entry (#0)" in format_ir_error(issues[0])
    assert "return $missing" in format_ir_error(issues[0])
    assert "block entry (#1)" in format_ir_error(issues[1])
    assert "$other: f32 = neg" in format_ir_error(issues[1])


@pytest.mark.parametrize("kind", ["global", "param", "empty", "no-entry", "returns"])
def test_non_instruction_diagnostics_have_honest_context(kind):
    program = Program()
    function = Function("main")
    program.add_function(function)
    if kind == "global":
        program.global_values = [Value("weight", "f32")]
    elif kind == "param":
        function.params = [Value("input", "f32")]
    elif kind == "returns":
        function.returns = [Value("a"), Value("b")]
    if kind != "no-entry":
        block = function.new_block("entry")
        if kind != "empty":
            block.add(Instruction(OpCode.RETURN))
    issue = verify_ir(program)[1][0]
    rendered = format_ir_error(issue)
    assert issue.instruction_index is None
    assert "instruction=" not in rendered
    assert "-->" in rendered and "note:" in rendered
    expected = {"global": "global #0", "param": "param #0", "empty": "<empty block>",
                "no-entry": "<no basic blocks>", "returns": "return signature:"}[kind]
    assert expected in rendered


def test_bad_opcode_and_dtype_still_render():
    program = make_example("valid")
    instruction = program.functions[0].blocks[0].instructions[0]
    instruction.opcode = "unknown"
    instruction.dest.dtype = "f32"
    issues = verify_ir(program)[1]
    assert any(i.rule == "type-consistency" for i in issues)
    assert all("note:" in format_ir_error(i) for i in issues)


@pytest.mark.parametrize("tty,no_color,escape", [(True, False, True), (False, False, False), (True, True, False)])
def test_color_follows_stream_and_no_color(monkeypatch, tty, no_color, escape):
    class Stream(io.StringIO):
        def isatty(self):
            return tty
    monkeypatch.delenv("NO_COLOR", raising=False)
    if no_color:
        monkeypatch.setenv("NO_COLOR", "1")
    result = render_ir_error(diagnostic("undefined"), stream=Stream())
    assert ("\033[" in result) is escape


def test_old_diagnostics_without_context_still_render():
    issue = VerificationError(ErrorLevel.ERROR, "bad IR", rule="type-consistency")
    assert format_ir_error(issue) == str(issue)


def test_terminal_control_characters_are_escaped():
    program = make_example("undefined")
    program.functions[0].blocks[0].instructions[0].operands[0].name = "bad\n\033[31m"
    output = format_ir_error(verify_ir(program)[1][0])
    assert "\033" not in output
    assert r"\n" in output


def test_cli_renders_ir_error_once_and_preserves_compact_api(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(CompilerDriver, "_parse", lambda *args: make_example("undefined"))
    target = tmp_path / "out.s"
    assert main(["bad.onnx", "--verify-ir", "-o", str(target)]) == 1
    output = capsys.readouterr().err
    assert output.count("[ERROR][def-before-use]") == 1
    assert "--> 0 | return $missing: f32" in output
    assert "note: Register external values" in output
    assert not target.exists()
    result = CompilerDriver(CompilerConfig(verify_ir=True)).compile("bad.onnx", str(target))
    assert result.errors == [str(result.ir_diagnostics[0])]
    assert result.ir_diagnostics[0].context


def test_warning_context_retains_stage_and_cli_does_not_duplicate(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(CompilerDriver, "_parse", lambda *args: make_example("unreachable"))
    monkeypatch.setattr(CompilerDriver, "_generate_code", lambda *args: "generated")
    assert main(["warning.onnx", "--verify-ir", "-o", str(tmp_path / "out.s")]) == 0
    output = capsys.readouterr().err
    assert output.count("[WARNING][control-flow-integrity]") == 2  # Two checkpoints.
    assert "stage=after-parse" in output and "stage=before-codegen" in output
    assert "block dead (#1)" in output
    assert output.count("note: Check incoming branches") == 2


def test_unicode_names_use_display_width_for_caret():
    from scratchv.analysis.ir_diagnostics import _display_width
    program = make_example("undefined")
    program.functions[0].blocks[0].instructions[0].operands[0].name = "未定义"
    output = format_ir_error(verify_ir(program)[1][0]).splitlines()
    index = next(i for i, line in enumerate(output) if "--> 0" in line)
    marker = output[index + 1]
    assert marker.index("^") == _display_width(output[index].split("$", 1)[0])
    assert "^~~~~~~" in marker  # '$' plus three double-width characters.
