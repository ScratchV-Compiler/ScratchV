"""Topic21 U01–U45 and A01–A18: direct shared-IR contract tests."""

import copy
import math

import pytest

from scratchv.analysis.ir_verifier import (
    ErrorLevel, IRVerifier, OPCODE_SPECS, VerificationError, verify_ir,
)
from scratchv.ir.types import BasicBlock, DataType as D, Function, Instruction as I, OpCode as O, Program, Value as V


def program(*instructions, params=(), returns=()):
    p = Program()
    f = Function("main", params=list(params), returns=list(returns))
    p.add_function(f)
    b = f.new_block("entry")
    b.instructions = list(instructions)
    return p


def literal(value=1, dtype=D.FLOAT32, name="literal"):
    return V(name, dtype, True, value)


def straight():
    a, x, y = V("a"), V("x"), V("y")
    return program(I(O.ADD, x, [a, literal()]), I(O.MUL, y, [x, a]),
                   I(O.RETURN, operands=[y]), params=[a])


def loop(nested=False):
    instructions = [I(O.FOR, V("i", D.INT32), attrs=dict(start=0, end=3, step=1))]
    if nested:
        instructions.append(I(O.FOR, V("j", D.INT32), attrs=dict(start=0, end=2, step=1)))
    instructions += [I(O.ADD, V("x", D.INT32), [V("i", D.INT32), literal(1, D.INT32)]), I(O.ENDFOR)]
    if nested:
        instructions.append(I(O.ENDFOR))
    return program(*instructions, I(O.RETURN))


def diamond():
    a, slot, x, y, result = [V(n) for n in ("a", "slot", "x", "y", "result")]
    p = program(I(O.ALLOCA, slot), I(O.BR_IF, operands=[a, a], attrs={"cmp_op": ">="},
                                    target="left,right"), params=[a])
    f = p.functions[0]
    for name, instructions in (
        ("left", [I(O.SUB, x, [a, a]), I(O.STORE, operands=[slot, x]), I(O.BR, target="merge")]),
        ("right", [I(O.SUB, y, [a, a]), I(O.STORE, operands=[slot, y]), I(O.BR, target="merge")]),
        ("merge", [I(O.LOAD, result, [slot]), I(O.RETURN, operands=[result])]),
    ):
        f.new_block(name).instructions = instructions
    return p


def issues(p, rule, passed=False):
    ok, found = verify_ir(p, stage="test")
    assert ok is passed, found
    selected = [i for i in found if i.rule == rule]
    assert selected, found
    assert all(i.stage == "test" for i in found)
    return selected


def test_u01_u14_u17_u37_u40_u41_legal_inputs():
    p = straight()
    p.global_values.append(V("global"))
    p.functions[0].blocks[0].name = "start"
    p.functions.append(copy.deepcopy(p.functions[0]))
    p.functions[1].name = "other"
    assert verify_ir(p) == (True, [])
    assert verify_ir(Program()) == (True, [])
    assert verify_ir(program(I(O.RETURN))) == (True, [])


@pytest.mark.parametrize("variant", ["undefined", "forward", "self", "constant"])
def test_u02_u03_u04_u08_definition_order(variant):
    p = straight()
    b = p.functions[0].blocks[0]
    if variant == "undefined":
        p.functions[0].params = []
    elif variant == "forward":
        b.instructions[0], b.instructions[1] = b.instructions[1], b.instructions[0]
    elif variant == "self":
        b.instructions[0].operands[0] = V("x")
    else:
        x = literal(1, name="x")
        b.instructions = [I(O.NEG, V("y"), [x]), I(O.LOAD_CONST, x, attrs={"value": 1}), I(O.RETURN)]
    found = issues(p, "def-before-use")
    assert found[0].instruction_index == 0
    if variant != "undefined":
        assert len(found) == 1
        assert found[0].value_name == "x"


def test_u05_u06_u07_u24_u38_diamond_memory_and_dominance():
    p = diamond()
    assert verify_ir(p) == (True, [])
    p.functions[0].blocks[1:3] = reversed(p.functions[0].blocks[1:3])
    assert verify_ir(p) == (True, [])
    merge = p.functions[0].blocks[-1]
    merge.instructions[0] = I(O.ADD, V("result"), [V("x"), V("a")])
    found = issues(p, "def-before-use")
    assert len(found) == 1
    assert (found[0].block_name, found[0].instruction_index) == ("merge", 0)
    p = diamond()
    p.functions[0].blocks[1].instructions[1].operands[1] = literal(1, D.INT32)
    assert len(issues(p, "type-consistency")) == 1


@pytest.mark.parametrize("target", [None, "", "missing", "other-function"])
def test_u09_u10_u11_u12_missing_target_suppresses_graph(target):
    p = program(I(O.BR, target=target))
    p.functions[0].new_block("exit").add(I(O.RETURN))
    other = Function("other")
    other.new_block("other-function").add(I(O.RETURN))
    p.add_function(other)
    found = issues(p, "label-existence")
    assert len(found) == 1
    assert all(i.level is ErrorLevel.ERROR for i in verify_ir(p)[1])
    p.functions[0].blocks[0].instructions[0].target = "exit"
    assert verify_ir(p) == (True, [])


@pytest.mark.parametrize("name", ["", "  ", "entry"])
def test_u13_a01_duplicate_positions(name):
    p = program(I(O.RETURN, operands=[V("undefined")]))
    b = BasicBlock(name)
    b.add(I(O.RETURN, operands=[V("undefined")]))
    p.functions[0].blocks.append(b)
    assert len(issues(p, "label-existence")) == 1
    assert len(issues(p, "def-before-use")) == 2


def test_u15_u16_empty_and_unterminated():
    found = issues(program(), "block-termination")
    assert len(found) == 1 and found[0].instruction_index is None
    p = straight()
    p.functions[0].blocks[0].instructions.pop()
    found = issues(p, "block-termination")
    assert len(found) == 1 and found[0].instruction_index == 1


@pytest.mark.parametrize("case", ["mixed", "result", "count", "dest", "reference", "return-type", "return-count"])
def test_u18_u19_u20_u25_u26_types(case):
    p = straight()
    b = p.functions[0].blocks[0]
    if case == "mixed":
        b.instructions[0].operands[1] = literal(1, D.INT32)
    elif case == "result":
        b.instructions[1].dest.dtype = D.INT32
    elif case == "count":
        b.instructions[0].operands = []
    elif case == "dest":
        b.instructions[-1].dest = V("unexpected")
    elif case == "reference":
        b.instructions[0].operands[0] = V("a", D.INT32)
    elif case == "return-type":
        p.functions[0].returns = [V("result", D.INT32)]
    else:
        p.functions[0].returns = [V("result")]
        b.instructions[-1].operands = []
    issues(p, "type-consistency")


@pytest.mark.parametrize("operands,attrs,valid", [
    ([literal(1, D.INT32)], {}, True), ([literal(1, D.INT64)], {}, True),
    ([literal(1)], {}, False), ([literal(), literal()], {"cmp_op": "<"}, True),
    ([literal(), literal()], {"cmp_op": "eq"}, False),
    ([literal(1, D.INT32)], {"cmp_op": "=="}, False),
])
def test_u21_u22_u28_conditions(operands, attrs, valid):
    p = program(I(O.BR_IF, operands=operands, attrs=attrs, target="exit, exit"))
    p.functions[0].new_block("exit").add(I(O.RETURN))
    assert verify_ir(p)[0] is valid


@pytest.mark.parametrize("target", ["left", "left,", "left,right,other", None, ",right"])
def test_u27_branch_shape(target):
    p = program(I(O.BR_IF, operands=[literal(1, D.INT32)], target=target))
    assert len(issues(p, "control-flow-integrity")) == 1
    assert not any(i.rule == "label-existence" for i in verify_ir(p)[1])


@pytest.mark.parametrize("op", [O.RETURN, O.BR, O.BR_IF])
def test_u29_u30_after_terminator(op):
    inst = I(op, target="exit" if op == O.BR else "exit,exit" if op == O.BR_IF else None,
             operands=[literal(1, D.INT32)] if op == O.BR_IF else [])
    p = program(inst, I(O.NEG, V("x"), [literal()]))
    p.functions[0].new_block("exit").add(I(O.RETURN))
    found = verify_ir(p)[1]
    assert {i.rule for i in found} == {"control-flow-integrity", "block-termination"}
    assert all(i.instruction_index == 1 for i in found)
    p.functions[0].blocks[0].instructions[-1] = I(O.RETURN)
    assert len(issues(p, "control-flow-integrity")) == 1


def test_u31_u32_u33_unreachable():
    p = program(I(O.RETURN))
    p.functions[0].new_block("dead").add(I(O.BR, target="dead2"))
    p.functions[0].new_block("dead2").add(I(O.BR, target="dead"))
    found = issues(p, "control-flow-integrity", passed=True)
    assert len(found) == 2 and all(i.level is ErrorLevel.WARNING for i in found)
    p.functions[0].blocks[1].instructions.insert(0, I(O.NEG, V("x"), [V("missing")]))
    assert len(issues(p, "def-before-use")) == 1


@pytest.mark.parametrize("kind", ["instruction", "branch", "param", "global", "shadow"])
def test_u34_u35_u36_a02_duplicate_definitions(kind):
    p = straight()
    f = p.functions[0]
    if kind == "instruction":
        f.blocks[0].instructions[1].dest = V("x")
        f.blocks[0].instructions[-1].operands = [V("x")]
    elif kind == "branch":
        p = diamond()
        p.functions[0].blocks[2].instructions[0].dest = V("x")
        p.functions[0].blocks[2].instructions[1].operands[1] = V("x")
    elif kind == "param":
        f.params.append(V("a"))
    elif kind == "global":
        p.global_values = [V("g"), V("g")]
        p.add_function(copy.deepcopy(f))
    else:
        p.global_values = [V("a")]
    found = issues(p, "ssa-validity")
    assert len(found) == 1
    assert "first definition" in found[0].message
    assert not any(i.rule == "def-before-use" for i in verify_ir(p)[1])


def test_u39_a12_a17_collect_other_functions():
    p = loop()
    p.functions[0].blocks[0].instructions.pop(2)
    p.add_function(Function("empty"))
    found = verify_ir(p)[1]
    assert {i.rule for i in found} >= {"control-flow-integrity", "entry-existence"}
    assert found[-1].function_name == "empty"


@pytest.mark.parametrize("nested", [False, True])
def test_u42a_u42c_a15_a16_loop_zero_path(nested):
    p = loop(nested)
    assert verify_ir(p) == (True, [])
    p.functions[0].blocks[0].instructions[-1].operands = [V("x", D.INT32)]
    found = issues(p, "def-before-use")
    assert len(found) == 1 and found[0].value_name == "x"
    assert found[0].block_name == "entry"


@pytest.mark.parametrize("inst", [I(O.ENDFOR), I(O.FOR, V("i", D.INT32), attrs=dict(start=0, end=1, step=1))])
def test_u42b_bad_pairing(inst):
    assert len(issues(program(inst, I(O.RETURN)), "control-flow-integrity")) == 1


@pytest.mark.parametrize("representation", ["label", "phi"])
def test_u42d_unsupported(representation):
    p = program(I(O.RETURN))
    if representation == "label":
        p.functions[0].blocks[0].instructions.insert(0, I(O.LABEL))
    else:
        p.functions[0].blocks[0].phi_nodes = [I(O.ADD)]
    found = issues(p, "control-flow-integrity")
    assert len(found) == 1 and "unsupported" in found[0].message


def test_u43_u44_explicit_loop():
    p = program(I(O.BR, target="header"))
    f = p.functions[0]
    f.new_block("header").add(I(O.BR_IF, operands=[literal(1, D.INT32)], target="body,exit"))
    f.new_block("body").add(I(O.BR, target="header"))
    f.new_block("exit").add(I(O.RETURN))
    assert verify_ir(p) == (True, [])
    f.blocks[2].instructions[0].target = "entry"
    assert len(issues(p, "control-flow-integrity")) == 1


@pytest.mark.parametrize("dtype,bits", [(D.INT32, 32), (D.INT64, 64)])
@pytest.mark.parametrize("case", ["min", "max", "low", "high", "bool", "fraction", "none"])
def test_u23_u45_integer_constants(dtype, bits, case):
    value = {"min": -(1 << (bits - 1)), "max": (1 << (bits - 1)) - 1,
             "low": -(1 << (bits - 1)) - 1, "high": 1 << (bits - 1),
             "bool": True, "fraction": 1.5, "none": None}[case]
    p = program(I(O.RETURN, operands=[literal(value, dtype)]))
    passed, found = verify_ir(p)
    assert passed is (case in ("min", "max"))
    assert all(i.rule == "type-consistency" for i in found)


@pytest.mark.parametrize("left,right,valid", [
    (math.nan, math.nan, True), (math.inf, math.inf, True),
    (-math.inf, math.inf, False), (0., -0., False), (-0., -0., True),
    (1., 1.00000000001, False),
])
def test_a03_a04_a13_exact_metadata(left, right, valid):
    p = program(I(O.LOAD_CONST, literal(right, name="x"), attrs={"value": left}), I(O.RETURN))
    assert verify_ir(p)[0] is valid


def test_a05_a14_invalid_dtype_and_a09_independent_errors():
    p = straight()
    p.functions[0].params.append(V("unused", "f32"))
    assert len(issues(p, "type-consistency")) == 1
    x = V("x", is_constant=True)
    p = program(I(O.NEG, V("y"), [x]), I(O.LOAD_CONST, x, attrs={"value": 1}), I(O.RETURN))
    found = verify_ir(p)[1]
    assert {i.rule for i in found} == {"type-consistency", "def-before-use"}
    p = program(I(O.ADD, V("x", "f32"), [V("a", "f32"), V("a", "f32")]), I(O.RETURN), params=[V("a", "f32")])
    assert len(issues(p, "type-consistency")) == 3


def test_a06_a11_repeat_readonly_and_name_identity():
    p = straight()
    p.functions[0].blocks[0].instructions[1].operands[0] = V("x")
    snapshot = copy.deepcopy(p)
    identities = [id(i) for i in p.functions[0].blocks[0].instructions]
    verifier = IRVerifier(p)
    assert verifier.verify() == verifier.verify() == []
    assert p.functions == snapshot.functions or p.dump() == snapshot.dump()
    assert p.functions[0].params == snapshot.functions[0].params
    assert identities == [id(i) for i in p.functions[0].blocks[0].instructions]
    p.functions[0].blocks[0].instructions.pop(0)
    assert verifier.verify()[0].rule == "def-before-use"


# Independent contract table: verify every opcode registration, but exercise
# shared signature validation only once per distinct signature.
_SIGNATURE_GROUPS = [
    ((O.ADD, O.SUB, O.MUL, O.DIV, O.MATMUL, O.DOT), (2, 2, True, "T")),
    ((O.NEG, O.RELU, O.MAXPOOL, O.RESHAPE, O.TRANSPOSE), (1, 1, True, "T")),
    ((O.EXP, O.GELU, O.SIGMOID, O.SOFTMAX), (1, 1, True, "F")),
    ((O.CONV, O.GEMM), (3, 3, True, "T")),
    ((O.CONCAT,), (1, None, True, "T")),
    ((O.LOAD_CONST, O.ALLOCA, O.FOR), (0, 0, True, None)),
    ((O.LOAD,), (1, 1, True, None)),
    ((O.STORE,), (2, 2, False, None)),
    ((O.BR, O.ENDFOR), (0, 0, False, None)),
    ((O.BR_IF,), (1, 2, False, None)),
    ((O.RETURN,), (0, 1, False, None)),
]
_EXPECTED_SIGNATURES = {
    opcode: signature for opcodes, signature in _SIGNATURE_GROUPS for opcode in opcodes
}


def signature_program(opcode, dtype=None):
    if dtype is None:
        dtype = D.INT32 if opcode in (O.FOR, O.BR_IF) else D.FLOAT32
    minimum, _, has_dest, _ = _EXPECTED_SIGNATURES[opcode]
    operands = [literal(1, dtype, f"c{i}") for i in range(minimum)]
    dest = V("result", dtype) if has_dest else None
    inst = I(opcode, dest, operands, attrs={"value": 1, "start": 0, "end": 2, "step": 1})
    inst.target = "exit,exit" if opcode == O.BR_IF else "exit" if opcode == O.BR else None
    instructions = [inst]
    if opcode == O.FOR:
        instructions.append(I(O.ENDFOR))
    elif opcode == O.ENDFOR:
        instructions.insert(0, I(O.FOR, V("i", D.INT32), attrs=dict(start=0, end=1, step=1)))
    if opcode not in (O.RETURN, O.BR, O.BR_IF):
        instructions.append(I(O.RETURN))
    p = program(*instructions)
    if opcode in (O.BR, O.BR_IF):
        p.functions[0].new_block("exit").add(I(O.RETURN))
    return p, inst


def test_all_opcode_signatures_are_registered_correctly():
    assert set(OPCODE_SPECS) == set(_EXPECTED_SIGNATURES)
    assert set(_EXPECTED_SIGNATURES) | {O.LABEL} == set(O)
    for opcode, expected in _EXPECTED_SIGNATURES.items():
        spec = OPCODE_SPECS[opcode]
        assert (spec.min_operands, spec.max_operands, spec.has_dest, spec.family) == expected, opcode


@pytest.mark.parametrize("opcode", list(_EXPECTED_SIGNATURES), ids=lambda op: op.value)
def test_each_opcode_accepts_a_legal_instruction(opcode):
    p, _ = signature_program(opcode)
    assert verify_ir(p) == (True, [])


@pytest.mark.parametrize("opcode,signature", [
    (opcodes[0], signature) for opcodes, signature in _SIGNATURE_GROUPS
], ids=[opcodes[0].value for opcodes, _ in _SIGNATURE_GROUPS])
def test_signature_count_and_destination_boundaries(opcode, signature):
    """One representative per signature; special opcode rules have own tests."""
    minimum, maximum, has_dest, _ = signature
    p, inst = signature_program(opcode)
    dest = inst.dest
    inst.dest = None if has_dest else V("unexpected")
    assert any(i.rule == "type-consistency" and "result" in i.message for i in verify_ir(p)[1])
    inst.dest = dest
    if maximum is not None:
        inst.operands = [literal()] * (maximum + 1)
        assert any("operands, got" in i.message for i in verify_ir(p)[1])
    if minimum:
        inst.operands = [literal()] * (minimum - 1)
        assert any("operands, got" in i.message for i in verify_ir(p)[1])


@pytest.mark.parametrize("opcode", [O.ADD, O.EXP], ids=["numeric-family", "float-family"])
@pytest.mark.parametrize("dtype", list(D), ids=lambda dtype: dtype.value)
def test_shared_type_families(opcode, dtype):
    p, _ = signature_program(opcode, dtype)
    passed, found = verify_ir(p)
    valid = opcode == O.ADD or dtype in (D.FLOAT32, D.FLOAT64)
    assert passed is valid
    assert (not found) if valid else all(i.rule == "type-consistency" for i in found)


def test_error_format_and_frozen():
    error = VerificationError(ErrorLevel.ERROR, "bad", "main", "entry", 0, "x", "def-before-use", "after-parse")
    assert "stage=after-parse" in str(error) and "instruction=0" in str(error)
    with pytest.raises(AttributeError):
        error.message = "changed"


@pytest.mark.parametrize("value", [None, True, 1.5, -1, 0])
def test_for_invalid_step(value):
    p = loop()
    p.functions[0].blocks[0].instructions[0].attrs["step"] = value
    assert all(i.rule == "type-consistency" for i in verify_ir(p)[1])
    assert not verify_ir(p)[0]


def test_unknown_opcode_multi_return_and_locals_are_declarations():
    p = program(I("future-op"), I(O.RETURN))
    assert len(issues(p, "type-consistency")) == 1
    p = program(I(O.RETURN, operands=[V("x")]), returns=[V("a"), V("b")])
    p.functions[0].locals = [V("x")]
    found = verify_ir(p)[1]
    assert {i.rule for i in found} == {"type-consistency", "def-before-use"}


def test_unreachable_def_cannot_supply_reachable_use():
    p = program(I(O.BR, target="exit"))
    f = p.functions[0]
    f.new_block("dead").instructions = [I(O.NEG, V("x"), [literal()]), I(O.BR, target="exit")]
    f.new_block("exit").add(I(O.RETURN, operands=[V("x")]))
    assert len(issues(p, "def-before-use")) == 1
    assert any(i.level is ErrorLevel.WARNING for i in verify_ir(p)[1])


def test_invalid_graph_still_checks_local_order_without_cross_block_guesses():
    p = straight()
    b = p.functions[0].blocks[0]
    b.instructions[0], b.instructions[1] = b.instructions[1], b.instructions[0]
    b.instructions[-1] = I(O.BR, target="missing")
    assert len(issues(p, "def-before-use")) == 1
    assert len(issues(p, "label-existence")) == 1


def test_duplicate_block_names_keep_distinct_analysis_positions():
    p = program(I(O.RETURN, operands=[V("x")]))
    duplicate = BasicBlock("entry")
    duplicate.instructions = [I(O.NEG, V("x"), [literal()]), I(O.RETURN)]
    p.functions[0].blocks.append(duplicate)
    # The graph is invalid. A definition in a different, identically named
    # block must not be mistaken for a same-segment forward reference.
    assert [i.rule for i in verify_ir(p)[1]] == ["label-existence"]


def test_readonly_complete_snapshot_with_nan_and_object_identity():
    def snapshot(obj):
        if isinstance(obj, float) and math.isnan(obj):
            return "NaN"
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        if isinstance(obj, (D, O)):
            return obj.value
        if isinstance(obj, dict):
            return (id(obj), tuple((key, snapshot(value)) for key, value in obj.items()))
        if isinstance(obj, (tuple, list)):
            return (id(obj), tuple(snapshot(value) for value in obj))
        return (id(obj), snapshot(vars(obj)))

    p = loop(nested=True)
    f = p.functions[0]
    p.global_values = [literal(math.nan, name="global")]
    f.params = [V("parameter", shape=(2, 3))]
    f.locals = [V("declared", shape=(4,))]
    f.returns = [V("result")]
    f.blocks[0].instructions[-1].operands = [literal(math.nan)]
    before = snapshot(p)
    verifier = IRVerifier(p)
    assert verifier.verify() == verifier.verify() == []
    assert snapshot(p) == before
    f.blocks[0].phi_nodes = [I(O.ADD, V("phi"))]
    before = snapshot(p)
    assert not verify_ir(p)[0]
    assert snapshot(p) == before


@pytest.mark.parametrize("key,value", [("start", None), ("end", True), ("end", 2**31), ("start", 1.5)])
def test_for_attribute_boundaries(key, value):
    p = loop()
    attrs = p.functions[0].blocks[0].instructions[0].attrs
    if value is None:
        del attrs[key]
    else:
        attrs[key] = value
    assert len(issues(p, "type-consistency")) == 1


def test_storage_load_return_signatures_and_concat():
    p = program(I(O.ALLOCA, V("slot")), I(O.LOAD, V("x", D.INT32), [V("slot")]), I(O.RETURN))
    assert len(issues(p, "type-consistency")) == 1
    p = program(I(O.CONCAT, V("x"), [literal(), literal(), literal()]), I(O.RETURN))
    assert verify_ir(p) == (True, [])
    p.functions[0].blocks[0].instructions[0].operands[-1] = literal(1, D.INT32)
    assert len(issues(p, "type-consistency")) == 1


def test_branch_into_loop_body_does_not_define_induction_variable():
    p = loop()
    f = p.functions[0]
    insts = f.blocks[0].instructions
    f.blocks[0].name = "init"
    f.blocks[0].instructions = insts[:1]
    f.new_block("body").instructions = insts[1:]
    start = BasicBlock("start")
    start.add(I(O.BR, target="body"))
    f.blocks.insert(0, start)
    found = issues(p, "def-before-use")
    assert len(found) == 1 and found[0].value_name == "i"
    assert found[0].block_name == "body"


def test_for_induction_variable_requires_i32():
    p, _ = signature_program(O.FOR, D.FLOAT32)
    found = issues(p, "type-consistency")
    assert len(found) == 1 and "FOR result must be i32" in found[0].message


def test_unreachable_predecessor_does_not_hide_valid_definition():
    p = program(I(O.NEG, V("x"), [literal()]), I(O.BR, target="exit"))
    f = p.functions[0]
    f.new_block("dead").add(I(O.BR, target="exit"))
    f.new_block("exit").add(I(O.RETURN, operands=[V("x")]))
    passed, found = verify_ir(p)
    assert passed
    assert len(found) == 1 and found[0].level is ErrorLevel.WARNING


def test_upstream_adapter_failure_is_a_diagnostic(monkeypatch):
    import scratchv.analysis.ir_verifier as module

    def fail(_):
        raise ValueError("duplicate basic block name: for_hdr1")

    monkeypatch.setattr(module, "IRCFGAdapter", fail)
    passed, found = verify_ir(straight(), stage="after-parse")
    assert not passed
    assert len(found) == 1
    assert found[0].rule == "control-flow-integrity"
    assert found[0].stage == "after-parse"
    assert "cannot build unified CFG" in found[0].message


def test_endfor_only_original_block_is_reachable():
    p = loop()
    f = p.functions[0]
    instructions = f.blocks[0].instructions
    closing = next(i for i, inst in enumerate(instructions) if inst.opcode == O.ENDFOR)
    f.blocks[0].instructions = instructions[:closing]
    f.new_block("closing").instructions = instructions[closing:closing + 1]
    f.new_block("exit").instructions = instructions[closing + 1:]
    assert verify_ir(p) == (True, [])
