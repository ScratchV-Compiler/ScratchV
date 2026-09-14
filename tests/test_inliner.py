"""Tests for the Topic 15 IR inliner."""

from scratchv.analysis.ir_verifier import ErrorLevel, IRVerifier
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import OpCode
from scratchv.optimizer.inline import Inliner as InlinerAlias
from scratchv.optimizer.inliner import Inliner, InlinerConfig


def _build_inc_callee(b: IRBuilder):
    a = b.make_value(name="a")
    b_val = b.make_value(name="b")
    inc = b.new_function("inc", params=[a, b_val])
    b.new_block("entry")
    t = b.add(a, b_val)
    b.ret(t)
    return inc


def _build_inc_caller(b: IRBuilder, args_pairs):
    main = b.new_function("main")
    b.new_block("entry")
    results = []
    for pair in args_pairs:
        results.append(b.call("inc", list(pair)))
    if len(results) == 1:
        b.ret(results[0])
    else:
        total = b.add(results[0], results[1])
        b.ret(total)
    return main, results


def _assert_no_call(program) -> None:
    assert all(
        ins.opcode is not OpCode.CALL
        for func in program.functions
        for block in func.blocks
        for ins in block.instructions
    )


def _assert_verifier_clean(program) -> None:
    errors = [
        e for e in IRVerifier(program).verify()
        if e.level is ErrorLevel.ERROR
    ]
    assert errors == []


def _block_by_name(func, name):
    return next(block for block in func.blocks if block.name == name)


class TestInlineBasic:
    def test_alias_module_exports_same_class(self):
        assert InlinerAlias is Inliner

    def test_single_site_substitutes_args(self):
        b = IRBuilder()
        _build_inc_callee(b)
        main = b.new_function("main")
        b.new_block("entry")
        x = b.make_const(2.0)
        y = b.make_const(3.0)
        r = b.call("inc", [x, y])
        b.ret(r)

        inl = Inliner(b.program, InlinerConfig(max_instrs=8))
        assert inl.run() == 1
        assert inl.stats["inlined"] == 1
        assert inl.stats["rejected"] == 0
        assert set(inl.stats) == {"inlined", "rejected", "rounds"}

        instrs = [i for blk in main.blocks for i in blk.instructions]
        assert all(i.opcode is not OpCode.CALL for i in instrs)
        add = next(i for i in instrs if i.opcode is OpCode.ADD)
        assert add.operands[0] is x
        assert add.operands[1] is y
        cont = _block_by_name(main, "main_inl0_cont")
        ret = cont.instructions[-1]
        assert ret.opcode is OpCode.RETURN
        assert ret.operands[0] is add.dest
        _assert_verifier_clean(b.program)

    def test_multi_site_independent_clones(self):
        b = IRBuilder()
        _build_inc_callee(b)
        main = b.new_function("main")
        b.new_block("entry")
        x = b.make_const(2.0)
        y = b.make_const(3.0)
        r1 = b.call("inc", [x, y])
        r2 = b.call("inc", [x, x])
        s = b.add(r1, r2)
        b.ret(s)

        inl = Inliner(b.program, InlinerConfig(max_instrs=8))
        assert inl.run() == 2
        assert inl.stats["inlined"] == 2
        assert inl.stats["rejected"] == 0

        assert [blk.name for blk in main.blocks] == [
            "entry", "inc_entry_inl0", "main_inl0_cont",
            "inc_entry_inl1", "main_inl1_cont",
        ]
        _assert_no_call(b.program)

        add0 = _block_by_name(main, "inc_entry_inl0").instructions[0]
        add1 = _block_by_name(main, "inc_entry_inl1").instructions[0]
        assert add0.opcode is OpCode.ADD
        assert add1.opcode is OpCode.ADD
        assert add0.dest is not add1.dest
        assert add0.operands[0] is x and add0.operands[1] is y
        assert add1.operands[0] is x and add1.operands[1] is x

        ret = _block_by_name(main, "main_inl1_cont").instructions[-1]
        assert ret.opcode is OpCode.RETURN
        assert ret.operands[0] is s
        s_instr = next(
            i for i in main.blocks[-1].instructions if i.dest is s)
        assert s_instr.opcode is OpCode.ADD
        assert s_instr.operands[0] is add0.dest
        assert s_instr.operands[1] is add1.dest
        _assert_verifier_clean(b.program)

    def test_multi_site_with_internal_constants_renamed(self):
        b = IRBuilder()
        a = b.make_value(name="a")
        b.new_function("scale", params=[a])
        b.new_block("entry")
        c = b.load_const(2.0)
        t = b.mul(a, c)
        b.ret(t)

        b.new_function("main")
        b.new_block("entry")
        x = b.make_const(3.0)
        r1 = b.call("scale", [x])
        r2 = b.call("scale", [x])
        s = b.add(r1, r2)
        b.ret(s)

        inl = Inliner(b.program, InlinerConfig())
        assert inl.run() == 2
        assert inl.stats["inlined"] == 2

        clone0 = _block_by_name(b.program.functions[-1], "scale_entry_inl0")
        clone1 = _block_by_name(b.program.functions[-1], "scale_entry_inl1")
        c0 = clone0.instructions[0]
        c1 = clone1.instructions[0]
        assert c0.opcode is OpCode.LOAD_CONST
        assert c1.opcode is OpCode.LOAD_CONST
        assert c0.dest is not c
        assert c1.dest is not c
        assert c0.dest is not c1.dest
        assert c0.dest.name == f"{c.name}_inl0"
        assert c1.dest.name == f"{c.name}_inl1"
        _assert_verifier_clean(b.program)

    def test_first_block_branches_to_clone(self):
        b = IRBuilder()
        _build_inc_callee(b)
        main = b.new_function("main")
        b.new_block("entry")
        x = b.make_const(1.0)
        y = b.make_const(2.0)
        b.ret(b.call("inc", [x, y]))

        inl = Inliner(b.program, InlinerConfig(max_instrs=8))
        assert inl.run() == 1
        entry = main.blocks[0]
        br = entry.instructions[-1]
        assert br.opcode is OpCode.BR
        assert br.target == "inc_entry_inl0"
        assert main.blocks[1].name == "inc_entry_inl0"


class TestInlineReject:
    @staticmethod
    def _build_reject_program():
        b = IRBuilder()
        a = b.make_value(name="a")
        b.new_function("f", params=[a])
        b.new_block("entry")
        b.load_const(1.0)
        b.load_const(2.0)
        b.load_const(3.0)
        t = b.add(a, a)
        b.ret(t)

        main = b.new_function("main")
        b.new_block("entry")
        x = b.make_const(1.0)
        r = b.call("f", [x])
        b.ret(r)
        return b, main

    def test_body_too_large_keeps_call(self):
        b, main = self._build_reject_program()
        before = b.program.dump()

        inl = Inliner(b.program, InlinerConfig(max_instrs=4))
        assert inl.run() == 0
        assert inl.stats["inlined"] == 0
        assert inl.stats["rejected"] == 1
        assert b.program.dump() == before
        assert len(main.blocks) == 1
        assert any("body_too_large" in w and "5 > 4" in w
                   for w in inl.warnings)

    def test_threshold_boundary_inlines(self):
        b, _main = self._build_reject_program()
        inl = Inliner(b.program, InlinerConfig(max_instrs=5))
        assert inl.run() == 1
        assert inl.stats["inlined"] == 1
        assert inl.stats["rejected"] == 0
        _assert_no_call(b.program)
        _assert_verifier_clean(b.program)

    def test_recursive_rejected(self):
        b = IRBuilder()
        a = b.make_value(name="a")
        b.new_function("f", params=[a])
        b.new_block("entry")
        r = b.call("f", [a])
        b.ret(r)
        b.new_function("main")
        b.new_block("entry")
        x = b.make_const(1.0)
        rm = b.call("f", [x])
        b.ret(rm)
        before = b.program.dump()

        inl = Inliner(b.program, InlinerConfig())
        assert inl.run() == 0
        assert inl.stats["inlined"] == 0
        assert inl.stats["rejected"] == 2
        assert b.program.dump() == before
        assert any("recursive_callee" in w for w in inl.warnings)

    def test_mutual_recursion_rejected(self):
        b = IRBuilder()
        pa = b.make_value(name="a")
        b.new_function("p", params=[pa])
        b.new_block("entry")
        r_p = b.call("q", [pa])
        b.ret(r_p)
        qa = b.make_value(name="a")
        b.new_function("q", params=[qa])
        b.new_block("entry")
        r_q = b.call("p", [qa])
        b.ret(r_q)
        b.new_function("main")
        b.new_block("entry")
        x = b.make_const(1.0)
        r_m = b.call("p", [x])
        b.ret(r_m)
        before = b.program.dump()

        inl = Inliner(b.program, InlinerConfig())
        assert inl.run() == 0
        assert inl.stats["inlined"] == 0
        assert inl.stats["rejected"] == 3
        assert b.program.dump() == before
        assert sum(1 for w in inl.warnings if "recursive_callee" in w) == 3

    def test_single_site_only_inlines_single_site(self):
        b = IRBuilder()
        _build_inc_callee(b)
        b.new_function("main")
        b.new_block("entry")
        x = b.make_const(1.0)
        y = b.make_const(2.0)
        r = b.call("inc", [x, y])
        b.ret(r)

        inl = Inliner(b.program, InlinerConfig(single_site_only=True))
        assert inl.run() == 1
        assert inl.stats["inlined"] == 1
        assert inl.stats["rejected"] == 0
        _assert_no_call(b.program)

    def test_single_site_only_rejects_multiple_sites(self):
        b = IRBuilder()
        _build_inc_callee(b)
        b.new_function("main")
        b.new_block("entry")
        x = b.make_const(1.0)
        y = b.make_const(2.0)
        r1 = b.call("inc", [x, y])
        r2 = b.call("inc", [x, x])
        s = b.add(r1, r2)
        b.ret(s)
        before = b.program.dump()

        inl = Inliner(b.program, InlinerConfig(single_site_only=True))
        assert inl.run() == 0
        assert inl.stats["inlined"] == 0
        assert inl.stats["rejected"] == 2
        assert b.program.dump() == before
        assert all("multiple_call_sites" in w for w in inl.warnings)

    def test_growth_budget_stops_cloning(self):
        b = IRBuilder()
        _build_inc_callee(b)
        b.new_function("main")
        b.new_block("entry")
        x = b.make_const(1.0)
        y = b.make_const(2.0)
        r1 = b.call("inc", [x, y])
        r2 = b.call("inc", [x, x])
        s = b.add(r1, r2)
        b.ret(s)

        inl = Inliner(b.program, InlinerConfig(growth_budget=2))
        assert inl.run() == 1
        assert inl.stats["inlined"] == 1
        assert inl.stats["rejected"] == 1
        assert any("growth_budget_exceeded" in w for w in inl.warnings)

    def test_argc_mismatch_rejected(self):
        b = IRBuilder()
        _build_inc_callee(b)
        b.new_function("main")
        b.new_block("entry")
        x = b.make_const(1.0)
        r = b.call("inc", [x])
        b.ret(r)

        inl = Inliner(b.program, InlinerConfig())
        assert inl.run() == 0
        assert inl.stats["rejected"] == 1
        assert any("argc_mismatch" in w for w in inl.warnings)

    def test_malformed_argc_rejected(self):
        b = IRBuilder()
        _build_inc_callee(b)
        main = b.new_function("main")
        b.new_block("entry")
        x = b.make_const(1.0)
        y = b.make_const(2.0)
        r = b.call("inc", [x, y])
        b.ret(r)
        main.blocks[0].instructions[0].attrs["argc"] = 5

        inl = Inliner(b.program, InlinerConfig())
        assert inl.run() == 0
        assert inl.stats["rejected"] == 1
        assert any("malformed_call" in w for w in inl.warnings)

    def test_callee_not_found_rejected(self):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        x = b.make_const(1.0)
        r = b.call("missing", [x])
        b.ret(r)

        inl = Inliner(b.program, InlinerConfig())
        assert inl.run() == 0
        assert inl.stats["rejected"] == 1
        assert any("callee_not_found" in w for w in inl.warnings)

    def test_ret_arity_mismatch_rejected(self):
        b = IRBuilder()
        v = b.make_value(name="v")
        b.new_function("report", params=[v])
        b.new_block("entry")
        b.ret()
        b.new_function("main")
        b.new_block("entry")
        x = b.make_const(1.0)
        r = b.call("report", [x])
        b.ret(r)

        inl = Inliner(b.program, InlinerConfig())
        assert inl.run() == 0
        assert inl.stats["rejected"] == 1
        assert any("ret_arity_mismatch" in w for w in inl.warnings)

    def test_valued_callee_without_dest_rejected(self):
        b = IRBuilder()
        a = b.make_value(name="a")
        b.new_function("f", params=[a])
        b.new_block("entry")
        t = b.add(a, a)
        b.ret(t)
        b.new_function("main")
        b.new_block("entry")
        x = b.make_const(1.0)
        b.call("f", [x], has_ret=False)
        b.ret(x)

        inl = Inliner(b.program, InlinerConfig())
        assert inl.run() == 0
        assert inl.stats["rejected"] == 1
        assert any("ret_arity_mismatch" in w for w in inl.warnings)

    def test_tail_call_rejected(self):
        b = IRBuilder()
        _build_inc_callee(b)
        b.new_function("main")
        b.new_block("entry")
        x = b.make_const(1.0)
        y = b.make_const(2.0)
        r = b.call("inc", [x, y], is_tail=True)
        b.ret(r)

        inl = Inliner(b.program, InlinerConfig())
        assert inl.run() == 0
        assert inl.stats["rejected"] == 1
        assert any("tail_unsupported" in w for w in inl.warnings)

    def test_loop_body_rejected(self):
        b = IRBuilder()
        a = b.make_value(name="a")
        b.new_function("f", params=[a])
        b.new_block("entry")
        b.for_loop(0, 4)
        b.endfor()
        b.ret(a)
        b.new_function("main")
        b.new_block("entry")
        x = b.make_const(1.0)
        r = b.call("f", [x])
        b.ret(r)

        inl = Inliner(b.program, InlinerConfig())
        assert inl.run() == 0
        assert inl.stats["rejected"] == 1
        assert any("loop_body_unsupported" in w for w in inl.warnings)

    def test_multiple_valued_returns_rejected(self):
        b = IRBuilder()
        a = b.make_value(name="a")
        b.new_function("f", params=[a])
        b.new_block("entry")
        t1 = b.add(a, a)
        b.ret(t1)
        b.new_block("other")
        t2 = b.mul(a, a)
        b.ret(t2)
        b.new_function("main")
        b.new_block("entry")
        x = b.make_const(1.0)
        r = b.call("f", [x])
        b.ret(r)

        inl = Inliner(b.program, InlinerConfig())
        assert inl.run() == 0
        assert inl.stats["rejected"] == 1
        assert any("multiple_valued_returns" in w for w in inl.warnings)


class TestInlineBlocks:
    def test_void_multi_return_redirects(self):
        b = IRBuilder()
        v = b.make_value(name="v")
        b.new_function("report", params=[v])
        b.new_block("entry")
        c = b.load_const(0.0)
        b.br_if(c, "done1", "done2")
        b.new_block("done1")
        b.ret()
        b.new_block("done2")
        b.ret()

        main = b.new_function("main")
        b.new_block("entry")
        x = b.load_const(1.0)
        b.call("report", [x], has_ret=False)
        b.ret(x)

        inl = Inliner(b.program, InlinerConfig())
        assert inl.run() == 1
        assert inl.stats["inlined"] == 1
        assert inl.stats["rejected"] == 0
        _assert_no_call(b.program)

        clone_entry = _block_by_name(main, "report_entry_inl0")
        assert all(i.opcode is not OpCode.RETURN
                   for i in clone_entry.instructions)
        c_clone = clone_entry.instructions[0]
        assert c_clone.opcode is OpCode.LOAD_CONST
        assert c_clone.dest is not c
        assert c_clone.dest.name == f"{c.name}_inl0"
        br_if = clone_entry.instructions[1]
        assert br_if.opcode is OpCode.BR_IF
        assert br_if.target == "report_done1_inl0,report_done2_inl0"

        for name in ("report_done1_inl0", "report_done2_inl0"):
            done = _block_by_name(main, name)
            assert len(done.instructions) == 1
            assert done.instructions[0].opcode is OpCode.BR
            assert done.instructions[0].target == "main_inl0_cont"

        cont = _block_by_name(main, "main_inl0_cont")
        ret = cont.instructions[-1]
        assert ret.opcode is OpCode.RETURN
        assert ret.operands[0] is x
        _assert_verifier_clean(b.program)

    def test_multi_block_valued_return(self):
        b = IRBuilder()
        v = b.make_value(name="v")
        b.new_function("f", params=[v])
        b.new_block("entry")
        c = b.load_const(1.0)
        b.br_if(c, "left", "right")
        b.new_block("left")
        t1 = b.add(v, v)
        b.br("join")
        b.new_block("right")
        b.mul(v, v)
        b.br("join")
        b.new_block("join")
        b.ret(t1)

        main = b.new_function("main")
        b.new_block("entry")
        x = b.make_const(3.0)
        r = b.call("f", [x])
        b.ret(r)

        inl = Inliner(b.program, InlinerConfig())
        assert inl.run() == 1
        assert inl.stats["inlined"] == 1
        _assert_no_call(b.program)

        entry = _block_by_name(main, "f_entry_inl0")
        br_if = entry.instructions[-1]
        assert br_if.opcode is OpCode.BR_IF
        assert br_if.target == "f_left_inl0,f_right_inl0"

        left = _block_by_name(main, "f_left_inl0")
        add_clone = left.instructions[0]
        assert add_clone.opcode is OpCode.ADD
        assert add_clone.operands[0] is x and add_clone.operands[1] is x
        assert left.instructions[-1].target == "f_join_inl0"

        right = _block_by_name(main, "f_right_inl0")
        assert right.instructions[-1].target == "f_join_inl0"

        cont = _block_by_name(main, "main_inl0_cont")
        ret = cont.instructions[-1]
        assert ret.opcode is OpCode.RETURN
        assert ret.operands[0] is add_clone.dest
        _assert_verifier_clean(b.program)

    def test_nested_call_in_clone_is_inlined(self):
        b = IRBuilder()
        a = b.make_value(name="a")
        b.new_function("double", params=[a])
        b.new_block("entry")
        t = b.add(a, a)
        b.ret(t)

        m = b.make_value(name="m")
        b.new_function("quad", params=[m])
        b.new_block("entry")
        d = b.call("double", [m])
        q = b.call("double", [d])
        b.ret(q)

        b.new_function("main")
        b.new_block("entry")
        x = b.make_const(1.0)
        r = b.call("quad", [x])
        b.ret(r)

        inl = Inliner(b.program, InlinerConfig(max_instrs=8))
        assert inl.run() == 3
        assert inl.stats["inlined"] == 3
        assert inl.stats["rejected"] == 0
        _assert_no_call(b.program)
        _assert_verifier_clean(b.program)

    def test_call_at_end_of_block_warns(self):
        b = IRBuilder()
        a = b.make_value(name="a")
        b.new_function("f", params=[a])
        b.new_block("entry")
        t = b.add(a, a)
        b.ret(t)

        main = b.new_function("main")
        b.new_block("entry")
        x = b.make_const(1.0)
        b.call("f", [x])

        inl = Inliner(b.program, InlinerConfig())
        assert inl.run() == 1
        assert any("call at end of block" in w for w in inl.warnings)
        cont = _block_by_name(main, "main_inl0_cont")
        assert cont.instructions == []


class TestPipelineIntegration:
    def test_optimizer_pipeline_inlines_and_reports(self):
        from scratchv.compiler import CompilerConfig, CompilerDriver

        b = IRBuilder()
        _build_inc_callee(b)
        b.new_function("main")
        b.new_block("entry")
        x = b.make_const(1.0)
        y = b.make_const(2.0)
        r = b.call("inc", [x, y])
        b.ret(r)

        driver = CompilerDriver(CompilerConfig(
            optimize_level="basic", inline=True, inline_max_instrs=8))
        result = driver._run_optimizations(b.program)

        assert result.changes == 1
        assert "[inliner]" in result.message
        _assert_no_call(b.program)

    def test_optimizer_pipeline_surfaces_rejections(self):
        from scratchv.compiler import CompilerConfig, CompilerDriver

        b = IRBuilder()
        a = b.make_value(name="a")
        b.new_function("f", params=[a])
        b.new_block("entry")
        r = b.call("f", [a])
        b.ret(r)
        b.new_function("main")
        b.new_block("entry")
        x = b.make_const(1.0)
        rm = b.call("f", [x])
        b.ret(rm)

        driver = CompilerDriver(CompilerConfig(
            optimize_level="basic", inline=True))
        result = driver._run_optimizations(b.program)

        assert result.changes == 0
        assert any("recursive_callee" in w for w in result.warnings)
        assert any("inliner: skip" in w for w in result.warnings)

    def test_inline_requires_optimize_warning(self, tmp_path):
        from scratchv.compiler import CompilerConfig, CompilerDriver

        driver = CompilerDriver(CompilerConfig(
            optimize_level="none", inline=True))
        result = driver.compile(
            input_path="",
            output_path=str(tmp_path / "out.s"),
            dsl_source="c = add(a, b)\nreturn c\n",
        )

        assert result.success
        assert any("inliner requires" in w for w in result.warnings)

    def test_cli_flags_map_to_config(self):
        from scratchv.main import args_to_config, build_arg_parser

        args = build_arg_parser().parse_args([
            "--inline", "--inline-max-instrs", "7",
            "--inline-single-site", "--minimal-call-codegen",
            "input.dsl",
        ])
        config = args_to_config(args)

        assert config.inline is True
        assert config.inline_max_instrs == 7
        assert config.inline_single_site is True
        assert config.minimal_call_codegen is True
