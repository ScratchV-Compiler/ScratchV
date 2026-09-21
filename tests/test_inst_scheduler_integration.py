"""Compiler/CLI integration and real RISC-V assembly execution comparisons."""

import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter

import pytest

from scratchv.backend.inst_scheduler import (
    InstructionScheduler,
    ScheduleConfig,
    parse_instructions,
    schedule_assembly,
)
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.main import args_to_config, build_arg_parser

SOURCE = "lw t0, 0(a0)\nadd t1, t0, t2\nmul t3, t3, t4\nmul t5, t3, t6\nret\n"
DSL = "x = add(a, b)\ny = mul(x, c)\nreturn y\n"


@pytest.mark.parametrize("allocator", ["linear", "greedy", "naive"])
@pytest.mark.parametrize("dag", [False, True])
def test_actual_compiler_paths_use_scheduler(tmp_path, allocator, dag):
    config = CompilerConfig(
        reg_alloc=allocator,
        use_dag_isel=dag,
        schedule=True,
        schedule_strict=True,
        schedule_report=True,
    )
    driver = CompilerDriver(config)
    original = driver._generate_code(driver._parse("", DSL))
    result = driver.compile("", str(tmp_path / "out.s"), dsl_source=DSL)
    assert result.success, result.errors
    assert result.stats["schedule"]["input_instructions"] > 0
    assert result.stats["schedule"]["modeled_instructions"] > 0
    assert "Scheduling Report" in result.stats["schedule"]["report"]
    assert result.stats["register_map"] == driver._last_register_map
    assert not any("Scheduling Report" in warning for warning in result.warnings)
    # Preserve all lines supplied by code generation, including labels.
    assert Counter(result.output_text.splitlines()) == Counter(original.splitlines())
    returns = [inst for inst in parse_instructions(result.output_text)
               if inst.opcode in {"ret", "jalr"}]
    assert len(returns) == 1
    assert returns[0].uses == {"x1"} and not returns[0].defines


def test_disabled_has_identical_output_and_never_calls_scheduler(tmp_path, monkeypatch):
    driver = CompilerDriver(CompilerConfig())
    expected = driver._generate_code(driver._parse("", DSL))

    def forbidden(*args, **kwargs):
        pytest.fail("disabled scheduling must not execute")

    monkeypatch.setattr("scratchv.backend.inst_scheduler.schedule_assembly", forbidden)
    result = driver.compile("", str(tmp_path / "out.s"), dsl_source=DSL)
    assert result.success
    assert result.output_text == expected
    assert "schedule" not in result.stats


def test_stats_are_per_compilation_and_strict_failure_leaves_output_intact(
    tmp_path, monkeypatch
):
    driver = CompilerDriver(CompilerConfig(schedule=True))
    monkeypatch.setattr(driver, "_generate_code", lambda program: SOURCE)
    out = tmp_path / "out.s"
    first = driver.compile("", str(out), dsl_source=DSL)
    second = driver.compile("", str(out), dsl_source=DSL)
    assert (
        first.stats["schedule"]["saved_cycles"]
        == second.stats["schedule"]["saved_cycles"]
        == 2
    )
    monkeypatch.setattr(InstructionScheduler, "schedule", lambda self, dag: [])
    restored = driver.compile("", str(out), dsl_source=DSL)
    assert restored.success
    assert restored.output_text == SOURCE
    assert any("restored" in warning for warning in restored.warnings)
    out.write_text("existing output")
    driver.config.schedule_strict = True
    failed = driver.compile("", str(out), dsl_source=DSL)
    assert not failed.success
    assert "Scheduling failed" in failed.errors[0]
    assert out.read_text() == "existing output"


@pytest.mark.parametrize(
    "config",
    [
        CompilerConfig(backend="llvm", schedule=True),
        CompilerConfig(schedule_strict=True),
        CompilerConfig(schedule_report=True),
    ],
)
def test_invalid_options_fail_before_parsing_or_writing(tmp_path, config):
    result = CompilerDriver(config).compile("does-not-exist", str(tmp_path / "out"))
    assert not result.success
    assert "schedule" in result.errors[0]
    assert not (tmp_path / "out").exists()


def test_cli_flags_and_module_preserve_text_and_report_json(tmp_path):
    args = build_arg_parser().parse_args(
        ["--schedule", "--schedule-strict", "--schedule-report"]
    )
    config = args_to_config(args)
    assert config.schedule and config.schedule_strict and config.schedule_report
    source, output, report = (
        tmp_path / name for name in ("in.s", "out.s", "report.json")
    )
    source.write_bytes((".text\nmain:\n" + SOURCE).replace("\n", "\r\n").encode())
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scratchv.backend.inst_scheduler",
            str(source),
            "-o",
            str(output),
            "--strict",
            "--report",
            "--report-json",
            str(report),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Scheduling Report" in result.stderr
    assert output.read_bytes().startswith(b".text\r\nmain:\r\n")
    assert json.loads(report.read_text())["stats"]["saved_cycles"] == 2


def test_composes_with_other_assembly_passes():
    driver = CompilerDriver(
        CompilerConfig(
            schedule=True,
            schedule_strict=True,
            peephole_asm=True,
            const_merge=True,
            beautify_asm=True,
            count_instr=True,
        )
    )
    stats, warnings = {}, []
    result = driver._run_asm_passes(".text\nf:\n" + SOURCE, warnings, stats)
    assert "f:" in result and ".text" in result
    assert stats["schedule"]["saved_cycles"] == 2


@pytest.fixture
def execute_riscv(tmp_path):
    """Use actual assembler/linker and CPU execution, never the IR executor."""
    if not shutil.which("clang") or not shutil.which("qemu-riscv32"):
        if os.environ.get("SCRATCHV_REQUIRE_RISCV_EXECUTION") == "1":
            pytest.fail("CI requires clang and qemu-riscv32; execution cannot be skipped")
        pytest.skip("RISC-V execution checks require clang and qemu-riscv32")
    sequence = 0

    def execute(body, *, march="rv32imfd", prefix="", suffix=""):
        nonlocal sequence
        sequence += 1
        source = tmp_path / f"case{sequence}.s"
        binary = tmp_path / f"case{sequence}"
        # Dump all working integer registers, FP status and the shared data area.
        # Code/link addresses are deliberately not treated as business outputs.
        capture = "\n".join(f"sw t{i}, {i * 4}(a6)" for i in range(7))
        program = (
            ".text\n.globl _start\n_start:\nla a0, data\n"
            + "\n".join(f"li t{i}, {i + 1}" for i in range(7))
            + "\n"
            + prefix
            + body
            + "\nla a6, result\n"
            + capture
            + "\n"
            "frflags a5\nsw a5, 28(a6)\n"
            "li a0, 1\nla a1, result\nli a2, 96\nli a7, 64\necall\n"
            "li a0, 0\nli a7, 93\necall\n" + suffix + "\n"
            ".data\n.balign 8\nresult:\n.zero 32\ndata:\n"
            + ".word 9, 11, 13, 15, 17, 19, 21, 23\n.zero 32\n"
        )
        source.write_text(program)
        compiled = subprocess.run(
            [
                "clang",
                "--target=riscv32-unknown-linux-gnu",
                f"-march={march}",
                "-mabi=ilp32d",
                "-nostdlib",
                "-static",
                "-fuse-ld=lld",
                "-Wl,--no-relax",
                str(source),
                "-o",
                str(binary),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert compiled.returncode == 0, compiled.stderr
        ran = subprocess.run(
            ["qemu-riscv32", str(binary)], capture_output=True, timeout=10, check=False
        )
        assert ran.returncode == 0, ran.stderr
        assert len(ran.stdout) == 96
        return ran.stdout

    return execute


def test_linear_branch_target_executes_a_backward_loop(execute_riscv):
    from scratchv.backend.machine_types import MachineInstr, MachineOp, MachineOperand
    from scratchv.backend.regalloc_linear import LinearScanAllocator, block_from_machine_instrs

    counter = MachineOperand.vreg("counter")
    block = block_from_machine_instrs([
        MachineInstr(MachineOp.LI, counter, MachineOperand.immediate(3)),
        MachineInstr(MachineOp.LABEL, comment=".Lcounter"),
        MachineInstr(MachineOp.ADDI, counter, counter, MachineOperand.immediate(-1)),
        MachineInstr(MachineOp.BNEZ, counter, comment=".Lcounter"),
    ])
    assembly = LinearScanAllocator(phys_regs=["t0"]).emit(block)
    assert execute_riscv(assembly) == execute_riscv("li t0, 0\n")


def test_scheduled_machine_targets_survive_cfg_and_execution(execute_riscv):
    from scratchv.analysis.adapters import MachineCFGAdapter
    from scratchv.analysis.cfg import build_cfg
    from scratchv.analysis.cfg_validation import verify_cfg
    from scratchv.backend.asm_emit import AsmEmitter
    from scratchv.backend.inst_scheduler import SchedInst, machine_instrs_from_scheduled

    instructions = [
        SchedInst(0, ".label", target=".entry"),
        SchedInst(1, "li", ["t0", "3"]),
        SchedInst(2, ".label", target=".loop"),
        SchedInst(3, "addi", ["t0", "t0", "-1"]),
        SchedInst(4, "bnez", ["t0", ".loop"]),
        SchedInst(5, "j", [".done"]),
        SchedInst(6, "li", ["t0", "99"]),
        SchedInst(7, ".label", target=".done"),
    ]
    machine = machine_instrs_from_scheduled(instructions)
    cfg = build_cfg(MachineCFGAdapter("roundtrip", machine))
    assert verify_cfg(cfg) == []
    assert any(edge.source == edge.target == ".loop" for edge in cfg.edges)
    assert any(edge.target == ".done" for edge in cfg.edges)
    assert execute_riscv(AsmEmitter(machine).emit()) == execute_riscv("li t0, 0\n")


@pytest.mark.parametrize("outer,inner", [(4, 2), (0, 3), (3, 0)])
@pytest.mark.parametrize("enabled", [False, True])
def test_nested_for_returns_outer_induction_value(execute_riscv, tmp_path, outer, inner, enabled):
    source = f"for i = 0, {outer}\nfor j = 0, {inner}\nendfor\nendfor\nreturn i\n"
    result = CompilerDriver(CompilerConfig(reg_alloc="linear", schedule=enabled)).compile(
        "", str(tmp_path / "nested.s"), dsl_source=source,
    )
    assert result.success, result.errors
    state = execute_riscv("call main\nmv t0, a0\n", suffix=result.output_text)
    assert int.from_bytes(state[:4], "little") == outer


@pytest.mark.parametrize(
    "body",
    [
        "lw t0, 0(a0)\nadd t1, t0, t2\naddi t3, t3, 1\n",
        "add t1, t0, t2\nmul t0, t3, t4\nsw t0, 0(a0)\nlw t1, 0(a0)\n",
        "sw t0, 0(a0)\nbeq t1, t1, done\nli t0, 99\ndone:\nadd t3, t0, t2\n",
        (
            "li t5, 4\nloop:\nlw t0, 0(a0)\nadd t1, t0, t2\naddi t3, t3, 1\n"
            "sw t1, 0(a0)\naddi t5, t5, -1\nbnez t5, loop\n"
        ),
        (
            "addi sp, sp, -16\nsw t0, 0(sp)\nlw t1, 0(sp)\nadd t2, t1, t3\n"
            "mul t4, t5, t6\naddi sp, sp, 16\n"
        ),
        (
            "fcvt.s.w f0, t0\nfcvt.s.w f1, t1\nfdiv.s f2, f0, f1\n"
            "addi t3, t3, 1\nfadd.s f3, f2, f0\nfsw f3, 0(a0)\nfcvt.w.s t4, f3, rtz\n"
        ),
        (
            "fcvt.d.w f0, t0\nfcvt.d.w f1, t1\nfdiv.d f2, f0, f1\n"
            "addi t3, t3, 1\nfadd.d f3, f2, f0\nfsd f3, 0(a0)\nfcvt.w.d t4, f3, rtz\n"
        ),
    ],
)
def test_real_execution_preserves_registers_memory_control_and_float(
    execute_riscv, body
):
    result = schedule_assembly(body, ScheduleConfig(strict=True))
    assert execute_riscv(body) == execute_riscv(result.asm_text)


def test_call_boundary_and_return(execute_riscv):
    body = "lw t0, 0(a0)\nadd t1, t0, t2\naddi t3, t3, 1\ncall helper\nadd t4, t0, t1\n"
    helper = "helper:\nli t0, 17\nret\n"
    result = schedule_assembly(body, ScheduleConfig(strict=True))
    assert "call helper" in result.asm_text
    assert execute_riscv(body, suffix=helper) == execute_riscv(
        result.asm_text, suffix=helper
    )


@pytest.mark.parametrize("seed", range(10))
def test_random_legal_blocks_execute_equivalently(execute_riscv, seed):
    rng = random.Random(seed)
    body = SOURCE.replace("ret\n", "")
    for _ in range(50):
        dst, a, b = (f"t{rng.randrange(7)}" for _ in range(3))
        op = rng.choice(["add", "sub", "mul", "div", "rem", "lw", "sw"])
        if op in {"lw", "sw"}:
            body += f"{op} {dst}, {rng.randrange(8) * 4}(a0)\n"
        else:
            body += f"{op} {dst}, {a}, {b}\n"
    result = schedule_assembly(body, ScheduleConfig(strict=True))
    assert execute_riscv(body) == execute_riscv(result.asm_text)


def test_conv_machine_loop_executes_equivalently(execute_riscv):
    # Small integer 1D convolution inner loop. Both load streams must keep order.
    body = (
        "li t4, 0\nli t5, 4\nmv a3, a0\nconv:\n"
        "lw t0, 0(a3)\nadd t1, t0, t2\nlw t3, 4(a3)\n"
        "mul t6, t1, t3\nadd t4, t4, t6\naddi a3, a3, 4\n"
        "addi t5, t5, -1\nbnez t5, conv\nsw t4, 0(a0)\n"
    )
    result = schedule_assembly(body, ScheduleConfig(strict=True))
    assert result.changed
    assert execute_riscv(body) == execute_riscv(result.asm_text)
