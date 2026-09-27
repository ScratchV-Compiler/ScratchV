"""LLVM timing contract and standalone execution regressions."""

from copy import deepcopy
import json
import shutil
import subprocess

import pytest

from benchmarks.schedule_analysis import cfg_liveness
from scratchv.backend import llvm_mca
from scratchv.backend.inst_scheduler import parse_instructions, schedule_assembly, ScheduleConfig
from scratchv.standalone.onnx_to_riscv_standalone import RISCVEmitter, rv_addi, rv_bne, rv_j


from scratchv.backend import schedule_semantics as sem


def instruction_samples() -> dict[str, str]:
    samples = {}
    def add(ops, args):
        samples.update({op: f"{op} {args}".rstrip() for op in ops})
    add(sem.INTEGER_BINARY | sem.MULTIPLY | sem.DIVIDE, "t0, t1, t2")
    add(sem.INTEGER_IMMEDIATE, "t0, t1, 1")
    add(["mv", "neg", "not", "seqz", "snez", "sltz", "sgtz"], "t0, t1")
    add(["nop", "ret"], "")
    add(["li", "lui"], "t0, 1")
    add(sem.LOADS | sem.STORES, "t0, 0(a0)")
    add(["flw", "fld", "fsw", "fsd"], "f0, 0(a0)")
    add(sem.BRANCHES, "t0, t1, .Ltarget")
    add(sem.ZERO_BRANCHES, "t0, .Ltarget")
    add(["j"], ".Ltarget")
    add(["jal"], "zero, .Ltarget")
    add(["jr"], "t0")
    add(["jalr"], "zero, t0, 0")
    for suffix in ("s", "d"):
        add([f"{op}.{suffix}" for op in
             ("fadd", "fsub", "fmul", "fdiv", "fmin", "fmax", "fsgnj", "fsgnjn", "fsgnjx")],
            "f0, f1, f2")
        add([f"{op}.{suffix}" for op in ("fsqrt", "fmv", "fabs", "fneg")], "f0, f1")
        add([f"{op}.{suffix}" for op in ("fmadd", "fmsub", "fnmadd", "fnmsub")], "f0, f1, f2, f3")
        add([f"{op}.{suffix}" for op in ("feq", "flt", "fle")], "t0, f1, f2")
        add([f"fclass.{suffix}"], "t0, f0")
        for integer in ("w", "wu"):
            add([f"fcvt.{integer}.{suffix}"], "t0, f0")
            add([f"fcvt.{suffix}.{integer}"], "f0, t0")
    add(["fcvt.s.d", "fcvt.d.s"], "f0, f1")
    add(["fmv.x.w", "fmv.x.s"], "t0, f0")
    add(["fmv.w.x", "fmv.s.x"], "f0, t0")
    return dict(sorted(samples.items()))


def measure(source):
    return llvm_mca.metrics(parse_instructions(source))


@pytest.mark.parametrize("source", [
    "add zero,a0,a1\n" * 128,
    "lw t0,0(a0)\nadd t1,t0,a1\n",
    "lw t0,0(a0)\nmul t1,t0,a1\n",
    "div t0,a0,a1\nadd t1,a2,a3\ndiv t2,a4,a5\n",
    "fadd.s f0,f1,f2\nfadd.s f3,f4,f5\n",
])
def test_report_is_backed_by_unmodified_llvm_output(source):
    result = measure(source)
    raw = subprocess.check_output(
        [llvm_mca.executable_path(), *llvm_mca.FLAGS, *llvm_mca.OPTIONS],
        input=result["source"], text=True, timeout=30)
    assert result["llvm_output"] == json.loads(raw)
    timeline = json.loads(raw)["CodeRegions"][0]["TimelineView"]["TimelineInfo"]
    assert result["issue_cycles"] == [row["CycleIssued"] for row in timeline]
    assert result["critical_path"] is None


@pytest.mark.parametrize("sample", list(instruction_samples().values()))
def test_supported_instruction_forms_have_complete_llvm_timelines(sample):
    result = measure(sample)
    region = result["llvm_output"]["CodeRegions"][0]
    assert region["SummaryView"]["Instructions"] == 1
    assert len(region["InstructionInfoView"]["InstructionList"]) == 1


@pytest.mark.parametrize("sample", ["max t0,t1,t2", "li t0,4096", "li.d f0,1.0"])
def test_unsupported_forms_never_receive_an_invented_latency(sample):
    with pytest.raises(llvm_mca.LLVMError, match="Unsupported instruction"):
        measure(sample)


def test_actual_resource_occupancy_and_consumer_bypass():
    assert measure("add zero,a0,a1\n" * 128)["peak_parallelism"] == 2
    assert measure("lw t0,0(a0)\nadd t1,t0,a1")["issue_cycles"] == [0, 1]
    assert measure("lw t0,0(a0)\nmul t1,t0,a1")["issue_cycles"] == [0, 3]
    assert measure("mul t0,a0,a1\nmul t1,a2,a3")["issue_cycles"] == [0, 1]
    assert measure("div t0,a0,a1\nadd t1,a2,a3\ndiv t2,a4,a5")["cycles"] == 131


def test_bubbles_exclude_drain_and_aggregate_by_issue_span():
    result = measure("lw t0,0(a0)\nmul t1,t0,a1")
    assert (result["cycles"], result["issue_span"], result["bubbles"], result["drain_cycles"]) == (6, 4, 2, 2)
    stats = schedule_assembly("lw t0,0(a0)\nmul t1,t0,a1\nnext:\nadd t0,a0,a1\n").stats
    combined = stats["metrics_before"]
    assert combined["bubble_ratio"] == 2 / 5
    assert combined["issue_span"] == 5
    assert combined["critical_path_max"] is None
    assert "Critical path: N/A" in schedule_assembly("add t0,a0,a1").report()


@pytest.mark.parametrize("strict", [False, True])
def test_missing_tool_is_an_error_not_a_fallback(strict):
    with pytest.raises(llvm_mca.LLVMError, match="not found"):
        schedule_assembly("add t0,a0,a1", ScheduleConfig(strict=strict, llvm_mca="/missing/llvm-mca"))


def test_disabled_scheduler_does_not_require_llvm(tmp_path):
    from scratchv.compiler import CompilerConfig, CompilerDriver
    driver = CompilerDriver(CompilerConfig(llvm_mca="/missing/llvm-mca"))
    assert driver._run_asm_passes("add t0,a0,a1", []) == "add t0,a0,a1"


def test_explicit_executable_overrides_environment(monkeypatch):
    path = llvm_mca.executable_path()
    monkeypatch.setenv("LLVM_MCA", "/missing/llvm-mca")
    result = schedule_assembly("add t0,a0,a1", ScheduleConfig(llvm_mca=path))
    assert result.stats["model_metadata"]["executable"] == path


@pytest.mark.parametrize("version", ["18.1.3", "18.1.8", "19.1.0", "22.1.8"])
def test_reports_actual_llvm_version_without_pinning_release(monkeypatch, version):
    monkeypatch.setattr(llvm_mca, "_run", lambda *args: f"LLVM (http://llvm.org/):\n  LLVM version {version}\n")
    monkeypatch.setattr(llvm_mca, "_identity", lambda path: (version, 0, 0))
    monkeypatch.setattr(llvm_mca, "executable_path", lambda *args: "/tools/llvm-mca")
    metadata = llvm_mca.model_metadata()
    assert metadata["llvm_version"] == version
    assert metadata["version_output"] == f"LLVM version {version}"
    assert f"llvmorg-{version}/" in metadata["source_url"]


def test_unrecognized_version_output_is_rejected(monkeypatch):
    monkeypatch.setattr(llvm_mca, "_run", lambda *args: "not an LLVM tool\n")
    with pytest.raises(llvm_mca.LLVMError, match="Unrecognized LLVM version"):
        llvm_mca._version(("invalid-version", 0, 0))


def test_default_executable_is_not_version_pinned(monkeypatch):
    monkeypatch.delenv("LLVM_MCA", raising=False)
    monkeypatch.setattr(llvm_mca.shutil, "which", lambda name: f"/tools/{name}")
    assert llvm_mca.executable_path() == "/tools/llvm-mca"


@pytest.mark.parametrize("failure", ["diagnostic", "exit", "timeout"])
def test_subprocess_failures_cannot_produce_success(monkeypatch, failure):
    def run(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired("llvm-mca", 30)
        return subprocess.CompletedProcess(args[0], 1 if failure == "exit" else 0,
                                           "{}", "unsupported instruction" if failure == "diagnostic" else "")
    monkeypatch.setattr(llvm_mca.subprocess, "run", run)
    with pytest.raises(llvm_mca.LLVMError):
        llvm_mca._run(["llvm-mca"])


@pytest.mark.parametrize("returncode,extra", [(0, ""), (1, ""), (0, "\nerror: unsupported instruction")])
def test_only_static_return_notice_is_accepted(monkeypatch, returncode, extra):
    diagnostic = ("warning: found a return instruction in the input assembly sequence.\n"
                  "note: program counter updates are ignored.\n")
    monkeypatch.setattr(llvm_mca.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], returncode, "{}", diagnostic + extra))
    if returncode or extra:
        with pytest.raises(llvm_mca.LLVMError):
            llvm_mca._run(["llvm-mca"], "ret\n")
    else:
        assert llvm_mca._run(["llvm-mca"], "ret\n") == "{}"


@pytest.mark.parametrize("failure", ["omitted", "expanded", "incomplete", "cpu", "json"])
def test_partial_or_wrong_analyses_are_rejected(monkeypatch, failure):
    original = measure("lw t0,0(a0)\nmul t1,t0,a1")["llvm_output"]
    raw = deepcopy(original)
    region = raw["CodeRegions"][0]
    if failure == "omitted":
        region["TimelineView"]["TimelineInfo"].pop()
    elif failure == "expanded":
        region["SummaryView"]["TotaluOps"] += 1
    elif failure == "incomplete":
        region["TimelineView"]["TimelineInfo"][0]["CycleExecuted"] = -1
    elif failure == "cpu":
        raw["TargetInfo"]["CPUName"] = "other"
    else:
        raw = {}
    monkeypatch.setattr(llvm_mca, "_analyze", lambda *args: raw)
    with pytest.raises(llvm_mca.LLVMError, match="Invalid llvm-mca analysis"):
        measure("lw t0,0(a0)\nmul t1,t0,a1")


def test_cached_raw_outputs_are_not_mutable_shared_state():
    first = measure("add t0,a0,a1")
    first["llvm_output"]["CodeRegions"].clear()
    assert measure("add t0,a0,a1")["llvm_output"]["CodeRegions"]


def test_numeric_targets_are_symbolized_only_with_machine_word_layout():
    emitter = RISCVEmitter()
    emitter.emit(rv_addi(5, 0, 1))
    emitter.emit(rv_bne(5, 0, 8))  # Unnamed target at instruction 3.
    emitter.emit(rv_addi(6, 0, 2))
    emitter.emit(rv_addi(7, 0, 3))
    source = emitter.disassemble(symbolic=True)
    assert "bne t0, zero, .Lsv_3" in source
    assert ".Lsv_3:" in source
    assert schedule_assembly(source).stats["coverage_ratio"] == 1
    # Arbitrary user assembly still has no known byte layout.
    assert schedule_assembly(emitter.disassemble()).stats["coverage_ratio"] == 0


def test_undefined_or_out_of_range_branch_never_silently_resolves():
    emitter = RISCVEmitter()
    emitter.emit_jump(rv_j, "missing")
    with pytest.raises(ValueError, match="Undefined"):
        emitter.resolve_fixups()
    emitter = RISCVEmitter()
    emitter.emit(rv_bne(5, 0, 2))
    with pytest.raises(ValueError, match="fixed-width"):
        emitter.disassemble(symbolic=True)


def test_cfg_liveness_follows_loop_backedge_and_reports_unknown_effects():
    source = "li t0,3\nloop:\nadd t1,t1,a0\naddi t0,t0,-1\nbnez t0,loop\nsw t1,0(a1)\nret\n"
    report = cfg_liveness(source)
    assert report["status"] == "completed"
    assert report["iterations"] > 1
    assert "x10" in report["instructions"][3]["live_out"]  # Needed by next iteration.
    assert cfg_liveness("max t0,t1,t2\nret")["status"] == "not_modeled"


def test_standalone_permutation_preserves_machine_encoding(tmp_path):
    if not shutil.which("clang") or not shutil.which("ld.lld"):
        pytest.skip("Requires RISC-V assembler and linker")
    from benchmarks.cnn_schedule_execution import assemble_listing
    from scratchv.standalone.onnx_to_riscv_standalone import rv_lw, rv_add
    emitter = RISCVEmitter()
    emitter.emit(rv_lw(5, 10, 0))
    emitter.emit(rv_add(6, 5, 7))
    emitter.emit(rv_addi(28, 28, 1))
    emitter.emit_jump(rv_j, "done")
    emitter.label("done")
    emitter.resolve_fixups()
    emitter.schedule()
    asm = tmp_path / "scheduled.s"
    asm.write_text(emitter.disassemble(symbolic=True))
    assert assemble_listing(asm, tmp_path) == emitter.to_bytes()


def test_full_cnn_baseline_crash_is_not_an_equivalence_pass(tmp_path, monkeypatch):
    import subprocess
    from benchmarks import cnn_schedule_execution as execution
    monkeypatch.setattr(execution.shutil, "which", lambda name: f"/tools/{name}")
    monkeypatch.setattr(execution.subprocess, "check_output", lambda *args, **kwargs: "QEMU test\n")
    runs = []

    def run(command, **kwargs):
        runs.append(command)
        crashed = command[0].endswith("qemu-riscv32")
        return subprocess.CompletedProcess(command, -11 if crashed else 0, b"", b"SIGSEGV" if crashed else b"")

    monkeypatch.setattr(execution.subprocess, "run", run)
    result = execution.execute_pair(tmp_path / "before.bin", tmp_path / "after.bin",
                                    {"input_elements": 1, "output_elements": 1, "workspace_bytes": 4}, tmp_path)
    assert result["status"] == "baseline_failed"
    assert result["equivalence_verified"] is False
    assert result["samples"] == []
    assert sum(command[0].endswith("qemu-riscv32") for command in runs) == 1


def test_full_cnn_register_mismatch_fails_even_if_outputs_match(tmp_path, monkeypatch):
    import subprocess
    from benchmarks import cnn_schedule_execution as execution
    monkeypatch.setattr(execution.shutil, "which", lambda name: f"/tools/{name}")
    monkeypatch.setattr(execution.subprocess, "check_output", lambda *args, **kwargs: "tool test\n")

    def run(command, **kwargs):
        data = bytes(648)
        if command[0].endswith("qemu-riscv32") and "after" in command[-1]:
            data = b"\x01" + data[1:]
        return subprocess.CompletedProcess(command, 0, data, b"")

    monkeypatch.setattr(execution.subprocess, "run", run)
    result = execution.execute_pair(tmp_path / "before.bin", tmp_path / "after.bin",
                                    {"input_elements": 1, "output_elements": 1, "workspace_bytes": 4}, tmp_path)
    assert result["status"] == "failed"
    assert all(not row["equal"] for row in result["samples"])
    assert all(row["before"]["output_q16"] == row["after"]["output_q16"] for row in result["samples"])
