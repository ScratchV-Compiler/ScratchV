"""Real small-guest checks for external-runtime hart ownership and barriers."""
from dataclasses import replace
import os
import shutil
import subprocess

import numpy as np
import pytest

from scratchv.backend.tensor_c_codegen import TensorCArtifact, TensorCCodegen, TensorSpec
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, Value
from scratchv.runtime.riscv_external import build_external, run_external
from scratchv.runtime.riscv_harts import resolve_harts
from scratchv.runtime.riscv_tensor import RiscVToolchain, discover_toolchain
from scratchv.runtime.weight_bundle import MIB, plan_guest_memory, write_weight_bundle


SOURCE = r"""
typedef unsigned long usize;
typedef unsigned long long u64;
extern void sv_parallel_for(usize, void (*)(void *, usize, usize), void *);
struct context { u64 *out; u64 value; };
static void record(void *opaque, usize begin, usize end) {
    struct context *context=opaque;
    usize hart, stack, fcsr;
    __asm__ volatile("csrr %0, mhartid" : "=r"(hart));
    __asm__ volatile("mv %0, sp" : "=r"(stack));
    volatile float left=1.5f, right=2.0f, add=0.25f;
    union { float f; unsigned bits; } result;
    result.f=__builtin_fmaf(left, right, add);
    __asm__ volatile("csrr %0, fcsr" : "=r"(fcsr));
#if SCRATCHV_HARTS > 1
    usize mie, mstatus;
    __asm__ volatile("csrr %0, mie" : "=r"(mie));
    __asm__ volatile("csrr %0, mstatus" : "=r"(mstatus));
    if (!(mie & 8) || (mstatus & 8)) fcsr |= 0x80000000UL;
#endif
    for (usize i=begin; i<end; ++i) context->out[i]+=context->value;
    context->out[65+hart]=stack;
    context->out[69+hart]=result.bits;
    context->out[73+hart]=fcsr;
}
int scratchv_run_external(const void *const inputs[], const void *const weights[],
                         void *workspace, usize bytes, void *output) {
    u64 *out=output;
    for (usize i=0; i<77; ++i) out[i]=0;
    struct context context={out, 0};
    for (usize pass=1; pass<=32; ++pass) {
        context.value=pass;
        sv_parallel_for(65, record, &context);
    }
    sv_parallel_for(0, record, &context);
    context.value=1000;
    sv_parallel_for(1, record, &context);
    context.value=2000;
    sv_parallel_for(3, record, &context);
    return 0;
}
"""


def artifact(source=SOURCE):
    return TensorCArtifact(source, (), TensorSpec("out", DataType.INT64, (77,)),
                           0, 0, function_name="scratchv_run_external",
                           constant_storage="external")


@pytest.fixture
def llvm_nm():
    executable = shutil.which("llvm-nm") or shutil.which("llvm-nm-18")
    if not executable:
        pytest.skip("ELF function placement checks require llvm-nm")
    return executable


def _defined_symbols(llvm_nm, elf):
    result = subprocess.run(
        [llvm_nm, "--defined-only", "--print-size", "--format=posix", str(elf)],
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    return {fields[0]: (int(fields[2], 16), int(fields[3], 16))
            for line in result.stdout.splitlines()
            if len(fields := line.split()) == 4}


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="explicit RV64 tools required")
@pytest.mark.parametrize("harts", [1, 4])
def test_expf_stays_in_one_guest_page_when_generated_code_grows(tmp_path, llvm_nm, harts):
    data = np.linspace(-8, 8, 68, dtype=np.float32).reshape(4, 17)
    expected = np.exp(data - data.max(axis=-1, keepdims=True))
    expected /= expected.sum(axis=-1, keepdims=True)
    outputs = []
    for negation_pairs in (0, 8):
        directory = tmp_path / f"negation_pairs_{negation_pairs}"
        directory.mkdir()
        x = Value("x", shape=data.shape)
        builder = IRBuilder()
        builder.new_function("main", [x])
        builder.new_block("entry")
        value = x
        for _ in range(negation_pairs):
            # Separate noinline kernels vary code placement without changing
            # Softmax inputs, as real graph and MatMul tile changes can do.
            value = builder.neg(builder.neg(value))
        builder.ret(builder.softmax(value))
        model = TensorCCodegen(builder.program, constant_storage="external",
                               kernel_calls=True).generate()
        write_weight_bundle(model.external_weights, model.external_initializers,
                            directory / "weights")
        executable = build_external(model, directory / "weights", directory / "build",
                                    discover_toolchain(), harts=harts)
        address, size = _defined_symbols(llvm_nm, executable.elf_path)["expf"]
        # Check the linked function, not linker-script text. A hot expf that
        # crosses a guest page loses direct TCG block chaining across that boundary.
        assert size > 0
        assert address % 4096 == 0, f"expf is not page aligned: {address:#x}"
        assert address // 4096 == (address + size - 1) // 4096
        actual, report = run_external(executable, {"x": data}, directory / "run", timeout=15)
        assert report["passed"] and report["harts"] == harts
        np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)
        outputs.append(actual.view(np.uint32))
    np.testing.assert_array_equal(outputs[0], outputs[1])


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="explicit RV64 tools required")
def test_external_does_not_retain_unused_expf(tmp_path, llvm_nm):
    write_weight_bundle((), (), tmp_path / "weights")
    executable = build_external(artifact(), tmp_path / "weights", tmp_path / "build",
                                discover_toolchain(), harts=1)
    assert "expf" not in _defined_symbols(llvm_nm, executable.elf_path)
    # This tiny guest fits in the startup page. An empty aligned output section
    # must not introduce a page-sized hole when no expf input section is linked.
    loaded_end = max(segment["address"] + segment["memsz"]
                     for segment in executable.evidence["segments"])
    assert loaded_end - executable.layout["code_base"] < 4096


@pytest.mark.parametrize("harts", [True, False, 0, 3, 5, 12, 1.0, "2", None])
def test_invalid_hart_counts_fail_before_building(tmp_path, harts):
    with pytest.raises(ValueError, match="harts"):
        resolve_harts(harts)
    with pytest.raises(ValueError, match="harts"):
        plan_guest_memory(harts=harts)
    with pytest.raises(ValueError, match="harts"):
        build_external(artifact(), tmp_path / "absent", tmp_path / "build",
                       RiscVToolchain(("unused",), "unused"), harts=harts)
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize("harts", [1, 2, 4])
def test_each_hart_has_a_disjoint_reserved_stack(harts):
    layout=plan_guest_memory(harts=harts)
    assert layout["stack_top"]-layout["stack_bottom"] == harts*MIB
    assert layout["code_limit"] == layout["stack_bottom"]
    assert layout["input_base"] == layout["stack_top"]
    if harts == 1:
        assert layout == plan_guest_memory()
    with pytest.raises(ValueError, match="stack"):
        plan_guest_memory(code_capacity=harts*MIB, harts=harts)


@pytest.mark.parametrize("count,expected", [(1, 1), (2, 2), (3, 2), (4, 4), (12, 4)])
def test_auto_harts_uses_available_affinity_count_not_cpu_ids_or_omp(monkeypatch, count, expected):
    calls = []
    monkeypatch.setattr(os, "sched_getaffinity",
                        lambda pid: calls.append(pid) or {7 + 2*i for i in range(count)},
                        raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setattr(os, "cpu_count", lambda: pytest.fail("affinity must take precedence"))
    monkeypatch.setattr(os, "process_cpu_count",
                        lambda: pytest.fail("affinity must take precedence"), raising=False)
    assert resolve_harts("auto") == expected
    assert calls == [0]


@pytest.mark.parametrize("harts", [1, 2, 4])
def test_manual_harts_never_probes_host_cpu(monkeypatch, harts):
    def unexpected(*args):
        pytest.fail("manual hart selection must not inspect CPU availability")
    for name in ("sched_getaffinity", "process_cpu_count", "cpu_count"):
        monkeypatch.setattr(os, name, unexpected, raising=False)
    assert resolve_harts(harts) == harts


@pytest.mark.parametrize("affinity", ["missing", "error", "empty"])
def test_auto_harts_falls_back_to_process_cpu_count(monkeypatch, affinity):
    if affinity == "missing":
        monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    elif affinity == "empty":
        monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(), raising=False)
    else:
        def unavailable(_):
            raise OSError("affinity unavailable")
        monkeypatch.setattr(os, "sched_getaffinity", unavailable, raising=False)
    monkeypatch.setattr(os, "process_cpu_count", lambda: 3, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: pytest.fail("process CPU count is available"))
    assert resolve_harts("auto") == 2


@pytest.mark.parametrize("count,expected", [(12, 4), (None, 1)])
def test_auto_harts_uses_cpu_count_when_process_cpu_count_is_unavailable(monkeypatch, count, expected):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: count)
    assert resolve_harts("auto") == expected


@pytest.mark.parametrize("unavailable", ["unknown", "error"])
def test_auto_harts_is_conservative_when_process_cpu_count_is_unknown(monkeypatch, unavailable):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    def count():
        if unavailable == "error":
            raise OSError("CPU count unavailable")
        return None
    monkeypatch.setattr(os, "process_cpu_count", count, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: pytest.fail("do not overrule process CPU count"))
    assert resolve_harts("auto") == 1


def test_auto_harts_must_be_resolved_before_planning_memory():
    with pytest.raises(ValueError, match="harts"):
        plan_guest_memory(harts="auto")


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="explicit RV64 tools required")
@pytest.mark.parametrize("harts", [1, 2, 4, "auto"])
def test_real_harts_complete_repeated_uneven_and_empty_tasks(tmp_path, monkeypatch, harts):
    expected_harts = resolve_harts(harts)
    model=artifact()
    write_weight_bundle((), (), tmp_path / "weights")
    executable=build_external(model, tmp_path / "weights", tmp_path / "build",
                              discover_toolchain(), harts=harts)
    assert type(executable.harts) is int and executable.harts == expected_harts
    if harts == "auto":
        # The built ELF and its layout stay fixed even if availability changes.
        monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0}, raising=False)
    output, report=run_external(executable, {}, tmp_path / "run", timeout=15)
    expected=np.full(65, sum(range(1, 33)), dtype=np.int64)
    expected[0]+=1000
    expected[:3]+=2000
    np.testing.assert_array_equal(output[:65], expected)
    for hart in range(expected_harts):
        top=executable.layout["stack_top"]-hart*MIB
        assert top-MIB <= output[65+hart] < top
        assert output[69+hart] == 0x40500000  # exactly 3.25f
        assert output[73+hart] == 0  # initialized RNE, exact operations
    assert report["passed"] and report["harts"] == expected_harts
    assert report["command"][report["command"].index("-smp")+1] == str(expected_harts)
    if expected_harts > 1:
        assert "tcg,thread=multi" in report["command"]
    changed=replace(executable, harts=2 if expected_harts == 1 else 1)
    with pytest.raises(ValueError, match="hart"):
        run_external(changed, {}, tmp_path / "mismatch", timeout=1)
    assert not (tmp_path / "mismatch").exists()


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="explicit RV64 tools required")
def test_worker_trap_is_reported_by_master_without_waiting_for_timeout(tmp_path):
    source=SOURCE.replace(
        "struct context *context=opaque;",
        'struct context *context=opaque; usize id; '
        '__asm__ volatile("csrr %0, mhartid" : "=r"(id)); '
        'if (id == 1) { for (volatile usize delay=0; delay<5000; ++delay) {} '
        '__asm__ volatile("unimp"); }')
    write_weight_bundle((), (), tmp_path / "weights")
    executable=build_external(artifact(source), tmp_path / "weights", tmp_path / "build",
                              discover_toolchain(), harts=2)
    with pytest.raises(RuntimeError, match="guest failed"):
        run_external(executable, {}, tmp_path / "run", timeout=15)


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="explicit RV64 tools required")
def test_four_hart_worker_timeout_reaps_owned_qemu_and_saves_failure(tmp_path, monkeypatch):
    import json
    from scratchv.runtime import riscv_external as runtime

    source = SOURCE.replace(
        "struct context *context=opaque;",
        'struct context *context=opaque; usize id; '
        '__asm__ volatile("csrr %0, mhartid" : "=r"(id)); '
        'if (id == 3) for (;;) __asm__ volatile("wfi");')
    write_weight_bundle((), (), tmp_path / "weights")
    executable = build_external(artifact(source), tmp_path / "weights", tmp_path / "build",
                                discover_toolchain(), harts=4)
    original = runtime.subprocess.Popen
    processes = []

    def launch(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    try:
        with pytest.raises(TimeoutError, match="RV64 forward exceeded"):
            run_external(executable, {}, tmp_path / "run", timeout=1)
        assert len(processes) == 1 and processes[0].poll() is not None
        report = json.loads((tmp_path / "run/run.json").read_text())
        assert report["status"] == "FAIL" and report["passed"] is False
        assert report["harts"] == 4 and report["guest_completed"] is False
        assert report["error"].startswith("TimeoutError: RV64 forward exceeded")
        assert report["qemu_wall_s"] >= 1
    finally:
        for process in processes:
            if process.poll() is None:
                runtime._stop_process_tree(process)
                process.wait(timeout=5)


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="explicit RV64 tools required")
@pytest.mark.parametrize("harts", [2, 4])
def test_doorbells_before_wait_coalesce_and_spurious_wakes_do_not_repeat_tasks(tmp_path, harts):
    # Odd passes keep hart 0 in its callback until every worker has signalled,
    # exercising pending/coalesced notifications before the master's join.
    # Even passes delay workers after hart 0's callback, exercising sleeping
    # joins. Extra rings between tasks wake idle workers without a new task.
    source = SOURCE.replace(
        "struct context { u64 *out; u64 value; };",
        r"""
static void ring(usize hart) {
    __asm__ volatile("fence rw,ow" ::: "memory");
    ((volatile unsigned *)0x02000000UL)[hart]=1;
}
struct context { u64 *out; u64 value; u64 ready[4]; };
""").replace(
        '    volatile float left=1.5f, right=2.0f, add=0.25f;',
        r"""
    if (context->value <= 32) {
        if (context->value & 1) {
            if (hart) {
                ring(0); ring(0); ring(0);
                __atomic_store_n(&context->ready[hart], context->value, __ATOMIC_RELEASE);
            } else {
                for (usize worker=1; worker<SCRATCHV_HARTS; ++worker)
                    while (__atomic_load_n(&context->ready[worker], __ATOMIC_ACQUIRE)
                           != context->value) {}
            }
        } else {
            if (!hart)
                __atomic_store_n(&context->ready[0], context->value, __ATOMIC_RELEASE);
            else {
                while (__atomic_load_n(&context->ready[0], __ATOMIC_ACQUIRE)
                       != context->value) {}
                for (volatile usize delay=0; delay<5000; ++delay) {}
            }
        }
    }
    volatile float left=1.5f, right=2.0f, add=0.25f;
""").replace(
        "        context.value=pass;",
        r"""
        for (usize repeat=0; repeat<3; ++repeat) {
            for (usize worker=1; worker<SCRATCHV_HARTS; ++worker) ring(worker);
            for (volatile usize delay=0; delay<5000; ++delay) {}
        }
        context.value=pass;
""")
    write_weight_bundle((), (), tmp_path / "weights")
    executable = build_external(artifact(source), tmp_path / "weights", tmp_path / "build",
                                discover_toolchain(), harts=harts)
    output, report = run_external(executable, {}, tmp_path / "run", timeout=15)
    expected = np.full(65, sum(range(1, 33)), dtype=np.int64)
    expected[0] += 1000
    expected[:3] += 2000
    np.testing.assert_array_equal(output[:65], expected)
    assert report["passed"] and report["harts"] == harts
    np.testing.assert_array_equal(output[73:73+harts], 0)
