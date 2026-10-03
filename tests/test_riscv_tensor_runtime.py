"""Binary ABI, failure propagation, and optional real RV64 math execution."""

from types import SimpleNamespace
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np
import pytest

from scratchv.runtime.riscv_tensor import (
    FRAME_END, FRAME_HEADER, FRAME_MAGIC, INPUT_BASE, INPUT_CAPACITY,
    RiscVTensorExecutionError, RiscVTensorTimeoutError,
    _checksum, _run_process, build_riscv_tensor, decode_tensor_frame, discover_toolchain,
    input_layout, pack_inputs, run_riscv_tensor, validate_riscv_elf,
)


def spec(name, shape, dtype):
    dtype = np.dtype(dtype)
    return SimpleNamespace(name=name, shape=shape, numpy_dtype=dtype,
                           nbytes=int(np.prod(shape)) * dtype.itemsize)


def frame(payload, status=0):
    return (FRAME_HEADER.pack(FRAME_MAGIC, status, len(payload)) + payload
            + struct.pack("<I", _checksum(payload)) + FRAME_END)


def test_pack_inputs_keeps_int64_ids_and_float32_bits_with_alignment():
    specs = (spec("ids", (1,), "int64"), spec("mask", (2,), "float32"))
    ids = np.array([2**45 + 3], dtype=np.int64)
    mask = np.array([-0.0, np.finfo(np.float32).min], dtype=np.float32)
    offsets, length = input_layout(specs)
    assert offsets == (0, 16)
    assert length == 24
    packed = pack_inputs(specs, {"ids": ids, "mask": mask})
    assert struct.unpack_from("<q", packed)[0] == 2**45 + 3
    assert packed[8:16] == b"\0" * 8
    assert packed[16:] == mask.tobytes()


@pytest.mark.parametrize("inputs", [
    {"ids": np.ones((2,), dtype=np.int64)},
    {"ids": np.ones((1,), dtype=np.int32)},
    {},
    {"ids": np.ones((1,), dtype=np.int64), "extra": np.zeros(1)},
])
def test_pack_rejects_wrong_shape_dtype_and_names(inputs):
    with pytest.raises(ValueError):
        pack_inputs((spec("ids", (1,), "int64"),), inputs)


def test_layout_rejects_duplicate_names_and_loader_overflow():
    with pytest.raises(ValueError, match="unique"):
        input_layout((spec("x", (1,), "int64"), spec("x", (1,), "float32")))
    with pytest.raises(ValueError, match="64 MiB"):
        input_layout((spec("large", (INPUT_CAPACITY + 1,), "uint8"),))


def test_checksum_matches_fnv1a_known_vector():
    assert _checksum(b"hello") == 0x4F9F2CAB


def test_frame_decoding_preserves_every_float32_bit():
    bits = np.array([0, 0x80000000, 0x3F800000, 0x7F800000, 0x7FC12345], dtype=np.uint32)
    result = decode_tensor_frame(frame(bits.tobytes()), spec("y", (5,), "float32"))
    np.testing.assert_array_equal(result.view(np.uint32), bits)


@pytest.mark.parametrize("corruption", ["magic", "length", "checksum", "end", "truncate", "extra"])
def test_frame_rejects_invalid_or_partial_guest_output(corruption):
    data = bytearray(frame(np.array([1.0, 2.0], dtype=np.float32).tobytes()))
    if corruption == "magic":
        data[0] ^= 1
    elif corruption == "length":
        data[12] ^= 1
    elif corruption == "checksum":
        data[FRAME_HEADER.size] ^= 1
    elif corruption == "end":
        data[-1] ^= 1
    elif corruption == "truncate":
        data = data[:-1]
    else:
        data += b"extra"
    with pytest.raises(ValueError):
        decode_tensor_frame(bytes(data), spec("y", (2,), "float32"))


def test_guest_error_cannot_be_misreported_as_valid_empty_output():
    with pytest.raises(RuntimeError, match="failure status 258"):
        decode_tensor_frame(frame(b"", status=258), spec("y", (2,), "float32"))


def _assert_posix_process_terminated(pid, *, starttime=None, timeout=1.0):
    # SIGKILL, pipe closure and orphan reaping are not one atomic event. Both
    # X (dead) and Z (zombie) have stopped executing. Never accept a live state.
    deadline = time.monotonic() + timeout
    state = "unknown"
    while True:
        try:
            status = Path(f"/proc/{pid}/stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
        else:
            # comm may contain spaces and ')'; state follows the final ')'.
            fields = status.rsplit(")", 1)[1].split()
            state = fields[0]
            if starttime is not None and int(fields[19]) != starttime:
                return  # The original child exited; this PID has been reused.
            if state in {"Z", "X"}:
                return
        if time.monotonic() >= deadline:
            raise AssertionError(f"Timed-out compiler child is still running: pid={pid}, state={state}")
        time.sleep(0.01)


@pytest.mark.parametrize("read_error", [FileNotFoundError, ProcessLookupError])
def test_process_exit_check_accepts_reaping_during_proc_read(monkeypatch, read_error):
    checked = []

    def read_reaped_process(self):
        raise read_error("Child was reaped while reading /proc")

    def check_reaped_process(pid, signal):
        checked.append((pid, signal))
        raise ProcessLookupError("Child has exited")

    monkeypatch.setattr(Path, "read_text", read_reaped_process)
    monkeypatch.setattr(os, "kill", check_reaped_process)
    _assert_posix_process_terminated(123)
    assert checked == [(123, 0)]


@pytest.mark.parametrize("state", ["R", "S", "D", "T", "t", "I"])
def test_process_exit_check_rejects_live_child(monkeypatch, state):
    monkeypatch.setattr(Path, "read_text", lambda self: f"123 (worker Z ) name) {state} 0")
    with pytest.raises(AssertionError, match="still running"):
        _assert_posix_process_terminated(123, timeout=0)


@pytest.mark.parametrize("state", ["Z", "X"])
@pytest.mark.parametrize("comm", ["python", "worker a", "worker Z ) name"])
def test_process_exit_check_accepts_dead_states_and_comm_spaces(monkeypatch, state, comm):
    monkeypatch.setattr(Path, "read_text", lambda self: f"123 ({comm}) {state} 0")
    _assert_posix_process_terminated(123, timeout=0)


def test_process_exit_check_waits_for_exit_transition(monkeypatch):
    statuses = iter(["123 (python) R 0", "123 (python) X 0"])
    monkeypatch.setattr(Path, "read_text", lambda self: next(statuses))
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    _assert_posix_process_terminated(123)


def test_process_exit_check_distinguishes_reused_pid(monkeypatch):
    fields = ["R", *(["0"] * 18), "200"]
    monkeypatch.setattr(Path, "read_text", lambda self: "123 (python) " + " ".join(fields))
    _assert_posix_process_terminated(123, starttime=100, timeout=0)
    with pytest.raises(AssertionError, match="still running"):
        _assert_posix_process_terminated(123, starttime=200, timeout=0)


def test_timeout_terminates_only_owned_process_tree(tmp_path):
    # The child inherits the compiler's stdout pipe. Killing only the direct
    # process would leave it alive and keep communicate() blocked indefinitely.
    source = (
        "import subprocess,sys,time,json; from pathlib import Path; "
        "child=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(60)']); "
        "started=int(Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]) "
        "if sys.platform=='linux' else None; "
        "print(json.dumps([child.pid,started]),flush=True); time.sleep(60)"
    )
    # A separate process/group must survive cleanup of the timed-out command.
    unrelated = subprocess.Popen(
        [sys.executable, "-B", "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=os.name != "nt",
    )
    try:
        with pytest.raises(subprocess.TimeoutExpired) as captured:
            _run_process([sys.executable, "-B", "-c", source], cwd=tmp_path, timeout=5)
        assert unrelated.poll() is None, "Timeout cleanup killed an unrelated process"
        pid, started = json.loads(captured.value.output.strip())
        _assert_timed_child_terminated(pid, started)
    finally:
        if unrelated.poll() is None:
            unrelated.kill()
        unrelated.wait(timeout=5)


def _assert_timed_child_terminated(pid, started):
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if handle:
            try:
                exit_code = ctypes.c_ulong()
                assert kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                assert exit_code.value != 259, "Timed-out compiler child is still running"
            finally:
                kernel.CloseHandle(handle)
    else:
        _assert_posix_process_terminated(pid, starttime=started)


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_interruption_stops_actual_owned_process(monkeypatch, tmp_path, interruption):
    communicate = subprocess.Popen.communicate
    interrupted = []

    def interrupt_once(process, *args, **kwargs):
        if not interrupted:
            interrupted.append(process)
            raise interruption(143)
        return communicate(process, *args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "communicate", interrupt_once)
    with pytest.raises(interruption):
        _run_process([sys.executable, "-B", "-c", "import time;time.sleep(60)"],
                     cwd=tmp_path, timeout=30)
    assert len(interrupted) == 1
    assert interrupted[0].poll() is not None


def test_termination_signal_requests_normal_cleanup():
    from scratchv.runtime.riscv_tensor import _terminate_invocation
    with pytest.raises(SystemExit) as caught:
        _terminate_invocation(15, None)
    assert caught.value.code == 143


@pytest.mark.parametrize("field,value", [
    ("class", 1), ("endian", 2), ("machine", 62), ("entry", INPUT_BASE), ("abi", 0),
])
def test_validate_elf_rejects_wrong_target_or_address(tmp_path, field, value):
    header = bytearray(64)
    header[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<H", header, 18, 243)
    struct.pack_into("<Q", header, 24, 0x80000000)
    struct.pack_into("<I", header, 48, 4)
    if field == "class":
        header[4] = value
    elif field == "endian":
        header[5] = value
    elif field == "machine":
        struct.pack_into("<H", header, 18, value)
    elif field == "entry":
        struct.pack_into("<Q", header, 24, value)
    else:
        struct.pack_into("<I", header, 48, value)
    path = tmp_path / "incorrect.elf"
    path.write_bytes(header)
    with pytest.raises(ValueError):
        validate_riscv_elf(path)


@pytest.fixture
def timed_guest(tmp_path, monkeypatch):
    from scratchv.runtime import riscv_tensor as runtime

    elf = tmp_path / "model.elf"
    elf.write_bytes(b"test executable identity")
    executable = SimpleNamespace(
        elf_path=elf, elf_sha256=hashlib.sha256(elf.read_bytes()).hexdigest(),
        inputs=(spec("x", (2,), "float32"),), output=spec("y", (2,), "float32"),
        toolchain=SimpleNamespace(qemu="test-qemu"),
    )
    clock = SimpleNamespace(now=100.0, reads=0)

    def now():
        clock.reads += 1
        return clock.now

    # Replacing the module binding avoids changing pytest's own clock.
    monkeypatch.setattr(runtime, "time", SimpleNamespace(perf_counter=now))
    feed = {"x": np.array([1.0, 2.0], np.float32)}
    return runtime, executable, feed, clock


def test_process_wall_time_excludes_input_preparation_and_output_decoding(tmp_path, monkeypatch, timed_guest):
    runtime, executable, feed, clock = timed_guest
    original_pack, original_decode = runtime.pack_inputs, runtime.decode_tensor_frame
    original_write = Path.write_bytes

    def prepare(*args):
        clock.now += 10.0
        return original_pack(*args)

    def write(path, data):
        # Disk preparation and log persistence must not count as simulation.
        if path.name == "inputs.bin":
            clock.now += 7.0
        elif path.name in ("qemu.stdout", "qemu.stderr"):
            clock.now += 40.0
        return original_write(path, data)

    def decode(*args):
        clock.now += 30.0
        return original_decode(*args)

    observed = []

    def process(command, *, cwd, timeout):
        observed.append(tuple(command))
        assert clock.now == 117.0
        clock.now += 3.25
        (cwd / "uart.bin").write_bytes(frame(feed["x"].tobytes()))
        return subprocess.CompletedProcess(command, 0, b"stdout", b"stderr")

    monkeypatch.setattr(runtime, "pack_inputs", prepare)
    monkeypatch.setattr(runtime, "decode_tensor_frame", decode)
    monkeypatch.setattr(Path, "write_bytes", write)
    monkeypatch.setattr(runtime, "_run_process", process)
    result = run_riscv_tensor(executable, feed, tmp_path / "run", timeout=9)
    assert result.elapsed_s == 3.25 and clock.reads == 2
    assert result.command == observed[0]
    np.testing.assert_array_equal(result.output, feed["x"])


@pytest.mark.parametrize("outcome,message", [
    ("nonzero", r"QEMU failed \(7\): rejected"),
    ("nonzero_valid_frame", r"QEMU failed \(7\): rejected"),
    ("guest", "failure status 3"),
    ("bad_frame", "truncated or missing"),
    ("missing_uart", "without producing UART"),
    ("log_error", "cannot save log"),
])
def test_failed_qemu_attempt_retains_measured_wall_time(tmp_path, monkeypatch, timed_guest, outcome, message):
    runtime, executable, feed, clock = timed_guest
    original_write = Path.write_bytes
    commands = []

    def write(path, data):
        if path.name == "qemu.stdout" and outcome == "log_error":
            clock.now += 90.0
            raise PermissionError("cannot save log")
        return original_write(path, data)

    def process(command, *, cwd, timeout):
        commands.append(tuple(command))
        clock.now += 4.5
        if outcome == "guest":
            (cwd / "uart.bin").write_bytes(frame(b"", status=3))
        elif outcome == "bad_frame":
            (cwd / "uart.bin").write_bytes(b"broken")
        elif outcome in ("nonzero_valid_frame", "log_error"):
            (cwd / "uart.bin").write_bytes(frame(feed["x"].tobytes()))
        code = 7 if outcome.startswith("nonzero") or outcome == "guest" else 0
        return subprocess.CompletedProcess(command, code, b"partial stdout", b"rejected")

    monkeypatch.setattr(Path, "write_bytes", write)
    monkeypatch.setattr(runtime, "_run_process", process)
    with pytest.raises(RiscVTensorExecutionError, match=message) as captured:
        run_riscv_tensor(executable, feed, tmp_path / "run")
    error = captured.value
    assert isinstance(error, RuntimeError)
    assert error.elapsed_s == 4.5 and error.status == "runtime_error"
    assert error.command == commands[0] and clock.reads == 2
    assert error.__cause__ is not None


@pytest.mark.parametrize("outcome,message", [
    ("guest", "failure status 3"), ("nonzero", r"QEMU failed \(7\): rejected"),
    ("bad_frame", "truncated or missing"),
])
def test_qemu_log_failure_keeps_primary_error_and_other_log(
        tmp_path, monkeypatch, timed_guest, outcome, message):
    runtime, executable, feed, clock = timed_guest
    original_write = Path.write_bytes

    def write(path, data):
        if path.name == "qemu.stdout":
            raise PermissionError("stdout is locked")
        return original_write(path, data)

    def process(command, *, cwd, timeout):
        clock.now += 2.5
        if outcome == "guest":
            (cwd / "uart.bin").write_bytes(frame(b"", status=3))
        elif outcome == "bad_frame":
            (cwd / "uart.bin").write_bytes(b"broken")
        return subprocess.CompletedProcess(command, 7 if outcome != "bad_frame" else 0,
                                           b"partial stdout", b"rejected")

    monkeypatch.setattr(Path, "write_bytes", write)
    monkeypatch.setattr(runtime, "_run_process", process)
    run_dir = tmp_path / "run"
    with pytest.raises(RiscVTensorExecutionError, match=message) as captured:
        run_riscv_tensor(executable, feed, run_dir)
    assert "stdout is locked" in str(captured.value)
    assert captured.value.elapsed_s == 2.5 and clock.reads == 2
    assert (run_dir / "qemu.stderr").read_bytes() == b"rejected"


@pytest.mark.parametrize("timeout", [False, True])
def test_compiler_log_failure_keeps_primary_error_and_other_log(tmp_path, monkeypatch, timeout):
    from scratchv.runtime import riscv_tensor as runtime

    original_write = Path.write_bytes

    def write(path, data):
        if path.name == "compiler.stdout":
            raise PermissionError("compiler stdout is locked")
        return original_write(path, data)

    def process(command, **kwargs):
        if timeout:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"], output=b"partial", stderr=b"rejected")
        return subprocess.CompletedProcess(command, 11, b"partial", b"rejected")

    artifact = SimpleNamespace(source="int scratchv_run(void){return 0;}", inputs=(),
                               output=spec("result", (1,), "float32"), function_name="scratchv_run")
    tools = SimpleNamespace(cc=("fixture-clang",), qemu="fixture-qemu", is_zig=False)
    monkeypatch.setattr(Path, "write_bytes", write)
    monkeypatch.setattr(runtime, "_run_process", process)
    error = TimeoutError if timeout else RuntimeError
    message = "compilation exceeded 2" if timeout else r"compilation failed \(11\)"
    with pytest.raises(error, match=message) as captured:
        build_riscv_tensor(artifact, tmp_path / "build", tools, timeout=2)
    assert "compiler stdout is locked" in str(captured.value)
    assert (tmp_path / "build/compiler.stderr").read_bytes() == b"rejected"


@pytest.mark.parametrize("log_error", [False, True])
def test_timeout_reports_actual_time_including_cleanup(tmp_path, monkeypatch, timed_guest, log_error):
    runtime, executable, feed, clock = timed_guest
    original_write = Path.write_bytes
    commands = []

    def process(command, *, cwd, timeout):
        commands.append(tuple(command))
        assert timeout == 2.0
        # _run_process returns its TimeoutExpired only after tree cleanup.
        clock.now += 6.75
        raise subprocess.TimeoutExpired(command, timeout, output=b"partial", stderr=b"timed out")

    def write(path, data):
        if path.name == "qemu.stdout":
            clock.now += 50.0
            if log_error:
                raise PermissionError("cannot preserve timeout log")
        return original_write(path, data)

    monkeypatch.setattr(Path, "write_bytes", write)
    monkeypatch.setattr(runtime, "_run_process", process)
    with pytest.raises(RiscVTensorTimeoutError, match="execution exceeded 2.0 seconds") as captured:
        run_riscv_tensor(executable, feed, tmp_path / "run", timeout=2.0)
    error = captured.value
    assert isinstance(error, TimeoutError)
    assert error.elapsed_s == 6.75 and error.timeout_s == 2.0
    assert error.status == "timeout" and error.command == commands[0]
    assert clock.reads == 2 and isinstance(error.__cause__, subprocess.TimeoutExpired)
    if log_error:
        assert "cannot preserve timeout log" in str(error)
    else:
        assert (tmp_path / "run/qemu.stdout").read_bytes() == b"partial"
    assert (tmp_path / "run/qemu.stderr").read_bytes() == b"timed out"


@pytest.mark.parametrize("failure", ["elf", "input", "stale", "write_input", "spawn"])
def test_attempt_not_started_has_no_simulation_time(tmp_path, monkeypatch, timed_guest, failure):
    runtime, executable, feed, clock = timed_guest
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    if failure == "elf":
        executable.elf_path.write_bytes(b"changed ELF")
    elif failure == "input":
        feed["x"] = np.ones(3, np.float32)
    elif failure == "stale":
        (run_dir / "uart.bin").write_bytes(b"prior evidence")
    elif failure == "write_input":
        (run_dir / "inputs.bin").mkdir()

    started = []

    def process(*args, **kwargs):
        started.append(True)
        assert failure == "spawn"
        raise FileNotFoundError("qemu executable disappeared before launch")

    monkeypatch.setattr(runtime, "_run_process", process)
    with pytest.raises((ValueError, OSError)) as captured:
        run_riscv_tensor(executable, feed, run_dir)
    assert not isinstance(captured.value, (RiscVTensorExecutionError, RiscVTensorTimeoutError))
    assert not hasattr(captured.value, "elapsed_s")
    assert len(started) == int(failure == "spawn")
    assert clock.reads == int(failure == "spawn")


@pytest.mark.integration
def test_real_rv64_musl_math_binary_inputs_and_guest_failure(tmp_path):
    try:
        tools = discover_toolchain()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    if not tools.is_zig:
        pytest.skip("Pinned Zig/musl toolchain is required for the math execution check")
    values = np.array([-255, -128, -3, -1, -.001, 0, .001, .5, 1, 2, 16, 128, 255],
                      dtype=np.float32)
    count = len(values)
    source = """
extern float expf(float); extern float logf(float); extern float powf(float,float);
extern float sinf(float); extern float cosf(float); extern float sqrtf(float);
int scratchv_run(const void *const inputs[], void *output) {
    const float *x = inputs[0]; const long *tag = inputs[1]; float *r = output;
    if (*tag != 35184372088835L) return 9;
    for (int i=0; i<N; ++i) {
        float p = (x[i]<0 ? -x[i] : x[i]) + 0.125f;
        r[i]=expf(-p); r[N+i]=logf(p); r[2*N+i]=powf(p,.37f);
        r[3*N+i]=sinf(x[i]); r[4*N+i]=cosf(x[i]); r[5*N+i]=sqrtf(p);
    }
    return 0;
}
""".replace("N", str(count))
    artifact = SimpleNamespace(
        source=source, inputs=(spec("x", (count,), "float32"), spec("tag", (1,), "int64")),
        output=spec("result", (6, count), "float32"), workspace_bytes=0,
        function_name="scratchv_run", compile_flags=("-fno-strict-aliasing",),
    )
    executable = build_riscv_tensor(artifact, tmp_path / "build", tools, timeout=300)
    assert executable.tool_versions["compiler"]
    assert "QEMU" in executable.tool_versions["qemu"]
    assert executable.tool_versions["math_library"] == "Zig-bundled musl"
    for index, array in enumerate((values, values[::-1].copy())):
        inputs = {"x": array, "tag": np.array([2**45 + 3], dtype=np.int64)}
        result = run_riscv_tensor(executable, inputs, tmp_path / f"run_{index}", timeout=30)
        positive = np.abs(array) + np.float32(.125)
        expected = np.stack([np.exp(-positive), np.log(positive),
                             np.power(positive, np.float32(.37)), np.sin(array),
                             np.cos(array), np.sqrt(positive)])
        np.testing.assert_allclose(result.output, expected, atol=1e-6, rtol=1e-6)
    with pytest.raises(FileExistsError, match="stale QEMU output"):
        run_riscv_tensor(executable, inputs, tmp_path / "run_0")
    inputs["tag"][0] = 0
    with pytest.raises(RuntimeError, match="failure status 9"):
        run_riscv_tensor(executable, inputs, tmp_path / "guest_error", timeout=30)
    executable.elf_path.write_bytes(executable.elf_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed after"):
        run_riscv_tensor(executable, inputs, tmp_path / "changed_elf")
