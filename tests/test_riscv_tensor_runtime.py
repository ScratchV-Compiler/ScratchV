"""Binary ABI, failure propagation, and optional real RV64 math execution."""

from types import SimpleNamespace
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

from scratchv.runtime.riscv_tensor import (
    FRAME_END, FRAME_HEADER, FRAME_MAGIC, INPUT_BASE, INPUT_CAPACITY,
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


def _assert_posix_process_terminated(pid):
    # A killed Linux orphan can remain a zombie until PID 1 reaps it. Reaping
    # can also happen while /proc is being read, so do not use exists() first.
    try:
        status = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    else:
        assert status.split()[2] == "Z", "Timed-out compiler child is still running"


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


@pytest.mark.parametrize("state", ["R", "S"])
def test_process_exit_check_rejects_live_child(monkeypatch, state):
    monkeypatch.setattr(Path, "read_text", lambda self: f"123 (python) {state} 0")
    with pytest.raises(AssertionError, match="still running"):
        _assert_posix_process_terminated(123)


def test_timeout_terminates_only_owned_process_tree(tmp_path):
    # The child inherits the compiler's stdout pipe. Killing only the direct
    # process would leave it alive and keep communicate() blocked indefinitely.
    source = (
        "import subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(60)']); "
        "print(child.pid,flush=True); time.sleep(60)"
    )
    with pytest.raises(subprocess.TimeoutExpired) as captured:
        _run_process([sys.executable, "-B", "-c", source], cwd=tmp_path, timeout=5)
    pid = int(captured.value.output.strip())
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
        _assert_posix_process_terminated(pid)


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
