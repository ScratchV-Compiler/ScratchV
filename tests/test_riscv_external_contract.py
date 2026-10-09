"""Independent ELF, completion protocol and local QMP contract checks.

No mocked numerical execution or complete-model claim is made by these tests.
"""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import socket
import struct
import threading
import time
from types import SimpleNamespace

import pytest

from scratchv.backend.tensor_c_codegen import TensorCCodegen
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import Value
from scratchv.runtime import riscv_external as runtime
from scratchv.runtime import weight_transport as transport
from scratchv.runtime.riscv_tensor import RiscVToolchain
from scratchv.runtime.weight_bundle import plan_guest_memory, write_weight_bundle


def _elf(path, layout, *, entry=None, machine=243, flags=5, segments=None):
    base = layout["code_base"]
    if segments is None:
        segments = [(1, 5, 256, base, base, 8, 16, 1)]
    data = bytearray(512)
    ident = b"\x7fELF\x02\x01\x01" + bytes(9)
    struct.pack_into("<16sHHIQQQIHHHHHH", data, 0, ident, 2, machine, 1,
                     base if entry is None else entry, 64, 0, flags, 64, 56,
                     len(segments), 0, 0, 0)
    for index, segment in enumerate(segments):
        struct.pack_into("<IIQQQQQQ", data, 64 + 56 * index, *segment)
    data[256:264] = b"\x13\x00\x00\x00" * 2
    path.write_bytes(data)
    return bytes(data)


def test_external_elf_binds_validated_bytes_and_bss_extent(tmp_path):
    layout = plan_guest_memory()
    binary = tmp_path / "model.elf"
    expected = _elf(binary, layout)
    evidence = runtime.validate_external_elf(binary, layout)
    assert evidence["sha256"] == hashlib.sha256(expected).hexdigest()
    assert evidence["bytes"] == len(expected)
    assert evidence["entry"] == layout["code_base"]
    assert evidence["segments"] == [dict(address=layout["code_base"], filesz=8,
                                         memsz=16, flags=5)]


@pytest.mark.parametrize("failure", ["machine", "soft_float", "entry_bss", "entry_data",
                                    "bss_stack", "outside_ram", "virtual_physical",
                                    "filesz_memsz", "filesz_file", "overlap", "alignment",
                                    "no_load", "table_truncated", "not_elf"])
def test_external_elf_rejects_invalid_load_contract(tmp_path, failure):
    layout = plan_guest_memory()
    binary = tmp_path / "model.elf"
    base = layout["code_base"]
    options = {}
    segment = [1, 5, 256, base, base, 8, 16, 1]
    segments = [segment]
    if failure == "machine":
        options["machine"] = 62
    elif failure == "soft_float":
        options["flags"] = 1
    elif failure == "entry_bss":
        options["entry"] = base + 12
    elif failure == "entry_data":
        segment[1] = 6
    elif failure == "bss_stack":
        segment[6] = layout["code_limit"] - base + 1
    elif failure == "outside_ram":
        segment[3] = segment[4] = base - 1
    elif failure == "virtual_physical":
        segment[3] += 64
    elif failure == "filesz_memsz":
        segment[5] = 17
    elif failure == "filesz_file":
        segment[2] = 511
    elif failure == "overlap":
        segments.append([1, 6, 320, base + 8, base + 8, 8, 16, 1])
    elif failure == "alignment":
        segment[7] = 3
    elif failure == "no_load":
        segment[0] = 4
    _elf(binary, layout, segments=segments, **options)
    if failure == "table_truncated":
        data = bytearray(binary.read_bytes())
        struct.pack_into("<Q", data, 32, len(data) - 1)
        binary.write_bytes(data)
    elif failure == "not_elf":
        binary.write_bytes(b"not an ELF")
    with pytest.raises(ValueError):
        runtime.validate_external_elf(binary, layout)


def test_external_elf_limit_is_strictly_less_than_100_mb(tmp_path):
    binary = tmp_path / "model.elf"
    with binary.open("wb") as stream:
        stream.truncate(runtime.MAX_ELF_BYTES)
    with pytest.raises(ValueError, match="smaller"):
        runtime.validate_external_elf(binary, plan_guest_memory())


@pytest.mark.parametrize("offset,fmt,value", [(20, "<I", 0), (52, "<H", 63),
                                             (32, "<Q", 32), (48, "<I", 13)])
def test_malformed_elf_header_cannot_pass_build_gate(tmp_path, offset, fmt, value):
    path = tmp_path / "invalid.elf"
    layout = plan_guest_memory()
    data = bytearray(_elf(path, layout))
    struct.pack_into(fmt, data, offset, value)
    path.write_bytes(data)
    with pytest.raises(ValueError):
        runtime.validate_external_elf(path, layout)


@pytest.mark.parametrize("status", [1, 2, 3, 0x80000005])
def test_completion_rejects_guest_status_even_with_full_length(status):
    with pytest.raises(RuntimeError, match="guest failed"):
        runtime.decode_completion(runtime.FRAME.pack(runtime.MAGIC, status, 4096), 4096)


@pytest.mark.parametrize("mutation", ["empty", "short", "trailing", "magic", "length"])
def test_completion_requires_one_exact_success_frame(mutation):
    frame = runtime.FRAME.pack(runtime.MAGIC, 0, 4096)
    frame = {"empty": b"", "short": frame[:-1], "trailing": frame + b"!",
             "magic": b"BADMAGIC" + frame[8:],
             "length": runtime.FRAME.pack(runtime.MAGIC, 0, 4095)}[mutation]
    with pytest.raises(ValueError):
        runtime.decode_completion(frame, 4096)


def test_completion_accepts_exact_success_without_converting_output():
    assert runtime.decode_completion(runtime.FRAME.pack(runtime.MAGIC, 0, 4096), 4096) is None


@contextmanager
def _qmp_server(respond):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(5)
    errors = []

    def serve():
        try:
            with listener.accept()[0] as connection:
                connection.settimeout(5)
                with connection.makefile("rwb", buffering=0) as stream:
                    def send(value):
                        stream.write(json.dumps(value).encode("utf-8") + b"\r\n")
                    send({"QMP": {"version": {}, "capabilities": []}})
                    capability = json.loads(stream.readline())
                    assert capability["execute"] == "qmp_capabilities"
                    send({"return": {}, "id": capability["id"]})
                    request = json.loads(stream.readline())
                    respond(request, send)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        yield listener.getsockname()[1]
    finally:
        worker.join(timeout=6)
        listener.close()
        assert not worker.is_alive(), "Test QMP server did not terminate"
        if errors:
            raise errors[0]


def test_qmp_ignores_events_and_matches_command_id():
    def respond(request, send):
        assert request["execute"] == "query-status"
        send({"event": "STOP", "data": {}})
        send({"return": {"stale": True}, "id": request["id"] - 1})
        send({"return": {"status": "paused"}, "id": request["id"]})
    with _qmp_server(respond) as port:
        client = runtime.QMP(port, time.monotonic() + 5)
        try:
            assert client.command("query-status") == {"status": "paused"}
        finally:
            client.close()


def test_qmp_error_response_is_not_a_successful_memory_dump():
    def respond(request, send):
        assert request["execute"] == "human-monitor-command"
        send({"error": {"class": "GenericError", "desc": "cannot write output"},
              "id": request["id"]})
    with _qmp_server(respond) as port:
        client = runtime.QMP(port, time.monotonic() + 5)
        try:
            with pytest.raises(RuntimeError, match="cannot write output"):
                client.command("human-monitor-command", {"command-line": "pmemsave 0x80000000 4 output.bin"})
        finally:
            client.close()


@pytest.mark.parametrize("command", ["quit", "stop", "human-monitor-command"])
@pytest.mark.parametrize("disconnect", ["eof", "reset"])
def test_qmp_disconnect_is_allowed_only_after_sending_quit(monkeypatch, command, disconnect):
    def respond(request, send):
        assert request["execute"] == command
        # Leave without a command response, as QEMU's documented quit permits.
    with _qmp_server(respond) as port:
        client = runtime.QMP(port, time.monotonic() + 5)
        if disconnect == "reset":
            def reset():
                raise ConnectionResetError("peer reset while reading reply")
            monkeypatch.setattr(client, "read", reset)
        try:
            if command == "quit":
                assert client.command(command) is None
            else:
                with pytest.raises(EOFError if disconnect == "eof" else ConnectionResetError):
                    client.command(command)
        finally:
            client.close()


def test_qmp_quit_error_reply_is_not_accepted_as_shutdown():
    def respond(request, send):
        send({"error": {"class": "GenericError", "desc": "quit rejected"},
              "id": request["id"]})
    with _qmp_server(respond) as port:
        client = runtime.QMP(port, time.monotonic() + 5)
        try:
            with pytest.raises(RuntimeError, match="quit rejected"):
                client.command("quit")
        finally:
            client.close()


@pytest.mark.parametrize("failure", ["send-reset", "short-write", "timeout", "malformed", "arguments"])
def test_qmp_quit_does_not_hide_send_protocol_or_timeout_errors(failure):
    client = runtime.QMP.__new__(runtime.QMP)
    client.serial, client.deadline = 0, time.monotonic() + 5
    client.socket = SimpleNamespace(settimeout=lambda timeout: None)
    writes = []

    def write(payload):
        writes.append(json.loads(payload))
        if failure == "send-reset":
            raise ConnectionResetError("quit was not sent")
        return len(payload) - (failure == "short-write")

    def read():
        if failure == "timeout":
            raise TimeoutError("quit reply timed out")
        if failure == "malformed":
            raise ValueError("malformed reply")
        raise EOFError("peer closed")

    client.stream = SimpleNamespace(write=write)
    client.read = read
    expected = {"send-reset": ConnectionResetError, "short-write": RuntimeError,
                "timeout": TimeoutError, "malformed": ValueError, "arguments": EOFError}[failure]
    with pytest.raises(expected):
        client.command("quit", {} if failure == "arguments" else None)
    assert writes[0]["execute"] == "quit"


def test_qmp_read_rejects_incomplete_reply_even_if_json_is_valid():
    client = runtime.QMP.__new__(runtime.QMP)
    client.deadline = time.monotonic() + 5
    client.socket = SimpleNamespace(settimeout=lambda timeout: None)
    client.stream = SimpleNamespace(readline=lambda limit: b'{"return": {}, "id": 1}')
    with pytest.raises(RuntimeError, match="Truncated"):
        client.read()


@pytest.mark.parametrize("failure", ["malformed_greeting", "capability_error", "capability_timeout"])
def test_qmp_constructor_closes_connection_when_handshake_fails(failure):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(3)
    closed = threading.Event()
    errors = []

    def serve():
        try:
            with listener.accept()[0] as connection:
                connection.settimeout(1)
                if failure == "malformed_greeting":
                    connection.sendall(b"not JSON\r\n")
                else:
                    connection.sendall(b'{"QMP":{"version":{},"capabilities":[]}}\r\n')
                    request = bytearray()
                    while not request.endswith(b"\n"):
                        part = connection.recv(1)
                        if not part:
                            raise AssertionError("Client disconnected before capabilities")
                        request.extend(part)
                    request = json.loads(request)
                    if failure == "capability_error":
                        reply = {"error": {"class": "GenericError", "desc": "handshake failed"},
                                 "id": request["id"]}
                        connection.sendall(json.dumps(reply).encode() + b"\r\n")
                # Keep the accepted socket open until the failed client closes.
                if connection.recv(1) == b"":
                    closed.set()
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        expected = {"malformed_greeting": ValueError, "capability_error": RuntimeError,
                    "capability_timeout": TimeoutError}[failure]
        deadline = time.monotonic() + (0.15 if failure == "capability_timeout" else 2)
        # Retain the exception/traceback so garbage collection cannot hide an
        # unclosed socket owned by the failed QMP constructor's local self.
        with pytest.raises(expected) as error:
            runtime.QMP(listener.getsockname()[1], deadline)
        assert error.value is not None
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert closed.is_set(), f"Failed constructor retained its QMP connection: {errors}"
        assert not errors
    finally:
        listener.close()
        worker.join(timeout=2)


def _unlaunchable_executable(tmp_path):
    value = Value("x", shape=(1,))
    builder = IRBuilder()
    builder.new_function("main", [value])
    builder.new_block("entry")
    builder.ret(value)
    artifact = TensorCCodegen(builder.program, constant_storage="external").generate()
    bundle = tmp_path / "weights"
    manifest = write_weight_bundle(artifact.external_weights, artifact.external_initializers, bundle)
    layout = plan_guest_memory(input_bytes=4, output_bytes=4)
    elf = tmp_path / "model.elf"
    _elf(elf, layout)
    evidence = runtime.validate_external_elf(elf, layout)
    tools = RiscVToolchain(("unused-compiler",), str(tmp_path / "missing-qemu-executable"))
    return runtime.ExternalExecutable(elf, bundle, manifest, layout, artifact.inputs,
                                      artifact.output, artifact.external_weights, tools, evidence)


def test_elf_mutation_is_rejected_before_launch(tmp_path):
    import numpy as np
    executable = _unlaunchable_executable(tmp_path)
    executable.elf_path.write_bytes(b"mutated after build")
    with pytest.raises(ValueError, match="ELF changed"):
        runtime.run_external(executable, {"x": np.ones((1,), np.float32)}, tmp_path / "run")


def test_launcher_failure_persists_fail_evidence(tmp_path):
    import numpy as np
    executable = _unlaunchable_executable(tmp_path)
    run_dir = tmp_path / "run"
    with pytest.raises(OSError):
        runtime.run_external(executable, {"x": np.ones((1,), np.float32)}, run_dir)
    report = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert report["passed"] is False and report["status"] == "FAIL"
    assert report["guest_completed"] is False
    assert "error" in report and report["elapsed_s"] >= 0


@pytest.mark.parametrize("stat_failure", [False, True], ids=["truncated", "unobservable"])
def test_postcheck_failure_refreshes_residual_evidence_without_replacing_error(
        tmp_path, monkeypatch, stat_failure):
    """Exercise report/lifecycle handling with a stub guest, not numerical execution."""
    import numpy as np

    x, w = Value("x", shape=(2,)), Value("w", shape=(2,))
    builder = IRBuilder()
    builder.new_function("main", [x])
    builder.new_block("entry")
    builder.program.global_values.append(w)
    builder.ret(builder.add(x, w))
    artifact = TensorCCodegen(builder.program, {"w": np.ones(2, np.float32)},
                             constant_storage="external").generate()
    bundle = tmp_path / "weights"
    manifest = write_weight_bundle(artifact.external_weights, artifact.external_initializers, bundle)
    original_weights = (bundle / "weights.bin").read_bytes()
    assert manifest["total_bytes"] == 8
    layout = plan_guest_memory(input_bytes=8, output_bytes=8, weight_bytes=8,
                               workspace_bytes=artifact.workspace_bytes)
    elf = tmp_path / "model.elf"
    _elf(elf, layout)
    executable = runtime.ExternalExecutable(
        elf, bundle, manifest, layout, artifact.inputs, artifact.output,
        artifact.external_weights, RiscVToolchain(("unused",), "stub-qemu"),
        runtime.validate_external_elf(elf, layout))
    out = tmp_path / "run"

    class Process:
        returncode = None
        pid = 1234

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = 0
            return 0

    class QMP:
        def __init__(self, *args):
            pass

        def command(self, command, *args):
            if command == "human-monitor-command":
                (out / "output.bin").write_bytes(bytes(8))
                return ""
            return {}

        def close(self):
            pass

    class Monitor:
        def __init__(self, *args):
            pass

        def start(self):
            pass

        def close(self):
            pass

        def report(self):
            return {}

    def launch(*args, cwd, **kwargs):
        (Path(cwd) / "uart.bin").write_bytes(runtime.FRAME.pack(runtime.MAGIC, 0, 8))
        return Process()

    verify, lstat = runtime.validate_transport, Path.lstat
    original_errors = []

    def inaccessible(path, *args, **kwargs):
        if path.parent == out / "weight-transport" and path.suffix == ".bin":
            raise PermissionError("injected residual stat failure")
        return lstat(path, *args, **kwargs)

    def truncate_then_verify(directory, report):
        path = directory / report["files"][0]["file"]
        path.write_bytes(path.read_bytes()[:-1])
        try:
            return verify(directory, report)
        except ValueError as exc:
            original_errors.append(exc)
            if stat_failure:
                monkeypatch.setattr(Path, "lstat", inaccessible)
            raise

    monkeypatch.setattr(transport, "CHUNK_BYTES", 4)
    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    monkeypatch.setattr(runtime, "QMP", QMP)
    monkeypatch.setattr(runtime, "_QemuMemoryMonitor", Monitor)
    monkeypatch.setattr(runtime, "_windows_job", lambda process: None)
    monkeypatch.setattr(runtime, "validate_transport", truncate_then_verify)
    monkeypatch.setattr(transport, "remove_transport", lambda *args: pytest.fail("must retain copies"))
    monkeypatch.setattr(runtime, "remove_transport", lambda *args: pytest.fail("must retain copies"))
    with pytest.raises(ValueError, match="size mismatch") as caught:
        runtime.run_external(executable, {"x": np.ones(2, np.float32)}, out)
    assert original_errors == [caught.value]
    report = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert not report["passed"] and report["status"] == "FAIL" and report["guest_completed"]
    assert report["error"] == "ValueError: Weight transport size mismatch"
    residual = report["weight_transport"]
    assert residual["status"] == "FAIL" and residual["stage"] == "postcheck"
    assert residual["retained_bytes"] == (None if stat_failure else 7)
    assert residual["retained_observation"] == ("unavailable" if stat_failure else "complete")
    assert residual["retained"] is (None if stat_failure else True)
    assert residual["retained_observed_bytes"] == (0 if stat_failure else 7)
    if stat_failure:
        assert len(residual["retained_observation_errors"]) == 2
        assert any("residual stat failure" in error for error in report["cleanup_errors"])
    else:
        assert not residual["retained_observation_errors"]
    assert json.loads((out / "weight-transport/transport.json").read_text(encoding="utf-8")) == residual
    assert (out / "weight-transport/weights-00000.bin").read_bytes() == original_weights[:3]
    assert (out / "weight-transport/weights-00001.bin").read_bytes() == original_weights[4:]
    assert (bundle / "weights.bin").read_bytes() == original_weights


@pytest.mark.parametrize("failure", ["report_stat", "rss_report"])
@pytest.mark.parametrize("has_primary", [False, True], ids=["completed", "primary_error"])
def test_cleanup_observation_failure_preserves_error_and_finishes_resource_cleanup(
        tmp_path, monkeypatch, failure, has_primary):
    """Fault injection for cleanup ownership/reporting; no guest arithmetic."""
    import numpy as np

    executable = _unlaunchable_executable(tmp_path)
    out = tmp_path / "run"
    events = []
    cause = ValueError("primary QMP stop failure")

    class Process:
        returncode = None
        pid = 1234

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            events.append("process-wait")
            self.returncode = 0
            return 0

    class QMP:
        def __init__(self, *args):
            pass

        def command(self, command, *args):
            if command == "stop" and has_primary:
                raise cause
            if command == "human-monitor-command":
                (out / "output.bin").write_bytes(bytes(4))
                return ""
            return {}

        def close(self):
            events.append("qmp-close")

    class Monitor:
        def __init__(self, *args):
            pass

        def start(self):
            pass

        def close(self):
            events.append("monitor-close")

        def report(self):
            events.append("monitor-report")
            if failure == "rss_report":
                raise OSError("injected RSS report failure")
            return {}

    def launch(*args, cwd, **kwargs):
        (Path(cwd) / "uart.bin").write_bytes(runtime.FRAME.pack(runtime.MAGIC, 0, 4))
        return Process()

    stat = Path.stat

    def inaccessible_report(path, *args, **kwargs):
        if path == out / "weight-transport/transport.json":
            raise PermissionError("injected report path stat failure")
        return stat(path, *args, **kwargs)

    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    monkeypatch.setattr(runtime, "QMP", QMP)
    monkeypatch.setattr(runtime, "_QemuMemoryMonitor", Monitor)
    monkeypatch.setattr(runtime, "_windows_job", lambda process: (
        SimpleNamespace(CloseHandle=lambda handle: events.append("job-close") or 1), 99))
    monkeypatch.setattr(runtime, "_stop_process_tree", lambda process, job: events.append("tree-kill"))
    if failure == "report_stat":
        monkeypatch.setattr(Path, "stat", inaccessible_report)
    with pytest.raises(ValueError if has_primary else RuntimeError) as caught:
        runtime.run_external(executable, {"x": np.ones(1, np.float32)}, out)
    if has_primary:
        assert caught.value is cause
        assert "tree-kill" in events
    else:
        assert "Runtime cleanup failed" in str(caught.value)
    assert all(event in events for event in ["qmp-close", "monitor-close", "monitor-report",
                                             "process-wait", "job-close"])
    report = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert not report["passed"] and report["status"] == "FAIL"
    expected = "report path stat failure" if failure == "report_stat" else "RSS report failure"
    assert any(expected in error for error in report["cleanup_errors"])
    if has_primary:
        assert report["error"] == "ValueError: primary QMP stop failure"
    # Transport observation/reporting is attempted even after monitor failure.
    assert report["weight_transport"]["retained_observation"] == "complete"
    assert report["weight_transport"]["retained_bytes"] == 0
