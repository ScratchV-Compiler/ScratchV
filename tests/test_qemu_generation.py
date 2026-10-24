"""Generation ABI, live guest cancellation, and reusable 1/4-hart small graphs."""
import json
import os
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from scratchv.backend.tensor_c_codegen import TensorCCodegen, TensorSpec
from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.ir.types import DataType
from scratchv.runtime import qemu_generation as adapter
from scratchv.runtime import riscv_external as runtime
from scratchv.runtime.generation import GenerationCancelledError, generate
from scratchv.runtime.llm_inputs import prepare_inputs
from scratchv.runtime.qemu_generation import QemuForward
from scratchv.runtime.riscv_tensor import discover_toolchain
from scratchv.runtime.weight_bundle import write_weight_bundle


class SmallTokenizer:
    model_vocab_size = 8
    pad_token_id = 0
    generation_eos_token_ids = (6,)

    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return [int(part) for part in text.split()]

    def validate_token_ids(self, tokens):
        result = list(tokens)
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
               or not 0 <= x < 8 for x in result):
            raise ValueError("invalid small token")
        return result

    def decode(self, tokens, *, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return " ".join(str(x) for x in tokens if not skip_special_tokens or x not in (0, 6))


def make_small_onnx(path):
    """A real input-dependent 256-row graph, with a small vocabulary/weights."""
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    embedding = np.zeros((8, 128), np.float32)
    embedding[np.arange(8), np.arange(8)] = 1
    projection = np.zeros((128, 8), np.float32)
    projection[np.arange(8), [1, 2, 3, 4, 6, 6, 6, 6]] = 8
    graph = helper.make_graph([
        helper.make_node("Gather", ["embedding", "input_ids"], ["hidden"], axis=0),
        helper.make_node("MatMul", ["hidden", "projection"], ["logits"]),
    ], "w5-small-generation", [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 256]),
        helper.make_tensor_value_info("attention_mask", TensorProto.FLOAT, [1, 1, 256, 256]),
    ], [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 256, 8])], [
        numpy_helper.from_array(embedding, "embedding"),
        numpy_helper.from_array(projection, "projection"),
    ])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    onnx.checker.check_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    return path


def abi_executable(**changes):
    base = dict(inputs=(TensorSpec("input_ids", DataType.INT64, (1, 256)),
                        TensorSpec("attention_mask", DataType.FLOAT32, (1, 1, 256, 256))),
                output=TensorSpec("logits", DataType.FLOAT32, (1, 256, 8)),
                evidence={"qemu_workspace_layout": "none"}, harts=1)
    base.update(changes)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("changes,error", [
    ({"evidence": {"qemu_workspace_layout": "two-buffer-pad-4k"}}, "layout=none"),
    ({"inputs": ()}, "exactly"),
    ({"inputs": (TensorSpec("input_ids", DataType.INT32, (1,256)),
                  TensorSpec("attention_mask", DataType.FLOAT32, (1,1,256,256)))}, "shape/dtype"),
    ({"output": TensorSpec("logits", DataType.FLOAT32, (1,8))}, "full FP32"),
    ({"output": TensorSpec("logits", DataType.INT64, (1,256,8))}, "full FP32"),
])
def test_rejects_incompatible_abi_before_creating_directory(tmp_path, changes, error):
    with pytest.raises(ValueError, match=error):
        QemuForward(abi_executable(**changes), tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("kwargs,exception", [
    ({"timeout": 0}, ValueError), ({"timeout": float("nan")}, ValueError),
    ({"timeout": True}, TypeError), ({"deadline": float("inf")}, ValueError),
    ({"cancelled": 1}, TypeError),
])
def test_invalid_options(tmp_path, kwargs, exception):
    with pytest.raises(exception):
        QemuForward(abi_executable(), tmp_path / "runs", **kwargs)


def test_serial_adapter_reuses_executable_and_keeps_full_output_contract(tmp_path, monkeypatch):
    exe = abi_executable()
    backend = QemuForward(exe, tmp_path / "runs")
    calls = []
    def execute(executable, inputs, run_dir, **kwargs):
        assert executable is exe
        assert not backend._lock.acquire(blocking=False)
        calls.append((inputs["input_ids"].copy(), run_dir, kwargs))
        result = np.zeros((1, 256, 8), np.float32)
        if len(calls) == 2:
            result = result[:, :1, :]
        return result, {"passed": True}
    monkeypatch.setattr(adapter, "run_external", execute)
    prepared = prepare_inputs([1], pad_token_id=0, vocab_size=8)
    backend(prepared)
    with pytest.raises(ValueError, match="fixed ABI"):
        backend(prepared)
    backend(prepared)
    assert [x[1].name for x in calls] == ["step-0000", "step-0001", "step-0002"]
    assert [r["passed"] for r in backend.runs] == [True, False, True]
    assert all(x[2]["cancelled"] is None for x in calls)
    with pytest.raises(FileExistsError):
        QemuForward(exe, tmp_path / "runs")


def test_adapter_nonoverlap_cancel_deadline_and_original_exception(tmp_path, monkeypatch):
    backend = QemuForward(abi_executable(), tmp_path / "runs")
    prepared = prepare_inputs([1], pad_token_id=0, vocab_size=8)
    backend._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="overlapping"):
            backend(prepared)
    finally:
        backend._lock.release()
    backend.cancelled = lambda: True
    with pytest.raises(GenerationCancelledError):
        backend(prepared)
    backend.cancelled = None
    backend.deadline = time.monotonic() - 1
    with pytest.raises(TimeoutError, match="generation"):
        backend(prepared)
    assert backend.runs == []
    backend.deadline = None
    error = RuntimeError("backend error")
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(adapter, "run_external", fail)
    with pytest.raises(RuntimeError) as actual:
        backend(prepared)
    assert actual.value is error
    assert backend.runs[0]["status"] == "FAIL"


@pytest.mark.parametrize("phase", ["greeting", "partial", "reply"])
def test_qmp_cancellation_interrupts_missing_or_partial_response(phase):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(3)
    ready, stop = threading.Event(), threading.Event()
    errors = []
    def serve():
        try:
            with listener.accept()[0] as connection:
                connection.settimeout(3)
                with connection.makefile("rwb", buffering=0) as stream:
                    if phase != "greeting":
                        stream.write(b'{"QMP":{}}\n')
                        request = json.loads(stream.readline())
                        stream.write(json.dumps({"return":{}, "id":request["id"]}).encode()+b'\n')
                        request = json.loads(stream.readline())
                        assert request["execute"] == "stop"
                        if phase == "partial":
                            stream.write(b'{"return":')
                    ready.set()
                    assert stop.wait(3)
        except BaseException as exc:
            errors.append(exc)
    worker = threading.Thread(target=serve)
    worker.start()
    client = None
    began = time.monotonic()
    try:
        with pytest.raises(GenerationCancelledError):
            client = runtime.QMP(listener.getsockname()[1], began + 5, cancelled=ready.is_set)
            client.command("stop")
        assert time.monotonic() - began < 1
    finally:
        if client is not None:
            client.close()
        stop.set()
        worker.join(4)
        listener.close()
    assert not worker.is_alive()
    assert not errors


def _assert_reaped(report):
    assert report["qemu_reaped"]
    with pytest.raises(ProcessLookupError):
        os.kill(report["qemu_pid"], 0)
    with pytest.raises(ProcessLookupError):
        os.killpg(report["qemu_pid"], 0)
    assert not report.get("cleanup_errors")


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="requires explicit SCRATCHV_QEMU/CC")
@pytest.mark.parametrize("harts", [1, 4])
def test_real_qemu_multitoken_reuse_numerics_timeout_and_cancel(tmp_path, monkeypatch, harts):
    import onnxruntime as ort
    path = make_small_onnx(tmp_path / "model.onnx")
    parser = ONNXParser()
    program = parser.parse(str(path))
    model = TensorCCodegen(program, initializers=parser.initializers,
                         constant_storage="external").generate()
    bundle = tmp_path / "weights"
    write_weight_bundle(model.external_weights, model.external_initializers, bundle)
    original_harness = runtime.guest_harness
    def waiting_fixture(*args):
        source = original_harness(*args)
        # Token 7 deliberately parks this test guest, so timeout/cancel happen
        # while a real QEMU process is alive instead of before it is launched.
        return source.replace("    int status=", '    if (((const long *)inputs[0])[0] == 7) for (;;) __asm__ volatile("wfi");\n    int status=')
    monkeypatch.setattr(runtime, "guest_harness", waiting_fixture)
    exe = runtime.build_external(model, bundle, tmp_path / "build", discover_toolchain(), harts=harts)
    backend = QemuForward(exe, tmp_path / "runs", timeout=10)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    comparisons = []
    def checked(prepared):
        actual = backend(prepared)
        expected = session.run(None, prepared.as_feed())[0]
        np.testing.assert_array_equal(actual, expected)
        comparisons.append({"valid_length": prepared.valid_length, "elements": actual.size,
                            "max_abs": 0.0})
        return actual
    first = generate("1", tokenizer=SmallTokenizer(), forward=checked, max_new_tokens=8)
    assert first.generated_token_ids == (2, 3, 4, 6) and first.stop_reason == "eos"
    second = generate("3", tokenizer=SmallTokenizer(), forward=checked, max_new_tokens=1)
    assert second.generated_token_ids == (4,) and second.stop_reason == "max_new_tokens"
    assert len(backend.runs) == 5
    for report in backend.runs:
        _assert_reaped(report)
        assert report["harts"] == harts and report["qemu_workspace_layout"] == "none"
        assert report["weight_transport"]["retained_bytes"] == 0
    waiting = prepare_inputs([7], pad_token_id=0, vocab_size=8)
    backend.timeout = 0.2
    with pytest.raises(TimeoutError, match="forward exceeded"):
        backend(waiting)
    _assert_reaped(backend.runs[-1])
    backend.timeout = 10
    original_wait = runtime._UartCompletionWaiter.wait
    started = []
    cancel_at = [None]
    def record_wait(self, process, *args, **kwargs):
        assert process.poll() is None
        started.append(process.pid)
        cancel_at[0] = time.monotonic() + 0.05
        return original_wait(self, process, *args, **kwargs)
    monkeypatch.setattr(runtime._UartCompletionWaiter, "wait", record_wait)
    backend.cancelled = lambda: cancel_at[0] is not None and time.monotonic() >= cancel_at[0]
    before = time.monotonic()
    with pytest.raises(GenerationCancelledError):
        backend(waiting)
    assert time.monotonic() - before < 2
    _assert_reaped(backend.runs[-1])
    assert started == [backend.runs[-1]["qemu_pid"]]
    backend.cancelled = None
    third = generate("4", tokenizer=SmallTokenizer(), forward=checked, max_new_tokens=2)
    assert third.generated_token_ids == (6,) and third.stop_reason == "eos"
    _assert_reaped(backend.runs[-1])
    assert all(run["command"][run["command"].index("-kernel")+1] == str(exe.elf_path)
               for run in backend.runs)
    (tmp_path / "generation-check.json").write_text(json.dumps({
        "harts": harts, "full_model": False, "qemu_workspace_layout": "none",
        "first_tokens": list(first.generated_token_ids), "different_context": list(second.generated_token_ids),
        "after_abort_tokens": list(third.generated_token_ids), "numeric_comparisons": comparisons,
        "one_elf_reused": True, "one_bundle_reused": True, "runs": backend.runs,
    }, indent=2)+"\n")


def test_cancellable_qmp_preserves_fragmented_events_and_matching_id():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(3)
    errors = []
    def serve():
        try:
            with listener.accept()[0] as connection:
                connection.settimeout(3)
                with connection.makefile("rwb", buffering=0) as stream:
                    stream.write(b'{"QMP":{}}\n')
                    capability = json.loads(stream.readline())
                    stream.write(json.dumps({"return":{}, "id":capability["id"]}).encode()+b'\n')
                    request = json.loads(stream.readline())
                    payload = (json.dumps({"event":"STOP"})+'\n'+
                               json.dumps({"return":{}, "id":request["id"]-1})+'\n'+
                               json.dumps({"return":{"status":"paused"}, "id":request["id"]})+'\n').encode()
                    stream.write(payload[:7])
                    time.sleep(0.01)
                    stream.write(payload[7:])
        except BaseException as exc:
            errors.append(exc)
    worker = threading.Thread(target=serve)
    worker.start()
    client = None
    try:
        client = runtime.QMP(listener.getsockname()[1], time.monotonic()+3, cancelled=lambda: False)
        assert client.command("stop") == {"status":"paused"}
    finally:
        if client is not None:
            client.close()
        worker.join(4)
        listener.close()
    assert not worker.is_alive()
    assert not errors


@pytest.mark.parametrize("contents,error", [
    ("null", TimeoutError("original timeout")),
    ("[1]", GenerationCancelledError("original cancellation")),
    ('"not an object"', KeyboardInterrupt("original interruption")),
    ("{", RuntimeError("original backend error")),
    ("[" * 2000 + "0" + "]" * 2000, TimeoutError("timeout with deeply nested report")),
    ('{"elapsed_s": NaN}', TimeoutError("timeout with nonfinite report")),
    ('{"samples": [Infinity, -Infinity]}', GenerationCancelledError("cancel with nonfinite report")),
    (None, TimeoutError("original timeout with unreadable report")),
])
def test_report_read_failure_preserves_backend_error_and_adapter_reuse(
    tmp_path, monkeypatch, contents, error,
):
    backend = QemuForward(abi_executable(), tmp_path / "runs")
    prepared = prepare_inputs([1], pad_token_id=0, vocab_size=8)
    calls = []
    original_read = Path.read_text

    def read(path, *args, **kwargs):
        if contents is None and path.name == "run.json":
            raise PermissionError("diagnostic report unreadable")
        return original_read(path, *args, **kwargs)

    def execute(executable, inputs, run_dir, **kwargs):
        calls.append((run_dir.name, inputs["input_ids"].copy()))
        if len(calls) == 1:
            run_dir.mkdir()
            (run_dir / "run.json").write_text("{}" if contents is None else contents)
            raise error
        result = np.zeros((1, 256, 8), dtype=np.float32)
        return result, {"passed": True}

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(adapter, "run_external", execute)
    with pytest.raises(type(error)) as caught:
        backend(prepared)
    assert caught.value is error
    assert len(backend.runs) == 1
    assert backend.runs[0]["passed"] is False
    assert backend.runs[0]["error"] == f"{type(error).__name__}: {error}"
    assert backend.runs[0]["report_read_error"]
    json.dumps(backend.runs, allow_nan=False)  # The CLI's finally report must still serialize.
    np.testing.assert_array_equal(backend(prepared), np.zeros((1, 256, 8), np.float32))
    assert [run["passed"] for run in backend.runs] == [False, True]
    assert [call[0] for call in calls] == ["step-0000", "step-0001"]
    np.testing.assert_array_equal(calls[0][1], calls[1][1])


@pytest.mark.parametrize("error", [
    GenerationCancelledError("cancelled during second token"),
    TimeoutError("timeout during second token"),
    RuntimeError("backend failed during second token"),
])
def test_conversation_and_reused_backend_rollback_together_after_partial_turn(
    tmp_path, monkeypatch, error,
):
    pytest.importorskip("jinja2")
    from scratchv.runtime.conversation import ChatMessage, Conversation

    backend = QemuForward(abi_executable(), tmp_path / "runs")
    chat = Conversation(SmallTokenizer(),
                        "{% for m in messages %}{{ m.content }} {% endfor %}")
    buffer = np.empty((1, 256, 8), dtype=np.float32)
    seen = []

    def execute(executable, inputs, run_dir, **kwargs):
        run_dir.mkdir()
        # The test's nonzero prefix denotes valid token IDs; padding is zero.
        prefix = tuple(int(x) for x in inputs["input_ids"][0] if x != 0)
        seen.append(prefix)
        if len(seen) == 3:
            (run_dir / "run.json").write_text("null")
            raise error
        buffer.fill(0)
        token = {1: 2, 2: 4, 4: 6}[len(seen)]
        buffer[0, len(prefix) - 1, token] = 10
        return buffer, {"passed": True}

    monkeypatch.setattr(adapter, "run_external", execute)
    first = chat.turn("1", forward=backend, max_new_tokens=1)
    committed = chat.history
    assert committed == (ChatMessage("user", "1"), ChatMessage("assistant", "2"))
    with pytest.raises(type(error)) as caught:
        chat.turn("3", forward=backend, max_new_tokens=3)
    assert caught.value is error
    assert chat.history == committed
    assert seen == [(1,), (1, 2, 3), (1, 2, 3, 4)]

    retry = chat.turn("5", forward=backend, max_new_tokens=2)
    assert seen[-1] == (1, 2, 5)  # No failed user 3 or partial assistant 4.
    assert retry.generated_token_ids == (6,) and retry.stop_reason == "eos"
    assert chat.history == committed + (ChatMessage("user", "5"), ChatMessage("assistant", ""))
    assert [run["passed"] for run in backend.runs] == [True, True, False, True]
    assert [Path(run["run_dir"]).name for run in backend.runs] == [
        "step-0000", "step-0001", "step-0002", "step-0003",
    ]
    buffer.fill(np.nan)
    assert first.generated_token_ids == (2,) and first.generated_text == "2"
    assert retry.generated_token_ids == (6,) and retry.generated_text == ""
