"""Host C owns its compiler tree through timeout, cancellation and worker exit.

Only short Python stand-ins are executed; no model or C compiler is needed.
"""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from probes.w4_qwen3_full import host_diagnostic as host


pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"),
                                reason="Linux Host C process-group ownership")
ROOT = Path(__file__).resolve().parents[1]


def _read_until(path, process, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return json.loads(path.read_text())
        assert process.poll() is None, "fixture exited before its compiler became ready"
        time.sleep(0.01)
    pytest.fail("fixture compiler did not become ready")


def _running(pid):
    try:
        # A killed orphan can briefly remain a zombie until the system reaps it.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def _assert_stopped(pids):
    deadline = time.monotonic() + 5
    while any(_running(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not [pid for pid in pids if _running(pid)], "owned descendant survived cleanup"


def _kill_group(pgid):
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _scripts(directory, outcome):
    """Exercise actual main/build_library, replacing only the model worker body."""
    compiler = directory / "fake-cc"
    compiler.write_text(f"#!{sys.executable}\n" + """
import json, os, pathlib, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
pipes = {} if os.environ['HOST_TEST_OUTCOME'] == 'compiler_pipe_timeout' else {
    'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL}
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], **pipes)
record = {'compiler': os.getpid(), 'compiler_group': os.getpgrp(), 'child': child.pid}
path = pathlib.Path(os.environ['HOST_TEST_READY'])
temporary = path.with_suffix('.tmp')
temporary.write_text(json.dumps(record))
temporary.replace(path)
print('compiler output before interruption', flush=True)
print('compiler error before interruption', file=sys.stderr, flush=True)
if os.environ['HOST_TEST_OUTCOME'] == 'worker_exit':
    sys.exit(7)
time.sleep(60)
""")
    compiler.chmod(0o700)
    worker = directory / "worker.py"
    worker.write_text("""
import json, os, pathlib, signal
from types import SimpleNamespace
from probes.w4_qwen3_full import host_diagnostic as host
signal.signal(signal.SIGTERM, signal.SIG_IGN)
pathlib.Path(os.environ['HOST_TEST_WORKER']).write_text(json.dumps({'worker': os.getpid()}))
artifact = SimpleNamespace(constant_storage='external', source='/* no C compilation */',
                           compile_flags=())
host.build_library(artifact, pathlib.Path(os.environ['HOST_TEST_BUILD']),
                   [os.environ['HOST_TEST_COMPILER']],
                   timeout=2 if os.environ['HOST_TEST_OUTCOME'].startswith('compiler_') else 60,
                   supervised=True)
""")
    supervisor = directory / "supervisor.py"
    supervisor.write_text("""
import os, sys
from probes.w4_qwen3_full import host_diagnostic as host
run_process = host._run_process
def run_fixture(command, **kwargs):
    return run_process([sys.executable, os.environ['HOST_TEST_SCRIPT']], **kwargs)
host._run_process = run_fixture
raise SystemExit(host.main(['--model-dir', os.environ['HOST_TEST_MODEL'],
                           '--output-dir', os.environ['HOST_TEST_OUTPUT'],
                           '--timeout', '4' if os.environ['HOST_TEST_OUTCOME'] == 'outer_timeout' else '30']))
""")
    environment = dict(os.environ, PYTHONPATH=str(ROOT), HOST_TEST_OUTCOME=outcome,
        HOST_TEST_READY=str(directory / "ready.json"), HOST_TEST_WORKER=str(directory / "worker.json"),
        HOST_TEST_SCRIPT=str(worker), HOST_TEST_BUILD=str(directory / "build"),
        HOST_TEST_COMPILER=str(compiler), HOST_TEST_MODEL=str(directory / "unused-model"),
        HOST_TEST_OUTPUT=str(directory / "report"))
    return supervisor, environment


@pytest.mark.parametrize("outcome", ["outer_timeout", "sigterm", "sigint", "compiler_timeout",
                                    "compiler_pipe_timeout", "worker_exit", "stopped_worker"])
def test_host_supervisor_reclaims_compiler_and_descendants(tmp_path, outcome):
    supervisor, environment = _scripts(tmp_path, outcome)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 start_new_session=True)
    process = subprocess.Popen([sys.executable, str(supervisor)], cwd=ROOT, env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    worker = record = None
    try:
        record = _read_until(tmp_path / "ready.json", process)
        worker = json.loads((tmp_path / "worker.json").read_text())["worker"]
        assert record["compiler_group"] == worker
        if outcome == "stopped_worker":
            os.kill(worker, signal.SIGSTOP)
            process.send_signal(signal.SIGTERM)
        elif outcome in {"sigterm", "sigint"}:
            process.send_signal(signal.SIGTERM if outcome == "sigterm" else signal.SIGINT)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 1, (stdout, stderr)
        _assert_stopped([worker, record["compiler"], record["child"]])
        assert unrelated.poll() is None, "cleanup crossed the owned process group"
        report = json.loads((tmp_path / "report/report.json").read_text())
        assert report["status"] == "HOST_FAIL" and report["passed"] is False
        assert not report["full_qemu_forward_executed"] and not any(report["gates"].values())
        if outcome == "outer_timeout":
            assert "TimeoutExpired" in report["error"]
        elif outcome in {"sigterm", "stopped_worker"}:
            assert "SystemExit: 143" in report["error"]
        elif outcome == "sigint":
            assert "KeyboardInterrupt" in report["error"]
        elif outcome.startswith("compiler_"):
            assert b"compilation exceeded" in (tmp_path / "report/worker.stderr").read_bytes()
        if outcome.startswith("compiler_") or outcome == "worker_exit":
            assert b"compiler output" in (tmp_path / "build/compiler.stdout").read_bytes()
            assert b"compiler error" in (tmp_path / "build/compiler.stderr").read_bytes()
    finally:
        if process.poll() is None:
            _kill_group(process.pid)
        process.communicate(timeout=5)
        if worker is None and (tmp_path / "worker.json").exists():
            worker = json.loads((tmp_path / "worker.json").read_text())["worker"]
        if worker is not None:
            _kill_group(worker)
        if record is not None and record["compiler_group"] != worker:
            _kill_group(record["compiler_group"])
        unrelated.kill()
        unrelated.wait(timeout=5)


def test_standalone_compiler_timeout_still_owns_a_separate_group(tmp_path):
    compiler = tmp_path / "fake-cc"
    compiler.write_text(f"#!{sys.executable}\n" + """
import json, os, pathlib, sys, time
pathlib.Path('standalone.json').write_text(json.dumps({'pid': os.getpid(), 'group': os.getpgrp()}))
print('standalone compiler', flush=True)
time.sleep(60)
""")
    compiler.chmod(0o700)
    from types import SimpleNamespace
    artifact = SimpleNamespace(constant_storage="external", source="/* unused */", compile_flags=())
    with pytest.raises(TimeoutError, match="Host C compilation exceeded"):
        host.build_library(artifact, tmp_path / "build", [str(compiler)], timeout=1)
    record = json.loads((tmp_path / "build/standalone.json").read_text())
    assert record["pid"] == record["group"] != os.getpgrp()
    _assert_stopped([record["pid"]])
    assert b"standalone compiler" in (tmp_path / "build/compiler.stdout").read_bytes()


def test_real_worker_marks_compilation_as_supervised(tmp_path, monkeypatch):
    """Skip expensive preparation but execute the real worker/build call boundary."""
    from types import SimpleNamespace
    import numpy as np
    import onnx

    monkeypatch.setattr(host, "source_evidence", lambda: {})
    monkeypatch.setattr(host.assets, "verify_files", lambda _: {})
    monkeypatch.setattr(onnx, "load", lambda *_, **__: object())
    monkeypatch.setattr(host, "audit_graph_structure", lambda _: {"passed": True, "layer_count": 28})
    monkeypatch.setattr(host, "input_cases", lambda: [("full_seed_0", 1, {"x": np.zeros(1, np.float32)})])
    monkeypatch.setattr(host, "ort_reference", lambda _, feed, path: np.save(path, feed["x"]))
    monkeypatch.setattr(host, "ONNXParser", lambda: SimpleNamespace(
        parse=lambda *_, **__: object(), initializers={}))
    artifact = SimpleNamespace(workspace_bytes=0, constant_bytes=0, external_weights=(),
                               source="", output=SimpleNamespace(shape=(1,)))
    monkeypatch.setattr(host, "TensorCCodegen", lambda *_, **__: SimpleNamespace(generate=lambda: artifact))
    monkeypatch.setattr(host, "compiler_command", lambda _: ["unused"])
    calls = []

    def stop_before_compilation(artifact, directory, compiler, timeout, *, supervised=False):
        calls.append(supervised)
        raise TimeoutError("injected compiler deadline")

    monkeypatch.setattr(host, "build_library", stop_before_compilation)
    out = tmp_path / "worker"
    args = SimpleNamespace(output_dir=out, model_dir=tmp_path / "unused-model", case="full_seed_0",
                           matmul_policy="sequential", cc=None, compile_timeout=1)
    assert host.worker(args) == 1
    assert calls == [True]
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "HOST_FAIL" and report["stage"] == "host_shared_library"
    assert "injected compiler deadline" in report["error"]
    assert not report["full_host_forward_executed"] and not report["full_qemu_forward_executed"]
