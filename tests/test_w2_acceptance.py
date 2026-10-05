"""Acceptance orchestration must not turn partial, stale or failed evidence into PASS."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

SPEC = importlib.util.spec_from_file_location("w2_acceptance", Path(__file__).parents[1] / "scripts/run_w2_acceptance.py")
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def arguments(tmp_path, gates=None):
    return argparse.Namespace(output_dir=tmp_path / "fresh", gates=list(runner.GATES) if gates is None else gates,
                              python=sys.executable, tokenizer_dir=tmp_path / "assets", tokenizer_mode="verify",
                              model_dir=tmp_path / "model", model_mode="verify", cc="compiler", qemu="emulator",
                              qemu_timeout=180, parse_timeout=300)


def runtime_fixture(tmp_path):
    root = tmp_path / "source"
    corpus = root / "probes/w2_runtime/corpus.json"
    corpus.parent.mkdir(parents=True)
    names = [f"case_{i}" for i in range(20)]
    corpus.write_text(json.dumps({"cases": [{"name": name} for name in names]}), encoding="utf-8")
    folder = tmp_path / "run/runtime"
    folder.mkdir(parents=True)
    for name in ("report.md", "report.html"):
        (folder / name).write_text("PASS", encoding="utf-8")
    sources = {runner.SCRIPTS["runtime"]: "a" * 64}
    data = {"gate": "unit:runtime", "passed": True, "stage": "complete", "source_sha256": sources,
            "tokenizer_cases": [{"name": name, "passed": True, "checks": 24} for name in names],
            "runtime_cases": [{"name": str(i), "passed": True} for i in range(5)]}
    return root, runner.Gate("runtime", (), folder), sources, data


def validate_fixture(fixture, mutate=lambda data: None):
    root, gate, sources, data = fixture
    mutate(data)
    (gate.output / "report.json").write_text(json.dumps(data), encoding="utf-8")
    return runner.validate_gate_report(gate, sources, root)


def test_plan_is_fresh_serial_and_explicit(tmp_path):
    args = arguments(tmp_path)
    plan = runner.build_plan(args)
    assert [gate.name for gate in plan] == list(runner.GATES)
    for gate in plan:
        assert gate.command[:4] == (sys.executable, "-B", "-X", "utf8")
        assert str(gate.output) in " ".join(gate.command)
    assert plan[5].dependencies == ("small-ir",)
    assert str(args.output_dir / "small-ir") in plan[5].command
    assert "--mode" in plan[-1].command and "verify" in plan[-1].command
    assert "compiler" in plan[1].command and "emulator" in plan[1].command


def test_frontend_real_pytest_tmp_path_in_fresh_gate_directory(tmp_path):
    """Pytest's basetemp.mkdir has no parents=True; its parent must already exist."""
    source = tmp_path / "source"
    tests = source / "tests"
    tests.mkdir(parents=True)
    (tests / "test_qwen3_frontend_patterns.py").write_text(
        "import pytest\n@pytest.mark.parametrize('number', range(22))\n"
        "def test_real_tmp_path(tmp_path, number):\n"
        "    path = tmp_path / str(number)\n"
        "    path.write_text('fresh')\n"
        "    assert path.read_text() == 'fresh'\n", encoding="utf-8")
    args = arguments(tmp_path, ["frontend"])
    args.output_dir.mkdir()
    gate = runner.build_plan(args, source)[0]
    assert not gate.output.exists()
    result = runner.run_gate(gate, {}, timeout=30, root=source)
    assert result["status"] == "PASS", Path(result["log"]).read_text(encoding="utf-8")
    assert result["validation"]["tests"] == 22


@pytest.mark.parametrize("chosen,expected", [(["small-qemu"], ["small-ir", "small-qemu"]),
                                               (["runtime-model"], ["runtime", "runtime-model"]),
                                               (["frontend"], ["frontend"])])
def test_subset_auto_adds_only_required_dependencies(tmp_path, chosen, expected):
    assert [gate.name for gate in runner.build_plan(arguments(tmp_path, chosen))] == expected


@pytest.mark.parametrize("status,expected", [("PASS", "PASS"), ("NOT_RUN", "PARTIAL"), ("BLOCKED", "FAIL"), ("FAIL", "FAIL")])
def test_overall_requires_every_gate(status, expected):
    report = {"gates": [{"name": name, "status": "PASS"} for name in runner.GATES]}
    report["gates"][-1]["status"] = status
    runner.finalize_status(report)
    assert report["status"] == expected
    assert report["passed"] is (expected == "PASS")


def test_top_level_source_error_blocks_pass():
    report = {"error": "source changed", "gates": [{"status": "PASS"} for _ in runner.GATES]}
    runner.finalize_status(report)
    assert report["status"] == "FAIL"


def test_git_unavailable_archive_is_not_failed(monkeypatch, tmp_path):
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("git")))
    result = runner.checkout_evidence(tmp_path)
    assert result["head"] is None and result["status"] == "unknown"


def test_new_source_files_have_fingerprints(tmp_path):
    path = tmp_path / "scratchv/runtime/new.py"
    path.parent.mkdir(parents=True)
    path.write_text("x = 1\n")
    before = runner.source_fingerprints(tmp_path)
    path.write_text("x = 2\n")
    assert before != runner.source_fingerprints(tmp_path)


def test_complete_runtime_evidence(tmp_path):
    assert validate_fixture(runtime_fixture(tmp_path))["tokenizer_comparisons"] == 480


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(gate="different"), lambda d: d.update(passed=False),
    lambda d: d.update(stage="tokenizer"), lambda d: d["tokenizer_cases"].pop(),
    lambda d: d["runtime_cases"].pop(), lambda d: d["tokenizer_cases"][0].update(checks=23),
    lambda d: d["tokenizer_cases"][1].update(name="case_0"),
    lambda d: d["runtime_cases"][0].update(passed=False),
    lambda d: d.update(source_sha256={}),
    lambda d: d.update(source_sha256={runner.SCRIPTS["runtime"]: "wrong"}),
])
def test_runtime_missing_or_inconsistent_evidence_fails(tmp_path, mutation):
    with pytest.raises(ValueError):
        validate_fixture(runtime_fixture(tmp_path), mutation)


@pytest.mark.parametrize("number", [None, float("nan"), float("inf"), -1, 1e-5, 0.1, True])
def test_invalid_numeric_evidence_fails(number):
    with pytest.raises(ValueError):
        runner.check_numeric_tree({"passed": True, "nested": [{"passed": True, "max_abs": number}]})


@pytest.mark.parametrize("field", ["seconds", "qemu_process_wall_seconds"])
def test_nonfinite_timing_cannot_break_machine_report(field):
    with pytest.raises(ValueError):
        runner.check_numeric_tree({"passed": True, field: float("nan")})


@pytest.mark.parametrize("tag", ["skipped", "failure", "error"])
def test_frontend_skips_are_failures(tmp_path, tag):
    cases = [f'<testcase classname="patterns" name="test{i}">{"<" + tag + "/>" if i == 0 else ""}</testcase>' for i in range(22)]
    (tmp_path / "tests.xml").write_text("<testsuites><testsuite>" + "".join(cases) + "</testsuite></testsuites>")
    with pytest.raises(ValueError):
        runner.validate_gate_report(runner.Gate("frontend", (), tmp_path), {})


def test_existing_output_is_not_read_as_pass(tmp_path):
    folder = tmp_path / "old"
    folder.mkdir()
    (folder / "report.json").write_text('{"passed":true}')
    result = runner.run_gate(runner.Gate("runtime", (), folder), {}, timeout=1)
    assert result["status"] == "FAIL" and "already exists" in result["error"]


def test_zero_exit_without_report_is_failure(tmp_path):
    gate = runner.Gate("runtime", (sys.executable, "-c", "print('PASS')"), tmp_path / "gate")
    result = runner.run_gate(gate, {}, timeout=10)
    assert result["returncode"] == 0 and result["status"] == "FAIL"
    assert "PASS" in Path(result["log"]).read_text()


def test_child_error_is_preserved(tmp_path):
    folder = tmp_path / "gate"
    code = "import pathlib,json; p=pathlib.Path(" + repr(str(folder)) + ");p.mkdir();(p/'report.json').write_text(json.dumps({'passed':False,'stage':'assets','error':'Checksum mismatch'}));raise SystemExit(1)"
    result = runner.run_gate(runner.Gate("runtime", (sys.executable, "-c", code), folder), {}, timeout=10)
    assert result["status"] == "FAIL" and "Checksum mismatch" in result["error"]
    assert result["child_stage"] == "assets"


def test_timeout_is_failure(tmp_path):
    result = runner.run_gate(runner.Gate("runtime", (sys.executable, "-c", "import time;time.sleep(30)"), tmp_path / "gate"), {}, timeout=0.2)
    assert result["status"] == "FAIL" and "exceeded" in result["error"]


@pytest.mark.parametrize("failure", [KeyboardInterrupt("cancelled"), SystemExit(143)])
def test_cancellation_reclaims_live_gate_and_preserves_exception(tmp_path, monkeypatch, failure):
    real_popen = subprocess.Popen
    children = []

    def launch(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        if args[0][0] != sys.executable:
            return process
        children.append(process)
        original_wait = process.wait

        def interrupt(*args, **kwargs):
            process.wait = original_wait
            raise failure

        process.wait = interrupt
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    try:
        with pytest.raises(type(failure)) as caught:
            runner.run_gate(runner.Gate("frontend", (sys.executable, "-c", "import time;time.sleep(60)"),
                                        tmp_path / "frontend"), {}, timeout=90)
        assert caught.value is failure
        row = caught.value.w2_gate_result
        assert row["status"] == "FAIL" and row["interrupted"] is True
        assert row["seconds"] >= 0
        assert type(failure).__name__ in row["error"]
        assert children[0].poll() is not None
    finally:
        for process in children:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


def test_cancellation_cleanup_and_log_errors_preserve_original(tmp_path, monkeypatch):
    failure = KeyboardInterrupt("primary interruption")

    class Process:
        returncode = None

        def poll(self):
            return None

        def wait(self, **kwargs):
            raise failure

    class Log:
        def close(self):
            raise OSError("secondary close error")

    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: Log())
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(runner, "terminate_process_tree", lambda process: (_ for _ in ()).throw(OSError("cleanup failed")))
    with pytest.raises(KeyboardInterrupt) as caught:
        runner.run_gate(runner.Gate("frontend", ("unused",), tmp_path / "frontend"), {}, timeout=10)
    assert caught.value is failure
    row = failure.w2_gate_result
    assert "primary interruption" in row["error"]
    assert "cleanup failed" in row["cleanup_error"]
    assert "secondary close error" in row["log_error"]
    assert row["status"] == "FAIL" and row["passed"] is False


@pytest.mark.parametrize("failure, code", [(KeyboardInterrupt(), 130), (SystemExit(143), 143)])
def test_cancelled_main_saves_failure_report_without_running_next_gate(tmp_path, monkeypatch, failure, code):
    monkeypatch.setattr(runner, "checkout_evidence", lambda: {"head": None})
    monkeypatch.setattr(runner, "source_fingerprints", lambda: {})
    launched = []

    def cancel(gate, *args, **kwargs):
        launched.append(gate.name)
        failure.w2_gate_result = {"name": gate.name, "status": "FAIL", "passed": False,
                                 "interrupted": True, "error": "original cancellation"}
        raise failure

    monkeypatch.setattr(runner, "run_gate", cancel)
    previous_term = signal.getsignal(signal.SIGTERM)
    out = tmp_path / "cancelled"
    assert runner.main(["--output-dir", str(out)]) == code
    assert signal.getsignal(signal.SIGTERM) == previous_term
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "FAIL" and report["interrupted"] is True
    assert launched == ["frontend"]
    assert report["gates"][0]["error"] == "original cancellation"
    assert all(row["status"] == "NOT_RUN" for row in report["gates"][1:])
    assert (out / "report.md").is_file() and (out / "report.html").is_file()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux supervisor signal and nested-session integration")
@pytest.mark.parametrize("cancel_signal", [signal.SIGINT, signal.SIGTERM])
def test_real_acceptance_supervisor_cancellation_cleans_nested_worker(tmp_path, cancel_signal):
    pidfile = tmp_path / "worker.json"
    out = tmp_path / "acceptance"
    worker = ("import os,time,json;from pathlib import Path;"
              "pid=os.getpid();stat=Path(f'/proc/{pid}/stat').read_text();"
              f"Path({str(pidfile)!r}).write_text(json.dumps([pid,stat.rsplit(')',1)[1].split()[19]]));"
              "time.sleep(60)")
    probe = ("import sys;from scratchv.runtime.riscv_tensor import _run_process;"
             f"_run_process([sys.executable,'-B','-c',{worker!r}],cwd=None,timeout=50)")
    supervisor = (
        "import sys;from scripts import run_w2_acceptance as r;"
        "r.source_fingerprints=lambda:{};r.checkout_evidence=lambda:{'head':None};"
        f"r.build_plan=lambda a:[r.Gate('frontend',(sys.executable,'-B','-c',{probe!r}),a.output_dir/'frontend')];"
        f"raise SystemExit(r.main(['--output-dir',{str(out)!r},'--gates','frontend']))")
    unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True)
    process = subprocess.Popen([sys.executable, "-B", "-c", supervisor], cwd=runner.ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    pid = None
    try:
        deadline = time.monotonic() + 15
        while True:
            assert process.poll() is None, "Supervisor exited before worker readiness"
            assert time.monotonic() < deadline, "Worker did not become ready"
            try:
                pid, started = json.loads(pidfile.read_text())
                break
            except (FileNotFoundError, ValueError):
                time.sleep(0.01)
        process.send_signal(cancel_signal)
        stdout, stderr = process.communicate(timeout=25)
        assert process.returncode == 128 + cancel_signal, (stdout, stderr)
        report = json.loads((out / "report.json").read_text())
        assert report["status"] == "FAIL" and report["interrupted"] is True
        assert report["gates"][0]["interrupted"] is True
        assert report["gates"][0]["status"] == "FAIL"
        assert all(row["status"] == "NOT_RUN" for row in report["gates"][1:])
        assert unrelated.poll() is None
        deadline = time.monotonic() + 5
        while True:
            try:
                fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            except FileNotFoundError:
                break
            if fields[0] in {"Z", "X"} or fields[19] != started:
                break
            assert time.monotonic() < deadline, "Detached worker survived supervisor cancellation"
            time.sleep(0.01)
    finally:
        for child in (process, unrelated):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
        if pid is not None:
            try:
                fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                if fields[19] == started and fields[0] not in {"Z", "X"}:
                    os.kill(pid, signal.SIGKILL)
            except FileNotFoundError:
                pass


@pytest.mark.skipif(sys.platform != "linux", reason="Linux nested-session SIGTERM integration")
def test_supervisor_cancels_nested_worker_session(tmp_path):
    """Real probe -> _run_process -> different session, not a mocked killpg."""
    pidfile = tmp_path / "ready.json"
    worker = ("import os,time,json;from pathlib import Path;"
              "pid=os.getpid();stat=Path(f'/proc/{pid}/stat').read_text();"
              f"Path({str(pidfile)!r}).write_text(json.dumps([pid,stat.rsplit(')',1)[1].split()[19]]));"
              "time.sleep(60)")
    probe = ("import sys;from scratchv.runtime.riscv_tensor import _run_process;"
             f"_run_process([sys.executable,'-B','-c',{worker!r}],cwd=None,timeout=50)")
    unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"],
                                 start_new_session=True)
    process = subprocess.Popen([sys.executable, "-B", "-c", probe], cwd=runner.ROOT,
                               start_new_session=True)
    pid = None
    try:
        deadline = time.monotonic() + 15
        while not pidfile.exists():
            assert process.poll() is None, "Probe exited before worker readiness"
            assert time.monotonic() < deadline, "Worker did not become ready"
            time.sleep(0.01)
        # Existence can precede completion of the tiny JSON write.
        while True:
            try:
                pid, started = json.loads(pidfile.read_text())
                break
            except ValueError:
                assert time.monotonic() < deadline
                time.sleep(0.01)
        assert os.getpgid(pid) != process.pid
        runner.terminate_process_tree(process)
        assert process.returncode == 143
        assert unrelated.poll() is None
        deadline = time.monotonic() + 5
        while True:
            try:
                fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            except (FileNotFoundError, ProcessLookupError):
                break
            if fields[0] in {"Z", "X"} or fields[19] != started:
                break
            assert time.monotonic() < deadline, "Detached worker survived supervisor cleanup"
            time.sleep(0.01)
    finally:
        for child in (process, unrelated):
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
        if pid is not None:
            try:
                fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                if fields[19] == started and fields[0] not in {"Z", "X"}:
                    os.kill(pid, 9)
            except (FileNotFoundError, ProcessLookupError):
                pass


@pytest.mark.parametrize("filename", ["report.md", "report.html", "report.json"])
def test_report_write_failure_cannot_publish_pass(tmp_path, monkeypatch, filename):
    original = Path.replace
    def replace(path, target):
        if Path(target).name == filename:
            raise OSError("disk failed")
        return original(path, target)
    monkeypatch.setattr(Path, "replace", replace)
    report = {"status": "PASS", "passed": True, "gates": []}
    runner.write_reports(tmp_path, report)
    assert report["status"] == "FAIL" and not report["passed"]
    if (tmp_path / "report.json").exists():
        assert json.loads((tmp_path / "report.json").read_text())["passed"] is False


def test_report_secondary_error_preserves_primary(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk failed")))
    report = {"status": "FAIL", "passed": False, "gates": [], "error": "original numerical divergence"}
    runner.write_reports(tmp_path, report)
    assert report["error"] == "original numerical divergence"


def test_subset_main_is_partial_and_exit_two(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "checkout_evidence", lambda: {"head": None})
    monkeypatch.setattr(runner, "source_fingerprints", lambda: {"source": "hash"})
    monkeypatch.setattr(runner, "run_gate", lambda gate, *a, **k: {"name": gate.name, "status": "PASS", "passed": True})
    out = tmp_path / "new"
    assert runner.main(["--output-dir", str(out), "--gates", "frontend"]) == 2
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "PARTIAL" and not report["passed"]


def test_source_mutation_during_run_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "checkout_evidence", lambda: {"head": None})
    values = iter([{"source": "old"}, {"source": "new"}])
    monkeypatch.setattr(runner, "source_fingerprints", lambda: next(values))
    monkeypatch.setattr(runner, "run_gate", lambda gate, *a, **k: {"name": gate.name, "status": "PASS", "passed": True})
    assert runner.main(["--output-dir", str(tmp_path / "new"), "--gates", "frontend"]) == 1


def test_missing_assets_block_and_do_not_launch_dependent_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "checkout_evidence", lambda: {"head": None})
    monkeypatch.setattr(runner, "source_fingerprints", lambda: {})
    monkeypatch.setattr(runner, "run_gate", lambda *a, **k: pytest.fail("must not run without required assets"))
    out = tmp_path / "new"
    assert runner.main(["--output-dir", str(out), "--gates", "runtime-model"]) == 1
    data = json.loads((out / "report.json").read_text())
    assert [row["status"] for row in data["gates"] if row["name"] in {"runtime", "runtime-model"}] == ["BLOCKED", "BLOCKED"]


def test_help_has_no_heavy_imports(tmp_path):
    code = "import runpy,sys;sys.argv=['run_w2_acceptance.py','--help'];\ntry: runpy.run_path(" + repr(str(Path(runner.__file__))) + ",run_name='__main__')\nexcept SystemExit: pass\nassert not ({'torch','numpy','onnx','transformers'} & sys.modules.keys())"
    result = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
