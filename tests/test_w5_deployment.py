"""Exercise deployment orchestration without requiring a container engine."""
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/deploy_w5.sh"


@pytest.fixture
def setup(tmp_path):
    assets = tmp_path / "model assets"
    assets.mkdir()
    output = tmp_path / "new output"
    calls = tmp_path / "calls.jsonl"
    engine = tmp_path / "container engine"
    engine.write_text(
        "#!" + sys.executable + "\n"
        "import json, os, signal, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "with open(os.environ['ENGINE_CALLS'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1] == os.environ.get('ENGINE_BLOCK_STAGE'):\n"
        "    worker_code = 'import time; time.sleep(60)'\n"
        "    if os.environ.get('ENGINE_IGNORE_TERM'):\n"
        "        signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "        worker_code = 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'\n"
        "    worker = subprocess.Popen([sys.executable, '-c', worker_code])\n"
        "    Path(os.environ['ENGINE_PROCESS_FILE']).write_text(json.dumps([os.getpid(), worker.pid]))\n"
        "    time.sleep(60)\n"
        "sys.exit(int(os.environ.get('ENGINE_' + sys.argv[1].upper() + '_EXIT', '0')))\n"
    )
    engine.chmod(0o755)
    env = dict(os.environ, SCRATCHV_CONTAINER_ENGINE=str(engine),
               SCRATCHV_W5_ASSETS=str(assets), SCRATCHV_W5_OUTPUT=str(output),
               ENGINE_CALLS=str(calls))
    env.pop("SCRATCHV_W5_BASE_IMAGE", None)
    return env, calls, output


def run(env, *args):
    return subprocess.run(["bash", str(SCRIPT), *args], env=env,
                          text=True, capture_output=True, timeout=10)


def recorded(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_unavailable_engine_does_not_create_output(setup):
    env, calls, output = setup
    env["SCRATCHV_CONTAINER_ENGINE"] += "-absent"
    result = run(env, "--help")
    assert result.returncode == 127
    assert "unavailable" in result.stderr
    assert not calls.exists()
    assert not output.exists()


def test_unready_engine_does_not_build_or_create_output(setup):
    env, calls, output = setup
    env["ENGINE_INFO_EXIT"] = "1"
    result = run(env)
    assert result.returncode == 1
    assert "not ready" in result.stderr
    assert recorded(calls) == [["info"]]
    assert not output.exists()


def test_missing_assets_do_not_build_or_run(setup):
    env, calls, output = setup
    env["SCRATCHV_W5_ASSETS"] += "-absent"
    result = run(env)
    assert result.returncode == 2
    assert "Assets directory does not exist" in result.stderr
    assert recorded(calls) == [["info"]]
    assert not output.exists()


def test_build_failure_never_starts_generation(setup):
    env, calls, output = setup
    env["ENGINE_BUILD_EXIT"] = "7"
    result = run(env, "--prompt", "The capital of France is")
    assert result.returncode == 7
    assert [call[0] for call in recorded(calls)] == ["info", "build"]


def test_arguments_mounts_and_invoking_user_are_preserved(setup):
    env, calls, output = setup
    args = ["--model", "/assets/model.onnx", "--tokenizer-dir", "/assets/tokenizer",
            "--prompt", "quotes '$HOME'; * stay literal", "--harts", "auto",
            "--output-dir", "/output/second run"]
    result = run(env, *args)
    assert result.returncode == 0, result.stderr
    invocations = recorded(calls)
    assert [call[0] for call in invocations] == ["info", "build", "run"]
    build = invocations[1]
    assert build[build.index("--build-arg") + 1] == "BASE_IMAGE=ubuntu:24.04"
    command = invocations[-1]
    assert "--rm" in command and "--init" in command
    assert command[command.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert f"type=bind,src={env['SCRATCHV_W5_ASSETS']},dst=/assets,readonly" in command
    assert f"type=bind,src={output},dst=/output" in command
    assert command[-len(args):] == args
    image_index = command.index("scratchv-w5:local")
    assert command[image_index + 1:image_index + 3] == ["--output-dir", "/output/run"]
    assert "SCRATCHV_ZIG_CACHE=/output/.zig-cache" in command


@pytest.mark.parametrize("value,expected", [
    ("", "ubuntu:24.04"),
    ("public.ecr.aws/ubuntu/ubuntu:24.04", "public.ecr.aws/ubuntu/ubuntu:24.04"),
    ("literal $(printf not-expanded); *", "literal $(printf not-expanded); *"),
])
def test_base_image_override_is_one_literal_build_argument(setup, value, expected):
    env, calls, output = setup
    env["SCRATCHV_W5_BASE_IMAGE"] = value
    result = run(env, "--help")
    assert result.returncode == 0, result.stderr
    invocations = recorded(calls)
    assert [call[0] for call in invocations] == ["info", "build", "run"]
    build = invocations[1]
    assert build.count("--build-arg") == 1
    assert build[build.index("--build-arg") + 1] == "BASE_IMAGE=" + expected
    assert not any(arg.startswith("BASE_IMAGE=") for arg in invocations[-1])


def test_model_failure_exit_status_is_propagated(setup):
    env, calls, output = setup
    env["ENGINE_RUN_EXIT"] = "9"
    assert run(env).returncode == 9


def test_mount_delimiter_is_rejected_before_build(setup):
    env, calls, output = setup
    bad_assets = Path(env["SCRATCHV_W5_ASSETS"]).with_name("bad,assets")
    bad_assets.mkdir()
    env["SCRATCHV_W5_ASSETS"] = str(bad_assets)
    result = run(env)
    assert result.returncode == 2
    assert "commas" in result.stderr
    assert recorded(calls) == [["info"]]


def test_container_copy_inputs_exist_and_do_not_include_assets():
    # Detect a missing packaged entry point without claiming an image was built.
    dockerfile = ROOT / "deployment/w5/Dockerfile"
    for line in dockerfile.read_text().splitlines():
        if not line.startswith("COPY "):
            continue
        for source in shlex.split(line)[1:-1]:
            assert (ROOT / source).exists(), source
            assert source not in {".", "assets", "output", ".venv", ".git"}


def test_packaged_cli_help_resolves_transitive_modules(tmp_path):
    # Use only the image COPY inputs; no host model or generated ELF is copied.
    # This catches missing probe/manifest dependencies before a Docker build.
    dockerfile = ROOT / "deployment/w5/Dockerfile"
    for line in dockerfile.read_text().splitlines():
        if not line.startswith("COPY "):
            continue
        parts = shlex.split(line)
        for source in parts[1:-1]:
            src = ROOT / source
            target = tmp_path / source
            target.parent.mkdir(parents=True, exist_ok=True)
            if src.is_file():
                shutil.copyfile(src, target)
            elif source == "probes":
                for item in src.rglob("*"):
                    if item.is_file() and item.suffix in {".py", ".json"}:
                        dest = target / item.relative_to(src)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(item, dest)
            else:
                shutil.copytree(src, target,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    env = dict(os.environ, PYTHONPATH=str(tmp_path), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, str(tmp_path / "scripts/run_inference.py"), "--help"],
                            cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--reuse-build" in result.stdout
    assert "--reference-suite" in result.stdout


@pytest.mark.parametrize("stage,signum,ignore_term", [
    ("info", signal.SIGTERM, False),
    ("info", signal.SIGINT, False),
    ("build", signal.SIGTERM, False),
    ("build", signal.SIGINT, False),
    ("build", signal.SIGTERM, True),
])
def test_cancel_stage_reaps_client_and_stops_helpers_without_starting_model(
    setup, tmp_path, stage, signum, ignore_term,
):
    """A signal to Bash alone must reach the owned container-client group."""
    env, calls, output = setup
    process_file = tmp_path / "stage-processes.json"
    env.update(ENGINE_BLOCK_STAGE=stage, ENGINE_PROCESS_FILE=str(process_file))
    if ignore_term:
        env["ENGINE_IGNORE_TERM"] = "1"
    process = subprocess.Popen(["bash", str(SCRIPT)], env=env, start_new_session=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    owned = []
    try:
        deadline = time.monotonic() + 5
        while not process_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert process_file.exists(), "container-client stage did not start"
        owned = json.loads(process_file.read_text())
        assert os.getpgid(owned[0]) == owned[0]
        process.send_signal(signum)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 128 + signum, (stdout, stderr)
        # The direct client must be reaped, not left a zombie owned by Bash.
        with pytest.raises(ProcessLookupError):
            os.kill(owned[0], 0)
        # A helper can briefly await host PID 1's reap; it must not be running.
        status = Path(f"/proc/{owned[1]}/stat")
        if status.exists():
            assert status.read_text().rsplit(")", 1)[1].split()[0] == "Z"
        assert [call[0] for call in recorded(calls)] == (
            ["info"] if stage == "info" else ["info", "build"]
        )
        if stage == "info":
            assert not output.exists()
    finally:
        # Ensure a failed regression cannot leave its own blocking substitutes.
        for pgid in ([owned[0]] if owned else []) + [process.pid]:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=5)
