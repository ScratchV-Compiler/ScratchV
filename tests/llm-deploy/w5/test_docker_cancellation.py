"""Opt-in real Docker signal propagation, guest cleanup, and executable reuse."""
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import subprocess
import time
import uuid

import pytest


ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.skipif(
    os.environ.get("SCRATCHV_W5_DOCKER_TESTS") != "1",
    reason="set SCRATCHV_W5_DOCKER_TESTS=1 for real Docker small-graph checks",
)


def _docker(*args, check=True):
    return subprocess.run(
        ["docker", *args], text=True, capture_output=True, timeout=30, check=check,
    )


@pytest.fixture(scope="module")
def docker_assets(tmp_path_factory):
    # Explicit opt-in must fail on an unavailable engine, never silently skip.
    assert shutil.which("docker"), "Docker is required with SCRATCHV_W5_DOCKER_TESTS=1"
    assert Path("/proc/self/stat").exists(), "These checks require a local Linux Docker engine"
    ready = _docker("info", "--format", "{{.ServerVersion}}", check=False)
    assert ready.returncode == 0, ready.stderr
    assets = tmp_path_factory.mktemp("docker-small-assets")
    helper = runpy.run_path(str(ROOT / "tests/test_qemu_generation.py"))
    helper["make_small_onnx"](assets / "model.onnx")

    from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers
    tokenizer_dir = assets / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer = Tokenizer(models.WordLevel({str(i): i for i in range(8)}, unk_token="5"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.add_special_tokens([
        AddedToken(value, special=True, normalized=False) for value in ("0", "6")
    ])
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    added = {
        str(token_id): {
            name: getattr(token, name)
            for name in ("content", "lstrip", "rstrip", "single_word", "normalized", "special")
        }
        for token_id, token in tokenizer.get_added_tokens_decoder().items()
    }
    metadata = {
        "tokenizer_config.json": {
            "tokenizer_class": "Qwen2Tokenizer", "pad_token": "0", "eos_token": "6",
            "unk_token": None, "bos_token": None, "added_tokens_decoder": added,
            "additional_special_tokens": [],
            "chat_template": "{% for message in messages %}{{ message['content'] }} {% endfor %}",
        },
        "generation_config.json": {"pad_token_id": 0, "eos_token_id": [6]},
        "config.json": {"model_type": "qwen3", "vocab_size": 8},
    }
    for name, value in metadata.items():
        (tokenizer_dir / name).write_text(json.dumps(value), encoding="utf-8")

    # Pause only actual generation guests. Tool discovery and FMA preflight run
    # normally; no fake QEMU or production harness change is involved.
    wrapper = assets / "qemu-control"
    wrapper.write_text(
        "#!/usr/bin/env python\n"
        "import os, sys\n"
        "args = sys.argv[1:]\n"
        "if '/steps/' in os.getcwd() and os.path.exists('/assets/park'):\n"
        "    args += ['-S']\n"
        "os.execv('/usr/bin/qemu-system-riscv64', ['qemu-system-riscv64', *args])\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    image = f"scratchv-w5-cancellation:{uuid.uuid4().hex[:12]}"
    yield assets, image
    # Only this fixture's unique image is eligible for cleanup.
    _docker("image", "rm", image, check=False)


def _owned_containers(image, output):
    found = []
    listing = _docker("ps", "-aq", "--filter", f"ancestor={image}")
    for cid in listing.stdout.split():
        result = _docker("inspect", cid, check=False)
        if result.returncode:
            continue
        record = json.loads(result.stdout)[0]
        if any(m.get("Source") == str(output.resolve()) and m.get("Destination") == "/output"
               for m in record["Mounts"]):
            found.append(cid)
    return found


def _process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        status = Path(f"/proc/{pid}/status").read_text()
    except FileNotFoundError:
        return None
    nspid = next((line.split()[1:] for line in status.splitlines()
                  if line.startswith("NSpid:")), [str(pid)])
    return {"pid": int(pid), "pgid": int(fields[2]), "starttime": fields[19],
            "container_pid": int(nspid[-1])}


def _wait_parked(client, image, output):
    deadline = time.monotonic() + 600  # Includes the deployment script's image build.
    while time.monotonic() < deadline:
        assert client.poll() is None, f"Deployment exited before paused guest: {client.returncode}"
        for cid in _owned_containers(image, output):
            top = _docker("top", cid, "-eo", "pid,pgid,comm,args", check=False)
            if top.returncode:
                continue
            rows = [line.split(None, 3) for line in top.stdout.splitlines()[1:]]
            guests = [row for row in rows if len(row) == 4
                      and "qemu-system-riscv64" in row[3] and "-S" in row[3].split()]
            if guests:
                processes = [_process_identity(row[0]) for row in rows]
                processes = [item for item in processes if item is not None]
                guest = _process_identity(guests[0][0])
                assert guest is not None, "Paused guest disappeared unexpectedly"
                return cid, processes, guest
        time.sleep(0.2)
    pytest.fail("Timed out waiting for real QEMU -S inside the deployment container")


def _assert_reaped(cid, processes, guest, evidence):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        live = [item for item in processes if _process_identity(item["pid"]) == item]
        present = _docker("inspect", cid, check=False).returncode == 0
        if not live and not present:
            break
        time.sleep(0.1)
    assert not live, f"Container host processes survived cancellation: {live}"
    assert not present, f"--rm container survived cancellation: {cid}"
    time.sleep(0.3)
    assert all(_process_identity(item["pid"]) != item for item in processes)
    assert _docker("inspect", cid, check=False).returncode != 0
    # Inspect the actual guest process group independently of report assertions.
    group_members = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            identity = _process_identity(entry.name)
            if identity is not None and identity["pgid"] == guest["pgid"]:
                group_members.append(identity)
    assert not group_members, f"QEMU process group still has members: {group_members}"
    evidence.update(container_removed=True, host_processes_reaped=True,
                    qemu_group_empty=True, delayed_recheck_seconds=0.3)


def _report(output, name):
    return json.loads((output / name / "report.json").read_text(encoding="utf-8"))


def _assert_pass(record, harts):
    assert record["status"] == "PASS" and record["passed"] is True
    assert record["harts"] == harts and record["qemu_workspace_layout"] == "none"
    assert record["qemu_forwards"] == 4
    assert record["cases"][0]["generated_token_ids"] == [2, 3, 4, 6]
    assert record["cases"][0]["generated_text"] == "2 3 4"
    assert len(record["numeric_steps"]) == 4
    assert all(step["passed"] and step["max_abs"] == 0 for step in record["numeric_steps"])
    assert all(run["qemu_reaped"] and run["guest_completed"] for run in record["runs"])


@pytest.mark.parametrize("harts", [1, 4], ids=["default-single", "manual-four"])
def test_external_signals_reap_guest_and_reuse_build(tmp_path, docker_assets, harts):
    assets, image = docker_assets
    output = tmp_path / "deployment"
    output.mkdir()
    env = dict(os.environ, SCRATCHV_W5_ASSETS=str(assets), SCRATCHV_W5_OUTPUT=str(output),
               SCRATCHV_W5_IMAGE=image,
               SCRATCHV_W5_BASE_IMAGE=os.environ.get(
                   "SCRATCHV_W5_BASE_IMAGE", "public.ecr.aws/ubuntu/ubuntu:24.04"))
    env["SCRATCHV_CONTAINER_ENGINE"] = "docker"
    common = ["--tokenizer-dir", "/assets/tokenizer", "--prompt", "1",
              "--max-new-tokens", "8", "--verify-onnx", "/assets/model.onnx",
              "--qemu", "/assets/qemu-control", "--matmul-policy", "blocked_fma",
              "--timeout", "90", "--total-timeout", "180", "--build-timeout", "180"]
    # The single-core baseline really exercises the CLI default.
    hart_args = [] if harts == 1 else ["--harts", "4"]
    active = None
    evidence = {"harts": harts, "image": image, "cancellations": []}

    def launch(name, source, selection):
        log = (tmp_path / f"{name}.log").open("w", encoding="utf-8")
        try:
            return subprocess.Popen(
                ["bash", str(ROOT / "scripts/deploy_w5.sh"), *source, *common, *selection,
                 "--output-dir", f"/output/{name}"],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        finally:
            log.close()

    try:
        active = launch("baseline", ["--model", "/assets/model.onnx"], hart_args)
        assert active.wait(timeout=600) == 0, (tmp_path / "baseline.log").read_text()[-6000:]
        baseline = _report(output, "baseline")
        _assert_pass(baseline, harts)
        original = json.loads((output / "baseline/executable.json").read_text())
        evidence["baseline"] = baseline
        reuse = ["--reuse-build", "/output/baseline/executable.json"]
        auto = int(_docker("run", "--rm", "--entrypoint", "python", image, "-c",
                          "from scratchv.runtime.riscv_harts import resolve_harts; "
                          "print(resolve_harts('auto'))").stdout.strip())
        evidence["available_auto_harts"] = auto
        recovery_args = ["--harts", "auto"] if harts == 4 and auto == 4 else hart_args

        for sig in (signal.SIGINT, signal.SIGTERM):
            name = sig.name.lower()
            (assets / "park").touch()
            active = launch(name, reuse, hart_args)
            cid, processes, guest = _wait_parked(active, image, output)
            cancellation = {"signal": sig.name, "client_pid": active.pid, "container_id": cid,
                            "host_processes": processes, "qemu_host_process": guest}
            evidence["cancellations"].append(cancellation)
            # Exercise Docker's real signal proxy. killpg would bypass this chain.
            active.send_signal(sig)
            cancellation["exit_code"] = active.wait(timeout=30)
            assert cancellation["exit_code"] == 128 + int(sig), (
                tmp_path / f"{name}.log").read_text()[-6000:]
            _assert_reaped(cid, processes, guest, cancellation)
            cancelled = _report(output, name)
            cancellation["report"] = cancelled
            assert cancelled["status"] == "CANCELLED" and not cancelled["passed"]
            assert cancelled["harts"] == harts and cancelled["qemu_workspace_layout"] == "none"
            assert cancelled["qemu_forwards"] == 0 and len(cancelled["runs"]) == 1
            run = cancelled["runs"][0]
            assert run["qemu_reaped"] and not run["guest_completed"]
            assert run["qemu_pid"] == guest["container_pid"]
            (assets / "park").unlink()

            recovery_name = f"{name}-recovery"
            active = launch(recovery_name, reuse, recovery_args)
            assert active.wait(timeout=600) == 0, (tmp_path / f"{recovery_name}.log").read_text()[-6000:]
            recovered = _report(output, recovery_name)
            _assert_pass(recovered, harts)
            assert recovered["cases"][0]["generated_token_ids"] == baseline["cases"][0]["generated_token_ids"]
            saved = json.loads((output / recovery_name / "executable.json").read_text())
            assert all(saved[key] == original[key] for key in ("elf", "bundle_dir", "qemu", "harts"))
            assert recovered["harts_requested"] == ("auto" if recovery_args == ["--harts", "auto"] else harts)
            cancellation["recovery"] = recovered
        evidence["passed"] = True
    finally:
        (assets / "park").unlink(missing_ok=True)
        # Failure cleanup cannot satisfy the assertions above, and targets only
        # our unique image plus output mount and our separately started client.
        for cid in _owned_containers(image, output):
            _docker("rm", "-f", cid, check=False)
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGKILL)
            active.wait(timeout=15)
        (tmp_path / "docker-cancellation-summary.json").write_text(
            json.dumps(evidence, indent=2) + "\n", encoding="utf-8",
        )
