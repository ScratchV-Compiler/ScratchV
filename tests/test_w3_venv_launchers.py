"""Exercise launcher-selected Python environments without executing model work."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import venv

import pytest

from probes.w3_qwen3_full import run as full_gate
from scripts import run_w3_preparation as preparation


@pytest.mark.skipif(sys.platform != "linux", reason="Linux symlinked venv interpreter contract")
@pytest.mark.parametrize("launcher", ["full-worker", "preparation"])
@pytest.mark.parametrize("relative", [False, True], ids=["absolute-python", "relative-python"])
def test_w3_commands_execute_inside_selected_venv(tmp_path, monkeypatch, launcher, relative):
    selected = tmp_path / "selected-venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(selected)
    python = selected / "bin/python"
    assert python.is_symlink()
    # A dependency installed only in the chosen venv makes escaping it observable.
    site = subprocess.check_output(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        text=True,
    ).strip()
    (Path(site) / "w3_venv_marker.py").write_text("VALUE = 'selected environment'\n")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(tmp_path)
    chosen_python = os.path.relpath(python) if relative else str(python)
    args = SimpleNamespace(python=chosen_python, model_dir=tmp_path / "model",
                           source_dir=tmp_path / "source", cc=None, qemu=None,
                           qemu_timeout=5, worker_timeout=5, max_worker_memory_gib=1)
    output = tmp_path / "output"
    output.mkdir()
    # Replace only the expensive worker script with a dependency/prefix probe.
    # The launchers still construct their real commands and start real Python.
    script = ("import json, sys, w3_venv_marker\n"
              "print(json.dumps({'prefix': sys.prefix, 'base_prefix': sys.base_prefix, "
              "'dependency': w3_venv_marker.VALUE}))\n")
    if launcher == "full-worker":
        worker = workspace / "probes/w3_qwen3_full/worker.py"
        worker.parent.mkdir(parents=True)
        worker.write_text(script)
        monkeypatch.setattr(full_gate, "ROOT", workspace)
        row = full_gate.run_worker(args, output, "ort", tmp_path / "inputs.npz")
        assert row["passed"], (row, (output / "ort.log").read_text())
        observations = [json.loads((output / "ort.log").read_text())]
    else:
        observations = []
        for command in preparation.build_commands(args, output).values():
            worker = workspace / command[4]
            worker.parent.mkdir(parents=True, exist_ok=True)
            worker.write_text(script)
            child = subprocess.run(command, cwd=workspace, text=True, capture_output=True, timeout=5)
            assert child.returncode == 0, child.stderr
            observations.append(json.loads(child.stdout))
    for observed in observations:
        assert observed["prefix"] == str(selected)
        assert observed["base_prefix"] != observed["prefix"]
        assert observed["dependency"] == "selected environment"
