"""Environment readiness must not hide dependency, source or LFS failures."""
import hashlib
import importlib.metadata
import json
from types import SimpleNamespace

import pytest

from scripts import check_w1_repro_env as env


def requirements(tmp_path, text="torch==2.7.1\nnumpy==2.2.6\n"):
    path = tmp_path / "requirements/qwen3-small-probe.txt"
    path.parent.mkdir()
    path.write_text(text, encoding="utf-8")


def test_local_cpu_version_tag_is_allowed(tmp_path, monkeypatch):
    requirements(tmp_path)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: {"torch": "2.7.1+cpu", "numpy": "2.2.6"}[name])
    assert env.check_packages(tmp_path)["ok"]


@pytest.mark.parametrize("installed", [None, "2.3.0", "2.2.6rc1"])
def test_missing_or_drifted_dependency_fails(tmp_path, monkeypatch, installed):
    requirements(tmp_path, "numpy==2.2.6\n")
    def version(name):
        if installed is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed
    monkeypatch.setattr(importlib.metadata, "version", version)
    assert not env.check_packages(tmp_path)["ok"]


@pytest.mark.parametrize("text", ["", "numpy>=2.2.6\n"])
def test_empty_or_unrecognized_pins_fail(tmp_path, text):
    requirements(tmp_path, text)
    with pytest.raises(ValueError):
        env.check_packages(tmp_path)


@pytest.mark.parametrize("cuda,hip,ok", [(None, None, True), ("12.6", None, False), (None, "6.3", False)])
def test_gpu_build_is_not_the_cpu_reference(monkeypatch, cuda, hip, ok):
    torch = SimpleNamespace(__version__="2.7.1", version=SimpleNamespace(cuda=cuda, hip=hip))
    monkeypatch.setattr(env.importlib, "import_module", lambda name: torch)
    assert env.check_cpu_torch()["ok"] is ok


def lfs_pointer(data):
    return ("version https://git-lfs.github.com/spec/v1\n"
            f"oid sha256:{hashlib.sha256(data).hexdigest()}\nsize {len(data)}")


def model_file(tmp_path, data):
    path = tmp_path / env.MODEL
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    return path


def test_model_matches_committed_lfs_identity(tmp_path, monkeypatch):
    data = b"real object bytes"
    path = model_file(tmp_path, data)
    monkeypatch.setattr(env, "run", lambda command, root: lfs_pointer(data))
    assert env.check_model(tmp_path)["ok"]
    path.write_bytes(b"different bytes!!")
    assert not env.check_model(tmp_path)["ok"]


def test_unhydrated_lfs_pointer_fails(tmp_path, monkeypatch):
    pointer = lfs_pointer(b"real model")
    model_file(tmp_path, pointer.encode())
    monkeypatch.setattr(env, "run", lambda command, root: pointer)
    with pytest.raises(ValueError, match="still an LFS pointer"):
        env.check_model(tmp_path)


def test_missing_lfs_object_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "run", lambda command, root: lfs_pointer(b"model"))
    with pytest.raises(FileNotFoundError):
        env.check_model(tmp_path)


def test_invalid_committed_pointer_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "run", lambda command, root: "invalid pointer")
    with pytest.raises(ValueError, match="Committed probe"):
        env.check_model(tmp_path)


def stub_environment(monkeypatch, *, dirty=False):
    def run(command, root):
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return "a" * 40
        if command[:2] == ["git", "status"]:
            return " M scratchv/compiler.py" if dirty else ""
        return "No broken requirements found."
    monkeypatch.setattr(env, "run", run)
    monkeypatch.setattr(env.sys, "version_info", (3, 12, 0))
    for name in ("check_packages", "check_cpu_torch", "check_native_imports", "check_tools", "check_model"):
        monkeypatch.setattr(env, name, lambda *a: {"ok": True})


def test_collect_records_failure_and_continues(tmp_path, monkeypatch):
    stub_environment(monkeypatch)
    def missing(*args):
        raise FileNotFoundError("qemu missing")
    monkeypatch.setattr(env, "check_tools", missing)
    report = env.collect(tmp_path)
    assert not report["ready"]
    assert "qemu missing" in report["checks"]["toolchain"]["error"]
    assert report["checks"]["synthetic_model"]["ok"]


def test_dirty_checkout_is_explicit_and_optionally_fails(tmp_path, monkeypatch):
    stub_environment(monkeypatch, dirty=True)
    report = env.collect(tmp_path)
    assert report["ready"] and report["warnings"]
    assert not report["checks"]["repository"]["clean"]
    assert not env.collect(tmp_path, require_clean=True)["ready"]


@pytest.mark.parametrize("is_zig,version,ok", [(True, "0.14.1", True), (True, "0.15.0", False), (False, "clang 18", False)])
def test_unvalidated_compiler_does_not_pass_reference_preflight(tmp_path, monkeypatch, is_zig, version, ok):
    from scratchv.runtime import riscv_tensor
    tools = SimpleNamespace(is_zig=is_zig, cc=("compiler",), qemu="qemu-system-riscv64")
    monkeypatch.setattr(riscv_tensor, "discover_toolchain", lambda **kwargs: tools)
    monkeypatch.setattr(riscv_tensor, "toolchain_versions", lambda tools: {"compiler": version})
    monkeypatch.setattr(env, "run", lambda *args: "virt\nrv64\n")
    assert env.check_tools(tmp_path, None, None)["ok"] is ok


@pytest.mark.parametrize("machines,cpus", [("pc Standard PC", "x86 qemu64"), ("virt ARM Virtual Machine", "cortex-a53\nmax"), ("virtual-prefix", "rv64-suffix")])
def test_wrong_qemu_architecture_fails(tmp_path, monkeypatch, machines, cpus):
    from scratchv.runtime import riscv_tensor
    tools = SimpleNamespace(is_zig=True, cc=("zig",), qemu="wrong-qemu")
    monkeypatch.setattr(riscv_tensor, "discover_toolchain", lambda **kwargs: tools)
    monkeypatch.setattr(riscv_tensor, "toolchain_versions", lambda tools: {"compiler": "0.14.1"})
    monkeypatch.setattr(env, "run", lambda cmd, root: machines if "-machine" in cmd else cpus)
    assert not env.check_tools(tmp_path, None, None)["ok"]


def test_non_emulator_tool_fails(tmp_path, monkeypatch):
    from scratchv.runtime import riscv_tensor
    tools = SimpleNamespace(is_zig=True, cc=("zig",), qemu="qemu-img")
    monkeypatch.setattr(riscv_tensor, "discover_toolchain", lambda **kwargs: tools)
    monkeypatch.setattr(riscv_tensor, "toolchain_versions", lambda tools: {"compiler": "0.14.1"})
    def reject(*args):
        raise RuntimeError("unknown -machine option")
    monkeypatch.setattr(env, "run", reject)
    with pytest.raises(RuntimeError, match="unknown -machine"):
        env.check_tools(tmp_path, None, None)


@pytest.mark.parametrize("providers,ok", [(["CPUExecutionProvider"], True), (["CUDAExecutionProvider"], False)])
def test_native_runtime_requires_cpu_provider(monkeypatch, providers, ok):
    module = SimpleNamespace(__version__="test", get_available_providers=lambda: providers)
    monkeypatch.setattr(env.importlib, "import_module", lambda name: module)
    assert env.check_native_imports()["ok"] is ok


def test_git_status_leading_column_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(env.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=" M file.py\n", stderr=""))
    assert env.run(["git", "status"], tmp_path) == " M file.py"


def test_cli_preserves_failed_report_and_does_not_overwrite(tmp_path, monkeypatch):
    stub_environment(monkeypatch)
    monkeypatch.setattr(env, "check_packages", lambda root: {"ok": False})
    out = tmp_path / "report"
    assert env.main(["--output-dir", str(out)]) == 1
    before = (out / "report.json").read_bytes()
    report = json.loads(before)
    assert not report["ready"] and "no numeric" in report["scope"]
    with pytest.raises(SystemExit) as error:
        env.main(["--output-dir", str(out)])
    assert error.value.code == 2
    assert (out / "report.json").read_bytes() == before
