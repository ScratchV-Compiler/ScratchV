"""Corrupt reuse metadata and failure publication, without model downloads.

These deliberately tiny metadata fixtures are not complete-model execution.
ELF arithmetic and the real guest transport are covered by the external-runtime
tests; here only that validator is stubbed to isolate runner trust boundaries.
"""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from probes import w3_common as common
from probes.w4_qwen3_full import run
from scratchv.runtime.weight_bundle import plan_guest_memory, write_weight_bundle


def _saved_executable(tmp_path, monkeypatch):
    build = tmp_path / "build"
    build.mkdir()
    elf = build / "model.elf"
    elf.write_bytes(b"isolated runner fixture; not an executable")
    bundle = tmp_path / "weights"
    manifest = write_weight_bundle([], [], bundle)
    sources = {}
    for name in ("model.c", "guest.c", "start.S", "link.ld"):
        data = f"fixture source {name}\n".encode()
        (build / name).write_bytes(data)
        sources[name] = hashlib.sha256(data).hexdigest()
    evidence = {"bytes": elf.stat().st_size, "entry": 0x80000000,
                "segments": [], "sha256": hashlib.sha256(elf.read_bytes()).hexdigest()}
    monkeypatch.setattr(run, "validate_external_elf", lambda *_: dict(evidence))
    layout = plan_guest_memory(input_bytes=2048, output_bytes=256 * 151936 * 4,
                               workspace_bytes=64)
    record = {"elf": str(elf), "bundle_dir": str(bundle), "manifest": manifest,
              "layout": layout, "inputs": [{"name": "input_ids", "dtype": "i64", "shape": [1, 256]}],
              "weights": [], "output": {"name": "logits", "dtype": "f32", "shape": [1, 256, 151936]},
              "cc": ["zig", "cc"], "qemu": "qemu-system-riscv64",
              "evidence": {**evidence, "sources": sources}}
    path = tmp_path / "executable.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path, record


def test_unchanged_reuse_metadata_checks_bundle_layout_and_generated_sources(tmp_path, monkeypatch):
    path, record = _saved_executable(tmp_path, monkeypatch)
    executable = run.load_executable(path)
    assert executable.manifest == record["manifest"]
    assert executable.layout == record["layout"]
    assert executable.output.shape == (1, 256, 151936)


@pytest.mark.parametrize("omission", ["all", "model.c", "guest.c", "start.S", "link.ld"])
def test_reuse_rejects_missing_generated_source_evidence(tmp_path, monkeypatch, omission):
    path, record = _saved_executable(tmp_path, monkeypatch)
    if omission == "all":
        record["evidence"]["sources"] = {}
    else:
        del record["evidence"]["sources"][omission]
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises((ValueError, KeyError), match="source|Source"):
        run.load_executable(path)


def test_reuse_rejects_generated_source_changed_since_build(tmp_path, monkeypatch):
    path, record = _saved_executable(tmp_path, monkeypatch)
    (Path(record["elf"]).parent / "model.c").write_text("corrupted", encoding="utf-8")
    with pytest.raises(ValueError, match="source"):
        run.load_executable(path)


def test_reuse_rejects_corrupted_weight_bytes(tmp_path, monkeypatch):
    path, record = _saved_executable(tmp_path, monkeypatch)
    (Path(record["bundle_dir"]) / "weights.bin").write_bytes(b"extra")
    with pytest.raises(ValueError, match="length"):
        run.load_executable(path)


def test_reuse_rejects_memory_overlap(tmp_path, monkeypatch):
    path, record = _saved_executable(tmp_path, monkeypatch)
    record["layout"]["workspace_base"] = record["layout"]["code_base"]
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="layout"):
        run.load_executable(path)


def test_reuse_rejects_duplicate_json_keys(tmp_path, monkeypatch):
    path, record = _saved_executable(tmp_path, monkeypatch)
    data = json.dumps(record)
    path.write_text('{"qemu": "ambiguous alternative",' + data[1:], encoding="utf-8")
    with pytest.raises(ValueError, match="uplicate|JSON"):
        run.load_executable(path)


def test_reuse_rejects_bool_output_dimension_even_with_self_consistent_layout(tmp_path, monkeypatch):
    path, record = _saved_executable(tmp_path, monkeypatch)
    record["output"]["shape"] = [1, True, 151936]
    record["layout"] = plan_guest_memory(input_bytes=2048, output_bytes=151936 * 4,
                                         workspace_bytes=64)
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="shape|Shape|dimension|static"):
        run.load_executable(path)


def test_reuse_source_filename_cannot_escape_build_directory(tmp_path, monkeypatch):
    path, record = _saved_executable(tmp_path, monkeypatch)
    record["evidence"]["sources"]["../outside"] = "0" * 64
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="source"):
        run.load_executable(path)


def _report():
    return {"passed": True, "status": "PASS", "stage": "complete", "gates": {"numeric": True}}


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), object()])
def test_serialization_failure_cannot_leave_previous_pass_views(tmp_path, bad_value):
    run.write_reports(tmp_path, _report())
    report = _report()
    report["unexpected"] = bad_value
    with pytest.raises((ValueError, TypeError)):
        run.write_reports(tmp_path, report)
    if (tmp_path / "report.json").exists():
        saved = json.loads((tmp_path / "report.json").read_text())
        assert saved.get("passed") is not True
        assert saved.get("status") != "PASS"
    for filename in ("report.md", "report.html"):
        if (tmp_path / filename).exists():
            assert '"passed": true' not in (tmp_path / filename).read_text()


def _mock_preflight_and_reuse(tmp_path, monkeypatch, previous_sources=None):
    import onnx
    sources = {"scratchv/backend/tensor_c_codegen.py": "a" * 64,
               "scratchv/runtime/riscv_external.py": "b" * 64}
    identity = {"model.onnx": {"sha256": "c" * 64}}
    previous = {"model_files": identity, "source_sha256": sources if previous_sources is None else previous_sources,
                "matmul_policy": "sequential",
                "executable_sha256": hashlib.sha256(b"{}").hexdigest()}
    previous_dir = tmp_path / "previous"
    previous_dir.mkdir()
    (previous_dir / "report.json").write_text(json.dumps(previous), encoding="utf-8")
    executable_path = previous_dir / "executable.json"
    executable_path.write_text("{}", encoding="utf-8")
    compiler, qemu = tmp_path / "mock-cc", tmp_path / "mock-qemu"
    compiler.write_bytes(b"fixture compiler identity, never executed")
    qemu.write_bytes(b"fixture QEMU identity, never executed")
    executable = run.ExternalExecutable(previous_dir / "model.elf", previous_dir / "weights",
        {"sha256": "d" * 64}, {}, (), None, (),
        run.RiscVToolchain((str(compiler),), str(qemu)), {})
    monkeypatch.setattr(run, "source_evidence", lambda: {"source_sha256": dict(sources)})
    monkeypatch.setattr(common, "source_evidence", lambda: {"source_sha256": dict(sources)})
    monkeypatch.setattr(run, "save_source_snapshot", lambda *_: {})
    monkeypatch.setattr(run, "toolchain_versions", lambda *_: {})
    monkeypatch.setattr(run.assets, "verify_files", lambda *_: copy.deepcopy(identity))
    monkeypatch.setattr(onnx, "load", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(run, "audit_graph_structure", lambda *_: {"passed": True, "layer_count": 28})
    monkeypatch.setattr(run, "load_executable", lambda *_: executable)
    monkeypatch.setattr(run, "save_executable", lambda *_: "e" * 64)
    monkeypatch.setattr(run, "run_fma_probe", lambda _out, *, cc, qemu: {
        "passed": True, "status": "PASS", "tool_binary_sha256": {
            "cc": run.sha256_file(cc[0]), "qemu": run.sha256_file(qemu)}})
    output = tmp_path / "attempt"
    arguments = ["--model-dir", str(tmp_path / "fixed-model"), "--output-dir", str(output),
                 "--reuse-build", str(executable_path), "--matmul-policy", "sequential"]
    return arguments, output


@pytest.mark.parametrize("case_list", [[], [("full_seed_42", 256, {})]])
def test_missing_requested_case_never_passes_without_forward(tmp_path, monkeypatch, case_list):
    arguments, output = _mock_preflight_and_reuse(tmp_path, monkeypatch)
    monkeypatch.setattr(run, "input_cases", lambda: case_list)
    exit_code = run.main(arguments + ["--case", "full_seed_0"])
    saved = json.loads((output / "report.json").read_text())
    assert exit_code != 0
    assert saved["passed"] is False
    assert saved["gates"]["numeric:qemu-full-qwen3"] is False
    assert saved["full_qemu_forward_executed"] is False
    assert "case" in saved.get("error", "").lower()


@pytest.mark.parametrize("previous_sources", [{}, {"scratchv/backend/tensor_c_codegen.py": "a" * 64}])
def test_reuse_requires_all_current_production_source_hashes(tmp_path, monkeypatch, previous_sources):
    arguments, output = _mock_preflight_and_reuse(tmp_path, monkeypatch, previous_sources)
    exit_code = run.main(arguments + ["--build-only"])
    saved = json.loads((output / "report.json").read_text())
    assert exit_code != 0
    assert saved["passed"] is False
    assert "source" in saved.get("error", "").lower()


def test_reuse_cannot_change_metadata_without_updating_trusted_previous_report(tmp_path, monkeypatch):
    arguments, output = _mock_preflight_and_reuse(tmp_path, monkeypatch)
    (tmp_path / "previous/executable.json").write_text('{"qemu": "replacement"}', encoding="utf-8")
    assert run.main(arguments + ["--build-only"]) != 0
    saved = json.loads((output / "report.json").read_text())
    assert saved["passed"] is False
    assert "metadata hash" in saved.get("error", "").lower()


def test_keyboard_interrupt_preserves_failure_and_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "source_evidence", lambda: {"source_sha256": {}})
    monkeypatch.setattr(run, "save_source_snapshot", lambda *_: {})
    def interrupted(*_):
        raise KeyboardInterrupt("injected before model parsing")
    monkeypatch.setattr(run.assets, "verify_files", interrupted)
    output = tmp_path / "interrupted"
    assert run.main(["--model-dir", str(tmp_path / "model"), "--output-dir", str(output)]) != 0
    saved = json.loads((output / "report.json").read_text())
    assert saved["stage"] == "asset_hashes"
    assert saved["interrupted"] is True
    assert saved["passed"] is False
    assert not any(saved["gates"].values())


@pytest.mark.parametrize("damage", ["failed", "wrong_status", "compiler_hash", "qemu_hash", "missing_hash"])
def test_fma_failure_or_wrong_execution_binary_stops_before_ort_and_full_qemu(tmp_path, monkeypatch, damage):
    arguments, output = _mock_preflight_and_reuse(tmp_path, monkeypatch)
    calls = []
    def conformance(_out, *, cc, qemu):
        calls.append((tuple(cc), qemu))
        report = {"passed": True, "status": "PASS", "tool_binary_sha256": {
            "cc": run.sha256_file(cc[0]), "qemu": run.sha256_file(qemu)}}
        if damage == "failed":
            report["passed"] = False
        elif damage == "wrong_status":
            report["status"] = "FAIL"
        elif damage == "missing_hash":
            del report["tool_binary_sha256"]
        else:
            report["tool_binary_sha256"]["cc" if damage == "compiler_hash" else "qemu"] = "0" * 64
        return report
    monkeypatch.setattr(run, "run_fma_probe", conformance)
    reached = []
    monkeypatch.setattr(run, "ort_reference", lambda *_: reached.append("ort"))
    monkeypatch.setattr(run, "run_external", lambda *_args, **_kwargs: reached.append("full_qemu"))
    assert run.main(arguments) != 0
    saved = json.loads((output / "report.json").read_text())
    assert len(calls) == 1 and reached == []
    assert saved["stage"] == "fma_conformance"
    assert "FMA conformance" in saved["error"]
    assert saved["full_qemu_forward_executed"] is False
    assert saved["gates"]["numeric:qemu-full-qwen3"] is False


def test_reuse_explicit_qemu_overrides_runtime_and_rechecks_its_exact_binary(tmp_path, monkeypatch):
    arguments, output = _mock_preflight_and_reuse(tmp_path, monkeypatch)
    replacement = tmp_path / "replacement-qemu"
    replacement.write_bytes(b"a new runtime with a distinct binary identity")
    called, saved_executables = [], []
    def conformance(_out, *, cc, qemu):
        called.append(qemu)
        return {"passed": True, "status": "PASS", "tool_binary_sha256": {
            "cc": run.sha256_file(cc[0]), "qemu": run.sha256_file(qemu)}}
    monkeypatch.setattr(run, "run_fma_probe", conformance)
    monkeypatch.setattr(run, "save_executable", lambda exe, _out: saved_executables.append(exe) or "e" * 64)
    monkeypatch.setattr(run, "input_cases", lambda: [])  # Stop after the gate, before any model forward.
    assert run.main(arguments + ["--qemu", str(replacement)]) != 0
    saved = json.loads((output / "report.json").read_text())
    assert called == [str(replacement.resolve())]
    assert saved_executables[0].toolchain.qemu == str(replacement.resolve())
    assert saved["qemu_override"]["built_with"] == str(tmp_path / "mock-qemu")
    assert saved["qemu_override"]["execute_with"] == str(replacement.resolve())
    assert saved["execution_tool_binary_sha256"]["qemu"] == run.sha256_file(replacement)
    assert saved["fma_conformance"]["tool_binary_sha256"] == saved["execution_tool_binary_sha256"]
    assert "case" in saved["error"].lower()


def test_reuse_rejects_explicit_compiler_before_conformance_or_full_forward(tmp_path, monkeypatch):
    arguments, output = _mock_preflight_and_reuse(tmp_path, monkeypatch)
    called = []
    monkeypatch.setattr(run, "run_fma_probe", lambda *_args, **_kwargs: called.append("fma"))
    assert run.main(arguments + ["--cc", str(tmp_path / "new-cc")]) != 0
    saved = json.loads((output / "report.json").read_text())
    assert called == []
    assert "--cc" in saved["error"]
    assert saved["full_qemu_forward_executed"] is False


@pytest.mark.parametrize("build_only", [False, True])
@pytest.mark.parametrize("damage", ["unchanged", "changed", "missing", "read_error"])
def test_final_source_recheck_controls_success_and_build_only(tmp_path, monkeypatch, build_only, damage):
    """Synthetic forward isolates publication; it is not full-model evidence."""
    import numpy as np

    arguments, output = _mock_preflight_and_reuse(tmp_path, monkeypatch)
    before = common.source_evidence()["source_sha256"]
    after = dict(before)
    changed_name = "scratchv/backend/tensor_c_codegen.py"
    if damage == "changed":
        after[changed_name] = "f" * 64
    elif damage == "missing":
        del after[changed_name]
    checks = []

    def final_sources():
        checks.append(True)
        if damage == "read_error":
            raise OSError("injected final source read failure")
        return {"source_sha256": dict(after)}

    monkeypatch.setattr(common, "source_evidence", final_sources)
    if build_only:
        arguments += ["--build-only"]
    else:
        # Avoid model execution: this checks that successful earlier stages do
        # not bypass the final source contract or publish a false PASS on error.
        monkeypatch.setattr(run, "input_cases", lambda: [("full_seed_0", 256, {})])
        monkeypatch.setattr(run, "ort_reference", lambda _model, _feed, path:
                            np.save(path, np.zeros(1, np.float32), allow_pickle=False))
        monkeypatch.setattr(run, "run_external", lambda *_args, **_kwargs:
                            (np.zeros(1, np.float32), {"passed": True, "status": "PASS"}))
        monkeypatch.setattr(run, "compare_logits", lambda *_:
                            {"passed": True, "max_abs": 0.0})
    exit_code = run.main(arguments)
    saved = json.loads((output / "report.json").read_text())
    assert checks == [True]
    if damage == "unchanged":
        assert exit_code == 0
        assert saved["source_recheck"] == {"passed": True, "changed": {}}
        assert saved["status"] == ("BUILD_ONLY" if build_only else "PASS")
        assert saved["passed"] is (not build_only)
    else:
        assert exit_code == 1 and saved["status"] == "FAIL" and saved["passed"] is False
        assert saved["stage"] == "source_postcheck"
        assert saved["gates"]["numeric:qemu-full-qwen3"] is False
        if damage == "read_error":
            assert "injected final source read failure" in saved["error"]
        else:
            assert saved["source_recheck"]["passed"] is False
            assert saved["source_recheck"]["changed"] == {changed_name: {
                "before_sha256": before[changed_name], "after_sha256": after.get(changed_name)}}


def test_original_failure_does_not_attempt_final_source_recheck(tmp_path, monkeypatch):
    arguments, output = _mock_preflight_and_reuse(tmp_path, monkeypatch)
    checks = []

    def final_sources():
        checks.append(True)
        raise OSError("this secondary source failure must not replace the original")

    monkeypatch.setattr(common, "source_evidence", final_sources)
    monkeypatch.setattr(run, "input_cases", lambda: [])
    assert run.main(arguments) == 1
    saved = json.loads((output / "report.json").read_text())
    assert checks == []
    assert "Selected cases" in saved["error"]
    assert "source_recheck" not in saved
