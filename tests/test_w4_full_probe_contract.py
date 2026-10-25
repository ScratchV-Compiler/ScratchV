"""Corrupt reuse metadata and failure publication, without model downloads.

These deliberately tiny metadata fixtures are not complete-model execution.
ELF arithmetic and the real guest transport are covered by the external-runtime
tests; here only that validator is stubbed to isolate runner trust boundaries.
"""

import copy
import hashlib
import json
import os
from dataclasses import replace
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


@pytest.mark.parametrize("harts", [1, 2, 4])
def test_reused_executable_harts_and_layout_agree(tmp_path, monkeypatch, harts):
    path, record = _saved_executable(tmp_path, monkeypatch)
    record["harts"] = record["evidence"]["harts"] = harts
    record["layout"] = plan_guest_memory(input_bytes=2048, output_bytes=256 * 151936 * 4,
                                         workspace_bytes=64, harts=harts)
    path.write_text(json.dumps(record), encoding="utf-8")
    assert run.load_executable(path).harts == harts


@pytest.mark.parametrize("harts", [True, 0, 3, "2", "auto"])
def test_reused_executable_rejects_invalid_harts(tmp_path, monkeypatch, harts):
    path, record = _saved_executable(tmp_path, monkeypatch)
    record["harts"] = record["evidence"]["harts"] = harts
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="hart"):
        run.load_executable(path)


def test_reused_executable_rejects_hart_count_changed_without_rebuild(tmp_path, monkeypatch):
    path, record = _saved_executable(tmp_path, monkeypatch)
    record["harts"] = 4
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="hart"):
        run.load_executable(path)


@pytest.mark.parametrize("requested", [None, 2, 4, "auto"])
@pytest.mark.parametrize("layout", ["none", "two-buffer-pad-4k"])
def test_full_cli_resolves_harts_once_before_tiny_graph_build(tmp_path, monkeypatch, requested, layout):
    from scratchv.frontend import onnx_parser
    from scratchv.ir.builder import IRBuilder
    from scratchv.ir.types import Value

    arguments, output = _mock_preflight_and_reuse(tmp_path, monkeypatch)
    reuse_index = arguments.index("--reuse-build")
    del arguments[reuse_index:reuse_index + 2]
    x = Value("x", shape=(2, 3))
    builder = IRBuilder()
    builder.new_function("main", [x])
    builder.new_block("entry")
    target = builder.neg(x)
    if layout != "none":
        target = builder.neg(builder.neg(target))
    builder.ret(target)
    parser = SimpleNamespace(initializers={}, parse=lambda *a, **kw: builder.program)
    monkeypatch.setattr(onnx_parser, "ONNXParser", lambda: parser)
    monkeypatch.setattr(run, "discover_toolchain", lambda *a: object())
    probes = []
    def available_cpus(pid):
        assert requested == "auto"
        probes.append(pid)
        return {3, 5, 7}
    monkeypatch.setattr(os, "sched_getaffinity", available_cpus, raising=False)
    expected = 2 if requested == "auto" else requested or 1
    built = []
    def stopped_build(artifact, *a, **kw):
        built.append(kw["harts"])
        assert type(kw["harts"]) is int and kw["harts"] == expected
        assert artifact.qemu_workspace_layout == layout
        raise RuntimeError("intentional tiny-graph build stop")
    monkeypatch.setattr(run, "build_external", stopped_build)
    if requested is not None:
        arguments += ["--harts", str(requested)]
    if layout != "none":
        arguments += ["--qemu-workspace-layout", layout]
    assert run.main(arguments + ["--build-only"]) == 1
    saved = json.loads((output / "report.json").read_text())
    assert built == [expected]
    assert probes == ([0] if requested == "auto" else [])
    assert saved["harts_requested"] == (requested if requested is not None else 1)
    assert saved["harts"] == expected
    assert saved["qemu_workspace_layout_requested"] == saved["codegen"]["qemu_workspace_layout"] == layout
    assert "intentional tiny-graph build stop" in saved["error"]
    assert not saved["full_qemu_forward_executed"]


@pytest.mark.parametrize("built_harts", [2, 4])
def test_full_cli_auto_reuse_requires_resolved_harts_to_match_build(tmp_path, monkeypatch, built_harts):
    arguments, output = _mock_preflight_and_reuse(tmp_path, monkeypatch)
    existing = run.load_executable(None)
    monkeypatch.setattr(run, "load_executable", lambda *_: replace(existing, harts=built_harts))
    probes = []
    monkeypatch.setattr(os, "sched_getaffinity",
                        lambda pid: probes.append(pid) or {0, 1, 2, 3}, raising=False)
    result = run.main(arguments + ["--harts", "auto", "--build-only"])
    saved = json.loads((output / "report.json").read_text())
    assert probes == [0]
    assert saved["harts_requested"] == "auto" and saved["harts"] == 4
    assert not saved["full_qemu_forward_executed"]
    if built_harts == 4:
        assert result == 0 and saved["status"] == "BUILD_ONLY"
    else:
        assert result == 1 and "hart count" in saved["error"]
        assert saved["stage"] == "load_existing_build"


@pytest.mark.parametrize("built_layout", [None, "none", "two-buffer-pad-4k"])
@pytest.mark.parametrize("requested_layout", ["none", "two-buffer-pad-4k"])
def test_full_cli_reuse_requires_same_qemu_workspace_layout(
        tmp_path, monkeypatch, built_layout, requested_layout):
    arguments, output = _mock_preflight_and_reuse(tmp_path, monkeypatch)
    existing = run.load_executable(None)
    evidence = {} if built_layout is None else {"qemu_workspace_layout": built_layout}
    monkeypatch.setattr(run, "load_executable", lambda *_: replace(existing, evidence=evidence))
    result = run.main(arguments + ["--qemu-workspace-layout", requested_layout, "--build-only"])
    saved = json.loads((output / "report.json").read_text())
    assert saved["qemu_workspace_layout_requested"] == requested_layout
    assert not saved["full_qemu_forward_executed"]
    if (built_layout or "none") == requested_layout:
        assert result == 0 and saved["status"] == "BUILD_ONLY"
    else:
        assert result == 1 and "workspace layout" in saved["error"]
        assert saved["stage"] == "load_existing_build"


def test_full_cli_rejects_unknown_qemu_workspace_layout_before_output_creation(tmp_path):
    target = tmp_path / "report"
    with pytest.raises(SystemExit) as exc:
        run.main(["--model-dir", str(tmp_path), "--output-dir", str(target),
                  "--qemu-workspace-layout", "automatic-padding"])
    assert exc.value.code == 2
    assert not target.exists()


@pytest.mark.parametrize("tile", [None, "1x4", "2x4", "1x8"])
def test_full_cli_matmul_tile_reaches_tiny_graph_build(tmp_path, monkeypatch, tile):
    from scratchv.frontend import onnx_parser
    from scratchv.ir.builder import IRBuilder
    from scratchv.ir.types import Value

    arguments, output = _mock_preflight_and_reuse(tmp_path, monkeypatch)
    reuse_index = arguments.index("--reuse-build")
    del arguments[reuse_index:reuse_index + 2]
    x, w = Value("x", shape=(2, 3)), Value("w", shape=(3, 2))
    builder = IRBuilder()
    builder.new_function("main", [x, w])
    builder.new_block("entry")
    builder.ret(builder.matmul(x, w))
    parser = SimpleNamespace(initializers={}, parse=lambda *a, **kw: builder.program)
    monkeypatch.setattr(onnx_parser, "ONNXParser", lambda: parser)
    monkeypatch.setattr(run, "discover_toolchain", lambda *a: object())
    generated = []

    def stopped_build(artifact, *a, **kw):
        generated.append(artifact)
        assert kw["harts"] == 1 and artifact.qemu_workspace_layout == "none"
        raise RuntimeError("intentional tile build stop")

    monkeypatch.setattr(run, "build_external", stopped_build)
    if tile is not None:
        arguments += ["--matmul-tile", tile, "--matmul-policy", "blocked_fma"]
    assert run.main(arguments + ["--build-only"]) == 1
    saved = json.loads((output / "report.json").read_text())
    assert len(generated) == 1 and generated[0].matmul_tile == (tile or "1x4")
    assert generated[0].matmul_policy == ("sequential" if tile is None else "blocked_fma")
    assert saved["matmul_tile"] == saved["codegen"]["matmul_tile"] == (tile or "1x4")
    assert "intentional tile build stop" in saved["error"]
    assert not saved["full_qemu_forward_executed"]


@pytest.mark.parametrize("built_tile", [None, "1x4", "2x4", "1x8"])
@pytest.mark.parametrize("requested_tile", ["1x4", "2x4", "1x8"])
def test_full_cli_reuse_requires_same_matmul_tile(tmp_path, monkeypatch, built_tile, requested_tile):
    arguments, output = _mock_preflight_and_reuse(tmp_path, monkeypatch)
    previous_path = tmp_path / "previous" / "report.json"
    previous = json.loads(previous_path.read_text())
    previous["matmul_policy"] = "blocked_fma"
    if built_tile is not None:
        previous["matmul_tile"] = built_tile
    previous_path.write_text(json.dumps(previous))
    existing = run.load_executable(None)
    evidence = {} if built_tile is None else {"matmul_tile": built_tile}
    monkeypatch.setattr(run, "load_executable", lambda *_: replace(existing, evidence=evidence))
    result = run.main(arguments + ["--matmul-policy", "blocked_fma",
                                  "--matmul-tile", requested_tile, "--build-only"])
    saved = json.loads((output / "report.json").read_text())
    assert saved["matmul_tile"] == requested_tile
    assert not saved["full_qemu_forward_executed"]
    if (built_tile or "1x4") == requested_tile:
        assert result == 0 and saved["status"] == "BUILD_ONLY"
    else:
        assert result == 1 and "MatMul tile" in saved["error"]
        assert saved["stage"] == "load_existing_build"


@pytest.mark.parametrize("report_tile,build_tile", [
    (None, "2x4"), ("1x4", "2x4"), ("2x4", None), ("2x4", "1x4"), ("1x8", "2x4"),
])
def test_full_cli_reuse_checks_report_and_build_matmul_tile(
        tmp_path, monkeypatch, report_tile, build_tile):
    arguments, output = _mock_preflight_and_reuse(tmp_path, monkeypatch)
    previous_path = tmp_path / "previous" / "report.json"
    previous = json.loads(previous_path.read_text())
    previous["matmul_policy"] = "blocked_fma"
    if report_tile is not None:
        previous["matmul_tile"] = report_tile
    previous_path.write_text(json.dumps(previous))
    existing = run.load_executable(None)
    evidence = {} if build_tile is None else {"matmul_tile": build_tile}
    monkeypatch.setattr(run, "load_executable", lambda *_: replace(existing, evidence=evidence))
    result = run.main(arguments + ["--matmul-policy", "blocked_fma", "--matmul-tile", "2x4",
                                  "--build-only"])
    saved = json.loads((output / "report.json").read_text())
    assert result == 1 and "MatMul tile" in saved["error"]
    assert saved["stage"] == "load_existing_build"
    assert not saved["full_qemu_forward_executed"]


@pytest.mark.parametrize("tile", ["2x4", "1x8", "unknown"])
def test_full_cli_rejects_invalid_matmul_tile_before_output_creation(tmp_path, tile):
    target = tmp_path / "report"
    with pytest.raises(SystemExit) as exc:
        run.main(["--model-dir", str(tmp_path), "--output-dir", str(target),
                  "--matmul-tile", tile])
    assert exc.value.code == 2
    assert not target.exists()

@pytest.mark.parametrize("rhs_layout", [None, "original", "column-panel"])
def test_full_cli_rhs_layout_reaches_tiny_graph_build(tmp_path, monkeypatch, rhs_layout):
    import numpy as np
    from scratchv.frontend import onnx_parser
    from scratchv.ir.builder import IRBuilder
    from scratchv.ir.types import Value

    arguments, output = _mock_preflight_and_reuse(tmp_path, monkeypatch)
    reuse_index = arguments.index("--reuse-build")
    del arguments[reuse_index:reuse_index + 2]
    x, w = Value("x", shape=(2, 3)), Value("w", shape=(3, 2))
    builder = IRBuilder()
    builder.new_function("main", [x])
    builder.new_block("entry")
    builder.program.global_values.append(w)
    builder.ret(builder.matmul(x, w))
    parser = SimpleNamespace(initializers={"w": np.arange(6, dtype=np.float32).reshape(3, 2)},
                             parse=lambda *a, **kw: builder.program)
    monkeypatch.setattr(onnx_parser, "ONNXParser", lambda: parser)
    monkeypatch.setattr(run, "discover_toolchain", lambda *a: object())
    generated = []

    def stopped_build(artifact, *a, **kw):
        generated.append(artifact)
        assert kw["harts"] == 1 and artifact.qemu_workspace_layout == "none"
        raise RuntimeError("intentional RHS layout build stop")

    monkeypatch.setattr(run, "build_external", stopped_build)
    if rhs_layout is not None:
        arguments += ["--matmul-rhs-layout", rhs_layout, "--matmul-policy", "blocked_fma"]
    assert run.main(arguments + ["--build-only"]) == 1
    saved = json.loads((output / "report.json").read_text())
    assert len(generated) == 1
    model = generated[0]
    assert model.matmul_rhs_layout == (rhs_layout or "original")
    assert bool(model.matmul_rhs_panels) is (rhs_layout == "column-panel")
    assert saved["matmul_rhs_layout"] == saved["codegen"]["matmul_rhs_layout"] == model.matmul_rhs_layout
    assert saved["codegen"]["matmul_rhs_panels"] == list(model.matmul_rhs_panels)
    assert "intentional RHS layout build stop" in saved["error"]
    assert not saved["full_qemu_forward_executed"]


@pytest.mark.parametrize("built_layout", [None, "original", "column-panel"])
@pytest.mark.parametrize("requested_layout", ["original", "column-panel"])
def test_full_cli_reuse_requires_same_rhs_layout(
        tmp_path, monkeypatch, built_layout, requested_layout):
    arguments, output = _mock_preflight_and_reuse(tmp_path, monkeypatch)
    previous_path = tmp_path / "previous" / "report.json"
    previous = json.loads(previous_path.read_text())
    previous["matmul_policy"] = "blocked_fma"
    evidence = {}
    if built_layout is not None:
        previous["matmul_rhs_layout"] = built_layout
        previous["build"] = {"matmul_rhs_panels": []}
        evidence = {"matmul_rhs_layout": built_layout, "matmul_rhs_panels": []}
    previous_path.write_text(json.dumps(previous))
    existing = run.load_executable(None)
    monkeypatch.setattr(run, "load_executable", lambda *_: replace(existing, evidence=evidence))
    result = run.main(arguments + ["--matmul-policy", "blocked_fma",
                                  "--matmul-rhs-layout", requested_layout, "--build-only"])
    saved = json.loads((output / "report.json").read_text())
    assert saved["matmul_rhs_layout"] == requested_layout
    assert not saved["full_qemu_forward_executed"]
    if (built_layout or "original") == requested_layout:
        assert result == 0 and saved["status"] == "BUILD_ONLY"
    else:
        assert result == 1 and "MatMul RHS layout" in saved["error"]
        assert saved["stage"] == "load_existing_build"


@pytest.mark.parametrize("report_layout,build_layout", [
    (None, "column-panel"), ("original", "column-panel"),
    ("column-panel", None), ("column-panel", "original"),
])
def test_full_cli_reuse_checks_report_and_build_rhs_layout(
        tmp_path, monkeypatch, report_layout, build_layout):
    arguments, output = _mock_preflight_and_reuse(tmp_path, monkeypatch)
    previous_path = tmp_path / "previous" / "report.json"
    previous = json.loads(previous_path.read_text())
    previous["matmul_policy"] = "blocked_fma"
    if report_layout is not None:
        previous["matmul_rhs_layout"] = report_layout
    previous["build"] = {"matmul_rhs_panels": []}
    previous_path.write_text(json.dumps(previous))
    existing = run.load_executable(None)
    evidence = {"matmul_rhs_panels": []}
    if build_layout is not None:
        evidence["matmul_rhs_layout"] = build_layout
    monkeypatch.setattr(run, "load_executable", lambda *_: replace(existing, evidence=evidence))
    result = run.main(arguments + ["--matmul-policy", "blocked_fma",
                                  "--matmul-rhs-layout", "column-panel", "--build-only"])
    saved = json.loads((output / "report.json").read_text())
    assert result == 1 and "MatMul RHS layout" in saved["error"]
    assert saved["stage"] == "load_existing_build"
    assert not saved["full_qemu_forward_executed"]


@pytest.mark.parametrize("fault", ["missing-report", "missing-build", "different", "invalid-type"])
def test_full_cli_reuse_requires_matching_rhs_panel_metadata(tmp_path, monkeypatch, fault):
    arguments, output = _mock_preflight_and_reuse(tmp_path, monkeypatch)
    previous_path = tmp_path / "previous" / "report.json"
    previous = json.loads(previous_path.read_text())
    previous.update(matmul_policy="blocked_fma", matmul_rhs_layout="column-panel",
                    build={"matmul_rhs_panels": []})
    evidence = {"matmul_rhs_layout": "column-panel", "matmul_rhs_panels": []}
    if fault == "missing-report":
        previous.pop("build")
    elif fault == "missing-build":
        evidence.pop("matmul_rhs_panels")
    elif fault == "different":
        evidence["matmul_rhs_panels"] = [{"name": "wrong-physical-weight"}]
    else:
        previous["build"]["matmul_rhs_panels"] = evidence["matmul_rhs_panels"] = None
    previous_path.write_text(json.dumps(previous))
    existing = run.load_executable(None)
    monkeypatch.setattr(run, "load_executable", lambda *_: replace(existing, evidence=evidence))
    result = run.main(arguments + ["--matmul-policy", "blocked_fma",
                                  "--matmul-rhs-layout", "column-panel", "--build-only"])
    saved = json.loads((output / "report.json").read_text())
    assert result == 1 and "MatMul RHS panel metadata" in saved["error"]
    assert saved["stage"] == "load_existing_build"
    assert not saved["full_qemu_forward_executed"]


@pytest.mark.parametrize("layout", ["column-panel", "unknown"])
def test_full_cli_rejects_rhs_layout_before_output_creation(tmp_path, layout):
    target = tmp_path / "report"
    with pytest.raises(SystemExit) as exc:
        run.main(["--model-dir", str(tmp_path), "--output-dir", str(target),
                  "--matmul-rhs-layout", layout])
    assert exc.value.code == 2
    assert not target.exists()
