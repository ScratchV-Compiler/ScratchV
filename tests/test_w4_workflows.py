"""Execute W4 workflow gate scripts without downloading assets or running CI."""

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
W5_REQUIRED = (
    [f"test_real_qemu_multitoken_reuse_numerics_timeout_and_cancel[{h}]" for h in (1, 4)]
    + [f"test_explicit_mean_order_rv64_and_context_reuse[{w}-{h}]"
       for w in (13, 128, 1024) for h in (1, 4)]
    + [f"test_explicit_mean_overflow_propagates_and_build_remains_reusable[{h}]" for h in (1, 4)]
    + [f"test_reference_nonlinear_rv64_contexts_and_error_recovery[{h}]" for h in (1, 4)]
)

PATH = ROOT / ".github/workflows/w4-full-numeric.yml"
NIGHTLY_PATH = ROOT / ".github/workflows/w4-nightly.yml"
LLM_PATH = ROOT / ".github/workflows/llm-deploy.yml"
CI_PATH = ROOT / ".github/workflows/ci.yml"


def workflow(path=PATH):
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def step(label, path=PATH):
    return next(item for job in workflow(path)["jobs"].values() for item in job.get("steps", [])
                if item.get("name") == label)


def python_body(label, path=PATH):
    script = step(label, path)["run"]
    return re.split(r"python3? - <<'PY'\n", script, maxsplit=1)[1].rsplit("\nPY", 1)[0]


def _guard(expression, event):
    expression = expression.removeprefix("${{").removesuffix("}}").strip()
    tree = ast.parse(expression.replace("github.event_name", repr(event)), mode="eval").body
    assert isinstance(tree, ast.Compare) and len(tree.ops) == 1
    left, right = ast.literal_eval(tree.left), ast.literal_eval(tree.comparators[0])
    if isinstance(tree.ops[0], ast.NotEq):
        return left.casefold() != right.casefold()
    if isinstance(tree.ops[0], ast.Eq):
        return left.casefold() == right.casefold()
    raise AssertionError("Unsupported workflow guard")


def test_expensive_forward_is_not_automatic_on_pr_or_enabled_as_a_schedule():
    data = workflow()
    assert set(data["on"]) == {"workflow_call", "workflow_dispatch"}
    run_full = data["on"]["workflow_call"]["inputs"]["run_full"]
    assert run_full["type"] == "boolean" and run_full["default"] == "false"
    assert data["permissions"] == {"contents": "read"}
    assert data["concurrency"]["cancel-in-progress"] == "false"
    full = data["jobs"]["full-numeric"]
    assert full["needs"] == "unit"
    assert full["if"] == ("${{ !inputs.run_w5 && inputs.mode != 'w5-generation' && "
                          "(inputs.run_full || (github.event_name == 'workflow_dispatch' "
                          "&& inputs.mode == 'w4-forward')) }}")
    assert "if" not in data["jobs"]["unit"]


def test_single_ci_entry_calls_w4_unit_only_on_pull_requests():
    data = workflow(CI_PATH)
    assert "pull_request" in data["on"]
    calls = [job for job in data["jobs"].values()
             if job.get("uses") == "./.github/workflows/w4-full-numeric.yml"]
    assert len(calls) == 1
    caller = calls[0]
    assert caller["with"]["run_full"] == "false"
    assert caller["needs"] == "scope"
    event_guard, scope_guard = caller["if"].removeprefix("${{").removesuffix("}}").split("&&")
    for event in ("pull_request", "push", "schedule", "workflow_dispatch"):
        for selected in ("true", "false"):
            allowed = _guard(event_guard, event) and _guard(
                scope_guard.replace("needs.scope.outputs.w4", "github.event_name"), selected)
            assert allowed is (event == "pull_request" and selected == "true")
    assert "continue-on-error" not in caller


def test_same_ref_unit_and_full_calls_do_not_share_a_concurrency_group():
    concurrency = workflow()["concurrency"]
    group = concurrency["group"]
    assert "${{ inputs.run_full && 'full' || 'unit' }}" in group
    assert concurrency["cancel-in-progress"] == "false"
    fixed = group.replace("${{ github.ref }}", "refs/heads/candidate").replace(
        "${{ inputs.case || 'full_seed_0' }}", "full_seed_0").replace(
        "${{ inputs.run_w5 && 'w5' || inputs.mode || 'call' }}", "call")
    unit = fixed.replace("${{ inputs.run_full && 'full' || 'unit' }}", "unit")
    full = fixed.replace("${{ inputs.run_full && 'full' || 'unit' }}", "full")
    assert unit != full and "${{" not in unit + full
    assert group != workflow(NIGHTLY_PATH)["concurrency"]["group"]


def test_registered_llm_entry_calls_same_commit_w4_without_changing_w3_or_pr_defaults():
    data = workflow(LLM_PATH)
    inputs = data["on"]["workflow_dispatch"]["inputs"]
    assert inputs["run_w4_validation"]["type"] == "boolean"
    assert inputs["run_w4_validation"]["default"] == "false"
    assert inputs["w4_case"]["type"] == "choice"
    case = workflow()["on"]["workflow_call"]["inputs"]["case"]
    assert inputs["w4_case"]["default"] == case["default"] == "full_seed_0"
    assert inputs["w4_case"]["options"] == ["full_seed_0", "full_seed_42", "one_token",
                                          "short_17", "short_255", "changed_future", "changed_padding"]
    w4 = data["jobs"]["w4-validation"]
    assert w4["if"] == "${{ github.event_name == 'workflow_dispatch' && inputs.run_w4_validation && !inputs.run_w5_validation }}"
    assert w4["uses"] == "./.github/workflows/w4-full-numeric.yml"
    assert w4["with"] == {"run_full": "true", "case": "${{ inputs.w4_case || 'full_seed_0' }}"}
    assert "continue-on-error" not in w4
    for event in ("pull_request", "schedule", "workflow_dispatch"):
        for enabled in (False, True):
            guard = w4["if"].removeprefix("${{").removesuffix("}}")
            event_guard, flag, w5_guard = guard.split("&&")
            allowed = _guard(event_guard, event) and enabled
            assert flag.strip() == "inputs.run_w4_validation"
            assert w5_guard.strip() == "!inputs.run_w5_validation"
            assert allowed is (event == "workflow_dispatch" and enabled)
    assert inputs["run_w3_validation"]["default"] == "false"
    w3 = data["jobs"]["w3-validation"]
    assert w3["if"] == "${{ github.event_name == 'workflow_dispatch' && inputs.run_w3_validation && !inputs.run_w5_validation }}"
    assert w3["uses"] == "./.github/workflows/w3-nightly.yml"
    assert data["jobs"]["w2-acceptance"]["needs"] == ["llm-deploy", "full-qwen3-frontend"]
    assert data["permissions"] == {"contents": "read"}
    for job in ("unit", "full-numeric"):
        checkout = next(item for item in workflow()["jobs"][job]["steps"]
                        if item.get("uses", "").startswith("actions/checkout@"))
        assert checkout["with"]["ref"] == "${{ github.sha }}"


@pytest.mark.parametrize("event", ["pull_request", "schedule", "workflow_dispatch"])
@pytest.mark.parametrize("requested", ["", "false", "true"])
@pytest.mark.parametrize("result", ["success", "failure", "cancelled", "skipped", ""])
def test_llm_w4_scope_rejects_skipped_or_failed_explicit_execution(tmp_path, event, requested, result):
    active = event == "workflow_dispatch" and requested == "true"
    passed = active and result == "success"
    not_run = not active and result == "skipped"
    summary = tmp_path / "summary.md"
    status = workflow(LLM_PATH)["jobs"]["w4-execution-summary"]
    assert status["if"] == "always()" and status["needs"] == ["w4-validation"]
    script = python_body("Require explicitly requested W4 execution", LLM_PATH)
    completed = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
        env=dict(os.environ, GITHUB_EVENT_NAME=event, W4_REQUESTED=requested,
                 W4_RESULT=result, GITHUB_STEP_SUMMARY=str(summary)), capture_output=True, text=True)
    assert (completed.returncode == 0) is (passed or not_run), completed.stderr
    text = summary.read_text(encoding="utf-8")
    label = "PASS" if passed else "NOT RUN" if not_run else "FAIL"
    assert f"**{label}**" in text
    assert "<details>" in text and "w4-full-reports-*" in text
    assert "Nightly activation" in text and "same commit" in text
    if not_run:
        assert "not requested" in text and "not full QEMU acceptance" in text


def test_single_case_default_and_deadlines_leave_room_to_upload_failures():
    data = workflow()
    assert data["on"]["workflow_call"]["inputs"]["case"]["default"] == "full_seed_0"
    assert "all" not in workflow(LLM_PATH)["on"]["workflow_dispatch"]["inputs"]["w4_case"]["options"]
    job = data["jobs"]["full-numeric"]
    command = step("Build external weights and compare one full RV64 forward")
    job_minutes = int(job["timeout-minutes"])
    step_minutes = int(command["timeout-minutes"])
    outer_minutes = int(re.search(r"--kill-after=30s (\d+)m", command["run"]).group(1))
    guest_seconds = int(re.search(r"--timeout (\d+)", command["run"]).group(1))
    assert guest_seconds == 10800
    assert guest_seconds / 60 < outer_minutes < step_minutes < job_minutes <= 360
    assert job_minutes - step_minutes >= 30
    assert "--signal=INT" in command["run"]
    for item in job["steps"]:
        assert "continue-on-error" not in item


@pytest.fixture(scope="module")
def bash():
    for candidate in (os.environ.get("SCRATCHV_BASH"), shutil.which("bash")):
        if candidate and Path(candidate).is_file():
            result = subprocess.run([candidate, "--noprofile", "--norc", "-c", "exit 0"],
                                    capture_output=True)
            if result.returncode == 0:
                return candidate
    pytest.skip("A working Bash is required for actual Linux shell snippet execution")


@pytest.mark.parametrize("exit_code", [0, 1, 124, 130])
def test_shell_pipeline_preserves_failure_instead_of_tee_success(bash, tmp_path, exit_code):
    (tmp_path / "output/w4-ci").mkdir(parents=True)
    script = step("Build external weights and compare one full RV64 forward")["run"]
    wrapper = 'timeout() { printf "%s\\n" "$@" > "$ARG_LOG"; return "$MOCK_EXIT"; }\n'
    capture = tmp_path / "arguments.txt"
    env = dict(os.environ, SELECTED_CASE="full_seed_0", W4_OUTPUT_DIR="output/unique-full",
               MOCK_EXIT=str(exit_code), ARG_LOG=capture.as_posix())
    result = subprocess.run([bash, "--noprofile", "--norc", "-c", wrapper + script],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == exit_code, result.stderr
    args = capture.read_text().splitlines()
    assert args[args.index("--case") + 1] == "full_seed_0"
    assert args[args.index("--output-dir") + 1] == "output/unique-full"
    assert args[args.index("--cc") + 1] == "zig"
    assert args[args.index("--qemu") + 1] == "qemu-system-riscv64"
    assert args[args.index("--matmul-policy") + 1] == "blocked_fma"
    assert "--build-only" not in args and "--reuse-build" not in args
    assert int((tmp_path / "output/w4-ci/runner-exit-code.txt").read_text()) == exit_code


def _valid_report(tmp_path):
    sources = {}
    for name in ("probes/w4_qwen3_full/run.py", "scratchv/runtime/riscv_external.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# tiny CI gate source identity fixture\n")
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "source-snapshot.zip").write_bytes(b"tiny snapshot checksum fixture")
    return {"passed": True, "status": "PASS", "full_qemu_forward_executed": True,
            "git": {"head": "a" * 40, "dirty": False}, "source_sha256": sources,
            "source_snapshot": {"file": "source-snapshot.zip", "sha256": hashlib.sha256(
                (tmp_path / "source-snapshot.zip").read_bytes()).hexdigest()},
            "matmul_policy": "blocked_fma",
            "coverage": "full_seed_0", "cases_passed": 1, "seven_case_coverage": False,
            "execution_tool_binary_sha256": {"qemu": "0" * 64},
            "fma_conformance": {"passed": True, "status": "PASS", "tool_binary_sha256": {"qemu": "0" * 64},
                                "comparison": {
                "passed": True, "criterion": "exact binary32 bits and FCSR", "failed_rows": 0,
                "rows": [{"passed": True, "bits_match": True, "csr_match": True} for _ in range(12)]}},
            "gates": {key: True for key in ("build:riscv-full", "smoke:qemu-full", "numeric:qemu-full-qwen3")},
            "cases": [{"name": "full_seed_0", "passed": True,
                       "qemu": {"passed": True, "status": "PASS", "guest_completed": True, "exit_code": 0},
                       "numeric": {"passed": True, "atol": 1e-3,
                       "rtol": 0.0, "elements_compared": 256 * 151936, "max_abs": 0.0001}}]}


@pytest.mark.parametrize("damage", [None, "failed", "status", "forward", "build", "smoke", "numeric",
                                    "case", "duplicate_case", "atol", "rtol", "elements", "threshold",
                                    "nan", "inf", "bool", "text", "numeric_failed", "policy", "negative",
                                    "fma_missing", "fma_failed", "fma_status", "fma_comparison",
                                    "fma_criterion", "fma_count", "fma_failed_rows", "fma_bool_rows",
                                    "fma_row_failed", "fma_bits", "fma_csr", "fma_qemu_mismatch",
                                    "fma_qemu_absent", "fma_qemu_nonhex", "fma_checked_hash_absent"])
def test_actual_ci_report_gate_rejects_false_or_incomplete_acceptance(tmp_path, damage):
    report = _valid_report(tmp_path)
    numeric = report["cases"][0]["numeric"]
    if damage == "failed":
        report["passed"] = False
    elif damage == "status":
        report["status"] = "BUILD_ONLY"
    elif damage == "forward":
        report["full_qemu_forward_executed"] = False
    elif damage == "policy":
        report["matmul_policy"] = "sequential"
    elif damage and damage.startswith("fma_"):
        fma = report["fma_conformance"]
        if damage == "fma_missing":
            del report["fma_conformance"]
        elif damage == "fma_failed":
            fma["passed"] = False
        elif damage == "fma_status":
            fma["status"] = "FAIL"
        elif damage == "fma_comparison":
            fma["comparison"]["passed"] = False
        elif damage == "fma_criterion":
            fma["comparison"]["criterion"] = "max_abs < 1e-3"
        elif damage == "fma_count":
            fma["comparison"]["rows"].pop()
        elif damage in ("fma_failed_rows", "fma_bool_rows"):
            fma["comparison"]["failed_rows"] = 1 if damage == "fma_failed_rows" else False
        elif damage == "fma_qemu_mismatch":
            fma["tool_binary_sha256"]["qemu"] = "1" * 64
        elif damage == "fma_qemu_absent":
            del report["execution_tool_binary_sha256"]
        elif damage == "fma_qemu_nonhex":
            report["execution_tool_binary_sha256"]["qemu"] = fma["tool_binary_sha256"]["qemu"] = "x" * 64
        elif damage == "fma_checked_hash_absent":
            del fma["tool_binary_sha256"]
        else:
            key = {"fma_row_failed": "passed", "fma_bits": "bits_match", "fma_csr": "csr_match"}[damage]
            fma["comparison"]["rows"][0][key] = False
    elif damage in ("build", "smoke", "numeric"):
        key = {"build": "build:riscv-full", "smoke": "smoke:qemu-full", "numeric": "numeric:qemu-full-qwen3"}[damage]
        report["gates"][key] = False
    elif damage == "case":
        report["cases"][0]["name"] = "short_17"
    elif damage == "duplicate_case":
        report["cases"].append(copy.deepcopy(report["cases"][0]))
    elif damage == "atol":
        numeric["atol"] = 0.01
    elif damage == "rtol":
        numeric["rtol"] = 0.1
    elif damage == "elements":
        numeric["elements_compared"] = 151936
    elif damage == "numeric_failed":
        numeric["passed"] = False
    elif damage:
        numeric["max_abs"] = {"threshold": 1e-3, "nan": float("nan"), "inf": float("inf"),
                              "bool": True, "text": "0.0001", "negative": -0.01}[damage]
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    env = dict(os.environ, W4_OUTPUT_DIR=str(tmp_path), SELECTED_CASE="full_seed_0", GITHUB_SHA="a" * 40)
    result = subprocess.run([sys.executable, "-c", python_body("Require the complete forward and exact numerical gate")],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert (result.returncode == 0) is (damage is None), result.stderr


@pytest.mark.parametrize("result", ["pass", "skipped", "failure", "error", "missing", "duplicate"])
@pytest.mark.parametrize("required", ["test_external_rv64_matmul_and_reuse", "test_weight_pointer_above_32bit_limit",
    "test_real_rv64_fma_exact_conformance",
    "test_blocked_fma_host_and_rv64_are_bitwise_equal[left2-right2-1]",
    "test_blocked_fma_host_and_rv64_are_bitwise_equal[left2-right2-2]",
    "test_blocked_fma_host_and_rv64_are_bitwise_equal[left2-right2-4]",
    "test_blocked_fma_host_and_rv64_are_bitwise_equal[left2-right2-auto]",
    "test_column_panel_rv64_errors_recover_in_same_guest_workspace",
    "test_real_qemu_contiguous_loader_segments_and_success_cleanup[chunk31-3]",
    "test_post_execution_transport_corruption_fails_and_retains_copies",
    "test_success_copy_cleanup_failure_is_not_a_pass",
    "test_actual_shared_library_executes_kernel_calls_and_unloads[sequential]",
    "test_actual_shared_library_executes_kernel_calls_and_unloads[blocked_fma]"] + W5_REQUIRED)
def test_actual_junit_gate_requires_real_qemu_execution(tmp_path, result, required):
    folder = tmp_path / "output/w4-unit"
    folder.mkdir(parents=True)
    child = "" if result == "pass" else f"<{result}/>"
    case = f'<testcase name="{required}">{child}</testcase>'
    if result == "missing":
        case = ""
    elif result == "duplicate":
        case *= 2
    for name in ("test_external_rv64_matmul_and_reuse", "test_weight_pointer_above_32bit_limit",
                 "test_real_rv64_fma_exact_conformance",
                 "test_blocked_fma_host_and_rv64_are_bitwise_equal[left2-right2-1]",
                 "test_blocked_fma_host_and_rv64_are_bitwise_equal[left2-right2-2]",
                 "test_blocked_fma_host_and_rv64_are_bitwise_equal[left2-right2-4]",
                 "test_blocked_fma_host_and_rv64_are_bitwise_equal[left2-right2-auto]",
                 "test_column_panel_rv64_errors_recover_in_same_guest_workspace",
                 "test_real_qemu_contiguous_loader_segments_and_success_cleanup[chunk31-3]",
                 "test_post_execution_transport_corruption_fails_and_retains_copies",
                 "test_success_copy_cleanup_failure_is_not_a_pass",
                 "test_actual_shared_library_executes_kernel_calls_and_unloads[sequential]",
                 "test_actual_shared_library_executes_kernel_calls_and_unloads[blocked_fma]") + tuple(W5_REQUIRED):
        if name != required:
            case += f'<testcase name="{name}"/>'
    (folder / "tests.xml").write_text(f"<testsuite>{case}</testsuite>", encoding="utf-8")
    code = python_body("Run ABI, streaming bundle, runner and actual QEMU tests")
    completed = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert (completed.returncode == 0) is (result == "pass"), completed.stderr


@pytest.mark.parametrize("case,ram_gib,disk_gib,passed", [
    ("full_seed_0", 8, 15, True), ("short_17", 12, 20, True),
    ("full_seed_0", 7, 15, False), ("full_seed_0", 8, 14, False),
    ("all", 16, 50, False), ("full_seed_0; exit 0", 16, 50, False),
])
def test_resource_and_case_preflight_records_failure_before_launch(tmp_path, monkeypatch,
                                                                 case, ram_gib, disk_gib, passed):
    (tmp_path / "output/w4-ci").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SELECTED_CASE", case)
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setenv("GITHUB_ENV", str(tmp_path / "environment"))
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=disk_gib * 1024**3))
    real_read = Path.read_text
    def fake_memory(path, *args, **kwargs):
        if path.as_posix() == "/proc/meminfo":
            return f"MemTotal: 16777216 kB\nMemAvailable: {ram_gib * 1024**2} kB\n"
        return real_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", fake_memory)
    code = python_body("Validate case and record available Linux resources")
    if passed:
        exec(compile(code, "workflow-preflight", "exec"), {})
    else:
        with pytest.raises(SystemExit):
            exec(compile(code, "workflow-preflight", "exec"), {})
    evidence = json.loads((tmp_path / "output/w4-ci/resource-preflight.json").read_text())
    assert evidence["passed"] is passed
    assert evidence["case"] == case
    assert evidence["memory"]["MemAvailable"] == ram_gib * 1024**3


def test_reports_and_raw_outputs_survive_normal_failures_without_huge_weight_uploads():
    uploads = [item for item in workflow()["jobs"]["full-numeric"]["steps"]
               if item.get("uses", "").startswith("actions/upload-artifact@")]
    assert len(uploads) == 2 and all(item["if"] == "always()" for item in uploads)
    raw = next(item["with"] for item in uploads if item["with"]["retention-days"] == "3")
    for required in ("report.json", "inputs.npz", "ort.npy", "qemu/inputs.bin", "qemu/output.bin", "qemu/run.json",
                     "source-snapshot.zip", "observations.npy", "input.npy", "weight-transport/transport.json"):
        assert required in raw["path"]
    assert "weights.bin" not in raw["path"]
    assert "weights-*.bin" not in raw["path"] and "**/*.bin" not in raw["path"]
    assert all(item["with"]["retention-days"] in ("3", "30") for item in uploads)


def test_report_upload_retains_uppercase_and_lowercase_assembly_sources():
    patterns = step("Upload reports and failure diagnostics")["with"]["path"].splitlines()
    assert "output/w4-ci/**/*.s" in patterns
    assert "output/w4-ci/**/*.S" in patterns  # Linux must include the generated start.S.


def test_retained_audit_follows_acceptance_and_uses_the_same_raw_directory():
    steps = workflow()["jobs"]["full-numeric"]["steps"]
    gate = step("Require the complete forward and exact numerical gate")
    audit = step("Recompute retained numeric evidence without rerunning QEMU")
    upload = step("Upload reports and failure diagnostics")
    assert steps.index(gate) < steps.index(audit) < steps.index(upload)
    assert "if" not in audit and "continue-on-error" not in audit
    assert "-m probes.w4_qwen3_full.audit" in audit["run"]
    assert '--report-dir "$W4_OUTPUT_DIR"' in audit["run"]
    assert '--output-dir "${W4_OUTPUT_DIR}-audit"' in audit["run"]
    assert "--qemu" not in audit["run"]
    assert "output/w4-ci/**/*.json" in upload["with"]["path"]
    unit = step("Run ABI, streaming bundle, runner and actual QEMU tests")["run"]
    assert "tests/test_riscv_process_cleanup.py" in unit
    assert "tests/test_riscv_weight_transport.py" in unit


@pytest.mark.parametrize("ram_gib,disk_gib,passed", [(8, 8, True), (7, 8, False), (8, 7, False), (16, 20, True)])
def test_post_asset_preflight_records_real_shortfall_before_build(tmp_path, monkeypatch, ram_gib, disk_gib, passed):
    label = "Recheck free memory and disk after dependencies and assets"
    steps = workflow()["jobs"]["full-numeric"]["steps"]
    assert steps.index(step("Acquire the fixed original ONNX assets")) < steps.index(step(label))
    assert steps.index(step(label)) < steps.index(step("Build external weights and compare one full RV64 forward"))
    (tmp_path / "output/w4-ci").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=disk_gib * 1024**3))
    real_read = Path.read_text
    def fake_memory(path, *args, **kwargs):
        if path.as_posix() == "/proc/meminfo":
            return f"MemTotal: 16777216 kB\nMemAvailable: {ram_gib * 1024**2} kB\n"
        return real_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", fake_memory)
    code = python_body(label)
    if passed:
        exec(compile(code, "workflow-post-assets-preflight", "exec"), {})
    else:
        with pytest.raises(SystemExit):
            exec(compile(code, "workflow-post-assets-preflight", "exec"), {})
    evidence = json.loads((tmp_path / "output/w4-ci/resource-preflight-after-assets.json").read_text())
    assert evidence["passed"] is passed
    assert evidence["stage"] == "after_dependencies_and_assets"
    assert evidence["disk_free_bytes"] == disk_gib * 1024**3
    assert evidence["minimum_disk_free_bytes"] == evidence["minimum_available_memory_bytes"] == 8 * 1024**3
    assert 4 * 1024**3 < evidence["estimated_remaining_payload_bytes"] < 5 * 1024**3


def test_all_actual_bash_bodies_parse(bash):
    for path in (PATH, NIGHTLY_PATH):
        for job in workflow(path)["jobs"].values():
            for item in job.get("steps", []):
                if "run" in item:
                    result = subprocess.run([bash, "--noprofile", "--norc", "-n"], input=item["run"],
                                            capture_output=True, text=True)
                    assert result.returncode == 0, f"{path.name}/{item.get('name')}: {result.stderr}"


@pytest.mark.parametrize("damage", ["head", "dirty", "missing_git", "source_hash", "source_missing",
    "source_escape", "source_inventory", "snapshot_missing", "snapshot_hash", "snapshot_name",
    "coverage", "case_count", "case_bool_count", "seven_cases", "case_failed", "qemu_failed",
    "qemu_status", "qemu_missing", "qemu_not_completed", "qemu_exit", "qemu_bool_exit", "qemu_cleanup", "qemu_error"])
def test_ci_source_and_execution_provenance_cannot_be_swapped(tmp_path, damage):
    report = _valid_report(tmp_path)
    qemu = report["cases"][0]["qemu"]
    if damage == "head":
        report["git"]["head"] = "b" * 40
    elif damage == "dirty":
        report["git"]["dirty"] = True
    elif damage == "missing_git":
        del report["git"]
    elif damage == "source_hash":
        (tmp_path / "probes/w4_qwen3_full/run.py").write_bytes(b"altered")
    elif damage == "source_missing":
        (tmp_path / "probes/w4_qwen3_full/run.py").unlink()
    elif damage == "source_escape":
        report["source_sha256"]["../outside.py"] = "0" * 64
    elif damage == "source_inventory":
        report["source_sha256"] = {}
    elif damage == "snapshot_missing":
        (tmp_path / "source-snapshot.zip").unlink()
    elif damage == "snapshot_hash":
        report["source_snapshot"]["sha256"] = "0" * 64
    elif damage == "snapshot_name":
        report["source_snapshot"]["file"] = "../snapshot.zip"
    elif damage == "coverage":
        report["coverage"] = "all"
    elif damage in ("case_count", "case_bool_count"):
        report["cases_passed"] = 7 if damage == "case_count" else True
    elif damage == "seven_cases":
        report["seven_case_coverage"] = True
    elif damage == "case_failed":
        report["cases"][0]["passed"] = False
    elif damage == "qemu_missing":
        del report["cases"][0]["qemu"]
    else:
        key, value = {"qemu_failed": ("passed", False), "qemu_status": ("status", "FAIL"),
            "qemu_not_completed": ("guest_completed", False), "qemu_exit": ("exit_code", 1),
            "qemu_bool_exit": ("exit_code", False), "qemu_cleanup": ("cleanup_errors", ["still running"]),
            "qemu_error": ("error", "dump failed")}[damage]
        qemu[key] = value
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    result = subprocess.run([sys.executable, "-c", python_body("Require the complete forward and exact numerical gate")],
                            cwd=tmp_path, env=dict(os.environ, W4_OUTPUT_DIR=str(tmp_path),
                            SELECTED_CASE="full_seed_0", GITHUB_SHA="a" * 40), capture_output=True, text=True)
    assert result.returncode != 0


def test_w1_w3_handoff_is_bounded_and_uses_the_same_commit():
    data = workflow()
    unit = step("Run ABI, streaming bundle, runner and actual QEMU tests")["run"]
    for name in ("test_w1_qwen3_export.py", "test_llm_inputs.py", "test_w2_runtime_metadata.py", "test_w3_full_gate.py",
                 "test_tensor_boundary_contracts.py", "test_tensor_return_contract.py"):
        assert f"tests/{name}" in unit
        assert (ROOT / "tests" / name).is_file()
    assert "acquire_model" not in unit and "--case all" not in unit
    assert "probes.w3_qwen3_full.run" not in unit
    assert "requirements/qwen3-small-probe.txt" in step("Install W3-pinned CPU environment and Zig 0.14.1")["run"]
    assert "probes.w1_qwen3_export.run import acquire_model" in step("Acquire the fixed original ONNX assets")["run"]
    # The top-level CI handles every PR to main, including shared source and
    # workflow changes. This reusable workflow has no independent PR trigger.
    trigger = workflow(CI_PATH)["on"]["pull_request"]
    assert trigger["branches"] == ["main"]
    assert "paths" not in trigger and "paths-ignore" not in trigger
    paths = json.loads((ROOT / "scripts/ci_scopes.json").read_text())["w4"]
    assert "scratchv/**" in paths and "probes/w3_qwen3_full/**" in paths
    assert ".github/workflows/w4-nightly.yml" in paths
    assert "scripts/summarize_w4_ci.py" in paths and "tests/test_w4*.py" in paths
    for name in ("unit", "full-numeric"):
        checkout = next(s for s in data["jobs"][name]["steps"] if s.get("uses", "").startswith("actions/checkout@"))
        assert checkout["with"]["ref"] == "${{ github.sha }}"
        assert checkout["with"]["persist-credentials"] == "false"


@pytest.mark.parametrize("event", ["schedule", "workflow_dispatch", "workflow_call", "pull_request"])
@pytest.mark.parametrize("enabled,expected", [("", False), ("false", False), ("1", False),
                                               ("true", True), ("TRUE", True)])
def test_nightly_all_entries_require_explicit_repository_opt_in(event, enabled, expected):
    data = workflow(NIGHTLY_PATH)
    assert set(data["on"]) == {"schedule", "workflow_dispatch", "workflow_call"}
    full = data["jobs"]["full-numeric"]
    assert full["if"] == "${{ vars.W4_NIGHTLY_ENABLED == 'true' && github.event_name != 'pull_request' }}"
    guard, event_guard = full["if"].removeprefix("${{").removesuffix("}}").split("&&")
    allowed = _guard(guard.replace("vars.W4_NIGHTLY_ENABLED", "github.event_name"), enabled)
    allowed = allowed and _guard(event_guard, event)
    assert allowed is (expected and event != "pull_request")
    assert full["uses"] == "./.github/workflows/w4-full-numeric.yml"
    assert full["with"] == {"run_full": "true", "case": "${{ inputs.case || 'full_seed_0' }}"}
    assert event in data["on"] or event == "pull_request"  # A reusable caller retains its event.
    for trigger in ("workflow_call", "workflow_dispatch"):
        assert data["on"][trigger]["inputs"]["case"]["default"] == "full_seed_0"
    assert "all" not in data["on"]["workflow_dispatch"]["inputs"]["case"]["options"]
    assert data["permissions"] == {"contents": "read"}
    assert data["concurrency"]["cancel-in-progress"] == "false"
    assert data["concurrency"]["group"] != workflow()["concurrency"]["group"]
    status = data["jobs"]["nightly-status"]
    assert status["if"] == "${{ always() }}" and status["needs"] == "full-numeric"
    assert all("continue-on-error" not in job for job in data["jobs"].values())


@pytest.mark.parametrize("event", ["pull_request", "push", "workflow_dispatch", "schedule"])
@pytest.mark.parametrize("requested", ["false", "true"])
@pytest.mark.parametrize("unit", ["success", "failure", "cancelled", "skipped"])
@pytest.mark.parametrize("full", ["success", "failure", "cancelled", "skipped"])
def test_ci_summary_actual_bash_failure_matrix(bash, tmp_path, event, requested, unit, full):
    enabled = requested == "true"
    expected = unit == "success" and full == ("success" if enabled else "skipped")
    env = dict(os.environ, GITHUB_EVENT_NAME=event, UNIT_RESULT=unit, FULL_NUMERIC_RESULT=full,
               FULL_REQUESTED=requested, W5_REQUESTED="false", W5_GENERATION_RESULT="skipped",
               SELECTED_CASE="full_seed_0", GITHUB_SHA="a" * 40, GITHUB_RUN_ID="123", GITHUB_RUN_ATTEMPT="1",
               GITHUB_STEP_SUMMARY=(tmp_path / "summary.md").as_posix())
    script = step("Record W4 CI result without promoting skipped gates")["run"].replace("python3 -", '"' + sys.executable.replace("\\", "/") + '" -')
    result = subprocess.run([bash, "--noprofile", "--norc", "-c", script], cwd=tmp_path,
                            env=env, capture_output=True, text=True)
    assert (result.returncode == 0) is expected, result.stderr
    row = json.loads((tmp_path / "output/w4-status/status.json").read_text())
    assert row["passed"] is expected
    assert row["full_numeric_requested"] is enabled
    assert row["full_numeric_accepted"] is (expected and enabled)
    assert row["case"] == ("full_seed_0" if enabled else None)
    assert row["team_accepted"] is False and row["independent_reproduction"] is False
    assert row["commit"] == "a" * 40 and row["seven_case_coverage"] is False


@pytest.mark.parametrize("enabled", ["", "false", "true", "TRUE"])
@pytest.mark.parametrize("full", ["success", "failure", "cancelled", "skipped", ""])
def test_nightly_summary_preserves_disabled_and_rejects_every_missing_gate(bash, tmp_path, enabled, full):
    opted_in = enabled.casefold() == "true"
    passed = opted_in and full == "success"
    disabled = not opted_in and full == "skipped"
    env = dict(os.environ, NIGHTLY_ENABLED=enabled, FULL_NUMERIC_RESULT=full, SELECTED_CASE="full_seed_0",
               GITHUB_SHA="a" * 40, GITHUB_RUN_ID="123", GITHUB_RUN_ATTEMPT="1",
               GITHUB_STEP_SUMMARY=(tmp_path / "summary.md").as_posix())
    script = step("Require enabled nightly execution and preserve its scope", NIGHTLY_PATH)["run"].replace("python3 -", '"' + sys.executable.replace("\\", "/") + '" -')
    result = subprocess.run([bash, "--noprofile", "--norc", "-c", script], cwd=tmp_path,
                            env=env, capture_output=True, text=True)
    assert (result.returncode == 0) is (passed or disabled), result.stderr
    row = json.loads((tmp_path / "output/w4-nightly/status.json").read_text())
    assert row["passed"] is passed
    assert row["status"] == ("PASS" if passed else "DISABLED" if disabled else "FAIL")
    assert row["seven_case_coverage"] is False and row["team_accepted"] is False
    assert "DISABLED is not a numeric PASS" in (tmp_path / "summary.md").read_text()


def test_failure_summaries_keep_json_artifacts_and_all_prerequisites():
    for path, jobname, needs in ((PATH, "ci-status", ["unit", "full-numeric", "w5-generation"]),
                               (NIGHTLY_PATH, "nightly-status", "full-numeric")):
        job = workflow(path)["jobs"][jobname]
        assert job["if"] == "${{ always() }}" and job["needs"] == needs
        uploads = [s for s in job["steps"] if s.get("uses", "").startswith("actions/upload-artifact@")]
        assert len(uploads) == 1 and uploads[0]["if"] == "always()"
        assert uploads[0]["with"]["if-no-files-found"] == "error"
        assert uploads[0]["with"]["path"].endswith("/status.json")
        assert all("continue-on-error" not in s for s in job["steps"])


def _full_enabled(expression, event, mode, legacy, run_w5=False):
    # Evaluate only the fixed, repository-owned workflow boolean expressions.
    expression = expression.removeprefix("${{").removesuffix("}}").strip()
    for key, value in (("github.event_name", event), ("inputs.mode", mode), ("inputs.run_full", legacy), ("inputs.run_w5", run_w5)):
        expression = expression.replace(key, repr(value))
    return bool(eval(re.sub(r"!(?!=)", " not ", expression.replace("&&", " and ").replace("||", " or ")),
                     {"__builtins__": {}}))


@pytest.mark.parametrize("event,mode,legacy,w4,w5", [
    ("pull_request", "small", False, False, False),
    ("pull_request", "w5-generation", False, False, False),
    ("schedule", "w5-generation", False, False, False),
    ("workflow_dispatch", "small", False, False, False),
    ("workflow_dispatch", "w4-forward", False, True, False),
    ("workflow_dispatch", "w5-generation", False, False, True),
    ("workflow_call", "", True, True, False),
    ("workflow_dispatch", "", True, True, False),  # Existing LLM Deploy caller.
    ("schedule", "", True, True, False),  # Existing opt-in W4 nightly caller.
    ("workflow_dispatch", "w5-generation", True, False, False),
])
def test_w5_manual_selection_is_exclusive_and_old_callers_remain(event, mode, legacy, w4, w5):
    data = workflow()
    choice = data["on"]["workflow_dispatch"]["inputs"]["mode"]
    assert choice == {"description": "Small regressions by default; full modes are mutually exclusive",
                      "type": "choice", "default": "small",
                      "options": ["small", "w4-forward", "w5-generation"]}
    assert _full_enabled(data["jobs"]["full-numeric"]["if"], event, mode, legacy) is w4
    assert _full_enabled(data["jobs"]["w5-generation"]["if"], event, mode, legacy) is w5
    assert not (w4 and w5)
    assert set(data["on"]["workflow_call"]["inputs"]) == {"run_full", "run_w5", "case"}
    assert "schedule" not in data["on"] and "pull_request" not in data["on"]


def test_w5_manual_job_reuses_deployment_and_preserves_bounded_full_scope():
    job = workflow()["jobs"]["w5-generation"]
    assert job["needs"] == "unit"
    run = step("Run the frozen W5 suite through the deployment script")
    assert "bash scripts/deploy_w5.sh" in run["run"] and "--reference-suite" in run["run"]
    assert "--model /assets/onnx/model.onnx" in run["run"]
    assert "--verify-onnx /assets/onnx/model.onnx" in run["run"]
    for flag in ("--harts 4", "--matmul-policy blocked_fma", "--matmul-tile 2x4",
                 "--matmul-rhs-layout column-panel", "--mean-policy two-four-lane-last-axis",
                 "--nonlinear-policy fp32-reference"):
        assert flag in run["run"]
    assert "probes.w4_qwen3_full.run" not in run["run"] and "--max-new-tokens" not in run["run"]
    guest = int(re.search(r"--total-timeout (\d+)", run["run"]).group(1))
    outer = int(re.search(r"--kill-after=30s (\d+)m", run["run"]).group(1))
    assert 10540 < guest < outer * 60 < int(run["timeout-minutes"]) * 60
    assert int(run["timeout-minutes"]) + 40 <= int(job["timeout-minutes"]) <= 360
    bounded_steps = sum(int(item.get("timeout-minutes", 0)) for item in job["steps"])
    assert bounded_steps + 10 <= int(job["timeout-minutes"])  # setup, diagnostics and artifact upload
    assert job["steps"].index(step("Build the clean W5 deployment image")) < job["steps"].index(run)
    assert job["steps"].index(step("Acquire only the fixed W5 ONNX and tokenizer assets")) < job["steps"].index(run)
    upload = step("Upload W5 reports and failure diagnostics only")
    assert upload["if"] == "always()" and upload["with"]["retention-days"] == "30"
    assert not any(suffix in upload["with"]["path"] for suffix in (".bin", ".npy", ".onnx", ".elf", ".zip"))
    gate = step("Require complete W5 generation evidence")
    assert gate["if"] == "always()" and "scripts/check_w5_ci.py" in gate["run"]
    assert "--exit-code-file output/w5-ci/runner-exit-code.txt" in gate["run"]
    assert "--log-exit-code-file output/w5-ci/log-exit-code.txt" in gate["run"]
    assert all("continue-on-error" not in item for item in job["steps"])


@pytest.mark.parametrize("exit_code", [0, 1, 124, 130, 143])
@pytest.mark.parametrize("log_failure", [False, True], ids=["log-ok", "log-full"])
def test_w5_deployment_pipeline_preserves_actual_exit(bash, tmp_path, exit_code, log_failure):
    if log_failure and not Path("/dev/full").exists():
        pytest.skip("Real tee write-failure check requires /dev/full")
    output = tmp_path / "output/w5-ci"
    output.mkdir(parents=True)
    if log_failure:
        (output / "runner.log").symlink_to("/dev/full")
    script = step("Run the frozen W5 suite through the deployment script")["run"]
    # Replace only the expensive producer; execute the actual workflow and real tee.
    wrapper = ('timeout() { printf "%s\\n" "$@" > "$ARG_LOG"; '
               'printf "generation stdout\\n"; printf "generation stderr\\n" >&2; '
               'return "$MOCK_EXIT"; }\n')
    capture = tmp_path / "arguments.txt"
    result = subprocess.run([bash, "--noprofile", "--norc", "-c", wrapper + script], cwd=tmp_path,
        env=dict(os.environ, ARG_LOG=str(capture), MOCK_EXIT=str(exit_code)),
        capture_output=True, text=True, timeout=10)
    assert (result.returncode != 0) is (log_failure or exit_code != 0), result.stderr
    log_exit = int((output / "log-exit-code.txt").read_text())
    assert (log_exit != 0) is log_failure
    assert result.returncode == (exit_code or log_exit), result.stderr
    assert int((output / "runner-exit-code.txt").read_text()) == exit_code
    assert "generation stdout" in result.stdout and "generation stderr" in result.stdout
    if log_failure:
        assert "runner.log" in result.stderr
    else:
        assert (output / "runner.log").read_text() == "generation stdout\ngeneration stderr\n"
    args = capture.read_text().splitlines()
    assert "scripts/deploy_w5.sh" in args and "--reference-suite" in args
    assert args[args.index("--total-timeout") + 1] == "16500"


@pytest.mark.parametrize("status_file", ["runner-exit-code.txt", "log-exit-code.txt"])
def test_w5_deployment_pipeline_rejects_status_write_failure(bash, tmp_path, status_file):
    if not Path("/dev/full").exists():
        pytest.skip("Real status write-failure check requires /dev/full")
    output = tmp_path / "output/w5-ci"
    output.mkdir(parents=True)
    (output / status_file).symlink_to("/dev/full")
    script = step("Run the frozen W5 suite through the deployment script")["run"]
    wrapper = 'timeout() { printf "generation completed\\n"; return 0; }\n'
    result = subprocess.run([bash, "--noprofile", "--norc", "-c", wrapper + script],
        cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0, result.stderr
    assert (output / "runner.log").read_text() == "generation completed\n"
    assert result.stderr  # The status write, not generation or tee, must fail.
    if status_file == "log-exit-code.txt":
        assert (output / "runner-exit-code.txt").read_text() == "0\n"


@pytest.mark.parametrize("cpus,ram,disk,passed", [
    (4, 10, 10, True), (8, 16, 30, True),
    (3, 16, 30, False), (4, 9, 30, False), (4, 16, 9, False)])
def test_w5_resource_shortfall_fails_before_generation(tmp_path, monkeypatch, cpus, ram, disk, passed):
    (tmp_path / "output/w5-ci").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(cpus)))
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=disk * 1024**3))
    real_read = Path.read_text
    def memory(path, *args, **kwargs):
        if path.as_posix() == "/proc/meminfo":
            return f"MemTotal: 16777216 kB\\nMemAvailable: {ram * 1024**2} kB\\n".replace("\\n", "\n")
        return real_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", memory)
    code = python_body("Require four CPUs and capacity for 24 retained logits")
    if passed:
        exec(compile(code, "w5-resource-preflight", "exec"), {})
    else:
        with pytest.raises(SystemExit, match="generation NOT RUN"):
            exec(compile(code, "w5-resource-preflight", "exec"), {})
    row = json.loads((tmp_path / "output/w5-ci/resource-preflight.json").read_text())
    assert row["passed"] is passed and row["available_cpus"] == cpus


@pytest.mark.parametrize("unit,full,w5,requested,expected", [
    ("success", "skipped", "success", "true", True),
    ("success", "skipped", "skipped", "false", True),
    ("success", "skipped", "skipped", "true", False),
    ("success", "skipped", "failure", "true", False),
    ("success", "skipped", "cancelled", "true", False),
    ("success", "skipped", "", "true", False),
    ("failure", "skipped", "success", "true", False),
    ("cancelled", "skipped", "success", "true", False),
    ("skipped", "skipped", "success", "true", False),
    ("success", "success", "success", "true", False),
    ("success", "skipped", "success", "false", False),
])
def test_w5_summary_cannot_promote_skip_or_duplicate_full_execution(
        tmp_path, unit, full, w5, requested, expected):
    (tmp_path / "output/w4-status").mkdir(parents=True)
    env = dict(os.environ, UNIT_RESULT=unit, FULL_NUMERIC_RESULT=full, W5_GENERATION_RESULT=w5,
               FULL_REQUESTED="false", W5_REQUESTED=requested, SELECTED_CASE="full_seed_0",
               GITHUB_SHA="a" * 40, GITHUB_RUN_ID="123", GITHUB_RUN_ATTEMPT="1",
               GITHUB_STEP_SUMMARY=str(tmp_path / "summary.md"))
    result = subprocess.run([sys.executable, "-c", python_body("Record W4 CI result without promoting skipped gates")],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert (result.returncode == 0) is expected, result.stderr
    row = json.loads((tmp_path / "output/w4-status/status.json").read_text())
    assert row["w5_generation_accepted"] is (expected and requested == "true")
    assert row["full_numeric_accepted"] is False


@pytest.mark.parametrize("event,w5,w3,w4,passed", [
    ("workflow_dispatch", True, False, False, True),
    ("workflow_dispatch", True, True, False, False),
    ("workflow_dispatch", True, False, True, False),
    ("workflow_dispatch", True, True, True, False),
    ("workflow_dispatch", False, True, True, True),
    ("pull_request", False, False, False, True),
    ("schedule", False, False, False, True),
    ("schedule", True, False, False, False),
])
def test_registered_entry_rejects_conflicts_before_any_expensive_job(tmp_path, event, w5, w3, w4, passed):
    data = workflow(LLM_PATH)
    choice = data["on"]["workflow_dispatch"]["inputs"]["run_w5_validation"]
    assert choice["type"] == "boolean" and choice["default"] == "false"
    caller = data["jobs"]["w5-validation"]
    assert caller["uses"] == "./.github/workflows/w4-full-numeric.yml"
    assert caller["with"] == {"run_w5": "true"}
    for name in ("w5-validation", "w3-validation", "w4-validation", "llm-deploy",
                 "full-qwen3-onnx", "full-qwen3-frontend"):
        assert data["jobs"][name]["needs"] == "validation-selection"
    for name in ("llm-deploy", "full-qwen3-onnx", "full-qwen3-frontend", "w2-acceptance"):
        assert "!inputs.run_w5_validation" in data["jobs"][name]["if"]
    result = subprocess.run([sys.executable, "-c", python_body("Reject conflicting W5 and legacy full selections", LLM_PATH)],
        cwd=tmp_path, env=dict(os.environ, GITHUB_EVENT_NAME=event, W5_REQUESTED=str(w5).lower(),
                             W3_REQUESTED=str(w3).lower(), W4_REQUESTED=str(w4).lower()),
        capture_output=True, text=True)
    assert (result.returncode == 0) is passed, result.stderr


@pytest.mark.parametrize("event,mode,legacy,w5,allowed", [
    ("workflow_dispatch", "", False, True, True),
    ("workflow_dispatch", "small", False, False, True),
    ("workflow_dispatch", "w5-generation", False, False, True),
    ("workflow_dispatch", "", True, False, True),
    ("schedule", "", True, False, True),
    ("workflow_dispatch", "", True, True, False),
    ("workflow_dispatch", "w4-forward", False, True, False),
    ("workflow_dispatch", "w5-generation", True, False, False),
    ("pull_request", "", False, True, False),
    ("schedule", "", False, True, False),
])
def test_reusable_w5_input_is_manual_and_mutually_exclusive(tmp_path, event, mode, legacy, w5, allowed):
    data = workflow()
    option = data["on"]["workflow_call"]["inputs"]["run_w5"]
    assert option["type"] == "boolean" and option["default"] == "false"
    unit = data["jobs"]["unit"]
    first = unit["steps"][0]
    assert first["name"] == "Reject conflicting or nonmanual full-generation selections"
    result = subprocess.run([sys.executable, "-c", python_body(first["name"])], cwd=tmp_path,
        env=dict(os.environ, GITHUB_EVENT_NAME=event, RUN_MODE=mode,
                 W5_REQUESTED=str(w5).lower(), W4_REQUESTED=str(legacy).lower()),
        capture_output=True, text=True)
    assert (result.returncode == 0) is allowed, result.stderr
    w4_enabled = _full_enabled(data["jobs"]["full-numeric"]["if"], event, mode, legacy, w5)
    w5_enabled = _full_enabled(data["jobs"]["w5-generation"]["if"], event, mode, legacy, w5)
    assert not (w4_enabled and w5_enabled)
    if w5 and allowed:
        assert w5_enabled and not w4_enabled
    if not allowed:
        assert not w5_enabled and not w4_enabled


@pytest.mark.parametrize("selection,result,legacy,requested,expected", [
    ("success", "success", "skipped", "true", "PASS"),
    ("success", "skipped", "skipped", "true", "FAIL"),
    ("success", "cancelled", "skipped", "true", "FAIL"),
    ("success", "failure", "skipped", "true", "FAIL"),
    ("failure", "skipped", "skipped", "true", "FAIL"),
    ("cancelled", "skipped", "skipped", "true", "FAIL"),
    ("success", "success", "success", "true", "FAIL"),
    ("success", "success", "failure", "true", "FAIL"),
    ("success", "success", "cancelled", "true", "FAIL"),
    ("success", "skipped", "success", "false", "NOT RUN"),
    ("success", "success", "skipped", "false", "FAIL"),
])
def test_w5_entry_summary_requires_success_and_no_legacy_execution(
        tmp_path, selection, result, legacy, requested, expected):
    job = workflow(LLM_PATH)["jobs"]["w5-execution-summary"]
    assert job["if"] == "always()"
    results = {name: {"result": legacy} for name in job["needs"]}
    results["validation-selection"] = {"result": selection}
    results["w5-validation"] = {"result": result}
    summary = tmp_path / "summary.md"
    completed = subprocess.run([sys.executable, "-c", python_body("Require the requested W5-only execution", LLM_PATH)],
        cwd=tmp_path, env=dict(os.environ, GITHUB_EVENT_NAME="workflow_dispatch", W5_REQUESTED=requested,
                             W5_JOB_RESULTS=json.dumps(results), GITHUB_STEP_SUMMARY=str(summary)),
        capture_output=True, text=True)
    assert (completed.returncode == 0) is (expected != "FAIL"), completed.stderr
    assert f"**{expected}**" in summary.read_text()


@pytest.mark.parametrize("suffix", ["None", "Give brief answers."])
@pytest.mark.parametrize("outcome", ["pass", "skipped", "failure", "error", "missing", "duplicate"])
def test_w5_only_template_gate_uses_existing_image_and_rejects_missing_execution(tmp_path, suffix, outcome):
    import xml.etree.ElementTree as ET
    item = step("Require the official three-round template in the clean W5 image")
    command = item["run"]
    assert "docker run --rm --init" in command and '"$SCRATCHV_W5_IMAGE"' in command
    assert "SCRATCHV_QWEN3_TOKENIZER_DIR=/assets/tokenizer" in command
    assert "dst=/tmp/test_conversation.py,readonly" in command
    assert "--reference-suite" not in command and "scripts/run_inference.py" not in command
    assert "pip install" not in command and "docker build" not in command
    job = workflow()["jobs"]["w5-generation"]
    assert job["steps"].index(step("Acquire only the fixed W5 ONNX and tokenizer assets")) < job["steps"].index(item)
    assert job["steps"].index(item) < job["steps"].index(step("Run the frozen W5 suite through the deployment script"))
    suite = ET.Element("testsuite")
    for current in ("None", "Give brief answers."):
        if current == suffix and outcome == "missing":
            continue
        name = f"test_official_three_round_rendering_matches_transformers[{current}]"
        row = ET.SubElement(suite, "testcase", name=name)
        if current == suffix:
            if outcome in ("skipped", "failure", "error"):
                ET.SubElement(row, outcome)
            elif outcome == "duplicate":
                ET.SubElement(suite, "testcase", name=name)
    target = tmp_path / "output/w5-ci/multiturn-tests.xml"
    target.parent.mkdir(parents=True)
    ET.ElementTree(suite).write(target)
    result = subprocess.run([sys.executable, "-c", python_body(item["name"])], cwd=tmp_path,
                            capture_output=True, text=True)
    assert (result.returncode == 0) is (outcome == "pass"), result.stderr
