"""W5 CLI orchestration boundaries without a model, compiler or QEMU."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scratchv.runtime.conversation import Conversation
from scratchv.runtime.generation import GenerationCancelledError


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "scripts/run_inference.py"
    spec = spec_from_file_location("w5_inference_cli_tests", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tokenizer:
    model_vocab_size = 8
    tokenizer_vocab_size = 7
    pad_token_id = 0
    generation_eos_token_ids = (6,)

    def __init__(self):
        self.prompts = []

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        self.prompts.append(text)
        return [1, 2]

    def validate_token_ids(self, tokens):
        if any(type(token) is not int or not 0 <= token < 7 for token in tokens):
            raise ValueError("unknown token")
        return list(tokens)

    def decode(self, tokens, *, skip_special_tokens, clean_up_tokenization_spaces):
        return "x" * sum(token == 3 for token in tokens)


def logits(token=3):
    result = np.zeros((1, 256, 8), dtype=np.float32)
    result[0, :, token] = 1
    return result


@pytest.fixture
def runtime(cli, monkeypatch):
    tokenizer = Tokenizer()
    executable = SimpleNamespace(
        evidence={"matmul_policy": "sequential", "qemu_workspace_layout": "none"},
        toolchain=SimpleNamespace(cc="unused-cc", qemu="unused-qemu"), harts=1,
    )
    state = SimpleNamespace(tokenizer=tokenizer, exe=executable, prepared=[], adapters=[],
                            saved=[], fma=[], failure=None, oracle=None, harts=[])

    def prepare(args, out, harts):
        state.prepared.append((args, out, harts))
        return executable

    class Adapter:
        def __init__(self, exe, output, *, timeout):
            self.exe, self.output, self.timeout = exe, output, timeout
            self.vocab_size = tokenizer.model_vocab_size
            self.runs, self.inputs = [], []
            state.adapters.append(self)

        def __call__(self, prepared):
            self.inputs.append(prepared)
            if state.failure is not None:
                self.runs.append({"guest_completed": False})
                raise state.failure
            self.runs.append({"guest_completed": True})
            return logits()

    def fma(*args, **kwargs):
        state.fma.append((args, kwargs))
        return {"passed": True, "status": "PASS"}

    monkeypatch.setattr(cli.Qwen3Tokenizer, "from_directory", lambda path: tokenizer)
    monkeypatch.setattr(cli, "prepare_executable", prepare)
    monkeypatch.setattr(cli, "save_executable", lambda exe, out: state.saved.append((exe, out)) or "fixture-metadata-identity")
    monkeypatch.setattr(cli, "QemuForward", Adapter)
    monkeypatch.setattr(cli, "run_fma_probe", fma)
    monkeypatch.setattr(cli, "oracle_session", lambda path: SimpleNamespace(run=lambda names, feed: [state.oracle]))
    monkeypatch.setattr(cli, "resolve_harts", lambda requested: state.harts.append(requested) or (4 if requested == "auto" else requested))
    return state


def argv(tmp_path, *extra):
    return ["--model", "small.onnx", "--tokenizer-dir", "tokenizer", "--output-dir", str(tmp_path / "run"), *extra]


def report(tmp_path):
    return json.loads((tmp_path / "run/report.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("requested,resolved", [(None, 1), ("2", 2), ("4", 4), ("auto", 4)])
def test_raw_multitoken_one_build_one_adapter_and_harts_resolved_once(cli, runtime, tmp_path, requested, resolved):
    options = [] if requested is None else ["--harts", requested]
    assert cli.main(argv(tmp_path, "--prompt", " raw text ", "--max-new-tokens", "3", *options)) == 0
    value = report(tmp_path)
    assert value["status"] == "PASS" and value["passed"]
    assert value["full_model_gate"] == "NOT_RUN"
    assert value["qemu_workspace_layout"] == "none" and value["mean_policy"] == "sequential"
    assert value["nonlinear_policy"] == "libm"
    assert value["executable_sha256"] == "fixture-metadata-identity"
    assert value["harts"] == resolved
    assert runtime.harts == ["auto" if requested == "auto" else int(requested or 1)]
    assert len(runtime.prepared) == len(runtime.saved) == len(runtime.adapters) == 1
    assert len(runtime.fma) == 1
    assert runtime.adapters[0].exe is runtime.exe
    assert [item.valid_length for item in runtime.adapters[0].inputs] == [2, 3, 4]
    assert runtime.tokenizer.prompts == [" raw text "]
    assert value["cases"][0]["generated_token_ids"] == [3, 3, 3]
    assert value["qemu_forwards"] == 3


def test_three_chat_turns_reuse_adapter_and_commit_ordered_history(cli, runtime, tmp_path, monkeypatch):
    template = "{% for m in messages %}{{ m.role }}:{{ m.content }}|{% endfor %}assistant:"
    chat = Conversation(runtime.tokenizer, template, system_prompt="brief")
    monkeypatch.setattr(Conversation, "from_directory", lambda *a, **kw: chat)
    assert cli.main(argv(tmp_path, "--mode", "chat", "--system-prompt", "brief",
                         "--chat-turn", "first", "--chat-turn", "second", "--chat-turn", "third",
                         "--max-new-tokens", "2")) == 0
    value = report(tmp_path)
    assert [x["content"] for x in value["history"]] == ["brief", "first", "xx", "second", "xx", "third", "xx"]
    assert runtime.tokenizer.prompts == [
        "system:brief|user:first|assistant:",
        "system:brief|user:first|assistant:xx|user:second|assistant:",
        "system:brief|user:first|assistant:xx|user:second|assistant:xx|user:third|assistant:",
    ]
    assert len(runtime.adapters) == len(runtime.prepared) == 1
    assert value["qemu_forwards"] == 6 and len(value["cases"]) == 3


def test_build_only_never_executes_fma_or_forward_and_not_pass(cli, runtime, tmp_path):
    assert cli.main(argv(tmp_path, "--build-only")) == 0
    value = report(tmp_path)
    assert value["status"] == "BUILD_ONLY" and value["passed"] is False
    assert value["full_model_gate"] == "NOT_RUN" and value["qemu_forwards"] == 0
    assert runtime.fma == [] and runtime.adapters[0].inputs == []


@pytest.mark.parametrize("error,expected,status", [
    (GenerationCancelledError("cancelled"), 130, "CANCELLED"),
    (KeyboardInterrupt(), 130, "CANCELLED"),
    (TimeoutError("deadline expired"), 1, "FAIL"),
    (RuntimeError("backend failed"), 1, "FAIL"),
])
def test_cancel_failure_report_retains_attempt_without_claiming_success(cli, runtime, tmp_path, error, expected, status):
    runtime.failure = error
    assert cli.main(argv(tmp_path, "--prompt", "text", "--max-new-tokens", "2")) == expected
    value = report(tmp_path)
    assert value["status"] == status and value["passed"] is False
    assert value["cases"] == [] and value["full_model_gate"] == "NOT_RUN"
    assert value["forward_attempts"] == 1 and value["qemu_forwards"] == 0
    assert type(error).__name__ in value["error"]


def test_numeric_oracle_failure_is_reported_before_committing_tokens(cli, runtime, tmp_path):
    runtime.oracle = logits()
    runtime.oracle[0, -1, 0] = 0.1
    assert cli.main(argv(tmp_path, "--prompt", "text", "--max-new-tokens", "1", "--verify-onnx", "small.onnx")) == 1
    value = report(tmp_path)
    assert value["status"] == "FAIL" and not value["passed"]
    assert value["numeric_steps"][0]["passed"] is False
    assert value["cases"] == []


def test_numeric_comparison_checks_padding_rows_and_equal_argmax(cli):
    expected = logits()
    bad_padding = expected.copy()
    bad_padding[0, -1, 1] = cli.ATOL * 2
    assert cli.compare_step(bad_padding, expected, 2, 8)["passed"] is False
    nonfinite = expected.copy()
    nonfinite[0, -1, 0] = np.inf
    with pytest.raises(ValueError):
        cli.compare_step(nonfinite, expected, 2, 8)
    expected[0, 1, 4] = 1 + cli.ATOL / 4
    actual = expected.copy()
    actual[0, 1, 3], actual[0, 1, 4] = expected[0, 1, 4], expected[0, 1, 3]
    comparison = cli.compare_step(actual, expected, 2, 8)
    assert comparison["max_abs"] < cli.ATOL
    assert comparison["qemu_token_id"] != comparison["ort_token_id"]
    assert not comparison["passed"]


def matching_reuse_metadata(cli, monkeypatch, path):
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    path.with_name("report.json").write_text(json.dumps({"executable_sha256": "matching"}), encoding="utf-8")
    monkeypatch.setattr(cli, "sha256_file", lambda metadata: "matching")


@pytest.mark.parametrize("changes,harts", [
    ({"harts": 4}, 1),
    ({"matmul_policy": "blocked_fma"}, 1),
    ({"matmul_tile": "2x4"}, 1),
    ({"matmul_rhs_layout": "column-panel"}, 1),
    ({"qemu_workspace_layout": "two-buffer-pad-4k"}, 1),
])
def test_reuse_rejects_changed_configuration_or_padding(cli, monkeypatch, tmp_path, changes, harts):
    args = cli.parse_args(["--reuse-build", str(tmp_path / "old/executable.json"),
                           "--tokenizer-dir", "tokens", "--output-dir", str(tmp_path / "new"),
                           "--prompt", "text"])
    matching_reuse_metadata(cli, monkeypatch, args.reuse_build)
    evidence = {"matmul_policy": "sequential", "matmul_tile": "1x4", "matmul_rhs_layout": "original", "qemu_workspace_layout": "none"}
    evidence.update({k: v for k, v in changes.items() if k != "harts"})
    exe = SimpleNamespace(harts=changes.get("harts", 1), evidence=evidence)
    monkeypatch.setattr(cli, "load_executable", lambda path: exe)
    with pytest.raises(ValueError, match="differs"):
        cli.prepare_executable(args, tmp_path, harts)


def test_matching_reuse_returns_same_executable_without_compile(cli, monkeypatch, tmp_path):
    args = cli.parse_args(["--reuse-build", str(tmp_path / "old/executable.json"),
                           "--tokenizer-dir", "tokens", "--output-dir", str(tmp_path / "new"),
                           "--prompt", "text"])
    matching_reuse_metadata(cli, monkeypatch, args.reuse_build)
    exe = SimpleNamespace(harts=1, evidence={"matmul_policy": "sequential"})
    monkeypatch.setattr(cli, "load_executable", lambda path: exe)
    monkeypatch.setattr(cli, "build_external", lambda *a, **kw: pytest.fail("must not rebuild"))
    assert cli.prepare_executable(args, tmp_path, 1) is exe


@pytest.mark.parametrize("options", [
    ["--reference-suite"],
    ["--reference-suite", "--verify-onnx", "model.onnx", "--max-new-tokens", "8"],
    ["--reference-suite", "--verify-onnx", "model.onnx", "--mode", "chat"],
    ["--chat-turn", "hello"],
    ["--prompt", "hello", "--qemu-workspace-layout", "two-buffer-pad-4k"],
])
def test_cli_rejects_mutated_reference_contract_or_implicit_chat_padding(cli, tmp_path, options):
    with pytest.raises(SystemExit) as caught:
        cli.parse_args(argv(tmp_path, *options))
    assert caught.value.code == 2


def test_reference_session_refuses_small_graph_for_full_model_gate(cli, monkeypatch, tmp_path):
    import onnx
    from probes.w2_qwen3_parse import validation
    fixture = json.loads((cli.ROOT / "probes/w5_generation/references.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(onnx, "load", lambda *a, **kw: object())
    monkeypatch.setattr(validation, "audit_graph_structure", lambda graph: {"passed": True, "layer_count": 2})
    with pytest.raises(ValueError, match="complete 28-layer"):
        cli.reference_session(tmp_path / "onnx/model.onnx", fixture)


@pytest.mark.parametrize("audit,shape", [
    (None, (1, 256, 8)),
    (None, (1, 256, 151936)),
    ({"passed": True, "layer_count": 28}, (1, 256, 8)),
    ({"passed": True, "layer_count": 2}, (1, 256, 151936)),
    ({"passed": False, "layer_count": 28}, (1, 256, 151936)),
])
def test_full_gate_requires_executable_audit_and_complete_output(cli, audit, shape):
    evidence = {} if audit is None else {"w5_graph_audit": audit}
    exe = SimpleNamespace(evidence=evidence, output=SimpleNamespace(shape=shape))
    with pytest.raises(ValueError, match="complete graph evidence"):
        cli.require_full_executable(exe)


def test_full_gate_accepts_native_w5_complete_graph_audit(cli):
    exe = SimpleNamespace(evidence={"w5_graph_audit": {"passed": True, "layer_count": 28}},
                          output=SimpleNamespace(shape=(1, 256, 151936)))
    assert cli.require_full_executable(exe) is None


def test_legacy_w4_evidence_survives_two_reuses_without_rechecking_identity(cli, monkeypatch, tmp_path):
    from dataclasses import replace
    from scratchv.runtime.riscv_external import ExternalExecutable
    from scratchv.runtime.riscv_tensor import RiscVToolchain
    from scratchv.ir import DataType
    from scratchv.backend.tensor_c_codegen import TensorSpec

    legacy = tmp_path / "w4"
    legacy.mkdir()
    original = ExternalExecutable(
        tmp_path / "model.elf", tmp_path / "weights", {}, {}, (),
        TensorSpec("logits", DataType.FLOAT32, (1, 256, 151936)), (),
        RiscVToolchain(("cc",), "qemu"), {"harts": 4, "matmul_tile": "2x4",
        "matmul_rhs_layout": "column-panel", "qemu_workspace_layout": "none"}, harts=4,
    )
    identity = cli.save_executable(original, legacy)
    audit = {"passed": True, "layer_count": 28, "graph": "fixed Qwen3 graph"}
    (legacy / "report.json").write_text(json.dumps({"executable_sha256": identity,
        "matmul_policy": "blocked_fma", "graph_audit": audit}), encoding="utf-8")
    checked = []
    hash_file = cli.sha256_file
    monkeypatch.setattr(cli, "sha256_file", lambda path: checked.append(path) or hash_file(path))
    # Exercise the real metadata writer/identity link; lower-level ELF validation
    # is covered by W4. No compiler, model or guest process is needed here.
    monkeypatch.setattr(cli, "load_executable", lambda path: replace(original,
        evidence=json.loads(path.read_text(encoding="utf-8"))["evidence"]))
    monkeypatch.setattr(cli, "build_external", lambda *a, **kw: pytest.fail("must not rebuild"))
    current = legacy / "executable.json"
    for name in ("w5-first", "w5-second"):
        out = tmp_path / name
        out.mkdir()
        args = cli.parse_args(["--reuse-build", str(current), "--tokenizer-dir", "tokens",
            "--output-dir", str(out), "--reference-suite", "--verify-onnx", "onnx/model.onnx",
            "--harts", "4", "--matmul-policy", "blocked_fma", "--matmul-tile", "2x4",
            "--matmul-rhs-layout", "column-panel"])
        exe = cli.prepare_executable(args, out, 4)
        assert exe.evidence["matmul_policy"] == "blocked_fma"
        assert exe.evidence["w5_graph_audit"] == audit
        assert cli.require_full_executable(exe) is None
        assert exe.elf_path == original.elf_path and exe.bundle_dir == original.bundle_dir
        identity = cli.save_executable(exe, out)
        # W5's report contains no legacy graph_audit/matmul_policy fallback.
        (out / "report.json").write_text(json.dumps({"executable_sha256": identity}), encoding="utf-8")
        current = out / "executable.json"
    assert checked == [legacy / "executable.json", tmp_path / "w5-first/executable.json"]
    assert "matmul_policy" not in original.evidence and "w5_graph_audit" not in original.evidence


def test_verified_legacy_report_cannot_overwrite_native_failed_audit(cli, monkeypatch, tmp_path):
    from dataclasses import dataclass

    @dataclass
    class Executable:
        evidence: dict
        harts: int = 1

    metadata = tmp_path / "old/executable.json"
    matching_reuse_metadata(cli, monkeypatch, metadata)
    record = json.loads(metadata.with_name("report.json").read_text())
    record.update(matmul_policy="sequential", graph_audit={"passed": True, "layer_count": 28})
    metadata.with_name("report.json").write_text(json.dumps(record))
    exe = Executable({"w5_graph_audit": {"passed": False, "layer_count": 2}})
    monkeypatch.setattr(cli, "load_executable", lambda path: exe)
    args = cli.parse_args(["--reuse-build", str(metadata), "--tokenizer-dir", "tokens",
                           "--output-dir", str(tmp_path / "new"), "--prompt", "text"])
    reused = cli.prepare_executable(args, tmp_path, 1)
    assert reused.evidence["matmul_policy"] == "sequential"
    assert reused.evidence["w5_graph_audit"] == exe.evidence["w5_graph_audit"]


def test_reference_suite_rejects_small_executable_before_fma_or_oracle(cli, runtime, tmp_path):
    runtime.exe.output = SimpleNamespace(shape=(1, 256, 8))
    assert cli.main(argv(tmp_path, "--reference-suite", "--verify-onnx", "onnx/model.onnx")) == 1
    value = report(tmp_path)
    assert value["status"] == "FAIL" and value["full_model_gate"] == "NOT_RUN"
    assert "complete graph evidence" in value["error"]
    assert runtime.adapters == [] and runtime.fma == []


def test_signal_during_fma_cannot_be_swallowed_by_exception_handler(cli, runtime, tmp_path, monkeypatch):
    import signal
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    swallowed = []

    def fma(*args, **kwargs):
        try:
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        except Exception as exc:
            swallowed.append(exc)
            return {"passed": False, "status": "FAIL"}
        pytest.fail("signal did not abort FMA")

    monkeypatch.setattr(cli, "run_fma_probe", fma)
    assert cli.main(argv(tmp_path, "--prompt", "text", "--max-new-tokens", "1")) == 130
    value = report(tmp_path)
    assert value["status"] == "CANCELLED" and not value["passed"]
    assert "KeyboardInterrupt" in value["error"]
    assert value["stage"] == "fma_preflight" and value["qemu_forwards"] == 0
    assert not swallowed
    assert all(signal.getsignal(sig) is handler for sig, handler in previous.items())


@pytest.mark.parametrize("record", [None, {}, {"executable_sha256": "drifted"}])
def test_reuse_identity_is_required_before_loading_or_compiling(cli, monkeypatch, tmp_path, record):
    metadata = tmp_path / "old/executable.json"
    metadata.parent.mkdir()
    metadata.write_text("{}", encoding="utf-8")
    if record is not None:
        metadata.with_name("report.json").write_text(json.dumps(record), encoding="utf-8")
    args = cli.parse_args(["--reuse-build", str(metadata), "--tokenizer-dir", "tokens",
                           "--output-dir", str(tmp_path / "new"), "--prompt", "text"])
    monkeypatch.setattr(cli, "sha256_file", lambda path: "matching")
    monkeypatch.setattr(cli, "load_executable", lambda *a: pytest.fail("identity must be checked before load"))
    monkeypatch.setattr(cli, "build_external", lambda *a, **kw: pytest.fail("identity failure must not compile"))
    with pytest.raises(FileNotFoundError if record is None else ValueError):
        cli.prepare_executable(args, tmp_path, 1)


@pytest.fixture
def reference_suite(cli, runtime, monkeypatch, tmp_path):
    root = tmp_path / "fixture-root"
    (root / "probes/w5_generation").mkdir(parents=True)
    cases = [{"name": name, "prompt": name, "max_new_tokens": 2,
              "prompt_token_ids": [1, 2], "generated_token_ids": [3, 3],
              "generated_text": "xx", "stop_reason": "max_new_tokens",
              "required_substring": "x"} for name in ("france", "germany", "japan")]
    fixture = {"status": "frozen", "tokenizer": {"pad_token_id": 0,
        "generation_eos_token_ids": [6], "model_vocab_size": 8, "tokenizer_vocab_size": 7},
        "cases": cases}
    (root / "probes/w5_generation/references.json").write_text(json.dumps(fixture), encoding="utf-8")
    monkeypatch.setattr(cli, "ROOT", root)
    runtime.exe.evidence["w5_graph_audit"] = {"passed": True, "layer_count": 28}
    runtime.exe.output = SimpleNamespace(shape=(1, 256, 151936))
    runtime.oracle = logits()
    monkeypatch.setattr(cli, "reference_session", lambda *a: SimpleNamespace(run=lambda names, feed: [runtime.oracle]))
    snapshots = []
    atomic = cli.atomic_text

    def save(path, text):
        if path.name == "report.json":
            snapshots.append(json.loads(text))
        atomic(path, text)

    monkeypatch.setattr(cli, "atomic_text", save)
    return SimpleNamespace(fixture=fixture, snapshots=snapshots)


def test_full_suite_gate_records_running_then_pass(cli, runtime, reference_suite, tmp_path):
    assert cli.main(argv(tmp_path, "--reference-suite", "--verify-onnx", "onnx/model.onnx")) == 0
    value = report(tmp_path)
    assert value["full_model_gate"] == "PASS" and value["status"] == "PASS"
    assert len(value["numeric_steps"]) == 6 and len(value["cases"]) == 3
    assert all(row["reference_passed"] for row in value["cases"])
    assert all(row["numeric_step_count"] == row["forward_count"] == 2 for row in value["cases"])
    assert [(row["case"], row["step"], row["valid_length"]) for row in value["numeric_steps"]] == [
        (name, step, 2 + step) for name in ("france", "germany", "japan") for step in range(2)]
    assert reference_suite.snapshots[0]["full_model_gate"] == "RUNNING"
    assert reference_suite.snapshots[0]["qemu_forwards"] == 0


@pytest.mark.parametrize("failure,code,gate", [
    (TimeoutError("expired"), 1, "FAIL"),
    (GenerationCancelledError("cancelled"), 130, "CANCELLED"),
    (KeyboardInterrupt(), 130, "CANCELLED"),
])
def test_full_suite_running_error_has_terminal_gate(cli, runtime, reference_suite, tmp_path, failure, code, gate):
    runtime.failure = failure
    assert cli.main(argv(tmp_path, "--reference-suite", "--verify-onnx", "onnx/model.onnx")) == code
    value = report(tmp_path)
    assert value["full_model_gate"] == gate and value["status"] == gate
    assert value["stage"] == "generation" and value["forward_attempts"] == 1
    assert value["cases"] == []
    assert reference_suite.snapshots[0]["full_model_gate"] == "RUNNING"


def test_full_suite_numeric_failure_marks_failed_not_not_run(cli, runtime, reference_suite, tmp_path):
    runtime.oracle[0, -1, 0] = cli.ATOL * 2
    assert cli.main(argv(tmp_path, "--reference-suite", "--verify-onnx", "onnx/model.onnx")) == 1
    value = report(tmp_path)
    assert value["full_model_gate"] == "FAIL" and value["status"] == "FAIL"
    assert value["qemu_forwards"] == 1 and value["cases"] == []
    assert value["numeric_steps"][0]["passed"] is False


def test_full_suite_output_reference_failure_marks_failed(cli, runtime, reference_suite, tmp_path):
    fixture = reference_suite.fixture
    fixture["cases"][0]["generated_text"] = "wrong frozen text"
    (cli.ROOT / "probes/w5_generation/references.json").write_text(json.dumps(fixture), encoding="utf-8")
    assert cli.main(argv(tmp_path, "--reference-suite", "--verify-onnx", "onnx/model.onnx")) == 1
    value = report(tmp_path)
    assert value["full_model_gate"] == "FAIL" and value["qemu_forwards"] == 2
    assert value["cases"][0]["reference_passed"] is False
    assert all(row["passed"] for row in value["numeric_steps"])


@pytest.mark.parametrize("stage,cancel", [
    ("prepare", False), ("fma_preflight", False),
    ("fma_preflight", True), ("oracle_prepare", False),
])
def test_full_suite_pre_generation_failure_remains_not_run(cli, runtime, reference_suite, tmp_path, monkeypatch, stage, cancel):
    def fail(*args, **kwargs):
        raise KeyboardInterrupt("cancel") if cancel else ValueError("setup failed")

    target = {"prepare": "prepare_executable", "fma_preflight": "run_fma_probe",
              "oracle_prepare": "reference_session"}[stage]
    monkeypatch.setattr(cli, target, fail)
    assert cli.main(argv(tmp_path, "--reference-suite", "--verify-onnx", "onnx/model.onnx")) == (130 if cancel else 1)
    value = report(tmp_path)
    assert value["full_model_gate"] == "NOT_RUN" and value["stage"] == stage
    assert value["status"] == ("CANCELLED" if cancel else "FAIL")
    assert value["qemu_forwards"] == 0 and value["numeric_steps"] == []
    assert all(row["full_model_gate"] == "NOT_RUN" for row in reference_suite.snapshots)


@pytest.mark.parametrize("calls", [0, 1, 3])
def test_full_suite_rejects_missing_or_extra_forward_evidence(cli, runtime, reference_suite, tmp_path, monkeypatch, calls):
    from scratchv.runtime.generation import GenerationResult
    from scratchv.runtime.llm_inputs import prepare_inputs

    def faulty_generate(prompt, *, forward, **kwargs):
        history = [1, 2]
        for _ in range(calls):
            forward(prepare_inputs(history, pad_token_id=0, vocab_size=8))
            history.append(3)
        return GenerationResult((1, 2), (3, 3), "xx", "max_new_tokens")

    monkeypatch.setattr(cli, "generate", faulty_generate)
    assert cli.main(argv(tmp_path, "--reference-suite", "--verify-onnx", "onnx/model.onnx")) == 1
    value = report(tmp_path)
    assert value["full_model_gate"] == "FAIL" and not value["passed"]
    assert value["qemu_forwards"] == min(calls, 2)


@pytest.mark.parametrize("fault", ["wrong_length", "stale_history", "mask", "padding"])
def test_full_suite_rejects_wrong_input_history_even_when_logits_match(cli, runtime, reference_suite, tmp_path, monkeypatch, fault):
    from dataclasses import replace
    from scratchv.runtime.generation import GenerationResult
    from scratchv.runtime.llm_inputs import prepare_inputs

    def faulty_generate(prompt, *, forward, **kwargs):
        prepared = prepare_inputs([1, 2], pad_token_id=0, vocab_size=8)
        if fault == "wrong_length":
            prepared = replace(prepared, valid_length=1)
        elif fault == "stale_history":
            prepared.input_ids[0, 1] = 4
        elif fault == "mask":
            prepared.attention_mask[0, 0, 0, 1] = 0
        else:
            prepared.input_ids[0, -1] = 5
        forward(prepared)
        forward(prepare_inputs([1, 2, 3], pad_token_id=0, vocab_size=8))
        return GenerationResult((1, 2), (3, 3), "xx", "max_new_tokens")

    monkeypatch.setattr(cli, "generate", faulty_generate)
    assert cli.main(argv(tmp_path, "--reference-suite", "--verify-onnx", "onnx/model.onnx")) == 1
    value = report(tmp_path)
    assert value["full_model_gate"] == "FAIL" and value["qemu_forwards"] == 0


@pytest.mark.parametrize("same_argmax", [False, True])
def test_numeric_comparison_rejects_wrong_selected_row_independently(cli, monkeypatch, same_argmax):
    expected = logits()
    expected[0, 0, 3 if same_argmax else 4] = 2
    monkeypatch.setattr(cli, "last_valid_logits", lambda array, length, **kwargs: array[0, 0].copy())
    result = cli.compare_step(expected.copy(), expected, 2, 8)
    assert result["max_abs"] == 0 and result["qemu_token_id"] == result["ort_token_id"]
    assert result["selected_row_passed"] is False and result["passed"] is False


def test_full_suite_rejects_frozen_token_drift_with_equal_qemu_ort_logits(cli, runtime, reference_suite, tmp_path, monkeypatch):
    runtime.oracle = logits(4)
    original = cli.QemuForward

    class Adapter(original):
        def __call__(self, prepared):
            super().__call__(prepared)
            return logits(4)

    monkeypatch.setattr(cli, "QemuForward", Adapter)
    assert cli.main(argv(tmp_path, "--reference-suite", "--verify-onnx", "onnx/model.onnx")) == 1
    value = report(tmp_path)
    row = value["numeric_steps"][0]
    assert value["full_model_gate"] == "FAIL" and value["qemu_forwards"] == 1
    assert row["max_abs"] == 0 and row["selected_row_passed"] is True
    assert row["qemu_token_id"] == row["ort_token_id"] == 4
    assert row["reference_token_id"] == 3 and row["reference_token_passed"] is False
    assert row["case"] == "france" and row["step"] == 0 and row["valid_length"] == 2


@pytest.mark.parametrize("error,code,status", [
    (TimeoutError("expired after first forward"), 1, "FAIL"),
    (GenerationCancelledError("cancelled after first forward"), 130, "CANCELLED"),
])
def test_full_suite_abort_retains_completed_step_evidence(cli, runtime, reference_suite, tmp_path, monkeypatch, error, code, status):
    generate = cli.generate

    def abort_after_forward(prompt, *, forward, **kwargs):
        def guarded(prepared):
            forward(prepared)
            raise error
        return generate(prompt, forward=guarded, **kwargs)

    monkeypatch.setattr(cli, "generate", abort_after_forward)
    assert cli.main(argv(tmp_path, "--reference-suite", "--verify-onnx", "onnx/model.onnx")) == code
    value = report(tmp_path)
    assert value["full_model_gate"] == status and value["status"] == status
    assert value["cases"] == [] and value["qemu_forwards"] == 1
    assert value["numeric_steps"][0]["passed"] is True
    assert value["numeric_steps"][0]["case"] == "france"


@pytest.mark.parametrize("requested,cpu_count,resolved", [
    (None, 8, 1), ("1", 8, 1), ("2", 8, 2), ("4", 1, 4),
    ("auto", 1, 1), ("auto", 3, 2), ("auto", 12, 4),
])
def test_cli_uses_real_hart_resolver_without_changing_default(cli, runtime, tmp_path, monkeypatch, requested, cpu_count, resolved):
    from scratchv.runtime import riscv_harts
    affinity_reads = []
    monkeypatch.setattr(riscv_harts.os, "sched_getaffinity", lambda pid: affinity_reads.append(pid) or set(range(cpu_count)))
    monkeypatch.setattr(cli, "resolve_harts", riscv_harts.resolve_harts)
    options = [] if requested is None else ["--harts", requested]
    assert cli.main(argv(tmp_path, "--build-only", *options)) == 0
    value = report(tmp_path)
    assert value["harts"] == resolved and runtime.prepared[0][2] == resolved
    assert value["qemu_workspace_layout"] == "none" and value["qemu_forwards"] == 0
    assert affinity_reads == ([0] if requested == "auto" else [])


@pytest.mark.parametrize("built", [None, "sequential", "two-four-lane-last-axis"])
@pytest.mark.parametrize("requested", ["sequential", "two-four-lane-last-axis"])
def test_w5_reuse_mean_policy_requires_explicit_matching_build(cli, monkeypatch, tmp_path, built, requested):
    metadata = tmp_path / "old/executable.json"
    matching_reuse_metadata(cli, monkeypatch, metadata)
    record = json.loads(metadata.with_name("report.json").read_text())
    if built is not None:
        record["mean_policy"] = built
    metadata.with_name("report.json").write_text(json.dumps(record))
    evidence = {"matmul_policy": "sequential"}
    if built is not None:
        evidence["mean_policy"] = built
    exe = SimpleNamespace(harts=1, evidence=evidence)
    monkeypatch.setattr(cli, "load_executable", lambda path: exe)
    monkeypatch.setattr(cli, "build_external", lambda *a, **kw: pytest.fail("reuse must not compile"))
    args = cli.parse_args(["--reuse-build", str(metadata), "--tokenizer-dir", "tokens",
        "--output-dir", str(tmp_path / "out"), "--prompt", "text", "--mean-policy", requested])
    if (built or "sequential") == requested:
        assert cli.prepare_executable(args, tmp_path, 1) is exe
    else:
        with pytest.raises(ValueError, match="Mean policy"):
            cli.prepare_executable(args, tmp_path, 1)


def test_w5_reuse_mean_evidence_and_sibling_report_must_agree(cli, monkeypatch, tmp_path):
    metadata = tmp_path / "old/executable.json"
    matching_reuse_metadata(cli, monkeypatch, metadata)
    exe = SimpleNamespace(harts=1, evidence={"matmul_policy": "sequential", "mean_policy": "two-four-lane-last-axis"})
    monkeypatch.setattr(cli, "load_executable", lambda path: exe)
    args = cli.parse_args(["--reuse-build", str(metadata), "--tokenizer-dir", "tokens",
        "--output-dir", str(tmp_path / "out"), "--prompt", "text", "--mean-policy", "two-four-lane-last-axis"])
    with pytest.raises(ValueError, match="Mean policy"):
        cli.prepare_executable(args, tmp_path, 1)


@pytest.mark.parametrize("policy", ["sequential", "two-four-lane-last-axis"])
def test_w5_mean_policy_reaches_real_codegen_without_compiler(cli, monkeypatch, tmp_path, policy):
    from scratchv.frontend import onnx_parser
    from scratchv.ir.builder import IRBuilder
    from scratchv.ir.types import Value
    from scratchv.runtime.riscv_external import ExternalExecutable
    from scratchv.runtime.riscv_tensor import RiscVToolchain
    builder = IRBuilder()
    x = Value("x", shape=(2, 8))
    builder.new_function("main", [x]); builder.new_block("entry"); builder.ret(builder.neg(x))
    monkeypatch.setattr(onnx_parser, "ONNXParser", lambda: SimpleNamespace(initializers={}, parse=lambda *a, **kw: builder.program))
    toolchain = RiscVToolchain(("unused-cc",), "unused-qemu")
    monkeypatch.setattr(cli, "discover_toolchain", lambda *a: toolchain)
    built = []

    def build(artifact, bundle, out, tools, **kwargs):
        built.append(artifact)
        return ExternalExecutable(out / "model.elf", bundle, {}, {}, artifact.inputs,
            artifact.output, artifact.external_weights, tools, {"mean_policy": artifact.mean_policy})

    monkeypatch.setattr(cli, "build_external", build)
    args = cli.parse_args(argv(tmp_path, "--build-only", "--mean-policy", policy))
    exe = cli.prepare_executable(args, tmp_path, 1)
    assert len(built) == 1 and built[0].mean_policy == policy
    assert exe.evidence["mean_policy"] == policy and exe.harts == 1
    assert built[0].matmul_policy == "sequential" and built[0].qemu_workspace_layout == "none"


def test_w5_mean_policy_is_explicit_in_report(cli, runtime, tmp_path):
    assert cli.main(argv(tmp_path, "--build-only", "--mean-policy", "two-four-lane-last-axis")) == 0
    assert report(tmp_path)["mean_policy"] == "two-four-lane-last-axis"
    assert runtime.prepared[0][0].mean_policy == "two-four-lane-last-axis"


@pytest.mark.parametrize("built", [None, "libm", "fp32-reference"])
@pytest.mark.parametrize("requested", ["libm", "fp32-reference"])
def test_w5_reuse_nonlinear_policy_requires_explicit_matching_build(cli, monkeypatch, tmp_path, built, requested):
    metadata = tmp_path / "old/executable.json"
    matching_reuse_metadata(cli, monkeypatch, metadata)
    record = json.loads(metadata.with_name("report.json").read_text())
    if built is not None:
        record["nonlinear_policy"] = built
    metadata.with_name("report.json").write_text(json.dumps(record))
    evidence = {"matmul_policy": "sequential"}
    if built is not None:
        evidence["nonlinear_policy"] = built
    exe = SimpleNamespace(harts=1, evidence=evidence)
    monkeypatch.setattr(cli, "load_executable", lambda path: exe)
    monkeypatch.setattr(cli, "build_external", lambda *a, **kw: pytest.fail("reuse must not compile"))
    args = cli.parse_args(["--reuse-build", str(metadata), "--tokenizer-dir", "tokens",
        "--output-dir", str(tmp_path / "out"), "--prompt", "text", "--nonlinear-policy", requested])
    if (built or "libm") == requested:
        assert cli.prepare_executable(args, tmp_path, 1) is exe
    else:
        with pytest.raises(ValueError, match="nonlinear policy"):
            cli.prepare_executable(args, tmp_path, 1)


def test_w5_reuse_nonlinear_evidence_and_sibling_report_must_agree(cli, monkeypatch, tmp_path):
    metadata = tmp_path / "old/executable.json"
    matching_reuse_metadata(cli, monkeypatch, metadata)
    exe = SimpleNamespace(harts=1, evidence={"matmul_policy": "sequential", "nonlinear_policy": "fp32-reference"})
    monkeypatch.setattr(cli, "load_executable", lambda path: exe)
    args = cli.parse_args(["--reuse-build", str(metadata), "--tokenizer-dir", "tokens",
        "--output-dir", str(tmp_path / "out"), "--prompt", "text", "--nonlinear-policy", "fp32-reference"])
    with pytest.raises(ValueError, match="nonlinear policy"):
        cli.prepare_executable(args, tmp_path, 1)


@pytest.mark.parametrize("policy", ["libm", "fp32-reference"])
def test_w5_nonlinear_policy_reaches_real_codegen_without_compiler(cli, monkeypatch, tmp_path, policy):
    from scratchv.frontend import onnx_parser
    from scratchv.ir.builder import IRBuilder
    from scratchv.ir.types import Value
    from scratchv.runtime.riscv_external import ExternalExecutable
    from scratchv.runtime.riscv_tensor import RiscVToolchain
    builder = IRBuilder()
    x = Value("x", shape=(2, 8))
    builder.new_function("main", [x]); builder.new_block("entry"); builder.ret(builder.neg(x))
    monkeypatch.setattr(onnx_parser, "ONNXParser", lambda: SimpleNamespace(initializers={}, parse=lambda *a, **kw: builder.program))
    toolchain = RiscVToolchain(("unused-cc",), "unused-qemu")
    monkeypatch.setattr(cli, "discover_toolchain", lambda *a: toolchain)
    built = []

    def build(artifact, bundle, out, tools, **kwargs):
        built.append(artifact)
        return ExternalExecutable(out / "model.elf", bundle, {}, {}, artifact.inputs,
            artifact.output, artifact.external_weights, tools, {"nonlinear_policy": artifact.nonlinear_policy})

    monkeypatch.setattr(cli, "build_external", build)
    args = cli.parse_args(argv(tmp_path, "--build-only", "--nonlinear-policy", policy))
    exe = cli.prepare_executable(args, tmp_path, 1)
    assert len(built) == 1 and built[0].nonlinear_policy == policy
    assert exe.evidence["nonlinear_policy"] == policy and exe.harts == 1
    assert built[0].matmul_policy == "sequential" and built[0].qemu_workspace_layout == "none"


def test_w5_nonlinear_policy_is_explicit_in_report(cli, runtime, tmp_path):
    assert cli.main(argv(tmp_path, "--build-only", "--nonlinear-policy", "fp32-reference")) == 0
    assert report(tmp_path)["nonlinear_policy"] == "fp32-reference"
    assert runtime.prepared[0][0].nonlinear_policy == "fp32-reference"


@pytest.mark.parametrize("policy", ["unknown", "sequential", ""])
def test_w5_unknown_nonlinear_policy_fails_before_output(cli, tmp_path, policy):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv(tmp_path, "--build-only", "--nonlinear-policy", policy))
    assert exc.value.code == 2 and not (tmp_path / "run").exists()


def test_finite_extreme_logits_have_json_safe_diagnostic_margin(cli):
    expected = np.full((1, 256, 8), -np.finfo(np.float32).max, dtype=np.float32)
    expected[0, 1, 3] = np.finfo(np.float32).max
    with np.errstate(over="raise"):
        comparison = cli.compare_step(expected.copy(), expected, 2, 8)
    assert comparison["passed"] and comparison["max_abs"] == 0.0
    assert comparison["qemu_token_id"] == comparison["ort_token_id"] == 3
    assert comparison["ort_top2_margin"] == 2.0 * float(np.finfo(np.float32).max)
    json.dumps(comparison, allow_nan=False)
