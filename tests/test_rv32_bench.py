"""Tests for the RV32 full benchmark driver (topic 27).

Dynamic tests run only tiny synthetic models with explicit budgets; the big
``cnn.onnx`` full simulation is deliberately never executed here.
"""

import json
from copy import deepcopy

import pytest

from scratchv.standalone import bench_report, rv32_bench


# ───────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ───────────────────────────────────────────────────────────────────────────

def _build_mini_onnx(path):
    onnx = pytest.importorskip("onnx")
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.RandomState(0)
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 8, 8])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 6, 6])
    weight = numpy_helper.from_array(
        (rng.randn(1, 1, 3, 3).astype(np.float32) * 0.1), "W",
    )
    bias = numpy_helper.from_array(np.zeros(1, np.float32), "B")
    node = helper.make_node(
        "Conv", ["input", "W", "B"], ["output"],
        kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[1, 1],
    )
    graph = helper.make_graph([node], "mini", [inp], [out], [weight, bias])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.save(model, str(path))


@pytest.fixture(scope="module")
def mini_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("mini_rv32") / "mini.onnx"
    _build_mini_onnx(path)
    return path


@pytest.fixture(scope="module")
def mini_compiled(mini_model, tmp_path_factory):
    out = tmp_path_factory.mktemp("mini_build")
    info = rv32_bench.compile_scratchv(
        str(mini_model), str(out / "_sv.bin"), str(out / "_sv.s"),
    )
    assert info["status"] == "success", info
    return {"onnx": mini_model, "out": out, "info": info}


def _fake_model():
    return {
        "path": "mini.onnx", "sha256": "0" * 64, "bytes": 241,
        "input_name": "input", "input_shape": [1, 1, 8, 8],
        "output_name": "output", "output_shape": [1, 1, 6, 6],
        "initializer_count": 2, "weight_bytes": 40,
    }


def _fake_env():
    return {
        "python": "3.11.0", "numpy": "2.0.0",
        "tinyfive": "1.0.0", "llvmlite": None,
    }


def _fake_scratchv(*, completion="halted", limit=None, executed=100,
                   ops=None):
    if ops is None:
        ops = {
            "total": executed, "load": 7, "store": 3, "mul": 5,
            "add": 10, "madd": 0, "branch": 4,
        }
    return {
        "compile": {
            "status": "success", "binary": "_sv.bin", "binary_bytes": 680,
            "binary_sha256": "1" * 64, "code_bytes": 640, "data_offset": 640,
            "data_offset_source": "compiler_stdout", "data_bytes": 40,
            "workspace_bytes": 400, "static_insns": 160,
            "static_source": "asm_scan", "elapsed_s": 1.0,
        },
        "static_instruction_mix": {
            "source": "asm_scan", "load": 7, "store": 3, "mul": 5,
            "add": 10, "madd": 0, "branch": 4, "other": 0,
        },
        "dynamic": {
            "source": "simulated", "simulator": "tinyfive",
            "simulator_version": "1.0.0", "completion": completion,
            "limit": limit, "executed": executed, "timeout_s": 900.0,
            "elapsed_s": 1.0, "memory_size_bytes": 268435456,
            "input_seed": 42, "input_elements": 64, "halt_addr": 688,
            "ops": ops, "x_registers_used": 12, "x_usage_total": 30,
            "f_registers_used": 0, "per_label": None,
            "per_label_note": "tinyfive exe() exposes no per-PC trace",
            "last_error": None,
        },
        "output": {
            "addr": 201326592, "elements": 36,
            "raw_hex": "0x" + "00" * 144, "q16_16": [0.0] * 36,
            "completion": completion, "partial": completion != "halted",
        },
    }


def _unavailable_llvm(reason="llvm executable image pipeline not "
                              "implemented (topic 25 boundary)"):
    return {
        "compile": {
            "status": "skipped", "reason": "llvmlite not available",
            "isa_detected": None, "isa_mismatch": False,
            "static_insns": 0, "static_source": "asm_scan", "elapsed_s": 0.0,
        },
        "dynamic": {
            "source": "unavailable", "simulator": "tinyfive",
            "completion": "not_run", "reason": reason, "ops": None,
        },
    }


def _valid_fake_report():
    return rv32_bench.build_report(
        _fake_model(), _fake_env(), _fake_scratchv(), _unavailable_llvm(),
    )


# ───────────────────────────────────────────────────────────────────────────
# T0: defaults / budget semantics
# ───────────────────────────────────────────────────────────────────────────

def test_cli_defaults_to_full_simulation_no_hidden_truncation():
    args = rv32_bench.build_parser().parse_args(["model.onnx"])
    assert args.max_instructions == 0
    assert args.full is False
    assert args.mem_size == 268435456
    assert args.chunk_instructions == 10_000_000

    with pytest.raises(SystemExit) as excinfo:
        rv32_bench.main(["model.onnx", "--full", "--max-instructions", "5"])
    assert excinfo.value.code == rv32_bench.EXIT_USAGE


# ───────────────────────────────────────────────────────────────────────────
# T1: full run of a tiny model matches an independent emulator
# ───────────────────────────────────────────────────────────────────────────

def test_full_run_small_model_matches_reference(tmp_path, mini_model,
                                                mini_compiled):
    pytest.importorskip("tinyfive")
    out = tmp_path / "full"
    rc = rv32_bench.main([
        str(mini_model), "--full", "--quiet", "--output-dir", str(out),
        "--timeout", "120", "--chunk-instructions", "4096",
    ])
    assert rc == rv32_bench.EXIT_OK

    report = json.loads((out / "rv32_bench.json").read_text())
    dyn = report["scratchv"]["dynamic"]
    assert dyn["completion"] == "halted"
    assert dyn["executed"] > 0
    assert dyn["ops"]["total"] == dyn["executed"]
    assert any(
        "no automatic" in w for w in report["warnings"]
    ), report["warnings"]
    out_block = report["scratchv"]["output"]
    assert out_block["completion"] == "halted"
    assert out_block["partial"] is False
    assert isinstance(out_block["q16_16"], list)
    assert rv32_bench.audit_provenance(report) == []
    assert bench_report.validate_report_schema(report) == []

    from scratchv.standalone.benchmark import RV32EmulatorFast

    model_info = rv32_bench.get_model_info(str(mini_model))
    binary = (out / "_sv.bin").read_bytes()
    data_offset = report["scratchv"]["compile"]["data_offset"]
    emu = RV32EmulatorFast(mem_size_mb=256)
    emu.load_unified_binary(binary, data_offset, load_addr=0)
    emu.regs[2] = rv32_bench.SP_ADDR
    emu.regs[10] = rv32_bench.INPUT_ADDR
    emu.regs[11] = rv32_bench.OUTPUT_ADDR
    blob = rv32_bench.build_input_q16(model_info["input_elements"], 42)
    emu.mem[rv32_bench.INPUT_ADDR:rv32_bench.INPUT_ADDR + len(blob)] = blob
    perf = emu.run(max_instr=2_000_000_000)

    assert perf.total == dyn["executed"]
    assert perf.load_count == dyn["ops"]["load"]
    assert perf.store_count == dyn["ops"]["store"]
    assert perf.branch_total == dyn["ops"]["branch"]

    raw_reference = "".join(
        f"{emu.read_mem_i32(rv32_bench.OUTPUT_ADDR + 4 * i) & 0xFFFFFFFF:08x}"
        for i in range(model_info["output_elements"])
    )
    assert report["scratchv"]["output"]["raw_hex"] == "0x" + raw_reference

    rerun = rv32_bench.run_simulation(
        asm_path=str(out / "_sv.s"),
        binary_path=str(out / "_sv.bin"),
        data_offset=data_offset,
        workspace_bytes=report["scratchv"]["compile"]["workspace_bytes"],
        input_elements=model_info["input_elements"],
        output_elements=model_info["output_elements"],
        max_instructions=0, mem_size=268435456, timeout_s=60.0,
        chunk_instructions=4096, input_seed=42,
    )
    assert rerun["dynamic"]["completion"] == "halted"
    assert rerun["dynamic"]["executed"] == dyn["executed"]
    assert rerun["dynamic"]["ops"] == dyn["ops"]
    assert rerun["output"]["raw_hex"] == report["scratchv"]["output"]["raw_hex"]


# ───────────────────────────────────────────────────────────────────────────
# T2: budget truncation is labeled, never presented as full
# ───────────────────────────────────────────────────────────────────────────

def test_budget_exhausted_is_labeled(tmp_path, mini_model):
    pytest.importorskip("tinyfive")
    out = tmp_path / "budget"
    rc = rv32_bench.main([
        str(mini_model), "--max-instructions", "1000", "--quiet",
        "--output-dir", str(out), "--timeout", "120",
        "--chunk-instructions", "4096",
    ])
    assert rc == rv32_bench.EXIT_OK

    report = json.loads((out / "rv32_bench.json").read_text())
    dyn = report["scratchv"]["dynamic"]
    assert dyn["completion"] == "budget_exhausted"
    assert dyn["limit"] == 1000
    assert dyn["executed"] == 1000
    assert dyn["ops"]["total"] == 1000
    assert report["comparison"]["dynamic_instruction_ratio"] is None
    assert "budget_exhausted" in report["comparison"]["incomparable_reason"]
    out_block = report["scratchv"]["output"]
    assert out_block["completion"] == "budget_exhausted"
    assert out_block["partial"] is True
    assert isinstance(out_block["q16_16"], list)
    assert rv32_bench.audit_provenance(report) == []
    assert bench_report.validate_report_schema(report) == []

    markdown = (out / "rv32_bench.md").read_text()
    assert "[measured/budget]" in markdown
    assert "[measured/partial]" in markdown
    assert "dynamic instruction ratio" not in markdown

    rc_strict = rv32_bench.main([
        str(mini_model), "--max-instructions", "1000", "--quiet",
        "--fail-on-incomplete", "--output-dir", str(tmp_path / "budget2"),
        "--timeout", "120", "--chunk-instructions", "4096",
    ])
    assert rc_strict == rv32_bench.EXIT_INCOMPLETE


# ───────────────────────────────────────────────────────────────────────────
# T2b: wall-clock timeout keeps partial counters and partial output
# ───────────────────────────────────────────────────────────────────────────

def test_wall_clock_timeout_is_labeled_partial(tmp_path, mini_model):
    pytest.importorskip("tinyfive")
    out = tmp_path / "timeout"
    rc = rv32_bench.main([
        str(mini_model), "--quiet", "--output-dir", str(out),
        "--timeout", "0.05", "--chunk-instructions", "4096",
    ])
    assert rc == rv32_bench.EXIT_OK

    report = json.loads((out / "rv32_bench.json").read_text())
    dyn = report["scratchv"]["dynamic"]
    assert dyn["source"] == "simulated"
    assert dyn["completion"] == "timeout"
    assert dyn["limit"] is None
    assert dyn["executed"] > 0
    assert dyn["ops"]["total"] == dyn["executed"]
    assert report["comparison"]["dynamic_instruction_ratio"] is None
    assert "timeout" in report["comparison"]["incomparable_reason"]
    assert any("timeout" in w for w in report["warnings"])

    out_block = report["scratchv"]["output"]
    assert out_block["completion"] == "timeout"
    assert out_block["partial"] is True
    assert isinstance(out_block["q16_16"], list)
    assert rv32_bench.audit_provenance(report) == []
    assert bench_report.validate_report_schema(report) == []

    markdown = (out / "rv32_bench.md").read_text()
    assert "[measured/timeout]" in markdown
    assert "[measured/partial]" in markdown

    rc_strict = rv32_bench.main([
        str(mini_model), "--quiet", "--fail-on-incomplete",
        "--output-dir", str(tmp_path / "timeout_strict"),
        "--timeout", "0.05", "--chunk-instructions", "4096",
    ])
    assert rc_strict == rv32_bench.EXIT_INCOMPLETE


# ───────────────────────────────────────────────────────────────────────────
# T3: missing simulator never fabricates dynamic data
# ───────────────────────────────────────────────────────────────────────────

class _UnavailableProfiledMachine:
    available = False

    def __init__(self, mem_size=0):
        self.mem_size = mem_size


def test_report_requires_provenance(tmp_path, mini_model, monkeypatch):
    monkeypatch.setattr(
        rv32_bench, "ProfiledMachine", _UnavailableProfiledMachine,
    )
    out = tmp_path / "static"
    rc = rv32_bench.main([
        str(mini_model), "--allow-missing-simulator", "--quiet",
        "--output-dir", str(out),
    ])
    assert rc == rv32_bench.EXIT_OK

    report = json.loads((out / "rv32_bench.json").read_text())
    dyn = report["scratchv"]["dynamic"]
    assert dyn["source"] == "unavailable"
    assert dyn["ops"] is None
    assert dyn["completion"] == "not_run"
    assert dyn["reason"]
    assert report["scratchv"]["compile"]["static_insns"] > 0
    assert report["scratchv"]["compile"]["static_source"] == "asm_scan"
    assert report["comparison"]["dynamic_instruction_ratio"] is None
    out_block = report["scratchv"]["output"]
    assert out_block["completion"] == "not_run"
    assert out_block["partial"] is True
    assert out_block["raw_hex"] is None
    assert out_block["q16_16"] is None
    assert rv32_bench.audit_provenance(report) == []
    assert bench_report.validate_report_schema(report) == []

    markdown = (out / "rv32_bench.md").read_text()
    assert "[static]" in markdown
    assert "[unavailable]" in markdown

    rc_no_flag = rv32_bench.main([
        str(mini_model), "--quiet", "--output-dir", str(tmp_path / "static2"),
    ])
    assert rc_no_flag == rv32_bench.EXIT_NO_SIMULATOR


# ───────────────────────────────────────────────────────────────────────────
# T4: memory layout validation refuses impossible budgets
# ───────────────────────────────────────────────────────────────────────────

def test_memory_layout_validation(tmp_path, mini_model, capsys):
    out = tmp_path / "memfail"
    rc = rv32_bench.main([
        str(mini_model), "--mem-size", "1048576", "--output-dir", str(out),
    ])
    assert rc == rv32_bench.EXIT_LAYOUT
    stderr = capsys.readouterr().err
    assert "memory_layout_invalid" in stderr
    assert "need mem_size >=" in stderr
    assert not (out / "rv32_bench.json").exists()
    assert not (out / "rv32_bench.md").exists()
    assert not (out / "rv32_bench.html").exists()


# ───────────────────────────────────────────────────────────────────────────
# T5: labels cover all branch targets; size cross-check is strict
# ───────────────────────────────────────────────────────────────────────────

def test_parsed_labels_cover_all_branches(mini_compiled):
    asm = (mini_compiled["out"] / "_sv.s").read_text()
    data_offset = mini_compiled["info"]["data_offset"]
    labels = rv32_bench.parse_labels(asm, data_offset)

    assert labels.get(0) == "_start"
    assert "_done" in labels.values()
    assert mini_compiled["info"]["static_insns"] == data_offset // 4

    for _pc, op, operands in rv32_bench._iter_asm_lines(asm):
        if op in ("beq", "bne", "blt", "bge", "bltu", "bgeu", "j", "jal"):
            assert operands[-1] in labels or operands[-1].lstrip("+-").isdigit()

    with pytest.raises(rv32_bench.LabelParseError):
        rv32_bench.parse_labels(
            "_start:\n  bne t0, zero, +8\n  addi t0, t0, 1\n", 8,
        )
    with pytest.raises(rv32_bench.LabelParseError):
        rv32_bench.parse_labels("_start:\n  addi t0, t0, 1\n", 8)


# ───────────────────────────────────────────────────────────────────────────
# T6: LLVM RV64 output is flagged as incomparable
# ───────────────────────────────────────────────────────────────────────────

def test_llvm_riscv64_flagged_isa_mismatch():
    asm = (
        "  ld a0, 0(a1)\n"
        "  sd a0, 8(a1)\n"
        "  addiw a0, a0, 1\n"
        "  ret\n"
    )
    assert set(rv32_bench.detect_isa_mismatch(asm)) == {"ld", "sd", "addiw"}

    llvm = {
        "compile": {
            "status": "success",
            "reason": "rv64 mnemonics detected: addiw, ld, sd",
            "isa_detected": "riscv64", "isa_mismatch": True,
            "static_insns": 3, "static_source": "asm_scan", "elapsed_s": 0.0,
        },
        "dynamic": {
            "source": "unavailable", "simulator": "tinyfive",
            "completion": "not_run",
            "reason": "isa mismatch: rv64 mnemonics detected",
            "ops": None,
        },
    }
    report = rv32_bench.build_report(
        _fake_model(), _fake_env(), _fake_scratchv(), llvm,
    )
    assert report["llvm"]["compile"]["isa_detected"] == "riscv64"
    assert report["llvm"]["compile"]["isa_mismatch"] is True
    assert report["llvm"]["dynamic"]["source"] == "unavailable"
    assert report["comparison"]["dynamic_instruction_ratio"] is None
    assert "llvm" in report["comparison"]["incomparable_reason"]
    markdown = bench_report.render_markdown(report)
    assert "RV32IMF" not in markdown


# ───────────────────────────────────────────────────────────────────────────
# T6b: a failed LLVM compile is surfaced, not silently swallowed
# ───────────────────────────────────────────────────────────────────────────

def test_llvm_compile_failure_is_surfaced(tmp_path, mini_model, monkeypatch):
    failure_reason = "RuntimeError: no rv32 target available"

    def _failed(*_args, **_kwargs):
        return {
            "status": "failed", "reason": failure_reason,
            "isa_detected": None, "isa_mismatch": False,
            "static_insns": 0, "static_source": "asm_scan", "elapsed_s": 0.0,
        }

    monkeypatch.setattr(rv32_bench, "compile_llvm_rv32", _failed)
    out = tmp_path / "llvmfail"
    rc = rv32_bench.main([
        str(mini_model), "--quiet", "--output-dir", str(out),
        "--max-instructions", "500", "--timeout", "60",
        "--chunk-instructions", "4096",
    ])
    assert rc == rv32_bench.EXIT_OK

    report = json.loads((out / "rv32_bench.json").read_text())
    assert report["llvm"]["compile"]["status"] == "failed"
    assert any(
        "LLVM compilation failed" in w and failure_reason in w
        for w in report["warnings"]
    ), report["warnings"]
    assert failure_reason in report["llvm"]["dynamic"]["reason"]
    assert report["errors"] == []

    markdown = (out / "rv32_bench.md").read_text()
    assert failure_reason in markdown
    row = next(
        line for line in markdown.splitlines()
        if line.startswith("| static insns [asm_scan] |")
    )
    assert row.rstrip().endswith("| — |")


# ───────────────────────────────────────────────────────────────────────────
# T7: the provenance audit rejects static counts masquerading as dynamic
# ───────────────────────────────────────────────────────────────────────────

def test_audit_provenance_rejects_static_fallback():
    clean = _valid_fake_report()
    assert rv32_bench.audit_provenance(clean) == []

    static_masquerade = deepcopy(clean)
    static_masquerade["scratchv"]["dynamic"] = {
        "source": "simulated",
        "completion": "halted",
        "ops": {"total": 3841},
    }
    violations = rv32_bench.audit_provenance(static_masquerade)
    assert violations
    assert any("simulator" in v for v in violations)
    assert any("executed" in v for v in violations)

    ratio_from_truncated = deepcopy(clean)
    for side in ("scratchv", "llvm"):
        ratio_from_truncated[side]["dynamic"] = {
            "source": "simulated", "simulator": "tinyfive",
            "simulator_version": "1.0.0", "completion": "budget_exhausted",
            "limit": 1000, "executed": 1000, "memory_size_bytes": 268435456,
            "input_seed": 42,
            "ops": {"total": 1000, "load": 0, "store": 0, "mul": 0,
                    "add": 0, "madd": 0, "branch": 0},
        }
    ratio_from_truncated["scratchv"]["output"] = {
        "addr": 201326592, "elements": 36, "raw_hex": None,
        "q16_16": None, "completion": "budget_exhausted", "partial": True,
    }
    ratio_from_truncated["comparison"] = {
        "dynamic_instruction_ratio": 0.31, "incomparable_reason": None,
    }
    ratio_violations = rv32_bench.audit_provenance(ratio_from_truncated)
    assert any("incomplete/non-simulated" in v for v in ratio_violations)

    forged_halt = deepcopy(clean)
    forged_halt["scratchv"]["dynamic"].update(
        {"completion": "halted", "limit": 5000, "executed": 5000, "ops": {
            "total": 5000, "load": 0, "store": 0, "mul": 0,
            "add": 0, "madd": 0, "branch": 0,
        }},
    )
    assert any(
        "unverified halt" in v
        for v in rv32_bench.audit_provenance(forged_halt)
    )

    counters_disagree = deepcopy(clean)
    counters_disagree["scratchv"]["dynamic"].update(
        {"executed": 100, "ops": {
            "total": 9999, "load": 0, "store": 0, "mul": 0,
            "add": 0, "madd": 0, "branch": 0,
        }},
    )
    assert any(
        "ops.total" in v
        for v in rv32_bench.audit_provenance(counters_disagree)
    )

    budget_overrun = deepcopy(clean)
    budget_overrun["scratchv"]["dynamic"].update(
        {"completion": "halted", "limit": 50, "executed": 100, "ops": {
            "total": 100, "load": 0, "store": 0, "mul": 0,
            "add": 0, "madd": 0, "branch": 0,
        }},
    )
    assert any(
        "budget overrun" in v
        for v in rv32_bench.audit_provenance(budget_overrun)
    )

    unmasked_output = deepcopy(clean)
    unmasked_output["scratchv"]["dynamic"]["completion"] = "timeout"
    assert any(
        "output.partial" in v
        for v in rv32_bench.audit_provenance(unmasked_output)
    )


# ───────────────────────────────────────────────────────────────────────────
# T7b: output.q16_16 has one stable type (list) regardless of element count
# ───────────────────────────────────────────────────────────────────────────

def test_output_q16_16_is_always_a_list():
    class _StubMachine:
        available = True

        def __init__(self, words):
            self.words = words

        def read_mem_i32(self, addr):
            return self.words[addr // 4]

    single = rv32_bench._read_output(_StubMachine([-65536]), 0, 1)
    assert isinstance(single["q16_16"], list)
    assert single["q16_16"] == [-1.0]

    pair = rv32_bench._read_output(_StubMachine([65536, -131072]), 0, 2)
    assert pair["q16_16"] == [1.0, -2.0]


# ───────────────────────────────────────────────────────────────────────────
# T8: schema validation covers every required provenance key
# ───────────────────────────────────────────────────────────────────────────

def test_bench_report_schema_required_keys():
    report = _valid_fake_report()
    assert bench_report.validate_report_schema(report) == []

    empty_errors = bench_report.validate_report_schema({})
    joined = "\n".join(empty_errors)
    for key in (
        "schema_version", "model.sha256", "environment", "targets",
        "scratchv.compile", "scratchv.dynamic", "scratchv.output",
        "llvm.compile", "llvm.dynamic", "comparison",
    ):
        assert key in joined, f"{key} missing from {empty_errors}"

    no_static_source = deepcopy(report)
    del no_static_source["scratchv"]["compile"]["static_source"]
    assert any(
        "scratchv.compile.static_source" in e
        for e in bench_report.validate_report_schema(no_static_source)
    )

    no_sha = deepcopy(report)
    del no_sha["model"]["sha256"]
    assert any(
        "model.sha256" in e
        for e in bench_report.validate_report_schema(no_sha)
    )

    bad_ratio = deepcopy(report)
    bad_ratio["comparison"] = {"dynamic_instruction_ratio": 1.23}
    assert any(
        "comparison.incomparable_reason" in e
        for e in bench_report.validate_report_schema(bad_ratio)
    )

    no_output_partial = deepcopy(report)
    del no_output_partial["scratchv"]["output"]["partial"]
    assert any(
        "scratchv.output.partial" in e
        for e in bench_report.validate_report_schema(no_output_partial)
    )
