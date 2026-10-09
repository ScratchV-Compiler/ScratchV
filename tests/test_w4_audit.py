"""Synthetic retained-evidence tests; these are never model accuracy results."""
import json
from pathlib import Path

import numpy as np
import pytest

from probes.w3_common import sha256_file
from probes.w3_qwen3_full.cases import input_cases
from probes.w4_qwen3_full import audit, run
from scratchv.runtime.riscv_external import FRAME, MAGIC
from scratchv.runtime.riscv_tensor import pack_inputs


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-retained-arrays")
    case = root / "full_seed_0"
    qemu = case / "qemu"
    qemu.mkdir(parents=True)
    name, length, feed = input_cases()[0]
    np.savez(case / "inputs.npz", **feed)
    (qemu / "inputs.bin").write_bytes(pack_inputs(audit.SPECS, feed))
    expected = np.lib.format.open_memmap(case / "ort.npy", mode="w+", dtype=np.float32, shape=audit.SHAPE)
    expected[:] = 0
    expected.flush()
    with (qemu / "output.bin").open("wb") as stream:
        stream.truncate(audit.NBYTES)
    actual = np.memmap(qemu / "output.bin", mode="r", dtype="<f4", shape=audit.SHAPE)
    numeric = run.compare_logits(actual, expected)
    (qemu / "uart.bin").write_bytes(FRAME.pack(MAGIC, 0, audit.NBYTES))
    qr = {"guest_completed": True, "exit_code": 0, "passed": True, "status": "PASS",
          "output_sha256": sha256_file(qemu / "output.bin"),
          "input_sha256": sha256_file(qemu / "inputs.bin")}
    (qemu / "run.json").write_text(json.dumps(qr))
    report = {"passed": True, "status": "PASS", "coverage": "full_seed_0",
              "full_qemu_forward_executed": True,
              "gates": {key: True for key in ("build:riscv-full", "smoke:qemu-full", "numeric:qemu-full-qwen3")},
              "cases": [{"name": name, "valid_length": length, "passed": True,
                         "input_sha256": sha256_file(case / "inputs.npz"),
                         "ort_sha256": sha256_file(case / "ort.npy"), "qemu": qr, "numeric": numeric}]}
    (root / "report.json").write_text(json.dumps(report))
    del expected, actual
    return root


def test_retained_arrays_recompute_success(evidence):
    assert audit.audit(evidence)["passed"]


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r.update(cases=[]), "coverage"),
    (lambda r: r.update(passed=False), "gates"),
    (lambda r: r["cases"][0]["numeric"].update(max_abs=False), "comparison"),
    (lambda r: r["cases"][0].update(valid_length=17), "input"),
    (lambda r: r["cases"][0].update(ort_sha256="0" * 64), "hash"),
])
def test_tampered_evidence_rejected(evidence, mutation, match):
    path = evidence / "report.json"
    original = path.read_text()
    report = json.loads(original)
    mutation(report)
    try:
        path.write_text(json.dumps(report))
        with pytest.raises(ValueError, match=match):
            audit.audit(evidence)
    finally:
        path.write_text(original)


def test_absent_evidence_writes_failure(tmp_path):
    out = tmp_path / "audit"
    assert audit.main(["--report-dir", str(tmp_path / "absent"), "--output-dir", str(out)]) == 1
    assert json.loads((out / "report.json").read_text())["passed"] is False
