"""Retained-evidence contradictions and direct CLI bootstrap.

These reuse synthetic full-sized zero arrays only to exercise the auditor. They
are never a model execution or a W4 numerical acceptance result.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from probes.w4_qwen3_full import audit
from tests.test_w4_audit import evidence


@pytest.mark.parametrize("child", [
    {"passed": False, "status": "FAIL", "error": "RuntimeError: runtime cleanup failed",
     "cleanup_errors": ["qmp.close: OSError: injected close failure"]},
    {"passed": False, "status": "FAIL", "error": "Output validation failed"},
])
def test_retained_audit_rejects_failed_qemu_record_despite_valid_output(evidence, child):
    report_path = evidence / "report.json"
    run_path = evidence / "full_seed_0/qemu/run.json"
    original_report, original_run = report_path.read_text(), run_path.read_text()
    report, qemu = json.loads(original_report), json.loads(original_run)
    # A successful guest and exact numeric arrays cannot turn runtime cleanup
    # failure into a consistent all-gates PASS. Both copies match deliberately.
    qemu.update(child)
    report["cases"][0]["qemu"] = qemu
    try:
        report_path.write_text(json.dumps(report))
        run_path.write_text(json.dumps(qemu))
        with pytest.raises(ValueError, match="run|QEMU|qemu|completion|fail"):
            audit.audit(evidence)
    finally:
        report_path.write_text(original_report)
        run_path.write_text(original_run)


def test_audit_direct_script_cli_works_without_pythonpath():
    root = Path(__file__).resolve().parents[1]
    env = {key: value for key, value in os.environ.items() if key.upper() != "PYTHONPATH"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run([sys.executable, "-B", "probes/w4_qwen3_full/audit.py", "--help"],
                            cwd=root, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "--report-dir" in result.stdout and "--output-dir" in result.stdout
