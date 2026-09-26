"""Black-box tests for beautifier file and command-line interfaces."""

from __future__ import annotations

import subprocess
import sys

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_SCRIPT = PROJECT_ROOT / "scratchv" / "backend" / "asm_beautifier.py"
if not CLI_SCRIPT.is_file():
    CLI_SCRIPT = Path(__file__).with_name("asm_beautifier.py")
CLI_COMMAND = [sys.executable, str(CLI_SCRIPT)]


def run_cli(*args: str | Path) -> subprocess.CompletedProcess[str]:
    """Run the standalone script and capture its text streams."""

    return subprocess.run(
        [*CLI_COMMAND, *(str(arg) for arg in args)],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def test_cli_writes_to_stdout_without_output_path(tmp_path: Path) -> None:
    input_path = tmp_path / "input.s"
    input_path.write_text("add a0,a1,a2", encoding="utf-8")

    completed = run_cli(input_path)

    assert completed.returncode == 0
    assert "add       a0, a1, a2" in completed.stdout
    assert "# a0 = a1 + a2" in completed.stdout
    assert completed.stderr == ""


def test_cli_writes_to_output_file(tmp_path: Path) -> None:
    input_path = tmp_path / "input.s"
    output_path = tmp_path / "output.s"
    input_path.write_text("ret", encoding="utf-8")

    completed = run_cli(input_path, "--output", output_path)

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert "# return" in output_path.read_text(encoding="utf-8")


def test_cli_no_comments_disables_semantic_comments(tmp_path: Path) -> None:
    input_path = tmp_path / "input.s"
    input_path.write_text("add a0,a1,a2", encoding="utf-8")

    completed = run_cli(input_path, "--no-comments")

    assert completed.returncode == 0
    assert "a0 = a1 + a2" not in completed.stdout
    assert "add       a0, a1, a2" in completed.stdout


def test_cli_no_align_preserves_instruction_layout(tmp_path: Path) -> None:
    input_path = tmp_path / "input.s"
    input_path.write_text("add x1,x2,x3", encoding="utf-8")

    completed = run_cli(input_path, "--no-align")

    assert completed.returncode == 0
    assert completed.stdout.startswith("add x1,x2,x3  #")


def test_cli_abi_register_names_changes_only_comment(tmp_path: Path) -> None:
    input_path = tmp_path / "input.s"
    input_path.write_text("add x1,x2,x3", encoding="utf-8")

    completed = run_cli(input_path, "--abi-register-names")

    assert completed.returncode == 0
    assert "add       x1, x2, x3" in completed.stdout
    assert "# ra = sp + gp" in completed.stdout


def test_cli_missing_input_file_returns_one(tmp_path: Path) -> None:
    input_path = tmp_path / "missing.s"

    completed = run_cli(input_path)

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert str(input_path) in completed.stderr
    assert "Traceback" not in completed.stderr


def test_cli_invalid_utf8_returns_one(tmp_path: Path) -> None:
    input_path = tmp_path / "invalid.s"
    input_path.write_bytes(b"\xff")

    completed = run_cli(input_path)

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert str(input_path) in completed.stderr


def test_cli_unwritable_output_path_returns_one(tmp_path: Path) -> None:
    input_path = tmp_path / "input.s"
    output_path = tmp_path / "missing" / "output.s"
    input_path.write_text("ret", encoding="utf-8")

    completed = run_cli(input_path, "-o", output_path)

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert str(output_path) in completed.stderr
    assert not output_path.exists()


def test_cli_missing_required_argument_returns_two() -> None:
    completed = run_cli()

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "usage:" in completed.stderr
