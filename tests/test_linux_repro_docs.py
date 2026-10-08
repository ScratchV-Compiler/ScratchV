"""Check rejection of unusable public reproduction commands, not doc wording."""
import shutil

import pytest

from scripts import check_linux_repro_docs as docs


@pytest.mark.parametrize("content", [
    "```powershell\npython run.py\n```\n",
    "~~~pwsh\npython run.py\n~~~\n",
    "Use D:/private/models for the checkpoint\n",
    "python C:\\private\\run.py\n",
    "$env:OMP_NUM_THREADS = '1'\n",
    "$LASTEXITCODE\n",
    "venv/Scripts/python.exe\n",
    "```bash\nzig.exe version\n```\n",
    "```bash\necho unfinished\n",
])
def test_rejects_non_linux_or_broken_examples(tmp_path, content):
    path = tmp_path / "README.md"
    path.write_text(content, encoding="utf-8")
    assert docs.inspect_document(path)[0]


def test_linux_examples_and_truthful_history_are_allowed(tmp_path):
    path = tmp_path / "README.md"
    path.write_text('Historical Windows results are not Linux evidence.\n'
                    'https://example.com/source\n'
                    '```bash\nset -euo pipefail\n'
                    '"$SCRATCHV_PYTHON" run.py --model-dir /srv/models/qwen3\n```\n', encoding="utf-8")
    assert docs.inspect_document(path) == ([], 1)


def test_required_bash_cannot_silently_skip(monkeypatch):
    monkeypatch.setattr(docs.shutil, "which", lambda _: None)
    with pytest.raises(SystemExit) as exc:
        docs.main(["--check-bash"])
    assert exc.value.code == 1


def test_empty_input_is_not_success(tmp_path):
    with pytest.raises(SystemExit) as exc:
        docs.main(["--root", str(tmp_path)])
    assert exc.value.code == 1


def test_inventory_covers_shared_guides_and_benchmarks(tmp_path):
    names = ("README.md", "CONTRIBUTING.md", "docs/guide/setup.md",
             "docs/topics/topic/reproduce.md", "probes/w4/example.md", "benchmarks/sample/README.md")
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Linux reproduction\n", encoding="utf-8")
    assert {p.relative_to(tmp_path).as_posix() for p in docs.document_paths(tmp_path)} == set(names)


def test_actual_bash_rejects_bad_syntax_without_executing(tmp_path):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("Bash is required; Linux CI exercises this case")
    path = tmp_path / "README.md"
    marker = tmp_path / "must-not-exist"
    path.write_text(f'```bash\ntouch "{marker.as_posix()}"\nif then\n```\n', encoding="utf-8")
    problems, count = docs.inspect_document(path, bash=bash)
    assert count == 1 and any("invalid Bash" in item for item in problems)
    assert not marker.exists()
