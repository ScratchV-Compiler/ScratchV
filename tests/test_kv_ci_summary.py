"""CI must display measured KV evidence without promoting stale/partial runs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from scripts.summarize_kv_ci import render
from scripts.summarize_w2_ci import render as render_acceptance

ROOT = Path(__file__).resolve().parents[1]


def write(directory, name, value):
    path = directory/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    for variable in ('GITHUB_SHA', 'GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT'):
        monkeypatch.delenv(variable, raising=False)
    layout = dict(layers=2, kv_heads=2, head_dim=16, vocab_size=128, capacity=256, pad_token_id=0)
    row = lambda name, error: dict(case=name, passed=True, max_abs_by_output=[error, 2*error, 0, 0, 0])
    write(tmp_path, 'w6-kv-status.json', dict(schema='scratchv.kv-ci-outcome.v1', outcome='success',
                                           commit='abc', run_id='10', run_attempt='1'))
    write(tmp_path, 'w6-kv-cache/validation.json', dict(schema='scratchv.kv-export.v1',
        passed=True, checkpoint='seeded-small', layout=layout, seconds=7.5, reference=[row('prefill-1', 1e-6)],
        compiled=[row('ir-prefill-1', 2e-6), row('host-prefill-1', 3e-6),
                  row('qemu-1-prefill-1', 4e-6),
                  dict(case='qemu-1-greedy-generation', passed=True, generated_token_ids=[3]*4)],
        memory=dict(prefill_workspace_bytes=1000, decode_workspace_bytes=200, cache_bank_bytes=100)))
    write(tmp_path, 'w6-kv-cache/qemu-1/session.json', dict(schema='scratchv.kv-session.v1',
        passed=True, state='CLOSED', qemu_reaped=True, cores=1, cache_layout=layout,
        startup_s=0.5, sampled_peak_qemu_rss_bytes=5000,
        requests=[dict(kind=2, passed=True, forward_s=1.5),
                  dict(kind=3, passed=True, forward_s=0.2), dict(kind=3, passed=True, forward_s=0.4)]))
    write(tmp_path, 'w6-kv-cache/build-1/build.json', dict(cores=1, cache_layout=layout, compile_seconds=2.5))
    (tmp_path/'w6-kv-tests.xml').write_text('<testsuites><testsuite>'
        '<testcase name="reset"/><testcase name="timeout"/></testsuite></testsuites>')
    return tmp_path


def change(directory, name, **updates):
    path = directory/name
    value = json.loads(path.read_text())
    value.update(updates)
    write(directory, name, value)


def test_measured_numbers_and_scope_are_visible_in_acceptance(evidence):
    text = render_acceptance(evidence)
    kv = text.split('## KV cache', 1)[1]
    assert '**PASS**（CI 步骤：success）' in kv
    assert '通过 2，失败 0，错误 0，跳过 0' in kv
    assert '| QEMU / ORT | PASS / 1 | 4.000000000e-06 | 8.000000000e-06 |' in kv
    assert '7.500000' in kv and '2.500000' in kv
    assert 'Decode 合计（2 次，平均 0.300000 s） | 0.600000' in kv
    assert '| 两块缓存区 | 200 |' in kv
    assert '| 采样 QEMU 进程峰值 RSS | 5,000 |' in kv
    assert '进程回收：完成' in kv
    assert '完整 Qwen3-0.6B KV：**NOT RUN' in kv


@pytest.mark.parametrize('outcome', ['failure', 'cancelled', 'skipped'])
def test_successful_old_reports_do_not_override_actual_outcome(evidence, outcome):
    change(evidence, 'w6-kv-status.json', outcome=outcome)
    text = render(evidence)
    assert '**PASS**' not in text
    assert {'failure': '**FAIL**', 'cancelled': '**CANCELLED**', 'skipped': '**NOT RUN'}[outcome] in text
    if outcome == 'skipped':
        assert '4.000000000e-06' not in text


@pytest.mark.parametrize('name', ['w6-kv-status.json', 'w6-kv-tests.xml',
    'w6-kv-cache/validation.json', 'w6-kv-cache/qemu-1/session.json', 'w6-kv-cache/build-1/build.json'])
def test_missing_evidence_cannot_pass(evidence, name):
    (evidence/name).unlink()
    assert '**INCOMPLETE' in render(evidence)
    assert '**PASS**' not in render(evidence)


@pytest.mark.parametrize('name', ['w6-kv-status.json', 'w6-kv-tests.xml',
    'w6-kv-cache/validation.json', 'w6-kv-cache/qemu-1/session.json'])
@pytest.mark.parametrize('invalid', ['{', '[]', 'null'])
def test_malformed_artifacts_are_reported_instead_of_crashing(evidence, name, invalid):
    (evidence/name).write_text(invalid)
    assert '**INCOMPLETE' in render(evidence)


@pytest.mark.parametrize('tag', ['failure', 'error'])
def test_junit_failures_cannot_hide_behind_passing_model(evidence, tag):
    (evidence/'w6-kv-tests.xml').write_text(
        f'<testsuite><testcase><{tag}/></testcase></testsuite>')
    assert '**FAIL**' in render(evidence)


@pytest.mark.parametrize('field', ['passed', 'qemu_reaped'])
def test_incomplete_session_cannot_pass(evidence, field):
    change(evidence, 'w6-kv-cache/qemu-1/session.json', **{field: False})
    assert '**FAIL**' in render(evidence)


@pytest.mark.parametrize('rows', [[], [{'case': 'x', 'passed': True, 'max_abs_by_output': [0]}],
    [{'case': 'x', 'passed': True, 'max_abs_by_output': [float('nan')]*5}],
    [{'case': 'x', 'passed': True, 'max_abs_by_output': [10**500]*5}]])
def test_missing_or_nonfinite_comparisons_cannot_pass(evidence, rows):
    change(evidence, 'w6-kv-cache/validation.json', reference=rows)
    assert '**INCOMPLETE' in render(evidence)


def test_numeric_failure_is_displayed_even_when_top_level_flag_is_true(evidence):
    rows = [dict(case='prefill-1', passed=False, max_abs_by_output=[0.1]*5)]
    change(evidence, 'w6-kv-cache/validation.json', reference=rows)
    assert '**FAIL**' in render(evidence)
    assert '| 参考链（PyTorch / ORT / 完整重算） | FAIL / 1 |' in render(evidence)


@pytest.mark.parametrize('field,variable', [('commit', 'GITHUB_SHA'), ('run_id', 'GITHUB_RUN_ID'),
                                          ('run_attempt', 'GITHUB_RUN_ATTEMPT')])
def test_another_ci_run_is_not_accepted_as_current_evidence(evidence, monkeypatch, field, variable):
    monkeypatch.setenv(variable, 'different')
    assert '不属于本次提交或运行' in render(evidence)
    assert '**PASS**' not in render(evidence)


def test_acceptance_needs_no_installed_python_dependencies(evidence):
    result = subprocess.run([sys.executable, '-S', str(ROOT/'scripts/summarize_w2_ci.py'),
                             '--report-dir', str(evidence)], cwd=evidence,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert '## KV cache' in result.stdout


def test_workflow_preserves_and_downloads_this_runs_kv_evidence():
    workflow = yaml.load((ROOT/'.github/workflows/llm-deploy.yml').read_text(), Loader=yaml.BaseLoader)
    steps = workflow['jobs']['llm-deploy']['steps']
    probe = next(step for step in steps if step.get('id') == 'kv_cache')
    assert '--qemu --cores 1' in probe['run']
    status = next(step for step in steps if step.get('name') == 'Record KV cache execution outcome')
    assert status['if'] == 'always()'
    assert status['env']['KV_OUTCOME'] == '${{ steps.kv_cache.outcome }}'
    upload = next(step for step in steps if step.get('with', {}).get('name') == 'w6-kv-cache')
    assert upload['if'] == 'always()' and 'output/w6-kv-status.json' in upload['with']['path']
    acceptance = workflow['jobs']['w2-acceptance']['steps']
    download = next(step for step in acceptance if step.get('with', {}).get('name') == 'w6-kv-cache')
    assert download['if'] == 'always()' and download['with']['path'] == 'output'
    assert download['continue-on-error'] == 'true'
    assert any(step.get('run') == 'python3 scripts/summarize_w2_ci.py --report-dir output' for step in acceptance)


@pytest.mark.parametrize('outcome', ['success', 'failure', 'cancelled', 'skipped'])
def test_workflow_records_actual_outcome_even_after_failed_or_skipped_gate(tmp_path, outcome):
    workflow = yaml.load((ROOT/'.github/workflows/llm-deploy.yml').read_text(), Loader=yaml.BaseLoader)
    step = next(step for step in workflow['jobs']['llm-deploy']['steps']
                if step.get('name') == 'Record KV cache execution outcome')
    bash = shutil.which('bash')
    if not bash:
        pytest.skip('Bash required to execute the workflow command')
    env = dict(os.environ, KV_OUTCOME=outcome, GITHUB_SHA='abc', GITHUB_RUN_ID='42',
               GITHUB_RUN_ATTEMPT='3')
    result = subprocess.run([bash, '-eu', '-c', step['run']], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path/'output/w6-kv-status.json').read_text()) == dict(
        schema='scratchv.kv-ci-outcome.v1', outcome=outcome, commit='abc', run_id='42', run_attempt='3')
