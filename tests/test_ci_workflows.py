"""Keep one PR validation run without dropping gates or running full models."""
import json
from pathlib import Path

import yaml
import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / '.github/workflows'


def workflow(name):
    return yaml.load((WORKFLOWS / name).read_text(encoding='utf-8'), Loader=yaml.BaseLoader)


def test_only_ci_and_privileged_ai_review_automatically_handle_prs():
    triggers = {}
    for path in WORKFLOWS.glob('*.yml'):
        data = workflow(path.name)
        events = set(data['on'])
        if events & {'pull_request', 'pull_request_target'}:
            triggers[path.name] = events & {'pull_request', 'pull_request_target'}
    assert triggers == {'ci.yml': {'pull_request'}, 'ai-code-review.yml': {'pull_request_target'}}
    assert workflow('ci.yml')['permissions'] == {'contents': 'read'}
    assert workflow('ai-code-review.yml')['permissions']['pull-requests'] == 'write'


def test_reusable_jobs_share_the_callers_run_and_same_commit():
    jobs = workflow('ci.yml')['jobs']
    for name, filename in [('llm', 'llm-deploy.yml'), ('topic06', 'topic06-benchmark.yml')]:
        assert jobs[name]['uses'] == './.github/workflows/' + filename
        assert 'workflow_call' in workflow(filename)['on']
        assert not ({'pull_request', 'push'} & set(workflow(filename)['on']))
    assert jobs['llm']['needs'] == 'scope'
    assert jobs['llm']['if'] == "github.event_name == 'pull_request' && needs.scope.outputs.llm == 'true'"
    assert workflow('llm-deploy.yml')['jobs']['full-qwen3-onnx']['if'] == "github.event_name != 'pull_request'"
    assert 'workflow_dispatch' in workflow('llm-deploy.yml')['on']


def test_scope_and_acceptance_are_fail_closed_and_keep_all_required_dependencies():
    data = workflow('ci.yml')
    jobs = data['jobs']
    assert jobs['scope']['outputs'] == {name: '${{ steps.scope.outputs.' + name + ' }}'
                                       for name in ('llm', 'w4')}
    checkout = jobs['scope']['steps'][0]
    assert checkout['with']['fetch-depth'] == '0'
    assert checkout['with']['persist-credentials'] == 'false'
    gate = jobs['acceptance']
    assert gate['if'] == 'always()'
    assert set(gate['needs']) == {'scope', 'test', 'benchmark', 'topic06', 'llm'} | ({'w4'} if 'w4' in jobs else set())
    assert gate['steps'][-1]['env']['CI_JOB_RESULTS'] == '${{ toJSON(needs) }}'
    assert gate['steps'][-1]['run'] == 'python3 scripts/summarize_ci.py >> "$GITHUB_STEP_SUMMARY"'
    assert data['concurrency']['cancel-in-progress'] == "${{ github.event_name == 'pull_request' }}"
    assert 'github.run_id' in data['concurrency']['group']


def test_scopes_keep_required_frontend_coverage_and_no_dead_scope():
    scopes = json.loads((ROOT / 'scripts/ci_scopes.json').read_text(encoding='utf-8'))
    # Every configured scope must name a job in ci.yml, so a removed check
    # cannot leave a selector behind that nothing consumes.
    assert set(scopes) == {'llm', 'w4'}
    assert {'scratchv/**', 'probes/**', 'requirements/**'} <= set(scopes['llm'])
    assert bool(scopes['w4']) == (WORKFLOWS / 'w4-full-numeric.yml').exists()


def test_qwen_assets_select_acceptance_and_jobs_fetch_only_required_lfs_files():
    scopes = json.loads((ROOT / 'scripts/ci_scopes.json').read_text(encoding='utf-8'))
    assert 'assets/qwen3-0.6b/**' in scopes['llm']
    jobs = workflow('llm-deploy.yml')['jobs']
    for name in ('llm-deploy', 'full-qwen3-onnx', 'full-qwen3-frontend'):
        steps = jobs[name]['steps']
        assert steps[0]['with']['lfs'] == 'false'
        cache = next(step for step in steps if step.get('uses') == 'actions/cache@v4')
        assert cache['with']['path'] == '.git/lfs'
        pull = next(step['run'] for step in steps if 'git lfs pull' in step.get('run', ''))
        assert pull.index('git lfs install --local') < pull.index('git lfs pull')
        expected = 'tokenizer' if name == 'llm-deploy' else 'onnx'
        assert f'assets/qwen3-0.6b/{expected}/*' in pull
        if name == 'llm-deploy':
            assert 'probes/w1_tiny_transformer/out/tiny_transformer_2l.onnx' in pull
            assert 'assets/qwen3-0.6b/onnx' not in pull
        commands = '\n'.join(step.get('run', '') for step in steps)
        assert '--mode download' not in commands
        assert ('--mode verify' in commands) == (name == 'full-qwen3-onnx')


def test_the_test_job_writes_no_step_summary():
    """The step outcome and the test-reports artifact already say all of it."""
    steps = workflow('ci.yml')['jobs']['test']['steps']
    assert not [step for step in steps if 'GITHUB_STEP_SUMMARY' in step.get('run', '')]


def test_the_w1_w2_job_writes_no_step_summary():
    """Its gates report in the job log; only full-model validation adds a page."""
    data = workflow('llm-deploy.yml')
    writers = {name for name, job in data['jobs'].items()
               for step in job.get('steps', []) if 'GITHUB_STEP_SUMMARY' in step.get('run', '')}
    assert writers == {'full-qwen3-onnx', 'w3-execution-summary', 'w4-execution-summary'}


def test_linux_reproduction_check_runs_without_a_path_filter():
    """A docs-only PR selects no scope, so this check cannot be scoped."""
    steps = workflow('ci.yml')['jobs']['test']['steps']
    step = next(step for step in steps if step.get('name') == 'Check public instructions and Bash syntax')
    assert step['run'] == '$PY scripts/check_linux_repro_docs.py --check-bash'
    assert 'if' not in step
    assert not (WORKFLOWS / 'linux-reproduction-docs.yml').exists()


def test_long_compiler_and_topic_reports_are_folded_without_dropping_artifacts():
    for filename, job, step_name in [('ci.yml', 'benchmark', 'Write job summary'),
                                    ('topic06-benchmark.yml', 'topic06-benchmark', 'Publish benchmark summary')]:
        steps = workflow(filename)['jobs'][job]['steps']
        summary = next(step for step in steps if step.get('name') == step_name)
        assert summary['if'] == 'always()'
        assert summary['run'].count('<details>') == summary['run'].count('</details>') == 1
        assert any('actions/upload-artifact@' in step.get('uses', '') for step in steps)


def test_cleanup_stress_is_scheduled_or_manual_but_ordinary_test_remains():
    steps = workflow('llm-deploy.yml')['jobs']['llm-deploy']['steps']
    stress = next(step for step in steps if step.get('name') == 'Repeated Linux timeout cleanup regression')
    assert stress['if'] == "github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'"
    ordinary = next(step for step in steps if step.get('name') == 'Tensor compiler and runtime regressions')
    assert 'tests/test_riscv_tensor_runtime.py' in ordinary['run']
    assert ordinary['if'] == (
        "github.event_name != 'pull_request' || !inputs.ci_covers_generic_regressions")
    assert 'set -euo pipefail' in stress['run'] and 'seq 1 20' in stress['run']


def test_same_job_dedup_retains_full_suite_exclusions_and_distinct_smokes():
    steps = workflow('ci.yml')['jobs']['test']['steps']
    pytest_steps = [step for step in steps if '-m pytest' in step.get('run', '')]
    assert len(pytest_steps) == 2
    suite, excluded = pytest_steps
    assert 'pytest tests/' in suite['run'] and '--ignore=tests/test_simulator.py' in suite['run']
    assert suite['env']['SCRATCHV_REQUIRE_RISCV_EXECUTION'] == '1'
    assert 'tests/test_simulator.py::TestStubProfiledMachine' in excluded['run']
    assert 'tests/test_simulator.py::TestRealProfiledMachine::test_lw_preserves_all_four_bytes' in excluded['run']
    commands = '\n'.join(step.get('run', '') for step in steps)
    assert '-m scratchv.backend.asm_peephole' in commands
    assert '--backend ir --verify-ir' in commands
    deploy = workflow('llm-deploy.yml')['jobs']['llm-deploy']['steps']
    generic = next(step['run'] for step in deploy if step.get('name') == 'Tensor compiler and runtime regressions')
    pinned = next(step['run'] for step in deploy if step.get('name') == 'Runtime unit regressions in pinned environment')
    for path in ('tests/test_llm_inputs.py', 'tests/test_w2_runtime_gate.py',
                 'tests/test_w2_runtime_metadata.py', 'tests/test_w2_runtime_reporting.py'):
        assert path not in generic and path in pinned
    assert any(step.get('name') == 'Explicit Qwen3 artifact optimization integration' for step in deploy)


def test_cross_workflow_dedup_is_opt_in_pr_only_and_owned_by_required_test_job():
    ci = workflow('ci.yml')['jobs']
    deploy = workflow('llm-deploy.yml')
    option = deploy['on']['workflow_call']['inputs']['ci_covers_generic_regressions']
    assert option['type'] == 'boolean'
    assert option['required'] == 'false' and option['default'] == 'false'
    assert ci['llm']['with'] == {'ci_covers_generic_regressions': 'true'}
    assert ci['llm']['needs'] == 'scope'  # Keep deployment and generic tests parallel.
    generic = next(step for step in deploy['jobs']['llm-deploy']['steps']
                   if step.get('name') == 'Tensor compiler and runtime regressions')
    # Non-PR events always run; an independent PR caller defaults to running.
    assert generic['if'] == (
        "github.event_name != 'pull_request' || !inputs.ci_covers_generic_regressions")
    for path in WORKFLOWS.glob('*.yml'):
        for job in workflow(path.name)['jobs'].values():
            if job.get('uses') == './.github/workflows/llm-deploy.yml':
                delegated = job.get('with', {}).get('ci_covers_generic_regressions', 'false')
                assert delegated == ('true' if path.name == 'ci.yml' else 'false')
    owner = ci['test']
    assert 'if' not in owner and 'continue-on-error' not in owner
    suite = next(step for step in owner['steps'] if step.get('name') == 'Run all topic tests')
    assert 'if' not in suite and 'continue-on-error' not in suite
    assert 'pytest tests/' in suite['run']
    assert suite['env']['SCRATCHV_REQUIRE_RISCV_EXECUTION'] == '1'
    gate = ci['acceptance']
    assert gate['if'] == 'always()' and 'test' in gate['needs']
    assert 'continue-on-error' not in gate
    # test_ci_routing exercises the existing final gate with test failure,
    # cancellation, skipping and missing results; none may count as success.

def test_delegated_pr_keeps_real_rv64_musl_execution_in_system_qemu_environment():
    deploy = workflow('llm-deploy.yml')['jobs']['llm-deploy']['steps']
    actual = next(step for step in deploy if step.get('name') == 'Delegated PR RV64 musl execution regression')
    assert actual['if'] == "github.event_name == 'pull_request' && inputs.ci_covers_generic_regressions"
    selector = ('tests/test_riscv_tensor_runtime.py::'
                'test_real_rv64_musl_math_binary_inputs_and_guest_failure')
    assert selector in actual['run'] and 'set -euo pipefail' in actual['run']
    assert 'continue-on-error' not in actual
    tools = next(step for step in deploy if step.get('name') == 'Install RISC-V toolchain and QEMU')
    dependencies = next(step for step in deploy if step.get('name') == 'Install Python dependencies')
    assert 'qemu-system-riscv64' in tools['run'] and 'qemu-system-misc' in tools['run']
    assert '-e ".[all,llm-deploy]"' in dependencies['run']
    # The inverse condition on generic coverage avoids running this case twice
    # for independent reusable, schedule or manual entry points.
    generic = next(step for step in deploy if step.get('name') == 'Tensor compiler and runtime regressions')
    assert generic['if'] == "github.event_name != 'pull_request' || !inputs.ci_covers_generic_regressions"


@pytest.mark.parametrize('outcome', ['success', 'skipped', 'failure', 'error', 'missing', 'wrong-name', 'duplicate'])
def test_delegated_rv64_junit_check_requires_one_real_success(tmp_path, outcome):
    import subprocess
    import sys
    import xml.etree.ElementTree as ET

    steps = workflow('llm-deploy.yml')['jobs']['llm-deploy']['steps']
    step = next(step for step in steps if step.get('name') == 'Delegated PR RV64 musl execution regression')
    # Execute the actual short checker from the workflow on synthetic JUnit;
    # this does not invoke a compiler, QEMU or the numeric test itself.
    checker = step['run'].split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    suite = ET.Element('testsuite')
    if outcome != 'missing':
        name = ('wrong_test' if outcome == 'wrong-name' else
                'test_real_rv64_musl_math_binary_inputs_and_guest_failure')
        case = ET.SubElement(suite, 'testcase', name=name)
        if outcome in ('skipped', 'failure', 'error'):
            ET.SubElement(case, outcome)
        if outcome == 'duplicate':
            ET.SubElement(suite, 'testcase', name=name)
    target = tmp_path / 'output/riscv-math-regression/tests.xml'
    target.parent.mkdir(parents=True)
    ET.ElementTree(suite).write(target)
    result = subprocess.run([sys.executable, '-c', checker], cwd=tmp_path,
                            capture_output=True, text=True)
    assert (result.returncode == 0) is (outcome == 'success')
