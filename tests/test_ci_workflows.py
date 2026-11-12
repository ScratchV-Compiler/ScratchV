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
    assert workflow('llm-deploy.yml')['jobs']['full-qwen3-onnx']['if'] == "github.event_name != 'pull_request' && !inputs.run_w5_validation"
    assert 'workflow_dispatch' in workflow('llm-deploy.yml')['on']


def test_scope_and_acceptance_are_fail_closed_and_keep_all_required_dependencies():
    data = workflow('ci.yml')
    jobs = data['jobs']
    assert jobs['scope']['outputs'] == {name: '${{ steps.scope.outputs.' + name + ' }}'
                                       for name in ('llm', 'w4', 'docs_only')}
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
    assert writers == {'full-qwen3-onnx', 'full-validation-summary'}


@pytest.mark.parametrize("event,requested,changes,statuses", [
    ("pull_request", (), {}, ("NOT RUN", "NOT RUN", "NOT RUN")),
    ("schedule", ("W4",), {}, ("NOT RUN", "NOT RUN", "NOT RUN")),
    ("workflow_dispatch", ("W3", "W4"), {"w3-validation": "success", "w4-validation": "success"},
     ("PASS", "PASS", "NOT RUN")),
    ("workflow_dispatch", ("W3",), {}, ("FAIL", "NOT RUN", "NOT RUN")),
    ("workflow_dispatch", ("W4",), {"w4-validation": "failure"}, ("NOT RUN", "FAIL", "NOT RUN")),
    ("workflow_dispatch", ("W4",), {"w4-validation": "cancelled"}, ("NOT RUN", "FAIL", "NOT RUN")),
    ("pull_request", (), {"w4-validation": "success"}, ("NOT RUN", "FAIL", "NOT RUN")),
    ("workflow_dispatch", ("W5",), {"w5-validation": "success"}, ("NOT RUN", "NOT RUN", "PASS")),
    ("workflow_dispatch", ("W5",), {}, ("NOT RUN", "NOT RUN", "FAIL")),
    ("workflow_dispatch", ("W5",), {"w5-validation": "cancelled"}, ("NOT RUN", "NOT RUN", "FAIL")),
    ("workflow_dispatch", ("W5",), {"validation-selection": "failure"}, ("FAIL", "FAIL", "FAIL")),
    ("workflow_dispatch", ("W5",), {"w5-validation": "success", "llm-deploy": "success"},
     ("NOT RUN", "NOT RUN", "FAIL")),
    ("workflow_dispatch", ("W5",), {"w5-validation": "success", "full-qwen3-onnx": "failure"},
     ("NOT RUN", "NOT RUN", "FAIL")),
    ("workflow_dispatch", ("W5",), {"w5-validation": "success", "full-qwen3-frontend": "cancelled"},
     ("NOT RUN", "NOT RUN", "FAIL")),
    ("workflow_dispatch", ("W5",), {"w5-validation": "success", "w3-validation": "success"},
     ("FAIL", "NOT RUN", "FAIL")),
    ("workflow_dispatch", ("W5",), {"w5-validation": "success", "w4-validation": "success"},
     ("NOT RUN", "FAIL", "FAIL")),
    ("pull_request", (), {"w3-validation": None}, ("FAIL", "NOT RUN", "NOT RUN")),
])
def test_full_validation_summary(event, requested, changes, statuses, tmp_path):
    import os
    import subprocess

    job = workflow("llm-deploy.yml")["jobs"]["full-validation-summary"]
    assert job["if"] == "always()" and len(job["steps"]) == 1
    assert set(job["needs"]) == {
        "validation-selection", "llm-deploy", "full-qwen3-onnx", "full-qwen3-frontend",
        "w3-validation", "w4-validation", "w5-validation",
    }
    step = job["steps"][0]
    stages = ("W3", "W4", "W5")
    assert step["env"] == {"JOB_RESULTS": "${{ toJSON(needs) }}",
                          **{f"{stage}_REQUESTED": "${{ inputs.run_" + stage.lower() + "_validation }}"
                             for stage in stages}}
    outcomes = dict.fromkeys(job["needs"], "skipped")
    outcomes.update({"validation-selection": "success"}, **changes)
    results = {job: {"result": outcome} for job, outcome in outcomes.items() if outcome is not None}
    summary = tmp_path / "summary.md"
    env = dict(os.environ, GITHUB_EVENT_NAME=event, JOB_RESULTS=json.dumps(results),
               GITHUB_STEP_SUMMARY=str(summary))
    env.update({f"{stage}_REQUESTED": str(stage in requested).lower() for stage in stages})
    run = subprocess.run(["bash", "-e", "-c", step["run"]], cwd=tmp_path, env=env,
                         capture_output=True, text=True)
    assert (run.returncode == 0) == ("FAIL" not in statuses), run.stderr
    table = summary.read_text().splitlines()
    for stage, expected in zip(stages, statuses):
        row, = [line for line in table if line.startswith(f"| {stage} |")]
        actual = results.get(stage.lower() + "-validation", {}).get("result", "missing")
        assert f"| {actual} | **{expected}** |" in row
        assert row.endswith("— |") == (actual in ("skipped", "missing"))


def test_linux_reproduction_check_runs_without_a_path_filter():
    """Even a PR containing only reader guides has to pass the docs checks."""
    jobs = workflow('ci.yml')['jobs']
    steps = jobs['scope']['steps']
    assert 'if' not in jobs['scope']
    for name in ('test', 'benchmark', 'topic06'):
        assert jobs[name]['needs'] == 'scope'
        assert jobs[name]['if'] == "needs.scope.outputs.docs_only == 'false'"
    links = next(step for step in steps if step.get('name') == 'Check documentation links')
    assert links['if'] == "steps.scope.outputs.docs_only == 'true'"
    assert links['run'] == 'python3 scripts/check_docs_links.py'
    step = next(step for step in steps if step.get('name') == 'Check public instructions and Bash syntax')
    assert step['run'] == 'python3 scripts/check_linux_repro_docs.py --check-bash'
    assert 'if' not in step
    assert 'continue-on-error' not in step and 'continue-on-error' not in jobs['scope']
    deploy_steps = workflow('llm-deploy.yml')['jobs']['llm-deploy']['steps']
    repeated = next(item for item in deploy_steps if item.get('name') == 'docs:linux-reproduction')
    assert repeated['run'] == step['run']
    # Independent PR callers default to running it; manual/scheduled calls
    # always run it. Only CI's explicit delegation omits the second execution.
    assert repeated['if'] == (
        "github.event_name != 'pull_request' || !inputs.ci_covers_generic_regressions")
    assert jobs['llm']['needs'] == 'scope'
    assert 'scope' in jobs['acceptance']['needs']
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
    w4 = workflow('w4-full-numeric.yml')
    option = w4['on']['workflow_call']['inputs']['ci_covers_generic_regressions']
    assert option['type'] == 'boolean'
    assert option['required'] == 'false' and option['default'] == 'false'
    assert ci['w4']['with']['ci_covers_generic_regressions'] == 'true'
    for path in WORKFLOWS.glob('*.yml'):
        for job in workflow(path.name)['jobs'].values():
            if job.get('uses') == './.github/workflows/w4-full-numeric.yml':
                delegated = job.get('with', {}).get('ci_covers_generic_regressions', 'false')
                assert delegated == ('true' if path.name == 'ci.yml' else 'false')
    generic = next(step for step in deploy['jobs']['llm-deploy']['steps']
                   if step.get('name') == 'Tensor compiler and runtime regressions')
    # Non-PR events always run; an independent PR caller defaults to running.
    assert generic['if'] == (
        "github.event_name != 'pull_request' || !inputs.ci_covers_generic_regressions")
    frontend = next(step for step in deploy['jobs']['llm-deploy']['steps']
                    if step.get('name') == 'unit:frontend-ops')
    assert frontend['if'] == generic['if']
    assert 'tests/test_qwen3_frontend_patterns.py' in frontend['run']
    for path in WORKFLOWS.glob('*.yml'):
        for job in workflow(path.name)['jobs'].values():
            if job.get('uses') == './.github/workflows/llm-deploy.yml':
                delegated = job.get('with', {}).get('ci_covers_generic_regressions', 'false')
                assert delegated == ('true' if path.name == 'ci.yml' else 'false')
    owner = ci['test']
    assert owner['if'] == "needs.scope.outputs.docs_only == 'false'" and 'continue-on-error' not in owner
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


@pytest.mark.parametrize("suffix", ["None", "Give brief answers."])
@pytest.mark.parametrize("outcome", ["pass", "skipped", "failure", "error", "missing", "duplicate"])
def test_w5_official_three_round_template_must_execute(tmp_path, suffix, outcome):
    import subprocess
    import sys
    import xml.etree.ElementTree as ET

    steps = workflow("llm-deploy.yml")["jobs"]["llm-deploy"]["steps"]
    actual = next(step for step in steps if step.get("name") == "W5 official three-round conversation template")
    assert "tests/test_conversation.py::test_official_three_round_rendering_matches_transformers" in actual["run"]
    assert "if" not in actual and "continue-on-error" not in actual
    checker = actual["run"].split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
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
    directory = tmp_path / "output/w5-multiturn"
    directory.mkdir(parents=True)
    ET.ElementTree(suite).write(directory / "tests.xml")
    result = subprocess.run([sys.executable, "-c", checker], cwd=tmp_path, capture_output=True, text=True)
    assert (result.returncode == 0) is (outcome == "pass"), result.stderr


@pytest.mark.parametrize('event,delegated,omit_generic', [
    ('pull_request', 'true', True),
    ('pull_request', 'false', False),
    ('pull_request', None, False),
    ('workflow_dispatch', 'true', False),
    ('schedule', 'true', False),
    ('push', 'true', False),
])
def test_delegation_only_removes_pure_tests_from_actual_commands(
        tmp_path, event, delegated, omit_generic):
    """Execute the workflow Bash selection without starting pytest or QEMU."""
    import os
    import subprocess

    cases = [
        ('w4-full-numeric.yml', 'unit',
         'Run ABI, streaming bundle, runner and actual QEMU tests',
         {'tests/test_llm_inputs.py', 'tests/test_w2_runtime_metadata.py',
          'tests/test_w4_workflows.py', 'tests/test_w4_ci_summary.py'}),
        ('llm-deploy.yml', 'llm-deploy',
         'Runtime unit regressions in pinned environment',
         {'tests/test_llm_inputs.py', 'tests/test_w2_runtime_metadata.py'}),
    ]
    for filename, job, name, generic in cases:
        step = next(step for step in workflow(filename)['jobs'][job]['steps']
                    if step.get('name') == name)
        assert step['env']['CI_COVERS_GENERIC_REGRESSIONS'] == (
            '${{ inputs.ci_covers_generic_regressions }}')
        assert 'continue-on-error' not in step
        # The JUnit checker is tested separately; inspect the actual pytest argv.
        script = step['run'].split("python - <<'PY'", 1)[0]
        capture = tmp_path / (filename + '.args')
        wrapper = 'python() { printf "%s\\n" "$@" > "$CAPTURE"; return "${PYTEST_EXIT:-0}"; }\n'
        env = dict(os.environ, GITHUB_EVENT_NAME=event, QWEN3_PY='python',
                   CAPTURE=str(capture), PYTEST_EXIT='0')
        env.pop('CI_COVERS_GENERIC_REGRESSIONS', None)

        def invoke(flag, exit_code='0'):
            selected_env = dict(env, PYTEST_EXIT=exit_code)
            if flag is not None:
                selected_env['CI_COVERS_GENERIC_REGRESSIONS'] = flag
            result = subprocess.run(['bash', '--noprofile', '--norc', '-c', wrapper + script],
                                    cwd=tmp_path, env=selected_env, capture_output=True, text=True)
            assert result.returncode == int(exit_code), result.stderr
            return [arg for arg in capture.read_text().splitlines() if arg.startswith('tests/')]

        baseline = invoke('false')
        actual = invoke(delegated)
        assert generic <= set(baseline)
        assert len(actual) == len(set(actual))
        assert actual == [arg for arg in baseline if not (omit_generic and arg in generic)]
        if filename == 'w4-full-numeric.yml':
            assert ('tests/test_qemu_generation.py::'
                    'test_real_qemu_multitoken_reuse_numerics_timeout_and_cancel') in actual
            assert 'tests/test_riscv_process_cleanup.py' in actual
        else:
            assert 'tests/test_qwen3_tokenizer.py' in actual
            assert 'tests/test_w2_runtime_model.py' in actual
        if omit_generic:
            invoke(delegated, '7')  # Delegation must not swallow test failures.

def test_test_dependencies_resolve_once_with_fixed_cpu_stack():
    import tomllib

    steps = workflow('ci.yml')['jobs']['test']['steps']
    commands = '\n'.join(step.get('run', '') for step in steps)
    installs = [line.strip() for line in commands.splitlines() if 'pip install' in line]
    assert installs == [
        'python -m pip install --upgrade pip setuptools wheel',
        'python -m pip install -e ".[all,llm-deploy]"',
    ]
    assert 'python -m pip check' in commands
    assert "torch.__version__ == '2.7.1+cpu' and torch.version.cuda is None" in commands
    assert 'import torch, yaml, safetensors' in commands
    extras = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']['optional-dependencies']
    assert {'pytest==9.1.1', 'PyYAML==6.0.3', 'onnxruntime==1.22.1'} <= set(extras['llm-deploy'])
    assert any(item.startswith('torch @ https://download.pytorch.org/whl/cpu/')
               for item in extras['llm-deploy'])


def test_pip_download_caches_are_separate_and_keep_environment_checks():
    roles = [
        ('ci.yml', 'test'), ('ci.yml', 'benchmark'),
        ('llm-deploy.yml', 'llm-deploy'),
        ('llm-deploy.yml', 'full-qwen3-onnx'),
        ('llm-deploy.yml', 'full-qwen3-frontend'),
        ('w4-full-numeric.yml', 'unit'),
    ]
    keys, prefixes = [], []
    for filename, job in roles:
        steps = workflow(filename)['jobs'][job]['steps']
        caches = [step['with'] for step in steps
                  if step.get('uses', '').startswith('actions/cache@')
                  and step.get('with', {}).get('path') == '~/.cache/pip']
        assert len(caches) == 1
        cache = caches[0]
        assert '${{ runner.os }}' in cache['key'] and 'py3.12-' in cache['key']
        assert "hashFiles('pyproject.toml', 'requirements/*.txt')" in cache['key']
        assert cache['key'].startswith(cache['restore-keys'])
        keys.append(cache['key'])
        prefixes.append(cache['restore-keys'])
        python_setup = next(step for step in steps
                            if step.get('uses', '').startswith('actions/setup-python@'))
        assert python_setup['with']['python-version'] == '3.12'
        if filename != 'ci.yml' or job == 'test':
            assert any('pip check' in step.get('run', '') for step in steps)
    assert len(set(keys)) == len(keys)
    assert len(set(prefixes)) == len(prefixes)

    deploy_steps = workflow('llm-deploy.yml')['jobs']['llm-deploy']['steps']
    probe = next(step['run'] for step in deploy_steps
                 if step.get('name') == 'Install pinned CPU Qwen3 probe environment')
    assert 'python3 -m venv output/qwen3-probe-venv' in probe
    assert '"$QWEN3_PY" -m pip install -v -e ".[llm-deploy]"' in probe
    assert 'torch.version.cuda is None' in probe

def test_topic06_measures_once_but_keeps_required_category_failures():
    import shlex

    steps = workflow('topic06-benchmark.yml')['jobs']['topic06-benchmark']['steps']
    commands = '\n'.join(step.get('run', '') for step in steps)
    assert commands.count('scripts/run_topic06_benchmarks.py') == 1
    benchmark = next(step for step in steps if step.get('id') == 'benchmark')
    assert 'if' not in benchmark and 'continue-on-error' not in benchmark
    args = shlex.split(benchmark['run'].replace('\\\n', ' '))
    assert args == [
        'python', 'scripts/run_topic06_benchmarks.py',
        '--require-category', 'activation',
        '--require-category', 'elementwise',
        '--require-category', 'loop',
    ]
    report = next(step for step in steps
                  if step.get('name') == 'Generate full diagnostic benchmark report')
    assert report['run'] == 'python scripts/generate_topic06_report.py --full-report'
    assert report['if'] == (
        "${{ !cancelled() && (steps.benchmark.outcome == 'success' || "
        "steps.benchmark.outcome == 'failure') }}")
