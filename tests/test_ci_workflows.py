"""Keep one PR validation run without dropping gates or running full models."""
import json
from pathlib import Path

import yaml


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
    assert writers == {'full-qwen3-onnx', 'w3-execution-summary'}


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
