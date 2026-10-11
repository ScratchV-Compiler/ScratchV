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
    for name, filename in [('llm', 'llm-deploy.yml'), ('docs', 'linux-reproduction-docs.yml'),
                           ('topic06', 'topic06-benchmark.yml')]:
        assert jobs[name]['uses'] == './.github/workflows/' + filename
        assert 'workflow_call' in workflow(filename)['on']
        assert not ({'pull_request', 'push'} & set(workflow(filename)['on']))
    for name in ('llm', 'docs'):
        assert jobs[name]['needs'] == 'scope'
        assert jobs[name]['if'] == f"github.event_name == 'pull_request' && needs.scope.outputs.{name} == 'true'"
    assert workflow('llm-deploy.yml')['jobs']['full-qwen3-onnx']['if'] == "github.event_name != 'pull_request'"
    assert 'workflow_dispatch' in workflow('llm-deploy.yml')['on']


def test_scope_and_acceptance_are_fail_closed_and_keep_all_required_dependencies():
    data = workflow('ci.yml')
    jobs = data['jobs']
    assert jobs['scope']['outputs'] == {name: '${{ steps.scope.outputs.' + name + ' }}'
                                       for name in ('llm', 'docs', 'w4')}
    checkout = jobs['scope']['steps'][0]
    assert checkout['with']['fetch-depth'] == '0'
    assert checkout['with']['persist-credentials'] == 'false'
    gate = jobs['acceptance']
    assert gate['if'] == 'always()'
    assert set(gate['needs']) == {'scope', 'test', 'benchmark', 'topic06', 'docs', 'llm'} | ({'w4'} if 'w4' in jobs else set())
    assert gate['steps'][-1]['env']['CI_JOB_RESULTS'] == '${{ toJSON(needs) }}'
    assert gate['steps'][-1]['run'] == 'python3 scripts/summarize_ci.py >> "$GITHUB_STEP_SUMMARY"'
    assert data['concurrency']['cancel-in-progress'] == "${{ github.event_name == 'pull_request' }}"
    assert 'github.run_id' in data['concurrency']['group']


def test_scopes_keep_required_frontend_and_docs_coverage():
    scopes = json.loads((ROOT / 'scripts/ci_scopes.json').read_text(encoding='utf-8'))
    assert {'scratchv/**', 'probes/**', 'requirements/**'} <= set(scopes['llm'])
    assert {'docs/**', 'README.md', 'CONTRIBUTING.md'} <= set(scopes['docs'])
    assert bool(scopes['w4']) == (WORKFLOWS / 'w4-full-numeric.yml').exists()


def test_long_compiler_and_topic_reports_are_folded_without_dropping_artifacts():
    for filename, job, step_name in [('ci.yml', 'benchmark', 'Write job summary'),
                                    ('topic06-benchmark.yml', 'topic06-benchmark', 'Publish benchmark summary')]:
        steps = workflow(filename)['jobs'][job]['steps']
        summary = next(step for step in steps if step.get('name') == step_name)
        assert summary['if'] == 'always()'
        assert summary['run'].count('<details>') == summary['run'].count('</details>') == 1
        assert any('actions/upload-artifact@' in step.get('uses', '') for step in steps)
