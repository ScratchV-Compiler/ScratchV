"""CLI mode selection, build reuse, arithmetic contract, reporting and failures."""
from importlib.util import module_from_spec,spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from tests.test_generation import Tokenizer
from scratchv.runtime.kv_cache import KVCacheLayout
from scratchv.runtime.riscv_session import SessionExecutable
from scratchv.runtime.riscv_tensor import RiscVToolchain


@pytest.fixture
def cli():
    path=Path(__file__).resolve().parents[1]/'scripts/run_inference.py'
    spec=spec_from_file_location('kv_inference_cli_tests',path)
    module=module_from_spec(spec);spec.loader.exec_module(module)
    return module


def arguments(tmp_path):
    return ['--kv-cache','--reuse-build',str(tmp_path/'old'/'executable.json'),
            '--tokenizer-dir',str(tmp_path/'tokenizer'),'--output-dir',str(tmp_path/'out'),
            '--prompt','raw','--max-new-tokens','3']


def test_mode_selection_and_static_capacity(cli,tmp_path):
    args=cli.parse_args(arguments(tmp_path))
    assert args.kv_cache and args.kv_capacity==256
    for extra in (['--mode','chat'],['--decode-model','x'],['--kv-capacity','512']):
        with pytest.raises(SystemExit):cli.parse_args(arguments(tmp_path)+extra)
    with pytest.raises(SystemExit):
        cli.parse_args(['--kv-cache','--model','p.onnx','--tokenizer-dir','x','--output-dir','y','--prompt','a'])
    with pytest.raises(SystemExit):
        cli.parse_args(['--model','p.onnx','--decode-model','d.onnx','--tokenizer-dir','x','--output-dir','y','--prompt','a'])


@pytest.mark.parametrize('failure_phase', ['prefill', 'close'])
def test_reuse_completion_and_failure_reports(cli,tmp_path,monkeypatch,failure_phase):
    import scratchv.runtime.riscv_session as runtime
    path=tmp_path/'old';path.mkdir();(path/'executable.json').write_text('{}')
    (path/'report.json').write_text(json.dumps({'executable_sha256':cli.sha256_file(path/'executable.json')}))
    policies=dict(matmul_policy='sequential',matmul_tile='1x4',matmul_rhs_layout='original',
                  mean_policy='sequential',nonlinear_policy='libm')
    exe=SessionExecutable(path/'model.elf',path/'weights',{},(),{},KVCacheLayout(1,1,2,8),
        RiscVToolchain(('cc',),'qemu'),{'identity':'a'*64,'policies':policies},1)
    monkeypatch.setattr(runtime,'load_session_executable',lambda path:exe)
    monkeypatch.setattr(runtime,'save_session_executable',lambda exe,out:'b'*64)
    monkeypatch.setattr(cli.Qwen3Tokenizer,'from_directory',lambda path:Tokenizer())
    sessions=[]
    class Session:
        def __init__(self,*a,**k):
            self.report={'passed':False,'qemu_reaped':False};self.runs=[];sessions.append(self)
        def __enter__(self):return self
        def __exit__(self,kind,error,tb):self.report.update(passed=error is None,qemu_reaped=True,exit_code=0)
        def reset(self):pass
        def prefill(self,ids):return self.decode(1)
        def decode(self,token):
            self.runs.append({'passed':True})
            a=np.zeros((1,1,8),np.float32);a[0,0,3]=1;return a
    monkeypatch.setattr(runtime,'QemuSession',Session)
    assert cli.main(arguments(tmp_path))==0
    report=json.loads((tmp_path/'out'/'report.json').read_text())
    assert report['passed'] and report['kv_cache'] and report['session']['qemu_reaped']
    assert report['session']['exit_code']==0
    assert report['result']['generated_token_ids']==[3,3,3]
    assert len(sessions[0].runs)==3
    args=arguments(tmp_path)
    args[args.index(str(tmp_path/'out'))]=str(tmp_path/'failed')
    if failure_phase=='prefill':
        def failure(self,ids):raise RuntimeError('guest failure')
        monkeypatch.setattr(Session,'prefill',failure)
        message='guest failure'
    else:
        def failure(self,kind,error,tb):
            self.report.update(passed=False,qemu_reaped=True,exit_code=-9)
            raise RuntimeError('QEMU shutdown failed (-9)')
        monkeypatch.setattr(Session,'__exit__',failure)
        message='QEMU shutdown failed (-9)'
    assert cli.main(args)==1
    report=json.loads((tmp_path/'failed'/'report.json').read_text())
    assert not report['passed'] and report['session']['qemu_reaped']
    assert message in report['error']
    if failure_phase=='close':
        assert report['session']['exit_code']==-9


def test_common_compiler_cli_exposes_v2_and_reports_outputs(tmp_path):
    import onnx
    from onnx import helper,TensorProto
    from scratchv.compiler import CompilerConfig,CompilerDriver
    from scratchv.main import args_to_config,build_arg_parser
    model=helper.make_model(helper.make_graph([], 'identity',
        [helper.make_tensor_value_info('x',TensorProto.FLOAT,[2]),
         helper.make_tensor_value_info('y',TensorProto.INT64,[1])],
        [helper.make_tensor_value_info('y',TensorProto.INT64,[1]),
         helper.make_tensor_value_info('x',TensorProto.FLOAT,[2])]),
        opset_imports=[helper.make_opsetid('',18)],ir_version=10)
    path=tmp_path/'multi.onnx';onnx.save(model,path)
    args=build_arg_parser().parse_args([str(path),'--backend','tensor-c','--tensor-constant-storage','external',
        '--tensor-function-name','sv_test_v2','--tensor-abi-version','2'])
    driver=CompilerDriver(args_to_config(args));result=driver.compile(str(path),str(tmp_path/'multi.c'))
    assert result.success,result.errors
    assert result.stats['tensor']['abi_version']==2
    assert [s['name'] for s in result.stats['tensor']['outputs']]==['y','x']


@pytest.fixture
def cached_runtime(cli, tmp_path, monkeypatch):
    """Exercise the real cached token loop with a tiny resident-session double."""
    from dataclasses import replace
    import scratchv.runtime.riscv_session as runtime
    from tests.test_w5_inference_cli import Tokenizer as FixtureTokenizer

    path = tmp_path / 'old'
    path.mkdir()
    (path / 'executable.json').write_text('{}')
    (path / 'report.json').write_text(json.dumps({
        'executable_sha256': cli.sha256_file(path / 'executable.json')}))
    policies = dict(matmul_policy='sequential', matmul_tile='1x4', matmul_rhs_layout='original',
                    mean_policy='sequential', nonlinear_policy='libm')
    exe = SessionExecutable(path / 'model.elf', path / 'weights', {}, (), {},
                            KVCacheLayout(1, 1, 2, 8), RiscVToolchain(('cc',), 'qemu'),
                            {'identity': 'a'*64, 'policies': policies}, 1)
    tokenizer = FixtureTokenizer()
    root = tmp_path / 'fixture-root'
    references = root / 'probes/w5_generation/references.json'
    references.parent.mkdir(parents=True)
    fixture = {'status': 'frozen', 'tokenizer': {
        'pad_token_id': 0, 'generation_eos_token_ids': [6],
        'model_vocab_size': 8, 'tokenizer_vocab_size': 7},
        'cases': [{'name': name, 'prompt': name, 'max_new_tokens': 8,
                   'prompt_token_ids': [1, 2], 'generated_token_ids': [3]*8,
                   'generated_text': 'x'*8, 'stop_reason': 'max_new_tokens',
                   'required_substring': 'x'} for name in ('france', 'germany', 'japan')]}
    references.write_text(json.dumps(fixture))
    state = SimpleNamespace(exe=exe, sessions=[], saved=[], histories=[], tokenizer=tokenizer,
                            fixture=fixture, references=references, fail=None, actual_error=0.0,
                            oracle_token=3, actual_token=3, oracle_error=0.0, resolved=[], snapshots=[])

    def resolve(requested):
        state.resolved.append(requested)
        return 4 if requested == 'auto' else requested

    def oracle_run(names, feed):
        assert names == ['logits']
        history = list(feed['input_ids'][0])
        state.histories.append(history[:next((i for i, token in enumerate(history) if token == 0), 256)])
        value = np.zeros((1, 256, 8), np.float32)
        value[0, :, state.oracle_token] = 1
        value[0, :, 0] = state.oracle_error
        return [value]

    class Session:
        def __init__(self, executable, output, **kwargs):
            assert executable is state.exe
            self.report, self.runs, self.requests = {}, [], []
            state.sessions.append(self)
        def __enter__(self):
            self.report.update(startup_s=0.0)
            return self
        def __exit__(self, kind, error, tb):
            self.report.update(passed=error is None, qemu_reaped=True, exit_code=0)
            if state.fail == 'close':
                raise RuntimeError('close failed')
        def reset(self):
            self.requests.append(('reset', None))
        def prefill(self, ids):
            return self.forward('prefill', list(ids))
        def decode(self, token):
            return self.forward('decode', token)
        def forward(self, kind, value):
            self.requests.append((kind, value))
            if isinstance(state.fail, BaseException):
                raise state.fail
            output = np.zeros((1, 1, 8), np.float32)
            output[0, 0, state.actual_token] = 1
            output[0, 0, 0] = state.actual_error
            self.runs.append({'passed': True})
            return output

    monkeypatch.setattr(cli, 'ROOT', root)
    monkeypatch.setattr(cli, 'resolve_cores', resolve)
    monkeypatch.setattr(cli, 'require_full_cached_executable', lambda exe: None)
    monkeypatch.setattr(runtime, 'load_session_executable', lambda path: state.exe)
    monkeypatch.setattr(runtime, 'save_session_executable', lambda exe, out: state.saved.append(exe) or 'metadata')
    monkeypatch.setattr(runtime, 'build_session', lambda *a, **kw: pytest.fail('reuse must not rebuild'))
    monkeypatch.setattr(runtime, 'QemuSession', Session)
    monkeypatch.setattr(cli.Qwen3Tokenizer, 'from_directory', lambda path: tokenizer)
    monkeypatch.setattr(cli, 'reference_session', lambda *a: SimpleNamespace(run=oracle_run))
    monkeypatch.setattr(cli, 'oracle_session', lambda *a: SimpleNamespace(run=oracle_run))
    save = cli.atomic_text
    def record(path, text):
        state.snapshots.append(json.loads(text))
        save(path, text)
    monkeypatch.setattr(cli, 'atomic_text', record)
    state.replace_exe = lambda **changes: setattr(state, 'exe', replace(state.exe, **changes))
    return state


def suite_arguments(tmp_path, *extra):
    return arguments(tmp_path)[:-4] + ['--reference-suite', '--verify-onnx', 'onnx/model.onnx', *extra]


def read_report(tmp_path):
    return json.loads((tmp_path / 'out/report.json').read_text())


@pytest.mark.parametrize('option,requested,resolved', [([], 1, 1), (['--cores', '4'], 4, 4),
                                                        (['--harts', 'auto'], 'auto', 4)])
def test_frozen_suite_reuses_one_session_and_build_for_three_eight_token_cases(
        cli, cached_runtime, tmp_path, option, requested, resolved):
    state = cached_runtime
    state.replace_exe(cores=resolved)
    assert cli.main(suite_arguments(tmp_path, *option)) == 0
    report = read_report(tmp_path)
    assert report['status'] == report['full_model_gate'] == 'PASS'
    assert report['passed'] and report['session']['qemu_reaped']
    assert state.resolved == [requested]
    assert report['cores'] == report['harts'] == resolved
    assert len(state.sessions) == len(state.saved) == 1
    assert [row['name'] for row in report['cases']] == ['france', 'germany', 'japan']
    assert all(row['reference_passed'] and row['stop_reason'] == 'max_new_tokens'
               and row['numeric_step_count'] == row['forward_count'] == 8
               and row['prefill_count'] == 1 and row['decode_count'] == 7 for row in report['cases'])
    assert len(report['numeric_steps']) == 24
    assert [(r['case'], r['step'], r['valid_length'], r['phase']) for r in report['numeric_steps']] == [
        (case, step, step+2, 'prefill' if step == 0 else 'decode')
        for case in ('france', 'germany', 'japan') for step in range(8)]
    assert all(r['elements_compared'] == 8 and r['passed'] for r in report['numeric_steps'])
    assert state.histories == [[1, 2] + [3]*step for _ in range(3) for step in range(8)]
    assert state.sessions[0].requests == ([('reset', None), ('prefill', [1, 2])] + [('decode', 3)]*7)*3
    assert state.snapshots[0]['full_model_gate'] == 'RUNNING'
    for row in report['cases']:
        timing = row['timings']
        assert timing['generation_and_validation_s'] >= timing['generation_call_s'] >= timing['ort_s'] >= 0
        assert timing['numeric_comparison_s'] >= 0 and timing['result_validation_s'] >= 0
    first = report['cases'][0]['timings']
    assert first['first_case_with_session_start_s'] >= first['generation_and_validation_s']
    assert 'first_case_with_session_start_s' not in report['cases'][1]['timings']
    assert report['timings']['resident_session_s'] >= first['first_case_with_session_start_s']


@pytest.mark.parametrize('error,passes', [(np.float32(0.0005), True), (np.float32(0.001), False)])
def test_cached_numeric_tolerance_is_strict_including_exact_threshold(
        cli, cached_runtime, tmp_path, monkeypatch, error, passes):
    # Use exactly the representable FP32 endpoint as ATOL so the equality case
    # cannot accidentally pass/fail only because decimal 0.001 rounded upward.
    monkeypatch.setattr(cli, 'ATOL', float(np.float32(0.001)))
    cached_runtime.actual_error = error
    assert cli.main(arguments(tmp_path) + ['--verify-onnx', 'model.onnx']) == (0 if passes else 1)
    report = read_report(tmp_path)
    row = report['numeric_steps'][0]
    assert row['passed'] is passes
    assert row['greedy_equal']
    if not passes:
        assert row['max_abs'] == row['atol']
        assert len(report['numeric_steps']) == 1 and report['cases'] == []
    assert report['session']['qemu_reaped']


@pytest.mark.parametrize('fault', ['token', 'text', 'stop', 'prompt', 'missing_case', 'budget', 'extra_forward'])
def test_frozen_suite_rejects_contract_and_result_mismatches(cli, cached_runtime, tmp_path, monkeypatch, fault):
    state = cached_runtime
    case = state.fixture['cases'][0]
    if fault == 'token':
        case['generated_token_ids'][0] = 4
    elif fault == 'text':
        case['generated_text'] = 'wrong text'
    elif fault == 'stop':
        case['stop_reason'] = 'eos'
    elif fault == 'prompt':
        case['prompt_token_ids'] = [2, 1]
    elif fault == 'missing_case':
        state.fixture['cases'].pop()
    elif fault == 'budget':
        case['max_new_tokens'] = 7
    else:
        import scratchv.runtime.cached_generation as generation
        original = generation.generate_cached
        def extra(*args, **kwargs):
            result = original(*args, **kwargs)
            kwargs['session'].decode(3)
            return result
        monkeypatch.setattr(generation, 'generate_cached', extra)
    state.references.write_text(json.dumps(state.fixture))
    assert cli.main(suite_arguments(tmp_path)) == 1
    report = read_report(tmp_path)
    assert not report['passed']
    assert report['full_model_gate'] == ('NOT_RUN' if fault in ('missing_case', 'budget') else 'FAIL')
    if fault in ('text', 'stop'):
        assert not report['cases'][0]['reference_passed']
    if fault == 'token':
        assert not report['numeric_steps'][0]['reference_token_passed']
    if fault == 'prompt':
        assert state.sessions[0].requests == [('reset', None)]


@pytest.mark.parametrize('failure,code,status', [(TimeoutError('expired'), 1, 'FAIL'),
                                                 (KeyboardInterrupt('cancelled'), 130, 'CANCELLED'),
                                                 (SystemExit(143), 143, 'CANCELLED'),
                                                 ('close', 1, 'FAIL')])
def test_cached_suite_failure_closes_session_and_never_passes(cli, cached_runtime, tmp_path, failure, code, status):
    import signal
    prior = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    cached_runtime.fail = failure
    assert cli.main(suite_arguments(tmp_path)) == code
    report = read_report(tmp_path)
    assert report['status'] == report['full_model_gate'] == status
    assert not report['passed'] and report['session']['qemu_reaped']
    assert all(signal.getsignal(s) is handler for s, handler in prior.items())


def test_cached_final_report_write_failure_returns_failure(cli, cached_runtime, tmp_path, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(cli, 'atomic_text', fail)
    assert cli.main(arguments(tmp_path)) == 1
    assert cached_runtime.sessions[0].report['qemu_reaped']
    assert 'Report save failed: OSError: disk full' in capsys.readouterr().err


@pytest.mark.parametrize('key', ['lm_head_policy', 'attention_decode_policy'])
def test_reuse_rejects_different_optimization_policy(cli, cached_runtime, tmp_path, key):
    cached_runtime.exe.evidence['policies'][key] = 'vocab-blocked' if key == 'lm_head_policy' else 'kv-heads'
    assert cli.main(arguments(tmp_path)) == 1
    assert 'policies differ' in read_report(tmp_path)['error']
    assert not cached_runtime.sessions


def test_full_kv_suite_requires_complete_layout(cli):
    with pytest.raises(ValueError, match='complete 28-layer'):
        cli.require_full_cached_executable(SimpleNamespace(cache_layout=KVCacheLayout(1, 1, 2, 8)))
    assert cli.require_full_cached_executable(SimpleNamespace(
        cache_layout=KVCacheLayout(28, 8, 128, 151936, pad_token_id=151643))) is None


@pytest.mark.parametrize('extra', [['--lm-head-policy', 'vocab-blocked'],
                                    ['--attention-decode-policy', 'kv-heads']])
def test_optional_policies_require_their_arithmetic_configuration(cli, tmp_path, extra):
    with pytest.raises(SystemExit):
        cli.parse_args(arguments(tmp_path) + extra)
    valid = arguments(tmp_path) + ['--matmul-policy', 'blocked_fma', '--matmul-tile', '2x4',
                                   '--nonlinear-policy', 'fp32-reference'] + extra
    args = cli.parse_args(valid)
    assert args.kv_cache
    if extra[0] == '--attention-decode-policy':
        with pytest.raises(SystemExit):
            cli.parse_args([a for a in valid if a != '--kv-cache'])


def test_build_routes_decode_policy_only_to_decode_and_reuses_one_bundle(
        cli, cached_runtime, tmp_path, monkeypatch):
    from scratchv.frontend.onnx_parser import ONNXParser
    import scratchv.runtime.riscv_session as runtime
    calls, built = [], []
    monkeypatch.setattr(ONNXParser, 'parse', lambda self, path, **kw: path)
    monkeypatch.setattr(cli, 'discover_toolchain', lambda *a: cached_runtime.exe.toolchain)
    class Codegen:
        def __init__(self, program, initializers, **kwargs):
            self.kwargs = kwargs
            calls.append((program, kwargs))
        def generate(self):
            return SimpleNamespace(outputs=[SimpleNamespace(shape=(1, 1, 8)),
                SimpleNamespace(shape=(1, 1, 256, 2)), SimpleNamespace(shape=(1, 1, 256, 2))])
    def build(prefill, decode, cache, out, tools, **kwargs):
        built.append((prefill, decode, cache, kwargs))
        return cached_runtime.exe
    monkeypatch.setattr(cli, 'TensorCCodegen', Codegen)
    monkeypatch.setattr(runtime, 'build_session', build)
    assert cli.main(['--kv-cache', '--model', 'prefill.onnx', '--decode-model', 'decode.onnx',
        '--tokenizer-dir', 'tokenizer', '--output-dir', str(tmp_path / 'out'), '--build-only',
        '--matmul-policy', 'blocked_fma', '--matmul-tile', '2x4',
        '--nonlinear-policy', 'fp32-reference', '--lm-head-policy', 'vocab-blocked',
        '--attention-decode-policy', 'kv-heads']) == 0
    assert len(built) == len(cached_runtime.saved) == 1
    assert not cached_runtime.sessions
    assert [k['attention_decode_policy'] for _, k in calls] == ['original', 'kv-heads']
    assert all(k['lm_head_policy'] == 'vocab-blocked' and k['abi_version'] == 2 for _, k in calls)
    report = read_report(tmp_path)
    assert report['status'] == 'BUILD_ONLY' and not report['passed']
    assert report['build']['policies']['attention_decode_policy'] == 'kv-heads'
    assert report['build']['policies']['lm_head_policy'] == 'vocab-blocked'


@pytest.mark.parametrize('fault', ['greedy', 'nonfinite'])
def test_cached_numeric_checks_reject_greedy_drift_and_nonfinite_logits(
        cli, cached_runtime, tmp_path, fault):
    if fault == 'greedy':
        cached_runtime.actual_token = 4
    else:
        cached_runtime.actual_error = np.nan
    assert cli.main(arguments(tmp_path) + ['--verify-onnx', 'model.onnx']) == 1
    report = read_report(tmp_path)
    assert not report['passed'] and report['session']['qemu_reaped']
    if fault == 'greedy':
        assert not report['numeric_steps'][0]['greedy_equal']
    else:
        assert 'finite' in report['error']
