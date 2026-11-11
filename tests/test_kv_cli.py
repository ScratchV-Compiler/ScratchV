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


def test_reuse_completion_and_failure_reports(cli,tmp_path,monkeypatch):
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
        def __exit__(self,kind,error,tb):self.report.update(passed=error is None,qemu_reaped=True)
        def reset(self):pass
        def prefill(self,ids):return self.decode(1)
        def decode(self,token):
            self.runs.append({'passed':True})
            a=np.zeros((1,1,8),np.float32);a[0,0,3]=1;return a
    monkeypatch.setattr(runtime,'QemuSession',Session)
    assert cli.main(arguments(tmp_path))==0
    report=json.loads((tmp_path/'out'/'report.json').read_text())
    assert report['passed'] and report['kv_cache'] and report['session']['qemu_reaped']
    assert report['result']['generated_token_ids']==[3,3,3]
    assert len(sessions[0].runs)==3
    args=arguments(tmp_path)
    args[args.index(str(tmp_path/'out'))]=str(tmp_path/'failed')
    def failure(self,ids):raise RuntimeError('guest failure')
    monkeypatch.setattr(Session,'prefill',failure)
    assert cli.main(args)==1
    report=json.loads((tmp_path/'failed'/'report.json').read_text())
    assert not report['passed'] and report['session']['qemu_reaped']
    assert 'guest failure' in report['error']


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
