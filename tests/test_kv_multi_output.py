"""Ordered results, output lifetime, ABI safety and shared cache RAM contract."""
import ctypes
from dataclasses import replace
import shutil
import subprocess
import numpy as np
import onnx
from onnx import helper, TensorProto
import pytest
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, Value
from scratchv.analysis.ir_verifier import verify_ir
from scratchv.frontend.onnx_parser import ONNXParser, ONNXParseError
from scratchv.backend.tensor_c_codegen import TensorCCodegen, TensorCCodegenError
from scratchv.verification.ir_interpreter import IRInterpreter
from scratchv.runtime.weight_bundle import write_shared_weight_bundle, plan_session_memory


def graph():
    b = IRBuilder()
    x = Value('x', shape=(2,3))
    f = b.new_function('main',[x])
    b.new_block('entry')
    early = b.neg(x)
    later = b.add(early,x)
    final = b.neg(later)
    f.returns = [Value(n,shape=(2,3)) for n in ('early','input','alias','final')]
    b.ret_many([early,x,early,final])
    return b.program


def test_interpreter_returns_owned_ordered_aliases_and_accounts_for_copies():
    x = np.arange(6,dtype=np.float32).reshape(2,3)
    p = graph()
    assert verify_ir(p)[0]
    result = IRInterpreter(p).run({'x':x},collect_memory_stats=True,memory_mode='last_use')
    assert len(result.return_values)==4
    for a,b in zip(result.return_values,[-x,x,-x,np.zeros_like(x)]):
        np.testing.assert_array_equal(a,b)
        assert not np.shares_memory(a,x)
    assert not np.shares_memory(result.return_values[0],result.return_values[2])
    assert result.memory_stats['return_logical_bytes']==4*x.nbytes
    with pytest.raises(ValueError,match='return_values'):
        _ = result.return_value


def test_parser_keeps_declaration_order_and_fails_missing_result(tmp_path):
    model = helper.make_model(helper.make_graph(
        [helper.make_node('Neg',['x'],['a']),helper.make_node('Add',['a','x'],['b'])],
        'ordered',[helper.make_tensor_value_info('x',TensorProto.FLOAT,[2])],
        [helper.make_tensor_value_info(n,TensorProto.FLOAT,[2]) for n in ('b','x','a')]),
        opset_imports=[helper.make_opsetid('',18)],ir_version=10)
    path = tmp_path/'ordered.onnx'
    onnx.save(model,path)
    parser = ONNXParser()
    p = parser.parse(str(path))
    a = TensorCCodegen(p,constant_storage='external').generate()
    assert [s.name for s in a.outputs]==['b','x','a']
    results = IRInterpreter(p).run({'x':np.array([2,3],np.float32)}).return_values
    np.testing.assert_array_equal(results[0],[0,0])
    np.testing.assert_array_equal(results[1],[2,3])
    np.testing.assert_array_equal(results[2],[-2,-3])
    model.graph.output[-1].name='absent'
    onnx.save(model,path)
    with pytest.raises(ONNXParseError,match='no value binding'):
        parser.parse(str(path))


@pytest.mark.parametrize('change',['count','type','undefined','shape'])
def test_verifier_checks_every_return(change):
    p=graph()
    f=p.functions[0]
    if change=='count':
        f.returns.pop()
    elif change=='type':
        f.returns[-1].dtype=DataType.INT64
    elif change=='shape':
        f.returns[1].shape=(3,2)
    else:
        f.blocks[0].instructions[-1].operands[-1]=Value('missing')
    assert not verify_ir(p)[0]


@pytest.mark.parametrize('name',['bad-name','int','1test','sv_finite','sv_overlap'])
def test_entry_point_validation(name):
    with pytest.raises(TensorCCodegenError,match='entry point'):
        TensorCCodegen(graph(),constant_storage='external',function_name=name).generate()


def test_legacy_backends_refuse_multiple_results():
    from scratchv.backend.instruction_select import InstructionSelector
    from scratchv.backend.llvm_codegen import LLVMCodegen
    for operation in (lambda:InstructionSelector(graph()).run(),lambda:LLVMCodegen(graph()).emit()):
        with pytest.raises(ValueError,match='multiple return'):
            operation()
    with pytest.raises(TensorCCodegenError,match='external'):
        TensorCCodegen(graph()).generate()


def test_host_v2_keeps_all_outputs_live_and_checks_every_output(tmp_path):
    compiler=shutil.which('cc')
    if not compiler:
        pytest.skip('Host C compiler unavailable')
    artifact=TensorCCodegen(graph(),constant_storage='external',function_name='test_multi').generate()
    with pytest.raises(TensorCCodegenError,match='outputs'):
        _=artifact.output
    source=tmp_path/'model.c';source.write_text(artifact.source)
    library=tmp_path/'model.so'
    subprocess.run([compiler,'-shared','-fPIC','-O2',*artifact.compile_flags,str(source),'-o',str(library),'-lm'],
                   capture_output=True,check=True,timeout=60)
    loaded=ctypes.CDLL(str(library))
    function=loaded.test_multi
    ptr=ctypes.POINTER(ctypes.c_void_p)
    function.argtypes=[ptr,ptr,ctypes.c_void_p,ctypes.c_size_t,ptr]
    x=np.arange(6,dtype=np.float32).reshape(2,3)
    output=[np.full((2,3),123,np.float32) for _ in artifact.outputs]
    arena=np.empty((artifact.workspace_bytes+7)//8,np.uint64)
    pointers=lambda arrays:(ctypes.c_void_p*len(arrays))(*(a.ctypes.data for a in arrays))
    inputs=pointers([x]);weights=pointers([])
    args=[inputs,weights,arena.ctypes.data,artifact.workspace_bytes]
    assert function(*args,pointers(output))==0
    for a,b in zip(output,[-x,x,-x,np.zeros_like(x)]):
        np.testing.assert_array_equal(a,b)
    # Invalid late outputs must fail before the first output is copied.
    bad=pointers(output);bad[-1]=None
    output[0].fill(123)
    assert function(*args,bad)==1
    assert np.all(output[0]==123)
    bad=pointers(output);bad[-1]=x.ctypes.data
    assert function(*args,bad)==1
    bad=pointers(output);bad[-1]=output[0].ctypes.data
    assert function(*args,bad)==1
    bad=pointers(output);bad[-1]=arena.ctypes.data
    assert function(*args,bad)==1
    bad=pointers(output);bad[-1]=output[-1].ctypes.data+1
    assert function(*args,bad)==1
    assert function(inputs,weights,arena.ctypes.data,artifact.workspace_bytes-1,pointers(output))==1
    x[0,0]=np.nan
    assert function(*args,pointers(output))==2


def test_shared_weights_require_logical_and_physical_identity(tmp_path):
    from scratchv.backend.tensor_c_codegen import TensorSpec
    spec=TensorSpec('weight',DataType.FLOAT32,(2,))
    artifact=TensorCCodegen(graph(),constant_storage='external').generate()
    a=replace(artifact,external_weights=(spec,),external_initializers=(np.array([1,2],np.float32),))
    b=replace(a)
    c=replace(a,external_initializers=(np.array([1,3],np.float32),))
    d=replace(a,matmul_rhs_layout='column-panel')
    manifest,specs,maps=write_shared_weight_bundle([a,b,c,d],tmp_path/'weights')
    assert maps==((0,),(0,),(1,),(2,))
    assert len(specs)==len(manifest['tensors'])==3


def test_session_memory_uses_larger_workspace_and_disjoint_banks():
    layout=plan_session_memory(56*1024**2,prefill_input_bytes=265,decode_input_bytes=1040,
        prefill_workspace_bytes=12345,decode_workspace_bytes=6789,logits_bytes=151936*4,
        weight_bytes=2400000000,cores=4)
    assert layout['regions']['workspace']['nbytes']==12345
    assert layout['cache_bases'][1]-layout['cache_bases'][0]==56*1024**2
    assert layout['cache_bases'][1]+layout['cache_bank_bytes']<=layout['workspace_base']
    assert all(address%64==0 for address in layout['cache_bases'])
    assert layout['regions']['stack']['nbytes']==4*1024**2


def test_verifier_does_not_treat_scalar_boundary_as_unknown_shape():
    b=IRBuilder()
    x=Value('scalar',shape=())
    f=b.new_function('main',[x]);b.new_block('entry')
    f.returns=[Value('tensor',shape=(1,)),Value('scalar',shape=())]
    b.ret_many([x,x])
    passed,issues=verify_ir(b.program)
    assert not passed and any('returned shape' in issue.message for issue in issues)


def test_shared_panels_use_checkpoint_identity_across_graph_ssa_names(tmp_path):
    from scratchv.backend.tensor_c_codegen import TensorSpec
    base=TensorCCodegen(graph(),constant_storage='external').generate()
    array=np.arange(8,dtype=np.float32).reshape(1,2,4)
    def panel(name,source):
        return replace(base,external_weights=(TensorSpec(name,DataType.FLOAT32,array.shape),),
            external_initializers=(array,),matmul_rhs_layout='column-panel',
            matmul_rhs_panels=({'name':name,'source_weight':source,'folded_transpose':name,
                'logical_shape':[2,4],'physical_shape':[1,2,4],'panel_columns':4},))
    _,specs,mappings=write_shared_weight_bundle(
        [panel('v_12','checkpoint.weight'),panel('v_99','checkpoint.weight'),
         panel('v_12','different.weight')],tmp_path/'panels')
    assert len(specs)==2 and mappings==((0,),(0,),(1,))


def test_full_model_output_validation_does_not_expand_every_weight_pair():
    def source_bytes(count):
        b=IRBuilder();x=Value('x',shape=(1,))
        f=b.new_function('main',[x]);b.new_block('entry')
        weights=[Value(f'weight_{i}',shape=(1,)) for i in range(200)]
        b.program.global_values.extend(weights)
        f.returns=[Value(f'result_{i}',shape=(1,)) for i in range(count)]
        b.ret_many([x]*count)
        initializers={w.name:np.array([1],np.float32) for w in weights}
        return len(TensorCCodegen(b.program,initializers,constant_storage='external').generate().source)
    assert source_bytes(57) < 3*source_bytes(2)
