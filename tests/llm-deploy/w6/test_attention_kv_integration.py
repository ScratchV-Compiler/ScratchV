"""Actual KV decode shapes, GQA pairing, masking and transactional multi-output ABI."""
from dataclasses import replace
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import pytest

from scratchv.backend.tensor_c_codegen import TensorCCodegen, TensorCCodegenError
from scratchv.frontend.onnx_parser import ONNXParser
from tests.test_tensor_c_external import _PARALLEL_HOST_RUNTIME, host_cc, library, pointers


pytestmark = pytest.mark.skipif(os.name == 'nt', reason='Parallel Host C fixture uses pthreads')


def graph(path, *, shared_score=False, groups=2):
    tensor = lambda name, shape: helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
    heads = 8 * groups
    shapes = dict(q=(1, heads, 1, 128), k=(1, 8, 256, 128),
                  v=(1, 8, 256, 128), mask=(1, 1, 1, 256))
    arrays = dict(axes=np.array([2], np.int64), shape=np.array([1, heads, 256, 128], np.int64),
                  scale=np.array(1 / np.sqrt(128), np.float32))
    nodes = [helper.make_node('Neg', ['q'], ['query'])]
    for name in ('k', 'v'):
        nodes.extend([helper.make_node('Unsqueeze', [name, 'axes'], [name+'u']),
                      helper.make_node('Concat', [name+'u']*groups, [name+'c'], axis=2),
                      helper.make_node('Reshape', [name+'c', 'shape'], [name+'r'])])
    nodes.extend([
        helper.make_node('Transpose', ['kr'], ['kt'], perm=[0, 1, 3, 2]),
        helper.make_node('MatMul', ['query', 'kt'], ['score']),
        helper.make_node('Mul', ['score', 'scale'], ['scaled']),
        helper.make_node('Add', ['scaled', 'mask'], ['masked']),
        helper.make_node('Softmax', ['masked'], ['probability'], axis=-1),
        helper.make_node('Cast', ['probability'], ['prob'], to=TensorProto.FLOAT),
        helper.make_node('MatMul', ['prob', 'vr'], ['context']),
    ])
    outputs = [tensor('context', (1, heads, 1, 128)), tensor('k', shapes['k']), tensor('v', shapes['v'])]
    if shared_score:
        outputs.append(tensor('score', (1, heads, 1, 256)))
    model = helper.make_model(helper.make_graph(nodes, 'decode-attention',
        [tensor(name, shape) for name, shape in shapes.items()], outputs,
        [numpy_helper.from_array(value, name) for name, value in arrays.items()]),
        opset_imports=[helper.make_opsetid('', 18)], ir_version=10)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, path)
    parser = ONNXParser()
    program = parser.parse(str(path))
    return program, parser.initializers


def artifact(program, bindings, policy='kv-heads', nonlinear='fp32-reference'):
    return TensorCCodegen(program, bindings, constant_storage='external', kernel_calls=True,
        matmul_policy='blocked_fma', matmul_tile='2x4', nonlinear_policy=nonlinear,
        attention_decode_policy=policy).generate()


@pytest.mark.parametrize('cores,nonlinear', [(1, 'fp32-reference'), (4, 'fp32-reference'), (4, 'libm')])
def test_real_kv_shapes_mask_and_all_outputs_match_ort(tmp_path, host_cc, cores, nonlinear):
    path = tmp_path/'attention.onnx'
    program, bindings = graph(path)
    original = artifact(program, bindings, policy='original', nonlinear=nonlinear)
    fused = artifact(program, bindings, nonlinear=nonlinear)
    assert fused.source.count('/* w6_kv_attention_begin:') == 1
    assert fused.workspace_bytes < original.workspace_bytes
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    reference = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(321)
    feed = {name: rng.normal(0, .25, size=shape).astype(np.float32) for name, shape in
            [('q',(1,16,1,128)), ('k',(1,8,256,128)), ('v',(1,8,256,128))]}
    feed['mask'] = np.zeros((1,1,1,256), np.float32)
    compiled = []
    for label, model in [('original', original), ('fused', fused)]:
        compiled.append((label, replace(model, source=model.source+_PARALLEL_HOST_RUNTIME)))
    results = {}
    for label, model in compiled:
        weights = [np.ascontiguousarray(a) if a.ndim else a.copy() for a in model.external_initializers]
        arena = np.empty((model.workspace_bytes+7)//8, np.uint64)
        outputs = [np.full(s.shape, -999, s.numpy_dtype) for s in model.outputs]
        with library(tmp_path/label, model, [*host_cc, f'-DSCRATCHV_CORES={cores}', '-pthread']) as run:
            for length in (1, 5, 17, 255, 256):
                feed['mask'].fill(0)
                feed['mask'][...,length:] = np.finfo(np.float32).min
                status = run(pointers(list(feed.values())), pointers(weights), arena.ctypes.data,
                             arena.nbytes, pointers(outputs))
                assert status == 0
                expected = reference.run(None, feed)
                for actual, oracle in zip(outputs, expected):
                    np.testing.assert_allclose(actual, oracle, atol=1e-6, rtol=1e-5)
                if label == 'original':
                    results[length] = [a.copy() for a in outputs]
                else:
                    for actual, previous in zip(outputs, results[length]):
                        np.testing.assert_array_equal(actual.view(np.uint32), previous.view(np.uint32))
            # All Q/K/V heads differ. A head mapping or 256-stride bug cannot be
            # hidden by identical inputs, and each query head is checked above.


def test_masked_overflow_and_nonfinite_preserve_all_outputs_and_allow_reuse(tmp_path, host_cc):
    program, bindings = graph(tmp_path/'attention.onnx')
    model = artifact(program, bindings)
    model = replace(model, source=model.source+_PARALLEL_HOST_RUNTIME)
    q = np.zeros((1,16,1,128), np.float32)
    k = np.zeros((1,8,256,128), np.float32)
    v = np.ones_like(k)
    mask = np.full((1,1,1,256), np.finfo(np.float32).min, np.float32)
    mask[...,0] = 0
    weights = [a.copy() for a in model.external_initializers]
    arena = np.empty((model.workspace_bytes+7)//8, np.uint64)
    outputs = [np.full(s.shape, -999, s.numpy_dtype) for s in model.outputs]
    with library(tmp_path/'compiled', model, [*host_cc, '-DSCRATCHV_CORES=4', '-pthread']) as run:
        def call(size=arena.nbytes):
            return run(pointers([q,k,v,mask]),pointers(weights),arena.ctypes.data,size,pointers(outputs))
        assert call(arena.nbytes-1) == 1
        v[0,-1,-1,-1] = np.nan
        assert call() == 2
        v[0,-1,-1,-1] = 1
        # Even a masked future slot must preserve the ordinary path's failure
        # if a finite Q*K overflows before the mask is added.
        q[0,-1,0,0] = 2
        k[0,-1,-1,0] = np.finfo(np.float32).max
        assert call() == 2
        assert all(np.all(a == -999) for a in outputs)
        k.fill(0)
        assert call() == 0
        np.testing.assert_array_equal(outputs[0], 1)
        np.testing.assert_array_equal(outputs[1], k)
        np.testing.assert_array_equal(outputs[2], v)


@pytest.mark.parametrize('kwargs', [{'shared_score':True}, {'groups':1}])
def test_shared_intermediates_and_other_layouts_keep_original_implementation(tmp_path, kwargs):
    program, bindings = graph(tmp_path/'attention.onnx', **kwargs)
    model = artifact(program, bindings)
    assert '/* w6_kv_attention_begin:' not in model.source
    assert model.source == artifact(program, bindings, policy='original').source


def test_policy_requires_compatible_arithmetic(tmp_path):
    program, bindings = graph(tmp_path/'attention.onnx')
    with pytest.raises(TensorCCodegenError, match='blocked_fma/2x4'):
        TensorCCodegen(program,bindings,constant_storage='external',kernel_calls=True,
                      attention_decode_policy='kv-heads').generate()


def test_four_core_qemu_matches_original_and_ort(tmp_path):
    from scratchv.runtime.riscv_external import build_external, run_external
    from scratchv.runtime.riscv_tensor import discover_toolchain
    from scratchv.runtime.weight_bundle import write_weight_bundle
    path = tmp_path/'attention.onnx'
    program, bindings = graph(path)
    # The ordinary single-result runner suffices to exercise the fused kernel;
    # Host C above separately verifies publication of every ABI v2 result.
    function = program.functions[0]
    function.returns = function.returns[:1]
    function.blocks[0].instructions[-1].operands = function.blocks[0].instructions[-1].operands[:1]
    rng = np.random.default_rng(245)
    feed = {name:rng.normal(0,.25,size=shape).astype(np.float32) for name,shape in
            [('q',(1,16,1,128)),('k',(1,8,256,128)),('v',(1,8,256,128))]}
    feed['mask'] = np.zeros((1,1,1,256),np.float32)
    feed['mask'][...,17:] = np.finfo(np.float32).min
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    expected = ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider']).run(None,feed)[0]
    tools = discover_toolchain()
    previous = None
    for policy in ('original','kv-heads'):
        model = artifact(program,bindings,policy)
        directory = tmp_path/policy
        directory.mkdir()
        write_weight_bundle(model.external_weights,model.external_initializers,directory/'weights')
        executable = build_external(model,directory/'weights',directory/'build',tools,cores=4)
        actual, report = run_external(executable,feed,directory/'run',timeout=60)
        assert report['passed'] and executable.cores == 4
        np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-5)
        if previous is not None:
            np.testing.assert_array_equal(actual.view(np.uint32),previous.view(np.uint32))
        previous = actual
