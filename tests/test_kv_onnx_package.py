"""Publishing must preserve ONNX computations and every external tensor byte."""
import json

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort
import pytest

from scripts.package_qwen3_kv_onnx import package
from export_qwen3_onnx import external_tensors, sha256


@pytest.fixture
def exports(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    for mode, offset in [('prefill', 0), ('decode', 1)]:
        weight = numpy_helper.from_array(np.arange(16, dtype=np.float32).reshape(4, 4), 'weight')
        bias = numpy_helper.from_array(np.arange(4, dtype=np.float32)+offset, 'bias')
        nodes = [helper.make_node('MatMul', ['x', 'weight'], ['a']),
                 helper.make_node('Add', ['a', 'bias'], ['y'])]
        graph = helper.make_graph(nodes, mode,
            [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 4])], [weight, bias])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)], ir_version=10)
        onnx.save_model(model, source/f'{mode}.onnx', save_as_external_data=True,
                        all_tensors_to_one_file=True, location=f'{mode}.data', size_threshold=0)
    return source


def test_shared_shards_preserve_both_models_and_leave_sources_intact(exports, tmp_path):
    before = {p.name: sha256(p) for p in exports.iterdir()}
    target = tmp_path/'package'
    report = package(exports, target, shard_bytes=128)
    assert report['unique_external_tensors'] == 3  # shared matrix, distinct biases
    assert report['packed_weight_bytes'] < report['original_weight_bytes']
    assert len(list(target.glob('*.data'))) == 2
    assert all(p.stat().st_size <= 128 for p in target.glob('*.data'))
    for mode in ('prefill', 'decode'):
        old = onnx.load(exports/f'{mode}.onnx')
        new = onnx.load(target/f'{mode}.onnx')
        for a, b in zip(old.graph.initializer, new.graph.initializer):
            np.testing.assert_array_equal(numpy_helper.to_array(a), numpy_helper.to_array(b))
        x = {'x': np.array([[1, 2, 3, 4]], np.float32)}
        a = ort.InferenceSession(str(exports/f'{mode}.onnx')).run(None, x)
        b = ort.InferenceSession(str(target/f'{mode}.onnx')).run(None, x)
        np.testing.assert_array_equal(a[0], b[0])
    assert before == {p.name: sha256(p) for p in exports.iterdir()}
    for row in report['files']:
        assert sha256(target/row['name']) == row['sha256']
    assert json.loads((target/'manifest.json').read_text())['verification']['all_external_tensor_bytes_identical']


def test_existing_package_is_never_overwritten(exports, tmp_path):
    target = tmp_path/'package'
    package(exports, target, shard_bytes=128)
    before = sha256(target/'manifest.json')
    with pytest.raises(FileExistsError):
        package(exports, target, shard_bytes=128)
    assert sha256(target/'manifest.json') == before


@pytest.mark.parametrize('limit', [0, -1, True, 2*1024**3, 63])
def test_invalid_size_or_oversized_tensor_is_rejected(exports, tmp_path, limit):
    with pytest.raises(ValueError):
        package(exports, tmp_path/'package', shard_bytes=limit)
    assert not (tmp_path/'package/manifest.json').exists()


@pytest.mark.parametrize('corruption', ['truncated', 'escape'])
def test_invalid_source_range_or_path_cannot_produce_release(exports, tmp_path, corruption):
    if corruption == 'truncated':
        (exports/'prefill.data').write_bytes(b'bad')
    else:
        model = onnx.load(exports/'prefill.onnx', load_external_data=False)
        tensor = next(external_tensors(model))
        next(entry for entry in tensor.external_data if entry.key == 'location').value = '../outside.data'
        (tmp_path/'outside.data').write_bytes(b'bad')
        onnx.save_model(model, exports/'prefill.onnx')
    with pytest.raises(ValueError):
        package(exports, tmp_path/'package', shard_bytes=128)
    assert not (tmp_path/'package/manifest.json').exists()
