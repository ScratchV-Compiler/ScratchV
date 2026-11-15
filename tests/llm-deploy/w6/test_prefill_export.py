"""Short Prefill keeps valid logits/KV and the existing 256-position Decode ABI."""
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import pytest
import torch

from scripts import export_qwen3_kv_onnx as exporter
from scratchv.backend.tensor_c_codegen import TensorCCodegen
from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.runtime.kv_cache import prepare_decode, prepare_prefill


ATOL = 1e-5
RTOL = 1e-5


def torch_run(wrapper, feed):
    with torch.no_grad():
        return [value.numpy().copy() for value in wrapper(
            *(torch.from_numpy(value) for value in feed.values()))]


def compare(actual, expected):
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        np.testing.assert_allclose(left, right, atol=ATOL, rtol=RTOL)
    np.testing.assert_array_equal(actual[0].argmax(axis=-1), expected[0].argmax(axis=-1))


@pytest.fixture(scope='module')
def exported(tmp_path_factory):
    """Reuse one two-layer random-weight Qwen3 for every export and reference."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = exporter.load_model(seed=17)
        root = tmp_path_factory.mktemp('short-prefill-export')
        records = {}
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        with patch.object(exporter, 'load_model', return_value=model):
            for length in (256, 5, 1):
                directory = root / str(length)
                # Length one also exercises a feed shorter than the old three
                # export tokens, without exporting another unchanged Decode.
                mode = 'prefill' if length == 1 else None
                _, wrappers, layout, report = exporter.export_models(
                    directory, seed=17, exporter='legacy', mode=mode,
                    **({} if length == 256 else {'prefill_length': length}))
                sessions = {
                    name: ort.InferenceSession(str(directory / f'{name}.onnx'), options,
                                               providers=['CPUExecutionProvider'])
                    for name in report['graphs']
                }
                records[length] = dict(directory=directory, wrappers=wrappers, layout=layout,
                                       report=report, sessions=sessions)
        yield records
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize('length', [True, False, 0, -1, 257, 1.0, '5', None])
def test_invalid_prefill_length_is_rejected_before_loading_model(tmp_path, length):
    with pytest.raises(ValueError, match='prefill_length'):
        exporter.make_wrappers(None, prefill_length=length)
    with patch.object(exporter, 'load_model', side_effect=AssertionError('Must reject before load')):
        with pytest.raises(ValueError, match='prefill_length'):
            exporter.export_models(tmp_path / 'invalid', prefill_length=length)
    assert not (tmp_path / 'invalid').exists()


def test_export_shapes_parse_and_keep_fixed_cache_capacity(exported):
    artifacts = {}
    for length, record in exported.items():
        assert record['report']['prefill_length'] == length
        session = record['sessions']['prefill']
        assert [value.shape for value in session.get_inputs()] == [
            [1, length], [1, 1, length, length], [1]]
        layout = record['layout']
        assert [value.shape for value in session.get_outputs()] == [
            [1, 1, layout.vocab_size], *([list(layout.shape)] * len(layout.cache_names))]
        parser = ONNXParser()
        program = parser.parse(str(record['directory'] / 'prefill.onnx'))
        artifacts[length] = TensorCCodegen(
            program, parser.initializers, constant_storage='external', kernel_calls=True,
            abi_version=2).generate()
    base = exported[256]
    parser = ONNXParser()
    program = parser.parse(str(base['directory'] / 'decode.onnx'))
    decode = TensorCCodegen(program, parser.initializers, constant_storage='external',
                            kernel_calls=True, abi_version=2).generate()
    for length, artifact in artifacts.items():
        assert base['layout'].validate_graphs(artifact, decode) == length
    assert artifacts[5].workspace_bytes < artifacts[256].workspace_bytes
    # Changing Prefill length never changes the exported Decode interface.
    assert [(v.name, v.shape) for v in base['sessions']['decode'].get_inputs()] == [
        (v.name, v.shape) for v in exported[5]['sessions']['decode'].get_inputs()]


@pytest.mark.parametrize('length', [1, 5])
def test_every_valid_logit_row_and_cache_matches_256_prefill(exported, length):
    base, short = exported[256], exported[length]
    layout = base['layout']
    tokens = [1, 5, 7, 9, 11][:length]
    wide_feed = prepare_prefill(tokens, layout)
    short_feed = prepare_prefill(tokens, layout, prefill_length=length)
    for index in range(length):
        wide_feed['logit_index'][0] = short_feed['logit_index'][0] = index
        wide = base['sessions']['prefill'].run(None, wide_feed)
        compact = short['sessions']['prefill'].run(None, short_feed)
        host_wide = torch_run(base['wrappers'][0], wide_feed)
        host_compact = torch_run(short['wrappers'][0], short_feed)
        compare(wide, host_wide)
        compare(compact, host_compact)
        compare([compact[0]], [wide[0]])
        for actual, expected in zip(compact[1:], wide[1:]):
            np.testing.assert_allclose(actual[:, :, :length], expected[:, :, :length],
                                       atol=ATOL, rtol=RTOL)
            np.testing.assert_array_equal(actual[:, :, length:], 0)


def test_two_decode_steps_after_short_prefill_match_ort(exported):
    base, short = exported[256], exported[5]
    layout = base['layout']
    tokens = [1, 5, 7, 9, 11]
    wide_feed = prepare_prefill(tokens, layout)
    short_feed = prepare_prefill(tokens, layout, prefill_length=5)
    outputs = [base['sessions']['prefill'].run(None, wide_feed),
               short['sessions']['prefill'].run(None, short_feed),
               torch_run(base['wrappers'][0], wide_feed),
               torch_run(short['wrappers'][0], short_feed)]
    # The guest clears unused cache slots before publishing Prefill results.
    # Match that existing contract, not the old graph's discarded pad values.
    for result in outputs:
        for cache in result[1:]:
            cache[:, :, 5:] = 0
    for actual in outputs[1:]:
        compare(actual, outputs[0])
    for position in (5, 6):
        token = int(outputs[0][0][0, 0].argmax())
        feeds = [prepare_decode(token, position, layout, result[1:]) for result in outputs]
        outputs = [base['sessions']['decode'].run(None, feeds[0]),
                   short['sessions']['decode'].run(None, feeds[1]),
                   torch_run(base['wrappers'][1], feeds[2]),
                   torch_run(short['wrappers'][1], feeds[3])]
        for actual in outputs[1:]:
            compare(actual, outputs[0])
        for result in outputs:
            for cache in result[1:]:
                np.testing.assert_array_equal(cache[:, :, position+1:], 0)


def test_dynamo_short_prefill_matches_reference_and_compiles(exported, tmp_path):
    base = exported[256]
    directory = tmp_path / 'dynamo-5'
    with patch.object(exporter, 'load_model', return_value=base['wrappers'][0].model):
        _, _, layout, report = exporter.export_models(
            directory, seed=17, prefill_length=5, mode='prefill', exporter='dynamo')
    assert report['prefill_length'] == 5 and report['exporter'] == 'dynamo'
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(directory / 'prefill.onnx'), options,
                                   providers=['CPUExecutionProvider'])
    tokens = [1, 5, 7, 9, 11]
    wide_feed = prepare_prefill(tokens, layout)
    short_feed = prepare_prefill(tokens, layout, prefill_length=5)
    for index in range(5):
        wide_feed['logit_index'][0] = short_feed['logit_index'][0] = index
        wide = base['sessions']['prefill'].run(None, wide_feed)
        compact = session.run(None, short_feed)
        compare([compact[0]], [wide[0]])
        for actual, expected in zip(compact[1:], wide[1:]):
            np.testing.assert_allclose(actual[:, :, :5], expected[:, :, :5],
                                       atol=ATOL, rtol=RTOL)
            np.testing.assert_array_equal(actual[:, :, 5:], 0)
    artifacts = []
    for path in (directory / 'prefill.onnx', base['directory'] / 'decode.onnx'):
        parser = ONNXParser()
        program = parser.parse(str(path), mmap_external_data=True)
        artifacts.append(TensorCCodegen(
            program, parser.initializers, constant_storage='external', kernel_calls=True,
            abi_version=2).generate())
    assert layout.validate_graphs(*artifacts) == 5
