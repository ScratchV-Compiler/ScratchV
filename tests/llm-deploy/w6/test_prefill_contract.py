"""Short static Prefill inputs keep the 256-position Decode/cache contract."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scratchv.runtime import kv_protocol
from scratchv.runtime.kv_cache import KVCacheLayout, prepare_prefill
from scratchv.runtime.riscv_session import (
    QemuSession, SessionExecutable, _offsets, _validate_session_metadata,
    load_session_executable, save_session_executable,
)
from scratchv.runtime.riscv_tensor import RiscVToolchain
from scratchv.runtime.weight_bundle import plan_session_memory
from test_kv_session import artifact


LAYOUT = KVCacheLayout(1, 1, 2, 8)


def signatures(length):
    # Reuse the lifecycle test's tiny ABI v2 artifacts. Only signatures are
    # inspected here; these replacement artifacts are never compiled/executed.
    prefill, decode = artifact('prefill', LAYOUT), artifact('decode', LAYOUT)
    inputs = (replace(prefill.inputs[0], shape=(1, length)),
              replace(prefill.inputs[1], shape=(1, 1, length, length)), prefill.inputs[2])
    return replace(prefill, inputs=inputs), decode


def executable(directory, length, *, legacy=False):
    prefill, decode = signatures(length)
    _, pi = _offsets(prefill.inputs)
    _, di = _offsets(decode.inputs[:3])
    evidence = dict(identity='a'*64, prefill_input_bytes=pi, decode_input_bytes=di,
                    prefill_workspace_bytes=64, decode_workspace_bytes=64,
                    cache_layout=LAYOUT.to_dict())
    if not legacy:
        evidence['prefill_length'] = length
    layout = plan_session_memory(LAYOUT.bank_bytes, prefill_input_bytes=pi, decode_input_bytes=di,
        prefill_workspace_bytes=64, decode_workspace_bytes=64, logits_bytes=32, weight_bytes=0, cores=1)
    return SessionExecutable(directory/'model.elf', directory/'weights', {'total_bytes':0}, (),
                             layout, LAYOUT, RiscVToolchain(('cc',),'qemu'), evidence, 1)


@pytest.mark.parametrize('length,count', [(1,1), (5,1), (5,3), (5,5), (256,256)])
def test_short_inputs_preserve_ids_mask_and_last_logit_index(length, count):
    tokens = [0,1,2,3,4]*(count//5+1)
    tokens = tokens[:count]
    result = prepare_prefill(tokens, LAYOUT, prefill_length=length)
    full = prepare_prefill(tokens, LAYOUT)
    assert result['input_ids'].shape == (1,length)
    assert result['attention_mask'].shape == (1,1,length,length)
    np.testing.assert_array_equal(result['input_ids'],full['input_ids'][:,:length])
    np.testing.assert_array_equal(result['attention_mask'],full['attention_mask'][:,:,:length,:length])
    assert result['logit_index'].tolist() == [count-1]
    assert all(a.flags.c_contiguous for a in result.values())
    assert result['input_ids'].dtype == np.int64
    assert result['attention_mask'].dtype == np.float32


@pytest.mark.parametrize('bad,error', [(0,ValueError),(257,ValueError),(-1,ValueError),
                                      (True,TypeError),(5.0,TypeError),('5',TypeError)])
def test_prefill_length_is_a_bounded_integer(bad, error):
    with pytest.raises(error):
        prepare_prefill([1],LAYOUT,prefill_length=bad)


@pytest.mark.parametrize('tokens', [[],[1]*6,[8],[-1]])
def test_invalid_short_prompts_are_rejected_without_truncation(tokens):
    with pytest.raises(ValueError):
        prepare_prefill(tokens,LAYOUT,prefill_length=5)


@pytest.mark.parametrize('length', [1,5,256])
def test_signatures_allow_short_prefill_and_keep_complete_cache(length):
    prefill, decode = signatures(length)
    assert LAYOUT.validate_graphs(prefill,decode) == length
    assert all(s.shape == LAYOUT.shape for s in prefill.outputs[1:])
    bad_mask = replace(prefill, inputs=(prefill.inputs[0],
        replace(prefill.inputs[1],shape=(1,1,length,255)),prefill.inputs[2]))
    with pytest.raises(ValueError,match='KV ABI'):
        LAYOUT.validate_graphs(bad_mask,decode)
    short_output = replace(prefill,outputs=(
        replace(prefill.outputs[0],shape=(1,2,8)),*prefill.outputs[1:]))
    with pytest.raises(ValueError,match='KV ABI'):
        LAYOUT.validate_graphs(short_output,decode)
    bad_cache = replace(prefill,outputs=(prefill.outputs[0],
        replace(prefill.outputs[1],shape=(1,1,255,2)),prefill.outputs[2]))
    with pytest.raises(ValueError,match='KV ABI'):
        LAYOUT.validate_graphs(bad_cache,decode)


@pytest.mark.parametrize('length,legacy', [(5,False),(256,False),(256,True)])
def test_metadata_roundtrip_and_input_bytes_are_checked(tmp_path, length, legacy):
    exe = executable(tmp_path,length,legacy=legacy)
    assert exe.prefill_length == length
    _validate_session_metadata(exe)
    save_session_executable(exe,tmp_path)
    restored = load_session_executable(tmp_path/'executable.json')
    assert restored.prefill_length == length
    assert restored.cache_layout.capacity == 256
    record = json.loads((tmp_path/'executable.json').read_text())
    record['evidence']['prefill_length'] = 4 if length == 5 else 5
    (tmp_path/'executable.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='input bytes'):
        load_session_executable(tmp_path/'executable.json')


def test_short_metadata_cannot_drop_length_and_fall_back_to_256(tmp_path):
    exe = executable(tmp_path,5)
    exe.evidence.pop('prefill_length')
    with pytest.raises(ValueError,match='input bytes'):
        _validate_session_metadata(exe)


def test_host_rejects_long_prompt_before_request_and_session_remains_usable(tmp_path,monkeypatch):
    exe = executable(tmp_path,5)
    session = QemuSession(exe,tmp_path/'run')
    session.state = 'READY'
    calls = []
    def request(kind,payload=b'',*,length=None):
        calls.append((kind,payload,length))
        session.valid_length = length
        return np.zeros((1,1,8),np.float32)
    monkeypatch.setattr(session,'_request',request)
    with pytest.raises(ValueError,match='prefill_length=5'):
        session.prefill([1]*6)
    assert not calls and session.state == 'READY' and session.valid_length == 0
    session.prefill([1,2,3,4,5])
    assert calls[0][0] == kv_protocol.PREFILL and calls[0][2] == 5
    np.testing.assert_array_equal(np.frombuffer(calls[0][1],dtype='<i8'),[1,2,3,4,5])
    session.decode(6)
    assert calls[-1][0] == kv_protocol.DECODE and session.valid_length == 6
    session.valid_length = 256
    with pytest.raises(ValueError,match='below capacity'):
        session.decode(6)


def test_old_duck_typed_executable_keeps_default_length(tmp_path):
    exe = SimpleNamespace(cores=1,evidence={'identity':'a'*64},layout={},cache_layout=LAYOUT)
    session = QemuSession(exe,tmp_path/'run')
    assert session.prefill_length == session.report['prefill_length'] == 256
