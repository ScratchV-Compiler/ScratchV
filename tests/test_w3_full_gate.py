import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from probes.w3_qwen3_full.cases import CASE_NAMES, input_cases
from probes.w3_qwen3_full.comparison import compare_tensor, compare_positions
from probes.w3_qwen3_full import resources
from probes.w3_qwen3_full.run import checked_artifact, main


def test_inputs_cover_boundaries_and_isolate_mask_effects():
    cases = input_cases()
    assert tuple(x[0] for x in cases) == CASE_NAMES
    by_name = {name: (valid, feed) for name, valid, feed in cases}
    for _, length, feed in cases:
        assert feed['input_ids'].shape == (1,256)
        assert feed['input_ids'].dtype == np.int64
        for q in (0,16,17,63,64,254,255):
            visible = np.flatnonzero(feed['attention_mask'][0,0,q] == 0)
            np.testing.assert_array_equal(visible,np.arange(min(q+1,length)))
    a,b = by_name['full_seed_0'][1],by_name['changed_future'][1]
    np.testing.assert_array_equal(a['input_ids'][:,:64],b['input_ids'][:,:64])
    assert np.all(a['input_ids'][:,64:] != b['input_ids'][:,64:])
    a,b = by_name['short_17'][1],by_name['changed_padding'][1]
    np.testing.assert_array_equal(a['input_ids'][:,:17],b['input_ids'][:,:17])
    np.testing.assert_array_equal(a['attention_mask'],b['attention_mask'])
    assert np.all(a['input_ids'][:,17:] != b['input_ids'][:,17:])


def test_comparison_reports_padding_failure_even_if_valid_tokens_match():
    expected=np.zeros((1,7,3),dtype=np.float32)
    actual=expected.copy(); actual[0,6,2]=0.01
    row=compare_positions(actual,expected,3)
    assert not row['passed'] and row['valid_queries']['passed']
    assert not row['padding_queries']['passed']
    assert row['worst_index']==[0,6,2]


def test_chunk_boundary_and_strided_index_are_exact():
    a=np.zeros((2,9,7),dtype=np.float32)[:,::2,::2]
    b=a.copy(); b[1,4,3]=0.5
    row=compare_tensor(a,b,chunk_elements=3)
    assert row['worst_index']==[1,4,3]
    assert row['max_abs']==0.5 and row['actual_value']==0 and row['reference_value']==0.5


@pytest.mark.parametrize('invalid',[np.nan,np.inf,-np.inf])
def test_nonfinite_is_failure_with_json_safe_evidence(invalid):
    a=np.array([0,1,invalid],dtype=np.float32)
    row=compare_tensor(a,a,chunk_elements=1)
    assert not row['passed'] and row['worst_index']==[2]
    json.dumps(row,allow_nan=False)


def test_threshold_is_strict_and_empty_is_not_evidence():
    a=np.array([0.5],dtype=np.float32); b=np.array([0],dtype=np.float32)
    assert not compare_tensor(a,b,atol=0.5)['passed']
    assert compare_tensor(a,b,atol=0.5001)['passed']
    assert not compare_tensor(a[:0],b[:0])['passed']
    assert not compare_tensor(a.astype(np.float64),b)['passed']


def test_artifact_cannot_point_outside_case(tmp_path):
    target=tmp_path/'other.npy'; target.write_bytes(b'data')
    with pytest.raises(ValueError,match='Unexpected artifact path'):
        checked_artifact(tmp_path,{'path':'../other.npy'},'logits.npy')


@pytest.mark.parametrize('argument',['nan','inf','0','-1'])
def test_cli_rejects_invalid_resource_limits_before_output(tmp_path,argument):
    out=tmp_path/'out'
    with pytest.raises(SystemExit) as exc:
        main(['--model-dir',str(tmp_path),'--output-dir',str(out),'--worker-timeout',argument])
    assert exc.value.code==2 and not out.exists()


def test_memory_observer_reads_own_process():
    row=resources.process_memory(os.getpid())
    assert row['rss_bytes']>0
    assert row['private_commit_bytes'] is None or row['private_commit_bytes']>0


def spawn_sleeper():
    flags={'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=='nt' else {'start_new_session':True}
    return resources.spawn_owned([sys.executable,'-c','import time; time.sleep(30)'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,**flags)


@pytest.mark.parametrize('cause',['timeout','memory','observer','interrupt'])
def test_worker_limits_and_cancellation_stop_process_tree(monkeypatch,cause):
    process=spawn_sleeper(); row={}
    try:
        if cause=='memory':
            monkeypatch.setattr(resources,'process_memory',lambda pid:{'rss_bytes':100,'private_commit_bytes':None})
        elif cause in ('observer','interrupt'):
            def fail(pid):
                raise OSError('cannot monitor') if cause=='observer' else KeyboardInterrupt()
            monkeypatch.setattr(resources,'process_memory',fail)
        expected={'timeout':TimeoutError,'memory':MemoryError,'observer':OSError,'interrupt':KeyboardInterrupt}[cause]
        with pytest.raises(expected):
            resources.wait_bounded(process,timeout=0.15 if cause=='timeout' else 5,
                                   max_memory_bytes=10 if cause=='memory' else 2**40,row=row,interval=0.02)
        assert process.poll() is not None
        assert 'resource_monitor' in row and row['elapsed_seconds']>=0
    finally:
        if process.poll() is None:resources.terminate_process_tree(process)
