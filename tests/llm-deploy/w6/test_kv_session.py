"""Real QEMU lifecycle/failure gates. Run explicitly with the pinned toolchain."""
import os
import signal
import threading
import numpy as np
import pytest
from scratchv.backend.tensor_c_codegen import TensorCCodegen
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType as D, Value
from scratchv.runtime.kv_cache import KVCacheLayout
from scratchv.runtime.riscv_session import (build_session, QemuSession,
    save_session_executable, load_session_executable)
from scratchv.runtime.riscv_tensor import discover_toolchain
from scratchv.runtime.generation import GenerationCancelledError


def artifact(mode, layout, fail=False):
    b=IRBuilder()
    decode=mode=='decode'
    inputs=[Value('input_ids',D.INT64,shape=(1,1 if decode else 256)),
            Value('attention_mask',shape=(1,1,1 if decode else 256,256)),
            Value('position_ids' if decode else 'logit_index',D.INT64,shape=(1,1) if decode else (1,))]
    if decode:
        inputs += [Value('past.'+n,shape=layout.shape) for n in layout.cache_names]
    f=b.new_function(mode,inputs);b.new_block('entry')
    logits=Value('constant_logits',shape=(1,1,8))
    initializers={'constant_logits':np.arange(8,dtype=np.float32).reshape(1,1,8)}
    b.program.global_values.append(logits)
    outputs=[logits]
    if decode:
        # Change every output on each invocation; target/host compare bank parity.
        for source in inputs[3:]:
            outputs.append(b.add(source,b.make_const(1.0)))
        if fail:
            # Graph is legal; the model returns a numeric-domain failure at runtime.
            outputs[-1]=b.div(outputs[-1],b.make_const(0.0))
    else:
        for i,name in enumerate(layout.cache_names):
            v=Value(name+'.initial',shape=layout.shape)
            b.program.global_values.append(v)
            initializers[v.name]=np.full(layout.shape,i+1,np.float32)
            outputs.append(v)
    f.returns=[Value('logits',shape=(1,1,8)),*[Value(n,shape=layout.shape) for n in layout.cache_names]]
    b.ret_many(outputs)
    return TensorCCodegen(b.program,initializers,constant_storage='external',
                          function_name='sv_'+mode+'_v2',kernel_calls=True).generate()


@pytest.fixture(scope='module',params=[1,2,4])
def executable(request,tmp_path_factory):
    layout=KVCacheLayout(1,1,2,8)
    return build_session(artifact('prefill',layout),artifact('decode',layout),layout,
                         tmp_path_factory.mktemp('session')/'build',discover_toolchain(),cores=request.param)


def test_continuous_requests_reset_and_reuse_metadata(executable,tmp_path):
    save_session_executable(executable,tmp_path)
    exe=load_session_executable(tmp_path/'executable.json')
    with QemuSession(exe,tmp_path/'run',timeout=20) as s:
        pid=s.process.pid
        s.reset();s.prefill([1,2,3])
        for step in range(3):
            s.decode(4)
            cache=s.dump_cache()
            for i,a in enumerate(cache):
                np.testing.assert_array_equal(a[:,:,:3],i+2+step)
            assert s.process.pid==pid and s.valid_length==4+step
        s.reset()
        assert s.valid_length==0
        assert all(np.all(a==0) for a in s.dump_cache())
        s.prefill([3]);s.decode(2)
    s.close();s.close()
    assert s.report['passed'] and s.report['qemu_reaped']
    assert not s.monitor.thread.is_alive()


def test_cancelled_session_is_reaped(executable,tmp_path):
    cancelled=threading.Event()
    with pytest.raises(GenerationCancelledError):
        with QemuSession(executable,tmp_path/'cancel',timeout=20,cancelled=cancelled.is_set) as s:
            s.reset();s.prefill([1]);cancelled.set();s.decode(2)
    assert s.state=='FAULT' and s.valid_length==1 and s.report['qemu_reaped']
    assert not s.monitor.thread.is_alive()


def test_timeout_and_guest_exit_are_not_replayed(executable,tmp_path):
    for mode in ('timeout','exit'):
        s=QemuSession(executable,tmp_path/mode,timeout=20).start()
        s.reset();s.prefill([1])
        if mode=='timeout':
            # Pausing all harts leaves a valid guest unable to respond.
            s.qmp.command('stop');s.timeout=0.05
            error=TimeoutError
        else:
            s.process.kill();s.process.wait(timeout=5)
            error=RuntimeError
        with pytest.raises(error):s.decode(2)
        assert s.state=='FAULT' and s.valid_length==1 and s.report['qemu_reaped']
        assert not s.monitor.thread.is_alive()
        s.close()


@pytest.mark.skipif(os.name=='nt',reason='POSIX SIGTERM unwind')
def test_sigterm_unwinds_and_restores_handler(executable,tmp_path):
    previous=signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit) as error:
        with QemuSession(executable,tmp_path/'term',timeout=20) as s:
            s.reset();os.kill(os.getpid(),signal.SIGTERM)
    assert error.value.code==143
    assert signal.getsignal(signal.SIGTERM)==previous
    assert s.report['qemu_reaped']


def test_numeric_failure_does_not_publish_new_length(tmp_path):
    layout=KVCacheLayout(1,1,2,8)
    exe=build_session(artifact('prefill',layout),artifact('decode',layout,fail=True),layout,
                      tmp_path/'bad-build',discover_toolchain())
    s=QemuSession(exe,tmp_path/'bad-run',timeout=20).start()
    s.reset();s.prefill([1,2])
    with pytest.raises(ValueError,match='status=2'):
        s.decode(3)
    assert s.valid_length==2 and s.bank==0 and s.report['qemu_reaped']
