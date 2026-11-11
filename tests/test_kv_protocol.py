import struct
from types import SimpleNamespace
import numpy as np
import pytest
from scratchv.runtime import kv_protocol as p
from scratchv.runtime.kv_cache import KVCacheLayout
from scratchv.runtime.riscv_session import QemuSession


def test_stream_split_and_coalesced_messages():
    frames=[p.Frame(p.PREFILL,123,1,payload=struct.pack('<qq',1,2)),p.Frame(p.DONE,123,1,2,512)]
    data=b''.join(f.encode() for f in frames)
    decoder=p.FrameDecoder()
    result=[]
    for byte in data:
        result+=decoder.feed(bytes([byte]))
    assert result==frames
    assert p.FrameDecoder().feed(data)==frames


@pytest.mark.parametrize('offset',[0,8,12,24,40,48,52,56])
def test_invalid_header_or_checksum(offset):
    data=bytearray(p.Frame(p.DECODE,1,1,1,payload=struct.pack('<q',2)).encode())
    data[offset]^=255
    with pytest.raises(ValueError):
        p.FrameDecoder().feed(data)


def test_oversized_length_is_rejected_without_waiting_for_body():
    data=bytearray(p.Frame(p.RESET,1,1).encode())
    struct.pack_into('<I',data,16,p.MAX_PAYLOAD+1)
    with pytest.raises(ValueError,match='header'):
        p.FrameDecoder().feed(data)
    decoder=p.FrameDecoder();decoder.feed(b'SVKV')
    with pytest.raises(EOFError,match='Truncated'):
        decoder.eof()


class Serial:
    def __init__(self,response):
        self.response=response
        self.closed=False
        self.sent=[]
    def settimeout(self,value): pass
    def sendall(self,value): self.sent.append(value)
    def recv(self,size):
        value,self.response=self.response,b''
        return value
    def close(self): self.closed=True


class QMP:
    def __init__(self,fail=False): self.fail=fail
    def command(self,name,*args):
        if self.fail:
            raise OSError('memory transport failure')
        return ''
    def close(self): pass


def mock_session(tmp_path,monkeypatch,response):
    exe=SimpleNamespace(cores=1,evidence={'identity':'a'*64},layout={'logits_base':0x90000000},
                        cache_layout=KVCacheLayout(2,2,16,128))
    session=QemuSession(exe,tmp_path/'run',timeout=1)
    session.run_dir.mkdir()
    session.session_id=42
    session.state='READY'
    session.valid_length=3
    session.serial=Serial(response)
    session.qmp=QMP()
    session.process=SimpleNamespace(returncode=None,poll=lambda:None,wait=lambda timeout:None)
    monkeypatch.setattr('scratchv.runtime.riscv_session._stop_process_tree',lambda *a:None)
    monkeypatch.setattr(session,'_dump',lambda *a:np.zeros((1,1,128),np.float32).tobytes())
    return session


def test_success_commits_one_bank_after_transport(tmp_path,monkeypatch):
    s=mock_session(tmp_path,monkeypatch,p.Frame(p.DONE,42,1,4,512).encode())
    a=s.decode(2)
    assert a.shape==(1,1,128) and s.valid_length==4 and s.bank==1 and s.state=='READY'
    request=p.FrameDecoder().feed(s.serial.sent[0])[0]
    assert request.valid_length==3 and request.request==1


@pytest.mark.parametrize('response',[
    p.Frame(p.DONE,43,1,4,512),p.Frame(p.DONE,42,2,4,512),p.Frame(p.DONE,42,0,4,512),
    p.Frame(p.DONE,42,1,5,512),p.Frame(p.DONE,42,1,4,1024),p.Frame(p.ERROR,42,1,3,status=2),
    p.Frame(p.ACK,42,1,4,512)])
def test_bad_response_faults_without_host_commit(tmp_path,monkeypatch,response):
    s=mock_session(tmp_path,monkeypatch,response.encode())
    with pytest.raises(ValueError): s.decode(2)
    assert s.state=='FAULT' and s.valid_length==3 and s.bank==0
    assert s.serial.closed
    assert len(s.runs)==1 and not s.runs[0]['passed']
    assert 'request' in (s.run_dir/'session.json').read_text()
    with pytest.raises(RuntimeError,match='not usable'): s.decode(2)
    s.close();s.close()


@pytest.mark.parametrize('failure',['transport','numeric','eof','duplicate','cancelled','timeout'])
def test_uncertain_results_invalidate_session(tmp_path,monkeypatch,failure):
    packet=p.Frame(p.DONE,42,1,4,512).encode()
    s=mock_session(tmp_path,monkeypatch,packet)
    if failure=='transport': s.qmp=QMP(True)
    elif failure=='numeric':
        monkeypatch.setattr(s,'_dump',lambda *a:np.full((128,),np.nan,np.float32).tobytes())
    elif failure=='eof': s.serial.response=b''
    elif failure=='duplicate': s.serial.response=packet+packet
    elif failure=='cancelled': s.cancelled=lambda:True
    elif failure=='timeout': s.timeout=1e-12
    with pytest.raises((ValueError,OSError,EOFError,RuntimeError,TimeoutError)):
        s.decode(2)
    assert s.state=='FAULT' and s.valid_length==3 and s.bank==0 and s.serial.closed
