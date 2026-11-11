import numpy as np
import pytest
from scratchv.runtime.kv_cache import KVCacheLayout,prepare_prefill,prepare_decode,cached_logits
from scratchv.runtime.cached_generation import generate_cached
from scratchv.runtime.generation import GenerationCancelledError
from tests.test_generation import Tokenizer


class Session:
    def __init__(self,tokens=(3,4,3)):
        self.tokens=iter(tokens)
        self.calls=[]
        self.valid_length=0
    def reset(self):
        self.valid_length=0
        self.calls.append(('reset',))
    def row(self):
        a=np.zeros((1,1,8),np.float32)
        a[0,0,next(self.tokens)]=10
        return a
    def prefill(self,tokens):
        self.valid_length=len(tokens)
        self.calls.append(('prefill',tuple(tokens)))
        return self.row()
    def decode(self,token):
        self.calls.append(('decode',token,self.valid_length))
        self.valid_length+=1
        return self.row()


def test_cached_generation_pending_token_and_global_position():
    s=Session()
    r=generate_cached('abc',tokenizer=Tokenizer(),session=s,max_new_tokens=3)
    assert r.generated_token_ids==(3,4,3)
    assert r.generated_text=='3 4 3'
    assert r.stop_reason=='max_new_tokens'
    assert s.calls==[('reset',),('prefill',(1,2)),('decode',3,2),('decode',4,3)]
    assert s.valid_length==4  # History is length 5; last selected token is pending.


@pytest.mark.parametrize('length,budget,selected,reason,calls',[
    (2,0,(3,),'max_new_tokens',0),(256,2,(3,),'capacity',0),
    (255,2,(3,),'capacity',2),(254,4,(3,4),'capacity',3),
    (2,1,(3,),'max_new_tokens',2),(2,3,(5,),'eos',2)])
def test_generation_boundaries(length,budget,selected,reason,calls):
    s=Session(selected)
    r=generate_cached('x',tokenizer=Tokenizer([1]*length),session=s,max_new_tokens=budget)
    assert r.stop_reason==reason
    assert len(s.calls)==calls


@pytest.mark.parametrize('tokens',[[],[1]*257])
def test_invalid_prompts_never_touch_session(tokens):
    s=Session()
    with pytest.raises(ValueError):
        generate_cached('x',tokenizer=Tokenizer(tokens),session=s,max_new_tokens=1)
    assert not s.calls


@pytest.mark.parametrize('kwargs,error',[
    ({'cancelled':lambda:True},GenerationCancelledError),
    ({'timeout_seconds':0},TimeoutError),({'max_new_tokens':-1},ValueError)])
def test_abort_before_calls(kwargs,error):
    s=Session()
    arguments=dict(tokenizer=Tokenizer(),session=s,max_new_tokens=1)
    arguments.update(kwargs)
    with pytest.raises(error):
        generate_cached('x',**arguments)
    assert not s.calls


def test_bad_logit_shape_finite_and_undecodable_ids():
    for bad in (np.zeros((1,256,8),np.float32),np.full((1,1,8),np.nan,np.float32)):
        with pytest.raises(ValueError):
            cached_logits(bad,8)
    with pytest.raises(TypeError):
        cached_logits(np.zeros((1,1,8),np.float64),8)
    with pytest.raises(ValueError,match='absent'):
        generate_cached('x',tokenizer=Tokenizer(),session=Session([7]),max_new_tokens=1)


def test_inputs_masks_and_cache_arrays_are_explicit():
    layout=KVCacheLayout(2,2,16,128)
    p=prepare_prefill([1,2,3],layout)
    assert p['logit_index'].tolist()==[2]
    assert p['attention_mask'][0,0,2,2]==0
    assert p['attention_mask'][0,0,2,3]<0
    cache=[np.zeros(layout.shape,np.float32) for _ in layout.cache_names]
    d=prepare_decode(4,3,layout,cache)
    assert d['position_ids'].tolist()==[[3]]
    assert d['attention_mask'][0,0,0,3]==0
    assert d['attention_mask'][0,0,0,4]<0
    assert d['past.layer_0.key'] is cache[0]
    assert layout.bank_bytes==2*2*2*256*16*4
    with pytest.raises(ValueError):
        prepare_decode(4,256,layout,cache)
    cache[-1].fill(np.inf)
    with pytest.raises(ValueError):
        prepare_decode(4,3,layout,cache)
    assert KVCacheLayout(28,8,128,151936).bank_bytes==56*1024**2
