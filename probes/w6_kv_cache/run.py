#!/usr/bin/env python3
"""Small Qwen3 KV reference/compiler/target gate. Artifacts stay in output/."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scratchv.runtime.kv_cache import prepare_prefill, prepare_decode


def verify_reference(directory, model, wrappers, layout):
    import torch
    import onnxruntime as ort
    sessions = [ort.InferenceSession(str(directory / f'{mode}.onnx'),
                                    providers=['CPUExecutionProvider'])
                for mode in ('prefill', 'decode')]
    rows = []
    def compare(label, actual, expected):
        errors = []
        for name, a, b in zip(layout.output_names, actual, expected):
            b = b.detach().numpy() if hasattr(b, 'detach') else b
            if a.shape != b.shape or a.dtype != b.dtype or not np.isfinite(a).all():
                raise AssertionError(f'{label}: invalid {name}')
            err = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
            if err > 1e-5:
                raise AssertionError(f'{label}: {name}: {err} > 1e-5')
            errors.append(err)
        rows.append({'case': label, 'max_abs_by_output': errors, 'passed': True})
    with torch.no_grad():
        for length in (1, 17, 255):
            tokens = [int(t) for t in np.random.default_rng(length).integers(1, layout.vocab_size, length)]
            feed = prepare_prefill(tokens, layout)
            cached = sessions[0].run(None, feed)
            pytorch = wrappers[0](*(torch.from_numpy(v) for v in feed.values()))
            compare(f'prefill-{length}', cached, pytorch)
            reference = model(input_ids=torch.from_numpy(feed['input_ids']),
                              attention_mask=torch.from_numpy(feed['attention_mask']),
                              use_cache=True)
            expected = [reference.logits[:, length-1:length]]
            for pair in reference.past_key_values:
                expected.extend(pair)
            compare(f'upstream-prefill-{length}', cached, expected)
            for step in range(min(4, layout.capacity-length)):
                token = int(np.argmax(cached[0][0, 0]))
                past = [a.copy() for a in cached[1:]]
                before = [a.copy() for a in past]
                # Finite poisoned padding must never enter attention.
                for a in past:
                    a[:, :, len(tokens):, :] = 13.0
                decode_feed = prepare_decode(token, len(tokens), layout, past)
                cached = sessions[1].run(None, decode_feed)
                native = wrappers[1](*(torch.from_numpy(v) for v in decode_feed.values()))
                compare(f'decode-{length}-{step}', cached, native)
                for a, b in zip(past, before):
                    np.testing.assert_array_equal(a[:, :, :len(tokens)], b[:, :, :len(tokens)])
                    assert np.all(a[:, :, len(tokens):] == 13)
                tokens.append(token)
                fresh = prepare_prefill(tokens, layout)
                full = sessions[0].run(None, fresh)
                # Compare valid prefix per head, never a flattened prefix.
                compare(f'cached-vs-full-{length}-{step}',
                        [cached[0], *(a[:, :, :len(tokens)] for a in cached[1:])],
                        [full[0], *(a[:, :, :len(tokens)] for a in full[1:])])
                changed = {k: v.copy() for k, v in decode_feed.items()}
                changed['past.layer_0.value'][:, :, :len(tokens)-1] += 0.5
                perturbed = sessions[1].run(None, changed)
                if np.array_equal(perturbed[0], cached[0]):
                    raise AssertionError('Decode ignored caller cache')
    return rows


def compile_models(directory, layout):
    from scratchv.frontend.onnx_parser import ONNXParser
    from scratchv.backend.tensor_c_codegen import TensorCCodegen
    artifacts = []
    for mode in ('prefill', 'decode'):
        parser = ONNXParser()
        program = parser.parse(str(directory/f'{mode}.onnx'), mmap_external_data=True)
        artifacts.append(TensorCCodegen(program, parser.initializers, constant_storage='external',
            function_name=f'sv_{mode}_v2', kernel_calls=True,
            max_constant_bytes=3*1024**3, max_workspace_bytes=1024**3).generate())
    layout.validate_graphs(*artifacts)
    return artifacts


def verify_compiled(directory, layout, artifacts, *, qemu=False, cores=1):
    import ctypes
    import shutil
    import subprocess
    import onnxruntime as ort
    from scratchv.verification.ir_interpreter import IRInterpreter
    from scratchv.frontend.onnx_parser import ONNXParser
    rows = []
    functions, weights, arenas, libraries = [], [], [], []
    pointers = lambda arrays: (ctypes.c_void_p*len(arrays))(*(a.ctypes.data for a in arrays))
    for mode, artifact in zip(('prefill','decode'), artifacts):
        path = directory/f'{mode}.c'
        path.write_text(artifact.source)
        binary = directory/f'{mode}.so'
        compiler = shutil.which('cc')
        if compiler is None:
            raise RuntimeError('Host C compiler required for the compiled numeric gate')
        subprocess.run([compiler,'-shared','-fPIC','-O2','-std=c11',*artifact.compile_flags,
                        str(path),'-lm','-o',str(binary)], check=True,capture_output=True,timeout=180)
        library = ctypes.CDLL(str(binary.resolve()))
        libraries.append(library)
        function = getattr(library,artifact.function_name)
        function.restype = ctypes.c_int
        ptr = ctypes.POINTER(ctypes.c_void_p)
        function.argtypes = [ptr,ptr,ctypes.c_void_p,ctypes.c_size_t,ptr]
        functions.append(function)
        weights.append([np.ascontiguousarray(a) for a in artifact.external_initializers])
        arenas.append(np.zeros((artifact.workspace_bytes+7)//8,np.uint64))
    def host(mode, feed):
        i = 0 if mode=='prefill' else 1
        artifact = artifacts[i]
        output = [np.empty(s.shape,s.numpy_dtype) for s in artifact.outputs]
        status = functions[i](pointers(list(feed.values())),pointers(weights[i]),arenas[i].ctypes.data,
                              artifact.workspace_bytes,pointers(output))
        if status:
            raise AssertionError(f'{mode}: host C status {status}')
        return output
    def compare(label, actual, expected, length):
        errors = []
        for i, (a,b) in enumerate(zip(actual,expected)):
            if i:
                a,b = a[:,:,:length],b[:,:,:length]
            if a.shape != b.shape or not np.isfinite(a).all():
                raise AssertionError(f'{label}: invalid output {i}')
            error = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
            if error>1e-5:
                raise AssertionError(f'{label}: output {i} error {error}')
            errors.append(error)
        rows.append({'case':label,'max_abs_by_output':errors,'passed':True})
    oracles = [ort.InferenceSession(str(directory/f'{mode}.onnx'),providers=['CPUExecutionProvider'])
               for mode in ('prefill','decode')]
    target = None
    if qemu:
        from scratchv.runtime.riscv_session import build_session,QemuSession
        from scratchv.runtime.riscv_tensor import discover_toolchain
        executable = build_session(*artifacts,layout,directory/f'build-{cores}',discover_toolchain(),cores=cores)
        target = QemuSession(executable,directory/f'qemu-{cores}',timeout=120)
        target.start()
    try:
        for length in (1,17,255):
            tokens = np.random.default_rng(length).integers(1,layout.vocab_size,length).tolist()
            feed = prepare_prefill(tokens,layout)
            expected = oracles[0].run(None,feed)
            output = host('prefill',feed)
            compare(f'host-prefill-{length}',output,expected,length)
            parser = ONNXParser()
            program = parser.parse(str(directory/'prefill.onnx'))
            interpreted = IRInterpreter(program).run(feed,initializers=parser.initializers,
                memory_mode='last_use',collect_memory_stats=True).return_values
            compare(f'ir-prefill-{length}',interpreted,expected,length)
            if target:
                target.reset()
                actual_logits = target.prefill(tokens)
                actual = [actual_logits,*target.dump_cache()]
                compare(f'qemu-{cores}-prefill-{length}',actual,expected,length)
                for a in actual[1:]:
                    assert np.all(a[:,:,length:]==0)
            for step in range(min(4,layout.capacity-length)):
                token = int(np.argmax(expected[0]))
                feed = prepare_decode(token,len(tokens),layout,output[1:])
                expected = oracles[1].run(None,feed)
                output = host('decode',feed)
                tokens.append(token)
                compare(f'host-decode-{length}-{step}',output,expected,len(tokens))
                parser = ONNXParser()
                program = parser.parse(str(directory/'decode.onnx'))
                interpreted = IRInterpreter(program).run(feed,initializers=parser.initializers,
                    memory_mode='last_use',collect_memory_stats=True).return_values
                compare(f'ir-decode-{length}-{step}',interpreted,expected,len(tokens))
                if target:
                    actual_logits = target.decode(token)
                    actual = [actual_logits,*target.dump_cache()]
                    compare(f'qemu-{cores}-decode-{length}-{step}',actual,expected,len(tokens))
                    assert int(actual_logits.argmax())==int(expected[0].argmax())
        if target:
            from scratchv.runtime.cached_generation import generate_cached
            class Tokenizer:
                model_vocab_size = layout.vocab_size
                pad_token_id = layout.pad_token_id
                generation_eos_token_ids = ()
                def encode(self, text, **kwargs): return [1,2,3]
                def validate_token_ids(self, tokens): return list(tokens)
                def decode(self, tokens, **kwargs): return ' '.join(map(str,tokens))
            before = len(target.runs)
            result = generate_cached('raw',tokenizer=Tokenizer(),session=target,max_new_tokens=4)
            history = [1,2,3]
            expected_tokens = []
            for _ in range(4):
                output = oracles[0].run(None,prepare_prefill(history,layout))[0]
                token = int(output.argmax())
                expected_tokens.append(token)
                history.append(token)
            assert result.generated_token_ids==tuple(expected_tokens)
            calls = target.runs[before:]
            assert [r['kind'] for r in calls]==[1,2,3,3,3]
            assert target.valid_length==6
            rows.append({'case':f'qemu-{cores}-greedy-generation','passed':True,
                         'generated_token_ids':list(result.generated_token_ids)})
        if target:
            target.reset()
            assert target.valid_length==0
            for a in target.dump_cache():
                assert np.all(a==0)
    finally:
        if target:
            target.close()
    return rows


def benchmark(directory, wrappers, layout, artifacts, cores):
    """Same weights/FP32 policies/prompt/budget; measure QEMU, not hardware speed."""
    import torch
    from scratchv.frontend.onnx_parser import ONNXParser
    from scratchv.backend.tensor_c_codegen import TensorCCodegen
    from scratchv.runtime.riscv_external import build_external
    from scratchv.runtime.riscv_session import build_session, QemuSession
    from scratchv.runtime.riscv_tensor import discover_toolchain
    from scratchv.runtime.weight_bundle import write_weight_bundle
    from scratchv.runtime.qemu_generation import QemuForward
    from scratchv.runtime.generation import generate
    from scratchv.runtime.cached_generation import generate_cached
    class FullLogits(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.prefill = wrappers[0]
            self.register_buffer('all_indices',torch.arange(256))
        def forward(self,input_ids,attention_mask):
            return self.prefill(input_ids,attention_mask,self.all_indices)[0]
    feed = prepare_prefill(list(range(1,18)),layout)
    path = directory/'uncached.onnx'
    with torch.no_grad():
        torch.onnx.export(FullLogits().eval(),
            (torch.from_numpy(feed['input_ids']),torch.from_numpy(feed['attention_mask'])),path,
            input_names=['input_ids','attention_mask'],output_names=['logits'],opset_version=18,dynamo=False)
    parser = ONNXParser()
    program = parser.parse(str(path),mmap_external_data=True)
    full = TensorCCodegen(program,parser.initializers,constant_storage='external',kernel_calls=True,
                          max_workspace_bytes=1024**3,max_constant_bytes=3*1024**3).generate()
    write_weight_bundle(full.external_weights,full.external_initializers,directory/'uncached-weights')
    toolchain = discover_toolchain()
    executable = build_external(full,directory/'uncached-weights',directory/'uncached-build',toolchain,cores=cores)
    cached_executable = build_session(*artifacts,layout,directory/'cached-bench-build',toolchain,cores=cores)
    class Tokenizer:
        model_vocab_size=layout.vocab_size
        pad_token_id=layout.pad_token_id
        generation_eos_token_ids=()
        def encode(self,text,**kwargs): return list(range(1,18))
        def validate_token_ids(self,tokens): return list(tokens)
        def decode(self,tokens,**kwargs): return ' '.join(map(str,tokens))
    forward = QemuForward(executable,directory/'uncached-steps',timeout=120)
    start = time.perf_counter()
    uncached = generate('raw',tokenizer=Tokenizer(),forward=forward,max_new_tokens=4)
    uncached_seconds = time.perf_counter()-start
    start = time.perf_counter()
    selection = {}
    with QemuSession(cached_executable,directory/'cached-bench-run',timeout=120) as session:
        cached = generate_cached('raw',tokenizer=Tokenizer(),session=session,max_new_tokens=4,timings=selection)
    cached_seconds = time.perf_counter()-start
    assert cached.generated_token_ids==uncached.generated_token_ids
    result = {'scope':'QEMU/TCG wall time; not real RISC-V hardware', 'cores':cores,
              'warmups':0,'prompt_length':17,'max_new_tokens':4,'policies':'FP32 sequential/libm',
              'uncached_total_s':uncached_seconds,'cached_total_s':cached_seconds,
              'uncached_runs':forward.runs,'cached_session':session.report,'cached_generation_timings':selection,
              'generated_token_ids':list(cached.generated_token_ids),'passed':True}
    (directory/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--qemu', action='store_true')
    parser.add_argument('--benchmark', action='store_true',help='Also measure cached vs uncached QEMU generation')
    parser.add_argument('--cores', type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args()
    from scripts.export_qwen3_kv_onnx import export_models
    model, wrappers, layout, report = export_models(args.output_dir, model_dir=args.model_dir)
    report['reference'] = verify_reference(args.output_dir, model, wrappers, layout)
    artifacts = compile_models(args.output_dir, layout)
    report['compiled'] = verify_compiled(args.output_dir, layout, artifacts, qemu=args.qemu, cores=args.cores)
    if args.benchmark:
        report['benchmark'] = benchmark(args.output_dir,wrappers,layout,artifacts,args.cores)
    report['passed'] = True
    report['status'] = 'PASS'
    report['memory'] = {'prefill_workspace_bytes': artifacts[0].workspace_bytes,
                        'decode_workspace_bytes': artifacts[1].workspace_bytes,
                        'cache_bank_bytes': layout.bank_bytes}
    report['seconds'] = time.perf_counter() - started
    (args.output_dir / 'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print('PASS: PyTorch/ORT/IR/Host C' + ('/QEMU' if args.qemu else '') + ' KV multi-step gate')


if __name__ == '__main__':
    main()
