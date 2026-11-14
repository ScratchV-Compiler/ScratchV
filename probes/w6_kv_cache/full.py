#!/usr/bin/env python3
"""Process-isolated validation of the pinned, pretrained Qwen3-0.6B KV graphs.

Each stage is resumable in the same directory and publishes a separate report.
Large tensors/weights are never required to coexist with all reference engines.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scratchv.runtime.kv_cache import KVCacheLayout, prepare_prefill, prepare_decode
from scratchv.runtime.qwen3_tokenizer import Qwen3Tokenizer

# Match the pinned text tokenizer so the validated executable is reusable by
# scripts/run_inference.py without changing its padding contract.
LAYOUT = KVCacheLayout(28, 8, 128, 151936, pad_token_id=151643)
ATOL = 1e-3
POLICIES = dict(matmul_policy='blocked_fma', matmul_tile='2x4',
                matmul_rhs_layout='column-panel', mean_policy='two-four-lane-last-axis',
                nonlinear_policy='fp32-reference')


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def publish_summary(directory):
    """Keep a compact report beside executable.json for audited CLI reuse."""
    names = ('torch', 'ort-prefill', 'ort-decode', 'baseline', 'build', 'qemu', 'host')
    records = {name: json.loads((directory/f'{name}.json').read_text())
               for name in names if (directory/f'{name}.json').exists()}
    complete = len(records) == len(names) and all(r.get('passed') for r in records.values())
    failed = any(r.get('status') == 'FAIL' for r in records.values())
    result = dict(schema='scratchv.full-kv-validation.v1', model='Qwen/Qwen3-0.6B',
                  passed=complete, status='PASS' if complete else 'FAIL' if failed else 'INCOMPLETE',
                  full_model_gate='PASS_NUMERIC_RESOURCE' if complete else 'INCOMPLETE',
                  uncached_qemu_performance_comparison='NOT_RUN',
                  layout=LAYOUT.to_dict(), policies=POLICIES, atol=ATOL, stages={})
    for name, record in records.items():
        steps = record.get('steps', [])
        errors = [step['comparison']['max_abs'] if 'comparison' in step else step['max_abs']
                  for step in steps]
        result['stages'][name] = dict(status=record['status'], passed=record.get('passed', False),
            seconds=record.get('seconds'), peak_rss_bytes=record.get('peak_rss_bytes'),
            checked_steps=len(steps), max_abs=max(errors, default=None), report=f'{name}.json')
    build = records.get('build', {})
    for name in ('executable_sha256', 'weight_bytes', 'unshared_weight_bytes', 'layout'):
        if name in build:
            result['target_memory' if name == 'layout' else name] = build[name]
    qemu = records.get('qemu', {})
    for name in ('generated_token_ids', 'generated_text', 'generation', 'generation_timings'):
        if name in qemu:
            result[name] = qemu[name]
    if 'session' in qemu:
        result['qemu_session_report'] = 'qemu/session.json'
    result['scope'] = dict(reference='four cases, ten steps, all 57 outputs; uncached logits',
        host='single host core; text four steps and capacity boundary two steps',
        qemu=f'{qemu.get("cores", build.get("cores", 4))} target cores; text four steps, all 57 outputs, RESET and cleanup',
        timing='audited generation includes full cache dumps and numerical comparisons')
    save_json(directory/'report.json', result)


def output_path(directory, backend, case, step):
    path = directory/'traces'/backend/case/f'step-{step:02d}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_outputs(path, arrays):
    if len(arrays) != len(LAYOUT.output_names):
        raise ValueError('Expected all 57 results')
    np.savez(path, **dict(zip(LAYOUT.output_names, arrays)))


def load_outputs(path):
    with np.load(path) as data:
        return [data[name].copy() for name in LAYOUT.output_names]


def clear_padding(cache, length):
    for array in cache:
        array[:, :, length:, :] = 0


def compare(actual, expected, length, *, check_greedy=True):
    if len(actual) != 57 or len(expected) != 57:
        raise ValueError('Comparison must include all 57 results')
    rows = []
    for index, (name, a, b) in enumerate(zip(LAYOUT.output_names, actual, expected)):
        shape = (1, 1, LAYOUT.vocab_size) if index == 0 else LAYOUT.shape
        if a.shape != shape or b.shape != shape or a.dtype != np.float32 or b.dtype != np.float32:
            raise ValueError(f'{name}: invalid shape/dtype')
        # Validate the complete physical tensor; compare valid tokens per head.
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f'{name}: nonfinite result')
        av = a.ravel() if index == 0 else a[:, :, :length].ravel()
        bv = b.ravel() if index == 0 else b[:, :, :length].ravel()
        maximum, square, bad = 0.0, 0.0, 0
        for start in range(0, len(av), 65536):
            diff = np.abs(av[start:start+65536].astype(np.float64)-bv[start:start+65536].astype(np.float64))
            maximum = max(maximum, float(diff.max(initial=0)))
            square += float(np.dot(diff, diff))
            bad += int(np.count_nonzero(diff >= ATOL))
        rows.append(dict(name=name, max_abs=maximum, rmse=float(np.sqrt(square/len(av))),
                         elements=len(av), elements_at_or_above_atol=bad, passed=maximum < ATOL))
    greedy_equal = int(actual[0].argmax()) == int(expected[0].argmax())
    result = dict(passed=all(row['passed'] for row in rows) and (greedy_equal or not check_greedy),
                  atol=ATOL, rtol=0, criterion='strict max_abs < 1e-3; all valid KV elements per head',
                  max_abs=max(row['max_abs'] for row in rows), greedy_equal=greedy_equal, outputs=rows)
    if not result['passed']:
        failures = [(r['name'], r['max_abs']) for r in rows if not r['passed']]
        raise AssertionError(f'Numerical comparison failed: {failures}; greedy_equal={greedy_equal}; '
                             f'max_abs={result["max_abs"]}')
    return result


def cases():
    tokenizer = Qwen3Tokenizer.from_directory(ROOT/'assets/qwen3-0.6b/tokenizer')
    ids = tokenizer.encode('The capital of France is', add_special_tokens=False)
    return [dict(name='paris', tokens=ids, steps=4),
            dict(name='one_token', tokens=[42], steps=2),
            dict(name='short_17', tokens=np.random.default_rng(17).integers(1,151643,17).tolist(),steps=2),
            dict(name='boundary_255', tokens=np.random.default_rng(255).integers(1,151643,255).tolist(),steps=2)]


def torch_stage(args, report):
    import torch
    from scripts.export_qwen3_kv_onnx import load_model, make_wrappers
    torch.set_num_threads(args.threads)
    model = load_model(args.model_dir)
    wrappers = make_wrappers(model)
    entries = []
    with torch.inference_mode():
        for case in cases():
            history = case['tokens'].copy()
            cache = None
            for step in range(case['steps']):
                started = time.perf_counter()
                mode = 'prefill' if step == 0 else 'decode'
                feed = (prepare_prefill(history, LAYOUT) if step == 0 else
                        prepare_decode(history[-1], len(history)-1, LAYOUT, cache))
                wrapper = wrappers[0 if step == 0 else 1]
                native = wrapper(*(torch.from_numpy(v) for v in feed.values()))
                actual = [v.numpy().copy() for v in native]
                # Independent upstream model with a fresh Cache on each full forward.
                full_feed = prepare_prefill(history, LAYOUT)
                upstream = model(input_ids=torch.from_numpy(full_feed['input_ids']),
                    attention_mask=torch.from_numpy(full_feed['attention_mask']), use_cache=True)
                expected = [upstream.logits[:,len(history)-1:len(history)].numpy()]
                for pair in upstream.past_key_values:
                    expected.extend(v.numpy() for v in pair)
                comparison = compare(actual, expected, len(history))
                selected = int(actual[0].argmax())
                path = output_path(args.output_dir,'torch',case['name'],step)
                save_outputs(path, actual)
                entry = dict(case=case['name'],step=step,mode=mode,tokens=history.copy(),
                             valid_length=len(history), selected_token=selected, comparison=comparison,
                             seconds=time.perf_counter()-started,
                             trace=str(path.relative_to(args.output_dir)))
                entries.append(entry)
                report['steps'] = entries
                save_json(args.output_dir/'torch.json',report)
                print(f'TORCH {case["name"]} {mode} length={len(history)} max_abs={comparison["max_abs"]:.8g} token={selected}',flush=True)
                cache = actual[1:]
                clear_padding(cache,len(history))
                if step+1 < case['steps']:
                    history.append(selected)
                del native,upstream,expected,actual
    tokenizer = Qwen3Tokenizer.from_directory(ROOT/'assets/qwen3-0.6b/tokenizer')
    ids = [r['selected_token'] for r in entries if r['case']=='paris']
    report['paris'] = dict(prompt='The capital of France is',generated_token_ids=ids,
                          generated_text=tokenizer.decode(ids,skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=False))


def ort_session(path, threads):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads=threads
    options.inter_op_num_threads=1
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_cpu_mem_arena=False
    options.enable_mem_pattern=False
    return ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])


def ort_stage(args, report, mode):
    session = ort_session(args.output_dir/f'{mode}.onnx',args.threads)
    golden = json.loads((args.output_dir/'torch.json').read_text())['steps']
    report['steps'] = []
    cache, last_case = None, None
    for entry in golden:
        if entry['mode'] != mode:
            continue
        if mode == 'prefill':
            feed=prepare_prefill(entry['tokens'],LAYOUT)
        else:
            if entry['case'] != last_case:
                previous=load_outputs(output_path(args.output_dir,'ort',entry['case'],0))
                cache=previous[1:]
                clear_padding(cache,len(entry['tokens'])-1)
                del previous
            feed=prepare_decode(entry['tokens'][-1],len(entry['tokens'])-1,LAYOUT,cache)
        started=time.perf_counter()
        actual=session.run(None,feed)
        expected=load_outputs(args.output_dir/entry['trace'])
        comparison=compare(actual,expected,entry['valid_length'])
        if mode=='decode':
            # Verify immutable caller inputs and masking with independent reruns.
            before=[a.copy() for a in cache]
            again=session.run(None,feed)
            for a,b in zip(cache,before):np.testing.assert_array_equal(a,b)
            compare(again,actual,entry['valid_length'])
            if entry['case']=='paris' and entry['step']==1:
                poisoned=[a.copy() for a in cache]
                for a in poisoned:a[:,:,entry['valid_length']-1:]=13
                poisoned_feed=prepare_decode(entry['tokens'][-1],entry['valid_length']-1,LAYOUT,poisoned)
                compare(session.run(None,poisoned_feed),actual,entry['valid_length'])
                changed=[a.copy() for a in cache]
                changed[1][:,:,:entry['valid_length']-1]+=0.5
                delta=session.run(None,prepare_decode(entry['tokens'][-1],entry['valid_length']-1,LAYOUT,changed))[0]
                if np.array_equal(delta,actual[0]):raise AssertionError('Decode ignored historical values')
                report['cache_immutability_padding_and_sensitivity']=True
            del before,again
        save_outputs(output_path(args.output_dir,'ort',entry['case'],entry['step']),actual)
        report['steps'].append(dict(case=entry['case'],step=entry['step'],valid_length=entry['valid_length'],
                                    comparison=comparison,seconds=time.perf_counter()-started))
        print(f'ORT {mode} {entry["case"]} step={entry["step"]} max_abs={comparison["max_abs"]:.8g}',flush=True)
        cache=actual[1:]
        clear_padding(cache,entry['valid_length'])
        last_case=entry['case']
        del actual,expected


def baseline_stage(args, report):
    session=ort_session(ROOT/'assets/qwen3-0.6b/onnx/model.onnx',args.threads)
    golden=json.loads((args.output_dir/'torch.json').read_text())['steps']
    report['steps']=[]
    for entry in golden:
        feed=prepare_prefill(entry['tokens'],LAYOUT)
        actual=session.run(None,{k:feed[k] for k in ('input_ids','attention_mask')})[0]
        with np.load(args.output_dir/entry['trace']) as archive:expected=archive['logits']
        row=actual[:,entry['valid_length']-1:entry['valid_length']]
        if not np.isfinite(actual).all():raise ValueError('Nonfinite uncached logits')
        delta=np.abs(row.astype(np.float64)-expected.astype(np.float64))
        maximum=float(delta.max())
        greedy=int(row.argmax())==int(expected.argmax())
        if maximum>=ATOL or not greedy:raise AssertionError(f'Uncached baseline differs: {maximum}, greedy={greedy}')
        report['steps'].append(dict(case=entry['case'],step=entry['step'],max_abs=maximum,
                                    greedy_equal=greedy,passed=True))
        print(f'BASELINE {entry["case"]} step={entry["step"]} max_abs={maximum:.8g}',flush=True)
        del actual,expected


def compile_artifacts(directory):
    from scratchv.frontend.onnx_parser import ONNXParser
    from scratchv.backend.tensor_c_codegen import TensorCCodegen
    artifacts=[]
    for mode in ('prefill','decode'):
        parser=ONNXParser()
        program=parser.parse(str(directory/f'{mode}.onnx'),mmap_external_data=True)
        artifact=TensorCCodegen(program,parser.initializers,constant_storage='external',
            function_name=f'sv_{mode}_v2',kernel_calls=True,max_constant_bytes=3*1024**3,
            max_workspace_bytes=1024**3,**POLICIES).generate()
        artifacts.append(artifact)
        print(f'CODEGEN {mode}: workspace={artifact.workspace_bytes}, outputs={len(artifact.outputs)}',flush=True)
    LAYOUT.validate_graphs(*artifacts)
    return artifacts


def build_stage(args,report):
    from scratchv.runtime.riscv_session import build_session,save_session_executable
    from scratchv.runtime.riscv_tensor import discover_toolchain
    artifacts=compile_artifacts(args.output_dir)
    for mode,a in zip(('prefill','decode'),artifacts):
        (args.output_dir/f'{mode}.c').write_text(a.source)
    toolchain=discover_toolchain()
    executable=build_session(*artifacts,LAYOUT,args.output_dir/'build',toolchain,cores=args.cores,timeout=900)
    executable=replace(executable,evidence={**executable.evidence,'policies':POLICIES})
    report['executable_sha256']=save_session_executable(executable,args.output_dir)
    report['build']=executable.evidence
    report['layout']=executable.layout
    report['weight_bytes']=executable.manifest['total_bytes']
    report['unshared_weight_bytes']=sum(a.constant_bytes for a in artifacts)
    print(f'BUILD PASS: shared={report["weight_bytes"]} vs separate={report["unshared_weight_bytes"]}',flush=True)


def host_stage(args,report):
    import ctypes
    import shutil
    import subprocess
    from scratchv.runtime.riscv_session import load_session_executable
    exe=load_session_executable(args.output_dir/'executable.json')
    rows=exe.evidence['weight_mappings']
    arrays=[np.memmap(exe.bundle_dir/'weights.bin',dtype=np.dtype({'f32':'<f4','i32':'<i4','i64':'<i8'}[s['dtype']]),
                      mode='r',offset=s['offset'],shape=tuple(s['shape'])) for s in exe.manifest['tensors']]
    arena=np.empty((exe.layout['regions']['workspace']['nbytes']+7)//8,np.uint64)
    ptr=ctypes.POINTER(ctypes.c_void_p)
    pointers=lambda a:(ctypes.c_void_p*len(a))(*(x.ctypes.data for x in a))
    functions,libraries=[],[]
    for mode in ('prefill','decode'):
        source=args.output_dir/f'{mode}.c';binary=args.output_dir/f'{mode}.so'
        subprocess.run([shutil.which('cc'),'-shared','-fPIC','-O2','-march=native','-std=c11',
            '-fno-fast-math','-ffp-contract=off','-fno-strict-aliasing',str(source),'-lm','-o',str(binary)],
            check=True,capture_output=True,timeout=900)
        library=ctypes.CDLL(str(binary));libraries.append(library)
        function=getattr(library,f'sv_{mode}_v2');function.restype=ctypes.c_int
        function.argtypes=[ptr,ptr,ctypes.c_void_p,ctypes.c_size_t,ptr];functions.append(function)
    report['steps']=[]
    cache=None
    for entry in json.loads((args.output_dir/'torch.json').read_text())['steps']:
        # Full C verifies both ordinary text and the final physical cache slot.
        if entry['case'] not in ('paris','boundary_255'):continue
        mode=0 if entry['mode']=='prefill' else 1
        feed=prepare_prefill(entry['tokens'],LAYOUT) if mode==0 else prepare_decode(
            entry['tokens'][-1],entry['valid_length']-1,LAYOUT,cache)
        output=[np.empty((1,1,LAYOUT.vocab_size),np.float32),
                *[np.empty(LAYOUT.shape,np.float32) for _ in LAYOUT.cache_names]]
        started=time.perf_counter()
        status=functions[mode](pointers(list(feed.values())),pointers([arrays[i] for i in rows[mode]]),
                               arena.ctypes.data,arena.nbytes,pointers(output))
        if status:raise RuntimeError(f'Host C returned {status}')
        expected=load_outputs(output_path(args.output_dir,'ort',entry['case'],entry['step']))
        comparison=compare(output,expected,entry['valid_length'])
        report['steps'].append(dict(case=entry['case'],step=entry['step'],comparison=comparison,
                                    seconds=time.perf_counter()-started))
        save_outputs(output_path(args.output_dir,'host',entry['case'],entry['step']),output)
        print(f'HOST {entry["case"]} step={entry["step"]} max_abs={comparison["max_abs"]:.8g}',flush=True)
        cache=output[1:];clear_padding(cache,entry['valid_length'])


def qemu_stage(args,report):
    from scratchv.runtime.riscv_session import QemuSession,load_session_executable
    from probes.w4_qwen3_full.fma_conformance import run_probe
    exe=load_session_executable(args.output_dir/'executable.json')
    if exe.cores != args.cores or exe.evidence.get('policies') != POLICIES:
        raise ValueError('Reused target core count or arithmetic policies differ')
    report['fma_conformance']=run_probe(args.output_dir/'fma-conformance',cc=exe.toolchain.cc,qemu=exe.toolchain.qemu)
    if not report['fma_conformance'].get('passed'):raise AssertionError('FMA preflight failed')
    entries=[r for r in json.loads((args.output_dir/'torch.json').read_text())['steps'] if r['case']=='paris']
    report['steps']=[]
    session=QemuSession(exe,args.output_dir/'qemu',timeout=args.timeout)
    try:
        from dataclasses import asdict
        from scratchv.runtime.cached_generation import generate_cached
        tokenizer=Qwen3Tokenizer.from_directory(ROOT/'assets/qwen3-0.6b/tokenizer')
        class AuditedSession:
            def reset(self):
                return session.reset()
            def check(self,logits,started):
                entry=entries[len(report['steps'])]
                actual=[logits,*session.dump_cache()]
                expected=load_outputs(output_path(args.output_dir,'ort','paris',entry['step']))
                comparison=compare(actual,expected,entry['valid_length'])
                for array in actual[1:]:
                    if not np.all(array[:,:,entry['valid_length']:]==0):
                        raise AssertionError('Nonzero unused cache tail')
                if session.valid_length!=entry['valid_length']:
                    raise AssertionError('Host/guest length mismatch')
                token=int(logits.argmax())
                if token!=entry['selected_token']:raise AssertionError('Greedy token differs')
                save_outputs(output_path(args.output_dir,'qemu','paris',entry['step']),actual)
                report['steps'].append(dict(step=entry['step'],mode=entry['mode'],
                    valid_length=session.valid_length,selected_token=token,comparison=comparison,
                    seconds=time.perf_counter()-started))
                report['session']=session.report
                save_json(args.output_dir/'qemu.json',report)
                print(f'QEMU {entry["mode"]} step={entry["step"]} length={session.valid_length} '
                      f'max_abs={comparison["max_abs"]:.8g} token={token}',flush=True)
                return logits
            def prefill(self,tokens):
                if list(tokens)!=entries[0]['tokens']:
                    raise AssertionError('Tokenizer prompt differs from reference')
                started=time.perf_counter()
                return self.check(session.prefill(tokens),started)
            def decode(self,token):
                entry=entries[len(report['steps'])]
                if token!=entry['tokens'][-1]:raise AssertionError('Pending token differs')
                started=time.perf_counter()
                return self.check(session.decode(token),started)
        with session:
            timings={}
            result=generate_cached('The capital of France is',tokenizer=tokenizer,
                session=AuditedSession(),max_new_tokens=4,timings=timings)
            if result.generated_token_ids!=tuple(entry['selected_token'] for entry in entries):
                raise AssertionError('Generated tokens differ from reference')
            report['generation']=asdict(result)
            report['generation_timings']=timings
            if session.valid_length!=len(result.prompt_token_ids)+len(result.generated_token_ids)-1:
                raise AssertionError('Last selected token must remain pending')
            session.reset()
            if session.valid_length!=0 or any(np.any(array) for array in session.dump_cache()):
                raise AssertionError('RESET did not clear complete cache bank')

    finally:
        report['session']=session.report
    report['generated_token_ids']=[r['selected_token'] for r in report['steps']]
    tokenizer=Qwen3Tokenizer.from_directory(ROOT/'assets/qwen3-0.6b/tokenizer')
    report['generated_text']=tokenizer.decode(report['generated_token_ids'],skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--model-dir',type=Path,default=ROOT/'output/qwen3-0.6b-source')
    parser.add_argument('--stage',required=True,choices=('torch','ort-prefill','ort-decode','baseline','build','host','qemu'))
    parser.add_argument('--cores',type=int,choices=(1,2,4),default=4)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--timeout',type=float,default=3600)
    args=parser.parse_args()
    args.output_dir=args.output_dir.resolve();args.model_dir=args.model_dir.resolve()
    report_path=args.output_dir/f'{args.stage}.json'
    if report_path.exists() and json.loads(report_path.read_text()).get('passed'):
        raise FileExistsError(f'Already successful: {report_path}')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    report=dict(schema='scratchv.full-kv-stage.v1',stage=args.stage,passed=False,status='RUNNING',
        model='Qwen/Qwen3-0.6B',layers=28,dtype='float32',capacity=256,cores=args.cores,
        pad_token_id=LAYOUT.pad_token_id,
        policies=POLICIES,atol=ATOL)
    save_json(report_path,report)
    try:
        if args.stage=='torch':torch_stage(args,report)
        elif args.stage.startswith('ort-'):ort_stage(args,report,args.stage.removeprefix('ort-'))
        elif args.stage=='baseline':baseline_stage(args,report)
        elif args.stage=='build':build_stage(args,report)
        elif args.stage=='host':host_stage(args,report)
        elif args.stage=='qemu':qemu_stage(args,report)
        report.update(passed=True,status='PASS')
    except BaseException as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        raise
    finally:
        report['seconds']=time.perf_counter()-started
        report['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        save_json(report_path,report)
        publish_summary(args.output_dir)
    print(f'PASS {args.stage} ({report["seconds"]:.3f}s)',flush=True)


if __name__=='__main__':main()
