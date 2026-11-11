"""Resident bare-metal KV session. All model arithmetic executes in RV64/QEMU."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import signal
import socket
import struct
import subprocess
import threading
import time

import numpy as np
from . import kv_protocol as protocol
from .kv_cache import KVCacheLayout, cached_logits, prepare_prefill
from .llm_inputs import _integer
from .riscv_external import (QMP, _check_cancelled, _QemuMemoryMonitor,
                             sha256_file, validate_external_elf)
from .riscv_tensor import (LINKER_SCRIPT, RiscVToolchain, _creation_flags, _run_process,
                           _save_process_logs, _stop_process_tree, _terminate_invocation,
                           _windows_job, toolchain_versions, STARTUP_ASSEMBLY as SINGLE_STARTUP)
from .riscv_harts import CORE_DEFINITIONS, STARTUP_ASSEMBLY, WORKER_SOURCE, validate_cores
from .weight_bundle import (MIB, plan_session_memory, validate_weight_bundle,
                            write_shared_weight_bundle)
from .weight_transport import stage_transport, remove_transport


def _offsets(specs):
    offsets, cursor = [], 0
    for spec in specs:
        cursor = (cursor+63)//64*64
        offsets.append(cursor)
        cursor += spec.nbytes
    return tuple(offsets), cursor


@dataclass(frozen=True)
class SessionExecutable:
    elf_path: Path
    bundle_dir: Path
    manifest: dict
    weights: tuple
    layout: dict
    cache_layout: KVCacheLayout
    toolchain: RiscVToolchain
    evidence: dict
    cores: int


def _guest_source(prefill, decode, manifest, mappings, layout, cache, identity):
    po, _ = _offsets(prefill.inputs)
    do, _ = _offsets(decode.inputs[:3])
    def weights(indices):
        return ','.join(f'(const void *)0x{layout["weights_base"]+manifest["tensors"][i]["offset"]:x}UL'
                        for i in indices) or '0'
    pptrs = ','.join(f'(const void *)0x{layout["prefill_input_base"]+offset:x}UL' for offset in po)
    dptrs = ','.join(f'(const void *)0x{layout["decode_input_base"]+offset:x}UL' for offset in do)
    tensor_bytes = int(np.prod(cache.shape))*4
    # WORKER_SOURCE's trap path uses frame(status, size), which always faults.
    source = CORE_DEFINITIONS + r'''
#include <stdint.h>
#include <stddef.h>
typedef unsigned long usize;
typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;
void *memcpy(void *dst, const void *src, usize n) {
    u8 *d=dst; const u8 *s=src; for (usize i=0;i<n;++i) d[i]=s[i]; return dst;
}
void *memset(void *dst, int v, usize n) {
    u8 *d=dst; for (usize i=0;i<n;++i) d[i]=(u8)v; return dst;
}
static u64 session_id, request_id;
static u32 valid_length, bank;
static u8 header[56], payload[2048];
static void uart_byte(u8 v) {
    volatile u8 *u=(volatile u8 *)0x10000000UL;
    while (!(u[5]&0x20)) {} u[0]=v;
}
static u8 uart_read(void) {
    volatile u8 *u=(volatile u8 *)0x10000000UL;
    while (!(u[5]&1)) {} return u[0];
}
static u32 get32(const u8 *s) {
    u32 v=0; for (u32 i=0;i<4;++i) v|=(u32)s[i]<<(8*i); return v;
}
static u64 get64(const u8 *s) {
    u64 v=0; for (u32 i=0;i<8;++i) v|=(u64)s[i]<<(8*i); return v;
}
static void put32(u8 *s,u32 v) { for(u32 i=0;i<4;++i)s[i]=(u8)(v>>(8*i)); }
static void put64(u8 *s,u64 v) { for(u32 i=0;i<8;++i)s[i]=(u8)(v>>(8*i)); }
static u32 hash(u32 v,const u8 *s,u32 n) {
    for(u32 i=0;i<n;++i)v=(v^s[i])*16777619U; return v;
}
static void send_frame(u32 kind,u32 status,const u8 *data,u32 bytes,u32 logits) {
    u8 h[56]; memset(h,0,56); memcpy(h,"SVKV0001",8);
    put32(h+8,1);put32(h+12,kind);put32(h+16,bytes);put32(h+20,status);
    put64(h+24,session_id);put64(h+32,request_id);
    put32(h+40,valid_length);put32(h+44,logits);put32(h+52,256);
    put32(h+48,hash(hash(2166136261U,h,56),data,bytes));
    __asm__ volatile("fence rw,rw" ::: "memory");
    for(u32 i=0;i<56;++i)uart_byte(h[i]);
    for(u32 i=0;i<bytes;++i)uart_byte(data[i]);
}
static void frame(u32 status,u64 unused) {
    (void)unused;send_frame(13,status,0,0,0);
    for(;;)__asm__ volatile("wfi");
}
''' + WORKER_SOURCE + f'''
extern int {prefill.function_name}(const void *const *,const void *const *,void *,usize,void *const *);
extern int {decode.function_name}(const void *const *,const void *const *,void *,usize,void *const *);
static const u8 build_id[]="{identity}";
void sv_main(void) {{
    session_id=*(const u64 *)0x{layout['prefill_input_base']:x}UL;
    const void *pinputs[]={{ {pptrs} }};
    const void *dinputs[{len(decode.inputs)}]={{ {dptrs} }};
    const void *pweights[]={{ {weights(mappings[0])} }};
    const void *dweights[]={{ {weights(mappings[1])} }};
    void *outputs[{len(cache.output_names)}];
    u8 *banks[2]={{(u8 *)0x{layout['cache_bases'][0]:x}UL,(u8 *)0x{layout['cache_bases'][1]:x}UL}};
    send_frame(10,0,build_id,64,0);
    for(;;) {{
        for(u32 i=0;i<56;++i)header[i]=uart_read();
        u32 bytes=get32(header+16),kind=get32(header+12),digest=get32(header+48);
        if(bytes>2048)frame(100,0);
        for(u32 i=0;i<bytes;++i)payload[i]=uart_read();
        put32(header+48,0);
        if(memcmp_placeholder) {{}}
        for(u32 i=0;i<8;++i)if(header[i]!=(u8)"SVKV0001"[i])frame(101,0);
        if(get32(header+8)!=1 || get32(header+20) || get32(header+44) ||
           get32(header+52)!=256 || get64(header+24)!=session_id ||
           get64(header+32)!=request_id+1 || get32(header+40)!=valid_length ||
           hash(hash(2166136261U,header,56),payload,bytes)!=digest)frame(102,0);
        request_id=get64(header+32);
        if(kind==4) {{
            if(bytes)frame(103,0);
            send_frame(11,0,0,0,0);for(;;)__asm__ volatile("wfi");
        }}
        if(kind==1) {{
            if(bytes)frame(104,0);
            memset(banks[0],0,{layout['cache_bank_bytes']}UL);
            memset(banks[1],0,{layout['cache_bank_bytes']}UL);
            valid_length=0;bank=0;send_frame(11,0,0,0,0);continue;
        }}
        u32 next_length,target;
        if(kind==2) {{
            if(valid_length || !bytes || bytes%8 || bytes>2048)frame(105,0);
            next_length=bytes/8;target=0;
        }} else if(kind==3) {{
            if(!valid_length || valid_length>=256 || bytes!=8)frame(106,0);
            next_length=valid_length+1;target=1-bank;
        }} else {{frame(107,0);continue;}}
        for(u32 i=0;i<bytes/8;++i)if(get64(payload+8*i)>={cache.vocab_size}ULL)frame(108,0);
        outputs[0]=(void *)0x{layout['logits_base']:x}UL;
        for(u32 i=0;i<{len(cache.cache_names)};++i) {{
            outputs[i+1]=banks[target]+i*{tensor_bytes}UL;
            dinputs[i+3]=banks[bank]+i*{tensor_bytes}UL;
        }}
        int status;
        if(kind==2) {{
            int64_t *ids=(int64_t *)pinputs[0];
            float *mask=(float *)pinputs[1];
            for(u32 i=0;i<256;++i)ids[i]=i<next_length ? (int64_t)get64(payload+8*i) : {cache.pad_token_id};
            for(u32 q=0;q<256;++q)for(u32 k=0;k<256;++k)
                mask[q*256+k]=(k<=q && k<next_length) ? 0.0f : -0x1.fffffep127f;
            *(int64_t *)pinputs[2]=next_length-1;
            status={prefill.function_name}(pinputs,pweights,(void *)0x{layout['workspace_base']:x}UL,
                                           {layout['regions']['workspace']['nbytes']}UL,outputs);
        }} else {{
            *(int64_t *)dinputs[0]=(int64_t)get64(payload);
            *(int64_t *)dinputs[2]=valid_length;
            for(u32 k=0;k<256;++k)((float *)dinputs[1])[k]=k<=valid_length ? 0.0f : -0x1.fffffep127f;
            status={decode.function_name}(dinputs,dweights,(void *)0x{layout['workspace_base']:x}UL,
                                         {layout['regions']['workspace']['nbytes']}UL,outputs);
        }}
        sv_check_workers();
        if(status)frame((u32)status,0);
        /* Check every result before publishing bank/length, including padding. */
        for(u32 i=0;i<{len(cache.output_names)};++i) {{
            u32 count=i ? {tensor_bytes//4}U : {cache.vocab_size}U;
            const u32 *bits=(const u32 *)outputs[i];
            for(u32 j=0;j<count;++j)if((bits[j]&0x7f800000U)==0x7f800000U)frame(109,0);
        }}
        if(kind==2)for(u32 i=1;i<{len(cache.output_names)};++i)
            for(u32 h=0;h<{cache.kv_heads};++h)
                memset((float *)outputs[i]+(h*256+next_length)*{cache.head_dim},0,
                       (256-next_length)*{cache.head_dim*4}UL);
        bank=target;valid_length=next_length;
        send_frame(12,0,0,0,{cache.vocab_size*4});
    }}
}}
'''
    return source.replace('        if(memcmp_placeholder) {}\n', '')


def build_session(prefill, decode, cache_layout, build_dir, toolchain, *, cores=1, timeout=600):
    """Link separately compiled Prefill/Decode into one code-only ELF and bundle."""
    cores = validate_cores(cores)
    cache_layout.validate_graphs(prefill, decode)
    if prefill.function_name == decode.function_name:
        raise ValueError('Prefill and Decode entry points must differ')
    build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=False)
    manifest, weights, mappings = write_shared_weight_bundle((prefill, decode), build_dir/'weights')
    _, pi = _offsets(prefill.inputs)
    _, di = _offsets(decode.inputs[:3])
    layout = plan_session_memory(cache_layout.bank_bytes, prefill_input_bytes=pi,
        decode_input_bytes=di, prefill_workspace_bytes=prefill.workspace_bytes,
        decode_workspace_bytes=decode.workspace_bytes, logits_bytes=prefill.outputs[0].nbytes,
        weight_bytes=manifest['total_bytes'], cores=cores)
    identity = hashlib.sha256((prefill.source+decode.source+json.dumps(manifest,sort_keys=True)
                              +json.dumps(cache_layout.to_dict(),sort_keys=True)+str(cores)).encode()).hexdigest()
    linker = LINKER_SCRIPT.replace('0x1c000000', hex(layout['code_capacity']))
    if cores != 1:
        linker = linker.replace('0x100000', hex(cores*MIB))
    sources = {'prefill.c': prefill.source, 'decode.c': decode.source,
               'guest.c': _guest_source(prefill,decode,manifest,mappings,layout,cache_layout,identity),
               'start.S': SINGLE_STARTUP if cores == 1 else STARTUP_ASSEMBLY, 'link.ld': linker}
    for name, source in sources.items():
        (build_dir/name).write_text(source)
    target = ['-target','riscv64-linux-musl','-static'] if toolchain.is_zig else ['--target=riscv64-unknown-elf','-fuse-ld=lld','-nostdlib']
    arch = ['-mcpu=generic_rv64+m+a+f+d+c+zicsr+zifencei'] if toolchain.is_zig else ['-march=rv64gc']
    command = [*toolchain.cc,*target,'-O2','-std=c11',*arch,'-mabi=lp64d','-mcmodel=medany',
               '-msmall-data-limit=0','-mno-relax','-ffreestanding','-fno-builtin','-fno-stack-protector',
               '-fno-pie','-fPIC' if toolchain.is_zig else '-fno-pic',
               '-fno-fast-math','-ffp-contract=off','-fno-strict-aliasing','-ffunction-sections','-fdata-sections',
               f'-DSCRATCHV_CORES={cores}',f'-DSCRATCHV_HARTS={cores}',
               'prefill.c','decode.c','guest.c','start.S','-Wl,-T,link.ld','-Wl,-e,sv_boot',
               '-Wl,--gc-sections','-Wl,--build-id=none',*(['-lm'] if toolchain.is_zig else []),'-o','model.elf']
    env = dict(os.environ)
    temporary = build_dir/'tmp'
    temporary.mkdir()
    env.update(TMP=str(temporary),TEMP=str(temporary),TMPDIR=str(temporary))
    if toolchain.is_zig:
        env['ZIG_GLOBAL_CACHE_DIR'] = str(Path(os.environ.get('SCRATCHV_ZIG_CACHE',
            _default_cache_path())).resolve())
        env['ZIG_LOCAL_CACHE_DIR'] = str(build_dir/'.zig-cache')
    start = time.perf_counter()
    try:
        result = _run_process(command,cwd=build_dir,env=env,timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _save_process_logs(build_dir,'compiler',exc.stdout,exc.stderr)
        raise TimeoutError(f'KV RV64 build exceeded {timeout}s') from exc
    _save_process_logs(build_dir,'compiler',result.stdout,result.stderr)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-12000:])
    evidence = validate_external_elf(build_dir/'model.elf',layout)
    evidence.update(identity=identity,cores=cores,prefill_input_bytes=pi,decode_input_bytes=di,compile_seconds=time.perf_counter()-start,
        cache_layout=cache_layout.to_dict(),weight_mappings=mappings,command=command,
        prefill_workspace_bytes=prefill.workspace_bytes,decode_workspace_bytes=decode.workspace_bytes,
        versions=toolchain_versions(toolchain),
        sources={name:sha256_file(build_dir/name) for name in sources})
    (build_dir/'build.json').write_text(json.dumps(evidence,indent=2)+'\n')
    (build_dir/'layout.json').write_text(json.dumps(layout,indent=2)+'\n')
    return SessionExecutable(build_dir/'model.elf',build_dir/'weights',manifest,weights,layout,
                             cache_layout,toolchain,evidence,cores)


def _default_cache_path():
    return str(Path(__file__).resolve().parents[2]/'output'/'.riscv-zig-cache')


class QemuSession:
    """A serial, fail-closed resident guest with transactional double-bank KV.

    Each response is matched to its nonce, build, request and committed length.
    Timeout, EOF, cancellation, guest error or output-read failure permanently
    faults the session; no command is automatically replayed.
    """
    def __init__(self, executable, run_dir, *, timeout=3600.0, cancelled=None, deadline=None):
        for name, value in (('timeout',timeout),('deadline',deadline)):
            if value is None and name=='deadline':
                continue
            if isinstance(value, (bool,np.bool_)) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
            if name=='timeout' and value<=0:
                raise ValueError('timeout must be positive')
        if cancelled is not None and not callable(cancelled):
            raise TypeError('cancelled must be callable')
        self.executable, self.timeout, self.cancelled, self.deadline = executable,float(timeout),cancelled,deadline
        self.run_dir = Path(run_dir).resolve()
        self.state = 'NEW'
        self.valid_length = self.bank = self.request_id = 0
        self.session_id = secrets.randbits(64) or 1
        self.process = self.qmp = self.job = self.monitor = self.serial = self.listener = None
        self.stdout = self.stderr = self._previous_term = None
        self.decoder = protocol.FrameDecoder()
        self._lock = threading.Lock()
        self._disposed = False
        self.report = {'schema':'scratchv.kv-session.v1','passed':False,'state':self.state,
                       'cores':executable.cores,'identity':executable.evidence['identity'],
                       'cache_layout':executable.cache_layout.to_dict(),
                       'memory':executable.layout,'requests':[],'weight_transport':{}}
        self.runs = self.report['requests']

    def _remaining(self):
        _check_cancelled(self.cancelled)
        remaining = self._operation_deadline-time.monotonic()
        if self.deadline is not None:
            remaining = min(remaining,self.deadline-time.monotonic())
        if remaining<=0:
            raise TimeoutError('KV session deadline exceeded')
        return min(remaining,0.05)

    def _read_frame(self):
        while True:
            wait = self._remaining()
            if self.process.poll() is not None:
                raise RuntimeError(f'QEMU exited before response ({self.process.returncode})')
            self.serial.settimeout(wait)
            try:
                data = self.serial.recv(4096)
            except socket.timeout:
                continue
            if not data:
                self.decoder.eof()
            frames = self.decoder.feed(data)
            if frames:
                if len(frames)!=1 or self.decoder.buffer:
                    raise ValueError('Unsolicited/duplicate KV response')
                return frames[0]

    def start(self):
        if self.state!='NEW':
            raise RuntimeError(f'Cannot start session in {self.state}')
        self.run_dir.mkdir(parents=True,exist_ok=False)
        start = time.perf_counter()
        self._operation_deadline = time.monotonic()+self.timeout
        try:
            self._remaining()
            exe = self.executable
            _validate_session_metadata(exe)
            if exe.cores!=validate_cores(exe.evidence['cores']) or exe.layout['regions']['stack']['nbytes']!=exe.cores*MIB:
                raise ValueError('Session build core count/stack mismatch')
            if validate_external_elf(exe.elf_path,exe.layout)['sha256']!=exe.evidence['sha256']:
                raise ValueError('Session ELF changed after build')
            if validate_weight_bundle(exe.bundle_dir,exe.weights)!=exe.manifest:
                raise ValueError('Session weight bundle changed after build')
            self.listener = socket.socket()
            self.listener.bind(('127.0.0.1',0))
            self.listener.listen(1)
            serial_port = self.listener.getsockname()[1]
            with socket.socket() as reservation:
                reservation.bind(('127.0.0.1',0))
                qmp_port = reservation.getsockname()[1]
            (self.run_dir/'boot.bin').write_bytes(struct.pack('<Q',self.session_id))
            layout = exe.layout
            command = [exe.toolchain.qemu,'-machine','virt','-cpu','rv64','-accel',
                'tcg' if exe.cores==1 else 'tcg,thread=multi','-smp',str(exe.cores),'-m',str(layout['memory_mib']),
                '-bios','none','-display','none','-monitor','none','-no-reboot',
                '-serial',f'tcp:127.0.0.1:{serial_port},nodelay=on',
                '-qmp',f'tcp:127.0.0.1:{qmp_port},server=on,wait=off,nodelay=on',
                '-kernel',str(exe.elf_path),'-device',
                f'loader,file=boot.bin,addr=0x{layout["prefill_input_base"]:x},force-raw=on']
            stage_start = time.perf_counter()
            stage_transport(exe.bundle_dir/'weights.bin',exe.manifest,self.run_dir/'weight-transport',
                            self.report['weight_transport'])
            self.report['weight_staging_s'] = time.perf_counter()-stage_start
            for row in self.report['weight_transport']['files']:
                path = str(self.run_dir/'weight-transport'/row['file']).replace(',',',,')
                command += ['-device',f'loader,file={path},addr=0x{layout["weights_base"]+row["offset"]:x},force-raw=on']
            self._remaining()
            if os.name!='nt' and threading.current_thread() is threading.main_thread():
                self._previous_term = signal.getsignal(signal.SIGTERM)
                signal.signal(signal.SIGTERM,_terminate_invocation)
            self.stdout = (self.run_dir/'qemu.stdout').open('wb')
            self.stderr = (self.run_dir/'qemu.stderr').open('wb')
            options = {'creationflags':_creation_flags()}
            if os.name!='nt':
                options['start_new_session'] = True
            self.process = subprocess.Popen(command,cwd=self.run_dir,stdout=self.stdout,stderr=self.stderr,**options)
            self.job = _windows_job(self.process)
            self.report.update(command=command,qemu_pid=self.process.pid)
            self.monitor = _QemuMemoryMonitor(self.process)
            self.monitor.start()
            while self.serial is None:
                self.listener.settimeout(self._remaining())
                if self.process.poll() is not None:
                    raise RuntimeError('QEMU exited before UART connection')
                try:
                    self.serial,_ = self.listener.accept()
                except socket.timeout:
                    continue
            self.serial.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            ready = self._read_frame()
            if (ready.kind!=protocol.READY or ready.session!=self.session_id or ready.request!=0
                    or ready.status or ready.valid_length or ready.logits_bytes
                    or ready.payload.decode('ascii')!=exe.evidence['identity']):
                raise ValueError('Invalid READY/build/session identity')
            self.qmp = QMP(qmp_port,self._operation_deadline,cancelled=self._cancelled)
            self.state = 'READY'
            self.report['startup_s'] = time.perf_counter()-start
            return self
        except BaseException as exc:
            self._fault(exc)
            raise

    def _cancelled(self):
        if self.deadline is not None and time.monotonic()>=self.deadline:
            raise TimeoutError('Generation deadline exceeded')
        return self.cancelled is not None and self.cancelled()

    def _fault(self, exc):
        self.state = 'FAULT'
        self.report['error'] = f'{type(exc).__name__}: {exc}'
        self._dispose()

    def _dump(self, address, size, name):
        path = self.run_dir/name
        reply = self.qmp.command('human-monitor-command',{'command-line':f'pmemsave 0x{address:x} {size} {name}'})
        if reply.strip():
            raise RuntimeError(f'QMP memory dump failed: {reply}')
        data = path.read_bytes()
        if len(data)!=size:
            raise ValueError('QMP output byte count mismatch')
        return data

    def _request(self, kind, payload=b'', *, length=None):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Overlapping KV session requests are unsupported')
        if self.state != 'READY':
            self._lock.release()
            raise RuntimeError(f'Session is not usable ({self.state})')
        row = {'kind':kind,'request':self.request_id+1,'committed_before':self.valid_length,'passed':False}
        start = time.perf_counter()
        try:
            self._operation_deadline = time.monotonic()+self.timeout
            self._remaining()
            self.state = 'EXECUTING'
            self.request_id += 1
            packet = protocol.Frame(kind,self.session_id,self.request_id,self.valid_length,payload=payload).encode()
            self.serial.settimeout(min(30.0,max(self._remaining(),0.001)))
            self.serial.sendall(packet)
            response = self._read_frame()
            row['forward_s'] = time.perf_counter()-start
            expected_kind = protocol.DONE if kind in (protocol.PREFILL,protocol.DECODE) else protocol.ACK
            expected_length = self.valid_length if length is None else length
            size = self.executable.cache_layout.vocab_size*4 if expected_kind==protocol.DONE else 0
            if (response.kind!=expected_kind or response.status or response.payload
                    or response.session!=self.session_id or response.request!=self.request_id
                    or response.valid_length!=expected_length or response.logits_bytes!=size):
                raise ValueError(f'Invalid KV response: kind={response.kind} status={response.status}, '
                                 f'request={response.request}, length={response.valid_length}')
            self.state = 'RESULT_READY'
            self.qmp.deadline = self._operation_deadline
            output = None
            if size:
                transfer = time.perf_counter()
                self.qmp.command('stop')
                data = self._dump(self.executable.layout['logits_base'],size,f'logits-{self.request_id:04d}.bin')
                output = np.frombuffer(data,dtype='<f4').reshape(1,1,-1).copy()
                cached_logits(output,self.executable.cache_layout.vocab_size)
                self.qmp.command('cont')
                row['output_transfer_s'] = time.perf_counter()-transfer
            self._remaining()
            self.valid_length = expected_length
            if kind in (protocol.RESET,protocol.PREFILL):
                self.bank = 0
            elif kind==protocol.DECODE:
                self.bank = 1-self.bank
            self.state = 'READY'
            row.update(passed=True,committed_after=self.valid_length)
            return output
        except BaseException as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
            self._fault(exc)
            raise
        finally:
            row['total_s'] = time.perf_counter()-start
            self.runs.append(row)
            if self._disposed and self.run_dir.is_dir():
                try:
                    (self.run_dir/'session.json').write_text(json.dumps(self.report,indent=2)+'\n')
                except OSError:
                    pass
            self._lock.release()

    def reset(self):
        return self._request(protocol.RESET,length=0)

    def prefill(self, prompt_ids):
        feed = prepare_prefill(prompt_ids,self.executable.cache_layout)
        if self.valid_length:
            raise ValueError('Prefill requires RESET')
        length = int(feed['logit_index'][0])+1
        return self._request(protocol.PREFILL,feed['input_ids'][0,:length].astype('<i8').tobytes(),length=length)

    def decode(self, token_id):
        token = _integer(token_id,'token_id',maximum=self.executable.cache_layout.vocab_size-1)
        if not 1<=self.valid_length<self.executable.cache_layout.capacity:
            raise ValueError('Decode requires a nonempty cache below capacity')
        return self._request(protocol.DECODE,struct.pack('<q',token),length=self.valid_length+1)

    def dump_cache(self, layer=None):
        """Explicit debug-only KV transfer; ordinary generation reads logits only."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Overlapping KV session requests are unsupported')
        try:
            if self.state!='READY':
                raise RuntimeError('Cannot inspect a failed/closed session')
            cache = self.executable.cache_layout
            indices = range(2*cache.layers) if layer is None else range(
                2*_integer(layer,'layer',maximum=cache.layers-1),2*int(layer)+2)
            self._operation_deadline = time.monotonic()+self.timeout
            self._remaining()
            self.qmp.deadline = self._operation_deadline
            self.qmp.command('stop')
            results = []
            size = cache.bank_bytes//(2*cache.layers)
            for i in indices:
                data = self._dump(self.executable.layout['cache_bases'][self.bank]+i*size,size,
                                  f'cache-{self.request_id:04d}-{i:02d}.bin')
                a = np.frombuffer(data,dtype='<f4').reshape(cache.shape).copy()
                if not np.isfinite(a).all():
                    raise ValueError('Nonfinite debug cache')
                results.append(a)
            self.qmp.command('cont')
            return tuple(results)
        except BaseException as exc:
            self._fault(exc)
            raise
        finally:
            self._lock.release()

    def _dispose(self):
        if self._disposed:
            return
        self._disposed = True
        errors = []
        def cleanup(label, fn):
            try:
                fn()
            except BaseException as exc:
                errors.append(f'{label}: {type(exc).__name__}: {exc}')
        if self.qmp is not None:
            cleanup('QMP close',self.qmp.close)
        if self.serial is not None:
            cleanup('UART close',self.serial.close)
        if self.listener is not None:
            cleanup('listener close',self.listener.close)
        if self.process is not None:
            cleanup('process group termination',lambda:_stop_process_tree(self.process,self.job))
            cleanup('process reap',lambda:self.process.wait(timeout=10))
        if self.monitor is not None:
            cleanup('RSS monitor',self.monitor.close)
            self.report.update(self.monitor.report())
        if self.job is not None:
            cleanup('Windows job',lambda:self.job[0].CloseHandle(self.job[1]))
        for handle in (self.stdout,self.stderr):
            if handle is not None:
                cleanup('log handle',handle.close)
        if self._previous_term is not None:
            cleanup('SIGTERM restore',lambda:signal.signal(signal.SIGTERM,self._previous_term))
        self.report.update(state=self.state,valid_length=self.valid_length,
                           qemu_reaped=self.process is not None and self.process.returncode is not None,
                           passed=self.state=='CLOSED' and not errors)
        if self.report['passed'] and 'files' in self.report['weight_transport']:
            cleanup('weight transport',lambda:remove_transport(self.run_dir/'weight-transport',self.report['weight_transport']))
        if errors:
            self.report.update(passed=False,cleanup_errors=errors)
        if self.run_dir.is_dir():
            cleanup('session report',lambda:(self.run_dir/'session.json').write_text(json.dumps(self.report,indent=2)+'\n'))
        if errors and self.state!='FAULT':
            raise RuntimeError('; '.join(errors))

    def close(self):
        if self._disposed:
            return
        try:
            if self.state=='READY':
                self._request(protocol.CLOSE)
                self.qmp.cancelled = None
                self.qmp.deadline = time.monotonic()+5
                self.qmp.command('quit')
                self.process.wait(timeout=5)
            self.state = 'CLOSED'
        except BaseException as exc:
            self._fault(exc)
            raise
        finally:
            self._dispose()

    def __enter__(self):
        return self.start()

    def __exit__(self, kind, exc, traceback):
        if exc is not None:
            self._fault(exc)
        else:
            self.close()


def save_session_executable(executable, directory):
    """Persist metadata; hashes bind the later loader to build-time ELF/weights."""
    record = {'schema':'scratchv.kv-executable.v1','elf_path':str(executable.elf_path),
        'bundle_dir':str(executable.bundle_dir),'manifest':executable.manifest,
        'weights':[{'name':s.name,'dtype':s.dtype.value,'shape':list(s.shape)} for s in executable.weights],
        'layout':executable.layout,'cache_layout':executable.cache_layout.to_dict(),
        'cc':list(executable.toolchain.cc),'qemu':executable.toolchain.qemu,
        'evidence':executable.evidence,'cores':executable.cores}
    path = Path(directory)/'executable.json'
    path.write_text(json.dumps(record,indent=2)+'\n')
    return sha256_file(path)


def load_session_executable(path):
    from scratchv.backend.tensor_c_codegen import TensorSpec
    from scratchv.ir.types import DataType
    record = json.loads(Path(path).read_text())
    if record['schema']!='scratchv.kv-executable.v1':
        raise ValueError('Unsupported KV executable schema')
    weights = tuple(TensorSpec(s['name'],DataType(s['dtype']),tuple(s['shape'])) for s in record['weights'])
    executable = SessionExecutable(Path(record['elf_path']),Path(record['bundle_dir']),record['manifest'],
        weights,record['layout'],KVCacheLayout(**record['cache_layout']),
        RiscVToolchain(tuple(record['cc']),record['qemu']),record['evidence'],validate_cores(record['cores']))
    _validate_session_metadata(executable)
    return executable


def _validate_session_metadata(executable):
    evidence = executable.evidence
    cache = executable.cache_layout
    expected = plan_session_memory(cache.bank_bytes,
        prefill_input_bytes=evidence['prefill_input_bytes'],decode_input_bytes=evidence['decode_input_bytes'],
        prefill_workspace_bytes=evidence['prefill_workspace_bytes'],
        decode_workspace_bytes=evidence['decode_workspace_bytes'],logits_bytes=cache.vocab_size*4,
        weight_bytes=executable.manifest['total_bytes'],cores=executable.cores)
    if expected!=executable.layout or cache.to_dict()!=evidence['cache_layout']:
        raise ValueError('Session layout differs from its build metadata')
