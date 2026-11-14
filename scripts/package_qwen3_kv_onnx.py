#!/usr/bin/env python3
"""Repack two KV graphs into shared <=1 GiB ONNX external-data shards.

Copies tensor bytes verbatim, preserving graph operations and embedded tensors.
Only tensors with equal names, types, shapes and SHA-256 share storage. Source
exports are read-only; the destination must not already exist.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from export_qwen3_onnx import external_tensors, sha256

SHARD_BYTES = 1024**3
CHUNK_BYTES = 8*1024**2


def tensor_source(graph, tensor):
    fields = {entry.key: entry.value for entry in tensor.external_data}
    path = (graph.parent/fields['location']).resolve()
    if not path.is_relative_to(graph.parent) or not path.is_file():
        raise ValueError(f'External tensor escapes export directory: {tensor.name}')
    offset = int(fields.get('offset', 0))
    length = int(fields.get('length', path.stat().st_size-offset))
    if offset < 0 or length < 0 or offset+length > path.stat().st_size:
        raise ValueError(f'Invalid external tensor range: {tensor.name}')
    return path, offset, length


def range_hash(path, offset, length):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        stream.seek(offset)
        while length:
            block = stream.read(min(CHUNK_BYTES, length))
            if not block:
                raise EOFError(f'Truncated tensor data: {path}')
            digest.update(block)
            length -= len(block)
    return digest.hexdigest()


def graph_fingerprint(model):
    import onnx
    normalized = onnx.ModelProto()
    normalized.CopyFrom(model)
    for tensor in external_tensors(normalized):
        del tensor.external_data[:]
    return hashlib.sha256(normalized.SerializeToString(deterministic=True)).hexdigest()


def package(source_dir, output_dir, *, shard_bytes=SHARD_BYTES):
    import onnx
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if type(shard_bytes) is not int or not 1 <= shard_bytes <= SHARD_BYTES:
        raise ValueError('Shard size must be between 1 byte and 1 GiB')
    graphs = [(source_dir/f'{mode}.onnx', onnx.load(source_dir/f'{mode}.onnx', load_external_data=False))
              for mode in ('prefill', 'decode')]
    output_dir.mkdir(parents=True, exist_ok=False)
    shared, records, files = {}, {}, []
    original_files, destination, shard, cursor = set(), None, 0, 0
    try:
        for path, model in graphs:
            before = graph_fingerprint(model)
            tensor_records = []
            for tensor in external_tensors(model):
                source, offset, length = tensor_source(path, tensor)
                if length > shard_bytes:
                    raise ValueError(f'Tensor exceeds shard size: {tensor.name} ({length} bytes)')
                original_files.add(source)
                digest = range_hash(source, offset, length)
                key = (tensor.name, tensor.data_type, tuple(tensor.dims), digest)
                if key not in shared:
                    aligned = (cursor+63)//64*64
                    if destination is None or aligned+length > shard_bytes:
                        if destination is not None:
                            destination.close()
                        shard += 1
                        filename = f'weights-{shard:05d}.data'
                        destination = (output_dir/filename).open('xb')
                        files.append(output_dir/filename)
                        cursor = aligned = 0
                    destination.write(b'\0'*(aligned-cursor))
                    with source.open('rb') as stream:
                        stream.seek(offset)
                        remaining = length
                        copied = hashlib.sha256()
                        while remaining:
                            block = stream.read(min(CHUNK_BYTES, remaining))
                            if not block:
                                raise EOFError(f'Truncated tensor: {tensor.name}')
                            destination.write(block)
                            copied.update(block)
                            remaining -= len(block)
                    if copied.hexdigest() != digest:
                        raise ValueError('Source tensor changed during packaging')
                    shared[key] = (filename, aligned, length)
                    cursor = aligned+length
                filename, start, count = shared[key]
                del tensor.external_data[:]
                for field, value in (('location', filename), ('offset', start), ('length', count)):
                    entry = tensor.external_data.add()
                    entry.key, entry.value = field, str(value)
                tensor_records.append((tensor.name, filename, start, count, digest))
            if destination is not None:
                destination.flush()
            target = output_dir/path.name
            onnx.save_model(model, target)
            onnx.checker.check_model(str(target))
            saved = onnx.load(target, load_external_data=False)
            if graph_fingerprint(saved) != before:
                raise AssertionError(f'Packaging changed graph operations: {path.name}')
            records[path.stem] = dict(source_graph_sha256=sha256(path),
                computation_sha256=before, external_tensors=len(tensor_records),
                _tensors=tensor_records,
                inputs=[value.name for value in saved.graph.input],
                outputs=[value.name for value in saved.graph.output])
            print(f'PACKED {path.name}: {len(tensor_records)} external tensors', flush=True)
        if destination is not None:
            destination.close()
            destination = None
        # Verify the saved references, not just the in-memory copy plan.
        verified = {}
        for path, _ in graphs:
            saved = onnx.load(output_dir/path.name, load_external_data=False)
            expected = records[path.stem].pop('_tensors')
            actual = list(external_tensors(saved))
            if len(actual) != len(expected):
                raise AssertionError('External tensor count changed')
            for tensor, (name, filename, offset, length, digest) in zip(actual, expected):
                source, start, count = tensor_source(output_dir/path.name, tensor)
                if tensor.name != name or (source.name, start, count) != (filename, offset, length):
                    raise AssertionError('Saved tensor reference differs from copy plan')
                key = (source, start, count)
                if key not in verified:
                    verified[key] = range_hash(*key)
                if verified[key] != digest:
                    raise AssertionError(f'Tensor bytes changed: {name}')
        manifest = dict(schema='scratchv.kv-onnx-package.v1', graphs=records,
            onnx_version=importlib.metadata.version('onnx'), max_shard_bytes=shard_bytes,
            original_weight_bytes=sum(path.stat().st_size for path in original_files),
            packed_weight_bytes=sum(path.stat().st_size for path in files),
            unique_external_tensors=len(shared),
            verification=dict(graphs_identical_except_external_references=True,
                              all_external_tensor_bytes_identical=True),
            files=[dict(name=path.name, bytes=path.stat().st_size, sha256=sha256(path))
                   for path in [*(output_dir/path.name for path, _ in graphs), *files]])
        for mode in ('prefill', 'decode'):
            metadata = source_dir/f'export-{mode}.json'
            if metadata.exists():
                export = json.loads(metadata.read_text())
                manifest.setdefault('export', {})[mode] = {
                    key: export[key] for key in ('layout', 'packages', 'exporter')}
        (output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        return manifest
    finally:
        if destination is not None:
            destination.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    result = package(args.source_dir, args.output_dir)
    print(f'PASS: {result["packed_weight_bytes"]} shared weight bytes in <=1 GiB shards')
