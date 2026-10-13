"""Versioned, streaming external tensor weights and an explicit RV64 RAM plan.

A bundle has fixed public files ``manifest.json`` and ``weights.bin``. Hashes detect damage,
not provenance: callers must bind ``expected_specs`` and the model identity to a
trusted source. Validation is a snapshot, not a lock for a later QEMU loader.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

import numpy as np


SCHEMA = "scratchv.weights"
VERSION = 1
ALIGNMENT = 64
CHUNK_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
UINT64_MAX = (1 << 64) - 1
MIB = 1024 * 1024
_DTYPES = {"f32": np.dtype("<f4"), "i32": np.dtype("<i4"), "i64": np.dtype("<i8")}
_TOP_KEYS = {"schema", "version", "endian", "alignment", "total_bytes", "sha256", "tensors"}
_TENSOR_KEYS = {"index", "name", "dtype", "shape", "offset", "nbytes", "sha256"}
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class WeightBundleError(ValueError):
    """An external weight bundle violates its declared storage contract."""


def _integer(value, label, *, maximum=UINT64_MAX, numpy_ok=False):
    valid = type(value) is int or (numpy_ok and isinstance(value, np.integer))
    if not valid or isinstance(value, (bool, np.bool_)) or not 0 <= value <= maximum:
        raise WeightBundleError(f"{label} must be a nonnegative integer <= {maximum}")
    return int(value)


def _align(value, alignment=ALIGNMENT):
    return (value + alignment - 1) // alignment * alignment


def _shape(value, label, *, numpy_ok=False):
    if not isinstance(value, (list, tuple)) or len(value) > 16:
        raise WeightBundleError(f"{label} must have at most 16 static dimensions")
    result = [_integer(d, label, maximum=(1 << 31) - 1, numpy_ok=numpy_ok) for d in value]
    if math.prod(result) > (1 << 31) - 1:
        raise WeightBundleError(f"{label} exceeds the static tensor element limit")
    return result


def _name(value):
    if not isinstance(value, str) or not value or len(value) > 65536 or "\0" in value:
        raise WeightBundleError("Tensor name must be a nonempty string without NUL")
    return value


def _spec_metadata(specs):
    result, names = [], set()
    for index, spec in enumerate(specs):
        try:
            name = _name(spec.name)
            dtype = getattr(spec.dtype, "value", spec.dtype)
            if not isinstance(dtype, str) or dtype not in _DTYPES:
                raise WeightBundleError(f"Unsupported dtype for {name!r}: {dtype!r}")
            shape = _shape(spec.shape, f"Shape of {name!r}", numpy_ok=True)
            numpy_dtype = np.dtype(spec.numpy_dtype)
            if numpy_dtype.newbyteorder("<") != _DTYPES[dtype]:
                raise WeightBundleError(f"Spec dtype/numpy_dtype disagree for {name!r}")
            nbytes = math.prod(shape) * _DTYPES[dtype].itemsize
            if _integer(spec.nbytes, f"nbytes of {name!r}", numpy_ok=True) != nbytes:
                raise WeightBundleError(f"Spec nbytes/shape disagree for {name!r}")
        except (AttributeError, TypeError) as exc:
            raise WeightBundleError(f"Invalid tensor spec at index {index}") from exc
        if name in names:
            raise WeightBundleError(f"Tensor names must be unique: {name!r}")
        names.add(name)
        result.append({"index": index, "name": name, "dtype": dtype,
                       "shape": shape, "nbytes": nbytes})
    return result


def _iter_array_chunks(array, dtype):
    """C-order, little-endian blocks, bounded even for strided/mapped arrays."""
    iterator = np.nditer(array, flags=["external_loop", "buffered", "zerosize_ok"],
                         op_flags=["readonly"], op_dtypes=[dtype], casting="equiv",
                         order="C", buffersize=CHUNK_BYTES // dtype.itemsize)
    with iterator:
        for block in iterator:
            if dtype.kind == "f" and not np.isfinite(block).all():
                raise WeightBundleError("Nonfinite floating-point weight")
            yield block.tobytes(order="C")


def _reject_links(path):
    """Reject symlinks/junctions in existing components, including the root."""
    path = Path(os.path.abspath(os.fspath(path)))
    for component in (*reversed(path.parents), path):
        if component.is_symlink() or (hasattr(component, "is_junction") and component.is_junction()):
            raise WeightBundleError(f"Bundle path must not traverse a link: {component}")
    return path


def write_weight_bundle(specs, arrays, directory):
    """Write a new bundle directory without copying the complete model.

    ``arrays`` is a name->ndarray mapping or an iterable in ``specs`` order.
    Parent directories must exist; even an empty existing output is rejected.
    On failure a new directory may contain partial weights, but no valid manifest
    is published. The caller owns and must keep source arrays stable while writing.
    """
    metadata = _spec_metadata(specs)
    names = [entry["name"] for entry in metadata]
    if isinstance(arrays, Mapping):
        if set(arrays) != set(names):
            raise WeightBundleError("Array names must exactly match tensor specs")
        array_iter = (arrays[name] for name in names)
    else:
        array_iter = iter(arrays)
    directory = _reject_links(directory)
    directory.mkdir(exist_ok=False)
    total_hash = hashlib.sha256()
    tensors, cursor = [], 0
    with (directory / "weights.bin").open("xb") as stream:
        for entry in metadata:
            try:
                array = next(array_iter)
            except StopIteration as exc:
                raise WeightBundleError("Fewer arrays than tensor specs") from exc
            dtype = _DTYPES[entry["dtype"]]
            if not isinstance(array, np.ndarray):
                raise WeightBundleError(f"Weight {entry['name']!r} must be an ndarray")
            if list(array.shape) != entry["shape"] or array.dtype.newbyteorder("<") != dtype:
                raise WeightBundleError(f"Weight {entry['name']!r} shape/dtype mismatch")
            offset = _align(cursor)
            padding = b"\0" * (offset - cursor)
            stream.write(padding)
            total_hash.update(padding)
            tensor_hash, written = hashlib.sha256(), 0
            for chunk in _iter_array_chunks(array, dtype):
                stream.write(chunk)
                tensor_hash.update(chunk)
                total_hash.update(chunk)
                written += len(chunk)
            if written != entry["nbytes"]:
                raise WeightBundleError("Weight changed size while being written")
            tensors.append({**entry, "offset": offset, "sha256": tensor_hash.hexdigest()})
            cursor = offset + written
        sentinel = object()
        if next(array_iter, sentinel) is not sentinel:
            raise WeightBundleError("More arrays than tensor specs")
        stream.flush()
        os.fsync(stream.fileno())
    manifest = {"schema": SCHEMA, "version": VERSION, "endian": "little",
                "alignment": ALIGNMENT, "total_bytes": cursor,
                "sha256": total_hash.hexdigest(), "tensors": tensors}
    payload = (json.dumps(manifest, indent=2, ensure_ascii=True) + "\n").encode("utf-8")
    if len(payload) > MAX_MANIFEST_BYTES:
        raise WeightBundleError("Manifest exceeds size limit")
    # Publish only after durable tensor and manifest writes. A hard link is an
    # atomic, no-replace publication within the same directory on Linux/Windows.
    # Any pre-publication failure leaves no manifest.json for a reader to accept.
    pending = directory / "manifest.pending.json"
    with pending.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(pending, directory / "manifest.json")
    try:
        pending.unlink()
    except OSError:
        # The published bundle is complete. Cleanup failure is not data failure;
        # validators only consume the two fixed public filenames.
        pass
    return manifest


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise WeightBundleError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise WeightBundleError(f"Invalid JSON numeric constant: {value}")


def _fingerprint(info):
    # CPython/Windows lstat and fstat can report creation/change time differently.
    # POSIX ctime is comparable and adds a useful mutation check there.
    change_time = info.st_ctime_ns if os.name != "nt" else None
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, change_time


def _open_regular(path):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise WeightBundleError(f"Expected regular file: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    stream = os.fdopen(fd, "rb")
    if _fingerprint(before) != _fingerprint(os.fstat(stream.fileno())):
        stream.close()
        raise WeightBundleError(f"File changed while opening: {path.name}")
    return stream, before


def _check_unchanged(stream, path, before):
    if (_fingerprint(os.fstat(stream.fileno())) != _fingerprint(before)
            or _fingerprint(path.lstat()) != _fingerprint(before)):
        raise WeightBundleError(f"File changed during validation: {path.name}")


def _sha(value, label):
    if not isinstance(value, str) or not _HEX_SHA256.fullmatch(value):
        raise WeightBundleError(f"Invalid SHA-256 for {label}")


def _validate_manifest(manifest, expected_specs):
    if not isinstance(manifest, dict) or set(manifest) != _TOP_KEYS:
        raise WeightBundleError("Manifest must contain exactly the supported schema fields")
    if manifest["schema"] != SCHEMA or _integer(manifest["version"], "version") != VERSION:
        raise WeightBundleError("Unsupported weight bundle schema/version")
    if manifest["endian"] != "little" or _integer(manifest["alignment"], "alignment") != ALIGNMENT:
        raise WeightBundleError("Unsupported endian/alignment")
    total_bytes = _integer(manifest["total_bytes"], "total_bytes")
    _sha(manifest["sha256"], "weights.bin")
    tensors = manifest["tensors"]
    if not isinstance(tensors, list):
        raise WeightBundleError("tensors must be an ordered list")
    names, cursor = set(), 0
    for index, tensor in enumerate(tensors):
        if not isinstance(tensor, dict) or set(tensor) != _TENSOR_KEYS:
            raise WeightBundleError("Tensor entry contains unsupported schema fields")
        if _integer(tensor["index"], "tensor index") != index:
            raise WeightBundleError("Tensor indices must be unique, ordered and contiguous")
        name = _name(tensor["name"])
        if name in names:
            raise WeightBundleError("Tensor names must be unique")
        names.add(name)
        dtype = tensor["dtype"]
        if not isinstance(dtype, str) or dtype not in _DTYPES:
            raise WeightBundleError(f"Unsupported tensor dtype: {dtype!r}")
        if not isinstance(tensor["shape"], list):
            raise WeightBundleError("Manifest tensor shape must be a list")
        shape = _shape(tensor["shape"], f"Shape of {name!r}")
        nbytes = _integer(tensor["nbytes"], "tensor nbytes")
        if nbytes != math.prod(shape) * _DTYPES[dtype].itemsize:
            raise WeightBundleError("Tensor shape/dtype/nbytes mismatch")
        offset = _integer(tensor["offset"], "tensor offset")
        if offset != _align(cursor):
            raise WeightBundleError("Tensor offsets must be aligned, ordered and non-overlapping")
        cursor = offset + nbytes
        if cursor > total_bytes:
            raise WeightBundleError("Tensor exceeds weights.bin bounds")
        _sha(tensor["sha256"], name)
    if cursor != total_bytes:
        raise WeightBundleError("total_bytes does not match final tensor boundary")
    if expected_specs is not None:
        expected = _spec_metadata(expected_specs)
        actual = [{key: tensor[key] for key in ("index", "name", "dtype", "shape", "nbytes")}
                  for tensor in tensors]
        if actual != expected:
            raise WeightBundleError("Manifest tensors differ from trusted expected specs")


def validate_weight_bundle(directory, expected_specs=None):
    """Validate fixed bundle files, all byte hashes, padding and finite fp32.

    Checks are bounded-memory and strict, but cannot authenticate a self-authored
    manifest or guarantee that a file stays unchanged after this call returns.
    """
    directory = _reject_links(directory)
    if not directory.is_dir():
        raise WeightBundleError("Bundle directory does not exist")
    try:
        manifest_path = directory / "manifest.json"
        manifest_stream, manifest_before = _open_regular(manifest_path)
        with manifest_stream:
            if manifest_before.st_size > MAX_MANIFEST_BYTES:
                raise WeightBundleError("Manifest exceeds size limit")
            payload = manifest_stream.read(MAX_MANIFEST_BYTES + 1)
            if len(payload) > MAX_MANIFEST_BYTES:
                raise WeightBundleError("Manifest exceeds size limit")
            manifest = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object,
                                  parse_constant=_reject_constant)
            _validate_manifest(manifest, expected_specs)
            weights_path = directory / "weights.bin"
            weights_stream, weights_before = _open_regular(weights_path)
            with weights_stream:
                if weights_before.st_size != manifest["total_bytes"]:
                    raise WeightBundleError("weights.bin length mismatch (truncated or extra bytes)")
                total_hash, cursor = hashlib.sha256(), 0
                for tensor in manifest["tensors"]:
                    padding = weights_stream.read(tensor["offset"] - cursor)
                    if len(padding) != tensor["offset"] - cursor or any(padding):
                        raise WeightBundleError("Invalid or truncated alignment padding")
                    total_hash.update(padding)
                    remaining, tensor_hash = tensor["nbytes"], hashlib.sha256()
                    while remaining:
                        chunk = weights_stream.read(min(remaining, CHUNK_BYTES))
                        if len(chunk) != min(remaining, CHUNK_BYTES):
                            raise WeightBundleError("Truncated weight tensor")
                        if tensor["dtype"] == "f32" and not np.isfinite(np.frombuffer(chunk, dtype="<f4")).all():
                            raise WeightBundleError("Nonfinite floating-point weight")
                        tensor_hash.update(chunk)
                        total_hash.update(chunk)
                        remaining -= len(chunk)
                    if tensor_hash.hexdigest() != tensor["sha256"]:
                        raise WeightBundleError(f"Tensor SHA-256 mismatch: {tensor['name']!r}")
                    cursor = tensor["offset"] + tensor["nbytes"]
                if weights_stream.read(1) or total_hash.hexdigest() != manifest["sha256"]:
                    raise WeightBundleError("weights.bin SHA-256 mismatch")
                _check_unchanged(weights_stream, weights_path, weights_before)
            _check_unchanged(manifest_stream, manifest_path, manifest_before)
            return manifest
    except WeightBundleError:
        raise
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise WeightBundleError(f"Cannot validate weight bundle: {exc}") from exc


def plan_guest_memory(code_capacity=128 * MIB, workspace_bytes=0, weight_bytes=0,
                      input_bytes=0, output_bytes=0, *, ram_base=0x80000000):
    """Plan non-overlapping RV64 regions; code_capacity includes a 1 MiB stack.

    All bases are 64-byte aligned. RAM is rounded up to MiB with at least 16 MiB
    and at least 5% spare capacity. This plan does not replace ELF bounds checks.
    """
    sizes = {"code_capacity": code_capacity, "workspace_bytes": workspace_bytes,
             "weight_bytes": weight_bytes, "input_bytes": input_bytes,
             "output_bytes": output_bytes, "ram_base": ram_base}
    sizes = {key: _integer(value, key) for key, value in sizes.items()}
    code_capacity, ram_base = sizes["code_capacity"], sizes["ram_base"]
    if ram_base % ALIGNMENT or code_capacity % ALIGNMENT:
        raise WeightBundleError("RAM base and code capacity must be 64-byte aligned")
    if code_capacity < MIB + ALIGNMENT:
        raise WeightBundleError("Code capacity must include code plus a 1 MiB stack")
    regions = {}
    cursor = ram_base
    for name, nbytes in (("code", code_capacity - MIB), ("stack", MIB),
                         ("inputs", sizes["input_bytes"]), ("outputs", sizes["output_bytes"]),
                         ("workspace", sizes["workspace_bytes"]), ("weights", sizes["weight_bytes"])):
        base = _align(cursor)
        end = base + nbytes
        if end > UINT64_MAX:
            raise WeightBundleError("Guest address range overflows uint64")
        regions[name] = {"base": base, "end": end, "nbytes": nbytes}
        cursor = end
    used_bytes = cursor - ram_base
    margin_bytes = max(16 * MIB, (used_bytes + 19) // 20)
    ram_bytes = _align(used_bytes + margin_bytes, MIB)
    if ram_base + ram_bytes > UINT64_MAX:
        raise WeightBundleError("Guest RAM including margin overflows uint64")
    return {"ram_base": ram_base, "ram_end": ram_base + ram_bytes,
            "ram_bytes": ram_bytes, "memory_mib": ram_bytes // MIB,
            "alignment": ALIGNMENT, "used_bytes": used_bytes,
            "margin_bytes": ram_bytes - used_bytes, "regions": regions,
            "code_base": ram_base, "code_capacity": code_capacity,
            "code_limit": regions["code"]["end"],
            "stack_bottom": regions["stack"]["base"], "stack_top": regions["stack"]["end"],
            "input_base": regions["inputs"]["base"], "output_base": regions["outputs"]["base"],
            "workspace_base": regions["workspace"]["base"], "weights_base": regions["weights"]["base"]}
