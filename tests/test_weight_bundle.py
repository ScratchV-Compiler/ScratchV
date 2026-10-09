"""Actual bounded file I/O, corrupt bundles, and RV64 guest-memory boundaries."""

import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from scratchv.backend.tensor_c_codegen import TensorSpec
from scratchv.ir.types import DataType
from scratchv.runtime import weight_bundle as wb


def _spec(name, shape, dtype=DataType.FLOAT32):
    return TensorSpec(name, dtype, shape)


def _bundle(tmp_path):
    specs = (_spec("matrix", (2, 3)), _spec("ids", (2,), DataType.INT64))
    arrays = {"matrix": np.arange(6, dtype=np.float32).reshape(2, 3),
              "ids": np.array([2**45 + 1, -(2**60)], dtype=np.int64)}
    directory = tmp_path / "bundle"
    manifest = wb.write_weight_bundle(specs, arrays, directory)
    return directory, manifest, specs, arrays


def _save(directory, manifest):
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _rehash(directory, manifest):
    data = (directory / "weights.bin").read_bytes()
    manifest["sha256"] = hashlib.sha256(data).hexdigest()
    for entry in manifest["tensors"]:
        manifest_slice = data[entry["offset"]:entry["offset"] + entry["nbytes"]]
        entry["sha256"] = hashlib.sha256(manifest_slice).hexdigest()
    _save(directory, manifest)


def test_write_real_bundle_preserves_bits_order_and_64_byte_padding(tmp_path):
    directory, manifest, specs, arrays = _bundle(tmp_path)
    assert wb.validate_weight_bundle(directory, specs) == manifest
    assert set(manifest) == {"schema", "version", "endian", "alignment", "total_bytes", "sha256", "tensors"}
    assert manifest["schema"] == "scratchv.weights"
    assert manifest["version"] == 1
    assert manifest["endian"] == "little"
    assert manifest["alignment"] == 64
    assert [t["index"] for t in manifest["tensors"]] == [0, 1]
    assert [t["offset"] for t in manifest["tensors"]] == [0, 64]
    assert manifest["total_bytes"] == 80
    data = (directory / "weights.bin").read_bytes()
    assert data[:24] == arrays["matrix"].tobytes()
    assert data[24:64] == bytes(40)
    assert data[64:] == arrays["ids"].astype("<i8").tobytes()
    assert manifest["sha256"] == hashlib.sha256(data).hexdigest()


def test_scalar_zero_sized_and_empty_model_are_unambiguous(tmp_path):
    specs = (_spec("zero", (0, 4)), _spec("scalar", ()), _spec("lastzero", (0,)))
    arrays = [np.zeros((0, 4), np.float32), np.array(-0.0, np.float32), np.zeros(0, np.float32)]
    directory = tmp_path / "zero"
    manifest = wb.write_weight_bundle(specs, iter(arrays), directory)
    assert wb.validate_weight_bundle(directory, specs) == manifest
    assert [t["offset"] for t in manifest["tensors"]] == [0, 0, 64]
    assert manifest["total_bytes"] == 64
    assert (directory / "weights.bin").read_bytes()[:4] == b"\0\0\0\x80"
    empty = tmp_path / "empty"
    assert wb.write_weight_bundle([], [], empty)["total_bytes"] == 0
    assert wb.validate_weight_bundle(empty)["tensors"] == []


def test_memmap_strides_and_foreign_endian_stream_in_bounded_blocks(tmp_path, monkeypatch):
    # More than three streaming chunks, with noncontiguous C-order traversal.
    backing = np.memmap(tmp_path / "source.bin", mode="w+", dtype=">f4", shape=(1025, 1026))
    backing[:] = np.arange(1026, dtype=np.float32)
    view = backing[:, ::-1]
    chunks = []
    original = wb._iter_array_chunks

    def observed(array, dtype):
        assert array is view
        for chunk in original(array, dtype):
            chunks.append(len(chunk))
            assert 0 < len(chunk) <= wb.CHUNK_BYTES
            yield chunk

    monkeypatch.setattr(wb, "_iter_array_chunks", observed)
    directory = tmp_path / "stream"
    spec = _spec("mapped", view.shape)
    manifest = wb.write_weight_bundle([spec], {"mapped": view}, directory)
    assert len(chunks) >= 4
    assert sum(chunks) == view.nbytes
    assert wb.validate_weight_bundle(directory, [spec]) == manifest
    emitted = np.memmap(directory / "weights.bin", mode="r", dtype="<f4", shape=view.shape)
    np.testing.assert_array_equal(emitted[0], view[0])
    np.testing.assert_array_equal(emitted[-1], view[-1])
    # Closing mappings is needed for pytest's Windows temporary-directory cleanup.
    emitted._mmap.close()
    backing._mmap.close()


def test_arrays_iterable_is_consumed_one_tensor_at_a_time(tmp_path):
    size = wb.CHUNK_BYTES
    specs = [_spec("first", (size,), DataType.INT32), _spec("second", (2,), DataType.INT32)]
    directory = tmp_path / "iterable"

    def arrays():
        yield np.arange(size, dtype=np.int32)
        assert (directory / "weights.bin").stat().st_size == size * 4
        yield np.array([7, 8], dtype=np.int32)

    manifest = wb.write_weight_bundle(specs, arrays(), directory)
    assert wb.validate_weight_bundle(directory, specs) == manifest


def test_existing_output_is_never_overwritten(tmp_path):
    directory, manifest, specs, arrays = _bundle(tmp_path)
    before = (directory / "weights.bin").read_bytes()
    with pytest.raises(FileExistsError):
        wb.write_weight_bundle(specs, arrays, directory)
    assert wb.validate_weight_bundle(directory, specs) == manifest
    assert (directory / "weights.bin").read_bytes() == before
    empty = tmp_path / "existing_empty"
    empty.mkdir()
    with pytest.raises(FileExistsError):
        wb.write_weight_bundle([], [], empty)


def test_manifest_is_not_published_when_final_fsync_fails(tmp_path, monkeypatch):
    directory = tmp_path / "interrupted"
    original = wb.os.fsync
    calls = []

    def failed_second_fsync(fd):
        calls.append(fd)
        if len(calls) == 2:
            raise OSError("injected manifest fsync failure")
        original(fd)

    monkeypatch.setattr(wb.os, "fsync", failed_second_fsync)
    with pytest.raises(OSError, match="fsync failure"):
        wb.write_weight_bundle([_spec("x", (1,))], [np.ones(1, np.float32)], directory)
    assert not (directory / "manifest.json").exists()
    with pytest.raises(wb.WeightBundleError):
        wb.validate_weight_bundle(directory)


def test_manifest_publication_cannot_overwrite_a_concurrent_file(tmp_path, monkeypatch):
    directory = tmp_path / "publication"
    original = wb.os.link

    def colliding_link(source, destination):
        destination.write_bytes(b"existing concurrent artifact")
        original(source, destination)

    monkeypatch.setattr(wb.os, "link", colliding_link)
    with pytest.raises(FileExistsError):
        wb.write_weight_bundle([], [], directory)
    assert (directory / "manifest.json").read_bytes() == b"existing concurrent artifact"


@pytest.mark.parametrize("arrays", [[], [np.ones(1, np.float32), np.ones(1, np.float32)],
                                  [np.ones(2, np.float32)], [np.ones(1, np.float64)],
                                  [np.array([np.nan], np.float32)], [np.array([np.inf], np.float32)],
                                  [[1.0]], {"extra": np.ones(1, np.float32)}])
def test_bad_array_input_never_publishes_a_manifest(tmp_path, arrays):
    directory = tmp_path / "failed"
    with pytest.raises(wb.WeightBundleError):
        wb.write_weight_bundle([_spec("a", (1,))], arrays, directory)
    assert not (directory / "manifest.json").exists()
    with pytest.raises(wb.WeightBundleError):
        wb.validate_weight_bundle(directory)


@pytest.mark.parametrize("field,value", [("name", ""), ("dtype", "f64"), ("shape", (True,)),
                                         ("shape", (-1,)), ("shape", (2**31,)),
                                         ("nbytes", True), ("nbytes", 8),
                                         ("numpy_dtype", np.dtype("int32"))])
def test_spec_metadata_must_be_consistent_and_not_bool(tmp_path, field, value):
    data = dict(name="x", dtype="f32", shape=(1,), numpy_dtype=np.dtype("float32"), nbytes=4)
    data[field] = value
    with pytest.raises(wb.WeightBundleError):
        wb.write_weight_bundle([SimpleNamespace(**data)], [np.ones(1, np.float32)], tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_duplicate_specs_and_extra_mapping_arrays_rejected_before_creation(tmp_path):
    with pytest.raises(wb.WeightBundleError, match="unique"):
        wb.write_weight_bundle([_spec("x", (1,))] * 2, [], tmp_path / "duplicates")
    with pytest.raises(wb.WeightBundleError, match="exactly match"):
        wb.write_weight_bundle([_spec("x", (1,))], {"x": np.zeros(1, np.float32), "y": np.zeros(1)},
                               tmp_path / "extra")


@pytest.mark.parametrize("field,value", [("schema", "other"), ("version", 2), ("version", True),
                                         ("endian", "big"), ("alignment", 32), ("alignment", True),
                                         ("total_bytes", True), ("total_bytes", -1),
                                         ("total_bytes", 2**64), ("total_bytes", 81),
                                         ("sha256", "not-a-hash"), ("tensors", {})])
def test_manifest_header_is_strict(tmp_path, field, value):
    directory, manifest, _, _ = _bundle(tmp_path)
    manifest[field] = value
    _save(directory, manifest)
    with pytest.raises(wb.WeightBundleError):
        wb.validate_weight_bundle(directory)


@pytest.mark.parametrize("field,value", [("index", 0), ("index", True), ("name", "matrix"),
                                         ("name", ""), ("dtype", "u64"), ("dtype", []),
                                         ("shape", [True]), ("shape", [-2]), ("shape", [3]),
                                         ("shape", [2.0]), ("shape", [2**31]),
                                         ("offset", 0), ("offset", 1), ("offset", 128),
                                         ("offset", True), ("nbytes", True), ("nbytes", 8),
                                         ("sha256", "0" * 64)])
def test_manifest_tensor_identity_layout_and_hash_are_strict(tmp_path, field, value):
    directory, manifest, _, _ = _bundle(tmp_path)
    manifest["tensors"][1][field] = value
    _save(directory, manifest)
    with pytest.raises(wb.WeightBundleError):
        wb.validate_weight_bundle(directory)


def test_expected_specs_detect_self_consistent_but_wrong_tensor_contract(tmp_path):
    directory, manifest, specs, _ = _bundle(tmp_path)
    manifest["tensors"][0]["shape"] = [3, 2]
    _save(directory, manifest)
    # Self consistency alone does not prove this is the graph's correct weight.
    assert wb.validate_weight_bundle(directory)["tensors"][0]["shape"] == [3, 2]
    with pytest.raises(wb.WeightBundleError, match="trusted expected specs"):
        wb.validate_weight_bundle(directory, specs)


@pytest.mark.parametrize("damage", ["truncated", "extra", "data", "padding", "total_hash"])
def test_binary_corruption_is_detected(tmp_path, damage):
    directory, manifest, _, _ = _bundle(tmp_path)
    path = directory / "weights.bin"
    data = bytearray(path.read_bytes())
    if damage == "truncated":
        data = data[:-1]
    elif damage == "extra":
        data += b"x"
    elif damage == "data":
        data[0] ^= 1
    elif damage == "padding":
        data[30] = 1
    else:
        manifest["sha256"] = "0" * 64
    path.write_bytes(data)
    if damage == "padding":
        # Even replacing both hashes cannot legalize noncanonical padding.
        _rehash(directory, manifest)
    else:
        _save(directory, manifest)
    with pytest.raises(wb.WeightBundleError):
        wb.validate_weight_bundle(directory)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_rejected_even_when_all_hashes_are_updated(tmp_path, value):
    directory, manifest, _, _ = _bundle(tmp_path)
    with (directory / "weights.bin").open("r+b") as stream:
        stream.write(np.array([value], dtype="<f4").tobytes())
    _rehash(directory, manifest)
    with pytest.raises(wb.WeightBundleError, match="Nonfinite"):
        wb.validate_weight_bundle(directory)


@pytest.mark.parametrize("document", ['{"schema": "a", "schema": "b"}', '{"version": NaN}',
                                      '{"version": Infinity}', '{', '[]'])
def test_duplicate_json_keys_and_invalid_json_fail_closed(tmp_path, document):
    directory, _, _, _ = _bundle(tmp_path)
    (directory / "manifest.json").write_text(document, encoding="utf-8")
    with pytest.raises(wb.WeightBundleError):
        wb.validate_weight_bundle(directory)


def test_manifest_paths_cannot_escape_fixed_filenames(tmp_path):
    directory, manifest, _, _ = _bundle(tmp_path)
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"not a weight file")
    manifest["weights_file"] = "../secret.bin"
    _save(directory, manifest)
    with pytest.raises(wb.WeightBundleError, match="schema fields"):
        wb.validate_weight_bundle(directory)
    assert secret.read_bytes() == b"not a weight file"


def test_symlinked_weight_file_and_bundle_directory_are_rejected(tmp_path):
    directory, _, specs, arrays = _bundle(tmp_path)
    original = directory / "weights.bin"
    target = tmp_path / "linked-data.bin"
    original.rename(target)
    try:
        original.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Host does not permit creating symlinks: {exc}")
    with pytest.raises(wb.WeightBundleError, match="regular file"):
        wb.validate_weight_bundle(directory)
    linked_directory = tmp_path / "linked-bundle"
    linked_directory.symlink_to(directory, target_is_directory=True)
    with pytest.raises(wb.WeightBundleError, match="link"):
        wb.validate_weight_bundle(linked_directory)
    with pytest.raises(wb.WeightBundleError, match="link"):
        wb.write_weight_bundle(specs, arrays, linked_directory / "new")


@pytest.mark.skipif(os.name != "nt", reason="Windows directory junction behavior")
def test_windows_junction_bundle_path_is_rejected_without_touching_target(tmp_path):
    import _winapi

    directory, _, specs, arrays = _bundle(tmp_path)
    original = {path.name: path.read_bytes() for path in directory.iterdir()}
    root = tmp_path.resolve()
    linked_directory = root / "junction-bundle"
    assert directory.resolve().is_relative_to(root)
    assert linked_directory.parent == root and not linked_directory.exists()
    _winapi.CreateJunction(str(directory.resolve()), str(linked_directory))
    try:
        assert linked_directory.is_junction()
        assert linked_directory.resolve() == directory.resolve()
        with pytest.raises(wb.WeightBundleError, match="link"):
            wb.validate_weight_bundle(linked_directory)
        with pytest.raises(wb.WeightBundleError, match="link"):
            wb.write_weight_bundle(specs, arrays, linked_directory / "new")
        assert not (directory / "new").exists()
    finally:
        # rmdir removes the junction itself, never the nonempty target tree.
        if linked_directory.is_junction():
            os.rmdir(linked_directory)
    assert not linked_directory.exists()
    assert directory.is_dir()
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == original
    wb.validate_weight_bundle(directory)


def test_detects_mutation_between_hashing_and_final_stat(tmp_path, monkeypatch):
    directory, _, _, _ = _bundle(tmp_path)
    original = wb._check_unchanged

    def mutate_after_read(stream, path, before):
        if path.name == "weights.bin":
            with path.open("ab") as writer:
                writer.write(b"changed after hashing")
        return original(stream, path, before)

    monkeypatch.setattr(wb, "_check_unchanged", mutate_after_read)
    with pytest.raises(wb.WeightBundleError, match="changed during validation"):
        wb.validate_weight_bundle(directory)


def test_validation_result_does_not_claim_future_immutability(tmp_path):
    directory, _, _, _ = _bundle(tmp_path)
    wb.validate_weight_bundle(directory)
    with (directory / "weights.bin").open("ab") as stream:
        stream.write(b"later mutation")
    with pytest.raises(wb.WeightBundleError):
        wb.validate_weight_bundle(directory)


def test_manifest_size_is_bounded_before_json_parsing(tmp_path, monkeypatch):
    directory, _, _, _ = _bundle(tmp_path)
    monkeypatch.setattr(wb, "MAX_MANIFEST_BYTES", 8)
    with pytest.raises(wb.WeightBundleError, match="size limit"):
        wb.validate_weight_bundle(directory)


def test_plan_has_aligned_disjoint_regions_and_ram_margin_for_large_weights():
    plan = wb.plan_guest_memory(workspace_bytes=768 * wb.MIB + 3,
                                weight_bytes=2500 * wb.MIB + 4,
                                input_bytes=17, output_bytes=152000 * 4)
    assert plan["code_base"] == 0x80000000
    assert plan["code_capacity"] == 128 * wb.MIB
    assert plan["stack_top"] == plan["code_base"] + plan["code_capacity"]
    assert plan["stack_top"] - plan["stack_bottom"] == wb.MIB
    assert plan["code_limit"] == plan["stack_bottom"]
    regions = list(plan["regions"].values())
    assert list(plan["regions"]) == ["code", "stack", "inputs", "outputs", "workspace", "weights"]
    for region in regions:
        assert region["base"] % 64 == 0
        assert region["end"] - region["base"] == region["nbytes"]
    for previous, following in zip(regions, regions[1:]):
        assert previous["end"] <= following["base"]
    assert plan["ram_end"] > regions[-1]["end"]
    assert plan["ram_bytes"] == plan["memory_mib"] * wb.MIB
    assert plan["margin_bytes"] >= 16 * wb.MIB
    assert plan["margin_bytes"] * 20 >= plan["used_bytes"]
    assert plan["regions"]["weights"]["end"] > 2**32


@pytest.mark.parametrize("kwargs", [dict(code_capacity=wb.MIB), dict(code_capacity=wb.MIB + 1),
                                    dict(code_capacity=True), dict(workspace_bytes=True),
                                    dict(weight_bytes=-1), dict(input_bytes=1.0),
                                    dict(output_bytes=np.bool_(True)), dict(ram_base=True),
                                    dict(ram_base=0x80000001), dict(weight_bytes=2**64),
                                    dict(ram_base=2**64 - 64), dict(weight_bytes=2**64 - 1),
                                    dict(weight_bytes=2**64 - 0x80000000 - 128 * wb.MIB - 1)])
def test_plan_rejects_invalid_dimensions_and_uint64_overflow(kwargs):
    with pytest.raises(wb.WeightBundleError):
        wb.plan_guest_memory(**kwargs)


def test_zero_size_layout_and_small_custom_code_reservation():
    plan = wb.plan_guest_memory(code_capacity=2 * wb.MIB)
    assert plan["code_limit"] - plan["code_base"] == wb.MIB
    assert plan["input_base"] == plan["output_base"] == plan["workspace_base"] == plan["weights_base"]
    assert plan["memory_mib"] == 18
