"""Opt-in weight borrowing / last-use execution never weaken default or D8."""

import weakref

import numpy as np
import pytest

from benchmarks.ir_interpreter_cases import builder
from scratchv.frontend.onnx_parser import ONNXParseError, ONNXParser
from scratchv.ir.types import DataType as D, Value
from scratchv.verification.ir_interpreter import IRExecutionError, IRInterpreter


def test_last_use_releases_chain_without_changing_result_or_steps():
    x = Value("x", shape=(1024,))
    b = builder(x)
    current = x
    for _ in range(20):
        current = b.neg(current)
    b.ret(current)
    inputs = {"x": np.arange(1024, dtype=np.float32)}
    interp = IRInterpreter(b.program)
    old = interp.run(inputs, collect_memory_stats=True)
    new = interp.run(inputs, memory_mode="last_use", collect_memory_stats=True)
    np.testing.assert_array_equal(old.return_value, new.return_value)
    assert old.executed_steps == new.executed_steps
    assert old.diagnostics == new.diagnostics
    assert old.memory_stats["retained_values"] == 21
    assert new.memory_stats["retained_values"] == 1
    assert new.memory_stats["reclaimed_bindings"] == 20
    assert new.memory_stats["peak_numpy_storage_bytes"] == 2 * inputs["x"].nbytes
    assert old.memory_stats["peak_numpy_storage_bytes"] == 22 * inputs["x"].nbytes
    assert not np.shares_memory(new.return_value, inputs["x"])


def test_repeated_operand_residual_and_alias_views_survive_last_use():
    x = Value("x", shape=(2, 3))
    b = builder(x)
    doubled = b.add(x, x)
    view = b.transpose(b.reshape(doubled, (3, 2)), (1, 0))
    result = b.add(view, x)
    b.ret(result)
    source = np.arange(6, dtype="float32").reshape(2, 3)
    interp = IRInterpreter(b.program)
    old = interp.run({"x": source})
    new = interp.run({"x": source}, memory_mode="last_use", collect_memory_stats=True)
    np.testing.assert_array_equal(old.return_value, new.return_value)
    assert new.memory_stats["retained_values"] == 1
    assert not np.shares_memory(new.return_value, source)


def test_borrowed_weights_are_shared_readonly_and_returns_are_copied(monkeypatch):
    import scratchv.verification.ir_interpreter as module

    weight = Value("w", shape=(2, 2))
    b = builder()
    b.program.global_values.append(weight)
    b.ret(b.transpose(weight, (1, 0)))
    source = np.arange(4, dtype="float32").reshape(2, 2)
    source.setflags(write=False)
    original_compute = module.compute
    seen = []

    def check(instr, operands):
        seen.append((np.shares_memory(operands[0], source), operands[0].flags.writeable))
        return original_compute(instr, operands)

    monkeypatch.setattr(module, "compute", check)
    result = IRInterpreter(b.program).run({}, initializers={"w": source},
        memory_mode="last_use", copy_initializers=False, collect_memory_stats=True)
    assert seen == [(True, False)]
    assert not np.shares_memory(result.return_value, source)
    result.return_value[0, 0] = 99
    assert source[0, 0] == 0 and not source.flags.writeable
    assert result.memory_stats["copy_initializers"] is False
    # Default still owns a separate initializer copy.
    IRInterpreter(b.program).run({}, initializers={"w": source})
    assert seen[-1] == (False, True)


def test_borrowing_rejects_writable_arrays_without_changing_flags():
    weight = Value("w", shape=(2,))
    b = builder()
    b.program.global_values.append(weight)
    b.ret(weight)
    source = np.ones(2, dtype=np.float32)
    with pytest.raises(IRExecutionError, match="borrowed initializer must be read-only"):
        IRInterpreter(b.program).run({}, initializers={"w": source}, copy_initializers=False)
    assert source.flags.writeable


@pytest.mark.parametrize("options", [{"memory_mode": "last_use"}, {"copy_initializers": False}])
@pytest.mark.parametrize("kind", ["branch", "loop", "memory"])
def test_optimized_modes_fail_closed_for_control_flow_and_memory(kind, options):
    b = builder()
    if kind == "branch":
        b.br("next")
        b.new_block("next")
    elif kind == "loop":
        b.for_loop(0, 2)
        b.endfor()
    else:
        b.alloca(4, D.INT32)
    b.ret()
    assert IRInterpreter(b.program).run({}).return_value is None
    with pytest.raises(IRExecutionError, match="memory optimization requires"):
        IRInterpreter(b.program).run({}, **options)


def test_observer_runs_before_release_is_readonly_and_does_not_pin_values():
    x = Value("x", shape=(32,))
    b = builder(x)
    dead = b.neg(x)
    y = b.add(x, x)
    z = b.neg(y)
    b.ret(z)
    seen, refs = [], []

    def observe(name, array):
        assert not array.flags.writeable
        assert array.base is not None
        seen.append(name)
        refs.append(weakref.ref(array.base))
        if name == y.name:
            assert refs[0]() is None  # Unused dead result was really released.

    result = IRInterpreter(b.program).run({"x": np.ones(32, dtype="float32")},
                                          memory_mode="last_use", observer=observe)
    assert seen == [dead.name, y.name, z.name]
    assert all(reference() is None for reference in refs)
    np.testing.assert_array_equal(result.return_value, np.full(32, -2, dtype="float32"))


def test_observer_failure_has_ir_location():
    b = builder()
    output = b.neg(b.make_const(1.0))
    b.ret(output)

    def broken(name, array):
        raise OSError("checkpoint disk full")

    with pytest.raises(IRExecutionError, match="ObserverError.*checkpoint disk full") as caught:
        IRInterpreter(b.program).run({}, memory_mode="last_use", observer=broken)
    assert caught.value.value_name == output.name
    assert caught.value.instruction_index == 0


def test_unused_operation_still_raises_numeric_error():
    b = builder()
    b.div(b.make_const(1.0), b.make_const(0.0))
    b.ret(b.make_const(7.0))
    for mode in ("retain_all", "last_use"):
        with pytest.raises(IRExecutionError, match="NumericError.*division by zero") as caught:
            IRInterpreter(b.program).run({}, memory_mode=mode)
        assert caught.value.instruction_index == 0


@pytest.mark.parametrize("options", [
    {"copy_initializers": 1}, {"memory_mode": None}, {"memory_mode": []},
    {"memory_mode": "automatic"}, {"observer": 1},
])
def test_invalid_options_are_rejected(options):
    b = builder()
    b.ret()
    with pytest.raises(IRExecutionError, match="InvalidOptions"):
        IRInterpreter(b.program).run({}, **options)


def external_model(tmp_path, *, fields=None, shape=(2, 3), trailing=0):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    data = np.arange(6, dtype="float32")
    (tmp_path / "weights.bin").write_bytes(b"HEAD" + data.tobytes() + b"T" * trailing)
    tensor = TensorProto(name="weight", data_type=TensorProto.FLOAT, dims=shape,
                         data_location=TensorProto.EXTERNAL)
    for key, value in fields if fields is not None else [
        ("location", "weights.bin"), ("offset", "4"), ("length", "24")
    ]:
        entry = tensor.external_data.add()
        entry.key, entry.value = key, value
    graph = helper.make_graph([helper.make_node("Neg", ["weight"], ["output"])],
        "external", [], [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        initializer=[tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    path = tmp_path / "model.onnx"
    path.write_bytes(model.SerializeToString())
    return path, data.reshape(shape)


def test_external_mapping_matches_default_and_borrowed_execution(tmp_path):
    path, data = external_model(tmp_path)
    normal, mapped = ONNXParser(), ONNXParser()
    normal.parse(str(path))
    program = mapped.parse(str(path), mmap_external_data=True)
    weight = mapped.initializers["weight"]
    assert isinstance(weight, np.memmap) and weight.mode == "r"
    assert not weight.flags.writeable
    np.testing.assert_array_equal(weight, normal.initializers["weight"])
    result = IRInterpreter(program).run({}, initializers=mapped.initializers,
                                       copy_initializers=False, memory_mode="last_use")
    np.testing.assert_array_equal(result.return_value, -data)
    assert not np.shares_memory(result.return_value, weight)
    with pytest.raises(ValueError):
        weight[0, 0] = 999


def test_external_mapping_does_not_require_python39_path_api(tmp_path, monkeypatch):
    from pathlib import Path

    import scratchv.frontend.onnx_parser as parser_module

    class Python38Path(type(Path())):
        @property
        def is_relative_to(self):
            raise AttributeError("is_relative_to is unavailable on Python 3.8")

        def relative_to(self, *other):
            # Newer pathlib implementations internally call is_relative_to;
            # delegate through an ordinary Path while hiding that newer API
            # from the parser under test, rather than patching global pathlib.
            return Path(str(self)).relative_to(*other)

    path, expected = external_model(tmp_path)
    parser = ONNXParser()
    monkeypatch.setattr(parser_module, "Path", Python38Path)
    assert not hasattr(Python38Path(), "is_relative_to")
    parser.parse(str(path), mmap_external_data=True)
    weight = parser.initializers["weight"]
    assert isinstance(weight, np.memmap) and not weight.flags.writeable
    np.testing.assert_array_equal(weight, expected)


@pytest.mark.parametrize("fields, message", [
    ([("location", "../weights.bin")], "model directory"),
    ([("location", "C:\\weights.bin")], "model directory"),
    ([("location", "weights.bin"), ("location", "weights.bin")], "Duplicate external"),
    ([("location", "weights.bin"), ("offset", "-1")], "nonnegative integer"),
    ([("location", "weights.bin"), ("length", "20")], "disagrees"),
    ([("location", "weights.bin"), ("offset", "5")], "exceeds file"),
    ([("location", "missing.bin")], "exceeds file"),
    ([("location", "weights.bin"), ("length", "24"), ("unknown", "x")], "Unsupported external"),
])
def test_external_mapping_rejects_unsafe_or_mismatched_metadata(tmp_path, fields, message):
    path, _ = external_model(tmp_path, fields=fields)
    with pytest.raises(ONNXParseError, match=message):
        ONNXParser().parse(str(path), mmap_external_data=True)


def test_default_external_loader_option_is_strict(tmp_path):
    path, _ = external_model(tmp_path)
    with pytest.raises(ONNXParseError, match="mmap_external_data must be boolean"):
        ONNXParser().parse(str(path), mmap_external_data=1)


def test_external_constant_node_uses_same_readonly_mapping(tmp_path):
    import onnx
    from onnx import helper

    path, expected = external_model(tmp_path)
    model = onnx.load(str(path), load_external_data=False)
    tensor = model.graph.initializer[0]
    node = helper.make_node("Constant", [], ["weight"], value=tensor)
    model.graph.node.insert(0, node)
    del model.graph.initializer[:]
    path.write_bytes(model.SerializeToString())
    parser = ONNXParser()
    parser.parse(str(path), mmap_external_data=True)
    assert isinstance(parser.initializers["weight"], np.memmap)
    assert not parser.initializers["weight"].flags.writeable
    np.testing.assert_array_equal(parser.initializers["weight"], expected)


def test_mapper_does_not_materialize_proto_or_read_outside_declared_range(tmp_path, monkeypatch):
    import onnx

    path, expected = external_model(tmp_path, trailing=64)
    tensor = onnx.load(str(path), load_external_data=False).graph.initializer[0]
    parser = ONNXParser()
    parser._base_dir = str(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("mmap loader must not materialize external protobuf data")

    monkeypatch.setattr(onnx.external_data_helper, "load_external_data_for_tensor", forbidden)
    array = parser._map_external_initializer(tensor)
    assert not tensor.HasField("raw_data")
    assert array.nbytes == 24 and array.offset == 4
    np.testing.assert_array_equal(array, expected)


def test_parser_mapping_option_resets_between_calls(tmp_path):
    path, _ = external_model(tmp_path)
    parser = ONNXParser()
    parser.parse(str(path), mmap_external_data=True)
    assert isinstance(parser.initializers["weight"], np.memmap)
    parser.parse(str(path))
    assert not isinstance(parser.initializers["weight"], np.memmap)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
def test_mapping_all_supported_numeric_types_scalar_and_empty_shapes(tmp_path, dtype, shape):
    from onnx import TensorProto, helper

    values = np.arange(np.prod(shape, dtype=int), dtype=dtype).reshape(shape)
    (tmp_path / "weights.bin").write_bytes(b"HEAD" + values.tobytes())
    tensor_dtype = helper.np_dtype_to_tensor_dtype(values.dtype)
    tensor = TensorProto(name="weight", data_type=tensor_dtype, dims=shape,
                         data_location=TensorProto.EXTERNAL)
    for key, value in [("location", "weights.bin"), ("offset", "4"),
                       ("length", str(values.nbytes))]:
        entry = tensor.external_data.add()
        entry.key, entry.value = key, value
    graph = helper.make_graph([helper.make_node("Neg", ["weight"], ["output"])],
        "external_types", [], [helper.make_tensor_value_info("output", tensor_dtype, shape)],
        initializer=[tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    path = tmp_path / "model.onnx"
    path.write_bytes(model.SerializeToString())
    parser = ONNXParser()
    program = parser.parse(str(path), mmap_external_data=True)
    assert not parser.initializers["weight"].flags.writeable
    result = IRInterpreter(program).run({}, initializers=parser.initializers,
                                       memory_mode="last_use", copy_initializers=False)
    assert result.return_value.dtype == values.dtype
    assert result.return_value.shape == shape
    np.testing.assert_array_equal(result.return_value, -values)


def test_memmap_backing_accounting_includes_mapping_alignment_and_deduplicates_views(tmp_path):
    import mmap

    from scratchv.verification.ir_interpreter import _MemoryObservation, _backing_storage

    offset = mmap.ALLOCATIONGRANULARITY + 4
    path = tmp_path / "aligned.bin"
    path.write_bytes(b"x" * offset + np.arange(6, dtype="float32").tobytes())
    mapped = np.memmap(path, mode="r", dtype="float32", offset=offset, shape=(2, 3))
    alias = mapped.T
    owner, mapped_bytes = _backing_storage(mapped)
    assert owner == _backing_storage(alias)[0]
    assert mapped_bytes == mapped.nbytes + offset % mmap.ALLOCATIONGRANULARITY
    logical, storage = _MemoryObservation.snapshot({"mapped": mapped, "alias": alias})
    assert logical == 2 * mapped.nbytes
    assert storage == mapped_bytes  # Mapping span, deliberately not physical RSS.
    second = np.memmap(path, mode="r", dtype="float32", offset=offset, shape=(2, 3))
    assert _backing_storage(second)[0] != owner
    assert _MemoryObservation.snapshot({"first": mapped, "second": second})[1] == 2 * mapped_bytes


def test_executed_dead_nonfinite_constant_is_not_skipped():
    b = builder()
    b.load_const(float("nan"))
    b.ret(b.make_const(7.0))
    for mode in ("retain_all", "last_use"):
        with pytest.raises(IRExecutionError, match="NumericError.*nonfinite"):
            IRInterpreter(b.program).run({}, memory_mode=mode)


def test_unreferenced_nan_initializer_keeps_original_binding_scope():
    unused = Value("unused", shape=(2,))
    b = builder()
    b.program.global_values.append(unused)
    b.ret(b.make_const(7.0))
    values = np.array([np.nan, np.nan], dtype="float32")
    values.setflags(write=False)
    # Neither mode validates/evaluates global tensors never referenced by this
    # function; dead *instructions* remain evaluated, as tested separately.
    for mode in ("retain_all", "last_use"):
        result = IRInterpreter(b.program).run({}, initializers={"unused": values},
                                             memory_mode=mode, copy_initializers=False)
        assert result.return_value == 7


def test_mapper_rejects_symlink_resolving_outside_model_directory(tmp_path, monkeypatch):
    from pathlib import Path

    from onnx import TensorProto

    base = tmp_path / "model"
    base.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(np.ones(1, dtype="float32").tobytes())
    link_path = base / "weights.bin"
    original = Path.resolve

    def emulate_symlink(path, *args, **kwargs):
        # Cross-platform exercise of the resolved-path boundary; creating real
        # Windows symlinks can require a privilege unrelated to this contract.
        if path == link_path:
            return outside
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", emulate_symlink)
    tensor = TensorProto(name="weight", data_type=TensorProto.FLOAT, dims=(1,),
                         data_location=TensorProto.EXTERNAL)
    entry = tensor.external_data.add()
    entry.key, entry.value = "location", "weights.bin"
    parser = ONNXParser()
    parser._base_dir = str(base)
    with pytest.raises(ONNXParseError, match="escapes the model directory"):
        parser._map_external_initializer(tensor)


def test_observer_accidental_write_fails_and_preserves_borrowed_weight():
    weight = Value("w", shape=(2, 2))
    b = builder()
    b.program.global_values.append(weight)
    b.ret(b.transpose(weight, (1, 0)))
    source = np.arange(4, dtype="float32").reshape(2, 2)
    expected = source.copy()
    source.setflags(write=False)

    def observe(name, array):
        array[0, 0] = 999

    with pytest.raises(IRExecutionError, match="ObserverError.*read-only"):
        IRInterpreter(b.program).run({}, initializers={"w": source},
            memory_mode="last_use", copy_initializers=False, observer=observe)
    np.testing.assert_array_equal(source, expected)
