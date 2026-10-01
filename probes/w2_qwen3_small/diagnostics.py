"""Observe named FP32 checkpoints in one IR execution without compiler hooks.

The diagnostic graph preserves every original output and adds a flattened
concatenation as its first output, matching ScratchV's single-return contract.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _names(names, label):
    names = list(names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{label} must contain nonempty checkpoint names")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} contains duplicate checkpoint names")
    return names


def build_diagnostic_model(model: onnx.ModelProto,
                           references: Mapping[str, np.ndarray]):
    """Copy a static model and pack all outputs in reference insertion order."""
    if not isinstance(references, Mapping):
        raise ValueError("references must be a mapping")
    names = _names(references, "references")
    outputs = list(model.graph.output)
    output_names = _names((value.name for value in outputs), "graph outputs")
    if set(names) != set(output_names):
        raise ValueError("references must exactly match graph outputs")
    by_name = {value.name: value for value in outputs}
    schema, offset = [], 0
    for name in names:
        reference = references[name]
        if not isinstance(reference, np.ndarray) or reference.dtype != np.dtype("float32"):
            raise ValueError(f"checkpoint {name} must be an FP32 NumPy array")
        info = by_name[name].type
        if not info.HasField("tensor_type") or info.tensor_type.elem_type != TensorProto.FLOAT:
            raise ValueError(f"checkpoint {name} must have FP32 tensor output type")
        tensor = info.tensor_type
        if not tensor.HasField("shape") or any(
                not dim.HasField("dim_value") or dim.dim_value < 0 for dim in tensor.shape.dim):
            raise ValueError(f"checkpoint {name} must have a static shape")
        shape = tuple(dim.dim_value for dim in tensor.shape.dim)
        if reference.shape != shape:
            raise ValueError(f"checkpoint {name} reference shape disagrees with graph output")
        size = math.prod(shape)
        schema.append(dict(name=name, shape=list(shape), offset=offset, size=size))
        offset += size

    diagnostic = copy.deepcopy(model)
    graph = diagnostic.graph
    reserved = {value.name for value in [*graph.input, *graph.output,
                                         *graph.value_info, *graph.initializer]}
    reserved.update(name for node in graph.node for name in [node.name, *node.input, *node.output])

    def fresh(base):
        candidate, suffix = base, 1
        while candidate in reserved:
            candidate = f"{base}_{suffix}"
            suffix += 1
        reserved.add(candidate)
        return candidate

    shape_name = fresh("__trace_flat_shape")
    graph.initializer.append(numpy_helper.from_array(np.array([-1], dtype=np.int64), shape_name))
    flat_names = []
    for index, entry in enumerate(schema):
        flattened = fresh(f"__trace_flat_{index}")
        graph.node.append(helper.make_node("Reshape", [entry["name"], shape_name], [flattened],
                                           name=fresh(f"__trace_reshape_{index}")))
        flat_names.append(flattened)
    packed = fresh("trace_pack")
    graph.node.append(helper.make_node("Concat", flat_names, [packed], axis=0,
                                       name=fresh("__trace_concat")))
    del graph.output[:]
    graph.output.append(helper.make_tensor_value_info(packed, TensorProto.FLOAT, [offset]))
    graph.output.extend(copy.deepcopy(outputs))
    return diagnostic, schema


def _nonnegative_integer(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def unpack_trace(array: np.ndarray, schema: list[dict]) -> dict[str, np.ndarray]:
    """Validate the entire schema and return independent checkpoint arrays."""
    if not isinstance(array, np.ndarray) or array.dtype != np.dtype("float32") or array.ndim != 1:
        raise ValueError("trace_pack must be a one-dimensional FP32 NumPy array")
    if not isinstance(schema, (list, tuple)) or not schema:
        raise ValueError("schema must contain checkpoints")
    if any(not isinstance(entry, Mapping) or set(entry) != {"name", "shape", "offset", "size"}
           for entry in schema):
        raise ValueError("schema entries must contain exactly name/shape/offset/size")
    _names((entry["name"] for entry in schema), "schema")
    offset, validated = 0, []
    for entry in schema:
        shape = entry["shape"]
        if not isinstance(shape, (list, tuple)):
            raise ValueError("checkpoint shape must be an integer sequence")
        shape = tuple(_nonnegative_integer(dim, "shape dimension") for dim in shape)
        start = _nonnegative_integer(entry["offset"], "offset")
        size = _nonnegative_integer(entry["size"], "size")
        if start != offset or size != math.prod(shape):
            raise ValueError("schema offsets must be contiguous and sizes must match shapes")
        validated.append((entry["name"], shape, start, size))
        offset += size
    if array.shape != (offset,):
        raise ValueError("trace_pack length disagrees with schema total size")
    return {name: array[start:start + size].reshape(shape).copy()
            for name, shape, start, size in validated}


def _json_scalar(value):
    value = value.item() if isinstance(value, np.generic) else value
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("+Inf" if value > 0 else "-Inf")
    return value


def tensor_diff(actual, expected, atol=1e-5) -> dict:
    """Compare exact shape/type and finite values with strict max_abs < atol.

    Empty, equally shaped/typed tensors pass with max_abs=0 and no worst index.
    Nonfinite values are represented by strings in the JSON diagnostic.
    """
    if isinstance(atol, (bool, np.bool_)) or not isinstance(atol, Real):
        raise ValueError("atol must be finite and positive")
    atol = float(atol)
    if not math.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be finite and positive")
    arrays = [value if isinstance(value, np.ndarray) else None for value in (actual, expected)]
    finite = [bool(np.isfinite(value).all())
              if value is not None and value.dtype.kind in "biuf" else False for value in arrays]
    actual_array, expected_array = arrays
    comparable = all(value is not None for value in arrays)
    shape_matches = comparable and actual.shape == expected.shape
    dtype_matches = comparable and actual.dtype == expected.dtype
    report = dict(passed=False, max_abs=None,
                  actual_shape=list(actual.shape) if actual_array is not None else None,
                  expected_shape=list(expected.shape) if expected_array is not None else None,
                  actual_dtype=str(actual.dtype) if actual_array is not None else None,
                  expected_dtype=str(expected.dtype) if expected_array is not None else None,
                  shape_matches=bool(shape_matches), dtype_matches=bool(dtype_matches),
                  actual_finite=finite[0], expected_finite=finite[1], finite=all(finite),
                  empty=bool(comparable and actual.size == expected.size == 0),
                  worst_index=None, actual_value=None, expected_value=None, reason=None, atol=atol)
    if not comparable:
        report["reason"] = "outputs must be NumPy arrays"
        return report
    if not shape_matches or not dtype_matches:
        report["reason"] = "shape mismatch" if not shape_matches else "dtype mismatch"
        return report
    if actual.dtype.kind not in "biuf":
        report["reason"] = "outputs must have numeric dtypes"
        return report
    if not all(finite):
        invalid = ~np.isfinite(actual) | ~np.isfinite(expected)
        index = np.unravel_index(int(np.flatnonzero(invalid)[0]), actual.shape)
        report.update(worst_index=[int(dim) for dim in index],
                      actual_value=_json_scalar(actual[index]),
                      expected_value=_json_scalar(expected[index]), reason="nonfinite output or reference")
        return report
    if not actual.size:
        report.update(passed=True, max_abs=0.0)
        return report
    if actual.dtype.kind in "biu":
        differences = np.abs(actual.astype(object) - expected.astype(object))
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            differences = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    index = np.unravel_index(int(np.argmax(differences)), actual.shape)
    maximum = float(differences[index])
    passed = math.isfinite(maximum) and maximum < atol
    report.update(passed=passed, max_abs=maximum if math.isfinite(maximum) else None,
                  worst_index=[int(dim) for dim in index],
                  actual_value=_json_scalar(actual[index]), expected_value=_json_scalar(expected[index]),
                  reason=None if passed else ("absolute error exceeds tolerance"
                                              if math.isfinite(maximum) else "absolute error overflow"))
    return report


def compare_outputs(actual: Mapping[str, np.ndarray], expected: Mapping[str, np.ndarray],
                    order: Sequence[str], atol=1e-5) -> dict:
    """Compare every checkpoint in explicit diagnostic order without dropping keys."""
    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        raise ValueError("actual and expected outputs must be mappings")
    if isinstance(order, (str, bytes)) or not isinstance(order, Sequence):
        raise ValueError("checkpoint order must be a sequence of names")
    names = _names(order, "checkpoint order")
    if set(actual) != set(names) or set(expected) != set(names):
        raise ValueError("actual/expected keys must exactly match checkpoint order")
    checkpoints = [dict(name=name, **tensor_diff(actual[name], expected[name], atol)) for name in names]
    first = next((checkpoint["name"] for checkpoint in checkpoints if not checkpoint["passed"]), None)
    return dict(passed=first is None, first_divergence=first, checkpoints=checkpoints)
