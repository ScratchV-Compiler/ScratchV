"""Read-only evidence for complete ONNX -> IR translation and weight binding.

This deliberately audits the fixed, single-output Qwen graph. It does not run
the model or claim numerical equivalence. External tensors are hashed in chunks
without loading another copy of the complete weights.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path, PureWindowsPath

import numpy as np
import onnx

from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.ir.types import DataType, OpCode
from probes.w2_qwen3_parse.validation import make_control_constant_reader


class AuditError(ValueError):
    """Translation or binding evidence is incomplete or inconsistent."""


_TYPES = {1: DataType.FLOAT32, 6: DataType.INT32,
          7: DataType.INT64, 11: DataType.FLOAT64}
_CHUNK_BYTES = 1024 * 1024


def _require(condition, message):
    if not condition:
        raise AuditError(message)


def _value_record(value):
    return {"ir_name": value.name, "shape": list(value.shape),
            "dtype": value.dtype.value}


class AuditedONNXParser(ONNXParser):
    """The production parser, with per-node observations rather than new lowering."""

    def __init__(self, progress=None):
        super().__init__()
        self.progress = progress
        self.node_records = []
        self.current_context = {"stage": "not_started"}
        self._node_instructions = []

    def parse(self, model_path):
        self.node_records = []
        self._node_instructions = []
        self.current_context = {"stage": "parse", "model_path": str(model_path)}
        if self.progress:
            self.progress(dict(self.current_context))
        program = super().parse(model_path)
        self.current_context = {"stage": "parsed", "node_count": len(self.node_records)}
        if self.progress:
            self.progress(dict(self.current_context))
        return program

    def _translate_node(self, node, output_names):
        index = len(self.node_records)
        block = self.builder.current_block
        start = len(block.instructions)
        self.current_context = {"stage": "translate_node", "node_index": index,
                                "node_name": node.name, "op_type": node.op_type,
                                "inputs": list(node.input), "outputs": list(node.output)}
        if self.progress:
            self.progress(dict(self.current_context))
        record = {**self.current_context, "ir_start": start, "status": "running",
                  "input_values": {name: _value_record(self._value_map[name])
                                   for name in node.input if name in self._value_map}}
        try:
            super()._translate_node(node, output_names)
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            self.node_records.append(record)
            raise
        instructions = block.instructions[start:]
        record.update(status="translated", ir_end=len(block.instructions),
                      output_values={name: _value_record(self._value_map[name])
                                     for name in node.output if name in self._value_map},
                      instructions=[{"opcode": inst.opcode.value,
                                     "dest": inst.dest.name if inst.dest else None,
                                     "operands": [value.name for value in inst.operands],
                                     "attrs": dict(inst.attrs)} for inst in instructions])
        self.node_records.append(record)
        self._node_instructions.append(instructions)


def _array_hash(data):
    """Hash logical C-order little-endian bytes with bounded temporary buffers."""
    digest = hashlib.sha256()
    array = np.asarray(data)
    little_dtype = array.dtype.newbyteorder("<")
    if array.flags.c_contiguous and array.dtype == little_dtype:
        view = memoryview(array.reshape(-1)).cast("B")
        for offset in range(0, len(view), _CHUNK_BYTES):
            digest.update(view[offset:offset + _CHUNK_BYTES])
    else:
        iterator = np.nditer(array, flags=["external_loop", "buffered", "zerosize_ok"],
                             op_flags=["readonly"], op_dtypes=[little_dtype],
                             order="C", buffersize=max(1, _CHUNK_BYTES // array.itemsize))
        for chunk in iterator:
            digest.update(chunk.tobytes(order="C"))
    return digest.hexdigest()


def _external_hash(tensor, model_dir, expected_bytes):
    entries = list(tensor.external_data)
    fields = {item.key: item.value for item in entries}
    _require(len(fields) == len(entries), f"{tensor.name}: duplicate external-data keys")
    location = fields.get("location", "")
    windows = PureWindowsPath(location)
    _require(bool(location) and "\x00" not in location and not windows.drive
             and not windows.is_absolute() and not Path(location).is_absolute(),
             f"{tensor.name}: invalid external-data location")
    root = Path(model_dir).resolve()
    path = (root / location.replace("\\", "/")).resolve()
    _require(path.is_relative_to(root) and path.is_file(),
             f"{tensor.name}: external data missing or outside model directory: {location}")
    for key in ("offset", "length"):
        if key in fields:
            _require(fields[key].isascii() and fields[key].isdecimal(),
                     f"{tensor.name}: invalid external-data {key}")
    offset = int(fields.get("offset", 0))
    size = path.stat().st_size
    length = int(fields.get("length", size - offset))
    _require(offset <= size and length == expected_bytes and offset + length <= size,
             f"{tensor.name}: external-data range/length mismatch")
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as stream:
        stream.seek(offset)
        while remaining:
            chunk = stream.read(min(_CHUNK_BYTES, remaining))
            _require(bool(chunk), f"{tensor.name}: truncated external data")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest(), {"location": location, "offset": offset, "length": length}


def _tensor_spec(tensor, model_dir):
    _require(tensor.data_type in _TYPES, f"{tensor.name}: unsupported tensor dtype")
    dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type))
    shape = tuple(tensor.dims)
    _require(all(dim >= 0 for dim in shape), f"{tensor.name}: negative tensor dimension")
    size = math.prod(shape) * dtype.itemsize
    if tensor.data_location == onnx.TensorProto.EXTERNAL:
        sha, external = _external_hash(tensor, model_dir, size)
    else:
        sha = _array_hash(onnx.numpy_helper.to_array(tensor))
        external = None
    return dtype, shape, size, sha, external


def _onnx_attributes(node):
    """Read source protobuf directly, independently of production lowering."""
    attributes = {attribute.name: onnx.helper.get_attribute_value(attribute)
                  for attribute in node.attribute}
    _require(len(attributes) == len(node.attribute), f"{node.name}: duplicate ONNX attributes")
    return attributes


def _constant_spec(node, model_dir):
    attrs = _onnx_attributes(node)
    _require(len(attrs) == 1, f"{node.name}: Constant requires exactly one attribute")
    key, data = next(iter(attrs.items()))
    if key == "value":
        return _tensor_spec(data, model_dir)
    if key in ("value_int", "value_ints"):
        array = np.asarray(data, dtype=np.int64)
    elif key in ("value_float", "value_floats"):
        array = np.asarray(data, dtype=np.float32)
    else:
        raise AuditError(f"{node.name}: unsupported Constant attribute {key}")
    return array.dtype, array.shape, array.nbytes, _array_hash(array), None


def _same_literal(left, right):
    if left == right:
        return left != 0 or math.copysign(1, left) == math.copysign(1, right)
    return (isinstance(left, float) and isinstance(right, float)
            and math.isnan(left) and math.isnan(right))


def _audit_scalar_load(instruction, data, dtype, context):
    """Cover auxiliary LOAD_CONST results which have no ONNX value-map entry."""
    destination = instruction.dest
    _require(instruction.opcode == OpCode.LOAD_CONST and destination is not None
             and not instruction.operands and not instruction.target
             and destination.dtype == dtype and destination.shape == ()
             and destination.is_constant is True
             and _same_literal(destination.const_value, data.item())
             and set(instruction.attrs) == {"value"}
             and _same_literal(instruction.attrs.get("value"), data.item()),
             f"{context}: scalar instruction metadata mismatch")


def _expected_lowering(node, parser, read_control):
    """Independent structural contract for the operations in the fixed graph."""
    attrs = _onnx_attributes(node)
    names = [name for name in node.input if name]
    args = [parser._value_map[name] for name in names]
    op = node.op_type
    ir_attrs = {}
    operands = args
    if op in {"Abs", "Add", "Cast", "Cos", "Div", "Exp", "Gather", "MatMul",
              "Mul", "Neg", "Pow", "Reciprocal", "Relu", "Sigmoid", "Sin",
              "Softmax", "Sqrt", "Sub", "Transpose", "Concat", "Reshape",
              "Expand", "Unsqueeze", "Slice", "ReduceMean"}:
        opcode = "reduce_mean" if op == "ReduceMean" else op.lower()
    else:
        raise AuditError(f"{node.name}: no independent lowering contract for {op}")

    def integers(index):
        # Never ask the parser/IR/cache to certify its own shape evaluation.
        # This reader resolves the original ONNX graph with bounded NumPy ops.
        name = node.input[index]
        array = read_control(name)
        _require(array.ndim == 1 and array.dtype.kind in "iu",
                 f"{node.name}: {name} is not a constant integer vector")
        return tuple(int(value) for value in array)

    if op == "Cast":
        _require(set(attrs) == {"to"} and attrs["to"] in _TYPES,
                 f"{node.name}: unsupported Cast attributes")
        # CAST encodes its target in dest.dtype, not an instruction attribute.
        ir_attrs = {}
    elif op in ("Gather", "Softmax"):
        ir_attrs = {"axis": attrs.get("axis", 0 if op == "Gather" else -1)}
    elif op == "MatMul":
        a, b = args
        if len(a.shape) == len(b.shape) == 2 and all(d > 0 for d in a.shape + b.shape):
            ir_attrs = {"m": a.shape[0], "n": b.shape[1], "k": a.shape[1]}
    elif op in ("Reshape", "Expand"):
        _require(not attrs.get("allowzero", 0), f"{node.name}: unsupported allowzero")
        ir_attrs = {"shape": integers(1)}
        operands = args[:1]
    elif op == "Unsqueeze":
        ir_attrs = {"axes": integers(1) if len(args) > 1 else tuple(attrs["axes"])}
        operands = args[:1]
    elif op == "Slice":
        for index, key in enumerate(("starts", "ends", "axes", "steps"), 1):
            if len(node.input) > index and node.input[index]:
                ir_attrs[key] = integers(index)
            elif key in attrs:
                ir_attrs[key] = tuple(attrs[key])
        operands = args[:1]
    elif op == "ReduceMean":
        axes = integers(1) if len(args) > 1 else attrs.get("axes")
        if not axes and attrs.get("noop_with_empty_axes", 0):
            return None, args[:1], {}
        ir_attrs = {"keepdims": bool(attrs.get("keepdims", 1))}
        if axes is not None:
            ir_attrs["axes"] = tuple(axes)
        operands = args[:1]
    elif op == "Transpose" and "perm" in attrs:
        ir_attrs = {"perm": tuple(attrs["perm"])}
    elif op == "Concat":
        ir_attrs = {"axis": attrs["axis"]}
    return opcode, operands, ir_attrs


def audit_parsed_model(model, parser, program, model_dir):
    """Raise AuditError on any gap; otherwise return serializable evidence."""
    graph = model.graph
    _require(len(graph.output) == 1, "fixed-graph audit requires exactly one graph output")
    _require(program is parser.builder.program, "program is not the observed parser result")
    _require(len(program.functions) == 1 and len(program.functions[0].blocks) == 1,
             "fixed-graph audit requires one function and one basic block")
    function = program.functions[0]
    instructions = function.blocks[0].instructions
    _require(len(parser.node_records) == len(graph.node)
             and len(parser._node_instructions) == len(graph.node),
             "translated node count does not cover the ONNX graph")
    read_control = make_control_constant_reader(model)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    metadata = {value.name: value.type.tensor_type for value in
                [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]}
    expected_names = {value.name for value in [*graph.input, *graph.initializer]}
    initializer_names = {value.name for value in graph.initializer}
    _require(len(initializer_names) == len(graph.initializer), "duplicate initializer names")
    def progress(stage, **context):
        parser.current_context = {"stage": stage, **context}
        if parser.progress:
            parser.progress(dict(parser.current_context))

    bindings = []
    for value in graph.initializer:
        progress("audit_source_tensor", tensor_name=value.name, source="initializer")
        bindings.append((value.name, "initializer", _tensor_spec(value, model_dir)))
    for node in graph.node:
        _require(len(node.output) == 1 and bool(node.output[0]),
                 f"{node.name}: audit requires one nonempty node output")
        _require(node.output[0] not in expected_names, f"{node.name}: duplicate ONNX definition")
        expected_names.add(node.output[0])
        if node.op_type == "Constant":
            progress("audit_source_tensor", tensor_name=node.output[0], source="Constant")
            bindings.append((node.output[0], "Constant", _constant_spec(node, model_dir)))
    _require(set(parser._value_map) == expected_names,
             "ONNX -> IR value mapping has missing or unexpected names")
    _require(graph.output[0].name in expected_names,
             "graph output does not have an ONNX definition")
    for name in expected_names - initializer_names:
        info = metadata.get(name)
        _require(info is not None and info.HasField("shape")
                 and all(dim.HasField("dim_value") for dim in info.shape.dim),
                 f"{name}: missing static ONNX shape")
        _require(info.elem_type in _TYPES, f"{name}: unsupported ONNX dtype")
        value = parser._value_map[name]
        _require(value.shape == tuple(dim.dim_value for dim in info.shape.dim)
                 and value.dtype == _TYPES[info.elem_type], f"{name}: IR shape/dtype mismatch")

    globals_by_name = {value.name: value for value in program.global_values}
    binding_names = {name for name, _, _ in bindings}
    _require(len(globals_by_name) == len(program.global_values)
             and set(globals_by_name) == binding_names,
             "IR global definitions are missing, duplicated, or unexpected")
    _require(set(parser.initializers) == binding_names,
             "initializer bindings are missing or unexpected")
    tensor_rows = []
    for name, source, (dtype, shape, size, expected_sha, external) in bindings:
        progress("audit_bound_tensor", tensor_name=name, source=source)
        array = parser.initializers[name]
        _require(isinstance(array, np.ndarray), f"{name}: binding must be a NumPy array")
        value = globals_by_name[name]
        _require(value is parser._value_map[name], f"{name}: global/value mapping identity mismatch")
        _require(array.dtype == dtype and array.shape == shape and value.shape == shape
                 and value.dtype == _TYPES[onnx.helper.np_dtype_to_tensor_dtype(dtype)],
                 f"{name}: tensor binding shape/dtype mismatch")
        _require(value.is_constant == (len(shape) == 0), f"{name}: scalar metadata mismatch")
        if len(shape) == 0:
            _require(_same_literal(value.const_value, array.item()), f"{name}: scalar value mismatch")
        actual_sha = _array_hash(array)
        _require(actual_sha == expected_sha, f"{name}: bound tensor content SHA256 mismatch")
        tensor_rows.append({"name": name, "source": source, "shape": list(shape),
                            "dtype": str(dtype), "bytes": size, "sha256": actual_sha,
                            "external": external})
    expected_params = [parser._value_map[value.name] for value in graph.input
                       if value.name not in initializer_names]
    _require(len(function.params) == len(expected_params)
             and all(a is b for a, b in zip(function.params, expected_params)),
             "function parameters do not match ONNX inputs")
    expected_param_names = [value.name for value in graph.input if value.name not in initializer_names]
    _require([value.name for value in function.params] == expected_param_names,
             "function parameter names do not match ONNX input bindings")
    _require(not any(value.is_constant for value in function.params),
             "function parameters must not be constants")

    scalar_initializers = [value for value in graph.initializer if len(value.dims) == 0]
    cursor = len(scalar_initializers)
    for inst, tensor in zip(instructions[:cursor], scalar_initializers):
        data = parser.initializers[tensor.name]
        _audit_scalar_load(inst, data, parser._value_map[tensor.name].dtype,
                           f"{tensor.name}: scalar initializer")
    alias_rows = []
    for index, (node, record, observed) in enumerate(zip(
            graph.node, parser.node_records, parser._node_instructions)):
        progress("audit_node", node_index=index, node_name=node.name, op_type=node.op_type)
        context = f"node #{index} {node.name} ({node.op_type})"
        _require(record.get("status") == "translated" and record.get("node_index") == index
                 and record.get("node_name") == node.name and record.get("op_type") == node.op_type
                 and record.get("inputs") == list(node.input) and record.get("outputs") == list(node.output),
                 f"{context}: node observation mismatch")
        end = record.get("ir_end")
        _require(record.get("ir_start") == cursor and isinstance(end, int)
                 and end == cursor + len(observed)
                 and all(a is b for a, b in zip(instructions[cursor:end], observed))
                 and len(instructions[cursor:end]) == len(observed),
                 f"{context}: emitted IR range changed or has a gap")
        cursor = end
        output = parser._value_map[node.output[0]]
        if node.op_type == "Constant":
            _require(not node.input, f"{context}: Constant has inputs")
            array = parser.initializers[node.output[0]]
            _require(len(observed) == (1 if array.ndim == 0 else 0),
                     f"{context}: Constant emitted unexpected instructions")
            if observed:
                _audit_scalar_load(observed[0], array, output.dtype, f"{context}: Constant")
            continue
        if node.op_type == "Identity":
            _require(len(node.input) == 1, f"{context}: invalid Identity inputs")
            opcode, operands, ir_attrs = None, [parser._value_map[node.input[0]]], {}
        else:
            opcode, operands, ir_attrs = _expected_lowering(node, parser, read_control)
        if opcode is None:
            _require(not observed and output is operands[0], f"{context}: incorrect alias binding")
            alias_rows.append({"onnx_name": node.output[0], "input": node.input[0],
                               "ir_name": output.name})
        else:
            _require(len(observed) == 1, f"{context}: node did not emit exactly one instruction")
            inst = observed[0]
            _require(inst.opcode.value == opcode and inst.dest is output,
                     f"{context}: wrong IR opcode or output binding")
            _require(len(inst.operands) == len(operands)
                     and all(a is b for a, b in zip(inst.operands, operands)),
                     f"{context}: wrong IR dependency bindings")
            _require(inst.attrs == ir_attrs, f"{context}: IR attributes mismatch")
    _require(len(instructions) == cursor + 1, "unexpected instructions outside node ranges and RETURN")
    returns = [inst for inst in instructions if inst.opcode == OpCode.RETURN]
    output_name = graph.output[0].name
    _require(len(returns) == 1 and instructions[-1] is returns[0]
             and len(returns[0].operands) == 1
             and returns[0].operands[0] is parser._value_map[output_name],
             "actual RETURN does not bind the unique ONNX graph output")
    progress("audit_complete", node_count=len(graph.node), binding_count=len(bindings))
    return {"passed": True, "node_count": len(graph.node), "value_count": len(expected_names),
            "ir_instruction_count": len(instructions), "global_count": len(globals_by_name),
            "initializer_count": len(graph.initializer),
            "constant_count": len(bindings) - len(graph.initializer),
            "binding_count": len(bindings), "binding_bytes": sum(row["bytes"] for row in tensor_rows),
            "tensor_bindings": tensor_rows, "alias_count": len(alias_rows), "aliases": alias_rows,
            "return": {"onnx_name": output_name, **_value_record(parser._value_map[output_name])},
            "node_records": parser.node_records}
