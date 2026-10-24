"""Read typed initializer encodings before protobuf/Python float conversion.

Only the ModelProto -> GraphProto -> TensorProto path is inspected here.
ONNX remains responsible for parsing the model and loading external data.
Wire format: https://protobuf.dev/programming-guides/encoding/
"""

from pathlib import Path


def _varint(data: bytes, position: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if position >= len(data):
            raise ValueError("truncated ONNX protobuf varint")
        byte = data[position]
        position += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            return value, position
    raise ValueError("invalid ONNX protobuf varint")


def _field(data: bytes, position: int):
    tag, position = _varint(data, position)
    number, wire = tag >> 3, tag & 7
    start = position
    if number == 0:
        raise ValueError("invalid ONNX protobuf field number")
    if wire == 0:
        _, position = _varint(data, position)
    elif wire in (1, 5):
        position += 8 if wire == 1 else 4
    elif wire == 2:
        size, start = _varint(data, position)
        position = start + size
    elif wire == 3:
        # Unknown legacy groups remain legal protobuf fields. Skip them,
        # including nested groups, without interpreting their numeric values.
        while True:
            child, child_wire, _, position = _field(data, position)
            if child_wire == 4:
                if child != number:
                    raise ValueError("mismatched ONNX protobuf group")
                break
    elif wire != 4:
        raise ValueError(f"unsupported ONNX protobuf wire type: {wire}")
    if position > len(data):
        raise ValueError("truncated ONNX protobuf field")
    return number, wire, data[start:position], position


def _fields(data: bytes):
    """Yield field number, wire type and untouched payload bytes."""
    position = 0
    while position < len(data):
        number, wire, payload, position = _field(data, position)
        if wire == 4:
            raise ValueError("unexpected ONNX protobuf end group")
        yield number, wire, payload


def _int32(payload: bytes) -> int:
    value, _ = _varint(payload, 0)
    value &= 0xffffffff
    return value - (1 << 32) if value & (1 << 31) else value


def typed_initializer_bits(model_path: str, onnx) -> dict[tuple[str, int], int]:
    """Preserve packed/unpacked float_data and double_data scalar encodings.

    Reading the original file also avoids the pure-Python protobuf decoder's
    float promotion. A serialized Python TensorProto is already too late.
    raw_data/external_data continue through ONNX's byte-preserving loader.
    """
    model_fields = onnx.ModelProto.DESCRIPTOR.fields_by_name
    graph_fields = onnx.GraphProto.DESCRIPTOR.fields_by_name
    tensor_fields = onnx.TensorProto.DESCRIPTOR.fields_by_name
    result = {}
    for number, wire, graph in _fields(Path(model_path).read_bytes()):
        if number != model_fields["graph"].number or wire != 2:
            continue
        for number, wire, tensor in _fields(graph):
            if number != graph_fields["initializer"].number or wire != 2:
                continue
            fields = list(_fields(tensor))
            # Only correctly typed wire fields participate in protobuf decoding.
            # Singular strings/scalars use their final recognized occurrence.
            name = ""
            dtype = None
            location = onnx.TensorProto.DEFAULT
            has_raw = False
            for number, wire, payload in fields:
                if number == tensor_fields["name"].number and wire == 2:
                    name = payload.decode("utf-8")
                elif number == tensor_fields["data_type"].number and wire == 0:
                    dtype = _int32(payload)
                elif number == tensor_fields["data_location"].number and wire == 0:
                    value = _int32(payload)
                    # Unknown proto2 enum values are unknown fields, so do not
                    # overwrite the previous recognized data_location.
                    if value in (onnx.TensorProto.DEFAULT, onnx.TensorProto.EXTERNAL):
                        location = value
                elif number == tensor_fields["raw_data"].number and wire == 2:
                    has_raw = True
            # external_data is metadata; DEFAULT keeps using inline storage.
            if has_raw or location == onnx.TensorProto.EXTERNAL:
                continue
            for field, field_dtype, size, fixed_wire in (
                ("float_data", onnx.TensorProto.FLOAT, 4, 5),
                ("double_data", onnx.TensorProto.DOUBLE, 8, 1),
            ):
                if dtype != field_dtype:
                    continue
                payload = b"".join(payload for number, wire, payload in fields
                                   if number == tensor_fields[field].number
                                   and wire in (2, fixed_wire))
                if len(payload) == size:
                    result[name, dtype] = int.from_bytes(payload, "little")
    return result
