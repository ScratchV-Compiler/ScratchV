"""Independent ONNX wire-format helpers shared by Topic 28 tests.

Fixtures build protobuf bytes directly, so tests never depend on the
production wire reader and never use Python float values to represent the
original signaling/payload bits.
"""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto as TP, helper


def varint(value: int) -> bytes:
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    return bytes(result + bytes([value]))


def length_field(number: int, payload: bytes) -> bytes:
    return varint(number * 8 + 2) + varint(len(payload)) + payload


def integer_field(number: int, value: int) -> bytes:
    return varint(number * 8) + varint(value)


def typed_tensor(nt: int, raw: int, packed: bool, name: str = "a") -> bytes:
    """Serialize a float/double TensorProto with an explicit raw bit pattern."""
    width, field, wire = (4, 4, 5) if nt == TP.FLOAT else (8, 10, 1)
    payload = raw.to_bytes(width, "little")
    prefix = TP(name=name, data_type=nt).SerializeToString()
    return prefix + (
        length_field(field, payload) if packed else varint(field * 8 + wire) + payload
    )


def save_graph(graph, tmp_path: Path, name: str = "input.onnx"):
    """Check an opset-13 graph, save it, and return (path, model)."""
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    path = tmp_path / name
    onnx.save(model, path)
    return path, model
