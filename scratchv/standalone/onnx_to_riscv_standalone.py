#!/usr/bin/env python3
"""
ScratchV: Library-free ONNX → RISC-V RV32IM complete encoding pipeline.
========================================================

Converts an ONNX CNN model to self-contained RISC-V machine code.
NO external libraries required — Python 3.8+ standard library only.
The generated RISC-V binary runs bare-metal with zero dependencies.

Usage:
    python onnx_to_riscv_standalone.py models/graph/cnn.onnx -o output.bin

Pipeline:
    1. Parse ONNX protobuf (manual wire-format parser, no `onnx` package)
    2. Extract model graph, weights, shapes
    3. Convert float32 weights → Q16.16 fixed-point
    4. Memory planning (assign addresses for all tensors)
    5. Generate inline RISC-V RV32IM machine code (Conv, Gemm, MaxPool,
       ReLU, Sigmoid, Reshape — all with nested loops)
    6. Emit flat RISC-V binary (position-independent, ready to flash)

Output:
    - output.bin  : RISC-V RV32IM flat binary (code + embedded weights)
    - output.s    : Human-readable RISC-V assembly (for verification)

RISC-V binary ABI (bare-metal):
    On entry:
      a0 = pointer to input tensor (float32, NCHW layout)
      a1 = pointer to output buffer (at least 4 bytes for scalar output)
    The binary is position-independent (uses auipc for data addressing).
    Returns via jalr zero, ra, 0.

Arithmetic: Q16.16 fixed-point throughout (float × 65536 → int32).
    Multiplication uses MULH + MUL + shift for full 64-bit precision.

Author: ScratchV standalone compiler
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import struct
import sys

# ═══════════════════════════════════════════════════════════════════════════
# Part 1: Minimal Protobuf Wire-Format Parser
# ═══════════════════════════════════════════════════════════════════════════
# ONNX uses Google Protocol Buffers v2 wire format.
# Wire types: 0=varint, 1=64-bit, 2=length-delimited, 5=32-bit
# Each field: tag = (field_number << 3) | wire_type


class ProtoReader:
    """Reads protobuf wire-format from a byte buffer."""

    def __init__(self, data: bytes, pos: int = 0, end: int | None = None):
        self.data = data
        self.pos = pos
        self.end = end if end is not None else len(data)

    def _read_varint(self) -> int:
        result = 0
        shift = 0
        while self.pos < self.end:
            byte = self.data[self.pos]
            self.pos += 1
            result |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                break
            shift += 7
        return result

    def _read_fixed32(self) -> int:
        val = struct.unpack_from("<I", self.data, self.pos)[0]
        self.pos += 4
        return val

    def _read_fixed64(self) -> int:
        val = struct.unpack_from("<Q", self.data, self.pos)[0]
        self.pos += 8
        return val

    def _read_bytes(self, n: int) -> bytes:
        val = self.data[self.pos : self.pos + n]
        self.pos += n
        return val

    def parse_message(self) -> dict[int, object]:
        """Parse a protobuf message. Returns dict of field_number → value."""
        fields: dict[int, object] = {}
        while self.pos < self.end:
            tag = self._read_varint()
            field_num = tag >> 3
            wire_type = tag & 0x7

            if wire_type == 0:  # varint
                val: object = self._read_varint()
            elif wire_type == 1:  # 64-bit
                val = self._read_fixed64()
            elif wire_type == 2:  # length-delimited
                length = self._read_varint()
                val = self._read_bytes(length)
            elif wire_type == 5:  # 32-bit
                val = self._read_fixed32()
            else:
                raise ValueError(
                    f"Unknown wire type {wire_type} at pos {self.pos}"
                )

            existing = fields.get(field_num)
            if existing is None:
                fields[field_num] = val
            elif isinstance(existing, list):
                existing.append(val)  # type: ignore[union-attr]
            else:
                fields[field_num] = [existing, val]
        return fields


def _parse_packed_varints(data: bytes) -> list[int]:
    """Parse a packed repeated int64 field (wire type 2 containing varints)."""
    result: list[int] = []
    pos = 0
    while pos < len(data):
        val = 0
        shift = 0
        while pos < len(data):
            byte = data[pos]
            pos += 1
            val |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                break
            shift += 7
        result.append(val)
    return result


def _safe_decode(b: object) -> str:
    """Decode bytes to string."""
    if isinstance(b, bytes):
        return b.decode("utf-8", errors="replace")
    return str(b)


# ═══════════════════════════════════════════════════════════════════════════
# Part 2: ONNX Model Extractor
# ═══════════════════════════════════════════════════════════════════════════
# Extracts the computation graph, weights/biases, and tensor shapes
# from a raw ONNX protobuf binary.


# ONNX data type constants
ONNX_FLOAT = 1
ONNX_INT32 = 6
ONNX_INT64 = 7


class TensorInfo:
    """Describes a tensor: name, shape, data type, and optional data."""

    def __init__(self):
        self.name: str = ""
        self.shape: tuple[int, ...] = ()
        self.data_type: int = ONNX_FLOAT
        self.data: bytes = b""  # raw float32 bytes
        self.fixed_data: list[int] = []  # Q16.16 fixed-point values

    @property
    def num_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def size_bytes(self) -> int:
        return self.num_elements * 4


class NodeInfo:
    """Describes an ONNX graph node (operator)."""

    def __init__(self):
        self.op_type: str = ""
        self.name: str = ""
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.attrs: dict[str, object] = {}


class ONNXModel:
    """Extracted ONNX model: graph, weights, input/output specs."""

    def __init__(self):
        self.nodes: list[NodeInfo] = []
        self.initializers: dict[str, TensorInfo] = {}
        self.inputs: list[TensorInfo] = []
        self.outputs: list[TensorInfo] = []
        self.graph_name: str = ""
        self._value_shapes: dict[str, tuple[int, ...]] = {}

    @classmethod
    def from_file(cls, path: str) -> ONNXModel:
        """Parse an ONNX protobuf file."""
        with open(path, "rb") as f:
            data = f.read()

        model = cls()
        root = ProtoReader(data).parse_message()

        # ModelProto: graph = field 7
        if 7 not in root:
            raise ValueError("ONNX ModelProto missing graph (field 7)")

        graph_data = root[7]
        if not isinstance(graph_data, bytes):
            raise ValueError("GraphProto must be length-delimited")

        graph = ProtoReader(graph_data).parse_message()

        model.graph_name = _safe_decode(graph.get(2, ""))

        # Parse nodes (field 1)
        if 1 in graph:
            node_list = graph[1]
            if not isinstance(node_list, list):
                node_list = [node_list]
            for nd in node_list:
                if not isinstance(nd, bytes):
                    continue
                model.nodes.append(model._parse_node(nd))

        # Parse initializers (field 5) — weights and biases
        if 5 in graph:
            init_list = graph[5]
            if not isinstance(init_list, list):
                init_list = [init_list]
            for idata in init_list:
                if not isinstance(idata, bytes):
                    continue
                tensor = model._parse_tensor(idata)
                model.initializers[tensor.name] = tensor

        # Parse graph inputs (field 11)
        if 11 in graph:
            inp_list = graph[11]
            if not isinstance(inp_list, list):
                inp_list = [inp_list]
            for idata in inp_list:
                if not isinstance(idata, bytes):
                    continue
                tensor = model._parse_value_info(idata)
                if tensor.name not in model.initializers:
                    model.inputs.append(tensor)

        # Parse graph outputs (field 12)
        if 12 in graph:
            out_list = graph[12]
            if not isinstance(out_list, list):
                out_list = [out_list]
            for odata in out_list:
                if not isinstance(odata, bytes):
                    continue
                model.outputs.append(model._parse_value_info(odata))

        model._infer_shapes()
        return model

    def _parse_node(self, data: bytes) -> NodeInfo:
        """Parse a NodeProto message."""
        f = ProtoReader(data).parse_message()
        node = NodeInfo()
        node.name = _safe_decode(f.get(3, ""))

        # op_type (field 4)
        node.op_type = _safe_decode(f.get(4, ""))

        # inputs (field 1, repeated string)
        if 1 in f:
            inp = f[1]
            if isinstance(inp, list):
                node.inputs = [_safe_decode(x) for x in inp if isinstance(x, bytes)]
            elif isinstance(inp, bytes):
                node.inputs = [_safe_decode(inp)]

        # outputs (field 2, repeated string)
        if 2 in f:
            out = f[2]
            if isinstance(out, list):
                node.outputs = [_safe_decode(x) for x in out if isinstance(x, bytes)]
            elif isinstance(out, bytes):
                node.outputs = [_safe_decode(out)]

        # attributes (field 5, repeated AttributeProto)
        if 5 in f:
            attr_list = f[5]
            if not isinstance(attr_list, list):
                attr_list = [attr_list]
            for ad in attr_list:
                if not isinstance(ad, bytes):
                    continue
                self._parse_attribute(ad, node.attrs)

        return node

    @staticmethod
    def _parse_attribute(data: bytes, attrs: dict[str, object]) -> None:
        """Parse an AttributeProto into the attrs dict."""
        af = ProtoReader(data).parse_message()
        name = _safe_decode(af.get(1, ""))
        # type = field 20
        if 3 in af:
            attrs[name] = af[3]  # int (varint)
        elif 2 in af:
            fval = af[2]
            if isinstance(fval, int):
                attrs[name] = struct.unpack("<f", struct.pack("<I", fval))[0]
            else:
                attrs[name] = fval
        elif 4 in af:
            attrs[name] = _safe_decode(af[4])  # string
        elif 7 in af:
            # floats (packed)
            f7 = af[7]
            if isinstance(f7, bytes):
                n = len(f7) // 4
                attrs[name] = list(struct.unpack(f"<{n}f", f7))
            else:
                attrs[name] = f7
        elif 8 in af:
            # ints (packed varint)
            i8 = af[8]
            if isinstance(i8, bytes):
                attrs[name] = _parse_packed_varints(i8)
            else:
                attrs[name] = i8

    def _parse_tensor(self, data: bytes) -> TensorInfo:
        """Parse a TensorProto message (initializer).

        ONNX TensorProto field numbers:
          field 1: dims (repeated int64, packed)
          field 2: data_type (int32)
          field 4: float_data (repeated float, packed)
          field 5: int32_data (repeated int32, packed)
          field 7: int64_data (repeated int64, packed)
          field 8: name (string)
          field 9: raw_data (bytes)
        """
        f = ProtoReader(data).parse_message()
        tensor = TensorInfo()

        # name (field 8)
        tensor.name = _safe_decode(f.get(8, ""))

        # data_type (field 2)
        tensor.data_type = f.get(2, ONNX_FLOAT) if isinstance(f.get(2), int) else ONNX_FLOAT

        # dims (field 1, packed int64)
        if 1 in f:
            d = f[1]
            if isinstance(d, bytes):
                dims = _parse_packed_varints(d)
                if dims:
                    tensor.shape = tuple(dims)
            elif isinstance(d, list):
                tensor.shape = tuple(int(x) if isinstance(x, int) else 0 for x in d)
            elif isinstance(d, int):
                tensor.shape = (d,)

        # If dims field is empty, try to parse shape from name or data size
        if not tensor.shape:
            tensor.shape = self._parse_shape_from_name(tensor.name, tensor.data_type, f)

        # raw_data (field 9)
        if 9 in f:
            rd = f[9]
            if isinstance(rd, bytes):
                tensor.data = rd

        # float_data (field 4, packed float32)
        if not tensor.data and 4 in f:
            fd = f[4]
            if isinstance(fd, bytes):
                tensor.data = fd
            elif isinstance(fd, list):
                arr = []
                for v in fd:
                    if isinstance(v, int):
                        arr.append(struct.pack("<I", v))
                    elif isinstance(v, float):
                        arr.append(struct.pack("<f", v))
                tensor.data = b"".join(arr)

        # int32_data (field 5, packed — fallback for INT32 tensors)
        # Negative int32 values are encoded as sign-extended varints; wrap them
        # back to 32-bit two's complement before packing.
        if not tensor.data and 5 in f and tensor.data_type == ONNX_INT32:
            d5 = f[5]
            if isinstance(d5, bytes):
                vals = _parse_packed_varints(d5)
            elif isinstance(d5, list):
                vals = [int(v) if isinstance(v, int) else 0 for v in d5]
            else:
                vals = []
            out = []
            for v in vals:
                u = v & 0xFFFFFFFF
                if u >= 0x80000000:
                    u -= 0x100000000
                out.append(struct.pack("<i", u))
            tensor.data = b"".join(out)

        # int64_data (field 7, packed — fallback for INT64 tensors)
        if not tensor.data and 7 in f:
            id7 = f[7]
            if isinstance(id7, bytes):
                # Packed varints — convert each varint to 8-byte LE
                vals = _parse_packed_varints(id7)
                out = []
                for v in vals:
                    out.append(struct.pack("<q", v))
                tensor.data = b"".join(out)

        return tensor

    @staticmethod
    def _parse_shape_from_name(
        name: str, data_type: int, fields: dict[int, object]
    ) -> tuple[int, ...]:
        """Try to infer tensor shape from its name string."""
        # PPL Quantization Tool stores shapes as "[d0, d1, d2, ...]" in the name
        if name.startswith("[") and name.endswith("]"):
            try:
                parts = name[1:-1].split(",")
                return tuple(int(p.strip()) for p in parts)
            except (ValueError, IndexError):
                pass

        # Single number name = 1D tensor
        try:
            n = int(name.strip())
            # Verify by checking raw_data size
            if 9 in fields and isinstance(fields[9], bytes):
                raw_size = len(fields[9])
                if data_type == ONNX_FLOAT and raw_size == n * 4:
                    return (n,)
                if data_type == ONNX_INT32 and raw_size == n * 4:
                    return (n,)
                if data_type == ONNX_INT64 and raw_size == n * 8:
                    return (n,)
            return (n,)
        except ValueError:
            pass

        # Fallback: infer from raw_data size
        if 9 in fields and isinstance(fields[9], bytes):
            raw_size = len(fields[9])
            if data_type == ONNX_FLOAT:
                return (raw_size // 4,)
            if data_type == ONNX_INT32:
                return (raw_size // 4,)
            if data_type == ONNX_INT64:
                return (raw_size // 8,)

        return ()

    @staticmethod
    def _parse_value_info(data: bytes) -> TensorInfo:
        """Parse a ValueInfoProto (graph input/output)."""
        f = ProtoReader(data).parse_message()
        tensor = TensorInfo()
        tensor.name = _safe_decode(f.get(1, ""))

        # type (field 2) → TypeProto → tensor_type (field 1)
        if 2 in f:
            type_data = f[2]
            if isinstance(type_data, bytes):
                tf = ProtoReader(type_data).parse_message()
                if 1 in tf:
                    tensor_data = tf[1]
                    if isinstance(tensor_data, bytes):
                        ttf = ProtoReader(tensor_data).parse_message()
                        tensor.data_type = (
                            ttf.get(1, ONNX_FLOAT)
                            if isinstance(ttf.get(1), int)
                            else ONNX_FLOAT
                        )
                        # shape (field 2) → TensorShapeProto
                        if 2 in ttf:
                            shape_data = ttf[2]
                            if isinstance(shape_data, bytes):
                                sf = ProtoReader(shape_data).parse_message()
                                dims = []
                                dim_list = sf.get(1, [])
                                if not isinstance(dim_list, list):
                                    dim_list = [dim_list]
                                for dd in dim_list:
                                    if isinstance(dd, bytes):
                                        df = ProtoReader(dd).parse_message()
                                        if 1 in df:
                                            dims.append(
                                                df[1] if isinstance(df[1], int) else 1
                                            )
                                        elif 2 in df:
                                            dims.append(-1)  # symbolic
                                tensor.shape = tuple(dims)
        return tensor

    def _infer_shapes(self) -> None:
        """Propagate tensor shapes through the computation graph."""
        # Start from known shapes (initializers and inputs)
        for name, t in self.initializers.items():
            if t.shape:
                self._value_shapes[name] = t.shape

        for inp in self.inputs:
            if inp.shape:
                self._value_shapes[inp.name] = inp.shape

        # Forward propagation through nodes
        for node in self.nodes:
            self._infer_node_shape(node)

    def _infer_node_shape(self, node: NodeInfo) -> None:
        """Infer output shape for a single node."""
        # Try to get input shapes
        input_shapes = []
        for inp in node.inputs:
            if inp in self._value_shapes:
                input_shapes.append(self._value_shapes[inp])
            elif inp in self.initializers:
                input_shapes.append(self.initializers[inp].shape)
            else:
                input_shapes.append(())

        op = node.op_type

        if op == "Conv":
            self._infer_conv_shape(node, input_shapes)
        elif op == "MaxPool":
            self._infer_pool_shape(node, input_shapes)
        elif op == "Relu":
            if input_shapes and input_shapes[0]:
                self._set_output_shape(node, input_shapes[0])
        elif op == "Gemm":
            self._infer_gemm_shape(node, input_shapes)
        elif op == "Sigmoid":
            if input_shapes and input_shapes[0]:
                self._set_output_shape(node, input_shapes[0])
        elif op == "Reshape":
            self._infer_reshape_shape(node, input_shapes)
        elif op == "Fwht":
            if input_shapes and input_shapes[0]:
                self._set_output_shape(node, input_shapes[0])
        elif op == "SpmmCsr":
            self._infer_spmm_shape(node, input_shapes)
        elif op == "WinogradConv":
            self._infer_conv_shape(node, input_shapes)

    def _infer_conv_shape(
        self, node: NodeInfo, shapes: list[tuple[int, ...]]
    ) -> None:
        """Infer Conv2D output shape: NCHW format."""
        if len(shapes) < 2 or not shapes[0]:
            return
        x_shape = shapes[0]  # (N, C, H, W)
        w_shape = shapes[1] if len(shapes) > 1 else ()  # (C_out, C_in, K, K)

        out_channels = w_shape[0] if len(w_shape) >= 1 else 1

        # Prefer the actual weight kernel dims over the kernel_shape attr /
        # default 3, otherwise non-3x3 kernels (e.g. 5x5) are mis-sized.
        if len(w_shape) >= 4 and w_shape[2] > 0 and w_shape[3] > 0:
            kh, kw = int(w_shape[2]), int(w_shape[3])
        elif len(w_shape) >= 3 and w_shape[2] > 0:
            kh = kw = int(w_shape[2])
        else:
            kernel = node.attrs.get("kernel_shape", [3, 3])
            if isinstance(kernel, list) and len(kernel) >= 2:
                kh, kw = int(kernel[0]), int(kernel[1])
            else:
                kh = kw = 3

        stride = node.attrs.get("strides", [1, 1])
        if isinstance(stride, list) and len(stride) >= 2:
            sh, sw = int(stride[0]), int(stride[1])
        else:
            sh = sw = 1

        pads = node.attrs.get("pads", [0, 0, 0, 0])
        if isinstance(pads, list) and len(pads) >= 4:
            ph = int(pads[0])
            pw = int(pads[1]) if len(pads) > 1 else ph
        else:
            ph = pw = 0

        if len(x_shape) >= 4:
            h_in, w_in = x_shape[2], x_shape[3]
            h_out = (h_in + 2 * ph - kh) // sh + 1
            w_out = (w_in + 2 * pw - kw) // sw + 1
            n = x_shape[0] if len(x_shape) > 0 else 1
            self._set_output_shape(node, (n, out_channels, h_out, w_out))

    def _infer_pool_shape(
        self, node: NodeInfo, shapes: list[tuple[int, ...]]
    ) -> None:
        """Infer MaxPool output shape."""
        if not shapes or not shapes[0]:
            return
        x_shape = shapes[0]

        kernel = node.attrs.get("kernel_shape", [2, 2])
        if isinstance(kernel, list) and len(kernel) >= 2:
            kh, kw = int(kernel[0]), int(kernel[1])
        else:
            kh = kw = 2

        stride = node.attrs.get("strides", [2, 2])
        if isinstance(stride, list) and len(stride) >= 2:
            sh, sw = int(stride[0]), int(stride[1])
        else:
            sh = sw = 2

        if len(x_shape) >= 4:
            h_in, w_in = x_shape[2], x_shape[3]
            # ceil_mode affects rounding
            ceil_mode = node.attrs.get("ceil_mode", 0)
            if ceil_mode:
                h_out = (h_in - kh + sh) // sh
                w_out = (w_in - kw + sw) // sw
            else:
                h_out = (h_in - kh) // sh + 1
                w_out = (w_in - kw) // sw + 1
            self._set_output_shape(
                node, (x_shape[0], x_shape[1], h_out, w_out)
            )

    def _infer_gemm_shape(
        self, node: NodeInfo, shapes: list[tuple[int, ...]]
    ) -> None:
        """Infer Gemm output shape: Y = alpha * A * B + beta * C.

        - A shape: (M, K)
        - B shape (transB=0): (K, N) → output (M, N), N = B.shape[1]
        - B shape (transB=1): (N, K) → A × B^T = (M, K) × (K, N) = (M, N)
          N = B.shape[0]
        """
        if len(shapes) < 2:
            return
        a_shape = shapes[0]
        b_shape = shapes[1]
        trans_b = node.attrs.get("transB", 0)

        m = a_shape[0] if len(a_shape) > 0 else 1
        if isinstance(trans_b, int) and trans_b and len(b_shape) >= 2:
            # B is (N, K) → output N = B.shape[0]
            n = b_shape[0]
        elif len(b_shape) >= 2:
            # B is (K, N) → output N = B.shape[1]
            n = b_shape[1]
        elif len(b_shape) >= 1:
            n = b_shape[0]
        else:
            n = 1
        self._set_output_shape(node, (m, n))

    def _infer_reshape_shape(
        self, node: NodeInfo, shapes: list[tuple[int, ...]]
    ) -> None:
        """Infer Reshape output shape from target shape tensor.

        The second input to Reshape is a shape tensor (INT64) whose VALUES
        specify the output dimensions. We need to read those values to
        determine the output shape.
        """
        # Try to read the target shape from the second input initializer
        if len(node.inputs) >= 2 and node.inputs[1] in self.initializers:
            target = self.initializers[node.inputs[1]]
            if target.data and target.data_type == ONNX_INT64:
                n = len(target.data) // 8
                dims = list(struct.unpack(f"<{n}q", target.data))
                # Replace 0 with inferred from input, replace -1 with 1
                # (0 means "keep this dimension", but we approximate)
                input_shape = shapes[0] if len(shapes) >= 1 and shapes[0] else ()
                resolved = []
                for i, d in enumerate(dims):
                    if d == 0 and i < len(input_shape):
                        resolved.append(input_shape[i])
                    elif d == -1 or d == 0:
                        resolved.append(1)
                    else:
                        resolved.append(int(d))
                self._set_output_shape(node, tuple(resolved))
                return

        # Fallback: if second input shape is known and >0 dims, use it
        if len(shapes) >= 2 and shapes[1]:
            self._set_output_shape(node, shapes[1])

    def _infer_spmm_shape(
        self, node: NodeInfo, shapes: list[tuple[int, ...]]
    ) -> None:
        """Infer SpmmCsr output (M, N) from row_ptr length and dense B shape.

        Inputs: (values, col_indices, row_ptr, B); M = len(row_ptr) - 1,
        N = B.shape[1] (or 1 for a 1-D B / SpMV).
        """
        if len(shapes) < 4 or not shapes[3]:
            return
        row_shape = shapes[2]
        m = (row_shape[0] - 1) if row_shape else 0
        b_shape = shapes[3]
        n = b_shape[1] if len(b_shape) >= 2 else 1
        self._set_output_shape(node, (m, n))

    def _set_output_shape(
        self, node: NodeInfo, shape: tuple[int, ...]
    ) -> None:
        for out_name in node.outputs:
            self._value_shapes[out_name] = shape

    def get_shape(self, name: str) -> tuple[int, ...]:
        """Get the inferred shape for a named value."""
        if name in self._value_shapes:
            return self._value_shapes[name]
        if name in self.initializers:
            return self.initializers[name].shape
        return ()


# ═══════════════════════════════════════════════════════════════════════════
# Part 3: RISC-V RV32IM Instruction Encoder
# ═══════════════════════════════════════════════════════════════════════════
# Direct machine-code generation for RV32IM instructions.
# Each function returns a 32-bit integer (little-endian word).


# RISC-V opcodes
_RV_LOAD = 0b0000011
_RV_STORE = 0b0100011
_RV_BRANCH = 0b1100011
_RV_JALR = 0b1100111
_RV_JAL = 0b1101111
_RV_OP_IMM = 0b0010011
_RV_OP = 0b0110011
_RV_LUI = 0b0110111
_RV_AUIPC = 0b0010111

# funct3
_F3_ADD = 0b000
_F3_SLT = 0b010
_F3_XOR = 0b100
_F3_OR = 0b110
_F3_AND = 0b111
_F3_SRL = 0b101
_F3_BEQ = 0b000
_F3_BNE = 0b001
_F3_BLT = 0b100
_F3_BGE = 0b101
_F3_BLTU = 0b110
_F3_LW = 0b010
_F3_SW = 0b010
_F3_MUL = 0b000

# funct7
_F7_ADD = 0b0000000
_F7_SUB = 0b0100000
_F7_MUL = 0b0000001
_F7_MULH = 0b0000001  # MULH: funct7=1, funct3=001

# Register numbers
_R_ZERO = 0
_R_RA = 1
_R_SP = 2
_R_GP = 3
_R_FP = 8
_R_T0 = 5
_R_T1 = 6
_R_T2 = 7
_R_A0 = 10
_R_A1 = 11
_R_A2 = 12
_R_A3 = 13
_R_A4 = 14
_R_A5 = 15
_R_A6 = 16
_R_A7 = 17

# Saved registers s0-s11 (x8-x9, x18-x27)
# t3-t6 (x28-x31)
# We use s-registers for long-lived loop counters
# and t-registers for temporaries

# Register allocation plan:
#   s0  = input base pointer (preserved across layers)
#   s1  = output base pointer
#   s2  = weight base pointer
#   s3  = bias base pointer / general scratch
#   s4  = outer loop counter / stride
#   s5  = inner loop counter
#   s6  = accumulation register
#   s7  = temporary base pointer
#   s8  = loop bound
#   s9  = loop bound
#   s10 = loop bound
#   s11 = loop bound / address offset
#   t0-t4 = arithmetic temporaries
#   t5-t6 = address temporaries / short-lived

# RISC-V ABI register names
_ABI_NAMES = {
    0: "zero", 1: "ra", 2: "sp", 3: "gp", 4: "tp",
    5: "t0", 6: "t1", 7: "t2",
    8: "s0", 9: "s1",
    10: "a0", 11: "a1", 12: "a2", 13: "a3",
    14: "a4", 15: "a5", 16: "a6", 17: "a7",
    18: "s2", 19: "s3", 20: "s4", 21: "s5",
    22: "s6", 23: "s7", 24: "s8", 25: "s9",
    26: "s10", 27: "s11",
    28: "t3", 29: "t4", 30: "t5", 31: "t6",
}


def _sext(val: int, bits: int) -> int:
    """Sign-extend to given bit width."""
    mask = (1 << bits) - 1
    val &= mask
    if val >> (bits - 1):
        val -= 1 << bits
    return val


def _u32(val: int) -> int:
    """Ensure 32-bit unsigned."""
    return val & 0xFFFFFFFF


# ── Instruction encoders ──────────────────────────────────────────────────


def rv_rtype(rd: int, rs1: int, rs2: int, funct3: int, funct7: int) -> int:
    """R-type: ADD, SUB, MUL, MULH, SLT, etc."""
    return _u32(
        (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | _RV_OP
    )


def rv_itype(
    rd: int, rs1: int, imm: int, funct3: int, opcode: int = _RV_OP_IMM
) -> int:
    """I-type: ADDI, LW, JALR, SRAI, SLLI, etc."""
    return _u32(
        (_sext(imm, 12) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
    )


def rv_stype(rs1: int, rs2: int, imm: int, funct3: int) -> int:
    """S-type: SW."""
    imm = _sext(imm, 12)
    return _u32(
        ((imm >> 5) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | ((imm & 0x1F) << 7)
        | _RV_STORE
    )


def rv_btype(rs1: int, rs2: int, imm: int, funct3: int) -> int:
    """B-type: BEQ, BNE, BLT, BGE, BLTU."""
    imm = _sext(imm, 13)
    b12 = (imm >> 12) & 1
    b10_5 = (imm >> 5) & 0x3F
    b4_1 = (imm >> 1) & 0xF
    b11 = (imm >> 11) & 1
    return _u32(
        (b12 << 31)
        | (b10_5 << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | (b4_1 << 8)
        | (b11 << 7)
        | _RV_BRANCH
    )


def rv_utype(rd: int, imm: int, opcode: int = _RV_LUI) -> int:
    """U-type: LUI, AUIPC."""
    return _u32((_sext(imm, 20) << 12) | (rd << 7) | opcode)


def rv_jtype(rd: int, imm: int) -> int:
    """J-type: JAL."""
    imm = _sext(imm, 21)
    b20 = (imm >> 20) & 1
    b10_1 = (imm >> 1) & 0x3FF
    b11 = (imm >> 11) & 1
    b19_12 = (imm >> 12) & 0xFF
    return _u32(
        (b20 << 31)
        | (b19_12 << 12)
        | (b11 << 20)
        | (b10_1 << 21)
        | (rd << 7)
        | _RV_JAL
    )


# ── Convenience instruction builders ──────────────────────────────────────


def rv_add(rd: int, rs1: int, rs2: int) -> int:
    return rv_rtype(rd, rs1, rs2, _F3_ADD, _F7_ADD)


def rv_sub(rd: int, rs1: int, rs2: int) -> int:
    return rv_rtype(rd, rs1, rs2, _F3_ADD, _F7_SUB)


def rv_mul(rd: int, rs1: int, rs2: int) -> int:
    return rv_rtype(rd, rs1, rs2, _F3_MUL, _F7_MUL)


def rv_mulh(rd: int, rs1: int, rs2: int) -> int:
    """MULH: signed upper 32 bits of 64-bit product."""
    return rv_rtype(rd, rs1, rs2, 0b001, _F7_MULH)


def rv_div(rd: int, rs1: int, rs2: int) -> int:
    return rv_rtype(rd, rs1, rs2, 0b100, _F7_MUL)


def rv_slt(rd: int, rs1: int, rs2: int) -> int:
    return rv_rtype(rd, rs1, rs2, _F3_SLT, _F7_ADD)


def rv_slti(rd: int, rs1: int, imm: int) -> int:
    return rv_itype(rd, rs1, imm, _F3_SLT)


def rv_xor(rd: int, rs1: int, rs2: int) -> int:
    return rv_rtype(rd, rs1, rs2, _F3_XOR, _F7_ADD)


def rv_or(rd: int, rs1: int, rs2: int) -> int:
    return rv_rtype(rd, rs1, rs2, _F3_OR, _F7_ADD)


def rv_and(rd: int, rs1: int, rs2: int) -> int:
    return rv_rtype(rd, rs1, rs2, _F3_AND, _F7_ADD)


def rv_slli(rd: int, rs1: int, shamt: int) -> int:
    """SLLI: shift left logical immediate."""
    return rv_itype(rd, rs1, shamt & 0x1F, 0b001)


def rv_srli(rd: int, rs1: int, shamt: int) -> int:
    """SRLI: shift right logical immediate."""
    return rv_itype(rd, rs1, shamt & 0x1F, _F3_SRL)


def rv_srai(rd: int, rs1: int, shamt: int) -> int:
    """SRAI: shift right arithmetic immediate."""
    return rv_itype(rd, rs1, (shamt & 0x1F) | (1 << 10), _F3_SRL)


def rv_addi(rd: int, rs1: int, imm: int) -> int:
    return rv_itype(rd, rs1, imm, _F3_ADD)


def rv_lw(rd: int, rs1: int, offset: int) -> int:
    return rv_itype(rd, rs1, offset, _F3_LW, _RV_LOAD)


def rv_sw(rs1: int, rs2: int, offset: int) -> int:
    return rv_stype(rs1, rs2, offset, _F3_SW)


def rv_beq(rs1: int, rs2: int, offset: int) -> int:
    return rv_btype(rs1, rs2, offset, _F3_BEQ)


def rv_bne(rs1: int, rs2: int, offset: int) -> int:
    return rv_btype(rs1, rs2, offset, _F3_BNE)


def rv_blt(rs1: int, rs2: int, offset: int) -> int:
    return rv_btype(rs1, rs2, offset, _F3_BLT)


def rv_bge(rs1: int, rs2: int, offset: int) -> int:
    return rv_btype(rs1, rs2, offset, _F3_BGE)


def rv_bltu(rs1: int, rs2: int, offset: int) -> int:
    return rv_btype(rs1, rs2, offset, _F3_BLTU)


def rv_lui(rd: int, imm: int) -> int:
    return rv_utype(rd, imm, _RV_LUI)


def rv_auipc(rd: int, imm: int) -> int:
    return rv_utype(rd, imm, _RV_AUIPC)


def rv_jal(rd: int, offset: int) -> int:
    return rv_jtype(rd, offset)


def rv_jalr(rd: int, rs1: int, offset: int) -> int:
    return rv_itype(rd, rs1, offset, 0b000, _RV_JALR)


def rv_j(offset: int) -> int:
    """Unconditional jump (JAL with rd=x0)."""
    return rv_jal(_R_ZERO, offset)


def rv_ret() -> int:
    """Return: JALR zero, ra, 0."""
    return rv_jalr(_R_ZERO, _R_RA, 0)


def rv_li(rd: int, imm: int) -> tuple[int, ...]:
    """Load immediate (pseudo-instruction). Returns 1 or 2 words.

    12-bit signed immediate: ADDI x0, imm
    Otherwise: LUI + ADDI sequence
    """
    if -2048 <= imm <= 2047:
        return (rv_addi(rd, _R_ZERO, imm),)
    # lui rd, upper20 ; addi rd, rd, lower12
    # Handle sign extension: if lower12 is negative, add 1 to upper
    upper = (imm + 0x800) >> 12
    lower = imm - (upper << 12)
    return (rv_lui(rd, upper), rv_addi(rd, rd, lower))


def rv_mv(rd: int, rs: int) -> int:
    """Move: ADDI rd, rs, 0."""
    return rv_addi(rd, rs, 0)


def rv_nop() -> int:
    return rv_addi(_R_ZERO, _R_ZERO, 0)


def rv_qmul(rd: int, rs1: int, rs2: int, tmp1: int, tmp2: int) -> list[int]:
    """Q16.16 fixed-point multiply: rd = (rs1 * rs2) >> 16 (full precision).

    Uses MULH + MUL for 64-bit intermediate:
      mulh tmp1, rs1, rs2   # upper 32 bits
      mul  tmp2, rs1, rs2   # lower 32 bits
      slli tmp1, tmp1, 16   # high << 16
      srli tmp2, tmp2, 16   # low >> 16
      or   rd, tmp1, tmp2   # combine
    """
    return [
        rv_mulh(tmp1, rs1, rs2),
        rv_mul(tmp2, rs1, rs2),
        rv_slli(tmp1, tmp1, 16),
        rv_srli(tmp2, tmp2, 16),
        rv_or(rd, tmp1, tmp2),
    ]


def rv_qmul_simple(rd: int, rs1: int, rs2: int) -> int:
    """Q16.16 multiply (simple, may overflow for large values).

    Uses MUL + SRAI:
      mul rd, rs1, rs2
      srai rd, rd, 16
    """
    return rv_mul(rd, rs1, rs2)


def rv_qmul_postshift(rd: int, rs: int) -> list[int]:
    """Post-shift after rv_qmul_simple: srai rd, rs, 16."""
    return [rv_srai(rd, rs, 16)]


# ═══════════════════════════════════════════════════════════════════════════
# Part 4: Fixed-Point Conversion
# ═══════════════════════════════════════════════════════════════════════════

Q16 = 16
Q_SCALE = 1 << Q16  # 65536


def float32_to_q16(raw_bytes: bytes) -> list[int]:
    """Convert IEEE 754 float32 bytes to Q16.16 fixed-point list."""
    n = len(raw_bytes) // 4
    floats = struct.unpack(f"<{n}f", raw_bytes)
    result = []
    for f in floats:
        # Clamp to avoid overflow
        if math.isnan(f) or math.isinf(f):
            result.append(0)
        else:
            scaled = int(f * Q_SCALE)
            if scaled > 0x7FFFFFFF:
                scaled = 0x7FFFFFFF
            elif scaled < -0x80000000:
                scaled = -0x80000000
            result.append(scaled & 0xFFFFFFFF)
    return result


def int64_to_shape(raw_bytes: bytes) -> tuple[int, ...]:
    """Convert INT64 raw bytes to shape tuple."""
    n = len(raw_bytes) // 8
    vals = struct.unpack(f"<{n}q", raw_bytes)
    return tuple(int(v) for v in vals)


# ═══════════════════════════════════════════════════════════════════════════
# Winograd F(2,3) transforms (compile-time helpers, pure Python)
# ═══════════════════════════════════════════════════════════════════════════
# 2D minimal filtering: Y = A^T [ (G g G^T) ⊙ (B^T d B) ] A
# F(2,3): input tile 4x4, kernel 3x3, output 2x2.
# G contains 1/2 entries; B^T and A^T are integer {0, ±1}.

_WINO_F23_BT = ((1, 0, -1, 0), (0, 1, 1, 0), (0, -1, 1, 0), (0, 1, 0, -1))
_WINO_F23_G = ((1.0, 0.0, 0.0), (0.5, 0.5, 0.5),
               (0.5, -0.5, 0.5), (0.0, 0.0, 1.0))
_WINO_F23_AT = ((1.0, 1.0, 1.0, 0.0), (0.0, 1.0, -1.0, -1.0))


def winograd_f23_kernel_transform(g):
    """Return U = G g G^T (4x4) for a 3x3 kernel g (nested sequences)."""
    P = [[sum(_WINO_F23_G[i][k] * g[k][j] for k in range(3)) for j in range(3)]
         for i in range(4)]
    return [[sum(P[i][k] * _WINO_F23_G[j][k] for k in range(3)) for j in range(4)]
            for i in range(4)]


def _prepare_winograd_kernels(model) -> None:
    """Fold WinogradConv 3x3 kernels into the Winograd domain at compile time.

    For each WinogradConv node with FLOAT weight [Cout, Cin, 3, 3], compute
    U = [Cout, Cin, 4, 4] and register it as a new FLOAT initializer. The node
    stores the tensor name in ``attrs["_winograd_u"]`` for the code generator.
    """
    for node in model.nodes:
        if node.op_type != "WinogradConv" or len(node.inputs) < 2:
            continue
        w = model.initializers.get(node.inputs[1])
        if w is None or not w.data or len(w.shape) != 4:
            continue
        cout, cin, kh, kw = w.shape
        if (kh, kw) != (3, 3):
            raise ValueError("WinogradConv currently supports 3x3 kernels only")

        floats = struct.unpack(f"<{cout * cin * 9}f", w.data)
        u_bytes = bytearray()
        for oc in range(cout):
            for ic in range(cin):
                base = (oc * cin + ic) * 9
                g = [[floats[base + r * 3 + c] for c in range(3)] for r in range(3)]
                u = winograd_f23_kernel_transform(g)
                for r in range(4):
                    for c in range(4):
                        u_bytes += struct.pack("<f", u[r][c])

        tensor = TensorInfo()
        tensor.name = f"{w.name}__wino23"
        tensor.shape = (cout, cin, 4, 4)
        tensor.data_type = ONNX_FLOAT
        tensor.data = bytes(u_bytes)
        model.initializers[tensor.name] = tensor
        node.attrs["_winograd_u"] = tensor.name


# ═══════════════════════════════════════════════════════════════════════════
# Part 5: Memory Planner
# ═══════════════════════════════════════════════════════════════════════════


class MemoryPlan:
    """Assigns memory addresses for all tensors in the computation graph.

    Strategy:
      - Weight data: embedded after code (known at compile time)
      - Input buffer: caller-provided (a0 on entry)
      - Output buffer: caller-provided (a1 on entry) for final output
      - Intermediate tensors: workspace area, reuse where possible
      - Stack: grows down from top of workspace
    """

    def __init__(self):
        # Map from tensor name → byte offset in data section
        self.weight_offsets: dict[str, int] = {}
        # Map from tensor name → byte offset in workspace
        self.workspace_offsets: dict[str, int] = {}
        # Total data section size (bytes)
        self.data_size: int = 0
        # Total workspace size (bytes)
        self.workspace_size: int = 0

    def layout_weights(self, initializers: dict[str, TensorInfo]) -> bytes:
        """Layout all weight tensors sequentially in the data section.

        Returns the concatenated Q16.16 weight data as bytes.
        """
        offset = 0
        data_parts: list[bytes] = []

        for name, tensor in initializers.items():
            if not tensor.data:
                continue

            # Convert to Q16.16
            if tensor.data_type == ONNX_FLOAT:
                q16_vals = float32_to_q16(tensor.data)
            elif tensor.data_type == ONNX_INT32:
                # INT32 tensors (e.g. CSR col_indices / row_ptr) are stored
                # verbatim as one 32-bit word per element — do NOT run them
                # through the Q16.16 conversion.
                n = len(tensor.data) // 4
                int_vals = struct.unpack(f"<{n}i", tensor.data)
                q16_vals = [v & 0xFFFFFFFF for v in int_vals]
            elif tensor.data_type == ONNX_INT64:
                # INT64 tensors (like Reshape target shapes) — keep as-is
                q16_vals = []
                n = len(tensor.data) // 8
                int_vals = struct.unpack(f"<{n}q", tensor.data)
                for v in int_vals:
                    q16_vals.append(v & 0xFFFFFFFF)
                    q16_vals.append((v >> 32) & 0xFFFFFFFF)
            else:
                continue

            tensor.fixed_data = q16_vals

            # Align to 4 bytes
            if offset % 4 != 0:
                pad = 4 - (offset % 4)
                data_parts.append(b"\x00" * pad)
                offset += pad

            self.weight_offsets[name] = offset

            # Pack Q16.16 values as little-endian int32
            packed = struct.pack(f"<{len(q16_vals)}I", *q16_vals)
            data_parts.append(packed)
            offset += len(packed)

        self.data_size = offset
        return b"".join(data_parts)

    def alloc_workspace(self, name: str, num_elements: int) -> int:
        """Allocate workspace for a tensor. Simple bump allocator.

        Returns byte offset within workspace.
        """
        # Align to 4 bytes
        if self.workspace_size % 4 != 0:
            self.workspace_size += 4 - (self.workspace_size % 4)

        offset = self.workspace_size
        self.workspace_offsets[name] = offset
        self.workspace_size += num_elements * 4
        return offset

    def get_weight_offset(self, name: str) -> int:
        """Get byte offset of a weight tensor in the data section."""
        return self.weight_offsets.get(name, 0)

    def get_workspace_offset(self, name: str) -> int:
        """Get byte offset of a tensor in the workspace."""
        return self.workspace_offsets.get(name, 0)


# ═══════════════════════════════════════════════════════════════════════════
# Part 6: RISC-V Code Generator
# ═══════════════════════════════════════════════════════════════════════════
# Generates inline RISC-V RV32IM machine code for each CNN operator.
# All operators are implemented with nested loops — no runtime library calls.


class RISCVEmitter:
    """Emits RISC-V instructions into a code buffer with label tracking."""

    def __init__(self, compact_li32: bool = False):
        self.code: list[int] = []  # list of 32-bit instruction words
        self.labels: dict[str, int] = {}  # label name → instruction index
        self.label_history: list[str] = []
        self.pending_fixups: list[tuple[int, str, str]] = []  # (idx, kind, label)
        self.comments: dict[int, str] = {}  # instruction index → comment
        self.protected_indices: set[int] = set()
        self.compact_li32 = compact_li32
        self.label_prefix = ""

    def _emit(self, word: int, comment: str = "") -> int:
        """Emit one instruction word. Returns its index."""
        idx = len(self.code)
        self.code.append(word)
        if comment:
            self.comments[idx] = comment
        return idx

    def emit(self, word: int, comment: str = "") -> int:
        return self._emit(word, comment)

    def emit_many(self, words: list[int], comment: str = "") -> None:
        for i, w in enumerate(words):
            c = comment if i == 0 else ""
            self._emit(w, c)

    def label(self, name: str) -> None:
        """Define a label at the current position."""
        name = self.label_prefix + name
        if name in self.labels:
            raise ValueError(f"Duplicate label '{name}'")
        self.label_history.append(name)
        self.labels[name] = len(self.code)

    def finalize(self, strategy=None) -> None:
        """Finalize code through an optional symbolic encoding strategy."""
        if strategy is None:
            self.resolve_fixups()
        else:
            strategy(self)

    def emit_branch(self, op_builder, rs1: int, rs2: int,
                    label: str, comment: str = "") -> int:
        """Emit a branch instruction with label fixup."""
        idx = self._emit(op_builder(rs1, rs2, 0), comment)
        self.pending_fixups.append((idx, "b", self.label_prefix + label))
        return idx

    def emit_jump(self, op_builder, label: str, comment: str = "") -> int:
        """Emit a jump instruction with label fixup."""
        idx = self._emit(op_builder(0), comment)
        self.pending_fixups.append((idx, "j", self.label_prefix + label))
        return idx

    def emit_jal(self, rd: int, label: str, comment: str = "") -> int:
        idx = self._emit(rv_jal(rd, 0), comment)
        self.pending_fixups.append((idx, "j", self.label_prefix + label))
        return idx

    def emit_li(self, rd: int, imm: int, comment: str = "") -> None:
        """Emit load-immediate (may be 1 or 2 instructions)."""
        words = rv_li(rd, imm)
        self.emit_many(list(words), comment)

    def emit_li32(self, rd: int, imm: int, comment: str = "") -> None:
        """Emit a 32-bit immediate using LUI + ADDI sequence.

        The baseline form always emits two instructions.  With constant-load
        compaction enabled, it uses the canonical one-or-two instruction
        lowering used for a merged source-level ``li`` pseudo-instruction.
        """
        if self.compact_li32:
            self.emit_li(rd, imm, comment)
            return

        # Split into upper 20 bits and lower 12 bits
        # ADDI sign-extends the 12-bit immediate, so we need to handle carry
        imm_u32 = imm & 0xFFFFFFFF
        upper = (imm_u32 + 0x800) >> 12
        lower = _sext(imm_u32 & 0xFFF, 12)
        # Adjust upper if lower is negative (ADDI will subtract)
        if lower < 0:
            # The ADDI will sign-extend and subtract, so we bump upper
            upper_adjusted = upper
            self._emit(rv_lui(rd, upper_adjusted & 0xFFFFF), comment)
            self._emit(rv_addi(rd, rd, lower & 0xFFF), "")
        else:
            self._emit(rv_lui(rd, upper & 0xFFFFF), comment)
            self._emit(rv_addi(rd, rd, lower & 0xFFF), "")

    def emit_qmul(self, rd: int, rs1: int, rs2: int,
                  tmp1: int = _R_T0, tmp2: int = _R_T1,
                  comment: str = "") -> None:
        """Emit full-precision Q16.16 multiply sequence (5 instructions)."""
        self.emit_many(rv_qmul(rd, rs1, rs2, tmp1, tmp2), comment)

    def emit_qmul_srai(self, rd: int, rs1: int, rs2: int,
                       tmp1: int = _R_T0,
                       comment: str = "") -> None:
        """Emit Q16.16 multiply with MUL + post SRAI (2 instructions).

        WARNING: may overflow if product exceeds 32 bits.
        Safe when both operands are in range [-2^15, 2^15].
        """
        self._emit(rv_qmul_simple(rd, rs1, rs2), comment)
        self._emit(rv_srai(rd, rd, 16), "")

    def resolve_fixups(self) -> None:
        """Resolve all branch/jump fixups."""
        for idx, kind, label in self.pending_fixups:
            if label not in self.labels:
                raise ValueError(f"Undefined label '{label}' at instruction {idx}")
            target_idx = self.labels[label]
            byte_offset = (target_idx - idx) * 4
            limit = 4096 if kind == "b" else 1048576
            if not -limit <= byte_offset < limit:
                raise ValueError(f"Out-of-range {kind} target '{label}': {byte_offset} bytes")
            word = self.code[idx]

            if kind == "b":
                # Reconstruct branch with correct offset
                rs1 = (word >> 15) & 0x1F
                rs2 = (word >> 20) & 0x1F
                funct3 = (word >> 12) & 0x7
                self.code[idx] = rv_btype(rs1, rs2, byte_offset, funct3)
            elif kind == "j":
                if (word & 0x7F) == _RV_JAL:
                    rd = (word >> 7) & 0x1F
                    self.code[idx] = rv_jal(rd, byte_offset)
                else:
                    self.code[idx] = rv_j(byte_offset)
        self.pending_fixups.clear()

    def disassemble(self, *, symbolic: bool = False) -> str:
        """Render numeric debug output or a fixed-width symbolic assembly listing.

        Symbolic targets are decoded from the resolved words, including targets
        with no original source label. Labels use instruction addresses so names
        containing ONNX operator paths cannot become invalid assembler symbols.
        """
        targets = set(self.labels.values()) if symbolic else set()
        decoded = [_disasm_one(word) for word in self.code]
        if symbolic:
            for i, word in enumerate(self.code):
                if word & 0x7F in {_RV_BRANCH, _RV_JAL}:
                    prefix, literal = decoded[i].rsplit(" ", 1)
                    offset = int(literal)
                    target = i + offset // 4
                    if offset % 4 or not 0 <= target <= len(self.code):
                        raise ValueError(f"Invalid fixed-width branch target at instruction {i}")
                    targets.add(target)
                    decoded[i] = f"{prefix} .Lsv_{target}"
        lines = []
        for i, word in enumerate(self.code):
            if symbolic:
                if i in targets:
                    lines.append(f".Lsv_{i}:")
            else:
                for name, idx in self.labels.items():
                    if idx == i:
                        lines.append(f"{name}:")

            asm = decoded[i]
            comment = self.comments.get(i, "")
            if comment:
                asm = f"{asm:<32s} # {comment}"
            lines.append(f"  {asm}")
        if symbolic and len(self.code) in targets:
            lines.append(f".Lsv_{len(self.code)}:")
        return "\n".join(lines)

    def schedule(self, llvm_mca: str | None = None):
        """Apply verified local permutations to the exact emitted machine words."""
        from collections import defaultdict, deque
        from scratchv.backend.inst_scheduler import ScheduleConfig, parse_instructions, schedule_assembly

        source = self.disassemble(symbolic=True)
        original = parse_instructions(source)
        if len(original) != len(self.code):
            raise ValueError("Scheduling requires a decoded instruction for every machine word")
        result = schedule_assembly(source, ScheduleConfig(strict=True, llvm_mca=llvm_mca))
        words = defaultdict(deque)
        for index, inst in enumerate(original):
            words[inst.raw_line].append((self.code[index], self.comments.get(index, "")))
        reordered = [words[inst.raw_line].popleft() for inst in parse_instructions(result.asm_text)]
        if len(reordered) != len(self.code) or any(words.values()):
            raise ValueError("Scheduled machine-word permutation is incomplete")
        self.code = [word for word, _ in reordered]
        self.comments = {i: comment for i, (_, comment) in enumerate(reordered) if comment}
        return result

    def to_bytes(self) -> bytes:
        """Encode all instructions as little-endian binary."""
        return struct.pack(f"<{len(self.code)}I", *self.code)


def _disasm_one(word: int) -> str:
    """Minimal RISC-V disassembler for debugging."""
    opcode = word & 0x7F
    rd = (word >> 7) & 0x1F
    funct3 = (word >> 12) & 0x7
    rs1 = (word >> 15) & 0x1F
    rs2 = (word >> 20) & 0x1F
    funct7 = (word >> 25) & 0x7F
    rdn = _ABI_NAMES.get(rd, f"x{rd}")
    rs1n = _ABI_NAMES.get(rs1, f"x{rs1}")
    rs2n = _ABI_NAMES.get(rs2, f"x{rs2}")

    if opcode == _RV_OP_IMM:
        imm = _sext((word >> 20) & 0xFFF, 12)
        if funct3 == _F3_ADD:
            if rd == _R_ZERO and rs1 == _R_ZERO and imm == 0:
                return "nop"
            if rs1 == _R_ZERO:
                return f"li {rdn}, {imm}"
            elif imm == 0:
                return f"mv {rdn}, {rs1n}"
            return f"addi {rdn}, {rs1n}, {imm}"
        elif funct3 == _F3_SLT:
            return f"slti {rdn}, {rs1n}, {imm}"
        elif funct3 == _F3_XOR:
            return f"xori {rdn}, {rs1n}, {imm}"
        elif funct3 == _F3_OR:
            return f"ori {rdn}, {rs1n}, {imm}"
        elif funct3 == _F3_AND:
            return f"andi {rdn}, {rs1n}, {imm}"
        elif funct3 == _F3_SRL:
            if funct7 == 0b0100000:
                shamt = imm & 0x1F
                return f"srai {rdn}, {rs1n}, {shamt}"
            shamt = imm & 0x1F
            return f"srli {rdn}, {rs1n}, {shamt}"
        elif funct3 == 0b001:
            shamt = imm & 0x1F
            return f"slli {rdn}, {rs1n}, {shamt}"
        return f"op_imm_{funct3:03b} {rdn}, {rs1n}, {imm}"

    elif opcode == _RV_OP:
        if funct7 == _F7_ADD:
            if funct3 == _F3_ADD:
                return f"add {rdn}, {rs1n}, {rs2n}"
            elif funct3 == _F3_SLT:
                return f"slt {rdn}, {rs1n}, {rs2n}"
            elif funct3 == _F3_XOR:
                return f"xor {rdn}, {rs1n}, {rs2n}"
            elif funct3 == _F3_OR:
                return f"or {rdn}, {rs1n}, {rs2n}"
            elif funct3 == _F3_AND:
                return f"and {rdn}, {rs1n}, {rs2n}"
        elif funct7 == _F7_SUB:
            return f"sub {rdn}, {rs1n}, {rs2n}"
        elif funct7 == _F7_MUL:
            if funct3 == _F3_MUL:
                return f"mul {rdn}, {rs1n}, {rs2n}"
            elif funct3 == 0b001:
                return f"mulh {rdn}, {rs1n}, {rs2n}"
            elif funct3 == 0b100:
                return f"div {rdn}, {rs1n}, {rs2n}"
        return f"op_{funct7:07b}_{funct3:03b} {rdn}, {rs1n}, {rs2n}"

    elif opcode == _RV_LOAD:
        imm = _sext((word >> 20) & 0xFFF, 12)
        return f"lw {rdn}, {imm}({rs1n})"

    elif opcode == _RV_STORE:
        imm = ((word >> 25) << 5) | ((word >> 7) & 0x1F)
        imm = _sext(imm, 12)
        return f"sw {rs2n}, {imm}({rs1n})"

    elif opcode == _RV_BRANCH:
        b4_1 = (word >> 8) & 0xF
        b10_5 = (word >> 25) & 0x3F
        b11 = (word >> 7) & 1
        b12 = (word >> 31) & 1
        imm = (b12 << 12) | (b11 << 11) | (b10_5 << 5) | (b4_1 << 1)
        imm = _sext(imm, 13)
        mnemonic = {_F3_BEQ: "beq", _F3_BNE: "bne", _F3_BLT: "blt",
                     _F3_BGE: "bge", _F3_BLTU: "bltu"}.get(funct3, f"b{funct3:03b}")
        return f"{mnemonic} {rs1n}, {rs2n}, {imm:+d}"

    elif opcode == _RV_JALR:
        imm = _sext((word >> 20) & 0xFFF, 12)
        if rd == _R_ZERO and rs1 == _R_RA and imm == 0:
            return "ret"
        elif rd == _R_ZERO:
            return f"jr {rs1n}"
        return f"jalr {rdn}, {rs1n}, {imm}"

    elif opcode == _RV_JAL:
        b20 = (word >> 31) & 1
        b10_1 = (word >> 21) & 0x3FF
        b11 = (word >> 20) & 1
        b19_12 = (word >> 12) & 0xFF
        imm = (b20 << 20) | (b19_12 << 12) | (b11 << 11) | (b10_1 << 1)
        imm = _sext(imm, 21)
        if rd == _R_ZERO:
            return f"j {imm:+d}"
        return f"jal {rdn}, {imm:+d}"

    elif opcode == _RV_LUI:
        imm = _sext(word >> 12, 20)
        return f"lui {rdn}, 0x{imm & 0xFFFFF:x}"

    elif opcode == _RV_AUIPC:
        imm = _sext(word >> 12, 20)
        return f"auipc {rdn}, 0x{imm & 0xFFFFF:x}"

    return f".word 0x{word:08x}"


# ═══════════════════════════════════════════════════════════════════════════
# Part 7: CNN Layer Code Generators
# ═══════════════════════════════════════════════════════════════════════════


class CNNRISCVGenerator:
    """Generates complete RISC-V code for all CNN operators."""

    def __init__(
        self,
        model: ONNXModel,
        memory: MemoryPlan,
        compact_constants: bool = False,
        finalize_strategy=None,
    ):
        self.model = model
        self.mem = memory
        self.emit = RISCVEmitter(compact_li32=compact_constants)
        self.finalize_strategy = finalize_strategy

        # Wide register aliases for readability
        # t0-t6: x5-x7, x28-x31 → temporaries
        # s0-s11: x8-x9, x18-x27 → saved / loop counters
        self.T0, self.T1, self.T2 = 5, 6, 7
        self.T3, self.T4, self.T5, self.T6 = 28, 29, 30, 31
        self.S0, self.S1 = 8, 9
        self.S2, self.S3, self.S4, self.S5 = 18, 19, 20, 21
        self.S6, self.S7, self.S8, self.S9 = 22, 23, 24, 25
        self.S10, self.S11 = 26, 27

    def generate(self) -> bytes:
        """Generate the complete RISC-V binary for the CNN model."""
        # ── Entry point ────────────────────────────────────────────────
        self.emit.label("_start")

        # Save return address (bare-metal may not need this, but safe)
        # We use the stack for saving/restoring
        # Stack grows down from STACK_TOP (defined by linker/memory layout)
        # Initialize sp from the high address passed by convention

        # Save input/output pointers
        self.emit.emit(rv_mv(self.S0, _R_A0), "s0 = input_ptr")
        self.emit.emit(rv_mv(self.S1, _R_A1), "s1 = output_ptr")

        # Initialize GP (global pointer) with the data section base.
        # We use AUIPC to get PC-relative address.
        # During code generation, we don't know the exact offset yet,
        # so we'll use a placeholder that gets resolved when linking code+data.
        # For now, emit a NOP placeholder — the caller handles AUIPC setup.
        init_label_idx = len(self.emit.code)
        self.emit.label("_init_data_base")
        # The data base address loading is handled at binary assembly time.
        # We embed a dummy AUIPC + ADDI that gets patched:
        #   auipc gp, 0        → placeholder
        #   addi gp, gp, 0     → placeholder
        self.emit.emit(rv_auipc(_R_GP, 0), "gp = data base (patched at link time)")
        self.emit.emit(rv_addi(_R_GP, _R_GP, 0), "data offset (patched)")
        self.emit.protected_indices.update({init_label_idx, init_label_idx + 1})

        # ── Copy input from caller buffer to workspace ─────────────────
        # The input data is at s0 (caller-provided, float32/Q16.16 format).
        # We copy it to the workspace so subsequent layers can read uniformly.
        if self.model.inputs:
            input_name = self.model.inputs[0].name
            input_shape = self.model.get_shape(input_name)
            input_el = 1
            for d in input_shape:
                input_el *= d
            ws_offset = self.mem.get_workspace_offset(input_name)
            if input_el > 0:
                # Loop to copy input_el words from s0 to sp+ws_offset
                self.emit.label("_copy_input")
                self.emit.emit(rv_addi(self.S7, _R_ZERO, 0),
                               f"i=0 ({input_el} elements)")
                copy_loop_l = "_input_copy_loop"
                copy_done_l = "_input_copy_done"
                self.emit.label(copy_loop_l)
                # Load from source (s0 + i*4)
                self.emit.emit(rv_slli(self.T3, self.S7, 2), "offset = i*4")
                self.emit.emit(rv_add(self.T3, self.S0, self.T3), "src addr")
                self.emit.emit(rv_lw(self.T1, self.T3, 0), "load src[i]")
                # Store to workspace (sp + ws_offset + i*4)
                self.emit.emit_li32(self.T6, ws_offset, f"ws_offset={ws_offset}")
                self.emit.emit(rv_add(self.T3, _R_SP, self.T6), "ws base")
                self.emit.emit(rv_slli(self.T5, self.S7, 2), "")
                self.emit.emit(rv_add(self.T3, self.T3, self.T5), "dst addr")
                self.emit.emit(rv_sw(self.T3, self.T1, 0), "store to workspace")
                # i++
                self.emit.emit(rv_addi(self.S7, self.S7, 1), "i++")
                self.emit.emit_li32(self.T6, input_el, f"n={input_el}")
                self.emit.emit(rv_slt(self.T4, self.S7, self.T6), "i < n?")
                self.emit.emit_branch(rv_bne, self.T4, _R_ZERO, copy_loop_l, "loop")
                self.emit.label(copy_done_l)

        # Process each node in topological order
        for node in self.model.nodes:
            self._generate_node(node)

        # ── Copy output from workspace to caller buffer ─────────────────
        # After the last layer, copy the final output to a1 (caller buffer)
        if self.model.outputs:
            last_output = self.model.outputs[0].name
            out_shape = self.model.get_shape(last_output)
            out_el = 1
            for d in out_shape:
                out_el *= d
            ws_offset = self.mem.get_workspace_offset(last_output)
            if out_el > 0 and ws_offset >= 0:
                self.emit.label("_copy_output")
                self.emit.emit(rv_addi(self.S7, _R_ZERO, 0),
                               f"copy output ({out_el} elements)")
                out_copy_loop = "_out_copy_loop"
                out_copy_done = "_out_copy_done"
                self.emit.label(out_copy_loop)
                # Load from workspace (sp + ws_offset + i*4)
                self.emit.emit_li32(self.T6, ws_offset, f"ws_offset={ws_offset}")
                self.emit.emit(rv_add(self.T3, _R_SP, self.T6), "ws base")
                self.emit.emit(rv_slli(self.T5, self.S7, 2), "")
                self.emit.emit(rv_add(self.T3, self.T3, self.T5), "src addr")
                self.emit.emit(rv_lw(self.T1, self.T3, 0), "load from ws")
                # Store to output buffer (s1 + i*4)
                self.emit.emit(rv_slli(self.T3, self.S7, 2), "offset = i*4")
                self.emit.emit(rv_add(self.T3, self.S1, self.T3), "dst addr")
                self.emit.emit(rv_sw(self.T3, self.T1, 0), "store to output")
                # i++
                self.emit.emit(rv_addi(self.S7, self.S7, 1), "i++")
                self.emit.emit_li32(self.T6, out_el, f"n={out_el}")
                self.emit.emit(rv_slt(self.T4, self.S7, self.T6), "i < n?")
                self.emit.emit_branch(rv_bne, self.T4, _R_ZERO, out_copy_loop, "loop")
                self.emit.label(out_copy_done)

        # ── Exit ───────────────────────────────────────────────────────
        self.emit.label("_done")
        self.emit.emit(rv_ret(), "return")
        self.emit.emit(rv_nop(), "")

        # Resolve internal branch/jump fixups, or run an injected symbolic
        # finalizer before relocation when a benchmark strategy is supplied.
        self.emit.finalize(self.finalize_strategy)

        return self.emit.to_bytes()

    def _generate_node(self, node: NodeInfo) -> None:
        """Dispatch to the appropriate code generator for this node type."""
        op = node.op_type.lower()
        self.emit.label(f"_op_{node.name or op}_{len(self.emit.labels)}")
        self.emit.emit(rv_nop(), f"--- {node.op_type}: "
                       f"{', '.join(node.inputs[:2])} -> {', '.join(node.outputs[:1])}")

        handler = getattr(self, f"_gen_{op}", None)
        if handler is None:
            raise ValueError(f"Unsupported op: {node.op_type}")
        # Operator-local loop names must never bind to a later operator.
        previous_prefix = self.emit.label_prefix
        self.emit.label_prefix = f"_node_{len(self.emit.code)}_"
        try:
            handler(node)
        finally:
            self.emit.label_prefix = previous_prefix

    def _get_weight_addr(self, name: str, dst_reg: int) -> None:
        """Load address of a weight tensor into dst_reg.

        Base address (gp) + byte offset.
        Uses LUI+ADDI for the offset if it exceeds 12 bits.
        """
        offset = self.mem.get_weight_offset(name)
        if -2048 <= offset <= 2047:
            self.emit.emit(rv_addi(dst_reg, _R_GP, offset),
                           f"addr of {name} (gp+{offset})")
        else:
            # Need to compute full offset
            self.emit.emit_li32(self.T6, offset, f"offset for {name}")
            self.emit.emit(rv_add(dst_reg, _R_GP, self.T6),
                           f"addr of {name}")

    def _get_workspace_addr(self, name: str, dst_reg: int) -> None:
        """Load workspace address for a tensor into dst_reg.

        For intermediate tensors: we use sp-relative addressing.
        The workspace is below the stack, so addresses are sp + offset.
        """
        offset = self.mem.get_workspace_offset(name)
        if -2048 <= offset <= 2047:
            self.emit.emit(rv_addi(dst_reg, _R_SP, offset),
                           f"ws addr of {name}")
        else:
            self.emit.emit_li32(self.T6, offset, f"ws offset for {name}")
            self.emit.emit(rv_add(dst_reg, _R_SP, self.T6),
                           f"ws addr of {name}")

    # ── Conv2D ─────────────────────────────────────────────────────────

    def _gen_conv(self, node: NodeInfo) -> None:
        """Generate optimized inline Conv2D: NCHW layout, nested loops.

        Optimizations:
          - Pointer-walking in innermost kw loop (11 instr/MAC vs ~42 original)
          - Boundary checks skipped when pads=0 (saves ~10 instr/MAC)
          - All constants preloaded (saves ~10 emit_li per MAC)
          - Strength reduction: stride*h/w computed once per oh/ow; used as base
          - Pointer adjustment per kh row: input_ptr += (W-K)*4

        for oc in range(C_out):
          for oh in range(H_out):
            for ow in range(W_out):
              acc = bias[oc]
              for ic in range(C_in):
                for kh in range(K):
                  for kw in range(K):
                    acc += input[ic, ih, iw] * weight[oc, ic, kh, kw]
              output[oc, oh, ow] = acc
        """
        x_name = node.inputs[0]
        w_name = node.inputs[1]
        # Conv bias is optional; a missing third input means "no bias" (zero).
        b_name = node.inputs[2] if len(node.inputs) > 2 and node.inputs[2] else None
        out_name = node.outputs[0]

        x_shape = self.model.get_shape(x_name)
        w_shape = self.model.get_shape(w_name)

        N = x_shape[0] if len(x_shape) > 0 else 1
        C_in = x_shape[1] if len(x_shape) > 1 else 1
        H = x_shape[2] if len(x_shape) > 2 else 1
        W = x_shape[3] if len(x_shape) > 3 else 1

        C_out = w_shape[0] if len(w_shape) > 0 else 1
        K = w_shape[2] if len(w_shape) > 2 else 3

        attrs = node.attrs
        stride = attrs.get("strides", [1, 1])
        sh = int(stride[0]) if isinstance(stride, list) else 1
        sw = int(stride[1]) if isinstance(stride, list) and len(stride) > 1 else sh

        pads = attrs.get("pads", [0, 0, 0, 0])
        ph = int(pads[0]) if isinstance(pads, list) else 0
        pw = int(pads[1]) if isinstance(pads, list) and len(pads) > 1 else ph

        H_out = (H + 2 * ph - K) // sh + 1
        W_out = (W + 2 * pw - K) // sw + 1

        no_pad = (ph == 0 and pw == 0)

        # Allocate workspace
        out_elements = N * C_out * H_out * W_out
        self.mem.alloc_workspace(out_name, out_elements)

        # Load base addresses
        self._get_workspace_addr(x_name, self.S2)   # s2 = input base
        self._get_weight_addr(w_name, self.S3)       # s3 = weight base
        if b_name is not None:
            self._get_weight_addr(b_name, self.S4)   # s4 = bias base
        self._get_workspace_addr(out_name, self.S5)  # s5 = output base

        # ── Preload loop-invariant constants ──────────────────────────
        # We use a2-a7 (x12-x17) for preloaded constants — free in bare-metal
        # after a0/a1 are saved to s0/s1 at entry.
        K_REG = 12          # a2: kernel size K
        STRIDE_H_REG = 13   # a3: stride_h (sh)
        STRIDE_W_REG = 14   # a4: stride_w (sw)
        W_REG = 15          # a5: input width W
        HW_REG = 16         # a6: H * W (channel stride)
        C_IN_REG = 17       # a7: C_in

        self.emit.emit_li(K_REG, K, f"K = {K}")
        self.emit.emit_li(STRIDE_H_REG, sh, f"sh = {sh}")
        self.emit.emit_li(STRIDE_W_REG, sw, f"sw = {sw}")
        self.emit.emit_li(W_REG, W, f"W = {W}")
        self.emit.emit_li32(HW_REG, H * W, f"H*W = {H*W}")
        self.emit.emit_li(C_IN_REG, C_in, f"C_in = {C_in}")

        # ── Register layout ────────────────────────────────────────────
        S2_IN_BASE       = self.S2   # s2: input base
        S3_W_BASE        = self.S3   # s3: weight base
        S4_BIAS_BASE     = self.S4   # s4: bias base
        S5_OUT_BASE      = self.S5   # s5: output base
        OH_REG           = self.S6   # s6: oh loop
        OW_REG           = self.S7   # s7: ow loop
        OC_REG           = self.S8   # s8: oc loop
        IC_REG           = self.S9   # s9: ic loop
        KH_REG           = self.S10  # s10: kh loop
        KW_REG           = self.S11  # s11: kw loop

        ACC_REG          = self.T0   # t0: acc
        VAL_REG          = self.T1   # t1: input value
        TMP_REG          = self.T2   # t2: temp / product
        IN_PTR           = self.T3   # t3: running input pointer
        COND_REG         = self.T4   # t4: condition / temp
        WT_PTR           = self.T5   # t5: running weight pointer
        T6_REG           = self.T6   # t6: general temporary
        # Entry saved a0/a1 in s0/s1. Keep spatial bases out of MAC/address
        # scratch registers, whose values change inside the ow/ic loops.
        IH_BASE_REG      = _R_A0
        IW_BASE_REG      = _R_A1

        L = self.emit.label

        # Product constants (computed at compile time, loaded when needed)
        H_out_W_out = H_out * W_out
        C_in_K_K = C_in * K * K
        K_K = K * K
        W_minus_K_times_4 = (W - K) * 4

        # Batch (N) is handled by emitting the whole computation N times with a
        # compile-time slice offset, rather than a runtime loop: N is known at
        # codegen time and is small (<= a few). N==1 emits exactly the old code.
        in_batch_stride = C_in * H * W
        out_batch_stride = C_out * H_out * W_out

        for _n in range(N):
            # Unique label prefix per batch copy (labels repeat across copies).
            previous_prefix = self.emit.label_prefix
            self.emit.label_prefix = f"{previous_prefix}n{_n}_"
            if _n > 0:
                # Advance both working bases by one sample stride. Bases carry
                # over between copies, so add an incremental stride (not n*).
                self.emit.emit_li32(TMP_REG, in_batch_stride,
                                    f"batch n={_n} in stride")
                self.emit.emit(rv_slli(TMP_REG, TMP_REG, 2), "*4")
                self.emit.emit(rv_add(S2_IN_BASE, S2_IN_BASE, TMP_REG),
                               "input base += slice")
                self.emit.emit_li32(TMP_REG, out_batch_stride,
                                    f"batch n={_n} out stride")
                self.emit.emit(rv_slli(TMP_REG, TMP_REG, 2), "*4")
                self.emit.emit(rv_add(S5_OUT_BASE, S5_OUT_BASE, TMP_REG),
                               "output base += slice")

            # ── oc loop ────────────────────────────────────────────────
            self.emit.emit(rv_addi(OC_REG, _R_ZERO, 0), f"oc=0 (C_out={C_out})")
            L("_conv_oc_loop")

            # ── oh loop ────────────────────────────────────────────────────
            self.emit.emit(rv_addi(OH_REG, _R_ZERO, 0), f"oh=0 (H_out={H_out})")
            L("_conv_oh_loop")

            # ih_base = oh * stride_h (precompute for this oh row)
            self.emit.emit(rv_mul(IH_BASE_REG, OH_REG, STRIDE_H_REG),
                           "ih_base = oh * stride_h")

            # ── ow loop ────────────────────────────────────────────────────
            self.emit.emit(rv_addi(OW_REG, _R_ZERO, 0), f"ow=0 (W_out={W_out})")
            L("_conv_ow_loop")

            # iw_base = ow * stride_w (precompute for this column)
            self.emit.emit(rv_mul(IW_BASE_REG, OW_REG, STRIDE_W_REG),
                           "iw_base = ow * stride_w")

            # Each output element starts a new reduction, including its bias.
            if b_name is not None:
                self.emit.emit(rv_slli(TMP_REG, OC_REG, 2), "offset = oc*4")
                self.emit.emit(rv_add(TMP_REG, S4_BIAS_BASE, TMP_REG), "+ bias_base")
                self.emit.emit(rv_lw(ACC_REG, TMP_REG, 0), "acc = bias[oc]")
            else:
                self.emit.emit(rv_addi(ACC_REG, _R_ZERO, 0), "acc = 0 (no bias)")

            if no_pad and K == 3:
                # ══════════════════════════════════════════════════════════════
                # Unrolled K=3 no-pad fast path: fully unroll kw+kh loops
                # + pointer-increment ic iteration (no address recalculation).
                # ~7 instr/MAC vs ~12 before.  2026-06-08
                # ══════════════════════════════════════════════════════════════
                # ic_advance = bytes to advance IN_PTR to next ic's same spatial pos
                # K*K loads plus K-1 row skips move (K-1)*W + K elements.
                # The next channel's window starts H*W elements from this one.
                ic_advance_bytes = (H * W - (K - 1) * W - K) * 4
                IC_ADV_REG = KH_REG   # s10 freed: no kh counter needed
                ROW_ADV_REG = KW_REG  # s11 freed: no kw counter needed

                self.emit.emit_li32(IC_ADV_REG, ic_advance_bytes,
                                   f"ic_adv={ic_advance_bytes}")
                self.emit.emit_li32(ROW_ADV_REG, W_minus_K_times_4,
                                   f"row_adv={W_minus_K_times_4}")

                # ic = 0
                self.emit.emit(rv_addi(IC_REG, _R_ZERO, 0),
                               f"ic=0 (C_in={C_in})")

                # ── Compute starting pointers for ic=0 (once, outside loop) ─
                # IN_PTR = input_base + (ih_base * W + iw_base) * 4
                self.emit.emit(rv_mul(T6_REG, IH_BASE_REG, W_REG),
                               "inoff = ih_base * W")
                self.emit.emit(rv_add(T6_REG, T6_REG, IW_BASE_REG),
                               "+ iw_base")
                self.emit.emit(rv_slli(T6_REG, T6_REG, 2),
                               "* 4")
                self.emit.emit(rv_add(IN_PTR, S2_IN_BASE, T6_REG),
                               "in_ptr = &input[0, ih_base, iw_base]")
                # WT_PTR = weight_base + oc * C_in*K*K * 4
                self.emit.emit_li32(T6_REG, C_in_K_K, f"oc_wt_step={C_in_K_K}")
                self.emit.emit(rv_mul(T6_REG, OC_REG, T6_REG), "wt_oc_base")
                self.emit.emit(rv_slli(T6_REG, T6_REG, 2),
                               f"wtoff = oc * {C_in_K_K} * 4")
                self.emit.emit(rv_add(WT_PTR, S3_W_BASE, T6_REG),
                               "wt_ptr = &weight[oc, 0, 0, 0]")

                L("_conv_ic_loop")

                # ── Fully unrolled 3×3 MAC block (9 MACs, no loop control) ─
                # Row 0 (kh=0): MAC 0,1,2 then row advance
                for _ in range(3):
                    self.emit.emit(rv_lw(VAL_REG, IN_PTR, 0),
                                   "load input")
                    self.emit.emit(rv_lw(T6_REG, WT_PTR, 0),
                                   "load weight")
                    self.emit.emit(rv_mul(T6_REG, VAL_REG, T6_REG),
                                   "Q16.16 mul")
                    self.emit.emit(rv_srai(T6_REG, T6_REG, 16),
                                   ">>16")
                    self.emit.emit(rv_add(ACC_REG, ACC_REG, T6_REG),
                                   "acc += prod")
                    self.emit.emit(rv_addi(IN_PTR, IN_PTR, 4),
                                   "in_ptr++")
                    self.emit.emit(rv_addi(WT_PTR, WT_PTR, 4),
                                   "wt_ptr++")
                # Row advance: in_ptr += (W-K)*4
                self.emit.emit(rv_add(IN_PTR, IN_PTR, ROW_ADV_REG),
                               "in_ptr += row_adv to next row")

                # Row 1 (kh=1): MAC 3,4,5 then row advance
                for _ in range(3):
                    self.emit.emit(rv_lw(VAL_REG, IN_PTR, 0),
                                   "load input")
                    self.emit.emit(rv_lw(T6_REG, WT_PTR, 0),
                                   "load weight")
                    self.emit.emit(rv_mul(T6_REG, VAL_REG, T6_REG),
                                   "Q16.16 mul")
                    self.emit.emit(rv_srai(T6_REG, T6_REG, 16),
                                   ">>16")
                    self.emit.emit(rv_add(ACC_REG, ACC_REG, T6_REG),
                                   "acc += prod")
                    self.emit.emit(rv_addi(IN_PTR, IN_PTR, 4),
                                   "in_ptr++")
                    self.emit.emit(rv_addi(WT_PTR, WT_PTR, 4),
                                   "wt_ptr++")
                # Row advance
                self.emit.emit(rv_add(IN_PTR, IN_PTR, ROW_ADV_REG),
                               "in_ptr += row_adv to next row")

                # Row 2 (kh=2): MAC 6,7,8 (last row, no row advance after)
                for _ in range(3):
                    self.emit.emit(rv_lw(VAL_REG, IN_PTR, 0),
                                   "load input")
                    self.emit.emit(rv_lw(T6_REG, WT_PTR, 0),
                                   "load weight")
                    self.emit.emit(rv_mul(T6_REG, VAL_REG, T6_REG),
                                   "Q16.16 mul")
                    self.emit.emit(rv_srai(T6_REG, T6_REG, 16),
                                   ">>16")
                    self.emit.emit(rv_add(ACC_REG, ACC_REG, T6_REG),
                                   "acc += prod")
                    self.emit.emit(rv_addi(IN_PTR, IN_PTR, 4),
                                   "in_ptr++")
                    self.emit.emit(rv_addi(WT_PTR, WT_PTR, 4),
                                   "wt_ptr++")

                # ── Advance IN_PTR to next ic's same spatial position ─────
                # WT_PTR already at next ic (weights contiguous across ic)
                self.emit.emit(rv_add(IN_PTR, IN_PTR, IC_ADV_REG),
                               "in_ptr += ic_advance")
                # ic++ and loop
                self.emit.emit(rv_addi(IC_REG, IC_REG, 1), "ic++")
                self.emit.emit(rv_slt(COND_REG, IC_REG, C_IN_REG), "ic < C_in?")
                self.emit.emit_branch(rv_bne, COND_REG, _R_ZERO,
                                      "_conv_ic_loop", "loop ic")

            else:
                # ══════════════════════════════════════════════════════════════
                # General loop path (K ≠ 3 or padded)
                # ══════════════════════════════════════════════════════════════
                # ── ic loop ────────────────────────────────────────────────
                self.emit.emit(rv_addi(IC_REG, _R_ZERO, 0), f"ic=0 (C_in={C_in})")
                L("_conv_ic_loop")

                # Step 1: IN_PTR = input_base + ic * H*W * 4
                self.emit.emit(rv_mul(VAL_REG, IC_REG, HW_REG),
                               "ic_offset = ic * H*W")
                self.emit.emit(rv_slli(VAL_REG, VAL_REG, 2),
                               "ic_byte_off = ic_offset * 4")
                self.emit.emit(rv_add(IN_PTR, S2_IN_BASE, VAL_REG),
                               "in_ptr = input_base + ic*H*W*4")

                # Step 2: WT_PTR = weight_base + oc * C_in*K*K * 4
                self.emit.emit_li32(VAL_REG, C_in_K_K, f"oc_wt_step={C_in_K_K}")
                self.emit.emit(rv_mul(VAL_REG, OC_REG, VAL_REG), "wt_oc_base")
                self.emit.emit(rv_slli(VAL_REG, VAL_REG, 2),
                               f"wt_oc_byte = {C_in_K_K}*4")
                self.emit.emit(rv_add(WT_PTR, S3_W_BASE, VAL_REG),
                               "wt_ptr = weight_base + oc_wt_byte")

                # Step 3: IN_PTR += (ih_base * W + iw_base) * 4
                self.emit.emit(rv_mul(VAL_REG, IH_BASE_REG, W_REG),
                               "ih_off = ih_base * W")
                self.emit.emit(rv_add(VAL_REG, VAL_REG, IW_BASE_REG),
                               "row_off = ih*W + iw_base")
                self.emit.emit(rv_slli(VAL_REG, VAL_REG, 2),
                               "row_byte = row_off * 4")
                self.emit.emit(rv_add(IN_PTR, IN_PTR, VAL_REG),
                               "in_ptr = &input[ic, ih_base, iw_base]")

                # Step 4: WT_PTR += ic * K*K * 4
                self.emit.emit_li32(T6_REG, K_K, f"K*K = {K_K}")
                self.emit.emit(rv_mul(VAL_REG, IC_REG, T6_REG),
                               f"ic_wt_off = ic * {K_K}")
                self.emit.emit(rv_slli(VAL_REG, VAL_REG, 2),
                               "ic_wt_byte = ic_wt_off * 4")
                self.emit.emit(rv_add(WT_PTR, WT_PTR, VAL_REG),
                               "wt_ptr += ic * K*K * 4")

                # ── kh loop ────────────────────────────────────────────────
                self.emit.emit(rv_addi(KH_REG, _R_ZERO, 0), f"kh=0 (K={K})")
                L("_conv_kh_loop")

                # ── kw loop (innermost) ────────────────────────────────────
                self.emit.emit(rv_addi(KW_REG, _R_ZERO, 0), f"kw=0 (K={K})")
                L("_conv_kw_loop")

                if no_pad:
                    # No-padding fast path: pointer-walking
                    self.emit.emit(rv_lw(VAL_REG, IN_PTR, 0),
                                   "load input[ic,ih,iw]")
                    self.emit.emit(rv_lw(T6_REG, WT_PTR, 0),
                                   "load weight[oc,ic,kh,kw]")
                    self.emit.emit(rv_mul(T6_REG, VAL_REG, T6_REG),
                                   "Q16.16 mul")
                    self.emit.emit(rv_srai(T6_REG, T6_REG, 16),
                                   ">> 16")
                    self.emit.emit(rv_add(ACC_REG, ACC_REG, T6_REG),
                                   "acc += product")
                    self.emit.emit(rv_addi(IN_PTR, IN_PTR, 4),
                                   "in_ptr += 4")
                    self.emit.emit(rv_addi(WT_PTR, WT_PTR, 4),
                                   "wt_ptr += 4")
                    self.emit.emit(rv_addi(KW_REG, KW_REG, 1), "kw++")
                    self.emit.emit(rv_slt(COND_REG, KW_REG, K_REG), "kw < K?")
                    self.emit.emit_branch(rv_bne, COND_REG, _R_ZERO,
                                          "_conv_kw_loop", "loop kw")
                else:
                    # Padding path: check coordinates before any input load.
                    self.emit.emit(rv_mul(TMP_REG, OH_REG, STRIDE_H_REG),
                                   "tmp = oh * stride_h")
                    self.emit.emit(rv_add(TMP_REG, TMP_REG, KH_REG), "tmp += kh")
                    self.emit.emit(rv_addi(TMP_REG, TMP_REG, -ph),
                                   f"tmp -= {ph}")

                    skip_label = f"_conv_skip_{len(self.emit.labels)}"
                    self.emit.emit(rv_slti(COND_REG, TMP_REG, 0), "ih < 0?")
                    self.emit.emit_branch(rv_bne, COND_REG, _R_ZERO, skip_label,
                                          "skip if ih < 0")
                    self.emit.emit_li(T6_REG, H, f"H = {H}")
                    self.emit.emit(rv_slt(COND_REG, TMP_REG, T6_REG), "ih < H?")
                    self.emit.emit_branch(rv_beq, COND_REG, _R_ZERO, skip_label,
                                          "skip if ih >= H")

                    self.emit.emit(rv_mul(VAL_REG, OW_REG, STRIDE_W_REG),
                                   "val = ow * stride_w")
                    self.emit.emit(rv_add(VAL_REG, VAL_REG, KW_REG), "val += kw")
                    self.emit.emit(rv_addi(VAL_REG, VAL_REG, -pw),
                                   f"val -= {pw}")
                    self.emit.emit(rv_slti(COND_REG, VAL_REG, 0), "iw < 0?")
                    self.emit.emit_branch(rv_bne, COND_REG, _R_ZERO, skip_label,
                                          "skip if iw < 0")

                    self.emit.emit_li(T6_REG, W, f"W = {W}")
                    self.emit.emit(rv_slt(COND_REG, VAL_REG, T6_REG), "iw < W?")
                    self.emit.emit_branch(rv_beq, COND_REG, _R_ZERO, skip_label,
                                          "skip if iw >= W")

                    self.emit.emit(rv_mul(T6_REG, IC_REG, HW_REG),
                                   "addr = ic*H*W")
                    self.emit.emit(rv_mul(COND_REG, TMP_REG, W_REG),
                                   "tmp = ih*W")
                    self.emit.emit(rv_add(T6_REG, T6_REG, COND_REG),
                                   "addr += ih*W")
                    self.emit.emit(rv_add(T6_REG, T6_REG, VAL_REG),
                                   "addr += iw")
                    self.emit.emit(rv_slli(T6_REG, T6_REG, 2), "addr *= 4")
                    self.emit.emit(rv_add(T6_REG, S2_IN_BASE, T6_REG),
                                   "addr += input_base")
                    self.emit.emit(rv_lw(VAL_REG, T6_REG, 0),
                                   "load input[ic,ih,iw]")

                    # WT_PTR already points at weight[oc, ic, 0, 0].
                    self.emit.emit(rv_mul(T6_REG, KH_REG, K_REG), "addr = kh*K")
                    self.emit.emit(rv_add(T6_REG, T6_REG, KW_REG),
                                   "addr += kw")
                    self.emit.emit(rv_slli(T6_REG, T6_REG, 2), "addr *= 4")
                    self.emit.emit(rv_add(T6_REG, WT_PTR, T6_REG),
                                   "addr += weight[oc,ic] base")
                    self.emit.emit(rv_lw(COND_REG, T6_REG, 0),
                                   "load weight[oc,ic,kh,kw]")

                    self.emit.emit(rv_mul(T6_REG, VAL_REG, COND_REG),
                                   "Q16.16 mul")
                    self.emit.emit(rv_srai(T6_REG, T6_REG, 16),
                                   ">> 16")
                    self.emit.emit(rv_add(ACC_REG, ACC_REG, T6_REG),
                                   "acc += product")

                    self.emit.label(skip_label)

                    self.emit.emit(rv_addi(KW_REG, KW_REG, 1), "kw++")
                    self.emit.emit(rv_slt(COND_REG, KW_REG, K_REG), "kw < K?")
                    self.emit.emit_branch(rv_bne, COND_REG, _R_ZERO,
                                          "_conv_kw_loop", "loop kw")

                # ── After kw loop: advance for next kh row ─────────────────
                if no_pad:
                    self.emit.emit(rv_addi(IN_PTR, IN_PTR, W_minus_K_times_4),
                                   f"in_ptr += (W({W})-K({K}))*4 = {W_minus_K_times_4}")

                # ── Increment kh ────────────────────────────────────────────
                self.emit.emit(rv_addi(KH_REG, KH_REG, 1), "kh++")
                self.emit.emit(rv_slt(COND_REG, KH_REG, K_REG), "kh < K?")
                self.emit.emit_branch(rv_bne, COND_REG, _R_ZERO,
                                      "_conv_kh_loop", "loop kh")

                # ── Increment ic ────────────────────────────────────────────
                self.emit.emit(rv_addi(IC_REG, IC_REG, 1), "ic++")
                self.emit.emit(rv_slt(COND_REG, IC_REG, C_IN_REG), "ic < C_in?")
                self.emit.emit_branch(rv_bne, COND_REG, _R_ZERO,
                                      "_conv_ic_loop", "loop ic")

            # ── Store output[oc, oh, ow] = acc ─────────────────────────────
            # out_offset = oc*H_out*W_out + oh*W_out + ow
            self.emit.emit_li(TMP_REG, H_out_W_out, f"H_out*W_out={H_out_W_out}")
            self.emit.emit(rv_mul(TMP_REG, OC_REG, TMP_REG),
                           "addr = oc * H_out_W_out")
            self.emit.emit_li(T6_REG, W_out, f"W_out={W_out}")
            self.emit.emit(rv_mul(T6_REG, OH_REG, T6_REG),
                           "tmp = oh * W_out")
            self.emit.emit(rv_add(TMP_REG, TMP_REG, T6_REG),
                           "addr += oh * W_out")
            self.emit.emit(rv_add(TMP_REG, TMP_REG, OW_REG),
                           "addr += ow")
            self.emit.emit(rv_slli(TMP_REG, TMP_REG, 2), "addr *= 4")
            self.emit.emit(rv_add(TMP_REG, S5_OUT_BASE, TMP_REG),
                           "+ output_base")
            self.emit.emit(rv_sw(TMP_REG, ACC_REG, 0),
                           "store output[oc,oh,ow]")

            # ── Increment ow ────────────────────────────────────────────────
            self.emit.emit(rv_addi(OW_REG, OW_REG, 1), "ow++")
            self.emit.emit_li(T6_REG, W_out, f"W_out={W_out}")
            self.emit.emit(rv_slt(COND_REG, OW_REG, T6_REG), "ow < W_out?")
            self.emit.emit_branch(rv_bne, COND_REG, _R_ZERO,
                                  "_conv_ow_loop", "loop ow")

            # ── Increment oh ────────────────────────────────────────────────
            self.emit.emit(rv_addi(OH_REG, OH_REG, 1), "oh++")
            self.emit.emit_li(T6_REG, H_out, f"H_out={H_out}")
            self.emit.emit(rv_slt(COND_REG, OH_REG, T6_REG), "oh < H_out?")
            self.emit.emit_branch(rv_bne, COND_REG, _R_ZERO,
                                  "_conv_oh_loop", "loop oh")

            # ── Increment oc ────────────────────────────────────────────────
            self.emit.emit(rv_addi(OC_REG, OC_REG, 1), "oc++")
            self.emit.emit_li(T6_REG, C_out, f"C_out={C_out}")
            self.emit.emit(rv_slt(COND_REG, OC_REG, T6_REG), "oc < C_out?")
            self.emit.emit_branch(rv_bne, COND_REG, _R_ZERO,
                                  "_conv_oc_loop", "loop oc")

            # restore label prefix after this batch copy
            self.emit.label_prefix = previous_prefix

    # ── ReLU ───────────────────────────────────────────────────────────

    def _gen_relu(self, node: NodeInfo) -> None:
        """Element-wise ReLU: output = max(input, 0).

        Branch-free implementation:
          slti mask, src, 0    → mask = (src < 0) ? 1 : 0
          addi mask, mask, -1  → mask = (src < 0) ? 0 : 0xFFFFFFFF
          and  dst, src, mask  → dst = (src >= 0) ? src : 0
        """
        x_name = node.inputs[0]
        out_name = node.outputs[0]
        shape = self.model.get_shape(x_name)
        num_el = 1
        for d in shape:
            num_el *= d
        self.mem.alloc_workspace(out_name, num_el)

        self._get_workspace_addr(x_name, self.S2)
        self._get_workspace_addr(out_name, self.S5)

        i_reg = self.S6
        addr_reg = self.T3
        val_reg = self.T1
        mask_reg = self.T0
        cond_reg = self.T4

        L = self.emit.label
        self.emit.emit(rv_addi(i_reg, _R_ZERO, 0), f"relu ({num_el} elements)")

        loop_l = "_relu_loop"
        L(loop_l)
        # Load x[i]
        self.emit.emit(rv_slli(addr_reg, i_reg, 2), "")
        self.emit.emit(rv_add(addr_reg, self.S2, addr_reg), "")
        self.emit.emit(rv_lw(val_reg, addr_reg, 0), "x = input[i]")

        # Branch-free ReLU: mask = (x<0)?1:0; mask=~mask+1; x = x & mask
        self.emit.emit(rv_slti(mask_reg, val_reg, 0), "mask = (x < 0)?1:0")
        self.emit.emit(rv_addi(mask_reg, mask_reg, -1), "mask = (x<0)?0:-1")
        self.emit.emit(rv_and(val_reg, val_reg, mask_reg), "x = max(x,0)")

        # Store output[i]
        self.emit.emit(rv_slli(addr_reg, i_reg, 2), "")
        self.emit.emit(rv_add(addr_reg, self.S5, addr_reg), "")
        self.emit.emit(rv_sw(addr_reg, val_reg, 0), "output[i] = x")

        # i++ and loop
        self.emit.emit(rv_addi(i_reg, i_reg, 1), "i++")
        self.emit.emit_li32(self.T6, num_el, f"n={num_el}")
        self.emit.emit(rv_slt(cond_reg, i_reg, self.T6), "i < n?")
        self.emit.emit_branch(rv_bne, cond_reg, _R_ZERO, loop_l, "loop")
        self.emit.label("_relu_done")

    # ── MaxPool ────────────────────────────────────────────────────────

    def _gen_maxpool(self, node: NodeInfo) -> None:
        """2D MaxPool: NCHW layout with nested loops.

        For each output position (oc,oh,ow), compute max over
        the kernel window input[oc, oh*stride:oh*stride+Kh, ow*stride:ow*stride+Kw].
        """
        x_name = node.inputs[0]
        out_name = node.outputs[0]
        x_shape = self.model.get_shape(x_name)

        N = x_shape[0] if len(x_shape) > 0 else 1
        C = x_shape[1] if len(x_shape) > 1 else 1
        H = x_shape[2] if len(x_shape) > 2 else 1
        W = x_shape[3] if len(x_shape) > 3 else 1

        attrs = node.attrs
        kernel = attrs.get("kernel_shape", [2, 2])
        Kh = int(kernel[0]) if isinstance(kernel, list) else 2
        Kw = int(kernel[1]) if isinstance(kernel, list) and len(kernel) > 1 else Kh

        stride = attrs.get("strides", [2, 2])
        Sh = int(stride[0]) if isinstance(stride, list) else 2
        Sw = int(stride[1]) if isinstance(stride, list) and len(stride) > 1 else Sh

        H_out = (H - Kh) // Sh + 1
        W_out = (W - Kw) // Sw + 1
        out_elements = N * C * H_out * W_out
        self.mem.alloc_workspace(out_name, out_elements)

        self._get_workspace_addr(x_name, self.S2)
        self._get_workspace_addr(out_name, self.S5)

        # Registers for loop counters
        oc_reg = self.S6   # output channel
        oh_reg = self.S7   # output height
        ow_reg = self.S8   # output width
        kh_reg = self.S9   # kernel height
        kw_reg = self.S10  # kernel width
        max_reg = self.T0  # current max value
        val_reg = self.T1  # loaded value
        addr_reg = self.T3 # address computation
        cmp_reg = self.T4 # comparison result
        tmp_reg = self.T5 # temporary

        L = self.emit.label

        # oc = 0
        self.emit.emit(rv_addi(oc_reg, _R_ZERO, 0), f"oc=0 (C={C})")
        L("_mp_oc_loop")

        # oh = 0
        self.emit.emit(rv_addi(oh_reg, _R_ZERO, 0), f"oh=0 (H_out={H_out})")
        L("_mp_oh_loop")

        # ow = 0
        self.emit.emit(rv_addi(ow_reg, _R_ZERO, 0), f"ow=0 (W_out={W_out})")
        L("_mp_ow_loop")

        # max_val = INT32_MIN (−2^31)
        self.emit.emit_li32(max_reg, -2147483648, "max = INT32_MIN")

        # kh = 0
        self.emit.emit(rv_addi(kh_reg, _R_ZERO, 0), f"kh=0 (Kh={Kh})")
        L("_mp_kh_loop")

        # kw = 0
        self.emit.emit(rv_addi(kw_reg, _R_ZERO, 0), f"kw=0 (Kw={Kw})")
        L("_mp_kw_loop")

        # Compute input index: ih = oh*Sh + kh, iw = ow*Sw + kw
        self.emit.emit_li(tmp_reg, Sh, f"Sh={Sh}")
        self.emit.emit(rv_mul(tmp_reg, oh_reg, tmp_reg), "tmp = oh * Sh")
        self.emit.emit(rv_add(tmp_reg, tmp_reg, kh_reg), "ih = oh*Sh + kh")

        self.emit.emit_li(addr_reg, Sw, f"Sw={Sw}")
        self.emit.emit(rv_mul(addr_reg, ow_reg, addr_reg), "addr = ow * Sw")
        self.emit.emit(rv_add(addr_reg, addr_reg, kw_reg), "iw = ow*Sw + kw")

        # input_offset = oc*H*W + ih*W + iw
        self.emit.emit_li(self.T6, H * W, f"H*W={H*W}")
        self.emit.emit(rv_mul(cmp_reg, oc_reg, self.T6), "offset = oc*H*W")
        self.emit.emit_li(self.T6, W, f"W={W}")
        self.emit.emit(rv_mul(self.T6, tmp_reg, self.T6), "tmp = ih*W")
        self.emit.emit(rv_add(cmp_reg, cmp_reg, self.T6), "offset += ih*W")
        self.emit.emit(rv_add(cmp_reg, cmp_reg, addr_reg), "offset += iw")
        self.emit.emit(rv_slli(cmp_reg, cmp_reg, 2), "offset *= 4")
        self.emit.emit(rv_add(cmp_reg, self.S2, cmp_reg), "addr = input + offset")
        self.emit.emit(rv_lw(val_reg, cmp_reg, 0), "x = input[oc,ih,iw]")

        # if val > max: max = val
        self.emit.emit(rv_slt(cmp_reg, max_reg, val_reg), "max < val?")
        skip_label = f"_mp_noupdate_{len(self.emit.labels)}"
        self.emit.emit_branch(rv_beq, cmp_reg, _R_ZERO, skip_label, "skip if max >= val")
        self.emit.emit(rv_mv(max_reg, val_reg), "max = val")
        self.emit.label(skip_label)

        # kw++
        self.emit.emit(rv_addi(kw_reg, kw_reg, 1), "kw++")
        self.emit.emit_li(self.T6, Kw, f"Kw={Kw}")
        self.emit.emit(rv_slt(cmp_reg, kw_reg, self.T6), "kw < Kw?")
        self.emit.emit_branch(rv_bne, cmp_reg, _R_ZERO, "_mp_kw_loop", "loop kw")

        # kh++
        self.emit.emit(rv_addi(kh_reg, kh_reg, 1), "kh++")
        self.emit.emit_li(self.T6, Kh, f"Kh={Kh}")
        self.emit.emit(rv_slt(cmp_reg, kh_reg, self.T6), "kh < Kh?")
        self.emit.emit_branch(rv_bne, cmp_reg, _R_ZERO, "_mp_kh_loop", "loop kh")

        # Store output[oc,oh,ow] = max
        # output_offset = oc*H_out*W_out + oh*W_out + ow
        self.emit.emit_li(addr_reg, H_out * W_out, f"Hout*Wout={H_out*W_out}")
        self.emit.emit(rv_mul(addr_reg, oc_reg, addr_reg), "addr = oc*Hout*Wout")
        self.emit.emit_li(self.T6, W_out, f"Wout={W_out}")
        self.emit.emit(rv_mul(self.T6, oh_reg, self.T6), "tmp = oh*Wout")
        self.emit.emit(rv_add(addr_reg, addr_reg, self.T6), "addr += oh*Wout")
        self.emit.emit(rv_add(addr_reg, addr_reg, ow_reg), "addr += ow")
        self.emit.emit(rv_slli(addr_reg, addr_reg, 2), "addr *= 4")
        self.emit.emit(rv_add(addr_reg, self.S5, addr_reg), "addr += output")
        self.emit.emit(rv_sw(addr_reg, max_reg, 0), "store output[oc,oh,ow]")

        # ow++
        self.emit.emit(rv_addi(ow_reg, ow_reg, 1), "ow++")
        self.emit.emit_li(self.T6, W_out, f"Wout={W_out}")
        self.emit.emit(rv_slt(cmp_reg, ow_reg, self.T6), "ow < Wout?")
        self.emit.emit_branch(rv_bne, cmp_reg, _R_ZERO, "_mp_ow_loop", "loop ow")

        # oh++
        self.emit.emit(rv_addi(oh_reg, oh_reg, 1), "oh++")
        self.emit.emit_li(self.T6, H_out, f"Hout={H_out}")
        self.emit.emit(rv_slt(cmp_reg, oh_reg, self.T6), "oh < Hout?")
        self.emit.emit_branch(rv_bne, cmp_reg, _R_ZERO, "_mp_oh_loop", "loop oh")

        # oc++
        self.emit.emit(rv_addi(oc_reg, oc_reg, 1), "oc++")
        self.emit.emit_li(self.T6, C, f"C={C}")
        self.emit.emit(rv_slt(cmp_reg, oc_reg, self.T6), "oc < C?")
        self.emit.emit_branch(rv_bne, cmp_reg, _R_ZERO, "_mp_oc_loop", "loop oc")

        self.emit.label("_mp_done")

    # ── Gemm (Fully Connected) ─────────────────────────────────────────

    def _gen_gemm(self, node: NodeInfo) -> None:
        """Gemm: Y = alpha * A * B + beta * C.

        For a typical FC layer with transB=1:
          A: (M, K), B: (N, K) stored as (N, K), C: (N,) bias
          Y: (M, N)
          Y[i, j] = bias[j] + sum_k(A[i, k] * B[j, k])
        """
        a_name = node.inputs[0]
        w_name = node.inputs[1]
        # Gemm bias (C) is optional; a missing third input means "no bias".
        b_name = node.inputs[2] if len(node.inputs) > 2 and node.inputs[2] else None
        out_name = node.outputs[0]

        a_shape = self.model.get_shape(a_name)
        w_shape = self.model.get_shape(w_name)

        trans_b = node.attrs.get("transB", 0)
        trans_b_flag = bool(trans_b) if isinstance(trans_b, int) else False

        M = a_shape[0] if len(a_shape) > 0 else 1
        K = a_shape[1] if len(a_shape) > 1 else 1

        if trans_b_flag:
            N = w_shape[0] if len(w_shape) > 0 else 1
            # weight is (N, K), so K must match
        else:
            N = w_shape[1] if len(w_shape) > 1 else 1
            # weight is (K, N)

        out_elements = M * N
        self.mem.alloc_workspace(out_name, out_elements)

        self._get_workspace_addr(a_name, self.S2)
        self._get_weight_addr(w_name, self.S3)
        if b_name is not None:
            self._get_weight_addr(b_name, self.S4)
        self._get_workspace_addr(out_name, self.S5)

        # Nested loops: for i in range(M), for j in range(N)
        i_reg = self.S6
        j_reg = self.S7
        k_reg = self.S8
        acc_reg = self.T0
        addr_reg = self.T3
        cond_reg = self.T4
        val_a_reg = self.T5
        val_w_reg = self.T6

        L = self.emit.label

        # i = 0
        self.emit.emit(rv_addi(i_reg, _R_ZERO, 0), f"i=0 (M={M})")
        L("_gemm_i_loop")

        # j = 0
        self.emit.emit(rv_addi(j_reg, _R_ZERO, 0), f"j=0 (N={N})")
        L("_gemm_j_loop")

        # Load bias[j] → acc (or zero when Gemm has no C input)
        if b_name is not None:
            self.emit.emit(rv_slli(addr_reg, j_reg, 2), "addr = j*4")
            self.emit.emit(rv_add(addr_reg, self.S4, addr_reg), "addr += bias_base")
            self.emit.emit(rv_lw(acc_reg, addr_reg, 0), "acc = bias[j]")
        else:
            self.emit.emit(rv_addi(acc_reg, _R_ZERO, 0), "acc = 0 (no bias)")

        # k = 0
        self.emit.emit(rv_addi(k_reg, _R_ZERO, 0), f"k=0 (K={K})")
        L("_gemm_k_loop")

        # Load A[i, k]
        self.emit.emit_li(addr_reg, K, f"K={K}")
        self.emit.emit(rv_mul(addr_reg, i_reg, addr_reg), "addr = i*K")
        self.emit.emit(rv_add(addr_reg, addr_reg, k_reg), "addr += k")
        self.emit.emit(rv_slli(addr_reg, addr_reg, 2), "addr *= 4")
        self.emit.emit(rv_add(addr_reg, self.S2, addr_reg), "addr += A_base")
        self.emit.emit(rv_lw(val_a_reg, addr_reg, 0), "load A[i,k]")

        # Load W[j, k] (since transB, weight is N×K)
        self.emit.emit_li(addr_reg, K, f"K={K}")
        self.emit.emit(rv_mul(addr_reg, j_reg, addr_reg), "addr = j*K")
        self.emit.emit(rv_add(addr_reg, addr_reg, k_reg), "addr += k")
        self.emit.emit(rv_slli(addr_reg, addr_reg, 2), "addr *= 4")
        self.emit.emit(rv_add(addr_reg, self.S3, addr_reg), "addr += W_base")
        self.emit.emit(rv_lw(val_w_reg, addr_reg, 0), "load W[j,k]")

        # MAC: acc += A[i,k] * W[j,k]
        self.emit.emit_qmul_srai(self.T1, val_a_reg, val_w_reg, self.T2,
                                 "Q16.16: A[i,k] * W[j,k]")
        self.emit.emit(rv_add(acc_reg, acc_reg, self.T1), "acc += product")

        # k++
        self.emit.emit(rv_addi(k_reg, k_reg, 1), "k++")
        self.emit.emit_li(self.T6, K, f"K={K}")
        self.emit.emit(rv_slt(cond_reg, k_reg, self.T6), "k < K?")
        self.emit.emit_branch(rv_bne, cond_reg, _R_ZERO, "_gemm_k_loop", "loop k")

        # Store output[i, j] = acc
        self.emit.emit_li(addr_reg, N, f"N={N}")
        self.emit.emit(rv_mul(addr_reg, i_reg, addr_reg), "addr = i*N")
        self.emit.emit(rv_add(addr_reg, addr_reg, j_reg), "addr += j")
        self.emit.emit(rv_slli(addr_reg, addr_reg, 2), "addr *= 4")
        self.emit.emit(rv_add(addr_reg, self.S5, addr_reg), "addr += output_base")
        self.emit.emit(rv_sw(addr_reg, acc_reg, 0), "store Y[i,j]")

        # j++
        self.emit.emit(rv_addi(j_reg, j_reg, 1), "j++")
        self.emit.emit_li(self.T6, N, f"N={N}")
        self.emit.emit(rv_slt(cond_reg, j_reg, self.T6), "j < N?")
        self.emit.emit_branch(rv_bne, cond_reg, _R_ZERO, "_gemm_j_loop", "loop j")

        # i++
        self.emit.emit(rv_addi(i_reg, i_reg, 1), "i++")
        self.emit.emit_li(self.T6, M, f"M={M}")
        self.emit.emit(rv_slt(cond_reg, i_reg, self.T6), "i < M?")
        self.emit.emit_branch(rv_bne, cond_reg, _R_ZERO, "_gemm_i_loop", "loop i")

    # ── Sigmoid ────────────────────────────────────────────────────────

    def _gen_sigmoid(self, node: NodeInfo) -> None:
        """Sigmoid using integer piecewise approximation.

        sigmoid(x) ≈ 1/(1+exp(-x))

        Q16.16 approximation:
          x ≤ -3*65536 → 0
          x ≥ +3*65536 → 65536
          else → 32768 + x/8  (linear approximation around x=0)
        """
        x_name = node.inputs[0]
        out_name = node.outputs[0]
        shape = self.model.get_shape(x_name)

        num_el = 1
        for d in shape:
            num_el *= d
        self.mem.alloc_workspace(out_name, num_el)

        self._get_workspace_addr(x_name, self.S2)
        self._get_workspace_addr(out_name, self.S5)

        L = self.emit.label
        i_reg = self.S6
        addr_reg = self.T3
        val_reg = self.T1
        dst_reg = self.T0
        cond_reg = self.T4

        self.emit.emit(rv_addi(i_reg, _R_ZERO, 0), f"i=0 ({num_el} elements)")
        L("_sigmoid_loop")

        # Load x[i]
        self.emit.emit(rv_slli(addr_reg, i_reg, 2), "addr = i*4")
        self.emit.emit(rv_add(addr_reg, self.S2, addr_reg), "")
        self.emit.emit(rv_lw(val_reg, addr_reg, 0), "x = input[i]")

        # if x <= -196608 (-3 in Q16.16): result = 0
        self.emit.emit_li32(self.T6, -196608, "-3*65536")
        self.emit.emit(rv_slt(cond_reg, val_reg, self.T6), "x < -3.0?")
        skip1_label = f"_sig_skip1_{len(self.emit.labels)}"
        self.emit.emit_branch(rv_beq, cond_reg, _R_ZERO, skip1_label, "skip if not")
        self.emit.emit(rv_addi(dst_reg, _R_ZERO, 0), "result = 0")
        store_label = f"_sig_store_{len(self.emit.labels)}"
        self.emit.emit_jump(rv_j, store_label, "goto store")

        self.emit.label(skip1_label)

        # if x >= 196608 (3 in Q16.16): result = 65536
        self.emit.emit_li32(self.T6, 196608, "3*65536")
        self.emit.emit(rv_slt(cond_reg, val_reg, self.T6), "x < 3.0?")
        skip2_label = f"_sig_skip2_{len(self.emit.labels)}"
        self.emit.emit_branch(rv_bne, cond_reg, _R_ZERO, skip2_label, "skip if x < 3")
        self.emit.emit_li32(dst_reg, 65536, "result = 1.0 (Q16.16)")
        self.emit.emit_jump(rv_j, store_label, "goto store")

        self.emit.label(skip2_label)

        # Linear: result = 32768 + x/8
        self.emit.emit(rv_srai(dst_reg, val_reg, 3), "x / 8")
        self.emit.emit_li(self.T6, 32768, "0.5 (Q16.16)")
        self.emit.emit(rv_add(dst_reg, dst_reg, self.T6), "+ 0.5 (Q16.16)")
        # Clamp to [0, 65536]
        self.emit.emit(rv_slti(cond_reg, dst_reg, 0), "result < 0?")
        cl_label = f"_sig_cl_{len(self.emit.labels)}"
        self.emit.emit_branch(rv_beq, cond_reg, _R_ZERO, cl_label, "")
        self.emit.emit(rv_addi(dst_reg, _R_ZERO, 0), "clamp to 0")
        self.emit.label(cl_label)

        self.emit.label(store_label)

        # Store output[i]
        self.emit.emit(rv_slli(addr_reg, i_reg, 2), "addr = i*4")
        self.emit.emit(rv_add(addr_reg, self.S5, addr_reg), "")
        self.emit.emit(rv_sw(addr_reg, dst_reg, 0), "store output[i]")

        # i++
        self.emit.emit(rv_addi(i_reg, i_reg, 1), "i++")
        self.emit.emit_li32(self.T6, num_el, f"num_el={num_el}")
        self.emit.emit(rv_slt(cond_reg, i_reg, self.T6), "i < num_el?")
        self.emit.emit_branch(rv_bne, cond_reg, _R_ZERO, "_sigmoid_loop", "loop sigmoid")

    # ── Reshape ────────────────────────────────────────────────────────

    def _gen_reshape(self, node: NodeInfo) -> None:
        """Reshape is a no-op: just copy the tensor reference.

        We copy the data to a new workspace location (identity copy).
        """
        x_name = node.inputs[0]
        out_name = node.outputs[0]
        shape = self.model.get_shape(x_name)

        num_el = 1
        for d in shape:
            num_el *= d

        self.mem.alloc_workspace(out_name, num_el)

        self._get_workspace_addr(x_name, self.S2)
        self._get_workspace_addr(out_name, self.S5)

        i_reg = self.S6
        addr_src = self.T3
        addr_dst = self.T5
        val_reg = self.T1
        cond_reg = self.T4

        L = self.emit.label

        self.emit.emit(rv_addi(i_reg, _R_ZERO, 0), f"reshape copy ({num_el} elements)")
        L("_reshape_loop")

        # Load
        self.emit.emit(rv_slli(addr_src, i_reg, 2), "")
        self.emit.emit(rv_add(addr_src, self.S2, addr_src), "")
        self.emit.emit(rv_lw(val_reg, addr_src, 0), "load src[i]")

        # Store
        self.emit.emit(rv_slli(addr_dst, i_reg, 2), "")
        self.emit.emit(rv_add(addr_dst, self.S5, addr_dst), "")
        self.emit.emit(rv_sw(addr_dst, val_reg, 0), "store dst[i]")

        # i++
        self.emit.emit(rv_addi(i_reg, i_reg, 1), "i++")
        self.emit.emit_li32(self.T6, num_el, f"num_el={num_el}")
        self.emit.emit(rv_slt(cond_reg, i_reg, self.T6), "i < num_el?")
        self.emit.emit_branch(rv_bne, cond_reg, _R_ZERO, "_reshape_loop", "loop")

    # ── FWHT ───────────────────────────────────────────────────────────

    def _gen_fwht(self, node: NodeInfo) -> None:
        """Scalar iterative FWHT over the flattened input (N a power of two).

        Algorithm (standard in-place butterfly on Q16.16 values):
          for length in 1, 2, 4, ..., N/2:
            for i in range(0, N, 2*length):
              for j in range(length):
                u = a[i+j]; v = a[i+j+length]
                a[i+j] = u+v; a[i+j+length] = u-v

        The input is first copied to the output workspace, then transformed
        in place there (input and output are distinct buffers).
        """
        x_name = node.inputs[0]
        out_name = node.outputs[0]
        shape = self.model.get_shape(x_name)

        num_el = 1
        for d in shape:
            num_el *= d

        if num_el <= 0 or (num_el & (num_el - 1)) != 0:
            raise ValueError(
                f"Fwht requires a power-of-two element count, got {num_el}"
            )

        direction = node.attrs.get("direction", "forward")
        if direction not in ("forward", "inverse"):
            raise ValueError(
                f"Fwht direction must be 'forward' or 'inverse', got {direction!r}"
            )
        # Inverse = forward Hadamard (H is symmetric, H^2 = N*I) then scale 1/N.
        # N is a power of two, so 1/N is an arithmetic right shift by log2(N).
        inv_shift = num_el.bit_length() - 1

        self.mem.alloc_workspace(out_name, num_el)

        self._get_workspace_addr(x_name, self.S2)
        self._get_workspace_addr(out_name, self.S5)

        L = self.emit.label

        # ── Copy input → output workspace ──────────────────────────────
        i_reg = self.S6
        addr_src = self.T3
        addr_dst = self.T5
        val_reg = self.T1
        cond_reg = self.T4

        self.emit.emit(rv_addi(i_reg, _R_ZERO, 0), f"fwht copy ({num_el} el)")
        L("_fwht_copy_loop")
        self.emit.emit(rv_slli(addr_src, i_reg, 2), "")
        self.emit.emit(rv_add(addr_src, self.S2, addr_src), "")
        self.emit.emit(rv_lw(val_reg, addr_src, 0), "load src[i]")
        self.emit.emit(rv_slli(addr_dst, i_reg, 2), "")
        self.emit.emit(rv_add(addr_dst, self.S5, addr_dst), "")
        self.emit.emit(rv_sw(addr_dst, val_reg, 0), "store dst[i]")
        self.emit.emit(rv_addi(i_reg, i_reg, 1), "i++")
        self.emit.emit_li32(self.T6, num_el, f"n={num_el}")
        self.emit.emit(rv_slt(cond_reg, i_reg, self.T6), "i < n?")
        self.emit.emit_branch(rv_bne, cond_reg, _R_ZERO, "_fwht_copy_loop", "loop")

        # ── Butterflies (in place on output workspace) ─────────────────
        len_reg = self.S6       # current butterfly length
        i_reg = self.S7         # block start index
        j_reg = self.S8         # offset inside block
        idx_reg = self.S9       # i + j
        step_reg = self.S10     # 2 * length
        n_reg = self.S11        # N (loop-invariant)

        addr_a = self.T3
        addr_b = self.T5
        u_reg = self.T0
        v_reg = self.T1
        cond_reg = self.T4
        tmp_reg = self.T6

        self.emit.emit_li32(n_reg, num_el, f"N={num_el}")
        self.emit.emit(rv_addi(len_reg, _R_ZERO, 1), "length = 1")

        L("_fwht_len_loop")
        self.emit.emit(rv_slt(cond_reg, len_reg, n_reg), "length < N?")
        self.emit.emit_branch(rv_beq, cond_reg, _R_ZERO, "_fwht_done", "exit")
        self.emit.emit(rv_slli(step_reg, len_reg, 1), "step = 2*length")

        self.emit.emit(rv_addi(i_reg, _R_ZERO, 0), "i = 0")
        L("_fwht_i_loop")
        self.emit.emit(rv_addi(j_reg, _R_ZERO, 0), "j = 0")
        L("_fwht_j_loop")

        # addr_a = out + (i+j)*4 ; addr_b = out + (i+j+length)*4
        self.emit.emit(rv_add(idx_reg, i_reg, j_reg), "idx = i+j")
        self.emit.emit(rv_slli(addr_a, idx_reg, 2), "idx*4")
        self.emit.emit(rv_add(addr_a, self.S5, addr_a), "addr_a")
        self.emit.emit(rv_add(tmp_reg, idx_reg, len_reg), "idx+length")
        self.emit.emit(rv_slli(tmp_reg, tmp_reg, 2), "(idx+length)*4")
        self.emit.emit(rv_add(addr_b, self.S5, tmp_reg), "addr_b")

        self.emit.emit(rv_lw(u_reg, addr_a, 0), "u = a[idx]")
        self.emit.emit(rv_lw(v_reg, addr_b, 0), "v = a[idx+length]")
        self.emit.emit(rv_add(tmp_reg, u_reg, v_reg), "u+v")
        self.emit.emit(rv_sw(addr_a, tmp_reg, 0), "a[idx] = u+v")
        self.emit.emit(rv_sub(tmp_reg, u_reg, v_reg), "u-v")
        self.emit.emit(rv_sw(addr_b, tmp_reg, 0), "a[idx+length] = u-v")

        # j++ ; j < length?
        self.emit.emit(rv_addi(j_reg, j_reg, 1), "j++")
        self.emit.emit(rv_slt(cond_reg, j_reg, len_reg), "j < length?")
        self.emit.emit_branch(rv_bne, cond_reg, _R_ZERO, "_fwht_j_loop", "loop j")

        # i += step ; i < N?
        self.emit.emit(rv_add(i_reg, i_reg, step_reg), "i += step")
        self.emit.emit(rv_slt(cond_reg, i_reg, n_reg), "i < N?")
        self.emit.emit_branch(rv_bne, cond_reg, _R_ZERO, "_fwht_i_loop", "loop i")

        # length *= 2 ; loop
        self.emit.emit(rv_slli(len_reg, len_reg, 1), "length *= 2")
        self.emit.emit_jump(rv_j, "_fwht_len_loop", "loop length")

        L("_fwht_done")

        # ── Inverse scaling: a[i] = a[i] >> log2(N) ────────────────────
        if direction == "inverse" and inv_shift > 0:
            i_reg = self.S6
            addr = self.T3
            val = self.T1
            cond = self.T4
            self.emit.emit(rv_addi(i_reg, _R_ZERO, 0), "inv-scale i=0")
            L("_fwht_scale_loop")
            self.emit.emit(rv_slli(addr, i_reg, 2), "")
            self.emit.emit(rv_add(addr, self.S5, addr), "")
            self.emit.emit(rv_lw(val, addr, 0), "load a[i]")
            self.emit.emit(rv_srai(val, val, inv_shift), f"a[i] >>= {inv_shift}")
            self.emit.emit(rv_sw(addr, val, 0), "store a[i]")
            self.emit.emit(rv_addi(i_reg, i_reg, 1), "i++")
            self.emit.emit_li32(self.T6, num_el, f"n={num_el}")
            self.emit.emit(rv_slt(cond, i_reg, self.T6), "i < n?")
            self.emit.emit_branch(rv_bne, cond, _R_ZERO, "_fwht_scale_loop", "loop")

    # ── Sparse matrix multiply (CSR) ───────────────────────────────────

    def _gen_spmmcsr(self, node: NodeInfo) -> None:
        """CSR sparse matrix × dense matrix (SpMM; N=1 degenerates to SpMV).

        Inputs: (values[q16, nnz], col_indices[int32, nnz],
                 row_ptr[int32, M+1], B[q16, K, N]) -> C[q16, M, N]

        for i in range(M):
            for n in range(N): C[i,n] = 0
            for j in row_ptr[i] .. row_ptr[i+1]-1:
                a = values[j]; k = col_indices[j]
                for n in range(N):
                    C[i,n] += (a * B[k,n]) >> 16
        """
        v_name = node.inputs[0]
        col_name = node.inputs[1]
        row_name = node.inputs[2]
        b_name = node.inputs[3]
        out_name = node.outputs[0]

        row_shape = self.model.get_shape(row_name)
        M = (row_shape[0] - 1) if row_shape else 0
        b_shape = self.model.get_shape(b_name)
        N = b_shape[1] if len(b_shape) >= 2 else 1

        self.mem.alloc_workspace(out_name, M * N)

        base_v = self.S2
        base_col = self.S3
        base_row = self.S4
        base_b = self.S5
        base_c = self.S6

        i_reg = self.S7
        j_reg = self.S8
        jend_reg = self.S9
        n_reg = self.S10
        ncol_reg = self.S11

        acc = self.T0
        a_reg = self.T1
        k_reg = self.T2
        addr = self.T3
        cond = self.T4
        tmp = self.T6

        self._get_weight_addr(v_name, base_v)
        self._get_weight_addr(col_name, base_col)
        self._get_weight_addr(row_name, base_row)
        self._get_workspace_addr(b_name, base_b)
        self._get_workspace_addr(out_name, base_c)

        self.emit.emit_li(ncol_reg, N, f"N={N}")

        L = self.emit.label

        # for i in range(M) — condition at top so M=0 is a no-op
        self.emit.emit(rv_addi(i_reg, _R_ZERO, 0), f"i=0 (M={M})")
        L("_spmm_i_loop")
        self.emit.emit_li32(tmp, M, f"M={M}")
        self.emit.emit(rv_slt(cond, i_reg, tmp), "i < M?")
        self.emit.emit_branch(rv_beq, cond, _R_ZERO, "_spmm_done", "exit")

        # j = row_ptr[i]; j_end = row_ptr[i+1]
        self.emit.emit(rv_slli(addr, i_reg, 2), "i*4")
        self.emit.emit(rv_add(addr, base_row, addr), "&row_ptr[i]")
        self.emit.emit(rv_lw(j_reg, addr, 0), "j = row_ptr[i]")
        self.emit.emit(rv_lw(jend_reg, addr, 4), "j_end = row_ptr[i+1]")

        # Zero the C row: for n in range(N)
        self.emit.emit(rv_addi(n_reg, _R_ZERO, 0), "n=0 (zero)")
        L("_spmm_zero_loop")
        self.emit.emit(rv_mul(tmp, i_reg, ncol_reg), "i*N")
        self.emit.emit(rv_add(tmp, tmp, n_reg), "+ n")
        self.emit.emit(rv_slli(tmp, tmp, 2), "*4")
        self.emit.emit(rv_add(tmp, base_c, tmp), "&C[i,n]")
        self.emit.emit(rv_sw(tmp, _R_ZERO, 0), "C[i,n] = 0")
        self.emit.emit(rv_addi(n_reg, n_reg, 1), "n++")
        self.emit.emit(rv_slt(cond, n_reg, ncol_reg), "n < N?")
        self.emit.emit_branch(rv_bne, cond, _R_ZERO, "_spmm_zero_loop", "loop")

        # for j in [j, j_end)
        L("_spmm_j_loop")
        self.emit.emit(rv_slt(cond, j_reg, jend_reg), "j < j_end?")
        self.emit.emit_branch(rv_beq, cond, _R_ZERO, "_spmm_j_done", "exit")

        # a = values[j]
        self.emit.emit(rv_slli(addr, j_reg, 2), "j*4")
        self.emit.emit(rv_add(addr, base_v, addr), "&values[j]")
        self.emit.emit(rv_lw(a_reg, addr, 0), "a = values[j]")
        # k = col_indices[j]  (indirect load -> addresses B's row)
        self.emit.emit(rv_slli(addr, j_reg, 2), "j*4")
        self.emit.emit(rv_add(addr, base_col, addr), "&col[j]")
        self.emit.emit(rv_lw(k_reg, addr, 0), "k = col[j]")

        # for n in range(N)
        self.emit.emit(rv_addi(n_reg, _R_ZERO, 0), "n=0")
        L("_spmm_n_loop")
        # B[k,n]
        self.emit.emit(rv_mul(tmp, k_reg, ncol_reg), "k*N")
        self.emit.emit(rv_add(tmp, tmp, n_reg), "+ n")
        self.emit.emit(rv_slli(tmp, tmp, 2), "*4")
        self.emit.emit(rv_add(tmp, base_b, tmp), "&B[k,n]")
        self.emit.emit(rv_lw(tmp, tmp, 0), "load B[k,n]")
        self.emit.emit(rv_mul(tmp, a_reg, tmp), "a*B (low32)")
        self.emit.emit(rv_srai(tmp, tmp, 16), ">>16")
        # C[i,n] += prod
        self.emit.emit(rv_mul(addr, i_reg, ncol_reg), "i*N")
        self.emit.emit(rv_add(addr, addr, n_reg), "+ n")
        self.emit.emit(rv_slli(addr, addr, 2), "*4")
        self.emit.emit(rv_add(addr, base_c, addr), "&C[i,n]")
        self.emit.emit(rv_lw(acc, addr, 0), "old C")
        self.emit.emit(rv_add(acc, acc, tmp), "+= prod")
        self.emit.emit(rv_sw(addr, acc, 0), "store C[i,n]")
        # n++
        self.emit.emit(rv_addi(n_reg, n_reg, 1), "n++")
        self.emit.emit(rv_slt(cond, n_reg, ncol_reg), "n < N?")
        self.emit.emit_branch(rv_bne, cond, _R_ZERO, "_spmm_n_loop", "loop")

        # j++
        self.emit.emit(rv_addi(j_reg, j_reg, 1), "j++")
        self.emit.emit_jump(rv_j, "_spmm_j_loop", "loop j")

        L("_spmm_j_done")
        # i++
        self.emit.emit(rv_addi(i_reg, i_reg, 1), "i++")
        self.emit.emit_jump(rv_j, "_spmm_i_loop", "loop i")

        L("_spmm_done")

    # ── Winograd F(2,3) convolution ─────────────────────────────────────

    def _gen_winogradconv(self, node: NodeInfo) -> None:
        """2D Winograd F(2,3) convolution (stride=1, SAME padding).

        Y_tile = A^T [ sum_ic (G g G^T) ⊙ (B^T d B) ] A
        Kernel transform U = G g G^T is folded at compile time (2.2) into the
        initializer named in ``attrs["_winograd_u"]``.
        """
        x_name = node.inputs[0]
        w_name = node.inputs[1]
        b_name = node.inputs[2] if len(node.inputs) > 2 and node.inputs[2] else None
        out_name = node.outputs[0]

        x_shape = self.model.get_shape(x_name)
        w_shape = self.model.get_shape(w_name)
        N = x_shape[0] if len(x_shape) > 0 else 1
        Cin = x_shape[1] if len(x_shape) > 1 else 1
        H = x_shape[2] if len(x_shape) > 2 else 1
        W = x_shape[3] if len(x_shape) > 3 else 1
        Cout = w_shape[0] if len(w_shape) > 0 else 1
        KH = w_shape[2] if len(w_shape) > 2 else 3

        attrs = node.attrs
        pads = list(attrs.get("pads", [1, 1, 1, 1]))[:4]
        if pads != [1, 1, 1, 1]:
            raise ValueError("WinogradConv requires pads=[1,1,1,1] (SAME, stride 1)")
        if list(attrs.get("strides", [1, 1]))[:2] != [1, 1]:
            raise ValueError("WinogradConv requires strides=[1,1]")
        if KH != 3:
            raise ValueError("WinogradConv supports 3x3 kernels only")
        u_name = attrs.get("_winograd_u")
        if not u_name:
            raise ValueError("WinogradConv missing precomputed kernel (2.2)")

        TH = (H + 1) // 2
        TW = (W + 1) // 2
        Hout, Wout = H, W

        d_name = f"{out_name}__d"
        v_name = f"{out_name}__v"
        acc_name = f"{out_name}__acc"
        for nm in (d_name, v_name, acc_name):
            self.mem.alloc_workspace(nm, 16)
        self.mem.alloc_workspace(out_name, N * Cout * Hout * Wout)

        base_in = self.S2
        base_u = self.S3
        base_out = self.S4
        base_d = self.S5
        base_v = self.S6
        base_acc = self.S7
        base_bias = self.S8

        n_reg = self.S9
        oc_reg = self.S10
        ti_reg = self.S11
        tj_reg = self.T4
        ic_reg = self.T5

        T0, T1, T2, T3, T6 = self.T0, self.T1, self.T2, self.T3, self.T6

        self._get_workspace_addr(x_name, base_in)
        self._get_weight_addr(u_name, base_u)
        self._get_workspace_addr(out_name, base_out)
        self._get_workspace_addr(d_name, base_d)
        self._get_workspace_addr(v_name, base_v)
        self._get_workspace_addr(acc_name, base_acc)
        if b_name is not None:
            self._get_weight_addr(b_name, base_bias)

        BT = _WINO_F23_BT
        Bm = [[BT[b][a] for b in range(4)] for a in range(4)]  # B[a][b] = BT[b][a]
        AT = _WINO_F23_AT
        A = [[AT[b][a] for b in range(2)] for a in range(4)]   # A[a][b] = AT[b][a]

        E = self.emit.emit
        Br = self.emit.emit_branch
        J = self.emit.emit_jump
        L = self.emit.label
        li = self.emit.emit_li

        # for n
        E(rv_addi(n_reg, _R_ZERO, 0), f"n=0 (N={N})")
        L("_wg_n_loop")
        E(rv_addi(oc_reg, _R_ZERO, 0), f"oc=0 (Cout={Cout})")
        L("_wg_oc_loop")
        E(rv_addi(ti_reg, _R_ZERO, 0), f"ti=0 (TH={TH})")
        L("_wg_ti_loop")
        E(rv_addi(tj_reg, _R_ZERO, 0), f"tj=0 (TW={TW})")
        L("_wg_tj_loop")

        # zero the 4x4 Winograd-domain accumulator
        for p in range(16):
            E(rv_sw(base_acc, _R_ZERO, 4 * p), "" if p else "acc[:] = 0")

        E(rv_addi(ic_reg, _R_ZERO, 0), f"ic=0 (Cin={Cin})")
        L("_wg_ic_loop")

        # ── load padded 4x4 input tile into dbuf ──────────────────────
        for r in range(4):
            for c in range(4):
                E(rv_slli(T0, ti_reg, 1), "")
                E(rv_addi(T0, T0, r - 1), "")                 # gi = 2*ti-1+r
                E(rv_slli(T1, tj_reg, 1), "")
                E(rv_addi(T1, T1, c - 1), "")                 # gj = 2*tj-1+c
                E(rv_addi(T2, _R_ZERO, 0), "")                # default 0
                skip = f"_wg_dskip_{r}_{c}"
                Br(rv_blt, T0, _R_ZERO, skip, "")             # gi < 0
                Br(rv_blt, T1, _R_ZERO, skip, "")             # gj < 0
                li(T6, H)
                E(rv_slt(T3, T0, T6), "")
                Br(rv_beq, T3, _R_ZERO, skip, "")             # gi >= H
                li(T6, W)
                E(rv_slt(T3, T1, T6), "")
                Br(rv_beq, T3, _R_ZERO, skip, "")             # gj >= W
                li(T3, Cin)
                E(rv_mul(T3, n_reg, T3), "")
                E(rv_add(T3, T3, ic_reg), "")
                li(T6, H)
                E(rv_mul(T3, T3, T6), "")
                E(rv_add(T3, T3, T0), "")
                li(T6, W)
                E(rv_mul(T3, T3, T6), "")
                E(rv_add(T3, T3, T1), "")
                E(rv_slli(T3, T3, 2), "")
                E(rv_add(T3, base_in, T3), "")
                E(rv_lw(T2, T3, 0), "")
                L(skip)
                E(rv_sw(base_d, T2, 4 * (r * 4 + c)), "")

        # ── input transform: vbuf = B^T dbuf B (signed sums of dbuf) ───
        for i in range(4):
            for j in range(4):
                E(rv_addi(T2, _R_ZERO, 0), "")
                for a in range(4):
                    for b in range(4):
                        coef = BT[i][a] * Bm[b][j]
                        if coef == 0:
                            continue
                        E(rv_lw(T0, base_d, 4 * (a * 4 + b)), "")
                        E(rv_add(T2, T2, T0) if coef > 0 else rv_sub(T2, T2, T0), "")
                E(rv_sw(base_v, T2, 4 * (i * 4 + j)), "")

        # ── Winograd-domain MAC: acc[p] += (U[oc,ic,p] * V[p]) >> 16 ───
        li(T0, Cin)
        E(rv_mul(T0, oc_reg, T0), "")
        E(rv_add(T0, T0, ic_reg), "")
        E(rv_slli(T0, T0, 6), "((oc*Cin+ic)*16)*4")
        E(rv_add(T0, base_u, T0), "&U[oc,ic,0]")
        for p in range(16):
            E(rv_lw(T1, T0, 4 * p), "")
            E(rv_lw(T2, base_v, 4 * p), "")
            # Full-precision Q16.16 multiply: (U*V) >> 16 using MULH+MUL.
            # Winograd intermediates (V, U) can exceed 2^15, so the low-32
            # MUL + SRAI used elsewhere would wrap; the 64-bit product >>16
            # stays in range and matches the ideal reference.
            E(rv_mul(T6, T1, T2), "mul lo")
            E(rv_mulh(T1, T1, T2), "mul hi")
            E(rv_srli(T6, T6, 16), "logical >>16 (low word can be negative)")
            E(rv_slli(T1, T1, 16), "")
            E(rv_or(T1, T1, T6), "(U*V)>>16")
            E(rv_lw(T6, base_acc, 4 * p), "")
            E(rv_add(T6, T6, T1), "")
            E(rv_sw(base_acc, T6, 4 * p), "")

        E(rv_addi(ic_reg, ic_reg, 1), "ic++")
        li(T6, Cin)
        E(rv_slt(T3, ic_reg, T6), "")
        Br(rv_bne, T3, _R_ZERO, "_wg_ic_loop", "loop ic")

        # ── output transform Y = A^T acc A, + bias, store in-bounds ────
        for oy in range(2):
            for ox in range(2):
                E(rv_addi(T2, _R_ZERO, 0), "")
                for a in range(4):
                    for b in range(4):
                        coef = AT[oy][a] * AT[ox][b]
                        if coef == 0:
                            continue
                        E(rv_lw(T0, base_acc, 4 * (a * 4 + b)), "")
                        E(rv_add(T2, T2, T0) if coef > 0 else rv_sub(T2, T2, T0), "")
                if b_name is not None:
                    E(rv_slli(T3, oc_reg, 2), "")
                    E(rv_add(T3, base_bias, T3), "")
                    E(rv_lw(T0, T3, 0), "bias[oc]")
                    E(rv_add(T2, T2, T0), "")
                # bounds: oh = 2*ti+oy < H, ow = 2*tj+ox < W
                oskip = f"_wg_oskip_{oy}_{ox}"
                E(rv_slli(T0, ti_reg, 1), "")
                E(rv_addi(T0, T0, oy), "oh")
                li(T6, H)
                E(rv_slt(T3, T0, T6), "")
                Br(rv_beq, T3, _R_ZERO, oskip, "")
                E(rv_slli(T1, tj_reg, 1), "")
                E(rv_addi(T1, T1, ox), "ow")
                li(T6, W)
                E(rv_slt(T3, T1, T6), "")
                Br(rv_beq, T3, _R_ZERO, oskip, "")
                li(T3, Cout)
                E(rv_mul(T3, n_reg, T3), "")
                E(rv_add(T3, T3, oc_reg), "")
                li(T6, H)
                E(rv_mul(T3, T3, T6), "")
                E(rv_add(T3, T3, T0), "")
                li(T6, W)
                E(rv_mul(T3, T3, T6), "")
                E(rv_add(T3, T3, T1), "")
                E(rv_slli(T3, T3, 2), "")
                E(rv_add(T3, base_out, T3), "")
                E(rv_sw(T3, T2, 0), "")
                L(oskip)

        E(rv_addi(tj_reg, tj_reg, 1), "tj++")
        li(T6, TW)
        E(rv_slt(T3, tj_reg, T6), "")
        Br(rv_bne, T3, _R_ZERO, "_wg_tj_loop", "loop tj")

        E(rv_addi(ti_reg, ti_reg, 1), "ti++")
        li(T6, TH)
        E(rv_slt(T3, ti_reg, T6), "")
        Br(rv_bne, T3, _R_ZERO, "_wg_ti_loop", "loop ti")

        E(rv_addi(oc_reg, oc_reg, 1), "oc++")
        li(T6, Cout)
        E(rv_slt(T3, oc_reg, T6), "")
        Br(rv_bne, T3, _R_ZERO, "_wg_oc_loop", "loop oc")

        E(rv_addi(n_reg, n_reg, 1), "n++")
        li(T6, N)
        E(rv_slt(T3, n_reg, T6), "")
        Br(rv_bne, T3, _R_ZERO, "_wg_n_loop", "loop n")

        L("_wg_done")

# ═══════════════════════════════════════════════════════════════════════════
# Part 8: Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════


def _emit_platform_fwht() -> str:
    """Size-independent FWHT kernel (platform ABI: a0=in, a1=out, a2=N)."""
    lines = [
        "# ScratchV platform kernel: FWHT (size-independent, a2 = N)",
        "    .option norelax",
        "    .text",
        "    .globl cnn_entry",
        "    .type cnn_entry, @function",
        "cnn_entry:",
        "    mv   t0, a0                 # src = in",
        "    mv   t1, a1                 # dst = out",
        "    mv   t2, a2                 # N",
        "    mv   t3, zero               # i",
        "_p_fwht_copy:",
        "    lw   t4, 0(t0)",
        "    sw   t4, 0(t1)",
        "    addi t0, t0, 4",
        "    addi t1, t1, 4",
        "    addi t3, t3, 1",
        "    blt  t3, t2, _p_fwht_copy",
        "    li   t5, 1                  # length = 1",
        "_p_fwht_len:",
        "    bge  t5, t2, _p_fwht_done",
        "    slli t6, t5, 1              # step = 2*length",
        "    li   s2, 0                  # i = block start",
        "_p_fwht_i:",
        "    bge  s2, t2, _p_fwht_next_len",
        "    li   s3, 0                  # j = 0",
        "_p_fwht_j:",
        "    bge  s3, t5, _p_fwht_next_block",
        "    add  s4, s2, s3             # idx = i + j",
        "    slli s5, s4, 2",
        "    add  s6, a1, s5             # &a[idx]",
        "    add  s7, s4, t5             # idx + length",
        "    slli s7, s7, 2",
        "    add  s8, a1, s7             # &a[idx+length]",
        "    lw   s9, 0(s6)              # u",
        "    lw   s10, 0(s8)             # v",
        "    add  s11, s9, s10           # u+v",
        "    sw   s11, 0(s6)",
        "    sub  s11, s9, s10           # u-v",
        "    sw   s11, 0(s8)",
        "    addi s3, s3, 1",
        "    j    _p_fwht_j",
        "_p_fwht_next_block:",
        "    add  s2, s2, t6",
        "    j    _p_fwht_i",
        "_p_fwht_next_len:",
        "    slli t5, t5, 1",
        "    j    _p_fwht_len",
        "_p_fwht_done:",
        "    ret",
        "    .size cnn_entry, . - cnn_entry",
    ]
    return "\n".join(lines) + "\n"


def _emit_platform_conv() -> str:
    """Size-independent direct Conv2D kernel (platform ABI).

    a0 = input (NCHW), a1 = output (NCHW), a2 = param block
    (N,C_in,H,W,C_out,K,pad,stride,H_out,W_out,w_ptr,b_ptr).
    Semantics per docs/输入规范.md section 3.5.
    """
    lines = [
        "# ScratchV platform kernel: direct Conv2D (size-independent)",
        "    .option norelax",
        "    .text",
        "    .globl cnn_entry",
        "    .type cnn_entry, @function",
        "cnn_entry:",
        "    lw   s0, 0(a2)              # N",
        "    lw   s1, 4(a2)              # C_in",
        "    lw   s2, 8(a2)              # H",
        "    lw   s3, 12(a2)             # W",
        "    lw   s4, 16(a2)             # C_out",
        "    lw   s5, 20(a2)             # K",
        "    lw   s6, 24(a2)             # pad",
        "    lw   s7, 32(a2)             # H_out",
        "    lw   s8, 36(a2)             # W_out",
        "    lw   s9, 40(a2)             # w_ptr",
        "    lw   s10, 44(a2)            # b_ptr",
        "    mv   s11, a0                # in base",
        "    mv   a7, a1                 # out base",
        "    li   t0, 0                  # n = 0",
        "_p_conv_n:",
        "    bge  t0, s0, _p_conv_done",
        "    li   t1, 0                  # oc = 0",
        "_p_conv_oc:",
        "    bge  t1, s4, _p_conv_next_n",
        "    li   t2, 0                  # oh = 0",
        "_p_conv_oh:",
        "    bge  t2, s7, _p_conv_next_oc",
        "    li   t3, 0                  # ow = 0",
        "_p_conv_ow:",
        "    bge  t3, s8, _p_conv_next_oh",
        "    li   t6, 0                  # acc = 0",
        "    beq  s10, zero, _p_conv_nobias",
        "    slli a3, t1, 2",
        "    add  a3, s10, a3",
        "    lw   t6, 0(a3)              # acc = bias[oc]",
        "_p_conv_nobias:",
        "    li   t4, 0                  # ic = 0",
        "_p_conv_ic:",
        "    bge  t4, s1, _p_conv_store",
        "    li   t5, 0                  # kh = 0",
        "_p_conv_kh:",
        "    bge  t5, s5, _p_conv_next_ic",
        "    li   a6, 0                  # kw = 0",
        "_p_conv_kw:",
        "    bge  a6, s5, _p_conv_next_kh",
        "    add  a3, t2, t5",
        "    sub  a3, a3, s6             # ih",
        "    add  a4, t3, a6",
        "    sub  a4, a4, s6             # iw",
        "    blt  a3, zero, _p_conv_kw_skip",
        "    bge  a3, s2, _p_conv_kw_skip",
        "    blt  a4, zero, _p_conv_kw_skip",
        "    bge  a4, s3, _p_conv_kw_skip",
        "    mul  a5, t0, s1",
        "    add  a5, a5, t4",
        "    mul  a5, a5, s2",
        "    add  a5, a5, a3",
        "    mul  a5, a5, s3",
        "    add  a5, a5, a4",
        "    slli a5, a5, 2",
        "    add  a5, s11, a5",
        "    lw   a5, 0(a5)              # in value",
        "    mul  a4, t1, s1",
        "    add  a4, a4, t4",
        "    mul  a4, a4, s5",
        "    add  a4, a4, t5",
        "    mul  a4, a4, s5",
        "    add  a4, a4, a6",
        "    slli a4, a4, 2",
        "    add  a4, s9, a4",
        "    lw   a4, 0(a4)              # weight value",
        "    mul  a5, a5, a4",
        "    srai a5, a5, 16",
        "    add  t6, t6, a5             # acc += prod",
        "_p_conv_kw_skip:",
        "    addi a6, a6, 1",
        "    j    _p_conv_kw",
        "_p_conv_next_kh:",
        "    addi t5, t5, 1",
        "    j    _p_conv_kh",
        "_p_conv_next_ic:",
        "    addi t4, t4, 1",
        "    j    _p_conv_ic",
        "_p_conv_store:",
        "    mul  a3, t0, s4",
        "    add  a3, a3, t1",
        "    mul  a3, a3, s7",
        "    add  a3, a3, t2",
        "    mul  a3, a3, s8",
        "    add  a3, a3, t3",
        "    slli a3, a3, 2",
        "    add  a3, a7, a3",
        "    sw   t6, 0(a3)",
        "    addi t3, t3, 1",
        "    j    _p_conv_ow",
        "_p_conv_next_oh:",
        "    addi t2, t2, 1",
        "    j    _p_conv_oh",
        "_p_conv_next_oc:",
        "    addi t1, t1, 1",
        "    j    _p_conv_oc",
        "_p_conv_next_n:",
        "    addi t0, t0, 1",
        "    j    _p_conv_n",
        "_p_conv_done:",
        "    ret",
        "    .size cnn_entry, . - cnn_entry",
    ]
    return "\n".join(lines) + "\n"


def _emit_platform_spmm() -> str:
    """Size-independent CSR SpMM kernel (platform ABI).

    a0 = CSR block (values_ptr, col_ptr, row_ptr), a1 = out (M*N), a2 = params
    (M, K, N, nnz, B_ptr). Semantics per docs/输入规范.md section 4.5.
    """
    lines = [
        "# ScratchV platform kernel: CSR SpMM (size-independent)",
        "    .option norelax",
        "    .text",
        "    .globl cnn_entry",
        "    .type cnn_entry, @function",
        "cnn_entry:",
        "    lw   t2, 0(a0)              # values_ptr",
        "    lw   t3, 4(a0)              # col_ptr",
        "    lw   t4, 8(a0)              # row_ptr",
        "    lw   t5, 0(a2)              # M",
        "    lw   t6, 8(a2)              # N",
        "    lw   s2, 16(a2)             # B_ptr",
        "    mv   a0, t2                 # values",
        "    mv   a2, t3                 # col",
        "    mv   a3, t4                 # row_ptr",
        "    mv   a4, t5                 # M",
        "    mv   a5, t6                 # N",
        "    mv   a6, s2                 # B",
        "    li   s3, 0                  # i = 0",
        "_p_spmm_i:",
        "    bge  s3, a4, _p_spmm_done",
        "    slli t0, s3, 2",
        "    add  t0, a3, t0             # &row_ptr[i]",
        "    lw   s5, 0(t0)              # j = row_ptr[i]",
        "    lw   s6, 4(t0)              # j_end = row_ptr[i+1]",
        "    li   s4, 0                  # n = 0",
        "_p_spmm_zero:",
        "    mul  t0, s3, a5",
        "    add  t0, t0, s4",
        "    slli t0, t0, 2",
        "    add  t0, a1, t0",
        "    sw   zero, 0(t0)",
        "    addi s4, s4, 1",
        "    blt  s4, a5, _p_spmm_zero",
        "_p_spmm_j:",
        "    bge  s5, s6, _p_spmm_next_i",
        "    slli t1, s5, 2",
        "    add  t1, a0, t1",
        "    lw   t1, 0(t1)              # a = values[j]",
        "    slli t0, s5, 2",
        "    add  t0, a2, t0",
        "    lw   t0, 0(t0)              # k = col[j]",
        "    li   s4, 0                  # n = 0",
        "_p_spmm_n:",
        "    mul  t2, t0, a5",
        "    add  t2, t2, s4",
        "    slli t2, t2, 2",
        "    add  t2, a6, t2",
        "    lw   t2, 0(t2)              # B[k,n]",
        "    mul  t2, t1, t2",
        "    srai t2, t2, 16",
        "    mul  t3, s3, a5",
        "    add  t3, t3, s4",
        "    slli t3, t3, 2",
        "    add  t3, a1, t3",
        "    lw   t4, 0(t3)",
        "    add  t4, t4, t2",
        "    sw   t4, 0(t3)",
        "    addi s4, s4, 1",
        "    blt  s4, a5, _p_spmm_n",
        "    addi s5, s5, 1",
        "    j    _p_spmm_j",
        "_p_spmm_next_i:",
        "    addi s3, s3, 1",
        "    j    _p_spmm_i",
        "_p_spmm_done:",
        "    ret",
        "    .size cnn_entry, . - cnn_entry",
    ]
    return "\n".join(lines) + "\n"


def emit_platform_asm(generator, weight_data: bytes, model=None) -> str:
    """Render a platform-standard, clang-assemblable listing (``--platform-asm``).

    If *model* is a single size-independent-supported problem node (currently
    FWHT), emit the dedicated runtime-sized kernel; otherwise emit the
    fixed-shape standard listing (B-1).
    """
    if model is not None and len(model.nodes) == 1 and model.nodes[0].op_type == "Fwht":
        return _emit_platform_fwht()
    if model is not None and len(model.nodes) == 1 and model.nodes[0].op_type == "SpmmCsr":
        return _emit_platform_spmm()
    if model is not None and len(model.nodes) == 1 and model.nodes[0].op_type == "Conv":
        return _emit_platform_conv()
    return _emit_platform_listing(generator, weight_data)


def _emit_platform_listing(generator, weight_data: bytes) -> str:
    """Fixed-shape platform-standard listing (B-1).

    Differences from the default debug listing:
      - global entry symbol ``cnn_entry`` (not ``_start``);
      - branch/jump targets are real labels (not numeric offsets);
      - the ``_init_data_base`` placeholder pair becomes ``la gp, __data_start``;
      - weights/constants are emitted inline as ``.word`` in a ``.data`` section
        (no ``.incbin`` / ``.include``), preceded by ``__data_start``.

    Platform contract: a0 = input base, a1 = output base, a2 = size scalar;
    sp is already set by the caller.
    """
    import re as _re

    emit = generator.emit
    dis = emit.disassemble(symbolic=True).splitlines()

    # Prefer the emitter's own label names; else a generated .Lpc_<idx>.
    idx_to_name: dict[int, str] = {}
    for name, idx in emit.labels.items():
        if idx not in idx_to_name:
            idx_to_name[idx] = name

    def label_for(idx: int) -> str:
        if idx == 0:
            return "cnn_entry"
        return idx_to_name.get(idx, f".Lpc_{idx}")

    out: list[str] = []
    for line in dis:
        stripped = line.strip()
        if stripped.endswith(":") and stripped[:-1].startswith(".Lsv_"):
            idx = int(stripped[len(".Lsv_"):-1])
            out.append(f"{label_for(idx)}:")
        else:
            out.append(_re.sub(r"\.Lsv_(\d+)", lambda m: label_for(int(m.group(1))), line))

    # Replace the auipc/addi placeholder pair with a standard la.
    fixed: list[str] = []
    skip_next = False
    for line in out:
        if skip_next:
            skip_next = False
            continue
        if "auipc gp, 0x0" in line:
            indent = line[: len(line) - len(line.lstrip())]
            fixed.append(f"{indent}la gp, __data_start      # data section base")
            skip_next = True  # drop the following addi/mv placeholder
            continue
        fixed.append(line)

    header = [
        "# Generated by ScratchV standalone --platform-asm",
        "# Platform contract: a0=input base, a1=output base, a2=size; sp preset.",
        "# Assemble: clang -march=rv32im -mabi=ilp32 -nostdlib -c this.s",
        "",
        "    .option norelax",
        "    .text",
        "    .globl cnn_entry",
        "    .type cnn_entry, @function",
    ]
    tail = ["    .size cnn_entry, . - cnn_entry", ""]

    data_lines = ["", "    .data", "    .balign 4", "__data_start:"]
    if weight_data:
        words = struct.unpack(f"<{len(weight_data) // 4}I", weight_data)
        for i in range(0, len(words), 8):
            chunk = ", ".join(f"0x{w:08x}" for w in words[i:i + 8])
            data_lines.append(f"    .word {chunk}")
    else:
        data_lines.append("    .word 0")

    return "\n".join(header + fixed + tail + data_lines) + "\n"


def patch_gp_data_base(
    generator: CNNRISCVGenerator,
    code_bytes: bytes,
    *,
    sync_listing: bool = False,
    strict: bool = False,
) -> tuple[bytes, dict[str, object]]:
    """Patch GP using the final code size and optionally sync the listing."""
    if len(code_bytes) % 4:
        code_bytes += b"\x00" * (4 - len(code_bytes) % 4)

    data_offset = len(code_bytes)
    candidate = getattr(generator, "emit", None)
    emitter = candidate if hasattr(candidate, "labels") else generator
    init_label_idx = emitter.labels.get("_init_data_base", -1)
    if init_label_idx < 0:
        if strict:
            raise ValueError("missing _init_data_base label")
        return code_bytes, {
            "applied": False,
            "code_size": data_offset,
            "data_offset": data_offset,
            "validation": True,
        }

    auipc_word_idx = init_label_idx
    addi_word_idx = init_label_idx + 1
    code_word_list = list(struct.unpack(f"<{len(code_bytes) // 4}I", code_bytes))
    if addi_word_idx >= len(code_word_list):
        raise ValueError("_init_data_base pair is incomplete")

    auipc_pc = auipc_word_idx * 4
    delta = data_offset - auipc_pc
    upper = (delta + 0x800) >> 12
    lower = _sext(delta & 0xFFF, 12)
    code_word_list[auipc_word_idx] = rv_auipc(_R_GP, upper & 0xFFFFF)
    code_word_list[addi_word_idx] = rv_addi(_R_GP, _R_GP, lower & 0xFFF)
    patched = struct.pack(f"<{len(code_word_list)}I", *code_word_list)

    if sync_listing:
        emitter.code = code_word_list
    validation = code_word_list[auipc_word_idx] == rv_auipc(
        _R_GP, upper & 0xFFFFF
    ) and code_word_list[addi_word_idx] == rv_addi(_R_GP, _R_GP, lower & 0xFFF)
    if strict and not validation:
        raise ValueError("GP patch validation failed")
    return patched, {
        "applied": True,
        "code_size": data_offset,
        "data_offset": data_offset,
        "auipc_index": auipc_word_idx,
        "addi_index": addi_word_idx,
        "delta": delta,
        "validation": validation,
    }


def _run_tinyfive_sim(
    output_bin: str, code_bytes: bytes, memory: MemoryPlan,
    model: ONNXModel, data_offset: int, max_instr: int,
) -> None:
    """Run the compiled binary through TinyFive's RV32IM simulator.

    Loads the binary into TinyFive's ProfiledMachine, sets up the
    bare-metal ABI registers, generates random Q16.16 input, and
    executes for *max_instr* cycles.  Reports TinyFive's built-in
    performance counters (total, load, store, mul, add, branch).
    """
    import random
    import struct

    print("\n[tinyfive] Running TinyFive RV32IM simulation...")
    try:
        from scratchv.simulator.tinyfive import ProfiledMachine
    except ImportError:
        print("  ERROR: scratchv.simulator.tinyfive not found", file=sys.stderr)
        return

    m = ProfiledMachine(mem_size=256 * 1024 * 1024)  # 256 MB
    if not m.available:
        print("  ERROR: TinyFive not installed. Run: pip install tinyfive numpy",
              file=sys.stderr)
        return

    # ── Load binary ──────────────────────────────────────────────────
    code_words = list(struct.unpack(f"<{len(code_bytes)//4}I", code_bytes))
    m.load_binary(code_words, origin=0)

    # Load weight data at data_offset
    with open(output_bin, "rb") as f:
        full_binary = f.read()
    data_bytes = full_binary[len(code_bytes):]
    m.load_data(data_bytes, data_offset)
    print(f"  Code: {len(code_words)} words at 0x0")
    print(f"  Data: {len(data_bytes):,} bytes at 0x{data_offset:x}")

    # ── Set up bare-metal ABI registers ──────────────────────────────
    m.set_reg(2, 128 * 1024 * 1024)   # sp = 128 MB
    m.set_reg(3, data_offset)          # gp = data section base

    # Input buffer: place at 160 MB
    INPUT_ADDR = 160 * 1024 * 1024
    m.set_reg(10, INPUT_ADDR)          # a0 = input ptr

    # Output buffer: place at 192 MB
    OUTPUT_ADDR = 192 * 1024 * 1024
    m.set_reg(11, OUTPUT_ADDR)         # a1 = output ptr

    # Generate random Q16.16 input
    input_shape = model.get_shape(model.inputs[0].name) if model.inputs else ()
    input_el = 1
    for d in input_shape:
        input_el *= d
    random.seed(42)
    for i in range(input_el):
        val = int((random.random() - 0.5) * 0.2 * 65536)  # ±0.1 in Q16.16
        m.write_mem_i32(INPUT_ADDR + i * 4, val)

    print(f"  Input:  {input_el:,} Q16.16 elements at 0x{INPUT_ADDR:x}")
    print(f"  Output: buffer at 0x{OUTPUT_ADDR:x}")
    print(f"  sp=0x{128*1024*1024:x}  gp=0x{data_offset:x}")

    # ── Run simulation ───────────────────────────────────────────────
    print(f"  Running (max {max_instr:,} instructions)...")
    import time
    t0 = time.perf_counter()
    m.run(instructions=max_instr)
    elapsed = time.perf_counter() - t0

    # ── Report ───────────────────────────────────────────────────────
    perf = m.get_perf()
    mips = perf["total"] / elapsed / 1_000_000 if elapsed > 0 else 0
    print("\n  ── TinyFive Simulation Results ──")
    print(f"  Instructions executed: {perf['total']:,}")
    print(f"  Wall time:            {elapsed:.2f}s")
    print(f"  Simulated MIPS:       {mips:.1f}")
    print("")
    print("  ── Performance Counters (TinyFive built-in) ──")
    print(f"  {'Counter':<12s} {'Count':>15s} {'%':>8s}")
    print(f"  {'─'*12} {'─'*15} {'─'*8}")
    total = max(perf['total'], 1)
    for key, label in [('load', 'Load'), ('store', 'Store'), ('mul', 'Mul'),
                        ('add', 'Add/ALU'), ('madd', 'Mul-Add'),
                        ('branch', 'Branch')]:
        val = perf.get(key, 0)
        print(f"  {label:<12s} {val:>15,} {val/total*100:>7.1f}%")
    print(f"  {'─'*12} {'─'*15} {'─'*8}")
    print(f"  {'TOTAL':<12s} {total:>15,} {100.0:>7.1f}%")

    # ── Output value ─────────────────────────────────────────────────
    out_val = m.read_mem_i32(OUTPUT_ADDR)
    out_float = out_val / 65536.0
    print(f"\n  Output Q16.16: {out_val}  (≈ {out_float:.6f} float)")

    # ── Register state at end ────────────────────────────────────────
    print("\n  ── Final Register State ──")
    for name, idx in [('ra', 1), ('sp', 2), ('gp', 3), ('a0', 10), ('a1', 11),
                       ('t0', 5), ('t1', 6), ('t2', 7)]:
        print(f"  {name:>4s} (x{idx:2d}): 0x{m.get_reg(idx):08x}")


def convert_onnx_to_riscv(
    onnx_path: str, output_bin: str, output_asm: str = "",
    benchmark: bool = False, max_instr: int = 2_000_000_000,
    estimate: bool = False, report: bool = False,
    uarch: str = "basic",
    tinyfive_sim: bool = False, tinyfive_max_instr: int = 100_000_000,
    const_merge: bool = False,
    schedule: bool = False,
    metadata: dict | None = None,
    llvm_mca: str | None = None,
    symbolic_asm: bool = False,
    platform_asm: bool = False,
) -> int:
    """Full pipeline: ONNX model → RISC-V RV32IM binary.

    Args:
        onnx_path: Path to .onnx model file.
        output_bin: Path for output RISC-V binary.
        output_asm: Path for output assembly listing.
        benchmark: If True, run the generated binary through the RV32IM
            emulator and print detailed performance metrics.
        max_instr: Max instructions for benchmark emulation (avoids
            infinite loops).

    Returns: 0 on success, non-zero on error.
    """
    print("ScratchV: Library-free ONNX → RISC-V RV32IM Pipeline")
    print(f"{'='*60}")
    print(f"  Input:  {onnx_path}")

    # ── Step 1: Parse ONNX protobuf ────────────────────────────────────
    print("\n[1/5] Parsing ONNX protobuf (manual wire-format parser)...")
    model = ONNXModel.from_file(onnx_path)
    print(f"  Graph: {model.graph_name}")
    print(f"  Nodes: {len(model.nodes)}")
    print(f"  Initializers: {len(model.initializers)}")
    for i, node in enumerate(model.nodes):
        print(f"    [{i}] {node.op_type}: "
              f"{', '.join(node.inputs[:2])} → {', '.join(node.outputs[:1])}")

    # ── Step 2: Layout weights and plan memory ─────────────────────────
    print("\n[2/5] Converting weights to Q16.16 fixed-point and planning memory...")
    _prepare_winograd_kernels(model)   # fold WinogradConv 3x3 kernels (if any)
    memory = MemoryPlan()
    weight_data = memory.layout_weights(model.initializers)
    print(f"  Weight data: {memory.data_size:,} bytes ({memory.data_size/1024/1024:.1f} MB)")

    # Allocate workspace for input tensor
    if model.inputs:
        input_name = model.inputs[0].name
        input_shape = model.get_shape(input_name)
        input_el = 1
        for d in input_shape:
            input_el *= d
        memory.alloc_workspace(input_name, input_el)
        print(f"  Input '{input_name}': shape={input_shape}, {input_el:,} elements")

    # ── Step 3: Generate RISC-V code ───────────────────────────────────
    print("\n[3/5] Generating inline RISC-V RV32IM machine code...")
    constant_merge_report = None
    if const_merge:
        # Generate both variants from identical pre-codegen memory plans.  The
        # generator allocates layer workspaces, so sharing one plan would make
        # the second run observe mutated offsets and invalidate the A/B result.
        baseline_memory = copy.deepcopy(memory)
        baseline_generator = CNNRISCVGenerator(model, baseline_memory)
        baseline_code_bytes = baseline_generator.generate()

        optimized_memory = copy.deepcopy(memory)
        generator = CNNRISCVGenerator(
            model, optimized_memory, compact_constants=True,
        )
        code_bytes = generator.generate()
        memory = optimized_memory

        if baseline_memory.workspace_offsets != memory.workspace_offsets:
            raise AssertionError("constant-merge A/B changed workspace layout")

        # Run the public assembly pass over the exact baseline listing.  Its
        # categorized statistics describe the source transformation, while
        # the two generated binaries provide the real machine-code metrics.
        from scratchv.backend.const_merge import merge_constants_detailed

        asm_before = baseline_generator.emit.disassemble()
        asm_after, merge_stats = merge_constants_detailed(asm_before)

        def count_listing_instructions(asm_text: str) -> int:
            count = 0
            for raw_line in asm_text.splitlines():
                code = raw_line.split("#", 1)[0].strip()
                if not code or code.endswith(":") or code.startswith("."):
                    continue
                count += 1
            return count

        source_before = count_listing_instructions(asm_before)
        source_after = count_listing_instructions(asm_after)
        if len(code_bytes) > len(baseline_code_bytes):
            raise AssertionError("constant merge unexpectedly increased code size")

        constant_merge_report = {
            "enabled": True,
            "source_transform_path": "backend.const_merge public assembly pass",
            "machine_codegen_path": "RISCVEmitter(compact_li32=True)",
            "machine_metrics_are_public_pass_output": False,
            "used": merge_stats.total_changes > 0,
            "candidate_pairs": merge_stats.candidate_pairs,
            "merged_pairs": merge_stats.merged_pairs,
            "redundant_lui_removed": merge_stats.redundant_lui_removed,
            "iterations": merge_stats.iterations,
            "source_instructions_before": source_before,
            "source_instructions_after": source_after,
            "source_instruction_reduction": source_before - source_after,
            "machine_instructions_before": len(baseline_code_bytes) // 4,
            "machine_instructions_after": len(code_bytes) // 4,
            "machine_instruction_reduction": (
                len(baseline_code_bytes) - len(code_bytes)
            ) // 4,
            "code_size_before": len(baseline_code_bytes),
            "code_size_after": len(code_bytes),
            "code_size_reduction": len(baseline_code_bytes) - len(code_bytes),
        }
    else:
        generator = CNNRISCVGenerator(model, memory)
        code_bytes = generator.generate()

    print(f"  Code size: {len(code_bytes):,} bytes ({len(code_bytes)//4} instructions)")
    if constant_merge_report is not None:
        cm = constant_merge_report
        print(
            "  Constant merge: "
            f"{cm['code_size_before']:,} → {cm['code_size_after']:,} bytes; "
            f"{cm['machine_instructions_before']} → "
            f"{cm['machine_instructions_after']} machine instructions"
        )
    print(f"  Workspace: {memory.workspace_size:,} bytes "
          f"({memory.workspace_size/1024/1024:.1f} MB)")

    # ── Step 4: Assemble binary ────────────────────────────────────────
    print("\n[4/5] Assembling flat binary...")
    # The binary layout:
    #   [code_bytes][weight_data_bytes]
    # Code is at offset 0, data immediately follows (4-byte aligned).

    # Ensure 4-byte alignment between code and data
    code_len = len(code_bytes)
    if code_len % 4 != 0:
        pad = 4 - (code_len % 4)
        code_bytes += b"\x00" * pad
        code_len = len(code_bytes)

    # Compute the data offset (byte distance from code start)
    data_offset = code_len

    # Patch the AUIPC+ADDI sequence at _init_data_base to point to data
    # The AUIPC sets rd = pc + (imm << 12)
    # At _init_data_base label, pc = label_instruction_address
    # We want gp = code_start + data_offset
    # gp = auipc_result + addi_immediate
    # auipc_result = pc_of_auipc + (upper_imm << 12)
    # So: code_start + data_offset = pc_auipc + (upper_imm << 12) + addi_imm
    # upper_imm = (data_offset - pc_auipc - addi_imm) >> 12

    # Find the _init_data_base label index
    init_label_idx = generator.emit.labels.get("_init_data_base", -1)
    if init_label_idx >= 0:
        auipc_pc = init_label_idx * 4  # byte address of the AUIPC instruction
        auipc_word_idx = init_label_idx
        addi_word_idx = init_label_idx + 1

        # The GP should point to code_start + data_offset
        # In a position-independent binary, code_start is not known at link time.
        # We set GP = AUIPC(PC) + 0 for the AUIPC, and ADDI gp, gp, data_offset
        # Since the AUIPC gives PC, and PC = code_start + auipc_pc,
        # we need: gp = code_start + data_offset = PC + (data_offset - auipc_pc)
        delta = data_offset - auipc_pc

        # AUIPC: set upper 20 bits
        upper = (delta + 0x800) >> 12
        # ADDI: add lower 12 bits (signed)
        lower = _sext(delta & 0xFFF, 12)

        # Patch the instructions
        code_word_list = list(struct.unpack(f"<{len(code_bytes)//4}I", code_bytes))
        code_word_list[auipc_word_idx] = rv_auipc(_R_GP, upper & 0xFFFFF)
        code_word_list[addi_word_idx] = rv_addi(_R_GP, _R_GP, lower & 0xFFF)
        code_bytes = struct.pack(f"<{len(code_word_list)}I", *code_word_list)
        if schedule or symbolic_asm:
            generator.emit.code = code_word_list

    if schedule:
        scheduling = generator.emit.schedule(llvm_mca=llvm_mca)
        code_bytes = generator.emit.to_bytes()
        print(scheduling.report())

    binary = code_bytes + weight_data
    print(f"  Total binary: {len(binary):,} bytes ({len(binary)/1024/1024:.1f} MB)")
    print("  Code offset: 0x00000000")
    print(f"  Data offset: 0x{data_offset:08x} ({data_offset:,} bytes)")

    # ── Step 5: Write output ───────────────────────────────────────────
    print("\n[5/5] Writing output files...")
    with open(output_bin, "wb") as f:
        f.write(binary)
    print(f"  Binary: {output_bin} ({len(binary):,} bytes)")

    # Disassembly for verification
    if platform_asm:
        asm_text = emit_platform_asm(generator, weight_data, model)
    else:
        asm_text = generator.emit.disassemble(symbolic=schedule or symbolic_asm)
    asm_path = output_asm or output_bin.replace(".bin", ".s")
    with open(asm_path, "w") as f:
        f.write(asm_text)
    print(f"  Assembly: {asm_path}")
    if metadata is not None:
        output_elements = 1 if model.outputs else 0
        if model.outputs:
            for dimension in model.get_shape(model.outputs[0].name):
                output_elements *= dimension
        metadata.update(code_bytes=len(code_bytes), workspace_bytes=memory.workspace_size,
                        input_elements=input_el if model.inputs else 0,
                        output_elements=output_elements)

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    print(f"  Code:   {len(code_bytes):,} bytes ({len(code_bytes)//4} instructions)")
    print(f"  Data:   {memory.data_size:,} bytes ({memory.data_size/1024/1024:.1f} MB)")
    print(f"  Total:  {len(binary):,} bytes ({len(binary)/1024/1024:.1f} MB)")
    print(f"  Input:  {input_name if model.inputs else 'unknown'}")
    print(f"          shape={input_shape if model.inputs else 'unknown'}")
    print(f"  Output: {model.outputs[0].name if model.outputs else 'unknown'}")
    print(f"  Workspace needed: {memory.workspace_size/1024/1024:.1f} MB")
    print("\n  Bare-metal ABI:")
    print("    a0 → input tensor (float32, Q16.16 converted)")
    print("    a1 → output buffer")
    print("    gp → data section base (set by binary on entry)")
    print("    sp → stack pointer (caller must initialize)")
    print("    returns via jalr zero, ra, 0")
    print(f"{'='*60}")

    # ── Optional: Analytical estimation ──────────────────────────────────
    if estimate or report:
        print("\n[estimate] Analytical instruction count estimation...")
        try:
            from scratchv.standalone.benchmark import estimate_cnn_model, print_estimate
            est = estimate_cnn_model()
            print_estimate(est)
        except ImportError:
            print("  ERROR: benchmark module not found", file=sys.stderr)

    # ── Optional: Generate CI reports (HTML, JSON, GitHub summary) ───────
    if report:
        print("\n[report] Generating CI benchmark reports...")
        try:
            from scratchv.standalone.bench_report import (
                generate_github_summary,
                generate_html_report,
                generate_json_report,
            )
            from scratchv.standalone.benchmark import estimate_cnn_model
            est_data = estimate_cnn_model()
            code_len = len(code_bytes)

            os.makedirs("benchmark_reports", exist_ok=True)
            model_name = os.path.basename(onnx_path)

            # HTML report
            with open("benchmark_reports/benchmark.html", "w") as f:
                f.write(generate_html_report(
                    code_size=code_len,
                    static_insns=code_len // 4,
                    est_data=est_data,
                    model_name=model_name,
                    optimization=constant_merge_report,
                ))
            print("  HTML: benchmark_reports/benchmark.html")

            # JSON report
            with open("benchmark_reports/benchmark.json", "w") as f:
                f.write(generate_json_report(
                    code_size=code_len,
                    static_insns=code_len // 4,
                    est_data=est_data,
                    model_name=model_name,
                    optimization=constant_merge_report,
                ))
            print("  JSON: benchmark_reports/benchmark.json")

            # GitHub Actions job summary
            with open("benchmark_reports/github_summary.md", "w") as f:
                f.write(generate_github_summary(
                    code_size=code_len,
                    static_insns=code_len // 4,
                    est_data=est_data,
                    optimization=constant_merge_report,
                ))
            print("  Summary: benchmark_reports/github_summary.md")
        except ImportError as e:
            print(f"  ERROR: {e}", file=sys.stderr)

    # ── Optional: Benchmark ───────────────────────────────────────────────
    if benchmark:
        print("\n[benchmark] Running RISC-V emulation with performance counters...")
        code_size = len(code_bytes)

        # Build label address map from emitter (for per-operator stats)
        label_addrs = {}
        for name, idx in generator.emit.labels.items():
            label_addrs[idx * 4] = name  # PC = instruction_index * 4

        # Generate random Q16.16 input data matching input shape
        input_shape = model.get_shape(model.inputs[0].name) if model.inputs else ()
        input_el = 1
        for d in input_shape:
            input_el *= d
        # Generate random input in Q16.16 (small values, ±0.1 range)
        import random
        random.seed(42)
        input_q16 = []
        for _ in range(input_el):
            val = int((random.random() - 0.5) * 0.2 * 65536)  # ±0.1 in Q16.16
            input_q16.append(val)
        input_bytes = struct.pack(f"<{input_el}i", *input_q16)

        try:
            from scratchv.standalone.benchmark import (
                PROFILES,
                format_benchmark_report,
                run_benchmark,
            )
            uarch_obj = PROFILES.get(uarch, PROFILES["basic"])
            perf = run_benchmark(
                binary_path=output_bin,
                code_size=code_size,
                input_data=input_bytes,
                max_instr=max_instr,
                label_addrs=label_addrs,
                uarch=uarch_obj,
                verbose=True,
            )
            report = format_benchmark_report(perf, output_bin, code_size)
            print(report)
        except ImportError:
            print("  ERROR: benchmark module not found", file=sys.stderr)
        except Exception as e:
            print(f"  Benchmark failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    # ── Optional: TinyFive simulation ────────────────────────────────────
    if tinyfive_sim:
        _run_tinyfive_sim(output_bin, code_bytes, memory,
                          model, data_offset, tinyfive_max_instr)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ScratchV: Library-free ONNX → RISC-V RV32IM binary compiler"
    )
    parser.add_argument("model", help="Path to ONNX model file (.onnx)")
    parser.add_argument(
        "-o", "--output", default="output.bin",
        help="Output binary file path (default: output.bin)"
    )
    parser.add_argument(
        "--asm", default="",
        help="Output assembly file path (default: <output>.s)"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run generated binary through RV32IM emulator and print "
             "detailed performance metrics (instruction mix, C/M ratio, "
             "branch stats, per-layer breakdown, MIPS)"
    )
    parser.add_argument(
        "--estimate", action="store_true",
        help="Print analytical instruction count estimation (instant, "
             "no emulation needed)"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate CI benchmark reports (HTML, JSON, GitHub summary) "
             "in benchmark_reports/ directory"
    )
    parser.add_argument(
        "--const-merge", action="store_true",
        help="Enable constant-load merging and report real baseline/optimized "
             "machine-code size differences"
    )
    parser.add_argument("--schedule", action="store_true",
                        help="Schedule physical-register instructions before writing the binary")
    parser.add_argument("--symbolic-asm", action="store_true",
                        help="Emit reassemblable symbolic targets for scheduling analysis")
    parser.add_argument("--platform-asm", action="store_true",
                        help="Emit platform-standard assembly: global entry cnn_entry, "
                             "label branch targets, inline .word weights, la gp (clang-"
                             "assemblable, -march=rv32im -nostdlib)")
    parser.add_argument(
        "--uarch", default="basic", choices=["single", "fast", "basic", "slow"],
        help="Microarchitecture profile for cycle-accurate emulation: "
             "single (CPI=1), fast (mul=1 div=4), basic (mul=4 div=34), "
             "slow (mul=32 div=34 lw=5) (default: basic)"
    )
    parser.add_argument(
        "--max-instr", type=int, default=2_000_000_000,
        help="Max instructions for benchmark emulation (default: 2 billion)"
    )
    parser.add_argument(
        "--tinyfive", action="store_true",
        help="Run compiled binary through TinyFive RV32IM simulator "
             "(library-backed, independent verification)"
    )
    parser.add_argument(
        "--tinyfive-max-instr", type=int, default=5_000_000,
        help="Max instructions for TinyFive simulation (default: 5 million)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: model file not found: {args.model}", file=sys.stderr)
        return 1

    output_asm = args.asm or args.output.replace(".bin", ".s")
    if output_asm == args.output:
        output_asm = args.output + ".s"

    return convert_onnx_to_riscv(
        args.model, args.output, output_asm,
        benchmark=args.benchmark,
        max_instr=args.max_instr,
        estimate=args.estimate,
        report=args.report,
        uarch=args.uarch,
        tinyfive_sim=args.tinyfive,
        tinyfive_max_instr=args.tinyfive_max_instr,
        const_merge=args.const_merge,
        schedule=args.schedule,
        symbolic_asm=args.symbolic_asm,
        platform_asm=args.platform_asm,
    )


if __name__ == "__main__":
    sys.exit(main())
