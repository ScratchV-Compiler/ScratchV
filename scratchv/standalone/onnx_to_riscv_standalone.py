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

import struct
import sys
import os
import argparse
import math
from typing import Optional, Union

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

    def _infer_conv_shape(
        self, node: NodeInfo, shapes: list[tuple[int, ...]]
    ) -> None:
        """Infer Conv2D output shape: NCHW format."""
        if len(shapes) < 2 or not shapes[0]:
            return
        x_shape = shapes[0]  # (N, C, H, W)
        w_shape = shapes[1] if len(shapes) > 1 else ()  # (C_out, C_in, K, K)

        out_channels = w_shape[0] if len(w_shape) >= 1 else 1

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

    def __init__(self):
        self.code: list[int] = []  # list of 32-bit instruction words
        self.labels: dict[str, int] = {}  # label name → instruction index
        self.pending_fixups: list[tuple[int, str, str]] = []  # (idx, kind, label)
        self.comments: dict[int, str] = {}  # instruction index → comment

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
        self.labels[name] = len(self.code)

    def emit_branch(self, op_builder, rs1: int, rs2: int,
                    label: str, comment: str = "") -> int:
        """Emit a branch instruction with label fixup."""
        idx = self._emit(op_builder(rs1, rs2, 0), comment)
        self.pending_fixups.append((idx, "b", label))
        return idx

    def emit_jump(self, op_builder, label: str, comment: str = "") -> int:
        """Emit a jump instruction with label fixup."""
        idx = self._emit(op_builder(0), comment)
        self.pending_fixups.append((idx, "j", label))
        return idx

    def emit_jal(self, rd: int, label: str, comment: str = "") -> int:
        idx = self._emit(rv_jal(rd, 0), comment)
        self.pending_fixups.append((idx, "j", label))
        return idx

    def emit_li(self, rd: int, imm: int, comment: str = "") -> None:
        """Emit load-immediate (may be 1 or 2 instructions)."""
        words = rv_li(rd, imm)
        self.emit_many(list(words), comment)

    def emit_li32(self, rd: int, imm: int, comment: str = "") -> None:
        """Emit a 32-bit immediate using LUI + ADDI sequence.

        Always emits 2 instructions (even if imm fits in 12 bits),
        for cases where we need the full 32-bit value.
        """
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
                print(f"WARNING: undefined label '{label}' — fixup at instr {idx}")
                continue
            target_idx = self.labels[label]
            byte_offset = (target_idx - idx) * 4
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

    def disassemble(self) -> str:
        """Produce human-readable RISC-V assembly for verification."""
        lines = []
        for i, word in enumerate(self.code):
            # Label?
            for name, idx in self.labels.items():
                if idx == i:
                    lines.append(f"{name}:")

            asm = _disasm_one(word)
            comment = self.comments.get(i, "")
            if comment:
                asm = f"{asm:<32s} # {comment}"
            lines.append(f"  {asm}")
        return "\n".join(lines)

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

    def __init__(self, model: ONNXModel, memory: MemoryPlan):
        self.model = model
        self.mem = memory
        self.emit = RISCVEmitter()

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
        self.emit.label("_init_data_base")
        # The data base address loading is handled at binary assembly time.
        # We embed a dummy AUIPC + ADDI that gets patched:
        #   auipc gp, 0        → placeholder
        #   addi gp, gp, 0     → placeholder
        self.emit.emit(rv_auipc(_R_GP, 0), "gp = data base (patched at link time)")
        self.emit.emit(rv_addi(_R_GP, _R_GP, 0), "data offset (patched)")

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

        # Resolve internal branch/jump fixups
        self.emit.resolve_fixups()

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
        handler(node)

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
        """Generate inline Conv2D: NCHW layout, direct convolution with loops.

        for oc in range(C_out):
          for oh in range(H_out):
            for ow in range(W_out):
              acc = bias[oc]   # Q16.16
              for ic in range(C_in):
                for kh in range(K):
                  for kw in range(K):
                    ih = oh*S + kh - P
                    iw = ow*S + kw - P
                    if 0 <= ih < H and 0 <= iw < W:
                      acc += input[ic,ih,iw] * weight[oc,ic,kh,kw]
              acc = acc >> 16 (normalize)
              output[oc,oh,ow] = acc
        """
        x_name = node.inputs[0]
        w_name = node.inputs[1]
        b_name = node.inputs[2]
        out_name = node.outputs[0]

        x_shape = self.model.get_shape(x_name)
        w_shape = self.model.get_shape(w_name)

        # Dimensions (NCHW)
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

        # Allocate workspace for output
        out_elements = N * C_out * H_out * W_out
        self.mem.alloc_workspace(out_name, out_elements)

        # Load base addresses
        # Input tensor: s0 points to input (caller-provided for first layer,
        # workspace for subsequent layers)
        self._get_workspace_addr(x_name, self.S2)  # s2 = input base
        self._get_weight_addr(w_name, self.S3)      # s3 = weight base
        self._get_weight_addr(b_name, self.S4)      # s4 = bias base
        self._get_workspace_addr(out_name, self.S5) # s5 = output base

        H_out_reg = self.S6   # oh loop counter
        W_out_reg = self.S7   # ow loop counter
        C_out_reg = self.S8   # oc loop counter
        C_in_reg = self.S9    # ic loop counter
        Kh_reg = self.S10     # kh loop counter
        Kw_reg = self.S11     # kw loop counter

        acc_reg = self.T0     # accumulation
        val_reg = self.T1     # loaded value
        tmp_reg = self.T2     # temporary
        addr_reg = self.T3    # address computation
        cond_reg = self.T4    # condition check

        L = self.emit.label  # shorthand

        # Initialize oc = 0
        self.emit.emit(rv_addi(C_out_reg, _R_ZERO, 0), f"oc = 0 (C_out={C_out})")
        L("_conv_oc_loop")

        # Load bias[oc]
        self.emit.emit(rv_slli(addr_reg, C_out_reg, 2), "addr_reg = oc * 4")
        self.emit.emit(rv_add(addr_reg, self.S4, addr_reg), "addr_reg = bias_base + oc*4")
        self.emit.emit(rv_lw(acc_reg, addr_reg, 0), "acc = bias[oc]")

        # Initialize oh = 0
        self.emit.emit(rv_addi(H_out_reg, _R_ZERO, 0), f"oh = 0 (H_out={H_out})")
        L("_conv_oh_loop")

        # Initialize ow = 0
        self.emit.emit(rv_addi(W_out_reg, _R_ZERO, 0), f"ow = 0 (W_out={W_out})")
        L("_conv_ow_loop")

        # Initialize ic = 0
        self.emit.emit(rv_addi(C_in_reg, _R_ZERO, 0), f"ic = 0 (C_in={C_in})")
        L("_conv_ic_loop")

        # Initialize kh = 0
        self.emit.emit(rv_addi(Kh_reg, _R_ZERO, 0), f"kh = 0 (K={K})")
        L("_conv_kh_loop")

        # Initialize kw = 0
        self.emit.emit(rv_addi(Kw_reg, _R_ZERO, 0), f"kw = 0 (K={K})")
        L("_conv_kw_loop")

        # --- Inner MAC computation ---
        # Compute ih = oh * stride + kh - pad
        # iw = ow * stride + kw - pad
        self.emit.emit_li(self.T6, sh, f"stride_h = {sh}")
        self.emit.emit(rv_mul(tmp_reg, H_out_reg, self.T6), "tmp = oh * stride_h")
        self.emit.emit(rv_add(tmp_reg, tmp_reg, Kh_reg), "tmp += kh")
        self.emit.emit(rv_addi(tmp_reg, tmp_reg, -ph), f"tmp -= pad_h ({ph}) → ih")
        # tmp_reg = ih

        # Check if 0 <= ih < H
        self.emit.emit(rv_slti(cond_reg, tmp_reg, 0), "cond = (ih < 0)")
        skip_label = f"_conv_skip_{len(self.emit.labels)}"
        self.emit.emit_branch(rv_bne, cond_reg, _R_ZERO, skip_label, "skip if ih < 0")
        self.emit.emit_li(self.T6, H, f"H = {H}")
        self.emit.emit(rv_slt(cond_reg, tmp_reg, self.T6), "cond = (ih < H)")
        self.emit.emit_branch(rv_beq, cond_reg, _R_ZERO, skip_label, "skip if ih >= H")

        # Compute iw = ow * stride_w + kw - pad_w
        self.emit.emit_li(self.T6, sw, f"stride_w = {sw}")
        self.emit.emit(rv_mul(val_reg, W_out_reg, self.T6), "val = ow * stride_w")
        self.emit.emit(rv_add(val_reg, val_reg, Kw_reg), "val += kw")
        self.emit.emit(rv_addi(val_reg, val_reg, -pw), f"val -= pad_w ({pw}) → iw")

        # Check if 0 <= iw < W
        self.emit.emit(rv_slti(cond_reg, val_reg, 0), "cond = (iw < 0)")
        self.emit.emit_branch(rv_bne, cond_reg, _R_ZERO, skip_label, "skip if iw < 0")
        self.emit.emit_li(self.T6, W, f"W = {W}")
        self.emit.emit(rv_slt(cond_reg, val_reg, self.T6), "cond = (iw < W)")
        self.emit.emit_branch(rv_beq, cond_reg, _R_ZERO, skip_label, "skip if iw >= W")

        # Load input[ic, ih, iw]
        # offset = ic * H * W + ih * W + iw
        self.emit.emit_li(addr_reg, H * W, f"H*W = {H*W}")
        self.emit.emit(rv_mul(addr_reg, C_in_reg, addr_reg), "addr = ic * H * W")
        self.emit.emit_li(self.T5, W, f"W = {W}")
        self.emit.emit(rv_mul(self.T5, tmp_reg, self.T5), "tmp5 = ih * W")
        self.emit.emit(rv_add(addr_reg, addr_reg, self.T5), "addr += ih * W")
        self.emit.emit(rv_add(addr_reg, addr_reg, val_reg), "addr += iw")
        self.emit.emit(rv_slli(addr_reg, addr_reg, 2), "addr *= 4 (byte offset)")
        self.emit.emit(rv_add(addr_reg, self.S2, addr_reg), "addr += input_base")
        self.emit.emit(rv_lw(val_reg, addr_reg, 0), "load input[ic,ih,iw]")

        # Load weight[oc, ic, kh, kw]
        # weight_offset = oc*C_in*K*K + ic*K*K + kh*K + kw
        self.emit.emit_li(addr_reg, C_in * K * K, f"C_in*K*K = {C_in*K*K}")
        self.emit.emit(rv_mul(addr_reg, C_out_reg, addr_reg), "addr = oc * C_in*K*K")
        self.emit.emit_li(self.T5, K * K, f"K*K = {K*K}")
        self.emit.emit(rv_mul(self.T5, C_in_reg, self.T5), "tmp5 = ic * K*K")
        self.emit.emit(rv_add(addr_reg, addr_reg, self.T5), "addr += ic * K*K")
        self.emit.emit_li(self.T5, K, f"K = {K}")
        self.emit.emit(rv_mul(self.T5, Kh_reg, self.T5), "tmp5 = kh * K")
        self.emit.emit(rv_add(addr_reg, addr_reg, self.T5), "addr += kh * K")
        self.emit.emit(rv_add(addr_reg, addr_reg, Kw_reg), "addr += kw")
        self.emit.emit(rv_slli(addr_reg, addr_reg, 2), "addr *= 4 (byte offset)")
        self.emit.emit(rv_add(addr_reg, self.S3, addr_reg), "addr += weight_base")
        self.emit.emit(rv_lw(self.T5, addr_reg, 0), "load weight[oc,ic,kh,kw]")

        # MAC: acc += input_val * weight_val (Q16.16 multiply)
        self.emit.emit_qmul_srai(self.T6, val_reg, self.T5, self.T4,
                                 "Q16.16: input * weight")
        self.emit.emit(rv_add(acc_reg, acc_reg, self.T6), "acc += input*weight")

        self.emit.label(skip_label)

        # Increment kw
        self.emit.emit(rv_addi(Kw_reg, Kw_reg, 1), "kw++")
        self.emit.emit_li(self.T6, K, f"K = {K}")
        self.emit.emit(rv_slt(cond_reg, Kw_reg, self.T6), "kw < K?")
        self.emit.emit_branch(rv_bne, cond_reg, _R_ZERO, "_conv_kw_loop", "loop kw")

        # Increment kh
        self.emit.emit(rv_addi(Kh_reg, Kh_reg, 1), "kh++")
        self.emit.emit_li(self.T6, K, f"K = {K}")
        self.emit.emit(rv_slt(cond_reg, Kh_reg, self.T6), "kh < K?")
        self.emit.emit_branch(rv_bne, cond_reg, _R_ZERO, "_conv_kh_loop", "loop kh")

        # Increment ic
        self.emit.emit(rv_addi(C_in_reg, C_in_reg, 1), "ic++")
        self.emit.emit_li(self.T6, C_in, f"C_in = {C_in}")
        self.emit.emit(rv_slt(cond_reg, C_in_reg, self.T6), "ic < C_in?")
        self.emit.emit_branch(rv_bne, cond_reg, _R_ZERO, "_conv_ic_loop", "loop ic")

        # Store output[oc, oh, ow] = acc
        # output_offset = oc*H_out*W_out + oh*W_out + ow
        self.emit.emit_li(addr_reg, H_out * W_out, f"H_out*W_out = {H_out*W_out}")
        self.emit.emit(rv_mul(addr_reg, C_out_reg, addr_reg), "addr = oc * H_out*W_out")
        self.emit.emit_li(self.T6, W_out, f"W_out = {W_out}")
        self.emit.emit(rv_mul(self.T6, H_out_reg, self.T6), "tmp = oh * W_out")
        self.emit.emit(rv_add(addr_reg, addr_reg, self.T6), "addr += oh * W_out")
        self.emit.emit(rv_add(addr_reg, addr_reg, W_out_reg), "addr += ow")
        self.emit.emit(rv_slli(addr_reg, addr_reg, 2), "addr *= 4")
        self.emit.emit(rv_add(addr_reg, self.S5, addr_reg), "addr += output_base")
        self.emit.emit(rv_sw(addr_reg, acc_reg, 0), "store output[oc,oh,ow] = acc")

        # Increment ow
        self.emit.emit(rv_addi(W_out_reg, W_out_reg, 1), "ow++")
        self.emit.emit_li(self.T6, W_out, f"W_out = {W_out}")
        self.emit.emit(rv_slt(cond_reg, W_out_reg, self.T6), "ow < W_out?")
        self.emit.emit_branch(rv_bne, cond_reg, _R_ZERO, "_conv_ow_loop", "loop ow")

        # Increment oh
        self.emit.emit(rv_addi(H_out_reg, H_out_reg, 1), "oh++")
        self.emit.emit_li(self.T6, H_out, f"H_out = {H_out}")
        self.emit.emit(rv_slt(cond_reg, H_out_reg, self.T6), "oh < H_out?")
        self.emit.emit_branch(rv_bne, cond_reg, _R_ZERO, "_conv_oh_loop", "loop oh")

        # Increment oc
        self.emit.emit(rv_addi(C_out_reg, C_out_reg, 1), "oc++")
        self.emit.emit_li(self.T6, C_out, f"C_out = {C_out}")
        self.emit.emit(rv_slt(cond_reg, C_out_reg, self.T6), "oc < C_out?")
        self.emit.emit_branch(rv_bne, cond_reg, _R_ZERO, "_conv_oc_loop", "loop oc")

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
        b_name = node.inputs[2]
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

        # Load bias[j] → acc
        self.emit.emit(rv_slli(addr_reg, j_reg, 2), "addr = j*4")
        self.emit.emit(rv_add(addr_reg, self.S4, addr_reg), "addr += bias_base")
        self.emit.emit(rv_lw(acc_reg, addr_reg, 0), "acc = bias[j]")

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
        self.emit.emit(rv_addi(dst_reg, dst_reg, 32768), "+ 0.5 (Q16.16)")
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
        out_shape = self.model.get_shape(out_name)

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

# ═══════════════════════════════════════════════════════════════════════════
# Part 8: Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════


def convert_onnx_to_riscv(
    onnx_path: str, output_bin: str, output_asm: str = "",
    benchmark: bool = False, max_instr: int = 2_000_000_000,
    estimate: bool = False, report: bool = False,
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
    print(f"ScratchV: Library-free ONNX → RISC-V RV32IM Pipeline")
    print(f"{'='*60}")
    print(f"  Input:  {onnx_path}")

    # ── Step 1: Parse ONNX protobuf ────────────────────────────────────
    print(f"\n[1/5] Parsing ONNX protobuf (manual wire-format parser)...")
    model = ONNXModel.from_file(onnx_path)
    print(f"  Graph: {model.graph_name}")
    print(f"  Nodes: {len(model.nodes)}")
    print(f"  Initializers: {len(model.initializers)}")
    for i, node in enumerate(model.nodes):
        print(f"    [{i}] {node.op_type}: "
              f"{', '.join(node.inputs[:2])} → {', '.join(node.outputs[:1])}")

    # ── Step 2: Layout weights and plan memory ─────────────────────────
    print(f"\n[2/5] Converting weights to Q16.16 fixed-point and planning memory...")
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
    print(f"\n[3/5] Generating inline RISC-V RV32IM machine code...")
    generator = CNNRISCVGenerator(model, memory)
    code_bytes = generator.generate()
    print(f"  Code size: {len(code_bytes):,} bytes ({len(code_bytes)//4} instructions)")
    print(f"  Workspace: {memory.workspace_size:,} bytes "
          f"({memory.workspace_size/1024/1024:.1f} MB)")

    # ── Step 4: Assemble binary ────────────────────────────────────────
    print(f"\n[4/5] Assembling flat binary...")
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

    binary = code_bytes + weight_data
    print(f"  Total binary: {len(binary):,} bytes ({len(binary)/1024/1024:.1f} MB)")
    print(f"  Code offset: 0x00000000")
    print(f"  Data offset: 0x{data_offset:08x} ({data_offset:,} bytes)")

    # ── Step 5: Write output ───────────────────────────────────────────
    print(f"\n[5/5] Writing output files...")
    with open(output_bin, "wb") as f:
        f.write(binary)
    print(f"  Binary: {output_bin} ({len(binary):,} bytes)")

    # Disassembly for verification
    asm_text = generator.emit.disassemble()
    asm_path = output_asm or output_bin.replace(".bin", ".s")
    with open(asm_path, "w") as f:
        f.write(asm_text)
    print(f"  Assembly: {asm_path}")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Pipeline complete!")
    print(f"  Code:   {len(code_bytes):,} bytes ({len(code_bytes)//4} instructions)")
    print(f"  Data:   {memory.data_size:,} bytes ({memory.data_size/1024/1024:.1f} MB)")
    print(f"  Total:  {len(binary):,} bytes ({len(binary)/1024/1024:.1f} MB)")
    print(f"  Input:  {input_name if model.inputs else 'unknown'}")
    print(f"          shape={input_shape if model.inputs else 'unknown'}")
    print(f"  Output: {model.outputs[0].name if model.outputs else 'unknown'}")
    print(f"  Workspace needed: {memory.workspace_size/1024/1024:.1f} MB")
    print(f"\n  Bare-metal ABI:")
    print(f"    a0 → input tensor (float32, Q16.16 converted)")
    print(f"    a1 → output buffer")
    print(f"    gp → data section base (set by binary on entry)")
    print(f"    sp → stack pointer (caller must initialize)")
    print(f"    returns via jalr zero, ra, 0")
    print(f"{'='*60}")

    # ── Optional: Analytical estimation ──────────────────────────────────
    if estimate or report:
        print(f"\n[estimate] Analytical instruction count estimation...")
        try:
            from scratchv.standalone.benchmark import estimate_cnn_model, print_estimate
            est = estimate_cnn_model()
            print_estimate(est)
        except ImportError:
            print("  ERROR: benchmark module not found", file=sys.stderr)

    # ── Optional: Generate CI reports (HTML, JSON, GitHub summary) ───────
    if report:
        print(f"\n[report] Generating CI benchmark reports...")
        try:
            from scratchv.standalone.benchmark import estimate_cnn_model
            from scratchv.standalone.bench_report import (
                generate_html_report, generate_json_report, generate_github_summary,
            )
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
                ))
            print(f"  HTML: benchmark_reports/benchmark.html")

            # JSON report
            with open("benchmark_reports/benchmark.json", "w") as f:
                f.write(generate_json_report(
                    code_size=code_len,
                    static_insns=code_len // 4,
                    est_data=est_data,
                    model_name=model_name,
                ))
            print(f"  JSON: benchmark_reports/benchmark.json")

            # GitHub Actions job summary
            with open("benchmark_reports/github_summary.md", "w") as f:
                f.write(generate_github_summary(
                    code_size=code_len,
                    static_insns=code_len // 4,
                    est_data=est_data,
                ))
            print(f"  Summary: benchmark_reports/github_summary.md")
        except ImportError as e:
            print(f"  ERROR: {e}", file=sys.stderr)

    # ── Optional: Benchmark ───────────────────────────────────────────────
    if benchmark:
        print(f"\n[benchmark] Running RISC-V emulation with performance counters...")
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
            from scratchv.standalone.benchmark import run_benchmark, format_benchmark_report
            perf = run_benchmark(
                binary_path=output_bin,
                code_size=code_size,
                input_data=input_bytes,
                max_instr=max_instr,
                label_addrs=label_addrs,
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
        "--max-instr", type=int, default=2_000_000_000,
        help="Max instructions for benchmark emulation (default: 2 billion)"
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
    )


if __name__ == "__main__":
    sys.exit(main())
