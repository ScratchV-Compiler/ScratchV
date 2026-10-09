"""ONNX model parser: reads an ONNX protobuf file and emits IR."""

from __future__ import annotations

import math
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from scratchv.ir.types import DataType, Value, Program
from scratchv.ir.builder import IRBuilder


class ONNXParseError(Exception):
    """Raised when ONNX parsing fails."""
    pass


class ONNXParser:
    """Parses an ONNX model file into an IR Program.

    Requires the ``onnx`` Python package.
    """

    def __init__(self):
        self.builder = IRBuilder()
        self._value_map: dict[str, Value] = {}
        # Pass these arrays to IRInterpreter.run(initializers=...). Program
        # keeps typed global definitions; the execution API binds their data.
        self.initializers: dict[str, np.ndarray] = {}
        self._tensor_info = {}
        self._producers = {}
        self._constant_cache = {}
        self._base_dir = ""
        self._mmap_external_data = False
        self._opsets: dict[str, int] = {}

    def parse(self, model_path: str, *, mmap_external_data: bool = False) -> Program:
        """Parse ONNX, optionally mapping external weights as read-only arrays.

        The default loader is unchanged. The explicit mmap mode avoids
        materializing external weights in TensorProto.raw_data. External files
        must be within the model directory, with valid byte ranges matching
        the tensor shape and dtype. The caller must keep those files unchanged
        while this parser's initializer arrays are in use. Mapping is not an
        integrity check; model/weight hashes must be validated separately.
        Runtime inputs require explicit static shapes after ONNX inference;
        symbolic dimensions or unknown rank are rejected, not treated as zero.
        Inputs also listed as initializers use the initializer's fixed shape.
        """
        if not isinstance(mmap_external_data, bool):
            raise ONNXParseError("mmap_external_data must be boolean")
        try:
            import onnx
        except ImportError:
            raise ONNXParseError(
                "The 'onnx' Python package is required. Install it with:\n"
                "  pip install onnx"
            )

        self.builder = IRBuilder()
        self._value_map = {}
        self.initializers = {}
        self._tensor_info = {}
        self._producers = {}
        self._constant_cache = {}
        self._base_dir = str(Path(model_path).resolve().parent)
        self._mmap_external_data = mmap_external_data
        # Inspect/infer the graph before materializing external weights. This
        # avoids serializing a >2 GiB protobuf just to infer tensor shapes.
        model = onnx.load(model_path, load_external_data=False)
        self._opsets = {item.domain: item.version for item in model.opset_import}
        domains = sorted({node.domain for node in model.graph.node
                          if node.domain not in ("", "ai.onnx")})
        if domains:
            raise ONNXParseError(f"Unsupported ONNX domains: {', '.join(domains)}")
        missing = sorted({
            node.op_type for node in model.graph.node
            if not hasattr(self, f"_handle_{node.op_type.lower()}")
        })
        if missing:
            raise ONNXParseError(f"Unsupported ONNX op types: {', '.join(missing)}")
        try:
            model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        except onnx.shape_inference.InferenceError as exc:
            raise ONNXParseError(f"ONNX shape inference failed: {exc}") from exc
        graph = model.graph
        # Protobuf returns dim_value=0 when a dimension is symbolic or absent.
        # Check presence before reading it: an explicit zero-sized tensor is
        # static, while a missing shape field is unknown rank, not a scalar.
        input_shapes = {}
        initializer_names = {init.name for init in graph.initializer}
        for inp in graph.input:
            if inp.name in initializer_names:
                # Legacy exports list fixed weights as graph inputs as well.
                # This parser binds those as initializers, not parameters;
                # their tensor data defines the shape even if input metadata
                # remains symbolic after ONNX shape inference.
                continue
            tensor = inp.type.tensor_type
            if not tensor.HasField("shape") or any(
                not dim.HasField("dim_value") or dim.dim_value < 0
                for dim in tensor.shape.dim
            ):
                raise ONNXParseError(
                    f"Input {inp.name!r} requires a static nonnegative shape; "
                    "dynamic dimensions and unknown rank are unsupported"
                )
            input_shapes[inp.name] = tuple(dim.dim_value for dim in tensor.shape.dim)
        self._tensor_info = {
            value.name: value.type.tensor_type
            for value in [*graph.input, *graph.value_info, *graph.output]
        }
        # Globals/parameters retain ONNX names, while builder temporaries use
        # v_N. Reserve the entire graph before emitting anything, including
        # Constant nodes encountered later, so SSA names cannot collide.
        names = {value.name for value in [*graph.input, *graph.initializer,
                                          *graph.value_info, *graph.output]}
        names.update(name for node in graph.node for name in [*node.input, *node.output])
        self.builder._name_counter = max(
            (int(name[2:]) for name in names
             if name.startswith("v_") and name[2:].isascii() and name[2:].isdecimal()),
            default=0,
        )

        # Create IR function from ONNX graph
        func_name = graph.name or "main"
        func = self.builder.new_function(func_name)
        self.builder.new_block("entry")  # entry block

        # Map ONNX initializers (constants) to IR values
        for init in graph.initializer:
            if mmap_external_data and onnx.external_data_helper.uses_external_data(init):
                arr = self._map_external_initializer(init)
            else:
                arr = onnx.numpy_helper.to_array(init, base_dir=self._base_dir)
            self._bind_constant(init.name, arr)

        # Map graph inputs to function params
        for inp in graph.input:
            if inp.name in self._value_map:
                continue  # already defined as initializer
            dtype = DataType.FLOAT32
            if inp.type.tensor_type.elem_type:
                dtype = self._dtype(inp.type.tensor_type.elem_type)
            val = self.builder.make_value(name=inp.name, dtype=dtype)
            val.shape = input_shapes[inp.name]
            func.params.append(val)
            self._value_map[inp.name] = val

        # Process graph outputs
        output_names = {o.name for o in graph.output}

        # Process nodes
        for node in graph.node:
            self._translate_node(node, output_names)

        # Add return if we have outputs
        for o in graph.output:
            if o.name in self._value_map:
                self.builder.ret(self._value_map[o.name])
                break
        else:
            self.builder.ret()

        return self.builder.program

    def _map_external_initializer(self, tensor):
        """Validate before mapping; never populate the TensorProto raw_data."""
        import numpy as np
        from onnx import helper

        self._dtype(tensor.data_type)  # Keep the frontend's supported dtype set.
        fields = {}
        for entry in tensor.external_data:
            if entry.key in fields:
                raise ONNXParseError(f"Duplicate external field for {tensor.name}: {entry.key}")
            fields[entry.key] = entry.value
        unknown = set(fields) - {"location", "offset", "length", "checksum"}
        if unknown:
            raise ONNXParseError(f"Unsupported external fields for {tensor.name}: {sorted(unknown)}")
        location = fields.get("location", "")
        windows = PureWindowsPath(location)
        path = Path(location)
        if (not location or path.is_absolute() or windows.is_absolute()
                or windows.drive or ".." in path.parts or ".." in windows.parts):
            raise ONNXParseError(f"External path must stay within the model directory: {tensor.name}")
        # Recognize either separator on every host; resolve symlinks as well.
        base = Path(self._base_dir).resolve()
        candidate = (base / Path(*windows.parts)).resolve()
        try:
            candidate.relative_to(base)
        except ValueError as exc:
            raise ONNXParseError(f"External path escapes the model directory: {tensor.name}") from exc
        if candidate == base:
            raise ONNXParseError(f"External path escapes the model directory: {tensor.name}")
        if tensor.HasField("raw_data") or tensor.HasField("segment"):
            raise ONNXParseError(f"Ambiguous raw or segmented external tensor: {tensor.name}")
        shape = tuple(tensor.dims)
        if any(dimension < 0 for dimension in shape):
            raise ONNXParseError(f"Negative external tensor dimension: {tensor.name}")
        dtype = np.dtype(helper.tensor_dtype_to_np_dtype(tensor.data_type)).newbyteorder("<")
        expected = math.prod(shape) * dtype.itemsize

        def size_field(name, default):
            value = fields.get(name)
            if value is None:
                return default
            if not value or not value.isascii() or not value.isdecimal():
                raise ONNXParseError(f"External {name} must be a nonnegative integer: {tensor.name}")
            return int(value)

        offset = size_field("offset", 0)
        length = size_field("length", expected)
        if length != expected:
            raise ONNXParseError(f"External length disagrees with tensor shape/dtype: {tensor.name}")
        try:
            if not candidate.is_file() or offset + expected > candidate.stat().st_size:
                raise ONNXParseError(f"External tensor byte range exceeds file: {tensor.name}")
            if expected == 0:
                array = np.empty(shape, dtype=dtype)
                array.setflags(write=False)
                return array
            return np.memmap(candidate, mode="r", dtype=dtype, offset=offset, shape=shape, order="C")
        except (OSError, ValueError, OverflowError) as exc:
            raise ONNXParseError(f"Cannot map external tensor {tensor.name}: {exc}") from exc

    def _translate_node(self, node, output_names: set[str]) -> None:
        """Translate a single ONNX node to IR instructions."""
        op_type = node.op_type
        if node.domain not in ("", "ai.onnx"):
            raise ONNXParseError(f"Unsupported ONNX domain: {node.domain}")
        inputs = [self._get_value(name) for name in node.input if name]
        outputs = node.output

        handler = getattr(self, f"_handle_{op_type.lower()}", None)
        if handler is None:
            raise ONNXParseError(f"Unsupported ONNX op type: {op_type}")

        start = len(self.builder.current_block.instructions)
        handler(node, inputs, outputs)
        for instruction in self.builder.current_block.instructions[start:]:
            if instruction.dest is not None:
                self._producers[instruction.dest.name] = instruction
        for name in outputs:
            info = self._tensor_info.get(name)
            if info is not None:
                value = self._value_map[name]
                if info.elem_type:
                    value.dtype = self._dtype(info.elem_type)
                if all(dim.HasField("dim_value") for dim in info.shape.dim):
                    value.shape = tuple(dim.dim_value for dim in info.shape.dim)

    def _get_value(self, name: str) -> Value:
        if name not in self._value_map:
            raise ONNXParseError(f"ONNX input has no preceding definition: {name}")
        return self._value_map[name]

    def _define_outputs(self, outputs: list[str],
                        value: Value | None = None) -> Value:
        """Register output names for a node."""
        if value is None:
            value = self.builder.make_value()
        for name in outputs:
            self._value_map[name] = value
        return value

    @staticmethod
    def _attributes(node) -> dict:
        from onnx import helper
        return {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}

    @staticmethod
    def _dtype(elem_type: int) -> DataType:
        if elem_type not in (1, 6, 7, 11):
            raise ONNXParseError(f"Unsupported ONNX element type: {elem_type}")
        return DataType.from_onnx(elem_type)

    def _bind_constant(self, name: str, data) -> Value:
        from onnx import helper
        dtype = self._dtype(helper.np_dtype_to_tensor_dtype(data.dtype))
        value = self.builder.make_value(name=name, dtype=dtype,
                                        is_constant=data.ndim == 0,
                                        const_value=data.item() if data.ndim == 0 else None)
        value.shape = tuple(data.shape)
        self.builder.program.global_values.append(value)
        self.initializers[name] = data
        self._value_map[name] = value
        if data.ndim == 0:
            self.builder.load_const(data.item(), dtype)
        return value

    def _constant_array(self, value: Value):
        """Resolve only shape/axis dependencies, not whole-model constant folding.

        Exporters encode these as Constant -> Cast/Abs/Reshape chains. Keep
        computed values separate from initializer bindings: intermediate IR
        definitions are not globals accepted by IRInterpreter.run().
        """
        import numpy as np
        from scratchv.verification.ir_numpy_ops import DTYPES, OpError, compute

        if value.name in self.initializers:
            return self.initializers[value.name]
        if value.name in self._constant_cache:
            return self._constant_cache[value.name]
        if value.is_constant:
            return np.asarray(value.const_value, dtype=DTYPES[value.dtype])
        instruction = self._producers.get(value.name)
        allowed = {"cast", "abs", "reshape", "concat", "unsqueeze", "transpose",
                   "slice", "gather", "add", "sub", "mul", "div", "neg", "expand"}
        if instruction is None or instruction.opcode.value not in allowed:
            raise ONNXParseError(
                f"Expected a constant integer vector for shape/axes: {value.name}")
        operands = [self._constant_array(operand) for operand in instruction.operands]
        if any(data.size > 4096 for data in operands):
            raise ONNXParseError("Shape/axis constant expression exceeds 4096 elements")
        try:
            if instruction.opcode.value == "expand":
                requested = instruction.attrs["shape"]
                # Python integers avoid int64 product overflow, and the actual
                # broadcast shape can be larger than the requested shape. The
                # first guard also avoids NumPy's platform-sized shape limit.
                if (math.prod(requested) > 4096 or math.prod(
                        np.broadcast_shapes(operands[0].shape, requested)) > 4096):
                    raise ONNXParseError("Shape/axis constant expression exceeds 4096 elements")
            result = compute(instruction, operands)
        except (OpError, ValueError, TypeError, FloatingPointError, OverflowError) as exc:
            raise ONNXParseError(f"Invalid shape/axis constant {value.name}: {exc}") from exc
        self._constant_cache[value.name] = result
        return result

    def _constant_ints(self, value: Value) -> tuple[int, ...]:
        data = self._constant_array(value)
        if data.ndim != 1 or data.dtype.kind not in "iu":
            raise ONNXParseError(
                f"Expected a constant integer vector for shape/axes: {value.name}"
            )
        return tuple(int(item) for item in data)

    # --- Operator handlers ---

    def _handle_constant(self, node, inputs, outputs):
        import numpy as np
        from onnx import external_data_helper, numpy_helper
        attrs = self._attributes(node)
        if len(attrs) != 1 or len(outputs) != 1 or inputs:
            raise ONNXParseError("Constant requires one value attribute and one output")
        key, value = next(iter(attrs.items()))
        if key == "value":
            if self._mmap_external_data and external_data_helper.uses_external_data(value):
                data = self._map_external_initializer(value)
            else:
                data = numpy_helper.to_array(value, base_dir=self._base_dir)
        elif key in ("value_int", "value_ints"):
            data = np.asarray(value, dtype=np.int64)
        elif key in ("value_float", "value_floats"):
            data = np.asarray(value, dtype=np.float32)
        else:
            raise ONNXParseError(f"Unsupported Constant attribute: {key}")
        self._bind_constant(outputs[0], data)

    def _handle_identity(self, node, inputs, outputs):
        self._define_outputs(outputs, inputs[0])

    def _handle_cast(self, node, inputs, outputs):
        attrs = self._attributes(node)
        if set(attrs) != {"to"}:
            raise ONNXParseError("Cast supports only the numeric 'to' attribute")
        self._define_outputs(outputs, self.builder.cast(inputs[0], self._dtype(attrs["to"])))

    def _handle_abs(self, node, inputs, outputs):
        self._define_outputs(outputs, self.builder.abs(inputs[0]))

    def _handle_cos(self, node, inputs, outputs):
        self._define_outputs(outputs, self.builder.cos(inputs[0]))

    def _handle_sin(self, node, inputs, outputs):
        self._define_outputs(outputs, self.builder.sin(inputs[0]))

    def _handle_pow(self, node, inputs, outputs):
        self._define_outputs(outputs, self.builder.pow(inputs[0], inputs[1]))

    def _handle_reciprocal(self, node, inputs, outputs):
        self._define_outputs(outputs, self.builder.reciprocal(inputs[0]))

    def _handle_add(self, node, inputs: list[Value],
                    outputs: list[str]) -> None:
        # Preserve per-operation rounding and integer width. Shape constants
        # are evaluated lazily by _constant_array using these same IR kernels.
        self._define_outputs(outputs, self.builder.add(inputs[0], inputs[1]))

    def _handle_mul(self, node, inputs: list[Value],
                    outputs: list[str]) -> None:
        self._define_outputs(outputs, self.builder.mul(inputs[0], inputs[1]))

    def _handle_sub(self, node, inputs: list[Value],
                    outputs: list[str]) -> None:
        a, b = inputs[0], inputs[1]
        result = self.builder.sub(a, b)
        self._define_outputs(outputs, result)

    def _handle_div(self, node, inputs: list[Value],
                    outputs: list[str]) -> None:
        a, b = inputs[0], inputs[1]
        result = self.builder.div(a, b)
        self._define_outputs(outputs, result)

    def _handle_relu(self, node, inputs: list[Value],
                     outputs: list[str]) -> None:
        result = self.builder.relu(inputs[0])
        self._define_outputs(outputs, result)

    def _handle_matmul(self, node, inputs: list[Value],
                       outputs: list[str]) -> None:
        a, b = inputs[0], inputs[1]
        if len(a.shape) == len(b.shape) == 2 and all(d > 0 for d in a.shape + b.shape):
            result = self.builder.matmul(a, b, a.shape[0], b.shape[1], a.shape[1])
        else:
            # Batched/vector ONNX MatMul uses NumPy broadcasting semantics.
            # m/n/k describe only the legacy 2-D/flattened-matrix contract.
            result = self.builder.matmul(a, b)
        self._define_outputs(outputs, result)

    def _handle_gelu(self, node, inputs: list[Value],
                     outputs: list[str]) -> None:
        # ONNX defaults to the exact erf formulation, while this IR's GELU
        # deliberately implements the tanh approximation. Never substitute it
        # for the default just because both operations share a name.
        attrs = self._attributes(node)
        if set(attrs) - {"approximate"} or attrs.get("approximate", b"none") != b"tanh":
            raise ONNXParseError("Gelu supports only approximate='tanh'; exact Gelu is unsupported")
        result = self.builder.gelu(inputs[0])
        self._define_outputs(outputs, result)

    def _handle_softmax(self, node, inputs: list[Value],
                        outputs: list[str]) -> None:
        # Before opset 13 Softmax flattens dimensions from axis onward and
        # defaults to axis=1; the IR normalizes a single axis. Supporting the
        # modern default for a legacy graph silently produces wrong values.
        if self._opsets.get(node.domain, 0) < 13:
            raise ONNXParseError("Softmax requires ONNX opset >= 13; legacy flatten semantics are unsupported")
        axis = -1
        for attr in node.attribute:
            if attr.name == "axis":
                axis = attr.i
        result = self.builder.softmax(inputs[0], axis=axis)
        self._define_outputs(outputs, result)

    def _handle_maxpool(self, node, inputs: list[Value],
                        outputs: list[str]) -> None:
        kernel = 2
        stride = 2
        for attr in node.attribute:
            if attr.name == "kernel_shape":
                kernel = attr.ints[0]
            if attr.name == "strides":
                stride = attr.ints[0]
        result = self.builder.maxpool(inputs[0], kernel, stride)
        self._define_outputs(outputs, result)

    def _handle_neg(self, node, inputs: list[Value],
                    outputs: list[str]) -> None:
        result = self.builder.neg(inputs[0])
        self._define_outputs(outputs, result)

    def _handle_exp(self, node, inputs: list[Value],
                    outputs: list[str]) -> None:
        result = self.builder.exp(inputs[0])
        self._define_outputs(outputs, result)

    def _handle_conv(self, node, inputs: list[Value],
                     outputs: list[str]) -> None:
        x, w, b = inputs[0], inputs[1], inputs[2]
        out_channels = w.shape[0] if len(w.shape) > 0 else 1
        kernel_size = w.shape[2] if len(w.shape) > 2 else 3
        stride = 1
        padding = 1
        for attr in node.attribute:
            if attr.name == "kernel_shape":
                kernel_size = attr.ints[0]
            elif attr.name == "strides":
                stride = attr.ints[0]
            elif attr.name == "pads":
                padding = attr.ints[0]
        result = self.builder.conv(x, w, b, out_channels,
                                   kernel_size, stride, padding)
        self._define_outputs(outputs, result)

    def _handle_gemm(self, node, inputs: list[Value],
                     outputs: list[str]) -> None:
        attrs = self._attributes(node)
        if set(attrs) - {"alpha", "beta", "transA", "transB"}:
            raise ONNXParseError("Unsupported Gemm attributes")
        if len(inputs) not in (2, 3):
            raise ONNXParseError("Gemm requires two matrices and an optional bias")
        a, w = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else self.builder.make_const(0, a.dtype)
        result = self.builder.gemm(
            a, w, b, bool(attrs.get("transA", 0)), bool(attrs.get("transB", 0)),
            alpha=attrs.get("alpha", 1.0), beta=attrs.get("beta", 1.0),
        )
        self._define_outputs(outputs, result)

    def _handle_sigmoid(self, node, inputs: list[Value],
                        outputs: list[str]) -> None:
        result = self.builder.sigmoid(inputs[0])
        self._define_outputs(outputs, result)

    def _handle_reshape(self, node, inputs: list[Value],
                        outputs: list[str]) -> None:
        if self._attributes(node).get("allowzero", 0):
            raise ONNXParseError("Reshape allowzero=1 is not supported by the IR")
        shape = self._constant_ints(inputs[1])
        result = self.builder.reshape(inputs[0], shape)
        self._define_outputs(outputs, result)

    def _handle_gather(self, node, inputs, outputs):
        axis = self._attributes(node).get("axis", 0)
        self._define_outputs(outputs, self.builder.gather(inputs[0], inputs[1], axis))

    def _handle_sqrt(self, node, inputs, outputs):
        self._define_outputs(outputs, self.builder.sqrt(inputs[0]))

    def _handle_reducemean(self, node, inputs, outputs):
        attrs = self._attributes(node)
        axes = self._constant_ints(inputs[1]) if len(inputs) > 1 else attrs.get("axes")
        if not axes and attrs.get("noop_with_empty_axes", 0):
            self._define_outputs(outputs, inputs[0])
        else:
            self._define_outputs(outputs, self.builder.reduce_mean(
                inputs[0], axes, bool(attrs.get("keepdims", 1))))

    def _handle_transpose(self, node, inputs, outputs):
        self._define_outputs(outputs, self.builder.transpose(
            inputs[0], self._attributes(node).get("perm")))

    def _handle_concat(self, node, inputs, outputs):
        self._define_outputs(outputs, self.builder.concat(
            inputs, self._attributes(node)["axis"]))

    def _handle_slice(self, node, inputs, outputs):
        attrs = self._attributes(node)
        # Preserve optional input positions: steps may be present without axes.
        for index, key in enumerate(("starts", "ends", "axes", "steps"), start=1):
            if len(node.input) > index and node.input[index]:
                attrs[key] = self._constant_ints(self._get_value(node.input[index]))
        self._define_outputs(outputs, self.builder.slice(
            inputs[0], attrs["starts"], attrs["ends"], attrs.get("axes"), attrs.get("steps")))

    def _handle_unsqueeze(self, node, inputs, outputs):
        axes = (self._constant_ints(inputs[1]) if len(inputs) > 1
                else self._attributes(node)["axes"])
        self._define_outputs(outputs, self.builder.unsqueeze(inputs[0], axes))

    def _handle_expand(self, node, inputs, outputs):
        self._define_outputs(outputs, self.builder.expand(
            inputs[0], self._constant_ints(inputs[1])))
