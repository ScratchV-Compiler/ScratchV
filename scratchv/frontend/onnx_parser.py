"""ONNX model parser: reads an ONNX protobuf file and emits IR."""

from __future__ import annotations

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

    def parse(self, model_path: str) -> Program:
        """Parse an ONNX model file and return an IR Program."""
        try:
            import onnx
        except ImportError:
            raise ONNXParseError(
                "The 'onnx' Python package is required. Install it with:\n"
                "  pip install onnx"
            )

        model = onnx.load(model_path)
        graph = model.graph

        # Create IR function from ONNX graph
        func_name = graph.name or "main"
        func = self.builder.new_function(func_name)
        self.builder.new_block("entry")  # entry block

        # Map ONNX initializers (constants) to IR values
        for init in graph.initializer:
            arr = onnx.numpy_helper.to_array(init)
            dtype = DataType.from_onnx(init.data_type)
            val = self.builder.make_value(name=init.name, dtype=dtype)
            if arr.size == 1:
                val = self.builder.make_value(
                    name=init.name, dtype=dtype, is_constant=True,
                    const_value=float(arr.item()),
                )
            # Emit a load_const for scalar initializers
            if arr.size == 1:
                self.builder.load_const(float(arr.item()), dtype)
            else:
                # Multi-element tensor: store pointer info in attrs
                val.is_constant = False
                val.shape = tuple(arr.shape)
            self._value_map[init.name] = val

        # Map graph inputs to function params
        for inp in graph.input:
            if inp.name in self._value_map:
                continue  # already defined as initializer
            dtype = DataType.FLOAT32
            if inp.type.tensor_type.elem_type:
                dtype = DataType.from_onnx(inp.type.tensor_type.elem_type)
            val = self.builder.make_value(name=inp.name, dtype=dtype)
            # Infer shape from ONNX type
            shape_dims = list(inp.type.tensor_type.shape.dim)
            val.shape = tuple(d.dim_value for d in shape_dims)
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

    def _translate_node(self, node, output_names: set[str]) -> None:
        """Translate a single ONNX node to IR instructions."""
        op_type = node.op_type
        inputs = [self._get_value(name) for name in node.input if name]
        outputs = node.output

        handler = getattr(self, f"_handle_{op_type.lower()}", None)
        if handler is None:
            raise ONNXParseError(f"Unsupported ONNX op type: {op_type}")

        handler(node, inputs, outputs)

    def _get_value(self, name: str) -> Value:
        if name not in self._value_map:
            val = self.builder.make_value(name=name)
            self._value_map[name] = val
        return self._value_map[name]

    def _define_outputs(self, outputs: list[str],
                        value: Value | None = None) -> Value:
        """Register output names for a node."""
        if value is None:
            value = self.builder.make_value()
        for name in outputs:
            self._value_map[name] = value
        return value

    # --- Operator handlers ---

    def _handle_add(self, node, inputs: list[Value],
                    outputs: list[str]) -> None:
        a, b = inputs[0], inputs[1]
        if a.is_constant and b.is_constant:
            assert a.const_value is not None
            assert b.const_value is not None
            result = self.builder.make_value(
                is_constant=True,
                const_value=a.const_value + b.const_value,
            )
        else:
            result = self.builder.add(a, b)
        self._define_outputs(outputs, result)

    def _handle_mul(self, node, inputs: list[Value],
                    outputs: list[str]) -> None:
        a, b = inputs[0], inputs[1]
        if a.is_constant and b.is_constant:
            assert a.const_value is not None
            assert b.const_value is not None
            result = self.builder.make_value(
                is_constant=True,
                const_value=a.const_value * b.const_value,
            )
        else:
            result = self.builder.mul(a, b)
        self._define_outputs(outputs, result)

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
        m = a.shape[0] if len(a.shape) > 0 else 1
        k = a.shape[1] if len(a.shape) > 1 else 1
        n = b.shape[1] if len(b.shape) > 1 else 1
        result = self.builder.matmul(a, b, m, n, k)
        self._define_outputs(outputs, result)

    def _handle_gelu(self, node, inputs: list[Value],
                     outputs: list[str]) -> None:
        result = self.builder.gelu(inputs[0])
        self._define_outputs(outputs, result)

    def _handle_softmax(self, node, inputs: list[Value],
                        outputs: list[str]) -> None:
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
        a, w, b = inputs[0], inputs[1], inputs[2]
        trans_a = False
        trans_b = False
        for attr in node.attribute:
            if attr.name == "transA":
                trans_a = attr.i != 0
            elif attr.name == "transB":
                trans_b = attr.i != 0
        result = self.builder.gemm(a, w, b, trans_a, trans_b)
        self._define_outputs(outputs, result)

    def _handle_sigmoid(self, node, inputs: list[Value],
                        outputs: list[str]) -> None:
        result = self.builder.sigmoid(inputs[0])
        self._define_outputs(outputs, result)

    def _handle_reshape(self, node, inputs: list[Value],
                        outputs: list[str]) -> None:
        # The second input contains the target shape
        shape = ()
        if len(inputs) > 1 and inputs[1].is_constant:
            # shape is a constant — extract it from attrs
            shape = inputs[1].shape
        result = self.builder.reshape(inputs[0], shape)
        self._define_outputs(outputs, result)
